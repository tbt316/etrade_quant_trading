from __future__ import annotations
import asyncio
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import certifi
import ssl

from .config import get_settings, resolve_premium_field, resolve_premium_mode
from .cache_io import (
    stored_option_chain,
    stored_option_price,
    merge_nested_dicts_with_count,
    record_unsaved_option_entries,
)
from .http_safety import redacted_request_url, safe_exception_summary

import requests

log = logging.getLogger(__name__)


def _resolve_api_key(explicit_key: Optional[str], settings_key: str) -> str:
    """Resolve a Polygon key without importing local credential modules."""
    for candidate in (explicit_key, settings_key, os.getenv("POLYGON_API_KEY")):
        key = (candidate or "").strip()
        if key:
            return key
    raise RuntimeError(
        "Missing Polygon API key. Pass api_key explicitly or set POLYGON_API_KEY."
    )


@dataclass(frozen=True)
class _Retry:
    retries: int = 1
    backoff_factor: float = 0.5  # exponential backoff base


class PolygonAPIClient:
    """Async client for Polygon.io option chains & quotes.

    Responsibilities
    ---------------
    - Manage a shared aiohttp session with TLS
    - Fetch option chains (v3/reference/options/contracts)
    - Fetch option quotes/trades/open-close based on premium mode
    - Write results into shared caches (stored_option_chain / stored_option_price)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        max_concurrent_requests: int = 10,
        retries: int = _Retry.retries,
        backoff_factor: float = _Retry.backoff_factor,
    ) -> None:
        s = get_settings()
        self.api_key = _resolve_api_key(
            api_key,
            getattr(s, "polygon_api_key", ""),
        )
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session: Optional[aiohttp.ClientSession] = None
        self._force_chain_update = s.option_chain_force_update
        self._premium_mode = resolve_premium_mode(s)
        self._premium_field = resolve_premium_field(s)
        self._debug_quote_poll = bool(getattr(s, "debug_polygon_quote", False))

    # ---------------- Context manager lifecycle ----------------
    async def __aenter__(self) -> "PolygonAPIClient":
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=self.ssl_context))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    # ---------------- Chains ----------------
    async def get_option_chain_async(
        self,
        ticker: str,
        expiration_date: str,
        call_put: str,
        as_of_date: str,
        *,
        force_update: bool = False,
    ) -> Dict[float, str]:
        """Return {strike_price: option_symbol} for (ticker, expiry, call/put, as_of).
        Uses memory cache unless forced.
        """
        t = ticker.upper()
        exp_key = str(expiration_date)
        asof_key = str(as_of_date)
        cp_key = call_put.lower()

        # Try memory unless forced
        if not (self._force_chain_update or force_update):
            try:
                exp_bucket = stored_option_chain[t][exp_key]
                asof_bucket = exp_bucket[asof_key]
                d = asof_bucket.get(cp_key)
                if isinstance(d, dict) and d:
                    return d
                if asof_bucket.get("_empty"):
                    return {}
            except Exception:
                pass

        # Otherwise query Polygon
        print(f"[DEBUG] Fetching option chain from Polygon: {t} exp={exp_key} as_of={asof_key} cp={cp_key}")
        strike_dict = await self._query_polygon_for_option_chain_async(t, exp_key, cp_key, asof_key)

        # Persist into nested cache and track unsaved entries
        leaf = (
            stored_option_chain
            .setdefault(t, {})
            .setdefault(exp_key, {})
            .setdefault(asof_key, {})
            .setdefault(cp_key, {})
        )
        added = merge_nested_dicts_with_count(leaf, strike_dict or {})
        record_unsaved_option_entries(t, added)
        return strike_dict or {}

    async def _query_polygon_for_option_chain_async(
        self,
        ticker: str,
        expiration_date: str,
        call_put: str,
        as_of: str,
    ) -> Dict[float, str]:
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params_base = {
            "underlying_ticker": ticker,
            "expiration_date": expiration_date,
            "as_of": as_of,
            "contract_type": call_put,
            "apiKey": self.api_key,
            "limit": 500,
        }
        results: Dict[float, str] = {}
        for order in ("asc", "desc"):
            params = dict(params_base)
            params["order"] = order
            try:
                async with self.semaphore:
                    async with self.session.get(url, params=params) as resp:
                        print(
                            "[DEBUG] Querying Polygon for option chain: "
                            f"{redacted_request_url(url, params)}"
                        )
                        resp.raise_for_status()
                        data = await resp.json()
                for item in data.get("results", []) or []:
                    sp = item.get("strike_price")
                    sym = item.get("ticker")
                    if sp is not None and sym:
                        results[sp] = sym
                        # print(f"[DEBUG] Fetched option: {ticker} {call_put} exp={expiration_date} as_of={as_of} strike={sp} symbol={sym}")
            except aiohttp.ClientResponseError as error:
                print(
                    f"Polygon chain error ({ticker} {expiration_date} "
                    f"{call_put} {as_of}): {safe_exception_summary(error)}"
                )
            except Exception as error:
                print(
                    "Unexpected Polygon chain error: "
                    f"{safe_exception_summary(error)}"
                )
        return results

    async def get_option_contracts_in_range(
        self,
        ticker: str,
        *,
        call_put: str,
        as_of: str,
        exp_start: str,
        exp_end: str,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        limit: int = 1000,
    ) -> Dict[str, Dict[float, str]]:
        """Return {expiration_date: {strike_price: option_symbol}} for expiries in [exp_start, exp_end]."""
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "as_of": as_of,
            "contract_type": call_put,
            "expiration_date.gte": exp_start,
            "expiration_date.lte": exp_end,
            "order": "asc",
            "limit": int(limit),
            "apiKey": self.api_key,
        }
        if strike_price_gte is not None:
            try:
                params["strike_price.gte"] = float(strike_price_gte)
            except Exception:
                pass
        if strike_price_lte is not None:
            try:
                params["strike_price.lte"] = float(strike_price_lte)
            except Exception:
                pass
        by_exp: Dict[str, Dict[float, str]] = {}
        try:
            async with self.semaphore:
                async with self.session.get(url, params=params) as resp:
                    print(
                        "[DEBUG] Querying Polygon contracts range: "
                        f"{redacted_request_url(url, params)}"
                    )
                    resp.raise_for_status()
                    data = await resp.json()
            for item in data.get("results", []) or []:
                exp_s = str(item.get("expiration_date"))
                sp = item.get("strike_price")
                sym = item.get("ticker")
                if not exp_s or sp is None or not sym:
                    continue
                m = by_exp.setdefault(exp_s, {})
                m[float(sp)] = sym
        except Exception as error:
            print(
                f"[WARN] contracts range fetch failed for {ticker} "
                f"{call_put} {as_of}: {safe_exception_summary(error)}"
            )
        return by_exp

    async def get_option_chains_batch_async(
        self,
        ticker: str,
        unique_chain_requests: List[Tuple[str, str, str]],
        *,
        force_update: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, str]]]]]:
        """Batch fetch chains for [(expiration_date, as_of_date, call_put), ...]."""
        tasks = [
            asyncio.create_task(
                self.get_option_chain_async(ticker, exp, cp, as_of, force_update=force_update)
            )
            for (exp, as_of, cp) in unique_chain_requests
        ]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        out: Dict[str, Dict[str, Dict[str, Dict[str, Dict[float, str]]]]] = {ticker: {}}
        for (exp, as_of, cp), result in zip(unique_chain_requests, fetched):
            if isinstance(result, Exception):
                log.error(
                    "Error fetching option chain: %s",
                    safe_exception_summary(result),
                )
                result = {}
            try:
                strikes_count = len(result) if isinstance(result, dict) else 0
                print(
                    f"[DEBUG] Chain summary: {ticker} exp={exp} as_of={as_of} cp={cp} strikes={strikes_count}"
                )
            except Exception:
                pass
            out[ticker].setdefault(exp, {}).setdefault(as_of, {})[cp] = result
        return out

    # ---------------- Quotes / Prices ----------------
    async def get_option_prices_batch_async(
        self,
        ticker: str,
        options_list: List[Dict[str, Any]],
        skip_write: bool = False,
    ) -> List[Dict[str, Any]]:
        """Batch fetch prices for a deduplicated list of options.

        Each option dict must contain: strike_price, call_put, expiration_date, quote_timestamp, option_ticker
        """
        # Deduplicate
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for opt in options_list:
            key = (
                opt.get("strike_price"),
                opt.get("call_put"),
                opt.get("expiration_date"),
                opt.get("quote_timestamp"),
                opt.get("option_ticker"),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(opt)

        async def _one(opt: Dict[str, Any]) -> Dict[str, Any]:
            return await self._query_and_store_option_price(
                ticker=ticker,
                strike_price=float(opt["strike_price"]),
                call_put=str(opt["call_put"]).lower(),
                expiration_date=str(opt["expiration_date"]),
                pricing_date=str(opt["quote_timestamp"]),
                option_ticker=str(opt.get("option_ticker")),
                skip_write=skip_write,
            )

        tasks = [asyncio.create_task(_one(o)) for o in deduped]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                log.error("Error fetching option price: %s", r)
                out.append({})
            else:
                out.append(r)
        return out

    async def _query_and_store_option_price(
        self,
        *,
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        option_ticker: str,
        skip_write: bool = False,
    ) -> Dict[str, Any]:
        """Fetch price using the configured premium field and write into cache."""
        premium_field = self._premium_field  # 'mid_price' | 'trade_price' | 'close_price'
        mode = self._premium_mode

        # Compose endpoint based on premium type
        target_ns = 0
        cutoff_ns = 0
        window_lo_ns: int | None = None
        window_hi_ns: int | None = None
        
        # Default query params
        params = {
            "apiKey": self.api_key,
        }
        
        if mode == "trade":
            url = f"https://api.polygon.io/v3/trades/{option_ticker}"
            params.update({
                "order": "desc",
                "sort": "timestamp",
                "limit": 100,
            })
            parse = None
        elif mode in ("mid", "quote"):
            url = f"https://api.polygon.io/v3/quotes/{option_ticker}"
            params.update({
                "order": "desc",
                "sort": "timestamp",
                "limit": 100,
            })
            parse = None
        else:  # close
            url = f"https://api.polygon.io/v1/open-close/{option_ticker}/{pricing_date}"
            parse = self._parse_open_close

        # If we are using trades/quotes, convert pricing_date into a window
        # from 9:00 AM to target time.
        query_specs: List[Tuple[str, Optional[int], Optional[int]]] = []
        if parse is None:  # trade or mid
            from datetime import datetime, time, timedelta
            from .config import get_settings
            try:
                if len(pricing_date) > 10:
                    dt = datetime.strptime(pricing_date, "%Y-%m-%d %H:%M:%S")
                else:
                    d0 = datetime.strptime(pricing_date, "%Y-%m-%d")
                    s = get_settings()
                    hh, mm, ss = [int(x) for x in (s.premium_time_target or "12:45:00").split(":")]
                    dt = datetime.combine(d0.date(), time(hh, mm, ss))
                
                # Start at 9:00 AM on the same day
                start_dt = datetime.combine(dt.date(), time(9, 0, 0))
                
                window_lo_ns = int(start_dt.timestamp() * 1_000_000_000)
                window_hi_ns = int(dt.timestamp() * 1_000_000_000)
                target_ns = window_hi_ns
                
                # replace generic timestamp with explicit bounds
                params.pop("timestamp", None)
                params["timestamp.gte"] = str(window_lo_ns)
                params["timestamp.lte"] = str(window_hi_ns)
                
                # We only need one spec now since we are just querying the range desc
                query_specs = [("desc", window_lo_ns, window_hi_ns)]
                
            except Exception:
                # if parsing fails, keep original params (though this path is unlikely to work well with new requirements)
                target_ns = 0
                cutoff_ns = 0

        # Retry loop
        params_for_log: Dict[str, Any] | None = dict(params)
        payload_for_log: Dict[str, Any] | None = None
        session = self.session
        if session is None:
            raise RuntimeError("PolygonAPIClient session is not initialized")
        for attempt in range(1, self.retries + 1):
            print(f"[DEBUG] Polygon price fetch attempt #{attempt} for {option_ticker} @{pricing_date}")
            try:
                samples: List[Dict[str, Any]] = []
                if parse is not None:
                    async with self.semaphore:
                        async with session.get(url, params=params) as resp:
                            print(
                                "[DEBUG] Querying Polygon for option price: "
                                f"{redacted_request_url(url, params)}"
                            )
                            resp.raise_for_status()
                            data = await resp.json()
                    payload = parse(data)
                    if payload:
                        samples.append(dict(payload))
                else:
                    combined_results: List[Dict[str, Any]] = []
                    seen_ts: set[int] = set()
                    effective_specs = query_specs or [("desc", None, None)]

                    async def _fetch_one(spec_index: int, req_params: Dict[str, Any]) -> Dict[str, Any]:
                        async with self.semaphore:
                            async with session.get(url, params=req_params) as resp:
                                resp.raise_for_status()
                                data_part = await resp.json()
                        return {"spec_index": spec_index, "params": req_params, "data": data_part}

                    tasks: List["asyncio.Task[Dict[str, Any]]"] = []
                    for idx_spec, (order, lower, upper) in enumerate(effective_specs):
                        req_params = dict(params)
                        req_params["order"] = order
                        if lower is not None:
                            req_params["timestamp.gte"] = str(lower)
                        if upper is not None:
                            req_params["timestamp.lte"] = str(upper)
                        tasks.append(asyncio.create_task(_fetch_one(idx_spec, req_params)))

                    fetch_results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
                    for fr in fetch_results:
                        if isinstance(fr, Exception):
                            log.warning(
                                "Polygon price window fetch failed: %s",
                                safe_exception_summary(fr),
                            )
                            continue
                        params_for_log = dict(fr.get("params") or params)
                        data_part = fr.get("data") or {}
                        for item in data_part.get("results") or []:
                            try:
                                ts_key = int(item.get("sip_timestamp") or 0)
                            except Exception:
                                ts_key = 0
                            if ts_key > 0:
                                if ts_key in seen_ts:
                                    continue
                                seen_ts.add(ts_key)
                            combined_results.append(item)
                    results = combined_results
                    best = None
                    if mode == "trade":
                        for t in results:
                            try:
                                ts = int(t.get("sip_timestamp") or 0)
                                price = float(t.get("price") or 0.0)
                                size = int(t.get("size") or 0)
                            except Exception:
                                continue
                            if ts <= 0 or price <= 0:
                                continue
                            dist = abs(ts - target_ns) if target_ns else 0
                            key = (dist, -ts)
                            cand = {
                                "trade_size": size,
                                "trade_price": price,
                                "sip_timestamp": ts,
                                "target_timestamp": target_ns,
                            }
                            samples.append(dict(cand))
                            if (best is None) or (key < best[0]):
                                best = (key, cand)
                        payload = best[1] if best else {}
                    else:  # mid_price via quotes
                        for q in results:
                            try:
                                ask = float(q.get("ask_price") or 0.0)
                                bid = float(q.get("bid_price") or 0.0)
                                ts = int(q.get("sip_timestamp") or 0)
                                ask_size = int(q.get("ask_size") or 0)
                                bid_size = int(q.get("bid_size") or 0)
                            except Exception:
                                continue
                            if ask <= 0 or bid <= 0 or ts <= 0:
                                continue
                            mid = round((ask + bid) / 2.0, 3)
                            dist = abs(ts - target_ns) if target_ns else 0
                            key = (dist, -ts)
                            cand = {
                                "ask_price": ask,
                                "bid_price": bid,
                                "ask_size": ask_size,
                                "bid_size": bid_size,
                                "mid_price": mid,
                                "sip_timestamp": ts,
                                "target_timestamp": target_ns,
                            }
                            samples.append(dict(cand))
                            if (best is None) or (key < best[0]):
                                best = (key, cand)
                        payload = best[1] if best else {}
                if payload:
                    if samples:
                        samples = sorted(
                            [s for s in samples if isinstance(s, dict)],
                            key=lambda x: PolygonAPIClient._sample_timestamp(x) or 0,
                        )
                        payload["_samples"] = samples
                    
                    write_date = pricing_date
                    # We do NOT overwrite write_date with sip_timestamp here, 
                    # because the cache lookup expects to find data under the target date (pricing_date).
                    # The sip_timestamp is already preserved inside the payload.

                    if not skip_write:
                        self._write_option_payload(
                            ticker, strike_price, call_put, expiration_date, write_date, payload, samples
                        )
                        
                        # Print detailed pricing information
                        def _fmt_ts(ns_val: Any) -> str:
                            try:
                                return datetime.fromtimestamp(int(ns_val) / 1_000_000_000).strftime("%H:%M:%S")
                            except Exception:
                                return str(ns_val)
                        
                        details = {
                            "ask_price": payload.get("ask_price"),
                            "bid_price": payload.get("bid_price"),
                            "ask_size": payload.get("ask_size"),
                            "bid_size": payload.get("bid_size"),
                            "mid_price": payload.get("mid_price"),
                            "trade_price": payload.get("trade_price"),
                            "trade_size": payload.get("trade_size"),
                            "sip_timestamp": _fmt_ts(payload.get("sip_timestamp")),
                            "target_timestamp": _fmt_ts(target_ns),
                            "_samples": f"{len(samples)} samples" if samples else "0 samples",
                        }
                        # Remove None values for cleaner output
                        details = {k: v for k, v in details.items() if v is not None}
                        
                        print(
                            f"Stored {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, "
                            f"Pricing:{write_date}:{details}"
                        )
                    else:
                        # Data fetched but not written to cache (skip_write=True)
                        # Caller will validate and write if needed
                        print(
                            f"[DEBUG] Skipped write (validation pending) for {ticker},Strike:{strike_price},"
                            f"{call_put},Expire:{expiration_date}, Pricing:{write_date}"
                        )
                    return payload
                
                if not skip_write:
                    self._write_invalid_option(
                        ticker,
                        strike_price,
                        call_put,
                        expiration_date,
                        pricing_date,
                        premium_field,
                        target_timestamp=target_ns,
                    )
                    print(
                        f"Stored invalid {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, Pricing:{pricing_date}"
                    )
                return {}
            except Exception as error:
                if attempt >= self.retries:
                    print(
                        f"Max retries exceeded for {ticker} {call_put} "
                        f"{strike_price}@{pricing_date}: "
                        f"{safe_exception_summary(error)}"
                    )
                    return {}
                wait = self.backoff_factor * (2 ** (attempt - 1))
                log.warning(
                    "Attempt %d failed (%s). Retrying in %.2fs",
                    attempt,
                    safe_exception_summary(error),
                    wait,
                )
                await asyncio.sleep(wait)
                try:
                    print(
                        f"[DEBUG] Polygon retry #{attempt} for {ticker} {call_put} {strike_price} @{pricing_date} "
                        f"(wait {wait:.2f}s): {safe_exception_summary(error)}"
                    )
                except Exception:
                    pass

        return {}

    # ---------------- Parsers ----------------
    @staticmethod
    def _parse_trade(data: Dict[str, Any]) -> Dict[str, Any]:
        results = data.get("results") or []
        if not results:
            return {}
        t = results[0]
        return {
            "trade_size": t.get("size", 0),
            "trade_price": t.get("price", 0.0),
            "sip_timestamp": t.get("sip_timestamp", 0),
        }

    @staticmethod
    def _parse_quote_mid(data: Dict[str, Any]) -> Dict[str, Any]:
        results = data.get("results") or []
        ask = bid = ask_size = bid_size = 0
        ts_ns = 0
        i = 0
        while i < len(results) and (ask <= 0 or bid <= 0):
            q = results[i]
            ask = q.get("ask_price", 0.0)
            bid = q.get("bid_price", 0.0)
            ask_size = q.get("ask_size", 0)
            bid_size = q.get("bid_size", 0)
            # capture a representative SIP timestamp if present on this quote
            try:
                ts_ns = int(q.get("sip_timestamp") or 0)
            except Exception:
                ts_ns = 0
            i += 1
        if ask > 0 and bid > 0:
            mid = round((ask + bid) / 2.0, 3)
            return {
                "ask_price": ask,
                "bid_price": bid,
                "ask_size": ask_size,
                "bid_size": bid_size,
                "mid_price": mid,
                "sip_timestamp": ts_ns,
            }
        return {}

    @staticmethod
    def _parse_open_close(data: Dict[str, Any]) -> Dict[str, Any]:
        close = data.get("close")
        if close is None:
            return {}
        return {
            "close_price": round(float(close), 3),
            "close_volume": data.get("volume", 0),
        }

    # ---------------- Cache writers ----------------
    @staticmethod
    def _sample_timestamp(sample: Dict[str, Any]) -> Optional[int]:
        """Return a usable timestamp (ns) for sample ordering/dedup."""
        if not isinstance(sample, dict):
            return None
        for key in ("sip_timestamp", "participant_timestamp", "target_timestamp"):
            try:
                val = int(sample.get(key) or 0)
                if val > 0:
                    return val
            except Exception:
                continue
        return None

    @staticmethod
    def _merge_sample_list(
        existing: Optional[List[Dict[str, Any]]],
        incoming: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge two lists of samples keyed by timestamp, keeping earliest entries ordered."""
        ordered: Dict[int, Dict[str, Any]] = {}
        if isinstance(existing, list):
            for item in existing:
                ts = PolygonAPIClient._sample_timestamp(item)
                if ts is None or ts in ordered:
                    continue
                ordered[ts] = dict(item)
        for item in incoming:
            ts = PolygonAPIClient._sample_timestamp(item)
            if ts is None or ts in ordered:
                continue
            ordered[ts] = dict(item)
        return [ordered[k] for k in sorted(ordered.keys())]

    @staticmethod
    def _write_option_payload(
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        payload: Dict[str, Any],
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        t = ticker.upper()
        strike_key = round(float(strike_price), 2)
        # Normalize the pricing date bucket to YYYY-MM-DD to keep cache consistent
        pricing_bucket = str(pricing_date)[:10]
        leaf = (
            stored_option_price
            .setdefault(t, {})
            .setdefault(pricing_bucket, {})
            .setdefault(strike_key, {})
            .setdefault(expiration_date, {})
            .setdefault(call_put, {})
        )
        payload_to_store = dict(payload)
        if samples:
            merged_samples = PolygonAPIClient._merge_sample_list(
                leaf.get("_samples") if isinstance(leaf, dict) else None,
                [dict(s) for s in samples if isinstance(s, dict)],
            )
            if merged_samples:
                payload_to_store["_samples"] = merged_samples
        added = merge_nested_dicts_with_count(leaf, payload_to_store)
        record_unsaved_option_entries(t, added)
        try:
            ts = int(payload.get("target_timestamp") or 0)
            if ts and "_invalid_targets" in leaf:
                targets = leaf.get("_invalid_targets")
                if isinstance(targets, list) and ts in targets:
                    targets.remove(ts)
        except Exception:
            pass

    def _debug_print_quote(
        self,
        *,
        option_ticker: str,
        target_ns: Optional[int],
        window_lo_ns: Optional[int],
        window_hi_ns: Optional[int],
        payload: Dict[str, Any],
        liquidity_ok: bool | None = None,
        label: str | None = None,
    ) -> None:
        if not self._debug_quote_poll:
            return
        try:
            ask = float(payload.get("ask_price") or 0.0)
            bid = float(payload.get("bid_price") or 0.0)
            ask_size = int(payload.get("ask_size") or 0)
            bid_size = int(payload.get("bid_size") or 0)
            mid = float(payload.get("mid_price") or 0.0)
        except Exception:
            return
        if ask <= 0 or bid <= 0 or mid <= 0:
            return
        spread = ask - bid

        def _fmt(ns_val: Optional[int]) -> str:
            if ns_val is None:
                return "n/a"
            try:
                return datetime.fromtimestamp(int(ns_val) / 1_000_000_000).strftime("%H:%M:%S")
            except Exception:
                return "n/a"

        target_str = _fmt(target_ns)
        window_str = f"{_fmt(window_lo_ns)}→{_fmt(window_hi_ns)}"
        color = "\033[36m"
        if liquidity_ok is False:
            color = "\033[31m"
        reset = "\033[0m"
        label_text = f"{label} " if label else ""
        print(
            f"{color}[QUOTE-DEBUG] {label_text}{option_ticker} target={target_str} window=[{window_str}] "
            f"mid={mid:.3f} spread={spread:.3f} ask={ask:.3f}({ask_size}) bid={bid:.3f}({bid_size}){reset}"
        )

    @staticmethod
    def _write_invalid_option(
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        premium_field: str,
        *,
        target_timestamp: int | None = None,
    ) -> None:
        t = ticker.upper()
        strike_key = round(float(strike_price), 2)
        pricing_bucket = str(pricing_date)[:10]
        if premium_field == "trade_price":
            invalid = {
                "trade_size": 0,
                "trade_price": 0.0,
                "sip_timestamp": 0,
                "target_timestamp": target_timestamp or 0,
            }
        elif premium_field == "close_price":
            invalid = {
                "close_price": 0.0,
                "close_volume": 0,
                "target_timestamp": target_timestamp or 0,
            }
        else:
            invalid = {
                "ask_price": 0.0,
                "bid_price": 0.0,
                "ask_size": 0,
                "bid_size": 0,
                "mid_price": 0.0,
                "target_timestamp": target_timestamp or 0,
            }
        leaf = (
            stored_option_price
            .setdefault(t, {})
            .setdefault(pricing_bucket, {})
            .setdefault(strike_key, {})
            .setdefault(expiration_date, {})
            .setdefault(call_put, {})
        )
        added = merge_nested_dicts_with_count(leaf, invalid)
        try:
            ts = int(target_timestamp or 0)
            if ts:
                targets = leaf.setdefault("_invalid_targets", [])
                if ts not in targets:
                    targets.append(ts)
        except Exception:
            pass
        record_unsaved_option_entries(t, added)
