from __future__ import annotations
import asyncio
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
import certifi
import ssl

from .config import get_settings, PREMIUM_FIELD_MAP
from .cache_io import stored_option_chain, stored_option_price, merge_nested_dicts

import requests
from typing import Optional

def _resolve_api_key() -> str:
    """
    Prefer env var POLYGON_API_KEY.
    Fall back to legacy polygonio_config.py if present.
    """
    key = os.getenv("POLYGON_API_KEY")
    if key:
        return key

    # Legacy fallback (not recommended): polygonio_config.py in repo root
    try:
        import importlib
        cfg = importlib.import_module("polygonio_config")
        key = getattr(cfg, "POLYGON_API_KEY", None) or getattr(cfg, "API_KEY", None)
    except Exception:
        key = None

    if not key:
        raise RuntimeError(
            "Missing Polygon API key. Set POLYGON_API_KEY in your environment "
            "(preferred), or provide polygonio_config.py with POLYGON_API_KEY."
        )
    return key

log = logging.getLogger(__name__)

def _load_polygon_key_from_config() -> str:
    """Load POLYGON_API_KEY from env or ~/.etrade_quant/config.json.
    Env var wins. Returns empty string if not found or on error.
    """
    key = os.getenv("POLYGON_API_KEY", "").strip()
    if key:
        return key
    try:
        key = polygonio_config.API_KEY
        if key:
            return key
    except Exception as e:
        log.warning("Failed to read Polygon key from")
    return ""


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
        # prefer explicit arg -> settings -> config/env fallback
        self.api_key = api_key or getattr(s, "polygon_api_key", "") or _load_polygon_key_from_config()
        if not self.api_key:
            log.warning("Polygon API key is empty")
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session: Optional[aiohttp.ClientSession] = None
        self._force_chain_update = s.option_chain_force_update
        self._premium_field = PREMIUM_FIELD_MAP.get(s.premium_price_mode, "trade_price")

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
                d = stored_option_chain[t][exp_key][asof_key][cp_key]
                if isinstance(d, dict):
                    return d
            except Exception:
                pass

        # Otherwise query Polygon
        strike_dict = await self._query_polygon_for_option_chain_async(t, exp_key, cp_key, asof_key)

        # Persist into nested cache
        stored_option_chain.setdefault(t, {}).setdefault(exp_key, {}).setdefault(asof_key, {})[cp_key] = strike_dict or {}
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
                        resp.raise_for_status()
                        data = await resp.json()
                        print(f"Fetched option chain: {ticker} {call_put} exp={expiration_date} as_of={as_of} order={order}")
                for item in data.get("results", []) or []:
                    sp = item.get("strike_price")
                    sym = item.get("ticker")
                    if sp is not None and sym:
                        results[sp] = sym
            except aiohttp.ClientResponseError as e:
                log.error("Polygon chain error (%s %s %s %s): %s", ticker, expiration_date, call_put, as_of, e)
            except Exception as e:
                log.error("Unexpected chain error: %s", e)
        return results

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
                log.error("Error fetching option chain: %s", result)
                result = {}
            out[ticker].setdefault(exp, {}).setdefault(as_of, {})[cp] = result
        return out

    # ---------------- Quotes / Prices ----------------
    async def get_option_prices_batch_async(
        self,
        ticker: str,
        options_list: List[Dict[str, Any]],
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
    ) -> Dict[str, Any]:
        """Fetch price using the configured premium field and write into cache."""
        premium_field = self._premium_field  # 'mid_price' | 'trade_price' | 'close_price'

        # Compose endpoint based on premium type
        if premium_field == "trade_price":
            url = f"https://api.polygon.io/v3/trades/{option_ticker}"
            params = {
                "timestamp": pricing_date,
                "order": "desc",
                "sort": "timestamp",
                "limit": 10,
                "apiKey": self.api_key,
            }
            parse = self._parse_trade
        elif premium_field == "mid_price":
            url = f"https://api.polygon.io/v3/quotes/{option_ticker}"
            params = {
                "timestamp": pricing_date,
                "order": "desc",
                "sort": "timestamp",
                "limit": 10,
                "apiKey": self.api_key,
            }
            parse = self._parse_quote_mid
        else:  # close
            url = f"https://api.polygon.io/v1/open-close/{option_ticker}/{pricing_date}"
            params = {"apiKey": self.api_key}
            parse = self._parse_open_close

        # Retry loop
        for attempt in range(1, self.retries + 1):
            try:
                async with self.semaphore:
                    async with self.session.get(url, params=params) as resp:
                        if resp.status != 200:
                            full = f"{url}?{urlencode(params)}"
                            log.warning("Non-200 from Polygon: %s -> %s", full, resp.status)
                            self._write_invalid_option(ticker, strike_price, call_put, expiration_date, pricing_date, premium_field)
                            return {}
                        data = await resp.json()
                payload = parse(data)
                if payload:
                    self._write_option_payload(ticker, strike_price, call_put, expiration_date, pricing_date, payload)
                    print(f"Stored {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, Pricing:{pricing_date}:{payload}")
                    return payload
                # No valid data → write invalid marker
                self._write_invalid_option(ticker, strike_price, call_put, expiration_date, pricing_date, premium_field)
                return {}
            except Exception as e:
                if attempt >= self.retries:
                    log.error("Max retries exceeded for %s %s %s@%s: %s", ticker, call_put, strike_price, pricing_date, e)
                    return {}
                wait = self.backoff_factor * (2 ** (attempt - 1))
                log.warning("Attempt %d failed (%s). Retrying in %.2fs", attempt, e, wait)
                await asyncio.sleep(wait)

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
        i = 0
        while i < len(results) and (ask <= 0 or bid <= 0):
            q = results[i]
            ask = q.get("ask_price", 0.0)
            bid = q.get("bid_price", 0.0)
            ask_size = q.get("ask_size", 0)
            bid_size = q.get("bid_size", 0)
            i += 1
        if ask > 0 and bid > 0:
            mid = round((ask + bid) / 2.0, 3)
            return {
                "ask_price": ask,
                "bid_price": bid,
                "ask_size": ask_size,
                "bid_size": bid_size,
                "mid_price": mid,
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
    def _write_option_payload(
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        payload: Dict[str, Any],
    ) -> None:
        t = ticker.upper()
        strike_key = round(float(strike_price), 2)
        stored_option_price.setdefault(t, {}).setdefault(pricing_date, {}).setdefault(strike_key, {}).setdefault(expiration_date, {}).setdefault(call_put, {}).update(payload)
        log.debug("Stored %s %s K=%s exp=%s on %s: %s", t, call_put, strike_key, expiration_date, pricing_date, payload)

    @staticmethod
    def _write_invalid_option(
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        premium_field: str,
    ) -> None:
        t = ticker.upper()
        strike_key = round(float(strike_price), 2)
        if premium_field == "trade_price":
            invalid = {"trade_size": 0, "trade_price": 0.0, "sip_timestamp": 0}
        elif premium_field == "close_price":
            invalid = {"close_price": 0.0, "close_volume": 0}
        else:
            invalid = {"ask_price": 0.0, "bid_price": 0.0, "ask_size": 0, "bid_size": 0, "mid_price": 0.0}
        stored_option_price.setdefault(t, {}).setdefault(pricing_date, {}).setdefault(strike_key, {}).setdefault(expiration_date, {}).setdefault(call_put, {}).update(invalid)
