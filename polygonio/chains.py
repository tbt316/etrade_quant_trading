from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Iterable

import asyncio

from .config import get_settings, resolve_premium_field
from .cache_io import load_stored_option_data, stored_option_chain, stored_option_price
from .poly_client import PolygonAPIClient


def _to_date(d: str | date | None) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _ds(d: str | date | None) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, str):
        return d
    return d.strftime("%Y-%m-%d")


def _price_field() -> str:
    return resolve_premium_field(get_settings())


def _target_timestamp_ns(as_of_str: str) -> tuple[Optional[int], Optional[int]]:
    settings = get_settings()
    win = max(1, int(getattr(settings, "premium_time_window_secs", 60)))
    target_str = getattr(settings, "premium_time_target", "12:30:00") or "12:30:00"

    try:
        target_time = time.fromisoformat(target_str)
    except ValueError:
        parts = [int(part) for part in target_str.replace(" ", "").split(":") if part]
        while len(parts) < 3:
            parts.append(0)
        try:
            target_time = time(parts[0], parts[1], parts[2])
        except Exception:
            target_time = time(12, 30, 0)

    try:
        if len(as_of_str) > 10:
            dt_target = datetime.strptime(as_of_str, "%Y-%m-%d %H:%M:%S")
        else:
            base_date = datetime.strptime(as_of_str, "%Y-%m-%d")
            dt_target = datetime.combine(base_date.date(), target_time)
    except Exception:
        return None, None

    target_ns = int(dt_target.timestamp() * 1_000_000_000)
    window_ns = win * 1_000_000_000
    return target_ns, window_ns


def _timestamp_matches_window(
    *,
    as_of_str: str,
    payload: Dict[str, Any],
    premium_field: str,
) -> bool:
    """Return True if *payload* falls within the configured premium time window."""

    if premium_field == "close_price":
        # Close price does not have an intraday timestamp requirement
        return premium_field in payload and payload.get(premium_field) is not None

    if not isinstance(payload, dict):
        return False

    ts = payload.get("sip_timestamp")
    if ts is None and premium_field == "trade_price":
        ts = payload.get("target_timestamp")

    try:
        ts_val = int(ts)
    except (TypeError, ValueError):
        return False

    if ts_val <= 0:
        return False

    target_ns, window_ns = _target_timestamp_ns(as_of_str)
    if target_ns is None or window_ns is None:
        return False

    return target_ns - window_ns <= ts_val <= target_ns + window_ns


def _cached_price_usable(
    *,
    data: Dict[str, Any] | None,
    premium_field: str,
    as_of_str: str,
) -> bool:
    if not isinstance(data, dict):
        return False
    value = data.get(premium_field)
    try:
        if value is None or float(value) <= 0.0:
            return False
    except Exception:
        return False
    return _timestamp_matches_window(as_of_str=as_of_str, payload=data, premium_field=premium_field)


def _should_skip_fetch_due_to_invalid(
    *,
    data: Dict[str, Any] | None,
    premium_field: str,
    as_of_str: str,
) -> bool:
    if not isinstance(data, dict):
        return False

    target_ns, _ = _target_timestamp_ns(as_of_str)
    if target_ns is None:
        return False

    try:
        marker_ns = int(data.get("target_timestamp") or 0)
    except (TypeError, ValueError):
        marker_ns = 0

    if marker_ns != target_ns:
        return False

    try:
        val = data.get(premium_field)
        if val is None or float(val) <= 0.0:
            return True
    except Exception:
        return True

    if premium_field == "trade_price" and int(data.get("trade_size") or 0) == 0:
        return True

    return False


FALLBACK_MAX_WEEKS = 5


def _has_chain_for(ticker: str, as_of_s: str, exp_s: str) -> Tuple[bool, Dict[float, Any], Dict[float, Any]]:
    bucket = stored_option_chain.get(ticker, {}).get(exp_s, {}).get(as_of_s, {})
    calls = bucket.get("call") or {}
    puts = bucket.get("put") or {}
    return (len(calls) > 0 and len(puts) > 0, calls, puts)


def find_available_expiration(
    ticker: str,
    as_of: str | date,
    target_expiration: str | date,
    max_weeks: int = FALLBACK_MAX_WEEKS,
) -> Tuple[Optional[str], Optional[Dict[float, Any]], Optional[Dict[float, Any]]]:
    as_of_s = _ds(as_of)
    target_d = _to_date(target_expiration)
    if as_of_s is None or target_d is None:
        return None, None, None

    # exact
    ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(target_d))
    if ok:
        return _ds(target_d), calls, puts

    # forward
    for k in range(1, max_weeks + 1):
        cand = target_d + timedelta(days=7 * k)
        ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(cand))
        if ok:
            return _ds(cand), calls, puts

    # backward
    for k in range(1, max_weeks + 1):
        cand = target_d - timedelta(days=7 * k)
        ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(cand))
        if ok:
            return _ds(cand), calls, puts

    return None, None, None


@lru_cache(maxsize=2048)
def _get_price_bucket(ticker: str, as_of_s: str) -> Dict[float, Dict[str, Dict[str, Dict[str, float]]]]:
    return stored_option_price.get(ticker, {}).get(as_of_s, {})


def _extract_chain_for_expiration(
    *,
    ticker: str,
    as_of_s: str,
    expiration_s: str,
) -> Tuple[Dict[float, str], Dict[float, str]]:
    bucket = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {})
    call_symbols = bucket.get("call") or {}
    put_symbols = bucket.get("put") or {}
    return call_symbols, put_symbols


def _extract_premiums_for_strikes(
    *,
    ticker: str,
    as_of_s: str,
    expiration_s: str,
    strikes: Iterable[float],
    opt_type: str,
) -> Dict[float, float]:
    pf = _price_field()
    price_bucket = _get_price_bucket(ticker, as_of_s)
    out: Dict[float, float] = {}
    for k in strikes:
        exp_map = price_bucket.get(k, {})
        type_map = exp_map.get(expiration_s, {}).get(opt_type, {})
        val = type_map.get(pf)
        if val is not None:
            try:
                out[float(k)] = float(val)
            except Exception:
                pass
    return out


def _window_strikes(
    *,
    strikes: Iterable[float],
    spot: Optional[float],
    force_otm: bool,
    is_call: bool,
) -> Iterable[float]:
    """
    Return all provided strikes, optionally enforcing out-of-the-money
    filtering. The previous implementation trimmed strikes using
    ``option_range``; this function now intentionally ignores any range
    constraint so the caller receives the full chain for coverage checks.
    """

    filt = []
    for k in strikes:
        try:
            kf = float(k)
        except Exception:
            continue

        if force_otm and spot is not None and spot > 0:
            if is_call and kf < spot:
                continue
            if not is_call and kf > spot:
                continue

        filt.append(kf)

    return sorted(set(filt))


@dataclass(frozen=True)
class ChainResult:
    ticker: str
    as_of: str
    expiration: str
    call_options: Dict[float, Dict[str, Any]]
    put_options: Dict[float, Dict[str, Any]]


def get_option_chain_for_date(
    *,
    ticker: str,
    as_of_str: Optional[str] = None,
    expiration_str: Optional[str] = None,
    as_of: Optional[str | date] = None,
    expiration: Optional[str | date] = None,
    spot: Optional[float] = None,
    option_range: Optional[float] = None,
    force_otm: Optional[bool] = None,
) -> Optional[ChainResult]:
    # `option_range` is accepted for legacy callers but intentionally ignored
    # so that the full strike chain is returned. Range checks are performed by
    # upstream consumers.
    settings = get_settings()
    force_otm = settings.force_otm if force_otm is None else force_otm

    as_of_s = _ds(as_of_str) or _ds(as_of)
    expiration_s = _ds(expiration_str) or _ds(expiration)
    if as_of_s is None or expiration_s is None:
        return None

    call_syms = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("call") or {}
    put_syms  = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("put") or {}
    if not call_syms or not put_syms:
        # Attempt to fetch missing chain data from Polygon
        async def _fetch() -> None:
            async with PolygonAPIClient() as client:
                reqs = [(expiration_s, as_of_s, "call"), (expiration_s, as_of_s, "put")]
                await client.get_option_chains_batch_async(ticker, reqs)

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                new_loop.run_until_complete(_fetch())
                new_loop.close()
            else:
                loop.run_until_complete(_fetch())
        except Exception:
            pass

        call_syms = (
            stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("call")
            or call_syms
        )
        put_syms = (
            stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("put")
            or put_syms
        )

    if not call_syms or not put_syms:
        chosen_exp, calls_fallback, puts_fallback = find_available_expiration(
            ticker=ticker, as_of=as_of_s, target_expiration=expiration_s
        )
        if not chosen_exp:
            return None
        expiration_s = chosen_exp
        call_syms = calls_fallback or {}
        put_syms  = puts_fallback or {}

    all_call_strikes = list(call_syms.keys())
    all_put_strikes = list(put_syms.keys())

    win_call_strikes = _window_strikes(
        strikes=all_call_strikes, spot=spot, force_otm=force_otm, is_call=True
    )
    win_put_strikes = _window_strikes(
        strikes=all_put_strikes, spot=spot, force_otm=force_otm, is_call=False
    )

    call_prem = _extract_premiums_for_strikes(
        ticker=ticker, as_of_s=as_of_s, expiration_s=expiration_s, strikes=win_call_strikes, opt_type="call"
    )
    put_prem = _extract_premiums_for_strikes(
        ticker=ticker, as_of_s=as_of_s, expiration_s=expiration_s, strikes=win_put_strikes, opt_type="put"
    )

    call_options: Dict[float, Dict[str, Any]] = {}
    for k in win_call_strikes:
        sym = call_syms.get(k)
        prem = call_prem.get(k)
        if sym is not None and prem is not None:
            call_options[float(k)] = {"symbol": sym, "premium": float(prem)}

    put_options: Dict[float, Dict[str, Any]] = {}
    for k in win_put_strikes:
        sym = put_syms.get(k)
        prem = put_prem.get(k)
        if sym is not None and prem is not None:
            put_options[float(k)] = {"symbol": sym, "premium": float(prem)}

    return ChainResult(
        ticker=ticker,
        as_of=as_of_s,
        expiration=expiration_s,
        call_options=call_options,
        put_options=put_options,
    )

# -----------------------------------------------------------------------------
# Legacy-compatible async API
# -----------------------------------------------------------------------------
from datetime import datetime, timedelta


async def pull_option_chain_data(
    ticker: str,
    call_put: str,
    expiration_str: str,
    as_of_str: str,
    close_price: float | None = None,
    *,
    client=None,
    force_otm: bool = False,
    force_update: bool = False,
    fetch_prices: bool = False,
    expiry_window_days: int = 0,
):
    """
    Legacy-compatible wrapper used by pricing/recursive_backtest.

    Returns a 5-tuple:
        (all_call_data, all_put_data, call_opts, put_opts, strike_range)

    - *_opts* are lists of meta dicts aligned by index with *_data* lists.
    - *_data* rows are the stored price dicts containing fields like
      'ask_price','bid_price','mid_price','close_price','trade_price', etc.
    - *strike_range* is a dict like
      ``{"call": {"min_strike": ..., "max_strike": ...}, "put": {...}}``
      describing the strike coverage irrespective of quote availability.

    This function now fetches data only for the requested option type(s) and
    makes a single attempt for the provided expiration date. Callers are
    responsible for trying alternate expirations if needed.
    """
    cp = (call_put or "call_put_both").lower()
    need_calls = cp in {"call", "call_put_both", "call_put", "both"}
    need_puts = cp in {"put", "call_put_both", "call_put", "both"}

    call_syms: Dict[float, str] = {}
    put_syms: Dict[float, str] = {}
    bucket: Dict[str, Any] = {}
    as_of_date: Optional[date] = None

    def _parse_date_safe(val: Optional[str]) -> Optional[date]:
        if not val:
            return None
        try:
            return datetime.strptime(val, "%Y-%m-%d").date()
        except Exception:
            return None

    try:
        as_of_key = as_of_str[:10] if as_of_str and len(as_of_str) >= 10 else as_of_str
        as_of_date = _parse_date_safe(as_of_key)
    except Exception:
        as_of_date = None

    def _expiration_before_as_of(expiration_s: Optional[str]) -> bool:
        if not as_of_date:
            return False
        exp_date = _parse_date_safe(expiration_s)
        return bool(exp_date and exp_date < as_of_date)

    if _expiration_before_as_of(expiration_str):
        try:
            print(
                f"[CHAINS-DEBUG] Skipping expired chain {ticker} exp={expiration_str} as_of={as_of_str} "
                f"need_calls={need_calls} need_puts={need_puts}"
            )
        except Exception:
            pass
        return [], [], [], [], None

    if client is not None:
        reqs = []
        if need_calls:
            reqs.append((expiration_str, as_of_str, "call"))
        if need_puts:
            reqs.append((expiration_str, as_of_str, "put"))
        try:
            chain_data = await client.get_option_chains_batch_async(
                ticker, reqs, force_update=force_update
            )
        except Exception:
            chain_data = {}
        bucket = chain_data.get(ticker, {}).get(expiration_str, {}).get(as_of_str, {})

    if not bucket:
        bucket = stored_option_chain.get(ticker, {}).get(expiration_str, {}).get(as_of_str, {})

    call_syms = bucket.get("call") or {}
    put_syms  = bucket.get("put") or {}

    # If exact expiry has no symbols and a window is provided, query a date range and select closest expiry
    if ((need_calls and not call_syms) or (need_puts and not put_syms)) and client is not None and expiry_window_days > 0:
        try:
            dt_target = datetime.strptime(expiration_str, "%Y-%m-%d").date()
            start = (dt_target - timedelta(days=int(expiry_window_days))).strftime("%Y-%m-%d")
            end = (dt_target + timedelta(days=int(expiry_window_days))).strftime("%Y-%m-%d")
            calls_by_exp: Dict[str, Dict[float, str]] = {}
            puts_by_exp: Dict[str, Dict[float, str]] = {}
            if need_calls:
                calls_by_exp = await client.get_option_contracts_in_range(
                    ticker, call_put="call", as_of=as_of_str, exp_start=start, exp_end=end
                )
            if need_puts:
                puts_by_exp = await client.get_option_contracts_in_range(
                    ticker, call_put="put", as_of=as_of_str, exp_start=start, exp_end=end
                )
            # choose the nearest expiry having needed sides
            def _to_date(s: str):
                try:
                    return datetime.strptime(s, "%Y-%m-%d").date()
                except Exception:
                    return None
            target = dt_target
            exp_candidates = set()
            if need_calls:
                exp_candidates |= set(calls_by_exp.keys())
            if need_puts:
                exp_candidates &= set(puts_by_exp.keys()) if exp_candidates else set(puts_by_exp.keys())
            if exp_candidates:
                best = None
                chosen_exp = None
                for exp_s in exp_candidates:
                    d = _to_date(exp_s)
                    if not d:
                        continue
                    if as_of_date and d < as_of_date:
                        continue
                    diff = abs((d - target).days)
                    if best is None or diff < best:
                        best = diff
                        chosen_exp = exp_s
                if chosen_exp:
                    if _expiration_before_as_of(chosen_exp):
                        chosen_exp = None
                    else:
                        expiration_str = chosen_exp  # update to chosen expiry
                        call_syms = calls_by_exp.get(chosen_exp, {}) if need_calls else {}
                        put_syms = puts_by_exp.get(chosen_exp, {}) if need_puts else {}
                        # persist to cache under chosen expiry
                        try:
                            leaf = (
                                stored_option_chain
                                .setdefault(ticker, {})
                                .setdefault(chosen_exp, {})
                                .setdefault(as_of_str, {})
                            )
                            if need_calls:
                                leaf["call"] = dict(call_syms)
                            if need_puts:
                                leaf["put"] = dict(put_syms)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[CHAINS-DEBUG] range selection failed: {e}")

    if (need_calls and not call_syms) or (need_puts and not put_syms):
        try:
            avail_exps = list((stored_option_chain.get(ticker, {}) or {}).keys())
            asof_map = (stored_option_chain.get(ticker, {}).get(expiration_str, {}) or {})
            asof_keys = list(asof_map.keys()) if isinstance(asof_map, dict) else []
            print(
                f"[CHAINS-DEBUG] No chain symbols for {ticker} exp={expiration_str} as_of={as_of_str} "
                f"need_calls={need_calls} need_puts={need_puts}. "
                f"available_expirations={len(avail_exps)} sample={avail_exps[:5]} as_of_keys={asof_keys[:5]}"
            )
        except Exception:
            print(
                f"[CHAINS-DEBUG] No chain symbols for {ticker} exp={expiration_str} as_of={as_of_str} (need_calls={need_calls}, need_puts={need_puts})"
            )
        return [], [], [], [], None

    all_call_strikes = (
        sorted(float(k) for k in call_syms.keys()) if need_calls else []
    )
    all_put_strikes = (
        sorted(float(k) for k in put_syms.keys()) if need_puts else []
    )

    win_call_strikes = (
        _window_strikes(
            strikes=call_syms.keys(),
            spot=close_price,
            force_otm=force_otm,
            is_call=True,
        )
        if need_calls else []
    )
    win_put_strikes = (
        _window_strikes(
            strikes=put_syms.keys(),
            spot=close_price,
            force_otm=force_otm,
            is_call=False,
        )
        if need_puts else []
    )

    # Limit strikes to within +/- 5% of the spot close to avoid polling the entire chain
    try:
        if close_price is not None:
            lo = float(close_price) * 0.95
            hi = float(close_price) * 1.05
            if win_call_strikes:
                win_call_strikes = [float(k) for k in win_call_strikes if lo <= float(k) <= hi]
            if win_put_strikes:
                win_put_strikes = [float(k) for k in win_put_strikes if lo <= float(k) <= hi]
    except Exception:
        # If filtering fails for any reason, keep the original windows
        pass

    strike_range: Optional[Dict[str, Dict[str, float]]] = {}
    if need_calls and all_call_strikes:
        strike_range["call"] = {
            "min_strike": float(min(all_call_strikes)),
            "max_strike": float(max(all_call_strikes)),
        }
    if need_puts and all_put_strikes:
        strike_range["put"] = {
            "min_strike": float(min(all_put_strikes)),
            "max_strike": float(max(all_put_strikes)),
        }
    if not strike_range:
        strike_range = None

    pf_bucket = stored_option_price.get(ticker, {}).get(as_of_str, {})
    pf = _price_field()

    call_opts = [
        {"strike_price": float(k), "expiration_date": expiration_str, "option_ticker": call_syms[k]}
        for k in win_call_strikes
    ] if need_calls else []
    put_opts = [
        {"strike_price": float(k), "expiration_date": expiration_str, "option_ticker": put_syms[k]}
        for k in win_put_strikes
    ] if need_puts else []

    all_call_data: list[dict] = []
    all_put_data: list[dict] = []
    reqs: list[dict[str, Any]] = []
    call_missing_idx: list[int] = []
    put_missing_idx: list[int] = []

    for i, opt in enumerate(call_opts):
        strike = opt["strike_price"]
        data = pf_bucket.get(strike, {}).get(expiration_str, {}).get("call", {})
        use_cached = (
            not force_update
            and _cached_price_usable(data=data, premium_field=pf, as_of_str=as_of_str)
        )
        skip_fetch = (
            not force_update
            and _should_skip_fetch_due_to_invalid(data=data, premium_field=pf, as_of_str=as_of_str)
        )
        if use_cached:
            all_call_data.append(data)
        elif skip_fetch:
            all_call_data.append(data or {})
        else:
            all_call_data.append(None)
            if need_calls:
                reqs.append({
                    "strike_price": strike,
                    "call_put": "call",
                    "expiration_date": expiration_str,
                    "quote_timestamp": as_of_str,
                    "option_ticker": opt["option_ticker"],
                })
                call_missing_idx.append(i)

    for i, opt in enumerate(put_opts):
        strike = opt["strike_price"]
        data = pf_bucket.get(strike, {}).get(expiration_str, {}).get("put", {})
        use_cached = (
            not force_update
            and _cached_price_usable(data=data, premium_field=pf, as_of_str=as_of_str)
        )
        skip_fetch = (
            not force_update
            and _should_skip_fetch_due_to_invalid(data=data, premium_field=pf, as_of_str=as_of_str)
        )
        if use_cached:
            all_put_data.append(data)
        elif skip_fetch:
            all_put_data.append(data or {})
        else:
            all_put_data.append(None)
            if need_puts:
                reqs.append({
                    "strike_price": strike,
                    "call_put": "put",
                    "expiration_date": expiration_str,
                    "quote_timestamp": as_of_str,
                    "option_ticker": opt["option_ticker"],
                })
                put_missing_idx.append(i)

    fetch_needed = False
    if fetch_prices:
        if force_update:
            fetch_needed = bool(reqs)
        else:
            if call_opts and len(call_missing_idx) > 0.1 * len(call_opts):
                fetch_needed = True
            if put_opts and len(put_missing_idx) > 0.1 * len(put_opts):
                fetch_needed = True

    try:
        print(
            f"[CHAINS-DEBUG] Pricing cache summary exp={expiration_str} as_of={as_of_str} "
            f"call_opts={len(call_opts)} cached_calls={len(call_opts)-len(call_missing_idx)} missing_calls={len(call_missing_idx)} "
            f"put_opts={len(put_opts)} cached_puts={len(put_opts)-len(put_missing_idx)} missing_puts={len(put_missing_idx)} "
            f"fetch_needed={fetch_needed}"
        )
    except Exception:
        pass

    if fetch_needed and reqs and client is not None:
        try:
            fetched = await client.get_option_prices_batch_async(ticker, reqs)
        except Exception:
            fetched = []
        j = 0
        for idx in call_missing_idx:
            all_call_data[idx] = fetched[j] if j < len(fetched) else {}
            j += 1
        for idx in put_missing_idx:
            all_put_data[idx] = fetched[j] if j < len(fetched) else {}
            j += 1

    def _valid(prem: dict) -> bool:
        try:
            return float(prem.get(pf, 0.0)) > 0.0
        except Exception:
            return False

    call_filtered = [(opt, prem) for opt, prem in zip(call_opts, all_call_data) if _valid(prem)]
    put_filtered = [(opt, prem) for opt, prem in zip(put_opts, all_put_data) if _valid(prem)]

    if not call_filtered and not put_filtered:
        try:
            missing_calls = sum(1 for d in all_call_data if not _valid(d)) if all_call_data else 0
            missing_puts = sum(1 for d in all_put_data if not _valid(d)) if all_put_data else 0
            print(
                f"[CHAINS-DEBUG] No valid premiums for {ticker} exp={expiration_str} as_of={as_of_str}. "
                f"call_opts={len(call_opts)} invalid_calls={missing_calls} put_opts={len(put_opts)} invalid_puts={missing_puts} "
                f"pf={pf}. Consider enabling force_update or checking cache."
            )
        except Exception:
            print(
                f"[CHAINS-DEBUG] No valid premiums for {ticker} exp={expiration_str} as_of={as_of_str}."
            )

    call_opts, all_call_data = zip(*call_filtered) if call_filtered else ([], [])
    put_opts, all_put_data = zip(*put_filtered) if put_filtered else ([], [])
    call_opts, all_call_data = list(call_opts), list(all_call_data)
    put_opts, all_put_data = list(put_opts), list(all_put_data)

    return all_call_data, all_put_data, call_opts, put_opts, strike_range
