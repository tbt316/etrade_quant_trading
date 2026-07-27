from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import time as _time
import asyncio
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cache_io import stored_option_chain, stored_option_price
from .config import get_settings
from .option_math import calculate_implied_volatility
from .poly_client import PolygonAPIClient
from .recursive_backtest import _price_from_data, _calendar_expiry_candidates



def nearest_strike(strikes: Iterable[float], spot: float) -> float:
    return min(strikes, key=lambda k: abs(k - spot))


def calc_implied_vol(
    *,
    spot: float,
    strike: float,
    premium: float,
    dte: int,
    option_type: str,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> Optional[float]:
    try:
        return calculate_implied_volatility(
            close_price=float(spot),
            strike_price=float(strike),
            option_price=float(premium),
            days_to_expire=float(max(dte, 1)),
            risk_free_rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
            option_type=option_type,
        )
    except Exception:
        return None


def forward_factor(
    *, iv_front: float, iv_back: float, dte_front: int, dte_back: int
) -> Optional[Tuple[float, float]]:
    t1 = dte_front / 365.0
    t2 = dte_back / 365.0
    if t2 <= t1 or iv_front is None or iv_back is None:
        return None
    v1 = iv_front**2
    v2 = iv_back**2
    forward_var = (v2 * t2 - v1 * t1) / (t2 - t1)
    if forward_var <= 0:
        return None
    forward_vol = forward_var**0.5
    if forward_vol == 0:
        return None
    ff = (iv_front - forward_vol) / forward_vol
    return ff, forward_vol


def _sample_bid_ask(sample: Dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    try:
        bid = sample.get("bid_price")
        ask = sample.get("ask_price")
        if bid is None:
            bid = sample.get("bid")
        if ask is None:
            ask = sample.get("ask")
        mid = sample.get("mid_price")
        if mid is None and bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0
        return (
            None if bid is None else float(bid),
            None if ask is None else float(ask),
            None if mid is None else float(mid),
        )
    except Exception:
        return None, None, None


_QUOTE_DEBUG_RESET = "\033[0m"
_QUOTE_DEBUG_CYAN = "\033[36m"
_QUOTE_DEBUG_GREEN = "\033[32m"
_QUOTE_DEBUG_YELLOW = "\033[33m"
_QUOTE_DEBUG_RED = "\033[31m"


def _fmt_timestamp(ts_ns: Optional[int]) -> str:
    if not ts_ns:
        return "n/a"
    try:
        return datetime.fromtimestamp(ts_ns / 1_000_000_000).strftime("%H:%M:%S")
    except Exception:
        return "n/a"


def _quote_summary_line(sample: Dict[str, Any]) -> str:
    if not isinstance(sample, dict):
        return "n/a"
    ask = float(sample.get("ask_price") or 0.0)
    bid = float(sample.get("bid_price") or 0.0)
    ask_sz = int(sample.get("ask_size") or 0)
    bid_sz = int(sample.get("bid_size") or 0)
    mid = float(sample.get("mid_price") or ((ask + bid) / 2 if ask and bid else 0.0))
    spread = (ask - bid) if ask and bid else 0.0
    ts = PolygonAPIClient._sample_timestamp(sample)
    return f"[{_fmt_timestamp(ts)} mid={mid:.3f} spread={spread:.3f}] ask={ask:.3f}({ask_sz}) bid={bid:.3f}({bid_sz})"


def _log_pair_debug(
    label: str,
    front: Dict[str, Any],
    back: Dict[str, Any],
    delta_ns: int,
    *,
    color: str = _QUOTE_DEBUG_YELLOW,
    extra: str | None = None,
) -> None:
    extra_text = f"{extra} " if extra else ""
    if bool(getattr(get_settings(), "debug_polygon_quote", False)):
        print(
            f"{color}[QUOTE-DEBUG] {label} delta={delta_ns / 1_000_000_000:.3f}s "
            f"{extra_text}front={_quote_summary_line(front)} back={_quote_summary_line(back)}{_QUOTE_DEBUG_RESET}"
        )


def _sample_sizes(sample: Dict[str, Any]) -> tuple[float | None, float | None]:
    try:
        bid_sz = sample.get("bid_size")
        ask_sz = sample.get("ask_size")
        if bid_sz is None:
            bid_sz = sample.get("bidsize") or sample.get("bid_size_contracts")
        if ask_sz is None:
            ask_sz = sample.get("asksize") or sample.get("ask_size_contracts")
        return (
            None if bid_sz is None else float(bid_sz),
            None if ask_sz is None else float(ask_sz),
        )
    except Exception:
        return None, None


def _quote_liquidity_status_sample(sample: Dict[str, Any]) -> tuple[bool, bool]:
    settings = get_settings()
    limit = float(getattr(settings, "bid_ask_size_limit", 0.0) or 0.0)
    max_pct = float(getattr(settings, "premium_max_spread_pct", 0.0) or 0.0)
    bid, ask, mid = _sample_bid_ask(sample or {})
    if limit <= 0:
        limit = 0.0
    bid_sz, ask_sz = _sample_sizes(sample or {})
    spread_data = bid is not None and ask is not None
    size_data = bid_sz is not None and ask_sz is not None
    spread_ok = True
    if max_pct > 0:
        if not spread_data:
            spread_ok = False
        else:
            try:
                bid_val = float(bid)
                ask_val = float(ask)
                if ask_val <= bid_val:
                    spread_ok = False
                else:
                    mid_val = float(mid if mid is not None else (bid_val + ask_val) / 2.0)
                    if mid_val <= 0:
                        spread_ok = False
                    else:
                        spread_pct = (ask_val - bid_val) / mid_val
                        spread_ok = spread_pct <= max_pct
            except Exception:
                spread_ok = False
    size_ok = True
    if limit > 0:
        if not size_data:
            size_ok = False
        else:
            try:
                size_ok = float(bid_sz) >= limit and float(ask_sz) >= limit
            except Exception:
                size_ok = False
    ok = spread_ok and size_ok
    fail_with_data = ((not spread_ok) and spread_data) or ((not size_ok) and size_data)
    return ok, fail_with_data


def _sample_liquidity_ok(sample: Dict[str, Any]) -> bool:
    return _quote_liquidity_status_sample(sample)[0]


def _sample_liquidity_fail_hard(sample: Dict[str, Any]) -> bool:
    return _quote_liquidity_status_sample(sample)[1]


def extract_samples_from_quote(
    quote: Dict[str, Any],
    *,
    target_ns: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not isinstance(quote, dict):
        return samples
    raw_samples = quote.get("_samples")
    if isinstance(raw_samples, list):
        for item in raw_samples:
            if isinstance(item, dict):
                sample = dict(item)
                sample.setdefault("target_timestamp", target_ns)
                samples.append(sample)
    base = {
        k: v
        for k, v in quote.items()
        if isinstance(k, str) and not k.startswith("_") and k != "_samples"
    }
    if base:
        sample = dict(base)
        sample.setdefault("target_timestamp", target_ns)
        samples.append(sample)
    return samples


def filter_window_samples(
    samples: Iterable[Dict[str, Any]],
    *,
    premium_field: str,
    window_lo_ns: int,
    window_hi_ns: int,
    require_liquidity: bool = True,
) -> List[Dict[str, Any]]:
    windowed: List[Dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        ts = PolygonAPIClient._sample_timestamp(sample)
        if ts is None:
            continue
        if window_lo_ns and ts < window_lo_ns:
            continue
        if window_hi_ns and ts > window_hi_ns:
            continue
        price = _price_from_data(sample, premium_field)
        if price is None or price <= 0:
            continue
        liq_ok = _sample_liquidity_ok(sample)
        liq_fail_hard = _sample_liquidity_fail_hard(sample)
        if require_liquidity and not liq_ok:
            continue
        sample_copy = dict(sample)
        sample_copy.setdefault("_price", price)
        sample_copy["_liquidity_ok"] = liq_ok
        sample_copy["_liquidity_fail_hard"] = liq_fail_hard
        windowed.append(sample_copy)
    return windowed


def dedup_samples_by_timestamp(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[int, Dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        ts = PolygonAPIClient._sample_timestamp(sample)
        if ts is None or ts in dedup:
            continue
        dedup[ts] = dict(sample)
    return [dedup[k] for k in sorted(dedup.keys())]

def select_best_sample_pair(
    front_samples: List[Dict[str, Any]],
    back_samples: List[Dict[str, Any]],
    *,
    premium_field: str,
    target_ns: int,          # kept for signature compatibility; not used
    pair_delta_ns: int,
    prefer_liquidity: bool = False,  # kept for signature compatibility; not used
    debug_pairs: bool = False,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Select the front/back pair with the smallest timestamp delta, subject to:
      - both timestamps present,
      - both prices > 0 for the given premium_field,
      - |ts_front - ts_back| <= pair_delta_ns.

    Distance to target_ns and liquidity ranking are ignored.
    """
    best_pair: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    best_delta: Optional[int] = None
    want_debug = bool(debug_pairs or getattr(get_settings(), "debug_polygon_quote", False))

    for front in front_samples:
        ts_front = PolygonAPIClient._sample_timestamp(front)
        if ts_front is None:
            continue

        price_front = front.get("_price")
        if price_front is None:
            price_front = _price_from_data(front, premium_field)
        if price_front is None or price_front <= 0:
            continue

        for back in back_samples:
            ts_back = PolygonAPIClient._sample_timestamp(back)
            if ts_back is None:
                continue

            delta_pair = abs(ts_front - ts_back)
            if delta_pair > pair_delta_ns:
                if want_debug:
                    _log_pair_debug(
                        "delta_skip",
                        front,
                        back,
                        delta_pair,
                        color=_QUOTE_DEBUG_YELLOW,
                        extra="delta>limit",
                    )
                continue

            price_back = back.get("_price")
            if price_back is None:
                price_back = _price_from_data(back, premium_field)
            if price_back is None or price_back <= 0:
                continue

            # Candidate passes all criteria
            if want_debug:
                _log_pair_debug(
                    "candidate",
                    front,
                    back,
                    delta_pair,
                    color=_QUOTE_DEBUG_CYAN,
                    extra=f"delta={delta_pair}",
                )

            if best_delta is None or delta_pair < best_delta:
                best_delta = delta_pair
                best_pair = (dict(front), dict(back))
                if want_debug:
                    _log_pair_debug(
                        "best_update",
                        front,
                        back,
                        delta_pair,
                        color=_QUOTE_DEBUG_GREEN,
                        extra=f"delta={delta_pair}",
                    )
    return best_pair

async def ensure_premium(
    *,
    entry: Dict[str, Any],
    premium_field: str,
    client: PolygonAPIClient,
    option_type: str,
    underlying: str,
    as_of_str: str,
    debug: bool = False,
    force_refresh: bool = False,
    skip_write: bool = False,
) -> Optional[float]:
    if debug:
        print(
            "[CAL-UTILS] Ensuring premium for "
            f"{underlying} {option_type} as_of={as_of_str} field={premium_field}"
        )

    settings = get_settings()

    def _compute_window() -> tuple[Optional[int], Optional[int], Optional[int]]:
        try:
            if len(as_of_str) > 10:
                dt = datetime.strptime(as_of_str, "%Y-%m-%d %H:%M:%S")
            else:
                base_date = datetime.strptime(as_of_str, "%Y-%m-%d")
                hh, mm, ss = [int(x) for x in (settings.premium_time_target or "12:45:00").split(":")]
                dt = datetime.combine(base_date.date(), datetime.min.time()).replace(hour=hh, minute=mm, second=ss)
            win = max(1, int(getattr(settings, "premium_time_window_secs", 60)))
            target_ns_local = int(dt.timestamp() * 1_000_000_000)
            window = win * 1_000_000_000
            return target_ns_local, target_ns_local - window, target_ns_local + window
        except Exception:
            return None, None, None

    target_ns, window_lo_ns, window_hi_ns = _compute_window()

    def _timestamp_in_window(ts_val: Any) -> bool:
        try:
            ts_int = int(ts_val)
        except Exception:
            return False
        if window_lo_ns is None or window_hi_ns is None:
            return True
        return window_lo_ns <= ts_int <= window_hi_ns

    def _quote_timestamp(data: Dict[str, Any]) -> Optional[int]:
        ts_val = data.get("sip_timestamp")
        if ts_val is None and premium_field == "trade_price":
            ts_val = data.get("target_timestamp")
        try:
            return int(ts_val)
        except Exception:
            return None

    quote = entry.get("quote", {}) if isinstance(entry, dict) else {}
    samples_present = isinstance(quote.get("_samples"), list) and len(quote["_samples"]) > 0
    price = _price_from_data(quote, premium_field)
    if not samples_present:
        price = None
    if price is not None:
        # We accept the price even if outside window, consistent with fallback logic
        return price

    meta = entry.get("meta", {}) if isinstance(entry, dict) else {}
    strike = meta.get("strike_price")
    expiration = meta.get("expiration_date") or meta.get("expiration")
    option_ticker = meta.get("option_ticker")
    if strike is None or expiration is None or not option_ticker:
        if debug:
            print(
                "[CAL-UTILS] Missing metadata for premium fetch "
                f"(strike={strike}, expiration={expiration}, ticker={option_ticker})"
            )
        return None

    def _lookup_price_cache() -> Optional[Dict[str, Any]]:
        # Pull previously persisted Polygon quotes keyed by date/strike/expiry
        tkey = underlying.upper()
        by_date = stored_option_price.get(tkey, {})
        if not isinstance(by_date, dict):
            return None
        price_bucket = by_date.get(as_of_str[:10])
        if not isinstance(price_bucket, dict):
            return None
        try:
            strike_key = round(float(strike), 2)
        except Exception:
            return None
        strike_bucket = price_bucket.get(strike_key)
        if not isinstance(strike_bucket, dict):
            strike_bucket = price_bucket.get(f"{strike_key:.2f}")
        if not isinstance(strike_bucket, dict):
            strike_bucket = price_bucket.get(str(strike_key))
        if not isinstance(strike_bucket, dict):
            return None
        expiry_bucket = strike_bucket.get(str(expiration))
        if not isinstance(expiry_bucket, dict):
            return None
        keys = []
        if isinstance(option_type, str):
            keys.extend({option_type, option_type.lower(), option_type.upper()})
        for key in keys:
            node = expiry_bucket.get(key)
            if isinstance(node, dict) and node:
                return node
        return None

    def _select_best_cached_sample(
        quote: Dict[str, Any]
    ) -> tuple[Optional[Tuple[float, Dict[str, Any]]], bool]:
        """Return (price, payload) for the first liquidity-valid sample inside window and flag when candidates exist."""
        if not quote:
            return None, False
        candidates: List[Dict[str, Any]] = []
        core = {k: v for k, v in quote.items() if k != "_samples"}
        if core:
            candidates.append(dict(core))
        samples = quote.get("_samples")
        if isinstance(samples, list):
            for s in samples:
                if isinstance(s, dict):
                    candidates.append(dict(s))
        best_key: Optional[Tuple[int, int]] = None
        best_payload: Optional[Dict[str, Any]] = None
        best_price: Optional[float] = None
        candidates_checked = False
        
        # Track best absolute sample (ignoring window) as fallback
        fallback_key: Optional[Tuple[int, int]] = None
        fallback_payload: Optional[Dict[str, Any]] = None
        fallback_price: Optional[float] = None

        for payload in candidates:
            ts_val = _quote_timestamp(payload)
            price_val = _price_from_data(payload, premium_field)
            if price_val is None or price_val <= 0:
                continue
            candidates_checked = True
            liq_ok = _sample_liquidity_ok(payload)
            if not liq_ok:
                continue
            
            dist = abs((ts_val or 0) - (target_ns or 0)) if target_ns is not None else 0
            key: Tuple[int, int] = (dist, -(ts_val or 0))
            
            # Update fallback (best valid sample regardless of window)
            if fallback_key is None or key < fallback_key:
                fallback_key = key
                fallback_payload = payload
                fallback_price = price_val

            if ts_val is not None and not _timestamp_in_window(ts_val):
                continue

            if best_key is None or key < best_key:
                best_key = key
                best_payload = payload
                best_price = price_val
                
        if best_payload and best_price is not None:
            return (best_price, best_payload), True
            
        # Fallback: if we have a valid sample outside the window, use it
        # This matches poly_client behavior which returns the closest sample
        if fallback_payload and fallback_price is not None:
             return (fallback_price, fallback_payload), True
             
        return None, candidates_checked

    cached_leaf = None
    if not force_refresh:
        cached_leaf = _lookup_price_cache()
    if cached_leaf:
        invalid_targets = cached_leaf.get("_invalid_targets")
        if (
            isinstance(invalid_targets, list)
            and target_ns is not None
            and any(int(t or 0) == int(target_ns) for t in invalid_targets)
        ):
            if debug:
                print(
                    f"[CAL-UTILS] invalid cache hit for {option_ticker} "
                    f"(target={target_ns}); skipping Polygon fetch"
                )
            return None
        cached_quote = {
            k: v for k, v in cached_leaf.items() if k != "_invalid_targets"
        }
        
        # Check if cached data has validation failure marker
        validation_failed = cached_quote.get("_validation_failed")
        if validation_failed:
            # Print red alert message
            print(
                f"\033[91m[ALERT] Cached data for {option_ticker} failed validation check: {validation_failed}. "
                f"Skipping API fetch (data unusable for trading).\033[0m"
            )
            return None
        
        best_entry, had_candidates = _select_best_cached_sample(cached_quote)
        if best_entry:
            price_candidate, payload = best_entry
            quote_dest = entry.setdefault("quote", {})
            quote_dest.clear()
            quote_dest.update(payload)
            quote_dest["_cache_source"] = "stored_price"
            if isinstance(cached_quote.get("_samples"), list):
                quote_dest["_samples"] = [
                    dict(s) for s in cached_quote["_samples"] if isinstance(s, dict)
                ]
            if debug:
                print(f"[CAL-UTILS] price cache hit {option_ticker} price={price_candidate}")
            return price_candidate
        if had_candidates:
            if target_ns is not None:
                invalid_targets = cached_leaf.setdefault("_invalid_targets", [])
                if isinstance(invalid_targets, list):
                    invalid_targets.append(int(target_ns))
            if debug:
                print(
                    f"[CAL-UTILS] cached price for {option_ticker} "
                    "failed liquidity; refetching"
                )
        if debug and not had_candidates:
            print(f"[CAL-UTILS] cached price for {option_ticker} outside window; refetching")

    if not force_refresh:
        try:
            if len(as_of_str) > 10:
                dt = datetime.strptime(as_of_str, "%Y-%m-%d %H:%M:%S")
            else:
                base_date = datetime.strptime(as_of_str, "%Y-%m-%d")
                hh, mm, ss = [int(x) for x in (settings.premium_time_target or "12:45:00").split(":")]
                dt = datetime.combine(base_date.date(), datetime.min.time()).replace(hour=hh, minute=mm, second=ss)
            win = max(1, int(getattr(settings, "premium_time_window_secs", 60)))
            target_ns = int(dt.timestamp() * 1_000_000_000)
            lo_ns = target_ns - win * 1_000_000_000
            hi_ns = target_ns + win * 1_000_000_000

            tkey = underlying.upper()
            chain_cache = stored_option_chain.setdefault(tkey, {})
            exp_cache = chain_cache.setdefault(expiration, {})
            as_of_bucket = exp_cache.setdefault(as_of_str[:10], {})
            cp_keys = []
            if isinstance(option_type, str):
                cp_keys = [option_type, option_type.lower()]
            else:
                cp_keys = [option_type]
            for call_put in cp_keys:
                leaf = as_of_bucket.get(call_put)
                if isinstance(leaf, dict) and leaf:
                    price_candidate = _price_from_data(leaf, premium_field)
                    ts = leaf.get("sip_timestamp")
                    if ts is None and premium_field == "trade_price":
                        ts = leaf.get("target_timestamp")
                    if (
                        price_candidate is not None
                        and isinstance(ts, (int, float))
                        and lo_ns <= int(ts) <= hi_ns
                    ):
                        entry.setdefault("quote", {}).update(leaf)
                        entry["quote"]["_cache_source"] = "chain_cache"
                        if debug:
                            print(f"[CAL-UTILS] cache hit {option_ticker} price={price_candidate}")
                        return price_candidate
        except Exception as exc:
            if debug:
                print(f"[CAL-UTILS] cache lookup failed: {exc}")

    req = [
        {
            "strike_price": strike,
            "call_put": option_type,
            "expiration_date": expiration,
            "quote_timestamp": as_of_str,
            "option_ticker": option_ticker,
        }
    ]
    if debug:
        print(
            f"[CAL-UTILS] Fetching premium from API for {option_ticker} "
            f"(strike={strike}, exp={expiration}, type={option_type})"
        )
    fetched = await client.get_option_prices_batch_async(underlying, req, skip_write=skip_write)
    payload = fetched[0] if fetched else {}
    price = _price_from_data(payload, premium_field)
    if price is not None:
        entry.setdefault("quote", {}).update(payload)
        return price
    if debug:
        print(f"[CAL-UTILS] API fetch returned no price for {option_ticker}")
    return None


@dataclass
class CalendarPairResult:
    front_date: date
    back_date: date
    strike_front: float
    strike_back: float
    entry_front: Dict[str, Any]
    entry_back: Dict[str, Any]
    premium_front: float
    premium_back: float
    used_ts_front: Optional[int]
    used_ts_back: Optional[int]
    chosen_front: Dict[str, Any]
    chosen_back: Dict[str, Any]
    target_dt: datetime
    target_ns: int
    window_lo_ns: int
    window_hi_ns: int
    front_samples: List[Dict[str, Any]]
    back_samples: List[Dict[str, Any]]
    timings: Dict[str, Any] = field(default_factory=dict)


async def _gather_calendar_pair_internal(
    *,
    ticker: str,
    as_of_date: date,
    spot: float,
    front_target: int,
    back_target: int,
    weekday: str,
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str = "call",
    debug: bool = False,
    strike_lower_override: Optional[float] = None,
    strike_upper_override: Optional[float] = None,
) -> Tuple[Optional[CalendarPairResult], Optional[str]]:
    if spot is None:
        return None, "no_spot"

    gather_total_start = _time.perf_counter()
    gather_timings: Dict[str, Any] = {"attempts": []}

    def _record_timing(label: str, start_ts: float) -> None:
        try:
            gather_timings[label] = (_time.perf_counter() - start_ts) * 1000.0
        except Exception:
            pass

    expiries = _calendar_expiry_candidates(
        as_of=as_of_date,
        weekday=weekday,
        front_target=front_target,
        back_target=back_target,
        ticker=ticker,
    )
    if not expiries:
        if debug:
            print(f"[CAL-UTILS] No calendar expiries for {ticker} on {as_of_date}")
        return None, "no_expiries"
    front_date, back_date = expiries
    gather_timings["expiry_candidates"] = [d.isoformat() for d in expiries if isinstance(d, date)]

    settings = get_settings()
    expiry_window_days = max(1, int(getattr(settings, "premium_expiry_window_days", 20)))
    strike_band_pct = max(0.0, float(getattr(settings, "premium_strike_spot_pct", 0.02)))
    spot_val = float(spot)
    if strike_lower_override is not None and strike_upper_override is not None:
        strike_lower = max(0.0, float(strike_lower_override))
        strike_upper = max(strike_lower, float(strike_upper_override))
    else:
        strike_lower = max(0.0, spot_val * (1.0 - strike_band_pct))
        strike_upper = spot_val * (1.0 + strike_band_pct)

    if strike_upper - strike_lower <= 1e-6:
        wiggle = max(0.05, strike_lower * 0.001 if strike_lower else 0.05)
        strike_lower = max(0.0, strike_lower - wiggle)
        strike_upper = strike_upper + wiggle

    tkey = ticker.upper()
    scache = stored_option_chain.setdefault(tkey, {})

    async def _first_expiry_with_contracts(target_date: date, *, min_date: date | None = None) -> Tuple[date | None, Dict[float, str]]:
        as_of_str = as_of_date.strftime("%Y-%m-%d")
        exp_s = target_date.strftime("%Y-%m-%d")
        exp_bucket = scache.setdefault(exp_s, {})
        as_bucket = exp_bucket.setdefault(as_of_str, {})
        as_bucket.pop("_alias_to", None)

        def _cached_syms(exp_str: str) -> Dict[float, str]:
            return dict(
                scache.get(exp_str, {})
                .get(as_of_str, {})
                .get(option_type, {})
                or {}
            )

        cached_syms = _cached_syms(exp_s)
        if cached_syms:
            as_bucket.pop("_fallback_exp", None)
            as_bucket.pop("_empty", None)
            return target_date, cached_syms

        fallback_exp = as_bucket.get("_fallback_exp")
        if fallback_exp:
            fb_syms = _cached_syms(fallback_exp)
            if fb_syms:
                try:
                    fb_date = datetime.strptime(fallback_exp, "%Y-%m-%d").date()
                except Exception:
                    fb_date = None
                if fb_date:
                    return fb_date, fb_syms

        if as_bucket.get("_empty"):
            return None, {}

        rng_start_dt = target_date - timedelta(days=expiry_window_days)
        rng_end_dt = target_date + timedelta(days=expiry_window_days)
        rng_start = rng_start_dt.strftime("%Y-%m-%d")
        rng_end = rng_end_dt.strftime("%Y-%m-%d")
        try:
            by_exp = await client.get_option_contracts_in_range(
                ticker,
                call_put=option_type,
                as_of=as_of_str,
                exp_start=rng_start,
                exp_end=rng_end,
                strike_price_gte=strike_lower,
                strike_price_lte=strike_upper,
            )
        except Exception:
            by_exp = {}

        for exp_key, mapping in by_exp.items():
            exp_cache = scache.setdefault(str(exp_key), {}).setdefault(as_of_str, {})
            if mapping:
                exp_cache[option_type] = dict(mapping)
                exp_cache.pop("_empty", None)
            else:
                exp_cache.setdefault("_empty", True)

        best_date: Optional[date] = None
        best_key: Optional[Tuple[int, int, date]] = None
        chosen_syms: Dict[float, str] = {}
        for exp_key, mapping in by_exp.items():
            if not mapping:
                continue
            try:
                d = datetime.strptime(str(exp_key), "%Y-%m-%d").date()
            except Exception:
                continue
            if min_date and d <= min_date:
                continue
            diff = abs((d - target_date).days)
            ahead = 0 if d >= target_date else 1
            key = (diff, ahead, d)
            if best_key is None or key < best_key:
                best_key = key
                best_date = d
                chosen_syms = dict(mapping)

        if best_date:
            as_bucket["_fallback_exp"] = best_date.strftime("%Y-%m-%d")
            as_bucket.pop("_empty", None)
            return best_date, chosen_syms

        as_bucket["_empty"] = True
        return None, {}

    front_contract_start = _time.perf_counter()
    gather_timings["front_contracts_offset_ms"] = (
        (front_contract_start - gather_total_start) * 1000.0
    )
    front_date_actual, calls_front_syms = await _first_expiry_with_contracts(front_date)
    _record_timing("front_contracts_ms", front_contract_start)
    if front_date_actual is None or not calls_front_syms:
        if debug:
            print(f"[CAL-UTILS] No contracts for front expiry {front_date} ({ticker} {as_of_date})")
        return None, "no_front_contracts"
    back_contract_start = _time.perf_counter()
    gather_timings["back_contracts_offset_ms"] = (
        (back_contract_start - gather_total_start) * 1000.0
    )
    back_date_actual, calls_back_syms = await _first_expiry_with_contracts(
        back_date, min_date=front_date_actual
    )
    _record_timing("back_contracts_ms", back_contract_start)
    if back_date_actual is None or not calls_back_syms:
        if debug:
            print(f"[CAL-UTILS] No contracts for back expiry {back_date} ({ticker} {as_of_date})")
        return None, "no_back_contracts"

    def _collect_candidates(mapping: Dict[float, str]) -> List[float]:
        out: List[float] = []
        for k in mapping.keys():
            try:
                out.append(float(k))
            except Exception:
                continue
        return sorted(out)

    front_candidates = _collect_candidates(calls_front_syms)
    back_candidates = _collect_candidates(calls_back_syms)
    gather_timings["front_strike_candidates"] = len(front_candidates)
    gather_timings["back_strike_candidates"] = len(back_candidates)

    if not front_candidates or not back_candidates:
        if debug:
            print(f"[CAL-UTILS] Empty strike candidates for {ticker} on {as_of_date}")
        return None, "no_atm_strike"

    cents_front = [int(round(k * 100)) for k in front_candidates]
    cents_back = [int(round(k * 100)) for k in back_candidates]
    common = set(cents_front).intersection(cents_back)

    def _find_key_by_cents(mapping: Dict[float, str], cents: int) -> Optional[float]:
        best_k = None
        best_d = None
        for k in mapping.keys():
            try:
                c = int(round(float(k) * 100))
                d = abs(c - cents)
                if best_d is None or d < best_d:
                    best_d = d
                    best_k = float(k)
                    if d == 0:
                        break
            except Exception:
                continue
        return best_k

    def _strike_pair_candidates(max_pairs: int = 25) -> List[Tuple[float, float]]:
        pairs: List[Tuple[float, float]] = []
        seen: set[Tuple[float, float]] = set()
        spot_val = float(spot)
        if common:
            common_sorted = sorted(common, key=lambda c: abs((c / 100.0) - spot_val))
            for cents in common_sorted:
                f_val = _find_key_by_cents(calls_front_syms, cents)
                b_val = _find_key_by_cents(calls_back_syms, cents)
                if f_val is None or b_val is None:
                    continue
                key = (float(f_val), float(b_val))
                if key in seen:
                    continue
                pairs.append(key)
                seen.add(key)
                if len(pairs) >= max_pairs:
                    return pairs
        front_sorted = sorted(front_candidates, key=lambda v: abs(float(v) - spot_val))
        back_sorted = sorted(back_candidates, key=lambda v: abs(float(v) - spot_val))
        for f in front_sorted:
            for b in back_sorted:
                key = (float(f), float(b))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    strike_pair_start = _time.perf_counter()
    gather_timings["strike_pair_offset_ms"] = (
        (strike_pair_start - gather_total_start) * 1000.0
    )
    strike_pairs = _strike_pair_candidates()
    _record_timing("strike_pair_build_ms", strike_pair_start)
    gather_timings["strike_pair_count"] = len(strike_pairs)
    if not strike_pairs:
        if debug:
            print(f"[CAL-UTILS] No strike pair candidates for {ticker} on {as_of_date}")
        return None, "no_strike_pair"

    window_secs = max(1, int(getattr(settings, "premium_time_window_secs", 60)))
    target_time_s = getattr(settings, "premium_time_target", "12:30:00") or "12:30:00"
    pair_delta_secs = int(getattr(settings, "premium_pair_delta_secs", 60))
    strict_window_secs = int(getattr(settings, "premium_strict_window_secs", 0))
    backoff_minutes = max(1, int(getattr(settings, "premium_time_backoff_minutes", 30)))
    backoff_steps = max(0, int(getattr(settings, "premium_time_backoff_steps", 3)))
    try:
        th, tm, ts = [int(x) for x in target_time_s.split(":")]
    except Exception:
        th, tm, ts = 12, 30, 0
    base_dt = datetime.combine(as_of_date, datetime.min.time()).replace(hour=th, minute=tm, second=ts)
    pair_delta_ns = pair_delta_secs * 1_000_000_000
    
    # If strict window is enabled, only attempt target time once (no backoff retry)
    if strict_window_secs > 0:
        backoff_offsets = [0]
    else:
        backoff_offsets = [0] + [-(i * backoff_minutes * 60) for i in range(1, backoff_steps + 1)]

    async def _attempt_pair(
        target_dt: datetime,
        offset_index: int,
        entry_front_meta: Dict[str, Any],
        entry_back_meta: Dict[str, Any],
    ) -> Optional[CalendarPairResult]:
        target_str_local = target_dt.strftime("%Y-%m-%d %H:%M:%S")
        target_ns_local = int(target_dt.timestamp() * 1_000_000_000)
        window_lo_ns_local = target_ns_local - window_secs * 1_000_000_000
        window_hi_ns_local = target_ns_local + window_secs * 1_000_000_000

        entry_front_attempt = {"meta": dict(entry_front_meta)}
        entry_back_attempt = {"meta": dict(entry_back_meta)}

        attempt_info: Dict[str, Any] = {
            "offset_index": offset_index,
            "target_timestamp": target_ns_local,
            "result": "pending",
        }
        try:
            attempt_info["strike_front"] = float(entry_front_meta.get("strike_price", 0.0))
        except Exception:
            pass
        try:
            attempt_info["strike_back"] = float(entry_back_meta.get("strike_price", 0.0))
        except Exception:
            pass
        attempts_list = gather_timings.setdefault("attempts", [])
        attempts_list.append(attempt_info)
        attempt_info["attempt_index"] = len(attempts_list) - 1

        def _mark_invalid(meta: Dict[str, Any]) -> None:
            try:
                client._write_invalid_option(
                    ticker=meta.get("option_ticker") or ticker,
                    strike_price=float(meta.get("strike_price", 0)),
                    call_put=option_type,
                    expiration_date=str(meta.get("expiration_date")),
                    pricing_date=target_str_local,
                    premium_field=premium_field,
                    target_timestamp=target_ns_local,
                )
            except Exception:
                pass

        def _log_quote_failure(label: str, option_ticker: str | None, quote: Dict[str, Any]) -> None:
            if not isinstance(quote, dict):
                return
            ticker_label = option_ticker or ticker
            label_text = label or ""
            if getattr(client, "_debug_quote_poll", False):
                try:
                    client._debug_print_quote(
                        option_ticker=ticker_label,
                        target_ns=target_ns_local,
                        window_lo_ns=window_lo_ns_local,
                        window_hi_ns=window_hi_ns_local,
                        payload=quote,
                        liquidity_ok=False,
                        label=label_text,
                    )
                except Exception:
                    pass
            elif debug:
                label_part = f"{label_text} " if label_text else ""
                print(
                    f"{_QUOTE_DEBUG_RED}[QUOTE-DEBUG] {label_part}{ticker_label} failed liquidity "
                    f"target={target_str_local} | {_quote_summary_line(quote)}{_QUOTE_DEBUG_RESET}"
                )
                breakpoint()

        attempt_start = _time.perf_counter()
        attempt_info["attempt_start_offset_ms"] = (
            (attempt_start - gather_total_start) * 1000.0
        )
        if debug:
            print(f"[CAL-UTILS] Fetch premiums for {ticker} target={target_str_local} offset_index={offset_index}")
        ensure_parallel_start = _time.perf_counter()
        await asyncio.gather(
            ensure_premium(
                entry=entry_front_attempt,
                premium_field=premium_field,
                client=client,
                option_type=option_type,
                underlying=ticker,
                as_of_str=target_str_local,
                debug=debug,
                skip_write=True,
            ),
            ensure_premium(
                entry=entry_back_attempt,
                premium_field=premium_field,
                client=client,
                option_type=option_type,
                underlying=ticker,
                as_of_str=target_str_local,
                debug=debug,
                skip_write=True,
            ),
        )
        attempt_info["ensure_parallel_ms"] = (_time.perf_counter() - ensure_parallel_start) * 1000.0
        
        front_ok, front_fail_hard = _quote_liquidity_status_sample(
            entry_front_attempt.get("quote", {}) if isinstance(entry_front_attempt, dict) else {}
        )
        attempt_info["front_liquidity_ok"] = front_ok
        attempt_info["front_liquidity_fail_hard"] = front_fail_hard
        if not front_ok:
            if "liquidity_rejected" not in gather_timings:
                gather_timings["liquidity_rejected"] = "front"
            _log_quote_failure("front", entry_front_meta.get("option_ticker"), entry_front_attempt.get("quote", {}))
            
            # Write the data to cache with liquidity failure marker
            q_front = entry_front_attempt.get("quote", {})
            if q_front and not q_front.get("_cache_source"):
                q_front["_validation_failed"] = "liquidity"
                try:
                    strike_price = float(entry_front_meta["strike_price"])
                    expiration_date = str(entry_front_meta["expiration_date"])
                    ts = q_front.get("sip_timestamp")
                    write_date = target_str_local
                    if ts:
                        try:
                            write_date = datetime.fromtimestamp(ts / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
                    client._write_option_payload(
                        ticker, strike_price, option_type, expiration_date, write_date,
                        dict(q_front), q_front.get("_samples", [])
                    )
                except Exception:
                    pass
            
        back_ok, back_fail_hard = _quote_liquidity_status_sample(
            entry_back_attempt.get("quote", {}) if isinstance(entry_back_attempt, dict) else {}
        )
        attempt_info["back_liquidity_ok"] = back_ok
        attempt_info["back_liquidity_fail_hard"] = back_fail_hard
        if not back_ok:
            if "liquidity_rejected" not in gather_timings:
                gather_timings["liquidity_rejected"] = "back"
            _log_quote_failure("back", entry_back_meta.get("option_ticker"), entry_back_attempt.get("quote", {}))
            
            # Write the data to cache with liquidity failure marker
            q_back = entry_back_attempt.get("quote", {})
            if q_back and not q_back.get("_cache_source"):
                q_back["_validation_failed"] = "liquidity"
                try:
                    strike_price = float(entry_back_meta["strike_price"])
                    expiration_date = str(entry_back_meta["expiration_date"])
                    ts = q_back.get("sip_timestamp")
                    write_date = target_str_local
                    if ts:
                        try:
                            write_date = datetime.fromtimestamp(ts / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
                    client._write_option_payload(
                        ticker, strike_price, option_type, expiration_date, write_date,
                        dict(q_back), q_back.get("_samples", [])
                    )
                except Exception:
                    pass

        q_front_attempt = entry_front_attempt.get("quote", {}) if isinstance(entry_front_attempt, dict) else {}
        q_back_attempt = entry_back_attempt.get("quote", {}) if isinstance(entry_back_attempt, dict) else {}

        front_sample_start = _time.perf_counter()
        front_samples_attempt = dedup_samples_by_timestamp(
            filter_window_samples(
                extract_samples_from_quote(q_front_attempt, target_ns=target_ns_local),
                premium_field=premium_field,
                window_lo_ns=window_lo_ns_local,
                window_hi_ns=window_hi_ns_local,
            )
        )
        attempt_info["front_sample_ms"] = (_time.perf_counter() - front_sample_start) * 1000.0
        attempt_info["front_sample_count"] = len(front_samples_attempt)
        back_sample_start = _time.perf_counter()
        back_samples_attempt = dedup_samples_by_timestamp(
            filter_window_samples(
                extract_samples_from_quote(q_back_attempt, target_ns=target_ns_local),
                premium_field=premium_field,
                window_lo_ns=window_lo_ns_local,
                window_hi_ns=window_hi_ns_local,
            )
        )
        attempt_info["back_sample_ms"] = (_time.perf_counter() - back_sample_start) * 1000.0
        attempt_info["back_sample_count"] = len(back_samples_attempt)

        select_start = _time.perf_counter()
        debug_pairs = bool(settings.debug_polygon_quote or debug)
        best_pair_local = select_best_sample_pair(
            front_samples_attempt,
            back_samples_attempt,
            premium_field=premium_field,
            target_ns=target_ns_local,
            pair_delta_ns=pair_delta_ns,
            prefer_liquidity=True,
            debug_pairs=debug_pairs,
        )
        attempt_info["select_pair_ms"] = (_time.perf_counter() - select_start) * 1000.0
        if not best_pair_local:
            if debug and (not front_ok or not back_ok):
                print(
                    f"{_QUOTE_DEBUG_RED}[QUOTE-DEBUG] liquidity retry target={target_str_local} "
                    f"front={_quote_summary_line(q_front_attempt)} "
                    f"back={_quote_summary_line(q_back_attempt)}{_QUOTE_DEBUG_RESET}"
                )
            attempt_info["result"] = "no_pair"
            _mark_invalid(entry_front_meta)
            _mark_invalid(entry_back_meta)
            return None

        chosen_front_local, chosen_back_local = best_pair_local
        premium_front_local = chosen_front_local.get("_price") or _price_from_data(chosen_front_local, premium_field)
        premium_back_local = chosen_back_local.get("_price") or _price_from_data(chosen_back_local, premium_field)
        if premium_front_local is None or premium_back_local is None:
            attempt_info["result"] = "no_price"
            _mark_invalid(entry_front_meta)
            _mark_invalid(entry_back_meta)
            return None

        # Strict window enforcement for position opening
        if strict_window_secs > 0:
            front_ts = PolygonAPIClient._sample_timestamp(chosen_front_local)
            back_ts = PolygonAPIClient._sample_timestamp(chosen_back_local)
            strict_window_ns = strict_window_secs * 1_000_000_000
            
            front_delta = abs(front_ts - target_ns_local) if front_ts else float('inf')
            back_delta = abs(back_ts - target_ns_local) if back_ts else float('inf')
            
            if front_delta > strict_window_ns or back_delta > strict_window_ns:
                # Data is outside strict window - cache it but don't open position
                attempt_info["result"] = "strict_window_reject"
                attempt_info["front_delta_secs"] = front_delta / 1_000_000_000
                attempt_info["back_delta_secs"] = back_delta / 1_000_000_000
                
                # Cache the data for potential use in closing positions
                entry_front = dict(entry_front_attempt)
                entry_back = dict(entry_back_attempt)
                
                def _prepare_store_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    out: List[Dict[str, Any]] = []
                    for s in samples:
                        if not isinstance(s, dict):
                            continue
                        copy = dict(s)
                        copy.pop("_price", None)
                        copy.setdefault("target_timestamp", target_ns_local)
                        out.append(copy)
                    return out

                for entry, sample, samples_all in (
                    (entry_front, chosen_front_local, front_samples_attempt),
                    (entry_back, chosen_back_local, back_samples_attempt),
                ):
                    quote_dest = entry.setdefault("quote", {}) if isinstance(entry, dict) else {}
                    cleaned_sample = dict(sample)
                    cleaned_sample.pop("_price", None)
                    cleaned_sample["_validation_failed"] = "strict_window"  # Mark as failed strict window check
                    quote_dest.update(cleaned_sample)
                    quote_dest["_samples"] = _prepare_store_samples(samples_all)
                    try:
                        strike_price = float(entry["meta"]["strike_price"])
                        expiration_date = str(entry["meta"]["expiration_date"])
                        ts = cleaned_sample.get("sip_timestamp")
                        write_date = target_str_local
                        if ts:
                            try:
                                write_date = datetime.fromtimestamp(ts / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
                            except Exception:
                                pass

                        client._write_option_payload(
                            ticker,
                            strike_price,
                            option_type,
                            expiration_date,
                            write_date,
                            dict(cleaned_sample),
                            quote_dest["_samples"],
                        )
                    except Exception:
                        pass
                
                if debug:
                    print(
                        f"{_QUOTE_DEBUG_RED}[CAL-UTILS] Strict window reject: "
                        f"front_delta={front_delta/1_000_000_000:.1f}s, "
                        f"back_delta={back_delta/1_000_000_000:.1f}s, "
                        f"max={strict_window_secs}s. Data cached but position opening rejected.{_QUOTE_DEBUG_RESET}"
                    )
                
                return None

        attempt_info["result"] = "success"
        attempt_info["success_ts_front"] = PolygonAPIClient._sample_timestamp(chosen_front_local)
        attempt_info["success_ts_back"] = PolygonAPIClient._sample_timestamp(chosen_back_local)
        try:
            attempt_info["premium_front"] = float(premium_front_local)
        except Exception:
            pass
        try:
            attempt_info["premium_back"] = float(premium_back_local)
        except Exception:
            pass

        entry_front = dict(entry_front_attempt)
        entry_back = dict(entry_back_attempt)

        def _prepare_store_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for s in samples:
                if not isinstance(s, dict):
                    continue
                copy = dict(s)
                copy.pop("_price", None)
                copy.setdefault("target_timestamp", target_ns_local)
                out.append(copy)
            return out

        for entry, sample, samples_all in (
            (entry_front, chosen_front_local, front_samples_attempt),
            (entry_back, chosen_back_local, back_samples_attempt),
        ):
            quote_dest = entry.setdefault("quote", {}) if isinstance(entry, dict) else {}
            cleaned_sample = dict(sample)
            cleaned_sample.pop("_price", None)
            quote_dest.update(cleaned_sample)
            quote_dest["_samples"] = _prepare_store_samples(samples_all)
            try:
                strike_price = float(entry["meta"]["strike_price"])
                expiration_date = str(entry["meta"]["expiration_date"])
                ts = cleaned_sample.get("sip_timestamp")
                write_date = target_str_local
                if ts:
                    try:
                        write_date = datetime.fromtimestamp(ts / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass

                client._write_option_payload(
                    ticker,
                    strike_price,
                    option_type,
                    expiration_date,
                    write_date,
                    dict(cleaned_sample),
                    quote_dest["_samples"],
                )
                
                def _fmt_ts(ns_val: Any) -> str:
                    try:
                        return datetime.fromtimestamp(int(ns_val) / 1_000_000_000).strftime("%H:%M:%S")
                    except Exception:
                        return str(ns_val)

                details = {
                    "ask_price": cleaned_sample.get("ask_price"),
                    "bid_price": cleaned_sample.get("bid_price"),
                    "ask_size": cleaned_sample.get("ask_size"),
                    "bid_size": cleaned_sample.get("bid_size"),
                    "mid_price": cleaned_sample.get("mid_price"),
                    "sip_timestamp": _fmt_ts(cleaned_sample.get("sip_timestamp")),
                    "target_timestamp": _fmt_ts(target_ns_local),
                    "_samples": f"{len(quote_dest.get('_samples', []))} samples",
                }
                if not quote_dest.get("_cache_source"):
                    print(
                        f"Stored {ticker},Strike:{strike_price},{option_type},Expire:{expiration_date}, "
                        f"Pricing:{target_str_local}:{details}"
                    )
            except Exception:
                pass

        gather_timings["success_attempt_index"] = attempt_info["attempt_index"]
        gather_timings["success_offset_index"] = offset_index
        gather_timings["success_target"] = target_str_local
        gather_timings["attempt_count"] = len(gather_timings.get("attempts", []))
        gather_timings["total_ms"] = (_time.perf_counter() - gather_total_start) * 1000.0

        return CalendarPairResult(
            front_date=front_date_actual,
            back_date=back_date_actual,
            strike_front=float(entry_front_meta["strike_price"]),
            strike_back=float(entry_back_meta["strike_price"]),
            entry_front=entry_front,
            entry_back=entry_back,
            premium_front=float(premium_front_local),
            premium_back=float(premium_back_local),
            used_ts_front=PolygonAPIClient._sample_timestamp(chosen_front_local),
            used_ts_back=PolygonAPIClient._sample_timestamp(chosen_back_local),
            chosen_front=dict(chosen_front_local),
            chosen_back=dict(chosen_back_local),
            target_dt=target_dt,
            target_ns=target_ns_local,
            window_lo_ns=window_lo_ns_local,
            window_hi_ns=window_hi_ns_local,
            front_samples=front_samples_attempt,
            back_samples=back_samples_attempt,
            timings=gather_timings,
        )

    for strike_front, strike_back in strike_pairs:
        entry_front_meta = {
            "strike_price": float(strike_front),
            "expiration_date": front_date_actual.strftime("%Y-%m-%d"),
            "option_ticker": calls_front_syms.get(float(strike_front)),
        }
        entry_back_meta = {
            "strike_price": float(strike_back),
            "expiration_date": back_date_actual.strftime("%Y-%m-%d"),
            "option_ticker": calls_back_syms.get(float(strike_back)),
        }
        if not entry_front_meta["option_ticker"] or not entry_back_meta["option_ticker"]:
            continue
        for idx, offset in enumerate(backoff_offsets):
            attempt_dt = base_dt + timedelta(seconds=offset)
            result = await _attempt_pair(attempt_dt, idx, entry_front_meta, entry_back_meta)
            if result is not None:
                return result, None

    if debug:
        print(
            f"[CAL-UTILS] Failed to align premiums for {ticker} on {as_of_date} "
            f"(front={front_date_actual}, back={back_date_actual})"
        )
    print(f"\033[91m[CAL-UTILS] Failed to find valid pair for {ticker} on {as_of_date}\033[0m")
    return None, "no_premium_pair"


async def gather_calendar_pair(
    *,
    ticker: str,
    as_of_date: date,
    spot: float,
    front_target: int,
    back_target: int,
    weekday: str,
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str = "call",
    debug: bool = False,
) -> Tuple[Optional[CalendarPairResult], Optional[str]]:
    return await _gather_calendar_pair_internal(
        ticker=ticker,
        as_of_date=as_of_date,
        spot=spot,
        front_target=front_target,
        back_target=back_target,
        weekday=weekday,
        client=client,
        premium_field=premium_field,
        option_type=option_type,
        debug=debug,
    )


__all__ = [
    "nearest_strike",
    "calc_implied_vol",
    "forward_factor",
    "extract_samples_from_quote",
    "filter_window_samples",
    "dedup_samples_by_timestamp",
    "select_best_sample_pair",
    "ensure_premium",
    "CalendarPairResult",
    "gather_calendar_pair",
]
