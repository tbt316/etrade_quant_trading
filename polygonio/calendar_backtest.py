from __future__ import annotations

import asyncio
import time as _time
from itertools import cycle
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"

from .calendar_utils import (
    CalendarPairResult,
    calc_implied_vol,
    ensure_premium,
    forward_factor,
    gather_calendar_pair,
    extract_samples_from_quote,
    filter_window_samples,
    dedup_samples_by_timestamp,
    select_best_sample_pair,
)
from .cache_io import (
    cache_risk_free_rate,
    cache_risk_free_rates,
    get_cached_risk_free_rate,
    load_stored_option_data,
    option_data_unsaved_count,
    save_stored_option_data,
)
from .config import get_settings, resolve_premium_field
from .poly_client import PolygonAPIClient
from .prices import get_historical_prices


def _extract_bid_ask(entry: dict) -> tuple[float | None, float | None, float | None]:
    try:
        q = (entry or {}).get("quote") or {}
        bid = q.get("bid_price")
        ask = q.get("ask_price")
        if bid is None:
            bid = q.get("bid")
        if ask is None:
            ask = q.get("ask")
        mid = q.get("mid_price")
        if mid is None and bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0
        return (
            (None if bid is None else float(bid)),
            (None if ask is None else float(ask)),
            (None if mid is None else float(mid)),
        )
    except Exception:
        return None, None, None


def _quote_liquidity_status(entry: dict | None) -> tuple[bool, bool]:
    """Return (ok, fail_with_data) for spread/size checks."""
    settings = get_settings()
    max_pct = float(getattr(settings, "premium_max_spread_pct", 0.0) or 0.0)
    bid, ask, mid = _extract_bid_ask(entry or {})
    limit = float(getattr(settings, "bid_ask_size_limit", 0.0) or 0.0)
    bid_sz, ask_sz = _extract_sizes(entry or {})
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
            # Missing size data should not block pricing; treat as pass so we can
            # still evaluate the position rather than dropping the quotes entirely.
            size_ok = True
        else:
            try:
                size_ok = float(bid_sz) >= limit and float(ask_sz) >= limit
            except Exception:
                size_ok = False
    ok = spread_ok and size_ok
    fail_with_data = ((not spread_ok) and spread_data) or ((not size_ok) and size_data)
    return ok, fail_with_data


def _quote_liquidity_ok(entry: dict | None) -> bool:
    return _quote_liquidity_status(entry)[0]


def _exec_price_from_entry(entry: dict, *, side: str, fallback_price: float) -> float:
    """Return steered execution price using bid/ask when available.

    side: 'buy' or 'sell'
    steer in settings: 0.5 = mid; <0.5 worsens fills (buy→ask, sell→bid).
    Mapping: w_buy = 1 - steer; w_sell = steer. price = bid + w*(ask - bid)
    """
    settings = get_settings()
    steer = float(getattr(settings, "execution_steer", 0.5) or 0.5)
    steer = max(0.0, min(1.0, steer))
    bid, ask, mid = _extract_bid_ask(entry or {})
    if not _quote_liquidity_ok(entry):
        return float(fallback_price if fallback_price is not None else (mid or 0.0))
    try:
        if bid is not None and ask is not None and ask >= bid:
            w = (1.0 - steer) if side == "buy" else steer
            return float(bid) + float(w) * (float(ask) - float(bid))
    except Exception:
        pass
    return float(mid if mid is not None else fallback_price)


def _cal_bt_timing_enabled() -> bool:
    try:
        return bool(getattr(get_settings(), "debug_backtest_timing", False))
    except Exception:
        return False


def _cal_bt_timing_print(msg: str) -> None:
    if not _cal_bt_timing_enabled():
        return
    try:
        stamp = _time.perf_counter()
        print(f"[CALBT-TIME] {stamp:.6f} {msg}")
    except Exception:
        pass


@contextmanager
def _cal_bt_timing(section: str, *, extra: str | None = None):
    if not _cal_bt_timing_enabled():
        yield
        return
    label = section if extra is None else f"{section} {extra}"
    _cal_bt_timing_print(f"start {label}")
    start = _time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (_time.perf_counter() - start) * 1000.0
        _cal_bt_timing_print(f"done {label} {elapsed_ms:.1f}ms")


def _extract_sizes(entry: dict) -> tuple[float | None, float | None]:
    try:
        q = (entry or {}).get("quote") or {}
        bid_sz = q.get("bid_size")
        ask_sz = q.get("ask_size")
        # Some payloads may use different keys; try common alternates
        if bid_sz is None:
            bid_sz = q.get("bidsize") or q.get("bid_size_contracts")
        if ask_sz is None:
            ask_sz = q.get("asksize") or q.get("ask_size_contracts")
        return (None if bid_sz is None else float(bid_sz)), (
            None if ask_sz is None else float(ask_sz)
        )
    except Exception:
        return None, None


@dataclass
class CalendarBacktestConfig:
    ticker: str
    start_date: str
    end_date: str
    qty: int = 1
    ff_entry_threshold: float = 0.1
    ff_exit_threshold: float = 0.0
    take_profit_pct: float = 0.3
    stop_loss_pct: float = 1.0
    front_dte: Optional[int] = None
    back_dte: Optional[int] = None
    option_type: str = "call"
    debug: bool = False
    tickers: Optional[List[str]] = None
    max_daily_positions: int = 3

    def front_back_targets(self) -> tuple[int, int]:
        settings = get_settings()
        front_default = int(getattr(settings, "calendar_front_dte_default", 60))
        back_default = int(getattr(settings, "calendar_back_dte_default", 90))
        front = int(self.front_dte or front_default)
        back = int(self.back_dte or back_default)
        if back <= front:
            back = max(back, front + 5)
        return front, back


@dataclass
class CalendarPosition:
    id: int
    open_date: date
    target_dt: datetime
    ticker: str
    strike_front: float
    strike_back: float
    front_expiry: date
    back_expiry: date
    front_option: str
    back_option: str
    qty: int
    entry_premium_front: float
    entry_premium_back: float
    entry_debit_points: float
    entry_ff: float
    notes: Dict[str, Any] = field(default_factory=dict)
    entry_ts_front: Optional[datetime] = None
    entry_ts_back: Optional[datetime] = None
    close_date: Optional[date] = None
    close_premium_front: Optional[float] = None
    close_premium_back: Optional[float] = None
    close_reason: Optional[str] = None
    close_ts_front: Optional[datetime] = None
    close_ts_back: Optional[datetime] = None
    realized_pnl: float = 0.0
    entry_spot: Optional[float] = None
    close_spot: Optional[float] = None
    entry_iv_front: Optional[float] = None
    entry_iv_back: Optional[float] = None
    close_iv_front: Optional[float] = None
    close_iv_back: Optional[float] = None
    entry_iv_front_reason: Optional[str] = None
    entry_iv_back_reason: Optional[str] = None
    close_iv_front_reason: Optional[str] = None
    close_iv_back_reason: Optional[str] = None


@dataclass
class _LegPricingRequest:
    pos_id: int
    leg: str  # "front" or "back"
    ticker: str
    meta: Dict[str, Any]
    as_of_dt: datetime

    @property
    def is_open(self) -> bool:
        return self.close_date is None


_risk_free_history: Dict[str, Dict[date, float]] = {}
_risk_free_history_span: tuple[date, date] | None = None


def _prepare_risk_free_history(start: date, end: date) -> None:
    global _risk_free_history, _risk_free_history_span
    if start > end:
        return
    span = (start, end)
    if _risk_free_history_span == span and _risk_free_history:
        return
    symbols = ("^IRX", "^FVX", "^TNX")
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    new_history: Dict[str, Dict[date, float]] = {}
    for sym in symbols:
        mapping: Dict[date, float] = {}
        df = None
        try:
            df = get_historical_prices(sym, start_str, end_str)
        except Exception:
            df = None
        if df is not None and not df.empty:
            for raw_date, close_val in zip(df["date"], df["close"]):
                d_obj = raw_date.date() if hasattr(raw_date, "date") else raw_date
                if not isinstance(d_obj, date):
                    continue
                try:
                    rate_value = float(close_val) / 100.0
                except Exception:
                    continue
                mapping[d_obj] = max(0.0, rate_value)
        new_history[sym] = mapping
        if mapping:
            cache_map = {d.strftime("%Y-%m-%d"): rate for d, rate in mapping.items()}
            cache_risk_free_rates(sym, cache_map)
    _risk_free_history = new_history
    _risk_free_history_span = span


def _infer_risk_free(cur: date, dte_days: int) -> float:
    tenor = max(1, int(dte_days))
    if tenor <= 90:
        sym = "^IRX"
    elif tenor <= 365:
        sym = "^FVX"
    else:
        sym = "^TNX"
    ds = cur.strftime("%Y-%m-%d")
    history_map = _risk_free_history.get(sym)
    if history_map is not None:
        rate_value = history_map.get(cur)
        if rate_value is not None:
            return float(rate_value)
    cached = get_cached_risk_free_rate(sym, ds)
    if cached is not None:
        return float(cached)
    rate_value = 0.0
    try:
        df = get_historical_prices(sym, ds, ds)
        if df is not None and not df.empty:
            y = float(df.iloc[-1]["close"]) / 100.0
            rate_value = max(0.0, y)
    except Exception:
        pass
    cache_risk_free_rate(sym, ds, rate_value)
    return rate_value


def _ns_to_datetime(ns: Optional[int]) -> Optional[datetime]:
    if ns is None:
        return None
    try:
        return datetime.fromtimestamp(float(ns) / 1_000_000_000)
    except Exception:
        return None


def _format_ts(dt_obj: Optional[datetime]) -> Optional[str]:
    if not isinstance(dt_obj, datetime):
        return None
    try:
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _calc_leg_iv_value(
    *,
    spot: Optional[float],
    strike: float,
    premium: Optional[float],
    expiry: date,
    as_of: date,
    option_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[float], Optional[str]]:
    settings = get_settings()
    debug_solver = bool(getattr(settings, "debug_iv_solver", False))
    debug_iv = bool(getattr(settings, "debug_iv", False))

    def _handle_failure(
        reason: str, context: Optional[Dict[str, Any]] = None
    ) -> tuple[None, str]:
        if debug_solver:
            print(f"[IV-DEBUG] reason={reason} ctx={context}")
            breakpoint()
        return None, reason

    base_context: Dict[str, Any] = {
        "spot": spot,
        "strike": strike,
        "premium": premium,
        "expiry": expiry.isoformat() if isinstance(expiry, date) else expiry,
        "as_of": as_of.isoformat() if isinstance(as_of, date) else as_of,
        "option_type": option_type,
    }
    if context:
        base_context.update(context)

    if spot is None:
        return _handle_failure("missing_spot", base_context)
    if premium is None:
        return _handle_failure("missing_premium", base_context)
    dte_days = max(1, (expiry - as_of).days)
    base_context["dte_days"] = dte_days
    r = _infer_risk_free(as_of, dte_days)
    q_yield = float(getattr(settings, "dividend_yield_default", 0.0) or 0.0)
    base_context["risk_free_rate"] = r
    base_context["dividend_yield"] = q_yield
    trace_context = dict(base_context)
    trace_context["option_pricing_type"] = context.get("pricing_mode") if context else None
    if debug_iv:
        print(f"[IV-TRACE] starting solver with inputs: {trace_context}")
    try:
        value = calc_implied_vol(
            spot=float(spot),
            strike=float(strike),
            premium=float(premium),
            dte=dte_days,
            option_type=option_type,
            risk_free_rate=r,
            dividend_yield=q_yield,
        )
        if value is None:
            return _handle_failure("solver_failed", base_context)
        if debug_iv:
            print(f"[IV-TRACE] solver result={value} ctx={trace_context}")
            breakpoint()
        return value, None
    except Exception as exc:
        return _handle_failure(
            f"exception:{exc.__class__.__name__}",
            {**base_context, "error": str(exc)},
        )


async def _price_leg(
    *,
    ticker: str,
    option_type: str,
    premium_field: str,
    meta: Dict[str, Any],
    client: PolygonAPIClient,
    as_of_dt: datetime,
    debug: bool = False,
) -> Tuple[Optional[float], Optional[datetime], Dict[str, Any]]:
    entry = {"meta": dict(meta)}
    as_of_str = as_of_dt.strftime("%Y-%m-%d %H:%M:%S")
    force_refresh = False
    price: Optional[float] = None
    ts_dt: Optional[datetime] = None
    attempts = 0
    ok = False
    fail_with_data = False
    while True:
        attempts += 1
        price = await ensure_premium(
            entry=entry,
            premium_field=premium_field,
            client=client,
            option_type=option_type,
            underlying=ticker,
            as_of_str=as_of_str,
            debug=debug,
            force_refresh=force_refresh,
        )
        quote = entry.get("quote", {}) if isinstance(entry, dict) else {}
        ts_ns = PolygonAPIClient._sample_timestamp(quote)
        ts_dt = _ns_to_datetime(ts_ns)
        ok, fail_with_data = _quote_liquidity_status(entry)
        cache_source = quote.get("_cache_source") if isinstance(quote, dict) else None
        if ok or not fail_with_data or force_refresh or not cache_source:
            break
        # First failure with data that came from cache; try forcing refresh once.
        force_refresh = True
        entry.pop("quote", None)
        continue
    if not ok:
        if fail_with_data:
            entry["_liquidity_rejected_hard"] = True
        return None, None, entry
    return price, ts_dt, entry


def _select_matched_samples(
    *,
    entry_front: Dict[str, Any],
    entry_back: Dict[str, Any],
    target_dt: datetime,
    premium_field: str,
    pair_delta_override_secs: Optional[int] = None,
) -> Optional[Tuple[float, datetime, float, datetime]]:
    settings = get_settings()
    window_secs = max(1, int(getattr(settings, "premium_time_window_secs", 60)))
    pair_delta_secs = int(getattr(settings, "premium_pair_delta_secs", 60))
    target_ns = int(target_dt.timestamp() * 1_000_000_000)
    win_ns = window_secs * 1_000_000_000
    lo_ns = target_ns - win_ns
    hi_ns = target_ns + win_ns

    def _samples(entry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        quote = entry.get("quote", {}) if isinstance(entry, dict) else {}
        raw = extract_samples_from_quote(quote, target_ns=target_ns)
        full = dedup_samples_by_timestamp(
            filter_window_samples(
                raw,
                premium_field=premium_field,
                window_lo_ns=lo_ns,
                window_hi_ns=hi_ns,
                require_liquidity=False,
            )
        )
        ok = [sample for sample in full if sample.get("_liquidity_ok")]
        return ok, full

    front_samples_ok, front_samples_all = _samples(entry_front)
    back_samples_ok, back_samples_all = _samples(entry_back)
    if not front_samples_all or not back_samples_all:
        return None

    def _match(
        front_samples: List[Dict[str, Any]],
        back_samples: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if not front_samples or not back_samples:
            return None
        return select_best_sample_pair(
            front_samples,
            back_samples,
            premium_field=premium_field,
            target_ns=target_ns,
            pair_delta_ns=int(
                pair_delta_secs
                if pair_delta_override_secs is None
                else pair_delta_override_secs
            )
            * 1_000_000_000,
            prefer_liquidity=True,
        )

    pair = _match(front_samples_ok, back_samples_ok)
    if not pair:
        pair = _match(front_samples_all, back_samples_all)
    if not pair:
        return None
    front_sample, back_sample = pair
    front_price = front_sample.get("_price")
    back_price = back_sample.get("_price")
    if front_price is None or back_price is None:
        return None
    front_ts = _ns_to_datetime(PolygonAPIClient._sample_timestamp(front_sample))
    back_ts = _ns_to_datetime(PolygonAPIClient._sample_timestamp(back_sample))
    if front_ts is None or back_ts is None:
        return None
    return float(front_price), front_ts, float(back_price), back_ts


async def _matched_close_prices(
    *,
    pos: "CalendarPosition",
    client: PolygonAPIClient,
    front_meta: Dict[str, Any],
    back_meta: Dict[str, Any],
    target_dt: datetime,
    premium_field: str,
    option_type: str,
    prefetched_front: Optional[PrefetchedLeg] = None,
    prefetched_back: Optional[PrefetchedLeg] = None,
) -> Optional[Tuple[float, datetime, float, datetime]]:
    settings = get_settings()
    backoff_minutes = max(1, int(getattr(settings, "premium_time_backoff_minutes", 60)))
    backoff_steps = max(0, int(getattr(settings, "premium_time_backoff_steps", 3)))
    offsets = [0] + [-(i * backoff_minutes * 60) for i in range(1, backoff_steps + 1)]
    max_back_days = max(0, int(getattr(settings, "premium_close_max_back_days", 5)))
    day_offsets = [0] + [-(i + 1) for i in range(max_back_days)]
    pair_delta_secs = int(getattr(settings, "premium_pair_delta_secs", 60))
    pair_delta_multiplier = float(
        getattr(settings, "premium_close_pair_delta_multiplier", 5.0) or 1.0
    )
    pair_delta_close_secs = max(
        pair_delta_secs, int(pair_delta_secs * pair_delta_multiplier)
    )

    fallback: Optional[Tuple[float, datetime, float, datetime]] = None

    if prefetched_front and prefetched_back:
        # Build steered execution prices from prefetched entries as fallback
        f_price = _exec_price_from_entry(
            prefetched_front[2], side="buy", fallback_price=float(prefetched_front[0])
        )
        b_price = _exec_price_from_entry(
            prefetched_back[2], side="sell", fallback_price=float(prefetched_back[0])
        )
        fallback = (
            float(f_price),
            prefetched_front[1],
            float(b_price),
            prefetched_back[1],
        )
        matched_prefetched = _select_matched_samples(
            entry_front=prefetched_front[2],
            entry_back=prefetched_back[2],
            target_dt=target_dt,
            premium_field=premium_field,
            pair_delta_override_secs=pair_delta_close_secs,
        )
        if matched_prefetched:
            # matched_prefetched are mid prices; recompute steered from entries
            f_entry = prefetched_front[2]
            b_entry = prefetched_back[2]
            f_mid, f_ts, b_mid, b_ts = matched_prefetched
            f_exec = _exec_price_from_entry(
                f_entry, side="buy", fallback_price=float(f_mid)
            )
            b_exec = _exec_price_from_entry(
                b_entry, side="sell", fallback_price=float(b_mid)
            )
            return (float(f_exec), f_ts, float(b_exec), b_ts)

    for day_offset in day_offsets:
        day_dt = target_dt + timedelta(days=day_offset)
        if day_dt.date() < pos.open_date:
            break
        for offset in offsets:
            candidate_dt = day_dt + timedelta(seconds=offset)
            front_price, front_ts, front_entry = await _price_leg(
                ticker=pos.ticker,
                option_type=option_type,
                premium_field=premium_field,
                meta=front_meta,
                client=client,
                as_of_dt=candidate_dt,
                debug=False,
            )
            back_price, back_ts, back_entry = await _price_leg(
                ticker=pos.ticker,
                option_type=option_type,
                premium_field=premium_field,
                meta=back_meta,
                client=client,
                as_of_dt=candidate_dt,
                debug=False,
            )
            liquidity_rejected = bool(
                front_entry.get("_liquidity_rejected_hard")
                or back_entry.get("_liquidity_rejected_hard")
            )
            if front_price is not None and back_price is not None:
                fallback = (
                    float(front_price),
                    front_ts or candidate_dt,
                    float(back_price),
                    back_ts or candidate_dt,
                )
            matched = _select_matched_samples(
                entry_front=front_entry,
                entry_back=back_entry,
                target_dt=candidate_dt,
                premium_field=premium_field,
                pair_delta_override_secs=pair_delta_close_secs,
            )
            if matched:
                f_mid, f_ts, b_mid, b_ts = matched
                f_exec = _exec_price_from_entry(
                    front_entry, side="buy", fallback_price=float(f_mid)
                )
                b_exec = _exec_price_from_entry(
                    back_entry, side="sell", fallback_price=float(b_mid)
                )
                return (float(f_exec), f_ts, float(b_exec), b_ts)
            if liquidity_rejected:
                # Try another timestamp within the allowed window to locate quotes
                # that satisfy the liquidity constraints before giving up.
                continue
            if front_price is None or back_price is None:
                continue

    return fallback


PrefetchedLeg = Tuple[float, datetime, Dict[str, Any]]


async def _prefetch_leg_prices(
    requests: List[_LegPricingRequest],
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str,
) -> Dict[Tuple[int, str], PrefetchedLeg]:
    results: Dict[Tuple[int, str], PrefetchedLeg] = {}
    if not requests:
        return results

    async def _run(req: _LegPricingRequest):
        price, ts, entry = await _price_leg(
            ticker=req.ticker,
            option_type=option_type,
            premium_field=premium_field,
            meta=req.meta,
            client=client,
            as_of_dt=req.as_of_dt,
            debug=False,
        )
        return req, price, ts, entry

    tasks = [asyncio.create_task(_run(req)) for req in requests]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    for item in gathered:
        if isinstance(item, Exception):
            continue
        req, price, ts, entry = item
        if price is None or entry is None:
            continue
        key = (req.pos_id, req.leg)
        results[key] = (float(price), ts or req.as_of_dt, entry)
    return results


async def _collect_price_history_for_position(
    *,
    pos: "CalendarPosition",
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str,
    target_time: time,
) -> Dict[str, List[Tuple[datetime, float]]]:
    """Collect daily front/back leg prices between open and close dates."""
    front_meta = {
        "strike_price": pos.strike_front,
        "expiration_date": pos.front_expiry.strftime("%Y-%m-%d"),
        "option_ticker": pos.front_option,
    }
    back_meta = {
        "strike_price": pos.strike_back,
        "expiration_date": pos.back_expiry.strftime("%Y-%m-%d"),
        "option_ticker": pos.back_option,
    }
    front_points: List[Tuple[datetime, float]] = []
    back_points: List[Tuple[datetime, float]] = []
    spread_points: List[Tuple[datetime, float]] = []
    start_day = pos.open_date
    end_day = pos.close_date or pos.open_date
    cur_day = start_day
    while cur_day <= end_day:
        as_of_dt = datetime.combine(cur_day, target_time)
        try:
            front_price, front_ts, _ = await _price_leg(
                ticker=pos.ticker,
                option_type=option_type,
                premium_field=premium_field,
                meta=front_meta,
                client=client,
                as_of_dt=as_of_dt,
                debug=False,
            )
        except Exception:
            front_price, front_ts = None, None
        front_val = None
        if front_price is not None:
            front_val = float(front_price)
            front_points.append((front_ts or as_of_dt, front_val))
        try:
            back_price, back_ts, _ = await _price_leg(
                ticker=pos.ticker,
                option_type=option_type,
                premium_field=premium_field,
                meta=back_meta,
                client=client,
                as_of_dt=as_of_dt,
                debug=False,
            )
        except Exception:
            back_price, back_ts = None, None
        back_val = None
        if back_price is not None:
            back_val = float(back_price)
            back_points.append((back_ts or as_of_dt, back_val))
        if front_val is not None and back_val is not None:
            spread_points.append((as_of_dt, back_val - front_val))
        cur_day += timedelta(days=1)
    start_str = start_day.strftime("%Y-%m-%d")
    end_str = end_day.strftime("%Y-%m-%d")

    def _series_from_prices(sym: str) -> List[Tuple[datetime, float]]:
        data = get_historical_prices(
            sym,
            start_str,
            end_str,
            data_source="yfinance",
        )
        series: List[Tuple[datetime, float]] = []
        if data is not None and not data.empty:
            for d_raw, px in zip(data["date"], data["close"]):
                try:
                    d = d_raw.date() if hasattr(d_raw, "date") else d_raw
                    dt_point = datetime.combine(d, target_time)
                    series.append((dt_point, float(px)))
                except Exception:
                    continue
        return series

    spot_points = _series_from_prices(pos.ticker)
    vix_points = _series_from_prices("^VIX")

    return {
        "front": front_points,
        "back": back_points,
        "spread": spread_points,
        "spot": spot_points,
        "vix": vix_points,
    }


def _render_price_history_plot(
    *,
    plot_day: date,
    histories: List[Tuple["CalendarPosition", Dict[str, List[Tuple[datetime, float]]]]],
) -> None:
    """Render subplots showing price history for each provided position."""
    if not histories:
        return
    count = len(histories)
    rows = count + 1
    fig, axes = plt.subplots(rows, 1, figsize=(14, 4.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    def _annotate(axis, points: List[Tuple[datetime, float]], color: str) -> None:
        for ts, price in points:
            label = ts.strftime("%H:%M")
            axis.annotate(
                label,
                (ts, price),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                rotation=35,
                color=color,
            )

    spot_axis = axes_flat[0]
    spot_series_by_ticker: Dict[str, List[Tuple[datetime, float]]] = {}
    vix_series: List[Tuple[datetime, float]] = []
    for pos, series in histories:
        if series.get("spot"):
            spot_series_by_ticker.setdefault(pos.ticker, series.get("spot", []))
        if not vix_series and series.get("vix"):
            vix_series = series.get("vix", [])

    if spot_series_by_ticker:
        color_cycle = cycle(
            plt.rcParams.get(
                "axes.prop_cycle", plt.cycler(color=["tab:blue"])
            ).by_key()["color"]
        )
        for ticker, pts in spot_series_by_ticker.items():
            color = next(color_cycle)
            spot_axis.plot(
                [ts for ts, _ in pts],
                [val for _, val in pts],
                label=f"{ticker} Spot",
                color=color,
                marker=".",
            )
    else:
        spot_axis.text(
            0.5, 0.5, "No spot data", transform=spot_axis.transAxes, ha="center"
        )

    spot_axis.set_title("Spot & VIX")
    spot_axis.set_ylabel("Spot Price ($)")
    spot_axis.grid(True, linestyle="--", alpha=0.3)
    spot_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    spot_axis.tick_params(axis="x", rotation=30)

    legend_handles: List[Any] = []
    legend_labels: List[str] = []
    if vix_series:
        ax_vix = spot_axis.twinx()
        ax_vix.plot(
            [ts for ts, _ in vix_series],
            [val for _, val in vix_series],
            color="tab:red",
            linestyle="--",
            marker="^",
            label="VIX",
        )
        ax_vix.set_ylabel("VIX")
        ax_vix.grid(False)
        h2, l2 = ax_vix.get_legend_handles_labels()
        legend_handles.extend(h2)
        legend_labels.extend(l2)

    h1, l1 = spot_axis.get_legend_handles_labels()
    legend_handles = h1 + legend_handles
    legend_labels = l1 + legend_labels
    if legend_handles:
        spot_axis.legend(legend_handles, legend_labels, loc="best")

    for idx, (pos, series) in enumerate(histories):
        ax = axes_flat[idx + 1]
        front_points = series.get("front", [])
        back_points = series.get("back", [])
        front_label = (pos.front_option or "").strip() or f"{pos.ticker} front"
        back_label = (pos.back_option or "").strip() or f"{pos.ticker} back"
        spread_points = series.get("spread", [])
        plotted = False
        if front_points:
            ax.plot(
                [ts for ts, _ in front_points],
                [price for _, price in front_points],
                marker="o",
                label=front_label,
                color="tab:blue",
            )
            _annotate(ax, front_points, "tab:blue")
            plotted = True
        if back_points:
            ax.plot(
                [ts for ts, _ in back_points],
                [price for _, price in back_points],
                marker="s",
                label=back_label,
                color="tab:orange",
            )
            _annotate(ax, back_points, "tab:orange")
            plotted = True
        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        if not plotted:
            ax.text(0.5, 0.5, "No price data", transform=ax.transAxes, ha="center")
        ax.set_title(
            f"ID {pos.id} {pos.ticker} | Open {pos.open_date} → Close {pos.close_date} | PnL {pos.realized_pnl:.2f}"
        )
        ax.set_ylabel("Premium ($)")
        ax.set_xlabel("Date")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        if spread_points:
            ax_spread = ax.twinx()
            ax_spread.plot(
                [ts for ts, _ in spread_points],
                [val for _, val in spread_points],
                color="tab:green",
                marker="^",
                linestyle="--",
                label="Spread (back-front)",
            )
            _annotate(ax_spread, spread_points, "tab:green")
            ax_spread.set_ylabel("Spread ($)")
            ax_spread.grid(False)
            h2, l2 = ax_spread.get_legend_handles_labels()
            legend_handles.extend(h2)
            legend_labels.extend(l2)
        h1, l1 = ax.get_legend_handles_labels()
        legend_handles = h1 + legend_handles
        legend_labels = l1 + legend_labels
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="best")

    for idx in range(len(histories), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Loss Pair Price History — {plot_day.strftime('%Y-%m-%d')}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close(fig)


def _render_single_position_history(
    *, pos: "CalendarPosition", series: Dict[str, List[Tuple[datetime, float]]]
) -> None:
    """Render a per-position chart showing front/back price history."""
    front_points = series.get("front", [])
    back_points = series.get("back", [])
    if not front_points and not back_points:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    if front_points:
        ax.plot(
            [ts for ts, _ in front_points],
            [val for _, val in front_points],
            marker="o",
            label=(pos.front_option or "front").strip() or "front",
            color="tab:blue",
        )
        plotted = True
    if back_points:
        ax.plot(
            [ts for ts, _ in back_points],
            [val for _, val in back_points],
            marker="s",
            label=(pos.back_option or "back").strip() or "back",
            color="tab:orange",
        )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No price data", transform=ax.transAxes, ha="center")
    ax.set_title(
        f"{pos.ticker} ID {pos.id} | {pos.open_date} → {pos.close_date} | PnL {pos.realized_pnl:+.2f}"
    )
    ax.set_ylabel("Premium ($)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=30)
    if plotted:
        ax.legend(loc="best")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


async def _plot_history_for_closed_position(
    *,
    pos: "CalendarPosition",
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str,
    target_time: time,
) -> None:
    """Collect data + render chart for a single closed position."""
    if pos.close_date is None:
        return
    try:
        series = await _collect_price_history_for_position(
            pos=pos,
            client=client,
            premium_field=premium_field,
            option_type=option_type,
            target_time=target_time,
        )
    except Exception:
        return
    if series.get("front") or series.get("back"):
        _render_single_position_history(pos=pos, series=series)


async def _maybe_plot_loss_positions(
    *,
    positions: List["CalendarPosition"],
    client: PolygonAPIClient,
    premium_field: str,
    option_type: str,
    target_time: time,
    plot_day: date,
) -> None:
    """Fetch price history + render plots for loss-making positions."""
    if not positions:
        return
    histories: List[
        Tuple[CalendarPosition, Dict[str, List[Tuple[datetime, float]]]]
    ] = []
    for pos in positions:
        try:
            series = await _collect_price_history_for_position(
                pos=pos,
                client=client,
                premium_field=premium_field,
                option_type=option_type,
                target_time=target_time,
            )
        except Exception:
            continue
        if series.get("front") or series.get("back"):
            histories.append((pos, series))
    if histories:
        _render_price_history_plot(plot_day=plot_day, histories=histories)


def _print_loss_position_details(positions: List["CalendarPosition"]) -> None:
    if not positions:
        return

    def _fmt_iv(iv: Optional[float]) -> str:
        return f"{iv * 100:.2f}%" if isinstance(iv, (int, float)) else "n/a"

    for pos in positions:
        reason = pos.close_reason or "n/a"
        print(
            f"- [{pos.ticker}] id {pos.id} | reason {reason} | debit {pos.entry_debit_points:.2f} | P/L {pos.realized_pnl:+.2f}"
        )
        front_entry_ts = _format_ts(pos.entry_ts_front) or "n/a"
        front_close_ts = _format_ts(pos.close_ts_front) or "n/a"
        back_entry_ts = _format_ts(pos.entry_ts_back) or "n/a"
        back_close_ts = _format_ts(pos.close_ts_back) or "n/a"

        def _fmt_price(val: Optional[float]) -> str:
            return f"{val:.2f}" if isinstance(val, (int, float)) else "n/a"

        front_entry_p = _fmt_price(pos.entry_premium_front)
        front_close_p = _fmt_price(pos.close_premium_front)
        back_entry_p = _fmt_price(pos.entry_premium_back)
        back_close_p = _fmt_price(pos.close_premium_back)
        front_entry_iv = _fmt_iv(pos.entry_iv_front)
        front_close_iv = _fmt_iv(pos.close_iv_front)
        back_entry_iv = _fmt_iv(pos.entry_iv_back)
        back_close_iv = _fmt_iv(pos.close_iv_back)
        print(
            f"    {pos.front_option or 'front'}: {front_entry_p}@{front_entry_ts} [{front_entry_iv}] -> "
            f"{front_close_p}@{front_close_ts} [{front_close_iv}]"
        )
        print(
            f"    {pos.back_option or 'back'}: {back_entry_p}@{back_entry_ts} [{back_entry_iv}] -> "
            f"{back_close_p}@{back_close_ts} [{back_close_iv}]"
        )


async def simulate_calendar_backtest(config: CalendarBacktestConfig) -> Dict[str, Any]:
    settings = get_settings()
    premium_field = resolve_premium_field(settings)
    front_target, back_target = config.front_back_targets()
    gap_target_days = max(1.0, float(back_target - front_target))
    gap_tol_pct = float(getattr(settings, "calendar_gap_tolerance_pct", 0.3) or 0.0)
    gap_min_days = max(0.0, gap_target_days * (1.0 - gap_tol_pct))
    gap_max_days = gap_target_days * (1.0 + gap_tol_pct)
    weekday = getattr(settings, "calendar_weekday_default", "Friday")
    target_time_s = getattr(settings, "premium_time_target", "12:30:00") or "12:30:00"
    try:
        tt_h, tt_m, tt_s = [int(x) for x in target_time_s.split(":")]
    except Exception:
        tt_h, tt_m, tt_s = 12, 30, 0
    try:
        target_time_obj = time(tt_h, tt_m, tt_s)
    except Exception:
        target_time_obj = time(12, 30, 0)
    plot_history_enabled = bool(
        getattr(settings, "plot_option_pair_price_history", False)
    )
    capital_pct_limit = max(
        0.0, float(getattr(settings, "calendar_ticker_debt_pct", 0.05) or 0.0)
    )
    position_pct_limit = max(
        0.0, float(getattr(settings, "calendar_position_debt_pct", 0.05) or 0.0)
    )
    take_profit_pct_effective = float(
        config.take_profit_pct
        if config.take_profit_pct is not None
        else getattr(settings, "calendar_take_profit_pct", 0.3)
    )
    stop_loss_pct_effective = float(
        config.stop_loss_pct
        if config.stop_loss_pct is not None
        else getattr(settings, "calendar_stop_loss_pct", 1.0)
    )
    initial_capital_setting = max(
        0.0, float(getattr(settings, "initial_capital", 0.0) or 0.0)
    )

    start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").date()
    _prepare_risk_free_history(start_dt, end_dt)

    ticker_list: List[str] = []
    for raw in config.tickers or [config.ticker]:
        if raw:
            ticker_list.append(str(raw).upper())
    if not ticker_list:
        raise ValueError("No tickers provided for calendar backtest.")
    single_ticker_pool = len(ticker_list) == 1

    for ticker in ticker_list:
        load_stored_option_data(ticker, strike_type="atm")

    close_by_ticker: Dict[str, Dict[date, float]] = {}
    for ticker in ticker_list:
        hist = get_historical_prices(
            ticker,
            config.start_date,
            config.end_date,
            data_source="yfinance",
        )
        date_map: Dict[date, float] = {}
        if hist is not None and not hist.empty:
            for d_raw, px in zip(hist["date"], hist["close"]):
                try:
                    d = d_raw.date() if hasattr(d_raw, "date") else d_raw
                    date_map[d] = float(px)
                except Exception:
                    continue
        close_by_ticker[ticker] = date_map

    positions: List[CalendarPosition] = []
    closed_positions: List[CalendarPosition] = []
    daily_results: List[Dict[str, Any]] = []
    realized_total = 0.0
    next_id = 1
    closed_realized_total = 0.0

    cur = start_dt
    cum_wins = 0
    cum_closed = 0

    autosave_interval_secs = 5 * 60
    autosave_last = _time.monotonic()

    def _autosave_option_cache(force: bool = False) -> None:
        nonlocal autosave_last
        now = _time.monotonic()
        if not force and (now - autosave_last) < autosave_interval_secs:
            return
        saved_any = False
        for ticker in ticker_list:
            if not force and option_data_unsaved_count(ticker) <= 0:
                continue
            try:
                save_stored_option_data(ticker, strike_type="atm")
                saved_any = True
            except Exception as exc:
                print(f"Warning: Failed to auto-save option cache for {ticker}: {exc}")
        if saved_any or force:
            autosave_last = now

    async with PolygonAPIClient(api_key=settings.polygon_api_key) as client:
        while cur <= end_dt:
            day_realized = 0.0
            day_unrealized = 0.0
            open_value_total = 0.0
            opened_today: List[Dict[str, Any]] = []
            closed_today: List[Dict[str, Any]] = []
            closed_today_positions: List[CalendarPosition] = []

            target_dt = datetime.combine(cur, target_time_obj)

            todays_spots: Dict[str, Optional[float]] = {
                ticker: close_by_ticker.get(ticker, {}).get(cur)
                for ticker in ticker_list
            }
            has_spot_data = any(val is not None for val in todays_spots.values())
            if not has_spot_data:
                daily_results.append(
                    {
                        "date": cur.strftime("%Y-%m-%d"),
                        "spot": None,
                        "ff": None,
                        "forward_vol": None,
                        "front_dte": None,
                        "back_dte": None,
                        "opened": [],
                        "closed": [],
                        "open_count": len(positions),
                        "realized_pnl_day": 0.0,
                        "realized_pnl_cumulative": realized_total,
                        "instant_pnl_cumulative": realized_total,
                        "unrealized_pnl_total": 0.0,
                        "dte_gap": None,
                        "failure_reason": "no_spot_data",
                        "ticker_breakdown": [
                            {
                                "ticker": ticker,
                                "spot": None,
                                "ff": None,
                                "forward_vol": None,
                                "front_dte": None,
                                "back_dte": None,
                                "gap_dte": None,
                                "failure_reason": "no_spot",
                            }
                            for ticker in ticker_list
                        ],
                    }
                )
                cur += timedelta(days=1)
                continue

            day_label = cur.strftime("%Y-%m-%d")

            leg_requests: List[_LegPricingRequest] = []
            if positions:
                for pos in positions:
                    front_meta_prefetch = {
                        "strike_price": pos.strike_front,
                        "expiration_date": pos.front_expiry.strftime("%Y-%m-%d"),
                        "option_ticker": pos.front_option,
                    }
                    back_meta_prefetch = {
                        "strike_price": pos.strike_back,
                        "expiration_date": pos.back_expiry.strftime("%Y-%m-%d"),
                        "option_ticker": pos.back_option,
                    }
                    leg_requests.append(
                        _LegPricingRequest(
                            pos_id=pos.id,
                            leg="front",
                            ticker=pos.ticker,
                            meta=front_meta_prefetch,
                            as_of_dt=target_dt,
                        )
                    )
                    leg_requests.append(
                        _LegPricingRequest(
                            pos_id=pos.id,
                            leg="back",
                            ticker=pos.ticker,
                            meta=back_meta_prefetch,
                            as_of_dt=target_dt,
                        )
                    )

            with _cal_bt_timing("prefetch_leg_prices", extra=day_label):
                prefetched_leg_map = await _prefetch_leg_prices(
                    leg_requests,
                    client=client,
                    premium_field=premium_field,
                    option_type=config.option_type,
                )

            # Evaluate open positions for exits before opening new one
            still_open: List[CalendarPosition] = []
            with _cal_bt_timing("evaluate_open_positions", extra=day_label):
                for pos in positions:
                    ticker = pos.ticker
                    dte_front_remaining = max(0, (pos.front_expiry - cur).days)
                    front_meta = {
                        "strike_price": pos.strike_front,
                        "expiration_date": pos.front_expiry.strftime("%Y-%m-%d"),
                        "option_ticker": pos.front_option,
                    }
                    back_meta = {
                        "strike_price": pos.strike_back,
                        "expiration_date": pos.back_expiry.strftime("%Y-%m-%d"),
                        "option_ticker": pos.back_option,
                    }
                    prefetched_front = prefetched_leg_map.get((pos.id, "front"))
                    prefetched_back = prefetched_leg_map.get((pos.id, "back"))
                    matched_prices = await _matched_close_prices(
                        pos=pos,
                        client=client,
                        front_meta=front_meta,
                        back_meta=back_meta,
                        target_dt=target_dt,
                        premium_field=premium_field,
                        option_type=config.option_type,
                        prefetched_front=prefetched_front,
                        prefetched_back=prefetched_back,
                    )
                    unrealized = 0.0
                    mark_to_market_value = None
                    if matched_prices:
                        front_price, front_ts, back_price, back_ts = matched_prices

                        value_now_points = back_price - front_price
                        pnl_points = value_now_points - pos.entry_debit_points
                        unrealized = pnl_points * 100.0 * pos.qty
                        day_unrealized += unrealized
                        mark_to_market_value = value_now_points * 100.0 * pos.qty

                        exit_due_to_expiry = dte_front_remaining <= 0
                        exit_due_to_take_profit = False
                        exit_due_to_stop_loss = False
                        if take_profit_pct_effective > 0 and pos.entry_debit_points > 0:
                            if value_now_points >= pos.entry_debit_points * (
                                1.0 + take_profit_pct_effective
                            ):
                                exit_due_to_take_profit = True
                        if stop_loss_pct_effective > 0 and pos.entry_debit_points > 0:
                            if value_now_points <= pos.entry_debit_points * (
                                1.0 - stop_loss_pct_effective
                            ):
                                exit_due_to_stop_loss = True
                    else:
                        # No matched prices available after backoff. If the front leg
                        # has already expired, force-close the position to avoid
                        # repeatedly querying an expired contract. When debug_closure
                        # is enabled, drop into a breakpoint for investigation.
                        if dte_front_remaining <= 0:
                            if getattr(settings, "debug_closure", False):
                                try:
                                    print(
                                        f"[CLOSE-DEBUG] No quotes to close "
                                        f"pos_id={pos.id} {ticker} front_expiry={pos.front_expiry} "
                                        f"as_of={cur} – forcing expiry close with neutral P&L."
                                    )
                                except Exception:
                                    pass
                                breakpoint()
                            # Neutral assumption: close the spread at its entry debit
                            # when we cannot obtain reliable quotes at/after expiry.
                            value_now_points = float(pos.entry_debit_points)
                            pnl_points = 0.0
                            exit_due_to_expiry = True
                            exit_due_to_take_profit = False
                            exit_due_to_stop_loss = False
                            front_price = None
                            back_price = None
                            front_ts = None
                            back_ts = None
                            mark_to_market_value = value_now_points * 100.0 * pos.qty
                        else:
                            open_value_total += (
                                float(pos.entry_debit_points) * 100.0 * float(pos.qty)
                            )
                            still_open.append(pos)
                            continue

                    exit_reason = None
                    if exit_due_to_expiry:
                        exit_reason = "front_expiry"
                    elif exit_due_to_take_profit:
                        exit_reason = "take_profit"
                    elif exit_due_to_stop_loss:
                        exit_reason = "stop_loss"
                    if exit_reason:
                        entry_cash = (
                            float(pos.entry_debit_points) * 100.0 * float(pos.qty)
                        )
                        exit_cash = value_now_points * 100.0 * pos.qty
                        # Apply close commissions: two legs per spread
                        fee_per_leg = float(
                            getattr(settings, "option_trade_cost", 0.5) or 0.0
                        )
                        close_fees = 2.0 * fee_per_leg * float(pos.qty)
                        profit_dollars = (
                            pnl_points * 100.0 * pos.qty
                            - float(pos.notes.get("fees_open", 0.0))
                            - close_fees
                        )
                        realized_total += exit_cash
                        day_realized += exit_cash
                        if close_fees:
                            realized_total -= close_fees
                            day_realized -= close_fees
                        closed_realized_total += profit_dollars
                        day_unrealized -= unrealized
                        pos.close_spot = todays_spots.get(ticker)
                        pos.close_date = cur
                        pos.close_premium_front = front_price
                        pos.close_premium_back = back_price
                        pos.close_ts_front = front_ts
                        pos.close_ts_back = back_ts
                        pos.close_reason = exit_reason
                        pos.realized_pnl = profit_dollars
                        closed_positions.append(pos)
                        closed_today_positions.append(pos)
                        closed_today.append(
                            {
                                "id": pos.id,
                                "ticker": pos.ticker,
                                "strike_front": pos.strike_front,
                                "strike_back": pos.strike_back,
                                "front_expiry": pos.front_expiry.strftime("%Y-%m-%d"),
                                "back_expiry": pos.back_expiry.strftime("%Y-%m-%d"),
                                "debit_points": pos.entry_debit_points,
                                "qty": pos.qty,
                                "entry_cash": entry_cash,
                                "fees_open": float(pos.notes.get("fees_open", 0.0)),
                                "fees_close": close_fees,
                                "pnl": profit_dollars,
                                "reason": exit_reason,
                                "front_option": pos.front_option,
                                "back_option": pos.back_option,
                                "entry_premium_front": pos.entry_premium_front,
                                "entry_premium_back": pos.entry_premium_back,
                                "close_premium_front": pos.close_premium_front,
                                "close_premium_back": pos.close_premium_back,
                                "entry_iv_front": pos.entry_iv_front,
                                "entry_iv_back": pos.entry_iv_back,
                                "close_iv_front": pos.close_iv_front,
                                "close_iv_back": pos.close_iv_back,
                                "entry_iv_front_reason": pos.entry_iv_front_reason,
                                "entry_iv_back_reason": pos.entry_iv_back_reason,
                                "close_iv_front_reason": pos.close_iv_front_reason,
                                "close_iv_back_reason": pos.close_iv_back_reason,
                                "entry_ts_front": _format_ts(pos.entry_ts_front),
                                "entry_ts_back": _format_ts(pos.entry_ts_back),
                                "close_ts_front": _format_ts(pos.close_ts_front),
                                "close_ts_back": _format_ts(pos.close_ts_back),
                                "entry_spot": pos.entry_spot,
                                "close_spot": pos.close_spot,
                            }
                        )
                        if plot_history_enabled:
                            await _plot_history_for_closed_position(
                                pos=pos,
                                client=client,
                                premium_field=premium_field,
                                option_type=config.option_type,
                                target_time=target_time_obj,
                            )
                    else:
                        if mark_to_market_value is None:
                            mark_to_market_value = (
                                float(pos.entry_debit_points) * 100.0 * float(pos.qty)
                            )
                        open_value_total += mark_to_market_value
                        still_open.append(pos)
            # Persist only positions that remain open
            positions = still_open

            async def _evaluate_ticker(ticker: str) -> Dict[str, Any]:
                info: Dict[str, Any] = {
                    "ticker": ticker,
                    "spot": close_by_ticker.get(ticker, {}).get(cur),
                    "ff": None,
                    "forward_vol": None,
                    "front_dte": None,
                    "back_dte": None,
                    "gap_dte": None,
                    "failure_reason": None,
                    "pair_result": None,
                }
                timing_info: Dict[str, float] = {}
                info["timing"] = timing_info
                total_start = _time.perf_counter()
                try:
                    spot_val = info["spot"]
                    if spot_val is None:
                        info["failure_reason"] = "no_spot"
                        return info

                    gather_start = _time.perf_counter()
                    pair_result, failure_reason = await gather_calendar_pair(
                        ticker=ticker,
                        as_of_date=cur,
                        spot=float(spot_val),
                        front_target=front_target,
                        back_target=back_target,
                        weekday=weekday,
                        client=client,
                        premium_field=premium_field,
                        option_type=config.option_type,
                        debug=config.debug,
                    )
                    timing_info["gather_pair_ms"] = (
                        _time.perf_counter() - gather_start
                    ) * 1000.0
                    info["failure_reason"] = failure_reason
                    if pair_result is None:
                        return info
                    if getattr(pair_result, "timings", None):
                        timing_info["gather_detail"] = pair_result.timings

                    size_limit = float(
                        getattr(settings, "bid_ask_size_limit", 0.0) or 0.0
                    )
                    if size_limit > 0.0:
                        bf, af = _extract_sizes(pair_result.entry_front)
                        bb, ab = _extract_sizes(pair_result.entry_back)

                        def _too_small(val: float | None) -> bool:
                            return val is not None and val < size_limit

                        if (
                            _too_small(bf)
                            or _too_small(af)
                            or _too_small(bb)
                            or _too_small(ab)
                        ):
                            info["failure_reason"] = f"size_below_limit({size_limit})"
                            if getattr(
                                settings, "debug_calendar_pair_selection", False
                            ):
                                print(
                                    f"[CAL-Select] {ticker} {cur} skipped: size below limit. "
                                    f"front(bid={bf},ask={af}) back(bid={bb},ask={ab}) limit={size_limit}"
                                )
                            return info

                    dte_front = max(0, (pair_result.front_date - cur).days)
                    dte_back = max(0, (pair_result.back_date - cur).days)
                    info["front_dte"] = dte_front
                    info["back_dte"] = dte_back
                    gap_dte = float(dte_back - dte_front)
                    info["gap_dte"] = gap_dte
                    if gap_dte < gap_min_days or gap_dte > gap_max_days:
                        info["failure_reason"] = f"dte_gap_out_of_range({gap_dte:.1f})"
                        if getattr(settings, "debug_calendar_pair_selection", False):
                            print(
                                (
                                    f"[CAL-Select] {ticker} {cur} skipped: DTE spread out of spec. "
                                    f"front_dte={dte_front} back_dte={dte_back} gap={gap_dte:.1f} "
                                    f"allowed=[{gap_min_days:.1f}, {gap_max_days:.1f}] "
                                    f"targets={front_target}/{back_target} tol_pct={gap_tol_pct:.2f}"
                                )
                            )
                        return info

                    r_front = _infer_risk_free(cur, dte_front)
                    r_back = _infer_risk_free(cur, dte_back)
                    q_yield = float(
                        getattr(settings, "dividend_yield_default", 0.0) or 0.0
                    )

                    iv_start = _time.perf_counter()
                    iv_front = calc_implied_vol(
                        spot=float(spot_val),
                        strike=pair_result.strike_front,
                        premium=pair_result.premium_front,
                        dte=dte_front,
                        option_type=config.option_type,
                        risk_free_rate=r_front,
                        dividend_yield=q_yield,
                    )
                    iv_back = calc_implied_vol(
                        spot=float(spot_val),
                        strike=pair_result.strike_back,
                        premium=pair_result.premium_back,
                        dte=dte_back,
                        option_type=config.option_type,
                        risk_free_rate=r_back,
                        dividend_yield=q_yield,
                    )
                    timing_info["iv_calc_ms"] = (
                        _time.perf_counter() - iv_start
                    ) * 1000.0
                    if iv_front is None or iv_back is None:
                        return info

                    ff_start = _time.perf_counter()
                    ff_pair = forward_factor(
                        iv_front=iv_front,
                        iv_back=iv_back,
                        dte_front=dte_front,
                        dte_back=dte_back,
                    )
                    timing_info["ff_calc_ms"] = (
                        _time.perf_counter() - ff_start
                    ) * 1000.0
                    if ff_pair is None:
                        return info
                    ff_value, forward_vol_local = ff_pair
                    info["ff"] = ff_value
                    info["forward_vol"] = forward_vol_local
                    info["pair_result"] = pair_result
                    return info
                finally:
                    timing_info["total_ms"] = (
                        _time.perf_counter() - total_start
                    ) * 1000.0

            candidate_stage_start = _time.perf_counter()
            with _cal_bt_timing("evaluate_candidates", extra=day_label):
                ticker_tasks = [
                    asyncio.create_task(_evaluate_ticker(ticker))
                    for ticker in ticker_list
                ]
                after_task_creation = _time.perf_counter()
                if ticker_tasks:
                    candidate_infos = await asyncio.gather(*ticker_tasks)
                else:
                    candidate_infos = []
                after_gather = _time.perf_counter()
                candidate_map = {
                    item["ticker"]: item
                    for item in candidate_infos
                    if item.get("ticker")
                }
                primary_ticker = ticker_list[0]
                primary_info = candidate_map.get(primary_ticker) or (
                    candidate_infos[0] if candidate_infos else {}
                )
                primary_spot = primary_info.get("spot")
                ff_value_primary = primary_info.get("ff")
                forward_vol_primary = primary_info.get("forward_vol")
                front_dte_current = primary_info.get("front_dte")
                back_dte_current = primary_info.get("back_dte")
                gap_dte_current = primary_info.get("gap_dte")
                failure_reason_primary = primary_info.get("failure_reason")

                max_positions = max(0, config.max_daily_positions or 0)
                filter_start = after_gather
                valid_candidates = [
                    info
                    for info in candidate_infos
                    if info.get("pair_result") is not None
                    and info.get("ff") is not None
                    and float(info["ff"]) >= config.ff_entry_threshold
                ]
                valid_candidates.sort(
                    key=lambda x: x.get("ff", float("-inf")), reverse=True
                )
                after_filter = _time.perf_counter()
                selected_candidates: List[Dict[str, Any]] = []
                if max_positions > 0:
                    total_capital_base = max(
                        0.0, initial_capital_setting + realized_total
                    )
                    min_spread = float(
                        getattr(settings, "minimum_front_back_spread", 0.0) or 0.0
                    )
                    pending_entry_cash = 0.0
                    open_debt_by_ticker: Dict[str, float] = {}
                    for pos in positions:
                        entry_cash_active = (
                            float(pos.entry_debit_points) * 100.0 * pos.qty
                        )
                        if entry_cash_active:
                            open_debt_by_ticker[pos.ticker] = (
                                open_debt_by_ticker.get(pos.ticker, 0.0)
                                + entry_cash_active
                            )
                    for info in valid_candidates:
                        if len(selected_candidates) >= max_positions:
                            break
                        ticker = info.get("ticker")
                        pair_result = info.get("pair_result")
                        if pair_result is None:
                            continue
                        entry_debit_points = (
                            pair_result.premium_back - pair_result.premium_front
                        )
                        if min_spread > 0.0 and entry_debit_points < min_spread:
                            continue
                        per_contract_cash = float(entry_debit_points)
                        if per_contract_cash <= 0.0:
                            continue
                        per_contract_cash *= 100.0
                        allocated_qty = None
                        if capital_pct_limit > 0.0 and ticker:
                            current_debt = open_debt_by_ticker.get(ticker, 0.0)
                            current_total_capital = max(
                                0.0, total_capital_base - pending_entry_cash
                            )
                            if single_ticker_pool:
                                # Interpret as per-day allocation when only one ticker is in the pool
                                per_day_allowance = (
                                    current_total_capital * capital_pct_limit
                                )
                                allowed_cash = max(
                                    0.0, per_day_allowance - pending_entry_cash
                                )
                            else:
                                # Multi-ticker: enforce per-ticker open debt cap relative to capital
                                per_ticker_credit = (
                                    current_total_capital * capital_pct_limit
                                )
                                allowed_cash = max(
                                    0.0, per_ticker_credit - current_debt
                                )
                            if position_pct_limit > 0.0 and current_total_capital > 0.0:
                                per_position_cash = (
                                    current_total_capital * position_pct_limit
                                )
                                allowed_cash = (
                                    min(allowed_cash, per_position_cash)
                                    if allowed_cash > 0
                                    else per_position_cash
                                )
                            if allowed_cash < per_contract_cash:
                                continue
                            qty_possible = int(allowed_cash // per_contract_cash)
                            if qty_possible <= 0:
                                continue
                            allocated_qty = qty_possible
                            entry_cash_total = per_contract_cash * allocated_qty
                            pending_entry_cash += entry_cash_total
                            open_debt_by_ticker[ticker] = (
                                current_debt + entry_cash_total
                            )
                        else:
                            allocated_qty = max(1, config.qty or 1)
                        info["allocated_qty"] = allocated_qty
                        selected_candidates.append(info)
                selection_end = _time.perf_counter()

            stage_metrics = {
                "task_create_ms": (after_task_creation - candidate_stage_start)
                * 1000.0,
                "await_gather_ms": (after_gather - after_task_creation) * 1000.0,
                "filter_sort_ms": (after_filter - filter_start) * 1000.0,
                "selection_ms": (selection_end - after_filter) * 1000.0,
                "total_ms": (selection_end - candidate_stage_start) * 1000.0,
            }
            timing_threshold = float(
                getattr(settings, "candidate_timing_threshold_ms", 250.0) or 0.0
            )
            should_log_timing = (
                bool(getattr(settings, "debug_evaluate_candidates", False))
                or config.debug
                or (
                    timing_threshold > 0.0
                    and stage_metrics["total_ms"] >= timing_threshold
                )
            )
            if should_log_timing:
                print(
                    (
                        f"[CALBT-Candidates] {day_label} stage_ms "
                        f"create={stage_metrics['task_create_ms']:.1f} "
                        f"gather={stage_metrics['await_gather_ms']:.1f} "
                        f"filter={stage_metrics['filter_sort_ms']:.1f} "
                        f"select={stage_metrics['selection_ms']:.1f} "
                        f"total={stage_metrics['total_ms']:.1f}"
                    )
                )
                slow_candidates = sorted(
                    candidate_infos,
                    key=lambda info: (info.get("timing") or {}).get("total_ms", 0.0),
                    reverse=True,
                )
                for info in slow_candidates[:3]:
                    timing = info.get("timing") or {}
                    ticker_label = info.get("ticker") or "n/a"
                    status = (
                        "ok"
                        if info.get("pair_result") is not None
                        and info.get("failure_reason") in (None, "ok")
                        else info.get("failure_reason") or "n/a"
                    )
                    print(
                        (
                            f"[CALBT-Candidates] {day_label} ticker={ticker_label} "
                            f"total={timing.get('total_ms', 0.0):.1f}ms "
                            f"gather={timing.get('gather_pair_ms', 0.0):.1f}ms "
                            f"iv={timing.get('iv_calc_ms', 0.0):.1f}ms "
                            f"ff={timing.get('ff_calc_ms', 0.0):.1f}ms "
                            f"status={status}"
                        )
                    )
                    gather_detail = timing.get("gather_detail") or {}
                    attempts = gather_detail.get("attempts") or []
                    detail_attempt = None
                    idx = gather_detail.get("success_attempt_index")
                    if isinstance(idx, int) and 0 <= idx < len(attempts):
                        detail_attempt = attempts[idx]
                    elif attempts:
                        detail_attempt = attempts[-1]
                    if detail_attempt:
                        print(
                            (
                                f"[CALBT-Candidates] {day_label} ticker={ticker_label} gather_detail "
                                f"attempt={detail_attempt.get('attempt_index')} "
                                f"offset={detail_attempt.get('offset_index')} "
                                f"ensure_front={detail_attempt.get('ensure_front_ms', 0.0):.1f}ms "
                                f"ensure_back={detail_attempt.get('ensure_back_ms', 0.0):.1f}ms "
                                f"sample_front={detail_attempt.get('front_sample_ms', 0.0):.1f}ms/{detail_attempt.get('front_sample_count', 0)} "
                                f"sample_back={detail_attempt.get('back_sample_ms', 0.0):.1f}ms/{detail_attempt.get('back_sample_count', 0)} "
                                f"select={detail_attempt.get('select_pair_ms', 0.0):.1f}ms "
                                f"result={detail_attempt.get('result')}"
                            )
                        )
                        print(
                            (
                                f"[CALBT-Candidates] {day_label} ticker={ticker_label} gather_timeline "
                                f"front_offset={gather_detail.get('front_contracts_offset_ms', 0.0):.1f}ms "
                                f"front_dur={gather_detail.get('front_contracts_ms', 0.0):.1f}ms "
                                f"back_offset={gather_detail.get('back_contracts_offset_ms', 0.0):.1f}ms "
                                f"back_dur={gather_detail.get('back_contracts_ms', 0.0):.1f}ms "
                                f"strike_offset={gather_detail.get('strike_pair_offset_ms', 0.0):.1f}ms "
                                f"strike_dur={gather_detail.get('strike_pair_build_ms', 0.0):.1f}ms "
                                f"attempt_start={detail_attempt.get('attempt_start_offset_ms', 0.0):.1f}ms "
                                f"attempts={len(attempts)} "
                                f"total={gather_detail.get('total_ms', 0.0):.1f}ms"
                            )
                        )

            with _cal_bt_timing("open_positions", extra=day_label):
                for info in selected_candidates:
                    pair_result = info["pair_result"]
                    ff_value = info["ff"]
                    # Use steered execution prices for entry cash/debit
                    exec_front_sell = _exec_price_from_entry(
                        pair_result.entry_front,
                        side="sell",
                        fallback_price=pair_result.premium_front,
                    )
                    exec_back_buy = _exec_price_from_entry(
                        pair_result.entry_back,
                        side="buy",
                        fallback_price=pair_result.premium_back,
                    )
                    entry_debit_points = exec_back_buy - exec_front_sell
                    qty = int(info.get("allocated_qty") or max(1, config.qty or 1))
                    new_pos = CalendarPosition(
                        id=next_id,
                        open_date=cur,
                        target_dt=pair_result.target_dt,
                        ticker=info["ticker"],
                        strike_front=pair_result.strike_front,
                        strike_back=pair_result.strike_back,
                        front_expiry=pair_result.front_date,
                        back_expiry=pair_result.back_date,
                        front_option=pair_result.entry_front.get("meta", {}).get(
                            "option_ticker", ""
                        ),
                        back_option=pair_result.entry_back.get("meta", {}).get(
                            "option_ticker", ""
                        ),
                        qty=qty,
                        entry_premium_front=pair_result.premium_front,
                        entry_premium_back=pair_result.premium_back,
                        entry_debit_points=entry_debit_points,
                        entry_ff=ff_value,
                    )
                    new_pos.entry_ts_front = _ns_to_datetime(pair_result.used_ts_front)
                    new_pos.entry_ts_back = _ns_to_datetime(pair_result.used_ts_back)
                    new_pos.entry_spot = info.get("spot")
                    iv_front_val, iv_front_reason = _calc_leg_iv_value(
                        spot=new_pos.entry_spot,
                        strike=new_pos.strike_front,
                        premium=new_pos.entry_premium_front,
                        expiry=new_pos.front_expiry,
                        as_of=cur,
                        option_type=config.option_type,
                        context={
                            "stage": "entry",
                            "leg": "front",
                            "ticker": info.get("ticker"),
                            "date": cur.isoformat(),
                        },
                    )
                    new_pos.entry_iv_front = iv_front_val
                    new_pos.entry_iv_front_reason = iv_front_reason
                    iv_back_val, iv_back_reason = _calc_leg_iv_value(
                        spot=new_pos.entry_spot,
                        strike=new_pos.strike_back,
                        premium=new_pos.entry_premium_back,
                        expiry=new_pos.back_expiry,
                        as_of=cur,
                        option_type=config.option_type,
                        context={
                            "stage": "entry",
                            "leg": "back",
                            "ticker": info.get("ticker"),
                            "date": cur.isoformat(),
                        },
                    )
                    new_pos.entry_iv_back = iv_back_val
                    new_pos.entry_iv_back_reason = iv_back_reason
                    entry_cash = float(entry_debit_points) * 100.0 * float(qty)
                    # Apply open commissions: two legs per spread
                    fee_per_leg = float(
                        getattr(settings, "option_trade_cost", 0.5) or 0.0
                    )
                    open_fees = 2.0 * fee_per_leg * float(qty)
                    if entry_cash:
                        realized_total -= entry_cash
                        day_realized -= entry_cash
                    if open_fees:
                        realized_total -= open_fees
                        day_realized -= open_fees
                    new_pos.notes["fees_open"] = open_fees
                    positions.append(new_pos)
                    next_id += 1
                    opened_today.append(
                        {
                            "id": new_pos.id,
                            "ticker": info["ticker"],
                            "strike_front": new_pos.strike_front,
                            "strike_back": new_pos.strike_back,
                            "front_expiry": new_pos.front_expiry.strftime("%Y-%m-%d"),
                            "back_expiry": new_pos.back_expiry.strftime("%Y-%m-%d"),
                            "entry_debit_points": entry_debit_points,
                            "qty": qty,
                            "entry_cash": entry_cash,
                            "fees_open": open_fees,
                            "ff": ff_value,
                            "front_option": new_pos.front_option,
                            "back_option": new_pos.back_option,
                            "entry_premium_front": new_pos.entry_premium_front,
                            "entry_premium_back": new_pos.entry_premium_back,
                            "entry_iv_front": new_pos.entry_iv_front,
                            "entry_iv_back": new_pos.entry_iv_back,
                            "entry_iv_front_reason": new_pos.entry_iv_front_reason,
                            "entry_iv_back_reason": new_pos.entry_iv_back_reason,
                            "gap_dte": info.get("gap_dte"),
                            "entry_ts_front": _format_ts(new_pos.entry_ts_front),
                            "entry_ts_back": _format_ts(new_pos.entry_ts_back),
                            "opened_ts": _format_ts(
                                new_pos.entry_ts_front
                                or new_pos.entry_ts_back
                                or new_pos.target_dt
                            ),
                        }
                    )

            ticker_breakdown = [
                {
                    "ticker": info.get("ticker"),
                    "spot": info.get("spot"),
                    "ff": info.get("ff"),
                    "forward_vol": info.get("forward_vol"),
                    "front_dte": info.get("front_dte"),
                    "back_dte": info.get("back_dte"),
                    "gap_dte": info.get("gap_dte"),
                    "entry_iv_front_reason": info.get("entry_iv_front_reason"),
                    "entry_iv_back_reason": info.get("entry_iv_back_reason"),
                    "failure_reason": info.get("failure_reason"),
                }
                for info in candidate_infos
            ]

            with _cal_bt_timing("record_day", extra=day_label):
                daily_results.append(
                    {
                        "date": cur.strftime("%Y-%m-%d"),
                        "spot": None if primary_spot is None else float(primary_spot),
                        "ff": ff_value_primary,
                        "forward_vol": forward_vol_primary,
                        "front_dte": front_dte_current,
                        "back_dte": back_dte_current,
                        "opened": opened_today,
                        "closed": closed_today,
                        "open_count": len(positions),
                        "realized_pnl_day": day_realized,
                        "realized_pnl_cumulative": realized_total,
                        "instant_pnl_cumulative": realized_total + open_value_total,
                        "unrealized_pnl_total": day_unrealized,
                        "dte_gap": gap_dte_current,
                        "failure_reason": failure_reason_primary,
                        "ticker_breakdown": ticker_breakdown,
                    }
                )
            # Print in-loop summary for this date
            opened_count = len(positions)
            wins = sum(
                1
                for trade in closed_today
                if isinstance(trade.get("pnl"), (int, float))
                and float(trade.get("pnl")) > 0
            )
            cum_closed += len(closed_today)
            cum_wins += wins
            if cum_closed > 0:
                win_rate_pct = (cum_wins / cum_closed) * 100.0
                win_rate_str = f"{cum_wins}/{cum_closed} ({win_rate_pct:.1f}%)"
            else:
                win_rate_str = "0/0 (n/a)"
            cash_balance = initial_capital_setting + realized_total
            cash_balance_str = f"{cash_balance:+.2f}"
            pnl_str = f"{closed_realized_total:+.2f}"
            pnl_color = GREEN if closed_realized_total >= 0 else RED
            print(
                f"[Summary] {cur.strftime('%Y-%m-%d')} | opened={opened_count} | win_rate={win_rate_str} "
                f"| cash_balance={cash_balance_str} | P/L {pnl_color}{pnl_str}{RESET}"
            )

            _autosave_option_cache()

            cur += timedelta(days=1)

    for ticker in ticker_list:
        try:
            save_stored_option_data(ticker, strike_type="atm")
        except Exception:
            continue

    summary = {
        "ticker": ",".join(ticker_list),
        "start": config.start_date,
        "end": config.end_date,
        "realized_total": realized_total,
        "open_positions": [
            {
                "id": pos.id,
                "ticker": pos.ticker,
                "open_date": pos.open_date.strftime("%Y-%m-%d"),
                "strike_front": pos.strike_front,
                "strike_back": pos.strike_back,
                "front_expiry": pos.front_expiry.strftime("%Y-%m-%d"),
                "back_expiry": pos.back_expiry.strftime("%Y-%m-%d"),
                "entry_debit_points": pos.entry_debit_points,
                "entry_ff": pos.entry_ff,
            }
            for pos in positions
        ],
        "closed_positions": [
            {
                "id": pos.id,
                "ticker": pos.ticker,
                "open_date": pos.open_date.strftime("%Y-%m-%d"),
                "close_date": (
                    pos.close_date.strftime("%Y-%m-%d") if pos.close_date else None
                ),
                "strike_front": pos.strike_front,
                "strike_back": pos.strike_back,
                "front_expiry": pos.front_expiry.strftime("%Y-%m-%d"),
                "back_expiry": pos.back_expiry.strftime("%Y-%m-%d"),
                "entry_debit_points": pos.entry_debit_points,
                "close_reason": pos.close_reason,
                "pnl": pos.realized_pnl,
            }
            for pos in closed_positions
        ],
        "daily_results": daily_results,
    }
    return summary


def run_calendar_backtest(config: CalendarBacktestConfig) -> Dict[str, Any]:
    return asyncio.run(simulate_calendar_backtest(config))
