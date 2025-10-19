from __future__ import annotations
from typing import Any, Dict, List
from datetime import date, datetime

from .config import get_settings, PREMIUM_FIELD_MAP
from .cache_io import stored_option_price

def _to_datestr(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (date, datetime)):
        return x.strftime("%Y-%m-%d")
    return None

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
# 256-color "orange" (approx). Falls back gracefully on terminals without 256-color support.
ORANGE = "\033[38;5;208m"

def _fmt_legs(legs: List[Dict[str, Any]]) -> str:
    parts = []
    for leg in legs or []:
        action = leg.get("action", "")
        side = leg.get("side", "")
        strike = leg.get("strike", "")
        prem = leg.get("premium", "")
        abbrA = "S" if action == "sell" else "B"
        abbrS = "C" if side == "call" else "P"
        try:
            parts.append(f"{abbrA}{abbrS} {float(strike):g}@{float(prem):g}")
        except Exception:
            parts.append(f"{abbrA}{abbrS} {strike}@{prem}")
    return " | ".join(parts)

def _pos_to_line(
    p: Dict[str, Any],
    *,
    as_of: str | None = None,
    price_by_date: Dict[str, float] | None = None,
    vix_by_date: Dict[str, float] | None = None,
) -> str:
    """Format one position line.

    - Hide explicit strategy name (e.g. "put_credit_spread").
    - Include VIX close and underlying close for the target date (``as_of``).
    """
    legs = _fmt_legs(p.get("legs", []))
    qty = p.get("qty", 1)
    und = p.get("underlying", "") or p.get("ticker", "")
    exp = p.get("expiration", "") or p.get("expiration_date", "")
    opened_at = p.get("opened_at", "")

    # Resolve which date to use for printing prices
    ds = (as_of or _to_datestr(opened_at)) or ""
    vix = None
    px = None
    if price_by_date and ds in price_by_date:
        px = price_by_date.get(ds)
    if vix_by_date and ds in vix_by_date:
        vix = vix_by_date.get(ds)

    def _fmtf(x):
        try:
            return f"{float(x):g}"
        except Exception:
            return "?"

    vix_str = _fmtf(vix) if vix is not None else "?"
    px_str = _fmtf(px) if px is not None else "?"

    return f"{und} x{qty}  [VIX:{vix_str} px:{px_str}]  [{legs}]  open:{opened_at}  exp:{exp}"

def _sum_closed_profit_for_date(p: Dict[str, Any], ds: str) -> float | None:
    """Sum realized profit for legs that closed on ds. Returns None if unknown."""
    total = 0.0
    found = False
    # Per-leg realized fields
    try:
        if _to_datestr(p.get("call_closed_date")) == ds and (p.get("call_closed_profit") is not None):
            total += float(p.get("call_closed_profit") or 0.0)
            found = True
    except Exception:
        pass
    try:
        if _to_datestr(p.get("put_closed_date")) == ds and (p.get("put_closed_profit") is not None):
            total += float(p.get("put_closed_profit") or 0.0)
            found = True
    except Exception:
        pass
    # Whole-position fields (if ever provided by upstream)
    try:
        if _to_datestr(p.get("closed_at")) == ds and (p.get("profit") is not None):
            total += float(p.get("profit") or 0.0)
            found = True
    except Exception:
        pass
    return total if found else None


def _compute_expiry_close_price(side: str, strike: float, spot_close: float) -> float:
    if side == "put":
        return max(strike - spot_close, 0.0)
    else:
        return max(spot_close - strike, 0.0)


def _lookup_leg_price(
    p: Dict[str, Any],
    date_str: str,
    *,
    side: str,
    strike: Any,
) -> float | None:
    try:
        strike_f = float(strike)
    except Exception:
        return None

    ticker = (p.get("underlying") or p.get("symbol") or p.get("ticker") or "").upper()
    if not ticker:
        return None

    exp = _to_datestr(p.get("expiration") or p.get("expiration_date"))
    if not exp:
        return None

    strike_key = round(strike_f, 2)
    leg_data = (
        stored_option_price
        .get(ticker, {})
        .get(date_str, {})
        .get(strike_key, {})
        .get(exp, {})
        .get(side.lower(), {})
    )
    if not isinstance(leg_data, dict) or not leg_data:
        return None

    settings = get_settings()
    preferred = PREMIUM_FIELD_MAP.get(settings.premium_price_mode, "trade_price")
    for key in (preferred, "trade_price", "close_price", "mid_price", "price"):
        val = leg_data.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _format_leg_prices(
    p: Dict[str, Any],
    date_str: str,
    price_by_date: Dict[str, float],
    *,
    expiration: bool = False,
) -> str:
    spot_close = price_by_date.get(date_str) if expiration else None

    def fmt(val: Any) -> str:
        try:
            return f"{float(val):.2f}"
        except Exception:
            return "?"

    parts: List[str] = []

    def add_leg(label: str, open_field: str, close_field: str, strike_field: str, side: str) -> None:
        open_val = p.get(open_field)
        close_val = p.get(close_field)
        strike = p.get(strike_field)

        if close_val is None:
            if expiration and spot_close is not None and strike is not None:
                try:
                    close_val = _compute_expiry_close_price(side, float(strike), float(spot_close))
                except Exception:
                    close_val = None
            if close_val is None:
                close_val = _lookup_leg_price(p, date_str, side=side, strike=strike)

        if open_val is None and close_val is None:
            return
        strike_str = fmt(strike) if strike is not None else "?"
        open_str = fmt(open_val) if open_val is not None else "?"
        close_str = fmt(close_val) if close_val is not None else "?"
        parts.append(f"{label} {strike_str}: {open_str}\u2192{close_str}")

    add_leg("SP", "short_put_prem_open", "short_put_prem_today", "put_strike_sold", "put")
    add_leg("LP", "long_put_prem_open", "long_put_prem_today", "put_strike_bought", "put")
    add_leg("SC", "short_call_prem_open", "short_call_prem_today", "call_strike_sold", "call")
    add_leg("LC", "long_call_prem_open", "long_call_prem_today", "call_strike_bought", "call")

    return "; ".join(parts)

def _normalize_positions(
    obj: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Return ``(positions, per_day_map, debug)`` from varied inputs.

    Accepts a raw list of positions, a mapping containing ``positions`` and
    optional ``debug`` metadata, or an object with matching attributes.  Any
    available debug data is always returned so callers can surface engine
    summaries even when the result set is empty.
    """

    debug: Dict[str, Any] = {}
    per_day: Dict[str, List[Dict[str, Any]]] = {}

    if obj is None:
        pos: Any = []
    elif isinstance(obj, dict):
        debug = obj.get("debug", {})
        pos = obj.get("positions", [])
    else:
        debug = getattr(obj, "debug", {})
        pos = getattr(obj, "positions", obj)

    if isinstance(pos, dict):
        per_day = pos
        all_pos: List[Dict[str, Any]] = []
        for _d, _lst in pos.items():
            all_pos.extend(_lst or [])
        return all_pos, per_day, debug
    return pos or [], per_day, debug


def _fmt_engine_summary(debug: Dict[str, Any]) -> str:
    return (
        f"{DIM}[hint] Engine summary — days={debug.get('days_total')} "
        f"no_price={debug.get('days_no_price')} "
        f"earnings_skips={debug.get('days_skipped_earnings')} "
        f"expiries={debug.get('expiries_considered')} "
        f"positions_built={debug.get('positions_built')} "
        f"exceptions={debug.get('exceptions')}{RESET}"
    )

def print_opened_and_closed_for_date(results_or_positions: Any, date_str: str) -> None:
    all_pos, per_day, debug = _normalize_positions(results_or_positions)

    # Build quick lookups for VIX and underlying close if a full results dict is provided
    price_by_date: Dict[str, float] = {}
    vix_by_date: Dict[str, float] = {}
    if isinstance(results_or_positions, dict):
        try:
            for r in results_or_positions.get("price", []) or []:
                ds = _to_datestr(r.get("date")) or _to_datestr(r.get("as_of"))
                if ds and (r.get("close") is not None):
                    price_by_date[ds] = float(r.get("close"))
        except Exception:
            pass
        try:
            for r in results_or_positions.get("vix", []) or []:
                ds = _to_datestr(r.get("date")) or _to_datestr(r.get("as_of"))
                if ds and (r.get("close") is not None):
                    vix_by_date[ds] = float(r.get("close"))
        except Exception:
            pass

    todays = per_day.get(date_str, []) if per_day else []

    if not todays:
        todays = [
            p for p in all_pos
            if any(
                _to_datestr(p.get(k)) == date_str
                for k in (
                    "opened_at",
                    "expiration",
                    "expiration_date",
                    "closed_at",
                    "call_closed_date",
                    "put_closed_date",
                )
            )
        ]

    if not all_pos and not todays:
        print(f"{YELLOW}No positions found to report for {date_str}.{RESET}")
        if debug:
            print(_fmt_engine_summary(debug))
        return

    opened: List[Dict[str, Any]] = []
    closed_early: List[Dict[str, Any]] = []
    expired_today: List[Dict[str, Any]] = []
    scan_source = todays if todays else all_pos
    for item in scan_source:
        p = item.to_dict() if hasattr(item, "to_dict") else (dict(item) if isinstance(item, dict) else None)
        if not p: continue
        opened_at = _to_datestr(p.get("opened_at"))
        exp_s     = _to_datestr(p.get("expiration") or p.get("expiration_date"))
        closed_at = _to_datestr(p.get("closed_at"))
        call_cd   = _to_datestr(p.get("call_closed_date"))
        put_cd    = _to_datestr(p.get("put_closed_date"))

        if opened_at == date_str:
            opened.append(p)

        # natural expiry (avoid duplicate reporting if legs already closed earlier)
        if exp_s == date_str:
            def _lt(a: str | None, b: str) -> bool:
                try:
                    if not a or not b:
                        return False
                    return a < b
                except Exception:
                    return False

            call_already = (p.get("short_call_prem_open") in (None, 0)) or _lt(call_cd, date_str)
            put_already  = (p.get("short_put_prem_open") in (None, 0)) or _lt(put_cd, date_str)

            if not (call_already and put_already):
                expired_today.append(p)
                continue

        # early closures (per leg or whole position)
        if (closed_at == date_str) or (call_cd == date_str) or (put_cd == date_str):
            closed_early.append(p)

    print(f"{BOLD}=== {date_str} Daily Trades ==={RESET}")
    if opened:
        print(f"{ORANGE}{BOLD}Opened Positions:{RESET}")
        for p in opened:
            print(f"{ORANGE} + {_pos_to_line(p, as_of=date_str, price_by_date=price_by_date, vix_by_date=vix_by_date)}{RESET}")
    else:
        print(f"{DIM}Opened Positions: (none){RESET}")

    if closed_early:
        print(f"{BOLD}Closed Positions:{RESET}")
        for p in closed_early:
            profit = _sum_closed_profit_for_date(p, date_str)
            color = GREEN if (isinstance(profit, (int, float)) and profit > 0) else RED
            base_line = _pos_to_line(p, as_of=date_str, price_by_date=price_by_date, vix_by_date=vix_by_date)
            profit_str = f" | P/L {profit:+.2f}" if profit is not None else ""
            prices = _format_leg_prices(p, date_str, price_by_date, expiration=False)
            price_str = f" | Prices: {prices}" if prices else " | Prices: n/a"
            print(f"{color} - {base_line}{profit_str}{price_str}{RESET}")
    else:
        print(f"{DIM}Closed Positions: (none){RESET}")

    if expired_today:
        print(f"{CYAN}{BOLD}Expired Positions:{RESET}")
        for p in expired_today:
            profit = _sum_closed_profit_for_date(p, date_str)
            color = CYAN if (profit is None or profit >= 0) else YELLOW
            base_line = _pos_to_line(p, as_of=date_str, price_by_date=price_by_date, vix_by_date=vix_by_date)
            profit_str = f" | P/L {profit:+.2f}" if profit is not None else ""
            prices = _format_leg_prices(p, date_str, price_by_date, expiration=True)
            price_str = f" | Prices: {prices}" if prices else " | Prices: n/a"
            print(f"{color} - {base_line}{profit_str}{price_str}{RESET}")
    else:
        print(f"{DIM}Expired Positions: (none){RESET}")

    if not (opened or closed_early or expired_today) and debug:
        print(_fmt_engine_summary(debug))
