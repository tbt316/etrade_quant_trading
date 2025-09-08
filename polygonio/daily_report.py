from __future__ import annotations
from typing import Any, Dict, List
from datetime import date, datetime

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
    closed: List[Dict[str, Any]] = []
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

        # early closures (per leg or whole position)
        if (closed_at == date_str) or (call_cd == date_str) or (put_cd == date_str):
            closed.append(p)

        # natural expiry
        if exp_s == date_str:
            closed.append(p)

    print(f"{BOLD}=== {date_str} Daily Trades ==={RESET}")
    if opened:
        print(f"{ORANGE}{BOLD}Opened Positions:{RESET}")
        for p in opened:
            print(f"{ORANGE} + {_pos_to_line(p, as_of=date_str, price_by_date=price_by_date, vix_by_date=vix_by_date)}{RESET}")
    else:
        print(f"{DIM}Opened Positions: (none){RESET}")

    if closed:
        print(f"{BOLD}Closed Positions:{RESET}")
        for p in closed:
            profit = _sum_closed_profit_for_date(p, date_str)
            color = GREEN if (isinstance(profit, (int, float)) and profit > 0) else RED
            print(f"{color} - {_pos_to_line(p, as_of=date_str, price_by_date=price_by_date, vix_by_date=vix_by_date)}{RESET}")
    else:
        print(f"{DIM}Closed Positions: (none){RESET}")

    if not (opened or closed) and debug:
        print(_fmt_engine_summary(debug))
