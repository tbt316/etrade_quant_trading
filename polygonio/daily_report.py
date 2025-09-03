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

def _pos_to_line(p: Dict[str, Any]) -> str:
    legs = _fmt_legs(p.get("legs", []))
    qty = p.get("qty", 1)
    ttype = p.get("trade_type", "")
    und = p.get("underlying", "") or p.get("ticker", "")
    exp = p.get("expiration", "") or p.get("expiration_date", "")
    opened_at = p.get("opened_at", "")
    return f"{und} {ttype} x{qty}  [{legs}]  open:{opened_at}  exp:{exp}"

def _normalize_positions(obj: Any):
    debug = {}; per_day = {}
    if isinstance(obj, dict):
        debug = obj.get("debug", {})
        pos = obj.get("positions", [])
    else:
        pos = obj
    if isinstance(pos, dict):
        per_day = pos
        all_pos = []
        for _d, _lst in pos.items():
            all_pos.extend(_lst or [])
        return all_pos, per_day, debug
    return pos or [], per_day, debug

def print_opened_and_closed_for_date(results_or_positions: Any, date_str: str) -> None:
    all_pos, per_day, debug = _normalize_positions(results_or_positions)
    todays = per_day.get(date_str, []) if per_day else [
        p for p in all_pos
        if (isinstance(p, dict) and (p.get("opened_at") == date_str or p.get("expiration") == date_str or p.get("expiration_date") == date_str))
        or (getattr(p, "opened_at", None) == date_str)
    ]
    if not all_pos and not todays:
        print(f"{YELLOW}No positions found to report for {date_str}.{RESET}")
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
        print(f"{GREEN}{BOLD}Opened Positions:{RESET}")
        for p in opened: print(f"{GREEN} + {_pos_to_line(p)}{RESET}")
    else:
        print(f"{DIM}Opened Positions: (none){RESET}")

    if closed:
        print(f"{RED}{BOLD}Closed Positions:{RESET}")
        for p in closed: print(f"{RED} - {_pos_to_line(p)}{RESET}")
    else:
        print(f"{DIM}Closed Positions: (none){RESET}")

    if not (opened or closed) and debug:
        print(f"{DIM}[hint] Engine summary — days={debug.get('days_total')} no_price={debug.get('days_no_price')} "
              f"earnings_skips={debug.get('days_skipped_earnings')} expiries={debug.get('expiries_considered')} "
              f"positions_built={debug.get('positions_built')} exceptions={debug.get('exceptions')}{RESET}")