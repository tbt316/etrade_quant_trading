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
        print(_fmt_engine_summary(debug))


def plot_recursive_results(results: Dict[str, Any], *, show: bool = True, save_path: str | None = None) -> None:
    """Plot cumulative PnL and open position counts from backtest results.

    Parameters
    ----------
    results : dict
        Dictionary returned by ``monthly_recursive_backtest`` or
        ``backtest_options_sync_or_async``.
    show : bool, default True
        If True, display the plot using ``matplotlib.pyplot.show``.
    save_path : str, optional
        If provided, save the plot image to this path.

    Notes
    -----
    The function aggregates per-position ``*_closed_profit`` fields by their
    closing date to build a cumulative PnL curve.  Open position counts are
    taken from the ``pnl`` section of the results.
    """

    import matplotlib
    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import datetime

    pnl_rows = results.get("pnl") or []
    if not pnl_rows:
        print("No PnL data to plot.")
        return

    dates = [datetime.strptime(r.get("as_of"), "%Y-%m-%d") for r in pnl_rows]
    open_positions = [int(r.get("open_positions", 0)) for r in pnl_rows]

    profit_by_date: Dict[str, float] = {}
    for pos in results.get("positions") or []:
        profit = 0.0
        for k in ("call_closed_profit", "put_closed_profit"):
            try:
                profit += float(pos.get(k) or 0.0)
            except Exception:
                continue
        if profit == 0:
            continue
        close_date = None
        for k in ("call_closed_date", "put_closed_date", "closed_at", "expiration"):
            v = pos.get(k)
            if v:
                close_date = v.split("T")[0] if isinstance(v, str) else v
                break
        if close_date is None:
            continue
        profit_by_date.setdefault(close_date, 0.0)
        profit_by_date[close_date] += profit

    cumulative: List[float] = []
    running = 0.0
    for d in [dt.strftime("%Y-%m-%d") for dt in dates]:
        running += profit_by_date.get(d, 0.0)
        cumulative.append(running)

    fig, ax1 = plt.subplots()
    ax1.plot(dates, cumulative, color="tab:blue", label="Cumulative PnL")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PnL ($)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(dates, open_positions, color="tab:orange", label="Open Positions")
    ax2.set_ylabel("Open Positions", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

