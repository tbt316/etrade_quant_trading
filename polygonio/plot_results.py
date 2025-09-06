# etrade_quant_trading/polygonio/plot_results.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# Use the reference plotter exactly as requested
from polygonio_dailytrade_reference import plot_recursive_results  # noqa: E402


def _to_dt(x) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(x, fmt)
            except Exception:
                pass
    return None


def _collect_positions(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(res.get("positions") or [])


def _date_range(start: str, end: str) -> List[datetime]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    cur = s
    while cur <= e:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _build_price_df_from_result(res: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in (res.get("pnl") or []):
        ds = row.get("as_of")
        dt = _to_dt(ds)
        spot = row.get("spot")
        if dt is not None and isinstance(spot, (int, float)):
            rows.append((dt, float(spot)))
    if not rows:
        idx = _date_range(res.get("start"), res.get("end"))
        return pd.DataFrame({"close": np.nan}, index=pd.to_datetime(idx))
    rows.sort(key=lambda t: t[0])
    df = pd.DataFrame(rows, columns=["date", "close"]).set_index("date")
    return df


def _cumulative_realized_series(positions: List[Dict[str, Any]], dt_series: List[datetime]) -> List[float]:
    closed_cashflows = []
    for p in positions:
        for profit_key, date_key in [
            ("call_closed_profit", "call_closed_date"),
            ("put_closed_profit", "put_closed_date"),
        ]:
            pr = p.get(profit_key)
            cd = _to_dt(p.get(date_key))
            if cd is not None and isinstance(pr, (int, float)):
                closed_cashflows.append((datetime(cd.year, cd.month, cd.day), float(pr)))

    by_day: Dict[datetime, float] = {}
    for d, amt in closed_cashflows:
        by_day[d] = by_day.get(d, 0.0) + amt

    out = []
    running = 0.0
    for d in dt_series:
        running += by_day.get(datetime(d.year, d.month, d.day), 0.0)
        out.append(running)
    return out


def _cumulative_unrealized_fallback(pnl_cum_realized: List[float]) -> List[float]:
    # If you don’t have mark-to-market, just mirror realized so the plotter has a line.
    return list(pnl_cum_realized)


def _coerce_position_dates(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow-copied position dict with key date fields coerced to datetime.
    The reference plot expects pos['expiration'] to be a datetime/date object.
    """
    q = dict(p)  # shallow copy
    for k in ("expiration", "position_open_date", "opened_at", "put_closed_date", "call_closed_date"):
        if k in q:
            dt = _to_dt(q.get(k))
            if dt is not None:
                q[k] = dt
    # Ensure 'invalid_data' key exists (some branches check it)
    if "invalid_data" not in q:
        q["invalid_data"] = None
    return q


def _build_daily_results(positions: List[Dict[str, Any]], dt_series: List[datetime]) -> List[Dict[str, Any]]:
    """
    Build the per-day dicts expected by the reference plotter.
    Must include 'date' (YYYY-MM-DD), and positions must have datetime 'expiration'.
    """
    # Coerce date fields on all positions first
    norm = []
    for p in positions:
        q = _coerce_position_dates(p)
        q["_opened"] = _to_dt(q.get("position_open_date") or q.get("opened_at"))
        q["_exp"] = _to_dt(q.get("expiration"))
        norm.append(q)

    out: List[Dict[str, Any]] = []
    for d in dt_series:
        active = []
        for p in norm:
            opened = p["_opened"]
            exp = p["_exp"]
            put_closed = _to_dt(p.get("put_closed_date"))
            call_closed = _to_dt(p.get("call_closed_date"))

            # Active if at least one leg still open and not expired before d
            any_leg_open = not ((put_closed and put_closed <= d) and (call_closed and call_closed <= d))
            not_expired = (exp is None) or (exp.date() >= d.date())
            if (opened is None or opened <= d) and any_leg_open and not_expired:
                active.append(p)

        out.append({
            "date": d.strftime("%Y-%m-%d"),
            "active_positions": active,
            "opened_positions": [],
            "closed_positions": [],
        })
    return out


def _single_parameter_block_from_result(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        "start_date": res.get("start"),
        "end_date":   res.get("end"),
        "trade_type": res.get("trade_type"),
        "expiring_wks": res.get("expiring_wks"),
        "iron_condor_width": res.get("iron_condor_width"),
        "stop_profit_percent": res.get("stop_profit_percent"),
        "vix_correlation": res.get("vix_correlation"),
        "vix_threshold": res.get("vix_threshold"),
        "target_premium_otm": res.get("target_premium_otm"),
        "target_delta": res.get("target_delta"),
        "target_steer": res.get("target_steer"),
    }]


def build_plot_inputs(res: Dict[str, Any],
                      price_df: pd.DataFrame | None = None,
                      vix_df: pd.DataFrame | None = None):
    """
    Convert monthly_recursive_backtest(...) result -> args for plot_recursive_results(...).
    Returns:
      (ticker, final_pnl, daily_results, pnl_cum, pnl_cum_realized,
       start, end, parameter_history, df_dict)
    """
    ticker = res.get("ticker")
    start = res.get("start")
    end = res.get("end")

    dt_series = _date_range(start, end)
    positions = _collect_positions(res)

    pnl_cum_realized = _cumulative_realized_series(positions, dt_series)
    pnl_cum = _cumulative_unrealized_fallback(pnl_cum_realized)

    daily_results = _build_daily_results(positions, dt_series)
    parameter_history = _single_parameter_block_from_result(res)

    df = price_df if price_df is not None else _build_price_df_from_result(res)
    df_dict = {"df": df, "vix_df": (vix_df if vix_df is not None else pd.DataFrame())}

    final_pnl = float(pnl_cum_realized[-1]) if pnl_cum_realized else 0.0

    return (
        ticker,
        final_pnl,
        daily_results,
        pnl_cum,
        pnl_cum_realized,
        start,
        end,
        parameter_history,
        df_dict,
    )


def plot_from_backtest_results(res: Dict[str, Any],
                               *,
                               price_df: pd.DataFrame | None = None,
                               vix_df: pd.DataFrame | None = None):
    """
    Build inputs then call the reference plotter.

    Expected signature for plot_recursive_results:
      (ticker, final_pnl, daily_results, pnl_cumulative_series, pnl_cumulative_realized_series,
       parameter_history, global_start_date, global_end_date, df_dict)
    """
    (ticker, final_pnl, daily_results, pnl_cum, pnl_cum_realized,
     start, end, parameter_history, df_dict) = build_plot_inputs(res, price_df=price_df, vix_df=vix_df)

    return plot_recursive_results(
        ticker,
        final_pnl,
        daily_results,
        pnl_cum,
        pnl_cum_realized,
        parameter_history,
        start,
        end,
        df_dict,
    )
