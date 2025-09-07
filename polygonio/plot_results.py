# etrade_quant_trading/polygonio/plot_results.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, date, timedelta
import os
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

PLOT_DIR = "cover_call_plots"


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

        required_margin = 0.0
        for p in active:
            try:
                required_margin += float(p.get("required_margin") or 0.0)
            except Exception:
                required_margin += 0.0

        out.append({
            "date": d.strftime("%Y-%m-%d"),
            "active_positions": active,
            "opened_positions": [],
            "closed_positions": [],
            "required_margin": required_margin,
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
    Returns a tuple ordered exactly as ``plot_recursive_results`` expects:

    ``(ticker, final_pnl, daily_results, pnl_cum, pnl_cum_realized,
    parameter_history, start, end, df_dict)``
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
        parameter_history,
        start,
        end,
        df_dict,
    )


def plot_recursive_results(
    ticker: str,
    final_pnl: float,
    daily_results: List[Dict[str, Any]],
    pnl_cumulative_series: List[float],
    pnl_cumulative_realized_series: List[float],
    parameter_history: List[Dict[str, Any]],
    global_start_date: str,
    global_end_date: str,
    df_dict: Dict[str, pd.DataFrame],
) -> plt.Figure:
    """Create an eight-panel diagnostic plot and save it to disk.

    The layout and content mirror the heavy plotting function from the
    reference project.  The generated figure is written as a ``.png`` to
    :data:`PLOT_DIR` and the :class:`~matplotlib.figure.Figure` instance is
    returned for further inspection or testing.
    """

    # ------------------------------ fonts ---------------------------------
    _SCALE = 2
    size_keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ]
    # Use Matplotlib's default values as the baseline so repeated calls do not
    # exponentially scale the fonts.
    base_rc = plt.rcParamsDefault
    new_params: Dict[str, float] = {}
    for k in size_keys:
        v = base_rc.get(k, plt.rcParams.get(k))
        if isinstance(v, (int, float)):
            new_params[k] = v * _SCALE
        else:
            new_params[k] = FontProperties(size=v).get_size_in_points() * _SCALE
    plt.rcParams.update(new_params)

    def _fs(x: float) -> float:
        return x * _SCALE

    # --------------------------- basic series -----------------------------
    dt_series = [pd.to_datetime(r["date"]) for r in daily_results]
    price_df = df_dict.get("df", pd.DataFrame()).sort_index()
    vix_df = df_dict.get("vix_df", pd.DataFrame())

    required_margins = [r.get("required_margin", 0.0) for r in daily_results]
    req_series = pd.Series(required_margins, dtype="float64")
    cum_max_margin = req_series.expanding().max()

    pnl_series = pd.Series(pnl_cumulative_realized_series, dtype="float64")
    day_count = pd.Series(np.arange(1, len(pnl_series) + 1), dtype="float64")
    cumulative_ann_return = (
        pnl_series.div(cum_max_margin.replace(0, np.nan))
        .div(day_count)
        .mul(252)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    if len(cumulative_ann_return) >= 100:
        cumulative_ann_return[:100] = 0.0

    # ------------------------ per-day aggregations ------------------------
    n = len(dt_series)
    open_distance_calls: List[List[float]] = [[] for _ in range(n)]
    open_distance_puts: List[List[float]] = [[] for _ in range(n)]
    open_premium_calls: List[List[float]] = [[] for _ in range(n)]
    open_premium_puts: List[List[float]] = [[] for _ in range(n)]
    target_premium_calls: List[List[float]] = [[] for _ in range(n)]
    target_premium_puts: List[List[float]] = [[] for _ in range(n)]
    days_to_expiry_array: List[List[Optional[timedelta]]] = [[] for _ in range(n)]
    spread_width_array: List[List[float]] = [[] for _ in range(n)]
    call_closed_profit_array: List[List[float]] = [[] for _ in range(n)]
    put_closed_profit_array: List[List[float]] = [[] for _ in range(n)]

    call_dates: List[datetime] = []
    call_instance_distances: List[float] = []
    put_dates: List[datetime] = []
    put_instance_distances: List[float] = []
    min_distances: List[float] = []
    highlight_dates: List[datetime] = []
    highlight_distances: List[float] = []
    itm_amounts: List[float] = []
    itm_dates: List[datetime] = []
    itm_days_open: List[int] = []
    otm_dates: List[datetime] = []
    otm_days_open: List[int] = []
    otm_open_counts = [0] * n
    itm_now_otm_open_counts = [0] * n
    itm_open_counts = [0] * n
    iv_call_data: List[float] = []
    iv_put_data: List[float] = []

    close_lookup = price_df.get("close", pd.Series()).to_dict()
    get_close = close_lookup.get

    for idx, (date_dt, day) in enumerate(zip(dt_series, daily_results)):
        close_price = get_close(date_dt)
        if close_price is None or np.isnan(close_price):
            close_price = np.nan

        active_positions = day.get("active_positions", [])
        distances_today: List[float] = []
        itm_amount = 0.0
        opened_iv_calls: List[float] = []
        opened_iv_puts: List[float] = []

        for pos in active_positions:
            open_date = _to_dt(pos.get("position_open_date"))
            exp_date = _to_dt(pos.get("expiration"))
            call_closed = _to_dt(pos.get("call_closed_date"))
            put_closed = _to_dt(pos.get("put_closed_date"))

            short_call = pos.get("short_call_prem_open", 0) > 0
            short_put = pos.get("short_put_prem_open", 0) > 0

            if short_call and pos.get("call_strike_sold") is not None and call_closed is None and not np.isnan(close_price):
                strike = pos["call_strike_sold"]
                dist = (strike - close_price) / close_price * 100
                distances_today.append(dist)
                call_dates.append(date_dt)
                call_instance_distances.append(dist)
                if open_date and open_date.date() == date_dt.date():
                    open_distance_calls[idx].append((pos.get("open_distance_call") or 0) * 100)
                    open_premium_calls[idx].append(pos.get("short_call_prem_open", 0) - pos.get("long_call_prem_open", 0))
                    target_premium_calls[idx].append((pos.get("strike_target_call") or {}).get("premium_target"))
                    days_to_expiry_array[idx].append(exp_date - open_date if exp_date and open_date else None)
                    spread_width_array[idx].append(pos.get("call_strike_bought", 0) - pos.get("call_strike_sold", 0))

            if short_put and pos.get("put_strike_sold") is not None and put_closed is None and not np.isnan(close_price):
                strike = pos["put_strike_sold"]
                dist = -((strike - close_price) / close_price) * 100
                distances_today.append(dist)
                put_dates.append(date_dt)
                put_instance_distances.append(dist)
                if open_date and open_date.date() == date_dt.date():
                    open_distance_puts[idx].append((pos.get("open_distance_put") or 0) * 100)
                    open_premium_puts[idx].append(pos.get("short_put_prem_open", 0) - pos.get("long_put_prem_open", 0))
                    target_premium_puts[idx].append((pos.get("strike_target_put") or {}).get("premium_target"))
                    days_to_expiry_array[idx].append(exp_date - open_date if exp_date and open_date else None)
                    spread_width_array[idx].append(pos.get("put_strike_sold", 0) - pos.get("put_strike_bought", 0))

            if short_call and call_closed is not None and call_closed.date() == date_dt.date():
                call_closed_profit_array[idx].append(pos.get("call_closed_profit"))
            if short_put and put_closed is not None and put_closed.date() == date_dt.date():
                put_closed_profit_array[idx].append(pos.get("put_closed_profit"))

            if open_date and ((short_call and call_closed is None) or (short_put and put_closed is None)):
                days_open = (date_dt.date() - open_date.date()).days
                is_itm = False
                if short_call and pos.get("call_strike_sold") is not None and not np.isnan(close_price):
                    if (pos["call_strike_sold"] - close_price) / close_price * 100 < 0:
                        is_itm = True
                if short_put and pos.get("put_strike_sold") is not None and not np.isnan(close_price):
                    if -((pos["put_strike_sold"] - close_price) / close_price) * 100 < 0:
                        is_itm = True
                (itm_dates if is_itm else otm_dates).append(date_dt)
                (itm_days_open if is_itm else otm_days_open).append(days_open)

            if short_call and pos.get("call_strike_sold") is not None and not np.isnan(close_price):
                open_dist = (pos.get("open_distance_call") or 0) * 100
                current_dist = (pos["call_strike_sold"] - close_price) / close_price * 100
                if open_dist > 0:
                    (otm_open_counts if current_dist >= 0 else itm_now_otm_open_counts)[idx] += 1
                elif open_dist < 0:
                    itm_open_counts[idx] += 1
            if short_put and pos.get("put_strike_sold") is not None and not np.isnan(close_price):
                open_dist = (pos.get("open_distance_put") or 0) * 100
                current_dist = -((pos["put_strike_sold"] - close_price) / close_price) * 100
                if open_dist > 0:
                    (otm_open_counts if current_dist >= 0 else itm_now_otm_open_counts)[idx] += 1
                elif open_dist < 0:
                    itm_open_counts[idx] += 1

            if open_date and open_date.date() == date_dt.date():
                if isinstance(pos.get("iv_call"), (int, float)):
                    opened_iv_calls.append(pos["iv_call"])
                if isinstance(pos.get("iv_put"), (int, float)):
                    opened_iv_puts.append(pos["iv_put"])

            if exp_date and exp_date.date() == date_dt.date() and not np.isnan(close_price):
                if short_call and pos.get("call_strike_sold") is not None:
                    distc = (pos["call_strike_sold"] - close_price) / close_price * 100
                    if distc < 0:
                        itm_amount += max(close_price - pos["call_strike_sold"], 0) * 100
                if short_put and pos.get("put_strike_sold") is not None:
                    distp = -((pos["put_strike_sold"] - close_price) / close_price) * 100
                    if distp < 0:
                        itm_amount += max(pos["put_strike_sold"] - close_price, 0) * 100

        iv_call_data.append(np.mean(opened_iv_calls) if opened_iv_calls else np.nan)
        iv_put_data.append(np.mean(opened_iv_puts) if opened_iv_puts else np.nan)

        if distances_today:
            md = min(distances_today, key=abs)
            min_distances.append(md)
            if md < 0 and exp_date and exp_date.date() == date_dt.date():
                highlight_dates.append(date_dt)
                highlight_distances.append(md)
        else:
            min_distances.append(np.nan)
        itm_amounts.append(itm_amount if itm_amount else np.nan)

    # ---------------- strike mapping for price panel ----------------------
    sb_x: List[datetime] = []
    sb_y: List[float] = []
    pb_x: List[datetime] = []
    pb_y: List[float] = []
    call_dates_close: List[datetime] = []
    call_strikes: List[float] = []
    put_dates_close: List[datetime] = []
    put_strikes: List[float] = []
    call_open_close_strikes: List[tuple] = []
    put_open_close_strikes: List[tuple] = []

    unique_positions = {
        (
            pos.get("position_open_date"),
            pos.get("call_closed_date"),
            pos.get("put_closed_date"),
        ): pos
        for day in daily_results
        for pos in day.get("active_positions", [])
        if pos.get("expiration") is not None
    }

    for pos in unique_positions.values():
        open_date = _to_dt(pos.get("position_open_date"))
        if open_date is None:
            continue
        if pos.get("call_closed_date") is not None and pos.get("call_strike_sold") is not None:
            cd = _to_dt(pos["call_closed_date"])
            call_dates_close.append(cd)
            call_strikes.append(pos["call_strike_sold"])
            call_open_close_strikes.append((open_date, cd, pos["call_strike_sold"]))
            if pos.get("long_call_prem_open", 0) > 0 and pos.get("call_strike_bought"):
                sb_x.append(cd)
                sb_y.append(pos["call_strike_bought"])
        if pos.get("put_closed_date") is not None and pos.get("put_strike_sold") is not None:
            pclose = _to_dt(pos["put_closed_date"])
            put_dates_close.append(pclose)
            put_strikes.append(pos["put_strike_sold"])
            put_open_close_strikes.append((open_date, pclose, pos["put_strike_sold"]))
            if pos.get("long_put_prem_open", 0) > 0 and pos.get("put_strike_bought"):
                pb_x.append(pclose)
                pb_y.append(pos["put_strike_bought"])

    # ----------------------------- plotting -------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(70, 40))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

    # ax1: cumulative PnL and cumulative annualised return
    if dt_series and pnl_cumulative_series:
        ax1.plot(dt_series, pnl_cumulative_series, label=f"Cumulative PnL (Final={pnl_cumulative_series[-1]:.2f})", color="black", marker=".", linewidth=1, markersize=3)
        ax1.plot(dt_series, pnl_cumulative_realized_series, label=f"Cumulative PnL Realized (Final={pnl_cumulative_realized_series[-1]:.2f})", color="blue", marker=".", linewidth=1, markersize=3)
        ax1.set_ylabel("PnL ($)", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.legend(loc="upper left", fontsize=_fs(8))
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dt_series, cumulative_ann_return * 100, label="Cumulative Annualized Return", color="green", linewidth=1, marker=".", markersize=3)
        ax1_twin.set_ylabel("Cumulative Annualized Return (%)", color="green")
        ax1_twin.tick_params(axis="y", labelcolor="green")
        ax1_twin.legend(loc="upper right", fontsize=_fs(8))
        ax1.set_title(f"Cumulative PnL & Cumulative Annualized Return for {ticker}")
        ax1.set_xlabel("Date")
        ax1.grid(True)
    else:
        ax1.text(0.5, 0.5, "No PnL data to plot", transform=ax1.transAxes, ha="center")

    # ax2: closest strike distance & ITM amount
    if dt_series and min_distances and any(not np.isnan(d) for d in min_distances):
        ax2.plot(dt_series, min_distances, color="green", label="Min Distance to Open Options (OTM:+ / ITM:-)", linewidth=1, marker="o", markersize=1)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        if highlight_dates:
            ax2.scatter(highlight_dates, highlight_distances, color="red", marker="o", s=30, label="ITM Options Expiring Friday", rasterized=True)
        if call_dates:
            ax2.scatter(call_dates, call_instance_distances, color="red", s=2, alpha=0.3, zorder=1, rasterized=True)
        if put_dates:
            ax2.scatter(put_dates, put_instance_distances, color="blue", s=2, alpha=0.3, zorder=1, rasterized=True)

        def _flatten(dates, lists):
            d, v = [], []
            for dt, lst in zip(dates, lists):
                d.extend([dt] * len(lst))
                v.extend(lst)
            return d, v

        d_call, v_call = _flatten(dt_series, open_distance_calls)
        d_put, v_put = _flatten(dt_series, open_distance_puts)
        if d_call:
            ax2.scatter(d_call, v_call, color="orange", marker="^", s=30, alpha=0.6, label="Call Distances", rasterized=True)
        if d_put:
            ax2.scatter(d_put, v_put, color="purple", marker="v", s=30, alpha=0.6, label="Put Distances", rasterized=True)

        ax2.set_title(f"Closest Option-Strike Distance & ITM Amount for {ticker}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Distance (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.grid(True)
        ax2.legend(loc="upper left", fontsize=_fs(8))
        ax2_twin = ax2.twinx()
        ax2_twin.plot(dt_series, itm_amounts, color="red", label="Aggregate ITM Amount (Today Exp.)", linewidth=1, marker="o", markersize=10)
        ax2_twin.set_ylabel("ITM Amount ($)", color="blue")
        ax2_twin.tick_params(axis="y", labelcolor="blue")
        ax2_twin.legend(loc="upper right", fontsize=_fs(8))
    else:
        ax2.text(0.5, 0.5, "No open option distances to plot", transform=ax2.transAxes, ha="center")

    # ax3: underlying price + option strikes
    all_plot_dates = dt_series + call_dates_close + put_dates_close + sb_x + pb_x
    if all_plot_dates:
        start_plot = min(d for d in all_plot_dates if d is not None)
        end_plot = max(d for d in all_plot_dates if d is not None)
        daily_close_filtered = price_df["close"].loc[start_plot:end_plot]
        if not daily_close_filtered.empty:
            ax3.plot(daily_close_filtered.index, daily_close_filtered.values, color="black", label="Underlying Close", linewidth=1)
            for o, c, s in call_open_close_strikes:
                ax3.plot([o, c], [s, s], color="red", linestyle="--", linewidth=0.5)
            for o, c, s in put_open_close_strikes:
                ax3.plot([o, c], [s, s], color="blue", linestyle="--", linewidth=0.5)
            ax3.scatter([o for o, _, _ in call_open_close_strikes], [s for _, _, s in call_open_close_strikes], color="green", marker="o", s=10, label="Short Call Open", rasterized=True)
            ax3.scatter([o for o, _, _ in put_open_close_strikes], [s for _, _, s in put_open_close_strikes], color="red", marker="o", s=10, label="Short Put Open", rasterized=True)
            ax3.scatter(call_dates_close, call_strikes, color="purple", marker="s", s=10, label="Short Call Close", rasterized=True)
            ax3.scatter(put_dates_close, put_strikes, color="blue", marker="s", s=10, label="Short Put Close", rasterized=True)
            ax3.scatter(sb_x, sb_y, color="green", marker="^", s=5, label="Long Call Close", rasterized=True)
            ax3.scatter(pb_x, pb_y, color="red", marker="^", s=5, label="Long Put Close", rasterized=True)
            ax3.legend(loc="upper left", fontsize=_fs(8), framealpha=0.9)
            ax3.set_title("Option Strikes at Open and Close vs. Underlying Price")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Price")
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "No underlying data in range", transform=ax3.transAxes, ha="center")
    else:
        ax3.text(0.5, 0.5, "No strike data to plot", transform=ax3.transAxes, ha="center")

    # ax4: required margin & days open
    avg_margin = np.mean(required_margins) if required_margins else 0
    max_margin = np.max(required_margins) if required_margins else 0
    ax4.plot(dt_series, required_margins, color="green", label=f"Required Margin, avg={avg_margin:.0f}, max={max_margin:.0f}", linewidth=1, marker=".", markersize=3)
    ax4.set_title("Required Margin & Days Open")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Margin ($)", color="green")
    ax4.tick_params(axis="y", labelcolor="green")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend(loc="upper left", fontsize=_fs(8))
    ax4_twin = ax4.twinx()
    if itm_dates:
        ax4_twin.scatter(itm_dates, itm_days_open, color="red", marker="o", s=5, alpha=0.6, label="Days Open (ITM)", rasterized=True)
    if otm_dates:
        ax4_twin.scatter(otm_dates, otm_days_open, color="blue", marker="o", s=5, alpha=0.6, label="Days Open (OTM)", rasterized=True)
    ax4_twin.set_ylabel("Days Open", color="blue")
    ax4_twin.tick_params(axis="y", labelcolor="blue")
    ax4_twin.legend(loc="upper right", fontsize=_fs(8))

    # ax5: open premiums & position counts
    def _flatten_lists(date_list, list_of_lists):
        d, v = [], []
        for dte, lst in zip(date_list, list_of_lists):
            d.extend([dte] * len(lst))
            v.extend(lst)
        return d, v

    oc_dates, oc_vals = _flatten_lists(dt_series, open_premium_calls)
    tc_dates, tc_vals = _flatten_lists(dt_series, target_premium_calls)
    op_dates, op_vals = _flatten_lists(dt_series, open_premium_puts)
    tp_dates, tp_vals = _flatten_lists(dt_series, target_premium_puts)

    if oc_dates:
        ax5.scatter(oc_dates, oc_vals, marker="o", s=30, alpha=0.6, color="tab:blue", label="Open Call Premiums", rasterized=True)
    if tc_dates:
        ax5.scatter(tc_dates, tc_vals, marker="o", s=30, alpha=0.6, color="tab:red", label="Target Call Premiums", rasterized=True)
    if op_dates:
        ax5.scatter(op_dates, op_vals, marker="s", s=30, alpha=0.6, color="tab:orange", label="Open Put Premiums", rasterized=True)
    if tp_dates:
        ax5.scatter(tp_dates, tp_vals, marker="s", s=30, alpha=0.6, color="tab:purple", label="Target Put Premiums", rasterized=True)

    ax5.set_title("Open Premiums and Number of Opened Positions")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Premium ($)", color="black")
    ax5.tick_params(axis="y", labelcolor="black")
    ax5.grid(True)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(dt_series, otm_open_counts, color="blue", label="Positions OTM at Open", linewidth=1, linestyle="-.")
    ax5_twin.plot(dt_series, itm_now_otm_open_counts, color="red", label="Positions ITM Now, OTM at Open", linewidth=1, linestyle="--")
    ax5_twin.plot(dt_series, itm_open_counts, color="green", label="Positions ITM at Open", linewidth=2)
    ax5_twin.set_ylabel("Number of Positions Opened", color="black")
    ax5_twin.tick_params(axis="y", labelcolor="black")
    ax5_twin.legend(loc="upper right", fontsize=_fs(8))
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=_fs(8), framealpha=0.9)

    # ax6: VIX & IV
    if not vix_df.empty or any(~np.isnan(iv) for iv in iv_call_data + iv_put_data):
        vix_start = min(dt_series) if dt_series else vix_df.index.min()
        vix_end = max(dt_series) if dt_series else vix_df.index.max()
        ax6_twin = ax6.twinx()
        if any(~np.isnan(iv) for iv in iv_call_data):
            ax6_twin.plot(dt_series, iv_call_data, color="red", label="Call IV", linewidth=1, marker=".", markersize=3)
        if any(~np.isnan(iv) for iv in iv_put_data):
            ax6_twin.plot(dt_series, iv_put_data, color="blue", label="Put IV", linewidth=1, marker=".", markersize=3)
        ax6_twin.set_ylabel("Implied Volatility", color="black")
        ax6_twin.tick_params(axis="y", labelcolor="black")
        if any(~np.isnan(iv) for iv in iv_call_data) or any(~np.isnan(iv) for iv in iv_put_data):
            ax6_twin.legend(loc="upper right", fontsize=_fs(8))
        if not vix_df.empty:
            if not isinstance(vix_df.index, pd.DatetimeIndex):
                vix_df["date"] = pd.to_datetime(vix_df["date"])
                vix_df = vix_df.set_index("date")
            vix_filtered = vix_df.loc[vix_start:vix_end]
            if not vix_filtered.empty:
                ax6.plot(vix_filtered.index, vix_filtered["close"], color="purple", label="VIX Index", linewidth=1, marker=".", markersize=3)
                ax6.set_ylabel("VIX", color="purple")
                ax6.tick_params(axis="y", labelcolor="purple")
                ax6.legend(loc="upper left", fontsize=_fs(8))
        ax6.set_title(f"VIX Index and Implied Volatility for {ticker}")
        ax6.set_xlabel("Date")
        ax6.grid(True)
    else:
        ax6.text(0.5, 0.5, "No VIX or IV data to plot", transform=ax6.transAxes, ha="center")

    # ax7: Spread Width and Closed Profits
    sw_dates, sw_vals = _flatten_lists(dt_series, spread_width_array)
    put_profit_dates, put_profit_vals = _flatten_lists(dt_series, put_closed_profit_array)
    call_profit_dates, call_profit_vals = _flatten_lists(dt_series, call_closed_profit_array)
    ax7_twin = None
    if sw_dates or put_profit_dates or call_profit_dates:
        if sw_dates:
            ax7.scatter(sw_dates, sw_vals, color="blue", marker="o", s=30, alpha=0.6, label="Spread Width", rasterized=True)
        ax7.set_ylabel("Spread Width ($)", color="blue")
        ax7.tick_params(axis="y", labelcolor="blue")
        ax7_twin = ax7.twinx()
        if call_profit_dates:
            call_colors = ["green" if val > 0 else "red" for val in call_profit_vals]
            ax7_twin.scatter(call_profit_dates, call_profit_vals, c=call_colors, marker="^", s=30, alpha=0.6, label="Call Closed Profit", rasterized=True)
        if put_profit_dates:
            put_colors = ["green" if val > 0 else "red" for val in put_profit_vals]
            ax7_twin.scatter(put_profit_dates, put_profit_vals, c=put_colors, marker="s", s=30, alpha=0.6, label="Put Closed Profit", rasterized=True)
        ax7_twin.set_ylabel("Closed Profit ($)", color="black")
        ax7_twin.tick_params(axis="y", labelcolor="black")
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_twin.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=_fs(8), framealpha=0.9)
        ax7.set_title("Spread Width and Closed Profits")
        ax7.set_xlabel("Date")
        ax7.grid(True)
    else:
        ax7.text(0.5, 0.5, "No spread width or profit data", transform=ax7.transAxes, ha="center")

    # ax8: days to expiry
    dte_dates, dte_vals = [], []
    for d, lst in zip(dt_series, days_to_expiry_array):
        dte_dates.extend([d] * len(lst))
        dte_vals.extend([td.days for td in lst if td is not None])
    if dte_dates:
        ax8.scatter(dte_dates, dte_vals, color="orange", marker="o", s=30, alpha=0.6, label="Days to Expiry", rasterized=True)
        ax8.set_title("Days to Expiry")
        ax8.set_xlabel("Date")
        ax8.set_ylabel("Days")
        ax8.legend(fontsize=_fs(8))
        ax8.grid(True)
    else:
        ax8.text(0.5, 0.5, "No days to expiry data", transform=ax8.transAxes, ha="center")

    # --------------------------- x-axis formatting ------------------------
    if dt_series:
        min_date, max_date = min(dt_series), max(dt_series)
        for a in [ax1, ax2, ax3, ax4, ax5, ax5_twin, ax6, ax7, ax7_twin, ax8]:
            if a is not None:
                a.set_xlim(min_date, max_date)
                a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                for lbl in a.get_xticklabels():
                    lbl.set_rotation(45)
                    lbl.set_horizontalalignment("right")
    else:
        print("No dt_series available for x-axis formatting")

    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.20, wspace=0.15, top=0.96, bottom=0.04, left=0.04, right=0.98)

    # Shade parameter periods
    num_periods = len(parameter_history)
    colors = plt.cm.viridis(np.linspace(0, 1, num_periods)) if num_periods > 0 else []
    for i, params in enumerate(parameter_history):
        try:
            p_start = datetime.strptime(params["start_date"], "%Y-%m-%d")
            p_end = datetime.strptime(params["end_date"], "%Y-%m-%d")
            for a in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax7_twin, ax8]:
                if a is not None:
                    a.axvspan(p_start, p_end, alpha=0.15, color=colors[i % len(colors)])
        except Exception:
            pass

    fig.autofmt_xdate()
    plt.tight_layout(pad=3.0)

    os.makedirs(PLOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = os.path.join(PLOT_DIR, f"recursive_monthly_{ticker}_{global_start_date}_to_{global_end_date}_{timestamp}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=200)
    return fig


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
    args = build_plot_inputs(res, price_df=price_df, vix_df=vix_df)
    return plot_recursive_results(*args)
