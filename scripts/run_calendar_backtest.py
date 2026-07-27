from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

# Reuse the CLI color scheme from the daily report for consistency
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; ORANGE = "\033[38;5;208m"; CYAN = "\033[96m"

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from polygonio.calendar_backtest import (
    CalendarBacktestConfig,
    run_calendar_backtest,
)
from polygonio.cache_io import save_stored_option_data
from polygonio.config import get_settings
from polygonio.prices import get_historical_prices


def _print_daily(results: dict) -> None:
    daily_rows = results.get("daily_results", []) or []
    if not daily_rows:
        print(f"{DIM}No daily results to display{RESET}")
        return

    trades: list[dict[str, Any]] = []
    for row in daily_rows:
        date_str = row.get("date") or "n/a"
        for trade in row.get("closed", []) or []:
            pnl = trade.get("pnl")
            entry_cash = trade.get("entry_cash")
            if not isinstance(pnl, (int, float)):
                continue
            if not isinstance(entry_cash, (int, float)) or entry_cash <= 0:
                continue
            try:
                ret_pct = float(pnl) / float(entry_cash) * 100.0
            except Exception:
                continue
            trades.append(
                {
                    "ticker": trade.get("ticker") or "?",
                    "id": trade.get("id"),
                    "pnl": float(pnl),
                    "return_pct": ret_pct,
                    "reason": trade.get("reason") or trade.get("close_reason") or "",
                    "open_date": trade.get("open_date") or date_str,
                    "close_date": trade.get("close_date") or date_str,
                    "payload": trade,
                }
            )

    if not trades:
        print(f"{DIM}No closed trades to summarize{RESET}")
        return

    trades.sort(key=lambda t: t["return_pct"])
    losers = trades[:10]
    winners = list(reversed(trades[-10:]))

    def _fmt_iv(iv_val: Any, iv_reason: Any) -> str:
        if isinstance(iv_val, (int, float)):
            return f"{float(iv_val) * 100:.2f}%"
        if iv_reason:
            return f"n/a ({iv_reason})"
        return "n/a"

    def _print_block(title: str, block: list[dict[str, Any]]) -> None:
        print(f"{BOLD}{title}:{RESET}")
        if not block:
            print(f"{DIM}  (none){RESET}")
            return
        for trade in block:
            color = GREEN if trade["pnl"] >= 0 else RED
            pnl_str = f"{trade['pnl']:+.2f}"
            ret_str = f"{trade['return_pct']:+.2f}%"
            reason = trade["reason"] or "n/a"
            qty = trade["payload"].get("qty") or 1
            leg_front = trade["payload"].get("front_option") or "front"
            leg_back = trade["payload"].get("back_option") or "back"
            entry_front = trade["payload"].get("entry_premium_front")
            entry_back = trade["payload"].get("entry_premium_back")
            close_front = trade["payload"].get("close_premium_front")
            close_back = trade["payload"].get("close_premium_back")
            entry_ts_front = trade["payload"].get("entry_ts_front") or "n/a"
            entry_ts_back = trade["payload"].get("entry_ts_back") or "n/a"
            close_ts_front = trade["payload"].get("close_ts_front") or "n/a"
            close_ts_back = trade["payload"].get("close_ts_back") or "n/a"
            entry_iv_front = _fmt_iv(
                trade["payload"].get("entry_iv_front"),
                trade["payload"].get("entry_iv_front_reason"),
            )
            entry_iv_back = _fmt_iv(
                trade["payload"].get("entry_iv_back"),
                trade["payload"].get("entry_iv_back_reason"),
            )
            close_iv_front = _fmt_iv(
                trade["payload"].get("close_iv_front"),
                trade["payload"].get("close_iv_front_reason"),
            )
            close_iv_back = _fmt_iv(
                trade["payload"].get("close_iv_back"),
                trade["payload"].get("close_iv_back_reason"),
            )

            def _fmt_price(val: Any) -> str:
                return f"{float(val):.2f}" if isinstance(val, (int, float)) else "n/a"

            entry_spot = trade["payload"].get("entry_spot")
            close_spot = trade["payload"].get("close_spot")
            print(
                f"{color}  {trade['close_date']} [{trade['ticker']}] id {trade['id']} "
                f"| qty {qty} | pnl {pnl_str} ({ret_str}) | reason {reason}{RESET}"
            )
            print(
                f"    {leg_front}: {_fmt_price(entry_front)}@{entry_ts_front} [{entry_iv_front}] -> "
                f"{_fmt_price(close_front)}@{close_ts_front} [{close_iv_front}]"
            )
            print(
                f"    {leg_back}: {_fmt_price(entry_back)}@{entry_ts_back} [{entry_iv_back}] -> "
                f"{_fmt_price(close_back)}@{close_ts_back} [{close_iv_back}]"
            )
            print(
                f"    Spot: open {_fmt_price(entry_spot)} -> close {_fmt_price(close_spot)}"
            )

    _print_block("Top 10 winning trades (by return %)", winners)
    _print_block("Top 10 losing trades (by return %)", losers)


def _collect_trade_returns(results: dict) -> list[float]:
    rows = results.get("daily_results", []) or []
    returns: list[float] = []
    for row in rows:
        for trade in row.get("closed", []) or []:
            entry_cash = trade.get("entry_cash")
            if not isinstance(entry_cash, (int, float)) or entry_cash <= 0:
                continue
            pnl = trade.get("pnl")
            if not isinstance(pnl, (int, float)):
                continue
            try:
                returns.append(float(pnl) / float(entry_cash) * 100.0)
            except Exception:
                continue
    return returns


def _print_trade_summary(trade_returns: list[float]) -> None:
    if not trade_returns:
        print("Trade summary: no closed trades.")
        return
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100.0
    avg_return = statistics.mean(trade_returns)
    med_return = statistics.median(trade_returns)
    worst_loss = min(trade_returns)
    print(
        f"Trade summary: win rate {win_rate:.1f}% | average return {avg_return:.2f}% "
        f"| median return {med_return:.2f}% | worst loss {worst_loss:.2f}%"
    )


def plot_calendar_results(
    results: dict,
    ticker: str,
    path: Path,
    *,
    trade_returns: list[float] | None = None,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> None:
    if not _HAS_MATPLOTLIB:
        print(f"{RED}Matplotlib not installed; skipping plot generation{RESET}")
        return

    daily = results.get("daily_results") or []
    if not daily:
        print(f"{DIM}No daily data available for plotting{RESET}")
        return

    dates = []
    spots = []
    ffs = []
    forward_vols = []
    front_dtes = []
    back_dtes = []
    gap_dtes = []
    open_counts = []
    realized_cum = []
    realized_pnl_cum = []
    unrealized_total = []
    capital_curve = []
    total_capital_curve = []
    instant_values: list[float] = []

    def _nan_if_none(val: Any) -> float:
        try:
            if val is None:
                return math.nan
            return float(val)
        except Exception:
            return math.nan

    settings = get_settings()
    base_capital = float(getattr(settings, "initial_capital", 0.0) or 0.0)

    for row in daily:
        ds = row.get("date")
        try:
            dt = _dt.strptime(ds, "%Y-%m-%d")
        except Exception:
            continue
        spot_val_raw = row.get("spot")
        try:
            spot_val = float(spot_val_raw)
        except Exception:
            spot_val = None
        if spot_val is None:
            continue
        dates.append(dt)
        spots.append(spot_val)
        ffs.append(_nan_if_none(row.get("ff")))
        forward_vols.append(_nan_if_none(row.get("forward_vol")))
        front_val = _nan_if_none(row.get("front_dte"))
        back_val = _nan_if_none(row.get("back_dte"))
        front_dtes.append(front_val)
        back_dtes.append(back_val)
        gap_val = row.get("dte_gap")
        if isinstance(gap_val, (int, float)):
            gap = float(gap_val)
        elif math.isnan(front_val) or math.isnan(back_val):
            gap = math.nan
        else:
            gap = back_val - front_val
        gap_dtes.append(gap)
        open_counts.append(_nan_if_none(row.get("open_count")))
        # Track the raw engine-provided realized cumulative (affects cash curve)
        realized_raw = row.get("realized_pnl_cumulative")
        if isinstance(realized_raw, (int, float)):
            realized_cash_val = float(realized_raw)
        elif realized_cum:
            realized_cash_val = realized_cum[-1]
        else:
            realized_cash_val = 0.0
        realized_cum.append(realized_cash_val)
        # Build our own cumulative realized P/L purely from trade PnL (ignores entry cash)
        day_realized = row.get("realized_pnl_day")
        if isinstance(day_realized, (int, float)):
            current_realized = (realized_pnl_cum[-1] if realized_pnl_cum else 0.0) + float(day_realized)
        elif realized_pnl_cum:
            current_realized = realized_pnl_cum[-1]
        else:
            current_realized = 0.0
        realized_pnl_cum.append(current_realized)
        unrealized_val = _nan_if_none(row.get("unrealized_pnl_total"))
        unrealized_total.append(unrealized_val)
        try:
            rem_cash = base_capital + realized_cash_val
        except Exception:
            rem_cash = math.nan
        capital_curve.append(_nan_if_none(rem_cash))
        instant_raw = row.get("instant_pnl_cumulative")
        if isinstance(instant_raw, (int, float)):
            current_instant = float(instant_raw)
        elif instant_values:
            current_instant = instant_values[-1]
        else:
            current_instant = realized_cash_val
        instant_values.append(current_instant)
        total_capital_curve.append(_nan_if_none(base_capital + current_instant))

    if not dates:
        print(f"{DIM}No valid dates to plot; skipping plot generation{RESET}")
        return

    start_str = results.get("start") or dates[0].strftime("%Y-%m-%d")
    end_str = results.get("end") or dates[-1].strftime("%Y-%m-%d")

    def _price_map(symbol: str) -> dict:
        try:
            hist = get_historical_prices(symbol, start_str, end_str, data_source="yfinance")
        except Exception:
            return {}
        if hist is None or hist.empty:
            return {}
        out: dict = {}
        for raw_date, close in zip(hist.get("date", []), hist.get("close", [])):
            try:
                d = raw_date.date() if hasattr(raw_date, "date") else _dt.strptime(str(raw_date), "%Y-%m-%d").date()
                out[d] = float(close)
            except Exception:
                continue
        return out

    date_keys = [dt.date() for dt in dates]
    spy_map = _price_map("SPY")
    vix_map = _price_map("^VIX")
    spy_series = [_nan_if_none(spy_map.get(d)) for d in date_keys]
    vix_series = [_nan_if_none(vix_map.get(d)) for d in date_keys]

    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    gs = fig.add_gridspec(6, 1)
    ax_spot = fig.add_subplot(gs[0, 0])
    ax_ff = fig.add_subplot(gs[1, 0], sharex=ax_spot)
    ax_dte = fig.add_subplot(gs[2, 0], sharex=ax_spot)
    ax_positions = fig.add_subplot(gs[3, 0], sharex=ax_spot)
    ax_pnl = fig.add_subplot(gs[4, 0], sharex=ax_spot)
    ax_hist = fig.add_subplot(gs[5, 0])

    for ax in (ax_spot, ax_ff, ax_dte, ax_positions):
        ax.label_outer()

    ax_spot.plot(dates, spots, color="tab:blue", label=f"{ticker.upper()} Spot")
    ax_spot.plot(dates, spy_series, color="tab:orange", linestyle="--", label="SPY Spot")
    ax_spot.set_ylabel("Spot Price")
    ax_spot.set_title(f"{ticker.upper()} Calendar Backtest Overview")
    ax_spot.grid(alpha=0.3)
    ax_vix = ax_spot.twinx()
    ax_vix.plot(dates, vix_series, color="tab:red", label="VIX")
    ax_vix.set_ylabel("VIX")
    handles_spot, labels_spot = ax_spot.get_legend_handles_labels()
    handles_vix, labels_vix = ax_vix.get_legend_handles_labels()
    ax_spot.legend(handles_spot + handles_vix, labels_spot + labels_vix, loc="upper left")

    ax_ff.plot(dates, ffs, color="tab:purple", label="Forward Factor")
    ax_ff.plot(dates, forward_vols, color="tab:green", linestyle="--", label="Forward Vol")
    ax_ff.set_ylabel("FF / Fwd Vol")
    ax_ff.grid(alpha=0.3)
    ax_ff.legend(loc="upper right")

    ax_dte.plot(dates, front_dtes, color="tab:orange", label="Front DTE")
    ax_dte.plot(dates, back_dtes, color="tab:gray", label="Back DTE")
    ax_dte.set_ylabel("DTE")
    ax_dte.grid(alpha=0.3)
    ax_gap = ax_dte.twinx()
    ax_gap.plot(dates, gap_dtes, color="tab:blue", linestyle=":", label="Gap (Back-Front)")
    ax_gap.set_ylabel("Gap (days)")
    ax_gap.grid(False)
    h_dte, l_dte = ax_dte.get_legend_handles_labels()
    h_gap, l_gap = ax_gap.get_legend_handles_labels()
    ax_dte.legend(h_dte + h_gap, l_dte + l_gap, loc="upper right")

    label_positions = "Open Pairs" if "," not in (ticker or "") else "Open Positions"
    ax_positions.plot(dates, open_counts, color="tab:blue", label=label_positions)
    ax_positions.set_ylabel(label_positions)
    ax_positions.grid(alpha=0.3)
    ax_positions.legend(loc="upper left")

    final_realized = realized_pnl_cum[-1] if realized_pnl_cum else 0.0
    # ax_pnl.plot(
    #     dates,
    #     realized_pnl_cum,
    #     color="tab:green",
    #     label=f"Cum Realized P/L (${final_realized:,.0f})",
    # )
    # ax_pnl.plot(
    #     dates,
    #     unrealized_total,
    #     color="tab:red",
    #     linestyle="--",
    #     label="Open P/L",
    # )
    ax_pnl.plot(
        dates,
        capital_curve,
        color="tab:purple",
        linestyle="-.",
        label="Cash Balance",
    )
    ax_pnl.plot(
        dates,
        total_capital_curve,
        color="tab:brown",
        linestyle=":",
        label="Total Capital",
    )
    ax_pnl.set_ylabel("P/L ($)")
    ax_pnl.grid(alpha=0.3)
    ax_pnl.legend(loc="upper left")

    cleaned_realized = [
        val for val in realized_pnl_cum if isinstance(val, (int, float)) and not math.isnan(val)
    ]
    sharpe_text = "Sharpe n/a"
    if len(cleaned_realized) > 1:
        diffs = [
            cleaned_realized[i] - cleaned_realized[i - 1]
            for i in range(1, len(cleaned_realized))
        ]
        if diffs:
            stdev = statistics.pstdev(diffs)
            if stdev > 0:
                sharpe_val = statistics.mean(diffs) / stdev * math.sqrt(252)
                sharpe_text = f"Sharpe ≈ {sharpe_val:.2f}"

    capital_values = [
        val for val in capital_curve if isinstance(val, (int, float)) and not math.isnan(val)
    ]
    max_drawdown = 0.0
    if capital_values:
        running_max = capital_values[0]
        for val in capital_values:
            if val > running_max:
                running_max = val
            drawdown = running_max - val
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    dd_text = f"Max DD ≈ ${max_drawdown:.0f}" if max_drawdown > 0 else "Max DD n/a"

    ax_pnl.text(
        0.02,
        0.05,
        f"{sharpe_text} | {dd_text}",
        transform=ax_pnl.transAxes,
        fontsize=9,
        va="bottom",
    )

    ax_pnl.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pnl.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax_hist.set_title("Trade Return Distribution")
    if trade_returns:
        returns_array = [float(r) for r in trade_returns if isinstance(r, (int, float))]
        if returns_array:
            span_min = math.floor(min(returns_array))
            span_max = math.ceil(max(returns_array))
            if span_min == span_max:
                span_max = span_min + 1
            bin_edges = np.arange(span_min, span_max + 1, 1.0)
            ax_hist.hist(
                returns_array,
                bins=bin_edges,
                color="tab:cyan",
                edgecolor="black",
                alpha=0.7,
            )
        else:
            ax_hist.text(0.5, 0.5, "No trades closed", ha="center", va="center")
    else:
        ax_hist.text(0.5, 0.5, "No trades closed", ha="center", va="center")
    # tp_pct = abs(float(take_profit_pct)) if isinstance(take_profit_pct, (int, float)) else None
    # sl_pct = abs(float(stop_loss_pct)) if isinstance(stop_loss_pct, (int, float)) else None
    tp_pct, sl_pct = None,None
    bound_min = -sl_pct * 100.0 if sl_pct is not None else None
    bound_max = tp_pct * 100.0 if tp_pct is not None else None
    bound_lines = []
    if bound_min is not None or bound_max is not None:
        if bound_min is None and bound_max is not None:
            bound_min = -bound_max
        if bound_max is None and bound_min is not None:
            bound_max = max(bound_min * -1, 0.5)
        if bound_min is not None and bound_max is not None and bound_min != bound_max:
            ax_hist.set_xlim(bound_min, bound_max)
        if bound_max is not None:
            bound_lines.append(
                ax_hist.axvline(
                    bound_max,
                    color="green",
                    linestyle="--",
                    label="Take Profit",
                )
            )
        if bound_min is not None:
            bound_lines.append(
                ax_hist.axvline(
                    bound_min,
                    color="red",
                    linestyle="--",
                    label="Stop Loss",
                )
            )
    if bound_lines:
        ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.set_xlabel("Return (%)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    fig.autofmt_xdate()

    path = path if path.is_absolute() else Path("cover_call_plot") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"{DIM}Saved calendar backtest plot to {path}{RESET}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calendar spread backtest driven by Forward Factor.")
    parser.add_argument("-t", "--ticker", help="Underlying ticker symbol, e.g. SPY")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Explicit list of tickers to scan (overrides --ticker and config pool)",
    )
    parser.add_argument("-s", "--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-e", "--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--qty", type=int, default=1, help="Contracts per entry (default: 1)")
    parser.add_argument("--ff-entry", type=float, default=0.1, help="Forward Factor entry threshold")
    parser.add_argument("--ff-exit", type=float, default=0.0, help="Forward Factor exit threshold")
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit percentage of entry debit (defaults to config)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss percentage of entry debit (defaults to config)",
    )
    parser.add_argument(
        "--front-dte",
        type=int,
        default=None,
        help="Front leg target DTE (defaults to calendar_front_dte_default)",
    )
    parser.add_argument(
        "--back-dte",
        type=int,
        default=None,
        help="Back leg target DTE (defaults to calendar_back_dte_default)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write detailed results",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging output")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a PNG plot summarizing the backtest (saved to cover_call_plot by default)",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional custom path to save the plot (overrides --plot default location)",
    )
    parser.add_argument(
        "--max-daily-positions",
        type=int,
        default=None,
        help="Maximum calendar spreads to open per day (default: calendar_max_daily_positions in config)",
    )
    parser.add_argument(
        "--use-config-pool",
        action="store_true",
        help="Scan the tickers listed in config.calendar_ticker_pool instead of a single --ticker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    ticker_pool: list[str] = []
    if args.tickers:
        ticker_pool = [t.upper() for t in args.tickers if t]
    elif args.use_config_pool or not args.ticker:
        ticker_pool = [t.upper() for t in getattr(settings, "calendar_ticker_pool", []) if t]

    primary_ticker = args.ticker.upper() if args.ticker else (ticker_pool[0] if ticker_pool else None)
    if not primary_ticker:
        print("Error: provide --ticker or configure calendar_ticker_pool/--tickers.")
        return 1
    if ticker_pool and primary_ticker not in ticker_pool:
        ticker_pool.insert(0, primary_ticker)

    max_daily_positions = args.max_daily_positions or getattr(settings, "calendar_max_daily_positions", 1)

    cfg = CalendarBacktestConfig(
        ticker=primary_ticker,
        start_date=args.start,
        end_date=args.end,
        qty=args.qty,
        ff_entry_threshold=args.ff_entry,
        ff_exit_threshold=args.ff_exit,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        front_dte=args.front_dte,
        back_dte=args.back_dte,
        debug=args.debug,
        tickers=ticker_pool or None,
        max_daily_positions=max_daily_positions,
    )
    results = run_calendar_backtest(cfg)
    trade_returns = _collect_trade_returns(results)
    display_label = ",".join(cfg.tickers or [cfg.ticker])
    print(
        f"Calendar backtest for {display_label} "
        f"{cfg.start_date} → {cfg.end_date} | realized ${results.get('realized_total', 0.0):.2f}"
    )
    print(f"Open positions: {len(results.get('open_positions', []))}")
    print(f"Closed positions: {len(results.get('closed_positions', []))}")

    _print_daily(results)
    _print_trade_summary(trade_returns)

    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(results, indent=2))
        print(f"Wrote detailed results to {path}")

    plot_path: Path | None = None
    if args.plot_path:
        plot_path = Path(args.plot_path)
    elif args.plot:
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        label_slug = "_".join(cfg.tickers or [cfg.ticker]).upper()
        plot_path = Path(f"{label_slug}_calendar_backtest_{ts}.png")

    tp_effective = (
        float(cfg.take_profit_pct)
        if cfg.take_profit_pct is not None
        else float(getattr(settings, "calendar_take_profit_pct", 0.3) or 0.0)
    )
    sl_effective = (
        float(cfg.stop_loss_pct)
        if cfg.stop_loss_pct is not None
        else float(getattr(settings, "calendar_stop_loss_pct", 1.0) or 0.0)
    )

    if plot_path is not None:
        plot_calendar_results(
            results,
            display_label,
            plot_path,
            trade_returns=trade_returns,
            take_profit_pct=tp_effective,
            stop_loss_pct=sl_effective,
        )

    tickers_to_save = cfg.tickers or [cfg.ticker]
    for ticker in tickers_to_save:
        try:
            save_stored_option_data(ticker, strike_type="atm")
            print(f"{DIM}Persisted ATM option cache for {ticker}{RESET}")
        except Exception as exc:
            print(f"{RED}[WARN] Failed to persist option cache for {ticker}: {exc}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
