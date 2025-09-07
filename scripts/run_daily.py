# etrade_quant_trading/scripts/run_daily.py
# Note for Codex: install prerequisite packages specified in requirements.txt before running this script.
# Note for Codex: Use test_integration_run_daily.py for testing the changes, which will avoid calling 3rd part API. Do not use run_daily.py for testing.

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from polygonio.logging_setup import init_logging
from polygonio.recursive_backtest import monthly_recursive_backtest
from polygonio.daily_report import print_opened_and_closed_for_date
from polygonio.plot_results import plot_from_backtest_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run monthly_recursive_backtest and (optionally) plot results."
    )
    # Core backtest window
    p.add_argument("-t", "--ticker", required=True, help="Underlying ticker, e.g. SPY")
    p.add_argument("-s", "--start", required=True, help="Global start date, YYYY-MM-DD")
    p.add_argument("-e", "--end", required=True, help="Global end date, YYYY-MM-DD")

    # Strategy selection and knobs (mirror your engine’s parameters)
    p.add_argument("--trade-type", required=True, help="e.g. iron_condor, put_credit_spread, call_credit_spread")

    p.add_argument(
        "--weekday",
        dest="weekday",
        default="Friday",
        choices=["Friday", "Wednesday"],
        help="Which weekday expirations to target (default: Friday)",
    )
    p.add_argument(
        "--expiring-wks",
        dest="expiring_wks",
        type=int,
        default=1,
        help="Spacing between expiries in weeks (default: 1)",
    )
    p.add_argument(
        "--qty",
        dest="qty",
        type=int,
        default=1,
        help="Contracts per position (default: 1)",
    )
    p.add_argument(
        "--iron-condor-width",
        dest="iron_condor_width",
        type=float,
        default=None,
        help="Wing width for iron condor (points). If omitted, engine default is used.",
    )

    # Targeting / selection knobs (only applied if your engine uses them)
    p.add_argument(
        "--target-premium-otm",
        dest="target_premium_otm",
        type=float,
        default=None,
        help="Target premium (in $) or OTM-based premium target if your engine supports it.",
    )
    p.add_argument(
        "--target-delta",
        dest="target_delta",
        type=float,
        default=None,
        help="Target short-leg delta (e.g., 0.15).",
    )
    p.add_argument(
        "--target-steer",
        dest="target_steer",
        type=float,
        default=None,
        help="Optional steering factor used by your selector (engine-specific).",
    )

    # Risk management knobs
    p.add_argument(
        "--stop-profit-percent",
        dest="stop_profit_percent",
        type=float,
        default=None,
        help="Take-profit threshold (e.g., 0.5 = +50% profit on credit).",
    )
    p.add_argument(
        "--stop-loss-action",
        dest="stop_loss_action",
        default=None,
        help="Engine-specific stop-loss behavior keyword (e.g., 'close', 'hold').",
    )

    # Optional VIX adjustments
    p.add_argument(
        "--vix-threshold",
        dest="vix_threshold",
        type=float,
        default=None,
        help="Baseline VIX level used for premium adjustments.",
    )
    p.add_argument(
        "--vix-correlation",
        dest="vix_correlation",
        type=float,
        default=None,
        help="Correlation factor applied to premium targets when VIX deviates from the threshold (e.g., 0.05).",
    )

    # Plotting & logging
    p.add_argument(
        "--plot",
        action="store_true",
        help="After the backtest, render the reference-style plot using plot_recursive_results.",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    init_logging(level=args.log_level)

    # Run the backtest
    res = monthly_recursive_backtest(
        ticker=args.ticker,
        global_start_date=args.start,
        global_end_date=args.end,
        trade_type=args.trade_type,
        expiring_weekday=args.weekday,
        expiring_wks=args.expiring_wks,
        contract_qty=args.qty,
        iron_condor_width=args.iron_condor_width,
        target_premium_otm=args.target_premium_otm,
        target_delta=args.target_delta,
        target_steer=args.target_steer,
        stop_profit_percent=args.stop_profit_percent,
        stop_loss_action=args.stop_loss_action,
        vix_threshold=args.vix_threshold,
        vix_correlation=args.vix_correlation,
    )

    # Print opened/closed positions for each day in range
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        cur = start_dt
        while cur <= end_dt:
            ds = cur.strftime("%Y-%m-%d")
            print_opened_and_closed_for_date(res, ds)
            cur += timedelta(days=1)
    except Exception as e:
        print(f"[WARN] Daily report printing failed: {e}")

    plot_from_backtest_results(res)


if __name__ == "__main__":
    sys.exit(main())