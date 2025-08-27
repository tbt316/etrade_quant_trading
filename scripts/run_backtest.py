#!/usr/bin/env python3
"""Simple entry point to start a backtest (replacement for your old async main).

Examples:
  python -m scripts.run_backtest -t SPY -s 2022-01-01 -e 2022-12-31 \
    --lookback 5 \
    --hedge 0 0.1 0.2 \
    --multiplier 1 2 \
    --baseline 0.05 0.1 0.15 \
    --expiring-wks 1
"""
from __future__ import annotations
import argparse
import sys
from typing import List

from polygonio.backtest_core import BacktestConfig, run_monthly_backtest_cpu_bound
from polygonio.logging_setup import init_logging
from polygonio.config import get_settings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run monthly backtest")

    # core
    p.add_argument("-t", "--ticker", required=True, help="Underlying ticker, e.g. AAPL")
    p.add_argument("-s", "--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("-e", "--end",   required=True, help="End date YYYY-MM-DD")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    # strategy / sweep knobs (with sensible defaults)
    # NOTE: set defaults to match your typical runs; adjust anytime.
    p.add_argument("--lookback", type=int, default=5, help="Lookback months for the monthly backtest window")
    p.add_argument("--hedge", type=float, nargs="+", default=[0.0], help="List of hedge values to sweep (e.g. 0 0.1 0.2)")
    p.add_argument("--multiplier", type=float, nargs="+", default=[1.0], help="List of position multipliers (e.g. 1 2)")
    p.add_argument("--baseline", type=float, nargs="+", default=[0.05, 0.1, 0.15],
                   help="Target price baseline(s) to test (e.g. 0.05 0.1 0.15)")
    p.add_argument("--expiring-wks", type=int, default=1, help="Expiry spacing in weeks (legacy expiring_wk)")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    init_logging(level=args.log_level)

    # (Optional) pull defaults from config if you prefer central control
    # s = get_settings()
    # if args.lookback is None: args.lookback = s.lookback_months  # if you add to config

    cfg = BacktestConfig.from_settings(args.ticker, args.start, args.end)

    # Pass through all required legacy args:
    result = run_monthly_backtest_cpu_bound(
        cfg.ticker,
        cfg.start_date,
        cfg.end_date,
        kwargs=dict(
            lookback_months=args.lookback,
            hedge_values=args.hedge,
            multiplier_values=args.multiplier,
            target_price_baselines=args.baseline,
            expiring_wks=[args.expiring_wks] if isinstance(args.expiring_wks, int) else args.expiring_wks,
        ),
    )

    print("Backtest finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())