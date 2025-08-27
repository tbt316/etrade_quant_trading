#!/usr/bin/env python3
"""Simple entry point to start a backtest (replacement for your old async main).

Usage examples
--------------
$ python -m scripts.run_backtest --ticker AAPL --start 2023-01-01 --end 2023-12-31
$ python scripts/run_backtest.py -t SPY -s 2022-01-01 -e 2022-12-31

This uses the same monthly runner you had before (monthly_cpu_bound.run_monthly_backtest_cpu_bound)
through the adapter in polygonio.backtest_core.
"""
from __future__ import annotations
import argparse
import sys

from polygonio.backtest_core import BacktestConfig, run_monthly_backtest_cpu_bound
from polygonio.logging_setup import init_logging


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run monthly backtest")
    p.add_argument("--ticker", "-t", required=True, help="Underlying ticker, e.g. AAPL")
    p.add_argument("--start", "-s", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", "-e", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    init_logging(level=args.log_level)

    cfg = BacktestConfig.from_settings(args.ticker, args.start, args.end)

    # This line preserves your original behavior: it calls the exact same
    # monthly CPU-bound runner function your monolith used.
    result = run_monthly_backtest_cpu_bound(cfg.ticker, cfg.start_date, cfg.end_date)

    # You can print / persist the result here if desired.
    print("Backtest finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

