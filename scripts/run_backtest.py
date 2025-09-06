#!/usr/bin/env python3
from __future__ import annotations
import argparse
from polygonio.logging_setup import init_logging
from polygonio.recursive_backtest import monthly_recursive_backtest

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest runner aligned with dailytrade.py logic (monthly_recursive_backtest)."
    )
    p.add_argument("-t","--ticker", required=True, help="Underlying ticker, e.g. AAPL")
    p.add_argument("-s","--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("-e","--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--trade-type", required=True,
                   choices=["iron_condor","put_credit_spread","call_credit_spread","covered_call"])
    p.add_argument("--weekday", default="Friday", help="Which weekday to target for expiries (e.g., Friday)")
    p.add_argument("--expiring-wks", type=int, default=1, help="Weeks out for option expiry (1=next week)")
    p.add_argument("--qty", type=int, default=1, help="Contracts per leg")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--plot", action="store_true", help="Display a simple plot of results")
    p.add_argument("--save-plot", help="If provided, save plot image to this path")
    return p.parse_args()

def main():
    a = parse_args()
    init_logging(a.log_level)

    # Delegate exactly like dailytrade.py / run_daily.py
    res = monthly_recursive_backtest(
        ticker=a.ticker,
        global_start_date=a.start,
        global_end_date=a.end,
        trade_type=a.trade_type,
        expiring_weekday=a.weekday,
        expiring_wks=a.expiring_wks,
        contract_qty=a.qty,
    )
    if a.plot or a.save_plot:
        from polygonio.daily_report import plot_recursive_results
        plot_recursive_results(res, show=a.plot, save_path=a.save_plot)
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
