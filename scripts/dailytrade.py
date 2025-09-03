
# scripts/dailytrade.py
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from typing import Any, Dict

from etrade_quant_trading.polygonio.recursive_backtest import RecursionConfig, backtest_options_sync_or_async
from etrade_quant_trading.polygonio.daily_report import print_opened_and_closed_for_date

def parse_args():
    p = argparse.ArgumentParser(description="Daily trade reporter (opened/closed positions).")
    p.add_argument("-t", "--ticker", required=True, help="Underlying ticker")
    p.add_argument("-s", "--start", required=True, help="Backtest start date YYYY-MM-DD (inclusive)")
    p.add_argument("-e", "--end", required=True, help="Backtest end date YYYY-MM-DD (inclusive)")
    p.add_argument("--trade-type", required=True, help="Strategy type, e.g., 'iron_condor', 'put_credit_spread'")
    p.add_argument("--weekday", default="Friday", help="Target expiry weekday (default: Friday)")
    p.add_argument("--expiring-wks", type=int, default=1, help="Spacing between expiries (default: 1 week)")
    p.add_argument("--qty", type=int, default=1, help="Contracts per position")
    return p.parse_args()

def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def main():
    a = parse_args()
    cfg = RecursionConfig(
        ticker=a.ticker,
        global_start_date=a.start,
        global_end_date=a.end,
        trade_type=a.trade_type,
        expiring_weekday=a.weekday,
        expiring_wks=int(a.expiring_wks),
        contract_qty=int(a.qty),
    )

    # Run the engine once for the whole period
    results: Dict[str, Any] = backtest_options_sync_or_async(cfg)

    # Print opened/closed per day in the window
    start_dt = datetime.strptime(a.start, "%Y-%m-%d")
    end_dt = datetime.strptime(a.end, "%Y-%m-%d")
    for d in daterange(start_dt, end_dt):
        ds = d.strftime("%Y-%m-%d")
        print_opened_and_closed_for_date(results, ds)

if __name__ == "__main__":
    main()
