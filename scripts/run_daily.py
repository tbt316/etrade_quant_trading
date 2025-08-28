from __future__ import annotations
import argparse
from polygonio.logging_setup import init_logging
from polygonio.recursive_backtest import monthly_recursive_backtest

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t","--ticker", required=True)
    p.add_argument("-s","--start", required=True)
    p.add_argument("-e","--end", required=True)
    p.add_argument("--trade-type", required=True,
                   choices=["iron_condor","put_credit_spread","call_credit_spread","covered_call"])
    p.add_argument("--weekday", default="Friday")
    p.add_argument("--expiring-wks", type=int, default=1)
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main():
    a = parse_args()
    init_logging(a.log_level)
    res = monthly_recursive_backtest(
        ticker=a.ticker,
        global_start_date=a.start,
        global_end_date=a.end,
        trade_type=a.trade_type,
        expiring_weekday=a.weekday,
        expiring_wks=a.expiring_wks,
        contract_qty=a.qty,
    )
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
