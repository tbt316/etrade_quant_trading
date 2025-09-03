#!/usr/bin/env python3
from __future__ import annotations
import argparse
from polygonio.logging_setup import init_logging
from polygonio.recursive_backtest import monthly_recursive_backtest
from polygonio.daily_report import print_opened_and_closed_for_date

def parse_args():
    p = argparse.ArgumentParser(
        description="Run daily backtest with parity to dailytrade.py parameters."
    )
    # Required core args
    p.add_argument("-t","--ticker", required=True, help="Underlying ticker, e.g. SPY")
    p.add_argument("-s","--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("-e","--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--trade-type", required=True,
                   choices=["iron_condor","put_credit_spread","call_credit_spread","covered_call"])

    # Existing knobs
    p.add_argument("--weekday", default="Friday", help="Which weekday to target for expiries (e.g., Friday)")
    p.add_argument("--expiring-wks", type=int, default=1, help="Weeks out for option expiry (1=next week)")
    p.add_argument("--qty", type=int, default=1, help="Contracts per leg")

    # New parity args from dailytrade.py
    p.add_argument("--iron-condor-width", type=float, default=5.0,
                   help="Width for iron condor/spreads (strike distance)")
    p.add_argument("--target-premium-otm", type=float, default=None,
                   help="Target premium (OTM) used to set short strike(s)")
    p.add_argument("--target-delta", type=float, default=None,
                   help="Accepted but NOT used for strike selection (parity only)")
    p.add_argument("--target-steer", type=float, default=0.0,
                   help="Bias target premium between puts(+)/calls(-). "
                        "put target = otm*(1+steer), call target = otm*(1-steer)")
    p.add_argument("--stop-profit-percent", type=float, default=None,
                   help="Take-profit percent of initial credit, e.g., 0.5 = 50%")
    p.add_argument("--stop-loss-action", default=None,
                   help="Action on loss (plumbed for parity; specify rule to enable)")
    p.add_argument("--vix-threshold", type=float, default=None,
                   help="VIX gating threshold (no-op unless correlation set & logic enabled)")
    p.add_argument("--vix-correlation", choices=["gt","lt"], default=None,
                   help="VIX gating: compare VIX to threshold (no-op unless logic enabled)")

    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main():
    a = parse_args()
    init_logging(a.log_level)
    print(f"[DEBUG] run_daily args: ticker={a.ticker}, start={a.start}, end={a.end}, "
          f"trade_type={a.trade_type}, weekday={a.weekday}, expiring_wks={a.expiring_wks}, qty={a.qty}, "
          f"ic_width={a.iron_condor_width}, tgt_prem_otm={a.target_premium_otm}, tgt_delta={a.target_delta}, "
          f"steer={a.target_steer}, tp_pct={a.stop_profit_percent}, sl_action={a.stop_loss_action}, "
          f"vix_thr={a.vix_threshold}, vix_corr={a.vix_correlation}")

    print("[DEBUG] Invoking monthly_recursive_backtest...")
    res = monthly_recursive_backtest(
        ticker=a.ticker,
        global_start_date=a.start,
        global_end_date=a.end,
        trade_type=a.trade_type,
        expiring_weekday=a.weekday,
        expiring_wks=a.expiring_wks,
        contract_qty=a.qty,
        iron_condor_width=a.iron_condor_width,
        target_premium_otm=a.target_premium_otm,
        target_delta=a.target_delta,
        target_steer=a.target_steer,
        stop_profit_percent=a.stop_profit_percent,
        stop_loss_action=a.stop_loss_action,
        vix_threshold=a.vix_threshold,
        vix_correlation=a.vix_correlation,
    )

    # === Daily Opened/Closed report ===
    try:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(a.start, "%Y-%m-%d")
        end_dt = datetime.strptime(a.end, "%Y-%m-%d")
        cur = start_dt
        while cur <= end_dt:
            ds = cur.strftime("%Y-%m-%d")
            print_opened_and_closed_for_date(res, ds)
            cur += timedelta(days=1)
    except Exception as _e:
        print(f"[WARN] Daily report printing failed: {_e}")
        print("[DEBUG] run_daily finished. Result type:", type(res))
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
