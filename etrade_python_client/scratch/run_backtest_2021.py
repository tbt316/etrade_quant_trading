import asyncio
import os
import sys
import pandas as pd
from datetime import datetime

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels
from backtesting.backtest_runner import run_put_credit_spread_backtest

async def main():
    # 1. Fetch data and train HMM to get regimes
    print("Fetching historical data for HMM training...")
    df = fetch_historical_data()
    print("Training HMM...")
    hmm_model, k, feature_df = train_regime_hmm(df)
    regime_labels = get_regime_labels(hmm_model)
    regimes = {d.strftime("%Y-%m-%d"): int(v) for d, v in feature_df['HMM_State'].to_dict().items()}

    # 2. Run the backtest from 2021-01-01
    print("Starting backtest from 2021-01-01...")
    await run_put_credit_spread_backtest(
        underlying="SPY",
        start_date="2021-01-01",
        end_date="2026-04-25", # Current date in user metadata
        target_dte=42,
        close_dte=21,
        target_short_delta=-0.15,
        spread_width=20.0,
        plot=True,
        regimes=regimes,
        regime_labels=regime_labels,
        dynamic_delta_variant=True,
        enable_logging=True
    )

if __name__ == "__main__":
    asyncio.run(main())
