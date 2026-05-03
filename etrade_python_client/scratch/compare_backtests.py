import asyncio
import os
import sys
import pandas as pd
from datetime import datetime

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels
from backtesting.backtest_runner import run_put_credit_spread_backtest

async def compare_targets():
    df = fetch_historical_data()
    hmm_model, k, feature_df = train_regime_hmm(df)
    regime_labels = get_regime_labels(hmm_model)
    regimes = {d.strftime("%Y-%m-%d"): int(v) for d, v in feature_df['HMM_State'].to_dict().items()}

    common_params = {
        "underlying": "SPY",
        "start_date": "2021-01-01",
        "end_date": "2026-04-25",
        "regimes": regimes,
        "regime_labels": regime_labels,
        "dynamic_delta_variant": True,
        "panic_delta_multiplier": 4.0,
        "plot": False
    }

    print("Running Backtest 1 (70% target)...")
    res1 = await run_put_credit_spread_backtest(**common_params, early_profit_pct=0.70)
    
    # We need to manually set the profit target for panic trades in common_params logic
    # Wait, run_put_credit_spread_backtest uses the default 0.90 for panic if variant is on.
    # To compare, I need to modify the code or pass different values.
    # Actually, the user wants to compare the CURRENT code (which has 0.90 for panic)
    # with the PREVIOUS code (which had 0.70 for panic).
    
    # Let's check the current code state.
    pass

if __name__ == "__main__":
    # This script is just for reference. 
    # I can see the reasons from the logs provided.
    pass
