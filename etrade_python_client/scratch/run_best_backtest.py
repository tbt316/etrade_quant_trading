import asyncio
import os
import sys
import pandas as pd
from datetime import datetime

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels
from backtesting.backtest_runner import run_put_credit_spread_backtest

async def run_best():
    # 1. Fetch data
    print("Fetching historical data...")
    df_hist = fetch_historical_data()
    
    # 2. Parameters from optimization_results.csv (Best result)
    # short_delta: -0.14
    # panic_delta_mult: 4.0
    # panic_swap: False
    # k: 3
    
    k = 3
    short_delta = -0.14
    panic_delta_mult = 4.0
    panic_swap = False
    
    # 3. Train HMM for regimes
    print(f"Training HMM for K={k}...")
    best_hmm, best_k, feature_df = train_regime_hmm(df_hist, n_components=k)
    labels = get_regime_labels(best_hmm)
    reg_dict = {d.strftime('%Y-%m-%d'): s for d, s in feature_df['HMM_State'].to_dict().items()}
    
    # 4. Prepare prices
    start_date = "2020-01-01"
    end_date = "2021-04-25"
    
    spy_close = df_hist['SPY_Close'].copy()
    spy_close.index = spy_close.index.strftime("%Y-%m-%d")
    vix_close = df_hist['VIX_Close'].copy()
    vix_close.index = vix_close.index.strftime("%Y-%m-%d")
    
    mask = (spy_close.index >= start_date) & (spy_close.index <= end_date)
    spy_prices = spy_close[mask]
    vix_prices = vix_close[mask]
    
    # 5. Run Backtest
    print(f"\nRunning backtest with:")
    print(f"  short_delta: {short_delta}")
    print(f"  panic_delta_mult: {panic_delta_mult}")
    print(f"  panic_swap: {panic_swap}")
    print(f"  k: {k}")
    
    res = await run_put_credit_spread_backtest(
        underlying="SPY",
        start_date=start_date,
        end_date=end_date,
        target_short_delta=short_delta,
        panic_delta_multiplier=panic_delta_mult,
        panic_dte_target=63,
        panic_swap_enabled=panic_swap,
        regimes=reg_dict,
        regime_labels=labels,
        underlying_prices=spy_prices,
        vix_prices=vix_prices,
        backtest_qty=2,
        plot=True,
        enable_logging=True
    )
    
    print("\nBacktest complete.")

if __name__ == "__main__":
    asyncio.run(run_best())
