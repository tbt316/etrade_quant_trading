import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from live_trading.ev_engine import train_regime_hmm
from live_trading.data_ingestion import DataIngestor
from live_trading.market_sessions import latest_available_session_before
import logging

logging.basicConfig(level=logging.INFO)

async def run_analysis():
    print("🚀 Running Market Regime Analysis (2020-2026) - Optimized Window...")
    
    # Use a 1-year warmup for speed
    start_fetch = "2019-01-01"
    analysis_start = "2020-01-02"
    end_fetch = "2026-05-03" 
    
    ingestor = DataIngestor()
    
    print(f"📡 Fetching data from {start_fetch} to {end_fetch}...")
    df_raw = ingestor.fetch_yf_data(start_fetch, end_fetch)
    
    if df_raw.empty:
        print("❌ Failed to fetch data.")
        return

    # Train HMM in walk-forward mode
    # For speed, we will use a smaller warmup and a less frequent refit in this test
    print("🧠 Training HMM in walk-forward mode...")
    # Passing smaller warmup (default is 252 in ev_engine, let's stick to it but ensure data is enough)
    fit_end = latest_available_session_before(
        df_raw.index,
        analysis_start,
    )
    print(
        f"  Calibration {start_fetch} to {fit_end}; "
        f"OOS {analysis_start} to {end_fetch}"
    )
    hmm_model, k, results_df = train_regime_hmm(
        df_raw,
        expanding_window=True,
        fit_end=fit_end,
    )
    
    if hmm_model is None:
        print("❌ HMM Training failed.")
        return

    # Filter results to the 2020-2026 window
    analysis_df = results_df.loc[
        pd.Timestamp(analysis_start):pd.Timestamp(end_fetch)
    ].copy()
    
    print(f"✅ Analysis complete. Processed {len(analysis_df)} days.")
    
    # 1. State Characteristics
    print("\n--- Regime Characteristics ---")
    for state_id in sorted(analysis_df["HMM_State"].unique()):
        state_data = analysis_df[analysis_df['HMM_State'] == state_id]
        if not state_data.empty:
            label = state_data["Regime_Label"].dropna().iloc[-1]
            raw_subset = df_raw.loc[df_raw.index.intersection(state_data.index)]
            avg_vix = raw_subset['VIX_Close'].mean()
            if 'SPY_Close' in raw_subset.columns:
                ret_series = raw_subset['SPY_Close'].pct_change().dropna()
                avg_ret = ret_series.mean() * 252 * 100 
            else:
                avg_ret = 0.0
            
            count = len(state_data)
            print(f"Regime {state_id} ({label}):")
            print(f"  - Occurrences: {count} days ({count/len(analysis_df):.1%})")
            print(f"  - Avg VIX: {avg_vix:.2f}")
            print(f"  - Annualized Return: {avg_ret:.2f}%")

    # 2. Key Transitions
    print("\n--- Key Regime Transitions (2020-2026) ---")
    analysis_df['State_Change'] = analysis_df['HMM_State'].diff() != 0
    transitions = analysis_df[analysis_df['State_Change']]
    
    for date, row in transitions.iterrows():
        label = row['Regime_Label']
        print(f"📅 {date.strftime('%Y-%m-%d')}: Shifted to {label}")

    analysis_df.to_csv("scratch/regime_analysis_2020_2026_fast.csv")
    print("\n📊 Results saved to scratch/regime_analysis_2020_2026_fast.csv")

if __name__ == "__main__":
    asyncio.run(run_analysis())
