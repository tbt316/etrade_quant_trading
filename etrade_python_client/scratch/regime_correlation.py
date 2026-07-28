import pandas as pd
import numpy as np
import os
import sys
from live_trading.ev_engine import train_regime_hmm
from live_trading.data_ingestion import DataIngestor
from live_trading.market_sessions import latest_available_session_before
import asyncio

def analyze_regime_profiles():
    ingestor = DataIngestor()
    start = "2024-01-01"
    test_start = "2025-01-02"
    end = "2026-04-30"
    raw_df = ingestor.fetch_yf_data(start, end)
    if raw_df.empty:
        raise RuntimeError("REGIME_CORRELATION_MARKET_DATA_UNAVAILABLE")
    fit_end = latest_available_session_before(
        raw_df.index,
        test_start,
    )
    print(
        f"Calibration {start} to {fit_end}; "
        f"OOS {test_start} to {end}"
    )
    
    # We need a loop to call build_fused_dataset
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stationary_df = loop.run_until_complete(
        ingestor.build_fused_dataset(
            start,
            end,
            scale=False,
            fit_end=fit_end,
        )
    )
    
    # train_regime_hmm handles its own loop too
    best_hmm, best_k, final_df = train_regime_hmm(
        stationary_df,
        expanding_window=True,
        fit_end=fit_end,
    )
    if best_hmm is None or final_df.empty:
        raise RuntimeError("CAUSAL_REGIME_TRACE_UNAVAILABLE")
    final_df = final_df.loc[
        pd.Timestamp(test_start):pd.Timestamp(end)
    ].copy()
    if final_df.empty:
        raise RuntimeError("OUT_OF_SAMPLE_REGIME_TRACE_UNAVAILABLE")

    raw_data = raw_df.loc[final_df.index].copy()
    raw_data['HMM_State'] = final_df['HMM_State']
    raw_data['Regime_Label'] = final_df['Regime_Label']
    
    profile = raw_data.groupby('HMM_State').mean()
    
    print("\n" + "="*60)
    print("REGIME PROFILES & MARKET CORRELATIONS (2024-2026)")
    print("="*60)
    
    for state in range(best_k):
        state_labels = raw_data.loc[
            raw_data["HMM_State"] == state,
            "Regime_Label",
        ].dropna()
        name = (
            state_labels.iloc[-1]
            if not state_labels.empty
            else f"State {state}"
        )
        vix = profile.loc[state, 'VIX_Close'] if 'VIX_Close' in profile.columns else 0
        state_df = raw_data[raw_data['HMM_State'] == state]
        spy_ret = state_df['SPY_Close'].pct_change().mean() * 252 * 100
        
        print(f"\n{name}:")
        print(f"  • Avg VIX: {vix:.2f}")
        print(f"  • Annualized Mean Return: {spy_ret:.2f}%")
        
        if vix > 25:
            context = "Market Turmoil: High volatility, negative skew."
        elif vix > 20:
            context = "High Vol Chop: Sideways/defensive, elevated risk."
        elif spy_ret > 15 and vix < 15:
            context = "Robust Expansion: Strong bullish momentum, low fear."
        elif spy_ret > 0 and vix < 20:
            context = "Steady Growth: Moderate upside, stable environment."
        else:
            context = "Cautious/Defensive: Lower returns, higher relative risk."
            
        print(f"  • Correlation: {context}")

if __name__ == "__main__":
    analyze_regime_profiles()
