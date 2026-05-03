import os
import sys
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.getcwd())

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm, get_regime_labels

def run_sync(coro):
    """Helper to run async coroutines from sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return loop.run_until_complete(coro)

def generate_trace():
    # The target range for the review
    target_start = "2025-01-01"
    target_end = "2025-06-01"
    
    # We fetch deep history for "warmup" (fractional differencing and PCA stability)
    # The cache goes back to 2011/2014, so 2015 is a safe, deep warmup.
    fetch_start = "2015-01-01" 
    output_dir = "regime_review_2025"
    
    print(f"\n🚀 Starting Regime Trace Generation for {target_start} to {target_end}...")
    print(f"📡 Deep warmup starts at {fetch_start} to handle fractional differencing (d=0.2) and monthly macro indicators.")
    
    ingestor = DataIngestor()
    
    # 1. Fetch RAW data (Stationary but NOT scaled)
    print("📡 Fetching raw stationary feature set...")
    stationary_df = run_sync(ingestor.build_fused_dataset(fetch_start, target_end, scale=False))
    
    # Save the full stationary set for context, but we will focus on the target range
    stationary_df.to_csv(os.path.join(output_dir, "stationary_features_full.csv"))
    
    # 2. Fetch original levels (for review)
    print("📊 Fetching original price levels...")
    yf_df = ingestor.fetch_yf_data(fetch_start, target_end)
    fred_df = ingestor.fetch_fred_data(fetch_start, target_end)
    raw_levels = pd.concat([yf_df, fred_df], axis=1).ffill()
    raw_levels.to_csv(os.path.join(output_dir, "raw_price_levels.csv"))
    
    # 3. Train HMM (Causal Walk-Forward)
    print("🧠 Training HMM and extracting PCA indicators...")
    # We pass the full stationary_df to ensure enough data for BIC and HMM training
    best_hmm, best_k, feature_df = train_regime_hmm(stationary_df, n_components=None, expanding_window=True)
    
    if best_hmm is None:
        print("❌ HMM Training failed.")
        return

    # 4. Filter to target range for final review package
    print(f"✂️ Filtering results to target range: {target_start} to {target_end}")
    
    # CALCULATE PROBABILITIES (using the final model for the trace)
    pc_cols = [c for c in feature_df.columns if c.startswith('PC')]
    if not pc_cols:
         # Fallback if names are different
         pc_cols = ['PC1', 'PC2']
    
    # We use the final model to get consistent probabilities for the trace
    print("🔮 Calculating state probabilities...")
    probs = best_hmm.predict_proba(feature_df[pc_cols].values)
    for i in range(best_hmm.n_components):
        feature_df[f'prob_state_{i}'] = probs[:, i]

    feature_df = feature_df[feature_df.index >= target_start]
    raw_levels = raw_levels[raw_levels.index >= target_start]

    # Save results
    print("💾 Saving regime results (PCA + Probabilities + Labels)...")
    feature_df.to_csv(os.path.join(output_dir, "regime_results.csv"))
    raw_levels.to_csv(os.path.join(output_dir, "raw_price_levels.csv"))
    
    # 5. Extract Model Parameters (Final Snapshot)
    print("📝 Exporting final model parameters...")
    params = {
        "n_components": int(best_hmm.n_components),
        "startprob": best_hmm.startprob_.tolist(),
        "transmat": best_hmm.transmat_.tolist(),
        "feature_names": best_hmm.feature_names_
    }
    
    if hasattr(best_hmm, "means_"):
        params["means"] = best_hmm.means_.tolist()
    if hasattr(best_hmm, "covars_"):
        params["covars"] = best_hmm.covars_.tolist()
    if hasattr(best_hmm, "weights_"):
        params["weights"] = best_hmm.weights_.tolist()

    with open(os.path.join(output_dir, "model_parameters.json"), "w") as f:
        json.dump(params, f, indent=4)
        
    # 6. Save a manifest of files
    manifest = [
        "raw_price_levels.csv: Original market data (SPY, VIX, FRED Macros)",
        "stationary_features.csv: Pre-processed, fractionally differenced features",
        "regime_results.csv: PCA factors, HMM state probabilities, and semantic labels",
        "model_parameters.json: Final HMM transition matrix and emission parameters"
    ]
    with open(os.path.join(output_dir, "trace_manifest.txt"), "w") as f:
        f.write("\n".join(manifest))
    
    print(f"\n✅ Trace generation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    generate_trace()
