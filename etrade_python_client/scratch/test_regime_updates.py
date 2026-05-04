import asyncio
import pandas as pd
from datetime import datetime, timedelta
from live_trading.ev_engine import build_regime_return_arrays
import logging

logging.basicConfig(level=logging.INFO)

async def test_regime_detection():
    print("🚀 Starting Market Regime Detection Test...")
    # Use a shorter window for testing
    try:
        # We need enough data for the 252-day PCA warmup
        # Let's request 2 years
        regime_dict, hmm_model, daily_models = build_regime_return_arrays("test_key", force_refit=True)
        
        if hmm_model:
            print("✅ HMM Model trained successfully.")
            print(f"Number of components: {hmm_model.n_components}")
            print(f"Feature names: {hmm_model.feature_names_}")
            
            # Check if deterministic state mapping worked (implied by model existence and logs)
            # The logs should show "[Alignment] States remapped by variance"
            
            print("✅ Regime dictionary keys:", list(regime_dict.keys()))
            for k, v in regime_dict.items():
                print(f"  {k}: {len(v)} samples")
        else:
            print("❌ HMM Model training failed.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_regime_detection())
