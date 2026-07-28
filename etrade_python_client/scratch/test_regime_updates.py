import asyncio
import pandas as pd
from datetime import datetime, timedelta
from live_trading.ev_engine import build_regime_return_arrays
from live_trading.market_sessions import latest_completed_nyse_session
from live_trading.regime_taxonomy import RawHMMStateRef
import logging

logging.basicConfig(level=logging.INFO)

async def test_regime_detection():
    print("🚀 Starting Market Regime Detection Test...")
    # Use a shorter window for testing
    try:
        # We need enough data for the 252-day PCA warmup
        # Let's request 2 years
        as_of_date = latest_completed_nyse_session()
        regime_buckets, hmm_model, daily_models = (
            build_regime_return_arrays(
                datetime.fromisoformat(as_of_date).date().toordinal(),
                force_refit=True,
                as_of_date=as_of_date,
            )
        )
        
        if hmm_model:
            print("✅ HMM Model trained successfully.")
            print(f"Number of components: {hmm_model.n_components}")
            print(f"Feature names: {hmm_model.feature_names_}")
            
            # Check if deterministic state mapping worked (implied by model existence and logs)
            # The logs should show "[Alignment] States remapped by variance"
            
            for state in range(regime_buckets.state_count):
                bucket = regime_buckets.bucket_for(
                    RawHMMStateRef(
                        regime_buckets.taxonomy_id,
                        state,
                    )
                )
                print(f"  State_{state}: {len(bucket)} samples")
        else:
            print("❌ HMM Model training failed.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_regime_detection())
