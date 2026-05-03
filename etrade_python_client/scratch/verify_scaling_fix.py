import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import asyncio
from live_trading.data_ingestion import DataIngestor
import matplotlib.pyplot as plt

async def verify_fix():
    ingestor = DataIngestor()
    
    # Create fake data
    dates = pd.date_range("2014-01-01", "2021-01-01", freq="D")
    data = np.random.normal(0, 1, len(dates))
    
    # Outlier in 2020
    crash_idx = dates.get_loc("2020-03-15")
    data[crash_idx] = 5000.0 # Extreme
    
    df_with_outlier = pd.DataFrame(data, index=dates, columns=['Feature'])
    df_no_outlier = df_with_outlier.copy()
    df_no_outlier.iloc[crash_idx] = 0.0 # Remove outlier
    
    print("--- Testing GLOBAL SCALING (Old Way) ---")
    # Global fit on both
    scaled_global_with = ingestor.scale_features(df_with_outlier)
    scaled_global_no = ingestor.scale_features(df_no_outlier)
    
    mean_2015_with = scaled_global_with.loc["2015"].mean()[0]
    mean_2015_no = scaled_global_no.loc["2015"].mean()[0]
    
    print(f"2015 Mean (Global, Outlier present): {mean_2015_with:.6f}")
    print(f"2015 Mean (Global, Outlier absent):  {mean_2015_no:.6f}")
    print(f"Difference: {abs(mean_2015_with - mean_2015_no):.6f}")
    
    print("\n--- Testing CAUSAL SCALING (New Way) ---")
    # Causal fit on both
    scaled_causal_with = ingestor.scale_features(df_with_outlier, expanding=True, warmup=252)
    scaled_causal_no = ingestor.scale_features(df_no_outlier, expanding=True, warmup=252)
    
    c_mean_2015_with = scaled_causal_with.loc["2015"].mean()[0]
    c_mean_2015_no = scaled_causal_no.loc["2015"].mean()[0]
    
    print(f"2015 Mean (Causal, Outlier present): {c_mean_2015_with:.6f}")
    print(f"2015 Mean (Causal, Outlier absent):  {c_mean_2015_no:.6f}")
    print(f"Difference: {abs(c_mean_2015_with - c_mean_2015_no):.6f}")
    
    if abs(c_mean_2015_with - c_mean_2015_no) < 1e-10:
        print("\n✅ SUCCESS: Causal scaling is identical regardless of future outliers!")
    else:
        print("\n❌ FAILURE: Causal scaling still shows leakage.")

if __name__ == "__main__":
    asyncio.run(verify_fix())
