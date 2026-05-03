import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import logging
from live_trading.data_ingestion import DataIngestor

logging.basicConfig(level=logging.INFO)

def test_stationarity():
    ingestor = DataIngestor()
    
    # Create a dummy dataframe with a flat array and a wandering level
    dates = pd.date_range("2020-01-01", periods=100)
    df = pd.DataFrame({
        'Flat_Rate': [0.13] * 100,
        'Wandering_Level': np.cumsum(np.random.randn(100)) + 10,
        'Log_Return_Asset': np.random.randn(100) * 0.01
    }, index=dates)
    
    print("\n--- Raw Data ---")
    print(df.head())
    
    stationary_df = ingestor.ensure_stationarity(df)
    
    print("\n--- Stationary Data ---")
    print(stationary_df.head())
    
    # Check if Flat_Rate was transformed
    if stationary_df['Flat_Rate'].iloc[0] == 0.13:
        print("\n❌ FAILURE: Flat_Rate was NOT transformed (still 0.13)")
    else:
        print("\n✅ SUCCESS: Flat_Rate was transformed")

    # Check if Wandering_Level was transformed
    if abs(stationary_df['Wandering_Level'].iloc[50]) > 5:
        print("❌ FAILURE: Wandering_Level was likely NOT differenced (still large)")
    else:
        print("✅ SUCCESS: Wandering_Level was transformed")

if __name__ == "__main__":
    test_stationarity()
