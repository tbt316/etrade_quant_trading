import pandas as pd
import os

raw_path = 's_and_p_data/spy_vix_historical_raw.csv'
df = pd.read_csv(raw_path, index_col=0)

# Create SPY cache
spy_cache = df[['SPY_Close']].copy()
spy_cache.columns = ['Price']
spy_cache.to_csv('s_and_p_data/underlying_SPY.csv')
print("Created s_and_p_data/underlying_SPY.csv")

# Create VIX cache
vix_cache = df[['VIX_Close']].copy()
vix_cache.columns = ['Price']
vix_cache.to_csv('s_and_p_data/underlying_^VIX.csv')
print("Created s_and_p_data/underlying_^VIX.csv")
