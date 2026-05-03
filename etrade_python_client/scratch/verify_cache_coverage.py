import pandas as pd
import os
from datetime import datetime

cache_dir = "s_and_p_data/api_cache"
files = [f for f in os.listdir(cache_dir) if f.endswith(".parquet")]

print(f"{'Symbol':<20} | {'Start Date':<12} | {'End Date':<12} | {'Rows':<6}")
print("-" * 60)

for f in sorted(files):
    path = os.path.join(cache_dir, f)
    df = pd.read_parquet(path)
    symbol = f.replace(".parquet", "").replace("INDEX_", "^")
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    rows = len(df)
    print(f"{symbol:<20} | {start:<12} | {end:<12} | {rows:<6}")
