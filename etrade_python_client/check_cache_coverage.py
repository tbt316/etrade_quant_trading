import pandas as pd
import os

cache_dir = "s_and_p_data/api_cache"
for f in os.listdir(cache_dir):
    if f.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(cache_dir, f))
        print(f"{f}: {df.index.min()} to {df.index.max()} (Rows: {len(df)})")
