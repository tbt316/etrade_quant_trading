
import yfinance as yf
import json
import os
from datetime import datetime, timedelta

CACHE_FILE = "/Users/btian/EtradePythonClient/etrade_python_client/spy_vix_price_cache.json"
TRACKER_FILE = "/Users/btian/EtradePythonClient/etrade_python_client/spy_tracking_data.json"

def populate_cache():
    # 1. Get dates from tracker
    with open(TRACKER_FILE, 'r') as f:
        tracker = json.load(f)
    dates = list(tracker["daily_snapshots"].keys())
    if not dates:
        print("No dates found in tracker.")
        return
    
    start_date = min(dates)
    print(f"Fetching data from {start_date}...")
    
    # 2. Fetch data
    spy = yf.download('SPY', start=start_date, progress=False, auto_adjust=False)
    vix = yf.download('^VIX', start=start_date, progress=False, auto_adjust=False)
    
    # 3. Load cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {"SPY": {}, "VIX": {}}
    
    # 4. Populate
    def extract(df):
        res = {}
        # Handle MultiIndex if present
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df = df.droplevel(1, axis=1)
        
        col = 'Close'
        if col not in df.columns:
            for c in df.columns:
                if 'Close' in str(c):
                    col = c
                    break
        
        if col in df.columns:
            for idx, row in df.iterrows():
                dt = str(idx.date())
                res[dt] = round(float(row[col]), 2)
        return res

    spy_data = extract(spy)
    vix_data = extract(vix)
    
    cache["SPY"].update(spy_data)
    cache["VIX"].update(vix_data)
    
    # 5. Save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Populated cache with {len(spy_data)} SPY and {len(vix_data)} VIX entries.")

if __name__ == "__main__":
    populate_cache()
