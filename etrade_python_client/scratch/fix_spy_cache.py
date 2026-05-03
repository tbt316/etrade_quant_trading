
import json
import os

CACHE_FILE = "/Users/btian/EtradePythonClient/etrade_python_client/spy_vix_price_cache.json"

if not os.path.exists(CACHE_FILE):
    print("Cache file not found.")
else:
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    
    spy_data = cache.get("SPY", {})
    fixed_count = 0
    for date, price in spy_data.items():
        # If price is < 2000, it's likely a raw price (SPY hasn't been $2000 yet, 
        # and $2000 * 100 = $200,000 which is above our $100k peak)
        if price < 2000:
            spy_data[date] = round(price * 100, 2)
            fixed_count += 1
    
    if fixed_count > 0:
        cache["SPY"] = spy_data
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Fixed {fixed_count} SPY data points in cache (multiplied by 100).")
    else:
        print("No raw SPY prices found to fix.")
