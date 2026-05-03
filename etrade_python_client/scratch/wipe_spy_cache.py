
import json
import os

CACHE_FILE = "/Users/btian/EtradePythonClient/etrade_python_client/spy_vix_price_cache.json"

if not os.path.exists(CACHE_FILE):
    print("Cache file not found.")
else:
    with open(CACHE_FILE, 'r') as f:
        cache = json.load(f)
    
    # Wipe SPY cache to force refetch of raw prices
    cache["SPY"] = {}
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print("Wiped SPY price cache. It will be refetched with raw prices on next dashboard refresh.")
