import sys, os
import json
sys.path.append('/Users/btian/EtradePythonClient/etrade_python_client')
from live_trading.etrade_cover_call_new import oauth
from market.market_bo import Market

session, base_url = oauth(use_sandbox=False, auto_login=True)
symbol = "SPY:2026:05:29:PUT:620"
url = f"{base_url}/v1/market/quote/{symbol}.json"
response = session.get(url)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("Response Data:")
    print(json.dumps(data, indent=2))
else:
    print(f"Error: {response.text}")
