import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_portfolio_realtime():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/portfolio.json"
        params = {"view": "COMPLETE", "count": 10, "realTimeNAV": "true"}
        response = session.get(url, header_auth=True, params=params)
        
        if response.status_code == 200:
            data = response.json()
            positions = data.get("PortfolioResponse", {}).get("AccountPortfolio", [{}])[0].get("Position", [])
            for p in positions:
                sym = p.get('Product', {}).get('symbol')
                print(f"\n--- Symbol: {sym} ---")
                # Look for maintenanceMargin specifically
                comp = p.get("Complete", {})
                for k, v in comp.items():
                    if "margin" in k.lower() or "req" in k.lower():
                        print(f"  {k}: {v}")
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_portfolio_realtime()
