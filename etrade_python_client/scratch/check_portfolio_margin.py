import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_portfolio_margin():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        # Fetch portfolio with margin details if possible
        # Some accounts show margin requirement per position
        url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/portfolio.json"
        params = {"view": "MARGIN"}
        response = session.get(url, header_auth=True, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*40)
            print("PORTFOLIO MARGIN VIEW")
            print("="*40)
            # Just print a few positions to see the structure
            positions = data.get("PortfolioResponse", {}).get("AccountPortfolio", [{}])[0].get("Position", [])
            for p in positions[:5]:
                print(f"Symbol: {p.get('Product', {}).get('symbol')}")
                print(f"  Qty: {p.get('quantity')}")
                # Look for margin keys
                # Possible keys: maintenanceMargin, marginRequirement
                for key in p.keys():
                    if "margin" in key.lower():
                        print(f"  {key}: {p[key]}")
            print("="*40)
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_portfolio_margin()
