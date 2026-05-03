import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_portfolio_views():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        views = ["COMPLETE", "INTRADAY", "OPTIONSWATCH"]
        
        for v in views:
            print(f"\n--- Checking View: {v} ---")
            url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/portfolio.json"
            params = {"view": v}
            response = session.get(url, header_auth=True, params=params)
            
            if response.status_code == 200:
                data = response.json()
                positions = data.get("PortfolioResponse", {}).get("AccountPortfolio", [{}])[0].get("Position", [])
                if positions:
                    # Check first position for margin keys
                    p = positions[0]
                    print(f"Sample Position ({p.get('Product', {}).get('symbol')}):")
                    # Look for anything related to margin or requirement
                    found_margin = False
                    # Positions often have a 'Complete' or 'Intraday' sub-object
                    sub_obj = p.get(v.capitalize())
                    if not sub_obj:
                        # Sometimes it's lowercase or the whole object
                        sub_obj = p
                        
                    for key, val in sub_obj.items():
                        if "margin" in key.lower() or "req" in key.lower():
                            print(f"  {key}: {val}")
                            found_margin = True
                    
                    if not found_margin:
                        print("  No margin keys found in this view.")
                else:
                    print("  No positions returned.")
            else:
                print(f"  Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_portfolio_views()
