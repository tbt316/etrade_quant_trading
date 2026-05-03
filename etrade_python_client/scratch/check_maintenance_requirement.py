import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_maintenance_requirement():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/portfolio.json"
        params = {"view": "COMPLETE", "count": 100}
        response = session.get(url, header_auth=True, params=params)
        
        if response.status_code == 200:
            data = response.json()
            positions = data.get("PortfolioResponse", {}).get("AccountPortfolio", [{}])[0].get("Position", [])
            print(f"Checking {len(positions)} positions...")
            for p in positions:
                sym = p.get('Product', {}).get('symbol')
                comp = p.get("Complete", {})
                
                # Search for any value that looks like a requirement
                req = comp.get("maintenanceRequirement")
                margin = comp.get("maintenanceMargin")
                
                if req is not None or margin is not None:
                    print(f"Symbol: {sym} | Req: {req} | Margin: {margin}")
                
                # Check for other keys too
                for k, v in comp.items():
                    if "req" in k.lower() or "margin" in k.lower():
                        if v != 0 and v != False: # Only print non-zero/non-false
                            print(f"  {sym} - {k}: {v}")
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_maintenance_requirement()
