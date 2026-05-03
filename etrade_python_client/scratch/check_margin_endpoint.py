import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_margin_endpoint():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        # Test a direct margin endpoint if it exists
        url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/margin.json"
        print(f"Testing URL: {url}")
        response = session.get(url, header_auth=True)
        
        if response.status_code == 200:
            print("SUCCESS! Found a direct margin endpoint.")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed with {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_margin_endpoint()
