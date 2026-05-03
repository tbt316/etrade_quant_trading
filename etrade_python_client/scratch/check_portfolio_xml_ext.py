import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_portfolio_xml_ext():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        
        url = base_url + "/v1/accounts/" + acc.account["accountIdKey"] + "/portfolio.xml"
        print(f"Testing URL: {url}")
        response = session.get(url, header_auth=True)
        
        if response.status_code == 200:
            print("\n" + "="*40)
            print("XML PORTFOLIO RESPONSE")
            print("="*40)
            print(response.text[:2000])
            print("="*40)
        else:
            print(f"Error: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_portfolio_xml_ext()
