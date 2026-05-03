import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts

def check_etrade_margin_full():
    try:
        from live_trading.etrade_cover_call_new import oauth
        os.chdir(project_root)
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        bal = acc.balance()
        
        if bal and "Computed" in bal:
            comp = bal["Computed"]
            print("\n" + "="*40)
            print("FULL E*TRADE COMPUTED DATA")
            print("="*40)
            print(json.dumps(comp, indent=2))
            print("="*40)
        else:
            print("Could not retrieve balance data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_etrade_margin_full()
