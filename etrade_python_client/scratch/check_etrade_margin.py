import os
import sys
import json

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts
# We need to import oauth from the main script or wherever it is defined.
# It seems it's defined in core_api or the main script.
# Let's check etrade_cover_call_new.py imports.

def check_etrade_margin():
    try:
        from live_trading.etrade_cover_call_new import oauth
        
        # We need to be in the project root for paths to work
        os.chdir(project_root)
        
        session, base_url = oauth(use_sandbox=False, auto_login=True)
        acc = Accounts(session, base_url)
        acc.account_list(1) # Individual Brokerage
        bal = acc.balance()
        
        if bal and "Computed" in bal:
            comp = bal["Computed"]
            print("\n" + "="*40)
            print("OFFICIAL E*TRADE MARGIN DATA")
            print("="*40)
            print(f"Net Account Value:     ${comp.get('RealTimeValues', {}).get('totalAccountValue', 0):,.2f}")
            print(f"Maintenance Margin:    ${comp.get('maintenanceMargin', 0):,.2f}")
            print(f"Current Margin Balance: ${comp.get('currentMarginBalance', 0):,.2f}")
            print(f"House Margin Req:      ${comp.get('houseMarginRequirement', 0):,.2f}")
            print(f"Margin Buying Power:   ${comp.get('marginBuyingPower', 0):,.2f}")
            print("="*40)
            
            # Print all Computed keys for reference
            print("\nRaw Computed Keys:", list(comp.keys()))
        else:
            print("Could not retrieve balance data.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_etrade_margin()
