import sys
import os
import configparser
from datetime import datetime

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.etrade_cover_call_new import oauth
from accounts.accounts_bo import Accounts

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    username = config.get('ETRADE', 'USER').strip("'\"")
    password = config.get('ETRADE', 'PASS').strip("'\"")

    print(f"Attempting to analyze SPY options for account ending 8703...")

    try:
        # Get authenticated session
        # We set auto_login=True to use the automated flow if needed
        session, base_url = oauth(use_sandbox=False, auto_login=True, username=username, password=password)
        
        accounts_bo = Accounts(session, base_url)
        account_info = accounts_bo.account_list() 
        
        # Find account ending in 8703
        idx = -1
        for i, info in enumerate(account_info):
            account_id, desc, inst_type, id_key = info
            if account_id.endswith('8703'):
                idx = i
                print(f"Selected Account: {account_id} ({desc})")
                break
        
        if idx == -1:
            print("❌ Could not find account ending in 8703")
            return

        # Initialize the selected account in the accounts_bo object
        accounts_bo.account_list(idx)

        print("🔄 Fetching portfolio positions and calculating Greeks (this may take a moment)...")
        # portfolio() calculates delta, gamma, theta etc. for all options
        positions = accounts_bo.portfolio(print_enable=False)
        
        spy_calls_delta = 0
        spy_puts_delta = 0
        
        print("\n📊 SPY Options Position Breakdown:")
        print("=" * 80)
        print(f"{'Type':<6} | {'Strike':<8} | {'Expiry':<12} | {'Qty':<6} | {'Delta':<10} | {'Agg Delta':<10}")
        print("-" * 80)
        
        found_any = False
        for pos in positions:
            if pos.symbol == "SPY" and pos.security_type == "Option":
                found_any = True
                # pos.delta is per contract (0 to 1 for calls, -1 to 0 for puts)
                # Aggregated delta = delta * quantity
                # E*Trade Greeks are usually per share, so delta of 0.5 means 50 delta per contract.
                # However, many systems use 0.5 to mean 50. Let's check how accounts_bo handles it.
                # Looking at accounts_bo.py, it just uses the 'delta' from E*Trade.
                # E*Trade API returns delta as a float (e.g., 0.5).
                
                agg_delta = pos.delta * pos.quantity
                expiry_str = pos.expiration_date.strftime("%Y-%m-%d") if pos.expiration_date else "N/A"
                print(f"{pos.call_put:<6} | {pos.strike_price:<8.2f} | {expiry_str:<12} | {pos.quantity:<6} | {pos.delta:<10.4f} | {agg_delta:<10.4f}")
                
                if pos.call_put == "CALL":
                    spy_calls_delta += agg_delta
                elif pos.call_put == "PUT":
                    spy_puts_delta += agg_delta
        
        if not found_any:
            print("No SPY option positions found in this account.")
        else:
            print("=" * 80)
            print(f"🚀 Aggregated SPY Call Delta: {spy_calls_delta:>12.4f}")
            print(f"📉 Aggregated SPY Put Delta:  {spy_puts_delta:>12.4f}")
            print("-" * 80)
            print(f"⚖️  Net SPY Portfolio Delta:  {spy_calls_delta + spy_puts_delta:>12.4f}")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
