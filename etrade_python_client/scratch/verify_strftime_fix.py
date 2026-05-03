import sys
import os
from datetime import datetime, date

# Mock the environment
sys.path.insert(0, "/Users/btian/EtradePythonClient/etrade_python_client")

from accounts.accounts_bo import StockPosition, Accounts

def test_conflict_check_fix():
    print("🧪 Testing check_conflict_position fix...")
    
    # 1. Create a dummy Accounts instance (mocking minimal dependencies)
    class MockAccounts(Accounts):
        def __init__(self):
            pass # Skip complex init
        def portfolio(self, minimal=False):
            # Return a position with a REAL date object (as E*Trade usually does)
            return [
                StockPosition(
                    symbol="SPY", 
                    quantity=-25, 
                    security_type="Option", 
                    call_put="CALL", 
                    strike_price=760.0, 
                    expiration_date=date(2026, 5, 22)
                )
            ]

    accounts = MockAccounts()
    
    # 2. Test Case A: Pass a STRING expiration date (Simulating the dashboard bug)
    print("  Testing Case A: String expiration date...")
    s_leg_str = StockPosition(
        symbol="SPY", 
        quantity=-25, 
        security_type="Option", 
        call_put="CALL", 
        strike_price=760.0, 
        expiration_date="2026-05-22"
    )
    
    try:
        qty = accounts.check_conflict_position(s_leg_str, "sell", 25)
        print(f"  ✅ String case passed. Result quantity: {qty}")
    except AttributeError as e:
        print(f"  ❌ String case FAILED with AttributeError: {e}")
        return False

    # 3. Test Case B: Pass a DATE object (Normal flow)
    print("  Testing Case B: Date object expiration date...")
    s_leg_date = StockPosition(
        symbol="SPY", 
        quantity=-25, 
        security_type="Option", 
        call_put="CALL", 
        strike_price=760.0, 
        expiration_date=date(2026, 5, 22)
    )
    
    try:
        qty = accounts.check_conflict_position(s_leg_date, "sell", 25)
        print(f"  ✅ Date object case passed. Result quantity: {qty}")
    except AttributeError as e:
        print(f"  ❌ Date object case FAILED with AttributeError: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_conflict_check_fix()
    if success:
        print("\n✨ ALL TESTS PASSED. The 'strftime' fix is verified.")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED.")
        sys.exit(1)
