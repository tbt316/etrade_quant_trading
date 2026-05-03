
import os
import json
import sys
from datetime import datetime

# Mock the Accounts object and its dependencies to generate the HTML
class MockAccounts:
    def __init__(self):
        self.username = "test_user"
    
    def get_portfolio_summary_html(self, bypass_etrade=True):
        # We need to import the real Accounts but mock its network calls
        # For simplicity, I'll just copy the relevant part of the logic
        # OR I can try to run it if I can mock the session
        return "<html><body><h1>Mock Dashboard</h1></body></html>"

# Actually, it's easier to just read the code and see what HTML it generates
# But I already did that.

def check_cache_consistency():
    with open('spy_vix_price_cache.json', 'r') as f:
        cache = json.load(f)
    
    spy = cache.get("SPY", {})
    vix = cache.get("VIX", {})
    
    print(f"SPY entries: {len(spy)}")
    print(f"VIX entries: {len(vix)}")
    
    if spy:
        min_spy = min(spy.values())
        max_spy = max(spy.values())
        print(f"SPY Range: {min_spy} - {max_spy}")
    
    if vix:
        min_vix = min(vix.values())
        max_vix = max(vix.values())
        print(f"VIX Range: {min_vix} - {max_vix}")

if __name__ == "__main__":
    check_cache_consistency()
