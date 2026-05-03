import os
import sys
from datetime import datetime

# Add project root to sys.path
project_root = "/Users/btian/EtradePythonClient/etrade_python_client"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from accounts.accounts_bo import Accounts
from market.market_bo import Market
from order.order_bo import Order

def explain_margin():
    # Initialize components
    # We use mock or real accounts depending on what's available. 
    # Since we want to explain the $256k, we should fetch the current positions.
    acc = Accounts()
    # Assume we are already logged in or have cached session if possible.
    # If not, we might need to read the last generated screened_option_pairs.html to see what went into it.
    
    # Better approach: Read the same data the server is seeing.
    # Since I don't want to trigger a full API fetch (which might fail without interaction),
    # I'll try to find if there's a cached portfolio or just use the logic on the current screened list.
    
    print("--- Margin Calculation Breakdown (Max Risk Model) ---")
    
    # Let's try to fetch the actual portfolio if possible, but safely.
    try:
        acc.account_list(1)
        positions = acc.portfolio(print_enable=False)
        screened = acc.screen_option(positions)
        
        # Now replicate the _spy_margin_totals logic with printing
        from collections import defaultdict
        
        def _as_num(v, default=0.0):
            try: return float(v) if v is not None else float(default)
            except: return float(default)
            
        def _naked_margin(short_leg, qty_abs):
            if short_leg is None or qty_abs <= 0: return 0.0
            cp = (getattr(short_leg, "call_put", "") or "").upper()
            strike = _as_num(getattr(short_leg, "strike_price", None), 0.0)
            if cp == "CALL": return strike * 100.0 * qty_abs
            if cp == "PUT":
                underlying = _as_num(getattr(short_leg, "underlying_last_price", None), strike)
                premium = _as_num(getattr(short_leg, "last_price", None), 0.0)
                calc1 = 0.2 * underlying - (strike - underlying) + premium
                calc2 = 0.1 * strike + premium
                return max(calc1, calc2, 0.0) * 100.0 * qty_abs
            return 0.0

        groups = defaultdict(lambda: {"longs": [], "shorts": []})
        for entry in screened:
            for leg in (entry.get("long_lot"), entry.get("short_lot")):
                if not leg or (getattr(leg, "symbol", "") or "").upper() != "SPY": continue
                cp = (getattr(leg, "call_put", "") or "").upper()
                exp = getattr(leg, "expiration_date", None)
                qty = _as_num(getattr(leg, "quantity", 0), 0.0)
                if qty == 0: continue
                bucket = groups[(cp, exp)]
                bucket["longs" if qty > 0 else "shorts"].append({"leg": leg, "qty": abs(qty)})

        expiry_groups = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0})
        
        def _strike(rec): return _as_num(getattr(rec["leg"], "strike_price", None), 0.0)

        for (cp, exp), parts in groups.items():
            shorts = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["shorts"]]
            longs = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["longs"]]
            if not shorts: continue
            
            reverse = (cp == "PUT")
            shorts.sort(key=_strike, reverse=reverse)
            longs.sort(key=_strike, reverse=reverse)

            margin = 0.0
            for short in shorts:
                s_qty = short["qty"]
                while s_qty > 0 and longs:
                    best_idx = min(range(len(longs)), key=lambda idx: abs(_strike(short) - _strike(longs[idx])))
                    long = longs[best_idx]
                    pair_qty = min(s_qty, long["qty"])
                    strike_diff = abs(_strike(short) - _strike(long))
                    margin += strike_diff * pair_qty * 100.0
                    s_qty -= pair_qty
                    long["qty"] -= pair_qty
                    if long["qty"] <= 0: longs.pop(best_idx)
                if s_qty > 0:
                    margin += _naked_margin(short["leg"], s_qty)
            
            expiry_groups[exp][cp] = margin

        total_max_risk = 0.0
        print(f"{'Expiration':<12} | {'Call Margin':<12} | {'Put Margin':<12} | {'Max Risk (Max of Both)':<20}")
        print("-" * 65)
        for exp in sorted(expiry_groups.keys()):
            margins = expiry_groups[exp]
            call_m = margins["CALL"]
            put_m = margins["PUT"]
            max_m = max(call_m, put_m)
            total_max_risk += max_m
            print(f"{str(exp)[:10]:<12} | ${call_m:>10,.2f} | ${put_m:>10,.2f} | ${max_m:>18,.2f}")
        
        print("-" * 65)
        print(f"{'TOTAL':<12} | {'':<12} | {'':<12} | ${total_max_risk:>18,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    explain_margin()
