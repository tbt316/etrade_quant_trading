import json
from collections import defaultdict

def _as_num(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except:
        return default

def calculate_naked_margin(pos):
    cp = pos.get("call_put", "").upper()
    strike = _as_num(pos.get("strike_price"))
    underlying = _as_num(pos.get("underlying_last_price"), strike)
    premium = _as_num(pos.get("last_price"), 0.0)
    qty_abs = abs(_as_num(pos.get("quantity"), 0))
    
    if cp == "CALL":
        # Simplified: strike * 100 * qty
        return strike * 100.0 * qty_abs
    elif cp == "PUT":
        calc1 = 0.2 * underlying - (strike - underlying) + premium
        calc2 = 0.1 * strike + premium
        margin_per_contract = max(calc1, calc2, 0.0)
        return margin_per_contract * 100.0 * qty_abs
    return 0.0

data = json.load(open('spy_tracking_data.json'))
today = sorted(data['daily_snapshots'].keys())[-1]
snap = data['daily_snapshots'][today]
positions = snap['positions']

# Group by expiry
expiry_groups = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0, "DETAILS": []})
type_groups = defaultdict(list)

for pos in positions:
    key = (pos['expiration_date'], pos['call_put'])
    type_groups[key].append(pos)

print(f"--- MATH BREAKDOWN FOR {today} ---")

for (expiry, opt_type), group in type_groups.items():
    shorts = [dict(p) for p in group if p['quantity'] < 0]
    longs = [dict(p) for p in group if p['quantity'] > 0]
    
    type_margin = 0.0
    # Match Spreads
    for short in shorts:
        s_strike = short['strike_price']
        s_qty = abs(short['quantity'])
        for long in longs:
            l_strike = long['strike_price']
            l_qty = long['quantity']
            matched = min(s_qty, l_qty)
            if matched > 0:
                width = abs(s_strike - l_strike)
                m = width * matched * 100
                type_margin += m
                expiry_groups[expiry]["DETAILS"].append(f"  Spread: {opt_type} {s_strike}/{l_strike} x{int(matched)} -> ${m:,.0f}")
                s_qty -= matched
                long['quantity'] -= matched
                short['quantity'] = -s_qty
                if s_qty <= 0: break
    
    # Calculate Naked for leftovers
    for short in shorts:
        if abs(short['quantity']) > 0:
            m = calculate_naked_margin(short)
            type_margin += m
            expiry_groups[expiry]["DETAILS"].append(f"  Naked: {opt_type} {short['strike_price']} x{int(abs(short['quantity']))} -> ${m:,.0f}")

    expiry_groups[expiry][opt_type] = type_margin

grand_total = 0
for expiry in sorted(expiry_groups.keys()):
    v = expiry_groups[expiry]
    risk = max(v['CALL'], v['PUT'])
    grand_total += risk
    print(f"\nEXPIRY: {expiry}")
    print(f"  Call Margin: ${v['CALL']:,.2f}")
    print(f"  Put Margin:  ${v['PUT']:,.2f}")
    print(f"  MAX RISK:    ${risk:,.2f}")
    for d in v["DETAILS"]:
        print(d)

print(f"\nGRAND TOTAL CALCULATED MARGIN: ${grand_total:,.2f}")
