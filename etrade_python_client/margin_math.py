import json
from collections import defaultdict

data = json.load(open('spy_tracking_data.json'))
today = sorted(data['daily_snapshots'].keys())[-1]
snap = data['daily_snapshots'][today]
positions = snap['positions']

expiry_groups = defaultdict(lambda: {'CALL': 0.0, 'PUT': 0.0})
type_groups = defaultdict(list)

for pos in positions:
    key = (pos['expiration_date'], pos['call_put'])
    type_groups[key].append(pos)

print(f"Analysis for Snapshot Date: {today}")
print("-" * 50)

for (expiry, opt_type), group in type_groups.items():
    shorts = [dict(p) for p in group if p['quantity'] < 0]
    longs = [dict(p) for p in group if p['quantity'] > 0]
    
    type_margin = 0.0
    for short in shorts:
        s_strike = short['strike_price']
        s_qty = abs(short['quantity'])
        for long in longs:
            l_strike = long['strike_price']
            l_qty = long['quantity']
            matched = min(s_qty, l_qty)
            if matched > 0:
                width = abs(s_strike - l_strike)
                type_margin += width * matched * 100
                s_qty -= matched
                long['quantity'] -= matched
                if s_qty <= 0: break
    expiry_groups[expiry][opt_type] = type_margin

total_margin = 0
for expiry in sorted(expiry_groups.keys()):
    v = expiry_groups[expiry]
    risk = max(v['CALL'], v['PUT'])
    total_margin += risk
    print(f"{expiry}: Call=${v['CALL']:,.0f}, Put=${v['PUT']:,.0f} -> Risk=${risk:,.0f}")

print("-" * 50)
print(f"Total Combined Margin: ${total_margin:,.0f}")
