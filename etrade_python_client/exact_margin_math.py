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

# Group by expiry first
expiry_groups = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0, "DETAILS": []})
groups = defaultdict(lambda: {"longs": [], "shorts": []})

for pos in positions:
    sym = (pos.get("symbol", "") or "").upper()
    cp = (pos.get("call_put", "") or "").upper()
    exp = pos.get("expiration_date", None)
    qty = _as_num(pos.get("quantity", 0), 0.0)
    if qty == 0:
        continue
    
    bucket = groups[(sym, cp, exp)]
    bucket["longs" if qty > 0 else "shorts"].append({"leg": pos, "qty": abs(qty)})

def _strike(rec):
    return _as_num(rec["leg"].get("strike_price", None), 0.0)

print(f"--- SCREENED_OPTION_PAIRS MARGIN MATH BREAKDOWN FOR {today} ---")

for (sym, cp, exp), parts in groups.items():
    shorts = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["shorts"]]
    if not shorts:
        continue
    longs = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["longs"]]

    reverse = True if cp == "PUT" else False
    shorts.sort(key=lambda rec: _strike(rec), reverse=reverse)
    longs.sort(key=lambda rec: _strike(rec), reverse=reverse)

    expiry_type_margin = 0.0
    for short in shorts:
        while short["qty"] > 0 and longs:
            best_idx = min(
                range(len(longs)),
                key=lambda idx: abs(_strike(short) - _strike(longs[idx]))
            )
            long = longs[best_idx]
            pair_qty = min(short["qty"], long["qty"])
            if pair_qty <= 0:
                if long["qty"] <= 0:
                    longs.pop(best_idx)
                continue
            strike_diff = abs(_strike(short) - _strike(long))
            margin = strike_diff * pair_qty * 100.0
            expiry_type_margin += margin
            
            expiry_groups[exp]["DETAILS"].append(f"  Spread: {cp} {_strike(short)}/{_strike(long)} x{int(pair_qty)} -> ${margin:,.0f}")
            
            short["qty"] -= pair_qty
            long["qty"] -= pair_qty
            if long["qty"] <= 0:
                longs.pop(best_idx)

        if short["qty"] > 0:
            m = calculate_naked_margin(short["leg"], short["qty"])
            expiry_type_margin += m
            expiry_groups[exp]["DETAILS"].append(f"  Naked: {cp} {_strike(short)} x{int(short['qty'])} -> ${m:,.0f}")
    
    expiry_groups[exp][cp] += expiry_type_margin

grand_total = 0
for expiry in sorted(expiry_groups.keys()):
    v = expiry_groups[expiry]
    risk = max(v['CALL'], v['PUT'])
    grand_total += risk
    print(f"\nEXPIRY: {expiry}")
    print(f"  Call Margin: ${v['CALL']:,.2f}")
    print(f"  Put Margin:  ${v['PUT']:,.2f}")
    print(f"  MAX RISK:    ${risk:,.2f}")
    for d in sorted(v["DETAILS"]):
        print(d)

print(f"\nGRAND TOTAL CALCULATED MARGIN: ${grand_total:,.2f}")
