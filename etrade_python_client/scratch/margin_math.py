import re
from collections import defaultdict

def parse_and_explain():
    with open("/Users/btian/EtradePythonClient/etrade_python_client/screened_option_pairs.html", "r") as f:
        content = f.read()

    # Find the table body rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL)
    
    positions = []
    for row in rows:
        cols = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cols) < 13: continue
        
        def clean(s): return re.sub(r'<.*?>', '', s).strip()
        
        sym = clean(cols[1])
        if sym != "SPY": continue
        
        qty_val = float(clean(cols[2]))
        cp = clean(cols[6]).upper()
        strike_val = float(clean(cols[11]))
        exp = clean(cols[12])
        
        positions.append({"exp": exp, "cp": cp, "strike": strike_val, "qty": qty_val})

    print(f"Parsed {len(positions)} SPY legs.")
    
    # Expiry buckets
    expiry_map = defaultdict(lambda: {"CALL": [], "PUT": []})
    for p in positions:
        expiry_map[p["exp"]][p["cp"]].append(p)

    def calculate_side_margin(legs, cp):
        shorts = sorted([p for p in legs if p["qty"] < 0], key=lambda x: x["strike"], reverse=(cp == "PUT"))
        longs = sorted([p for p in legs if p["qty"] > 0], key=lambda x: x["strike"], reverse=(cp == "PUT"))
        
        margin = 0.0
        details = []
        
        # Simple pairing logic for explanation
        for short in shorts:
            s_qty = abs(short["qty"])
            while s_qty > 0 and longs:
                # Find best matching long (closest strike)
                best_idx = min(range(len(longs)), key=lambda idx: abs(short["strike"] - longs[idx]["strike"]))
                long = longs[best_idx]
                pair_qty = min(s_qty, long["qty"])
                strike_diff = abs(short["strike"] - long["strike"])
                pair_margin = strike_diff * pair_qty * 100.0
                margin += pair_margin
                details.append(f"  - Pair: {pair_qty}x {cp} {short['strike']}/{long['strike']} | Margin: ${pair_margin:,.2f}")
                s_qty -= pair_qty
                long["qty"] -= pair_qty
                if long["qty"] <= 0: longs.pop(best_idx)
            
            if s_qty > 0:
                # Naked logic (simplified estimate)
                naked_m = short["strike"] * 0.1 * 100.0 * s_qty # 10% naked estimate
                margin += naked_m
                details.append(f"  - Naked: {s_qty}x {cp} {short['strike']} | Margin: ${naked_m:,.2f} (est)")
        
        return margin, details

    total_max_risk = 0.0
    print("\n--- Detailed Margin Breakdown ---")
    for exp in sorted(expiry_map.keys()):
        print(f"\nExpiration: {exp}")
        c_margin, c_details = calculate_side_margin(expiry_map[exp]["CALL"], "CALL")
        p_margin, p_details = calculate_side_margin(expiry_map[exp]["PUT"], "PUT")
        
        if c_details:
            print(f"  Calls (Total: ${c_margin:,.2f}):")
            for d in c_details: print(d)
        if p_details:
            print(f"  Puts (Total: ${p_margin:,.2f}):")
            for d in p_details: print(d)
            
        max_m = max(c_margin, p_margin)
        total_max_risk += max_m
        print(f"  >> Expiry Max Risk = max(${c_margin:,.2f}, ${p_margin:,.2f}) = ${max_m:,.2f}")

    print("\n" + "="*40)
    print(f"FINAL TOTAL MARGIN = ${total_max_risk:,.2f}")
    print("="*40)

if __name__ == "__main__":
    parse_and_explain()
