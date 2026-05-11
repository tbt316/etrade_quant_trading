import json
from collections import defaultdict

TRACKER_FILE = "/Users/btian/EtradePythonClient/etrade_python_client/spy_tracking_data.json"

def recalculate_snapshots():
    with open(TRACKER_FILE, 'r') as f:
        data = json.load(f)
    
    updated_count = 0
    for date_key, snapshot in data.get("daily_snapshots", {}).items():
        positions = snapshot.get("positions", [])
        
        type_groups = defaultdict(list)
        for pos in positions:
            qty = pos.get('quantity', 0) or 0
            if qty == 0: continue
            
            # Extract fields handling both current snapshot and legacy formats
            expiry = pos.get('expiration_date', '') or pos.get('expiry_date', '')
            call_put = pos.get('call_put', '') or pos.get('option_type', '')
            key = (str(expiry), str(call_put).upper())
            
            type_groups[key].append({
                'qty': qty,
                'price': pos.get('last_price', 0) or pos.get('price', 0) or 0,
                'strike': pos.get('strike_price', 0) or pos.get('strike', 0) or 0
            })

        total_option_price = 0.0
        for key, group in type_groups.items():
            shorts = [p for p in group if p['qty'] < 0]
            longs = [p for p in group if p['qty'] > 0]
            
            opt_type = key[1]
            if opt_type == 'PUT':
                longs.sort(key=lambda x: x['strike'], reverse=True)
                shorts.sort(key=lambda x: x['strike'], reverse=True)
            else:
                longs.sort(key=lambda x: x['strike'])
                shorts.sort(key=lambda x: x['strike'])

            for short in shorts:
                total_option_price += short['qty'] * short['price'] * 100.0
                
                short_qty_abs = abs(short['qty'])
                for long in longs:
                    if long['qty'] <= 0: continue
                    matched_qty = min(short_qty_abs, long['qty'])
                    if matched_qty > 0:
                        total_option_price += matched_qty * long['price'] * 100.0
                        short_qty_abs -= matched_qty
                        long['qty'] -= matched_qty
                    if short_qty_abs <= 0:
                        break
        
        # Update snapshot if changed
        old_val = snapshot.get("total_option_price", 0.0)
        new_val = round(total_option_price, 2)
        if abs(old_val - new_val) > 0.01:
            snapshot["total_option_price"] = new_val
            updated_count += 1
            print(f"{date_key}: {old_val} -> {new_val}")
            
    if updated_count > 0:
        with open(TRACKER_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Updated {updated_count} snapshots.")
    else:
        print("No snapshots needed updating.")

if __name__ == "__main__":
    recalculate_snapshots()
