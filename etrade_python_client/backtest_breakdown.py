import json

logfile = 'backtest_logs/backtest_path_SPY_20260428_085812.json'
with open(logfile, 'r') as f:
    data = json.load(f)

history = data.get('path', [])
# Look for the last entry that HAS active trades (in case the last one is empty)
last_snap = None
for i in range(len(history)-1, -1, -1):
    if history[i].get('active_trades'):
        last_snap = history[i]
        break

if not last_snap:
    print("No active trades found in history.")
    exit(1)

print(f"Details for Date: {last_snap['date']}")
print("-" * 60)
print(f"{'Entry Date':<12} {'Exp Date':<12} {'Strikes':<15} {'Qty':<5} {'Margin ($)':<12}")
print("-" * 60)

total_margin = 0
total_qty = 0

for t in sorted(last_snap['active_trades'], key=lambda x: x['entry_date']):
    s = t['short_strike']
    l = t['long_strike']
    qty = t['num_contracts']
    margin = t['margin_required']
    
    total_margin += margin
    total_qty += qty
    
    strikes = f"{s}/{l}"
    print(f"{t['entry_date']:<12} {t['expiration']:<12} {strikes:<15} {qty:<5} {margin:,.2f}")

print("-" * 60)
print(f"{'TOTAL':<12} {'':<12} {'':<15} {total_qty:<5} {total_margin:,.2f}")
