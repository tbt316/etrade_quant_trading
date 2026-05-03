
import json

with open('/Users/btian/EtradePythonClient/etrade_python_client/backtest_logs/backtest_path_SPY_20260426_120632.json', 'r') as f:
    data = json.load(f)

path = data['path']
gaps = []
current_gap = []

for day in path:
    if day['margin'] < 1000:
        current_gap.append(day['date'])
    else:
        if len(current_gap) >= 2:
            gaps.append((current_gap[0], current_gap[-1], len(current_gap)))
        current_gap = []

if len(current_gap) >= 2:
    gaps.append((current_gap[0], current_gap[-1], len(current_gap)))

for start, end, length in gaps:
    print(f"Gap from {start} to {end} ({length} days)")
