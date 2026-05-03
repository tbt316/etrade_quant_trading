
import json
import pandas_market_calendars as mcal
from datetime import datetime

with open('/Users/btian/EtradePythonClient/etrade_python_client/backtest_logs/backtest_path_SPY_20260426_120632.json', 'r') as f:
    data = json.load(f)

path_dates = [day['date'] for day in data['path']]
start_date = path_dates[0]
end_date = path_dates[-1]

nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = [d.strftime('%Y-%m-%d') for d in schedule.index]

missing = [d for d in trading_days if d not in path_dates]
print(f"Missing {len(missing)} trading days:")
for d in missing:
    print(d)
