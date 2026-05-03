
import pandas_market_calendars as mcal
from datetime import datetime

nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2022-01-01', end_date='2022-01-31')
print(schedule)

is_open = nyse.is_open_on_minute(datetime(2022, 1, 19, 15, 0)) # 3 PM ET
print(f"Is NYSE open on 2022-01-19 15:00 ET? {is_open}")

date_to_check = '2022-01-19'
is_trading_day = date_to_check in schedule.index.strftime('%Y-%m-%d')
print(f"Is {date_to_check} in the NYSE schedule? {is_trading_day}")
