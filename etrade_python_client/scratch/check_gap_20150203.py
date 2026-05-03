import asyncio
import os
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache

async def check_data():
    cache = OptionDataCache()
    client = MassiveAPIClient(cache)
    await client.__aenter__()
    try:
        ticker = "O:SPY150320P00163000"
        date = "2015-02-03"
        
        print(f"Checking {ticker} on {date}...")
        
        # Check Quotes
        quote = await client.fetch_eod_quote(ticker, date)
        print(f"Quote: {quote}")
        
        # Check Sync Bars
        sync = await client.fetch_synchronized_ohlcv(ticker, "O:SPY150320P00142000", date)
        print(f"Sync Bars: {sync}")
        
        # Check Daily Aggs
        data = await client.fetch_contract_daily_bars(ticker, date, date, underlying="SPY", contract_type="put", strike=163.0, expiration="2015-03-20")
        print(f"Daily Agg Result: {data}")
        
        # Look at cache
        cached = cache.get_ohlcv(ticker, date)
        print(f"Cache state: {cached}")

    finally:
        await client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(check_data())
