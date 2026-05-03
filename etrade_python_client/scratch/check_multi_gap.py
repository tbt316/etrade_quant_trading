import asyncio
import os
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache

async def check_data():
    cache = OptionDataCache()
    client = MassiveAPIClient(cache)
    await client.__aenter__()
    try:
        # Check a few strikes on Feb 3, 2015
        date = "2015-02-03"
        strikes = [160, 170, 180, 190, 200]
        
        for strike in strikes:
            ticker = f"O:SPY150320P{int(strike*1000):08d}"
            print(f"Checking {ticker} ({strike}) on {date}...")
            data = await client.fetch_contract_daily_bars(ticker, date, date, underlying="SPY", contract_type="put", strike=strike, expiration="2015-03-20")
            if data:
                print(f"  Result: {data[0]['close']} (Vol: {data[0]['volume']})")
            else:
                print(f"  Result: NO DATA")

    finally:
        await client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(check_data())
