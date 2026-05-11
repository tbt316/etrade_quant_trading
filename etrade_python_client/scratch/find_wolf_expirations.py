
import asyncio
import os
from datetime import datetime
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache

async def find_expirations():
    cache = OptionDataCache()
    client = MassiveAPIClient(cache)
    await client.__aenter__()
    try:
        underlying = "WOLF"
        as_of = datetime.now().strftime("%Y-%m-%d")
        
        # We need to find expirations. fetch_contracts_list requires an expiration.
        # MassiveAPIClient doesn't have a fetch_expirations method, but fetch_option_snapshot returns all of them.
        
        snapshot = await client.fetch_option_snapshot(underlying)
        expirations = set()
        for opt in snapshot:
            details = opt.get('details', {})
            exp = details.get('expiration_date')
            if exp:
                expirations.add(exp)
        
        sorted_exp = sorted(list(expirations))
        print(f"Available expirations for {underlying}:")
        for exp in sorted_exp:
            print(exp)
            
    finally:
        await client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(find_expirations())
