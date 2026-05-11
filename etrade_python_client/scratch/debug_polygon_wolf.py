import asyncio
import aiohttp
from backtesting.massive_config import API_KEY, BASE_URL

async def debug_api():
    option_ticker = 'O:WOLF270115C00030000'
    from_date = '2024-01-01'
    to_date = '2026-05-06'
    url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{from_date}/{to_date}"
    params = {"adjusted": "false", "sort": "asc", "limit": 5000, "apiKey": API_KEY}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            if "results" in data:
                print(f"Result count: {len(data['results'])}")
                if len(data['results']) > 0:
                    print(f"First result: {data['results'][0]}")
                    print(f"Last result: {data['results'][-1]}")
            else:
                print("No results in data")
                print(data)

if __name__ == "__main__":
    asyncio.run(debug_api())
