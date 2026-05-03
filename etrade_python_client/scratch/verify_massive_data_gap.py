
import asyncio
import aiohttp
import json
from backtesting.massive_config import API_KEY, BASE_URL

async def verify_quote(ticker, date):
    url = f"{BASE_URL}/v3/quotes/{ticker}"
    params = {
        "timestamp.lte": f"{date}T23:59:59Z",
        "order": "desc",
        "sort": "timestamp",
        "limit": 5,
        "apiKey": API_KEY
    }
    
    print(f"\n--- Verifying {ticker} on {date} ---")
    print(f"URL: {url}")
    print(f"Params: {json.dumps(params, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            print("Response Snippet (first 2 results):")
            if "results" in data and data["results"]:
                print(json.dumps(data["results"][:2], indent=2))
            else:
                print(json.dumps(data, indent=2))
            
            if not data.get("results"):
                print(f"VERDICT: True. No quotes found for {ticker} on {date}.")
            else:
                # Check for bid/ask
                valid_quote = False
                for q in data["results"]:
                    if q.get("bid_price") and q.get("ask_price"):
                        valid_quote = True
                        print(f"Found valid quote: Bid={q['bid_price']}, Ask={q['ask_price']} at {q['participant_timestamp']}")
                        break
                if not valid_quote:
                    print(f"VERDICT: True. Quotes found but none have both Bid and Ask.")
                else:
                    print(f"VERDICT: False. Found valid quotes.")

async def main():
    date = "2022-01-19"
    tickers = ["O:SPY220218P00430000", "O:SPY220218P00410000"]
    for ticker in tickers:
        await verify_quote(ticker, date)

if __name__ == "__main__":
    asyncio.run(main())
