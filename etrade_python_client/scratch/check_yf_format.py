import yfinance as yf
from datetime import datetime

def test_yfinance_option():
    ticker = yf.Ticker("SPY")
    expirations = ticker.options
    if expirations:
        exp = expirations[0]
        print(f"Checking expiration: {exp}")
        chain = ticker.option_chain(exp)
        if not chain.calls.empty:
            sample_call = chain.calls.iloc[0]
            print(f"Sample Call Symbol: {sample_call['contractSymbol']}")
            print(f"Strike: {sample_call['strike']}")

if __name__ == "__main__":
    test_yfinance_option()
