import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,date,timezone
import math
import csv
import os
import matplotlib.pyplot as plt
import yfinance as yf
import polygonio_config  # Replace with your config filename
import pickle
import json
import asyncio
import aiohttp
import certifi
import ssl
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlencode
from monthly_cpu_bound import run_monthly_backtest_cpu_bound
from concurrent.futures import ProcessPoolExecutor
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import yfinance as yf
import random
from itertools import product
import bisect
import statistics
import pandas_market_calendars as mcal

new_data_entry_count = 0 
option_count = 0

# Configure logging
logging.basicConfig(
    filename='backtest_debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.CRITICAL
)

# SSL Context for aiohttp
ssl_context = ssl.create_default_context(cafile=certifi.where())

OPTION_TRADE_COST = 0.5
LOOKBACK_WINDOW = 5
USE_CLOSE_PRICE_SETTING = False
LOAD_MONTHLY_DATA = False
NUM_CORES = 8
VALIDATION_MONTH_FORWARD = 6
VOL_THRESHOLD = -1
SPREAD_COST = 0.1
MIN_PROFIT = 0.05 
MAX_PROFIT = 0.2
OPTION_CHAIN_FORCE_UPDATE = False
SKIP_EARNINGS = False
SKIP_MISSING_STRIKE_TRADE = False
OPTION_RANGE = 0.1
USE_TRADE_DATA = True
PRICE_INTERPOLATE = True
INITIAL_CAPITAL = 100000
COVER_CALL_MAX_POSITIONS = 20
IV_THRESHOLD_MIN = 0

PREMIUM_PRICE_MODE = "trade"   # choose from: "close_price", "mid_price", "trade_price"

nyse = mcal.get_calendar('NYSE')

# Map the mode to the actual column name
PREMIUM_FIELD_MAP = {
    "close": "close_price",
    "mid":   "mid_price",
    "trade":  "trade_price",      # or whatever the 3rd field is called
}
DELTA_FIELD_MAP = {
    "close": "close_price_delta",
    "mid":   "mid_price_delta",
    "trade":  "trade_price_delta",      # or whatever the 3rd field is called
}
# ---------------------------------------------------------
# 1. GLOBAL SETTINGS
# ---------------------------------------------------------

# Directory where we store .pkl data for each ticker
CACHE_DIR = "polygon_api_option_data"

# In-memory dictionaries keyed by ticker:
# stored_option_price[ticker][pricing_date][strike_price][expiration_date][call_put] = {
#     "mid_price": float,
#     "ask_price": float,
#     "bid_price": float,
#     "ask_size": int,
#     "bid_size": int,
#     "close_price": float,
#     "close_volume": int
# }
stored_option_price = {}

# stored_option_chain[ticker][expiration_date][as_of_date] = list_of_strikes
stored_option_chain = {}

# Directory where we store monthly backtest results
MONTHLY_BACKTEST_DIR = "monthly_backtest_data"
if not os.path.exists(MONTHLY_BACKTEST_DIR):
    os.makedirs(MONTHLY_BACKTEST_DIR)

# ---------------------------------------------------------
# 2. DIRECTORY + FILENAME HELPERS
# ---------------------------------------------------------

def ensure_cache_dir():
    """Make sure the 'polygon_api_option_data' directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_price_cache_file(ticker):
    """
    Return the pkl filepath for storing the option price cache of a specific ticker.
    Example: polygon_api_option_data/AAPL_stored_option_price.pkl
    """
    ticker_upper = ticker.upper()
    return os.path.join(CACHE_DIR, f"{ticker_upper}_stored_option_price.pkl")

def get_chain_cache_file(ticker):
    """
    Return the pkl filepath for storing the option chain cache of a specific ticker.
    Example: polygon_api_option_data/AAPL_stored_option_chain.pkl
    """
    ticker_upper = ticker.upper()
    return os.path.join(CACHE_DIR, f"{ticker_upper}_stored_option_chain.pkl")

def get_monthly_backtest_file(ticker: str, global_start_date: str, global_end_date: str) -> str:
    """
    Return the pkl filepath for storing monthly_recursive_backtest results.
    We embed the timestamp + parameters in the filename for uniqueness.
    """
    # You may adjust how the timestamp or parameters are appended
    # We'll store just the timestamp here and store parameters inside the file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monthly_{ticker}_{global_start_date}_to_{global_end_date}_{timestamp_str}.pkl"
    return os.path.join(MONTHLY_BACKTEST_DIR, filename)

# ---------------------------------------------------------
# 3. LOADING & SAVING DATA
# ---------------------------------------------------------

def load_stored_option_data(ticker):
    """
    Load existing data from .pkl files for this ticker, if present.
    We do NOT discard old data. Instead, we merge any on-disk data
    with what's in memory to ensure we keep it all.

    After calling this, stored_option_price[ticker] and 
    stored_option_chain[ticker] contain old + new data.
    """
    print("Loading stored option data for ticker:", ticker)
    global stored_option_price, stored_option_chain

    ensure_cache_dir()

    if ticker not in stored_option_price:
        stored_option_price[ticker] = {}
    if ticker not in stored_option_chain:
        stored_option_chain[ticker] = {}

    price_cache_file = get_price_cache_file(ticker)
    chain_cache_file = get_chain_cache_file(ticker)
    
    # Load price cache if present
    if os.path.isfile(price_cache_file):
        try:
            with open(price_cache_file, "rb") as f:
                if os.path.getsize(price_cache_file) > 0:  # Check if file is non-empty
                    on_disk_price_dict = pickle.load(f)
                    merge_nested_dicts(stored_option_price[ticker], on_disk_price_dict)
                else:
                    print(f"Warning: Price cache file {price_cache_file} is empty; skipping.")
        except EOFError:
            print(f"Warning: Failed to load {price_cache_file} due to EOFError; treating as empty.")
            stored_option_price[ticker] = {}

    # Load chain cache if present
    if os.path.isfile(chain_cache_file):
        try:
            with open(chain_cache_file, "rb") as f:
                if os.path.getsize(chain_cache_file) > 0:  # Check if file is non-empty
                    on_disk_price_dict = pickle.load(f)
                    merge_nested_dicts(stored_option_chain[ticker], on_disk_price_dict)
                else:
                    print(f"Warning: Chain cache file {chain_cache_file} is empty; skipping.")
        except EOFError:
            print(f"Warning: Failed to load {chain_cache_file} due to EOFError; treating as empty.")
            stored_option_chain[ticker] = {}
    else:
        stored_option_chain[ticker] = {}

    if stored_option_price[ticker]:
        # Sort the dates (assuming keys are date strings in "yyyy-mm-dd" format)
        last_date = sorted(stored_option_price[ticker].keys())[-1]
        last_date_prices = stored_option_price[ticker].get(last_date, {})
        return last_date_prices
    else:
        print("No option price data available.")
        return {}
    
from multiprocessing import Lock  # Use this instead of threading.Lock for ProcessPoolExecutor
# Global lock for file access (shared across processes)
file_lock = Lock()

def save_stored_option_data(ticker):
    """
    Re-pickle the entire dictionary for the ticker by first comparing with existing
    data in the file and merging if necessary. This ensures we do not lose any previously
    stored data.
    """
    global new_data_entry_count 
    new_data_entry_count = 0 #reset the counter
    global stored_option_price, stored_option_chain
    ensure_cache_dir()

    price_cache_file = get_price_cache_file(ticker)
    chain_cache_file = get_chain_cache_file(ticker)

    # Initialize existing data dictionaries
    existing_price = {}
    existing_chain = {}

    # Load existing price data with synchronization
    with file_lock:
        if os.path.isfile(price_cache_file):
            try:
                with open(price_cache_file, "rb") as f:
                    if os.path.getsize(price_cache_file) > 0:  # Check for non-empty file
                        existing_price = pickle.load(f)
                    else:
                        print(f"Warning: {price_cache_file} is empty; treating as empty dict.")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load {price_cache_file} ({e}); using empty dict.")

    # Load existing chain data with synchronization
    with file_lock:
        if os.path.isfile(chain_cache_file):
            try:
                with open(chain_cache_file, "rb") as f:
                    if os.path.getsize(chain_cache_file) > 0:
                        existing_chain = pickle.load(f)
                    else:
                        print(f"Warning: {chain_cache_file} is empty; treating as empty dict.")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load {chain_cache_file} ({e}); using empty dict.")

    # Ensure in-memory dictionaries exist
    if ticker not in stored_option_price:
        stored_option_price[ticker] = {}
    if ticker not in stored_option_chain:
        stored_option_chain[ticker] = {}

    # Merge in-memory data with file data
    merge_nested_dicts(existing_price, stored_option_price[ticker])
    merge_nested_dicts(existing_chain, stored_option_chain[ticker])

    # Write merged data back to files with synchronization
    with file_lock:
        with open(price_cache_file, "wb") as f:
            pickle.dump(existing_price, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Verify write
        if os.path.getsize(price_cache_file) == 0:
            print(f"Error: {price_cache_file} is still empty after write!")

    with file_lock:
        with open(chain_cache_file, "wb") as f:
            pickle.dump(existing_chain, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Verify write
        if os.path.getsize(chain_cache_file) == 0:
            print(f"Error: {chain_cache_file} is still empty after write!")

def merge_nested_dicts(original_dict, on_disk_dict):
    """
    Recursively merge on_disk_dict into original_dict so we don't lose
    any data. If there's a conflict, we let original_dict's data remain
    unless it too is a dict that can be recursively merged.
    """
    for key, val in on_disk_dict.items():
        if isinstance(val, dict) and key in original_dict and isinstance(original_dict[key], dict):
            merge_nested_dicts(original_dict[key], val)
        else:
            # If not a dict or key not present, overwrite or add
            original_dict[key] = val

# ---------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------

def find_closest_strike(option_list, option_data, target):
    premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]
    valid_options = []
    for opt, prem in zip(option_list, option_data):
        if not isinstance(prem, dict): 
            continue
        if prem.get(premium_field) is None or prem[premium_field] <= 0:
            continue
        valid_options.append(opt)
    if not valid_options:
        return None
    closest_option = min(valid_options, key=lambda opt: abs(opt['strike_price'] - target))
    return closest_option['strike_price']

def convert_polygon_to_etrade_ticker(polygon_ticker, metadata=None):
    """
    Converts a Polygon.io API ticker symbol to the correct E*TRADE API ticker symbol.
    
    Args:
        polygon_ticker (str): The ticker symbol from Polygon.io.
        metadata (dict, optional): Additional information related to the stock (e.g. company name, exchange).
    
    Returns:
        str: The converted ticker symbol appropriate for E*TRADE.
    """
    # Example: Known ticker corrections for differences between Polygon.io and E*TRADE.
    # Extend this dictionary as needed.
    known_fixes = {
        "BRK.B": "BRK-B",   # Polygon may use dot notation while E*TRADE expects a dash.
        "BF.B": "BF-B",
        "VIX": "^VIX",
        "VIXW": "^VIX",
        "SPX": "^GSPC",
        # Add more known mappings here
    }
    
    # Use the dictionary lookup first if there is an exact match.
    if polygon_ticker in known_fixes:
        return known_fixes[polygon_ticker]

    return polygon_ticker

def get_random_user_agent() -> str:
    """Return one random desktop / mobile User‑Agent."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 13_6_1)",
        "Mozilla/5.0 (iPad; CPU OS 13_6_1)",
    ]
    return random.choice(user_agents)

from typing import Union

def fetch_yfinance_data(
    ticker: str,
    start_date: Union[date, datetime],
    end_date: Union[date, datetime],
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download close data from Yahoo Finance and return a DataFrame with 'date' and 'close' columns.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    start_date, end_date : date | datetime
        The date range for the data.
    interval : {"1d", "1m"}
        The data interval ('1d' for daily, '1m' for 1-minute).
    auto_adjust : bool
        Whether to auto-adjust prices.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'date' (datetime.date objects) and 'close' (float).
    """
    if interval not in {"1d", "1m"}:
        raise ValueError("interval must be '1d' or '1m'")

    # Convert date/datetime to naive datetime
    def _to_naive_dt(dt: Union[date, datetime]) -> datetime:
        if isinstance(dt, date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        return dt

    start_dt = _to_naive_dt(start_date)
    end_dt = _to_naive_dt(end_date)

    # For intraday, extend end_dt to now if start == end
    if interval == "1m" and start_dt.date() == end_dt.date():
        end_dt = datetime.now()

    # Yahoo's chart API treats period2 as exclusive
    if interval == "1d":
        end_dt += timedelta(days=1)
    else:  # "1m"
        end_dt += timedelta(seconds=60)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; YahooFinanceClient/1.0)",
            "Accept": "application/json, text/plain, */*",
        }
    )

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={int(start_dt.timestamp())}"
        f"&period2={int(end_dt.timestamp())}"
        f"&interval={interval}"
        f"&events=history"
        f"&adjusted={str(auto_adjust).lower()}"
    )

    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    try:
        result = payload["chart"]["result"][0]
        ts = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unexpected response structure or no data returned")

    if not ts or closes is None:
        raise ValueError("No data returned from Yahoo Finance")

    # Create DataFrame with naive datetime
    idx = pd.to_datetime(ts, unit="s")
    df = pd.DataFrame({"datetime": idx, "close": closes}).dropna()

    # Extract date as datetime.date objects
    df["date"] = df["datetime"].dt.date
    # Keep only 'date' and 'close' columns
    df = df[["date", "close"]]

    return df

def get_historical_prices(ticker, start_date, end_date,
                          vol_lookback=5, data_source='yfinance'):
    """
    Fetch historical prices using either Yahoo Finance (yfinance) or Polygon.io and
    return a DataFrame with calculated metrics.
    """
    # ── cache setup ──────────────────────────────────────────────────────────────
    if data_source == 'yfinance':
        yf_ticker = convert_polygon_to_etrade_ticker(ticker)
        cache_dir = os.path.join("yfinance", "prices")
        cache_file = os.path.join(cache_dir, f"{yf_ticker}_prices.csv")
    elif data_source == 'polygon':
        cache_dir = os.path.join("polygon", "prices")
        cache_file = os.path.join(cache_dir, f"{ticker}_prices.csv")
    else:
        raise ValueError("Invalid data_source. Choose 'yfinance' or 'polygon'.")
    os.makedirs(cache_dir, exist_ok=True)

    # ── cache check ─────────────────────────────────────────────────────────────
    use_cache = False
    today            = date.today()
    requested_start  = datetime.strptime(start_date, "%Y-%m-%d").date()
    requested_end    = datetime.strptime(end_date,   "%Y-%m-%d").date()

    # 1) figure out what "today" really means for caching
    #    (if it's weekend, roll back to last Friday)
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        last_weekday = today - timedelta(days=(today.weekday() - 4))
    else:
        last_weekday = today

    # 2) if user asked for an end date on weekend, treat that as last_weekday
    effective_end = min(requested_end, last_weekday)

    if os.path.exists(cache_file):
        file_mod_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        try:
            cached_df     = pd.read_csv(cache_file, parse_dates=['date'])
            cached_start  = cached_df['date'].min().date()
            cached_end    = cached_df['date'].max().date()
            days_gap      = (cached_start - requested_start).days

            # Only use cache if:
            #   • it covers our lookback window (days_gap ≤ 7)
            #   • it extends through effective_end
            #   • AND either it was updated on last_weekday OR we only need data before last_weekday
            use_cache = (
                days_gap <= 7
                and cached_end >= effective_end
                and (file_mod_date >= last_weekday or requested_end < last_weekday)
            )

        except Exception as e:
            print(f"Warning: cache read error for {ticker}: {e}")

    df = None
    if use_cache:
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=['date'])
            df = cached_df.loc[
                (cached_df['date'].dt.date >= requested_start)
                & (cached_df['date'].dt.date <= requested_end)
            ].copy()
        except Exception as e:
            print(f"Warning: cache processing error for {ticker}: {e}")
            df, use_cache = None, False

    # ── fetch if cache unusable ─────────────────────────────────────────────────
    if not use_cache or df is None or df.empty:
        if data_source == 'yfinance':
            print(f"Fetching price data from yfinance for {ticker}")
            df = fetch_yfinance_data(yf_ticker, datetime.combine(requested_start, datetime.min.time()), datetime.combine(requested_end, datetime.min.time()))
            df.to_csv(cache_file, index=False)

        elif data_source == 'polygon':
            # ---------- Polygon (≤500‑bar chunks) ----------
            api_key   = polygonio_config.API_KEY
            max_bars  = 500
            chunk_start = requested_start
            all_chunks = []

            while chunk_start <= requested_end:
                chunk_end = min(requested_end, chunk_start + timedelta(days=max_bars - 1))
                s = chunk_start.strftime("%Y-%m-%d")
                e = chunk_end.strftime("%Y-%m-%d")
                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
                    f"/range/1/day/{s}/{e}"
                    f"?adjusted=false&sort=asc&limit={max_bars}&apiKey={api_key}"
                )

                print(f"Fetching Polygon data for {ticker}: {s} → {e}")
                try:
                    resp = requests.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    if 'results' in data and data['results']:
                        tmp = pd.DataFrame(data['results'])
                        tmp['date'] = pd.to_datetime(tmp['t'], unit='ms')
                        tmp = tmp[['date', 'c']].rename(columns={'c': 'close'})
                        all_chunks.append(tmp)
                    else:
                        print(f"  ⚠️  No data for {ticker} {s}–{e}")
                except Exception as e:
                    print(f"Polygon error {ticker} {s}–{e}: {e}")

                # advance to next slice
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.2)  # gentle on the API

            if not all_chunks:
                return pd.DataFrame(
                    columns=['date', 'close', 'returns', 'vol_20', 'MA', 'ticker']
                )

            df = (
                pd.concat(all_chunks, ignore_index=True)
                  .drop_duplicates(subset=['date'], keep='first')
                  .sort_values('date')
                  .reset_index(drop=True)
            )
            df.to_csv(cache_file, index=False)

    # ── post‑processing (unchanged) ─────────────────────────────────────────────
    if df is None or df.empty:
        return pd.DataFrame(
            columns=['date', 'close', 'returns', 'vol_20', 'MA', 'ticker']
        )
    df["returns"] = df["close"].pct_change()
    df["vol_20"]  = df["returns"].rolling(window=vol_lookback).std()
    df["MA"]      = df["close"].rolling(window=vol_lookback).mean()
    df["ticker"]  = ticker

    return df


def get_all_weekdays(weekday, start_date, end_date, expiring_wk, trading_dates_df):
    """
    Return a list of all Fridays (or Wednesdays) as datetime objects in [start_date, end_date],
    adjusted to the last trading day on or before each candidate date.

    Args:
        weekday (str): 'Friday' or 'Wednesday'
        start_date (datetime): Datetime object for the start date
        end_date (datetime): Datetime object for the end date
        expiring_wk (int): Number of weeks between expiration dates (converted to int, not used in loop)
        trading_dates_df (DataFrame): DataFrame with 'date' column containing trading datetimes
    """
    fridays = []
    current = start_date

    # Convert expiring_wk to int (though unused in the loop)
    expiring_wk = int(expiring_wk)

    # Get sorted list of trading datetimes from the DataFrame
    trading_datetimes = sorted(trading_dates_df['date'])

    # Set weekday index: 4 for Friday, 2 for Wednesday
    if weekday == 'Friday':
        weekday_index = 4
    elif weekday == 'Wednesday':
        weekday_index = 2
    else:
        raise ValueError("weekday must be 'Friday' or 'Wednesday'")

    # Find the first matching weekday >= start_date
    while current <= end_date and current.weekday() != weekday_index:
        current += timedelta(days=1)

    # Collect adjusted dates
    while current <= end_date:
        # Find the last trading day on or before current
        index = bisect.bisect_right(trading_datetimes, current) - 1
        if index >= 0:  # Ensure there’s a trading day before current
            last_trading_day = trading_datetimes[index]
            # Only append if the trading day is on or after start_date
            if last_trading_day >= start_date:
                fridays.append(last_trading_day)
        # Move to the next candidate date (7 days later)
        current += timedelta(days=7)

    return fridays

def get_earnings_dates(ticker, start_date, end_date):
    """
    Fetches all known earnings dates for 'ticker' using yfinance's
    get_earnings_dates(limit=...) and returns them as a set of datetime.date
    objects within the [start_date, end_date] window.
    
    Args:
        ticker (str): The stock symbol, e.g. "AAPL".
        start_date (str): Start of the date range in "YYYY-MM-DD" format.
        end_date (str): End of the date range in "YYYY-MM-DD" format.
        
    Returns:
        set[datetime.date]: All earnings dates in the specified range.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    earnings_dates = set()
    yf_ticker = convert_polygon_to_etrade_ticker(ticker)

    # Create directory for cached data if it doesn't exist
    cache_dir = os.path.join("yfinance", "earnings")
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(cache_dir, f"{yf_ticker}_earnings.csv")

    # Check if cache file exists and is from today
    use_cache = False
    today = datetime.today().date()
    if os.path.exists(cache_file):
        # Check when the file was last modified
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        use_cache = (file_mod_time == today)
        if use_cache == False:
            print(f"Warning: Earnings cache file {cache_file} is outdated, file_mod_time: {file_mod_time}, today: {today}")
    else:
        print(f"Warning: Earnings cache file {cache_file} does not exist")
        
    if use_cache:
        try:
            # Read from cache
            cached_data = pd.read_csv(cache_file, parse_dates=['earnings_date'])
            
            # Check if this is a "no earnings" indicator file
            if 'no_earnings_flag' in cached_data.columns and len(cached_data) == 1 and cached_data.iloc[0]['no_earnings_flag']:
                return earnings_dates
                
            for _, row in cached_data.iterrows():
                date_only = row['earnings_date'].date()
                if start_dt <= date_only <= end_dt:
                    earnings_dates.add(date_only)
            return earnings_dates
        except Exception as e:
            print(f"Warning: Error reading cached earnings dates for {ticker}. Error: {e}")
            # If error reading cache, fall back to API query
            use_cache = False

    # If cache is not usable, query yfinance API
    try:
        print("Fetching earnings dates from yfinance for", ticker)
        t = yf.Ticker(yf_ticker)
        df_earnings_dates = t.get_earnings_dates(limit=50)  # You can adjust the limit as needed
        
        if df_earnings_dates is not None and not df_earnings_dates.empty:
            # Create a DataFrame to store earnings dates
            cache_df = pd.DataFrame(columns=['earnings_date'])
            all_dates = []
            
            # Process the earnings dates
            for dt_obj in pd.to_datetime(df_earnings_dates.index):
                date_only = dt_obj.date()
                all_dates.append(date_only)
                # Filter for dates within [start_dt, end_dt]
                if start_dt <= date_only <= end_dt:
                    earnings_dates.add(date_only)
            
            # Save to cache
            cache_df['earnings_date'] = all_dates
            cache_df.to_csv(cache_file, index=False)
        else:
            # No earnings dates found - create a marker file
            no_earnings_df = pd.DataFrame({
                'earnings_date': [today],
                'no_earnings_flag': [True],
                'checked_until': [today.strftime("%Y-%m-%d")]
            })
            no_earnings_df.to_csv(cache_file, index=False)
    except Exception as e:
        print(f"Warning: Unable to fetch earnings dates for {ticker}. Error: {e}")
        
        # Create an error indicator file
        error_df = pd.DataFrame({
            'earnings_date': [today],
            'error_flag': [True],
            'error_message': [str(e)],
            'checked_until': [today.strftime("%Y-%m-%d")]
        })
        error_df.to_csv(cache_file, index=False)
        
    return earnings_dates


def calculate_delta(ticker: str, pricing_date: str, expiration_date: str, call_put: str, force_delta_update: bool = False):
    """
    Calculates the 'close_price_delta' and 'mid_price_delta' for each option in the chain.
    For each strike, finds one strike above and one below, computes forward and backward differences,
    averages them, and rounds to 4 decimal places. Skips calculation if deltas are already present unless force_delta_update is True.

    Args:
        ticker (str): The underlying asset ticker symbol.
        pricing_date (str): The date for pricing in 'YYYY-MM-DD' format.
        expiration_date (str): The expiration date in 'YYYY-MM-DD' format.
        call_put (str): 'call' or 'put'.
        force_delta_update (bool, optional): If True, recalculates and overwrites delta values even if they exist. Defaults to False.
    """
    if ticker not in stored_option_price or pricing_date not in stored_option_price[ticker]:
        return

    # Collect all strikes for this expiration_date and call_put
    strikes = []
    for strike in stored_option_price[ticker][pricing_date]:
        if (expiration_date in stored_option_price[ticker][pricing_date][strike] and
                call_put in stored_option_price[ticker][pricing_date][strike][expiration_date]):
            strikes.append(strike)

    if not strikes:
        return

    # Sort strikes in ascending order
    if call_put == 'call':
        strikes.sort()
    elif call_put == 'put':
        strikes.sort(reverse=True)

    # Calculate deltas for each strike
    for K in strikes:
        option_data = stored_option_price[ticker][pricing_date][K][expiration_date][call_put]

        # Skip if both deltas are present and force_delta_update is False
        if not force_delta_update and 'close_price_delta' in option_data and 'mid_price_delta' in option_data:
            continue

        # Find the closest strike below (K_prev) and above (K_next)
        K_prev = max([s for s in strikes if s < K], default=None)
        K_next = min([s for s in strikes if s > K], default=None)

        # Initialize deltas
        close_price_delta = 0.0
        mid_price_delta = 0.0
        close_price_count = 0
        mid_price_count = 0

        # Current prices
        price_K = option_data

        # Forward difference (using K_next)
        if K_next is not None:
            price_K_next = stored_option_price[ticker][pricing_date][K_next][expiration_date][call_put]
            if 'close_price' in price_K and 'close_price' in price_K_next and price_K['close_price'] is not None and price_K_next['close_price'] is not None:
                forward_close_delta = (price_K_next['close_price'] - price_K['close_price']) / (K_next - K)
                close_price_delta += forward_close_delta
                close_price_count += 1
            if 'mid_price' in price_K and 'mid_price' in price_K_next and price_K['mid_price'] is not None and price_K_next['mid_price'] is not None:
                forward_mid_delta = (price_K_next['mid_price'] - price_K['mid_price']) / (K_next - K)
                mid_price_delta += forward_mid_delta
                mid_price_count += 1

        # Backward difference (using K_prev)
        if K_prev is not None:
            price_K_prev = stored_option_price[ticker][pricing_date][K_prev][expiration_date][call_put]
            if 'close_price' in price_K and 'close_price' in price_K_prev and price_K['close_price'] is not None and price_K_prev['close_price'] is not None:
                # if price_K['close_price'] > price_K_prev['close_price'] and price_K_prev['close_price'] > 0:
                #     price_K['close_price'] = price_K_prev['close_price'] # to avoid error data, force the option that is further out of money to be the same as the one that is closer to the money
                backward_close_delta = (price_K['close_price'] - price_K_prev['close_price']) / (K - K_prev)
                close_price_delta += backward_close_delta
                close_price_count += 1
            if 'mid_price' in price_K and 'mid_price' in price_K_prev and price_K['mid_price'] is not None and price_K_prev['mid_price'] is not None:
                # if price_K['mid_price'] > price_K_prev['mid_price'] and price_K_prev['mid_price'] > 0:
                #     price_K['mid_price'] = price_K_prev['mid_price']
                backward_mid_delta = (price_K['mid_price'] - price_K_prev['mid_price']) / (K - K_prev)
                mid_price_delta += backward_mid_delta
                mid_price_count += 1

        # Average the deltas and round to 4 decimal places
        if close_price_count > 0:
            option_data['close_price_delta'] = round(close_price_delta / close_price_count, 4)
        else:
            option_data['close_price_delta'] = 0.0  # No valid differences available

        if mid_price_count > 0:
            option_data['mid_price_delta'] = round(mid_price_delta / mid_price_count, 4)
        else:
            option_data['mid_price_delta'] = 0.0  # No valid differences available

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

def calculate_implied_volatility(close_price, strike_price, option_price, days_to_expire, risk_free_rate, dividend_yield, option_type='call'):
    """
    Calculate implied volatility using the Black-Scholes model and Newton-Raphson method.
    
    Parameters:
    - close_price (float): Current stock price
    - strike_price (float): Option strike price
    - option_price (float): Market price of the option
    - days_to_expire (float): Days until option expiration
    - risk_free_rate (float): Annualized risk-free rate (e.g., 0.05 for 5%)
    - dividend_yield (float): Annualized dividend yield (e.g., 0.02 for 2%)
    - option_type (str): 'call' or 'put' (default: 'call')
    
    Returns:
    - float: Implied volatility (annualized) or None if calculation fails
    """
    # Convert days to years
    time_to_expiry = days_to_expire / 365.0
    
    # Black-Scholes option price function
    def black_scholes_price(sigma):
        d1 = (np.log(close_price / strike_price) + 
              (risk_free_rate - dividend_yield + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)
        
        if option_type.lower() == 'call':
            price = (close_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - 
                     strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                     close_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1))
        
        return price - option_price  # Difference between model and market price
    
    # Vega (sensitivity to volatility)
    def vega(sigma):
        d1 = (np.log(close_price / strike_price) + 
              (risk_free_rate - dividend_yield + 0.5 * sigma**2) * time_to_expiry) / (sigma * np.sqrt(time_to_expiry))
        return close_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry)
    
    # Input validation
    if (close_price <= 0 or strike_price <= 0 or option_price <= 0 or 
        time_to_expiry <= 0 or option_price >= close_price):
        return None
    
    try:
        # Initial guess for volatility (0.2 = 20%)
        implied_vol = newton(
            black_scholes_price, 
            x0=0.2, 
            fprime=vega, 
            maxiter=100, 
            tol=1e-6
        )
        
        # Ensure positive volatility and reasonable range
        if implied_vol <= 0 or implied_vol > 5.0:  # Cap at 500% volatility
            return None
            
        return implied_vol
    
    except (RuntimeError, OverflowError, ValueError):
        return None

async def pull_option_chain_data(ticker, call_put, expiration_str, as_of_str, close_price, client, force_otm=False, force_update=False):
    # First collect all needed chain data
    unique_chain_requests = [(expiration_str, as_of_str, "call"), (expiration_str, as_of_str, "put")]
    chain_data = await client.get_option_chains_batch_async(ticker, list(unique_chain_requests), force_update=force_update)

    # Get available strike dictionaries
    call_strike_dict = chain_data[ticker][expiration_str][as_of_str]["call"]
    put_strike_dict = chain_data[ticker][expiration_str][as_of_str]["put"]
    
    # Check if we have valid data
    if not call_strike_dict or not put_strike_dict:
        # print(f"No valid strikes found for {ticker} expiration {expiration_str}, as of {as_of_str}")
        return None, None, None, None, None

    # Prepare options for premium query
    call_options = []
    put_options = []
    
    if ticker in ['SPY','QQQ','TQQQ','SQQQ']:
        scale = 1
    elif ticker in ['SPX','NDX']:
        scale = 0.5
    else:
        scale = 1

    if force_otm:
        price_limit_percent = 0
    else:
        price_limit_percent = -0.5
    
    for strike in call_strike_dict.keys():
        if close_price * (1+price_limit_percent*scale) < strike < close_price * (1 + 0.5 * scale): 
            call_options.append({
                'strike_price': strike,
                'call_put': 'call',
                'expiration_date': expiration_str,
                'quote_timestamp': as_of_str,
                'option_ticker': call_strike_dict[strike]
            })
    for strike in put_strike_dict.keys():
        if close_price * (1 - 0.5*scale) < strike < close_price * (1 - scale * price_limit_percent):
            put_options.append({
                'strike_price': strike,
                'call_put': 'put',
                'expiration_date': expiration_str,
                'quote_timestamp': as_of_str,
                'option_ticker': put_strike_dict[strike]
            })

    # Compute min and max strike prices for calls and puts
    if call_options:
        call_strikes = [opt['strike_price'] for opt in call_options]
        min_call_strike = min(call_strikes)
        max_call_strike = max(call_strikes)
    else:
        min_call_strike = None
        max_call_strike = None

    if put_options:
        put_strikes = [opt['strike_price'] for opt in put_options]
        min_put_strike = min(put_strikes)
        max_put_strike = max(put_strikes)
    else:
        min_put_strike = None
        max_put_strike = None

    # Prepare arrays to store all option data
    all_call_data = []
    all_put_data = []
    call_options_to_fetch = []
    put_options_to_fetch = []
    threshold = 1 if force_update else 0
    # Process call options
    for i, opt in enumerate(call_options):
        strike = opt['strike_price']
        pricing_date = opt['quote_timestamp']
        expiration = opt['expiration_date']
        data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get('call', {})

        if data and ((data.get('close_price', -1) >= threshold and PREMIUM_PRICE_MODE == 'close') or ((data.get('mid_price',-1) >= threshold and PREMIUM_PRICE_MODE == 'mid')) or ( data.get('trade_price',-1)>=threshold and PREMIUM_PRICE_MODE == 'trade')) and OPTION_CHAIN_FORCE_UPDATE == False:
            all_call_data.append(data)
        else:
            if 'call' in call_put:
                call_options_to_fetch.append(opt)
            all_call_data.append(None)  # Placeholder

    # Process put options
    for i, opt in enumerate(put_options):
        strike = opt['strike_price']
        pricing_date = opt['quote_timestamp']
        expiration = opt['expiration_date']
        data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get('put', {})
        
        if data and ((data.get('close_price', -1) >= threshold and PREMIUM_PRICE_MODE == 'close') or ((data.get('mid_price',-1) >= threshold and PREMIUM_PRICE_MODE == 'mid')) or ( data.get('trade_price',-1)>=threshold and PREMIUM_PRICE_MODE == 'trade')) and OPTION_CHAIN_FORCE_UPDATE == False:
            all_put_data.append(data)
        else:
            if 'put' in call_put:
                put_options_to_fetch.append(opt)
            all_put_data.append(None)  # Placeholder

    # Create dictionary with min and max strike prices
    strike_range_dict = {
        'call': {'min_strike': min_call_strike, 'max_strike': max_call_strike},
        'put': {'min_strike': min_put_strike, 'max_strike': max_put_strike}
    } if min_call_strike and min_put_strike else None

    range_ok = strike_range_dict is not None and max_call_strike > close_price * (1+OPTION_RANGE) and min_put_strike < close_price * (1-OPTION_RANGE)
    # Fetch missing data in one batch
    if len(call_options_to_fetch) > 0.1 * len(call_options) or len(put_options_to_fetch) > 0.1 * len(put_options) or force_update:
        print(f"Fetching {len(call_options_to_fetch)} call options (total call options: {len(call_options)}) and {len(put_options_to_fetch)} put options (total put options: {len(put_options)}) for {ticker} on {as_of_str}")
        fetched_data = await client.get_option_prices_batch_async(ticker, call_options_to_fetch + put_options_to_fetch)
        # Update call data
        fetched_call_data = fetched_data[:len(call_options_to_fetch)]
        call_idx = 0
        if 'call' in call_put:
            for i in range(len(all_call_data)):
                if all_call_data[i] is None:
                    all_call_data[i] = fetched_call_data[call_idx]
                    call_idx += 1

        # Update put data
        fetched_put_data = fetched_data[len(call_options_to_fetch):]
        put_idx = 0
        if 'put' in call_put:
            for i in range(len(all_put_data)):
                if all_put_data[i] is None:
                    all_put_data[i] = fetched_put_data[put_idx]
                    put_idx += 1

    # calculate_delta(ticker, as_of_str, expiration_str, 'call', force_delta_update=False)
    # calculate_delta(ticker, as_of_str, expiration_str, 'put', force_delta_update=False)

    return all_call_data, all_put_data, call_options, put_options, strike_range_dict

import numpy as np
from scipy.interpolate import PchipInterpolator

import numpy as np
from scipy.interpolate import PchipInterpolator

async def interpolate_option_price(
    ticker: str,
    close_price_today: float,
    strike_price_to_interpolate: float,
    option_type: str,  # 'call' or 'put'
    expiration_date: str,  # Expected format: YYYY-MM-DD
    pricing_date: str,  # Expected format: YYYY-MM-DD
    stored_option_price: dict,
    premium_field: str,  # e.g., 'mid_price', 'close_price', 'trade_price'
    price_interpolate_flag: bool = True,
    max_strike_search_distance: float = 10.0,  # Max absolute distance from target strike to search
    interpolation_point_max_spread: float = 5.0,  # Max distance of found interpolation points
    client: object = None,  # Placeholder for client object
) -> float:
    """
    Interpolates or curve-fits the option price for a given strike if the direct price is not available.

    Improvements:
    1. Uses only data points with sip_timestamp within 60s of each other for both interpolation and extrapolation.
    2. Ensures monotonicity: for calls, price decreases with strike; for puts, price increases with strike.

    Args:
        ticker (str): Underlying asset ticker symbol.
        strike_price_to_interpolate (float): Strike price to interpolate.
        option_type (str): 'call' or 'put'.
        expiration_date (str): Option expiration date in 'YYYY-MM-DD' format.
        pricing_date (str): Pricing date in 'YYYY-MM-DD' format.
        stored_option_price (dict): Cache dictionary holding option prices.
        premium_field (str): Field in cached data for price (e.g., 'mid_price').
        price_interpolate_flag (bool): If False, skips interpolation.
        max_strike_search_distance (float): Max strike distance to search.
        interpolation_point_max_spread (float): Max spread of interpolation points.

    Returns:
        float: Interpolated/curve-fitted option premium, or 0.0 if estimation fails.
    """

    def is_valid_price_data(data_dict, field_name):
        if not isinstance(data_dict, dict) or data_dict.get(field_name) is None:
            return False
        price_value = data_dict.get(field_name, 0.0)
        if not isinstance(price_value, (int, float)) or price_value <= 0:
            return False
        if field_name == 'mid_price':
            return data_dict.get('ask_size', 0) > 0 and data_dict.get('bid_size', 0) > 0
        elif field_name == 'close_price':
            return data_dict.get('close_volume', 0) > 0
        elif field_name == 'trade_price':
            return data_dict.get('trade_size', 0) > 0
        return False

    def linear_extrapolate(x, x1, y1, x2, y2):
        """Simple linear extrapolation with bounds."""
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)

    # Direct lookup
    target_strike_rounded = round(strike_price_to_interpolate, 2)
    option_type_lower = option_type.lower()
    direct_data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(
        target_strike_rounded, {}).get(expiration_date, {}).get(option_type_lower, {})
    if is_valid_price_data(direct_data, premium_field):
        return direct_data.get(premium_field)

    if not price_interpolate_flag:
        print(f"\033[31mNo direct data for {ticker} {option_type_lower} K={target_strike_rounded} Exp={expiration_date} on {pricing_date}. Interpolation disabled.\033[0m")
        return 0.0

    # Fetch option chain data
    all_call_data, all_put_data, call_options, put_options, strike_range_dict = await pull_option_chain_data(
        ticker, option_type_lower, str(expiration_date), str(pricing_date), close_price_today, client=client)

    # Collect available data points
    available_price_array = []
    option_strikes = call_options if option_type_lower == 'call' else put_options
    option_premium = all_call_data if option_type_lower == 'call' else all_put_data

    if option_strikes is None or option_premium is None:
        print(f"\033[31mNo option chain data for {ticker} {option_type_lower} Exp={expiration_date} on {pricing_date}. Cannot interpolate K={target_strike_rounded}.\033[0m")
        return 0.0
    for opt, prem in zip(option_strikes, option_premium):
        if (prem is not None and prem.get(premium_field, 0.0) > 0.0 and
            abs(opt['strike_price'] - target_strike_rounded) / target_strike_rounded < 0.5):
            available_price_array.append({
                'strike': opt['strike_price'],
                'price': prem.get(premium_field),
                'timestamp': prem.get('sip_timestamp', 0) / 1e9  # Convert nanoseconds to seconds
            })

    if len(available_price_array) < 2:
        print(f"\033[31mInsufficient data points for {ticker} {option_type_lower} K={target_strike_rounded}.close: {close_price_today:.2f}, days to expire: {pricing_date}->{expiration_date}\033[0m")
        return 0.0

    # Sort by strike and extract data
    available_price_array.sort(key=lambda x: x['strike'])
    strikes = [opt['strike'] for opt in available_price_array]
    prices = [opt['price'] for opt in available_price_array]
    timestamps = [opt['timestamp'] for opt in available_price_array]
        # print("Available data points:")
        # for x in zip(strikes, prices, timestamps):
        #     print(x)

    # Calculate intrinsic value
    intrinsic_value = (max(close_price_today - target_strike_rounded, 0) if option_type_lower == 'call'
                       else max(target_strike_rounded - close_price_today, 0))

    # Temporal filtering
    selected_points = []
    time_window = 60
    while len(selected_points) < 4 and time_window < 300:
        if timestamps and max(timestamps) - min(timestamps) > time_window:
            median_timestamp = np.median(timestamps)
            selected_points = [opt for opt in available_price_array if abs(opt['timestamp'] - median_timestamp) <= time_window]
        # print("Insufficient points after temporal filter, expanding window")
        time_window += 30
    
    if len(selected_points) < 4:
        print("Insufficient points after temporal filter, use all available points")
        selected_points = available_price_array

    selected_strikes = [opt['strike'] for opt in selected_points]
    selected_prices = [opt['price'] for opt in selected_points]

    # Interpolation or extrapolation
    if min(selected_strikes) <= target_strike_rounded <= max(selected_strikes):
                # print(f"Interpolating for strike {target_strike_rounded} using selected points:")
                # for s, p in zip(selected_strikes, selected_prices):
                #     print(f"Strike: {s}, Price: {p}")
        interpolator = PchipInterpolator(selected_strikes, selected_prices)
        estimated_price = interpolator(target_strike_rounded)

        # Enforce monotonicity for puts
        if option_type_lower == 'put':
            # Find the closest strikes above and below in selected points
            lower_strike = max([s for s in selected_strikes if s < target_strike_rounded], default=None)
            higher_strike = min([s for s in selected_strikes if s > target_strike_rounded], default=None)

            if lower_strike is not None:
                lower_price = selected_prices[selected_strikes.index(lower_strike)]
                estimated_price = max(estimated_price, lower_price)  # Price >= lower strike price
            if higher_strike is not None:
                higher_price = selected_prices[selected_strikes.index(higher_strike)]
                estimated_price = min(estimated_price, higher_price)  # Price <= higher strike price
    else:
                # print(f"Extrapolating for strike {target_strike_rounded} using selected points:")
                # for s, p in zip(selected_strikes, selected_prices):
                #     print(f"Strike: {s}, Price: {p}")
        if len(selected_strikes) < 2:
            print(f"\033[31mInsufficient selected points for extrapolation for {ticker} {option_type_lower} K={target_strike_rounded}\033[0m")
            # breakpoint()
            return 0.0
        if target_strike_rounded < min(selected_strikes):
            x1, x2 = selected_strikes[0], selected_strikes[1]
            y1, y2 = selected_prices[0], selected_prices[1]
            estimated_price = linear_extrapolate(target_strike_rounded, x1, y1, x2, y2)
            # Bound extrapolation for puts
            if option_type_lower == 'put':
                estimated_price = min(estimated_price, y1)  # Price <= lowest strike price
        else:
            x1, x2 = selected_strikes[-2], selected_strikes[-1]
            y1, y2 = selected_prices[-2], selected_prices[-1]
            estimated_price = linear_extrapolate(target_strike_rounded, x1, y1, x2, y2)
            # Bound extrapolation for puts
            if option_type_lower == 'put':
                estimated_price = max(estimated_price, y2)  # Price >= highest strike price

    # Ensure price respects intrinsic value and non-negativity
    estimated_price = max(estimated_price, intrinsic_value, 0.0)
    print(f"\033[34mEstimated price for {ticker} {option_type_lower} K={target_strike_rounded}: {estimated_price:.2f}.close: {close_price_today:.2f}, days to expire: {pricing_date}->{expiration_date}\033[0m")
    return estimated_price

# ---------------------------------------------------------
# 5. PolygonAPIClient Class with Persistent Session
# ---------------------------------------------------------

class PolygonAPIClient:
    """
    A client to interact with Polygon.io's API for fetching option chains and option prices.
    Utilizes asynchronous requests with caching to optimize performance and reduce redundant API calls.
    """
    def __init__(
        self,
        api_key: str,
        max_concurrent_requests: int = 10,
        retries: int = 1,
        backoff_factor: float = 0.5
    ):
        """
        Initializes the PolygonAPIClient.

        :param api_key: Your Polygon.io API key.
        :param max_concurrent_requests: Maximum number of concurrent API requests.
        :param retries: Number of retry attempts for failed API calls.
        :param backoff_factor: Factor for exponential backoff between retries.
        """
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """
        Asynchronous context manager entry. Initializes the aiohttp session.
        """
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), connector=aiohttp.TCPConnector(ssl=self.ssl_context))
        logging.info("Initialized aiohttp ClientSession.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Asynchronous context manager exit. Closes the aiohttp session.
        """
        if self.session:
            await self.session.close()
            logging.info("Closed aiohttp ClientSession.")

    async def get_option_chain_async(
        self,
        ticker: str, 
        expiration_date: str, 
        call_put: str, 
        as_of_date: str,
        force_update: bool = False
    ) -> List[float]:
        """
        Asynchronously retrieve the sorted list of strikes from local dictionary if present,
        otherwise call the API, store the result, and re-save.
        
        Data structure:
        stored_option_chain[ticker][expiration_date][as_of_date][call_put] = [list of strikes...]
        """
        
        # Ensure we have loaded existing data for this ticker
        if ticker not in stored_option_chain:
            load_stored_option_data(ticker)
        
        exp_key = str(expiration_date)
        asof_key = str(as_of_date)
        callput_key = str(call_put).lower()  # Ensure consistency
        
        if OPTION_CHAIN_FORCE_UPDATE is False and force_update is False: 
            # Check in memory
            try:
                if (exp_key in stored_option_chain[ticker] and
                    asof_key in stored_option_chain[ticker][exp_key] and
                    callput_key in stored_option_chain[ticker][exp_key][asof_key]):
                    
                    strike_dict = stored_option_chain[ticker][exp_key][asof_key][callput_key]
                    
                    if len(strike_dict) > 0:
                        return strike_dict
                    elif len(strike_dict) == 0:
                        return strike_dict
            except Exception as e:
                logging.warning(f"Error checking memory for option chain: {e}")
        
        strike_dict = await self.query_polygon_for_option_chain_async(
            ticker, expiration_date, call_put, as_of_date
        )
        
        if len(strike_dict) >= 0:
            # Insert the fetched strike list into the nested dictionary
            stored_option_chain[ticker].setdefault(exp_key, {})
            stored_option_chain[ticker][exp_key].setdefault(asof_key, {})
            stored_option_chain[ticker][exp_key][asof_key][callput_key] = strike_dict
            
            # Save the updated cache to disk
            # save_stored_option_data(ticker)
        
        # Stop timing the entire function        
        return strike_dict

    async def query_polygon_for_option_chain_async(
        self, 
        ticker: str, 
        expiration_date: str, 
        call_put: str, 
        as_of: str
    ) -> Dict[float, str]:
        """
        Asynchronously returns a dictionary of strike prices to tickers from Polygon.io v3.
        If none found, returns an empty dict.
        """
        url = "https://api.polygon.io/v3/reference/options/contracts"

        print("Querying option chain from API: ", ticker, expiration_date, call_put, as_of)
        
        params_asc = {
            "underlying_ticker": ticker,
            "expiration_date": expiration_date,
            "as_of": as_of,
            "contract_type": call_put,
            "apiKey": self.api_key,
            "limit": 500,
            "order": "asc"
        }
        params_desc = {
            "underlying_ticker": ticker,
            "expiration_date": expiration_date,
            "as_of": as_of,
            "contract_type": call_put,
            "apiKey": self.api_key,
            "limit": 500,
            "order": "desc"
        }        
        try:
            async with self.semaphore:
                # Query with ascending order
                async with self.session.get(url, params=params_asc) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if "results" not in data:
                        logging.warning(f"No 'results' in asc response for {ticker}, {expiration_date}, {call_put}, {as_of}")
                        return {}
                    
                    strike_dict_asc = {item["strike_price"]: item["ticker"] for item in data["results"]}
                
                # Query with descending order
                async with self.session.get(url, params=params_desc) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if "results" not in data:
                        logging.warning(f"No 'results' in desc response for {ticker}, {expiration_date}, {call_put}, {as_of}")
                        return {}
                    
                    strike_dict_desc = {item["strike_price"]: item["ticker"] for item in data["results"]}

                # Merge dictionaries
                strike_dict = {**strike_dict_asc, **strike_dict_desc}
                return strike_dict
            
        except aiohttp.ClientResponseError as e:
            logging.error(f"RequestException fetching option chain for {ticker}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error fetching option chain for {ticker}: {e}")
            return {}

    async def get_option_chains_batch_async(
        self,
        ticker: str,
        unique_chain_requests: List[Tuple[str, str, str]],
        force_update: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Asynchronously fetches option chains for a batch of unique requests.

        :param ticker: The underlying asset ticker symbol.
        :param unique_chain_requests: A list of tuples containing 
                                    (expiration_date, as_of_date, call_put)
        :return: A nested dictionary of fetched option chain data
        """
        # First, deduplicate the requests
        # deduped_requests = list(set(unique_chain_requests))
        
        # Create tasks for each unique request
        tasks = []

        for expiration_str, as_of_str, call_put in unique_chain_requests:
            task = asyncio.create_task(
                self.get_option_chain_async(
                    ticker=ticker,
                    expiration_date=expiration_str,
                    call_put=call_put,
                    as_of_date=as_of_str,
                    force_update=force_update
                )
            )
            tasks.append(task)
        # Gather all requests concurrently
        fetched_data = await asyncio.gather(*tasks, return_exceptions=True)
        # Organize results into the chain_data structure
        chain_data = {ticker: {}}
        for (expiration_str, as_of_str, call_put), result in zip(unique_chain_requests, fetched_data):
            if isinstance(result, Exception):
                logging.error(f"Error fetching option chain: {result}")
                # Store empty result for this specific request
                result = {}
            
            # Store the result in the nested dictionary
            chain_data[ticker].setdefault(expiration_str, {}) \
                            .setdefault(as_of_str, {})[call_put] = result
        return chain_data

    async def query_polygon_for_option_price_async(
        self,
        ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        option_ticker: str
    ) -> Dict[str, Any]:
        """
        Asynchronously fetches and stores the mid-price or close price for a specific option.

        :param ticker: The underlying asset ticker symbol.
        :param strike_price: The strike price of the option.
        :param call_put: 'call' or 'put'.
        :param expiration_date: The expiration date in 'YYYY-MM-DD' format.
        :param pricing_date: The date for which the price is being fetched in 'YYYY-MM-DD' format.
        :return: A dictionary containing the fetched data or an empty dict if unavailable.
        """
        for attempt in range(1, self.retries + 1):
            try:
                # Determine which endpoint to use based on `use_close_price`
                # For this function, we'll assume `use_close_price=True`
                fetched_data = await self._fetch_and_store_option_data(
                    ticker, option_ticker, strike_price, call_put, expiration_date, pricing_date
                )
                if fetched_data:
                    return fetched_data
                else:
                    raise ValueError("No valid data received.")
            except Exception as e:
                if attempt == self.retries:
                    logging.error(f"Max retries exceeded for {ticker}, Strike: {strike_price}, Type: {call_put}. Error: {e}")
                    return {}
                wait_time = self.backoff_factor * (2 ** (attempt - 1))
                logging.warning(f"Attempt {attempt} failed for {ticker}, Strike: {strike_price}, Type: {call_put}. "
                                f"Retrying in {wait_time} seconds. Error: {e}")
                await asyncio.sleep(wait_time)

    async def _fetch_and_store_option_data(
        self,
        ticker: str,
        option_ticker: str,
        strike_price: float,
        call_put: str,
        expiration_date: str,
        pricing_date: str,
        premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]  # Determines which data to fetch and store
    ) -> Dict[str, Any]:
        """
        Helper method to fetch and store option data from Polygon.io.

        :param ticker: The underlying asset ticker symbol.
        :param option_ticker: The full option ticker symbol.
        :param strike_price: The strike price of the option.
        :param call_put: 'call' or 'put'.
        :param expiration_date: The expiration date in 'YYYY-MM-DD' format.
        :param pricing_date: The date for which the price is fetched in 'YYYY-MM-DD' format.
        :param use_close_price: If True, fetches close price and volume from the open-close endpoint.
                                If False, fetches ask, bid, and mid prices from the quotes endpoint.
        :return: A dictionary containing the fetched data or an empty dict if unavailable.
        """
        # Define key path components
        pricing_key = pricing_date  # 'YYYY-MM-DD'
        strike_key = round(strike_price, 2)  # e.g., '150.00'
        expiry_key = expiration_date  # 'YYYY-MM-DD'
        cp_key = call_put.lower()  # 'call' or 'put'
        global new_data_entry_count

        # Format the option symbol
        option_symbol = option_ticker

        # Set up the API endpoint and parameters
        if premium_field == "trade_price":
            url = f"https://api.polygon.io/v3/trades/{option_symbol}"
            params = {
                "timestamp": pricing_date,
                "order": "desc",
                "sort": "timestamp",
                "limit": 10,
                "apiKey": self.api_key,
            }

        elif premium_field == "mid_price":
            url = f"https://api.polygon.io/v3/quotes/{option_symbol}"
            params = {
                "timestamp": pricing_date,
                "order": "desc",
                "sort": "timestamp",
                "limit": 10,
                "apiKey": self.api_key,
            }
        else:
            url = f"https://api.polygon.io/v1/open-close/{option_symbol}/{pricing_date}"
            params = {
                "apiKey": self.api_key,
            }
        
        def update_memory_invalid_data(premium_field=PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]):
            """Update the stored option data with default values for invalid fetches."""
            if premium_field == 'trade_price':
                invalid_data = {
                    "trade_size": 0,
                    "trade_price": 0.00,
                    "sip_timestamp": 0
                }
            elif premium_field == 'close_price':
                invalid_data = {
                    "close_price": 0,
                    "close_volume": 0
                }
            else:
                invalid_data = {
                    "ask_price": 0,
                    "bid_price": 0,
                    "ask_size": 0,
                    "bid_size": 0,
                    "mid_price": 0
                }
            # Get or create the option data dictionary
            option_data = stored_option_price.setdefault(ticker.upper(), {}) \
                                            .setdefault(pricing_key, {}) \
                                            .setdefault(strike_key, {}) \
                                            .setdefault(expiry_key, {}) \
                                            .setdefault(cp_key, {})
            # Update with invalid data, preserving existing fields
            option_data.update(invalid_data)

        try:
            async with self.semaphore:
                async with self.session.get(url, params=params) as response:
                    # print(f"url: {url}, params: {params}")
                    if response.status != 200:
                        full_url = f"{url}?{urlencode(params)}"
                        update_memory_invalid_data(premium_field)
                        return {}

                    data = await response.json()
                    # print(f"Response data: {data}")
                        # Process quotes endpoint data
                    if premium_field == 'trade_price':
                        if 'results' in data and data['results']:
                            trade = data['results'][0]
                            trade_size = trade.get('size', 0)
                            trade_price = trade.get('price', 0.00)
                            sip_timestamp = trade.get('sip_timestamp', 0)
                            fetched_data = {
                                "trade_size": trade_size,
                                "trade_price": trade_price,
                                "sip_timestamp": sip_timestamp,
                            }
                            # Get or create the option data dictionary
                            option_data = stored_option_price.setdefault(ticker.upper(), {}) \
                                                            .setdefault(pricing_key, {}) \
                                                            .setdefault(strike_key, {}) \
                                                            .setdefault(expiry_key, {}) \
                                                            .setdefault(cp_key, {})
                            # Update with fetched data
                            option_data.update(fetched_data)
                            print(f"Stored {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, Pricing:{pricing_date}:{option_data}")
                            new_data_entry_count += 1
                            return fetched_data
                        else:
                            update_memory_invalid_data(premium_field)
                            return {}
                    elif premium_field == 'mid_price':
                        if 'results' in data and data['results']:
                            ask_price = 0
                            bid_price = 0
                            ask_size = 0
                            bid_size = 0
                            index = 0
                            while (ask_price == 0 or bid_price == 0) and index < len(data['results']):
                                quote = data['results'][index]
                                ask_price = quote.get('ask_price', 0.00)
                                bid_price = quote.get('bid_price', 0.00)
                                ask_size = quote.get('ask_size', 0)
                                bid_size = quote.get('bid_size', 0)
                                index += 1
                            if ask_price > 0 and bid_price > 0:
                                mid_price = round((ask_price + bid_price) / 2.0, 3)
                                fetched_data = {
                                    "ask_price": ask_price,
                                    "bid_price": bid_price,
                                    "ask_size": ask_size,
                                    "bid_size": bid_size,
                                    "mid_price": mid_price
                                }
                                # Get or create the option data dictionary
                                option_data = stored_option_price.setdefault(ticker.upper(), {}) \
                                                                .setdefault(pricing_key, {}) \
                                                                .setdefault(strike_key, {}) \
                                                                .setdefault(expiry_key, {}) \
                                                                .setdefault(cp_key, {})
                                # Update with fetched data
                                option_data.update(fetched_data)
                                print(f"Stored {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, Pricing:{pricing_date}:{option_data}")
                                new_data_entry_count += 1
                                return fetched_data
                            else:
                                update_memory_invalid_data(premium_field)
                                return {}
                        else:
                            update_memory_invalid_data(premium_field)
                            return {}
                    else:
                        # Process open-close endpoint data
                        if 'close' in data and data['close'] is not None:
                            close_price = round(float(data['close']), 3)
                            close_volume = data.get('volume', 0)
                            if close_price >= 0:
                                fetched_data = {
                                    "close_price": close_price,
                                    "close_volume": close_volume
                                }
                                # Get or create the option data dictionary
                                option_data = stored_option_price.setdefault(ticker.upper(), {}) \
                                                                .setdefault(pricing_key, {}) \
                                                                .setdefault(strike_key, {}) \
                                                                .setdefault(expiry_key, {}) \
                                                                .setdefault(cp_key, {})
                                # Update with fetched data
                                option_data.update(fetched_data)
                                print(f"Stored {ticker},Strike:{strike_price},{call_put},Expire:{expiration_date}, Pricing:{pricing_date}:{option_data}")
                                new_data_entry_count += 1
                                return fetched_data
                            else:
                                update_memory_invalid_data(premium_field)
                                return {}
                        else:
                            update_memory_invalid_data(premium_field)
                            return {}
        except aiohttp.ClientError as e:
            logging.error(f"ClientError while fetching {'close price' if premium_field else 'option data'} for {option_symbol}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error while fetching {'close price' if premium_field else 'option data'} for {option_symbol}: {e}")
            return {}
            
    async def get_option_prices_batch_async(
        self,
        ticker: str,
        options_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously fetches option data for a batch of options, after removing duplicates.

        :param ticker: The underlying asset ticker symbol.
        :param options_list: A list of option dictionaries containing 'strike_price', 'call_put', 'expiration_date', 'quote_timestamp'.
        :return: A list of dictionaries containing fetched data for each option.
        """
        # Step 1: Deduplicate options_list
        seen = set()
        deduped_options = []
        for option in options_list:
            key = (
                option['strike_price'],
                option['call_put'],
                option['expiration_date'],
                option['quote_timestamp'],
                option.get('option_ticker')  # safely handle missing 'option_ticker'
            )
            if key not in seen:
                seen.add(key)
                deduped_options.append(option)

        # Step 2: Create async tasks
        tasks = []
        if len(deduped_options) > 0:
            print(f"Starting to batch fetch {len(deduped_options)} unique options for {ticker}")
        for option in deduped_options:
            task = asyncio.create_task(
                self.query_polygon_for_option_price_async(
                    ticker=ticker,
                    strike_price=option['strike_price'],
                    call_put=option['call_put'],
                    expiration_date=option['expiration_date'],
                    pricing_date=option['quote_timestamp'],
                    option_ticker=option.get('option_ticker')
                )
            )
            tasks.append(task)

        # Step 3: Gather all requests concurrently
        fetched_data = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: Handle exceptions if any
        results = []
        for i, result in enumerate(fetched_data):
            if isinstance(result, Exception):
                logging.error(f"Error fetching option data: {result}")
                results.append({})
            else:
                results.append(result)
        print(f"Gathered {len(results)} results for {ticker}")
        return results


    def get_option_prices_batch_sync(
        self,
        ticker: str,
        options_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synchronously fetches option data for a batch of options.
        Note: This method uses asyncio's event loop to run asynchronous tasks synchronously.

        :param ticker: The underlying asset ticker symbol.
        :param options_list: A list of option dictionaries containing 'strike_price', 'call_put', 'expiration_date', 'quote_timestamp'.
        :return: A list of dictionaries containing fetched data for each option.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_option_prices_batch_async(ticker, options_list))

# ---------------------------------------------------------
# 6. Synchronous Option Price Fetching
# ---------------------------------------------------------

def get_option_quote(underlying_ticker, strike_price, call_put, expiration_date, quote_timestamp):
    """
    Poll the option quote from Polygon.io (v3) and return ask_price, bid_price, etc.
    """
    expiration_date_formatted = expiration_date[2:].replace("-", "")  
    strike_price_formatted = f"{int(strike_price * 1000):08d}"  
    if call_put == 'call':
        call_put_converted = 'C'
    elif call_put == 'put':
        call_put_converted = 'P'
    else:
        raise ValueError("call_put must be 'call' or 'put'")
    option_symbol = f"O:{underlying_ticker}{expiration_date_formatted}{call_put_converted}{strike_price_formatted}"

    url = f"https://api.polygon.io/v3/quotes/{option_symbol}"
    params = {
        "timestamp": quote_timestamp,
        "order": "desc",
        "sort": "timestamp",
        "limit": 1,  # Only the latest quote
        "apiKey": polygonio_config.API_KEY,  # load from config
    }

    try:
        response = requests.get(url, params=params, timeout=10, verify=certifi.where())
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            extracted_data = []
            for result in data["results"]:
                extracted_data.append({
                    "ask_price": result.get("ask_price"),
                    "bid_price": result.get("bid_price"),
                    "ask_size": result.get("ask_size"),
                    "bid_size": result.get("bid_size"),
                    "mid_price": round((result.get("ask_price", 0.0) + result.get("bid_price", 0.0)) / 2.0, 3)
                })
            return extracted_data
        else:
            logging.warning(f"No valid data found in the response for {option_symbol}: {data}")
            return {"error": "No valid data found in the response."}
    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException fetching {option_symbol}: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error fetching {option_symbol}: {e}")
        return {"error": str(e)}

def query_polygon_for_option_price(ticker, strike_price, call_put, expiration_date, pricing_date):
    """
    Returns a dictionary with relevant price data or empty dict if invalid or none.
    """
    logging.debug(f"Entering query_polygon_for_option_price_sync for {ticker}, "
                  f"Strike: {strike_price}, Type: {call_put}, "
                  f"Expiration: {expiration_date}, Pricing Date: {pricing_date}")
    start_time = time.time()
    data = get_option_quote(ticker, strike_price, call_put, expiration_date, pricing_date)
    elapsed_time = time.time() - start_time
    logging.debug(f"API call in query_polygon_for_option_price_sync for {ticker}. "
                  f"Time taken: {elapsed_time:.4f} seconds.")

    if isinstance(data, dict) and "error" in data:
        logging.error(f"Error in query_polygon_for_option_price_sync for {ticker}: {data['error']}")
        return {}
    if not data:
        logging.warning(f"No data in query_polygon_for_option_price_sync for {ticker}. Returning empty dict.")
        return {}

    quote = data[0]
    ask_price = quote.get("ask_price", 0.0)
    bid_price = quote.get("bid_price", 0.0)

    if ask_price <= 0 or bid_price <= 0:
        logging.warning(f"Invalid prices returned for {ticker}: ask={ask_price}, bid={bid_price}. Returning empty dict.")
        return {}

    # mid_price is already calculated in get_option_quote
    mid_price = quote.get("mid_price", 0.0)

    # Insert into nested dictionary
    pricing_key = str(pricing_date)
    strike_key  = round(strike_price, 2)
    expiry_key  = str(expiration_date)
    cp_key      = call_put.lower()

    fetched_data = {
        "ask_price": ask_price,
        "bid_price": bid_price,
        "ask_size": quote.get("ask_size", 0),
        "bid_size": quote.get("bid_size", 0),
        "mid_price": mid_price
    }

    stored_option_price.setdefault(ticker.upper(), {}).setdefault(pricing_key, {}) \
                       .setdefault(strike_key, {}).setdefault(expiry_key, {})[cp_key] = fetched_data

    logging.debug(f"Exiting query_polygon_for_option_price_sync for {ticker}. Fetched data: {fetched_data}")

    return fetched_data

# ---------------------------------------------------------
# 7. BATCH FETCHING FUNCTIONS
# ---------------------------------------------------------

def get_option_prices_batch_sync(ticker, options_list):
    """
    Synchronously fetch multiple option prices from the API, 
    one request at a time, using the existing `query_polygon_for_option_price`.
    """
    mid_prices = []
    for option in options_list:
        fetched_data = query_polygon_for_option_price(
            ticker=ticker,
            strike_price=option['strike_price'],
            call_put=option['call_put'],
            expiration_date=option['expiration_date'],
            pricing_date=option['quote_timestamp']
        )
        mid_price = fetched_data.get("mid_price", 0.0)
        mid_prices.append(mid_price)
    return mid_prices

def compute_metrics_from_daily_results(daily_results):
    dates = [row['date'] for row in daily_results]
    cumulative_pnl_realized = [row['cumulative_pnl_realized'] for row in daily_results]
    
    # Compute daily PnL
    daily_pnl = np.diff(cumulative_pnl_realized, prepend=cumulative_pnl_realized[0])
    
    # Compute Sharpe ratio
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe_ratio = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Compute maximum drawdown and period
    cumulative_pnl = [row['cumulative_pnl'] for row in daily_results]
    max_drawdown = 0.0
    max_drawdown_period = 0
    peak = cumulative_pnl[0]
    peak_date = dates[0]
    for i, pnl in enumerate(cumulative_pnl):
        if pnl > peak:
            peak = pnl
            peak_date = dates[i]
        drawdown = peak - pnl
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_period = (dates[i] - peak_date).days
    
    # Extract positions information
    positions_seen = set()
    num_positions_opened = 0
    holding_days_list = []
    for i in range(1, len(daily_results)):
        prev_positions = {(pos['position_open_date'], pos['expiration']) for pos in daily_results[i-1]['active_positions']}
        current_positions = {(pos['position_open_date'], pos['expiration']) for pos in daily_results[i]['active_positions']}
        new_positions = current_positions - prev_positions
        for pos_key in new_positions:
            pos = next(p for p in daily_results[i]['active_positions'] if (p['position_open_date'], p['expiration']) == pos_key)
            open_date = pos['position_open_date']
            expiration = pos['expiration']
            holding_days = (expiration - open_date).days
            holding_days_list.append(holding_days)
            num_positions_opened += 1
            positions_seen.add(pos_key)
    
    avg_holding_days = np.mean(holding_days_list) if holding_days_list else 0.0
    
    # Average required margin
    required_margins = [row['required_margin'] for row in daily_results if row['required_margin'] > 0]
    avg_required_margin = np.mean(required_margins) if required_margins else 0.0
    
    return {
        'sharpe_ratio': round(sharpe_ratio,2),
        'max_drawdown': round(max_drawdown,0),
        'max_drawdown_period': max_drawdown_period,
        'num_positions_opened': num_positions_opened,
        'avg_holding_days': round(avg_holding_days,0),
        'avg_required_margin': round(avg_required_margin,0),
    }

# ---------------------------------------------------------
# 9. MAIN BACKTEST FUNCTION
# ---------------------------------------------------------

async def backtest_options_sync_or_async(
    start_date: str,
    end_date: str,
    ticker: str,
    df_dict: Dict[str, pd.DataFrame],
    trade_parameter: Dict[str, Any],  # Single dictionary containing all trade parameters
    client: PolygonAPIClient,
    use_async: bool = True,
    carry_over_weekly_results: Optional[List[Dict[str, Any]]] = None,
    mode: str = "training",
    trade_type: str = "iron_condor",
) -> Tuple[Optional[float], Optional[List[Dict[str, Any]]], Optional[float], Optional[List[datetime]]]:
    global new_data_entry_count
    # Define premium_field
    premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]
    delta_field = DELTA_FIELD_MAP[PREMIUM_PRICE_MODE]

    roll_method = None
    expiring_wks = trade_parameter['expiring_wks']
    target_premium_otm = trade_parameter['target_premium_otm']
    target_steer = trade_parameter['target_steer']
    day_of_week = trade_parameter['day_of_week']
    stop_profit_percent = trade_parameter['stop_profit_percent']
    stop_loss_action = trade_parameter['stop_loss_action']
    iron_condor_width = trade_parameter['iron_condor_width']
    target_delta = trade_parameter['target_delta']
    vix_correlation = trade_parameter['vix_correlation']
    vix_threshold = trade_parameter['vix_threshold']
    trade_type = trade_parameter['trade_type']

    active_positions: List[Dict[str, Any]] = []
    total_pnl = 0.0
    sharpe_ratio = 0.0
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")

    # Carry over open positions from previous run, if any (with expiration beyond start_date)
    if carry_over_weekly_results:
        prev_positions = carry_over_weekly_results[-1].get('active_positions', [])
        for pos in prev_positions:
            if pos.get('expiration') and isinstance(pos['expiration'], datetime):
                if pos['expiration'] > start_dt and pos['invalid_data'] is None:
                    active_positions.append(pos)
                    # print(f"Carried over position expiring {pos['expiration'].date()} (call {pos.get('call_strike_sold')}, put {pos.get('put_strike_sold')})")

    # Get all earnings dates in [start_date, end_date]
    earnings_dates = get_earnings_dates(ticker, start_date, end_date) or set()  

    # -------------- Historical Data --------------
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    end_dt_extended = end_dt + pd.Timedelta(weeks=expiring_wks)

    # Ensure the 'date' column is in datetime format
    df = df_dict['df']
    vix_df = df_dict['vix_df']

    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Apply the mask to filter the DataFrame
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt_extended)
    hist_df = df[mask].reset_index(drop=True)

    # Set the date as index and reindex to fill any missing dates with forward fill
    vix_close_series = vix_df.set_index('date')['close'].reindex(pd.date_range(start_date, end_dt_extended)).ffill()
    # -------------- Fridays --------------
    fridays_list = get_all_weekdays('Friday',start_dt, end_dt, expiring_wk=expiring_wks,trading_dates_df=hist_df)
    wednesday_list = get_all_weekdays('Wednesday',start_dt, end_dt, expiring_wk=expiring_wks,trading_dates_df=hist_df)

    if ticker == 'VIX':
        expire_day_list = wednesday_list
    else:
        expire_day_list = fridays_list
    total_pnl = 0.0
    weekly_results = []
    daily_dates = []
    cumulative_pnls  = []
    arrow_data = []  # For negative weeks

    # ------------------------------------------------
    # 1) Pre-collect all data we need for each Friday
    # ------------------------------------------------
    needed_data = []  # list of dict, each dict holds info for that Friday's trade

    async def select_new_position_strikes(trade_date: datetime, expiration_date: datetime, open_atm_call=0, open_atm_put=0):
        """Determine the option strikes for the new iron condor position."""   
        short_call_prem_open, short_put_prem_open, long_call_prem_open, long_put_prem_open = 0, 0, 0, 0
        call_strike_sold, put_strike_sold, call_strike_bought, put_strike_bought, open_distance_call, open_distance_put = 0,0,0,0,0,0
        invalid_data_flag = None
        sc_data, sp_data, lc_data, lp_data = None, None, None, None      
        expiration_str = expiration_date.strftime("%Y-%m-%d")
        as_of_str = trade_date.strftime("%Y-%m-%d")
        # Underlying price on trade date
        close_price_pricing = float(hist_df.loc[hist_df['date'] == trade_date, 'close'].iloc[0])
        
        if trade_type == 'iron_condor':
            call_put = ['call', 'put']
        if trade_type in ['call_credit_spread','covered_call']:
            call_put = ['call']
        if trade_type == 'put_credit_spread':
            call_put = ['put']

        all_call_data, all_put_data, call_options ,put_options, strike_range_dict = await pull_option_chain_data(ticker, call_put, expiration_str, as_of_str,close_price_pricing,client=client)

        counter = 0
        this_expiration_date = expiration_date
        option_range_scale = 1

        # Get NYSE calendar
        while all_call_data is None or all_put_data is None:
            if this_expiration_date > trade_date:
                break
            counter+=1
            previous_expiration_date = this_expiration_date
            this_expiration_date -= timedelta(days=1) # Move by one day if it`s missed because of non-trading day
            if counter > 1 and this_expiration_date.weekday() != 4 and len(nyse.valid_days(previous_expiration_date, previous_expiration_date))>0: #if already attempted to shift by one day, then only look for Friday expiration
                continue
            expiration_str = this_expiration_date.strftime("%Y-%m-%d")
            all_call_data, all_put_data, call_options ,put_options, strike_range_dict = await pull_option_chain_data(ticker, call_put, expiration_str, as_of_str,close_price_pricing, client=client)

        if strike_range_dict is not None:
            if 'call' in call_put:
                atm_strike_call = find_closest_strike(call_options,all_call_data,close_price_pricing)
                atm_strike_call_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(atm_strike_call,2), {}).get(expiration_str, {}).get('call', {}) if atm_strike_call is not None else None
            else:
                atm_strike_call = None
                atm_strike_call_data = None
            if 'put' in call_put:
                atm_strike_put = find_closest_strike(put_options,all_put_data,close_price_pricing)
                atm_strike_put_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(atm_strike_put,2), {}).get(expiration_str, {}).get('put', {}) if atm_strike_put is not None else None
            else:
                atm_strike_put = None
                atm_strike_put_data = None

            iv_calculated_put = calculate_implied_volatility(close_price=close_price_pricing, strike_price=atm_strike_put, option_price=atm_strike_put_data.get(premium_field,0), days_to_expire=(expiration_date - trade_date).days, risk_free_rate=0, dividend_yield=0.0, option_type='put') if atm_strike_put is not None else None
            iv_calculated_call = calculate_implied_volatility(close_price=close_price_pricing, strike_price=atm_strike_call, option_price=atm_strike_call_data.get(premium_field,0), days_to_expire=(expiration_date - trade_date).days, risk_free_rate=0, dividend_yield=0.0, option_type='call') if atm_strike_call is not None else None
            if iv_calculated_put is not None:
                if iv_calculated_put < IV_THRESHOLD_MIN and open_atm_put == 0:
                    print(f"iv_calculated_put: {iv_calculated_put}, close_price_pricing: {close_price_pricing}, strike_price: {atm_strike_put}, option_price: {atm_strike_put_data.get(premium_field,0)}, days_to_expire: {(expiration_date - trade_date).days}")
                    return None
            if iv_calculated_call is not None:
                if iv_calculated_call < IV_THRESHOLD_MIN and open_atm_call == 0:
                    print(f"iv_calculated_call: {iv_calculated_call}, close_price_pricing: {close_price_pricing}, strike_price: {atm_strike_call}, option_price: {atm_strike_call_data.get(premium_field,0)}, days_to_expire: {(expiration_date - trade_date).days}")
                    return None
        else:
            iv_calculated_call, iv_calculated_put = None, None 
        # if iv_calculated_put is None:
        #     breakpoint()
                    
        range_ok = strike_range_dict is not None and strike_range_dict['call']['max_strike'] > close_price_pricing * (1 + OPTION_RANGE * option_range_scale  ) and strike_range_dict['put']['min_strike'] < close_price_pricing * ( 1 - OPTION_RANGE * option_range_scale )
        # if all_call_data is not None and all_put_data is not None and range_ok is False:
        #     print(f"strikes found but too few options for {ticker} on {trade_date} expiration {expiration_str} original expiration {expiration_date}")
        #     print(f"call max strike {strike_range_dict['call']['max_strike']} close price {close_price_pricing} range {OPTION_RANGE}")
        #     print(f"put min strike {strike_range_dict['put']['min_strike']} close price {close_price_pricing} range {OPTION_RANGE}")

        while all_call_data is None or all_put_data is None or range_ok is False:
            if this_expiration_date <= trade_date:
                break
            counter+=1
            previous_expiration_date = this_expiration_date
            this_expiration_date -= timedelta(days=1) # Move by one day if it`s missed because of non-trading day
            if counter > 1 and this_expiration_date.weekday() != 4 and len(nyse.valid_days(previous_expiration_date, previous_expiration_date))>0: #if already attempted to shift by one day, then only look for Friday expiration
                continue
            expiration_str = this_expiration_date.strftime("%Y-%m-%d")
            all_call_data, all_put_data, call_options ,put_options, strike_range_dict = await pull_option_chain_data(ticker, call_put, expiration_str, as_of_str,close_price_pricing, client=client)

            if strike_range_dict is not None:
                if strike_range_dict['call']['max_strike'] > close_price_pricing * (1 + OPTION_RANGE * option_range_scale  ) and strike_range_dict['put']['min_strike'] < close_price_pricing * ( 1 - OPTION_RANGE * option_range_scale ):
                    range_ok = True
                # if range_ok is False:
                #     print(f"strikes found but too few options for {ticker} on {trade_date} expiration {expiration_str} original expiration {expiration_date}")
                #     print(f"call max strike {strike_range_dict['call']['max_strike']} close price {close_price_pricing} range {OPTION_RANGE}")
                #     print(f"put min strike {strike_range_dict['put']['min_strike']} close price {close_price_pricing} range {OPTION_RANGE}")
            if counter > 30:
                break
        if all_call_data is not None and all_put_data is not None:
            print(f"Option chain data found for {ticker} on {trade_date} expiration {expiration_str} original expiration {expiration_date}")
            expiration_date = this_expiration_date # update the expiration date now

        if all_call_data is None or all_put_data is None:
            print(f"\033[93mNo option chain data found for {ticker} on {trade_date} expiration {expiration_str}\033[0m")
            invalid_data_flag = "no option chain data"
            # return None
        
        days_to_expire = (expiration_date - trade_date).days

        # if days_to_expire < 20 or expiration_date.weekday() < 3:
        #     breakpoint()

        if target_premium_otm is not None:
            target_premium_call_baseline = target_premium_otm * (days_to_expire)**0.5 * (1-target_steer)
            target_premium_put_baseline  = target_premium_otm * (days_to_expire)**0.5 * (1+target_steer)
        else:
            target_premium_call_baseline = None
            target_premium_put_baseline  = None

        if target_delta is not None:
            target_delta_call_baseline = -target_delta * (days_to_expire)**0.5 * (1-target_steer)
            target_delta_put_baseline  = target_delta * (days_to_expire)**0.5 * (1+target_steer)
        else:
            target_delta_call_baseline = None
            target_delta_put_baseline  = None

        # Adjust target premiums based on VIX/20
        VIX_THRESHOLD_UP = vix_threshold
        VIX_THRESHOLD_DOWN = vix_threshold

        adjust = vix_correlation
        # if vix_value > vix_threshold * 1.5:
        #     adjust = vix_correlation * 4

        # target_premium_call = max(0.05,target_premium_call_baseline * ( 1 - max((vix_value - vix_threshold),0) * adjust) if target_premium_call_baseline is not None else target_premium_call_baseline) #dial down the put premium when VIX is high (possible rebound)
        target_premium_call = max(target_premium_call_baseline/1.5,target_premium_call_baseline * ( 1 - (vix_value - vix_threshold) * adjust) if target_premium_call_baseline is not None else target_premium_call_baseline) if trade_type != 'put_credit_spread' else 0 #dial down the put premium when VIX is high (possible rebound)
        target_premium_put = max(target_premium_put_baseline/1.5,target_premium_put_baseline * ( 1 + (vix_value - vix_threshold) * adjust ) if target_premium_put_baseline is not None else target_premium_put_baseline) if trade_type not in ['call_credit_spread','covered_call'] else 0 #dial up the put premium when VIX is high (possible rebound)
        target_delta_call = target_delta_call_baseline * ( 1 + (vix_value - vix_threshold) * 0 ) if target_delta_call_baseline is not None else target_delta_call_baseline
        target_delta_put = target_delta_put_baseline * ( 1 + (vix_value - vix_threshold) * adjust ) if target_delta_put_baseline is not None else target_delta_put_baseline

        # Choose strikes based on target premium or delta criteria
        strike_target_call = {'premium_target': target_premium_call, 'delta_target': target_delta_call}
        strike_target_put  = {'premium_target': target_premium_put,  'delta_target': target_delta_put}

        # print(f"target_premium_call: {target_premium_call}, target_premium_put: {target_premium_put}, target_delta_call: {target_delta_call}, target_delta_put: {target_delta_put}")
        if invalid_data_flag is None:
            if target_premium_call != 0: 
                if open_atm_call == 0:
                    if trade_type == 'covered_call':
                        # call_strike_sold = find_closest_strike(call_options, all_call_data, close_price_pricing * 1.1)
                        call_strike_sold, price_call_strike = find_strike_custom(call_options, all_call_data, strike_target_call, stock_price=close_price_pricing)
                    else:
                        call_strike_sold, price_call_strike = find_strike_custom(call_options, all_call_data, strike_target_call, stock_price=close_price_pricing)
                else:
                    # call_strike_sold = find_closest_strike(call_options, all_call_data, close_price_pricing - min( 2 * open_atm_call, 2 * iron_condor_width )) # sell deep in the money call so that the long leg is at the money
                    if trade_type == 'covered_call':
                        call_strike_sold = find_closest_strike(call_options, all_call_data, close_price_pricing - open_atm_call + 5)
                    else:
                        call_strike_sold = find_closest_strike(call_options, all_call_data, close_price_pricing - 1.5 * iron_condor_width ) # sell deep in the money call so that the long leg is at the money
                if call_strike_sold is None:
                    print(f"No suitable strikes found on {trade_date.date()} (call_strike={call_strike_sold}, put_strike={put_strike_sold}), strike_target_call={strike_target_call}, strike_target_put={strike_target_put})")
                    invalid_data_flag = "no short strike"
                if open_atm_call == 0:
                    if trade_type == 'covered_call':
                        call_strike_bought = call_strike_sold
                    else:
                        call_strike_bought = find_closest_strike(call_options, all_call_data, call_strike_sold + iron_condor_width) if call_strike_sold is not None else None
                else:
                    if trade_type == 'covered_call':
                        call_strike_bought = call_strike_sold
                    else:
                        call_strike_bought = find_closest_strike(call_options, all_call_data, close_price_pricing) if call_strike_sold is not None else None
                if call_strike_bought == call_strike_sold and trade_type != 'covered_call':
                    # call_strike_bought = call_strike_sold + iron_condor_width
                    if call_strike_bought is not None:
                        option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int((call_strike_sold + iron_condor_width) * 1000):08d}"
                        print(f"\033[93mcall_strike_bought is same as call_strike_sold for {ticker} on {current_day.date()}->{expiration_date}, call_strike_sold={call_strike_sold}, close price: {close_price_pricing:.2f}, try ticker: {option_ticker}\033[0m")
                                # result = await client._fetch_and_store_option_data(ticker, option_ticker, call_strike_sold, 'call', expiration_str, as_of_str, use_close_price=True)
                                # print(f"call strike {call_strike_sold} checked again, fetched data: {result}")
                                # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int(call_strike_bought * 1000):08d}"
                                # result = await client._fetch_and_store_option_data(ticker, option_ticker, call_strike_bought, 'call', expiration_str, as_of_str, use_close_price=True)
                                # print(f"call strike {call_strike_bought} checked again, fetched data: {result}")
                    # breakpoint()
                    invalid_data_flag = 'long strike out or range'
                    if SKIP_MISSING_STRIKE_TRADE:
                        return None
            if target_premium_put != 0:
                if open_atm_put == 0:
                    put_strike_sold, price_put_strike = find_strike_custom(put_options, all_put_data, strike_target_put, stock_price=close_price_pricing)
                else:
                    
                    put_strike_sold = find_closest_strike(put_options, all_put_data, close_price_pricing + 1.5 * iron_condor_width ) # sell deep in the money put so that the long leg is at the money
                if put_strike_sold is None:
                    print(f"No suitable strikes found on {trade_date.date()} (call_strike={call_strike_sold}, put_strike={put_strike_sold}), strike_target_call={strike_target_call}, strike_target_put={strike_target_put})")
                    invalid_data_flag = "no short strike"
                if open_atm_put == 0:
                    put_strike_bought  = find_closest_strike(put_options, all_put_data,  put_strike_sold - iron_condor_width)  if put_strike_sold is not None else None
                else:
                    put_strike_bought  = find_closest_strike(put_options, all_put_data,  close_price_pricing)  if put_strike_sold is not None else None
                    # if put_strike_sold - put_strike_bought < 1 * iron_condor_width:
                    #     print(f"put_strike_sold - put_strike_bought < 1.5 * iron_condor_width for {ticker} on {current_day.date()}->{expiration_date}, put_strike_sold={put_strike_sold}, close price: {close_price_pricing:.2f}")
                        # breakpoint()
                if put_strike_bought == put_strike_sold:
                    if put_strike_bought is not None:
                        option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int((put_strike_sold - iron_condor_width) * 1000):08d}"
                        print(f"\033[93mput_strike_bought is same as put_strike_sold for {ticker} on {current_day.date()}->{expiration_date}, put_strike_sold={put_strike_sold}, close price: {close_price_pricing:.2f}, try ticker: {option_ticker}\033[0m")
                                # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_sold * 1000):08d}"
                                # result = await client._fetch_and_store_option_data(ticker, option_ticker, put_strike_sold, 'put', expiration_str, as_of_str, use_close_price=True)
                                # print(f"put strike {put_strike_sold} checked again, fetched data: {result}")
                                # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_bought * 1000):08d}"
                                # result = await client._fetch_and_store_option_data(ticker, option_ticker, put_strike_bought, 'put', expiration_str, as_of_str, use_close_price=True)
                                # print(f"put strike {put_strike_bought} checked again, fetched data: {result}")
                    # breakpoint()
                    invalid_data_flag = 'long strike out or range'
                    if SKIP_MISSING_STRIKE_TRADE:
                        return None

        if invalid_data_flag is None:
            new_legs = []   
            # Prepare leg info for the four legs (short call/put, long call/put)
            if target_premium_call != 0:
                option_ticker_sold = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int(call_strike_sold * 1000):08d}"
                option_ticker_bought = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int(call_strike_bought * 1000):08d}"
                new_legs.extend([
                    {'strike_price': call_strike_sold,  'call_put': 'call', 'expiration_date': expiration_str, 'quote_timestamp': as_of_str, 'option_ticker': option_ticker_sold},
                    {'strike_price': call_strike_bought,'call_put': 'call', 'expiration_date': expiration_str, 'quote_timestamp': as_of_str, 'option_ticker': option_ticker_bought}
                ])
            if target_premium_put != 0:
                option_ticker_sold = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_sold * 1000):08d}"
                option_ticker_bought = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_bought * 1000):08d}"
                new_legs.extend([
                    {'strike_price': put_strike_sold,   'call_put': 'put',  'expiration_date': expiration_str, 'quote_timestamp': as_of_str, 'option_ticker': option_ticker_sold},
                    {'strike_price': put_strike_bought, 'call_put': 'put',  'expiration_date': expiration_str, 'quote_timestamp': as_of_str, 'option_ticker': option_ticker_bought}
                ])
            
            options_to_fetch_filtered = []
            for option in new_legs:
                strike = option['strike_price']
                call_put_type = option['call_put']
                expiration = option['expiration_date']
                pricing_date = option['quote_timestamp']

                if strike is None:
                    print(f"Strike is None for {ticker} on {current_day.date()}")
                # Fetch from stored_option_price
                data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get(call_put_type.lower(), {})
                if not data:
                    options_to_fetch_filtered.append(option)
                    
            # Fetch prices for the four legs at entry
            entry_prices = await client.get_option_prices_batch_async(ticker, options_to_fetch_filtered)
            
            if target_premium_call != 0:
                sc_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(call_strike_sold,2), {}).get(expiration_str, {}).get('call', {})
                lc_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(call_strike_bought,2), {}).get(expiration_str, {}).get('call', {})
                short_call_prem_open = round(sc_data.get(premium_field, 0.0), 4) if sc_data and ( ( PREMIUM_PRICE_MODE == 'mid' and sc_data.get('ask_size', 0) > 0 and sc_data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and sc_data.get('close_volume', 0) > 0 ) or (PREMIUM_PRICE_MODE == 'trade' and sc_data.get('trade_size',0) > 0)) else 0
                long_call_prem_open = round(lc_data.get(premium_field, 0.0), 4) if lc_data and ( ( PREMIUM_PRICE_MODE == 'mid' and lc_data.get('ask_size', 0) > 0 and lc_data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and lc_data.get('close_volume', 0) > 0 ) or (PREMIUM_PRICE_MODE == 'trade' and lc_data.get('trade_size',0) > 0)) else 0
                if trade_type == 'covered_call':
                    long_call_prem_open = 0.0  # For covered call, we don't consider the long call premium
                if short_call_prem_open - long_call_prem_open <= 0:
                    print(f"\033[93mShort call premium is less than or equal to long put premium for {ticker} on {current_day.date()}->{expiration_str}, close price: {close_price_pricing:.2f}, call strike {call_strike_sold} short_call_prem_open={short_call_prem_open}, call strike {call_strike_bought} long_call_prem_open={long_call_prem_open}\033[0m")
                            # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int(call_strike_sold * 1000):08d}"
                            # result = await client._fetch_and_store_option_data(ticker, option_ticker, call_strike_sold, 'call', expiration_str, as_of_str, use_close_price=False)
                            # short_call_prem_open = result.get('close_price', 0.0) if result else 0.0
                            # print(f"call strike {call_strike_sold} checked again, fetched data: {result}")
                            # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}C{int(call_strike_bought * 1000):08d}"
                            # result = await client._fetch_and_store_option_data(ticker, option_ticker, call_strike_bought, 'call', expiration_str, as_of_str, use_close_price=False)
                            # long_call_prem_open = result.get('close_price', 0.0) if result else 0.0
                            # print(f"call strike {call_strike_bought} checked again, fetched data: {result}")
                    # breakpoint()
                    invalid_data_flag = "prem data reverse"
                    if SKIP_MISSING_STRIKE_TRADE:
                        return None
            if target_premium_put != 0:
                sp_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(put_strike_sold,2), {}).get(expiration_str, {}).get('put', {})
                lp_data = stored_option_price.get(ticker.upper(), {}).get(as_of_str, {}).get(round(put_strike_bought,2), {}).get(expiration_str, {}).get('put', {})
                # print(f"sp_data: {sp_data}, lp_data: {lp_data}")
                short_put_prem_open = round(sp_data.get(premium_field, 0.0), 4) if sp_data and ( ( PREMIUM_PRICE_MODE == 'mid' and sp_data.get('ask_size', 0) > 0 and sp_data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and sp_data.get('close_volume', 0) > 0 ) or (PREMIUM_PRICE_MODE == 'trade' and sp_data.get('trade_size',0) > 0)) else 0
                long_put_prem_open = round(lp_data.get(premium_field, 0.0), 4) if lp_data and ( ( PREMIUM_PRICE_MODE == 'mid' and lp_data.get('ask_size', 0) > 0 and lp_data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and lp_data.get('close_volume', 0) > 0 ) or (PREMIUM_PRICE_MODE == 'trade' and lp_data.get('trade_size',0) > 0)) else 0
                if short_put_prem_open - long_put_prem_open <= 0:
                                        # print(f"\033[93mShort put premium is less than or equal to long call premium for {ticker} on {current_day.date()}->{expiration_str}, close price: {close_price_pricing:.2f}, put strike {put_strike_sold} short_put_prem_open=${short_put_prem_open}, put strike {put_strike_bought} long_put_prem_open=${long_put_prem_open}\033[0m")
                                        # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_sold * 1000):08d}"
                                        # result = await client._fetch_and_store_option_data(ticker, option_ticker, put_strike_sold, 'put', expiration_str, as_of_str)
                                        # print(f"put strike {put_strike_sold} checked again, fetched data: {result}")
                                        # option_ticker = f"O:{ticker.upper()}{expiration_str[2:].replace('-','')}P{int(put_strike_bought * 1000):08d}"
                                        # result = await client._fetch_and_store_option_data(ticker, option_ticker, put_strike_bought, 'put', expiration_str, as_of_str)
                                        # print(f"put strike {put_strike_bought} checked again, fetched data: {result}")
                                        # for opt,prem in zip(put_options, all_put_data):
                                        #     print(f"put strike {opt['strike_price']} {opt['option_ticker']} {prem}")
                                        # breakpoint()
                    invalid_data_flag = "prem data reverse"
                    if SKIP_MISSING_STRIKE_TRADE:
                        return None

        open_distance_call = round((call_strike_sold - close_price_pricing) / close_price_pricing, 4) if call_strike_sold else 0
        open_distance_put =  round((close_price_pricing - put_strike_sold) / close_price_pricing, 4) if put_strike_sold else 0
        open_days_to_expiration = (expiration_date - trade_date).days
        distance_limit = 0.00 * (open_days_to_expiration**0.5)  # 1% per day

                # if open_distance_call and abs(open_distance_call) < distance_limit:
                #     print(f"Call strike distance {open_distance_call} breaches limit for {ticker} on {trade_date.date()}")
                #     call_strike_sold = None
                #     call_strike_bought = None
                #     sc_data = None
                #     lc_data = None
                # if open_distance_put and abs(open_distance_put) < distance_limit:
                #     print(f"Put strike distance {open_distance_put} breaches limit for {ticker} on {trade_date.date()}")
                #     put_strike_sold = None
                #     put_strike_bought = None
                #     sp_data = None
                #     lp_data = None
        # Build the new position dictionary with all relevant details
        global option_count
        option_count += 1

        position = {
            'position_open_date': trade_date,
            'expiration': expiration_date,
            'underly_price': close_price_pricing,
            'call_strike_sold': call_strike_sold,
            'put_strike_sold': put_strike_sold,
            'call_strike_bought': call_strike_bought,
            'put_strike_bought': put_strike_bought,
            'short_call_prem_open': short_call_prem_open,
            'short_put_prem_open': short_put_prem_open,
            'long_call_prem_open': long_call_prem_open,
            'long_put_prem_open': long_put_prem_open,
            'short_call_prem_today': short_call_prem_open,
            'short_put_prem_today': short_put_prem_open,
            'long_call_prem_today': long_call_prem_open,
            'long_put_prem_today': long_put_prem_open,
            'short_call_delta_open': sc_data.get(delta_field, 0.0) if sc_data else 0.0,
            'short_put_delta_open': sp_data.get(delta_field, 0.0) if sp_data else 0.0,
            'long_call_delta_open': lc_data.get(delta_field, 0.0) if lc_data else 0.0,
            'long_put_delta_open': lp_data.get(delta_field, 0.0) if lp_data else 0.0,
            'open_distance_call': round((call_strike_sold - close_price) / close_price, 4) if call_strike_sold else 0,
            'open_distance_put':  round((close_price - put_strike_sold) / close_price, 4) if put_strike_sold else 0,
            'required_margin': 0.0,
            'call_closed_by_stop': False, 'put_closed_by_stop': False,
            'call_stop_loss': 0.0, 'put_stop_loss': 0.0,
            'call_stop_date': None, 'put_stop_date': None,
            'position_hedged': False,
            'open_distance_call': open_distance_call,
            'open_distance_put': open_distance_put,
            'invalid_data': invalid_data_flag,
            'call_closed_date': None,
            'put_closed_date': None,
            'open_vix': vix_value,
            'strike_target_call': strike_target_call, 
            'strike_target_put': strike_target_put,
            'call_hold_to_expiration': open_atm_call>0,
            'put_hold_to_expiration': open_atm_put>0,
            'option_number': option_count,
            'iv_call': iv_calculated_call if iv_calculated_call else 0,
            'iv_put': iv_calculated_put if iv_calculated_put else 0,
        }
        # Calculate position margin requirement (max spread width exposure)
        if call_strike_sold and call_strike_bought:
            position['required_margin'] += max(0, call_strike_bought - call_strike_sold) * 100
        if put_strike_sold and put_strike_bought:
            position['required_margin'] += max(0, put_strike_sold - put_strike_bought) * 100
        return position

    def find_strike_custom(options_list, premiums_list, target, stock_price,
                        OTM=True, VOL_THRESHOLD_BYPASS=False):
        """
        Finds the option strike price based on the provided target criteria,
        first removing any options whose mid_price violates monotonicity
        (calls: non-increasing in strike; puts: non-increasing in descending strike).
        """
        if stock_price is None:
            print("Stock price is None, cannot find closest strike.")
            return None, 0

        # Decide what field we’re matching on
        if target.get("premium_target") is not None:
            search_type, target_value = "premium", target["premium_target"]
        elif target.get("delta_target") is not None:
            search_type, target_value = "delta", target["delta_target"]
            delta_field = DELTA_FIELD_MAP[PREMIUM_PRICE_MODE]
        else:
            search_type, target_value = "strike", stock_price

        premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]

        # Combine and pre-filter out missing / zero premiums
        opts = []
        for opt, prem in zip(options_list, premiums_list):
            if not isinstance(prem, dict): 
                continue
            if prem.get(premium_field) is None or prem[premium_field] <= 0:
                continue
            if search_type == "delta" and prem.get(delta_field) is None:
                continue
            opts.append((opt, prem))
        if not opts:
            return None, 0

        # Apply OTM filter
        if OTM:
            filtered = []
            for opt, prem in opts:
                cp = opt.get("call_put", "").lower()
                K = opt.get("strike_price")
                if cp == "call" and K >= stock_price:
                    filtered.append((opt, prem))
                elif cp == "put" and K <= stock_price:
                    filtered.append((opt, prem))
            if filtered:
                opts = filtered

        # Helper: strip out any mid_price that violates monotonicity
        def enforce_mid_monotonic(options):
            calls = [o for o in options if o[0]["call_put"].lower() == "call"]
            puts  = [o for o in options if o[0]["call_put"].lower() == "put"]
            keep = []

            premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE]
            # Calls: strikes ↑, mid_price must be non-increasing
            for opt, prem in sorted(calls, key=lambda x: x[0]["strike_price"]):
                mp = prem[premium_field]
                if not keep or mp <= keep[-1][1][premium_field]:
                    keep.append((opt, prem))

            # Puts: strikes ↓, mid_price must be non-increasing
            for opt, prem in sorted(puts, key=lambda x: x[0]["strike_price"], reverse=True):
                mp = prem[premium_field]
                if len(keep)==0 or mp <= keep[-1][1][premium_field]:
                    keep.append((opt, prem))

            return keep

        opts = enforce_mid_monotonic(opts)
        if not opts:
            return None, 0

        # Now proceed exactly as before, but operating on 'opts' instead of 'filtered_options'
        # — Sort by strike for search routines —
        sorted_opts = sorted(opts, key=lambda x: x[0]["strike_price"])

        # If we're picking by strike distance
        if search_type == "strike":
            # Find the one whose strike is closest to the stock price
            best_opt, best_p = min(
                opts,
                key=lambda x: abs(x[0]["strike_price"] - stock_price)
            )
            candidate_strike = best_opt["strike_price"]
            candidate_prem   = best_p[premium_field]
            vol              = best_p.get("close_volume", 0)
            vol_thr          = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10
            if vol < vol_thr:
                # bump to a higher-volume neighbor
                high_vol = [x for x in opts if x[1].get("close_volume", 0) >= vol_thr]
                if high_vol:
                    best_opt, best_p = min(
                        high_vol,
                        key=lambda x: abs(x[0]["strike_price"] - stock_price)
                    )
                    candidate_strike = best_opt["strike_price"]
                    candidate_prem   = best_p[premium_field]
                else:
                    print(f"Warning: low volume {vol} on strike {candidate_strike}")
            return candidate_strike, round(candidate_prem, 2)

        # Otherwise, premium or delta search; we'll walk up/down from the ATM strike
        value_field = premium_field if search_type == "premium" else delta_field

        # locate starting index
        atm_strike = min(sorted_opts, key=lambda x: abs(x[0]["strike_price"] - stock_price))[0]["strike_price"]
        try:
            start_idx = next(i for i, (o, _) in enumerate(sorted_opts) if o["strike_price"] >= atm_strike)
        except StopIteration:
            start_idx = len(sorted_opts) - 1

        best = {"diff": float("inf"), "strike": None, "prem": 0, "vol": 0}
        pairs = []

        for direction_up in (True, False):
            idx, prev_val = start_idx, None
            while 0 <= idx < len(sorted_opts):
                opt, prem = sorted_opts[idx]
                val = prem.get(value_field, 0)
                diff = abs(val - target_value)
                pairs.append((diff, opt["strike_price"], prem[premium_field], prem.get("close_volume", 0)))

                # pick best so far
                if diff < best["diff"]:
                    best.update(diff=diff, strike=opt["strike_price"],
                                prem=prem[premium_field], vol=prem.get("close_volume", 0))

                # interpolation break
                if prev_val is not None and ((prev_val < target_value < val) or (val < target_value < prev_val)):
                    break

                prev_val = val
                idx = idx + 1 if direction_up else idx - 1

        # fallback to nearest if nothing interpolated
        if best["strike"] is None and pairs:
            d, s, p, v = min(pairs, key=lambda x: x[0])
            best.update(diff=d, strike=s, prem=p, vol=v)

        # ensure volume
        vol_thr = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10
        if best["vol"] < vol_thr:
            high = [pp for pp in pairs if pp[3] >= vol_thr]
            if high:
                _, s, p, _ = min(high, key=lambda x: x[0])
                best.update(strike=s, prem=p)
            else:
                print(f"Warning: best match strike {best['strike']} vol {best['vol']} < {vol_thr}")

        return best["strike"], round(best["prem"], 2)

    def find_closest_premium_strike(options_list, premiums_list, target, stock_price, OTM=True, VOL_THRESHOLD_BYPASS=False):
        """
        Finds the option strike price whose premium is closest to the target premium.
        If target is None, returns the strike closest to the stock_price while still following the OTM rule.
        If no OTM options are found when OTM=True, falls back to the option with the strike closest to stock_price.

        Args:
            options_list (list): A list of dictionaries, each representing an option contract.
                                Expected keys: 'strike_price', 'expiration_date', 'call_put' ('call' or 'put').
            premiums_list (list): A list of dictionaries, each containing premium data for the corresponding option.
                                Expected keys: 'close_price', 'mid_price', 'close_volume'.
            target (float or None): The target premium value to find the closest match for. If None, select based on strike proximity.
            stock_price (float): The current price of the underlying stock. Required for OTM filtering.
            OTM (bool, optional): If True, only consider Out-of-the-Money options.
                                Requires 'call_put' key in options_list elements. Defaults to True.
            VOL_THRESHOLD_BYPASS (bool, optional): If True, uses a lower volume threshold (10). Defaults to False.

        Returns:
            tuple: A tuple containing:
                - strike (float or None): The strike price of the best matching option, or None if no suitable option found.
                - premium (float): The premium of the selected option (0 if no option found).
        """
        # Check if stock_price is provided
        if stock_price is None:
            print("Stock price is None, cannot find closest premium strike (required for OTM check if enabled).")
            return None, 0

        valid_pairs = []

        # Validate input lengths
        if len(options_list) != len(premiums_list):
            print(f"Warning: Length mismatch: {len(options_list)} options, {len(premiums_list)} premiums")

        if not options_list or not premiums_list:
            return None, 0

        # Combine options and premiums, filtering out invalid premium data
        options_with_premiums = []
        for opt, prem_data in zip(options_list, premiums_list):
            if isinstance(prem_data, dict) and premium_field in prem_data and prem_data[premium_field] is not None and prem_data[premium_field] > 0:
                options_with_premiums.append((opt, prem_data))

        if not options_with_premiums:
            return None, 0

        # Sort options by strike price
        sorted_options = sorted(options_with_premiums, key=lambda x: x[0]['strike_price'])

        # OTM Filtering if enabled (applies both for target provided or None)
        filtered_options = []
        if OTM:
            for opt, prem_data in sorted_options:
                if 'call_put' not in opt:
                    print(f"Warning: 'call_put' key missing for strike {opt.get('strike_price', 'N/A')}. Including in search.")
                    filtered_options.append((opt, prem_data))
                    continue
                call_put = opt['call_put'].lower()
                strike_price = opt['strike_price']
                is_otm = (call_put == 'call' and strike_price >= stock_price) or (call_put == 'put' and strike_price <= stock_price)
                if is_otm:
                    filtered_options.append((opt, prem_data))
        else:
            filtered_options = sorted_options
            print(f"stock_price: {stock_price} No OTM filtering applied.")

        # Use sorted_options as fallback if no filtered option is available
        if not filtered_options and sorted_options:
            filtered_options = sorted_options

        if not filtered_options:
            return None, 0

        # --- New branch: when target is None, select option with strike closest to stock_price ---
        if target is None:
            best_option, best_prem_data = min(filtered_options, key=lambda x: abs(x[0]['strike_price'] - stock_price))
            candidate_strike = best_option['strike_price']
            candidate_premium = best_prem_data.get(premium_field, 0)
            candidate_volume = best_prem_data.get("close_volume", 0)
            volume_threshold = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10

            if candidate_volume < volume_threshold:
                vol_candidates = [p for p in filtered_options if p[1].get("close_volume", 0) >= volume_threshold]
                if vol_candidates:
                    best_vol_candidate = min(vol_candidates, key=lambda x: abs(x[0]['strike_price'] - stock_price))
                    candidate_strike = best_vol_candidate[0]['strike_price']
                    candidate_premium = best_vol_candidate[1].get(premium_field, 0)
                else:
                    print(f"Warning: Selected strike {candidate_strike} has volume {candidate_volume} below threshold {volume_threshold}.")
            return candidate_strike, candidate_premium
        # --- End of target is None branch ---

        # If target is provided, perform bidirectional search for premium closest to target
        # Fallback: if no OTM options are found when OTM is True, use option with strike closest to stock_price
        if OTM and not filtered_options:
            # Find the option with strike closest to stock_price
            closest_option, closest_prem_data = min(sorted_options, key=lambda x: abs(x[0]['strike_price'] - stock_price))
            candidate_strike = closest_option['strike_price']
            candidate_premium = closest_prem_data.get(premium_field, 0)
            candidate_volume = closest_prem_data.get("close_volume", 0)
            volume_threshold = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10
            print(f"\033[91m Quote: {closest_option['quote_timestamp']} Expire: {closest_option['expiration_date']} close: {stock_price} Warning: No OTM options found. Selecting the option with strike closest to stock price.  \033[0m")
            for opt, prem_data in sorted_options:
                print(f"  - Option: Strike {opt['strike_price']}, {opt['call_put']} Premium: {prem_data.get(premium_field, 0)}, Volume: {prem_data.get('close_volume', 0)}")
            if candidate_volume < volume_threshold:
                print(f"Warning: Selected strike {candidate_strike} has volume {candidate_volume} below threshold {volume_threshold}.")
            return candidate_strike, candidate_premium
        else:
            search_list = filtered_options if OTM else sorted_options

        # Bidirectional search for premium closest to target
        start_idx_ref_strike = min(search_list, key=lambda x: abs(x[0]['strike_price'] - stock_price))[0]['strike_price']
        try:
            start_idx = next(i for i, (opt, _) in enumerate(search_list) if opt['strike_price'] >= start_idx_ref_strike)
        except StopIteration:
            start_idx = len(search_list) - 1

        best_candidate_info = {'diff': float('inf'), 'strike': None, 'premium': 0, 'vol': 0, 'exp': None}

        for direction_up in [True, False]:
            idx = start_idx
            prev_premium = None
            while 0 <= idx < len(search_list):
                opt, prem_data = search_list[idx]
                premium = prem_data.get(premium_field, 0)
                vol = prem_data.get("close_volume", 0)
                strike = opt['strike_price']
                exp_date = opt.get('expiration_date')
                current_diff = abs(premium - target)
                valid_pairs.append((current_diff, strike, premium, vol, exp_date))

                if prev_premium is not None:
                    if (prev_premium < target < premium) or (premium < target < prev_premium):
                        prev_idx = idx - 1 if direction_up else idx + 1
                        if 0 <= prev_idx < len(search_list):
                            prev_opt, prev_prem_data = search_list[prev_idx]
                            prev_strike = prev_opt['strike_price']
                            prev_vol = prev_prem_data.get("close_volume", 0)
                            prev_exp = prev_opt.get('expiration_date')
                            prev_diff = abs(prev_premium - target)
                            if prev_diff < best_candidate_info['diff']:
                                best_candidate_info = {'diff': prev_diff, 'strike': prev_strike, 'premium': prev_premium, 'vol': prev_vol, 'exp': prev_exp}
                        break

                if current_diff < best_candidate_info['diff']:
                    best_candidate_info = {'diff': current_diff, 'strike': strike, 'premium': premium, 'vol': vol, 'exp': exp_date}

                prev_premium = premium
                idx = idx + 1 if direction_up else idx - 1

        # Fallback if no better candidate is found
        if best_candidate_info['strike'] is None:
            if not valid_pairs:
                return None, 0
            min_diff_pair = min(valid_pairs, key=lambda x: x[0])
            best_candidate_info = {'diff': min_diff_pair[0], 'strike': min_diff_pair[1], 'premium': min_diff_pair[2], 'vol': min_diff_pair[3], 'exp': min_diff_pair[4]}

        # Volume threshold check
        volume_threshold = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10
        candidate_strike = best_candidate_info['strike']
        candidate_premium = best_candidate_info['premium']
        candidate_volume = best_candidate_info['vol']

        if candidate_volume < volume_threshold:
            vol_candidates = [p for p in valid_pairs if p[3] >= volume_threshold]
            if vol_candidates:
                best_vol_candidate = min(vol_candidates, key=lambda x: x[0])
                candidate_strike = best_vol_candidate[1]
                candidate_premium = best_vol_candidate[2]
            else:
                print(f"Warning: Best match strike {candidate_strike} (premium {candidate_premium:.2f}) has volume {candidate_volume}, below threshold {volume_threshold}.")

        return candidate_strike, round(candidate_premium,2)

    # Initialize daily results list
    daily_results: List[Dict[str, Any]] = []
    daily_dates: List[datetime] = []
    realized_total = 0.0  # track cumulative realized PnL
    # Iterate through each trading day
    trading_days = pd.to_datetime(hist_df['date'].unique()).to_pydatetime().tolist()
    trading_days.sort()
    if ticker not in stored_option_chain:
        load_stored_option_data(ticker)
    closed_position_days = []          # ⬅️  collect holding-period stats
    positions_opened_total = 0
    prev_realized_total  = 0.0         # used to derive daily realized gain

    for current_day in trading_days:
        if new_data_entry_count > 5000:
            save_stored_option_data(ticker)
        # Stop simulation after end_date if no open positions remain
        if current_day.date() > end_dt.date() and not active_positions:
            break

        vix_value = vix_close_series.get(current_day, None)

        # Check if VIX data is available; skip the trade if not
        if vix_value is None:
            vix_value = 0

        # Get underlying close price for the day
        close_price = float(hist_df.loc[hist_df['date'] == current_day, 'close'].iloc[0])
        open_atm_call, open_atm_put = [], []
        ## 1. Handle expiring positions on this day
        expiring_today = [pos for pos in active_positions if pos['expiration'].date() == current_day.date()]
        num_full_closed = len(expiring_today)  # Start with expired positions as full closes
        for pos in expiring_today:
            # Determine payoff for each leg at expiration
            sc_loss = max(close_price - pos['call_strike_sold'], 0) if pos['short_call_prem_open'] > 0 and not pos['call_closed_by_stop'] else 0
            lc_gain = max(close_price - pos['call_strike_bought'], 0) if pos['long_call_prem_open'] > 0 and not pos['call_closed_by_stop'] else 0
            call_loss_final = sc_loss - lc_gain
            sp_loss = max(pos['put_strike_sold'] - close_price, 0) if pos['short_put_prem_open'] > 0 and not pos['put_closed_by_stop'] else 0
            lp_gain = max(pos['put_strike_bought'] - close_price, 0) if pos['long_put_prem_open'] > 0 and not pos['put_closed_by_stop'] else 0
            put_loss_final = sp_loss - lp_gain
            if pos['call_closed_date'] is None:
                pos['call_closed_date'] = current_day
                pos['call_closed_profit'] = (pos['short_call_prem_open'] - pos['long_call_prem_open'] - call_loss_final) * 100
            if pos['put_closed_date'] is None:
                pos['put_closed_date'] = current_day
                pos['put_closed_profit'] = (pos['short_put_prem_open'] - pos['long_put_prem_open'] - put_loss_final) * 100
            # Realized PnL = initial credit + payoff - any stop-loss losses
            payoff = -(call_loss_final + put_loss_final) * 100
            if call_loss_final > 2 and trade_type != 'covered_call':
                open_atm_call.append(call_loss_final)
            if put_loss_final > 2:
                open_atm_put.append(put_loss_final)
            # stop_losses = pos.get('call_stop_loss', 0.0) + pos.get('put_stop_loss', 0.0)
            if payoff < 0:
                if mode == "validation":
                    print(f"\033[91mExpired {pos['position_open_date'].date()}->{current_day.date()} | {ticker} (call_loss={call_loss_final*100:.0f}, put_loss={put_loss_final*100:.0f}) {ticker} close: {close_price:.2f} | {pos['call_strike_sold']}<->{pos['call_strike_bought']} | {pos['put_strike_sold']}<->{pos['put_strike_bought']} | Open distance call: {pos['open_distance_call']:.2%} | Open distance put: {pos['open_distance_put']:.2%} | realized loss: ${payoff:.0f} realized total: ${realized_total:.0f}|\033[0m")
                    # breakpoint()
            # realized_pnl = payoff - stop_losses
            realized_pnl = payoff
            realized_total += realized_pnl
        # Remove expired positions
        # active_positions = [pos for pos in active_positions if pos not in expiring_today]

        open_legs = []
        required_margin = 0

        for pos in active_positions:
            if pos['expiration'] <= current_day or pos['invalid_data'] is not None:
                continue
            
            if stop_profit_percent != 0: #only check position value daily if there is stop loss requirement
                exp_key = pos['expiration'].strftime("%Y-%m-%d")
                unique_chain_requests = [(exp_key, current_day.strftime("%Y-%m-%d"), "call"), (exp_key, current_day.strftime("%Y-%m-%d"), "put")]
                chain_data = await client.get_option_chains_batch_async(ticker, list(unique_chain_requests),force_update=False)
                
                # Get available strike dictionaries
                call_strike_dict = chain_data[ticker][exp_key][current_day.strftime("%Y-%m-%d")]["call"]
                put_strike_dict = chain_data[ticker][exp_key][current_day.strftime("%Y-%m-%d")]["put"]

                # short legs (only if not closed by stop)
                if pos['short_call_prem_open'] > 0 and not pos['call_closed_by_stop']:
                    option_ticker = f"O:{ticker.upper()}{exp_key[2:].replace('-','')}C{int(pos['call_strike_sold'] * 1000):08d}"
                    open_legs.append({
                        "strike_price": pos["call_strike_sold"],
                        "call_put": "call",
                        "expiration_date": exp_key,
                        "quote_timestamp": current_day.strftime("%Y-%m-%d"),
                        "option_ticker": option_ticker
                    })
                if pos['short_put_prem_open'] > 0 and not pos['put_closed_by_stop']:
                    option_ticker = f"O:{ticker.upper()}{exp_key[2:].replace('-','')}P{int(pos['put_strike_sold'] * 1000):08d}"
                    open_legs.append({
                        "strike_price": pos["put_strike_sold"],
                        "call_put": "put",
                        "expiration_date": exp_key,
                        "quote_timestamp": current_day.strftime("%Y-%m-%d"),
                        "option_ticker": option_ticker
                    })
                # (long legs – optional but useful for mark‑to‑market)
                if pos.get("long_call_prem_open", 0) > 0 and not pos.get("call_closed_by_stop", False):
                    option_ticker = f"O:{ticker.upper()}{exp_key[2:].replace('-','')}C{int(pos['call_strike_bought'] * 1000):08d}"
                    open_legs.append({
                        "strike_price": pos["call_strike_bought"],
                        "call_put": "call",
                        "expiration_date": exp_key,
                        "quote_timestamp": current_day.strftime("%Y-%m-%d"),
                        "option_ticker": option_ticker
                    })
                if pos.get("long_put_prem_open", 0) > 0 and not pos.get("put_closed_by_stop", False):
                    option_ticker = f"O:{ticker.upper()}{exp_key[2:].replace('-','')}P{int(pos['put_strike_bought'] * 1000):08d}"
                    open_legs.append({
                        "strike_price": pos["put_strike_bought"],
                        "call_put": "put",
                        "expiration_date": exp_key,
                        "quote_timestamp": current_day.strftime("%Y-%m-%d"),
                        "option_ticker": option_ticker
                    })

            margin_calculated = False
            #do not duplicate margin for call and put spreads
            if not margin_calculated:
                required_margin += 100 * (pos['call_strike_bought'] - pos['call_strike_sold']) if pos['call_closed_by_stop'] is False and pos['call_strike_sold'] and pos['short_call_prem_open'] > 0 else 0
                margin_calculated = True if pos['call_closed_by_stop'] is False and pos['call_strike_sold'] and pos['short_call_prem_open'] > 0 else False
            if not margin_calculated:    
                required_margin += 100 * (pos['put_strike_sold'] - pos['put_strike_bought']) if pos['put_closed_by_stop'] is False and pos['put_strike_sold'] and pos['short_put_prem_open'] > 0 else 0
                margin_calculated = True if pos['put_closed_by_stop'] is False and pos['put_strike_sold'] and pos['short_put_prem_open'] > 0 else False

        options_to_fetch_filtered = []
        for option in open_legs:
            strike = option['strike_price']
            call_put_type = option['call_put']
            expiration = option['expiration_date']
            pricing_date = option['quote_timestamp']

            if strike is None:
                print(f"Strike is None for {ticker} on {current_day.date()}")
            # Fetch from stored_option_price
            data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get(call_put_type.lower(), {})
            if not data or data.get(premium_field, None) is None:
                options_to_fetch_filtered.append(option)
            # else:
            #     if not data.get(premium_field): # or data.get('ask_size', 0) == 0 or data.get('bid_size', 0) == 0:
            #         options_to_fetch_filtered.append(option)

                    # current_prices = {}
        if len(options_to_fetch_filtered)>0:
            print(f"Fetching {len(options_to_fetch_filtered)} option prices for {ticker} on {current_day.date()} to check today`s value")
            price_data = await client.get_option_prices_batch_async(ticker, options_to_fetch_filtered)
                        
        # -----------------------------------------------------------
        # 2.  PRICE‑BASED STOP‑PROFIT-LOSS ― evaluate WHOLE position
        # -----------------------------------------------------------
        if stop_profit_percent != 0 and active_positions:
            for pos in active_positions:
                if pos['expiration'] <= current_day or pos['invalid_data'] is not None:
                    continue                                  # expiring today already handled
                if pos.get('position_hedged'):                # rolled/hedged elsewhere
                    continue

                exp_key = pos['expiration'].strftime("%Y-%m-%d")

                # ---------- ENTRY CREDIT ( what you collected ) ----------
                entry_credit_call = (
                    pos['short_call_prem_open']
                    - pos['long_call_prem_open']
                ) * 100                                       # dollars/condor

                entry_credit_put = (
                    + pos['short_put_prem_open']
                    - pos['long_put_prem_open']
                ) * 100   

                # ---------- CURRENT COST TO CLOSE  ( remaining open legs only ) ----------
                close_call_cost = 0.0
                close_put_cost = 0.0
                short_put_prem_updated, short_call_prem_updated, long_put_prem_updated, long_call_prem_updated = False, False, False, False
                
                #   short CALL  (if still open)
                if pos['short_call_prem_open'] > 0 and pos['call_closed_date'] is None:
                    data = stored_option_price.get(ticker.upper(), {}).get(current_day.strftime("%Y-%m-%d"), {}).get(round(pos['call_strike_sold'],2), {}).get(exp_key, {}).get('call', {})
                    if data and data.get(premium_field, None) is not None and ( ( PREMIUM_PRICE_MODE == 'mid' and data.get('ask_size', 0) > 0 and data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and data.get('close_volume', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'trade' and data.get('trade_size', 0) > 0 ) ):
                        pos['short_call_prem_today'] = data.get(premium_field, None)
                        short_call_prem_updated = True
                    else:
                        short_call_interpolated_price = await interpolate_option_price(
                            ticker=ticker,
                            close_price_today=close_price,
                            strike_price_to_interpolate=pos['call_strike_sold'],
                            option_type='call',
                            expiration_date=exp_key,
                            pricing_date=current_day.strftime("%Y-%m-%d"),
                            stored_option_price=stored_option_price,
                            premium_field=premium_field,
                            price_interpolate_flag=PRICE_INTERPOLATE,
                            max_strike_search_distance=50,
                            interpolation_point_max_spread=6,
                            client=client,
                        )
                        # print(f"trade price interpolated: {round(pos['call_strike_sold'],2)} {short_call_interpolated_price}")
                        if short_call_interpolated_price != 0:
                            pos['short_call_prem_today'] = short_call_interpolated_price
                            short_call_prem_updated = True
                        # breakpoint()
                #   short PUT   (if still open)
                if pos['short_put_prem_open'] > 0 and pos['put_closed_date'] is None:
                    data = stored_option_price.get(ticker.upper(), {}).get(current_day.strftime("%Y-%m-%d"), {}).get(round(pos['put_strike_sold'],2), {}).get(exp_key, {}).get('put', {})
                    if data and data.get(premium_field, None) is not None and ( ( PREMIUM_PRICE_MODE == 'mid' and data.get('ask_size', 0) > 0 and data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and data.get('close_volume', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'trade' and data.get('trade_size', 0) > 0 ) ):
                        pos['short_put_prem_today'] = data.get(premium_field, None)
                        short_put_prem_updated = True
                    else:
                        short_put_interpolated_price = await interpolate_option_price(
                            ticker=ticker,
                            close_price_today=close_price,
                            strike_price_to_interpolate=pos['put_strike_sold'],
                            option_type='put',
                            expiration_date=exp_key,
                            pricing_date=current_day.strftime("%Y-%m-%d"),
                            stored_option_price=stored_option_price,
                            premium_field=premium_field,
                            price_interpolate_flag=PRICE_INTERPOLATE,
                            max_strike_search_distance=50,
                            interpolation_point_max_spread=6,
                            client=client,
                        )
                        # print(f"trade price interpolated: {round(pos['put_strike_sold'],2)} {short_put_interpolated_price}")
                        if short_put_interpolated_price != 0:
                            pos['short_put_prem_today'] = short_put_interpolated_price
                            short_put_prem_updated = True
                        # breakpoint()

                #   long CALL   (always include – you could sell it back)
                if pos.get('long_call_prem_open', 0) > 0 and pos['call_closed_date'] is None and close_call_cost is not None:
                    data = stored_option_price.get(ticker.upper(), {}).get(current_day.strftime("%Y-%m-%d"), {}).get(round(pos['call_strike_bought'],2), {}).get(exp_key, {}).get('call', {})
                    if data and data.get(premium_field, None) is not None and ( ( PREMIUM_PRICE_MODE == 'mid' and data.get('ask_size', 0) > 0 and data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and data.get('close_volume', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'trade' and data.get('trade_size', 0) > 0 ) ):
                        close_call_cost -= data.get(premium_field, 0)
                        pos['long_call_prem_today'] = data.get(premium_field, None)
                        long_call_prem_updated = True
                    else:
                        long_call_interpolated_price = await interpolate_option_price(
                            ticker=ticker,
                            close_price_today=close_price,
                            strike_price_to_interpolate=pos['call_strike_bought'],
                            option_type='call',
                            expiration_date=exp_key,
                            pricing_date=current_day.strftime("%Y-%m-%d"),
                            stored_option_price=stored_option_price,
                            premium_field=premium_field,
                            price_interpolate_flag=PRICE_INTERPOLATE,
                            max_strike_search_distance=50,
                            interpolation_point_max_spread=6,
                            client=client,
                        )
                        if long_call_interpolated_price != 0:
                            pos['long_call_prem_today'] = long_call_interpolated_price
                            long_call_prem_updated = True

                #   long PUT
                if pos.get('long_put_prem_open', 0) > 0 and pos['put_closed_date'] is None and close_put_cost is not None:
                    data = stored_option_price.get(ticker.upper(), {}).get(current_day.strftime("%Y-%m-%d"), {}).get(round(pos['put_strike_bought'],2), {}).get(exp_key, {}).get('put', {})
                    if data and data.get(premium_field, None) is not None and ( ( PREMIUM_PRICE_MODE == 'mid' and data.get('ask_size', 0) > 0 and data.get('bid_size', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'close' and data.get('close_volume', 0) > 0 ) or ( PREMIUM_PRICE_MODE == 'trade' and data.get('trade_size', 0) > 0 ) ):
                        close_put_cost -= data.get(premium_field, 0)
                        pos['long_put_prem_today'] = data.get(premium_field, None)
                        long_put_prem_updated = True
                    else:
                        short_put_interpolated_price = await interpolate_option_price(
                            ticker=ticker,
                            close_price_today=close_price,
                            strike_price_to_interpolate=pos['put_strike_bought'],
                            option_type='put',
                            expiration_date=exp_key,
                            pricing_date=current_day.strftime("%Y-%m-%d"),
                            stored_option_price=stored_option_price,
                            premium_field=premium_field,
                            price_interpolate_flag=PRICE_INTERPOLATE,
                            max_strike_search_distance=50,
                            interpolation_point_max_spread=6,
                            client=client,
                        )
                        if short_put_interpolated_price != 0:
                            pos['long_put_prem_today'] = short_put_interpolated_price
                            long_put_prem_updated = True

                    if short_put_prem_updated and long_put_prem_updated:
                        close_put_cost = pos['short_put_prem_today'] - pos['long_put_prem_today']
                    else:
                        close_put_cost = 0
                        # print(f"\033[93mNo data for {ticker} on {current_day.date()} at strike {pos['put_strike_bought']} expiring {exp_key}, short_put_prem_updated: {short_put_prem_updated}, long_put_prem_updated: {long_put_prem_updated}\033[0m")
                        # breakpoint()
                    if short_call_prem_updated and long_call_prem_updated:
                        close_call_cost = pos['short_call_prem_today'] - pos['long_call_prem_today']
                    else:
                        close_call_cost = 0
                        # print(f"\033[93mNo data for {ticker} on {current_day.date()} at strike {pos['call_strike_bought']} expiring {exp_key}, short_call_prem_updated: {short_call_prem_updated}, long_call_prem_updated: {long_call_prem_updated}\033[0m")
                        # breakpoint()
                # ---------- STOP-PROFIT TRIGGER ----------
                call_expiring_soon = pos.get('expiration') - current_day <= timedelta(days=trade_parameter['expiring_wks'] / 2 * 7)
                put_expiring_soon = pos.get('expiration') - current_day <= timedelta(days=trade_parameter['expiring_wks'] / 2 * 7)
                # call_expiring_soon = False
                # put_expiring_soon = False
                # ---- CALL LEG ----
                if not pos.get('call_closed_by_stop', False) and close_call_cost is not None and entry_credit_call > 0:
                    if pos.get('call_hold_to_expiration', False) is False:
                        call_triggered_by_profit        = 0 < close_call_cost * 100 <= stop_profit_percent * entry_credit_call
                        call_triggered_by_lost = close_call_cost * 100 >=  50 * entry_credit_call
                        call_triggered_by_expiration  = 0 < close_call_cost * 100 <= 2 * entry_credit_call
                        # print(f"call_triggered_by_expiration: {call_triggered_by_expiration}, call_triggered_by_profit: {call_triggered_by_profit}, call_triggered_by_lost: {call_triggered_by_lost}, {pos['call_strike_sold']}<->{pos['call_strike_bought']}, close_call_cost: {close_call_cost}, entry_credit_call: {entry_credit_call}, close: {close_price}, vix: {vix_value}") 
                    else:
                        call_triggered_by_profit        = 0 < close_call_cost * 100 <= 0.2 * entry_credit_call
                        call_triggered_by_lost = close_call_cost * 100 >=  50 * entry_credit_call
                        call_triggered_by_expiration  = False

                    if call_triggered_by_profit or (call_expiring_soon and call_triggered_by_expiration) or call_triggered_by_lost:
                        realised_loss   = round(-close_call_cost * 100, 2) - 2 * 0.5
                        realized_total += realised_loss
                        pos['call_closed_by_stop'] = True
                        pos['call_closed_date'] = current_day
                        num_full_closed += 1
                        required_margin += 100 * (pos['call_strike_bought'] - pos['call_strike_sold'])
                        
                        if mode == "validation":
                            net_result  = entry_credit_call + realised_loss
                            pos['call_closed_profit'] = net_result
                            color_code  = "\033[92m" if net_result >= 0 else "\033[91m"
                            if call_triggered_by_lost:
                                reason = "loss"
                            elif call_triggered_by_profit:
                                reason = "profit"
                            else:
                                reason = "expiration"

                            print(
                                f"{color_code}Call closed by {reason} for {ticker} on {current_day.date()} "
                                f"({pos['position_open_date'].date()}->{pos['expiration'].date()}) | Strike {pos['call_strike_sold']}<->{pos['call_strike_bought']} ${pos['short_call_prem_today']:.2f}<->${pos['long_call_prem_today']:.2f}| Open Distance {pos['open_distance_call']:.0%} | "
                                f"Price {close_price:.2f} | VIX: {vix_value:.1f} | Entry ${entry_credit_call:.2f} | "
                                f"Close Δ ${realised_loss:.2f} → Net ${net_result:.0f}\033[0m"
                            )

                # ---------- PUT LEG ----------
                if not pos.get('put_closed_by_stop', False) and close_put_cost is not None and entry_credit_put > 0:
                    if pos.get('put_hold_to_expiration', False) is False:
                        put_triggered_by_profit        = 0 < close_put_cost * 100 <= stop_profit_percent * entry_credit_put
                        put_triggered_by_lost = close_put_cost * 100 >=  50 * entry_credit_put
                        put_triggered_by_expiration  = 0 < close_put_cost * 100 <= 2 * entry_credit_put
                        # print(f"put_triggered_by_expiration: {put_triggered_by_expiration}, put_triggered_by_profit: {put_triggered_by_profit}, put_triggered_by_lost: {put_triggered_by_lost}, {pos['put_strike_sold']}<->{pos['put_strike_bought']}, close_put_cost: {close_put_cost}, entry_credit_put: {entry_credit_put}, close: {close_price}, vix: {vix_value}") 
                    else:
                        put_triggered_by_profit        = 0 < close_put_cost * 100 <= 0.2 * entry_credit_put
                        put_triggered_by_lost = close_put_cost * 100 >=  50 * entry_credit_put
                        put_triggered_by_expiration  = False

                    if put_triggered_by_profit or (put_expiring_soon and put_triggered_by_expiration) or put_triggered_by_lost:
                        realised_loss   = round(-close_put_cost * 100, 2) - 2 * 0.5
                        realized_total += realised_loss
                        pos['put_closed_by_stop'] = True
                        pos['put_closed_date'] = current_day
                        num_full_closed += 1
                        required_margin -= 100 * (pos['put_strike_sold'] - pos['put_strike_bought'])

                        if mode == "validation":
                            net_result  = entry_credit_put + realised_loss
                            pos['put_closed_profit'] = net_result
                            color_code  = "\033[92m" if net_result >= 0 else "\033[91m"
                            if put_triggered_by_lost:
                                reason = "loss"
                            elif put_triggered_by_profit:
                                reason = "profit"
                            else:
                                reason = "expiration"

                            print(
                                f"{color_code}Put closed by {reason} for {ticker} on {current_day.date()} "
                                f"({pos['position_open_date'].date()}->{pos['expiration'].date()}) | Strike {pos['put_strike_sold']}<->{pos['put_strike_bought']} ${pos['short_put_prem_today']:.2f}<->${pos['long_put_prem_today']:.2f}| Open Distance {pos['open_distance_put']:.0%} | "
                                f"Price {close_price:.2f} | VIX: {vix_value:.1f} | Entry ${entry_credit_put:.2f} | "
                                f"Close Δ ${realised_loss:.2f} → Net ${net_result:.0f}\033[0m"
                            )
                            
        aggregated_call_loss, aggregated_put_loss = 0, 0
        for loss in open_atm_call:
            aggregated_call_loss += loss
        for loss in open_atm_put:
            aggregated_put_loss += loss
        ## 3. Open a new position on the designated day_of_week
        # if current_day.strftime("%A") in trade_parameter['day_of_week'] or required_margin < 100 * iron_condor_width * len(trade_parameter['day_of_week']) * 2 * trade_parameter['expiring_wks']:
        if current_day.strftime("%A") in trade_parameter['day_of_week']:
            # Determine target expiration date (e.g., next Friday or specified offset)
            curr_wd = current_day.weekday()
            # target_weekday = curr_wd
            target_weekday = 4
            # Compute initial expiration candidate (expiring_wks weeks out)
            # days_offset ensures expiration falls on target_weekday
            days_offset = (target_weekday - curr_wd) % 7
            
            if aggregated_call_loss > 0 or aggregated_put_loss > 0:
                this_expiring_wks = trade_parameter['expiring_wks'] * 2
                print(f"open_atm_call: {aggregated_call_loss}, open_atm_put: {aggregated_put_loss}")
            else:
                this_expiring_wks = trade_parameter['expiring_wks']           
            expiration_candidate = current_day + pd.Timedelta(
                weeks=int(this_expiring_wks),
                days=days_offset
            )
            print(f"current_day: {current_day}, expiration_candidate: {expiration_candidate}, this_expiring_wks: {this_expiring_wks}, days_offset: {days_offset}")
            # Adjust to actual trading day on or before candidate (to handle holidays)
                    # trading_dates = list(df['date'])
                    # idx = bisect.bisect_right(trading_dates, expiration_candidate) - 1
                    # expire_day = pd.to_datetime(trading_dates[idx]) if idx >= 0 else None
            expire_day = expiration_candidate
            if expire_day and expire_day.date() > current_day.date():
                # Skip trade if earnings fall between entry and expiration
                if any(current_day.date() <= ed < expire_day.date() for ed in earnings_dates) and SKIP_EARNINGS:
                    print(f"Skipped new position on {current_day.date()} due to earnings before {expire_day.date()}")
                else:
                    counter=0
                    this_open_atm_call = open_atm_call
                    this_open_atm_put = open_atm_put

                    # while len(this_open_atm_call) > 0 or len(this_open_atm_put) > 0 or counter == 0:

                    available_capital = INITIAL_CAPITAL + realized_total
                    if trade_type == 'covered_call':
                        if len(active_positions) >= COVER_CALL_MAX_POSITIONS:
                            counter_max = 0
                        else:
                            counter_max = 1
                    else: 
                        if required_margin < available_capital * 0.4:
                            # if vix_value > 18:
                            #     counter_max = num_full_closed/2 if num_full_closed > 0 else 1
                            # else:
                            #     counter_max = num_full_closed/2 if num_full_closed > 0 else 1

                            if vix_value > 18:
                                counter_max = 2
                            else:
                                counter_max = 1

                        else:
                            counter_max = 0
                    aggregated_call_loss_start = aggregated_call_loss
                    aggregated_put_loss_start = aggregated_put_loss
                    while ( aggregated_call_loss > aggregated_call_loss_start * 0.02 or aggregated_put_loss > aggregated_put_loss_start * 0.02 or counter < counter_max ) and required_margin < available_capital * 0.5:
                        if counter > 100:
                            break
                        # call_valid = counter < len(this_open_atm_call)
                        # put_valid  = counter < len(this_open_atm_put)

                        # # Exit loop if both are exhausted and we're not at the initial (counter == 0)
                        # if not (call_valid or put_valid or counter == 0):
                        #     break

                        # this_call_target = this_open_atm_call[counter] if call_valid else 0
                        # this_put_target  = this_open_atm_put[counter]  if put_valid  else 0

                        this_call_target = aggregated_call_loss if aggregated_call_loss > 0 else 0
                        this_put_target  = aggregated_put_loss  if aggregated_put_loss > 0 else 0
                        
                        active_positions_after_today = [pos for pos in active_positions if pos['expiration'] > current_day and pos['invalid_data'] is None]
                        if trade_type == 'covered_call' and (aggregated_call_loss > 0 or aggregated_put_loss > 0):
                            this_call_target = aggregated_call_loss / (COVER_CALL_MAX_POSITIONS - len(active_positions_after_today)) if COVER_CALL_MAX_POSITIONS > len(active_positions_after_today) else aggregated_call_loss
                            print(f"this_call_target: {this_call_target}, this_put_target: {this_put_target}, len(active_positions_after_today): {len(active_positions_after_today)}, COVER_CALL_MAX_POSITIONS: {COVER_CALL_MAX_POSITIONS}")
                        new_pos = await select_new_position_strikes(current_day, expire_day, this_call_target, this_put_target)
                        counter += 1

                        if new_pos:
                            if new_pos['short_call_prem_open'] - new_pos['long_call_prem_open'] < 0.05:
                                new_pos['short_call_prem_open'] = 0
                                new_pos['long_call_prem_open'] = 0
                                new_pos['required_margin'] -= 100 * (new_pos['call_strike_bought'] - new_pos['call_strike_sold']) if new_pos['call_strike_sold'] and new_pos['short_call_prem_open'] > 0 else 0
                            if new_pos['short_put_prem_open'] - new_pos['long_put_prem_open'] < 0.05:
                                new_pos['short_put_prem_open'] = 0
                                new_pos['long_put_prem_open'] = 0
                                new_pos['required_margin'] -= 100 * (new_pos['put_strike_sold'] - new_pos['put_strike_bought']) if new_pos['put_strike_sold'] and new_pos['short_put_prem_open'] > 0 else 0
                            active_positions.append(new_pos)
                            if new_pos['invalid_data'] is None:
                                initial_credit = (new_pos['short_call_prem_open'] + new_pos['short_put_prem_open'] - new_pos['long_call_prem_open'] - new_pos['long_put_prem_open']) * 100 - 4 * 0.5
                                if initial_credit < 0:
                                    print(f"Initial credit for {ticker} on {current_day.date()} is negative: ${initial_credit:.2f}, skipping position")
                                    # breakpoint()
                                realized_total += initial_credit
                                if mode == "validation":
                                    print(f"\033[38;5;208mOpened new position on {current_day.date()} expiring {new_pos['expiration'].date()} "
                                        f"(call {new_pos['call_strike_sold']}-{new_pos['call_strike_bought']}, "
                                        f"${(new_pos['short_call_prem_open'] - new_pos['long_call_prem_open']) * 100:.0f} "
                                        f"put {new_pos['put_strike_sold']}-{new_pos['put_strike_bought']} "
                                        f"${(new_pos['short_put_prem_open'] - new_pos['long_put_prem_open']) * 100:.0f} "
                                        f"close: {close_price:.2f}), vix: {vix_value:.1f}, initial credit: ${initial_credit:.2f}, realized total: ${realized_total:.0f}\033[0m")
                                    
                                required_margin += 100 * (new_pos['call_strike_bought'] - new_pos['call_strike_sold']) if new_pos['call_closed_by_stop'] is False and new_pos['call_strike_sold'] and new_pos['short_call_prem_open'] > 0 else 0
                                required_margin += 100 * (new_pos['put_strike_sold'] - new_pos['put_strike_bought']) if new_pos['put_closed_by_stop'] is False and new_pos['put_strike_sold'] and new_pos['short_put_prem_open'] > 0 else 0
                                aggregated_call_loss -= new_pos['short_call_prem_open'] - new_pos['long_call_prem_open']
                                aggregated_put_loss -= new_pos['short_put_prem_open'] - new_pos['long_put_prem_open']
                    
                    # if num_full_closed > 2:
                    #     print(f"num_full_closed: {num_full_closed}, aggregated_call_loss: {aggregated_call_loss}, aggregated_put_loss: {aggregated_put_loss}")
                    #     breakpoint()

            # if open_atm_call or open_atm_put:
            #     breakpoint()

        # Calculate unrealized PnL for each open position
        unrealized_total = 0.0
        for pos in active_positions:
            if pos['expiration'] <= current_day or pos['invalid_data'] is not None:
                continue
            if stop_profit_percent == 0: #only evaluate unrealized PnL if there is stop loss requirement
                continue
            pos_unreal = 0.0
            # Short call leg (if it wasn't closed by stop earlier)
            if pos['short_call_prem_open'] > 0 and not pos.get('call_closed_by_stop', False):
                pos_unreal += (- pos['short_call_prem_today']) * 100
            # Short put leg
            if pos['short_put_prem_open'] > 0 and not pos.get('put_closed_by_stop', False):
                pos_unreal += (- pos['short_put_prem_today']) * 100
            # Long call leg
            if pos.get('long_call_prem_open', 0) > 0 and not pos.get('call_closed_by_stop', False):
                pos_unreal += (pos['long_call_prem_today']) * 100
            # Long put leg
            if pos.get('long_put_prem_open', 0) > 0 and not pos.get('put_closed_by_stop', False):
                pos_unreal += (pos['long_put_prem_today']) * 100
            # Store current position value (mark-to-market) in the position dict
            pos['position_value'] = round(pos_unreal, 2)
            unrealized_total += pos_unreal
            # print(f"added position value for unrealized_total (${unrealized_total}): {pos['position_open_date'].date()}->{pos['expiration'].date()} {pos['call_strike_sold']}<->{pos['call_strike_bought']} {pos['put_strike_sold']}<->{pos['put_strike_bought']} {pos_unreal:.2f} {close_price:.2f} {vix_value:.1f}")
        if unrealized_total > 0:
            print(f"Unrealized PnL for {ticker} on {current_day.date()} close: {close_price:.2f} is unrealistically low: ${unrealized_total:.2f}")
            for pos in active_positions:
                if pos['expiration'] <= current_day or pos['invalid_data'] is not None:
                    continue
                if stop_profit_percent == 0: #only evaluate unrealized PnL if there is stop loss requirement
                    continue
                print(f"short put strike: {pos['put_strike_sold']}, long put strike: {pos['put_strike_bought']}, close price: {close_price:.2f}")
                print(f"short put premium: {pos['short_put_prem_today']}, long put premium: {pos['long_put_prem_today']}, close price: {close_price:.2f}")
            # breakpoint()
        # Calculate realized PnL for each closed position        
        # Update cumulative PnL (realized + unrealized) and record daily result
        total_pnl = realized_total + unrealized_total
        if len(daily_results) > 0: # drop the first 30 days
            annualized_gain = ( realized_total - daily_results[-1]['cumulative_pnl_realized'] if daily_results else 0.0 ) / required_margin * 252 * 100 if required_margin > 0 and daily_results[-1]['cumulative_pnl_realized'] != 0 else 0.0
        else:
            annualized_gain = 0
        active_positions = [pos for pos in active_positions if pos['expiration'].date() >= current_day.date()] #and pos['invalid_data'] is None and ( ( not pos['call_closed_by_stop'] and pos['short_call_prem_open'] > 0 ) or ( not pos['put_closed_by_stop'] and pos['short_put_prem_open'] > 0 ) ) ] # do not propogate the invalid positions into next day
        import copy

        daily_results.append({
            "date": current_day,
            "cumulative_pnl": round(total_pnl, 2),
            "cumulative_pnl_realized": round(realized_total, 2),
            "required_margin": round(required_margin, 2),
            "return_annualized": round(annualized_gain, 2),
            "active_positions": copy.deepcopy(active_positions),
        })

        daily_dates.append(current_day)
        if mode == "validation":
            print(f"{current_day.date()} | Close Price: {close_price:.2f} Realized PnL: {realized_total:.0f} (Total PnL: {total_pnl:.0f}, Unrealized: {unrealized_total:.0f})")
            # if total_pnl < -100000:
            #     breakpoint()
            # print(active_positions)
        # save_stored_option_data(ticker)
    return total_pnl, daily_results

def fetch_price_from_memory(
    ticker: str, 
    strike: float, 
    call_put: str, 
    expiration_str: str, 
    trade_date_str: str, 
    use_close_price: bool = False
) -> Dict[str, Any]:
    """
    Helper to retrieve the price data from stored_option_price if available.
    Returns an empty dict if not found.

    :param ticker: The underlying asset ticker symbol.
    :param strike: The strike price of the option.
    :param call_put: 'call' or 'put'.
    :param expiration_str: The expiration date in 'YYYY-MM-DD' format.
    :param trade_date_str: The pricing date in 'YYYY-MM-DD' format.
    :param use_close_price: Indicates whether to fetch close price data.
    :return: A dictionary with the relevant price data or empty dict if not found.
    """
    # Navigate through the nested dictionary to get the data
    data = (
        stored_option_price.get(ticker.upper(), {})
        .get(trade_date_str, {})
        .get(round(strike, 2), {})
        .get(expiration_str, {})
        .get(call_put.lower(), {})
    )
    
    if not data:
        logging.warning(f"No stored data found for {ticker}, Strike: {strike}, Type: {call_put}, "
                        f"Expiration: {expiration_str}, Pricing Date: {trade_date_str}.")
        return {}
    
    # Depending on the fetching method, return relevant fields
    if use_close_price:
        return {
            "close_price": data.get("close_price", 0.0),
            "close_volume": data.get("close_volume", 0.0)
        }
    else:
        return {
            "ask_price": data.get("ask_price", 0.0),
            "bid_price": data.get("bid_price", 0.0),
            "ask_size": data.get("ask_size", 0),
            "bid_size": data.get("bid_size", 0),
            "mid_price": data.get("mid_price", 0.0)
        }

# ---------------------------------------------------------
# 11. MONTHLY RECURSIVE BACKTEST FUNCTION
# ---------------------------------------------------------

def parameters_match(loaded_dict: Dict[str, Any], current_dict: Dict[str, Any]) -> bool:
    """
    Check if the loaded parameters match the current parameters.
    Return True if they match exactly, or if the current date range is within the loaded date range.
    """
    # Define keys to check other than the date range
    keys_to_check = ["ticker", "lookback_months", "hedge_values",
                     "multiplier_values", "target_price_baselines", "expiring_wks","target_premium"]
    
    # Check non-date parameters
    for k in keys_to_check:
        if loaded_dict.get(k) != current_dict.get(k):
            return False

    # Check date range containment
    loaded_start = datetime.strptime(loaded_dict.get("global_start_date", ""), "%Y-%m-%d")
    loaded_end = datetime.strptime(loaded_dict.get("global_end_date", ""), "%Y-%m-%d")
    current_start = datetime.strptime(current_dict.get("global_start_date", ""), "%Y-%m-%d")
    current_end = datetime.strptime(current_dict.get("global_end_date", ""), "%Y-%m-%d")

    # Return True if the current date range is within the loaded date range
    if loaded_start <= current_start and loaded_end >= current_end:
        return True

    return False

def analyze_trading_combinations(combo_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    combo_stats = []
    
    for combo in combo_results:
        # Get returns and filter out None values
        returns = [r for r in combo['return_annualized_array'] if r is not None]
        
        if returns:  # Only process if we have valid returns
            num_valid_points = len(returns)
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else float('inf')
            min_return = np.min(returns)
            baseline_return = 0
            score = (avg_return - baseline_return) / std_return if std_return > 0 else -1000
            
            # Calculate equity curve and maximum drawdown
            equity_curve = np.cumsum(returns)
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = peak - equity_curve
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # Access parameters from the 'parameters' dictionary
            params = combo['parameters']

            combo_stats.append({
                'expiring_wk': params['expiring_wks'],
                'target_premium_otm': params['target_premium_otm'],
                'target_steer': params['target_steer'],
                'iron_condor_width': params['iron_condor_width'],
                'stop_loss_action': params['stop_loss_action'],
                'stop_profit_percent': params['stop_profit_percent'],
                'day_of_week': params['day_of_week'],
                'num_valid_points': num_valid_points,
                'avg_return': avg_return,
                'std_return': std_return,
                'score': score,
                'final_pnl': combo['final_pnl'],
                'sharpe': score,  # Using our calculated score as the Sharpe ratio
                'start_date': combo['daily_dates'][0],
                'end_date': combo['daily_dates'][-1],
                'target_delta': params['target_delta'],
                'vix_correlation': params['vix_correlation'],
                'vix_threshold': params['vix_threshold'],
                'trade_type': params['trade_type'],
                'max_drawdown': max_drawdown,
            })

    if not combo_stats:
        return None

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(combo_stats)
    
    # Select top 5 combos by score
    top_combos = results_df.sort_values('score', ascending=False).head(5)
    
    # Among the top 5, select the one with the lowest max_drawdown
    best_combo = top_combos.sort_values('max_drawdown', ascending=True).iloc[0]
    
    # Calculate median expiring_wk from top 5 combos by score
    median_expiring_wk = np.median(top_combos['expiring_wk'])
    
    # Return dictionary with best combo parameters and appended median values
    return {
        'parameters': {
            'expiring_wks': best_combo['expiring_wk'],
            'target_premium_otm': best_combo['target_premium_otm'],
            'target_steer': best_combo['target_steer'],
            'iron_condor_width': best_combo['iron_condor_width'],
            'stop_loss_action': best_combo['stop_loss_action'],
            'stop_profit_percent': best_combo['stop_profit_percent'],
            'day_of_week': best_combo['day_of_week'],
            'target_delta': best_combo['target_delta'],
            'vix_correlation': best_combo['vix_correlation'],
            'vix_threshold': best_combo['vix_threshold'],
            'trade_type': best_combo['trade_type'],
        },
        'sharpe': best_combo['sharpe'],
        'median_expiring_wk': median_expiring_wk,
    }

async def monthly_recursive_backtest(
    ticker: str,
    global_start_date: str,
    global_end_date: str,
    lookback_months: int,
    lookforward_months: int,
    trade_parameters: List[Dict[str, Any]],  # List of trade parameter dictionaries
    client: PolygonAPIClient,
    save_file: bool = False,
    trade_type: str = "iron_condor",
    input_df: Dict[str, pd.DataFrame] = None,
) -> Tuple[float, List[datetime], List[float], List[Dict[str, Any]], List[float], List[Dict[str, Any]]]:
    global new_data_entry_count
    # Convert input strings to datetime for slicing
    start_dt = datetime.strptime(global_start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(global_end_date, "%Y-%m-%d")

    # Prepare to store the "continuous" weekly results
    global_daily_dates = []
    global_daily_pnls = []
    global_daily_results = []
    global_cumulative_pnls = []
    global_cumulative_pnls_realized = []
    parameter_history = []  # To track when parameters change
    cumulative_pnl = 0.0
    cumulative_pnl_realized = 0.0

    # Helper to get the first day of each month
    def month_starts_between(start_dt, end_dt):
        """
        Returns a list of (year, month, day=1) datetimes for each month in [start_dt, end_dt].
        """
        dates = []
        current = datetime(start_dt.year, start_dt.month, 1)
        while current <= end_dt:
            dates.append(current)
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        return dates

    all_month_starts = month_starts_between(start_dt, end_dt)
    if not all_month_starts:
        print("No months found in the specified range!")
        return 0.0, [], [], []

    carry_over_weekly_results = []
    for i in range(0, len(all_month_starts), lookforward_months):
        trade_month_start = all_month_starts[i]
        if i <= len(all_month_starts) - 1 - lookforward_months:
            next_month_start = all_month_starts[i + lookforward_months]
        else:
            print(f"Last month to validate: {trade_month_start.strftime('%Y-%m')}")
            next_month_start = end_dt + timedelta(days=1)
        if trade_month_start < start_dt:
            if next_month_start <= start_dt:
                continue
            trade_month_start = start_dt
        if next_month_start > end_dt:
            if trade_month_start >= end_dt:
                break
            trade_month_end = end_dt
        else:
            trade_month_end = next_month_start - timedelta(days=1)
            if trade_month_end > end_dt:
                trade_month_end = end_dt

        training_end = trade_month_start - timedelta(days=1)
        training_start = training_end - timedelta(days=30 * lookback_months)

        if training_start < start_dt and lookback_months > 0:
            print(f"Training start date {training_start.strftime('%Y-%m-%d')} is before global start date {start_dt.strftime('%Y-%m-%d')}, skipping this month.")
            continue

        if training_end <= training_start and lookback_months > 0:
            print(f"Training end date {training_end.strftime('%Y-%m-%d')} is before training start date {training_start.strftime('%Y-%m-%d')}, skipping this month.")
            continue

        print(f"Training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}, trade period: {trade_month_start.strftime('%Y-%m-%d')} to {trade_month_end.strftime('%Y-%m-%d')}")

        combo_results = []
        if len(trade_parameters) > 1:
            for params in trade_parameters:
                final_pnl, daily_results = await backtest_options_sync_or_async(
                    start_date=training_start.strftime("%Y-%m-%d"),
                    end_date=training_end.strftime("%Y-%m-%d"),
                    ticker=ticker,
                    df_dict=input_df,
                    trade_parameter=params,  # Pass the entire params dictionary
                    client=client,
                    use_async=True,
                    carry_over_weekly_results=None,
                    mode="training",
                    trade_type=trade_type,
                )
                if final_pnl is None and daily_results is None:
                    print(f"{ticker} Training skipped: {training_start.strftime('%Y-%m-%d')} ~ {training_end.strftime('%Y-%m-%d')} params: {params}")
                    continue
                combo_results.append({
                    "parameters": params,
                    "final_pnl": final_pnl,
                    "daily_dates": [row["date"] for row in daily_results],
                    "cumulative_pnl": [row["cumulative_pnl"] for row in daily_results],
                    "cumulative_pnl_realized": [row["cumulative_pnl_realized"] for row in daily_results],
                    "return_annualized_array": [row["return_annualized"] for row in daily_results],
                    "required_margin_array": [row["required_margin"] for row in daily_results],
                    "daily_results": daily_results
                })
                print(f"{ticker} Training completed: {training_start.strftime('%Y-%m-%d')}~{training_end.strftime('%Y-%m-%d')} params: {params}")
                if new_data_entry_count > 1000:
                    save_stored_option_data(ticker)

            # ------------------------------
            # Plotting the Cumulative PnL for All Runs
            # ------------------------------
            PLOT_TRAINING_RESULT = True
            if PLOT_TRAINING_RESULT:
                if combo_results:
                    # Compute metrics for each combination
                    combo_metrics = []
                    for combo in combo_results:
                        metrics = compute_metrics_from_daily_results(combo['daily_results'])
                        combo_metrics.append({
                            'parameters': combo['parameters'],
                            'final_pnl': combo['final_pnl'],
                            **metrics
                        })

                    # Subplot 1: Realized Gain vs. Date for All Combinations
                    plt.figure(figsize=(18, 6))
                    for combo in combo_results:
                        dates = combo['daily_dates']
                        pnl_realized = combo['cumulative_pnl_realized']
                        plt.plot(dates, pnl_realized, label=str(combo['parameters']))

                    plt.title(f'Realized Gain vs. Date for All Parameter Combinations ({ticker}, {training_start.strftime("%Y-%m-%d")} to {training_end.strftime("%Y-%m-%d")})')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Realized PnL')

                    # Legend: vertical list at the top, centered
                    plt.legend(
                        loc='lower center',         # anchor to lower center of the legend box
                        bbox_to_anchor=(0.5, 1.02),  # place above the plot
                        frameon=True,
                        ncol=1                      # 1 column = vertical stacking
                    )

                    plt.grid(True)
                    plt.gcf().set_constrained_layout(True)
                    plt.savefig(f'training_results/training_realized_gain_{ticker}_{training_start.strftime("%Y%m%d")}_{training_end.strftime("%Y%m%d")}.png')
                    plt.close()


                    # Table of Metrics
                    metrics_df = pd.DataFrame(combo_metrics)
                    params_df = pd.DataFrame([combo['parameters'] for combo in combo_results])
                    metrics_df = pd.concat([params_df, metrics_df.drop('parameters', axis=1)], axis=1)
                    metrics_df = metrics_df.sort_values('final_pnl', ascending=False)
                    print(f"\nTraining Metrics for {ticker} ({training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}):")
                    print(metrics_df.to_string(index=False))
                    metrics_df.to_csv(f'training_results/training_metrics_{ticker}_{training_start.strftime("%Y%m%d")}_{training_end.strftime("%Y%m%d")}.csv', index=False)

            if not combo_results:
                continue

            best_combo = analyze_trading_combinations(combo_results)

            if best_combo is None:
                print(f"No valid trading combinations found for month {trade_month_start.strftime('%Y-%m')}, skip trading this month.")
                continue
            best_params = best_combo["parameters"]

            print(f"Selected best combo for month {trade_month_start.strftime('%Y-%m')} => params: {best_params}")
        else:
            print(f"There is only 1 trade parameters combo")
            best_params = trade_parameters[0]

        parameter_history.append({
            "month": trade_month_start.strftime("%Y-%m"),
            "start_date": trade_month_start.strftime("%Y-%m-%d"),
            "end_date": trade_month_end.strftime("%Y-%m-%d"),
            **best_params
        })

        if trade_month_end < trade_month_start:
            continue
        
        if global_daily_results:
            trade_month_start = global_daily_results[-1]["date"] + timedelta(days=1)
            print(f"trade_month_start: {trade_month_start} last traded day: {global_daily_results[-1]['date'] if global_daily_results else 'N/A'}")

        print(f"Running best combo for month {trade_month_start.strftime('%Y-%m-%d')} to {trade_month_end.strftime('%Y-%m-%d')} => params: {best_params}")

        final_pnl_m, daily_results_m = await backtest_options_sync_or_async(
            start_date=trade_month_start.strftime("%Y-%m-%d"),
            end_date=trade_month_end.strftime("%Y-%m-%d"),
            ticker=ticker,
            df_dict=input_df,
            trade_parameter=best_params,  # Pass the entire best_params dictionary
            client=client,
            use_async=True,
            carry_over_weekly_results=carry_over_weekly_results,
            mode="validation",
            trade_type=trade_type,
        )
        if new_data_entry_count > 1000:
            save_stored_option_data(ticker)

        if daily_results_m is None:
            continue

        validation_cumulative_pnl = [row["cumulative_pnl"] for row in daily_results_m]
        validation_cumulative_pnl_realized = [row["cumulative_pnl_realized"] for row in daily_results_m]
        validation_dates = [row["date"] for row in daily_results_m]
        carry_over_weekly_results = daily_results_m

        if validation_cumulative_pnl and validation_dates:
            # Handle overlapping dates first
            if global_daily_dates:
                if validation_dates[0] <= global_daily_dates[-1]:
                    overlap_idx = validation_dates.index(global_daily_dates[-1])
                    validation_dates = validation_dates[overlap_idx + 1:]
                    validation_cumulative_pnl = validation_cumulative_pnl[overlap_idx + 1:]
                    validation_cumulative_pnl_realized = validation_cumulative_pnl_realized[overlap_idx + 1:]

            # Calculate offset from previous month’s final cumulative PnL
            cumulative_offset = global_cumulative_pnls_realized[-1] if global_cumulative_pnls_realized else 0.0
            cumulative_offset_realized = global_cumulative_pnls_realized[-1] if global_cumulative_pnls_realized else 0.0

            # Adjust cumulative pnl for continuity
            adjusted_cumulative_pnls = [pnl + cumulative_offset for pnl in validation_cumulative_pnl]
            adjusted_cumulative_pnls_realized = [pnl + cumulative_offset_realized for pnl in validation_cumulative_pnl_realized]

            # Now extend global lists with properly aligned data
            global_daily_dates.extend(validation_dates)
            global_cumulative_pnls.extend(adjusted_cumulative_pnls)
            global_cumulative_pnls_realized.extend(adjusted_cumulative_pnls_realized)
            global_daily_results.extend(daily_results_m[-len(validation_dates):])

        else:
            print("No data for this month. Skipping accumulation.")

    final_val = global_cumulative_pnls[-1]

    save_file = False
    if save_file:
        filename = get_monthly_backtest_file(ticker, global_start_date, global_end_date)
        data_to_save = {
            "parameters": current_parameters,
            "final_pnl": final_val,
            "dt_series": global_daily_dates,
            "pnl_series": global_weekly_pnls,
            "pnl_cumulative_series": global_cumulative_pnls,
            "pnl_cumulative_realized_series": global_cumulative_pnls_realized,
            "parameter_history": parameter_history,
            "weekly_results": global_daily_results
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Saved monthly_recursive_backtest results to {filename}")
        
    return final_val, global_daily_dates, global_cumulative_pnls, parameter_history, global_daily_pnls,global_daily_results, global_cumulative_pnls_realized

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
from datetime import datetime
import os
import matplotlib.dates as mdates

def plot_recursive_results(
    ticker,
    final_pnl,
    daily_results,
    pnl_cumulative_series,
    pnl_cumulative_realized_series,
    parameter_history,
    global_start_date,
    global_end_date,
    df_dict,
):
    _SCALE = 2
    _size_keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ]
    new_params = {}
    for k in _size_keys:
        v = plt.rcParams[k]
        if isinstance(v, (int, float)):
            new_params[k] = v * _SCALE
        else:  # string such as "large"
            new_params[k] = FontProperties(size=v).get_size_in_points() * _SCALE
    plt.rcParams.update(new_params)

    # helper for explicit numeric literals later
    def _fs(x):
        return x * _SCALE

    # ── Performance metrics ─────────────────────────────────────────────────
    daily_realized = np.diff(pnl_cumulative_realized_series, prepend=0)
    mean_realized = np.mean(daily_realized)
    std_realized = np.std(daily_realized, ddof=1) if len(daily_realized) > 1 else np.nan
    sharpe_ratio = (
        mean_realized / std_realized * np.sqrt(252)
        if std_realized not in (0, np.nan)
        else np.nan
    )

    daily_instantaneous = np.diff(pnl_cumulative_series, prepend=0)
    mean_instantaneous = np.mean(daily_instantaneous)
    std_instantaneous = (
        np.std(daily_instantaneous, ddof=1) if len(daily_instantaneous) > 1 else np.nan
    )
    sharpe_ratio_instantaneous = (
        mean_instantaneous / std_instantaneous * np.sqrt(252)
        if std_instantaneous not in (0, np.nan)
        else np.nan
    )

    running_max = np.maximum.accumulate(pnl_cumulative_series)
    drawdowns = running_max - pnl_cumulative_series
    max_drawdown = drawdowns.max()

    margins_all = [
        row.get("required_margin", 1e8)
        for row in daily_results
        if row.get("required_margin") is not None
    ]
    avg_margin_overall = np.mean(margins_all) if margins_all else np.nan
    cum_return_overall = (
        pnl_cumulative_realized_series[-1] / avg_margin_overall
        if avg_margin_overall not in (0, np.nan)
        else np.nan
    )

    print(f"Recursive monthly approach => Final PnL for {ticker}: {final_pnl:.2f}")
    print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print(
        f"Cumulative Return: {cum_return_overall:.2%}, "
        f"Annual return: {cum_return_overall/len(pnl_cumulative_realized_series)*252:.2%}"
    )

    # ── Cumulative-annualized-return series with max margin ─────────────────
    required_margins = [
        row["required_margin"]
        for row in daily_results
        if row.get("required_margin") is not None
        and isinstance(row["required_margin"], (int, float))
    ]
    required_margins_series = pd.Series(required_margins, dtype="float64")
    cumulative_max_margin = (
        required_margins_series.expanding().max()
        if not required_margins_series.empty
        else pd.Series([0])
    )

    pnl_series = pd.Series(pnl_cumulative_realized_series, dtype="float64")
    day_count = pd.Series(np.arange(1, len(pnl_series) + 1), dtype="float64")

    cumulative_ann_return = (
        pnl_series.div(cumulative_max_margin)
        .div(day_count)
        .mul(252)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    cumulative_ann_return[:100] = 0.0
    if not cumulative_ann_return.empty:
        print(
            f"Final cumulative annualized return: "
            f"{cumulative_ann_return.iloc[-1] * 100:.2f}%"
        )

    # ── Invalid-data statistics ─────────────────────────────────────────────
    invalid_counter = Counter()
    total_position_count = 0
    for daily_result in daily_results:
        current_date = pd.to_datetime(daily_result.get("date")).date()
        for pos in daily_result["active_positions"]:
            try:
                pos_open_date = pd.to_datetime(pos.get("position_open_date")).date()
            except Exception:
                continue
            if pos_open_date == current_date:
                total_position_count += 1
                inv = pos.get("invalid_data")
                if inv is not None:
                    invalid_counter[inv] += 1

    if total_position_count == 0:
        print("No positions opened to check for invalid data.")
    else:
        print(f"Total positions opened: {total_position_count}\n")
        for inv_type, count in invalid_counter.items():
            pct = count / total_position_count * 100
            print(f"  • Invalid type '{inv_type}': {count} ({pct:.2f}%)")
        valid_count = total_position_count - sum(invalid_counter.values())
        print(
            f"  • Valid positions       : {valid_count} "
            f"({valid_count/total_position_count*100:.2f}%)"
        )

    # ── Unpack dataframes & date series ─────────────────────────────────────
    try:
        df = df_dict["df"]
        vix_df = df_dict["vix_df"]
        dt_series = [pd.to_datetime(row["date"]) for row in daily_results]
        print(f"dt_series length: {len(dt_series)}")
        print(f"Sample dt_series: {dt_series[:2] if dt_series else 'Empty'}")
    except KeyError as e:
        print(f"Error unpacking data: {e}")
        dt_series = []
        df = pd.DataFrame()
        vix_df = pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    # SPEED BOOST: one-shot close-price lookup
    close_price_lookup = df["close"].to_dict()
    get_close = close_price_lookup.get

    required_margins_for_plot = [
        row["required_margin"]
        for row in daily_results
        if row.get("required_margin") is not None
        and isinstance(row["required_margin"], (int, float))
    ]

    # ── Collect individual option distances ─────────────────────────────────
    call_dates, call_instance_distances, put_dates, put_instance_distances = (
        [],
        [],
        [],
        [],
    )
    for date, day in zip(dt_series, daily_results):
        close_price = get_close(date)
        if np.isnan(close_price):
            continue
        for pos in day.get("active_positions", []):
            if pos.get("invalid_data") is not None or pos["expiration"].date() <= date.date():
                continue
            if (
                pos["short_call_prem_open"] > 0
                and pos["call_closed_date"] is None
                and isinstance(pos.get("call_strike_sold"), (int, float))
            ):
                call_dates.append(date)
                call_instance_distances.append(
                    (pos["call_strike_sold"] - close_price) / close_price * 100
                )
            if (
                pos["short_put_prem_open"] > 0
                and pos["put_closed_date"] is None
                and isinstance(pos.get("put_strike_sold"), (int, float))
            ):
                put_dates.append(date)
                put_instance_distances.append(
                    -((pos["put_strike_sold"] - close_price) / close_price) * 100
                )

    # ── Compute closest strike distances, ITM amounts, etc. ─────────────────
    min_distances, highlight_dates, highlight_distances, itm_amounts = [], [], [], []
    for day, date in zip(daily_results, dt_series):
        close_price = get_close(date)
        distances, itm_amount = [], 0
        has_itm_friday_expiry = False

        active_positions = [
            pos
            for pos in day.get("active_positions", [])
            if pos["invalid_data"] is None
            and pos["expiration"].date() >= date.date()
            and (
                (not pos["call_closed_by_stop"] and pos["short_call_prem_open"] > 0)
                or (not pos["put_closed_by_stop"] and pos["short_put_prem_open"] > 0)
            )
        ]

        for pos in active_positions:
            exp_date = pd.to_datetime(pos.get("expiration")).date()
            current_date = date.date()
            is_today_expiry = exp_date == current_date

            if pos["short_call_prem_open"] > 0 and not pos["call_closed_by_stop"]:
                strike = pos["call_strike_sold"]
                dist = (strike - close_price) / close_price * 100
                distances.append(dist)
                if dist < 0 and is_today_expiry:
                    has_itm_friday_expiry = True
                    itm_amount += max(close_price - strike, 0) * 100
            if pos["long_call_prem_open"] > 0 and not pos["call_closed_by_stop"]:
                strike = pos["call_strike_bought"]
                dist = (strike - close_price) / close_price * 100
                distances.append(dist)
                if dist < 0 and is_today_expiry:
                    has_itm_friday_expiry = True
                    itm_amount -= max(close_price - strike, 0) * 100
            if pos["short_put_prem_open"] > 0 and not pos["put_closed_by_stop"]:
                strike = pos["put_strike_sold"]
                dist = -((strike - close_price) / close_price) * 100
                distances.append(dist)
                if dist < 0 and is_today_expiry:  # Fixed: Changed upcoming_expiry to is_today_expiry
                    has_itm_friday_expiry = True
                    itm_amount += max(strike - close_price, 0) * 100
            if pos["long_put_prem_open"] > 0 and not pos["put_closed_by_stop"]:
                strike = pos["put_strike_bought"]
                dist = -((strike - close_price) / close_price) * 100
                distances.append(dist)
                if dist < 0 and is_today_expiry:
                    has_itm_friday_expiry = True
                    itm_amount -= max(strike - close_price, 0) * 100

        if distances and not np.isnan(close_price):
            md = min(distances, key=abs)
            min_distances.append(md)
            if has_itm_friday_expiry and md < 0:
                highlight_dates.append(date)
                highlight_distances.append(md)
        else:
            min_distances.append(np.nan)
        itm_amounts.append(itm_amount if itm_amount != 0 else np.nan)

    # ── Premium scatter extraction ──────────────────────────────────────────
    open_distance_calls, open_premium_calls, target_premium_calls = [], [], []
    open_distance_puts, open_premium_puts, target_premium_puts = [], [], []
    days_to_expiry_array, spread_width_array = [], []
    call_closed_profit_array, put_closed_profit_array = [], []
    for day in daily_results:
        call_distances, call_prem, call_tgt = [], [], []
        put_distances, put_prem, put_tgt = [], [], []
        days_to_expiry, spread_width = [], []
        call_closed_profit, put_closed_profit = [], []

        for pos in day.get("active_positions", []):
            same_open_day = pos["position_open_date"] == day["date"]
            same_closed_day = pos["call_closed_date"] == day["date"] or pos["put_closed_date"] == day["date"]
            if (
                pos["open_distance_call"] is not None
                and pos["short_call_prem_open"] > 0
                and same_open_day
            ):
                call_distances.append(pos["open_distance_call"] * 100)
                call_prem.append(
                    pos["short_call_prem_open"] - pos["long_call_prem_open"]
                )
                call_tgt.append(pos["strike_target_call"]["premium_target"])
                days_to_expiry.append(pos["expiration"] - pos["position_open_date"])
                spread_width.append(pos["call_strike_bought"] - pos["call_strike_sold"])
            if (
                pos["open_distance_put"] is not None
                and pos["short_put_prem_open"] > 0
                and same_open_day
            ):
                put_distances.append(pos["open_distance_put"] * 100)
                put_prem.append(
                    pos["short_put_prem_open"] - pos["long_put_prem_open"]
                )
                put_tgt.append(pos["strike_target_put"]["premium_target"])
                days_to_expiry.append(pos["expiration"] - pos["position_open_date"])
                spread_width.append(pos["put_strike_sold"] - pos["put_strike_bought"])
            if (
                pos["short_call_prem_open"] > 0
                and pos["call_closed_date"] is not None
                and pos["call_closed_date"].date() == day["date"].date()
            ):
                call_closed_profit.append(
                    pos["call_closed_profit"]
                )

            if (
                pos['short_put_prem_open'] > 0
                and pos['put_closed_date'] is not None
                and pos['put_closed_date'].date() == day['date'].date()
            ):
                put_closed_profit.append(
                    pos["put_closed_profit"]
                )
        open_distance_calls.append(call_distances)
        open_distance_puts.append(put_distances)
        open_premium_calls.append(call_prem)
        open_premium_puts.append(put_prem)
        target_premium_calls.append(call_tgt)
        target_premium_puts.append(put_tgt)
        days_to_expiry_array.append(days_to_expiry)
        spread_width_array.append(spread_width)
        call_closed_profit_array.append(call_closed_profit)
        put_closed_profit_array.append(put_closed_profit)

    # ── Days-open per moneyness state ───────────────────────────────────────
    itm_dates, itm_days_open, otm_dates, otm_days_open = [], [], [], []
    for date, day in zip(dt_series, daily_results):
        current_date = date.date()
        close_price = get_close(date)
        if np.isnan(close_price):
            continue
        for pos in day.get("active_positions", []):
            if (
                (pos["short_call_prem_open"] == 0 or pos["call_closed_date"] is not None)
                and (
                    pos["short_put_prem_open"] == 0
                    or pos["put_closed_date"] is not None
                )
            ):
                continue
            open_date = pd.to_datetime(pos.get("position_open_date")).date()
            if open_date is None:
                continue
            days_open = (current_date - open_date).days
            is_itm = False

            if (
                pos["short_call_prem_open"] > 0
                and pos["call_closed_date"] is None
                and isinstance(pos.get("call_strike_sold"), (int, float))
            ):
                is_itm |= (
                    (pos["call_strike_sold"] - close_price) / close_price * 100 < 0
                )
            if (
                pos["long_call_prem_open"] > 0
                and pos["call_closed_date"] is None
                and isinstance(pos.get("call_strike_bought"), (int, float))
            ):
                is_itm |= (
                    (pos["call_strike_bought"] - close_price) / close_price * 100 < 0
                )
            if (
                pos["short_put_prem_open"] > 0
                and pos["put_closed_date"] is None
                and isinstance(pos.get("put_strike_sold"), (int, float))
            ):
                is_itm |= (
                    -((pos["put_strike_sold"] - close_price) / close_price) * 100 < 0
                )
            if (
                pos["long_put_prem_open"] > 0
                and pos["put_closed_date"] is None
                and isinstance(pos.get("put_strike_bought"), (int, float))
            ):
                is_itm |= (
                    -((pos["put_strike_bought"] - close_price) / close_price) * 100 < 0
                )

            if is_itm:
                itm_dates.append(date)
                itm_days_open.append(days_open)
            else:
                otm_dates.append(date)
                otm_days_open.append(days_open)

    # ── Position-type counts ────────────────────────────────────────────────
    otm_open_counts = [0] * len(dt_series)
    itm_now_otm_open_counts = [0] * len(dt_series)
    itm_open_counts = [0] * len(dt_series)
    for i, (date, day) in enumerate(zip(dt_series, daily_results)):
        close_price = get_close(date)
        if np.isnan(close_price):
            continue
        for pos in day.get("active_positions", []):
            if pos.get("invalid_data") is not None:
                continue
            if (
                pos["short_call_prem_open"] > 0
                and pos["call_closed_date"] is None
                and isinstance(pos.get("call_strike_sold"), (int, float))
            ):
                strike = pos["call_strike_sold"]
                open_dist = pos.get("open_distance_call", 0) * 100
                current_dist = (strike - close_price) / close_price * 100
                if open_dist > 0:
                    (otm_open_counts if current_dist >= 0 else itm_now_otm_open_counts)[
                        i
                    ] += 1
                elif open_dist < 0:
                    itm_open_counts[i] += 1
            if (
                pos["short_put_prem_open"] > 0
                and pos["put_closed_date"] is None
                and isinstance(pos.get("put_strike_sold"), (int, float))
            ):
                strike = pos["put_strike_sold"]
                open_dist = pos.get("open_distance_put", 0) * 100
                current_dist = -((strike - close_price) / close_price) * 100
                if open_dist > 0:
                    (otm_open_counts if current_dist >= 0 else itm_now_otm_open_counts)[
                        i
                    ] += 1
                elif open_dist < 0:
                    itm_open_counts[i] += 1

    # ── Extract IV data from positions ──────────────────────────────────────
    iv_call_data, iv_put_data = [], []
    for date, day in zip(dt_series, daily_results):
        call_ivs = [
            pos["iv_call"]
            for pos in day.get("active_positions", [])
            if pos["position_open_date"] == date
            and isinstance(pos.get("iv_call"), (int, float))
        ]
        put_ivs = [
            pos["iv_put"]
            for pos in day.get("active_positions", [])
            if pos["position_open_date"] == date
            and isinstance(pos.get("iv_put"), (int, float))
        ]
        iv_call_data.append(np.mean(call_ivs) if call_ivs else np.nan)
        iv_put_data.append(np.mean(put_ivs) if put_ivs else np.nan)

    # ============================== PLOTTING ================================
    fig, axes = plt.subplots(4, 2, figsize=(70, 40))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

    # ---- ax1: cumulative PnL & cumulative annualized return ----------------
    if dt_series and len(pnl_cumulative_series) > 0:
        ax1.plot(
            dt_series,
            pnl_cumulative_series,
            label=f"Cumulative PnL (Mark-to-Market) (Final={pnl_cumulative_series[-1]:.2f})",
            color="black",
            marker=".",
            linestyle="-",
            linewidth=1,
            markersize=3,
        )
        ax1.plot(
            dt_series,
            pnl_cumulative_realized_series,
            label=(
                f"Cumulative PnL Realized "
                f"(Final={pnl_cumulative_realized_series[-1]:.2f}, "
                f"Sharpe realized={sharpe_ratio:.2f},instanceous={sharpe_ratio_instantaneous:.2f}, "
                f"MaxDD={max_drawdown:.2f})"
            ),
            color="blue",
            marker=".",
            linestyle="-",
            linewidth=1,
            markersize=3,
        )
        ax1.set_ylabel("PnL ($)", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.legend(loc="upper left", fontsize=_fs(8))

        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            dt_series,
            cumulative_ann_return * 100,
            label="Cumulative Annualized Return",
            color="green",
            linewidth=1,
            marker=".",
            markersize=3,
        )
        ax1_twin.set_ylabel("Cumulative Annualized Return (%)", color="green")
        ax1_twin.tick_params(axis="y", labelcolor="green")
        ax1_twin.legend(loc="upper right", fontsize=_fs(8))

        ax1.set_title(f"Cumulative PnL & Cumulative Annualized Return for {ticker}")
        ax1.set_xlabel("Date")
        ax1.grid(True)
    else:
        ax1.text(0.5, 0.5, "No PnL data to plot", transform=ax1.transAxes, ha="center")

    # ---- ax2: closest strike distance & ITM amount -------------------------
    if dt_series and min_distances and any(not np.isnan(d) for d in min_distances):
        ax2.plot(
            dt_series,
            min_distances,
            color="green",
            label="Min Distance to Open Options (OTM:+ / ITM:-)",
            linewidth=1,
            marker="o",
            markersize=1,
        )
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        if highlight_dates:
            ax2.scatter(
                highlight_dates,
                highlight_distances,
                color="red",
                marker="o",
                s=30,
                label="ITM Options Expiring Friday",
                rasterized=True,
            )
        if call_dates:
            ax2.scatter(
                call_dates,
                call_instance_distances,
                color="red",
                s=2,
                alpha=0.3,
                zorder=1,
                rasterized=True,
            )
        if put_dates:
            ax2.scatter(
                put_dates,
                put_instance_distances,
                color="blue",
                s=2,
                alpha=0.3,
                zorder=1,
                rasterized=True,
            )
        # Vectorised open-distance scatters
        def _flatten(dates, lists):
            d, v = [], []
            for dt, lst in zip(dates, lists):
                d.extend([dt] * len(lst))
                v.extend(lst)
            return d, v

        d_call, v_call = _flatten(dt_series, open_distance_calls)
        d_put, v_put = _flatten(dt_series, open_distance_puts)

        if d_call:
            ax2.scatter(
                d_call,
                v_call,
                color="orange",
                marker="^",
                s=30,
                alpha=0.6,
                label="Call Distances",
                rasterized=True,
            )
        if d_put:
            ax2.scatter(
                d_put,
                v_put,
                color="purple",
                marker="v",
                s=30,
                alpha=0.6,
                label="Put Distances",
                rasterized=True,
            )

        ax2.set_title(f"Closest Option-Strike Distance & ITM Amount for {ticker}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Distance (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.grid(True)
        ax2.legend(loc="upper left")

        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            dt_series,
            itm_amounts,
            color="red",
            label="Aggregate ITM Amount (Today Exp.)",
            linewidth=1,
            marker="o",
            markersize=10,
        )
        ax2_twin.set_ylabel("ITM Amount ($)", color="blue")
        ax2_twin.tick_params(axis="y", labelcolor="blue")
        ax2_twin.legend(loc="upper right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No open option distances to plot",
            transform=ax2.transAxes,
            ha="center",
        )

    # ---- ax3: underlying price + option strikes ----------------------------
    sb_x, sb_y, pb_x, pb_y = [], [], [], []
    call_dates_close, call_strikes, put_dates_close, put_strikes = [], [], [], []
    call_open_close_strikes, put_open_close_strikes = [], []

    unique_positions = {
        (
            pos.get("position_open_date"),
            pos.get("call_closed_date"),
            pos.get("put_closed_date"),
        ): pos
        for day in daily_results
        for pos in day.get("active_positions", [])
        if pos.get("expiration") is not None
    }

    for pos in unique_positions.values():
        open_date = pos.get("position_open_date")
        if open_date is None:
            continue
        if pos.get("call_closed_date") is not None:
            if pos.get("long_call_prem_open", 0) > 0 and pos.get("call_strike_bought"):
                sb_x.append(pos["call_closed_date"])
                sb_y.append(pos["call_strike_bought"])
            if pos.get("short_call_prem_open", 0) > 0 and pos.get("call_strike_sold"):
                call_dates_close.append(pos["call_closed_date"])
                call_strikes.append(pos["call_strike_sold"])
                call_open_close_strikes.append(
                    (open_date, pos["call_closed_date"], pos["call_strike_sold"])
                )
        if pos.get("put_closed_date") is not None:
            if pos.get("long_put_prem_open", 0) > 0 and pos.get("put_strike_bought"):
                pb_x.append(pos["put_closed_date"])
                pb_y.append(pos["put_strike_bought"])
            if pos.get("short_put_prem_open", 0) > 0 and pos.get("put_strike_sold"):
                put_dates_close.append(pos["put_closed_date"])
                put_strikes.append(pos["put_strike_sold"])
                put_open_close_strikes.append(
                    (open_date, pos["put_closed_date"], pos["put_strike_sold"])
                )

    all_plot_dates = (
        dt_series + call_dates_close + put_dates_close + sb_x + pb_x
    )
    if all_plot_dates:
        start_plot = min(d for d in all_plot_dates if d is not None)
        end_plot = max(d for d in all_plot_dates if d is not None)
        daily_close_filtered = df["close"].loc[start_plot:end_plot]
        if not daily_close_filtered.empty:
            ax3.plot(
                daily_close_filtered.index,
                daily_close_filtered.values,
                color="black",
                label="Underlying Close",
                linewidth=1,
            )
            for open_date, close_date, strike in call_open_close_strikes:
                ax3.plot(
                    [open_date, close_date], [strike, strike],
                    color="red",
                    linestyle="--",
                    linewidth=0.5,
                )
            for open_date, close_date, strike in put_open_close_strikes:
                ax3.plot(
                    [open_date, close_date], [strike, strike],
                    color="blue",
                    linestyle="--",
                    linewidth=0.5,
                )

            ax3.scatter(
                [o for o, _, _ in call_open_close_strikes],
                [s for _, _, s in call_open_close_strikes],
                color="green",
                marker="o",
                s=10,
                label="Short Call Open",
                rasterized=True,
            )
            ax3.scatter(
                [o for o, _, _ in put_open_close_strikes],
                [s for _, _, s in put_open_close_strikes],
                color="red",
                marker="o",
                s=10,
                label="Short Put Open",
                rasterized=True,
            )
            ax3.scatter(
                call_dates_close,
                call_strikes,
                color="purple",
                marker="s",
                s=10,
                label="Short Call Close",
                rasterized=True,
            )
            ax3.scatter(
                put_dates_close,
                put_strikes,
                color="blue",
                marker="s",
                s=10,
                label="Short Put Close",
                rasterized=True,
            )
            ax3.scatter(
                sb_x,
                sb_y,
                color="green",
                marker="^",
                s=5,
                label="Long Call Close",
                rasterized=True,
            )

            ax3.legend(loc="upper left", fontsize=_fs(8), framealpha=0.9)
            ax3.set_title("Option Strikes at Open and Close vs. Underlying Price")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Price")
            ax3.grid(True)
        else:
            ax3.text(
                0.5,
                0.5,
                "No underlying data in range",
                transform=ax3.transAxes,
                ha="center",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No strike data to plot",
            transform=ax3.transAxes,
            ha="center",
        )

    # ---- ax4: required margin & days open ----------------------------------
    avg_margin = np.mean(required_margins_for_plot)
    max_margin = max(required_margins_for_plot)
    ax4.plot(
        dt_series,
        required_margins_for_plot,
        color="green",
        label=f"Required Margin, avg={avg_margin:.0f}, max={max_margin:.0f}",
        linewidth=1,
        marker=".",
        markersize=3,
    )
    ax4.set_title("Required Margin & Days Open")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Margin ($)", color="green")
    ax4.tick_params(axis="y", labelcolor="green")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend(loc="upper left")

    ax4_twin = ax4.twinx()
    if itm_dates:
        ax4_twin.scatter(
            itm_dates,
            itm_days_open,
            color="red",
            marker="o",
            s=5,
            alpha=0.6,
            label="Days Open (ITM)",
            rasterized=True,
        )
    if otm_dates:
        ax4_twin.scatter(
            otm_dates,
            otm_days_open,
            color="blue",
            marker="o",
            s=5,
            alpha=0.6,
            label="Days Open (OTM)",
            rasterized=True,
        )
    ax4_twin.set_ylabel("Days Open", color="blue")
    ax4_twin.tick_params(axis="y", labelcolor="blue")
    ax4_twin.legend(loc="upper right")

    # ---- ax5: open premiums & position counts ------------------------------
    def _flatten(date_list, list_of_lists):
        d, v = [], []
        for dte, lst in zip(date_list, list_of_lists):
            d.extend([dte] * len(lst))
            v.extend(lst)
        return d, v

    call_color, put_color = "tab:blue", "tab:orange"
    call_target_color, put_target_color = "tab:red", "tab:purple"

    oc_dates, oc_vals = _flatten(dt_series, open_premium_calls)
    tc_dates, tc_vals = _flatten(dt_series, target_premium_calls)
    op_dates, op_vals = _flatten(dt_series, open_premium_puts)
    tp_dates, tp_vals = _flatten(dt_series, target_premium_puts)

    if oc_dates:
        ax5.scatter(
            oc_dates,
            oc_vals,
            marker="o",
            s=30,
            alpha=0.6,
            color=call_color,
            label="Open Call Premiums",
            rasterized=True,
        )
    if tc_dates:
        ax5.scatter(
            tc_dates,
            tc_vals,
            marker="o",
            s=30,
            alpha=0.6,
            color=call_target_color,
            label="Target Call Premiums",
            rasterized=True,
        )
    if op_dates:
        ax5.scatter(
            op_dates,
            op_vals,
            marker="s",
            s=30,
            alpha=0.6,
            color=put_color,
            label="Open Put Premiums",
            rasterized=True,
        )
    if tp_dates:
        ax5.scatter(
            tp_dates,
            tp_vals,
            marker="s",
            s=30,
            alpha=0.6,
            color=put_target_color,
            label="Target Put Premiums",
            rasterized=True,
        )

    ax5.set_title("Open Premiums and Number of Opened Positions")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Premium ($)", color="black")
    ax5.tick_params(axis="y", labelcolor="black")
    ax5.grid(True)

    ax5_twin = ax5.twinx()
    ax5_twin.plot(
        dt_series,
        otm_open_counts,
        color="blue",
        label="Positions OTM at Open",
        linewidth=1,
        linestyle="-.",
    )
    ax5_twin.plot(
        dt_series,
        itm_now_otm_open_counts,
        color="red",
        label="Positions ITM Now, OTM at Open",
        linewidth=1,
        linestyle="--",
    )
    ax5_twin.plot(
        dt_series,
        itm_open_counts,
        color="green",
        label="Positions ITM at Open",
        linewidth=2,
        linestyle="-",
    )
    ax5_twin.set_ylabel("Number of Positions Opened", color="black")
    ax5_twin.tick_params(axis="y", labelcolor="black")
    ax5_twin.legend(loc="upper right")

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=_fs(8),
        framealpha=0.9,
    )

    # ---- ax6: VIX & IV ----------------------------------------------------
    if not vix_df.empty or any(~np.isnan(iv) for iv in iv_call_data + iv_put_data):
        vix_start = min(dt_series) if dt_series else vix_df["date"].min()
        vix_end = max(dt_series) if dt_series else vix_df["date"].max()
        ax6_twin = ax6.twinx()

        if any(~np.isnan(iv) for iv in iv_call_data):
            ax6_twin.plot(
                dt_series,
                iv_call_data,
                color="red",
                label="Call IV",
                linewidth=1,
                marker=".",
                markersize=3,
            )
        if any(~np.isnan(iv) for iv in iv_put_data):
            ax6_twin.plot(
                dt_series,
                iv_put_data,
                color="blue",
                label="Put IV",
                linewidth=1,
                marker=".",
                markersize=3,
            )
        ax6_twin.set_ylabel("Implied Volatility", color="black")
        ax6_twin.tick_params(axis="y", labelcolor="black")
        if any(~np.isnan(iv) for iv in iv_call_data) or any(~np.isnan(iv) for iv in iv_put_data):
            ax6_twin.legend(loc="upper right", fontsize=_fs(8))

        if not vix_df.empty:
            if not isinstance(vix_df.index, pd.DatetimeIndex):
                vix_df["date"] = pd.to_datetime(vix_df["date"])
                vix_df = vix_df.set_index("date")
            vix_filtered = vix_df.loc[vix_start:vix_end]
            if not vix_filtered.empty:
                ax6.plot(
                    vix_filtered.index,
                    vix_filtered["close"],
                    color="purple",
                    label="VIX Index",
                    linewidth=1,
                    marker=".",
                    markersize=3,
                )
                ax6.set_ylabel("VIX", color="purple")
                ax6.tick_params(axis="y", labelcolor="purple")
                ax6.legend(loc="upper left", fontsize=_fs(8))

        ax6.set_title(f"VIX Index and Implied Volatility for {ticker}")
        ax6.set_xlabel("Date")
        ax6.grid(True)
    else:
        ax6.text(
            0.5, 0.5, "No VIX or IV data to plot", transform=ax6.transAxes, ha="center"
        )

    # ---- ax7: Spread Width and Closed Profits ------------------------------
    def _flatten(dates, lists):
        d, v = [], []
        for dt, lst in zip(dates, lists):
            d.extend([dt] * len(lst))
            v.extend(lst)
        return d, v

    sw_dates, sw_vals = _flatten(dt_series, spread_width_array)
    put_profit_dates, put_profit_vals = _flatten(dt_series, put_closed_profit_array)
    call_profit_dates, call_profit_vals = _flatten(dt_series, call_closed_profit_array)

    if sw_dates or put_profit_dates or call_profit_dates:
        # Plot spread width on primary y-axis (left)
        if sw_dates:
            ax7.scatter(
                sw_dates,
                sw_vals,
                color="blue",
                marker="o",
                s=30,
                alpha=0.6,
                label="Spread Width",
                rasterized=True,
            )
        ax7.set_ylabel("Spread Width ($)", color="blue")
        ax7.tick_params(axis="y", labelcolor="blue")

        # Create second y-axis for profits (right)
        ax7_twin = ax7.twinx()
        # Plot call profits with color based on sign
        if call_profit_dates:
            call_colors = ['green' if val > 0 else 'red' for val in call_profit_vals]
            ax7_twin.scatter(
                call_profit_dates,
                call_profit_vals,
                c=call_colors,
                marker="^",
                s=30,
                alpha=0.6,
                label="Call Closed Profit",
                rasterized=True,
            )
        # Plot put profits with color based on sign
        if put_profit_dates:
            put_colors = ['green' if val > 0 else 'red' for val in put_profit_vals]
            ax7_twin.scatter(
                put_profit_dates,
                put_profit_vals,
                c=put_colors,
                marker="s",
                s=30,
                alpha=0.6,
                label="Put Closed Profit",
                rasterized=True,
            )
        ax7_twin.set_ylabel("Closed Profit ($)", color="black")
        ax7_twin.tick_params(axis="y", labelcolor="black")

        # Combine legends from both axes
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_twin.get_legend_handles_labels()
        ax7.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            fontsize=_fs(8),
            framealpha=0.9,
        )

        ax7.set_title("Spread Width and Closed Profits")
        ax7.set_xlabel("Date")
        ax7.grid(True)
    else:
        ax7.text(0.5, 0.5, "No spread width or profit data", transform=ax7.transAxes, ha="center")

    # ---- ax8: days-to-expiry scatter ---------------------------------------
    dte_dates, dte_vals = [], []
    for d, l in zip(dt_series, days_to_expiry_array):
        dte_dates.extend([d] * len(l))
        dte_vals.extend([td.days for td in l if td is not None])

    if dte_dates:
        ax8.scatter(
            dte_dates,
            dte_vals,
            color="orange",
            marker="o",
            s=30,
            alpha=0.6,
            label="Days to Expiry",
            rasterized=True,
        )
        ax8.set_title("Days to Expiry")
        ax8.set_xlabel("Date")
        ax8.set_ylabel("Days")
        ax8.legend()
        ax8.grid(True)
    else:
        ax8.text(
            0.5, 0.5, "No days to expiry data", transform=ax8.transAxes, ha="center"
        )

    # ── Global X-axis formatting & save plot ────────────────────────────────
    if dt_series:
        min_date, max_date = min(dt_series), max(dt_series)
        for a in [ax1, ax2, ax3, ax4, ax5, ax5_twin, ax6, ax7, ax7_twin, ax8]:
            if a is not None:
                a.set_xlim(min_date, max_date)
                a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                for lbl in a.get_xticklabels():
                    lbl.set_rotation(45)
                    lbl.set_horizontalalignment("right")
    else:
        print("No dt_series available for x-axis formatting")

    # Replace tight_layout with tighter manual spacing
    fig.autofmt_xdate()
    fig.subplots_adjust(
        hspace=0.20,
        wspace=0.15,
        top=0.96,
        bottom=0.04,
        left=0.04,
        right=0.98,
    )

    # Shade parameter-period regions
    num_periods = len(parameter_history)
    colors = plt.cm.viridis(np.linspace(0, 1, num_periods)) if num_periods > 0 else []
    for i, params in enumerate(parameter_history):
        try:
            p_start = datetime.strptime(params["start_date"], "%Y-%m-%d")
            p_end = datetime.strptime(params["end_date"], "%Y-%m-%d")
            for a in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax7_twin, ax8]:
                if a is not None:
                    a.axvspan(p_start, p_end, alpha=0.15, color=colors[i % len(colors)])

            valid_indices = [j for j, d in enumerate(dt_series) if d >= p_start]
            if not valid_indices:
                continue
            start_idx = min(valid_indices, key=lambda j: abs(dt_series[j] - p_start))
            if start_idx < len(pnl_cumulative_realized_series):
                y_val = pnl_cumulative_realized_series[start_idx]
                label_text = (
                    f"EW:{params.get('expiring_wks','N/A')},"
                    f"W:{params.get('iron_condor_width','N/A')},"
                    f"SL:{params.get('stop_profit_percent','N/A')},"
                    f"VIX:{round(params.get('vix_correlation',0),2)},"
                    f"{round(params.get('vix_threshold',0),2)}\n"
                    f"Prem:{params.get('target_premium_otm','') or params.get('target_delta','')},"
                    f"Steer:{params.get('target_steer','N/A')}"
                )
                mid_pt = p_start + (p_end - p_start) / 2
                ax1.annotate(
                    label_text,
                    xy=(mid_pt, y_val),
                    xycoords="data",
                    xytext=(0, 20),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=_fs(8),
                )
        except Exception as e:
            print(f"Error plotting parameter history annotation: {e}")

    fig.autofmt_xdate()
    plt.tight_layout(pad=3.0)

    plot_dir = "cover_call_plots"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = os.path.join(
        plot_dir,
        f"recursive_monthly_{ticker}_{global_start_date}_to_{global_end_date}_{timestamp}.png",
    )
    try:
        plt.savefig(filename, bbox_inches="tight", dpi=200)
        print(f"Plot saved to {filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close("all")

async def main():
    # Define the backtesting period
    global_start_date = "2022-01-01"
    global_end_date = datetime.now().strftime("%Y-%m-%d")

    # Configuration and placeholder variables
    # tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
    tickers = ['GOOGL','PLTR','MRNA','T','AMZN','AAPL','INTC','WMT','BAC','AMD','AVGO','DIS','PFE','UNH','GOOG','NKE','CSCO','FFIV','OMC','COR','ORCL','LVS','APH','SBUX','MSFT','TSLA','KO','MU','DAL','UBER','NFLX','GM','BA','VZ','C','SMCI','F','META','NEE','MDT','PEP','JPM','NWSA','PANW','CPRT','BMY','ADM','FCX']
    extra_ticker = ['QQQ','SPY','ARKK','TQQQ','SQQQ','SQ','PYPL','QCOM','OXY']
    tickers = tickers + extra_ticker
    tickers = ['AAPL','META','MSFT','AMZN','QQQ','JPM','QCOM','PEP']
    tickers = ['SPY']
    global_start_date = "2025-08-20"
    global_end_date   = datetime.now().strftime("%Y-%m-%d")
    global_end_date   = "2025-08-25"
    lookback_months   = 0
    lookforward_months  = 240
    target_premium_otm    = np.arange(0.1,0.3,0.05)
    target_premium_otm    = [0.25]
    # target_premium_otm    = [0.2]
    # target_premium_otm    = [None]
    target_delta = [0.025]
    target_delta = [0.015]
    target_delta = [None]
    # target_steer = [0]
    target_steer = [0.9]
    iron_condor_width = [15,20,25]
    iron_condor_width = [20]

    # target_premium = [0.1]
    expiring_wks      = [2,3,4]  # Your expiring weeks data
    expiring_wks      = [6]  # Your expiring weeks data
    vix_correlation    = [0.05,0.1,0.15,0]
    vix_correlation    = [0.05]
    # vix_correlation    = [0]
    vix_threshold      = [20,25,15]
    vix_threshold      = [20]
    # roll_methods      = ['close price','loss','roll']
    roll_methods      = [None]
    stop_loss_action = ['roll_in']  # 'roll' or 'close' or 'skip'
    stop_profit_percent = np.arange(0.2,0.8,0.2)
    stop_profit_percent = [0.2]  # Your stop loss percentage(s)
    day_of_week       = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    # day_of_week       = ['Friday']
    trade_type       = ['iron_condor','put_spread','put_credit_spread']
    trade_type       = ['put_credit_spread']

    trade_parameters = [
            {
                'expiring_wks': expiring_wks,
                'target_premium_otm': target_premium_otm,
                'target_steer': target_steer,
                'target_delta': target_delta,
                'iron_condor_width': iron_condor_width,
                'stop_loss_action': stop_loss_action,
                'stop_profit_percent': stop_profit_percent,
                'day_of_week': day_of_week,
                'vix_correlation': vix_correlation,
                'vix_threshold': vix_threshold,
                'trade_type': trade_type,
            }
            for expiring_wks, target_premium_otm, target_steer, target_delta, iron_condor_width, stop_loss_action, stop_profit_percent, vix_correlation, vix_threshold, trade_type in product(
                expiring_wks, target_premium_otm, target_steer, target_delta, iron_condor_width, stop_loss_action, stop_profit_percent, vix_correlation, vix_threshold, trade_type
            )
        ]

    # Ensure the log directory exists
    log_dir = "./option_test_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    recursive_results = {}
    async with PolygonAPIClient(api_key=polygonio_config.API_KEY, max_concurrent_requests=10) as client:
        for ticker in tickers:
            df = get_historical_prices(ticker, global_start_date, global_end_date)
            vix_df = get_historical_prices("VIX", global_start_date, global_end_date)

            print(f"\n--- Starting Backtest for {ticker} ---")
            # load_stored_option_data(ticker)  # Load cached data if available

            # fetch_data_1 = await client._fetch_and_store_option_data('SPY','O:SPY231020C00451000',451,'call','2023-10-20','2023-10-06',use_close_price=False)
            # fetch_data_2 = await client._fetch_and_store_option_data('SPY','O:SPY231020C00480000',480,'call','2023-10-20','2023-10-06',use_close_price=False)
            # fetch_data_3 = await client._fetch_and_store_option_data('AAPL','O:AAPL240614C00210000',205,'call','2024-06-14','2024-06-11',use_close_price=True)
            # fetch_data_4 = await client._fetch_and_store_option_data('AAPL','O:AAPL240614C00200000',205,'call','2024-06-14','2024-06-12',use_close_price=True)
            # print(f"fetch_data_1: {fetch_data_1}, fetch_data_2: {fetch_data_2}, fetch_data_3: {fetch_data_3}, fetch_data_4: {fetch_data_4}")
            # breakpoint()

            # Run the recursive monthly backtest
            final_pnl, dt_series, pnl_cumulative_series, parameter_history, pnl_series, details_m, pnl_cumulative_realized_series = await monthly_recursive_backtest(
                            ticker=ticker,
                            global_start_date=global_start_date,
                            global_end_date=global_end_date,
                            lookback_months=lookback_months,
                            lookforward_months=lookforward_months,
                            trade_parameters=trade_parameters,  # Pass the list of trade parameters
                            client=client,
                            save_file=True,
                            trade_type=trade_type,
                            input_df={'df': df, 'vix_df': vix_df},
                        )
            if new_data_entry_count > 100:
                save_stored_option_data(ticker)
            # Store the results
            recursive_results[ticker] = {
                "weekly_pnl": pnl_series,
                "weekly_pnl_cumulative": pnl_cumulative_series,
                "dates": dt_series,
                "parameter_history": parameter_history,
            }
            # In the main() function, replace the previous $SELECTION_PLACEHOLDER$ code with:
            plot_recursive_results(ticker, final_pnl, details_m, pnl_cumulative_series, pnl_cumulative_realized_series, parameter_history,
                                   global_start_date, global_end_date,df_dict={'df': df, 'vix_df': vix_df})

if __name__ == "__main__":
    asyncio.run(main())