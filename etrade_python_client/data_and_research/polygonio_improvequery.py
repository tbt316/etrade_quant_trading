import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import csv
import os
import matplotlib.pyplot as plt
import yfinance as yf
from data_and_research import polygonio_config  # Replace with your config filename
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
from backtesting.monthly_cpu_bound import run_monthly_backtest_cpu_bound
from concurrent.futures import ProcessPoolExecutor
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import yfinance as yf
import random
from itertools import product
import bisect

# Configure logging
# logging.basicConfig(
#     filename='backtest_debug.log',
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.CRITICAL
# )

# SSL Context for aiohttp
ssl_context = ssl.create_default_context(cafile=certifi.where())

OPTION_TRADE_COST = 0.5
LOOKBACK_WINDOW = 5
USE_CLOSE_PRICE_SETTING = False
LOAD_MONTHLY_DATA = False
NUM_CORES = 8
VALIDATION_MONTH_FORWARD = 3
VOL_THRESHOLD = -1
SPREAD_COST = 0.1
MIN_PROFIT = 0.05 
MAX_PROFIT = 0.2
OPTION_CHAIN_FORCE_UPDATE = False
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

def find_closest_strike(strike_prices, target_strike):
    """
    Given a list of strike prices, return the one closest to the target_strike.
    If the target strike is below the minimum or above the maximum, return the nearest boundary.
    """
    if not strike_prices:
        logging.error("No strike prices available to find closest match.")
        raise ValueError("No strike prices available to find closest match.")
    
    min_strike = min(strike_prices)
    max_strike = max(strike_prices)
    
    if target_strike <= min_strike:
        closest_strike = min_strike
        logging.debug(f"Target strike {target_strike} below min strike {min_strike}. Choosing {closest_strike}.")
    elif target_strike >= max_strike:
        closest_strike = max_strike
        logging.debug(f"Target strike {target_strike} above max strike {max_strike}. Choosing {closest_strike}.")
    else:
        closest_strike = min(strike_prices, key=lambda x: abs(x - target_strike))
        logging.debug(f"Target strike {target_strike}, Closest strike found: {closest_strike}")
    
    return round(closest_strike,2)

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

def get_historical_prices(ticker, start_date, end_date, vol_lookback=5):
    """
    Fetch historical prices using Yahoo Finance (yfinance) and return the 'Close' prices.
    Also adds today's data if available.
    
    Args:
        ticker (str): Stock ticker (e.g., "AAPL").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
    Returns:
        pd.DataFrame: A DataFrame with columns ['date', 'close', 'ticker'], sorted by date ascending.
    """
    yf_ticker = convert_polygon_to_etrade_ticker(ticker)
    
    # Create directory for cached data if it doesn't exist
    cache_dir = os.path.join("yfinance", "prices")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f"{yf_ticker}_prices.csv")
    
    # Check if cache file exists and is up-to-date
    use_cache = False
    today = datetime.today().date()
    requested_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    requested_end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    if os.path.exists(cache_file):
        # Check when the file was last modified
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        
        # Ensure today is also a date object with the same format
        today_date = datetime.now().date()
        
        try:
            # Read date range from cache
            cached_data = pd.read_csv(cache_file, parse_dates=['date'])
            cached_start = cached_data['date'].min().date()
            cached_end = cached_data['date'].max().date()
            
            # Use cache if it covers the requested date range (with tolerance) and was updated today
            # The cache covers the requested range if:
            # - cached_start is within 7 days after the requested_start
            # - cached_end is on or after the requested_end (has recent enough data)
            file_mod_str = file_mod_time.strftime('%Y-%m-%d')
            today_str = today_date.strftime('%Y-%m-%d')
            
            # Calculate days between cached_start and requested_start
            days_difference = (cached_start - requested_start).days
            
            use_cache = (days_difference <= 7 and  # Allow up to 7 days difference
                        cached_end >= requested_end and
                        (file_mod_str == today_str or requested_end < today_date))
            
        except Exception as e:
            print(f"Warning: Error reading cached price data for {ticker}. Error: {e}")
            use_cache = False
    
    df = None
        
    if use_cache:
        try:
            # Read from cache
            cached_data = pd.read_csv(cache_file, parse_dates=['date'])
            
            # Filter to requested date range
            df = cached_data[(cached_data['date'].dt.date >= requested_start) & 
                             (cached_data['date'].dt.date <= requested_end)].copy()
        except Exception as e:
            print(f"Warning: Error processing cached price data for {ticker}. Error: {e}")
            use_cache = False
    
    # If cache is not usable or doesn't cover the date range, query yfinance API
    if not use_cache or df is None or df.empty:
        try:
            # Download historical price data from yfinance
            print(f"Fetching price data from yfinance for {ticker}")
            raw_df = yf.download(yf_ticker, start=start_date, end=end_date, progress=True, auto_adjust=False)

            if raw_df.empty:
                print(f"Warning: No data returned for {ticker} from yfinance.")
                # Return empty DataFrame with correct structure if no data is available
                return pd.DataFrame(columns=['date', 'close', 'returns', 'vol_20', 'MA', 'ticker'])
            
            # Handle multi-level columns if they exist
            if isinstance(raw_df.columns, pd.MultiIndex):
                # Select Close column for the specific ticker
                df = raw_df['Close'][yf_ticker].to_frame()
            else:
                # If single-level columns, just select Close
                df = raw_df['Close'].to_frame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns
            df.columns = ['date', 'close']
            
            # Fetch today's data
            today = datetime.today().date()
            # Check if today's data is already present in df
            dates_in_df = [d.date() if isinstance(d, datetime) else d for d in df['date']]
            if today not in dates_in_df:
                # Query today's data by setting end date as tomorrow (yfinance end is exclusive)
                tomorrow = today + timedelta(days=1)
                today_df = yf.download(yf_ticker, start=str(today), end=str(tomorrow), progress=False, auto_adjust=False)
                if not today_df.empty:
                    if isinstance(today_df.columns, pd.MultiIndex):
                        today_df = today_df['Close'][yf_ticker].to_frame()
                    else:
                        today_df = today_df['Close'].to_frame()
                    today_df = today_df.reset_index()
                    today_df.columns = ['date', 'close']
                    # Append today's data to existing df
                    df = pd.concat([df, today_df], ignore_index=True)
            
            # Save complete data to cache
            df.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"Warning: Error fetching price data for {ticker} from yfinance. Error: {e}")
            if df is None:
                return pd.DataFrame(columns=['date', 'close', 'returns', 'vol_20', 'MA', 'ticker'])
    
    # Ensure df has been defined
    if df is None:
        return pd.DataFrame(columns=['date', 'close', 'returns', 'vol_20', 'MA', 'ticker'])
    
    # Remove duplicate dates
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate additional columns
    df["returns"] = df["close"].pct_change()
    df["vol_20"] = df["returns"].rolling(window=vol_lookback).std()
    df["MA"] = df["close"].rolling(window=vol_lookback).mean()
    
    # Change ticker info in the DataFrame to the original ticker
    df["ticker"] = ticker
    
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
        # Start timing the function
        start_time = time.time()
        logging.debug(f"Entering get_option_chain for {ticker}, Expiration: {expiration_date}, "
                    f"Type: {call_put}, As of: {as_of_date}")
        
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
                        elapsed_time = time.time() - start_time
                        logging.debug(f"Cache hit in get_option_chain for {ticker}. Time taken: {elapsed_time:.4f} seconds.")
                        return strike_dict
                    elif len(strike_dict) == 0:
                        elapsed_time = time.time() - start_time
                        logging.error(f"Empty option chain record {ticker} {expiration_date}, {call_put}, {as_of_date} Time taken: {elapsed_time:.4f} seconds.")
                        return strike_dict
            except Exception as e:
                logging.warning(f"Error checking memory for option chain: {e}")
        
        strike_dict = await self.query_polygon_for_option_chain_async(
            ticker, expiration_date, call_put, as_of_date
        )

        api_elapsed_time = time.time() - start_time
        logging.debug(f"API call in get_option_chain for {ticker}. Time taken: {api_elapsed_time:.4f} seconds.")
        
        if len(strike_dict) >= 0:
            # Insert the fetched strike list into the nested dictionary
            stored_option_chain[ticker].setdefault(exp_key, {})
            stored_option_chain[ticker][exp_key].setdefault(asof_key, {})
            stored_option_chain[ticker][exp_key][asof_key][callput_key] = strike_dict
            
            # Save the updated cache to disk
            # save_stored_option_data(ticker)
        else:
            logging.warning(f"Query option chain failed for {ticker}, {expiration_date}, {call_put}, {as_of_date}")
        
        # Stop timing the entire function
        elapsed_time = time.time() - start_time
        logging.debug(f"Exiting get_option_chain for {ticker}. Total time taken: {elapsed_time:.4f} seconds.")
        
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
                if as_of == '2023-01-26' and call_put.lower == "put":
                    print(list(strike_dict.keys()))  # Debug: print sorted strike prices
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
        deduped_requests = list(set(unique_chain_requests))
        
        # Create tasks for each unique request
        tasks = []

        for expiration_str, as_of_str, call_put in deduped_requests:
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
        for (expiration_str, as_of_str, call_put), result in zip(deduped_requests, fetched_data):
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
                    ticker, option_ticker, strike_price, call_put, expiration_date, pricing_date, use_close_price=USE_CLOSE_PRICE_SETTING
                )
                if fetched_data:
                    logging.info(f"Successfully fetched data for {ticker}, Strike: {strike_price}, Type: {call_put}, "
                                 f"Expiration: {expiration_date}, Pricing Date: {pricing_date}")
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
        use_close_price: bool = True  # Determines which data to fetch and store
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

        # Format the option symbol
        option_symbol = option_ticker

        # Set up the API endpoint and parameters
        if not use_close_price:
            url = f"https://api.polygon.io/v3/quotes/{option_symbol}"
            params = {
                "timestamp": pricing_date,
                "order": "desc",
                "sort": "timestamp",
                "limit": 100,
                "apiKey": self.api_key,
            }
        else:
            url = f"https://api.polygon.io/v1/open-close/{option_symbol}/{pricing_date}"
            params = {
                "apiKey": self.api_key,
            }

        def update_memory_invalid_data(use_close_price):
            """Update the stored option data with default values for invalid fetches."""
            if use_close_price:
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
                    logging.debug(f"Fetching {'close' if use_close_price else 'option data'} for {option_symbol}")
                    if response.status != 200:
                        full_url = f"{url}?{urlencode(params)}"
                        logging.error(f"Failed to fetch {'close price' if use_close_price else 'option data'} for {option_symbol}: HTTP {response.status}")
                        logging.error(f"Request URL: {full_url}")
                        update_memory_invalid_data(use_close_price)
                        return {}

                    data = await response.json()

                    if not use_close_price:
                        # Process quotes endpoint data
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
                            if ask_price > 0 or bid_price > 0:
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
                                print(f"Stored data for {ticker}, Strike: {strike_price},{call_put}, Expiration: {expiration_date}, Pricing: {pricing_date}: {option_data}")
                                logging.debug(f"Stored data for {ticker}, Strike: {strike_price}, Type: {call_put}, "
                                            f"Expiration: {expiration_date}, Pricing Date: {pricing_date}: {option_data}")
                                return fetched_data
                            else:
                                logging.warning(f"Invalid prices returned for {option_symbol}: ask={ask_price}, bid={bid_price}")
                                update_memory_invalid_data(use_close_price)
                                return {}
                        else:
                            logging.warning(f"No results found for {option_symbol}.")
                            update_memory_invalid_data(use_close_price)
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
                                print(f"Stored data for {ticker}, Strike: {strike_price},{call_put}, Expiration: {expiration_date}, Pricing: {pricing_date}: {option_data}")
                                logging.debug(f"Stored data for {ticker}, Strike: {strike_price}, Type: {call_put}, "
                                            f"Expiration: {expiration_date}, Pricing Date: {pricing_date}: {option_data}")
                                return fetched_data
                            else:
                                logging.warning(f"Invalid close price returned for {option_symbol}: close={close_price}")
                                update_memory_invalid_data(use_close_price)
                                return {}
                        else:
                            logging.warning(f"No close price found for {option_symbol} on {pricing_date}.")
                            update_memory_invalid_data(use_close_price)
                            return {}
        except aiohttp.ClientError as e:
            logging.error(f"ClientError while fetching {'close price' if use_close_price else 'option data'} for {option_symbol}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error while fetching {'close price' if use_close_price else 'option data'} for {option_symbol}: {e}")
            return {}
        
    async def get_option_prices_batch_async(
        self,
        ticker: str,
        options_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously fetches option data for a batch of options.

        :param ticker: The underlying asset ticker symbol.
        :param options_list: A list of option dictionaries containing 'strike_price', 'call_put', 'expiration_date', 'quote_timestamp'.
        :return: A list of dictionaries containing fetched data for each option.
        """
        tasks = []
        for option in options_list:
            task = asyncio.create_task(
                self.query_polygon_for_option_price_async(
                    ticker=ticker,
                    strike_price=option['strike_price'],
                    call_put=option['call_put'],
                    expiration_date=option['expiration_date'],
                    pricing_date=option['quote_timestamp'],
                    option_ticker = option['option_ticker'] if 'option_ticker' in option else None,
                )
            )
            tasks.append(task)
    
        # Gather all requests concurrently
        fetched_data = await asyncio.gather(*tasks, return_exceptions=True)
    
        # Handle exceptions if any
        results = []
        for i, result in enumerate(fetched_data):
            if isinstance(result, Exception):
                logging.error(f"Error fetching option data: {result}")
                results.append({})
            else:
                results.append(result)
    
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
    
    roll_method = None
    expiring_wks = trade_parameter['expiring_wks']
    target_premium_otm = trade_parameter['target_premium_otm']
    target_premium_steer = trade_parameter['target_premium_steer']
    day_of_week = trade_parameter['day_of_week']
    stop_loss_percent = trade_parameter['stop_loss_percent']
    stop_loss_action = trade_parameter['stop_loss_action']
    iron_condor_width = trade_parameter['iron_condor_width']
    target_delta = trade_parameter['target_delta']
    vix_corrleation = trade_parameter['vix_correlation']

    if target_premium_otm is not None:
        target_premium_call = target_premium_otm * (expiring_wks*5)**0.5 * (1-target_premium_steer)
        target_premium_put  = target_premium_otm * (expiring_wks*5)**0.5 * (1+target_premium_steer)
    else:
        target_premium_call = None
        target_premium_put  = None

    if target_delta is not None:
        target_delta_call = -target_delta
        target_delta_put  = target_delta
    else:
        target_delta_call = None
        target_delta_put  = None

    active_positions: List[Dict[str, Any]] = []
    total_pnl = 0.0
    sharpe_ratio = 0.0
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d")

    # Add carry-over positions that are still open
    if carry_over_weekly_results and carry_over_weekly_results[-1].get('active_positions', None) is not None:
        for pos in carry_over_weekly_results[-1].get('active_positions', None):
            if pos['expiration']:
                if pos['expiration'] > start_dt:
                    active_positions.append(pos)
                    print(f"Carried over position added: Expiration {pos['expiration']}, call sold {pos['call_strike_sold']}, {pos.get('call_rolled','N/A')} put sold {pos['put_strike_sold']}, {pos.get('put_rolled', 'N/A')}")
                
    # -------------- EARNINGS DATES --------------
    earnings_dates = get_earnings_dates(ticker, start_date, end_date)

    # -------------- Log file --------------
    log_filename = f"./option_test_log/option_backtest_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "event", "week_start", "expiration", "underlying_close",
        "short_call_strike", "short_call_price",
        "short_put_strike",  "short_put_price",
        "long_call_strike",  "long_call_price",
        "long_put_strike",   "long_put_price",
        "total_credit_debit",
        "total_expiration_payoff",
        "weekly_pnl",
        "cumulative_pnl"
    ]

    # -------------- Historical Data --------------
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=7)
    # Ensure the 'date' column is in datetime format
    df = df_dict['df']
    vix_df = df_dict['vix_df']
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Apply the mask to filter the DataFrame
    mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    hist_df = df[mask].reset_index(drop=True)

    # Set the date as index and reindex to fill any missing dates with forward fill
    vix_close_series = vix_df.set_index('date')['close'].reindex(pd.date_range(start_date, end_date)).ffill()

    # -------------- Fridays --------------
    fridays_list = get_all_weekdays('Friday',start_dt, end_dt, expiring_wk=expiring_wks,trading_dates_df=hist_df)
    wednesday_list = get_all_weekdays('Wednesday',start_dt, end_dt, expiring_wk=expiring_wks,trading_dates_df=hist_df)

    if ticker == 'VIX':
        expire_day_list = wednesday_list
    else:
        expire_day_list = fridays_list
    total_pnl = 0.0
    weekly_results = []
    weekly_dates = []
    cumulative_pnls  = []
    arrow_data = []  # For negative weeks

    # ------------------------------------------------
    # 1) Pre-collect all data we need for each Friday
    # ------------------------------------------------
    needed_data = []  # list of dict, each dict holds info for that Friday's trade

    def check_option_assignment(active_positions, ticker, current_day, mode="training"):
        call_loss = []
        put_loss = []
        row = hist_df.loc[hist_df["date"].dt.date == current_day.date()]
        if row.empty:
            return [], []
        close_price = row['close'].iloc[0]

                    # if len(active_positions) > 1:
                    #     print(f"Warning: Multiple positions expiring on {current_day.date()} for {ticker}")
                    #     for pos in active_positions:
                    #         print(f"  - Position: {pos['expiration'].date()}, "
                    #                 f"Call Strike Sold: ${pos['call_strike_sold']}, ${pos['short_call_prem_open']} "
                    #                 f"Put Strike Sold: ${pos['put_strike_sold']}, ${pos['short_put_prem_open']} ")
            
        for position in active_positions:

            if position['expiration'].date() != current_day.date():
                continue  # Skip positions not expiring today
            
                    # print(f"checking option assignment for {ticker} on {current_day.date()} with close price: {close_price:.2f}")
                    # print(f"call position: {position['call_strike_sold']} sold, {position['short_call_prem_open']} premium open, "
                    #       f"put position {position['put_strike_sold']} sold, {position['short_put_prem_open']} premium open")

            # Calculate call-side loss
            sc_loss = (max(close_price - position['call_strike_sold'], 0)
                    if position['short_call_prem_open'] > 0 and not position.get('call_closed_by_stop', False)
                    else 0)
            lc_gain = (max(close_price - position['call_strike_bought'], 0)
                    if position['long_call_prem_open'] > 0
                    else 0)
            call_loss_this = sc_loss - lc_gain
            
            # Calculate put-side loss
            sp_loss = (max(position['put_strike_sold'] - close_price, 0)
                    if position['short_put_prem_open'] > 0 and not position.get('put_closed_by_stop', False)
                    else 0)
            lp_gain = (max(position['put_strike_bought'] - close_price, 0)
                    if position['long_put_prem_open'] > 0
                    else 0)
            put_loss_this = sp_loss - lp_gain
            
            # Log assignments if they occur
            if call_loss_this > 0:
                if mode == "validation":
                    state = position.get('call_open_state', 'unknown')
                    print(f"\033[91mCall assigned!!! {ticker} on {current_day} call_loss: {call_loss_this:.2f}, "
                        f"close_price: {close_price:.2f}, call_strike_sold: {position['call_strike_sold']} "
                        f"'{state}' on {position['week_start']} open distance: {position['open_distance_call']} \033[0m")
                call_loss.append({'loss': call_loss_this, 'open_state': position.get('call_open_state')})
            
            if put_loss_this > 0:
                if mode == "validation":
                    state = position.get('put_open_state', 'unknown')
                    print(f"\033[91mPut assigned!!! {ticker} on {current_day} put_loss: {put_loss_this:.2f}, "
                        f"close_price: {close_price:.2f}, put_strike_sold: {position['put_strike_sold']} "
                        f"'{state}' on {position['week_start']} open distance: {position['open_distance_put']}\033[0m")
                put_loss.append({'loss': put_loss_this, 'open_state': position.get('put_open_state')})
        
        return call_loss, put_loss

    def find_strike_custom(options_list, premiums_list, target, stock_price, OTM=True, VOL_THRESHOLD_BYPASS=False):
        """
        Finds the option strike price based on the provided target criteria, with price monotonicity enforced for premium search.

        Args:
            options_list (list): A list of dictionaries, each representing an option contract.
                                Expected keys: 'strike_price', 'expiration_date', 'call_put' ('call' or 'put').
            premiums_list (list): A list of dictionaries, each containing premium and delta data for the corresponding option.
                                Expected keys: 'close_price', 'mid_price', 'close_volume', 'close_price_delta', 'mid_price_delta'.
            target (dict): A dictionary with optional keys "premium_target" and "delta_target".
                        - If "premium_target" is provided, finds the strike with premium closest to this value.
                        - If "delta_target" is provided, finds the strike with delta closest to this value.
                        - If both are None, finds the strike closest to the stock_price.
            stock_price (float): The current price of the underlying stock. Required for OTM filtering and strike-based search.
            OTM (bool, optional): If True, only consider Out-of-the-Money options. Defaults to True.
            VOL_THRESHOLD_BYPASS (bool, optional): If True, uses a lower volume threshold (10). Defaults to False.

        Returns:
            tuple: (strike, premium) where strike is the selected strike price, and premium is its corresponding premium.
                Returns (None, 0) if no suitable option is found.
        """
        # Check if stock_price is provided
        if stock_price is None:
            print("Stock price is None, cannot find closest strike (required for OTM check if enabled).")
            return None, 0

        # Determine search_type and target_value
        if target.get("premium_target") is not None:
            search_type = "premium"
            target_value = target["premium_target"]
        elif target.get("delta_target") is not None:
            search_type = "delta"
            target_value = target["delta_target"]
        else:
            search_type = "strike"
            target_value = stock_price

        # Define premium_field
        premium_field = "close_price" if USE_CLOSE_PRICE_SETTING else "mid_price"

        # Define delta_field if needed
        if search_type == "delta":
            delta_field = "close_price_delta" if USE_CLOSE_PRICE_SETTING else "mid_price_delta"

        # Combine options and premiums, filtering out invalid premium data
        options_with_premiums = []
        for opt, prem_data in zip(options_list, premiums_list):
            if isinstance(prem_data, dict) and premium_field in prem_data and prem_data[premium_field] is not None and prem_data[premium_field] > 0:
                if search_type == "delta":
                    if delta_field in prem_data and prem_data[delta_field] is not None:
                        options_with_premiums.append((opt, prem_data))
                else:
                    options_with_premiums.append((opt, prem_data))

        if not options_with_premiums:
            return None, 0

        # Apply OTM filtering if enabled
        filtered_options = []
        if OTM:
            for opt, prem_data in options_with_premiums:
                if 'call_put' not in opt:
                    print(f"Warning: 'call_put' key missing for strike {opt.get('strike_price', 'N/A')}. Including in search.")
                    filtered_options.append((opt, prem_data))
                    continue
                call_put = opt['call_put'].lower()
                strike_price = opt['strike_price']
                is_otm = (call_put == 'call' and strike_price >= stock_price) or (call_put == 'put' and strike_price <= stock_price)
                # delta_check_ok = abs(opt['close_price_delta']) >= 0.005 if USE_CLOSE_PRICE_SETTING else abs(opt['mid_price_delta']) >= 0.055
                if is_otm:
                    filtered_options.append((opt, prem_data))
        else:
            filtered_options = options_with_premiums

        # Fallback to all options if no OTM options found
        if not filtered_options and options_with_premiums:
            filtered_options = options_with_premiums

        if not filtered_options:
            return None, 0

        # Sort options by strike price for premium and delta searches
        sorted_options = sorted(filtered_options, key=lambda x: x[0]['strike_price'])

        if search_type == "strike":
            # For strike search, find the option with the strike closest to stock_price
            def get_diff(opt, prem_data):
                return abs(opt["strike_price"] - stock_price)
            
            best_option, best_prem_data = min(filtered_options, key=lambda x: get_diff(x[0], x[1]))
            candidate_strike = best_option['strike_price']
            candidate_premium = best_prem_data.get(premium_field, 0)
            candidate_volume = best_prem_data.get("close_volume", 0)
            volume_threshold = VOL_THRESHOLD if not VOL_THRESHOLD_BYPASS else 10

            if candidate_volume < volume_threshold:
                vol_candidates = [p for p in filtered_options if p[1].get("close_volume", 0) >= volume_threshold]
                if vol_candidates:
                    best_vol_candidate = min(vol_candidates, key=lambda x: get_diff(x[0], x[1]))
                    candidate_strike = best_vol_candidate[0]['strike_price']
                    candidate_premium = best_vol_candidate[1].get(premium_field, 0)
                else:
                    print(f"Warning: Selected strike {candidate_strike} has volume {candidate_volume} below threshold {volume_threshold}.")
            return candidate_strike, round(candidate_premium, 2)
        else:
            # For premium or delta search, use bidirectional search
            valid_pairs = []
            value_field = premium_field if search_type == "premium" else delta_field

            # For premium search, enforce monotonicity
            if search_type == "premium":
                call_put = sorted_options[0][0]['call_put'].lower()
                premiums = [prem_data[premium_field] for _, prem_data in sorted_options]
                adjusted_premiums = []
                if call_put == 'call':
                    # Premiums should be non-increasing for calls
                    adjusted_premiums.append(premiums[0])
                    for i in range(1, len(premiums)):
                        if premiums[i] > adjusted_premiums[-1]:
                            adjusted_premiums.append(adjusted_premiums[-1])
                            logging.warning(
                                f"Adjusted call premium for strike {sorted_options[i][0]['strike_price']} "
                                f"from {premiums[i]} to {adjusted_premiums[-1]} to enforce monotonicity."
                            )
                        else:
                            adjusted_premiums.append(premiums[i])
                else:  # put
                    # Premiums should be non-decreasing for puts
                    adjusted_premiums.append(premiums[0])
                    for i in range(1, len(premiums)):
                        if premiums[i] < adjusted_premiums[-1]:
                            adjusted_premiums.append(adjusted_premiums[-1])
                            logging.warning(
                                f"Adjusted put premium for strike {sorted_options[i][0]['strike_price']} "
                                f"from {premiums[i]} to {adjusted_premiums[-1]} to enforce monotonicity."
                            )
                        else:
                            adjusted_premiums.append(premiums[i])

            # Find the starting index (strike closest to stock_price)
            start_idx_ref_strike = min(sorted_options, key=lambda x: abs(x[0]['strike_price'] - stock_price))[0]['strike_price']
            try:
                start_idx = next(i for i, (opt, _) in enumerate(sorted_options) if opt['strike_price'] >= start_idx_ref_strike)
            except StopIteration:
                start_idx = len(sorted_options) - 1

            best_candidate_info = {'diff': float('inf'), 'strike': None, 'premium': 0, 'vol': 0, 'exp': None}

            for direction_up in [True, False]:
                idx = start_idx
                prev_value = None
                while 0 <= idx < len(sorted_options):
                    opt, prem_data = sorted_options[idx]
                    # Use adjusted premiums for premium search, otherwise use original value
                    value = adjusted_premiums[idx] if search_type == "premium" else prem_data.get(value_field, 0)
                    vol = prem_data.get("close_volume", 0)
                    strike = opt['strike_price']
                    exp_date = opt.get('expiration_date')
                    current_diff = abs(value - target_value)
                    valid_pairs.append((current_diff, strike, prem_data.get(premium_field, 0), vol, exp_date))

                    if prev_value is not None:
                        if (prev_value < target_value < value) or (value < target_value < prev_value):
                            prev_idx = idx - 1 if direction_up else idx + 1
                            if 0 <= prev_idx < len(sorted_options):
                                prev_opt, prev_prem_data = sorted_options[prev_idx]
                                prev_strike = prev_opt['strike_price']
                                prev_vol = prev_prem_data.get("close_volume", 0)
                                prev_exp = prev_opt.get('expiration_date')
                                prev_value = adjusted_premiums[prev_idx] if search_type == "premium" else prev_prem_data.get(value_field, 0)
                                prev_diff = abs(prev_value - target_value)
                                if prev_diff < best_candidate_info['diff']:
                                    best_candidate_info = {
                                        'diff': prev_diff,
                                        'strike': prev_strike,
                                        'premium': prev_prem_data.get(premium_field, 0),
                                        'vol': prev_vol,
                                        'exp': prev_exp
                                    }
                            break

                    if current_diff < best_candidate_info['diff']:
                        best_candidate_info = {
                            'diff': current_diff,
                            'strike': strike,
                            'premium': prem_data.get(premium_field, 0),
                            'vol': vol,
                            'exp': exp_date
                        }

                    prev_value = value
                    idx = idx + 1 if direction_up else idx - 1

            # Fallback if no better candidate is found
            if best_candidate_info['strike'] is None:
                if not valid_pairs:
                    return None, 0
                min_diff_pair = min(valid_pairs, key=lambda x: x[0])
                best_candidate_info = {
                    'diff': min_diff_pair[0],
                    'strike': min_diff_pair[1],
                    'premium': min_diff_pair[2],
                    'vol': min_diff_pair[3],
                    'exp': min_diff_pair[4]
                }

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

            return candidate_strike, round(candidate_premium, 2)
        
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
        premium_field = "close_price" if USE_CLOSE_PRICE_SETTING else "mid_price"

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

    async def pull_option_chain_data(ticker, expiration_str, as_of_str, close_price, force_otm=False, force_update=False):
        # First collect all needed chain data
        unique_chain_requests = [(expiration_str, as_of_str, "call"), (expiration_str, as_of_str, "put")]
        chain_data = await client.get_option_chains_batch_async(ticker, list(unique_chain_requests),force_update=force_update)

        # Get available strike dictionaries
        call_strike_dict = chain_data[ticker][expiration_str][as_of_str]["call"]
        put_strike_dict = chain_data[ticker][expiration_str][as_of_str]["put"]
        
        # Check if we have valid data
        if not call_strike_dict or not put_strike_dict:
            # print(f"No valid strikes found for {ticker} on {current_day.date()}")
            return None, None, None, None

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
            if close_price * (1+price_limit_percent*scale) < strike < close_price * ( 1 + 0.5 * scale): 
                call_options.append({
                    'strike_price': strike,
                    'call_put': 'call',
                    'expiration_date': expiration_str,
                    'quote_timestamp': as_of_str,
                    'option_ticker': call_strike_dict[strike]  # Add the option ticker
                })

        for strike in put_strike_dict.keys():
            if close_price * ( 1 - 0.5*scale ) < strike < close_price * (1 - scale * price_limit_percent):
                put_options.append({
                    'strike_price': strike,
                    'call_put': 'put',
                    'expiration_date': expiration_str,
                    'quote_timestamp': as_of_str,
                    'option_ticker': put_strike_dict[strike]  # Add the option ticker
                })

        # Prepare arrays to store all option data
        all_call_data = []
        all_put_data = []
        call_options_to_fetch = []
        put_options_to_fetch = []

        # Process call options
        for i, opt in enumerate(call_options):
            strike = opt['strike_price']
            pricing_date = opt['quote_timestamp']
            expiration = opt['expiration_date']
            data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get('call', {})
            
            if data and ( ( data.get('close_price', 0) > 0 and USE_CLOSE_PRICE_SETTING) or ((data.get('mid_price',0) > 0 and not USE_CLOSE_PRICE_SETTING)) ) and OPTION_CHAIN_FORCE_UPDATE == False:
            # if data:
                all_call_data.append(data)
            else:
                call_options_to_fetch.append(opt)
                all_call_data.append(None)  # Placeholder

        # Process put options
        for i, opt in enumerate(put_options):
            strike = opt['strike_price']
            pricing_date = opt['quote_timestamp']
            expiration = opt['expiration_date']
            data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get('put', {})
            
            if data and ( ( data.get('close_price', 0) > 0 and USE_CLOSE_PRICE_SETTING) or ((data.get('mid_price',0) > 0 and not USE_CLOSE_PRICE_SETTING)) ) and OPTION_CHAIN_FORCE_UPDATE == False:
                all_put_data.append(data)
            else:
                put_options_to_fetch.append(opt)
                all_put_data.append(None)  # Placeholder

        # Fetch missing data in one batch
        if len(call_options_to_fetch) > 0.1 * len(call_options) or len(put_options_to_fetch) > 0.1 * len(put_options):
            print(f"Fetching {len(call_options_to_fetch)} call options and {len(put_options_to_fetch)} put options for {ticker} on {current_day.date()}")
            fetched_data = await client.get_option_prices_batch_async(ticker, call_options_to_fetch + put_options_to_fetch)
            # Update call data
            fetched_call_data = fetched_data[:len(call_options_to_fetch)]
            call_idx = 0
            for i in range(len(all_call_data)):
                if all_call_data[i] is None:
                    all_call_data[i] = fetched_call_data[call_idx]
                    call_idx += 1

            # Update put data
            fetched_put_data = fetched_data[len(call_options_to_fetch):]
            put_idx = 0
            for i in range(len(all_put_data)):
                if all_put_data[i] is None:
                    all_put_data[i] = fetched_put_data[put_idx]
                    put_idx += 1

        calculate_delta(ticker, as_of_str, expiration_str, 'call', force_delta_update=True)
        calculate_delta(ticker, as_of_str, expiration_str, 'put', force_delta_update=True)

        return all_call_data, all_put_data, call_options ,put_options
    
    def fetch_pricing_date(day_of_week,current_day):
        if day_of_week == 'Friday':
            trade_date_str = current_day.strftime("%Y-%m-%d")
        else:
        # Map day names to numbers (0=Monday, 6=Sunday)
            day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            current_day_num = current_day.weekday()
            target_day_num = day_map.get(day_of_week, 4)  # Default to Friday if invalid
            days_to_add = (target_day_num - current_day_num) % 7
            pricing_date = current_day + timedelta(days=days_to_add)
            trade_date_str = pricing_date.strftime("%Y-%m-%d")

        return trade_date_str

    async def generate_roll_option(ticker,expiration_str,as_of_str,close_price, call_loss,put_loss,stock_price):
        print(f"Generating roll option for {ticker} on {current_day.date()} with close price: {close_price:.2f}")
        call_strike_sold = None
        put_strike_sold = None
        call_strike_bought = None
        put_strike_bought = None
        expiration_target = None
        call_loss_compensate = 0
        call_loss_this = 0
        MAX_ROLL_WEEK = 6
        expire_day = None
        # For call loss
        for entry in call_loss:
            loss = entry.get("loss")
            call_open_state = entry.get("open_state")
            # if loss > 0 and call_open_state == 'otm':
            if loss > 0:
                call_loss_compensate += loss
            call_loss_this += loss
        if call_loss_compensate > 0:
            if roll_method == 'close price':
                week_target = 1
                while call_strike_sold is None and week_target < MAX_ROLL_WEEK:
                    call_strikes_list = [opt['strike_price'] for opt in call_options]
                    call_strike_sold = find_closest_strike(call_strikes_list, close_price)
                    call_strike_bought = find_closest_strike(call_strikes_list, close_price*1.1)
                    week_target+=1
                    expiration_target = current_day + timedelta(weeks=week_target)
                    expiration_str = expiration_target.strftime("%Y-%m-%d")
                    all_call_data, _, call_options ,_ = await pull_option_chain_data(ticker, expiration_str, as_of_str, close_price, force_otm=True)
                if week_target == MAX_ROLL_WEEK:
                    print(f"Cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expiration_str} call strike to sell: {call_strike_sold}, call strike to buy: {call_strike_bought} compensate: {call_loss_compensate*0.8:.2f}")
            elif roll_method == 'loss':
                call_strike_sold,_ = find_closest_premium_strike(call_options, all_call_data, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate)*0.5, stock_price, VOL_THRESHOLD_BYPASS=True)
                call_strike_bought,_ = find_closest_premium_strike(call_options, all_call_data, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate), stock_price,VOL_THRESHOLD_BYPASS=True)
            elif roll_method == 'roll':
                # New roll_method: use option chain with 4-week expiration
                week_target = 1
                all_call_data_4wk = None
                call_strike_sold = 0
                while ( all_call_data_4wk is None or (call_strike_sold is not None and call_strike_sold < close_price) or (premium_sold-premium_buy) < call_loss_compensate) and week_target < MAX_ROLL_WEEK:
                    expiration_target = current_day + timedelta(weeks=week_target)
                    expiration_str_target = expiration_target.strftime("%Y-%m-%d")
                    all_call_data_4wk, all_put_data_4wk, call_options_4wk ,put_options_4wk = await pull_option_chain_data(ticker, expiration_str_target, as_of_str, close_price)
                    week_target+=1
                    if all_call_data_4wk is not None:
                        if week_target >= MAX_ROLL_WEEK: # reduce premium target if it is reaching the target week limit
                            premium_derating = 1
                            while call_strike_sold is None or (call_strike_sold is not None and call_strike_sold < close_price ) :
                                call_strike_sold, premium_sold = find_closest_premium_strike(
                                    call_options_4wk, all_call_data_4wk, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate)*premium_derating,stock_price, VOL_THRESHOLD_BYPASS=True
                                )
                                call_strike_bought, premium_buy = find_closest_premium_strike(
                                    call_options_4wk, all_call_data_4wk, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate)*premium_derating/2, stock_price,VOL_THRESHOLD_BYPASS=True
                                )
                                expire_day = expiration_target
                                premium_derating -= 0.05
                                if premium_derating <= 0:
                                    break
                        else:
                            call_strike_sold, premium_sold = find_closest_premium_strike(
                                call_options_4wk, all_call_data_4wk, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate), stock_price,VOL_THRESHOLD_BYPASS=True
                            )
                            call_strike_bought, premium_buy = find_closest_premium_strike(
                                call_options_4wk, all_call_data_4wk, max(max( adjusted_target_premium_call, 0.01), call_loss_compensate)/2, stock_price,VOL_THRESHOLD_BYPASS=True
                            )
                            expire_day = expiration_target
                    else:
                        print(f"all call data is None for {ticker} ({week_target}wk {expiration_str_target}) on {current_day.date()}")
                    earnings_dates = get_earnings_dates(ticker, current_day.strftime("%Y-%m-%d"),(current_day + timedelta(weeks=week_target)).strftime("%Y-%m-%d"))
                    if earnings_dates is not None:
                        print(f"SKIP due to earning, cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expire_day} call strike to sell: {call_strike_sold}, call strike to buy: {call_strike_bought} compensate: {call_loss_compensate*0.8:.2f}")
                        break
                if week_target >= MAX_ROLL_WEEK:
                    print(f"Cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expire_day} call strike to sell: {call_strike_sold}, premium: {premium_sold}, call strike to buy: {call_strike_bought} premium: {premium_buy}, net: {premium_sold-premium_buy} compensate: {call_loss_compensate*0.8:.2f}")
                else:
                    print(f"Target call strike found for {ticker} ({week_target-1}wk expiration {expire_day}), current friday: {current_day.date()}")
        # For put loss
        put_loss_compensate = 0
        put_loss_this = 0   
        for entry in put_loss:
            loss = entry.get("loss")
            put_open_state = entry.get("open_state")
            # if loss > 0 and put_open_state == 'otm':
            if loss > 0:
                put_loss_compensate += loss
            put_loss_this += loss
        if put_loss_compensate > 0:
            if roll_method == 'close price':
                week_target = 1
                while put_strike_sold is None and week_target < MAX_ROLL_WEEK:
                    put_strikes_list = [opt['strike_price'] for opt in put_options]
                    put_strike_sold = find_closest_strike(put_strikes_list, close_price)
                    put_strike_bought = find_closest_strike(put_strikes_list, close_price*0.9)
                    week_target+=1
                    expiration_target = current_day + timedelta(weeks=week_target)
                    expiration_str = expiration_target.strftime("%Y-%m-%d")
                    _, all_put_data, _ ,put_options = await pull_option_chain_data(ticker, expiration_str, as_of_str, close_price, force_otm=True)
                if week_target == MAX_ROLL_WEEK:
                    print(f"Cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expiration_str} put strike to sell: {put_strike_sold}, put strike to buy: {put_strike_bought} compensate: {put_loss_compensate*0.8:.2f}")
            elif roll_method == 'loss':
                put_strike_sold,_ = find_closest_premium_strike(put_options, all_put_data, max(max( adjusted_target_premium_put,0.01), put_loss_compensate)*0.5, stock_price,VOL_THRESHOLD_BYPASS=True)
                put_strike_bought,_ = find_closest_premium_strike(put_options, all_put_data, max(max( adjusted_target_premium_put,0.01), put_loss_compensate), stock_price,VOL_THRESHOLD_BYPASS=True)
            elif roll_method == 'roll':
                # New roll_method for put side using 4-week options
                week_target = 1
                all_put_data_4wk = None
                put_strike_sold = 0
                while ( all_put_data_4wk is None or (put_strike_sold is not None and put_strike_sold > close_price) or (premium_sold - premium_buy) < put_loss_compensate) and week_target < MAX_ROLL_WEEK:
                    expiration_target = current_day + timedelta(weeks=week_target)
                    expiration_str_target = expiration_target.strftime("%Y-%m-%d")
                    as_of_str = current_day.date().strftime("%Y-%m-%d")
                    _, all_put_data_4wk, _, put_options_4wk = await pull_option_chain_data(ticker, expiration_str_target, as_of_str,close_price, force_otm=True)
                    week_target += 1
                    if all_put_data_4wk is not None:
                        if week_target >= MAX_ROLL_WEEK:
                            premium_derating = 1
                            while put_strike_sold is None or (put_strike_sold is not None and put_strike_sold > close_price ):
                                put_strike_sold, premium_sold = find_closest_premium_strike(
                                    put_options_4wk, all_put_data_4wk, max(max( adjusted_target_premium_put,0.01), put_loss_compensate)*premium_derating, stock_price,VOL_THRESHOLD_BYPASS=True
                                )
                                put_strike_bought, premium_buy = find_closest_premium_strike(
                                    put_options_4wk, all_put_data_4wk, max(max( adjusted_target_premium_put,0.01), put_loss_compensate)/2, stock_price,VOL_THRESHOLD_BYPASS=True
                                )
                                expire_day = expiration_target
                                premium_derating -= 0.05
                                if premium_derating <= 0:
                                    break
                        else:
                            put_strike_sold, premium_sold = find_closest_premium_strike(
                                put_options_4wk, all_put_data_4wk, max(max( adjusted_target_premium_put,0.01), put_loss_compensate), stock_price,VOL_THRESHOLD_BYPASS=True
                            )
                            put_strike_bought, premium_buy = find_closest_premium_strike(
                                put_options_4wk, all_put_data_4wk, max(max( adjusted_target_premium_put,0.01), put_loss_compensate)/2, stock_price,VOL_THRESHOLD_BYPASS=True
                            )
                            expire_day = expiration_target
                        print(f"previous result doesn`t meet the requirement for {ticker} {close_price:.2f} {current_day} {expire_day} put strike sold: {put_strike_sold}, premium_sold: {premium_sold}, put strike buy: {put_strike_bought} premium: {premium_buy}, compensate: {put_loss_compensate*0.8:.2f}")
                    else:
                        print(f"all put data is None for {ticker} ({week_target}wk {expiration_str_target}) on {current_day.date()}")

                    earnings_dates = get_earnings_dates(ticker, current_day.strftime("%Y-%m-%d"),(current_day + timedelta(weeks=week_target)).strftime("%Y-%m-%d"))
                    if earnings_dates is not None:
                        print(f"SKIP due to earning, cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expire_day} call strike to sell: {call_strike_sold}, call strike to buy: {call_strike_bought} compensate: {call_loss_compensate*0.8:.2f}")
                        break

                if week_target >= MAX_ROLL_WEEK:
                    print(f"Cannot find the ideal roll option for {ticker} ({week_target}wk expiration) on {current_day.date()}. Settled with {expire_day} put strike to sell: {put_strike_sold}, premium: {premium_sold}, put strike to buy: {put_strike_bought} premium: {premium_buy}, net: {premium_sold-premium_buy} compensate: {put_loss_compensate*0.8:.2f}")
                else:
                    print(f"Target put strike found for {ticker} ({week_target-1}wk expiration {expire_day}), current friday: {current_day.date()}")

        return call_strike_sold, call_strike_bought, put_strike_sold, put_strike_bought, expire_day, call_loss_compensate, put_loss_compensate
        
    async def check_stop_loss(positions, current_date, client, mode="training", stop_loss_percent=0, stop_loss_action="close"):
        """
        Check if any positions need to be closed or rolled due to stop loss triggers
        
        Args:
            positions: List of open position dictionaries
            current_date: Date to check prices against
            client: PolygonAPIClient instance for fetching option prices
            mode: "training" or "validation" mode (affects logging)
            stop_loss_action: "close" to exit position, "roll" to roll to next week
        
        Returns:
            List of positions that had stop losses triggered (closed or rolled)
        """
        if not stop_loss_percent or not positions:
            return []  # No stop loss set or no positions to check

        # Get the current day's close price
        row = hist_df.loc[hist_df["date"].dt.date == current_date.date()]
        if row.empty:
            return []  # No price data for today
            
        close_price = row['close'].iloc[0]
        closed_or_rolled_positions = []

        for position in positions:
            triggered = False
            call_roll_data = {}
            put_roll_data = {}

            if position['expiration'] <= current_date or position['week_start'] >= current_date or position['position_hedged'] is True:
                continue

            # Check call side stop loss
            if position['short_call_prem_open'] > 0 and not position.get('call_closed_by_stop', False):
                call_distance = (position['call_strike_sold'] - close_price) / close_price
                hedge_distance = stop_loss_percent * position['open_distance_call']
                if call_distance < hedge_distance:
                    formatted_date = current_date.strftime("%Y-%m-%d")
                    formatted_exp = position['expiration'].strftime("%Y-%m-%d")
                    
                    # Fetch current premium to determine loss or roll cost
                    call_close_data_short = stored_option_price.get(ticker.upper(), {}).get(formatted_date, {}).get(
                        round(position['call_strike_sold'], 2), {}).get(formatted_exp, {}).get("call", {})
                    
                    call_close_data_long = stored_option_price.get(ticker.upper(), {}).get(formatted_date, {}).get(
                        round(position['call_strike_bought'], 2), {}).get(formatted_exp, {}).get("call", {})
                    
                    if not call_close_data_short:
                        if mode == "validation":
                            print(f"No call close data in cache for {ticker} on {formatted_date} at strike {position['call_strike_sold']} expiring {formatted_exp} - fetching from API")
                        option_ticker = f"O:{ticker.upper()}{formatted_exp[2:].replace('-','')}C{int(position['call_strike_sold'] * 1000):08d}"
                        try:
                            call_close_data_short = await client.query_polygon_for_option_price_async(
                                ticker=ticker,
                                strike_price=position['call_strike_sold'],
                                call_put='call',
                                expiration_date=formatted_exp,
                                pricing_date=formatted_date,
                                option_ticker=option_ticker
                            )
                        except Exception as e:
                            print(f"Error fetching call option data: {e}")
                    if not call_close_data_long:
                        if mode == "validation":
                            print(f"No call close data in cache for {ticker} on {formatted_date} at strike {position['call_strike_bought']} expiring {formatted_exp} - fetching from API")
                        option_ticker = f"O:{ticker.upper()}{formatted_exp[2:].replace('-','')}C{int(position['call_strike_bought'] * 1000):08d}"
                        try:
                            call_close_data_long = await client.query_polygon_for_option_price_async(
                                ticker=ticker,
                                strike_price=position['call_strike_bought'],
                                call_put='call',
                                expiration_date=formatted_exp,
                                pricing_date=formatted_date,
                                option_ticker=option_ticker
                            )
                        except Exception as e:
                            print(f"Error fetching call option data: {e}")

                    premium_field = "close_price" if USE_CLOSE_PRICE_SETTING else "mid_price"
                    call_close_short_prem = call_close_data_short.get(premium_field, 0.0) if call_close_data_short else 0.0
                    call_close_long_prem = call_close_data_long.get(premium_field, 0.0) if call_close_data_long else 0.0

                    if call_close_short_prem > 0 and close_price < position['call_strike_bought'] and position['call_strike_bought'] - position['call_strike_sold'] > 5: # no need to roll if price is already above long call
                        if stop_loss_action == "close":
                            # Close the position
                            call_stop_loss = call_close_short_prem - position['short_call_prem_open']
                            call_stop_loss_amount = call_stop_loss * 100
                            position['call_closed_by_stop'] = True
                            position['call_stop_loss'] = call_stop_loss_amount
                            position['call_stop_date'] = current_date
                            position['call_stop_price'] = close_price
                            if mode == "validation":
                                print(f"\033[91mCall stop loss triggered (closed) for {ticker} on {current_date}! "
                                    f"Strike: {position['call_strike_sold']}, Current price: {close_price:.2f}, "
                                    f"Distance: {call_distance:.2%}, Loss: ${call_stop_loss_amount:.2f}\033[0m")
                        elif stop_loss_action in ['roll_out','roll_in','roll_in_opposite']:
                            # Roll to next week's expiration

                            if stop_loss_action == 'roll_out':
                                next_expiration = position['expiration'] + timedelta(weeks=1)
                            elif stop_loss_action in ['roll_in','roll_in_opposite']:
                                next_expiration = position['expiration']
                            while next_expiration.weekday() != 4:  # Adjust to next Friday
                                next_expiration += timedelta(days=1)
                            next_exp_str = next_expiration.strftime("%Y-%m-%d")
                        
                            all_call_data = None
                            while all_call_data is None:
                                # Fetch option chain for next expiration
                                all_call_data, all_put_data, call_options, put_options = await pull_option_chain_data(
                                    ticker, next_exp_str, formatted_date, close_price, force_otm=False
                                )

                                if all_call_data is None:
                                    next_expiration += timedelta(weeks=1)
                                    while next_expiration.weekday() != 4:  # Adjust to next Friday
                                        next_expiration += timedelta(days=1)
                                    next_exp_str = next_expiration.strftime("%Y-%m-%d")
                            
                            if all_call_data:
                                new_option_type = None
                                if stop_loss_action == 'roll_in':
                                    new_strike_opposite, new_premium_opposite = find_closest_premium_strike(
                                        put_options, all_put_data, None, close_price*(1-hedge_distance*1.5), VOL_THRESHOLD_BYPASS=True
                                    )
                                    new_strike_buy_opposite, new_premium_buy_opposite = find_closest_premium_strike(
                                        put_options, all_put_data, None, new_strike_opposite - iron_condor_width, VOL_THRESHOLD_BYPASS=True
                                    )
                                    roll_cost = ( - new_premium_opposite + new_premium_buy_opposite) * 100
                                # if stop_loss_action == 'roll_in':
                                    new_strike, new_premium = find_closest_premium_strike(
                                        call_options, all_call_data, None, close_price*(1+hedge_distance*1.5), VOL_THRESHOLD_BYPASS=True
                                    )
                                    new_strike_buy, new_premium_buy = find_closest_premium_strike(
                                        call_options, all_call_data, None, new_strike + iron_condor_width, VOL_THRESHOLD_BYPASS=True
                                    )
                                    new_option_type = 'call'
                                    roll_cost += ( call_close_short_prem - call_close_long_prem - new_premium + new_premium_buy) * 100

                                new_call_strike_buy = new_strike_buy if new_option_type == 'call' else 0
                                new_call_strike = new_strike if new_option_type == 'call' else 0
                                new_put_strike = new_strike_opposite
                                new_put_strike_buy = new_strike_buy_opposite
                                short_put_prem_open = new_premium_opposite
                                long_put_prem_open = new_premium_buy_opposite
                                short_call_prem_open = new_premium 
                                long_call_prem_open = new_premium_buy 

                                print(f"previous spread: {position['call_strike_sold']}<->{position['call_strike_bought']}, {position['put_strike_sold']}<->{position['put_strike_bought']}, ")
                                print(f"new spread: {new_strike}<->{new_strike_buy}, {new_strike_opposite}<->{new_strike_buy_opposite}), ")
                                print(f"cost to close: ${call_close_short_prem:.2f}<->${call_close_long_prem:.2f} ")
                                print(f"premium for open: ${new_premium:.2f}<->${new_premium_buy:.2f}), ${new_premium_opposite:.2f}<->${new_premium_buy_opposite:.2f}")

                                if new_strike and ( ( stop_loss_action == 'roll_in' and new_strike > position['call_strike_sold'] ) or stop_loss_action == 'roll_in_opposite') :
                                    call_roll_data = {
                                        'new_strike': new_strike,
                                        'new_strike_buy': new_strike_buy,
                                        'new_premium': new_premium,
                                        'new_premium_buy': new_premium_buy,
                                        'new_expiration': next_expiration,
                                        'roll_cost': round(roll_cost,2),
                                        'old_strike': position['call_strike_sold'],
                                        'old_premium': call_close_short_prem
                                    }
                                    position['call_roll_data'] = call_roll_data
                                    # Create new position for rolled call
                                    new_position = {
                                        'week_start': current_date,
                                        'expiration': next_expiration,
                                        'put_strike_sold': new_put_strike,
                                        'put_strike_bought': new_put_strike_buy,
                                        'call_strike_sold': new_call_strike,
                                        'call_strike_bought': new_call_strike_buy,
                                        'short_put_prem_open': short_put_prem_open,
                                        'long_put_prem_open': long_put_prem_open,
                                        'short_call_prem_open': short_call_prem_open,
                                        'long_call_prem_open': long_call_prem_open,
                                        'put_closed_by_stop': False,
                                        'call_closed_by_stop': False,
                                        'position_hedged': False,
                                        'open_distance_call': (new_call_strike - close_price) / close_price,
                                        'open_distance_put': (close_price - new_put_strike) / close_price,
                                    }
                                    positions.append(new_position)
                                    # Set original call premiums to zero
                                    if stop_loss_action == 'roll_in':
                                        position['short_call_prem_open'] = 0
                                        position['long_call_prem_open'] = 0
                                    if mode == "validation":
                                        print(f"\033[92mCall stop loss triggered (rolled) for {ticker} on {current_date}! close: {close_price:.2f} "
                                            f"Old Strike: {position['call_strike_sold']} -> New Strike: {new_strike}, "
                                            f"Cost: ${roll_cost:.2f}\033[0m")
                                else:
                                    print(f"No suitable OTM call strike for rolling {ticker} on {current_date}, new strike: {new_strike}, old strike: {position['call_strike_sold']}")
                            else:
                                print(f"No call option chain data for rolling {ticker} on {next_exp_str}")
                        triggered = True
            # Check put side stop loss
            if position['short_put_prem_open'] > 0 and not position.get('put_closed_by_stop', False):
                put_distance = (close_price - position['put_strike_sold']) / close_price
                hedge_distance = stop_loss_percent * position['open_distance_put']
                if put_distance < hedge_distance:
                    formatted_date = current_date.strftime("%Y-%m-%d")
                    formatted_exp = position['expiration'].strftime("%Y-%m-%d")
                    put_close_data_short = stored_option_price.get(ticker.upper(), {}).get(formatted_date, {}).get(
                        round(position['put_strike_sold'], 2), {}).get(formatted_exp, {}).get("put", {})
                    put_close_data_long = stored_option_price.get(ticker.upper(), {}).get(formatted_date, {}).get(
                        round(position['put_strike_bought'], 2), {}).get(formatted_exp, {}).get("put", {})
                    if not put_close_data_short:
                        if mode == "validation":
                            print(f"No put close data in cache for {ticker} on {formatted_date} at strike {position['put_strike_sold']} - fetching from API")
                        option_ticker = f"O:{ticker.upper()}{formatted_exp[2:].replace('-','')}P{int(position['put_strike_sold'] * 1000):08d}"
                        try:
                            put_close_data_short = await client.query_polygon_for_option_price_async(
                                ticker=ticker,
                                strike_price=position['put_strike_sold'],
                                call_put='put',
                                expiration_date=formatted_exp,
                                pricing_date=formatted_date,
                                option_ticker=option_ticker
                            )
                        except Exception as e:
                            print(f"Error fetching put option data: {e}")
                    if not put_close_data_long:
                        if mode == "validation":
                            print(f"No put close data in cache for {ticker} on {formatted_date} at strike {position['put_strike_bought']} - fetching from API")
                        option_ticker = f"O:{ticker.upper()}{formatted_exp[2:].replace('-','')}P{int(position['put_strike_bought'] * 1000):08d}"
                        try:
                            put_close_data_long = await client.query_polygon_for_option_price_async(
                                ticker=ticker,
                                strike_price=position['put_strike_bought'],
                                call_put='put',
                                expiration_date=formatted_exp,
                                pricing_date=formatted_date,
                                option_ticker=option_ticker
                            )
                        except Exception as e:
                            print(f"Error fetching put option data: {e}")
                    premium_field = "close_price" if USE_CLOSE_PRICE_SETTING else "mid_price"
                    put_close_short_prem = put_close_data_short.get(premium_field, 0.0) if put_close_data_short else 0.0
                    put_close_long_prem = put_close_data_long.get(premium_field, 0.0) if put_close_data_long else 0.0
                    if put_close_short_prem > 0 and close_price > position['put_strike_bought'] and position['put_strike_bought'] - position['put_strike_sold'] < -5: # no need to roll if price is already below long put
                        if stop_loss_action == "close":
                            # Close the position
                            put_stop_loss = put_close_short_prem - position['short_put_prem_open']
                            put_stop_loss_amount = put_stop_loss * 100
                            position['put_closed_by_stop'] = True
                            position['put_stop_loss'] = put_stop_loss_amount
                            position['put_stop_date'] = current_date
                            position['put_stop_price'] = close_price
                            if mode == "validation":
                                print(f"\033[91mPut stop loss triggered (closed) for {ticker} on {current_date}! "
                                    f"Strike: {position['put_strike_sold']}, Current price: {close_price:.2f}, "
                                    f"Distance: {put_distance:.2%}, Loss: ${put_stop_loss_amount:.2f}\033[0m")
                        elif stop_loss_action in ['roll_out','roll_in','roll_in_opposite']:
                            # Roll to next week's expiration

                            if stop_loss_action == 'roll_out':
                                next_expiration = position['expiration'] + timedelta(weeks=1)
                            elif stop_loss_action in ['roll_in','roll_in_opposite']:
                                next_expiration = position['expiration']
                            while next_expiration.weekday() != 4:  # Adjust to next Friday
                                next_expiration += timedelta(days=1)
                            next_exp_str = next_expiration.strftime("%Y-%m-%d")

                            all_put_data = None
                            while all_put_data is None:
                                # Fetch option chain for next expiration
                                all_call_data, all_put_data, call_options, put_options,= await pull_option_chain_data(
                                    ticker, next_exp_str, formatted_date, close_price, force_otm=False
                                )
                                if all_put_data is None:
                                    next_expiration += timedelta(weeks=1)
                                    while next_expiration.weekday() != 4:  # Adjust to next Friday
                                        next_expiration += timedelta(days=1)
                                    next_exp_str = next_expiration.strftime("%Y-%m-%d")

                            if all_put_data:
                                # Target a strike further OTM (e.g., 5% above current price)
                                new_option_type = None
                                if stop_loss_action == 'roll_in':
                                    new_strike_opposite, new_premium_opposite = find_closest_premium_strike(
                                        call_options, all_call_data, None, close_price*(1+hedge_distance*1.5), VOL_THRESHOLD_BYPASS=True
                                    )
                                    new_strike_buy_opposite, new_premium_buy_opposite = find_closest_premium_strike(
                                        call_options, all_call_data, None, new_strike_opposite + iron_condor_width, VOL_THRESHOLD_BYPASS=True
                                    )
                                    roll_cost = ( - new_premium_opposite + new_premium_buy_opposite) * 100
                                    # new_strike, new_premium = find_closest_premium_strike(
                                    #     call_options, all_call_data, None, position['put_strike_bought'], VOL_THRESHOLD_BYPASS=True, OTM=False
                                    # )
                                    # new_strike_buy, new_premium_buy = find_closest_premium_strike(
                                    #     call_options, all_call_data, None, position['put_strike_sold'], VOL_THRESHOLD_BYPASS=True, OTM=False
                                    # )

                                # if stop_loss_action == 'roll_in':
                                    new_strike, new_premium = find_closest_premium_strike(
                                        put_options, all_put_data, None, close_price*(1-hedge_distance*1.5), VOL_THRESHOLD_BYPASS=True
                                    )
                                    new_strike_buy, new_premium_buy = find_closest_premium_strike(
                                        put_options, all_put_data, None, new_strike - iron_condor_width, VOL_THRESHOLD_BYPASS=True
                                    )
                                    roll_cost += ( put_close_short_prem - put_close_long_prem - new_premium + new_premium_buy) * 100
                                
                                new_call_strike_buy = new_strike_buy_opposite
                                new_call_strike = new_strike_opposite 
                                new_put_strike = new_strike
                                new_put_strike_buy = new_strike_buy
                                short_put_prem_open = new_premium_opposite
                                long_put_prem_open = new_premium_buy_opposite
                                short_call_prem_open = new_premium
                                long_call_prem_open = new_premium_buy

                                print(f"previous spread: {position['call_strike_sold']}<->{position['call_strike_bought']}, {position['put_strike_sold']}<->{position['put_strike_bought']}, ")
                                print(f"new spread: {new_strike}<->{new_strike_buy}, {new_strike_opposite}<->{new_strike_buy_opposite}), ")
                                print(f"cost to close: ${put_close_short_prem:.2f}<->${put_close_long_prem:.2f} ")
                                print(f"premium for open: ${new_premium:.2f}<->${new_premium_buy:.2f}), ${new_premium_opposite:.2f}<->${new_premium_buy_opposite:.2f}")
                                
                                # Check if the new strike is OTM and meets the criteria for rolling
                                if new_strike and ( ( new_strike < position['put_strike_sold'] and stop_loss_action == 'roll_in' ) or stop_loss_action == 'roll_in_opposite') :
                                    put_roll_data = {
                                        'new_strike': new_strike,
                                        'new_strike_buy': new_strike_buy,
                                        'new_premium': new_premium,
                                        'new_premium_buy': new_premium_buy,
                                        'new_expiration': next_expiration,
                                        'roll_cost': round(roll_cost,2),
                                        'old_strike': position['put_strike_sold'],
                                        'old_premium': put_close_short_prem
                                    }
                                    position['put_roll_data'] = put_roll_data
                                    # Create new position for rolled put
                                    new_position = {
                                        'week_start': current_date,
                                        'expiration': next_expiration,
                                        'put_strike_sold': new_put_strike,
                                        'put_strike_bought': new_put_strike_buy,
                                        'call_strike_sold': new_call_strike,
                                        'call_strike_bought': new_call_strike_buy,
                                        'short_put_prem_open': short_put_prem_open,
                                        'long_put_prem_open': long_put_prem_open,
                                        'short_call_prem_open': short_call_prem_open,
                                        'long_call_prem_open': long_call_prem_open,
                                        'put_closed_by_stop': False,
                                        'call_closed_by_stop': False,
                                        'position_hedged': False,
                                        'open_distance_call': (new_call_strike - close_price) / close_price,
                                        'open_distance_put': (close_price - new_put_strike) / close_price,  
                                    }
                                    positions.append(new_position)
                                    # Set original put premiums to zero
                                    if stop_loss_action == 'roll_in': # null the original position if roll in
                                        position['short_put_prem_open'] = 0
                                        position['long_put_prem_open'] = 0
                                    if mode == "validation":
                                        print(f"\033[92mput stop loss triggered (rolled) for {ticker} on {current_date}! close: {close_price:.2f} "
                                            f"Old Strike: {position['put_strike_sold']} -> New Strike: {new_strike}, "
                                            f"Cost: ${roll_cost:.2f}\033[0m")
                                else:
                                    print(f"No suitable OTM put strike for rolling {ticker} on {current_date},new_strike: {new_strike}, old_strike: {position['put_strike_sold']}")
                            else:
                                print(f"No put option chain data for rolling {ticker} on {next_exp_str}")
                        triggered = True

            if triggered:
                closed_or_rolled_positions.append(position)
        return closed_or_rolled_positions

    #------------------ Get Expiration Dates ------------------
    for i in range(len(expire_day_list)):
        current_day = expire_day_list[i]
        if i == len(expire_day_list) - expiring_wks:
            print(f"{ticker} {current_day} is the {expiring_wks} weeks from the last trading day, skipping....")
            break
        expire_day = expire_day_list[i + expiring_wks]

        trade_date_str = fetch_pricing_date(day_of_week,current_day)

        # Get close price & daily_vol
        row_expire = hist_df.loc[hist_df["date"] == current_day.strftime("%Y-%m-%d")]
        row_trade = hist_df.loc[hist_df["date"] == trade_date_str]
        if row_trade.empty:
            close_price_pricing = None
        else:
            close_price_pricing = row_trade['close'].iloc[0]
        if row_expire.empty:
            close_price_expiration = None
        else:
            close_price_expiration = row_expire['close'].iloc[0]

        if row_trade.empty or row_expire.empty:
            if row_trade.empty:
                this_skip_reason = "NO_CLOSE_PRICE_AT_TRADE"
            else:
                this_skip_reason = "NO_CLOSE_PRICE_AT_EXPIRE"
            print(f"No valid close price data: {ticker} {current_day}")
            needed_data.append({
                "week_start": current_day,
                "pricing_date": trade_date_str,
                "expiration": expire_day,
                "as_of_date": trade_date_str,
                "close_price_expiration": close_price_expiration,
                "close_price_pricing": close_price_pricing,
                "call_strike_sold": None,
                "put_strike_sold": None,
                "days_to_expire": None,
                "call_strike_method": "percent",
                "put_strike_method": "percent",
                "skip_reason": this_skip_reason
            })
            continue

        skip_earnings = any(
            current_day.date() <= ed < expire_day.date()
            for ed in earnings_dates
        )
        skip_earnings = False

        if skip_earnings: #TBD need to check for PnL
            needed_data.append({
                "week_start": current_day,
                "pricing_date": trade_date_str,
                "expiration": expire_day,
                "as_of_date": trade_date_str,
                "close_price_expiration": close_price_expiration,
                "close_price_pricing": close_price_pricing,
                "call_strike_sold": None,
                "put_strike_sold": None,
                "days_to_expire": None,
                "call_strike_method": "percent",
                "put_strike_method": "percent",
                "skip_reason": "SKIP_EARNINGS"
            })
            continue
        
        pricing_date_dt = datetime.strptime(trade_date_str, "%Y-%m-%d")
        days_to_expire = (expire_day - pricing_date_dt).days
        sqrt_d = math.sqrt(days_to_expire)

        call_strike_sold = 0
        put_strike_sold = 0  

        # Convert trade_date_str to datetime for VIX lookup
        trade_date_dt = pd.to_datetime(trade_date_str)
        # Get the VIX closing value for the trading day
        vix_value = vix_close_series.get(trade_date_dt, None)

        # Check if VIX data is available; skip the trade if not
        if vix_value is None:
            vix_value = 0

        # Adjust target premiums based on VIX/20
        VIX_THRESHOLD = 50

        if target_premium_call is not None:
            adjusted_target_premium_call = target_premium_call * ( 1 + max((vix_value - VIX_THRESHOLD),0) / 10)
            adjusted_target_premium_put = target_premium_put * ( 1 + max((vix_value - VIX_THRESHOLD),0) / 10)
        else:
            adjusted_target_premium_call = None
            adjusted_target_premium_put = None

        # Optional: Log adjusted premiums in validation mode for debugging
        if mode == "validation" and vix_value > VIX_THRESHOLD:
            print(f"Adjusted target premium call: {adjusted_target_premium_call:.4f}, put: {adjusted_target_premium_put:.4f} for VIX: {vix_value:.2f} on {trade_date_str}")
            
        # -------------- Set Strike Targets --------------
        if ( adjusted_target_premium_call is not None and  adjusted_target_premium_put is not None) or ( target_delta_call is not None and target_delta_put is not None):
            # Target premium logic
            expiration_str = expire_day.strftime("%Y-%m-%d")
            as_of_str = trade_date_str

            all_call_data, all_put_data, call_options ,put_options = await pull_option_chain_data(ticker, expiration_str, as_of_str,close_price_pricing)

            if all_call_data is None or all_put_data is None:
                needed_data.append({
                    "week_start": current_day,
                    "pricing_date": trade_date_str,
                    "expiration": expire_day,
                    "as_of_date": trade_date_str,
                    "close_price_expiration": close_price_expiration,
                    "close_price_pricing": close_price_pricing,
                    "call_strike_sold": None,
                    "put_strike_sold": None,
                    "days_to_expire": None,
                    "call_strike_method": "percent",
                    "put_strike_method": "percent",
                    "skip_reason": "NO_OPTION_CHAIN"
                })
                # print(f"!!!!!!no valid option chain data: {ticker} {current_day}!!!!!!!!!")
                continue

            call_strike_target = {
                'premium_target': adjusted_target_premium_call,
                'delta_target': target_delta_call
            }
            put_strike_target = {
                'premium_target': adjusted_target_premium_put,
                'delta_target': target_delta_put
            }

            call_strike_sold, price_call_strike = find_strike_custom(call_options, all_call_data, call_strike_target, stock_price=close_price_pricing)
            if call_strike_sold is None:
                call_strike_bought = None
                if mode == "validation":
                    print(f"No valid premium-based call strike for {ticker} expire {expiration_str} today {as_of_str} target premium:{call_strike_target},call_options:{len(call_options)}, all_call_data:{len(all_call_data)}, stock_price:{close_price_pricing:.2f}")
            else:
                # call_strike_bought, price_call_strike_bought = find_strike_custom(call_options, all_call_data, call_strike_target, stock_price=call_strike_sold + iron_condor_width)
                call_strike_dict = [row['strike_price'] for row in call_options]
                call_strike_bought = find_closest_strike(call_strike_dict, call_strike_sold + iron_condor_width)

            put_strike_sold, price_put_strike = find_strike_custom(put_options, all_put_data, put_strike_target, stock_price=close_price_pricing)
            if put_strike_sold is None:
                put_strike_bought = None
                if mode == "validation":
                    print(f"No valid premium-based put strike for {ticker} expire {expiration_str} today {as_of_str} target premium:{put_strike_target},put_options:{len(put_options)}, all_put_data:{len(all_put_data)},close_price:{close_price_pricing:.2f}")
            else:
                # put_strike_bought, price_put_strike_bought = find_strike_custom(put_options, all_put_data, put_strike_target, stock_price=put_strike_sold-iron_condor_width)
                put_strike_dict = [row['strike_price'] for row in put_options]
                put_strike_bought = find_closest_strike(put_strike_dict, put_strike_sold - iron_condor_width)
                    # if put_strike_sold < call_strike_sold - 100:
                    #     print(f"premium put strike: {put_strike_sold}, premium put strike bought: {put_strike_bought}, close: {close_price_pricing}")
                    #     _, all_put_data, _ ,put_options = await pull_option_chain_data(ticker, expiration_str, as_of_str,close_price_pricing*0.8, force_update=False)
                    #     put_strike_sold, price_put_strike = find_strike_custom(put_options, all_put_data, put_strike_target, stock_price=close_price_pricing)
                    #     put_strike_bought, price_put_strike_bought = find_strike_custom(put_options, all_put_data, put_strike_target, stock_price=put_strike_sold-iron_condor_width)
                    # print(f"{as_of_str} premium put strike: {put_strike_sold}, premium put strike bought: {put_strike_bought}, close: {close_price_pricing}, premium target: {put_strike_target}")
                    # print(f"{as_of_str} premium call strike: {call_strike_sold}, premium call strike bought: {call_strike_bought}, close: {close_price_pricing}, premium target: {call_strike_target}")
                    # breakpoint()

            # print(f"{as_of_str} {expiration_str} premium put strike: {put_strike_sold},${price_put_strike} premium put strike bought: {put_strike_bought}, ${price_call_strike} close: {close_price_pricing}, premium target: {put_strike_target}")
            # print(f"{as_of_str} {expiration_str} premium call strike: {call_strike_sold},${price_call_strike} premium call strike bought: {call_strike_bought}, ${price_call_strike} close: {close_price_pricing}, premium target: {call_strike_target}")
            # if call_strike_bought is None or put_strike_bought is None:
            #     breakpoint()
            # Track selection method
            call_method = None
            put_method = None

            skip_reason = None
            if call_strike_sold is None or put_strike_sold is None:
                skip_reason = "NO_STRIKES"

            needed_data.append({
                "week_start": current_day,
                "pricing_date": trade_date_str,
                "expiration": expire_day,
                "as_of_date": trade_date_str,
                "close_price_expiration": close_price_expiration,
                "close_price_pricing": close_price_pricing,
                "call_strike_sold": call_strike_sold,
                "put_strike_sold": put_strike_sold,
                "call_strike_bought": call_strike_bought,
                "put_strike_bought": put_strike_bought,
                "days_to_expire": days_to_expire,
                "call_strike_method": call_method,
                "put_strike_method": put_method,
                "skip_reason": skip_reason
            })
        
    if not needed_data:
        # print(f"No valid Friday data for {ticker} after skipping earnings or missing data.")
        return None,None,None,None

    # ------------------------------------------------
    # 2) Collect all expiration_date & as_of_date pairs
    # ------------------------------------------------
    unique_chain_requests = set()  # (expiration_str, as_of_date, call_put)
    for item in needed_data:
        if item['expiration'] is None:
            continue
        exp_str = item["expiration"].strftime("%Y-%m-%d")
        as_of_str = item["as_of_date"]
        # We'll fetch both call & put once, so just store them
        unique_chain_requests.add((exp_str, as_of_str, "call"))
        unique_chain_requests.add((exp_str, as_of_str, "put"))
 
    big_options_to_fetch = []  
    final_friday_data = []
    
    for item in needed_data:        
        skip_reason = item["skip_reason"]
        if skip_reason is not None:
            final_friday_data.append({
                "week_start": item["week_start"] if "week_start" in item else None,
                "expire_day": item["expiration"] if "expiration" in item else None,
                "pricing_date": item["as_of_date"] if "as_of_date" in item else None,
                "close_price_expiration": item["close_price_expiration"],
                "close_price_pricing": item["close_price_pricing"],
                "call_strike_sold": None,
                "put_strike_sold": None,
                "call_strike_bought": None,
                "put_strike_bought": None,
                "call_strike_method": item["call_strike_method"] if "call_strike_method" in item else None,
                "put_strike_method": item["put_strike_method"] if "put_strike_method" in item else None,
                "skip_reason": skip_reason
            })
            continue
        
        current_day = item["week_start"]
        expire_day     = item["expiration"]
        as_of_str      = item["as_of_date"]
        close_price_expiration = float(item["close_price_expiration"])
        close_price_pricing    = float(item["close_price_pricing"])
        call_strike_sold = item["call_strike_sold"]
        put_strike_sold  = item["put_strike_sold"]
        call_strike_bought = item["call_strike_bought"]
        put_strike_bought  = item["put_strike_bought"]
        call_method = item["call_strike_method"]
        put_method = item["put_strike_method"]

        expiration_str = expire_day.strftime("%Y-%m-%d")

        # Adjust pricing_date based on day_of_week input

        trade_date_str = fetch_pricing_date(day_of_week,current_day)

        call_strike = call_strike_sold
        put_strike = put_strike_sold
        call_strike_bought = call_strike_bought
        put_strike_bought = put_strike_bought

        trade_options = [
            {
                'strike_price': call_strike,
                'call_put': 'call',
                'expiration_date': expiration_str,
                'quote_timestamp': trade_date_str
            },
            {
                'strike_price': put_strike,
                'call_put': 'put',
                'expiration_date': expiration_str,
                'quote_timestamp': trade_date_str
            },
            {
                'strike_price': call_strike_bought,
                'call_put': 'call',
                'expiration_date': expiration_str,
                'quote_timestamp': trade_date_str
            },
            {
                'strike_price': put_strike_bought,
                'call_put': 'put',
                'expiration_date': expiration_str,
                'quote_timestamp': trade_date_str
            }
        ]
        # Check cache and separate options into cached and to-fetch
        options_to_fetch_filtered = []

        for option in trade_options:
            strike = option['strike_price']
            call_put_type = option['call_put']
            expiration = option['expiration_date']
            pricing_date = option['quote_timestamp']

            if strike is None:
                print(f"Strike is None for {ticker} on {current_day.date()}")
            # Fetch from stored_option_price
            data = stored_option_price.get(ticker.upper(), {}).get(pricing_date, {}).get(round(strike,2), {}).get(expiration, {}).get(call_put_type.lower(), {})
            if data:
                logging.debug(f"Option data already cached for {ticker}, Strike: {strike}, Type: {call_put_type}, "
                              f"Expiration: {expiration}, Pricing Date: {pricing_date}: {data}")
            else:
                # Price not cached; need to fetch
                options_to_fetch_filtered.append(option)
                logging.debug(f"Option data not cached for {ticker}, Strike: {strike}, Type: {call_put_type}, "
                              f"Expiration: {expiration}, Pricing Date: {pricing_date}. Marked for fetching.")

        # Add to final_friday_data regardless of whether prices are cached or need to be fetched
        final_friday_data.append({
            "week_start": current_day,
            "expire_day": expire_day,
            "pricing_date": trade_date_str,
            "close_price_expiration": close_price_expiration,
            "close_price_pricing": close_price_pricing,
            "call_strike_sold": call_strike,
            "put_strike_sold": put_strike,
            "call_strike_bought": call_strike_bought,
            "put_strike_bought": put_strike_bought,
            "call_strike_method": call_method,
            "put_strike_method": put_method,
            "skip_reason": None
        })

        big_options_to_fetch.extend(options_to_fetch_filtered)

    if not final_friday_data:
        print(f"No final data for {ticker}. Possibly chain fetch failed or no matches.")
        return None,None,None,None

    # ------------------------------------------------
    # 5) Kick off the async query for all needed quotes
    # ------------------------------------------------
    # Remove duplicates from big_options_to_fetch if you want
    deduped_options_to_fetch = []
    seen = set()
    for opt in big_options_to_fetch:
        key = (opt['strike_price'], opt['call_put'], opt['expiration_date'], opt['quote_timestamp'])
        if key not in seen:
            deduped_options_to_fetch.append(opt)
            seen.add(key)

    if len(deduped_options_to_fetch) > 0:
        print("Start querying option price batch: ", ticker, len(deduped_options_to_fetch))

    if use_async:
        fetched_option_data = await client.get_option_prices_batch_async(ticker, deduped_options_to_fetch)
    else:
        fetched_option_data = get_option_prices_batch_sync(ticker, deduped_options_to_fetch)

    # ------------------------------------------------
    # 6) Now compute PnL for each Friday
    # ------------------------------------------------
    total_pnl = 0.0
    weekly_results = []
    arrow_data = []
    weekly_dates = []
    cumulative_pnls  = []

    # if carry_over_weekly_results is not None:
    #     for week_result in carry_over_weekly_results:
    #         if week_result['expiration'] is not None and week_result['expiration'] >= expire_day_list[0]:
    #             weekly_results.append(week_result)
    #             print(f"carry over week_result: {week_result['expiration']},{week_result['call_strike_sold']},{week_result['put_strike_sold']},{week_result['weekly_pnl']}")

    for i in range(len(expire_day_list)):
        current_day = expire_day_list[i]
        if i == len(expire_day_list) - expiring_wks:
            break
        # Clear expired positions from active_positions list
        active_positions = [pos for pos in active_positions 
                        if pos['expiration'] >= current_day]
        item = None
        for this_item in final_friday_data:
            if this_item["week_start"] == expire_day_list[i]:
                item = this_item
                break
        if item is None:
            print(f"Item not found for {expire_day_list[i]}")
            breakpoint()

        weekly_pnl = 0.0
        weekly_open_credit_amt = 0
        weekly_expiration_payoff = 0
        required_margin = 0.0
        sc_prem, sp_prem, lc_prem, lp_prem, sc_prem_previous, sp_prem_previous, sc_volume, lc_volume, sp_volume, lp_volume, sc_delta, sp_delta, lc_delta, lp_delta = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
        current_day = item["week_start"]
        expire_day     = item["expire_day"]
        trade_date_str = item["pricing_date"]
        if item["close_price_expiration"] is not None:
            close_price_expiration = float(item["close_price_expiration"])
        else:
            close_price_expiration = None
        if item["close_price_pricing"] is not None:
            close_price_pricing = float(item["close_price_pricing"])
        else:   
            close_price_pricing = None
        call_strike    = item["call_strike_sold"]
        put_strike     = item["put_strike_sold"]
        call_strike_bought= item["call_strike_bought"]
        put_strike_bought = item["put_strike_bought"]
        call_method = item["call_strike_method"]
        put_method = item["put_strike_method"]
        skip_reason = item["skip_reason"]

        # After opening positions for a week, add this code to check for stop losses on each trading day
        call_stop_loss, put_stop_loss, call_roll_cost, put_roll_cost = 0,0,0,0
        if stop_loss_percent is not None and len(active_positions) > 0 and mode == "validation":
            # Generate a list of all trading days between position opening and expiration
            earliest_day = min(pos['week_start'] for pos in active_positions)
            latest_expiry = max(pos['expiration'] for pos in active_positions)
            
            # Generate a list of all trading days between the earliest open and latest expiration
            trading_days = pd.date_range(start=earliest_day, end=latest_expiry, freq='B').to_pydatetime()
            # For each trading day, check active positions for stop loss triggers
            for check_day in trading_days:
                closed_or_rolled_positions = await check_stop_loss(active_positions, check_day, client, mode=mode, stop_loss_percent=stop_loss_percent, stop_loss_action=stop_loss_action)

                for position in closed_or_rolled_positions:
                    call_stop_loss = position.get('call_stop_loss', 0)
                    put_stop_loss = position.get('put_stop_loss', 0)
                    call_roll_cost_this = position.get('call_roll_data', {}).get('roll_cost', 0)
                    put_roll_cost_this = position.get('put_roll_data', {}).get('roll_cost', 0)
                    
                    call_roll_cost += call_roll_cost_this
                    put_roll_cost += put_roll_cost_this

                    # if mode == "validation":
                    #     breakpoint()

                    # Add roll costs to expiration payoff if rolled
                    if stop_loss_action in ['roll_in','roll_in_opposite','roll_out']:
                        print( f"Rolled position on {check_day.date()} for {ticker}: call_roll_cost={call_roll_cost:.2f}, put_roll_cost={put_roll_cost:.2f}, total rolled cost: {call_roll_cost + put_roll_cost:.2f}")
                        for position in active_positions:
                            print(f"{check_day.date()} Remaining active position: {position['call_strike_sold']} {position['expiration']} call sold with prem {position['short_call_prem_open']} {position['put_strike_sold']},{position['expiration']} put sold with prem {position['short_put_prem_open']} ")
                        # breakpoint()
                    # Remove fully closed positions
                    if position.get('call_closed_by_stop', False) and position.get('put_closed_by_stop', False):
                        print(f" Closed both call and put positions on {check_day.date()} for {ticker}: call_stop_loss={call_stop_loss:.2f}, put_stop_loss={put_stop_loss:.2f}")
                        active_positions.remove(position)

        call_loss, put_loss = check_option_assignment(active_positions,ticker,expire_day_list[i],mode=mode)
        
        cumulative_call_loss = weekly_results[-1]['cumulative_call_loss'] if len(weekly_results) > 0 else 0
        cumulative_put_loss = weekly_results[-1]['cumulative_put_loss'] if len(weekly_results) > 0 else 0
        call_loss_this = 0
        put_loss_this = 0
        for loss in call_loss:
            cumulative_call_loss += loss['loss']
            call_loss_this += loss['loss']
        for loss in put_loss:
            cumulative_put_loss += loss['loss']
            put_loss_this += loss['loss']

        if ( call_loss_this <= 0 and put_loss_this <=0 ) and skip_reason is not None:
            weekly_results.append({
                "week_start": expire_day_list[i],
                "trade_day": trade_date_str,
                "expiration": None,
                "close_price_expiration": close_price_expiration,
                "close_price_pricing": close_price_pricing,
                "call_strike_sold": 0,
                "put_strike_sold": 0,
                "call_strike_bought": 0,
                "put_strike_bought": 0,
                "short_call_prem_open": 0,
                "short_put_prem_open": 0,
                "long_call_prem_open": 0,
                "long_put_prem_open": 0,
                "short_call_volume": 0,
                "short_put_volume": 0,
                "long_call_volume": 0,
                "long_put_volume": 0,
                "open_distance_call": 0,
                "open_distance_put": 0,
                "weekly_open_credit_amt": 0,
                "weekly_expiration_payoff": 0,
                "weekly_pnl": 0,
                "short_call_delta": 0,
                "short_put_delta": 0,
                "long_call_delta": 0,
                "long_put_delta": 0,
                "cumulative_pnl": weekly_results[-1]['cumulative_pnl'] if len(weekly_results) > 0 else 0,
                "previous_call_assigned": (call_loss_this > 0),
                "previous_put_assigned": (put_loss_this > 0),
                "call_strike_method": "percent",
                "put_strike_method": "percent",
                "required_margin":0.0,
                "return_annualized":0,
                "cumulative_call_loss": cumulative_call_loss,
                "cumulative_put_loss": cumulative_put_loss,
                "skip_reason": skip_reason,
                "call_problems": None,
                "put_problems": None,
            })
            continue
        
        expiration_str = expire_day.strftime("%Y-%m-%d")
        as_of_str = current_day.strftime("%Y-%m-%d")
        call_problems = []
        put_problems = []
        close_price_at_trade = close_price_pricing

        if ( call_loss_this > 0 or put_loss_this > 0 ) and roll_method is not None:
            print(f"{ticker} {current_day} {expire_day} call loss: {call_loss_this}, put loss: {put_loss_this}, roll method: {roll_method}")
            breakpoint()
            if mode == "validation":
                print(f"Generating roll option for {ticker} {current_day} {expire_day} call loss: {call_loss_this}, put loss: {put_loss_this}, roll method: {roll_method}")
            call_strike_sold_roll, call_strike_bought_roll, put_strike_sold_roll, put_strike_bought_roll, expire_day_roll, call_loss_compensate, put_loss_compensate = await generate_roll_option(ticker,expiration_str,as_of_str,close_price_expiration,call_loss,put_loss, stock_price=close_price_expiration)
            if call_strike_sold_roll is not None and call_strike_bought_roll is not None:
                call_strike = call_strike_sold_roll
                call_method = "premium"
                call_strike_bought = call_strike_bought_roll
                put_problems.append(f"Call loss")
                expire_day = expire_day_roll
                close_price_at_trade = close_price_expiration
            if put_strike_sold_roll is not None and put_strike_bought_roll is not None:
                put_strike = put_strike_sold_roll
                put_method = "premium" 
                put_strike_bought = put_strike_bought_roll
                call_problems.append(f"Put loss")
                expire_day = expire_day_roll
                close_price_at_trade = close_price_expiration
            if call_strike is not None or put_strike is not None:
                trade_date_str = current_day.strftime("%Y-%m-%d")  #start the trade at week start (Friday) if there is opened option assigned
                trade_date_previous_str = trade_date_str   #skip checking for IV decrease if need to roll for loss
        else:
            call_loss_compensate, put_loss_compensate = 0, 0
            # trade_date_str = fetch_pricing_date(day_of_week,current_day)
            week_day = datetime.strptime(trade_date_str, '%Y-%m-%d').weekday()
            if week_day == 0:
                trade_date_previous_str = (datetime.strptime(trade_date_str, '%Y-%m-%d')-timedelta(days=3)).strftime('%Y-%m-%d')
            else:
                trade_date_previous_str = (datetime.strptime(trade_date_str, '%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d')

        premium_field = "close_price" if USE_CLOSE_PRICE_SETTING else "mid_price"
        delta_field = "close_price_delta" if USE_CLOSE_PRICE_SETTING else "mid_price_delta"

        if call_strike is not None:
            sc_data = stored_option_price.get(ticker.upper(), {}).get(trade_date_str, {}).get(round(call_strike,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("call", {})
            lc_data = stored_option_price.get(ticker.upper(), {}).get(trade_date_str, {}).get(round(call_strike_bought,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("call", {})
            sc_data_previous = stored_option_price.get(ticker.upper(), {}).get(trade_date_previous_str, {}).get(round(call_strike,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("call", {})
            sc_prem = sc_data.get(premium_field, 0.0) if sc_data else 0.0
            lc_prem = lc_data.get(premium_field, 0.0) if lc_data else 0.0
            sc_delta = sc_data.get(delta_field, 0.0) if sc_data else 0.0
            lc_delta = lc_data.get(delta_field, 0.0) if lc_data else 0.0
            sc_prem_previous = sc_data_previous.get(premium_field, 0.0) if sc_data_previous else 0.0
            sc_volume = sc_data.get("close_volume", 0) if sc_data else 0
            lc_volume = lc_data.get("close_volume", 0) if lc_data else 0
        else:
            sc_prem = 0
            sc_volume = 0
        if put_strike is not None:
            sp_data = stored_option_price.get(ticker.upper(), {}).get(trade_date_str, {}).get(round(put_strike,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("put", {})
            lp_data = stored_option_price.get(ticker.upper(), {}).get(trade_date_str, {}).get(round(put_strike_bought,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("put", {})
            sp_data_previous = stored_option_price.get(ticker.upper(), {}).get(trade_date_previous_str, {}).get(round(put_strike,2), {}).get(expire_day.strftime("%Y-%m-%d"), {}).get("put", {})
            sp_prem_previous = sp_data_previous.get(premium_field, 0.0) if sp_data_previous else 0.0
            sp_prem = sp_data.get(premium_field, 0.0) if sp_data else 0.0
            lp_prem = lp_data.get(premium_field, 0.0) if lp_data else 0.0
            sp_delta = sp_data.get(delta_field, 0.0) if sp_data else 0.0
            lp_delta = lp_data.get(delta_field, 0.0) if lp_data else 0.0
            sp_volume = sp_data.get("close_volume", 0) if sp_data else 0
            lp_volume = lp_data.get("close_volume", 0) if lp_data else 0
        else:
            sp_prem = 0
            sp_volume = 0

        if mode == "validation":
            if sc_volume < VOL_THRESHOLD:
                random_cost_sc = random.uniform(0.01, 1)
            else:
                random_cost_sc = random.uniform(0.01, SPREAD_COST/4)
            if sp_volume < VOL_THRESHOLD:
                random_cost_sp = random.uniform(0.01, 1)
            else:
                random_cost_sp = random.uniform(0.01, SPREAD_COST/4)
            if lc_volume < VOL_THRESHOLD:
                random_cost_lc = random.uniform(0.01, 1)
            else:
                random_cost_lc = random.uniform(0.01, SPREAD_COST/4)
            if lp_volume < VOL_THRESHOLD:
                random_cost_lp = random.uniform(0.01, 1)
            else:
                random_cost_lp = random.uniform(0.01, SPREAD_COST/4)
        else:
            random_cost_sc, random_cost_sp, random_cost_lc, random_cost_lp = 0.0, 0.0, 0.0, 0.0

        # Booleans to track if each side is enabled
        call_enabled = True
        put_enabled  = True

                    # if ( 0 < sc_prem < MIN_PROFIT or 0 < sp_prem < MIN_PROFIT or sc_prem > MAX_PROFIT or sp_prem > MAX_PROFIT ) and mode == "training":
                    #     skip_episode_count += 1
                    # # elif MIN_PROFIT < sc_prem and sc_prem < MAX_PROFIT and MIN_PROFIT < sp_prem and MAX_PROFIT > sp_prem:
                    # #     skip_episode_count -= 1
                    #     print(f"skip episode count: {skip_episode_count},{current_day} sc_prem: {sc_prem:.2f},sp_prem: {sp_prem:.2f},lc_prem: {lc_prem:.2f},lp_prem: {lp_prem:.2f}")
                    # if skip_episode_count > 20:
                    #     return None, None, None, None 
        
                # sc_prem = sc_prem * (1 - random_cost_sc)
                # sp_prem = sp_prem * (1 - random_cost_sp)
                # lc_prem = lc_prem * (1 + random_cost_lc)
                # lp_prem = lp_prem * (1 + random_cost_lp)

        if trade_type in ['iron_condor','single_spread']:
            if sp_prem <= 0:
                put_problems.append("Premium=0")
            if sp_prem <= lp_prem and sp_prem + lp_prem > 0:
                # put_problems.append("Short premium <= Long premium")
                print(f"\033[91m Data corruption: {ticker} {current_day} close: ${close_price_pricing} sp_prem: {sp_prem:.4f}, put strike sold: {put_strike:.2f}, lp_prem: {lp_prem:.4f} put strike bought: {put_strike_bought:.2f} \033[0m")
                # breakpoint()
                lp_prem = sp_prem 
                lp_prem = 0 
            # if sp_prem_previous != 0 and sp_prem != 0 and sp_prem <= sp_prem_previous * 0.2:
            #     put_problems.append("IV decrease")
            if sc_prem <= 0:
                call_problems.append("Premium=0")
            # if sc_volume < VOL_THRESHOLD and call_loss_compensate <= 0:
            #     call_problems.append(f"Volume below threshold,{sc_volume}")
            # if lc_volume < VOL_THRESHOLD and call_hedge > 1 and call_loss_compensate <= 0:
            #     call_problems.append("Hedge volume below threshold")
            if sc_prem <= lc_prem and sc_prem + lc_prem > 0:
                # call_problems.append("Short premium <= Long premium")
                print(f"\033[91m Data corruption: {ticker} {current_day} close: ${close_price_pricing} sc_prem: {sc_prem:.2f}, call strike sold: {call_strike:.2f}, lc_prem: {lc_prem:.2f} call strike bought: {call_strike_bought:.2f} \033[0m")
                # breakpoint()
                lc_prem = sc_prem
                lc_prem = 0
            # if sc_prem_previous != 0 and sc_prem != 0 and sc_prem <= sc_prem_previous * 0.2:
            #     call_problems.append("IV decrease")
            

        if trade_type == 'long_call_only':
            sc_prem = 0.0
            sp_prem = 0.0
            lp_prem = 0.0
        elif trade_type == 'long_put_only':
            sc_prem = 0.0
            sp_prem = 0.0
            lc_prem = 0.0
        elif trade_type == 'short_call_only':
            sp_prem = 0.0
            lp_prem = 0.0
            lc_prem = 0.0
        elif trade_type == 'short_put_only':
            sc_prem = 0.0
            lc_prem = 0.0
            lp_prem = 0.0
        elif trade_type == 'double_long':
            sc_prem = 0.0
            sp_prem = 0.0
            
        if call_problems:
            # Log & skip the call side
            # print(f"Skipping CALL side for {ticker} on {trade_date_str} => {call_problems}, close_price: {close_price:.2f}, call_strike: {call_strike:.2f}, call_strike_bought: {call_strike_bought:.2f}")
            call_enabled = False
            # Set call side prem to 0 so it won't affect net credit
            sc_prem = 0.0
            lc_prem = 0.0
        else:
            cumulative_call_loss -= sc_prem - lc_prem
            if (cumulative_call_loss > 0 or call_loss_compensate > 0 )and mode == "validation":
                print(f"call strike sell {call_strike} buy {call_strike_bought} close: {close_price_at_trade:.2f} sc_prem: {sc_prem:.2f}, lc_prem: {lc_prem:.2f}, call_loss: {call_loss_compensate:.2f} cumulative_call_loss: {cumulative_call_loss:.2f} current friday: {current_day},exp: {expire_day}")
            
        if put_problems:
            # Log & skip the put side
            # print(f"Skipping PUT side for {ticker} on {trade_date_str} => {put_problems}, close_price: {close_price:.2f}, put_strike: {put_strike:.2f}, put_strike_bought: {put_strike_bought:.2f}")
            put_enabled = False
            # Set put side prem to 0 so it won't affect net credit
            sp_prem = 0.0
            lp_prem = 0.0
        else:
            cumulative_put_loss -= sp_prem - lp_prem
            if ( cumulative_put_loss > 0 or put_loss_this > 0 ) and mode == "validation":
                print(f"put strike sell {put_strike} buy {put_strike_bought} close: {close_price_at_trade:.2f} sp_prem: {sp_prem:.2f}, lp_prem: {lp_prem:.2f}, put_loss: {put_loss_compensate:.2f} cumulative_put_loss: {cumulative_put_loss:.2f} current friday: {current_day},exp: {expire_day}")

        position = {
            'week_start': current_day,
            'expiration': expire_day,
            'call_strike_sold': call_strike,
            'put_strike_sold': put_strike,
            'call_strike_bought': call_strike_bought,
            'put_strike_bought': put_strike_bought,
            'short_call_prem_open': sc_prem,
            'short_put_prem_open': sp_prem,
            'long_call_prem_open': lc_prem,
            'long_put_prem_open': lp_prem,
            'position_hedged': False,
            'open_distance_call': round((call_strike - close_price_at_trade)/close_price_at_trade,3) if call_strike is not None else None,
            'open_distance_put': round((close_price_at_trade - put_strike)/close_price_at_trade,3) if put_strike is not None else None,
            # 'call_open_state': call_open_state,
            # 'put_open_state': put_open_state,
        }
        if sc_prem > 0 or sp_prem > 0:
            active_positions.append(position)

        # If we get here, at least one side is valid => proceed
        weekly_open_credit = (sc_prem + sp_prem) - (lc_prem + lp_prem)

        # Possibly recalc trade cost, payoff, etc.
        non_zero_prem_count = sum(1 for prem in [sc_prem, sp_prem, lc_prem, lp_prem] if prem > 0)
        weekly_open_credit_amt = round(weekly_open_credit * 100, 2) - (non_zero_prem_count * OPTION_TRADE_COST)

        if not call_problems:
            required_margin += (call_strike_bought - call_strike) * 100
        if not put_problems:
            required_margin += (put_strike - put_strike_bought) * 100

        logging.debug(f"Weekly Open Credit Amount: {weekly_open_credit_amt}")
                    
        weekly_expiration_payoff = round(( -call_loss_this - put_loss_this ) * 100,2)
        weekly_expiration_payoff -= (call_stop_loss + put_stop_loss)
        weekly_expiration_payoff -= (call_roll_cost + put_roll_cost)

        return_annualized = 0
        for i in range(len(weekly_results)):
            if weekly_results[i]['expiration'] != current_day:
                continue
            realized_gain = weekly_results[i]['weekly_open_credit_amt'] + weekly_expiration_payoff
            if weekly_results[i]['required_margin'] > 0:
                return_annualized = (realized_gain / weekly_results[i]['required_margin']) * 100 / (current_day - weekly_results[i]['week_start']).days * 365 if required_margin > 0 else 0
            break
        
        logging.debug(f"Weekly Expiration Payoff: {weekly_expiration_payoff}")

        weekly_pnl = round(weekly_open_credit_amt + weekly_expiration_payoff,2)
        if weekly_pnl < 0 and mode == "validation":
            print(f"{current_day} weekly pnl: {weekly_pnl:.2f}, weekly_open_credit_amt: {weekly_open_credit_amt:.2f}, weekly_expiration_payoff: {weekly_expiration_payoff:.2f},sc_prem: {sc_prem:.2f}, sp_prem: {sp_prem:.2f}, lc_prem: {lc_prem:.2f}, lp_prem: {lp_prem:.2f}")
            if ( call_problems or put_problems ) and mode == "validation":
                print(f"call_problems: {call_problems}, put_problems: {put_problems}")
        logging.debug(f"Weekly PnL: {weekly_pnl}")

        total_pnl += weekly_pnl
        weekly_dates.append(current_day)
        cumulative_pnls.append(total_pnl)
        if total_pnl < -100000 and mode == "training":
            return None, None, None, None
        call_open_state=None
        put_open_state=None
        if call_strike is not None and call_strike <= close_price_at_trade:
            call_open_state = "itm"
        else:
            call_open_state = "otm"
        if put_strike is not None and put_strike >= close_price_at_trade:
            put_open_state = "itm"
        else:
            put_open_state = "otm"

        weekly_results.append({
            "week_start": current_day,
            "trade_day": trade_date_str,
            "expiration": expire_day,
            "close_price_expiration": round(close_price_expiration,2),
            "close_price_pricing": round(close_price_pricing,2),
            "call_strike_sold": call_strike,
            "call_open_state": call_open_state,
            "put_strike_sold": put_strike,
            "call_strike_bought": call_strike_bought,
            "put_open_state": put_open_state,
            "put_strike_bought": put_strike_bought,
            "short_call_prem_open": round(sc_prem,2),
            "short_put_prem_open": round(sp_prem,2),
            "long_call_prem_open": round(lc_prem,2),
            "long_put_prem_open": round(lp_prem,2),
            "short_call_volume": sc_volume,
            "short_put_volume": sp_volume,
            "long_call_volume": lc_volume,
            "long_put_volume": lp_volume,
            "short_call_delta": round(sc_delta,3) if sc_delta is not None else None,
            "short_put_delta": round(sp_delta,3) if sp_delta is not None else None,
            "long_call_delta": round(lc_delta,3) if lc_delta is not None else None,
            "long_put_delta": round(lp_delta,3) if lp_delta is not None else None,
            "open_distance_call": round((call_strike - close_price_pricing)/close_price_pricing,3) if call_strike is not None else None,
            "open_distance_put": round((close_price_pricing - put_strike)/close_price_pricing,3) if put_strike is not None else None,
            "weekly_open_credit_amt": weekly_open_credit_amt,
            "weekly_expiration_payoff": weekly_expiration_payoff,
            "weekly_pnl": weekly_pnl,
            "cumulative_pnl": total_pnl,
            "previous_call_assigned": (call_loss_compensate > 0),
            "previous_put_assigned": (put_loss_compensate > 0),
            "call_strike_method": call_method,
            "put_strike_method": put_method,
            "required_margin": required_margin,
            "return_annualized": return_annualized,
            "cumulative_call_loss": cumulative_call_loss,
            "cumulative_put_loss": cumulative_put_loss,
            "skip_reason": skip_reason,
            "call_problems": call_problems,
            "put_problems": put_problems,
            **({"call_closed_by_stop": True, "call_stop_loss": position.get('call_stop_loss', 0), 
                "call_stop_date": position.get('call_stop_date')} if position.get('call_closed_by_stop') else {}),
            **({"put_closed_by_stop": True, "put_stop_loss": position.get('put_stop_loss', 0), 
                "put_stop_date": position.get('put_stop_date')} if position.get('put_closed_by_stop') else {}),
            **({"call_rolled": True, "call_roll_data": position.get('call_roll_data')} if position.get('call_rolled') else {}),
            **({"put_rolled": True, "put_roll_data": position.get('put_roll_data')} if position.get('put_rolled') else {}),
            "active_positions": active_positions
        })

    # ---------------------------------------------------------
    # Add ARROWS for negative weeks and compute Sharpe Ratio
    # ---------------------------------------------------------
    if cumulative_pnls:
        PLOT_CUMULATIVE_PNL = False
        if PLOT_CUMULATIVE_PNL:
            plt.figure(figsize=(20, 12))
            plt.plot(weekly_dates, cumulative_pnls, label="Cumulative PnL", color='black', linewidth=2)

            # Determine y-axis range for arrow placement
            y_min, y_max = plt.ylim()
            y_range = y_max - y_min

            for (x_date, y_val, direction) in arrow_data:
                if direction == "up":
                    # Draw arrow pointing up
                    plt.annotate(
                        "Call Loss",
                        xy=(x_date, y_val),
                        xycoords="data",
                        xytext=(x_date, y_val - 0.05 * y_range),
                        textcoords="data",
                        arrowprops=dict(facecolor="red", arrowstyle="->"),
                        horizontalalignment="center",
                        color="red",
                    )
                elif direction == "down":
                    # Draw arrow pointing down
                    plt.annotate(
                        "Put Loss",
                        xy=(x_date, y_val),
                        xycoords="data",
                        xytext=(x_date, y_val + 0.05 * y_range),
                        textcoords="data",
                        arrowprops=dict(facecolor="blue", arrowstyle="->"),
                        horizontalalignment="center",
                        color="blue",
                    )

        # Compute Sharpe
        weekly_returns  = []
        prev_pnl        = 0.0
        for val in cumulative_pnls:
            inc = val - prev_pnl
            prev_pnl = val

        # Drop first if it's basically 0
        if weekly_returns and abs(weekly_returns[0]) < 1e-9:
            weekly_returns.pop(0)
            logging.debug("Dropped first weekly return as it was effectively zero.")

        if len(weekly_returns) > 1:
            mean_ret = np.mean(weekly_returns)
            std_ret  = np.std(weekly_returns, ddof=1)
            if std_ret > 1e-9:
                sharpe_ratio = (mean_ret / std_ret) * math.sqrt(52)
            else:
                sharpe_ratio = 0.0
            logging.debug(f"Mean Weekly Return: {mean_ret}, Weekly Return Std Dev: {std_ret}, Sharpe Ratio: {sharpe_ratio}")
        else:
            sharpe_ratio = 0.0
            logging.debug("Not enough weekly returns to calculate Sharpe Ratio.")

    else:
        # Optional: Log that no PnL data was available
        logging.info(f"No PnL data available for {ticker} from {start_date} to {end_date}. Sharpe Ratio set to 0.0.")
        sharpe_ratio = 0.0

    # Clear large intermediate data structures when no longer needed
    del big_options_to_fetch
    del final_friday_data

    return total_pnl, weekly_results, sharpe_ratio, weekly_dates

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
        # Get PNL and margin required arrays and filter out None values
        returns = [r for r in combo['return_annualized_array'] if r is not None]
        
        if returns:  # Only process if we have valid returns
            num_valid_points = len(returns)
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else float('inf')
            baseline_return = 0
            score = (avg_return - baseline_return) / std_return if std_return > 0 else -1000
            
            # Access parameters from the 'parameters' dictionary
            params = combo['parameters']

            combo_stats.append({
                'expiring_wk': params['expiring_wks'],
                'target_premium_otm': params['target_premium_otm'],
                'target_premium_steer': params['target_premium_steer'],
                'iron_condor_width': params['iron_condor_width'],
                'stop_loss_action': params['stop_loss_action'],
                'stop_loss_percent': params['stop_loss_percent'],
                'day_of_week': params['day_of_week'],
                'num_valid_points': num_valid_points,
                'avg_return': avg_return,
                'std_return': std_return,
                'score': score,
                'final_pnl': combo['final_pnl'],
                'sharpe': score,  # Using our calculated score as the Sharpe ratio
                'start_date': combo['weekly_dates'][0],
                'end_date': combo['weekly_dates'][-1],
                'target_delta': params['target_delta'],
            })

    if not combo_stats:
        return None

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(combo_stats)
    
    # Get the top 5 combos sorted by final_pnl descending
    top_combos_score = results_df.sort_values('final_pnl', ascending=False).head(5)
    top_combos = top_combos_score.sort_values('score', ascending=False).head(5)
    
    # Choose the best combo from top combos (highest score)
    best_combo = top_combos.iloc[0]
    
    # Calculate median values for each parameter among the top combos
    median_expiring_wk = np.median(top_combos['expiring_wk'])
    
    # Return dictionary with best combo parameters and appended median values
    return {
        'parameters': {
            'expiring_wks': best_combo['expiring_wk'],
            'target_premium_otm': best_combo['target_premium_otm'],
            'target_premium_steer': best_combo['target_premium_steer'],
            'iron_condor_width': best_combo['iron_condor_width'],
            'stop_loss_action': best_combo['stop_loss_action'],
            'stop_loss_percent': best_combo['stop_loss_percent'],
            'day_of_week': best_combo['day_of_week'],
            'target_delta': best_combo['target_delta'],
        },
        'sharpe': best_combo['sharpe'],
        'median_expiring_wk': median_expiring_wk,
    }


async def monthly_recursive_backtest(
    ticker: str,
    global_start_date: str,
    global_end_date: str,
    lookback_months: int,
    trade_parameters: List[Dict[str, Any]],  # List of trade parameter dictionaries
    client: PolygonAPIClient,
    save_file: bool = False,
    trade_type: str = "iron_condor",
    input_df: Dict[str, pd.DataFrame] = None,
) -> Tuple[float, List[datetime], List[float], List[Dict[str, Any]], List[float], List[Dict[str, Any]]]:

    # Convert input strings to datetime for slicing
    start_dt = datetime.strptime(global_start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(global_end_date, "%Y-%m-%d")

    # 1) If LOAD_MONTHLY_DATA is True, attempt to find a suitable file
    if LOAD_MONTHLY_DATA:
        for fname in os.listdir(MONTHLY_BACKTEST_DIR):
            if fname.endswith(".pkl") and f"monthly_{ticker}" in fname:
                try:
                    # Adjust filename parsing to handle extra timestamp
                    parts = fname.split("_")
                    file_start = parts[-5]
                    file_end = parts[-3]
                    file_start_dt = datetime.strptime(file_start, "%Y-%m-%d")
                    file_end_dt = datetime.strptime(file_end, "%Y-%m-%d")
                except (ValueError, IndexError):
                    print(f"File name {fname} does not match expected format. Skipping.")
                    continue

                # Check if the global start and end dates fall within the file's range
                if file_start_dt <= start_dt and file_end_dt >= end_dt:
                    fullpath = os.path.join(MONTHLY_BACKTEST_DIR, fname)
                    with open(fullpath, "rb") as f:
                        loaded_data = pickle.load(f)
                    loaded_parameters = loaded_data.get("parameters", {})

                    # Compare parameters to ensure they match
                    if parameters_match(loaded_parameters, current_parameters):
                        print(f"Loaded monthly backtest from file: {fname}")

                        loaded_dt_series = loaded_data["dt_series"]  # List[datetime]
                        loaded_pnl_series = loaded_data["pnl_series"]  # List[float]
                        loaded_cumulative_pnl_series = loaded_data["pnl_cumulative_series"]  # List[float]
                        loaded_details = loaded_data["weekly_results"]  # List[Dict[str, Any]]
                        # 2) Slice dt_series & pnl_series to [start_dt, end_dt]
                        dt_sliced, pnl_sliced, cumulative_pnl_sliced, weekly_results = [], [], [], []
                        for d, p, c, e in zip(loaded_dt_series, loaded_pnl_series,loaded_cumulative_pnl_series,loaded_details):
                            if start_dt <= d <= end_dt:
                                dt_sliced.append(d)
                                pnl_sliced.append(p)
                                cumulative_pnl_sliced.append(c)
                                weekly_results.append(e)

                        # 3) Recompute final PnL from the sliced portion
                        final_pnl_sliced = pnl_sliced[-1] if pnl_sliced else 0.0

                        # Return the sliced data
                        return (
                            final_pnl_sliced,
                            dt_sliced,
                            cumulative_pnl_sliced,
                            loaded_data["parameter_history"],
                            pnl_sliced,
                            weekly_results
                        )
                    else:
                        print(f"Parameters mismatch for {fname}, ignoring it...")

    # Convert strings to datetime
    start_dt = datetime.strptime(global_start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(global_end_date, "%Y-%m-%d")

    # Prepare to store the "continuous" weekly results
    global_weekly_dates = []
    global_weekly_pnls = []
    global_weekly_results = []
    global_cumulative_pnls = []
    parameter_history = []  # To track when parameters change
    cumulative_pnl = 0.0

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
    for i in range(0, len(all_month_starts), VALIDATION_MONTH_FORWARD):
        trade_month_start = all_month_starts[i]
        if i <= len(all_month_starts) - 1 - VALIDATION_MONTH_FORWARD:
            next_month_start = all_month_starts[i + VALIDATION_MONTH_FORWARD]
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

        if training_start < start_dt:
            continue

        if training_end <= training_start:
            continue
        print(f"Training period: {training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}, trade period: {trade_month_start.strftime('%Y-%m-%d')} to {trade_month_end.strftime('%Y-%m-%d')}")

        combo_results = []
        for params in trade_parameters:
            final_pnl, details, sharpe, weekly_dates = await backtest_options_sync_or_async(
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
            if final_pnl is None and details is None and weekly_dates is None:
                print(f"{ticker} Training skipped: {training_start.strftime('%Y-%m-%d')} ~ {training_end.strftime('%Y-%m-%d')} params: {params}")
                continue
            combo_results.append({
                "parameters": params,
                "final_pnl": final_pnl,
                "weekly_dates": weekly_dates,
                "weekly_pnls": [row["weekly_pnl"] for row in details],
                "return_annualized_array": [row["return_annualized"] for row in details],
                "required_margin_array": [row["required_margin"] for row in details]
            })
            print(f"{ticker} Training completed: {training_start.strftime('%Y-%m-%d')}~{training_end.strftime('%Y-%m-%d')} params: {params}")
            save_stored_option_data(ticker)

        # ------------------------------
        # Plotting the Cumulative PnL for All Runs
        # ------------------------------
        PLOT_TRAINING_RESULT = False
        if PLOT_TRAINING_RESULT:
            plt.figure(figsize=(15, 8))

            for idx, result in enumerate(combo_results):
                dates = result["weekly_dates"]
                pnls = result["weekly_pnls"]
                
                # Compute cumulative PnL
                cumulative_pnl = np.cumsum(pnls)

                if result['parameters']['target_premium_otm'] is not None:
                    label = (f"EW:{params['expiring_wks']},wing:{params['iron_condor_width']},stp:{params['stop_loss_percent']}\n"
                                f"premium:{params['target_premium_otm']:.3f},steer:{params['target_premium_steer']}")
                else:
                    label = (f"EW:{params['expiring_wks']},wing:{params['iron_condor_width']},stp:{params['stop_loss_percent']}\n"
                                f"premium:{params['target_delta']:.3f}")
                
                plt.plot(dates, cumulative_pnl, label=label)

            plt.title(f"Cumulative PnL Over Time for {ticker}", fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Cumulative PnL ($)", fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True)

            # Improve date formatting
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Display the plot
            plt.show()

        if not combo_results:
            continue

        best_combo = analyze_trading_combinations(combo_results)

        if best_combo is None:
            print(f"No valid trading combinations found for month {trade_month_start.strftime('%Y-%m')}, skip trading this month.")
            continue
        best_params = best_combo["parameters"]

        print(f"Selected best combo for month {trade_month_start.strftime('%Y-%m')} => params: {best_params}")

        parameter_history.append({
            "month": trade_month_start.strftime("%Y-%m"),
            "start_date": trade_month_start.strftime("%Y-%m-%d"),
            "end_date": trade_month_end.strftime("%Y-%m-%d"),
            **best_params
        })

        if trade_month_end < trade_month_start:
            continue
        
        if global_weekly_results:
            trade_month_start = global_weekly_results[-1]["week_start"] + timedelta(days=1)
            print(f"trade_month_start: {trade_month_start} last traded week start: {global_weekly_results[-1]['week_start'] if global_weekly_results else 'N/A'}")

        print(f"Running best combo for month {trade_month_start.strftime('%Y-%m-%d')} to {trade_month_end.strftime('%Y-%m-%d')} => params: {best_params}")
        
        final_pnl_m, details_m, sharpe_m, month_dates = await backtest_options_sync_or_async(
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

        if details_m is None:
            continue
        month_pnls = [row["weekly_pnl"] for row in details_m]
        month_dates = [row["week_start"] for row in details_m]

        carry_over_weekly_results = details_m

        if month_dates and month_pnls:
            # Iterate through each week's PnL in the month
            for p in month_pnls:
                cumulative_pnl += p
                global_cumulative_pnls.append(cumulative_pnl)
            
            # Handle overlapping dates if necessary
            if global_weekly_dates:
                # Check if the first date of the current month overlaps with the last date of global_weekly_dates
                if month_dates[0] <= global_weekly_dates[-1]:
                    print(f"Overlap detected between {month_dates[0]} and {global_weekly_dates[-1]}. Removing duplicate.")
                    # Remove the overlapping date from the current month_dates and month_pnls
                    overlap_index = month_dates.index(global_weekly_dates[-1])
                    del month_dates[overlap_index]
                    del month_pnls[overlap_index]
                    del global_cumulative_pnls[-1]
                    
            # Extend the global lists with the current month's data
            global_weekly_dates.extend(month_dates)
            global_weekly_pnls.extend(month_pnls)
            global_weekly_results.extend(details_m)
        else:
            print("No data for this month. Skipping accumulation.")

    final_val = 0.0
    if global_weekly_pnls:
        final_val = global_cumulative_pnls[-1]

    save_file = False
    if save_file:
        filename = get_monthly_backtest_file(ticker, global_start_date, global_end_date)
        data_to_save = {
            "parameters": current_parameters,
            "final_pnl": final_val,
            "dt_series": global_weekly_dates,
            "pnl_series": global_weekly_pnls,
            "pnl_cumulative_series": global_cumulative_pnls,
            "parameter_history": parameter_history,
            "weekly_results": global_weekly_results
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Saved monthly_recursive_backtest results to {filename}")

    last_week_start = None
    for result in global_weekly_results:
        print(f"week start:{ticker} trade: {result['trade_day']} expiration: {result['expiration']}, skip reason: {result['skip_reason']} pnl: {result['weekly_pnl']}, call: {result['call_strike_sold']}->{result['call_strike_bought']}, close: {result['close_price_pricing']} put: {result['put_strike_sold']}->{result['put_strike_bought']},${result['short_call_prem_open']} ${result['short_put_prem_open']},${result['long_call_prem_open']} ${result['long_put_prem_open']}, sc delta: {result['short_call_delta']}, sp delta: {result['short_put_delta']}, lc delta: {result['long_call_delta']}, lp delta: {result['long_put_delta']}")
        if ( result['short_call_prem_open'] == 0 or result['short_put_prem_open'] == 0 ) and result['skip_reason'] is None:
            print(f"short call/put premium open is 0, skip: {result['short_call_prem_open']} {result['short_put_prem_open']}, call problems: {result['call_problems']}, put problems: {result['put_problems']}")
        
    return final_val, global_weekly_dates, global_cumulative_pnls, parameter_history, global_weekly_pnls,global_weekly_results

def plot_top_n_tickers_pnl(
    top_tickers: List[str],
    avg_sharpe: Dict[str, float],
    filtered_dates: Dict[str, List[datetime]],
    filtered_pnl: Dict[str, List[float]],
    output_dir: str = "top_n_tickers"
):
    """
    Plots the PnL over time for the top tickers and saves the plots to the specified directory.

    Parameters:
        top_tickers (List[str]): List of selected top tickers based on Sharpe ratio.
        weekly_pnl_data (Dict[str, List[float]]): Dictionary with tickers as keys and cumulative weekly PnL as values.
        date_series (Dict[str, List[datetime]]): Dictionary with tickers as keys and corresponding date lists.
        avg_sharpe (Dict[str, float]): Dictionary of tickers and their corresponding Sharpe ratios.
        filtered_dates (Dict[str, List[datetime]]): Filtered dates for the top tickers.
        filtered_pnl (Dict[str, List[float]]): Filtered PnL data for the top tickers.
        output_dir (str, optional): Directory to save the plots. Defaults to "top_n_tickers".
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for ticker in top_tickers:
        # Use filtered dates and PnL for plotting
        pnl_series = filtered_pnl.get(ticker, [])
        dates = filtered_dates.get(ticker, [])
        sharpe_ratio = avg_sharpe.get(ticker, 0.0)

        if not pnl_series or not dates or len(pnl_series) != len(dates):
            print(f"Skipping plot for {ticker} due to insufficient or mismatched data.")
            continue

        # Plot PnL
        plt.figure(figsize=(12, 6))
        plt.plot(dates, pnl_series, label=f"Cumulative PnL", color='blue')
        plt.title(f"{ticker} - Cumulative PnL Over Time")
        plt.xlabel("Date")
        plt.ylabel("PnL ($)")
        plt.grid(True)

        # Add Sharpe ratio to the plot
        plt.text(
            0.05, 0.95, f"Sharpe Ratio: {sharpe_ratio:.2f}",
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

        plt.legend()

        # Save the plot
        plot_file = os.path.join(output_dir, f"{ticker}_pnl_plot.png")
        plt.savefig(plot_file)
        print(f"Saved plot for {ticker} to {plot_file}")

        # Close the plot to save memory
        plt.close()

def select_top_n_tickers_with_plots(
    weekly_pnl_data: Dict[str, List[float]],
    date_series: Dict[str, List[datetime]],
    selection_date: datetime,
    lookback_period: int = 12,  # in months
    top_n: int = 10,
    output_dir: str = "top_n_tickers"
) -> List[str]:
    """
    Selects the top N tickers based on Sharpe ratio and generates PnL plots.

    Parameters:
        weekly_pnl_data (Dict[str, List[float]]): Dictionary with tickers and cumulative weekly PnL as values.
        date_series (Dict[str, List[datetime]]): Dictionary with tickers and corresponding date lists.
        selection_date (datetime): Date for selection.
        lookback_period (int, optional): Lookback period in months. Defaults to 12.
        top_n (int, optional): Number of top tickers to select. Defaults to 10.
        output_dir (str, optional): Directory to save plots. Defaults to "top_n_tickers".

    Returns:
        List[str]: List of top N tickers based on Sharpe ratio.
    """
    avg_sharpe = {}
    lookback_start_date = selection_date - timedelta(days=30 * lookback_period)
    filtered_dates = {}
    filtered_pnl = {}

    for ticker, pnl_series in weekly_pnl_data.items():
        if ticker not in date_series:
            print(f"Date series missing for ticker {ticker}. Skipping.")
            continue

        dates = date_series[ticker]
        if len(dates) != len(pnl_series):
            print(f"Date and PnL series length mismatch for ticker {ticker}. Skipping.")
            continue

        # Filter PnL and dates to lookback period
        filtered_dates[ticker] = [
            date for date in dates if lookback_start_date <= date <= selection_date
        ]
        filtered_pnl[ticker] = [
            pnl_series[i]
            for i, date in enumerate(dates)
            if lookback_start_date <= date <= selection_date
        ]

        # Calculate weekly returns
        weekly_returns = [
            (filtered_pnl[ticker][i] - filtered_pnl[ticker][i - 1]) / max(abs(filtered_pnl[ticker][i - 1]), 1e-9)
            for i in range(1, len(filtered_pnl[ticker]))
        ]

        if len(weekly_returns) < 4:
            print(f"Insufficient weekly return data for {ticker}. Skipping.")
            continue

        mean_ret = np.mean(weekly_returns)
        std_ret = np.std(weekly_returns, ddof=1)
        sharpe_ratio = mean_ret / std_ret if std_ret > 1e-9 else 0.0
        avg_sharpe[ticker] = sharpe_ratio

        total_return = (filtered_pnl[ticker][-1] - filtered_pnl[ticker][0]) / max(abs(filtered_pnl[ticker][0]), 1e-9) if len(filtered_pnl[ticker]) > 1 else 0.0
        print(f"Ticker: {ticker}, Total Return: {total_return:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")

    sorted_tickers = sorted(avg_sharpe.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [ticker for ticker, _ in sorted_tickers[:top_n]]

    # Generate PnL plots for top tickers
    plot_top_n_tickers_pnl(top_tickers, avg_sharpe, filtered_dates, filtered_pnl, output_dir)

    return top_tickers

def aggregate_top_n_tickers_with_plots(
    weekly_pnl_data: Dict[str, List[float]],
    date_series: Dict[str, List[datetime]],
    selection_date: datetime,
    lookforward_period: int = 1,  # in months
    tickers: List[str] = [],
    output_dir: str = "top_n_tickers"
):
    """
    Aligns and aggregates the PnL data of the specified tickers over the given date range 
    and generates a plot for the aggregated PnL.

    Parameters:
        weekly_pnl_data (Dict[str, List[float]]): Dictionary with tickers as keys and cumulative weekly PnL as values.
        date_series (Dict[str, List[datetime]]): Dictionary with tickers as keys and corresponding date lists.
        selection_date (datetime): Start date for filtering the PnL data.
        lookforward_period (int, optional): Number of months to include after the selection date. Defaults to 1.
        tickers (List[str], optional): List of tickers to aggregate. Defaults to all available tickers.
        output_dir (str, optional): Directory to save plots. Defaults to "top_n_tickers".
    """
    lookforward_end_date = selection_date + timedelta(days=30 * lookforward_period)
    filtered_dates = {}
    filtered_pnl = {}

    if not tickers:
        tickers = list(weekly_pnl_data.keys())

    for ticker in tickers:
        if ticker not in weekly_pnl_data or ticker not in date_series:
            print(f"Data missing for ticker {ticker}. Skipping.")
            continue

        pnl_series = weekly_pnl_data[ticker]
        dates = date_series[ticker]

        if len(dates) != len(pnl_series):
            print(f"Date and PnL series length mismatch for ticker {ticker}. Skipping.")
            continue

        # Filter PnL and dates to the specified range
        filtered_dates[ticker] = [
            date for date in dates if selection_date <= date <= lookforward_end_date
        ]
        filtered_pnl[ticker] = [
            pnl_series[i]
            for i, date in enumerate(dates)
            if selection_date <= date <= lookforward_end_date
        ]

        if not filtered_pnl[ticker]:
            print(f"No data for ticker {ticker} in the specified range {selection_date} {lookforward_end_date}. Skipping.")
            continue

    # Align dates across tickers
    common_dates = sorted(set(date for dates in filtered_dates.values() for date in dates))
    aggregated_pnl = []

    # Align dates across tickers
    common_dates = sorted(set(date for dates in filtered_dates.values() for date in dates))
    aggregated_pnl = []
    last_known_pnl = {ticker: 0.0 for ticker in tickers}  # Keep track of last known PnL for each ticker

    for date in common_dates:
        daily_pnl = 0.0
        for ticker in tickers:
            if ticker in filtered_dates and date in filtered_dates[ticker]:
                idx = filtered_dates[ticker].index(date)
                last_known_pnl[ticker] = filtered_pnl[ticker][idx]  # Update last known PnL
                daily_pnl += last_known_pnl[ticker]
            else:
                daily_pnl += last_known_pnl[ticker]  # Use last known PnL value
        aggregated_pnl.append(daily_pnl)

    # Generate PnL plots for aggregated data
    aggregated_data = {"Aggregated": aggregated_pnl}
    aggregated_dates = {"Aggregated": common_dates}
    returns = [
        (aggregated_pnl[i] - aggregated_pnl[i - 1]) / max(abs(aggregated_pnl[i - 1]), 1e-9)
        for i in range(1, len(aggregated_pnl))
    ]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = mean_ret / std_ret if std_ret > 1e-9 else 0.0

    avg_sharpe = {"Aggregated": sharpe}  # Optional: Calculate Sharpe ratio for aggregated data.

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plot_top_n_tickers_pnl(
        ["Aggregated"], avg_sharpe, aggregated_dates, aggregated_data, output_dir
    )

    return common_dates, aggregated_pnl

def plot_recursive_results(ticker, final_pnl, details_m, dt_series, pnl_cumulative_series, parameter_history, global_start_date, global_end_date, df_dict):
    print(f"Recursive monthly approach => Final PnL for {ticker}: {final_pnl:.2f}")
    df = df_dict['df']
    vix_df = df_dict['vix_df']

    # Data extraction
    strike_dates = [row["expiration"] for row in details_m]
    close_prices = [row["close_price_pricing"] for row in details_m]
    sc_values    = [row["call_strike_sold"]    for row in details_m]
    pc_values    = [row["put_strike_sold"]     for row in details_m]
    sb_values    = [row["call_strike_bought"]  for row in details_m]
    pb_values    = [row["put_strike_bought"]   for row in details_m]
    sc_premiums  = [row["short_call_prem_open"]   for row in details_m]
    pc_premiums  = [row["short_put_prem_open"]    for row in details_m]
    sb_premiums  = [row["long_call_prem_open"]   for row in details_m]
    pb_premiums  = [row["long_put_prem_open"]    for row in details_m]
    required_margins = [row["required_margin"] for row in details_m]
    annualized_returns = [row["return_annualized"] for row in details_m]
    skip_reasons = [row.get("skip_reason", "") for row in details_m]
    start_dates = [row["week_start"] for row in details_m]
    cumulative_call_loss = [row["cumulative_call_loss"] for row in details_m]
    cumulative_put_loss = [row["cumulative_put_loss"] for row in details_m]

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(40, 22))
    axes = axes.flatten()
    ax1 = axes[0]  # Cumulative PnL
    ax3 = axes[2]  # Underlying Daily Close + Option Strikes
    ax2 = axes[1]  # Cumulative Losses
    ax4 = axes[3]  # Required Margin & Annualized Return
    ax5 = axes[4]  # Distance to Strike
    ax6 = axes[5]  # VIX Index

    # Skip reasons markers
    skip_reason_markers = {
        "NO_CLOSE_PRICE": "o", "NO_VOLATILITY": "s", "SKIP_EARNINGS": "^",
        "NO_STRIKES": "d", "NO_MATCH_OPTION": "X", "NO_OPTION_CHAIN": "P"
    }
    skip_reason_colors = {
        "NO_CLOSE_PRICE": "red", "NO_VOLATILITY": "green", "SKIP_EARNINGS": "blue",
        "NO_STRIKES": "orange", "NO_MATCH_OPTION": "purple", "NO_OPTION_CHAIN": "magenta"
    }
    for reason, marker in skip_reason_markers.items():
        dates = [date for date, sr in zip(start_dates, skip_reasons) if sr == reason]
        if dates:
            pnls = [0] * len(dates)
            ax1.scatter(dates, pnls, marker=marker, color=skip_reason_colors[reason], label=reason, zorder=3)
    legend_elements = [Line2D([0], [0], marker=marker, color=skip_reason_colors[reason],
                              linestyle='None', markersize=8, label=reason)
                       for reason, marker in skip_reason_markers.items()]
    ax1.legend(handles=legend_elements, loc="upper left", title="Skip Reasons")

    # TOP LEFT: Cumulative PnL (ax1)
    if dt_series and pnl_cumulative_series:
        ax1.plot(dt_series, pnl_cumulative_series, label=f"Cumulative PnL (Final={final_pnl:.2f})",
                 color='black', marker='o', linewidth=2)
        ax1.set_title(f"Cumulative PnL for {ticker}")
        ax1.set_ylabel("PnL ($)")
        ax1.set_xlabel("Date")
        ax1.grid(True)
        ax1.legend()

                    # start = pd.to_datetime(global_start_date)
                    # end   = pd.to_datetime(global_end_date)
                    # daily_dates = pd.date_range(start=start, end=end, freq='B')

                    # # 2. for each date, sum PnL of all legs still open that day
                    # daily_pnl = []
                    # for current_date in daily_dates:
                    #     day_pnl = 0.0
                    #     for row in details_m:
                    #         ws  = pd.to_datetime(row["week_start"])
                    #         exp = pd.to_datetime(row["expiration"])
                    #         # only include legs if the position is open
                    #         if ws <= current_date <= exp:
                    #             # SHORT CALL leg
                    #             sc = query_polygon_for_option_price(
                    #                 ticker,
                    #                 row["call_strike_sold"],
                    #                 'call',
                    #                 exp.strftime("%Y-%m-%d"),
                    #                 current_date.strftime("%Y-%m-%d")
                    #             ).get("mid_price", np.nan)
                    #             if not np.isnan(sc):
                    #                 day_pnl += (row["short_call_prem_open"] - sc) * 100

                    #             # LONG CALL leg
                    #             sb = query_polygon_for_option_price(
                    #                 ticker,
                    #                 row["call_strike_bought"],
                    #                 'call',
                    #                 exp.strftime("%Y-%m-%d"),
                    #                 current_date.strftime("%Y-%m-%d")
                    #             ).get("mid_price", np.nan)
                    #             if not np.isnan(sb):
                    #                 day_pnl += (sb - row["long_call_prem_open"]) * 100

                    #             # SHORT PUT leg
                    #             sp = query_polygon_for_option_price(
                    #                 ticker,
                    #                 row["put_strike_sold"],
                    #                 'put',
                    #                 exp.strftime("%Y-%m-%d"),
                    #                 current_date.strftime("%Y-%m-%d")
                    #             ).get("mid_price", np.nan)
                    #             if not np.isnan(sp):
                    #                 day_pnl += (row["short_put_prem_open"] - sp) * 100

                    #             # LONG PUT leg
                    #             pb = query_polygon_for_option_price(
                    #                 ticker,
                    #                 row["put_strike_bought"],
                    #                 'put',
                    #                 exp.strftime("%Y-%m-%d"),
                    #                 current_date.strftime("%Y-%m-%d")
                    #             ).get("mid_price", np.nan)
                    #             if not np.isnan(pb):
                    #                 day_pnl += (pb - row["long_put_prem_open"]) * 100

                    #             print(f"{current_date.strftime('%Y-%m-%d')}: {day_pnl}, {sb}, {pb}, {sc}, {sp}")
                    #     daily_pnl.append(day_pnl)

                    # print(f"{current_date.strftime('%Y-%m-%d')}: {daily_pnl}")
                    # breakpoint()

                    # ax1.plot(
                    #     daily_dates,
                    #     daily_pnl,
                    #     linestyle='--',
                    #     marker='',
                    #     label="Daily PnL"
                    # )
                    # ax1.legend(loc='upper left')
        
        # Labeling parameter changes and adding shaded regions
        num_periods = len(parameter_history)
        colors = plt.cm.viridis(np.linspace(0, 1, num_periods))
        for i, params in enumerate(parameter_history):
            period_start = datetime.strptime(params["start_date"], "%Y-%m-%d")
            period_end = datetime.strptime(params["end_date"], "%Y-%m-%d")
            start_index = min(range(len(dt_series)), key=lambda j: abs(dt_series[j] - period_start))
            y_val = pnl_cumulative_series[start_index]
            if params['target_premium_otm'] is not None:
                label_text = (f"EW:{params['expiring_wks']},wing:{params['iron_condor_width']},stp:{params['stop_loss_percent']}\n"
                            f"premium:{params['target_premium_otm']:.3f},steer:{params['target_premium_steer']}")
            else:
                label_text = (f"EW:{params['expiring_wks']},wing:{params['iron_condor_width']},stp:{params['stop_loss_percent']}\n"
                            f"premium:{params['target_delta']:.3f}")
            mid_point = period_start + (period_end - period_start) / 2
            ax1.annotate(label_text,
                         xy=(mid_point, y_val), xycoords='data',
                         xytext=(0, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                         horizontalalignment='center', verticalalignment='bottom')
            # Add shaded regions to all plots
            for a in [ax1, ax3, ax2, ax4, ax5, ax6]:
                a.axvspan(period_start, period_end, alpha=0.2, color=colors[i])

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # TOP RIGHT: Underlying Daily Close + Option Strikes (ax3)
    valid_strike_dates = [d for d in strike_dates if d is not None]
    if not valid_strike_dates:
        raise ValueError("No valid strike dates found.")
    start_date_plot = min(valid_strike_dates)
    end_date_plot = max(valid_strike_dates)
    daily_dates = pd.date_range(start=start_date_plot, end=end_date_plot, freq='B').to_pydatetime()
    hist_df = df
    hist_df = hist_df.set_index('date')
    daily_close = hist_df["close"].reindex(daily_dates).ffill().values
    ax3.plot(daily_dates, daily_close, color='black', label="Underlying Daily Close")
    if ticker == "VIX":
        weekday_index = 2
    else:
        weekday_index = 4
    friday_mask = np.array([d.weekday() == weekday_index for d in daily_dates])
    friday_dates = np.array(daily_dates)[friday_mask]
    friday_close = [price for date, price in zip(daily_dates, daily_close) if date.weekday() == weekday_index]
    ax3.scatter(friday_dates, friday_close, color='lime', s=10, label="Friday Close", zorder=10)
    sb_x, sb_y, pb_x, pb_y = [], [], [], []
    for i, week in enumerate(details_m):
        if week["long_call_prem_open"] > 0:
            sb_x.append(strike_dates[i])
            sb_y.append(week["call_strike_bought"])
        if week["long_put_prem_open"] > 0:
            pb_x.append(strike_dates[i])
            pb_y.append(week["put_strike_bought"])
    call_percent_dates, call_percent_strikes = [], []
    call_premium_dates, call_premium_strikes = [], []
    put_percent_dates, put_percent_strikes = [], []
    put_premium_dates, put_premium_strikes = [], []
    breach_dates, breach_prices = [], []
    for i, week in enumerate(details_m):
        if week["call_strike_method"] == "percent" and week["short_call_prem_open"] > 0:
            call_percent_dates.append(strike_dates[i])
            call_percent_strikes.append(sc_values[i])
        elif week["short_call_prem_open"] > 0:
            call_premium_dates.append(strike_dates[i])
            call_premium_strikes.append(sc_values[i])
        if week["put_strike_method"] == "percent" and week["short_put_prem_open"] > 0:
            put_percent_dates.append(strike_dates[i])
            put_percent_strikes.append(pc_values[i])
        elif week["short_put_prem_open"] > 0:
            put_premium_dates.append(strike_dates[i])
            put_premium_strikes.append(pc_values[i])
        if week.get('previous_call_assigned') or week.get('previous_put_assigned'):
            breach_dates.append(start_dates[i])
            breach_prices.append(week['close_price_expiration'])
    ax3.scatter(breach_dates, breach_prices, color='magenta', marker='x', s=40, zorder=5, label="Breach")
    ax3.scatter(call_percent_dates, call_percent_strikes, color='red', marker='o', s=35, label="Short Call (Percent)")
    ax3.scatter(call_premium_dates, call_premium_strikes, color='red', marker='s', s=25, label="Short Call (Premium)")
    ax3.scatter(put_percent_dates, put_percent_strikes, color='blue', marker='o', s=35, label="Short Put (Percent)")
    ax3.scatter(put_premium_dates, put_premium_strikes, color='blue', marker='s', s=25, label="Short Put (Premium)")
    ax3.scatter(sb_x, sb_y, color='green', marker='^', s=25, label="Long Call")
    ax3.scatter(pb_x, pb_y, color='orange', marker='v', s=25, label="Long Put")
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', markersize=4, label='Underlying Daily Close'),
        Line2D([0], [0], color='magenta', marker='x', linestyle='None', markersize=6, label='Breach'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=6, label='Short Call (Percent)'),
        Line2D([0], [0], color='red', marker='s', linestyle='None', markersize=6, label='Short Call (Premium)'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=6, label='Short Put (Percent)'),
        Line2D([0], [0], color='blue', marker='s', linestyle='None', markersize=6, label='Short Put (Premium)'),
        Line2D([0], [0], color='green', marker='^', linestyle='None', markersize=6, label='Long Call'),
        Line2D([0], [0], color='orange', marker='v', linestyle='None', markersize=6, label='Long Put')
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)
    ax3.set_title("Option Strikes vs. Underlying Daily Close")
    ax3.set_ylabel("Price")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax3.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Middle Left: Cumulative Losses (ax2)
    if cumulative_call_loss and cumulative_put_loss:
        ax2.plot(start_dates, cumulative_call_loss, marker='o', color='blue', label='Cumulative Call Loss')
        ax2.plot(start_dates, cumulative_put_loss, marker='o', color='red', label='Cumulative Put Loss')
        combined_loss = [c + p for c, p in zip(cumulative_call_loss, cumulative_put_loss)]
        ax2.plot(start_dates, combined_loss, linestyle='--', color='purple', label='Combined Loss')
    else:
        ax2.text(0.5, 0.5, "No Loss Data", transform=ax2.transAxes, ha='center')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Loss")
    ax2.set_title("Cumulative Losses")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Middle Right: Required Margin & Annualized Return (ax4)
    ax4.plot(strike_dates, required_margins, color='green', label="Required Margin", linewidth=2)
    ax4.set_title("Required Margin and Annualized Return")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Margin ($)", color='green')
    ax4.tick_params(axis='y', labelcolor='green')
    ax4.grid(True)
    ax4.legend(loc='upper left')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(strike_dates, annualized_returns, color='blue', label="Annualized Return", linewidth=2)
    ax4_twin.set_ylabel("Annualized Return (%)", color='blue')
    ax4_twin.tick_params(axis='y', labelcolor='blue')
    ax4_twin.legend(loc='upper right')
    ax4.yaxis.set_label_coords(-0.1, 0.5)
    ax4_twin.yaxis.set_label_coords(1.1, 0.5)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax4.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Bottom Left: Distance to Strike (ax5)
    def parse_date(date_obj):
        if isinstance(date_obj, str):
            return datetime.strptime(date_obj, "%Y-%m-%d")
        return date_obj

    effective_date_candidates = []
    for row in details_m:
        trade_day = row.get("trade_day")
        expiration = row.get("expiration")
        if trade_day and expiration:
            trade_day_dt = parse_date(trade_day)
            expiration_dt = parse_date(expiration)
            effective_date_candidates.extend([trade_day_dt, expiration_dt])

    if effective_date_candidates:
        overall_start = min(effective_date_candidates)
        overall_end = max(effective_date_candidates)
    else:
        overall_start = datetime.strptime(global_start_date, "%Y-%m-%d")
        overall_end = datetime.strptime(global_end_date, "%Y-%m-%d")

    daily_data = yf.download(ticker, start=overall_start.strftime("%Y-%m-%d"),
                             end=(overall_end + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
    daily_close = daily_data["Close"]

    effective_dates, call_distance, put_distance, open_distance_call_plot, open_distance_put_plot = [], [], [], [], []
    for row in details_m:
        trade_day = row.get("trade_day")
        expiration = row.get("expiration")
        call_strike = row.get("call_strike_sold")
        put_strike = row.get("put_strike_sold")
        open_distance_call = row.get("open_distance_call")
        open_distance_put = row.get("open_distance_put")

        if trade_day and expiration and call_strike and put_strike:
            current_date = parse_date(trade_day) + timedelta(days=1)
            end_date = parse_date(expiration)
            while current_date <= end_date:
                try:
                    close_price = daily_close.loc[current_date]
                except KeyError:
                    current_date += timedelta(days=1)
                    continue
                effective_dates.append(current_date)
                call_distance.append((call_strike - close_price) / close_price * 100)
                put_distance.append((close_price - put_strike) / close_price * 100)
                open_distance_call_plot.append(open_distance_call*100)
                open_distance_put_plot.append(open_distance_put*100)
                current_date += timedelta(days=1)

    if effective_dates:
        ax5.plot(effective_dates, open_distance_call_plot, marker='o', linestyle='-', color='red', label="Call Distance (%)")
        ax5.plot(effective_dates, open_distance_put_plot, marker='s', linestyle='--', color='blue', label="Put Distance (%)")
        ax5.set_title("Distance from Daily Close Price to Option Strike (%)")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Distance (%)")
        ax5.grid(True)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No effective options found", transform=ax5.transAxes, ha='center')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax5.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Bottom Right: VIX Index (ax6)
    ax6.plot(vix_df['date'], vix_df['close'], color='purple', label="VIX Index")
    ax6.set_title("VIX Index")
    ax6.set_xlabel("Date")
    ax6.set_ylabel("VIX")
    ax6.grid(True)
    ax6.legend()
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for label in ax6.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

    # Set x-limits and finalize plot
    all_dates = [date for date in (dt_series + strike_dates) if date is not None]
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        for a in [ax1, ax3, ax2, ax4, ax5, ax6]:
            a.set_xlim(min_date, max_date)
    fig.autofmt_xdate()
    plt.tight_layout(pad=3.0)
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H-%M')
    os.makedirs("cover_call_plots", exist_ok=True)
    plot_filename = f"cover_call_plots/recursive_monthly_{ticker}_{global_start_date}_to_{global_end_date}_{formatted_time}.png"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300, pad_inches=0.5)
    print(f"Plot saved to {plot_filename}")
    plt.close('all')

    # Clear temporary variables
    del daily_dates, daily_close, vix_df
    import gc
    gc.collect()


async def main():
    # Define the backtesting period
    global_start_date = "2022-01-01"
    global_end_date = datetime.now().strftime("%Y-%m-%d")

    # Configuration and placeholder variables
    # tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
    tickers = ['GOOGL','PLTR','MRNA','T','AMZN','AAPL','INTC','WMT','BAC','AMD','AVGO','DIS','PFE','UNH','GOOG','NKE','CSCO','FFIV','OMC','COR','ORCL','LVS','APH','SBUX','MSFT','TSLA','KO','MU','DAL','UBER','NFLX','GM','BA','VZ','C','SMCI','F','META','NEE','MDT','PEP','JPM','NWSA','PANW','CPRT','BMY','ADM','FCX']
    extra_ticker = ['QQQ','SPY','ARKK','TQQQ','SQQQ','SQ','PYPL','QCOM','OXY']
    tickers = tickers + extra_ticker
    tickers = ['AAPL','META','MSFT','AMZN','QQQ','JPM','QCOM','GLD']
    # tickers = ['SPY']
    global_start_date = "2023-01-01"
    global_end_date   = datetime.now().strftime("%Y-%m-%d")
    lookback_months   = 12
    target_premium_otm    = np.arange(0.06,0.1,0.01)
    target_premium_otm    = [0.1]
    # target_premium_otm    = [None]
    target_premium_steer = [0.3]
    target_delta = [0.015,0.02,0.025]
    target_delta = [0.015]
    target_delta = [None]
    # target_premium_steer = [0]
    # target_premium_steer = [0.1]
    iron_condor_width = [10,20,30]
    iron_condor_width = [20]

    # target_premium = [0.1]
    expiring_wks      = [2,3,4]  # Your expiring weeks data
    expiring_wks      = [4]  # Your expiring weeks data
    vix_correlation = [0,0.25,0.5]
    # roll_methods      = ['close price','loss','roll']
    roll_methods      = [None]
    stop_loss_action = ['roll_in']  # 'roll' or 'close' or 'skip'
    stop_loss_percent = np.arange(0.2,0.8,0.1)
    stop_loss_percent = [0.5]  # Your stop loss percentage(s)
    # day_of_week       = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    day_of_week       = ['Friday']
    trade_type       = 'iron_condor'

    trade_parameters = [
            {
                'expiring_wks': expiring_wks,
                'target_premium_otm': target_premium_otm,
                'target_premium_steer': target_premium_steer,
                'target_delta': target_delta,
                'iron_condor_width': iron_condor_width,
                'stop_loss_action': stop_loss_action,
                'stop_loss_percent': stop_loss_percent,
                'day_of_week': day_of_week,
                'vix_correlation': vix_correlation,
            }
            for expiring_wks, target_premium_otm, target_premium_steer, target_delta, iron_condor_width, stop_loss_action, stop_loss_percent, vix_correlation in product(
                expiring_wks, target_premium_otm, target_premium_steer, target_delta, iron_condor_width, stop_loss_action, stop_loss_percent, vix_correlation
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
            final_pnl, dt_series, pnl_cumulative_series, parameter_history, pnl_series, details_m,pnl_cumulative_realized_series = await monthly_recursive_backtest(
                            ticker=ticker,
                            global_start_date=global_start_date,
                            global_end_date=global_end_date,
                            lookback_months=lookback_months,
                            trade_parameters=trade_parameters,  # Pass the list of trade parameters
                            client=client,
                            save_file=True,
                            trade_type=trade_type,
                            input_df={'df': df, 'vix_df': vix_df},
                        )
            save_stored_option_data(ticker)
            # Store the results
            recursive_results[ticker] = {
                "weekly_pnl": pnl_series,
                "weekly_pnl_cumulative": pnl_cumulative_series,
                "dates": dt_series,
                "parameter_history": parameter_history,
            }
            # In the main() function, replace the previous $SELECTION_PLACEHOLDER$ code with:
            plot_recursive_results(ticker, final_pnl, details_m, dt_series, pnl_cumulative_series, parameter_history, global_start_date, global_end_date,df_dict={'df': df, 'vix_df': vix_df})


if __name__ == "__main__":
    asyncio.run(main())