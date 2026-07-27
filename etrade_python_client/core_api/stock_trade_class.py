from __future__ import annotations
from email.mime import base
from lib2to3.pgen2.token import OP
import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import argparse
import time as t
import re
from urllib.parse import parse_qs, urlparse, unquote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyetrade
import pyperclip
import random
import sys
import tempfile
from copy import deepcopy
import csv
from selenium.common.exceptions import NoSuchElementException, TimeoutException, SessionNotCreatedException
from itertools import product
from accounts.accounts_bo import Accounts
from market.market_bo import Market
from order.order_bo import Order
from live_trading.runtime_safety import RuntimeSafetyBoundary
from datetime import time
import pickle

pairs_trading_dir = '/Users/btian/pairs_trading'
sys.path.append(pairs_trading_dir)

# print('sys path: ', sys.path)
# Now you can import the module without the .py extension
try:
    from pairs_trading_bo import Position,OpenPosition,InSamplePairs,HistoricalBacktest,CointData
    import pairs_trading_bo
    from pair_trade_backtrade import LookBackTest,LookForwardTest,TrainingDataSet
    import pair_trade_backtrade
except ModuleNotFoundError as e:
    print("Error importing pairs_trading_bo:", e)


def get_bid_ask_spread(ticker_symbol):
    # Fetch 1-minute interval data for the last 5 days
    ticker = yf.Ticker(ticker_symbol)
    try:
        hist = ticker.history(period='5d', interval='1m')
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None
    
    # Check if data is retrieved
    if hist.empty:
        print(f"No data retrieved for {ticker_symbol}")
        return None

    # Calculate bid-ask spread using the high and low prices
    hist['BidAskSpread'] = hist['High'] - hist['Low']
    
    # Normalize the spread by dividing by the close price
    hist['NormalizedBidAskSpread'] = hist['BidAskSpread'] / hist['Close'] * 100
    
    # Estimate bid and ask prices
    hist['EstimatedAsk'] = hist['High']
    hist['EstimatedBid'] = hist['Low']
    
    # Round the floating point numbers to 3 decimal points
    hist = hist.round(3)
    
    # Add date and time columns for easier filtering
    hist['Date'] = hist.index.date
    hist['Time'] = hist.index.time

    # Print the entries for the last 5 minutes of each trading day
    for date in hist['Date'].unique():
        day_data = hist[hist['Date'] == date]
        last_5_min_data = day_data.between_time('15:55', '16:00')
        print(f"Last 5 minutes of trading day for {date}:")
        print(last_5_min_data[['High', 'Low', 'Close', 'Volume', 'BidAskSpread', 'NormalizedBidAskSpread']])
    
    # Calculate average normalized bid-ask spread
    average_normalized_spread = hist['NormalizedBidAskSpread'].mean()
    print(f"Average normalized bid-ask spread for {ticker_symbol} over the last 5 days: {average_normalized_spread:.3f}")

    return average_normalized_spread


class StockInfo:
    def __init__(self, get_history_spread, symbol, close_price, ask_price, ask_size, bid_price, bid_size, volume, last_trade_time):
        self.symbol = symbol
        self.close_price = round(close_price, 3)
        self.ask_price = round(ask_price, 3)
        self.ask_size = ask_size
        self.bid_price = round(bid_price, 3)
        self.bid_size = bid_size
        self.volume = volume
        if close_price > 0:
            self.ask_bid_spread = round(abs(ask_price - bid_price) / close_price * 100, 3)
        else:
            self.ask_bid_spread = 100.0
        if get_history_spread:
            self.ask_bid_spread_est = round(get_bid_ask_spread(symbol), 3)
        else:
            self.ask_bid_spread_est = 0.0
        self.trade_qty_limit = round(min(ask_price * ask_size, bid_price * bid_size), 3)
        self.last_trade_time = last_trade_time

    def __repr__(self):
        return (f"StockInfo(symbol={self.symbol}, close_price={self.close_price}, ask_price={self.ask_price}, "
                f"ask_size={self.ask_size}, bid_price={self.bid_price}, bid_size={self.bid_size}, volume={self.volume}, "
                f"last_trade_time={self.last_trade_time})")

def extract_stock_info(data, get_history_spread: bool, source='etrade'):
    stock_info_list = []
    try:
        quotes = data['QuoteResponse']['QuoteData']
        for quote in quotes:
            symbol = quote['Product']['symbol']
            close_price = float(quote['All']['lastTrade'])
            if source == 'etrade':
                ask_price = float(quote['All']['ask'])
                bid_price = float(quote['All']['bid'])
            elif source == 'yfinance':
                ticker = yf.Ticker(symbol)
                ticker_info = ticker.info
                ask_price = ticker_info.get('ask')
                bid_price = ticker_info.get('bid')
            
            ask_size = int(quote['All']['askSize'])
            bid_size = int(quote['All']['bidSize'])
            volume = int(quote['All']['totalVolume'])
            last_trade_time_epoch = int(quote['All']['timeOfLastTrade'])
            last_trade_time = datetime.fromtimestamp(last_trade_time_epoch)
            
            stock_info = StockInfo(get_history_spread, symbol, close_price, ask_price, ask_size, bid_price, bid_size, volume, last_trade_time)
            stock_info_list.append(stock_info)
    except KeyError as e:
        print(f"Key error: {e}")
    except (ValueError, TypeError) as e:
        print(f"Value or Type error: {e}")
    return stock_info_list

def import_ticker_from_csv(start_date_str, stock_universe='sp500'):
    csv_directory = '/Users/btian/pairs_trading/price_csv/'
    filename_list = {
        'sp500': 'S&P 500 Historical Components & Changes(04-08-2024).csv',
        'oil-etf': 'oil-etf.csv',
        'energy_etf': 'energy_etf.csv',
        'treasury_etf': 'treasury_etf.csv',
        'silicon_etf': 'silicon_etf.csv',
        'real_estate_etf': 'real_estate_etf.csv',
        'gold_etf': 'gold_etf.csv',
        'biotech_etf': 'biotech_etf.csv',
        'nasdaq': 'nasdaq.csv'
    }

    # Convert start_date_str to datetime object
    start_date = pd.to_datetime(start_date_str)

    if stock_universe == 'sp500':
        sp500 = []
        title = 'List of S&P 500 companies'
        filename = csv_directory + filename_list['sp500']
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
            for date in df['date']:
                if start_date >= date:
                    sp500 = df.loc[df['date'] == date, 'tickers'].values[0]
                    # print('Matched date for sp500 list: ', start_date, date, sp500[0:5])
                    break
        return sp500
    else:
        etf = []
        title = f'List of {stock_universe} companies'
        filename = csv_directory + filename_list[stock_universe]
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            df['tickers'] = df['tickers'].apply(lambda x: sorted(x.split(',')))
            for date in df['date']:
                if start_date >= date:
                    etf = df.loc[df['date'] == date, 'tickers'].values[0]
                    # print(f'Matched date for {stock_universe} list: ', start_date, date, etf[0:5])
                    break
        return etf

def log_stock_info(stock_universe, stock_info_list):
    # Create the directory if it doesn't exist
    directory = 'historical_quote'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Prepare the data for CSV
    data = [{
        'symbol': stock_info.symbol,
        'close_price': round(stock_info.close_price, 3),
        'ask_price': round(stock_info.ask_price, 3),
        'ask_size': stock_info.ask_size,
        'bid_price': round(stock_info.bid_price, 3),
        'bid_size': stock_info.bid_size,
        'volume': stock_info.volume,
        'ask_bid_spread': round(stock_info.ask_bid_spread, 3),
        'ask_bid_spread_est': round(stock_info.ask_bid_spread_est, 3),
        'trade_qty_limit': round(stock_info.trade_qty_limit, 3),
        'last_trade_time': stock_info.last_trade_time
    } for stock_info in stock_info_list]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Get current date in yyyy-mm-dd format
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Save to CSV with the date prefixed to the filename
    filename = os.path.join(directory, f'{current_date}_{stock_universe}.csv')
    
    if not os.path.isfile(filename):
        # If the file does not exist, write the header
        df.to_csv(filename, index=False)
    else:
        # If the file exists, append data without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)
    
    print(f"Stock information for {stock_universe} has been appended to {filename}")

def plot_spread_from_csv(stock_universe):
    # Define the directory and file name
    directory = 'historical_quote'

    # Get current date in yyyy-mm-dd format
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Save to CSV with the date prefixed to the filename
    filename = os.path.join(directory, f'{current_date}_{stock_universe}.csv')
    
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"No data file found for {stock_universe}.")
        return
    
    # Load the data from the CSV file
    df = pd.read_csv(filename)
    
    # Convert last_trade_time to datetime
    df['last_trade_time'] = pd.to_datetime(df['last_trade_time'])
    
    # Filter data to include only today's entries
    today = datetime.now().date()
    df_today = df[df['last_trade_time'].dt.date == today]
    
    if df_today.empty:
        print(f"No data available for today for {stock_universe}.")
        return
    
    # Exclude symbols with close price smaller than $1
    df_today = df_today[df_today['close_price'] >= 1]

    # Set the index to last_trade_time
    df_today.set_index('last_trade_time', inplace=True)
    
    # Resample data into 5-minute intervals for box plot
    df_resampled = df_today.resample('5min')
    
    # Prepare data for the boxplot
    boxplot_data = [group['ask_bid_spread'].values for _, group in df_resampled if not group.empty]
    
    # Generate box plots for each 5-minute interval
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    
    # Box Plot
    axs[0].boxplot(boxplot_data, positions=range(len(boxplot_data)), widths=0.6)
    time_labels = [t.strftime('%H:%M') for time, _ in df_resampled if not _.empty]
    axs[0].set_xticks(range(len(time_labels)))
    axs[0].set_xticklabels(time_labels, rotation=45, ha='right')
    axs[0].set_title(f'Ask-Bid Spread Box Plot for {stock_universe} (Every 5 mins)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Ask-Bid Spread (%)')
    
    # Calculate median spread and closing volume for each ticker
    median_spreads = df_today.groupby('symbol')['ask_bid_spread'].median()
    closing_volumes = df_today.groupby('symbol').apply(lambda x: x['volume'][-1])
    
    # Ensure the sizes match for plotting
    if len(median_spreads) != len(closing_volumes):
        print("Mismatch in lengths of median spreads and closing volumes.")
        return
    
    # Scatter Plot
    axs[1].scatter(closing_volumes, median_spreads, alpha=0.5)
    axs[1].set_title(f'Scatter Plot of Median Ask-Bid Spread vs Closing Volume for {stock_universe}')
    axs[1].set_xlabel('Closing Volume')
    axs[1].set_ylabel('Median Ask-Bid Spread (%)')
    
    for i, symbol in enumerate(median_spreads.index):
        axs[1].annotate(symbol, (closing_volumes[i], median_spreads[i]))
    
    plt.tight_layout()
    plt.show()

def analyze_spread_from_csv(stock_universes):
    directory = 'historical_quote'
    spread_results = {}

    for stock_universe in stock_universes:
        all_spreads = []
        
        # Iterate through all files in the directory for the stock universe
        for filename in os.listdir(directory):
            if stock_universe in filename:
                filepath = os.path.join(directory, filename)
                print(filepath)
                df = pd.read_csv(filepath)
                
                # Convert last_trade_time to datetime
                df['last_trade_time'] = pd.to_datetime(df['last_trade_time'])
                
                # Filter data to exclude the first and last 30 minutes of trading
                df = df[df['last_trade_time'].dt.time > time(7, 0)]
                df = df[df['last_trade_time'].dt.time < time(12, 30)]
                
                # Collect all spreads for the stock universe
                all_spreads.extend(df['ask_bid_spread'].values)
        
        if all_spreads:
            # Calculate the 90th percentile spread
            all_spreads_sorted = sorted(all_spreads)
            threshold_index = int(len(all_spreads_sorted) * 0.9)
            spread_90_percentile = all_spreads_sorted[threshold_index]

            # Store the result for the stock universe
            spread_results[stock_universe] = spread_90_percentile
    
    return spread_results


def calculate_average_spread_excluding_outliers(df):
    # Group by symbol and calculate the average spread excluding the top and bottom 10% data points
    spread_dict = {}
    for symbol, group in df.groupby('symbol'):
        # Sort the ask_bid_spread values
        sorted_spreads = group['ask_bid_spread'].sort_values()
        
        # Exclude the top and bottom 10%
        lower_bound = int(len(sorted_spreads) * 0.1)
        upper_bound = int(len(sorted_spreads) * 0.9)
        filtered_spreads = sorted_spreads.iloc[lower_bound:upper_bound]
        
        # Calculate the average spread
        average_spread = filtered_spreads.mean()
        spread_dict[symbol] = round(average_spread, 3)
    
    return spread_dict

def generate_spread_dict_csv(stock_universes):
    base_directory = 'historical_quote'
    output_directory = os.path.join(base_directory, 'spread_dict')
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for stock_universe in stock_universes:
        spread_dict = {}

        # Iterate through all files in the base directory for the stock universe
        for filename in os.listdir(base_directory):
            filepath = os.path.join(base_directory, filename)

            # Skip directories and files in the spread_dict directory
            if os.path.isdir(filepath) or 'spread_dict' in filepath:
                continue
            
            if stock_universe in filename:
                df = pd.read_csv(filepath)
                
                # Convert last_trade_time to datetime
                df['last_trade_time'] = pd.to_datetime(df['last_trade_time'])
                
                # Filter data to exclude the first and last 30 minutes of trading
                df = df[df['last_trade_time'].dt.time > time(12, 30)]
                df = df[df['last_trade_time'].dt.time < time(12, 45)]
                
                # Calculate average spread excluding outliers and update the spread_dict
                daily_spread_dict = calculate_average_spread_excluding_outliers(df)
                for symbol, avg_spread in daily_spread_dict.items():
                    if symbol in spread_dict:
                        spread_dict[symbol].append(avg_spread)
                    else:
                        spread_dict[symbol] = [avg_spread]
        
        # Calculate the final average spread for each symbol
        final_spread_dict = {symbol: round(sum(spreads)/len(spreads), 3) for symbol, spreads in spread_dict.items()}
        
        # Convert the final spread dictionary to DataFrame
        spread_df = pd.DataFrame(list(final_spread_dict.items()), columns=['symbol', 'average_spread'])

        # Save the DataFrame to CSV in the new output directory without the date in the filename
        output_filename = os.path.join(output_directory, f'{stock_universe}_spread_dict.csv')
        spread_df.to_csv(output_filename, index=False)
        
        print(f"Spread dictionary for {stock_universe} has been saved to {output_filename}")

def generate_trade_parameter_list(stock_universe,sector_key):
    baseline_only_mode = 0
    base_trade_parameter = {
        'num_pairs': 5, 
        'corr_cutoff': 0.75, 
        'crossing_cutoff': 4,
        'in_sample_days': 30, 
        'out_of_sample_days': 2, 
        'entry_delta': 0.5, 
        'exit_delta': 0, 
        'stop_loss_delta': -0.1,
        'in_sample_stat_en': 1,
        'cheat-on-close': 1,
        'sector_key': sector_key,
        'stock_universe': stock_universe,
        'slippage': 0.0005,
        'cash': 1000,
        'force_close_ratio': 20,
        'leverage_ratio': 2.01,
        'slippage_compensation': 0.0005,
        'pair_sort': 'p_value',
        'coint_lookback': 0,
        'p_value_max': 0.05,
        'back_window_ratio': 30
    }
    # ####treasury
    # tested_trade_parameter = {
    #     # 'leverage_ratio': [3],
    #     'in_sample_stat_en': [1],
    #     # 'stop_loss_delta': [-0.1,-0.2],
    #     # 'slippage_compensation': [0.005,0.01],
    #     # 'out_of_sample_days': [2,30],
    #     # 'out_of_sample_days': [2],
    #     # 'force_close_days': [21,42],
    #     'in_sample_days': [10,20,30], 
    #     # # # 'in_sample_days': [30,150], 
    #     # # # 'sector_key': ['sector','industry'],
    #     'num_pairs': [2,5],
    #     'corr_cutoff': [0.75],
    #     # 'entry_delta': [0.8,1,1.2,1.4,1.6,1.8,2],
    #     # 'entry_delta': [1.4,2,2.5],
    #     'entry_delta': [0.5,1,1.5,2],
    #     # 'entry_delta': [1,2,3],
    #     # 'exit_delta': [-2,0],
    #     # # 'stop_loss_delta': [2,4],
    #     # 'pair_sort': ['p_value','std_asc','std_desc'],
    #     # 'force_close_days':[21],
    #     # 'pair_sort': ['std_desc'],
    #     # 'pair_sort': ['std_desc'],
    #     # # 'slippage': [0.005],
    #     'back_window_ratio': [30,60,90],
    #     # 'cheat-on-close': [1],
    #     # 'coint_lookback': [4,0],
    #     # 'p_value_max': [0.05, 0.01],
    #     # 'cash': [1000, 10000],
    #     # 'slippage_compensation': [0.0001,0.0008,0.0015]
    #     'slippage_compensation': [0,0.0001]
    # }
    ####Nasdaq
    tested_trade_parameter = {
        # 'leverage_ratio': [3],
        'in_sample_stat_en': [1,0],
        'stop_loss_delta': [-0.05,-0.1],
        # 'slippage_compensation': [0.005,0.01],
        # 'out_of_sample_days': [2,30],
        # 'out_of_sample_days': [2],
        # 'force_close_days': [21,42],
        'in_sample_days': [30,60], 
        # 'in_sample_days': [30], 
        # # # 'in_sample_days': [30,150], 
        # # # 'sector_key': ['sector','industry'],
        'num_pairs': [5,10],
        'corr_cutoff': [0,0.75],
        'crossing_cutoff': [4,6,8,10],
        # 'entry_delta': [0.8,1,1.2,1.4,1.6,1.8,2],
        # 'entry_delta': [1.4,2,2.5],
        'entry_delta': [1,1.5,2],
        # 'entry_delta': [1,2,3],
        # 'exit_delta': [-2,0],
        # # 'stop_loss_delta': [2,4],
        'pair_sort': ['crossing','std_desc'],
        'force_close_ratio':[50,95],
        # 'pair_sort': ['std_desc'],
        # 'pair_sort': ['std_desc'],
        # # 'slippage': [0.005],
        'back_window_ratio': [50,95],
        # 'cheat-on-close': [1],
        # 'coint_lookback': [4,0],
        # 'p_value_max': [0.05, 0.01],
        # 'cash': [1000, 10000],
        # 'slippage_compensation': [0.0001,0.0008,0.0015]
        'slippage_compensation': [0.0001,0.0005,0.001]
        # 'slippage_compensation': [0.002]
    }

    # Split trade parameters based on the in_sample_stat_en value
    trade_params_with_back_window = {key: value for key, value in tested_trade_parameter.items() if key != 'in_sample_stat_en'}
    trade_params_without_back_window = {key: value for key, value in trade_params_with_back_window.items() if key != 'back_window_ratio'}
    # Generate combinations based on in_sample_stat_en
    combinations = []

    # Combinations with in_sample_stat_en == 1
    keys_with_back_window, values_with_back_window = zip(*trade_params_with_back_window.items())
    for v in product(*values_with_back_window):
        combo = dict(zip(keys_with_back_window, v))
        combo['in_sample_stat_en'] = 1
        combinations.append(combo)

    # Combinations with in_sample_stat_en == 0
    if tested_trade_parameter['in_sample_stat_en'] == [1,0]: 
        keys_without_back_window, values_without_back_window = zip(*trade_params_without_back_window.items())
        for v in product(*values_without_back_window):
            combo = dict(zip(keys_without_back_window, v))
            combo['in_sample_stat_en'] = 0
            combinations.append(combo)

    completed_combinations = 0
    # Merge each combination with the base trade parameters
    all_trade_parameters = []

    for combo in combinations:
        trade_param = deepcopy(base_trade_parameter)
        trade_param.update(combo)
        all_trade_parameters.append(trade_param)

    print('Total combinations: ', len(all_trade_parameters))

    if baseline_only_mode:
        all_trade_parameters = []
        all_trade_parameters.append(base_trade_parameter)
    
    return all_trade_parameters

def _decode_imessage_body(text, attributed_body):
    if text:
        return str(text)
    if not attributed_body:
        return ""

    if isinstance(attributed_body, memoryview):
        attributed_body = attributed_body.tobytes()
    if not isinstance(attributed_body, bytes):
        return str(attributed_body)

    decoded = attributed_body.decode("utf-8", errors="ignore")
    chunks = re.findall(r"[\x20-\x7E]{4,}", decoded)
    return " ".join(chunks)


def _imessage_datetime(raw_date):
    if raw_date is None:
        return None
    try:
        raw = float(raw_date)
    except (TypeError, ValueError):
        return None

    abs_raw = abs(raw)
    if abs_raw > 100_000_000_000_000:
        seconds_since_2001 = raw / 1_000_000_000
    elif abs_raw > 100_000_000_000:
        seconds_since_2001 = raw / 1_000_000
    else:
        seconds_since_2001 = raw

    return datetime.fromtimestamp(978307200 + seconds_since_2001)


def _extract_mfa_code_from_message(message_text, sender=None):
    if not message_text:
        return None

    match = re.search(r"\b(\d{6})\b", message_text)
    if not match:
        return None

    combined = f"{sender or ''} {message_text}".lower()
    default_hints = [
        "e*trade",
        "etrade",
        "e-trade",
        "morgan stanley",
        "security code",
        "verification code",
        "authentication code",
        "authorization code",
        "login code",
        "one-time",
        "one time",
        "passcode",
    ]
    extra_hints = [
        hint.strip().lower()
        for hint in os.getenv("ETRADE_MFA_MESSAGE_HINTS", "").split(",")
        if hint.strip()
    ]

    if not any(hint in combined for hint in default_hints + extra_hints):
        return None

    return match.group(1)


def _copy_messages_db(db_path):
    import shutil
    import tempfile

    tmp_dir = tempfile.TemporaryDirectory(prefix="etrade_messages_")
    tmp_db_path = os.path.join(tmp_dir.name, "chat.db")
    shutil.copy2(db_path, tmp_db_path)
    for suffix in ("-wal", "-shm"):
        src = f"{db_path}{suffix}"
        if os.path.exists(src):
            shutil.copy2(src, f"{tmp_db_path}{suffix}")
    return tmp_dir, tmp_db_path


def _wake_messages_app():
    if sys.platform != "darwin":
        return
    try:
        import subprocess
        subprocess.run(
            ["open", "-gj", "-a", "Messages"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        pass


def _get_latest_imessage_code(timeout_minutes=5, since=None, query_limit=500, log_status=False):
    """
    Attempts to read the latest E*TRADE security code from the macOS iMessage database.
    Requires 'Full Disk Access' permissions for the terminal/app.
    """
    db_path = os.path.expanduser("~/Library/Messages/chat.db")
    if not os.path.exists(db_path):
        if log_status:
            print(f"iMessage database not found at {db_path}.")
        return None

    db_snapshot = None
    try:
        try:
            db_snapshot = _copy_messages_db(db_path)
            read_path = db_snapshot[1]
        except Exception:
            read_path = db_path

        conn = sqlite3.connect(f"file:{read_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        query = """
            SELECT
                message.date,
                message.text,
                message.attributedBody,
                handle.id
            FROM
                message
            LEFT JOIN
                handle ON message.handle_id = handle.ROWID
            WHERE
                message.is_from_me = 0
            ORDER BY
                message.date DESC
            LIMIT ?
        """
        cursor.execute(query, (query_limit,))
        rows = cursor.fetchall()
        conn.close()

        cutoff = since or (datetime.now() - timedelta(minutes=timeout_minutes))
        stale_cutoff = datetime.now() - timedelta(minutes=timeout_minutes)
        if cutoff < stale_cutoff:
            cutoff = stale_cutoff

        latest_inbound_time = None
        latest_matching_time = None
        scanned_recent = 0
        scanned_matching = 0

        for raw_date, text, attributed_body, sender in rows:
            msg_time = _imessage_datetime(raw_date)
            if not msg_time:
                continue

            if latest_inbound_time is None or msg_time > latest_inbound_time:
                latest_inbound_time = msg_time

            message_text = _decode_imessage_body(text, attributed_body)
            candidate_code = _extract_mfa_code_from_message(message_text, sender=sender)
            if candidate_code:
                scanned_matching += 1
                if latest_matching_time is None or msg_time > latest_matching_time:
                    latest_matching_time = msg_time

            if msg_time >= cutoff:
                scanned_recent += 1
                if candidate_code:
                    print(f"Auto-extracted MFA code from iMessage (sent at {msg_time:%Y-%m-%d %H:%M:%S}).")
                    return candidate_code

        if log_status:
            cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            if latest_matching_time:
                age = datetime.now() - latest_matching_time
                age_minutes = max(0.0, age.total_seconds() / 60.0)
                print(
                    "No recent E*TRADE MFA SMS found in iMessage "
                    f"after {cutoff_text}. Latest matching SMS is {age_minutes:.1f} minutes old "
                    f"({latest_matching_time:%Y-%m-%d %H:%M:%S})."
                )
            elif latest_inbound_time:
                print(
                    "No E*TRADE MFA SMS found in recent iMessage rows. "
                    f"Latest inbound message is {latest_inbound_time:%Y-%m-%d %H:%M:%S}; "
                    f"scanned {len(rows)} rows, {scanned_recent} after cutoff."
                )
            else:
                print(f"No inbound iMessage rows found while scanning {len(rows)} rows after cutoff {cutoff_text}.")
    except sqlite3.OperationalError as e:
        if "unable to open database file" in str(e) or "access denied" in str(e).lower():
            print("Warning: Access to iMessage database denied. Please grant 'Full Disk Access' to your Terminal/IDE in System Settings.")
        else:
            print(f"Error reading iMessage database: {e}")
    except Exception as e:
        print(f"Unexpected error reading iMessage: {e}")
    finally:
        if db_snapshot:
            db_snapshot[0].cleanup()

    return None

class LoginFailureException(Exception):
    """Custom exception for login failures, optionally containing a screenshot path."""
    def __init__(self, message, screenshot_path=None):
        super().__init__(message)
        self.screenshot_path = screenshot_path

def get_token_automated(oauth_url, username=None, password=None, headless=True, browser='chrome'):
    """
    Automates the OAuth process using Selenium.
    Prioritizes Chrome in Headless mode for robustness against screen locking.
    Falls back to Safari if Chrome is unavailable (but Safari requires unlocked screen).
    """
    driver = None
    
    # Try Chrome first if requested
    if browser.lower() == 'chrome':
        try:
            print("Attempting to launch Chrome WebDriver...")
            options = webdriver.ChromeOptions()
            
            # Stealth flags and realistic User-Agent to bypass bot detection (Status 942)
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
            
            if headless:
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--window-size=1920,1080")
            
            # Reduce logging noise
            options.add_argument("--log-level=3")
            
            driver = webdriver.Chrome(options=options)
            
            # Remove the navigator.webdriver property
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
            
            print("Chrome WebDriver launched successfully.")
        except Exception as e:
            print(f"Failed to launch Chrome: {e}. Falling back to Safari...")
            browser = 'safari'  # Fallback

    # Fallback to Safari
    if browser.lower() == 'safari':
        try:
            print("Attempting to launch Safari WebDriver...")
            driver = webdriver.Safari()
        except SessionNotCreatedException as exc:
            raise RuntimeError(
                "Unable to start Safari WebDriver session. In Safari, open Preferences → Advanced, enable the Develop menu, "
                "then choose Develop → Allow Remote Automation. You may also need to run 'safaridriver --enable' once from the terminal."
            ) from exc

    if not driver:
        raise RuntimeError("No suitable WebDriver could be initialized.")

    try:
        driver.get(oauth_url)
        WebDriverWait(driver, 30).until(lambda d: d.execute_script("return document.readyState") == "complete")

        # Check for scheduled maintenance immediately after loading the page
        try:
            page_title = (driver.title or "").lower()
            page_text = driver.execute_script("return document.body ? document.body.innerText : '';").lower()
            maintenance_keywords = [
                "maintenance", "temporarily unavailable", "system unavailable", 
                "service unavailable", "down for maintenance", "scheduled maintenance", 
                "schedule maintenance", "scheduled maintainance", "schedule maintainance"
            ]
            if any(kw in page_title for kw in maintenance_keywords) or any(kw in page_text for kw in maintenance_keywords):
                print("🛑 E*TRADE Scheduled Maintenance detected via page content/title.")
                raise LoginFailureException("E*TRADE Scheduled Maintenance: The site is temporarily unavailable due to maintenance.")
        except LoginFailureException:
            raise
        except Exception as check_err:
            print(f"Could not check for maintenance page: {check_err}")

        print("Automating the login process.")

        def _find_first(locators, condition=EC.presence_of_element_located, timeout=20):
            try:
                # Use any_of to check all locators simultaneously instead of sequentially
                return WebDriverWait(driver, timeout).until(EC.any_of(*[condition(loc) for loc in locators]))
            except (TimeoutException, AttributeError, Exception):
                return None

        def _extract_token_from_text(text):
            if not text:
                return None

            match = re.search(r"oauth[_-]?verifier[:=\s]+([A-Za-z0-9-_]+)", text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            for line in text.splitlines():
                candidate = line.strip()
                if candidate.lower() in ['banking', 'log on', 'etrade', 'accounts', 'markets', 'research', 'transfer', 'support']:
                    continue
                if 6 <= len(candidate) <= 128 and re.fullmatch(r"[A-Za-z0-9-_]+", candidate):
                    return candidate
            return None

        def _select_remember_device_if_present():
            try:
                selected = driver.execute_script("""
                    const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                    const bodyText = normalize(document.body ? document.body.innerText : '');
                    if (!bodyText.includes('save this device') && !bodyText.includes('remember this device')) {
                        return false;
                    }

                    const chooseRadio = (radio) => {
                        if (!radio) {
                            return false;
                        }
                        radio.scrollIntoView({block: 'center', inline: 'center'});
                        radio.click();
                        radio.checked = true;
                        radio.dispatchEvent(new Event('input', {bubbles: true}));
                        radio.dispatchEvent(new Event('change', {bubbles: true}));
                        return true;
                    };

                    for (const label of Array.from(document.querySelectorAll('label'))) {
                        const labelText = normalize(label.innerText || label.textContent);
                        if (labelText.includes('yes') && labelText.includes('save this device')) {
                            const radio = label.htmlFor
                                ? document.getElementById(label.htmlFor)
                                : label.querySelector('input[type="radio"]');
                            if (radio) {
                                return chooseRadio(radio);
                            }
                            label.scrollIntoView({block: 'center', inline: 'center'});
                            label.click();
                            return true;
                        }
                    }

                    for (const radio of Array.from(document.querySelectorAll('input[type="radio"]'))) {
                        const parts = [
                            radio.getAttribute('aria-label'),
                            radio.value,
                            radio.id,
                            radio.name,
                            radio.nextElementSibling ? radio.nextElementSibling.innerText : '',
                            radio.parentElement ? radio.parentElement.innerText : '',
                        ];
                        const radioText = normalize(parts.join(' '));
                        if (radioText.includes('yes') && radioText.includes('save this device')) {
                            return chooseRadio(radio);
                        }
                    }

                    for (const el of Array.from(document.querySelectorAll('[role="radio"], button, span, div'))) {
                        const text = normalize(el.innerText || el.textContent);
                        if (text === 'yes, save this device.' || text === 'yes, save this device') {
                            el.scrollIntoView({block: 'center', inline: 'center'});
                            el.click();
                            return true;
                        }
                    }

                    return false;
                """)
                if selected:
                    print("Selected MFA remember-device option.")
                return bool(selected)
            except Exception as e:
                print(f"Could not select MFA remember-device option: {e}")
                return False


        username_locators = [
            (By.ID, "USER"),
            (By.ID, "user_orig"),
            (By.NAME, "USER"),
            (By.CSS_SELECTOR, "input[name='USER']"),
            (By.CSS_SELECTOR, "input[name='user']"),
        ]
        password_locators = [
            (By.ID, "password"),
            (By.ID, "password_orig"),
            (By.NAME, "PASSWORD"),
            (By.CSS_SELECTOR, "input[type='password']"),
        ]
        login_button_locators = [
            (By.ID, "mfaLogonButton"),
            (By.ID, "logOnbtn"),
            (By.CSS_SELECTOR, "button[type='submit']"),
        ]

        username_field = _find_first(username_locators)
        password_field = _find_first(password_locators)
        login_button = _find_first(login_button_locators, EC.element_to_be_clickable)

        if username_field and password_field and login_button and username and password:
            print("Login fields found. Using ActionChains for React-compatible input...")
            try:
                from selenium.webdriver.common.action_chains import ActionChains
                
                # Clear and focus username field, then type character by character
                username_field.click()
                t.sleep(0.3)
                
                # Clear existing content using keyboard shortcuts
                actions = ActionChains(driver)
                actions.key_down(Keys.COMMAND).send_keys('a').key_up(Keys.COMMAND).perform()
                t.sleep(0.1)
                actions = ActionChains(driver)
                actions.send_keys(Keys.DELETE).perform()
                t.sleep(0.1)
                
                # Type username character by character (more reliable for React)
                for char in username:
                    actions = ActionChains(driver)
                    actions.send_keys(char).perform()
                    t.sleep(0.01)  # Faster typing
                
                print(f"Username typed: {username[:3]}***")
                
                # Move to password field
                password_field.click()
                t.sleep(0.1)
                
                # Clear password field
                actions = ActionChains(driver)
                actions.key_down(Keys.COMMAND).send_keys('a').key_up(Keys.COMMAND).perform()
                t.sleep(0.05)
                actions = ActionChains(driver)
                actions.send_keys(Keys.DELETE).perform()
                t.sleep(0.05)
                
                # Type password character by character
                for char in password:
                    actions = ActionChains(driver)
                    actions.send_keys(char).perform()
                    t.sleep(0.01)
                
                print("Password typed: ***")
                
                # Verify values were set by checking the DOM
                u_val = driver.execute_script("return arguments[0].value;", username_field)
                p_val = driver.execute_script("return arguments[0].value;", password_field)
                
                if u_val:
                    print(f"Username field verified: {u_val[:3]}***")
                else:
                    print("WARNING: Username field still empty after typing!")
                    
                if p_val:
                    print("Password field verified: ***")
                else:
                    print("WARNING: Password field still empty after typing!")
                
                # Small delay before click
                t.sleep(0.2)
                
                print("Clicking login button...")
                actions = ActionChains(driver)
                actions.move_to_element(login_button).click().perform()
                
                # Wait for page transition after login
                print("Waiting for login to process...")
                # Removed fixed 5s sleep; state machine below will handle polling immediately.
                
            except Exception as e:
                print(f"Error interacting with login form: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Unable to locate login form automatically. Please complete login manually.")

        accept_locators = [
            (By.ID, "acceptSubmit"),
            (By.ID, "acceptBtn"),
            (By.CSS_SELECTOR, "button#accept"),
            (By.CSS_SELECTOR, "button[data-testid='oauth-accept']"),
            (By.NAME, "accept"),
        ]

        # MFA / Security Code Locators
        mfa_phone_choice_locators = [
            (By.XPATH, "//label[contains(., 'iPhone') or contains(., 'Text') or contains(., 'SMS')]"),
            (By.CSS_SELECTOR, "input[type='radio']"),
        ]
        mfa_input_locators = [
            (By.CSS_SELECTOR, "input[name='SecurityCode']"),
            (By.CSS_SELECTOR, "input.security-code"),
            (By.ID, "securityCode"),
            (By.ID, "passcode"),
            (By.CSS_SELECTOR, "input[name='otp']"),
            (By.CSS_SELECTOR, "input[name='passcode']"),
            (By.CSS_SELECTOR, "input[type='tel']"),
            (By.CSS_SELECTOR, "input[autocomplete='one-time-code']"),
            (By.CSS_SELECTOR, "input[inputmode='numeric']"),
            (By.CSS_SELECTOR, "input[placeholder*='Security Code']"),
            (By.CSS_SELECTOR, "input[placeholder*='Verification Code']"),
            (By.CSS_SELECTOR, "input[placeholder*='code']"),
        ]
        mfa_submit_locators = [
            (By.ID, "previewSubmit"),
            (By.CSS_SELECTOR, "button[data-testid='submit-security-code']"),
            (By.XPATH, "//button[contains(text(), 'Submit')]"),
            (By.XPATH, "//button[contains(text(), 'Verify')]"),
            (By.XPATH, "//button[contains(text(), 'Continue')]"),
            (By.CSS_SELECTOR, "input[type='submit']"),
        ]
        mfa_send_code_locators = [
            (By.ID, "sendOTPCodeBtn"),
            (By.CSS_SELECTOR, "button#sendOTPCodeBtn"),
            (By.XPATH, "//button[contains(text(), 'Send Code')]"),
            (By.XPATH, "//button[contains(text(), 'Send code')]"),
            (By.XPATH, "//button[contains(text(), 'Send text')]"),
            (By.XPATH, "//button[contains(text(), 'Text me')]"),
            (By.XPATH, "//input[contains(@value, 'Send Code')]"),
            (By.XPATH, "//input[contains(@value, 'Send code')]"),
        ]

        token_locators = [
            (By.CSS_SELECTOR, "input#oauth_verifier"),
            (By.CSS_SELECTOR, "input#oauthVerifier"),
            (By.CSS_SELECTOR, "input[name='verifier']"),
            (By.CSS_SELECTOR, "input[name='oauth_verifier']"),
            (By.CSS_SELECTOR, "div.api-inner-container input[type='text']"),
            (By.CSS_SELECTOR, "code"),
        ]

        initial_handles = driver.window_handles[:]
        
        # Unified state machine loop to find the token or required interaction
        print("Waiting for Token OR Consent OR MFA challenge...")
        
        token = None
        max_duration = int(os.getenv("ETRADE_LOGIN_WAIT_SECONDS", "180"))
        start_wait = t.time()
        mfa_code_requested_at = None
        
        while (t.time() - start_wait) < max_duration:
            # Check for scheduled maintenance inside loop
            try:
                page_title = (driver.title or "").lower()
                page_text = driver.execute_script("return document.body ? document.body.innerText : '';").lower()
                maintenance_keywords = [
                    "maintenance", "temporarily unavailable", "system unavailable", 
                    "service unavailable", "down for maintenance", "scheduled maintenance", 
                    "schedule maintenance", "scheduled maintainance", "schedule maintainance"
                ]
                if any(kw in page_title for kw in maintenance_keywords) or any(kw in page_text for kw in maintenance_keywords):
                    print("🛑 E*TRADE Scheduled Maintenance detected inside loop.")
                    raise LoginFailureException("E*TRADE Scheduled Maintenance: The site is temporarily unavailable due to maintenance.")
            except LoginFailureException:
                raise
            except Exception:
                pass

            # 1. Success Condition: Check for the Token first (it might be already there)
            # Try JS extraction first for speed
            try:
                js_token = driver.execute_script(
                    "var el = document.querySelector('div.api-inner-container input[type=\\'text\\']') || document.querySelector('input#oauth_verifier') || document.querySelector('input#oauthVerifier'); return el ? el.value : null;"
                )
                if js_token and str(js_token).strip() and len(str(js_token).strip()) > 3:
                    token = str(js_token).strip()
                    print("OAuth verifier obtained via browser automation.")
                    return token
            except:
                pass

            # 2. Proactive wait for any valid next state (Token, Consent, MFA, etc.)
            try:
                all_conditions = []
                for loc in token_locators: all_conditions.append(EC.presence_of_element_located(loc))
                for loc in accept_locators: all_conditions.append(EC.element_to_be_clickable(loc))
                for loc in mfa_phone_choice_locators: all_conditions.append(EC.element_to_be_clickable(loc))
                for loc in mfa_send_code_locators: all_conditions.append(EC.element_to_be_clickable(loc))
                for loc in mfa_input_locators: all_conditions.append(EC.visibility_of_element_located(loc))
                
                # Wait up to 2 seconds for ANY condition to be met
                WebDriverWait(driver, 2).until(EC.any_of(*all_conditions))
            except TimeoutException:
                pass

            # 3. Check for Token Element (Standard Selenium)
            token_element = _find_first(token_locators, EC.presence_of_element_located, timeout=0.1)
            if token_element:
                val = token_element.get_attribute("value") or token_element.text
                if val and len(val.strip()) > 3:
                    token = val.strip()
                    print("OAuth verifier obtained via browser automation.")
                    return token

            # 4. Check for Accept/Authorize Button
            accept_button = _find_first(accept_locators, EC.element_to_be_clickable, timeout=0.1)
            if accept_button:
                print("Consent button found. clicking...")
                try:
                    accept_button.click()
                    continue # Re-evaluate state immediately
                except:
                    pass

            # 5. Check for MFA device choice and "Send Code" button (Choice screen)
            mfa_phone_choice = _find_first(mfa_phone_choice_locators, EC.element_to_be_clickable, timeout=0.1)
            if mfa_phone_choice:
                try:
                    mfa_phone_choice.click()
                except:
                    pass

            mfa_send_code = _find_first(mfa_send_code_locators, EC.element_to_be_clickable, timeout=0.1)
            if mfa_send_code:
                if not mfa_code_requested_at or datetime.now() - mfa_code_requested_at > timedelta(seconds=30):
                    print("MFA 'Send Code' button found. Triggering SMS...")
                    try:
                        mfa_code_requested_at = datetime.now() - timedelta(seconds=10)
                        mfa_send_code.click()
                        _wake_messages_app()
                        continue # Re-evaluate state immediately
                    except:
                        pass

            # 6. Check for MFA Challenge (Input screen)
            mfa_input = _find_first(mfa_input_locators, EC.visibility_of_element_located, timeout=0.1)
            if mfa_input:
                print("\n" + "!"*40 + "\nMFA CHALLENGE DETECTED!\n" + "!"*40 + "\n")
                
                # Attempt auto-extraction from iMessage
                security_code = None
                print("Checking iMessage for security code...")
                _wake_messages_app()
                mfa_wait_seconds = int(os.getenv("ETRADE_MFA_WAIT_SECONDS", "90"))
                poll_count = max(1, int(mfa_wait_seconds / 2.5))
                code_since = (mfa_code_requested_at or datetime.now()) - timedelta(minutes=2)
                for i in range(poll_count): # SMS forwarding to Messages can lag.
                    security_code = _get_latest_imessage_code(
                        timeout_minutes=10,
                        since=code_since,
                        log_status=(i == 0 or i == poll_count - 1),
                    )
                    if security_code:
                        print("Found security code in iMessage.")
                        break
                    t.sleep(2.5)
                
                if not security_code:
                    if headless or not sys.stdin.isatty():
                        raise LoginFailureException("MFA code required, but no recent iMessage code was found.")
                    print("Auto-extraction failed. Falling back to manual input.")
                    security_code = input(">> Please enter code: ").strip()
                
                if security_code:
                    print("Submitting MFA code...")
                    mfa_input.clear()
                    mfa_input.send_keys(security_code)
                    _select_remember_device_if_present()
                    mfa_submit = _find_first(mfa_submit_locators, EC.element_to_be_clickable, timeout=1)
                    if mfa_submit:
                        mfa_submit.click()
                    else:
                        mfa_input.send_keys(Keys.RETURN)
                    # No fixed sleep here, loop will continue and wait for result
                    continue 
            
            # 7. Brief sleep before next state check
            t.sleep(0.5)
            print(f"Still waiting... ({int(t.time() - start_wait)}s elapsed)")

        # Timeout cleanup if we exit the loop without return
        print("Extraction flow timed out.")

        # Final extraction attempts if the loop timed out
        if not token:
            print("Unified loop timed out. Trying final fallbacks...")
            try:
                current_url = driver.current_url or ""
                parsed_query = parse_qs(urlparse(current_url).query)
                token_candidates = parsed_query.get('oauth_verifier') or parsed_query.get('verifier')
                if token_candidates:
                    token = unquote(token_candidates[0]).strip()
                    print("OAuth verifier obtained from the authorization redirect.")
                    return token
            except:
                pass

            # Body text extraction
            try:
                page_text = driver.execute_script("return document.body ? document.body.innerText : '';")
                token = _extract_token_from_text(page_text)
                if token:
                    print("OAuth verifier obtained from the authorization page.")
                    return token
            except:
                pass

            # Only now save failure info
            descriptor, screenshot_path = tempfile.mkstemp(
                prefix="login_failure_",
                suffix=".png",
                dir=os.getcwd(),
            )
            os.close(descriptor)
            try:
                driver.save_screenshot(screenshot_path)
                os.chmod(screenshot_path, 0o600)
                print(f"Extraction failed. Saved failure screenshot to: {screenshot_path}")
            except Exception:
                try:
                    os.unlink(screenshot_path)
                except OSError:
                    pass
                screenshot_path = None
            
            raise LoginFailureException("Unable to retrieve OAuth verifier token automatically.", screenshot_path=screenshot_path)

        return token


    finally:
        if driver:
            driver.quit()

    return token

class LiveTradeAgent:
    def __init__(self,
                agent_id = None,
                authenticated_session = None,
                base_url = None,
                selected_account = None,
                use_sandbox = None,
                expected_account_id_key = None,
                expected_account_id = None,
                expected_institution_type = None,
                consumer_key = None,
                runtime_safety: RuntimeSafetyBoundary | None = None,
                ):
        if agent_id is None:
            self.agent_id = self.generate_agent_id()
            print('Live Trade Agent ID: ', self.agent_id)
        else:
            self.agent_id = agent_id

        if authenticated_session is not None and base_url is not None:
            if runtime_safety is None:
                raise RuntimeError(
                    "Authenticated LiveTradeAgent construction requires a RuntimeSafetyBoundary"
                )
            selected = runtime_safety.account_selection_kwargs["selected_account_id"]
            expected = (
                runtime_safety.expected_account_id_key,
                runtime_safety.expected_account_id,
                runtime_safety.expected_institution_type,
            )
            supplied = (expected_account_id_key, expected_account_id, expected_institution_type)
            if (
                use_sandbox is not None and use_sandbox != runtime_safety.use_sandbox
            ) or (selected_account is not None and selected_account != selected) or any(
                provided is not None and provided != armed
                for provided, armed in zip(supplied, expected)
            ):
                raise RuntimeError("LiveTradeAgent arguments conflict with the RuntimeSafetyBoundary")
            self.use_sandbox = runtime_safety.use_sandbox
            self.selected_account = selected
            self.expected_account_id_key = runtime_safety.expected_account_id_key
            self.expected_account_id = runtime_safety.expected_account_id
            self.expected_institution_type = runtime_safety.expected_institution_type
            self.runtime_safety = runtime_safety
            self.consumer_key = consumer_key
            self._initialize_components(authenticated_session, base_url)
            self.runtime_safety.verify_account(self.account.account)
        else:
            self.use_sandbox = bool(use_sandbox)
            self.selected_account = selected_account
            self.expected_account_id_key = expected_account_id_key
            self.expected_account_id = expected_account_id
            self.expected_institution_type = expected_institution_type
            self.runtime_safety = None
            self.consumer_key = consumer_key
        self.today_stock_price = []
        self.today_stock_spread = []

    def _initialize_components(self, authenticated_session, base_url):
        """Initialize or refresh dependent services with a new authenticated session."""
        self.market = Market(
            authenticated_session, base_url, use_sandbox=self.use_sandbox, consumer_key=self.consumer_key
        )
        self.account = Accounts(
            authenticated_session,
            base_url,
            use_sandbox=self.use_sandbox,
            consumer_key=self.consumer_key,
        )
        self.account.account_list(
            self.selected_account,
            expected_account_id_key=self.expected_account_id_key,
            expected_account_id=self.expected_account_id,
            expected_institution_type=self.expected_institution_type,
        )
        account_selected = self.account.account
        self.order = Order(
            authenticated_session,
            account_selected,
            base_url,
            use_sandbox=self.use_sandbox,
            consumer_key=self.consumer_key,
            runtime_safety=self.runtime_safety,
        )

    def refresh_session(self, authenticated_session, base_url):
        """
        Refresh the agent with a new authenticated session.
        Keeps agent/account IDs but rebuilds market/account/order clients.
        """
        self._initialize_components(authenticated_session, base_url)
        if self.runtime_safety is None:
            raise RuntimeError("Authenticated LiveTradeAgent refresh requires a RuntimeSafetyBoundary")
        self.runtime_safety.verify_account(self.account.account)

    @staticmethod
    def generate_agent_id():
        return f"{random.randint(10000000, 99999999)}"

    def list_account_id(self):
        """
        List all account IDs associated with the authenticated user.
        
        :param session: An authenticated E*TRADE session object
        :return: A list of account IDs
        """
        account_info = self.account.account_list(self.selected_account)
        for this_account_info in account_info:
            print(this_account_info[0])

    def poll_market_data(self, ticker_list):
        chunk_size = 25
        counter = 0
        stockInfo: StockInfo = []

        for i in range(0, len(ticker_list), chunk_size):
            this_tickerlist = ticker_list[i:i + chunk_size]
            market_quote = self.market.get_quote(this_tickerlist)
            stockInfo = stockInfo + extract_stock_info(market_quote, get_history_spread=0, source='etrade')

        for this_stockInfo in stockInfo:
            if this_stockInfo.ask_bid_spread_est > 0.1:
                print(counter, this_stockInfo.symbol, this_stockInfo.ask_bid_spread, this_stockInfo.trade_qty_limit, this_stockInfo.ask_price, this_stockInfo.bid_price, this_stockInfo.close_price)
                counter += 1
            price = {
                'ticker': this_stockInfo.symbol,
                'close_price': this_stockInfo.close_price,
            }
            spread = {
                'ticker': this_stockInfo.symbol,
                'ask_bid_spread': this_stockInfo.ask_bid_spread
            }
            self.today_stock_price.append(price)
            self.today_stock_spread.append(spread)

    def merge_price_df(self, close_prices_df, full_prices_df, today_stock_price_df):
        today_date = datetime.now().strftime('%Y-%m-%d')
        today_date_pd = pd.to_datetime(today_date)
        
        # Extract new data for close prices
        new_close_data = {item['ticker']: item['close_price'] for item in today_stock_price_df}
        new_close_entry = pd.DataFrame(new_close_data, index=[today_date_pd])
        
        # Merge new close prices into close_prices_df
        close_prices_df = pd.concat([close_prices_df, new_close_entry])

        # Copy the last row of full_prices_df to a new row indexed by today's date
        last_row = full_prices_df.iloc[-1].copy()
        new_row = pd.DataFrame([last_row], index=[today_date_pd])
        
        # Update the 'Close' price in the new row to match today's stock prices
        for item in today_stock_price_df:
            ticker = item['ticker']
            close_price = item['close_price']
            new_row[(ticker, 'Open')] = close_price
            new_row[(ticker, 'High')] = close_price
            new_row[(ticker, 'Low')] = close_price
            new_row[(ticker, 'Close')] = close_price
            new_row[(ticker, 'Adj Close')] = close_price  # Assuming Adj Close should also be updated

        # Append the new row to full_prices_df
        full_prices_df = pd.concat([full_prices_df, new_row])

        return close_prices_df, full_prices_df

    def generate_etrade_order(self, positions: list[Position], custom_order_ids: list) -> list:
        print(f"Generating etrade orders, {len(positions)} positions")
        etrade_orders = []

        for position, custom_order_id in zip(positions, custom_order_ids):
            symbols = position.pair_str.split(':')
            stock_a = symbols[0]
            stock_b = symbols[1]

            order_data_A = {
                'symbol': stock_a,
                'quantity': position.shares_a,
                'submit_date': position.open_date.strftime('%Y-%m-%d'),
                'orderAction': '',
                'orderType': 'MARKET',
                'priceType': 'MARKET',
                'orderTerm': 'GOOD_FOR_DAY',
                'limitPrice': '',
                'client_order_id': custom_order_id
            }
            order_data_B = {
                'symbol': stock_b,
                'quantity': position.shares_b,
                'submit_date': position.open_date.strftime('%Y-%m-%d'),
                'orderAction': '',
                'orderType': 'MARKET',
                'priceType': 'MARKET',
                'orderTerm': 'GOOD_FOR_DAY',
                'limitPrice': '',
                'client_order_id': custom_order_id
            }

            if position.reason_to_close == 'None':
                open_or_close = 'open'
            else:
                open_or_close = 'close'
            if position.position_type == pairs_trading_bo.OpenPosition.LONG_A_SHORT_B:
                if open_or_close == 'open':
                    order_data_A['orderAction'] = 'BUY'
                    order_data_B['orderAction'] = 'SELL_SHORT'
                else:
                    order_data_A['orderAction'] = 'SELL'
                    order_data_B['orderAction'] = 'BUY_TO_COVER'
            elif position.position_type == pairs_trading_bo.OpenPosition.SHORT_A_LONG_B:
                if open_or_close == 'open':
                    order_data_A['orderAction'] = 'SELL_SHORT'
                    order_data_B['orderAction'] = 'BUY'
                else:
                    order_data_A['orderAction'] = 'BUY_TO_COVER'
                    order_data_B['orderAction'] = 'SELL'
            if position.shares_a != 0:
                etrade_orders.append(order_data_A)
            if position.shares_b != 0:
                etrade_orders.append(order_data_B)

        return etrade_orders

    def update_csv_with_order_info(self, order_id, etrade_order):
        file_path = f'live_trade_tracker/{self.agent_id}_order_book.csv'
        
        if not os.path.exists(file_path):
            print(f"No order book found for agent_id: {self.agent_id}")
            return
        
        # Read the CSV file into a list of dictionaries
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        # Update the row with the matching client_order_id
        for row in rows:
            if row['Custom Order ID'] == str(etrade_order['client_order_id']):
                row['Order ID'] = order_id
                row['Order Submitted Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                row['orderAction'] = etrade_order['orderAction']
                spread_dict = next((item for item in self.today_stock_spread if item['ticker'] == row['Symbol']), None)
                row['Ask_bid_spread'] = spread_dict['ask_bid_spread']
                break
        
        # Write the updated rows back to the CSV
        with open(file_path, mode='w', newline='') as file:
            fieldnames = reader.fieldnames
            if 'Order ID' not in fieldnames:
                fieldnames.append('Order ID')
            if 'Order Submitted Time' not in fieldnames:
                fieldnames.append('Order Submitted Time')
            if 'orderAction' not in fieldnames:
                fieldnames.append('orderAction')
            if 'Ask_bid_spread' not in fieldnames:
                fieldnames.append('Ask_bid_spread')

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def update_csv_with_execution_details(self, executed_orders: list):
        file_path = f'live_trade_tracker/{self.agent_id}_order_book.csv'
        
        if not os.path.exists(file_path):
            print(f"No order book found for agent_id: {self.agent_id}")
            return
        
        # Read the CSV file into a list of dictionaries
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        
        print('Number of rows before editting: ',len(rows))
        # Update the rows with the execution details
        for order in executed_orders:
            for row in rows:
                if row['Order ID'] == str(order['order_id']):
                    row['Executed Price'] = order['executed_price']
                    row['Executed Quantity'] = order['executed_quantity']
                    row['Executed Time'] = order['executed_date']
                    row['Executed Symbol'] = order['symbol']
                    if order['executed_price'] is not None:
                        if row['orderAction'] == 'BUY' or row['orderAction'] == 'BUY_TO_COVER': 
                            slippage = max(float(row['Price A']) , float(row['Price B'])) - float(order['executed_price'])
                        if row['orderAction'] == 'SELL' or row['orderAction'] == 'SELL_SHORT': 
                            slippage = float(order['executed_price']) - max(float(row['Price A']) , float(row['Price B'])) 
                        row['Price slip'] = round(slippage,3)
                    # print('matched! ',row['Symbol'])
                    break

        # Calculate Expected Gain for closed positions
        for row in rows:
            if row['Open/Close'] == 'close':
                pair_str = row['Pair String']
                symbol = row['Symbol']
                open_date = row['Open Date']
                for open_row in rows:
                    if (open_row['Pair String'] == pair_str and
                        open_row['Symbol'] == symbol and
                        open_row['Open Date'] == open_date and
                        open_row['Open/Close'] == 'open'):
                        price_close = float(row['Price A']) if float(row['Price A']) != 0 else float(row['Price B'])
                        qty_close = float(row['Shares A']) if float(row['Shares A']) != 0 else float(row['Shares B'])
                        price_open = float(open_row['Price A']) if float(open_row['Price A']) != 0 else float(open_row['Price B'])
                        if row['orderAction'] == 'SELL':
                            expected_gain = (price_close - price_open) * qty_close
                        elif row['orderAction'] == 'BUY_TO_COVER':
                            expected_gain = (price_open - price_close) * qty_close
                        row['Expected Gain'] = round(expected_gain, 3)
                    
        with open(file_path, mode='w', newline='') as file:
            # Determine the updated fieldnames
            fieldnames = reader.fieldnames
            if 'Executed Price' not in fieldnames:
                fieldnames.append('Executed Price')
            if 'Executed Quantity' not in fieldnames:
                fieldnames.append('Executed Quantity')
            if 'Executed Time' not in fieldnames:
                fieldnames.append('Executed Time')
            if 'Executed Symbol' not in fieldnames:
                fieldnames.append('Executed Symbol')
            if 'Price slip' not in fieldnames:
                fieldnames.append('Price slip')

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def check_short_availability(self,ticker_list):
        fail_ticker_list = []
        for ticker in ticker_list:
            order = {
                'symbol': ticker,
                'quantity': 10,
                'orderAction': 'SELL_SHORT',
                'orderType': 'MARKET',
                'priceType': 'MARKET',
                'orderTerm': 'GOOD_FOR_DAY',
                'limitPrice': '',
                'client_order_id': random.randint(1000000000, 9999999999)
            }
            error = self.order.preview_order(order)
            if error == "Hard_to_Borrow":
                print(f"{ticker} is hard to borrow")
                fail_ticker_list.append(ticker)

        print(f"Following ticker cannot be shorted: {fail_ticker_list}")
        return fail_ticker_list

    def place_order(self,
                    etrade_orders,
                    ):
        for order in etrade_orders:
            try:
                # Preview the order
                preview_response = self.order.preview_order(order)
                print(f'preview_response: {preview_response}')
                if 'PreviewIds' in preview_response and preview_response['PreviewIds']:
                    preview_id = preview_response['PreviewIds'][0]['previewId']
                    order['previewId'] = preview_id
                    
                    # Place the order
                    place_response = self.order.place_order(order)
                    print(f'place_response: {place_response}')
                    return place_response
                else:
                    raise Exception("Failed to preview order")
            except Exception as e:
                print("Error placing equity order:", e)
                return None

    def check_order_execution(self,
                            order_id,
                            account_id,
                            max_attempts = 10,
                            delay = 5):
        """
        Check if a submitted order has been executed.
        
        :param session: An authenticated E*TRADE session object
        :param order_id: The ID of the order to check
        :param account_id: The account ID associated with the order
        :param max_attempts: Maximum number of attempts to check the order status
        :param delay: Delay in seconds between each check
        :return: True if the order is executed, False otherwise
        """        
        for attempt in range(max_attempts):
            try:
                # Fetch the order details
                response = self.order_api.list_orders(account_id, ord_status="OPEN")
                
                # Check if the order is in the list of open orders
                for order in response['OrdersResponse']['Order']:
                    if order['orderId'] == order_id:
                        status = order['orderStatus']
                        
                        if status == 'EXECUTED':
                            print(f"Order {order_id} has been executed.")
                            return True
                        elif status in ['CANCELLED', 'REJECTED']:
                            print(f"Order {order_id} has been {status.lower()}.")
                            return False
                        else:
                            print(f"Order {order_id} is still {status.lower()}. Waiting...")
                            break
                else:
                    # If the order is not in the open orders list, it might be executed
                    executed_response = self.order_api.list_orders(account_id, ord_status="EXECUTED")
                    for executed_order in executed_response['OrdersResponse']['Order']:
                        if executed_order['orderId'] == order_id:
                            print(f"Order {order_id} has been executed.")
                            return True
                    
                    print(f"Order {order_id} not found in open or executed orders. It might have been cancelled or rejected.")
                    return False
                
            except Exception as e:
                print(f"Error checking order status: {e}")
            
            time.sleep(delay)
        
        print(f"Max attempts reached. Unable to confirm execution of order {order_id}.")
        return False

    
    def save_order_to_csv(self,
                        order_details):
        directory = 'live trade tracker/FrontEndAgent'
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f'FrontEndAgent_order_{self.agent_id}.csv')
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
        # Write headers if the file does not exist
        if not file_exists:
            headers = ["Variable", "Value"]
            writer.writerow(headers)
        
        # Write the object's __dict__ items
        for key, value in self.__dict__.items():
            writer.writerow([key, value])


class BackEndAgent:
    def __init__(self,
                start_date: str,
                LiveTradeAgent_id: str,
                stock_universe: str,
                sector_key: str,
                use_existing_file: bool,
                metric_selected: str,
                dynamic_trade_setting: bool,
                ) -> None:
        if LiveTradeAgent_id is None:
            self.agent_id = LiveTradeAgent.generate_agent_id()
        else:
            self.agent_id = LiveTradeAgent_id

        self.start_date = start_date
        # self.load_last_state()
        self.stock_universe = stock_universe
        self.sector_key = sector_key
        self.metric_selected = metric_selected
        self.close_prices_df = None
        
        # Create an array of traiding period "episodes" by specifying the training and validation days
        # validation_episode_days specifies the look forward days for backtest, training_episode_days specifies the look back days for backtest
        # e.g. Run validation_episode_days = 60 first and 
        training_start_date_dt = pd.to_datetime(start_date)
        validation_start_date_dt = pd.to_datetime(start_date)
        trading_days = yf.download('AAPL').index
        training_start_date_dt_index = min(range(len(trading_days)), key=lambda i: abs(trading_days[i] - training_start_date_dt))
        trading_days = trading_days[training_start_date_dt_index:]
        if not dynamic_trade_setting:
            training_episode_days = len(trading_days)-2
            validation_episode_days = 1
        else:
            training_episode_days = 252
            validation_episode_days = 20
        trade_episode_array = pair_trade_backtrade.generate_episode_period(start_date, trading_days, training_episode_days, validation_episode_days)

        # for episode in trade_episode_array:
        #     print(trade_episode_array)

        # Run backtests on all trade_param_list, save the results as training_data_set_array, which has detailed datas like daily returns
        self.back_test_agent = LookBackTest(
            start_date = self.start_date, 
            end_date = datetime.now().strftime('%Y-%m-%d'),
            today = datetime.now().strftime('%Y-%m-%d'),
            trade_param_list = generate_trade_parameter_list(self.stock_universe,self.sector_key),
            base_trade_param=generate_trade_parameter_list(self.stock_universe,self.sector_key)[0],
            stock_universe=self.stock_universe
            )

        # Load all the cerebro results from pkl or re-run the backtests
        cerebro_results = self.back_test_agent.param_batch_test(use_existing_file)
        # Divide the cerebro_results into the training episodes 
        training_data_set_array = self.back_test_agent.get_training_data_set_array(cerebro_results,trade_episode_array)

        # iterate through every single episode and find out the optimal strategy for each episode
        best_trade_parameter_array = []
        for episode in trade_episode_array:
            optimal_strategy = TrainingDataSet.optimize_strategy(training_data_set_array,self.metric_selected,episode)
            best_trade_parameter_array.append(optimal_strategy)
            # print(f'Optimal strategy chosen for')
            # print('key,value')
            # for key, value in optimal_strategy.trade_parameter.items():
            #     print(f"{key},{value}")
        breakpoint()
        # LookForwardTest can take a trade_parameter_array and update the trading strategy dynamically
        self.forward_test_agent = LookForwardTest(start_date = validation_start_date_dt.strftime('%Y-%m-%d'),
                                end_date = datetime.now().strftime('%Y-%m-%d'),
                                best_trade_param_array = best_trade_parameter_array,
                                test_type = 'validation',
                                agent_id=self.agent_id
        )

    @staticmethod
    def read_trade_param_overwrite(file_path):
        base_trade_parameter = {}
        
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                key = row['key']
                value = row['value']
                
                # Convert numerical values to appropriate types
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep value as a string if it can't be converted to int or float
                
                base_trade_parameter[key] = value
        
        return base_trade_parameter

    def load_last_state(self):
        directory = 'live trade tracker/BackEndAgent'
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f'BackEndAgent_{self.agent_id}.csv')
        
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No such file: '{filename}'")
        
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            
            if not rows or len(rows) <= 1:
                raise ValueError("The CSV file is empty or does not contain valid data.")
            
            # Read the latest entry from the CSV
            for i in range(1, len(rows), len(self.__dict__)):
                last_entry = rows[i:i+len(self.__dict__)]
            
            # Set the object's __dict__ items from the latest entry
            for key, value in last_entry:
                setattr(self, key, value)

    def save_last_state(self):

        directory = 'live trade tracker/BackEndAgent'
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f'BackEndAgent_{self.agent_id}.csv')
        file_exists = os.path.isfile(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
        # Write headers if the file does not exist
        if not file_exists:
            headers = ["Variable", "Value"]
            writer.writerow(headers)
        
        # Write the object's __dict__ items
        for key, value in self.__dict__.items():
            writer.writerow([key, value])

    def generate_trade_signal(self,
                            out_of_sample_df: pd.DataFrame(),
                            close_prices_df: pd.DataFrame(),
                            selected_pairs: list[CointData],
                            ):
        
        #load_coint_pair loads the cointegration parts at the start of each out_of_sample period
        self.selected_pairs,self.out_of_sample_df,self.in_sample_pair_buffer,self.kalman_spread_list = self.OutSamplePair.load_coint_pair(
            self.closes_prices_df_ix,
            self.date_index,
            self.all_transactions_df,
            self.holdings,
            self.count_map,
            self.holdings_l,
            self.holdings_date_l,
            self.close_prices_df,
            self.in_sample_pair_buffer,
            self.pairs_list_open)

        self.out_of_sample_index,self.end_ix,self.out_of_sample_ix,self.out_of_sample_day,self.day_transactions_l = self.OutSamplePair.out_of_sample_start(
            out_of_sample_df = self.out_of_sample_df,
            pairs_list = self.selected_pairs,
            holdings = self.holdings,
            in_sample_stat_en = self.OutSamplePair.in_sample_stat_en)

        self.holdings, day_trans_df, self.opening_pos, self.closing_pos, self.pairs_list_open = self.OutSamplePair.out_of_sample_step(
            out_of_sample_df,
            close_prices_df,
            selected_pairs,
            self.holdings,
            self.cash,
            long_pos,
            short_pos,
            self.OutSamplePair.in_sample_stat_en,
            self.out_of_sample_ix,
            self.closes_prices_df_ix,
            self.end_ix,
            self.out_of_sample_index,
            self.open_positions,
            self.day_transactions_l,
            self.under_leverage,
            self.kalman_spread_list,
            self.pairs_list_open)



# # Example usage
# stock_universes = ['sp500','gold_etf','silicon_etf','real_estate_etf','energy_etf','treasury_etf','nasdaq']
# generate_spread_dict_csv(stock_universes)


# # Example usage
# stock_universes = ['sp500','gold_etf','silicon_etf','real_estate_etf','energy_etf','treasury_etf','nasdaq']
# spread_results = analyze_spread_from_csv(stock_universes)

# for universe, spread in spread_results.items():
#     print(f"The 90th percentile spread for {universe} is {spread:.3f}%")

# breakpoint()

# # Example usage
# stock_universe = 'energy_etf'
# stock_universe_list = ['sp500','gold_etf','silicon_etf','real_estate_etf','energy_etf','treasury_etf','nasdaq']
# plot_spread_from_csv(stock_universe)
