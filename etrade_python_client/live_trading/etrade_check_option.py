#!/usr/bin/env python
from tabnanny import check
import time as t
from cgi import test
import argparse
from json.decoder import JSONDecodeError
from signal import signal
from tracemalloc import start
import pyetrade
import json
import ast
import os
import sys
from datetime import timedelta
from datetime import datetime
from logging.handlers import RotatingFileHandler
import logging
import pandas as pd
import yfinance as yf
from backtesting import backtest_bo
import matplotlib.pyplot as plt
import numpy as np
from itertools import product                                                               
# import option_price
import csv
from accounts.accounts_bo import StockPosition
from core_api.stock_trade_class import *
import webbrowser
from rauth import OAuth1Service
from logging.handlers import RotatingFileHandler
from accounts.accounts_bo import Accounts, calculate_std_dev
from market.market_bo import Market
from live_trading.runtime_safety import configure_owner_only_logger, read_owner_only_json, write_owner_only_json
import configparser
import multiprocessing
from typing import List
from backtesting import option_limit_backtest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any
# import backtrader as bt

# from backtesting.test import SMA
# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# logger settings
logger = configure_owner_only_logger('my_logger')
'''
    Grab the option expire dates and option chains for the specified symbol.
    Save as a JSON file

'''

# OAuth credentials are supplied by local config.ini or environment variables.
OAUTH_KEYS = {
    "sandbox": {
        "consumer_key": os.getenv("ETRADE_SANDBOX_CONSUMER_KEY") or config['DEFAULT'].get('SANDBOX_CONSUMER_KEY'),
        "consumer_secret": os.getenv("ETRADE_SANDBOX_CONSUMER_SECRET") or config['DEFAULT'].get('SANDBOX_CONSUMER_SECRET'),
    },
    "live": {
        "consumer_key": os.getenv("ETRADE_LIVE_CONSUMER_KEY") or config['DEFAULT'].get('PROD_CONSUMER_KEY'),
        "consumer_secret": os.getenv("ETRADE_LIVE_CONSUMER_SECRET") or config['DEFAULT'].get('PROD_CONSUMER_SECRET'),
    }
}


def configured_oauth_keys(use_sandbox):
    keys = OAUTH_KEYS[environment_key(use_sandbox)]
    if not all(isinstance(keys.get(name), str) and keys[name].strip() for name in ("consumer_key", "consumer_secret")):
        raise RuntimeError("E*TRADE client credentials are not configured")
    return {name: keys[name].strip() for name in ("consumer_key", "consumer_secret")}

# File to cache OAuth tokens so you don't have to re-authenticate each time
ETRADE_OAUTH_FILE = ".etrade_oauth"

def extract_ticker_ask_bid(data):
    result = []
    try:
        quotes = data['QuoteResponse']['QuoteData']
        for quote in quotes:
            symbol = quote['Product']['symbol']
            ask = float(quote['All']['ask'])
            bid = float(quote['All']['bid'])
            result.append({'symbol': symbol, 'ask': ask, 'bid': bid})
    except KeyError as e:
        print(f"Key error: {e}")
    except (ValueError, TypeError) as e:
        print(f"Value or Type error: {e}")
    return result

def environment_key(use_sandbox) -> str:
    return "sandbox" if use_sandbox else "live"

def get_etrade_oauth(use_sandbox) -> dict:
    try:
        return read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache")[environment_key(use_sandbox)]
    except (KeyError, TypeError, RuntimeError):
        print("Couldn't load cached OAuth safely.")
        return None

# Save the token, merging in with existing tokens
def save_etrade_oauth(token, use_sandbox) -> bool:
    try:
        tokens = read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache") if os.path.lexists(ETRADE_OAUTH_FILE) else {}
        tokens[environment_key(use_sandbox)] = token
        write_owner_only_json(ETRADE_OAUTH_FILE, tokens)
        return True
    except (KeyError, TypeError, RuntimeError):
        print("Couldn't save cached OAuth safely.")
        return False

def oauth(use_sandbox, auto_login = True):
    """Allows user authorization for the sample application with OAuth 1"""
    keys = configured_oauth_keys(use_sandbox)
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]
    if use_sandbox:
        base_url = "https://apisb.etrade.com"
    else:
        base_url = "https://api.etrade.com"
    
    etrade = OAuth1Service(
        name="etrade",
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        request_token_url="https://api.etrade.com/oauth/request_token",
        access_token_url="https://api.etrade.com/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url=base_url
    )

    tokens = get_etrade_oauth(use_sandbox)
    if isinstance(tokens, dict) and {"access_token", "access_token_secret"} <= set(tokens):
        session = etrade.get_session((tokens['access_token'], tokens['access_token_secret']))
        
        renew_url = f"{base_url}/oauth/renew_access_token"
        response = session.get(renew_url)
    
        if response.status_code == 200:
            if b'Access Token has been renewed' in response.content:
                print("Session renewed successfully. Continuing with existing tokens.")
                return session, base_url
            else:
                print("Unexpected response content. Starting new OAuth flow.")
        else:
            print(f"Session renewal failed. Status code: {response.status_code}")
    else:
        print("No existing tokens found. Starting new OAuth flow.")

    # If we get here, either there were no existing tokens or renewal failed
    # Start a new OAuth flow
    request_token, request_token_secret = etrade.get_request_token(
        params={"oauth_callback": "oob", "format": "json"})

    authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)

    if auto_login:
        text_code = get_token_via_safari(authorize_url,username=username,password=password)
    else:
        webbrowser.open(authorize_url)
        text_code = input("Please accept agreement and enter verification code from browser: ")

    session = etrade.get_auth_session(
        request_token,
        request_token_secret,
        params={"oauth_verifier": text_code}
    )

    # Save the new tokens
    tokens = {
        'access_token': session.access_token,
        'access_token_secret': session.access_token_secret
    }
    if not save_etrade_oauth(tokens, use_sandbox):
        raise RuntimeError("could not persist OAuth cache safely")

    print("New session created and tokens saved.")
    return session, base_url

def plot_option_data(ticker, date, strike_price, expiration_date, option_type):
    # Load the pickle file for the given ticker and date
    filename = f"history_option/{ticker}_{date}.pkl"
    df = pd.read_pickle(filename)
    print(f"Loaded data from {filename},df: {df[:100]}")
    expiration_date_obj = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    filtered_df = df[(df["Strike_Price"].astype(float) == float(strike_price)) & (df["Expiry_Date"] == expiration_date_obj) & (df["Option_Type"] == option_type)]
    print(f"Filtered data, filtered_df: {filtered_df}")
    # Check if any data was found
    if filtered_df.empty:
        print(f"No data found for the given ticker,{ticker} date {date}, strike price {strike_price}, and expiration date {expiration_date}.")
        return

    # Convert timestamp to datetime and set it as index
    filtered_df["Timestamp"] = pd.to_datetime(filtered_df["Timestamp"])
    filtered_df.set_index("Timestamp", inplace=True)

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # Plot the ask and bid prices against time
    axs[0].plot(filtered_df.index, filtered_df["Ask_Price"], label="Ask Price")
    axs[0].plot(filtered_df.index, filtered_df["Bid_Price"], label="Bid Price")
    axs[0].set_ylabel("Price")
    axs[0].legend()

    # Plot the ask-bid spread on the secondary y-axis
    ax2 = axs[0].twinx()
    ax2.plot(filtered_df.index, filtered_df["Ask_Price"] - filtered_df["Bid_Price"], color="green", label="Ask-Bid Spread")
    ax2.set_ylabel("Spread")
    ax2.legend()

    # Plot the ask and bid sizes against time
    axs[1].plot(filtered_df.index, filtered_df["Ask_Size"], label="Ask Size")
    axs[1].plot(filtered_df.index, filtered_df["Bid_Size"], label="Bid Size")
    axs[1].set_ylabel("Size")
    axs[1].legend()

    # Plot the open interest against time
    axs[2].plot(filtered_df.index, filtered_df["Open_Interest"])
    axs[2].set_ylabel("Open Interest")

    # Plot the volume against time
    axs[3].plot(filtered_df.index, filtered_df["Volume"])
    axs[3].set_ylabel("Volume")

    # Format x-axis to display time in HH:MM:SS format
    date_formatter = mdates.DateFormatter("%H:%M:%S")
    for ax in axs:
        ax.xaxis.set_major_formatter(date_formatter)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45], interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grab all the option chains for the specified symbol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', help='use sandbox?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--trade', help='start live trade?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_existing_file', help='re-use the backtest results?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--username', help='username for login', type=str, required=True)
    parser.add_argument('--password', help='password for login', type=str, required=True)
    args = parser.parse_args()
    use_sandbox = args.sandbox
    start_trade = args.trade
    use_existing_file = args.use_existing_file
    # use_sandbox = True
    username = args.username
    password = args.password
    bypass_etrade = False
    start_log = False
    end_log = False

    keys = OAUTH_KEYS[environment_key(use_sandbox)]
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]

    trade_executed_flag = False # flag to be reset every ticks, indicating the trade strategy had been exeucted for the current tick

    if not bypass_etrade: 
        session, base_url = oauth(use_sandbox, auto_login=True)
    else:
        session = base_url = None

    authenticated = 0
    # Define the start and end times
    # trade_start_time = datetime.strptime('12:45:00', '%H:%M:%S').time()
    # trade_end_time = datetime.strptime('12:50:00', '%H:%M:%S').time()



    # plot_option_data('AAPL','2025-01-30','280','2025-02-07','Call')


    open_trade_start_time = datetime.strptime('6:30:00', '%H:%M:%S').time()
    open_trade_end_time = datetime.strptime('13:00:00', '%H:%M:%S').time()
    close_trade_start_time = datetime.strptime('12:00:00', '%H:%M:%S').time()
    close_trade_end_time = datetime.strptime('13:00:00', '%H:%M:%S').time()

    datalog_start_time = datetime.strptime('06:29:00', '%H:%M:%S').time()
    datalog_end_time = datetime.strptime('13:15:00', '%H:%M:%S').time()

    last_renewal_time = datetime.now()

    stock_positions: List[StockPosition] = []
    
    if start_trade:
        if not bypass_etrade: 
            raise RuntimeError("Legacy live execution is disabled; use etrade_cover_call_new with a RuntimeSafetyBoundary")
        else:
            etrade_instance = LiveTradeAgent()
        print('Live trade agent id: ', etrade_instance.agent_id)

        accounts = Accounts(session, base_url)
        market = Market(session, base_url)

        accounts.account_list()

        tickers = ['GOOGL','PLTR','MRNA','T','AMZN','AAPL','INTC','WMT','BAC','AMD','AVGO','DIS','PFE','UNH','GOOG','NKE','CSCO','FFIV','OMC','COR','ORCL','LVS','APH','SBUX','MSFT','TSLA','KO','MU','DAL','UBER','NFLX','GM','BA','VZ','C','SMCI','F','META','NEE','MDT','PEP','JPM','NWSA','PANW','CPRT','BMY','ADM','FCX']
        extra_ticker = ['QQQ','SPY','ARKK','TQQQ','SQQQ','PYPL','QCOM','OXY']
        tickers = tickers + extra_ticker

        while True:
            # Get the current time
            now = datetime.now()
            current_time = now.time()
            if (now - last_renewal_time) >= timedelta(minutes=1):
                print("Renewing session...", now)
                session, base_url = oauth(use_sandbox)
                last_renewal_time = datetime.now()
                accounts = Accounts(session, base_url)
                accounts.account_list()

            if ( now.time() > close_trade_start_time and now.time() < close_trade_end_time ) or ( now.time() > open_trade_start_time and now.time() < open_trade_end_time ):
                for ticker in tickers:
                    accounts.get_option_chain_simple(ticker)
                    print(f"Option chain for {ticker} saved.")
                    df = pd.read_pickle(f"history_option/{ticker}_{datetime.now().strftime('%Y-%m-%d')}.pkl")
                    print(df)
                time_check = datetime.now()
                while time_check - now < timedelta(seconds=60):
                    t.sleep(1)
                    time_check = datetime.now()
            else:
                print("Trading hours are over.")
                t.sleep(60)
