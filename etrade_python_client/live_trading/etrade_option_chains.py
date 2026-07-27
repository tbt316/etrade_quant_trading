#!/usr/bin/env python

from cgi import test
import datetime as dt
import argparse
from json.decoder import JSONDecodeError
from signal import signal
import pyetrade
import json
import ast
import os
import sys
import webbrowser
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
from data_and_research import option_price
import time
import csv
import configparser
from live_trading.runtime_safety import configure_owner_only_logger, read_owner_only_json, write_owner_only_json

# from backtesting.test import SMA

LOOKBACK_DAY = 180

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

def find_best_strategy(test_results,condition='max profit',loss_threshold=5,sigma_threshold=1):
    if condition == 'max profit':
        best_strategy_filtered = test_results[test_results['Loss percent'] < loss_threshold ]
        best_strategy_filtered = best_strategy_filtered[(best_strategy_filtered['sigma'] < sigma_threshold) & (best_strategy_filtered['Strike'] > 0) ]
        if best_strategy_filtered.empty:
            best_strategy_index=0
            return 0
        else:
            best_strategy_index = best_strategy_filtered['Profit daily %'].idxmax()
    if condition == 'min loss percent':
        best_strategy_filtered = test_results[test_results['Loss percent'] < loss_threshold ]
        best_strategy_filtered = best_strategy_filtered[(best_strategy_filtered['sigma'] < sigma_threshold) & (best_strategy_filtered['Strike'] > 0) ]
        if best_strategy_filtered.empty:
            best_strategy_index=0
            return 0              
        else:
            best_strategy_index = best_strategy_filtered['Loss expect'].idxmin()
    if condition == 'min total loss':
        best_strategy_filtered = test_results[test_results['Loss percent'] < loss_threshold ]
        best_strategy_filtered = best_strategy_filtered[(best_strategy_filtered['sigma'] < sigma_threshold) & (best_strategy_filtered['Strike'] > 0) ]
        if best_strategy_filtered.empty:
            best_strategy_index=0
            return 0              
        else:
            best_strategy_index = best_strategy_filtered['Cumulative loss'].idxmin()
    if condition == 'min sigma':
        best_strategy_filtered = test_results[test_results['Loss percent'] < loss_threshold ]
        best_strategy_filtered = best_strategy_filtered[(best_strategy_filtered['sigma'] < sigma_threshold) & (best_strategy_filtered['Strike'] > 0) ]
        if best_strategy_filtered.empty:
            best_strategy_index=0
            return 0              
        else:
            best_strategy_index = best_strategy_filtered['sigma'].idxmin()

    return test_results.loc[best_strategy_index]
 


def option_expire_dates_from_xml(q) -> list:
    """ Take the returned array of XML objects and create a list of dt.dates.
        INPUT: q is the result of the api.get_option_expire_date() call. It is actually a dictionary
                derived from XML from the Etrade API.
        OUTPUT: a list of dt.date values representing the expiration dates
    """
    dates = [dt.date(
            int(this_date["year"]),
            int(this_date["month"]),
            int(this_date["day"]))
            for this_date in q['OptionExpireDateResponse']['ExpirationDate']
        ]
    return dates

def get_all_option_chains(api, underlying_symbol, max_expire_day=30) -> dict:
    """ Returns the all the option chains for the underlying_symbol with expiration_dates
        as the key. This requires two calls, one to get_option_expire_date, then
        to get all the expiration_dates and multiple calls to get_option_chains
        with defaults.

    """
    try:
        q = api.get_option_expire_date(underlying_symbol,resp_format='xml')
        expiration_dates = option_expire_dates_from_xml(q)
    except Exception:
        raise

    rtn = dict()
    for this_expiry_date in expiration_dates:
        time_to_expire=this_expiry_date-dt.date.today()
        day_to_expire=time_to_expire.days
        if day_to_expire > max_expire_day: 
            break
        else:
            q = api.get_option_chains(underlying_symbol, this_expiry_date)
            chains = q['OptionChainResponse']['OptionPair']
            print(".", end="", flush=True) # progress
            rtn[this_expiry_date] = [i['Put'] for i in chains] + [i['Call'] for i in chains]
    print('Length of return: ', len(rtn))

    # print(rtn[this_expiry_date][0])
    # time.sleep(3000)
    return rtn

def get_option_chains_date_strike(api, q ,date, skip_adjusted,chain_type,strike,no_of_strike) -> dict:
    """ Returns the all the option chains for the underlying_symbol with expiration_dates
        as the key. This requires two calls, one to get_option_expire_date, then
        to get all the expiration_dates and multiple calls to get_option_chains
        with defaults.

    """
    # try:
    #     q = api.get_option_expire_date(underlying_symbol,resp_format='xml')
    #     expiration_dates = option_expire_dates_from_xml(q)
    #     # for this_expiry_date in expiration_dates:
    #     #     print(this_expiry_date)
    # except Exception:
    #     raise
    rtn = dict()
    for i in range(len(q[date])):
        if float(q[date][i]['strikePrice']) > strike and q[date][i]['optionType'] == chain_type and chain_type == 'CALL': 
                print("match quote: ", q[date][i])
                break
        if float(q[date][i]['strikePrice']) < strike and q[date][i]['optionType'] == chain_type and chain_type == 'PUT': 
                print("match quote: ", q[date][i])
                break

    strikePrice=[]
    ask=[]
    bid=[]
    volume=[]
    lastPrice=[]
    displaySymbol=[]
    optionType=[]

    strikePrice.append(float(q[date][i]['strikePrice']))
    ask.append(float(q[date][i]['ask']))
    bid.append(float(q[date][i]['bid']))
    volume.append(int(q[date][i]['volume']))
    lastPrice.append(float(q[date][i]['lastPrice']))
    displaySymbol.append(str(q[date][i]['displaySymbol']))
    optionType.append(str(q[date][i]['optionType']))

    option_quote_list=pd.DataFrame({
        'strikePrice':strikePrice,
        'ask':ask,
        'bid':bid,
        'volume':volume,
        'lastPrice':lastPrice,
        'displaySymbol':displaySymbol,
        'optionType':optionType
    })

    # closest_value = None
    # min_difference = float('inf')

    # for this_strike in option_quote_list['strikePrice']:
    #     if this_strike > strike:
    #         option_target_list=option_quote_list[option_quote_list['strikePrice'] == this_strike]
    #         break
    print('Option_quote_list: ', option_quote_list)
    return option_quote_list


def strvals_to_real(q) -> dict:
    ''' given the input dictionary, produce an equivalent with all real numbers converted from
        strings. This should be iterative.
    '''
    rtn = dict()
    for k,v in q.items():
        if isinstance(v,str):
            try:
                rtn[k] = ast.literal_eval(v)
            except:
                rtn[k] = v
        elif isinstance(v,dict):
            rtn[k] = strvals_to_real(v)
        else:
            rtn[k] = v
    return rtn

def alter_quote_dict(quote) -> dict:
    ''' Put the etrade returned quote dict into the form that I'm used to seeing.
        Input form: 'Product' keys
                    'All' keys
    '''
    rtn = strvals_to_real(quote['All'])
    rtn['dateTimeUTC'] = int(quote['dateTimeUTC'])
    for k in ('dateTime','quoteStatus','ahFlag','hasMiniOptions'):
        try:
            rtn[k] = quote[k]
        except KeyError: # hasMiniOptions missing in sandbox
            continue
    rtn['securityType'] = quote['Product']['securityType']
    rtn['symbol'] = quote['Product']['symbol']
    return rtn

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grab all the option chains for the specified symbol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', help='use sandbox?', action=argparse.BooleanOptionalAction)
    # parser.add_argument('symbol', help='symbol name', type=str)
    # parser.add_argument('STD_COUNT', help='number of standard deviation', type=int)
    # parser.add_argument('MOVE_AVG_DAY', help='number of days look back for average', type=int)
    # parser.add_argument('OPTION_TARGET_DAY', help='Option quote date', type=int)
    args = parser.parse_args()
    use_sandbox = args.sandbox

    keys = configured_oauth_keys(use_sandbox)
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]

    try:
        token = get_etrade_oauth(use_sandbox)
        # print("Got cached token: {}".format(token))
        manager = pyetrade.authorization.ETradeAccessManager(
            consumer_key,
            consumer_secret,
            token['oauth_token'],
            token['oauth_token_secret']
        )
        manager.renew_access_token()
    except Exception as err:
        # print("Got {} when trying to get & renew cached OAuth tokens; getting new ones".format(err))
        oauth = pyetrade.ETradeOAuth(consumer_key, consumer_secret)
        print("Opening E*TRADE authorization; copy the five-character verifier.")
        webbrowser.open(oauth.get_request_token())
        API_token = input('E*TRADE token: ')
        oauth.get_access_token(API_token)
        token = oauth.access_token
        save_etrade_oauth(token, use_sandbox)

    window_size_ratio=6
    sigma=1.5
    strike_base='CLOSE_PRICE'
    option_strategy='CALL_ONLY'
    expire_period=10

    ticker='ENPH'

    api = pyetrade.market.ETradeMarket(consumer_key, consumer_secret,
                                       token['oauth_token'],
                                       token['oauth_token_secret'],
                                       dev=use_sandbox)

    option_strategy_test=['CALL_ONLY','PUT_ONLY','CALL_OR_PUT','CALL_AND_PUT']
    sigma_test=np.arange(0.5,1.5,0.1)
    strike_base_test=['CLOSE_PRICE','MOV_AVG']
    window_size_ratio_test=np.arange(2,10,2)
    expire_period_test=[5,10,20]

    # Compute Cartesian product of arrays
    test_results = list(product(option_strategy_test, sigma_test, strike_base_test, window_size_ratio_test, expire_period_test))

    # Convert the result into a DataFrame
    test_results = pd.DataFrame(test_results, columns=['option_strategy', 'sigma', 'strike_base', 'window_size_ratio', 'expire_period'])

    test_results['Close price']=0
    test_results['Strike']=0
    test_results['Option gain']=0
    test_results['Option gain est']=0
    test_results['Loss percent']=0
    test_results['Loss std']=0
    test_results['Average loss']=0
    test_results['Loss expect']=0
    test_results['Cumulative loss']=0
    test_results['Profit daily %']=0

    print(test_results)

    test_combination=len(test_results)
    # test_combination=2

    signals = np.empty(test_combination, dtype=object)
    option_gain = np.empty(test_combination, dtype=float)
    option_gain_est = np.empty(test_combination, dtype=float)

    option_quote='true'

    end_date=dt.date.today()
    offset = timedelta(days=365*4)
    start_date=end_date-offset
    stock_data = backtest_bo.fetch_stock_data(ticker, start_date, end_date)

    if option_quote == 'true': 
        q_expire_date = api.get_option_expire_date(ticker,resp_format='xml')
        expiration_dates = option_expire_dates_from_xml(q_expire_date)
        q_quote_all = get_all_option_chains(api,ticker,max(expire_period_test))

    # print('Quote all: ',q_quote_all)
    # q = api.get_option_chains(underlying_symbol,date,skip_adjusted,chain_type,strike,no_of_strike)

    for i in range(test_combination):
        
        print(".", end="", flush=True) # progress
        # print('Quote expire date: %s, %s' % (expire_period, quote_expire_date))

        # def backtest_strategy(ticker, window_size_ratio=10, sigma=3, baseline='MOV_AVG', option_strategy='CALL_ONLY', expire_period=30, metric_eval_period=180):
        signals[i]=backtest_bo.backtest_strategy(stock_data,test_results.loc[i,'window_size_ratio'],test_results.loc[i,'sigma'],test_results.loc[i,'strike_base'],test_results.loc[i,'option_strategy'],test_results.loc[i,'expire_period'])

        start_time = time.time()

        if option_quote == 'true':

            quote_expire_date=dt.date.today()
            for this_expiry_date in expiration_dates:
                time_to_expire=this_expiry_date-dt.date.today()
                day_to_expire=time_to_expire.days
                if day_to_expire > test_results.loc[i,'expire_period']:
                    quote_expire_date = this_expiry_date
                    break

            if signals[i]['Option type'].iloc[-1] == 'call_and_put': 
                try:
                    chains = get_option_chains_date_strike(api, q_quote_all ,quote_expire_date,True,'CALL',signals[i]['Option strike'].iloc[-1],10)
                except Exception as err:
                    print('%s failed get_all_option_chains %s' % (ticker,signals[i]['Option type'].iloc[-1]))
                    print(err)
                    sys.exit(1)
                try:
                    chains_2 = get_option_chains_date_strike(api, q_quote_all,quote_expire_date,True,'PUT',signals[i]['Option strike alt'].iloc[-1],10)
                except Exception as err:
                    print('%s failed get_all_option_chains %s' % (ticker,signals[i]['Option type'].iloc[-1]))
                    print(err)
                    sys.exit(1)       
                option_gain[i] = ((chains['ask'].iloc[-1]*50 + chains_2['ask'].iloc[-1]*50) + (chains['bid'].iloc[-1]*50 + chains_2['bid'].iloc[-1]*50))

                option_price_estimate = call_price_with_early_exercise = option_price.binomial_option_price_with_early_exercise(stock_data, signals[i]['Option strike'].iloc[-1], day_to_expire, 0.05 , 100, 'call')
                option_price_estimate_2 = call_price_with_early_exercise = option_price.binomial_option_price_with_early_exercise(stock_data,signals[i]['Option strike alt'].iloc[-1], day_to_expire, 0.05 , 100, 'put')
                option_gain_est[i] = ((option_price_estimate*100 + option_price_estimate_2*100) - option_gain[i])/option_gain[i]*100
            else:
                try:
                    # print('option type: %s, option strike: %s\n' % (signals[i]['Option type'].iloc[-1],signals[i]['Option strike'].iloc[-1]))
                    chains = get_option_chains_date_strike(api, q_quote_all ,quote_expire_date,True,signals[i]['Option type'].iloc[-1],signals[i]['Option strike'].iloc[-1],10)
                except Exception as err:
                    print('%s failed get_all_option_chains %s' % (ticker,signals[i]['Option type'].iloc[-1]))
                    print(err)
                    sys.exit(1)
                option_gain[i] = chains['ask'].iloc[-1]*50 + chains['bid'].iloc[-1]*50
                option_price_estimate = call_price_with_early_exercise = option_price.binomial_option_price_with_early_exercise(stock_data, signals[i]['Option strike'].iloc[-1], day_to_expire, 0.05 , 100, signals[i]['Option type'].iloc[-1])
                option_gain_est[i] = (option_price_estimate*100 - option_gain[i])/option_gain[i]*100
        

        end_time = time.time()
        # running_time_ms = (end_time - start_time) * 1000
        # print("Running time:", running_time_ms, "milliseconds")

        test_results.loc[i,'Close price']=signals[i]['Close price'].iloc[-1]
        test_results.loc[i,'Strike']=signals[i]['Option strike'].iloc[-1]
        test_results.loc[i,'Option gain']=option_gain[i]
        test_results.loc[i,'Option gain est']=option_gain_est[i]
        test_results.loc[i,'Loss percent']=signals[i]['Loss probability'].iloc[-1]
        test_results.loc[i,'Loss std']=signals[i]['Loss std'].iloc[-1]
        test_results.loc[i,'Average loss']=signals[i]['average loss'].iloc[-1]
        test_results.loc[i,'Loss expect']=signals[i]['average loss'].iloc[-1]*signals[i]['Loss probability'].iloc[-1]/100
        test_results.loc[i,'Cumulative loss']=signals[i]['Cumulative loss'].iloc[-1]
        test_results.loc[i,'Profit daily %']=(test_results.loc[i,'Option gain']-test_results.loc[i,'Loss expect'])/signals[i]['Close price'].iloc[-1]/test_results.loc[i,'expire_period']

    plt.close('all')
    
    print('***************Backtest restulss*****************\n')
    print(test_results)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"backtest_result_{ticker}_{timestamp}.csv"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pathname = os.path.join(script_dir, filename)
    test_results.to_csv(pathname, index=False,float_format='%.3f')
    print(f"Results have been written to {pathname}")

    best_strategy_condition=['max profit','min loss percent','min total loss','min sigma']

    df = pd.read_csv(pathname)        

    signals_plot = np.empty(len(best_strategy_condition), dtype=object)

    fig, axes = plt.subplots(nrows=2,ncols=len(best_strategy_condition),figsize=(30, 10))

    for i in range(len(best_strategy_condition)):

        best_strategy=find_best_strategy(df,best_strategy_condition[i],10,2)
        signals_plot[i]=backtest_bo.backtest_strategy(stock_data,best_strategy.loc['window_size_ratio'],best_strategy.loc['sigma'],best_strategy.loc['strike_base'],best_strategy.loc['option_strategy'],best_strategy.loc['expire_period'])
        print(best_strategy)
        axes[0][i].plot(signals_plot[i]['Loss probability'], label='Loss probability', color='blue')
        axes[0][i].set_ylabel('Loss probability', color='blue')
        axes[0][i].tick_params('y', colors='blue')

        ax2=axes[0][i].twinx()
        ax2.plot(signals_plot[i]['Cumulative loss'],label='Cumu loss', color='red')
        # ax2.plot(signals_plot[i]['Cumulative loss'], label='Cumulative loss')
        ax2.plot(signals_plot[i]['Loss std'], label='Loss std')
        ax2.plot(signals_plot[i]['average loss'], label='average loss')
        ax2.set_ylabel('Expected Loss', color='red')

        # Plot the other series on the secondary y-axis
        if best_strategy.loc['strike_base'] == 'MOV_AVG':
            axes[1][i].plot(signals_plot[i]['MA'], label='MA', color='green')
        axes[1][i].plot(signals_plot[i]['Close price'], label='Close price', color='red')
        axes[1][i].scatter(signals_plot[i].index,signals_plot[i]['Option strike'], s=1,label='Option strike', color='orange')
        axes[1][i].set_ylabel('MA', color='green')
        axes[1][i].tick_params('y', colors='green')

    # Add title and legend
    plt.title('Multiple Series with Different Y Scales')
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.show()
