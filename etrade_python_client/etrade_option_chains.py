#!/usr/bin/env python

import datetime as dt
import argparse
from json.decoder import JSONDecodeError
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
# from backtesting.test import SMA

LOOKBACK_DAY = 180
MOVE_AVG_DAY = 10
STD_COUNT =6
OPTION_TARGET_DAY=30

# logger settings
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("python_client.log", maxBytes=5*1024*1024, backupCount=3)
FORMAT = "%(asctime)-15s %(message)s"
fmt = logging.Formatter(FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)
'''
    Grab the option expire dates and option chains for the specified symbol.
    Save as a JSON file

'''

# FILL THESE IN WITH YOUR OAUTH KEYS AND SECRETS
# See https://developer.etrade.com/getting-started
OAUTH_KEYS = {
    "sandbox": {
        "consumer_key": "79a22325849d55ab995c67d9b5df9684",
        "consumer_secret": "b9711c32ff7e1ec28a609317e0bdcf149fa6f944e59f3c89348e5aa85fd1b32f",
    },
    "live": {
        "consumer_key": "73ae73ac0315a6520f31b9d081d7849a",
        "consumer_secret": "3058031dfb6e2a44b5d0ef0055ed46c74f333c08fafdfb6ead39d6637249b34f",
    }
}

# File to cache OAuth tokens so you don't have to re-authenticate each time
ETRADE_OAUTH_FILE = ".etrade_oauth"

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

def get_all_option_chains(api, underlying_symbol) -> dict:
    """ Returns the all the option chains for the underlying_symbol with expiration_dates
        as the key. This requires two calls, one to get_option_expire_date, then
        to get all the expiration_dates and multiple calls to get_option_chains
        with defaults.

    """
    try:
        q = api.get_option_expire_date(underlying_symbol,resp_format='xml')
        expiration_dates = option_expire_dates_from_xml(q)
        print(expiration_dates)
    except Exception:
        raise

    rtn = dict()
    for this_expiry_date in expiration_dates:
        q = api.get_option_chains(underlying_symbol, this_expiry_date)
        print(this_expiry_date)
        chains = q['OptionChainResponse']['OptionPair']
        print(".", end="", flush=True) # progress
        # print(q) # progress
        rtn[this_expiry_date] = [i['Put'] for i in chains] + [i['Call'] for i in chains]
    print()
    return rtn

def get_option_chains_date_strike(api, underlying_symbol,date,skip_adjusted,chain_type,strike,no_of_strike) -> dict:
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

    # q = api.get_option_chains(underlying_symbol,date,skip_adjusted,chain_type,strike)
    q = api.get_option_chains(underlying_symbol,date,skip_adjusted,chain_type,strike,no_of_strike)
    chains = q['OptionChainResponse']['OptionPair']
    if chain_type == "call":
        rtn[date] = [i['Call'] for i in chains]
    if chain_type == "put":
        rtn[date] = [i['Put'] for i in chains]
    if chain_type == "callput":
        rtn[date] = [i['Put'] for i in chains] + [i['Call'] for i in chains]

    strikePrice=[]
    ask=[]
    bid=[]
    volume=[]

    option_target_list=pd.DataFrame({
        'strikePrice':strikePrice,
        'ask':ask,
        'bid':bid,
        'volume':volume
    })

    for i in range(len(rtn[date])):
        strikePrice.append(float(rtn[date][i]['strikePrice']))
        ask.append(float(rtn[date][i]['ask']))
        bid.append(float(rtn[date][i]['bid']))
        volume.append(float(rtn[date][i]['volume']))

    option_quote_list=pd.DataFrame({
        'strikePrice':strikePrice,
        'ask':ask,
        'bid':bid,
        'volume':volume
    })


    closest_value = None
    min_difference = float('inf')

    for this_strike in option_quote_list['strikePrice']:
        if this_strike > strike:
            option_target_list=option_quote_list[option_quote_list['strikePrice'] == this_strike]
            break

    # print(option_quote_list)
    print(option_target_list)
    return rtn


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
        with open(ETRADE_OAUTH_FILE) as f:
            tokens = json.load(f)
            return tokens[environment_key(use_sandbox)]
    except (KeyError, TypeError, FileNotFoundError, JSONDecodeError) as err:
        print("Couldn't find/parse cached OAuth in {} ({}: {})".format(ETRADE_OAUTH_FILE, err))
        return None

# Save the token, merging in with existing tokens
def save_etrade_oauth(token, use_sandbox) -> bool:
    try:
        try:
            with open(ETRADE_OAUTH_FILE) as f:
                tokens = json.load(f)
        except FileNotFoundError:
            tokens = {}
        tokens[environment_key(use_sandbox)] = token
        with open(os.open(ETRADE_OAUTH_FILE, os.O_CREAT | os.O_WRONLY, 0o600), "w") as f:
            f.write(json.dumps(tokens))
    except (KeyError, JSONDecodeError) as err:
        print("Couldn't write cached OAuth in {} ({})".format(ETRADE_OAUTH_FILE, err))
        sys.exit(1)

def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    mean, std = data.rolling(n_lookback).mean(), data.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grab all the option chains for the specified symbol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', help='use sandbox?', action=argparse.BooleanOptionalAction)
    parser.add_argument('symbol', help='symbol name', type=str)
    args = parser.parse_args()
    use_sandbox = args.sandbox

    keys = OAUTH_KEYS[environment_key(use_sandbox)]
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
        print("Visit this URL and copy the five character token")
        print(oauth.get_request_token())
        API_token = input('E*TRADE token: ')
        oauth.get_access_token(API_token)
        token = oauth.access_token
        save_etrade_oauth(token, use_sandbox)

    history_data = yf.download(args.symbol,(dt.datetime.now()+timedelta(days=-90)).strftime("%Y-%m-%d"),dt.datetime.now().strftime("%Y-%m-%d"))['Adj Close']
    print("DataFrame dimensions:", history_data.shape)

    # print(history_data)

    sma=history_data.rolling(MOVE_AVG_DAY).mean()
    upper, lower = BBANDS(history_data, MOVE_AVG_DAY, STD_COUNT)
    upper.name="BBand Up"
    lower.name="BBand Down"

    sma.name="SMA"

    import matplotlib.pyplot as plt
    # Plot the close price of the AAPL
    sma.plot()
    history_data.plot()
    upper.plot(linestyle='dashed')
    lower.plot(linestyle='dashed')

    # Show the legend
    plt.legend()

    # Define the label for the title of the figure
    plt.title("Returns", fontsize=16)

    # Define the labels for x-axis and y-axis
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)

    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

    print("Moving avg: %s\n Bollinger UP: %s\n,Bollinger DOWN: %s\n" % (sma[-1],upper[-1],lower[-1]))

    api = pyetrade.market.ETradeMarket(consumer_key, consumer_secret,
                                       token['oauth_token'],
                                       token['oauth_token_secret'],
                                       dev=use_sandbox)

    # q = api.get_quote([args.symbol],require_earnings_date=True,resp_format='xml')     # a dict response
    # quote = alter_quote_dict(q['QuoteResponse']['QuoteData'])

    q = api.get_option_expire_date(args.symbol,resp_format='xml')
    expiration_dates = option_expire_dates_from_xml(q)
    quote_expire_date=dt.date.today()
    for this_expiry_date in expiration_dates:
        time_to_expire=this_expiry_date-dt.date.today()
        day_to_expire=time_to_expire.days
        if day_to_expire > OPTION_TARGET_DAY:
            quote_expire_date = this_expiry_date
            break
    print('Quote expire date: %s' % quote_expire_date)

    try:
        chains = get_option_chains_date_strike(api, args.symbol,quote_expire_date,True,"call",upper[-1],10)
        # chains = get_all_option_chains(api, quote['symbol'])#,[2024,03,01],True,"call",900,10)
    except Exception as err:
        print('%s failed get_all_option_chains' % (args.symbol))
        print(err)
        sys.exit(1)

    try:
        chains = get_option_chains_date_strike(api, args.symbol,quote_expire_date,True,"put",lower[-1],10)
    except Exception as err:
        print('%s failed get_all_option_chains' % (args.symbol))
        print(err)
        sys.exit(1)

    # plt.show()
# chains has key that is dt.date; convert key to str (JSON needs a string as the key, not a dt.date)
# also convert all values in each ordered dict to floats/ints
    # converted_chains = {'quote': quote}     # start of dictionary
    # for (expiry_date,chain_list) in chains.items():
    #     converted_chain_list = [ strvals_to_real(v) for v in chain_list]
    #     converted_chains[str(expiry_date)] = converted_chain_list               # use str(expiry_date) because eventually this will be saved as JSON

    # json_filename = '{}_chains_{}.json'.format(quote['symbol'], dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # with open(json_filename, 'wt') as f:
    #     json.dump(converted_chains,f)
    # print("Wrote results to '{}' (size {})".format(json_filename, os.stat(json_filename).st_size))