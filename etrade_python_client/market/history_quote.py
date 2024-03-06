
# Define the ticker list
import pandas as pd
from datetime import date
import yfinance as yf
from datetime import timedelta
import datetime as dt

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


tickers_list = ['SQQQ', 'WMT']

# Fetch the data

def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    mean, std = data.rolling(n_lookback).mean(), data.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower

#data = yf.download(tickers_list,'2024-1-1','2024-3-1')['Adj Close']
# data = yf.download(tickers_list,'2023-01-01','2024-03-01')['Adj Close']
data= yf.download(tickers_list,(dt.datetime.now()+timedelta(days=-90)).strftime("%Y-%m-%d"),dt.datetime.now().strftime("%Y-%m-%d"))['Adj Close']

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + dt.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - dt.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

# Print first 5 rows of the data
print(data.head()) 

sma10=SMA(data.iloc[:, 0],10)
sma10.name="SMA10"

import matplotlib.pyplot as plt
# Plot the close price of the AAPL
sma10.plot()
data.iloc[:,0].plot()

upper, lower = BBANDS(data.iloc[:,0], 10, 3)
upper.name="BBand Up"
lower.name="BBand Down"

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


aapl = yf.Ticker("AAPL")
print(aapl.options)
print(aapl.option_chain(aapl.options[3]))


plt.show()
