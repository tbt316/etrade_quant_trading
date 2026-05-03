from ast import Break
from email.mime import base
from traceback import print_exception
from tracemalloc import start
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
import numpy as np

def get_option_data(stock_symbol, expiration_date, option_type, strike):
    stock = yf.Ticker(stock_symbol)
    option_chain = stock.option_chain(expiration_date)
    options = getattr(option_chain, "calls" if option_type.startswith("call") else "puts")
    option_data = options[options["strike"] == strike]
    return option_data

def get_option_history_data(contract_symbol, days_before_expiration=30):
    option = yf.Ticker(contract_symbol)
    option_info = option.info
    option_expiration_date = dt.datetime.fromtimestamp(option_info["expireDate"])
    start_date = option_expiration_date - dt.timedelta(days=days_before_expiration)
    option_history = option.history(start=start_date)
    return option_history


def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Input variables
# stock_data: dataframe of the stock data from yfinance
# window_size_ratio: the look-back days for calculating moving average and standard deviation
# sigma: number of sigma for setting the strike print_exception
# baseline: use MOV_AVG or CLOSE_PRICE for strike price baseline
# option strategy: PUT_ONLY; CALL_ONLY; PUT_AND_CALL; PUT_OR_CALL
# expire_period: expire date
# skip 
# Test variables
# metric_eval_period: number of days look back to evaluate the breach probability and loss average
def backtest_strategy(stock_data_input, window_size_ratio=10, sigma=3, baseline='MOV_AVG', option_strategy='CALL_ONLY', expire_period=30, metric_eval_period=180):

    window_size=expire_period*window_size_ratio
    # end_date=dt.date.today()
    # offset = timedelta(days=metric_eval_period*10)
    # start_date=end_date-offset
    # # Define the DataFrame for stock price, moving average, standard deviation, maximum breach price
    # #==========================================================================================================
    # stock_data = fetch_stock_data(ticker, start_date, end_date)

    if metric_eval_period*10 < len(stock_data_input):
        stock_data=stock_data_input[-metric_eval_period*10:].copy()
    else:
        stock_data=stock_data_input.copy()
    signals = pd.DataFrame(index=stock_data.index)
    signals['Signal'] = 0  # Initialize the signal column
    signals['Loss probability'] = 0
    signals['Loss']=0
    signals['Close price']=0
    signals['Option strike']=0
    signals['Option strike alt']=0
    signals['Cumulative loss']=0
    signals['Loss std']=0
    signals['average loss']=0
    signals['Option type']=0

    # Calculate daily returns
    stock_data.loc[:,'Daily Return'] = stock_data['Close'].pct_change()
    # Normalized volatility to the option period
    volatility = stock_data['Daily Return'].rolling(window=window_size).std() * np.sqrt(expire_period)

    signals['MA'] = stock_data['Close'].rolling(window=window_size).mean()
    signals['Spread']=(stock_data['Close']-signals['MA'])/stock_data['Close'].rolling(window=window_size).std()

    if baseline == 'MOV_AVG':
        signals['Upper'] = signals['MA'] * ( 1 + sigma * volatility )
        signals['Down'] = signals['MA'] * ( 1 - sigma * volatility )
    if baseline == 'CLOSE_PRICE':
        signals['Upper'] = stock_data['Close'] * ( 1 + sigma * volatility )
        signals['Down'] = stock_data['Close'] * ( 1 - sigma * volatility )

    # To be rewrite ***********************
    # for i in range(len(signals)):
    #     if i < (len(signals)-expire_period):
    #         future_prices = stock_data['Close'].iloc[i+1:i+expire_period+1]
    #     else:
    #         future_prices=0
    #     signals.loc[signals.index[i],'Close price']=stock_data['Close'].iloc[i]
    #     if option_strategy == 'CALL_ONLY':
    #         breaches_band = future_prices > signals['Upper'].iloc[i]
    #         signals.loc[signals.index[i],'Option strike']=signals['Upper'].iloc[i] 
    #         signals.loc[signals.index[i],'Option type']='CALL'       
    #     if option_strategy == 'PUT_ONLY':
    #         breaches_band = future_prices < signals['Down'].iloc[i]
    #         signals.loc[signals.index[i],'Option strike']=signals['Down'].iloc[i]
    #         signals.loc[signals.index[i],'Option type']='PUT'        
    #     if option_strategy == 'CALL_AND_PUT':
    #         breaches_band_down = future_prices < signals['Down'].iloc[i]
    #         breaches_band_up = future_prices > signals['Upper'].iloc[i] #array of index that meets the conditions
    #         breaches_band = breaches_band_up + breaches_band_down
    #         # signals.loc[signals.index[i],'Option type']='call_and_put'    
    #         signals.loc[signals.index[i],'Option type']='call_and_put'    
    #         signals.loc[signals.index[i],'Option strike']=signals['Upper'].iloc[i]    
    #         signals.loc[signals.index[i],'Option strike alt']=signals['Down'].iloc[i]    
    #     if option_strategy == 'CALL_OR_PUT':
    #         trend=''
    #         if stock_data['Close'].iloc[i] > signals['MA'].iloc[i]:
    #             trend='up'
    #             breaches_band = future_prices < signals['Down'].iloc[i]
    #             signals.loc[signals.index[i],'Option strike']=signals['Down'].iloc[i]    
    #             signals.loc[signals.index[i],'Option type']='PUT'    
    #         else:
    #             trend='down'
    #             breaches_band = future_prices > signals['Upper'].iloc[i]
    #             signals.loc[signals.index[i],'Option strike']=signals['Upper'].iloc[i]    
    #             signals.loc[signals.index[i],'Option type']='CALL'    

    #     signal_sum=0
    #     for j in range(i-metric_eval_period,i):    #cumulative breach over 6 month
    #         signal_sum=signal_sum+signals['Signal'][j]
    #     if i>0 and i < (len(signals)-expire_period) :
    #         signals.loc[signals.index[i],'Loss probability'] = signal_sum/metric_eval_period*100
    #     elif i>0:
    #         signals.loc[signals.index[i],'Loss probability'] = signals['Loss probability'][i-1]

    #     if breaches_band.any() and i < (len(signals)-expire_period):
    #         if option_strategy == 'CALL_ONLY':
    #                 max_breach_price = future_prices[breaches_band].max()-signals['Upper'].iloc[i]
    #                 max_breach_date = signals.index[i]
    #                 # print('Max Breach detected: %s, %s' % (max_breach_date,max_breach_price))
    #                 signals.at[signals.index[i], 'Signal'] = 1
    #                 signals.at[signals.index[i], 'Loss'] = max_breach_price*100
    #         if option_strategy == 'PUT_ONLY':
    #                 min_breach_price = future_prices[breaches_band].min()-signals['Down'].iloc[i]
    #                 min_breach_date = signals.index[i]
    #                 # print('Min Breach detected: %s, %s' % (min_breach_date,min_breach_price))
    #                 signals.at[signals.index[i], 'Signal'] = 1
    #                 signals.at[signals.index[i], 'Loss'] = -min_breach_price*100
    #         if option_strategy == 'CALL_AND_PUT':
    #             max_breach_price=0
    #             min_breach_price=0
    #             if breaches_band_down.any():
    #                 min_breach_price = future_prices[breaches_band_down].min()-signals['Down'].iloc[i]
    #                 min_breach_date = signals.index[i]
    #                 # print('Min Breach detected: %s, %s' % (min_breach_date,min_breach_price))
    #             if breaches_band_up.any():
    #                 max_breach_price = future_prices[breaches_band_up].max()-signals['Upper'].iloc[i]
    #                 max_breach_date = signals.index[i]
    #                 # print('Max Breach detected: %s, %s' % (max_breach_date,max_breach_price))
    #             signals.at[signals.index[i], 'Signal'] = 1
    #             signals.at[signals.index[i], 'Loss'] = max_breach_price*100-min_breach_price*100           
    #         if option_strategy == 'CALL_OR_PUT':
    #             if trend == 'up':
    #                 min_breach_price = future_prices[breaches_band].min()-signals['Down'].iloc[i]
    #                 min_breach_date = signals.index[i]
    #                 # print('Min Breach detected: %s, %s' % (min_breach_date,min_breach_price))
    #                 signals.at[signals.index[i], 'Signal'] = 1
    #                 signals.at[signals.index[i], 'Loss'] = -min_breach_price*100
    #             if trend == 'down':
    #                 max_breach_price = future_prices[breaches_band].max()-signals['Upper'].iloc[i]
    #                 max_breach_date = signals.index[i]
    #                 # print('Max Breach detected: %s, %s' % (max_breach_date,max_breach_price))
    #                 signals.loc[signals.index[i], 'Signal'] = 1
    #                 signals.loc[signals.index[i], 'Loss'] = max_breach_price*100

    #     signals.at[signals.index[i],'Cumulative loss']=signals.at[signals.index[i-1],'Cumulative loss']+signals.at[signals.index[i],'Loss']/expire_period
    #     loss_occured = signals['Loss'].iloc[0:i]
    #     loss_occured = loss_occured[loss_occured !=0 ]
    #     # print(loss_occured)
    #     if loss_occured.any():
    #         signals.loc[signals.index[i],'average loss']=np.average(loss_occured)
    #     else:
    #         signals.loc[signals.index[i],'average loss']=0
    #     signals.at[signals.index[i],'Loss std']=loss_occured.std()
    # To be rewrite ***********************


    # Calculate future prices using rolling window
    # future_prices = stock_data['Close'].shift(-1).rolling(window=expire_period, min_periods=1).apply(lambda x: x[-1] if len(x) > 0 else 0)

    future_prices_max = np.max(np.lib.stride_tricks.sliding_window_view(stock_data['Close'], window_shape=expire_period), axis=1)
    future_prices_min = np.min(np.lib.stride_tricks.sliding_window_view(stock_data['Close'], window_shape=expire_period), axis=1)

    future_prices_max = np.concatenate((future_prices_max,stock_data['Close'].iloc[-expire_period+1:]))
    future_prices_min = np.concatenate((future_prices_min,stock_data['Close'].iloc[-expire_period+1:]))

    # Assign Close price
    signals['Close price'] = stock_data['Close']

    # Calculate breaches_band based on option_strategy
    if option_strategy == 'CALL_ONLY':
        breaches_band = future_prices_max > signals['Upper']
        signals['Option strike'] = signals['Upper']
        signals['Option type'] = 'CALL'
    elif option_strategy == 'PUT_ONLY':
        breaches_band = future_prices_min < signals['Down']
        signals['Option strike'] = signals['Down']
        signals['Option type'] = 'PUT'
    elif option_strategy == 'CALL_AND_PUT':
        breaches_band_down = future_prices_min < signals['Down']
        breaches_band_up = future_prices_max > signals['Upper']
        breaches_band = breaches_band_up + breaches_band_down
        signals['Option type'] = 'call_and_put'
        signals['Option strike'] = np.where(breaches_band_up, signals['Upper'], signals['Down'])
    elif option_strategy == 'CALL_OR_PUT':
        trend = np.where(stock_data['Close'] > signals['MA'], 'up', 'down')
        breaches_band = np.where(trend == 'up', future_prices_min < signals['Down'], future_prices_max > signals['Upper'])
        signals['Option type'] = np.where(trend == 'up', 'PUT', 'CALL')
        signals['Option strike'] = np.where(trend == 'up', signals['Down'], signals['Upper'])

    # Calculate signal based on breaches_band
    signals['Signal'] = 0
    if breaches_band.any():
        if option_strategy == 'CALL_ONLY':
            max_breach_price = future_prices_max - signals['Upper']
            signals.loc[breaches_band, 'Signal'] = 1
            signals.loc[breaches_band, 'Loss'] = max_breach_price * 100
        elif option_strategy == 'PUT_ONLY':
            min_breach_price = future_prices_min - signals['Down']
            signals.loc[breaches_band, 'Signal'] = 1
            signals.loc[breaches_band, 'Loss'] = -min_breach_price * 100
        elif option_strategy == 'CALL_AND_PUT':
            max_breach_price = (future_prices_max - signals['Upper']) * breaches_band_up
            min_breach_price = (future_prices_min - signals['Down']) * breaches_band_down
            signals.loc[breaches_band, 'Signal'] = 1
            signals.loc[breaches_band, 'Loss'] = (max_breach_price - min_breach_price) * 100
        elif option_strategy == 'CALL_OR_PUT':
            max_breach_price = (future_prices_max - signals['Upper']) * (trend == 'down')
            min_breach_price = (signals['Down']-future_prices_min) * (trend == 'up')
            signals.loc[breaches_band, 'Signal'] = 1
            signals.loc[breaches_band, 'Loss'] = (max_breach_price + min_breach_price) * 100

    # Calculate Loss probability
    signal_sum = signals['Signal'].rolling(window=metric_eval_period, min_periods=1).sum()
    signals['Loss probability'] = signal_sum.shift(1) / metric_eval_period * 100

    # Calculate Cumulative loss
    signals['Cumulative loss'] = signals['Loss'].cumsum() / expire_period

    # Calculate average loss and Loss std
    loss_occured = signals['Loss'].iloc[0:-1].where(signals['Loss'] != 0)
    signals['average loss'] = loss_occured.mean()
    signals['Loss std'] = loss_occured.std()


    #==========================================================================================================


    # Find out option target expiration date 
    #==========================================================================================================
    # target_expiration_date = dt.datetime.now() + dt.timedelta(days=30)

    # options = yf.Ticker(ticker).options

    # expiration_dates = [dt.datetime.strptime(expiration, '%Y-%m-%d') for expiration in options]
    # closest_expiration_date = min(expiration_dates, key=lambda date: abs(date - target_expiration_date))
    # # closest_expiration_date_str = closest_expiration_date.strftime('%Y-%m-%d')
    # #==========================================================================================================


    # # Get option quote
    # #==========================================================================================================
    # option_data=get_option_data(ticker,closest_expiration_date,'CALL',signals['Upper'])
    # for i, od in option_data.iterrows():
    #     print(od)
    #     contract_symbol = od["contractSymbol"]
    #     option_history = get_option_history_data(contract_symbol)
    #     print(option_history)
    #     first_option_history = option_history.iloc[0]
    #     first_option_history_date = option_history.index[0]
    #     first_option_history_close = first_option_history["Close"]
    #     print("For {}, the closing price was ${:.2f} on {}.".format(
    #         contract_symbol,
    #         first_option_history_close,
    #         first_option_history_date
    #     ))


    return signals

def main():
    # Example: Backtest for Apple Inc. (AAPL) stock
# def backtest_strategy(ticker, window_size_ratio=10, sigma=3, baseline='MOV_AVG', option_strategy='CALL_ONLY', expire_period=30):
    ticker = 'AAPL'
    window_size_ratio=8
    sigma=0.8
    baseline='CLOSE_PRICE'
    option_strategy='CALL_ONLY'
    expire_period=5

    signals = np.empty(4, dtype=object)

    # option_strategy_test=['CALL_ONLY','CALL_OR_PUT','PUT_ONLY','CALL_AND_PUT']
    option_strategy_test=['CALL_ONLY','CALL_OR_PUT']

    test_combination=2

    end_date=dt.date.today()
    offset = timedelta(days=365*0.5)
    start_date=end_date-offset
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    for i in range(test_combination):
        option_strategy=option_strategy_test[i]
        signals[i]=backtest_strategy(stock_data,window_size_ratio,sigma,baseline,option_strategy,expire_period)
    
    # signals = backtest_strategy(ticker,6,2,'CLOSE_PRICE','CALL_AND_PUT',10)
    # signals = backtest_strategy(ticker,6,2,'CLOSE_PRICE','CALL_AND_PUT',10)
    # signals = backtest_strategy(ticker,6,2,'CLOSE_PRICE','CALL_AND_PUT',10)
    # signals = backtest_strategy(ticker,6,2,'CLOSE_PRICE','CALL_AND_PUT',10)

    # print(signals[signals['Loss probability']>0])

    plt.close('all')

    # Plot multiple series with different y scales
    fig, axes = plt.subplots(nrows=3,ncols=test_combination,figsize=(30, 10))

    for i in range(test_combination):
        #Plot the first series on the primary y-axis
        axes[0][i].plot(signals[i]['Loss probability'], label='Loss probability', color='blue')
        axes[0][i].set_ylabel('Loss probability', color='blue')
        axes[0][i].tick_params('y', colors='blue')

        ax2=axes[0][i].twinx()
        ax2.plot(signals[i]['Loss'],label='Loss', color='red')
        # ax2.plot(signals[i]['Cumulative loss'], label='Cumulative loss')
        ax2.plot(signals[i]['Loss std'], label='Loss std')
        ax2.plot(signals[i]['average loss'], label='average loss')
        ax2.set_ylabel('Expected Loss', color='red')

        # Plot the other series on the secondary y-axis
        axes[1][i].plot(signals[i]['MA'], label='MA', color='green')
        axes[1][i].plot(signals[i]['Close price'], label='Close price', color='red')
        axes[1][i].scatter(signals[i].index,signals[i]['Option strike'], s=1,label='Option strike', color='orange')
        axes[1][i].set_ylabel('MA', color='green')
        axes[1][i].tick_params('y', colors='green')

        data = signals[i]
        Spread_at_loss=data[data['Loss']>0]['Spread']
        
        axes[2][i].hist(Spread_at_loss, bins=30, color='skyblue', edgecolor='black')
        # axes[2][i].hist(signals[i]['Spread'], bins=30)

    # Plot the other series on the secondary y-axis
    # ax3.plot(signals['Close price'], label='Close price', color='green')
    # ax3.set_ylabel('Close price', color='green')
    # ax3.tick_params('y', colors='red')



    # Add title and legend
    plt.title('Multiple Series with Different Y Scales')
    fig.tight_layout()
    fig.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()
