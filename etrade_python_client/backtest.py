from ast import Break
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def backtest_strategy(stock_data, window_size=10, num_days_to_check=30):
    signals = pd.DataFrame(index=stock_data.index)
    signals['Signal'] = 0  # Initialize the signal column
    signals['BreachOdd'] = 0

    # Calculate 10-day moving average and 3 standard deviations
    signals['MA'] = stock_data['Close'].rolling(window=window_size).mean()
    signals['Upper'] = signals['MA'] + 3 * stock_data['Close'].rolling(window=window_size).std()
    # signals['Upper'] = stock_data['Close'] + 3 * stock_data['Close'].rolling(window=window_size).std()

    for i in range(len(signals) - num_days_to_check):
        future_prices = stock_data['Close'].iloc[i+1:i+num_days_to_check+1]
        breaches_upper_band = future_prices > signals['Upper'].iloc[i]
        signal_sum=0
        for j in range(i):
            signal_sum=signal_sum+signals['Signal'][j]
        if i>0:
            signals['BreachOdd'][i] = signal_sum/i*100

        if breaches_upper_band.any():
            max_breach_price = future_prices[breaches_upper_band].max()
            max_breach_date = future_prices[breaches_upper_band].idxmax()
            # print('Breach detected: %s, %s,%s' % (max_breach_date,max_breach_price,signals['BreachOdd'][i]))
            signals.at[max_breach_date, 'Signal'] = 1

    return signals

def main():
    # Example: Backtest for Apple Inc. (AAPL) stock
    ticker = 'ENPH'
    start_date = '2018-01-01'
    end_date = '2023-12-31'

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    signals = backtest_strategy(stock_data)
    print(signals[signals['BreachOdd']>0])

    # Plot multiple series with different y scales
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first series on the primary y-axis
    ax1.plot(signals['BreachOdd'], label='BreachOdd', color='blue')
    ax1.set_ylabel('BreachOdd', color='blue')
    ax1.tick_params('y', colors='blue')

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Plot the other series on the secondary y-axis
    ax2.plot(signals['MA'], label='MA', color='green')
    ax2.set_ylabel('MA', color='green')
    ax2.tick_params('y', colors='green')

    ax3 = ax1.twinx()

    # Plot the other series on the secondary y-axis
    ax3.plot(signals['Signal'], label='Signal', color='green')
    ax3.set_ylabel('Signal', color='green')
    ax3.tick_params('y', colors='red')



    # Add title and legend
    plt.title('Multiple Series with Different Y Scales')
    fig.tight_layout()
    fig.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()
