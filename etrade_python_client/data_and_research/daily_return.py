import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
ticker = 'AAPL'

# Calculate the date range: past 5 years from the current date
end_date = datetime.now().date()
start_date = end_date - timedelta(days=5*365 + 1)  # Approximate 5 years, accounting for leap years

# Download historical stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns based on adjusted closing prices
data['Daily Return'] = data['Adj Close'].pct_change()

# Drop the first row with NaN value
data = data.dropna()

# Generate the histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Daily Return'], bins=50, color='blue', edgecolor='black')
plt.title(f'Histogram of Daily Returns for {ticker} (Past 5 Years)')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()