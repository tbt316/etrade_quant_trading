import yfinance as yf
import datetime

# Replace 'AAPL' with the desired stock symbol
symbol = 'AAPL'

# Get the current date
current_date = datetime.datetime.now()

# Calculate the expiration date 30 days from the current date
expiration_date = current_date + datetime.timedelta(days=32)

# Format the expiration date in 'YYYY-MM-DD' string format
expiration_date_str = expiration_date.strftime('%Y-%m-%d')

# Get the options for the specified stock
options = yf.Ticker(symbol).options

print(options)

# Filter options based on the expiration date
options_30_days = [option for option in options if expiration_date_str in option]

# Print the options with 30-day expiration
print(f"Options with {expiration_date_str} expiration date: {options_30_days}")
