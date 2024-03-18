import numpy as np
import yfinance as yf
from datetime import timedelta
import datetime as dt
from scipy import optimize
from math import sqrt, exp, log
from scipy.stats import norm

# Function to calculate option price using the binomial model with early exercise
def binomial_option_price_with_early_exercise(stock_data, strike, expire_days, r, n, option_type='call'):

    # end_date=dt.date.today()
    # end_date=purchase_date
    # offset = timedelta(days=30)
    # start_date=end_date-offset

    # stock_data = yf.download(ticker_symbol, start_date, end_date)

    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Close'].pct_change()

    # Calculate volatility (standard deviation of daily returns)
    # volatility = np.std(stock_data['Daily Return'])
    current_price=stock_data['Close'].iloc[-1]

    # Annualize volatility
    trading_days_per_year = 252  # Assuming 252 trading days in a year
    # annualized_volatility = volatility * np.sqrt(trading_days_per_year)
    volatility = np.std(stock_data['Daily Return']) * np.sqrt(trading_days_per_year)
    # print("Historical Volatility (daily):", volatility)

    # volatility = 0.1996

    # print("Latest price:", current_price)

    delta_t =expire_days / trading_days_per_year / n
    u = np.exp(volatility * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    # Create stock price tree
    stock_tree = np.zeros((n+1, n+1))
    for j in range(n+1):
        for i in range(j+1):
            stock_tree[i, j] = current_price * (u ** (j - i)) * (d ** i)
    
    # Create option price tree
    option_tree = np.zeros((n+1, n+1))
    for j in range(n+1):
        if option_type == 'call':
            option_tree[j, n] = max(0, stock_tree[j, n] - strike)
        else:
            option_tree[j, n] = max(0, strike - stock_tree[j, n])
    
    # Calculate option price at earlier nodes
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            if option_type == 'call':
                option_tree[i, j] = max(stock_tree[i, j] - strike, 
                                        np.exp(-r * delta_t) * (p * option_tree[i, j+1] + (1 - p) * option_tree[i+1, j+1]))
            else:
                option_tree[i, j] = max(strike - stock_tree[i, j], 
                                        np.exp(-r * delta_t) * (p * option_tree[i, j+1] + (1 - p) * option_tree[i+1, j+1]))
    print('Option price estimate: ',option_tree[0, 0])
    return option_tree[0, 0]

def black_scholes_call(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def call_price_difference(sigma, S, K, T, r, market_price):
    model_price = black_scholes_call(S, K, T, r, sigma)
    return model_price - market_price

def implied_volatility(S, K, T, r, market_price):
    # Initial guess for volatility
    initial_volatility = 0.3
    
    # Find implied volatility using bisection method
    implied_volatility, _ = optimize.brentq(call_price_difference, 0.01, 1.0, 
                                            args=(S, K, T, r, market_price),
                                            full_output=True)
    return implied_volatility

# Example usage
S = 172.62  # Current stock price
K = 185  # Strike price
T = 9 / 252  # Time to expiration in years
r = 0.05  # Risk-free rate
market_price = 0.18  # Market price of the call option

# Calculate implied volatility
implied_vol = implied_volatility(S, K, T, r, market_price)
print("Implied Volatility:", implied_vol)


# Define parameters
# S = 32.51  # Current stock price
strike = 185 # Strike price
expire_days = 25    # Time to expiration (in days)
r = 0.05   # Risk-free rate (daily)
# sigma = 0.217875  # Volatility (daily)
n = 100    # Number of time steps

# end_date=dt.date.today()
# offset = timedelta(days=365*6)
# start_date=end_date-offset
# stock_data = backtest_bo.fetch_stock_data(ticker, start_date, end_date)
# Calculate call option price with early exercise
# call_price_with_early_exercise = binomial_option_price_with_early_exercise(stock_data, strike, expire_days,r, n)

