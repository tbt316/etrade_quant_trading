import numpy as np
from scipy.stats import norm

def black_scholes_option_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    
    return option_price

# Define option parameters
S = 172.62  # Current stock price
K = 195  # Strike price
T = 20/252    # Time to expiration (in years)
r = 0.05 # Risk-free interest rate
sigma = 0.2003 # Implied volatility

# Calculate option price
call_price = black_scholes_option_price(S, K, T, r, sigma, option_type='call')
put_price = black_scholes_option_price(S, K, T, r, sigma, option_type='put')

print("Call option price:", call_price)
print("Put option price:", put_price)
