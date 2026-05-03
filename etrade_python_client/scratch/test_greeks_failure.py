import numpy as np
from backtesting.greeks_calculator import bs_gamma, bs_call_price, bs_put_price

def test_gamma_near_expiration():
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    q = 0.0
    
    print("Testing Gamma near expiration (ATM):")
    for T in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0]:
        g = bs_gamma(S, K, T, r, sigma, q)
        print(f"  T={T:.6f}, Gamma={g:.6f}")

def test_dividend_yield():
    S = 100.0
    K = 100.0
    T = 0.1
    r = 0.05
    sigma = 0.2
    q = 0.03
    
    print("\nTesting Dividend Yield (q=0.03):")
    p_with_q = bs_call_price(S, K, T, r, sigma, q)
    p_no_q = bs_call_price(S, K, T, r, sigma, 0.0)
    print(f"  Call Price with q: {p_with_q:.6f}")
    print(f"  Call Price no q:   {p_no_q:.6f}")
    if p_with_q == p_no_q:
        print("  FAILURE: Dividend yield seems to be ignored!")
    else:
        print("  SUCCESS: Dividend yield affects price.")

if __name__ == "__main__":
    test_gamma_near_expiration()
    test_dividend_yield()
