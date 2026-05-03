import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from backtesting.greeks_calculator import bs_gamma, bs_call_price

def test_gamma_near_expiration_patched():
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    q = 0.0
    
    print("Testing Patched Gamma near expiration (ATM):")
    # T values from 0.1 down to 0
    for T in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0]:
        g = bs_gamma(S, K, T, r, sigma, q)
        print(f"  T={T:.6f}, Gamma={g:.6f}")

def test_gamma_zero_vol_patched():
    S = 100.0
    K = 100.0
    r = 0.05
    T = 0.01
    q = 0.0
    
    print("\nTesting Patched Gamma with sigma=0 (ATM):")
    g = bs_gamma(S, K, T, r, 0.0, q)
    print(f"  sigma=0.0, Gamma={g:.6f}")

if __name__ == "__main__":
    test_gamma_near_expiration_patched()
    test_gamma_zero_vol_patched()
