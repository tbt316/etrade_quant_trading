import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from backtesting.greeks_calculator import bs_gamma

def test_gamma_risk_neutral():
    S = 100.0
    K = 100.0
    r = 0.05
    q = 0.05 # r = q
    sigma = 0.2
    
    print("Testing Patched Gamma (r=q) near expiration (ATM):")
    for T in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0]:
        g = bs_gamma(S, K, T, r, sigma, q)
        print(f"  T={T:.6f}, Gamma={g:.6f}")

if __name__ == "__main__":
    test_gamma_risk_neutral()
