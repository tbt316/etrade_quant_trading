import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from backtesting.greeks_calculator import bs_gamma

def test_gamma_zero_vol():
    S = 100.0
    K = 100.0
    r = 0.05
    T = 0.01
    q = 0.0
    
    print("Testing Gamma with sigma=0:")
    g = bs_gamma(S, K, T, r, 0.0, q)
    print(f"  sigma=0.0, Gamma={g:.6f}")

if __name__ == "__main__":
    test_gamma_zero_vol()
