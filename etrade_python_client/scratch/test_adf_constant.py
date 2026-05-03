import numpy as np
from statsmodels.tsa.stattools import adfuller

try:
    series = np.array([0.13] * 100)
    res = adfuller(series)
    print(f"ADF p-value: {res[1]}")
except Exception as e:
    print(f"ADF failed: {e}")
