import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

def test_leakage():
    # Create fake data: 2014-2021
    # 2014-2015: Low vol (0-1)
    # 2020: Massive outlier (100)
    dates = pd.date_range("2014-01-01", "2021-01-01", freq="D")
    data = np.random.normal(0, 1, len(dates))
    
    # The 2020 Crash outlier
    crash_idx = dates.get_loc("2020-03-15")
    data[crash_idx] = 1000.0 # Extreme outlier
    
    df = pd.DataFrame(data, index=dates, columns=['Feature'])
    
    # 1. Global Scaling
    scaler_global = RobustScaler()
    df['Global_Scaled'] = scaler_global.fit_transform(df[['Feature']])
    
    # 2. Causal Scaling (Expanding)
    causal_scaled = []
    warmup = 365
    for i in range(len(df)):
        if i < warmup:
            causal_scaled.append(np.nan)
            continue
        # Only fit on data UP TO i
        s = RobustScaler()
        s.fit(df[['Feature']].iloc[:i]) # Fit on history
        val = s.transform(df[['Feature']].iloc[[i]])[0][0] # Transform current
        causal_scaled.append(val)
    
    df['Causal_Scaled'] = causal_scaled
    
    # Check 2015 data
    df_2015 = df.loc["2015-01-01":"2015-12-31"]
    
    print("--- Leakage Analysis (2015 Mean Scaled Value) ---")
    print(f"Global Mean: {df_2015['Global_Scaled'].mean():.6f}")
    print(f"Causal Mean: {df_2015['Causal_Scaled'].mean():.6f}")
    
    if abs(df_2015['Global_Scaled'].mean() - df_2015['Causal_Scaled'].mean()) > 1e-3:
        print("\n🚨 CONFIRMED: Global scaling is leaking future information into 2015!")
        print("Because of the 2020 outlier, the median and IQR of the global dataset are different,")
        print("shifting the 2015 points relative to where they should be based only on 2014 data.")
    else:
        print("\n✅ No significant leakage detected (check outlier magnitude).")

if __name__ == "__main__":
    test_leakage()
