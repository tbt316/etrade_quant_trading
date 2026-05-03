import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t as student_t

# Add current directory to path
sys.path.append(os.getcwd())

# Re-implement the logic for standalone testing
PROBABILITY_MODEL = 'bootstrap' # 'bootstrap' or 'parametric'
VIX_WINDOW = 45 # 45-trading day forward window (~2 months)
MONTE_CARLO_SIMS = 5000

def fetch_historical_data():
    cache_path = 's_and_p_data/spy_vix_historical_raw.csv'
    if os.path.exists(cache_path):
        print("Using cached data.")
        df = pd.read_csv(cache_path, index_index=True, parse_dates=True)
        return df
    
    print("Fetching data from yfinance...")
    spy_df = yf.download('SPY', period='max', auto_adjust=True)
    vix_df = yf.download('^VIX', period='max', auto_adjust=True)
    
    # Handle both single and multi-level columns
    spy = spy_df['Close'] if 'Close' in spy_df.columns else spy_df.xs('Close', axis=1, level=0)
    vix = vix_df['Close'] if 'Close' in vix_df.columns else vix_df.xs('Close', axis=1, level=0)
    
    df = pd.DataFrame({'SPY': spy, 'VIX': vix}).dropna()
    os.makedirs('s_and_p_data', exist_ok=True)
    df.to_csv(cache_path)
    return df

def build_regime_return_arrays(df):
    df = df.copy()
    df['next_45d_ret'] = df['SPY'].shift(-VIX_WINDOW) / df['SPY'] - 1
    df['daily_ret'] = df['SPY'].pct_change()
    df = df.dropna()
    
    regimes = {
        'low': df[df['VIX'] < 15]['daily_ret'].values,
        'normal': df[(df['VIX'] >= 15) & (df['VIX'] < 20)]['daily_ret'].values,
        'high': df[(df['VIX'] >= 20) & (df['VIX'] < 30)]['daily_ret'].values,
        'extreme': df[df['VIX'] >= 30]['daily_ret'].values
    }
    return regimes

def get_probability_engine(regime_returns, model='bootstrap'):
    if model == 'bootstrap':
        def prob_func(strike_pct):
            # strike_pct is e.g. 0.95 for 5% OTM put
            sims = np.random.choice(regime_returns, size=(MONTE_CARLO_SIMS, VIX_WINDOW))
            paths = np.cumprod(1 + sims, axis=1)
            # Probability of touching or crossing the strike at any point (First-Passage Time)
            breached = np.any(paths <= strike_pct, axis=1)
            return np.mean(breached)
    else:
        # Student's t distribution
        params = student_t.fit(regime_returns)
        df_p, loc_p, scale_p = params
        scale_45d = scale_p * np.sqrt(VIX_WINDOW)
        def prob_func(strike_pct):
            return student_t.cdf(strike_pct - 1, df_p, loc=loc_p * VIX_WINDOW, scale=scale_45d)
            
    return prob_func

def main():
    df = fetch_historical_data()
    regimes = build_regime_return_arrays(df)
    
    current_vix = 18.5 # Normal
    regime_name = 'normal' if current_vix < 20 else 'high'
    returns = regimes[regime_name]
    
    print(f"\n--- Standalone Verification (VIX: {current_vix}, Regime: {regime_name}) ---")
    
    strikes = [0.98, 0.95, 0.90] # 2%, 5%, 10% OTM
    
    for model in ['bootstrap', 'parametric']:
        print(f"\nModel: {model}")
        prob_engine = get_probability_engine(returns, model=model)
        for s in strikes:
            p = prob_engine(s)
            print(f"  Strike {s*100:.0f}%: P(breach) = {p:.2%}")

if __name__ == "__main__":
    main()
