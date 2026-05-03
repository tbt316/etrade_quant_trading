import pandas as pd
import numpy as np
from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm, fetch_historical_data
import asyncio
import os

print("Fetching historical data...")
df = fetch_historical_data()

ingestor = DataIngestor()
start_date = df.index.min().strftime("%Y-%m-%d")
end_date = df.index.max().strftime("%Y-%m-%d")

print("Building raw and scaled datasets...")
# Get raw combined data
fred_df = ingestor.fetch_fred_data(start_date, end_date)
yf_df = ingestor.fetch_yf_data(start_date, end_date)
raw_combined = pd.concat([yf_df, fred_df], axis=1).ffill().dropna()

# Train HMM which handles the async call internally
print("Training HMM...")
best_hmm, best_k, feature_df = train_regime_hmm(df, n_components=3, expanding_window=True)

# Causal Probabilities (Forward Filtering)
pc_cols = [c for c in feature_df.columns if c.startswith('PC')]
features_scaled = feature_df[pc_cols].values
n_samples = len(features_scaled)
all_probs = np.zeros((n_samples, best_k))

print(f"Calculating causal probabilities for {n_samples} samples...")
for t in range(n_samples):
    # predict_proba on an expanding window returns the filtered probability at the last step
    all_probs[t] = best_hmm.predict_proba(features_scaled[:t+1])[-1]

for i in range(best_k):
    feature_df[f'Prob_State_{i}'] = all_probs[:, i]

# Merge
print("Merging dataframes...")
final_df = raw_combined.join(feature_df, how='inner')

out_path = '/Users/btian/EtradePythonClient/etrade_python_client/quant-review-0430/quant_review_data.csv'
final_df.to_csv(out_path)
print(f"Data saved to {out_path}")
