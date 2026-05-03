from live_trading.ev_engine import fetch_historical_data, train_regime_hmm
import traceback

df_hist = fetch_historical_data()
try:
    best_hmm, best_k, feature_df = train_regime_hmm(df_hist, n_components=None)
except Exception as e:
    traceback.print_exc()
