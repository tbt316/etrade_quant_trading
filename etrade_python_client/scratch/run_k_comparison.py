import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import brier_score_loss

# adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import (
    fetch_historical_data, prepare_hmm_features, train_regime_hmm, 
    calculate_probability_of_touch, fit_gmm
)

def run_k_comparison():
    print("="*60)
    print("HMM K-Value Prediction Accuracy Evaluation")
    print("="*60)
    print("Evaluating how different K (hidden states) perform on")
    print("out-of-sample (OOS) probability predictions of experiencing")
    print("a -5% mid-trade Maximum Adverse Excursion (Touch Risk) over 45 days.")
    print("="*60)
    
    df = fetch_historical_data()
    if df.empty:
        print("Data unavailable.")
        return
        
    horizon = 45
    threshold = -0.05
    
    # Target Construction
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    df['future_min_price'] = df['SPY_Low'].rolling(window=indexer, min_periods=1).min()
    df['future_mae_return'] = df['future_min_price'] / df['SPY_Close'] - 1
    df['Target_Breach'] = (df['future_mae_return'] <= threshold).astype(int)
    
    # Required for features
    df['Log_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    df = df.dropna(subset=['Log_Return', 'VIX_Close', 'VVIX_Close', 'Target_Breach'])
    
    features, df_feat = prepare_hmm_features(df)
    
    # 80/20 Train-Test Split (Walk Forward Simulation Base)
    train_size = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:train_size].copy()
    test_df = df_feat.iloc[train_size:].copy()
    train_features = features[:train_size]
    test_features = features[train_size:]
    
    print(f"Total valid samples: {len(df_feat)}")
    print(f"Training on {train_size} samples. Testing on {len(test_df)} out-of-sample points.")
    
    k_range = [1, 2, 3, 4, 5, 6]
    results = []
    
    print("\n--- Evaluating K Values ---")
    for k in k_range:
        print(f"Training K={k}...", end=" ", flush=True)
        t0 = time.time()
        
        # 1. Train HMM on Training Data
        hmm_model, _, trained_feature_df = train_regime_hmm(train_df, n_components=k)
        
        # 2. Build daily GMM parameters based on state assignments from training
        daily_models = []
        for state in range(k):
            # daily returns
            state_log_returns = trained_feature_df[trained_feature_df['HMM_State'] == state]['Log_Return'].values
            if len(state_log_returns) == 0:
                state_log_returns = np.array([0.0])
            daily_frac_returns = np.exp(state_log_returns) - 1
            daily_models.append(fit_gmm(daily_frac_returns, regime_label=f'Daily_State_{state}'))
            
        print(f"done in {time.time()-t0:.2f}s. Evaluating...", end=" ", flush=True)
        
        # 3. OOS Evaluation
        actuals = test_df['Target_Breach'].values
        
        if len(test_df) > 0:
            # FIX: Align with new scaling/PCA architecture
            test_scaled = hmm_model.scaler_.transform(test_df)
            test_pcs = hmm_model.fusion_.sparse_pca.transform(test_scaled)
            test_posteriors = hmm_model.predict_proba(test_pcs)
            
            # Predict P(Touch) for each independent State out to 45 Days
            state_touch_probs = []
            for state in range(k):
                one_hot = np.zeros(k)
                one_hot[state] = 1.0
                p_touch = calculate_probability_of_touch(
                    current_state_probs=one_hot,
                    trans_matrix=hmm_model.transmat_,
                    gmm_models=daily_models,
                    dte=horizon,
                    strike_pct_drop=threshold,
                    num_paths=1000,
                    option_type="put"
                )
                state_touch_probs.append(p_touch)
                
            state_touch_probs = np.array(state_touch_probs)
            
            # Final Day-by-Day Prediction is Dot Product of Posterior weights and State specific probs
            pred_probs = test_posteriors @ state_touch_probs
            
            # Calculate Brier Score
            valid_mask = ~np.isnan(actuals)
            y_true = actuals[valid_mask]
            y_pred = pred_probs[valid_mask].astype(float)
            
            if len(y_true) > 0:
                brier = brier_score_loss(y_true, y_pred)
            else:
                brier = np.nan
        else:
            brier = np.nan
            
        train_scaled = hmm_model.scaler_.transform(train_df)
        train_pcs = hmm_model.fusion_.sparse_pca.transform(train_scaled)
        train_loglik = hmm_model.score(train_pcs)
            
        print(f"OOS Brier Score: {brier:.4f}")
        results.append({
            'K': k,
            'Train_LogLikelihood': round(train_loglik, 2),
            'OOS_Brier': brier
        })
        
    print("\n" + "="*45)
    print("Summary Statistics:")
    print("="*45)
    res_df = pd.DataFrame(results).set_index('K')
    print(res_df)
    print("---------------------------------------------")
    print("* Lower Brier Score is better (measures prediction error)")
    print("* Higher Train_LogLikelihood means better model fit on training set")

if __name__ == '__main__':
    run_k_comparison()
