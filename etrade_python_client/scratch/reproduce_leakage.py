import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from hmmlearn import hmm
import sys
import os

# Mock PCAFusion to avoid dependencies for this unit test
class MockFusion:
    def __init__(self):
        self.sparse_pca = None
    def fit_transform(self, df):
        return df # No-op for test
    def inverse_transform(self, x):
        return x

def mock_get_regime_labels(hmm_model, pc_df=None):
    """Refactored version of the labeling logic for testing."""
    K = hmm_model.n_components
    labels = {}
    
    # In the real engine, we'd use inverse_transform. 
    # Here we assume the features are already [Log_Return, VIX] for simplicity.
    
    for i in range(K):
        # CURRENT LOGIC (Global Mean)
        # avg_ret, avg_vix = hmm_model.means_[i]
        
        # PROPOSED LOGIC (State-Specific Causal/Local Centroid)
        if pc_df is not None and 'HMM_State' in pc_df.columns:
            state_data = pc_df[pc_df['HMM_State'] == i]
            if len(state_data) > 0:
                # Use recent data if available, else global for this state
                # Here we use the whole state_data to demonstrate the 'Global' vs 'Parameter' difference
                # but in the real fix we might want to prioritize the recent observations.
                avg_ret = state_data['Log_Return'].mean()
                avg_vix = state_data['VIX_Close'].mean()
            else:
                avg_ret, avg_vix = hmm_model.means_[i]
        else:
            avg_ret, avg_vix = hmm_model.means_[i]

        # Archetype mapping (simplified)
        if avg_ret > 0.0005 and avg_vix < 15:
            name = "Robust Expansion"
        elif avg_ret > 0 and avg_vix < 22:
            name = "Emerging Expansion"
        elif avg_vix > 30:
            name = "Market Turmoil"
        elif abs(avg_ret) < 0.0005 and avg_vix > 20:
            name = "High Vol Chop"
        elif avg_ret < 0:
            name = "Cautious Decline"
        else:
            name = f"Regime {i}"
            
        labels[i] = f"{name} ({i})"
        
    return labels

def reproduce_leakage():
    print("Simulating 5 years of Expansion + 1 month of Crash...")
    
    # 1. Generate Expansion Data (1250 days)
    expansion_ret = np.random.normal(0.001, 0.005, 1250)
    expansion_vix = np.random.normal(15, 2, 1250)
    
    # 2. Generate Crash Data (20 days)
    crash_ret = np.random.normal(-0.02, 0.03, 20)
    crash_vix = np.random.normal(60, 10, 20)
    
    df = pd.DataFrame({
        'Log_Return': np.concatenate([expansion_ret, crash_ret]),
        'VIX_Close': np.concatenate([expansion_vix, crash_vix])
    })
    
    # Train a 2-state HMM
    model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
    model.fit(df.values)
    
    # Predict states
    states = model.predict(df.values)
    df['HMM_State'] = states
    
    print("\n--- Model Parameters (Hindsight/Global) ---")
    for i in range(model.n_components):
        m_ret, m_vix = model.means_[i]
        print(f"State {i}: Mean Ret={m_ret:.5f}, Mean VIX={m_vix:.2f}")

    # Demonstate the "Leakage" labels
    # We'll use a copy of the actual function logic but with the old global approach
    def get_old_labels(m):
        labs = {}
        for i in range(m.n_components):
            avg_ret, avg_vix = m.means_[i]
            if avg_ret > 0 and avg_vix < 22: name = "Emerging Expansion"
            elif avg_vix > 30: name = "Market Turmoil"
            else: name = "Other"
            labs[i] = name
        return labs

    old_labels = get_old_labels(model)
    print("\n--- Old Labeling (Using Model Means) ---")
    for i, l in old_labels.items():
        print(f"State {i} -> {l}")

    # Now let's see which state was active during the crash
    crash_states = df['HMM_State'].iloc[-20:].unique()
    print(f"\nCrash period states: {crash_states}")
    for s in crash_states:
        print(f"  State {s} Label: {old_labels[s]}")

    # Check for the failure condition
    for s in crash_states:
        if old_labels[s] == "Emerging Expansion":
            print(f"\n❌ REPRODUCED: State {s} active during crash is labeled 'Emerging Expansion'!")
            return True
            
    print("\n✅ Failed to reproduce with these parameters. (Maybe HMM separated them better?)")
    return False

if __name__ == "__main__":
    reproduce_leakage()
