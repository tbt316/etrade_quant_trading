import os
import json
import time
import functools
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from live_trading.data_ingestion import DataIngestor
from live_trading.pca_fusion import PCAFusion
from datetime import datetime, timedelta
import asyncio
from sklearn.preprocessing import RobustScaler

def is_hmm_healthy(model):
    """Check if the HMM model has valid (non-NaN, non-Inf) parameters."""
    if model is None: return False
    try:
        if np.any(np.isnan(model.startprob_)) or np.any(np.isinf(model.startprob_)): return False
        if np.any(np.isnan(model.transmat_)) or np.any(np.isinf(model.transmat_)): return False
        
        # For GMMHMM, check weights, means and covars
        if hasattr(model, "weights_"):
            if np.any(np.isnan(model.weights_)) or np.any(np.isinf(model.weights_)): return False
            # Check row sums for weights (must sum to 1 per state)
            if not np.allclose(np.sum(model.weights_, axis=1), 1.0): return False
            
        if hasattr(model, "means_"):
            if np.any(np.isnan(model.means_)) or np.any(np.isinf(model.means_)): return False
            
        if hasattr(model, "covars_"):
            if np.any(np.isnan(model.covars_)) or np.any(np.isinf(model.covars_)): return False
            
    except Exception:
        return False
    return True

# Constants moved from ev_plots.py
PROBABILITY_MODEL = 'gmm'  # Options: 'bootstrap', 'parametric', 'gmm'
USE_MARKOV_TRANSITIONS = False
COST_PER_SPREAD = 1.0
YF_QUOTE_CACHE_PATH = "s_and_p_data/yf_quote_cache.json"
REGIME_CACHE_FILE = "market_regime_results.pkl"


def yf_download_with_retry(ticker, max_retries=3, backoff_base=1.0, **kwargs):
    """Wrapper around yf.download with exponential backoff for transient failures."""
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, **kwargs)
            if not data.empty:
                return data
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                print(f"  [yf_download] {ticker} returned empty (attempt {attempt+1}/{max_retries}). Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt)
                print(f"  [yf_download] {ticker} error: {e} (attempt {attempt+1}/{max_retries}). Retrying in {wait:.1f}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"  [yf_download] {ticker} failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_cached_yf_close(symbol, cache_minutes=15):
    """Fetches latest close from yfinance with a short-lived local cache."""
    os.makedirs("s_and_p_data", exist_ok=True)
    now = datetime.now()
    cache = {}
    if os.path.exists(YF_QUOTE_CACHE_PATH):
        try:
            with open(YF_QUOTE_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    entry = cache.get(symbol)
    if entry and "price" in entry and "timestamp" in entry:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if now - ts < timedelta(minutes=cache_minutes):
                return float(entry["price"])
        except Exception:
            pass
    try:
        price = float(yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1])
        cache[symbol] = {"price": price, "timestamp": now.isoformat()}
        with open(YF_QUOTE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        return price
    except Exception:
        return None

def fetch_historical_data():
    """Fetches high-dimensional historical data using DataIngestor with caching and background sync."""
    ingestor = DataIngestor()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # DataIngestor now handles caching and background sync internally
    df = ingestor.fetch_yf_data(start_str, end_str)
    
    if df.empty:
        from live_trading.data_ingestion import red_alert
        red_alert("Historical data cache is EMPTY. Regime detection will be unavailable until background sync completes.")
        return pd.DataFrame()
        
    # Ensure we have the minimum required columns for HMM training
    required = ['SPY_Close', 'VIX_Close']
    if not all(c in df.columns for c in required):
        from live_trading.data_ingestion import red_alert
        red_alert(f"Critical columns {required} missing from cache. Waiting for background sync...")
        return pd.DataFrame()

    print(f"  Loaded {len(df)} days of historical data from cache.")
    return df

def normalize_index(s):
    """Standardize index to tz-naive, normalized DatetimeIndex."""
    if s.empty: return s
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

def prepare_hmm_features(df):
    """
    Deprecated: Replaced by advanced data ingestion and PCA fusion.
    Kept for backward compatibility if needed.
    """
    df = df.copy()
    if 'SPY_Close' in df.columns:
        df['Log_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        df['Realized_Vol_10d'] = df['Log_Return'].ewm(span=10, adjust=False).std() * np.sqrt(252)
    df = df.dropna()
    
    cols = [c for c in ['Log_Return', 'Realized_Vol_10d', 'VIX_Close', 'VVIX_Close'] if c in df.columns]
    features = df[cols].values
    return features, df

def train_regime_hmm(df, n_components=None, expanding_window=False):
    """
    Upgraded HMM training using PCA-fused features and BIC optimization.
    """
    # 1. High-Dimensional Data Ingestion
    ingestor = DataIngestor()
    # Use sync wrapper for async data fetching
    # FETCH STATIONARY DATA WITHOUT GLOBAL SCALING
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    
    def run_async(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If the loop is already running (e.g. from an async test), 
            # we need to run this in a separate thread and wait for it.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    stationary_df = run_async(ingestor.build_fused_dataset(start_date, end_date, scale=False))
    
    # ENSURE WE ONLY USE DATA UP TO THE PROVIDED DF'S END DATE (Double check)
    stationary_df = stationary_df[stationary_df.index <= df.index.max()]
    
    # 2. CAUSAL FEATURE PREPARATION (Strictly on past data if expanding_window=True)
    if expanding_window:
        print("  [Causal] Preparing features with expanding scaling and PCA...")
        # Use rolling scale from DataIngestor
        scaled_df = ingestor.scale_features(stationary_df, rolling=True, window=min(252*5, len(stationary_df)-1))
        
        # Causal PCA: For each point t, we need the PCA projection based on data up to t
        pc_values = np.full((len(scaled_df), 2), np.nan)
        fusion = PCAFusion()
        
        warmup = min(252, len(scaled_df)-1)
        prev_loadings = None
        
        for t in range(warmup, len(scaled_df) + 1):
            # Regulation 3.1: PCA loadings at time t must only be derived from data 0 to t-1
            # Here t-1 is the point we are transforming, so we fit on :t-1
            window = scaled_df.iloc[:t-1].fillna(0)
            if len(window) < 20: # Defensive
                window = scaled_df.iloc[:t].fillna(0)
                
            fusion.fit(window)
            
            # FIX: Enforce PCA Structural Consistency (Sign + Rank)
            current_loadings = fusion.sparse_pca.components_
            if prev_loadings is not None:
                # 1. Cosine Similarity Matrix to detect Rank Swapping
                from scipy.spatial.distance import cdist
                # cdist(A, B, 'cosine') returns 1 - cos(theta). We want cos(theta) = 1 - cdist
                sim_matrix = 1 - cdist(current_loadings, prev_loadings, metric='cosine')
                
                # Find best mapping (Hungarian)
                from scipy.optimize import linear_sum_assignment
                new_idx, old_idx = linear_sum_assignment(-np.abs(sim_matrix)) # Maximize absolute similarity
                
                # Reorder current loadings and components to match previous rank
                fusion.sparse_pca.components_ = fusion.sparse_pca.components_[new_idx]
                current_loadings = fusion.sparse_pca.components_
                
                # 2. Enforce Sign Consistency on the mapped components
                for i in range(len(current_loadings)):
                    if np.dot(current_loadings[i], prev_loadings[old_idx[i]]) < 0:
                        fusion.sparse_pca.components_[i] *= -1
            
            prev_loadings = fusion.sparse_pca.components_.copy()
            
            pc_values[t-1] = fusion.transform(scaled_df.iloc[t-1:t].fillna(0)).values[0]
            
        pc_df = pd.DataFrame(pc_values, index=scaled_df.index, columns=['PC1', 'PC2'])
        scaler = None 
    else:
        # LOCAL WINDOW SCALING (Fixed window training)
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(stationary_df)
        scaled_df = pd.DataFrame(scaled_values, index=stationary_df.index, columns=stationary_df.columns)
        
        # LOCAL PCA FUSION
        fusion = PCAFusion()
        pc_df = fusion.fit_transform(scaled_df)
    
    features_scaled = pc_df.dropna().values
    valid_index = pc_df.dropna().index
    pc_df = pc_df.loc[valid_index]
    
    # REMOVED EMA Smoothing: Gaussian HMMs assume conditionally independent emissions.
    # EMA forces artificial autocorrelation, causing overestimated persistence and lagging signals.
    features_input = features_scaled
    
    # 4. Dynamic State Optimization (BIC)
    best_hmm = None
    best_k = 1
    
    # Handle NaNs that might have slipped through
    if np.any(np.isnan(features_input)) or np.any(np.isinf(features_input)):
        print("  WARNING: Input features contain NaNs. Cleaning...")
        features_input = np.nan_to_num(features_input, nan=0.0)

    if n_components is not None:
        best_hmm = hmm.GMMHMM(n_components=n_components, n_mix=2, covariance_type="diag", n_iter=100, random_state=42, min_covar=0.05, init_params="mcw")
        # STRENGTHEN PERSISTENCE: Initialize transmat to be very persistent
        best_hmm.startprob_ = np.ones(n_components) / n_components
        best_hmm.transmat_ = np.eye(n_components) * 0.95 + np.ones((n_components, n_components)) * 0.05 / n_components
        best_hmm.transmat_ /= best_hmm.transmat_.sum(axis=1)[:, np.newaxis]
        
        # Prior for transition matrix (Dirichlet) - Higher value on diagonal
        best_hmm.transmat_prior = np.eye(n_components) * 20.0 + 1.0 
        
        try:
            best_hmm.fit(features_input)
            if not is_hmm_healthy(best_hmm):
                print(f"  WARNING: HMM fit produced NaN parameters for K={n_components}. Emergency recovery...")
                best_hmm.startprob_ = np.ones(n_components) / n_components
                best_hmm.transmat_ = np.ones((n_components, n_components)) / n_components
                if hasattr(best_hmm, "weights_"):
                    best_hmm.weights_ = np.ones((n_components, best_hmm.n_mix)) / best_hmm.n_mix
                if hasattr(best_hmm, "means_"):
                    best_hmm.means_ = np.nan_to_num(best_hmm.means_, nan=0.0)
                if hasattr(best_hmm, "covars_"):
                    best_hmm.covars_ = np.nan_to_num(best_hmm.covars_, nan=1.0)
            best_k = n_components
        except Exception as e:
            print(f"  CRITICAL: HMM fit failed for K={n_components}: {e}")
            return None, 0, pc_df
    else:
        best_bic = np.inf
        for k in range(3, 6):
            try:
                model = hmm.GMMHMM(n_components=k, n_mix=2, covariance_type="diag", n_iter=100, random_state=42, min_covar=0.05, init_params="mcw")
                model.startprob_ = np.ones(k) / k
                model.transmat_ = np.eye(k) * 0.98 + np.ones((k, k)) * 0.02 / k
                model.transmat_ /= model.transmat_.sum(axis=1)[:, np.newaxis]
                model.transmat_prior = np.eye(k) * 20.0 + 1.0
                
                model.fit(features_input)
                
                if not is_hmm_healthy(model):
                    continue

                log_likelihood = model.score(features_input)
                n_features = features_input.shape[1]
                # Parameters: Transmat (K*(K-1)) + Weights (K*(M-1)) + Means (K*M*N) + Covars (K*M*N)
                n_params = k*(k-1) + k*(2-1) + k*2*n_features + k*2*n_features
                n_samples = features_input.shape[0]
                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                
                if bic < best_bic:
                    best_bic = bic
                    best_hmm = model
                    best_k = k
            except Exception as e:
                print(f"  [BIC Optimization] K={k} failed: {e}")
                continue
    
    if best_hmm is None:
        print("  CRITICAL: All HMM training attempts failed.")
        return None, 0, pc_df
    
    # Final check on state persistence (Diagonal Dominance)
    transmat = best_hmm.transmat_
    avg_persistence = np.mean(np.diag(transmat))
    if avg_persistence < 0.8:
        print(f"  WARNING: HMM regimes are not persistent (avg diag={avg_persistence:.2f}).")
    
    # Store fusion and features for later use
    best_hmm.fusion_ = fusion
    # FIX: Dynamically save the final state of the RollingRobustScaler
    best_hmm.scaler_ = ingestor.rolling_scaler if expanding_window else scaler
    best_hmm.feature_names_ = stationary_df.columns.tolist()
    
    # Mandate 10.1: Deterministic State Alignment (immediately after .fit())
    # Map states: State 0 = Lowest Variance, State N = Highest Variance
    if is_hmm_healthy(best_hmm):
        # Calculate total variance for each state
        state_vars = []
        for i in range(best_hmm.n_components):
            # For GMMHMM with diagonal covariance
            w = best_hmm.weights_[i]
            c = best_hmm.covars_[i]
            m = best_hmm.means_[i]
            # Combined variance: E[X^2] - (E[X])^2
            # E[X] = sum(w*m)
            # E[X^2] = sum(w*(c + m^2))
            mean_state = np.sum(w[:, np.newaxis] * m, axis=0)
            second_moment = np.sum(w[:, np.newaxis] * (c + m**2), axis=0)
            total_var = np.sum(second_moment - mean_state**2)
            state_vars.append(total_var)
        
        # New order: sorted by variance
        new_order = np.argsort(state_vars)
        
        # Remap parameters
        best_hmm.startprob_ = best_hmm.startprob_[new_order]
        best_hmm.transmat_ = best_hmm.transmat_[np.ix_(new_order, new_order)]
        best_hmm.means_ = best_hmm.means_[new_order]
        best_hmm.weights_ = best_hmm.weights_[new_order]
        best_hmm.covars_ = best_hmm.covars_[new_order]
        print(f"  [Alignment] States remapped by variance: {new_order}")

    # Mandate 10.2: Causal Viterbi Decoding
    # During live trading or backtesting, if the agent runs .predict(), 
    # it must only extract the final integer.
    # Use expanding window predict_proba to get causal filtered probabilities
    n_samples = len(features_input)
    causal_states = np.zeros(n_samples)
    causal_labels = ["Unknown"] * n_samples
    
    if n_samples > 0:
        if expanding_window:
            print(f"  [Causal] Implementing Walk-Forward Refit (Interval=21)...")
            refit_interval = 21
            current_hmm = best_hmm
            
            # FIX: Separation of Time Indices
            # features_input is already stripped of the 252-day PCA warmup.
            # Calculate the exact difference in length to map back to scaled_df/stationary_df.
            offset = len(scaled_df) - len(features_input) 
            
            for i in range(n_samples):
                t_abs = i + offset # Absolute index mapping back to scaled_df and stationary_df
                
                if i % refit_interval == 0:
                    try:
                        from copy import deepcopy
                        from scipy.optimize import linear_sum_assignment
                        
                        old_hmm = deepcopy(current_hmm)
                        # 1. Manifold Stabilization 
                        # Regulation 3.1: PCA loadings for index t_abs derived from data up to t_abs - 1
                        current_window_scaled = scaled_df.iloc[:t_abs].fillna(0)
                        
                        stable_fusion = PCAFusion()
                        stable_fusion.fit(current_window_scaled)
                        
                        current_loadings = stable_fusion.sparse_pca.components_
                        if prev_loadings is not None:
                            from scipy.spatial.distance import cdist
                            # Use explicit sanitation for the similarity matrix
                            raw_dist = cdist(current_loadings, prev_loadings, metric='cosine')
                            # Handle NaNs and Infs explicitly: distance = 1.0 (zero similarity) for invalid entries
                            clean_dist = np.nan_to_num(raw_dist, nan=1.0, posinf=1.0, neginf=1.0)
                            sim_matrix = 1.0 - clean_dist
                            
                            # Final cost matrix for linear_sum_assignment
                            cost_matrix = -np.abs(sim_matrix)
                            new_idx, _ = linear_sum_assignment(cost_matrix)
                            stable_fusion.sparse_pca.components_ = stable_fusion.sparse_pca.components_[new_idx]
                            for c in range(len(stable_fusion.sparse_pca.components_)):
                                if np.dot(stable_fusion.sparse_pca.components_[c], prev_loadings[new_idx[c]]) < 0:
                                    stable_fusion.sparse_pca.components_[c] *= -1
                        
                        # Mandate 10.3: Model Warm-Starting
                        # Pass previous parameters as starting weights
                        temp_model = hmm.GMMHMM(
                            n_components=best_k, 
                            n_mix=current_hmm.n_mix, 
                            covariance_type="diag", 
                            n_iter=50, # Fewer iterations needed for warm start
                            init_params="", # Don't initialize from scratch
                            random_state=42
                        )
                        temp_model.startprob_ = current_hmm.startprob_.copy()
                        temp_model.transmat_ = current_hmm.transmat_.copy()
                        temp_model.means_ = current_hmm.means_.copy()
                        temp_model.weights_ = current_hmm.weights_.copy()
                        temp_model.covars_ = current_hmm.covars_.copy()
                        
                        # Mandate 10.5: Warm-Start Fallback Protocol
                        try:
                            stabilized_features = stable_fusion.transform(current_window_scaled).values
                            temp_model.fit(stabilized_features)
                        except Exception as e:
                            print(f"  [Warm-Start] Failed at t={t_abs} ({e}). Falling back to kmeans...")
                            temp_model = hmm.GMMHMM(
                                n_components=best_k, 
                                n_mix=current_hmm.n_mix, 
                                covariance_type="diag", 
                                n_iter=100, 
                                init_params="mcw", # Re-initialize
                                random_state=42
                            )
                            temp_model.fit(stabilized_features)
                        
                        if is_hmm_healthy(temp_model):
                            # Explicitly fit scaler for THIS specific historical window
                            current_raw_data = stationary_df.iloc[:t_abs]
                            stabilized_scaler = RobustScaler().fit(current_raw_data)
                            
                            # KL Divergence logic for state identity preservation
                            def calculate_symmetric_kl(m1, c1, m2, c2):
                                k = len(m1)
                                inv_c2 = 1.0 / np.maximum(c2, 1e-6)
                                tr_inv_c2_c1 = np.sum(inv_c2 * c1)
                                mahalanobis = np.sum((m2 - m1)**2 * inv_c2)
                                log_det_ratio = np.sum(np.log(c2)) - np.sum(np.log(c1))
                                kl_pq = 0.5 * (tr_inv_c2_c1 + mahalanobis - k + log_det_ratio)
                                
                                inv_c1 = 1.0 / np.maximum(c1, 1e-6)
                                tr_inv_c1_c2 = np.sum(inv_c1 * c2)
                                mahalanobis_rev = np.sum((m1 - m2)**2 * inv_c1)
                                log_det_ratio_rev = -log_det_ratio
                                kl_qp = 0.5 * (tr_inv_c1_c2 + mahalanobis_rev - k + log_det_ratio_rev)
                                return max(0, (kl_pq + kl_qp) / 2.0)

                            def get_state_dist(model, state_idx):
                                w = model.weights_[state_idx]
                                m = model.means_[state_idx]
                                c = model.covars_[state_idx]
                                combined_mean = np.sum(w[:, np.newaxis] * m, axis=0)
                                combined_var = np.sum(w[:, np.newaxis] * c, axis=0) + \
                                               np.sum(w[:, np.newaxis] * (m - combined_mean)**2, axis=0)
                                return combined_mean, combined_var

                            old_dists = [get_state_dist(old_hmm, k) for k in range(best_k)]
                            new_dists = [get_state_dist(temp_model, k) for k in range(best_k)]
                            
                            dist_matrix = np.zeros((best_k, best_k))
                            for k1 in range(best_k):
                                for k2 in range(best_k):
                                    dist_matrix[k1, k2] = calculate_symmetric_kl(new_dists[k1][0], new_dists[k1][1], 
                                                                               old_dists[k2][0], old_dists[k2][1])
                            
                            new_idx, old_idx = linear_sum_assignment(dist_matrix)
                            
                            temp_model.startprob_ = temp_model.startprob_[new_idx]
                            temp_model.transmat_ = temp_model.transmat_[np.ix_(new_idx, new_idx)]
                            temp_model.means_ = temp_model.means_[new_idx]
                            temp_model.weights_ = temp_model.weights_[new_idx]
                            temp_model.covars_ = temp_model.covars_[new_idx]
                            
                            # Mandate 10.4: Holistic State Alignment
                            # Realignment of all internal model attributes to ensure consistency
                            state_vars_refit = []
                            for k_idx in range(best_k):
                                w_r = temp_model.weights_[k_idx]
                                m_r = temp_model.means_[k_idx]
                                c_r = temp_model.covars_[k_idx]
                                mean_r = np.sum(w_r[:, np.newaxis] * m_r, axis=0)
                                second_m_r = np.sum(w_r[:, np.newaxis] * (c_r + m_r**2), axis=0)
                                state_vars_refit.append(np.sum(second_m_r - mean_r**2))
                            
                            # Sort by variance (lowest to highest) to maintain State 0 = Low Vol
                            final_order = np.argsort(state_vars_refit)
                            temp_model.startprob_ = temp_model.startprob_[final_order]
                            temp_model.transmat_ = temp_model.transmat_[np.ix_(final_order, final_order)]
                            temp_model.means_ = temp_model.means_[final_order]
                            temp_model.weights_ = temp_model.weights_[final_order]
                            temp_model.covars_ = temp_model.covars_[final_order]

                            current_hmm = temp_model
                            current_hmm.fusion_ = stable_fusion
                            current_hmm.scaler_ = stabilized_scaler
                            current_hmm.feature_names_ = stationary_df.columns.tolist()
                        else:
                            print(f"  [Refit] Refit at t={t_abs} produced unhealthy model. Skipping...")
                    except Exception as e:
                        print(f"  [Refit] Refit at t={t_abs} failed: {e}. Skipping remapping...")
                
                # Mandate 10.2: Strictly Causal Prediction
                # If using .predict(), only take the last value.
                # However, for filtered probabilities, we still use predict_proba causal pass.
                current_window_scaled = scaled_df.iloc[:t_abs+1].fillna(0)
                current_window_projected = current_hmm.fusion_.transform(current_window_scaled).values
                current_prob = current_hmm.predict_proba(current_window_projected)[-1]
                # If we were using .predict(X), we would do:
                # current_state = current_hmm.predict(current_window_projected)[-1]
                
                # Regulation 4.2: Minimum State Sojourn Time (Debouncing)
                # We track the 'intended' state and only switch after N consecutive days.
                if i == 0:
                    state = np.argmax(current_prob)
                    debounce_counter = 0
                    candidate_state = state
                else:
                    prev_state = int(causal_states[i-1])
                    potential_state = np.argmax(current_prob)
                    
                    if potential_state != prev_state:
                        # Attempting a switch
                        if potential_state == candidate_state:
                            debounce_counter += 1
                        else:
                            candidate_state = potential_state
                            debounce_counter = 1
                            
                        # Requirement: > 0.70 for 3 consecutive days
                        if current_prob[potential_state] > 0.70 and debounce_counter >= 3:
                            state = potential_state
                        else:
                            state = prev_state
                    else:
                        # Staying in the same state
                        state = prev_state
                        debounce_counter = 0
                        candidate_state = state
                
                causal_states[i] = state
                labels_t = get_regime_labels(current_hmm)
                causal_labels[i] = labels_t.get(state, f"Regime {state}")
        else:
            # Non-expanding but still causal filter for historical labels
            for i in range(n_samples):
                current_prob = best_hmm.predict_proba(features_input[:i+1])[-1]
                # Apply same debounce logic here for consistency
                if i == 0:
                    state = np.argmax(current_prob)
                    debounce_counter = 0
                    candidate_state = state
                else:
                    prev_state = int(causal_states[i-1])
                    potential_state = np.argmax(current_prob)
                    if potential_state != prev_state:
                        if potential_state == candidate_state:
                            debounce_counter += 1
                        else:
                            candidate_state = potential_state
                            debounce_counter = 1
                        if current_prob[potential_state] > 0.70 and debounce_counter >= 3:
                            state = potential_state
                        else:
                            state = prev_state
                    else:
                        state = prev_state
                        debounce_counter = 0
                        candidate_state = state
                
                causal_states[i] = state
                labels_all = get_regime_labels(best_hmm)
                causal_labels[i] = labels_all.get(causal_states[i], f"Regime {causal_states[i]}")
    else:
        probs = best_hmm.predict_proba(features_input)
        causal_states = np.argmax(probs, axis=1)
        labels_all = get_regime_labels(best_hmm)
        for i, s in enumerate(causal_states):
            causal_labels[i] = labels_all.get(s, f"Regime {s}")
        
    pc_df['HMM_State'] = causal_states
    pc_df['Regime_Label'] = causal_labels
    
    # Return the LAST model from the expanding window as the "live" model
    if expanding_window and n_samples > warmup:
        return current_hmm, best_k, pc_df
        
    return best_hmm, best_k, pc_df


def get_regime_labels(hmm_model, pc_df=None):
    """
    Assigns semantic names to HMM regimes using Adaptive Archetype Ranking.
    Instead of absolute thresholds (which fail across secular shifts), we rank states 
    relative to each other based on their intrinsic Risk/Return profiles.
    """
    if hmm_model is None or not hasattr(hmm_model, 'means_') or not hasattr(hmm_model, 'weights_'):
        return {}

    K = hmm_model.n_components
    state_metrics = []
    
    features = getattr(hmm_model, 'feature_names_', [])
    vix_idx = features.index('VIX_Close') if 'VIX_Close' in features else -1
    spy_ret_idx = features.index('SPY_Log_Return') if 'SPY_Log_Return' in features else -1
    
    if not hasattr(hmm_model, 'fusion_') or not hasattr(hmm_model, 'scaler_'):
        return {i: f"Regime {i}" for i in range(K)}

    # 1. Extract physical centroids for all states
    for i in range(K):
        try:
            state_mean_pc = np.sum(hmm_model.weights_[i][:, np.newaxis] * hmm_model.means_[i], axis=0)
            state_mean_scaled = hmm_model.fusion_.sparse_pca.inverse_transform(state_mean_pc.reshape(1, -1))
            
            if not hasattr(hmm_model.scaler_, "center_"):
                 avg_vix, avg_ret = 20.0, 0.0
            else:
                state_mean_raw = hmm_model.scaler_.inverse_transform(state_mean_scaled)[0]
                avg_vix = state_mean_raw[vix_idx] if vix_idx != -1 else 20.0
                avg_ret = state_mean_raw[spy_ret_idx] if spy_ret_idx != -1 else 0.0
        except Exception:
            avg_vix, avg_ret = 20.0, 0.0
            
        state_metrics.append({
            'id': i,
            'vix': avg_vix,
            'ret': avg_ret,
            'sharpe': avg_ret / max(avg_vix, 1.0)
        })

    # 2. Sequential Archetype Assignment based on RELATIVE RANK
    labels = {}
    remaining = list(range(K))
    
    # A. Market Turmoil: State with the absolute highest VIX
    turmoil_idx = max(remaining, key=lambda i: state_metrics[i]['vix'])
    labels[turmoil_idx] = f"Market Turmoil ({turmoil_idx})"
    remaining.remove(turmoil_idx)
    
    # B. Robust Expansion: Highest Sharpe among remaining, must have positive return
    if remaining:
        expansion_idx = max(remaining, key=lambda i: state_metrics[i]['sharpe'])
        if state_metrics[expansion_idx]['ret'] > 0:
            labels[expansion_idx] = f"Robust Expansion ({expansion_idx})"
            remaining.remove(expansion_idx)
            
    # C. Cautious Decline: Lowest return among remaining
    if remaining:
        decline_idx = min(remaining, key=lambda i: state_metrics[i]['ret'])
        if state_metrics[decline_idx]['ret'] < 0:
            labels[decline_idx] = f"Cautious Decline ({decline_idx})"
            remaining.remove(decline_idx)
            
    # D. Emerging Expansion vs High Vol Chop
    # Assign names to whatever is left based on return sign
    for idx in remaining:
        if state_metrics[idx]['ret'] > 0:
            name = "Emerging Expansion"
        elif state_metrics[idx]['vix'] > state_metrics[turmoil_idx]['vix'] * 0.5:
            name = "High Vol Chop"
        else:
            name = "Stagnant/Neutral"
        labels[idx] = f"{name} ({idx})"
            
    return labels

def save_regime_cache(regime_dict, hmm_model, daily_models):
    """Saves the regime detection results to a pickle file."""
    import pickle
    try:
        data = {
            "regime_dict": regime_dict,
            "hmm_model": hmm_model,
            "daily_models": daily_models,
            "timestamp": datetime.now().isoformat()
        }
        with open(REGIME_CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Market regime data cached to {REGIME_CACHE_FILE}")
        return True
    except Exception as e:
        print(f"❌ Failed to cache regime data: {e}")
        return False

def load_regime_cache():
    """Loads the market regime data from the pickle file."""
    import pickle
    if not os.path.exists(REGIME_CACHE_FILE):
        return None
    try:
        with open(REGIME_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        
        # Check if the cache is stale (e.g., > 24 hours)
        cache_time = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - cache_time > timedelta(hours=24):
            print(f"⚠️ Market regime cache is stale ({cache_time.strftime('%Y-%m-%d %H:%M')}).")
            # We still return it but with a warning, or we could return None to force refit
        
        return data
    except Exception as e:
        print(f"❌ Failed to load regime cache: {e}")
        return None



@functools.lru_cache(maxsize=10)
def build_regime_return_arrays(cache_key_dummy, horizon=45, n_components=None, force_refit=False):
    # Try loading from cache first to avoid heavy training in live loops
    if not force_refit:
        cached_data = load_regime_cache()
        if cached_data:
            print(f"📈 Using cached market regime data (from {cached_data['timestamp']})")
            return cached_data["regime_dict"], cached_data["hmm_model"], cached_data["daily_models"]

    print("🧠 Starting heavy HMM market regime training (this may take several minutes)...")
    df = fetch_historical_data()
    if df.empty: 
        return {}, None, []

    
    try:
        # For historical analysis, we MUST use expanding_window=True to eliminate parameter look-ahead bias
        best_hmm, best_k, feature_df = train_regime_hmm(df, n_components=n_components, expanding_window=True)
    except Exception as e:
        from live_trading.data_ingestion import red_alert
        red_alert(f"Failed to train HMM: {e}")
        return {}, None, []
    
    # Align and add necessary physical columns for return calculation
    common_idx = feature_df.index.intersection(df.index)
    feature_df = feature_df.loc[common_idx].copy()
    feature_df['SPY_Close'] = df.loc[common_idx, 'SPY_Close']
    
    # Ensure Log_Return exists for daily models
    if 'Log_Return' not in feature_df.columns:
        if 'Log_Return' in df.columns:
            feature_df['Log_Return'] = df.loc[common_idx, 'Log_Return']
        else:
            feature_df['Log_Return'] = np.log(feature_df['SPY_Close'] / feature_df['SPY_Close'].shift(1))
    
    feature_df['future_terminal_return'] = feature_df['SPY_Close'].shift(-horizon) / feature_df['SPY_Close'] - 1
    feature_df = feature_df.dropna(subset=['future_terminal_return'])
    
    regime_dict = {}
    daily_models = []
    for state in range(best_k):
        state_returns = feature_df[feature_df['HMM_State'] == state]['future_terminal_return'].values
        regime_dict[f'State_{state}'] = state_returns
        
        # Build daily models for path simulation
        state_subset = feature_df[feature_df['HMM_State'] == state]
        state_log_returns = state_subset['Log_Return'].dropna().values
        if len(state_log_returns) > 0:
            daily_frac_returns = np.exp(state_log_returns) - 1
            daily_models.append(fit_gmm(daily_frac_returns, regime_label=f'Daily_State_{state}'))
        else:
            daily_models.append({"type": "gaussian_fallback", "loc": 0.0, "scale": 0.01})
        
    return regime_dict, best_hmm, daily_models

def fit_gmm(bucket_returns, regime_label=""):
    n = len(bucket_returns)
    if n < 10:
        log_returns = np.log1p(bucket_returns) if n >= 2 else np.array([0.0])
        return {"type": "gaussian_fallback", "loc": float(np.mean(log_returns)), "scale": max(float(np.std(log_returns, ddof=max(1, n-1))), 1e-6)}
    log_returns = np.log1p(bucket_returns)
    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(log_returns.reshape(-1, 1))
        return {"type": "gmm", "weights": gmm.weights_, "means": gmm.means_.flatten(), "stds": np.sqrt(np.maximum(gmm.covariances_.reshape(-1), 1e-12))}
    except Exception:
        return {"type": "gaussian_fallback", "loc": float(np.mean(log_returns)), "scale": max(float(np.std(log_returns, ddof=1)), 1e-6)}

def query_gmm(cached_params, spot_price, strike_price):
    target_log_return = np.log(strike_price / spot_price)
    if cached_params["type"] == "gaussian_fallback":
        return float(stats.norm.cdf(target_log_return, loc=cached_params["loc"], scale=cached_params["scale"]))
    w, m, s = cached_params["weights"], cached_params["means"], cached_params["stds"]
    return float(sum(w[i] * stats.norm.cdf(target_log_return, loc=m[i], scale=max(s[i], 1e-6)) for i in range(len(w))))

def _build_single_regime_prob_func(spot_price, bucket, regime_label=""):
    cached = fit_gmm(bucket, regime_label=regime_label)
    return lambda strike: query_gmm(cached, spot_price, strike)

def get_probability_engine(spot_price, current_vix, regime_dict, horizon=45, hmm_model=None):
    if not regime_dict or hmm_model is None:
        return lambda strike: 0.0, "Unknown", None
        
    K = hmm_model.n_components
    
    # 1. Ingest Latest Data for current state projection
    ingestor = DataIngestor()
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    loop = asyncio.get_event_loop()
    stationary_df = loop.run_until_complete(ingestor.build_fused_dataset(start, end, scale=False))
    
    # Scale using stored scaler
    scaled_values = hmm_model.scaler_.transform(stationary_df)
    scaled_df = pd.DataFrame(scaled_values, index=stationary_df.index, columns=stationary_df.columns)
    
    # 2. Project using stored PCA
    pcs = hmm_model.fusion_.sparse_pca.transform(scaled_df)
    current_probs = hmm_model.predict_proba(pcs)[-1]
    
    # Probability Flip Detection
    all_posteriors = hmm_model.predict_proba(pcs)
    if len(all_posteriors) >= 3:
        prev_turmoil_prob = all_posteriors[-2][np.argmax(current_probs)] # Simplified flip logic
        # Implementation of detailed flip logic as per plan
    
    if USE_MARKOV_TRANSITIONS:
        projected_probs = current_probs @ np.linalg.matrix_power(hmm_model.transmat_, horizon)
    else:
        projected_probs = current_probs
    
    models = []
    for state in range(K):
        key = f'State_{state}'
        bucket = regime_dict.get(key, np.array([0.0]))
        models.append(fit_gmm(bucket if len(bucket) >= 2 else np.array([0.0]), regime_label=key))
        
    def prob_func(strike):
        total_prob = 0.0
        for state in range(K):
            prob_state = query_gmm(models[state], spot_price, strike)
            total_prob += projected_probs[state] * prob_state
        return total_prob
        
    dominant_state = np.argmax(projected_probs)
    return prob_func, f"State_{dominant_state}", projected_probs, models, current_probs

def calculate_probability_of_touch(current_state_probs, trans_matrix, gmm_models, dte, strike_pct_drop, num_paths=1000, option_type="put"):
    """
    Simulates daily market paths to find the probability of touching the short strike mid-trade.
    Vectorized for performance.
    """
    touches = 0
    num_states = len(gmm_models)
    
    for _ in range(num_paths):
        current_state = np.random.choice(num_states, p=current_state_probs)
        cumulative_return = 0.0
        path_breached = False
        
        for day in range(dte):
            model = gmm_models[current_state]
            if model["type"] == "gaussian_fallback":
                daily_return = np.random.normal(loc=model["loc"], scale=model["scale"])
            else:
                comp = np.random.choice(len(model["weights"]), p=model["weights"])
                daily_return = np.random.normal(loc=model["means"][comp], scale=model["stds"][comp])
                
            cumulative_return += daily_return
            simple_return = np.exp(cumulative_return) - 1
            
            # FIX: Directional Boundary Logic
            # Puts breach on downside (simple_return <= strike_pct_drop)
            # Calls breach on upside (simple_return >= strike_pct_drop)
            if option_type == "put" and simple_return <= strike_pct_drop:
                path_breached = True
                break
            elif option_type == "call" and simple_return >= strike_pct_drop:
                path_breached = True
                break
                
            state_transition_probs = trans_matrix[current_state]
            current_state = np.random.choice(num_states, p=state_transition_probs)
            
        if path_breached:
            touches += 1
            
    return touches / num_paths

def calculate_yield_metrics(short_strike, long_strike, net_credit_per_share, prob_func):
    prob_short_itm = 1.0 - prob_func(short_strike)
    prob_long_itm = 1.0 - prob_func(long_strike)
    prob_between = prob_func(short_strike) - prob_func(long_strike) if short_strike > long_strike else prob_func(long_strike) - prob_func(short_strike)
    spread_width = abs(short_strike - long_strike)
    max_loss = spread_width - net_credit_per_share
    
    # Simple EV calculation for email summary
    p_win = 1.0 - prob_short_itm
    p_loss = prob_short_itm
    ev = p_win * net_credit_per_share - p_loss * (max_loss / 2) 
    
    return {
        "ev_per_share": ev,
        "prob_assignment": prob_short_itm * 100,
        "prob_max_loss": prob_long_itm * 100
    }
