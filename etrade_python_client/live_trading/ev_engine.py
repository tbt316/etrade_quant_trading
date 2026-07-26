import os
import json
import time
import functools
import hashlib
import re
import multiprocessing as mp
from copy import deepcopy
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import logging
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from live_trading.data_ingestion import DataIngestor, RollingRobustScaler
from live_trading.pca_fusion import PCAFusion
from datetime import datetime, timedelta
import asyncio
from sklearn.preprocessing import RobustScaler
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def is_hmm_healthy(model):
    """Check if the HMM model has valid (non-NaN, non-Inf) parameters and
    valid probability simplex constraints."""
    if model is None: return False
    try:
        if np.any(np.isnan(model.startprob_)) or np.any(np.isinf(model.startprob_)): return False
        if not np.allclose(np.sum(model.startprob_), 1.0, atol=1e-4): return False
        if np.any(np.isnan(model.transmat_)) or np.any(np.isinf(model.transmat_)): return False
        if not np.allclose(np.sum(model.transmat_, axis=1), 1.0, atol=1e-4): return False
        
        # For GMMHMM, check weights, means and covars
        if hasattr(model, "weights_"):
            if np.any(np.isnan(model.weights_)) or np.any(np.isinf(model.weights_)): return False
            # Check row sums for weights (must sum to 1 per state)
            if not np.allclose(np.sum(model.weights_, axis=1), 1.0): return False
            
        if hasattr(model, "means_"):
            if np.any(np.isnan(model.means_)) or np.any(np.isinf(model.means_)): return False
            
        if hasattr(model, "covars_"):
            cov = model.covars_
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)): return False
            diags = np.diagonal(cov, axis1=1, axis2=2) if cov.ndim == 3 else cov
            if np.any(diags <= 0): return False
            
    except Exception:
        return False
    return True


def validate_hmm_quality(model, features=None,
                         min_occupancy=0.03,
                         max_var_ratio=1e6,
                         min_kl_separation=0.01):
    """Statistical quality assessment that catches degenerate models.

    Returns (is_valid, diagnostics_dict).  Unlike is_hmm_healthy() which
    only checks numerical validity, this checks whether the model actually
    learned multiple distinguishable states.
    """
    diag = {"passed": False, "reason": ""}
    if not is_hmm_healthy(model):
        diag["reason"] = "model fails basic health check"
        return False, diag

    K = model.n_components
    if K <= 1:
        diag["passed"] = True
        diag["reason"] = "K=1; no multi-state checks needed"
        return True, diag

    # --- 1. State occupancy (requires features) ---
    if features is not None:
        try:
            feats = np.asarray(features, dtype=float)
            if feats.ndim == 2 and len(feats) >= K:
                probs = model.predict_proba(feats)
                occupancy = probs.mean(axis=0)
                diag["state_occupancy"] = occupancy.tolist()
                if np.any(occupancy < min_occupancy):
                    diag["reason"] = (
                        f"state occupancy below {min_occupancy:.0%}: "
                        f"{occupancy.tolist()}"
                    )
                    return False, diag
                # Posterior entropy (bits)
                ent_per_row = -np.sum(
                    probs * np.log2(np.maximum(probs, 1e-12)), axis=1
                )
                diag["mean_posterior_entropy_bits"] = float(np.mean(ent_per_row))
                # If entropy is near zero across all rows, model is single-state
                if np.mean(ent_per_row) < 0.01:
                    diag["reason"] = (
                        f"posterior entropy near zero ({np.mean(ent_per_row):.4f} bits) "
                        "— model assigns all observations to one state"
                    )
                    return False, diag
        except Exception as e:
            logger.warning(f"validate_hmm_quality: occupancy check skipped ({e})")

    # --- 2. Pairwise emission separation (KL divergence) ---
    state_vars = []
    for i in range(K):
        _, var_i = _state_emission_mean_var(model, i)
        state_vars.append(float(np.sum(var_i)))

    diag["state_variances"] = state_vars

    min_kl = np.inf
    for i in range(K):
        mean_i, var_i = _state_emission_mean_var(model, i)
        for j in range(i + 1, K):
            mean_j, var_j = _state_emission_mean_var(model, j)
            kl = _symmetric_diag_gaussian_kl(mean_i, var_i, mean_j, var_j)
            if kl < min_kl:
                min_kl = kl
    diag["min_pairwise_kl"] = float(min_kl) if np.isfinite(min_kl) else 0.0
    if min_kl < min_kl_separation:
        diag["reason"] = (
            f"pairwise KL separation too low ({min_kl:.6f} < {min_kl_separation})"
        )
        return False, diag

    # --- 3. Covariance ratio ---
    if len(state_vars) >= 2:
        sorted_vars = sorted(v for v in state_vars if v > 0)
        if len(sorted_vars) >= 2:
            ratio = sorted_vars[-1] / max(sorted_vars[0], 1e-15)
            diag["covariance_ratio"] = float(ratio)
            if ratio > max_var_ratio:
                diag["reason"] = (
                    f"covariance ratio {ratio:.0f} exceeds limit {max_var_ratio:.0f}"
                )
                return False, diag

    diag["passed"] = True
    diag["reason"] = "all quality checks passed"
    return True, diag

def _state_emission_mean_var(model, state_idx):
    if hasattr(model, "weights_"):
        w = model.weights_[state_idx]
        c = model.covars_[state_idx]
        m = model.means_[state_idx]
        mean_state = np.sum(w[:, np.newaxis] * m, axis=0)
        second_moment = np.sum(w[:, np.newaxis] * (c + m**2), axis=0)
        var_state = second_moment - mean_state**2
        return mean_state, var_state

    mean_state = np.asarray(model.means_[state_idx], dtype=float)
    covar = np.asarray(model.covars_[state_idx], dtype=float)
    if covar.ndim == 2:
        var_state = np.diag(covar)
    else:
        var_state = covar
    return mean_state, var_state

def _reorder_hmm_states(model, order):
    """Apply one state permutation to all HMM parameters that carry state rows."""
    order = np.asarray(order, dtype=int)
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[np.ix_(order, order)]
    model.means_ = model.means_[order]
    if hasattr(model, "weights_"):
        model.weights_ = model.weights_[order]
        model.covars_ = model.covars_[order]
    elif hasattr(model, "_covars_"):
        model._covars_ = model._covars_[order]
    else:
        model.covars_ = model.covars_[order]
    return model


def _state_variance_order(model):
    """Return state indices sorted from quietest to most volatile emission."""
    state_vars = []
    for i in range(model.n_components):
        _, var_state = _state_emission_mean_var(model, i)
        state_vars.append(np.sum(var_state))
    return np.argsort(state_vars)


def _symmetric_diag_gaussian_kl(mean_a, var_a, mean_b, var_b):
    """Symmetric KL divergence between diagonal Gaussian approximations."""
    eps = 1e-8
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    var_a = np.maximum(np.asarray(var_a, dtype=float), eps)
    var_b = np.maximum(np.asarray(var_b, dtype=float), eps)
    kl_ab = 0.5 * np.sum(np.log(var_b / var_a) + (var_a + (mean_a - mean_b) ** 2) / var_b - 1.0)
    kl_ba = 0.5 * np.sum(np.log(var_a / var_b) + (var_b + (mean_b - mean_a) ** 2) / var_a - 1.0)
    return float(0.5 * (kl_ab + kl_ba))


def _align_hmm_states_to_previous(model, previous_model):
    """Reorder model states to the closest prior-model emissions."""
    if (
        model is None
        or previous_model is None
        or not is_hmm_healthy(model)
        or not is_hmm_healthy(previous_model)
        or model.n_components != previous_model.n_components
    ):
        return model, None, np.nan

    distances = np.zeros((model.n_components, model.n_components), dtype=float)
    for new_state in range(model.n_components):
        new_mean, new_var = _state_emission_mean_var(model, new_state)
        for old_state in range(previous_model.n_components):
            old_mean, old_var = _state_emission_mean_var(previous_model, old_state)
            distances[new_state, old_state] = _symmetric_diag_gaussian_kl(new_mean, new_var, old_mean, old_var)

    new_idx, old_idx = linear_sum_assignment(distances)
    order = np.empty(model.n_components, dtype=int)
    for new_state, old_state in zip(new_idx, old_idx):
        order[old_state] = new_state
    model = _reorder_hmm_states(model, order)
    return model, order.tolist(), float(distances[new_idx, old_idx].mean())


def _align_hmm_states_by_variance(model):
    """Deterministically reorder states from lowest to highest emission variance."""
    if model is None or not is_hmm_healthy(model):
        return model, None

    order = _state_variance_order(model)
    model = _reorder_hmm_states(model, order)
    return model, order.tolist()


def _fallback_pca_components(n_components, n_features, previous_loadings=None):
    """Deterministic PCA fallback used when a causal window is degenerate."""
    if previous_loadings is not None:
        prev = np.asarray(previous_loadings, dtype=float)
        if prev.ndim == 2 and prev.shape == (n_components, n_features):
            return prev.copy()

    components = np.zeros((n_components, n_features), dtype=float)
    for i in range(n_components):
        components[i, i % n_features] = 1.0
    return components


def _project_with_components(window_df, row_df, components):
    """Project a row using explicit PCA components and the window mean."""
    window_vals = np.asarray(window_df, dtype=float)
    row_vals = np.asarray(row_df, dtype=float)
    if window_vals.ndim != 2 or row_vals.ndim != 2:
        raise ValueError("Expected 2D inputs for fallback projection.")
    centered = row_vals - np.nanmean(window_vals, axis=0, keepdims=True)
    return centered @ np.asarray(components, dtype=float).T


def _finalize_hmm_alignment(model, previous_model=None, method="fresh", fallback_reason=None):
    """Apply prior continuity and variance sorting, then attach diagnostics."""
    prior_order = None
    prior_distance = np.nan
    if previous_model is not None:
        model, prior_order, prior_distance = _align_hmm_states_to_previous(model, previous_model)
    model, variance_order = _align_hmm_states_by_variance(model)
    if model is not None:
        means = []
        variances = []
        for state in range(model.n_components):
            mean_state, var_state = _state_emission_mean_var(model, state)
            means.append(np.asarray(mean_state, dtype=float).tolist())
            variances.append(float(np.sum(var_state)))
        model.refit_metadata_ = {
            "method": method,
            "fallback_reason": fallback_reason,
            "prior_alignment_order": prior_order,
            "prior_alignment_mean_skl": prior_distance,
            "variance_sort_order": variance_order,
            "state_means": means,
            "state_variances": variances,
        }
    return model

def _causal_pca_worker(args):
    """Worker function for parallel causal PCA computation.
    Processes a chunk of time steps, using anchor_loadings for sign/rank consistency.
    Returns (indices, pc_values, final_loadings) for the chunk.
    """
    chunk_indices, scaled_values, scaled_columns, anchor_loadings, n_components, alpha = args
    from live_trading.pca_fusion import PCAFusion
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    n_features = scaled_values.shape[1]
    pc_results = np.full((len(chunk_indices), n_components), np.nan)
    prev_loadings = anchor_loadings  # Start from the anchor

    for local_i, t in enumerate(chunk_indices):
        # Regulation 3.1: PCA loadings at time t derived from data 0..t-1
        window_end = t - 1 if t > 20 else t
        window_data = scaled_values[:window_end]
        if len(window_data) < 20:
            window_data = scaled_values[:t]

        fusion = PCAFusion(n_components=n_components, alpha=alpha, use_sparse=False)
        window_df = pd.DataFrame(window_data, columns=scaled_columns)
        point = pd.DataFrame(
            scaled_values[t-1:t],
            columns=scaled_columns,
        )

        try:
            fusion.fit(window_df)
            current_loadings = fusion.sparse_pca.components_
            if prev_loadings is not None:
                sim_matrix = 1 - cdist(current_loadings, prev_loadings, metric='cosine')
                new_idx, old_idx = linear_sum_assignment(-np.abs(sim_matrix))
                fusion.sparse_pca.components_ = fusion.sparse_pca.components_[new_idx]
                current_loadings = fusion.sparse_pca.components_
                for i in range(len(current_loadings)):
                    if np.dot(current_loadings[i], prev_loadings[old_idx[i]]) < 0:
                        fusion.sparse_pca.components_[i] *= -1

            prev_loadings = fusion.sparse_pca.components_.copy()
            pc_results[local_i] = fusion.transform(point).values[0]
        except Exception:
            fallback_loadings = _fallback_pca_components(
                n_components,
                len(scaled_columns),
                previous_loadings=prev_loadings,
            )
            pc_results[local_i] = _project_with_components(window_df, point, fallback_loadings)[0]
            prev_loadings = fallback_loadings.copy()

    return chunk_indices, pc_results, prev_loadings


def _make_hmm(k, n_iter=100, init_params="mc", persistence=0.98, n_mix=1, model_class="gaussian"):
    if model_class == "gmm":
        model = hmm.GMMHMM(
            n_components=k,
            n_mix=n_mix,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42,
            min_covar=0.05,
            init_params=init_params or "",
        )
    else:
        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42,
            min_covar=0.05,
            init_params=init_params or "",
        )
    if init_params:
        model.startprob_ = np.ones(k) / k
        model.transmat_ = np.eye(k) * persistence + np.ones((k, k)) * (1.0 - persistence) / k
        model.transmat_ /= model.transmat_.sum(axis=1)[:, np.newaxis]
    model.transmat_prior = np.eye(k) * 20.0 + 1.0
    return model

def _sanitize_hmm_params(model):
    """Last-resort cleanup when hmmlearn returns numerically ugly parameters."""
    if model is None:
        return None
    try:
        if hasattr(model, "startprob_"):
            model.startprob_ = np.nan_to_num(model.startprob_, nan=1.0 / model.n_components)
            s = model.startprob_.sum()
            model.startprob_ = model.startprob_ / s if s > 0 else np.ones(model.n_components) / model.n_components

        if hasattr(model, "transmat_"):
            model.transmat_ = np.nan_to_num(model.transmat_, nan=0.0, posinf=0.0, neginf=0.0)
            row_sums = model.transmat_.sum(axis=1, keepdims=True)
            bad_rows = row_sums.squeeze() <= 0
            if np.any(bad_rows):
                model.transmat_[bad_rows] = 1.0 / model.n_components
                row_sums = model.transmat_.sum(axis=1, keepdims=True)
            model.transmat_ = model.transmat_ / row_sums

        if hasattr(model, "weights_"):
            model.weights_ = np.nan_to_num(model.weights_, nan=1.0)
            weight_sums = model.weights_.sum(axis=1, keepdims=True)
            bad_rows = weight_sums.squeeze() <= 0
            if np.any(bad_rows):
                model.weights_[bad_rows] = 1.0
                weight_sums = model.weights_.sum(axis=1, keepdims=True)
            model.weights_ = model.weights_ / weight_sums

        if hasattr(model, "means_"):
            model.means_ = np.nan_to_num(model.means_, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(model, "covars_"):
            covars = np.nan_to_num(model.covars_, nan=1.0, posinf=1.0, neginf=1.0)
            covars = np.maximum(covars, 1e-3)
            if hasattr(model, "weights_"):
                model.covars_ = covars
            elif hasattr(model, "_covars_"):
                model._covars_ = np.diagonal(covars, axis1=1, axis2=2) if covars.ndim == 3 else covars
            else:
                model.covars_ = covars
    except Exception:
        return None
    return model if is_hmm_healthy(model) else None

def _fit_hmm_model(features, n_components=None, previous_model=None):
    """Fit an HMM on the provided historical feature block only."""
    features = np.asarray(features, dtype=float)
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError("Input features to HMM fit contain NaN or Inf values")
    if len(features) < 20:
        raise ValueError(f"Need at least 20 feature rows for HMM fit, got {len(features)}")
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("HMM fit requires a 2D feature matrix.")
    if np.nanstd(features) < 1e-10 or np.unique(features, axis=0).shape[0] < max(5, min(20, len(features) // 4)):
        raise ValueError("HMM feature block is too low-variance or too repetitive for stable fit.")

    warm_start_error = None
    if previous_model is not None:
        k = previous_model.n_components
        try:
            previous_is_gmm = hasattr(previous_model, "weights_")
            model = _make_hmm(
                k,
                n_iter=50,
                init_params="",
                n_mix=getattr(previous_model, "n_mix", 1),
                model_class="gmm" if previous_is_gmm else "gaussian",
            )
            model.startprob_ = previous_model.startprob_.copy()
            model.transmat_ = previous_model.transmat_.copy()
            model.means_ = previous_model.means_.copy()
            if previous_is_gmm:
                model.weights_ = previous_model.weights_.copy()
                model.covars_ = previous_model.covars_.copy()
            elif hasattr(previous_model, "_covars_"):
                model._covars_ = previous_model._covars_.copy()
            else:
                prev_covars = np.asarray(previous_model.covars_)
                model.covars_ = np.diagonal(prev_covars, axis1=1, axis2=2) if prev_covars.ndim == 3 else prev_covars.copy()
            model.fit(features)
            is_valid, diag = validate_hmm_quality(model, features=features)
            if is_valid:
                return _finalize_hmm_alignment(model, previous_model, method="warm_start"), k
            model = _sanitize_hmm_params(model)
            if model is not None:
                is_valid_sanitized, diag_sanitized = validate_hmm_quality(model, features=features)
                if is_valid_sanitized:
                    return _finalize_hmm_alignment(model, previous_model, method="warm_start_sanitized"), k
            warm_start_error = f"warm-start quality check failed: {diag.get('reason', '')}"
        except Exception as e:
            warm_start_error = str(e)
            print(f"  [Warm-Start] Failed ({e}). Falling back to kmeans...")

    if n_components is not None:
        last_error = None
        try:
            model = _make_hmm(n_components, persistence=0.95, model_class="gaussian")
            model.fit(features)
            if is_hmm_healthy(model):
                method = "fresh" if previous_model is None else "fallback_kmeans"
                return _finalize_hmm_alignment(
                    model,
                    previous_model,
                    method=method,
                    fallback_reason=warm_start_error,
                ), n_components
            model = _sanitize_hmm_params(model)
            if model is not None:
                method = "fresh_sanitized" if previous_model is None else "fallback_kmeans_sanitized"
                return _finalize_hmm_alignment(
                    model,
                    previous_model,
                    method=method,
                    fallback_reason=warm_start_error,
                ), n_components
        except Exception as e:
            last_error = e
        raise ValueError(f"HMM fit produced unhealthy parameters for K={n_components}: {last_error}")

    best_hmm = None
    best_k = 0
    best_bic = np.inf
    for k in range(1, 6):
        try:
            model = _make_hmm(k, model_class="gaussian")
            model.fit(features)
            if not is_hmm_healthy(model):
                model = _sanitize_hmm_params(model)
            if model is None:
                continue
            log_likelihood = model.score(features)
            n_features = features.shape[1]
            n_params = k * (k - 1) + 2 * k * n_features
            bic = -2 * log_likelihood + n_params * np.log(len(features))
            if bic < best_bic:
                best_bic = bic
                best_hmm = model
                best_k = k
        except Exception as e:
            print(f"  [BIC Optimization] K={k} failed: {e}")

    if best_hmm is None:
        raise ValueError("All HMM training attempts failed")
    return _finalize_hmm_alignment(best_hmm, previous_model, method="bic_fresh"), best_k

def _rolling_scaler_snapshot(raw_window, window=252 * 5):
    """Create a scaler snapshot from data available at one historical timestamp."""
    scaler = RollingRobustScaler(window=window)
    hist = np.asarray(raw_window.tail(window).values, dtype=float)
    scaler.history = hist
    scaler.center_ = np.median(hist, axis=0)
    q1 = np.percentile(hist, 25, axis=0)
    q3 = np.percentile(hist, 75, axis=0)
    scaler.scale_ = np.where((q3 - q1) == 0, 1.0, q3 - q1)
    return scaler


def _apply_causal_stress_overlay(feature_df, raw_df, base_state_count):
    """
    Add a causal, raw-market stress overlay on top of the unsupervised HMM state.

    HMMs are intentionally persistent and can under-react to abrupt drawdowns.
    These overlay columns preserve the raw HMM output while exposing a tradable
    detected regime that reacts to close-T SPY/VIX stress for next-session use.
    """
    if feature_df.empty or raw_df.empty:
        return feature_df

    required = {"SPY_Close", "VIX_Close"}
    if not required.issubset(raw_df.columns):
        return feature_df

    aligned = raw_df.reindex(feature_df.index)
    spy = aligned["SPY_Close"].astype(float)
    vix = aligned["VIX_Close"].astype(float)
    log_ret = np.log(spy / spy.shift(1))
    ret_5d = np.log(spy / spy.shift(5))
    drawdown_21d = spy / spy.rolling(21, min_periods=5).max() - 1.0

    panic_mask = (
        (vix >= 35.0)
        | ((vix >= 30.0) & (drawdown_21d <= -0.08))
        | (log_ret <= -0.045)
        | (ret_5d <= -0.075)
    )
    decline_mask = (
        ~panic_mask
        & (
            (vix >= 25.0)
            | (drawdown_21d <= -0.06)
            | (ret_5d <= -0.04)
        )
    )

    detected_state = pd.Series(0, index=feature_df.index, dtype=int)
    detected_label = pd.Series("Expansion (0)", index=feature_df.index, dtype=object)

    # Raw HMM archetype labels are relative to the current model. A low-vol,
    # rising market can still have a highest-VIX state named "Market Turmoil",
    # so final risk regimes require observable close-T stress confirmation.
    detected_state.loc[decline_mask] = 1
    detected_label.loc[decline_mask] = "Cautious Decline (1)"
    detected_state.loc[panic_mask] = 2
    detected_label.loc[panic_mask] = "Panic / Crisis (2)"

    feature_df["Detected_Regime_State"] = detected_state
    feature_df["Detected_Regime_Label"] = detected_label
    feature_df["Stress_Overlay"] = np.select(
        [panic_mask.fillna(False), decline_mask.fillna(False)],
        ["panic_crisis", "cautious_decline"],
        default="none",
    )
    for state in range(3):
        feature_df[f"detected_prob_state_{state}"] = (detected_state == state).astype(float)
    feature_df["Stress_21d_Drawdown"] = drawdown_21d
    feature_df["Stress_5d_Log_Return"] = ret_5d
    feature_df["Stress_1d_Log_Return"] = log_ret
    feature_df["Stress_VIX_Close"] = vix
    feature_df.attrs["stress_overlay"] = "close_T_for_next_session"
    return feature_df


def calendar_days_to_trading_days(calendar_days):
    """Approximate calendar-day option horizons on a trading-day index."""
    return max(1, int(round(float(calendar_days) * 252.0 / 365.0)))

# Constants moved from ev_plots.py
PROBABILITY_MODEL = 'gmm'  # Options: 'bootstrap', 'parametric', 'gmm'
USE_MARKOV_TRANSITIONS = False
COST_PER_SPREAD = 1.0
YF_QUOTE_CACHE_PATH = os.path.join(PROJECT_ROOT, "s_and_p_data", "yf_quote_cache.json")
REGIME_CACHE_FILE = os.path.join(PROJECT_ROOT, "market_regime_results.pkl")
REGIME_SNAPSHOT_CACHE_DIR = os.path.join(PROJECT_ROOT, "backtest_cache", "regime_snapshots")
REGIME_SNAPSHOT_PIPELINE_VERSION = 1
REGIME_HMM_REFIT_INTERVAL_DAYS = 20
GMM_MIN_OBS_PER_COMPONENT = 15
GMM_MAX_COMPONENTS = 5


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
    """Fetches high-dimensional historical data using DataIngestor with caching and synchronous sync."""
    ingestor = DataIngestor()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # DataIngestor now handles caching and synchronous sync when wait=True
    df = ingestor.fetch_yf_data(start_str, end_str, wait=True)
    
    if df.empty:
        from live_trading.data_ingestion import red_alert
        red_alert("Historical data fetch FAILED. Regime detection will be unavailable.")
        return pd.DataFrame()
        
    # Ensure we have the minimum required columns for HMM training
    required = ['SPY_Close', 'VIX_Close']
    if not all(c in df.columns for c in required):
        from live_trading.data_ingestion import red_alert
        red_alert(f"Critical columns {required} missing even after sync attempt.")
        return pd.DataFrame()

    print(f"  Loaded {len(df)} days of historical data.")
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

def train_regime_hmm(df, n_components=3, expanding_window=False, exclude_features=None, pca_components=None):
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
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        # If the loop is already running (e.g. from an async test),
        # run the coroutine in a worker thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    stationary_df = run_async(ingestor.build_fused_dataset(start_date, end_date, scale=False))
    
    # ENSURE WE ONLY USE DATA UP TO THE PROVIDED DF'S END DATE (Double check)
    stationary_df = stationary_df[stationary_df.index <= df.index.max()]
    
    if exclude_features:
        cols_to_drop = [c for c in exclude_features if c in stationary_df.columns]
        if cols_to_drop:
            stationary_df = stationary_df.drop(columns=cols_to_drop)
            print(f"  [Feature Selection] Dropped features: {cols_to_drop}")

    # Preserve raw VIX for overlay logic, but do not feed it into the HMM.
    model_df = stationary_df.drop(columns=['VIX_Close'], errors='ignore')
    if 'VIX_Close' in stationary_df.columns:
        print("  [Feature Selection] Dropped VIX_Close from HMM emissions; retained for overlays.")

    # 2. CAUSAL FEATURE PREPARATION (Strictly on past data if expanding_window=True)
    if expanding_window:
        print(f"  [Step 1/4] Rolling robust scaling on {len(model_df)} rows...", flush=True)
        _t0_scale = time.time()
        # Use rolling scale from DataIngestor
        scaled_df = ingestor.scale_features(model_df, rolling=True, window=min(252*5, len(model_df)-1))
        
        # Drop leading NaNs from rolling robust scaling to prevent zero-padding variance-collapse
        scaled_df = scaled_df.dropna()
        assert not scaled_df.isna().any().any(), "scaled_df contains NaNs after dropna safeguard"
        model_df = model_df.loc[scaled_df.index]
        stationary_df = stationary_df.loc[scaled_df.index]
        
        print(f"  [Step 1/4] Scaling done in {time.time()-_t0_scale:.1f}s.", flush=True)

        # Auto-select PCA components for ≥85% variance retention
        if pca_components is None:
            from sklearn.decomposition import PCA as _PCA
            _clean = scaled_df.dropna()
            if len(_clean) > 20:
                _auto_pca = _PCA()
                _auto_pca.fit(_clean.values)
                _cumvar = np.cumsum(_auto_pca.explained_variance_ratio_)
                pca_components = max(2, int(np.argmax(_cumvar >= 0.85) + 1))
                print(f"  [PCA] Auto-selected {pca_components} components "
                      f"(retains {_cumvar[pca_components-1]:.1%} variance, target ≥85%)", flush=True)
            else:
                pca_components = 2
                print(f"  [PCA] Insufficient data for auto-selection, using {pca_components} components", flush=True)
        
        # Causal PCA: For each point t, we need the PCA projection based on data up to t
        pc_values = np.full((len(scaled_df), pca_components), np.nan)
        fusion = PCAFusion(n_components=pca_components, use_sparse=False)
        
        warmup = min(252, len(scaled_df)-1)
        prev_loadings = None
        total_pca_steps = len(scaled_df) + 1 - warmup
        all_t_indices = list(range(warmup, len(scaled_df) + 1))
        
        n_workers = min(mp.cpu_count(), 8)
        # Shared numpy array for multiprocessing (converted from DataFrame)
        scaled_values = scaled_df.values
        scaled_columns = scaled_df.columns.tolist()
        
        print(f"  [Step 2/4] Causal PCA projection: {total_pca_steps} steps (warmup={warmup}), "
              f"using {n_workers} parallel workers...", flush=True)
        _t0_pca = time.time()
        
        # === PARALLEL CHUNKED CAUSAL PCA ===
        # Strategy: Divide time steps into N chunks. For each chunk, compute an
        # "anchor" loading at the chunk boundary (serially, fast) then parallelize
        # all steps within the chunk using that anchor for sign/rank consistency.
        
        # Step A: Compute anchor loadings at chunk boundaries (serial, ~N fits)
        chunk_size = max(50, total_pca_steps // n_workers)
        chunks = []
        for start in range(0, len(all_t_indices), chunk_size):
            end = min(start + chunk_size, len(all_t_indices))
            chunks.append(all_t_indices[start:end])
        
        print(f"    Split into {len(chunks)} chunks (avg {chunk_size} steps each)", flush=True)
        
        # Compute anchor loadings at each chunk boundary serially
        anchor_loadings_list = [None]  # First chunk has no anchor
        for chunk_idx in range(1, len(chunks)):
            boundary_t = chunks[chunk_idx][0]
            window_data = scaled_df.iloc[:boundary_t-1]
            if len(window_data) < 20:
                window_data = scaled_df.iloc[:boundary_t]
            anchor_fusion = PCAFusion(n_components=pca_components, use_sparse=False)
            try:
                anchor_fusion.fit(window_data)
                # Apply sign consistency against previous anchor
                if anchor_loadings_list[-1] is not None:
                    curr = anchor_fusion.sparse_pca.components_
                    prev = anchor_loadings_list[-1]
                    sim_matrix = 1 - cdist(curr, prev, metric='cosine')
                    new_idx, old_idx = linear_sum_assignment(-np.abs(sim_matrix))
                    anchor_fusion.sparse_pca.components_ = anchor_fusion.sparse_pca.components_[new_idx]
                    for i in range(len(anchor_fusion.sparse_pca.components_)):
                        if np.dot(anchor_fusion.sparse_pca.components_[i], prev[old_idx[i]]) < 0:
                            anchor_fusion.sparse_pca.components_[i] *= -1
                anchor_loadings_list.append(anchor_fusion.sparse_pca.components_.copy())
            except Exception:
                anchor_loadings_list.append(
                    _fallback_pca_components(
                        pca_components,
                        window_data.shape[1],
                        previous_loadings=anchor_loadings_list[-1],
                    )
                )
        
        print(f"    Anchor loadings computed in {time.time()-_t0_pca:.1f}s. Launching parallel PCA...", flush=True)
        
        # Step B: Dispatch chunks to worker pool
        # Use 'spawn' context on macOS for stability.
        # Safe because workers only use numpy/scipy/sklearn.
        worker_args = [
            (chunks[i], scaled_values, scaled_columns, anchor_loadings_list[i], pca_components, 0.1)
            for i in range(len(chunks))
        ]
        
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_causal_pca_worker, worker_args)
        
        # Step C: Reassemble results
        for chunk_indices, pc_chunk, final_loadings in results:
            for local_i, t in enumerate(chunk_indices):
                pc_values[t-1] = pc_chunk[local_i]
        
        # Keep the last anchor loadings for downstream use
        _, _, prev_loadings = results[-1]
        # Re-fit final fusion object on full data for downstream use
        fusion.fit(scaled_df)
        if prev_loadings is not None:
            curr = fusion.sparse_pca.components_
            sim_matrix = 1 - cdist(curr, prev_loadings, metric='cosine')
            new_idx, old_idx = linear_sum_assignment(-np.abs(sim_matrix))
            fusion.sparse_pca.components_ = fusion.sparse_pca.components_[new_idx]
            for i in range(len(fusion.sparse_pca.components_)):
                if np.dot(fusion.sparse_pca.components_[i], prev_loadings[old_idx[i]]) < 0:
                    fusion.sparse_pca.components_[i] *= -1
        
        elapsed_pca = time.time() - _t0_pca
        print(f"  [Step 2/4] Parallel Causal PCA done in {elapsed_pca:.1f}s "
              f"({n_workers} workers, {total_pca_steps} steps).", flush=True)
        pc_df = pd.DataFrame(pc_values, index=scaled_df.index, columns=[f'PC{i+1}' for i in range(pca_components)])
        scaler = None 
    else:
        # LOCAL WINDOW SCALING (Fixed window training)
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(model_df)
        scaled_df = pd.DataFrame(scaled_values, index=model_df.index, columns=model_df.columns)
        
        # Auto-select PCA components for ≥85% variance retention (non-expanding)
        if pca_components is None:
            from sklearn.decomposition import PCA as _PCA
            _auto_pca = _PCA()
            _auto_pca.fit(scaled_df.values)
            _cumvar = np.cumsum(_auto_pca.explained_variance_ratio_)
            pca_components = max(2, int(np.argmax(_cumvar >= 0.85) + 1))
            print(f"  [PCA] Auto-selected {pca_components} components "
                  f"(retains {_cumvar[pca_components-1]:.1%} variance, target ≥85%)", flush=True)

        # LOCAL PCA FUSION
        fusion = PCAFusion(n_components=pca_components, use_sparse=False)
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
    
    if np.any(np.isnan(features_input)) or np.any(np.isinf(features_input)):
        raise ValueError("Input features contain NaNs or Infs before HMM training")

    if expanding_window:
        refit_interval = REGIME_HMM_REFIT_INTERVAL_DAYS
        n_samples = len(features_input)
        n_refits = n_samples // refit_interval + 1
        print(f"  [Step 3/4] Walk-Forward HMM Refit: {n_samples} samples, ~{n_refits} refits (every {refit_interval}d)...", flush=True)
        _t0_wf = time.time()
        offset = len(scaled_df) - n_samples
        if n_samples == 0:
            return None, 0, pc_df

        causal_states = []
        causal_labels = []
        causal_probs = []
        causal_refit_ids = []
        causal_refit_dates = []
        causal_refit_methods = []
        causal_refit_fallbacks = []
        causal_prior_alignments = []
        causal_variance_orders = []
        causal_state_means = []
        causal_state_variances = []
        causal_semantic_drift = []
        current_hmm = None
        current_k = n_components
        current_fusion = None
        current_refit_id = None
        current_refit_date = None
        current_refit_metadata = {}
        previous_refit_metadata = None
        prev_refit_loadings = None
        debounce_counter = 0
        candidate_state = None
        _refit_count = 0
        next_refit_i = 0

        for i in range(n_samples):
            t_abs = i + offset

            if i >= next_refit_i:
                try:
                    _refit_count += 1
                    elapsed_wf = time.time() - _t0_wf
                    rate_wf = i / max(elapsed_wf, 0.01) if i > 0 else 1
                    eta_wf = (n_samples - i) / max(rate_wf, 0.01)
                    refit_date = scaled_df.index[t_abs].strftime('%Y-%m-%d') if t_abs < len(scaled_df) else '?'
                    print(f"    HMM refit #{_refit_count} at sample {i}/{n_samples} "
                          f"(date={refit_date}, {100*i/n_samples:.0f}%) "
                          f"| elapsed {elapsed_wf:.0f}s | ETA {eta_wf:.0f}s", flush=True)

                    current_window_scaled = scaled_df.iloc[:t_abs]
                    current_window_raw = model_df.iloc[:t_abs].dropna()
                    if len(current_window_scaled) < 20 or len(current_window_raw) < 20:
                        continue
                    if np.nanstd(current_window_scaled.values) < 1e-10:
                        continue

                    stable_fusion = PCAFusion(n_components=pca_components, use_sparse=False)
                    stable_fusion.fit(current_window_scaled)

                    current_loadings = stable_fusion.sparse_pca.components_
                    if prev_refit_loadings is not None:
                        raw_dist = cdist(current_loadings, prev_refit_loadings, metric='cosine')
                        clean_dist = np.nan_to_num(raw_dist, nan=1.0, posinf=1.0, neginf=1.0)
                        sim_matrix = 1.0 - clean_dist
                        new_idx, old_idx = linear_sum_assignment(-np.abs(sim_matrix))
                        stable_fusion.sparse_pca.components_ = stable_fusion.sparse_pca.components_[new_idx]
                        for c, old_c in enumerate(old_idx):
                            if np.dot(stable_fusion.sparse_pca.components_[c], prev_refit_loadings[old_c]) < 0:
                                stable_fusion.sparse_pca.components_[c] *= -1

                    refit_features = stable_fusion.transform(current_window_scaled).values
                    current_hmm, current_k = _fit_hmm_model(
                        refit_features,
                        n_components=current_k,
                        previous_model=current_hmm,
                    )
                    current_refit_id = _refit_count
                    current_refit_date = scaled_df.index[t_abs]
                    current_refit_metadata = getattr(current_hmm, "refit_metadata_", {}) or {}
                    curr_vars = np.asarray(current_refit_metadata.get("state_variances", []), dtype=float)
                    prev_vars = (
                        np.asarray(previous_refit_metadata.get("state_variances", []), dtype=float)
                        if previous_refit_metadata
                        else np.array([])
                    )
                    semantic_drift = False
                    if curr_vars.size and prev_vars.size == curr_vars.size:
                        rel_change = np.abs(curr_vars - prev_vars) / np.maximum(np.abs(prev_vars), 1e-8)
                        semantic_drift = bool(np.nanmax(rel_change) > 0.50)
                    current_refit_metadata["semantic_drift_flag"] = semantic_drift
                    previous_refit_metadata = deepcopy(current_refit_metadata)
                    current_hmm.fusion_ = stable_fusion
                    current_hmm.scaler_ = _rolling_scaler_snapshot(
                        current_window_raw,
                        window=min(252 * 5, len(current_window_raw)),
                    )
                    current_hmm.feature_names_ = model_df.columns.tolist()
                    current_fusion = stable_fusion
                    prev_refit_loadings = stable_fusion.sparse_pca.components_.copy()
                    next_refit_i = i + refit_interval
                except Exception as e:
                    print(f"  [Refit] Refit at t={t_abs} failed: {e}. Keeping previous model.", flush=True)
                    next_refit_i = i + refit_interval
                    if current_hmm is None:
                        continue

            if current_hmm is None or current_fusion is None:
                continue

            current_window_scaled = scaled_df.iloc[:t_abs + 1]
            if np.nanstd(current_window_scaled.values) < 1e-10:
                continue
            current_window_projected = current_fusion.transform(current_window_scaled).values
            current_prob = current_hmm.predict_proba(current_window_projected)[-1]

            potential_state = int(np.argmax(current_prob))
            if not causal_states:
                state = potential_state
                candidate_state = state
                debounce_counter = 0
            else:
                prev_state = int(causal_states[-1])
                if potential_state != prev_state:
                    if potential_state == candidate_state:
                        debounce_counter += 1
                    else:
                        candidate_state = potential_state
                        debounce_counter = 1
                    state = potential_state if current_prob[potential_state] > 0.70 and debounce_counter >= 3 else prev_state
                else:
                    state = prev_state
                    candidate_state = state
                    debounce_counter = 0

            causal_states.append(state)
            causal_probs.append(current_prob)
            labels_t = get_regime_labels(current_hmm)
            causal_labels.append(labels_t.get(state, f"Regime {state}"))
            causal_refit_ids.append(current_refit_id)
            causal_refit_dates.append(current_refit_date)
            causal_refit_methods.append(current_refit_metadata.get("method"))
            causal_refit_fallbacks.append(current_refit_metadata.get("fallback_reason"))
            causal_prior_alignments.append(json.dumps(current_refit_metadata.get("prior_alignment_order")))
            causal_variance_orders.append(json.dumps(current_refit_metadata.get("variance_sort_order")))
            causal_state_means.append(json.dumps(current_refit_metadata.get("state_means")))
            causal_state_variances.append(json.dumps(current_refit_metadata.get("state_variances")))
            causal_semantic_drift.append(bool(current_refit_metadata.get("semantic_drift_flag", False)))

        print(f"  [Step 3/4] Walk-Forward done in {time.time()-_t0_wf:.1f}s ({_refit_count} refits).", flush=True)

        if not causal_states:
            print("  CRITICAL: Walk-forward HMM never produced a valid model.")
            return None, 0, pc_df

        causal_probs_arr = np.vstack(causal_probs)
        result_index = pc_df.index[-len(causal_states):]
        pc_df = pc_df.loc[result_index].copy()
        pc_df['HMM_State'] = np.asarray(causal_states, dtype=int)
        pc_df['Regime_Label'] = causal_labels
        pc_df['HMM_Refit_ID'] = causal_refit_ids
        pc_df['HMM_Refit_Date'] = causal_refit_dates
        pc_df['HMM_Refit_Method'] = causal_refit_methods
        pc_df['HMM_Refit_Fallback_Reason'] = causal_refit_fallbacks
        pc_df['HMM_Prior_Alignment_Order'] = causal_prior_alignments
        pc_df['HMM_Variance_Sort_Order'] = causal_variance_orders
        pc_df['HMM_State_Means'] = causal_state_means
        pc_df['HMM_State_Variances'] = causal_state_variances
        pc_df['HMM_Semantic_Drift_Flag'] = causal_semantic_drift
        pc_df['Regime_Signal_Timestamp'] = 'close_T_for_next_session'
        pc_df.attrs['regime_signal_timestamp'] = 'close_T_for_next_session'
        pc_df.attrs['regime_inference_mode'] = 'walk_forward_refit'
        for k in range(current_k):
            pc_df[f'prob_state_{k}'] = causal_probs_arr[:, k]
        pc_df = _apply_causal_stress_overlay(pc_df, df, current_k)
        return current_hmm, current_k, pc_df

    try:
        best_hmm, best_k = _fit_hmm_model(
            features_input,
            n_components=n_components,
            previous_model=None,
        )
    except Exception as e:
        print(f"  CRITICAL: Non-expanding HMM training failed: {e}")
        return None, 0, pc_df
    
    # Final check on state persistence (Diagonal Dominance)
    transmat = best_hmm.transmat_
    avg_persistence = np.mean(np.diag(transmat))
    if avg_persistence < 0.8:
        print(f"  WARNING: HMM regimes are not persistent (avg diag={avg_persistence:.2f}).")
    
    # Store fusion and features for later use
    best_hmm.fusion_ = fusion
    best_hmm.scaler_ = ingestor.rolling_scaler if expanding_window else scaler
    best_hmm.feature_names_ = model_df.columns.tolist()

    # Mandate 10.2: Causal Viterbi Decoding
    # During live trading or backtesting, if the agent runs .predict(), 
    # it must only extract the final integer.
    # Use expanding window predict_proba to get causal filtered probabilities
    n_samples = len(features_input)
    causal_states = np.zeros(n_samples)
    causal_labels = ["Unknown"] * n_samples
    causal_probs = np.zeros((n_samples, best_k))
    
    if n_samples > 0:
        # Non-expanding causal filter for historical labels
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
            causal_probs[i] = current_prob
            labels_all = get_regime_labels(best_hmm)
            causal_labels[i] = labels_all.get(causal_states[i], f"Regime {causal_states[i]}")
    else:
        probs = best_hmm.predict_proba(features_input)
        causal_probs = probs
        causal_states = np.argmax(probs, axis=1)
        labels_all = get_regime_labels(best_hmm)
        for i, s in enumerate(causal_states):
            causal_labels[i] = labels_all.get(s, f"Regime {s}")
        
    pc_df['HMM_State'] = causal_states
    pc_df['Regime_Label'] = causal_labels
    pc_df['Regime_Signal_Timestamp'] = 'close_T_for_next_session' if expanding_window else 'full_sample_research_only'
    pc_df.attrs['regime_signal_timestamp'] = pc_df['Regime_Signal_Timestamp'].iloc[0] if len(pc_df) else ''
    pc_df.attrs['regime_inference_mode'] = 'walk_forward_refit' if expanding_window else 'full_sample_research_only'
    for k in range(best_k):
        pc_df[f'prob_state_{k}'] = causal_probs[:, k]
    
    # Return the LAST model from the expanding window as the "live" model
    if expanding_window and n_samples > warmup:
        return current_hmm, best_k, pc_df
        
    return best_hmm, best_k, pc_df


def get_regime_labels(hmm_model, pc_df=None):
    """
    Assigns semantic names to HMM regimes using Adaptive Archetype Ranking.
    Instead of absolute thresholds (which fail across secular shifts), we rank states 
    relative to each other based on their intrinsic Risk/Return profiles.
    
    Risk proxy cascade (since VIX_Close was removed from the feature set):
      1. VVIX_Close (vol-of-vol — higher VVIX = higher market fear)
      2. SPY_Yang_Zhang_21d (OHLC-based realized vol)
      3. Covariance trace (total state variance as last resort)
    """
    if hmm_model is None or not hasattr(hmm_model, 'means_'):
        return {}

    K = hmm_model.n_components
    state_metrics = []
    
    features = getattr(hmm_model, 'feature_names_', [])
    
    # Risk proxy cascade: VVIX > Yang-Zhang > fallback to covariance trace
    risk_idx = -1
    for candidate in ['VVIX_Close', 'SPY_Yang_Zhang_21d', 'VIX_Close']:
        if candidate in features:
            risk_idx = features.index(candidate)
            logger.info(f"Regime labeling using risk proxy: {candidate}")
            break
    
    spy_ret_idx = features.index('SPY_Log_Return') if 'SPY_Log_Return' in features else -1
    
    if not hasattr(hmm_model, 'fusion_') or not hasattr(hmm_model, 'scaler_'):
        return {i: f"Regime {i}" for i in range(K)}

    # 1. Extract physical centroids for all states
    for i in range(K):
        try:
            state_mean_pc, state_var_pc = _state_emission_mean_var(hmm_model, i)
            state_mean_scaled = hmm_model.fusion_.sparse_pca.inverse_transform(state_mean_pc.reshape(1, -1))
            
            if not hasattr(hmm_model.scaler_, "center_"):
                avg_risk, avg_ret = 20.0, 0.0
            else:
                state_mean_raw = hmm_model.scaler_.inverse_transform(state_mean_scaled)[0]
                if risk_idx != -1:
                    avg_risk = state_mean_raw[risk_idx]
                else:
                    # Last resort: use covariance trace (total state variance)
                    avg_risk = float(np.trace(state_var_pc)) if state_var_pc.ndim == 2 else float(np.sum(state_var_pc))
                    logger.info(f"State {i}: using covariance trace ({avg_risk:.4f}) as risk proxy")
                avg_ret = state_mean_raw[spy_ret_idx] if spy_ret_idx != -1 else 0.0
        except Exception as e:
            from live_trading.data_ingestion import red_alert
            red_alert(f"Failed to inverse transform state centroids: {e}")
            raise e
            
        state_metrics.append({
            'id': i,
            'risk': avg_risk,
            'ret': avg_ret,
            'sharpe': avg_ret / max(abs(avg_risk), 1.0)
        })

    # 2. Sequential Archetype Assignment based on RELATIVE RANK
    labels = {}
    remaining = list(range(K))
    
    # A. Market Turmoil: State with the absolute highest risk proxy
    turmoil_idx = max(remaining, key=lambda i: state_metrics[i]['risk'])
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
        elif state_metrics[idx]['risk'] > state_metrics[turmoil_idx]['risk'] * 0.5:
            name = "High Vol Chop"
        else:
            name = "Stagnant/Neutral"
        labels[idx] = f"{name} ({idx})"
            
    return labels

def _feature_hash(feature_names):
    payload = json.dumps(list(feature_names or []), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_cache_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "na"


def _regime_snapshot_path(metadata):
    os.makedirs(REGIME_SNAPSHOT_CACHE_DIR, exist_ok=True)
    parts = [
        f"v{REGIME_SNAPSHOT_PIPELINE_VERSION}",
        f"asof_{_safe_cache_token(metadata.get('as_of_date'))}",
        f"h_{_safe_cache_token(metadata.get('horizon_calendar_days'))}",
        f"th_{_safe_cache_token(metadata.get('trading_horizon'))}",
        f"k_{_safe_cache_token(metadata.get('requested_n_components'))}",
        f"prob_{_safe_cache_token(metadata.get('probability_model'))}",
        f"model_{_safe_cache_token(metadata.get('model_class'))}",
    ]
    return os.path.join(REGIME_SNAPSHOT_CACHE_DIR, "_".join(parts) + ".pkl")


def list_regime_snapshots():
    """Return cached regime snapshot metadata from the local snapshot directory."""
    import pickle

    snapshots = []
    if not os.path.isdir(REGIME_SNAPSHOT_CACHE_DIR):
        return snapshots

    for name in sorted(os.listdir(REGIME_SNAPSHOT_CACHE_DIR)):
        if not name.endswith(".pkl"):
            continue
        path = os.path.join(REGIME_SNAPSHOT_CACHE_DIR, name)
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            snapshots.append(
                {
                    "path": path,
                    "timestamp": payload.get("timestamp"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        except Exception:
            continue
    return snapshots


def save_regime_snapshot(regime_dict, hmm_model, daily_models, metadata=None):
    """Save a date-keyed regime snapshot for later lookup."""
    import pickle

    try:
        metadata = metadata or {}
        metadata = dict(metadata)
        metadata.setdefault("snapshot_pipeline_version", REGIME_SNAPSHOT_PIPELINE_VERSION)
        metadata.setdefault("training_data_end", metadata.get("as_of_date"))
        if hmm_model is not None:
            metadata.setdefault("fitted_n_components", getattr(hmm_model, "n_components", None))
            metadata.setdefault("feature_hash", _feature_hash(getattr(hmm_model, "feature_names_", [])))
            metadata.setdefault("feature_names", getattr(hmm_model, "feature_names_", []))

        data = {
            "regime_dict": regime_dict,
            "hmm_model": hmm_model,
            "daily_models": daily_models,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }
        path = _regime_snapshot_path(metadata)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Market regime snapshot cached to {path}")
        return path
    except Exception as e:
        print(f"❌ Failed to cache regime snapshot: {e}")
        return None


def load_regime_snapshot(expected_metadata=None):
    """Load a cached regime snapshot if the metadata matches exactly."""
    import pickle

    expected_metadata = expected_metadata or {}
    path = _regime_snapshot_path(expected_metadata)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        actual_metadata = data.get("metadata", {})
        for key, expected_value in expected_metadata.items():
            if expected_value is None:
                continue
            if actual_metadata.get(key) != expected_value:
                return None
        return data
    except Exception as e:
        print(f"❌ Failed to load regime snapshot {path}: {e}")
        return None

def save_regime_cache(regime_dict, hmm_model, daily_models, metadata=None):
    """Saves the regime detection results to a pickle file."""
    import pickle
    try:
        metadata = metadata or {}
        metadata = dict(metadata)
        metadata.setdefault("snapshot_pipeline_version", REGIME_SNAPSHOT_PIPELINE_VERSION)
        metadata.setdefault("training_data_end", metadata.get("as_of_date"))
        if hmm_model is not None:
            metadata.setdefault("fitted_n_components", getattr(hmm_model, "n_components", None))
            metadata.setdefault("feature_hash", _feature_hash(getattr(hmm_model, "feature_names_", [])))
            metadata.setdefault("feature_names", getattr(hmm_model, "feature_names_", []))
        data = {
            "regime_dict": regime_dict,
            "hmm_model": hmm_model,
            "daily_models": daily_models,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }
        with open(REGIME_CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Market regime data cached to {REGIME_CACHE_FILE}")
        save_regime_snapshot(regime_dict, hmm_model, daily_models, metadata=metadata)
        return True
    except Exception as e:
        print(f"❌ Failed to cache regime data: {e}")
        return False

def load_regime_cache(expected_metadata=None):
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
            print(f"⚠️ Market regime cache is stale ({cache_time.strftime('%Y-%m-%d %H:%M')}). Forcing a refit.")
            return None

        expected_metadata = expected_metadata or {}
        actual_metadata = data.get("metadata", {})
        for key, expected_value in expected_metadata.items():
            if expected_value is None:
                continue
            if actual_metadata.get(key) != expected_value:
                print(
                    f"⚠️ Market regime cache mismatch for {key}: "
                    f"cached={actual_metadata.get(key)} requested={expected_value}. Forcing a refit."
                )
                return None
        
        return data
    except Exception as e:
        print(f"❌ Failed to load regime cache: {e}")
        return None


def _hmm_refit_anchor_date(trading_index, as_of_date, refit_interval=REGIME_HMM_REFIT_INTERVAL_DAYS):
    """Map an evaluation date to the latest causal HMM refit anchor."""
    as_of_ts = pd.Timestamp(as_of_date).normalize()
    if refit_interval <= 0:
        return as_of_ts

    idx = pd.DatetimeIndex(trading_index).sort_values().unique()
    if len(idx) == 0:
        return as_of_ts

    pos = idx.searchsorted(as_of_ts, side="right") - 1
    if pos < 0:
        return idx[0].normalize()

    anchor_pos = (pos // refit_interval) * refit_interval
    anchor_pos = min(anchor_pos, pos)
    return idx[anchor_pos].normalize()


def _build_causal_regime_feature_frame(causal_df, hmm_model, as_of_ts, refit_date=None, exclude_features=None):
    """Score a causal dataframe with a cached HMM snapshot."""
    if hmm_model is None or not hasattr(hmm_model, "fusion_"):
        raise ValueError("Cached HMM snapshot is missing the fitted fusion pipeline.")

    ingestor = DataIngestor()
    start_date = causal_df.index.min().strftime("%Y-%m-%d")
    end_date = pd.Timestamp(as_of_ts).normalize().strftime("%Y-%m-%d")

    def run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    stationary_df = run_async(ingestor.build_fused_dataset(start_date, end_date, scale=False))
    stationary_df = stationary_df[stationary_df.index <= pd.Timestamp(as_of_ts)].copy()

    if exclude_features:
        cols_to_drop = [c for c in exclude_features if c in stationary_df.columns]
        if cols_to_drop:
            stationary_df = stationary_df.drop(columns=cols_to_drop)

    model_df = stationary_df.drop(columns=["VIX_Close"], errors="ignore")
    if "VIX_Close" in stationary_df.columns:
        print("  [Feature Selection] Dropped VIX_Close from cached HMM emissions; retained for overlays.")

    if model_df.empty:
        return pd.DataFrame()

    print(f"  [Step 1/4] Rolling robust scaling on {len(model_df)} rows...", flush=True)
    _t0_scale = time.time()
    scale_window = max(20, min(252 * 5, len(model_df) - 1))
    scaled_df = ingestor.scale_features(model_df, rolling=True, window=scale_window)
    # Drop leading NaNs from rolling robust scaling to prevent zero-padding variance-collapse
    scaled_df = scaled_df.dropna()
    if scaled_df.empty:
        return pd.DataFrame()
    print(f"  [Step 1/4] Scaling done in {time.time() - _t0_scale:.1f}s.", flush=True)

    try:
        pc_df = hmm_model.fusion_.transform(scaled_df)
    except Exception as e:
        raise ValueError(f"Failed to project causal features through cached HMM fusion: {e}") from e

    if not isinstance(pc_df, pd.DataFrame):
        pc_df = pd.DataFrame(pc_df, index=scaled_df.index)

    pc_df = pc_df.dropna().copy()
    if pc_df.empty:
        return pc_df

    pc_df.columns = [f"PC{i+1}" for i in range(pc_df.shape[1])]

    # Walk-forward prefix predict_proba (Mandate 12.5) with debounce logic
    n_samples = len(pc_df)
    causal_probs = np.zeros((n_samples, hmm_model.n_components))
    causal_states = np.zeros(n_samples, dtype=int)
    
    debounce_counter = 0
    candidate_state = None
    
    for i in range(n_samples):
        current_prob = hmm_model.predict_proba(pc_df.values[:i+1])[-1]
        causal_probs[i] = current_prob
        
        potential_state = int(np.argmax(current_prob))
        if i == 0:
            state = potential_state
            candidate_state = state
            debounce_counter = 0
        else:
            prev_state = int(causal_states[i-1])
            if potential_state != prev_state:
                if potential_state == candidate_state:
                    debounce_counter += 1
                else:
                    candidate_state = potential_state
                    debounce_counter = 1
                state = potential_state if current_prob[potential_state] > 0.70 and debounce_counter >= 3 else prev_state
            else:
                state = prev_state
                candidate_state = state
                debounce_counter = 0
        causal_states[i] = state

    labels_all = get_regime_labels(hmm_model)
    pc_df["HMM_State"] = causal_states
    pc_df["Regime_Label"] = [labels_all.get(int(s), f"Regime {int(s)}") for s in causal_states]
    pc_df["HMM_Refit_Date"] = pd.Timestamp(refit_date if refit_date is not None else as_of_ts).normalize()
    pc_df["HMM_Refit_Method"] = "cached_refit_20d"
    pc_df["HMM_Refit_Fallback_Reason"] = ""
    pc_df["HMM_Prior_Alignment_Order"] = ""
    pc_df["HMM_Variance_Sort_Order"] = ""
    pc_df["HMM_State_Means"] = ""
    pc_df["HMM_State_Variances"] = ""
    pc_df["HMM_Semantic_Drift_Flag"] = False
    pc_df["Regime_Signal_Timestamp"] = "close_T_for_next_session"
    pc_df.attrs["regime_signal_timestamp"] = "close_T_for_next_session"
    pc_df.attrs["regime_inference_mode"] = "cached_refit_20d"

    for k in range(causal_probs.shape[1]):
        pc_df[f"prob_state_{k}"] = causal_probs[:, k]

    pc_df = _apply_causal_stress_overlay(pc_df, stationary_df, hmm_model.n_components)
    return pc_df



@functools.lru_cache(maxsize=10)
def build_regime_return_arrays(cache_key_dummy, horizon=45, n_components=3, force_refit=False, as_of_date=None):
    trading_horizon = calendar_days_to_trading_days(horizon)
    as_of_ts = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp(datetime.now()).normalize()
    df = fetch_historical_data()
    if df.empty:
        return {}, None, []

    total_rows = len(df)
    causal_df = df.loc[df.index <= as_of_ts].copy()
    if causal_df.empty:
        return {}, None, []

    hmm_anchor_date = _hmm_refit_anchor_date(causal_df.index, as_of_ts, REGIME_HMM_REFIT_INTERVAL_DAYS)
    requested_metadata = {
        "as_of_date": hmm_anchor_date.strftime("%Y-%m-%d"),
        "training_data_end": hmm_anchor_date.strftime("%Y-%m-%d"),
        "snapshot_pipeline_version": REGIME_SNAPSHOT_PIPELINE_VERSION,
        "horizon_calendar_days": int(horizon),
        "trading_horizon": int(trading_horizon),
        "requested_n_components": n_components if n_components is not None else "auto",
        "probability_model": PROBABILITY_MODEL,
        "model_class": "GaussianHMM",
        "hmm_refit_interval_days": REGIME_HMM_REFIT_INTERVAL_DAYS,
    }
    print("🧠 Starting causal HMM regime scoring...", flush=True)
    print(
        f"  Config: horizon={horizon}cal/{trading_horizon}trd days, "
        f"K={n_components or 'auto(3-5)'}, as_of={as_of_ts.date()}, "
        f"hmm_refit_anchor={hmm_anchor_date.date()}, interval={REGIME_HMM_REFIT_INTERVAL_DAYS}d",
        flush=True,
    )
    print(f"  Loaded {total_rows} days of historical data; {len(causal_df)} rows eligible as of {as_of_ts.date()}.")
    _t0_total = time.time()

    cached_data = None
    if not force_refit:
        cached_data = load_regime_snapshot(requested_metadata)
        if cached_data is None:
            cached_data = load_regime_cache(requested_metadata)

    if cached_data and cached_data.get("hmm_model") is not None:
        best_hmm = cached_data["hmm_model"]
        print(f"📈 Using cached HMM snapshot (from {cached_data['timestamp']})")
    else:
        hmm_train_df = df.loc[df.index <= hmm_anchor_date].copy()
        if hmm_train_df.empty:
            return {}, None, []
        print("🧠 Starting heavy HMM market regime training (this may take several minutes)...")
        print(
            f"  Config: horizon={horizon}cal/{trading_horizon}trd days, "
            f"K={n_components or 'auto(3-5)'}, anchor={hmm_anchor_date.date()}",
            flush=True,
        )
        try:
            # For historical analysis, we MUST use expanding_window=True to eliminate parameter look-ahead bias
            best_hmm, best_k, _ = train_regime_hmm(hmm_train_df, n_components=n_components, expanding_window=True)
        except Exception as e:
            from live_trading.data_ingestion import red_alert
            red_alert(f"Failed to train HMM: {e}")
            return {}, None, []
        print(f"  [Step 4/4] Building return buckets & GMM fitting...", flush=True)

    try:
        feature_df = _build_causal_regime_feature_frame(causal_df, best_hmm, as_of_ts, refit_date=hmm_anchor_date)
    except Exception as e:
        from live_trading.data_ingestion import red_alert
        red_alert(f"Failed to score HMM snapshot: {e}")
        return {}, None, []

    if feature_df.empty:
        return {}, None, []

    best_k = getattr(best_hmm, "n_components", None) or len([c for c in feature_df.columns if c.startswith("prob_state_")])
    
    # Align and add necessary physical columns for return calculation
    common_idx = feature_df.index.intersection(causal_df.index)
    feature_df = feature_df.loc[common_idx].copy()
    feature_df['SPY_Close'] = causal_df.loc[common_idx, 'SPY_Close']
    
    # Ensure Log_Return exists for daily models
    if 'Log_Return' not in feature_df.columns:
        if 'Log_Return' in causal_df.columns:
            feature_df['Log_Return'] = causal_df.loc[common_idx, 'Log_Return']
        else:
            feature_df['Log_Return'] = np.log(feature_df['SPY_Close'] / feature_df['SPY_Close'].shift(1))
    
    # Mandate 11.1: Calendar vs. Trading Day Alignment
    feature_df['future_terminal_return'] = feature_df['SPY_Close'].shift(-trading_horizon) / feature_df['SPY_Close'] - 1
    feature_df = feature_df.dropna(subset=['future_terminal_return'])
    if not feature_df.empty:
        as_of_pos = feature_df.index.searchsorted(as_of_ts, side="right") - 1
        eligible_positions = np.arange(len(feature_df)) + trading_horizon <= as_of_pos
        feature_df = feature_df.iloc[eligible_positions].copy()
    
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
        
    # Summary
    labels = get_regime_labels(best_hmm)
    for state in range(best_k):
        n_obs = len(regime_dict.get(f'State_{state}', []))
        label = labels.get(state, f'State {state}')
        print(f"    State {state} ({label}): {n_obs} return observations", flush=True)
    total_elapsed = time.time() - _t0_total
    print(f"  ✅ Regime training complete in {total_elapsed:.1f}s (K={best_k})", flush=True)

    cache_metadata = dict(requested_metadata)
    cache_metadata["fitted_n_components"] = best_k
    cache_metadata["feature_hash"] = _feature_hash(getattr(best_hmm, "feature_names_", []))
    cache_metadata["feature_names"] = getattr(best_hmm, "feature_names_", [])
    if cached_data is None or cached_data.get("hmm_model") is None:
        save_regime_cache(regime_dict, best_hmm, daily_models, metadata=cache_metadata)
    return regime_dict, best_hmm, daily_models


def select_gmm_by_bic(log_returns, max_components=GMM_MAX_COMPONENTS, min_obs_per_component=GMM_MIN_OBS_PER_COMPONENT):
    """
    Select a 1D Gaussian mixture by BIC, with k=1 included as the baseline.

    The component cap prevents tiny buckets from being overfit while still
    allowing richer mixtures when the sample size supports them.
    """
    x = np.asarray(log_returns, dtype=float).reshape(-1, 1)
    x = x[np.isfinite(x).ravel()]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least two finite returns to fit a distribution")

    supported_k = max(1, n // max(1, int(min_obs_per_component)))
    max_k = int(min(max_components, supported_k, n))
    candidates = []
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(x)
            candidates.append(
                {
                    "k": k,
                    "model": gmm,
                    "bic": float(gmm.bic(x)),
                    "aic": float(gmm.aic(x)),
                }
            )
        except Exception:
            continue
    if not candidates:
        raise ValueError("No GMM candidates converged")
    return min(candidates, key=lambda item: item["bic"]), candidates


def fit_gmm(bucket_returns, regime_label=""):
    n = len(bucket_returns)
    if n < 10:
        log_returns = np.log1p(bucket_returns) if n >= 2 else np.array([0.0])
        return {"type": "gaussian_fallback", "loc": float(np.mean(log_returns)), "scale": max(float(np.std(log_returns, ddof=max(1, n-1))), 1e-6)}
    log_returns = np.log1p(bucket_returns)
    try:
        selected, candidates = select_gmm_by_bic(log_returns)
        gmm = selected["model"]
        return {
            "type": "gmm",
            "n_components": int(selected["k"]),
            "selection_criterion": "bic",
            "bic": float(selected["bic"]),
            "aic": float(selected["aic"]),
            "candidate_bic": {int(c["k"]): float(c["bic"]) for c in candidates},
            "weights": gmm.weights_,
            "means": gmm.means_.flatten(),
            "stds": np.sqrt(np.maximum(gmm.covariances_.reshape(-1), 1e-12)),
        }
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
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        stationary_df = asyncio.run(ingestor.build_fused_dataset(start, end, scale=False))
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, ingestor.build_fused_dataset(start, end, scale=False))
            stationary_df = future.result()
    
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
    
    trading_horizon = calendar_days_to_trading_days(horizon)

    if USE_MARKOV_TRANSITIONS:
        projected_probs = current_probs @ np.linalg.matrix_power(hmm_model.transmat_, trading_horizon)
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
    Fully vectorized: simulates all paths simultaneously using numpy broadcasting.
    """
    num_states = len(gmm_models)
    rng = np.random.default_rng()
    
    # Pre-generate all random state sequences (num_paths x dte)
    # Initial states
    states = rng.choice(num_states, size=num_paths, p=current_state_probs)
    
    # Pre-compute cumulative log returns for all paths
    cumulative_returns = np.zeros(num_paths)
    breached = np.zeros(num_paths, dtype=bool)
    
    for day in range(dte):
        # Mask: only simulate paths that haven't breached yet
        active = ~breached
        if not np.any(active):
            break
        
        active_states = states[active]
        n_active = np.sum(active)
        daily_returns = np.zeros(n_active)
        
        for s in range(num_states):
            mask_s = active_states == s
            count_s = np.sum(mask_s)
            if count_s == 0:
                continue
            
            model = gmm_models[s]
            if model["type"] == "gaussian_fallback":
                daily_returns[mask_s] = rng.normal(loc=model["loc"], scale=model["scale"], size=count_s)
            else:
                # Sample GMM components then draw from the selected Gaussian
                comps = rng.choice(len(model["weights"]), size=count_s, p=model["weights"])
                state_indices = np.where(mask_s)[0]
                for c_idx in range(len(model["weights"])):
                    comp_mask = comps == c_idx
                    n_comp = np.sum(comp_mask)
                    if n_comp > 0:
                        draws = rng.normal(loc=model["means"][c_idx], scale=model["stds"][c_idx], size=n_comp)
                        daily_returns[state_indices[comp_mask]] = draws
        
        # Update cumulative returns for active paths
        active_indices = np.where(active)[0]
        cumulative_returns[active_indices] += daily_returns
        simple_returns = np.exp(cumulative_returns[active_indices]) - 1
        
        # Check breach conditions
        if option_type == "put":
            new_breaches = simple_returns <= strike_pct_drop
        else:
            new_breaches = simple_returns >= strike_pct_drop
        
        breached[active_indices[new_breaches]] = True
        
        # State transitions for surviving paths
        still_active = active & ~breached
        if np.any(still_active):
            for s in range(num_states):
                s_mask = still_active & (states == s)
                n_trans = np.sum(s_mask)
                if n_trans > 0:
                    states[s_mask] = rng.choice(num_states, size=n_trans, p=trans_matrix[s])
    
    return np.sum(breached) / num_paths

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
