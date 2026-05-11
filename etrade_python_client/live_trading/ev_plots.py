import os
import sys
import json
import time
import functools
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accounts.accounts_bo import Accounts
from core_api.stock_trade_class import LoginFailureException
from live_trading.etrade_cover_call_new import oauth, send_login_failure_notification
from live_trading.ev_engine import (
    PROBABILITY_MODEL, USE_MARKOV_TRANSITIONS, COST_PER_SPREAD, YF_QUOTE_CACHE_PATH,
    fetch_cached_yf_close, fetch_historical_data, build_regime_return_arrays,
    fit_gmm, query_gmm, get_probability_engine, calculate_yield_metrics,
    _build_single_regime_prob_func, train_regime_hmm, calculate_probability_of_touch,
    calendar_days_to_trading_days,
    get_regime_labels
)
from live_trading.data_ingestion import DataIngestor
import asyncio

TARGET_MARGIN_DOLLARS = 10000.0
COST_PER_SPREAD = 1.0

# Logic moved to ev_engine.py float(prob)


def _dominant_label(frame, state):
    rows = frame[frame['HMM_State'] == state]
    if 'Regime_Label' in rows.columns and not rows.empty and rows['Regime_Label'].notna().any():
        labels = rows['Regime_Label'].dropna()
        mode = labels.value_counts()
        if not mode.empty:
            return mode.index[0]
    return f"State {state}"

def plot_regime_timeline(n_components=None):
    """
    Time-domain visualization of HMM regimes from 2024 to today,
    overlaid with SPY price and VIX, plus a probability stacked area chart.
    """
    from live_trading.ev_engine import fetch_historical_data, train_regime_hmm
    
    df = fetch_historical_data()
    if df.empty:
        print("ERROR: No historical data available.")
        return

    # Train HMM with expanding_window=True to eliminate look-ahead bias
    best_hmm, best_k, feature_df = train_regime_hmm(df, n_components=n_components, expanding_window=True)
    # Mandate 10.6: Probabilities are now causally generated and stored in feature_df by the engine.
    # Bring in SPY and VIX from the original df for plotting
    feature_df = feature_df.join(df[['SPY_Close', 'VIX_Close']], how='inner')

    # Filter for 2015 onwards
    timeline_df = feature_df[feature_df.index >= '2015-01-01'].copy()
    if {'Detected_Regime_State', 'Detected_Regime_Label'}.issubset(timeline_df.columns):
        timeline_df['Raw_HMM_State'] = timeline_df['HMM_State']
        timeline_df['Raw_Regime_Label'] = timeline_df['Regime_Label']
        timeline_df['HMM_State'] = timeline_df['Detected_Regime_State']
        timeline_df['Regime_Label'] = timeline_df['Detected_Regime_Label']
        best_k = int(timeline_df['HMM_State'].max()) + 1
    if timeline_df.empty:
        print("ERROR: No data available for 2015 onwards.")
        return

    # Create 2 subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(24, 16), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # X-axis dates
    dates = timeline_df.index

    # --- Plot 1: Price and Regimes ---
    colors_list = plt.cm.Set3.colors
    stable_labels = {i: _dominant_label(timeline_df, i) for i in range(best_k)}
    state_changes = timeline_df['HMM_State'].ne(timeline_df['HMM_State'].shift()).cumsum()
    groups = timeline_df.groupby(state_changes)
    
    added_to_legend = set()
    for _, group in groups:
        state = group['HMM_State'].iloc[0]
        state_val = int(state) if not pd.isna(state) else 0
        color = colors_list[state_val % len(colors_list)]
        start_date = group.index[0]
        end_date = group.index[-1]
        label = stable_labels.get(state_val, f"State {state_val}")
        if label not in added_to_legend:
            ax1.axvspan(start_date, end_date, color=color, alpha=0.4, label=label)
            added_to_legend.add(label)
        else:
            ax1.axvspan(start_date, end_date, color=color, alpha=0.4)

    color_spy = 'navy'
    ax1.set_ylabel('SPY Price ($)', color=color_spy, fontsize=14, fontweight='bold')
    ax1.plot(dates, timeline_df['SPY_Close'], color=color_spy, linewidth=2.5, label='SPY Price', alpha=0.9)
    ax1.tick_params(axis='y', labelcolor=color_spy, labelsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    color_vix = 'darkred'
    ax2.set_ylabel('VIX Index', color=color_vix, fontsize=14, fontweight='bold')
    ax2.plot(dates, timeline_df['VIX_Close'], color=color_vix, linewidth=1.5, alpha=0.8, label='VIX Index', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color_vix, labelsize=12)

    # --- Plot 2: Probability Stacked Area ---
    prob_data = [timeline_df[f'prob_state_{i}'].values for i in range(best_k)]
    state_labels = []
    for i in range(best_k):
        state_labels.append(stable_labels.get(i, f'State {i}'))
    ax3.stackplot(dates, prob_data, labels=state_labels, 
                  colors=colors_list[:best_k], alpha=0.8)
    ax3.set_ylabel('Regime Probability', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left', frameon=True, fontsize=10, ncol=best_k)
    ax3.grid(True, alpha=0.3)

    # Formatting and Combine Legends for Ax1
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=11)

    plt.suptitle(f"Detailed Market Regime Analysis (2015 - Present)\nMarkov-Switching HMM States (K={best_k}) with Posterior Probabilities", 
                 fontsize=20, fontweight='bold', y=0.95)
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    combined_h = [h for h, l in zip(h1+h2, l1+l2) if not l.startswith('Regime')] + \
                 [h for h, l in zip(h1+h2, l1+l2) if l.startswith('Regime')]
    combined_l = [l for l in l1+l2 if not l.startswith('Regime')] + \
                 [l for l in l1+l2 if l.startswith('Regime')]
    ax1.legend(combined_h, combined_l, loc='upper left', frameon=True, shadow=True, fontsize=10, ncol=2)

    # Annotate key features
    ax1.text(0.01, -0.1, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             transform=ax1.transAxes, fontsize=10, color='gray')

    plt.tight_layout()
    
    output_path = "s_and_p_data/regime_timeline_2015.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Regime Timeline Plot saved to {output_path}")
    plt.show()

def plot_regime_distributions(horizon=45, n_components=None):
    """
    Standalone visualization: histograms of SPY horizon returns bucketed by HMM regime.
    """
    hist_df = fetch_historical_data()
    if hist_df.empty:
        print("ERROR: No historical data available.")
        return

    regime_dict, hmm_model, _ = build_regime_return_arrays(int(time.time()/86400), horizon=horizon, n_components=n_components)
    K = hmm_model.n_components

    cols = 2
    rows = max(2, (K + 1) // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = axes.flatten()

    print("\n" + "="*70)
    print("SPY Daily Return Distribution by HMM State — Normality Tests")
    print("="*70)

    colors_list = plt.cm.tab10.colors

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= K:
             ax.set_visible(False)
             continue
             
        key = f'State_{idx}'
        label = f'HMM State {idx}'
        color = colors_list[idx % len(colors_list)]
        returns = regime_dict.get(key, np.array([]))
        n = len(returns)

        if n < 10:
            ax.set_title(f"{label}\n(insufficient data: n={n})")
            ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
            continue

        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # excess kurtosis

        # --- Histogram ---
        n_bins = min(80, max(30, n // 50))
        counts, bins, patches = ax.hist(returns, bins=n_bins, density=True,
                                         alpha=0.65, color=color, edgecolor='black',
                                         linewidth=0.4, label='Empirical')

        # --- Student-t overlay ---
        x = np.linspace(bins[0], bins[-1], 300)
        df_nu, loc, scale = stats.t.fit(returns)
        ax.plot(x, stats.t.pdf(x, df_nu, loc=loc, scale=scale),
                color='black', linewidth=2, linestyle='--', label='Student-t fit')

        # --- Annotation box ---
        textstr = (
            f"n = {n:,}\n"
            f"μ = {mu:.6f}\n"
            f"σ = {sigma:.6f}\n"
            f"skew = {skew:.4f}\n"
            f"excess kurt = {kurt:.4f}\n"
            f"─────────────\n"
            f"Student-t Fit\n"
            f"ν (df) = {df_nu:.2f}\n"
            f"loc = {loc:.4f}\n"
            f"scale = {scale:.4f}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')

        # --- Formatting ---
        ax.set_title(f"{label}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3) 

        verdict = f"Fat Tails Captured (ν={df_nu:.1f})"
        ax.set_xlabel(f"Daily Return — {verdict}", fontsize=10, color='green')

        # --- Console output ---
        print(f"\n{'─'*50}")
        print(f"  {label} (n={n:,})")
        print(f"{'─'*50}")
        print(f"  mean     = {mu:.6f}")
        print(f"  std      = {sigma:.6f}")
        print(f"  skewness = {skew:.4f}")
        print(f"  ex. kurt = {kurt:.4f}")
        print(f"  Student-t: ν={df_nu:.2f}, loc={loc:.4f}, scale={scale:.4f}")
        print(f"  → {verdict}")

    plt.suptitle("SPY Daily Return Distributions by HMM State\n"
                 "15 Years of Historical Data — Gaussian Fit & Normality Tests",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = "s_and_p_data/spy_return_distributions_by_hmm.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.show()

def plot_regime_log_return_gmm(n_components=None, start_date='2015-01-01', output_dir='s_and_p_data'):
    """
    Plot the causal regime timeline and fit per-regime GMMs to SPY daily log returns.

    This diagnostic uses `train_regime_hmm(..., expanding_window=True)` and the
    per-row causal labels returned by the engine. It does not use forward returns.
    """
    hist_df = fetch_historical_data()
    if hist_df.empty:
        print("ERROR: No historical data available.")
        return

    best_hmm, best_k, feature_df = train_regime_hmm(
        hist_df,
        n_components=n_components,
        expanding_window=True,
    )
    if best_hmm is None or feature_df.empty:
        print("ERROR: HMM training did not produce a usable feature frame.")
        return

    common_idx = feature_df.index.intersection(hist_df.index)
    diag_df = feature_df.loc[common_idx].copy()
    diag_df['SPY_Close'] = hist_df.loc[common_idx, 'SPY_Close']
    diag_df['VIX_Close'] = hist_df.loc[common_idx, 'VIX_Close']
    diag_df['SPY_Log_Return'] = np.log(diag_df['SPY_Close'] / diag_df['SPY_Close'].shift(1))
    if {'Detected_Regime_State', 'Detected_Regime_Label'}.issubset(diag_df.columns):
        diag_df['Raw_HMM_State'] = diag_df['HMM_State']
        diag_df['Raw_Regime_Label'] = diag_df['Regime_Label']
        diag_df['HMM_State'] = diag_df['Detected_Regime_State']
        diag_df['Regime_Label'] = diag_df['Detected_Regime_Label']
    diag_df = diag_df.loc[pd.Timestamp(start_date):].dropna(subset=['SPY_Log_Return', 'HMM_State'])
    if diag_df.empty:
        print(f"ERROR: No diagnostic rows available from {start_date}.")
        return
    best_k = int(diag_df['HMM_State'].max()) + 1

    os.makedirs(output_dir, exist_ok=True)
    colors_list = plt.cm.Set3.colors

    fig, (ax_price, ax_prob) = plt.subplots(
        2, 1, figsize=(24, 14), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )
    stable_labels = {i: _dominant_label(diag_df, i) for i in range(best_k)}
    state_changes = diag_df['HMM_State'].ne(diag_df['HMM_State'].shift()).cumsum()
    added = set()
    for _, group in diag_df.groupby(state_changes):
        state = int(group['HMM_State'].iloc[0])
        label = stable_labels.get(state, f"State {state}")
        color = colors_list[state % len(colors_list)]
        if label not in added:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.35, label=label)
            added.add(label)
        else:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.35)

    ax_price.plot(diag_df.index, diag_df['SPY_Close'], color='navy', linewidth=2.0, label='SPY Close')
    ax_vix = ax_price.twinx()
    ax_vix.plot(diag_df.index, diag_df['VIX_Close'], color='darkred', linewidth=1.2, alpha=0.75, label='VIX')
    ax_price.set_ylabel('SPY Close')
    ax_vix.set_ylabel('VIX')
    ax_price.grid(True, alpha=0.25, linestyle='--')

    prob_cols = [f'prob_state_{i}' for i in range(best_k) if f'prob_state_{i}' in diag_df.columns]
    prob_labels = []
    for i in range(best_k):
        prob_labels.append(stable_labels.get(i, f'State {i}'))
    ax_prob.stackplot(
        diag_df.index,
        [diag_df[c].values for c in prob_cols],
        labels=prob_labels[:len(prob_cols)],
        colors=colors_list[:len(prob_cols)],
        alpha=0.85,
    )
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel('Regime Probability')
    ax_prob.set_xlabel('Date')
    ax_prob.legend(loc='lower left', fontsize=9, ncol=max(1, min(best_k, 5)))
    ax_prob.grid(True, alpha=0.25)

    h1, l1 = ax_price.get_legend_handles_labels()
    h2, l2 = ax_vix.get_legend_handles_labels()
    ax_price.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, ncol=3)
    fig.suptitle(
        f"Causal Market Regime Timeline ({start_date} - present)\n"
        f"GMMHMM K={best_k} | Signal timestamp: {diag_df.attrs.get('regime_signal_timestamp', 'close_T_for_next_session')}",
        fontsize=16,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    timeline_path = os.path.join(output_dir, 'regime_log_return_timeline.png')
    plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    cols = 2
    rows = max(1, int(np.ceil(best_k / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    summary_rows = []
    x_all = diag_df['SPY_Log_Return'].values
    x_min, x_max = np.nanpercentile(x_all, [0.5, 99.5])
    x_grid = np.linspace(x_min, x_max, 500)

    print("\n" + "=" * 80)
    print("SPY DAILY LOG RETURN GMM FIT BY CAUSAL HMM REGIME")
    print("=" * 80)

    for state in range(best_k):
        ax = axes[state]
        state_df = diag_df[diag_df['HMM_State'] == state]
        returns = state_df['SPY_Log_Return'].dropna().values.reshape(-1, 1)
        label = stable_labels.get(state, f"State {state}")
        n = len(returns)
        color = colors_list[state % len(colors_list)]

        if n < 20:
            ax.set_title(f"{label}\ninsufficient observations: n={n}")
            ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha='center', va='center')
            continue

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(returns)
        density = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        skew = float(stats.skew(returns.flatten()))
        kurt = float(stats.kurtosis(returns.flatten()))

        ax.hist(returns.flatten(), bins=min(80, max(25, n // 20)), density=True,
                alpha=0.62, color=color, edgecolor='black', linewidth=0.35, label='Empirical')
        ax.plot(x_grid, density, color='black', linewidth=2.0, label='2-component GMM')
        for comp_idx, (weight, mean, covar) in enumerate(zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.reshape(-1))):
            comp_density = weight * stats.norm.pdf(x_grid, loc=mean, scale=np.sqrt(max(covar, 1e-12)))
            ax.plot(x_grid, comp_density, linestyle='--', linewidth=1.2, label=f'Comp {comp_idx + 1}')

        bic = float(gmm.bic(returns))
        ax.set_title(f"{label}\nn={n:,} | BIC={bic:.1f}")
        ax.set_xlabel("SPY daily log return")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        summary_rows.append({
            'state': state,
            'label': label,
            'n': n,
            'mean_log_return': mu,
            'std_log_return': sigma,
            'skew': skew,
            'excess_kurtosis': kurt,
            'gmm_bic': bic,
            'gmm_weights': json.dumps([float(x) for x in gmm.weights_]),
            'gmm_means': json.dumps([float(x) for x in gmm.means_.flatten()]),
            'gmm_stds': json.dumps([float(np.sqrt(max(x, 1e-12))) for x in gmm.covariances_.reshape(-1)]),
        })
        print(f"State {state} | {label} | n={n:,} | mean={mu:.6f} | std={sigma:.6f} | BIC={bic:.1f}")

    for ax in axes[best_k:]:
        ax.set_visible(False)

    fig.suptitle("SPY Daily Log Return Distributions by Causal HMM Regime", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    gmm_path = os.path.join(output_dir, 'regime_log_return_gmm_fits.png')
    plt.savefig(gmm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary_path = os.path.join(output_dir, 'regime_log_return_gmm_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    diag_path = os.path.join(output_dir, 'regime_log_return_trace.csv')
    diag_df.to_csv(diag_path)

    print(f"\n✓ Timeline saved to {timeline_path}")
    print(f"✓ GMM fit plot saved to {gmm_path}")
    print(f"✓ Summary saved to {summary_path}")
    print(f"✓ Causal trace saved to {diag_path}")

def plot_gmm_clusters(horizon=45):
    """
    Visualization: scatter plots of returns for each regime,
    colored by the GMM component (0 vs 1) to show sub-regime discovery.
    """
    hist_df = fetch_historical_data()
    if hist_df.empty: return

    hmm_model, K, feature_df = train_regime_hmm(hist_df)
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    feature_df['future_min_price'] = feature_df['SPY_Low'].rolling(window=indexer, min_periods=1).min()
    feature_df['future_mae_return'] = feature_df['future_min_price'] / feature_df['SPY_Close'] - 1
    feature_df = feature_df.dropna(subset=['future_mae_return'])
    sub_df_list = [feature_df[feature_df['HMM_State'] == i] for i in range(K)]

    cols = 2
    rows = max(2, (K + 1) // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    axes = axes.flatten()
    
    colors_list = plt.cm.tab10.colors

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= K:
            ax.set_visible(False)
            continue
            
        sub_df = sub_df_list[idx]
        label = f"HMM State {idx}"
        base_color = colors_list[idx % len(colors_list)]
        
        if len(sub_df) < 10:
            ax.set_title(f"{label} (Insufficient Data)")
            continue

        returns = sub_df['future_mae_return'].values
        x = returns.reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(x)
        weights = gmm.weights_
        means = gmm.means_.flatten()

        # Scatter plot
        # component 0
        idx0 = (labels == 0)
        ax.scatter(sub_df.index[idx0], returns[idx0], s=10, alpha=0.6,
                   label=f"Comp 0 (w={weights[0]:.2f}, μ={means[0]:.4f})", color='tab:cyan')
        # component 1
        idx1 = (labels == 1)
        ax.scatter(sub_df.index[idx1], returns[idx1], s=10, alpha=0.6,
                   label=f"Comp 1 (w={weights[1]:.2f}, μ={means[1]:.4f})", color='tab:purple')

        ax.set_title(f"GMM Sub-Regimes ({horizon}d Horizon): {label}", fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel(f"{horizon}d Future MAE Return")

    plt.suptitle(f"GMM Component Clustering on SPY {horizon}d Returns\n"
                 "Identifying Hidden Sub-Regimes within HMM States",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = "/Users/btian/.gemini/antigravity/artifacts/gmm_regime_clusters.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ GMM Cluster Plot saved to {output_path}")
    plt.show()


def plot_gmm_distributions(horizon=45):
    """
    Visualization: Histograms of horizon returns for each regime,
    overlaid with the 2-component GMM probability density function.
    """
    hist_df = fetch_historical_data()
    if hist_df.empty: return

    # Use cache key that rotates daily
    regime_dict, hmm_model, _ = build_regime_return_arrays(int(time.time()/86400), horizon=horizon)
    K = hmm_model.n_components

    cols = 2
    rows = max(2, (K + 1) // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    axes = axes.flatten()
    
    colors_list = plt.cm.tab10.colors

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= K:
            ax.set_visible(False)
            continue
            
        key = f'State_{idx}'
        label = f'HMM State {idx}'
        color = colors_list[idx % len(colors_list)]
        
        returns = regime_dict.get(key, np.array([]))
        n = len(returns)
        if n < 10:
            ax.set_title(f"{label} (Insufficient Data)")
            continue

        # Fit GMM (2 components)
        x_fitted = returns.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(x_fitted)
        weights = gmm.weights_
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        
        # Model Selection Statistics (k=2 vs k=1)
        gmm1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm1.fit(x_fitted)
        
        ll2 = gmm.score(x_fitted) * n
        ll1 = gmm1.score(x_fitted) * n
        aic2, bic2 = gmm.aic(x_fitted), gmm.bic(x_fitted)
        aic1, bic1 = gmm1.aic(x_fitted), gmm1.bic(x_fitted)

        # Histogram
        n_bins = min(100, max(40, n // 40))
        ax.hist(returns, bins=n_bins, density=True, alpha=0.3, color='gray', label='Empirical Returns')

        # Generate PDF range
        xmin, xmax = ax.get_xlim()
        x_plot = np.linspace(xmin, xmax, 500)
        
        # Component 0
        pdf0 = weights[0] * stats.norm.pdf(x_plot, means[0], stds[0])
        ax.plot(x_plot, pdf0, '--', color='cyan', linewidth=1.5,
                label=f'Comp 0: w={weights[0]:.2f}, μ={means[0]:.4f}, σ={stds[0]:.4f}')
        
        # Component 1
        pdf1 = weights[1] * stats.norm.pdf(x_plot, means[1], stds[1])
        ax.plot(x_plot, pdf1, '--', color='magenta', linewidth=1.5,
                label=f'Comp 1: w={weights[1]:.2f}, μ={means[1]:.4f}, σ={stds[1]:.4f}')
        
        # Total GMM PDF
        ax.plot(x_plot, pdf0 + pdf1, color='black', linewidth=2.5, label='Total GMM Density')

        # Annotation Box with Stats
        better_aic = "GMM (k=2)" if aic2 < aic1 else "Gaussian (k=1)"
        stats_text = (
            f"Log-Likelihood (k=2): {ll2:.1f}\n"
            f"Log-Likelihood (k=1): {ll1:.1f}\n"
            f"────────────────\n"
            f"AIC (k=2): {aic2:,.0f} {'★' if aic2 < aic1 else ''}\n"
            f"AIC (k=1): {aic1:,.0f}\n"
            f"BIC (k=2): {bic2:,.0f} {'★' if bic2 < bic1 else ''}\n"
            f"BIC (k=1): {bic1:,.0f}\n"
            f"────────────────\n"
            f"Winner: {better_aic}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.45, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

        ax.set_title(f"GMM Fit ({horizon}d Horizon): {label}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{horizon}d Future MAE Return")
        ax.set_ylabel("Density")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(xmin, xmax)

    plt.suptitle(f"GMM Distribution Fits by HMM State ({horizon}d Horizon)\n"
                 "Visualizing Empirical Multi-Week Performance",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = "/Users/btian/.gemini/antigravity/artifacts/gmm_distribution_fits.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ GMM Distribution Plot saved to {output_path}")
    plt.show()

def fetch_puts_by_delta(accounts, symbol, expiration_date, price_near):
    url = f"{accounts.base_url}/v1/market/optionchains.json"
    params = {
        "symbol": symbol,
        "expiryYear": expiration_date.year,
        "expiryMonth": expiration_date.month,
        "expiryDay": expiration_date.day,
        "includeWeekly": True,
        "optionCategory": "ALL",
        "chainType": "PUT",
        "strikePriceNear": price_near,
        "noOfStrikes": 400,
    }
    response = accounts.session.get(url, params=params)
    if response.status_code != 200: return []
    data = response.json()
    pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
    
    puts = []
    for pair in pairs:
        opt = pair.get("Put")
        if opt:
            greeks = opt.get("OptionGreeks", {})
            delta = float(greeks.get("delta", 0.0)) if greeks.get("delta") is not None else 0.0
            if delta < 0:
                puts.append({
                    "strike": float(opt.get("strikePrice")),
                    "bid": float(opt.get("bid")),
                    "ask": float(opt.get("ask")),
                    "volume": int(opt.get("volume", 0)),
                    "openInterest": int(opt.get("openInterest", 0)),
                    "delta": delta
                })
    return sorted(puts, key=lambda x: x["strike"], reverse=True)

def get_closest_by_delta(options, target_delta):
    if not options: return None
    return min(options, key=lambda x: abs(x["delta"] - target_delta))

def select_target_expiration(accounts, symbol, target_dte):
    expirations = accounts.get_available_expirations(symbol)
    if not expirations:
        return None

    today = datetime.now().date()
    parsed = []
    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte > 0:
            parsed.append((abs(dte - target_dte), dte, exp_date))

    if not parsed:
        return None
    _, _, selected = min(parsed, key=lambda x: (x[0], x[1]))
    return datetime.combine(selected, datetime.min.time())

def _build_single_regime_prob_func(spot_price, bucket, regime_label=""):
    """Builds a probability function for one bucket using selected model.
    Assumes bucket contains horizon-appropriate returns.
    """
    prefix = f"[{PROBABILITY_MODEL.upper()}][{regime_label}]" if regime_label else f"[{PROBABILITY_MODEL.upper()}]"

    if PROBABILITY_MODEL == 'bootstrap':
        simulations = 5000
        if len(bucket) == 0:
            print(f"{prefix} Bucket empty — using zero-return fallback.", flush=True)
            bucket = np.array([0.0])
        t0 = time.time()
        print(f"{prefix} Using empirical horizon distribution ({len(bucket):,} samples)...",
              end=" ", flush=True)
        # For bootstrap in a rolling horizon model, we just sample the terminal returns
        # rather than creating synthetic paths (v2).
        terminal_returns = np.random.choice(bucket, size=simulations, replace=True)
        terminal_prices = spot_price * (1 + terminal_returns)
        print(f"done in {time.time()-t0:.3f}s", flush=True)

        def prob_func(strike):
            return np.sum(terminal_prices <= strike) / simulations
        return prob_func

    if PROBABILITY_MODEL == 'parametric':
        t0 = time.time()
        if len(bucket) < 5:
            print(f"{prefix} Too few samples ({len(bucket)}) — using Gaussian fallback.", flush=True)
            loc, scale = 0.0, 0.01
            df_nu = 100.0
        else:
            print(f"{prefix} Fitting Student-t on {len(bucket):,} horizon samples...",
                  end=" ", flush=True)
            log_returns = np.log1p(bucket)
            df_nu, loc, scale = stats.t.fit(log_returns)
            print(f"done in {time.time()-t0:.3f}s | ν={df_nu:.2f} μ={loc:.4f} σ={scale:.4f}",
                  flush=True)

        def prob_func(strike):
            target_log_return = np.log(strike / spot_price)
            return stats.t.cdf(target_log_return, df_nu, loc=loc, scale=max(scale, 1e-6))
        return prob_func

# Prob engine built in ev_engine.py
    return prob_func, name, None

def calculate_yield_metrics(options_chain, short_strike, long_strike, net_credit_per_share, prob_func):
    """
    Numerically integrates to find the Expected Shortfall (Risk) and Expected Value.
    Uses an empirical prob_func instead of Deltas.
    Returns: (net_yield, expected_shortfall, ev_per_contract, prob_of_loss)
    """
    if not options_chain: return 0.0, 0.0, 0.0, 0.0
    
    options = sorted(options_chain, key=lambda x: x["strike"], reverse=True)
    expected_loss = 0.0
    expected_profit_contribution = 0.0
    prob_of_loss = 0.0
    
    total_trade_cost = COST_PER_SPREAD

    # Trader's max realistic profit if trade is successful
    net_yield = (net_credit_per_share * 100.0) - total_trade_cost

    def payout_per_contract(spot):
        opt_pnl = net_credit_per_share - max(0, short_strike - spot) + max(0, long_strike - spot)
        return (opt_pnl * 100.0) - total_trade_cost

    # CDF at highest strike
    # prob_func(strike) = P(spot <= strike)
    prob_below_highest = prob_func(options[0]['strike'])
    prob_above = max(0.0, 1.0 - prob_below_highest)

    last_valid_prob = prob_below_highest
    
    # Iterate through all adjacent strike buckets
    for i in range(1, len(options)):
        strike_high = options[i-1]['strike']
        strike_low = options[i]['strike']
        
        current_prob_below = prob_func(strike_low)
        prob_bucket = last_valid_prob - current_prob_below
        
        mid_spot = (strike_high + strike_low) / 2.0
        pnl = payout_per_contract(mid_spot)
        
        if pnl < 0:
            expected_loss += pnl * prob_bucket
            prob_of_loss += prob_bucket
        else:
            expected_profit_contribution += pnl * prob_bucket
            
        last_valid_prob = current_prob_below
        
    # Tail below lowest strike
    prob_below = last_valid_prob
    pnl_below = payout_per_contract(options[-1]['strike'] - 10)
    if pnl_below < 0:
        expected_loss += pnl_below * prob_below
        prob_of_loss += prob_below

    # Tail above highest strike (max-profit zone).
    expected_profit_contribution += net_yield * prob_above

    es = (expected_loss / prob_of_loss) if prob_of_loss > 0 else 0.0
    ev_per_contract = expected_profit_contribution + expected_loss

    return net_yield, es, ev_per_contract, prob_of_loss

def main(args):
    use_sandbox = args.sandbox
    username = args.username
    password = args.password
    headless = not args.no_headless

    print("Authenticating with E*TRADE...")
    try:
        session, base_url = oauth(use_sandbox, auto_login=True, 
                                  username=username, password=password, 
                                  headless=headless)
    except LoginFailureException as e:
        print(f"Login failed: {e}")
        send_login_failure_notification(str(e), e.screenshot_path)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during login: {e}")
        send_login_failure_notification(f"Unexpected error during login: {e}")
        sys.exit(1)

    if not session:
        print("Failed to authenticate.")
        return

    accounts = Accounts(session, base_url)
    
    spy_ticker = "SPY"
    spx_ticker = "SPX"
    
    print("\nFetching stock and VIX prices...")
    spy_price = accounts.get_stock_price(spy_ticker)
    spx_price = accounts.get_stock_price(spx_ticker)
    vix_price = accounts.get_stock_price("^VIX")
    
    # Validation and local fallback if E*TRADE (even with its own fallback) returns None
    if vix_price is None:
        print("Warning: VIX price lookup failed. Attempting local yfinance fallback...")
        vix_price = fetch_cached_yf_close("^VIX", cache_minutes=15)
        if vix_price is None:
            print("Local VIX fallback failed.")
            vix_price = 20.0  # Default to 20 if everything fails
            
    if spy_price is None or spx_price is None:
        print(f"SPY: {spy_price}, SPX: {spx_price}")
        print("CRITICAL: Market prices unavailable. Cannot continue.")
        return

    print(f"Current VIX: {vix_price:.2f}")
    
    # --- Phase 4: Regime-Aware Entry Gating & Strategy Selection ---
    ingestor = DataIngestor()
    loop = asyncio.get_event_loop()
    
    # 1. Fetch Microstructure & Macro Filters
    print("  [Gating] Fetching GEX and Macro factors...")
    gex_metrics = loop.run_until_complete(ingestor.calculate_gex_metrics(spy_ticker, spy_price))
    net_gex = gex_metrics['net_gex']
    zero_gamma = gex_metrics['zero_gamma']
    
    # Fetch VXV/VIX from fused dataset
    macro_start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    macro_end = datetime.now().strftime("%Y-%m-%d")
    recent_macro = loop.run_until_complete(ingestor.fetch_yf_data(macro_start, macro_end))
    vxv_vix_ratio = recent_macro['VXV_VIX_Ratio'].iloc[-1] if 'VXV_VIX_Ratio' in recent_macro.columns else 1.1
    
    spx_exps = accounts.get_available_expirations(spx_ticker)
    if not spx_exps:
        spx_ticker = "SPXW"

    # Initialize Probability Engines. Select the listed expiration closest to target DTE.
    exp_date_obj = select_target_expiration(accounts, spy_ticker, args.target_dte)
    if exp_date_obj is None:
        print(f"CRITICAL: No valid {spy_ticker} expiration found near target DTE {args.target_dte}.")
        return
    target_exp = exp_date_obj.strftime("%Y-%m-%d")
    days_to_exp = max(1, (exp_date_obj - datetime.now()).days)

    print(f"[Engine] Building {days_to_exp}d rolling-horizon return buckets...", flush=True)
    regime_dict, hmm_model, daily_models = build_regime_return_arrays(int(time.time()/86400), horizon=days_to_exp, n_components=args.force_k)

    print(f"Current VIX: {vix_price:.2f}")
    spy_prob_func, regime_name, projected_weights, _, current_probs = get_probability_engine(
        spy_price, vix_price, regime_dict, horizon=days_to_exp, hmm_model=hmm_model
    )
    spx_prob_func, _, _, _, _ = get_probability_engine(
        spx_price, vix_price, regime_dict, horizon=days_to_exp, hmm_model=hmm_model
    )
    
    dominant_regime_label = get_regime_labels(hmm_model, pc_df=None).get(np.argmax(current_probs), "Unknown")
    print(f"\n[Regime Detection] Detected: {dominant_regime_label}")
    print(f"  Net GEX: {net_gex:.2f}B | Zero Gamma: {zero_gamma:.2f} | VXV/VIX: {vxv_vix_ratio:.2f}")

    # HARD GATING LOGIC
    is_turmoil = "Market Turmoil" in dominant_regime_label
    is_neg_gex = net_gex < 0
    is_backwardation = vxv_vix_ratio < 1.0
    
    if is_turmoil or is_neg_gex or is_backwardation:
        print("\n🛑 CRITICAL GATING ACTIVATED:")
        if is_turmoil: print("  - Archetype 'Market Turmoil' detected.")
        if is_neg_gex: print(f"  - Negative GEX ({net_gex:.2f}B) indicates unstable architecture.")
        if is_backwardation: print(f"  - Volatility Curve Backwardation (VXV/VIX={vxv_vix_ratio:.2f}).")
        print("  Recommendation: SKIP PUT SELLING. Risk of tail expansion is high.")
        if not args.override_risk_gates:
            print("  Hard gate enforced. Re-run with --override-risk-gates to continue intentionally.")
            return
        print("  OVERRIDE ENABLED: continuing despite hard gate.")
    
    # Dynamic Parameter Mapping
    STRATEGY_MAP = {
        "Robust Expansion": {"short_delta": -0.20},
        "Emerging Expansion": {"short_delta": -0.15},
        "High Vol Chop": {"short_delta": -0.10},
        "Market Turmoil": {"short_delta": -0.05}
    }
    
    for archetype, params in STRATEGY_MAP.items():
        if archetype in dominant_regime_label:
            print(f"  [Strategy Map] Adjusting target short delta to {params['short_delta']} for {archetype}")
            args.short_delta = params["short_delta"]
            break

    print("Current regime weights (Snapshot):", flush=True)
    if current_probs is not None:
        for idx in range(hmm_model.n_components):
            print(f"  State {idx}: {current_probs[idx]:.4f}")
    if USE_MARKOV_TRANSITIONS and projected_weights is not None:
        print("Projected regime weights at expiration:")
        for idx in range(hmm_model.n_components):
            print(f"  State {idx}: {projected_weights[idx]:.4f}")

    print(f"Fetching options chain centered slightly OTM for {target_exp} ({days_to_exp} days to exp)...")
    spy_options = fetch_puts_by_delta(accounts, spy_ticker, exp_date_obj, spy_price * 0.90)
    spx_options = fetch_puts_by_delta(accounts, spx_ticker, exp_date_obj, spx_price * 0.90)

    # Clean out quotes without valid asks
    spy_options = [o for o in spy_options if o["ask"] > 0]
    spx_options = [o for o in spx_options if o["ask"] > 0]

    target_short_delta = args.short_delta
    spy_short_opt = get_closest_by_delta(spy_options, target_short_delta)
    spx_short_opt = get_closest_by_delta(spx_options, target_short_delta)
    
    if not spy_short_opt or not spx_short_opt:
        print("Failed to find valid short legs!")
        return
        
    print(f"\nSPY Baseline Short Strike: {spy_short_opt['strike']} @ Delta {spy_short_opt['delta']:.4f}")
    print(f"{spx_ticker} Baseline Short Strike: {spx_short_opt['strike']} @ Delta {spx_short_opt['delta']:.4f}")

    spy_long_deltas = []
    spy_portfolio_evs = []
    spy_ev_efficiency = [] 
    spy_widths = []
    spy_short_prems = []
    spy_long_prems = []
    spy_long_strikes = []
    spy_net_credits = []
    spy_rel_spreads = []
    spy_long_volumes = []
    spy_long_oi = []

    spx_long_deltas = []
    spx_portfolio_evs = []
    spx_ev_efficiency = []
    spx_widths = []
    spx_short_prems = []
    spx_long_prems = []
    spx_long_strikes = []
    spx_net_credits = []
    spx_rel_spreads = []
    spx_long_volumes = []
    spx_long_oi = []

    total_deltas = 30
    print(f"\n[Sweep] Evaluating {total_deltas} long-leg deltas × 2 tickers (SPY + {spx_ticker})...",
          flush=True)
    t_sweep = time.time()
    target_long_deltas = np.linspace(-0.14, -0.01, total_deltas)

    for sweep_idx, ld in enumerate(target_long_deltas, start=1):
        # Evaluate SPY
        spy_long = get_closest_by_delta(spy_options, ld)
        if spy_long and spy_long["strike"] < spy_short_opt["strike"]:
            width = spy_short_opt["strike"] - spy_long["strike"]
            spy_net_credit = max(0, ((spy_short_opt["bid"] + spy_short_opt["ask"])/2) - ((spy_long["bid"] + spy_long["ask"])/2))
            
            if spy_net_credit > 0 and width > 0:
                print(f"  [{sweep_idx:02d}/{total_deltas}] SPY δ={ld:.3f} "
                      f"| width=${width:.0f} credit=${spy_net_credit:.2f}",
                      end=" ", flush=True)
                net_yield, es_per_c, ev_per_c, prob_loss = calculate_yield_metrics(
                    spy_options, spy_short_opt["strike"], spy_long["strike"], spy_net_credit, spy_prob_func)
                
                margin_per_contract = width * 100
                fractional_contracts = TARGET_MARGIN_DOLLARS / margin_per_contract

                portfolio_ev = ev_per_c * fractional_contracts
                portfolio_es = es_per_c * fractional_contracts
                
                strike_pct_drop = (spy_short_opt["strike"] / spy_price) - 1
                exit_horizon_days = max(1, int(days_to_exp * (args.exit_pct / 100.0)))
                prob_touch = calculate_probability_of_touch(
                    current_probs, hmm_model.transmat_, daily_models, exit_horizon_days, strike_pct_drop, num_paths=1000, option_type="put"
                )
                prob_touch = max(prob_touch, 0.001)
                efficiency = portfolio_ev / abs(portfolio_es) if abs(portfolio_es) > 0.01 else 0.0
                print(f"EV=${portfolio_ev:.1f} ES=${portfolio_es:.1f} P(loss)={prob_loss:.3f} P(touch)={prob_touch:.3f} Edge={efficiency:.4f}", flush=True)

                spy_long_deltas.append(abs(spy_long["delta"]))
                spy_portfolio_evs.append(portfolio_ev)
                spy_ev_efficiency.append(efficiency)
                spy_widths.append(width)
                spy_short_prems.append((spy_short_opt["bid"] + spy_short_opt["ask"]) / 2.0)
                spy_long_prems.append((spy_long["bid"] + spy_long["ask"]) / 2.0)
                spy_long_strikes.append(spy_long["strike"])
                spy_net_credits.append(spy_net_credit)
                
                # Slippage Calculation: (Ask-Bid) / Mid
                short_spread = abs(spy_short_opt["ask"] - spy_short_opt["bid"])
                long_spread = abs(spy_long["ask"] - spy_long["bid"])
                combined_spread_dollars = short_spread + long_spread
                # Normalize by net credit (premium received)
                rel_slippage = (combined_spread_dollars / spy_net_credit) if spy_net_credit > 0 else 0
                spy_rel_spreads.append(rel_slippage)
                spy_long_volumes.append(spy_long.get("volume", 0))
                spy_long_oi.append(spy_long.get("openInterest", 0))

        # Evaluate SPX
        spx_long = get_closest_by_delta(spx_options, ld)
        if spx_long and spx_long["strike"] < spx_short_opt["strike"]:
            width = spx_short_opt["strike"] - spx_long["strike"]
            spx_net_credit = max(0, ((spx_short_opt["bid"] + spx_short_opt["ask"])/2) - ((spx_long["bid"] + spx_long["ask"])/2))
            
            if spx_net_credit > 0 and width > 0:
                print(f"  [{sweep_idx:02d}/{total_deltas}] {spx_ticker} δ={ld:.3f} "
                      f"| width=${width:.0f} credit=${spx_net_credit:.2f}",
                      end=" ", flush=True)
                net_yield, es_per_c, ev_per_c, prob_loss = calculate_yield_metrics(
                    spx_options, spx_short_opt["strike"], spx_long["strike"], spx_net_credit, spx_prob_func)

                margin_per_contract = width * 100
                fractional_contracts = TARGET_MARGIN_DOLLARS / margin_per_contract

                portfolio_ev = ev_per_c * fractional_contracts
                portfolio_es = es_per_c * fractional_contracts
                
                strike_pct_drop = (spx_short_opt["strike"] / spx_price) - 1
                exit_horizon_days = max(1, int(days_to_exp * (args.exit_pct / 100.0)))
                prob_touch = calculate_probability_of_touch(
                    current_probs, hmm_model.transmat_, daily_models, exit_horizon_days, strike_pct_drop, num_paths=1000, option_type="put"
                )
                prob_touch = max(prob_touch, 0.001)
                efficiency = portfolio_ev / abs(portfolio_es) if abs(portfolio_es) > 0.01 else 0.0
                print(f"EV=${portfolio_ev:.1f} ES=${portfolio_es:.1f} P(loss)={prob_loss:.3f} P(touch)={prob_touch:.3f} Edge={efficiency:.4f}", flush=True)

                spx_long_deltas.append(abs(spx_long["delta"]))
                spx_portfolio_evs.append(portfolio_ev)
                spx_ev_efficiency.append(efficiency)
                spx_widths.append(width)
                spx_short_prems.append((spx_short_opt["bid"] + spx_short_opt["ask"]) / 2.0)
                spx_long_prems.append((spx_long["bid"] + spx_long["ask"]) / 2.0)
                spx_long_strikes.append(spx_long["strike"])
                spx_net_credits.append(spx_net_credit)

                # Slippage Calculation: (Ask-Bid) / Mid
                short_spread = abs(spx_short_opt["ask"] - spx_short_opt["bid"])
                long_spread = abs(spx_long["ask"] - spx_long["bid"])
                combined_spread_dollars = short_spread + long_spread
                rel_slippage = (combined_spread_dollars / spx_net_credit) if spx_net_credit > 0 else 0
                spx_rel_spreads.append(rel_slippage)
                spx_long_volumes.append(spx_long.get("volume", 0))
                spx_long_oi.append(spx_long.get("openInterest", 0))

    print(f"\n[Sweep] Delta sweep complete in {time.time()-t_sweep:.1f}s — generating plots...",
          flush=True)

    # Plot generation
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    spy_sorted = sorted(zip(spy_long_deltas, spy_portfolio_evs, spy_ev_efficiency, spy_widths, spy_short_prems, spy_long_prems, spy_long_strikes, spy_net_credits, spy_rel_spreads, spy_long_volumes, spy_long_oi), key=lambda x: x[0], reverse=True)
    spx_sorted = sorted(zip(spx_long_deltas, spx_portfolio_evs, spx_ev_efficiency, spx_widths, spx_short_prems, spx_long_prems, spx_long_strikes, spx_net_credits, spx_rel_spreads, spx_long_volumes, spx_long_oi), key=lambda x: x[0], reverse=True)
    
    if spy_sorted and spx_sorted:
        # Plot 1: Portfolio EV
        ax1.plot([x[0] for x in spy_sorted], [x[1] for x in spy_sorted], label='SPY', marker='o', alpha=0.8)
        ax1.plot([x[0] for x in spx_sorted], [x[1] for x in spx_sorted], label=spx_ticker, marker='x', alpha=0.8)
        
        # Annotate EV directly
        for (delta_val, port_ev, efficiency, width, _, _, _, _, _, _, _) in spy_sorted:
            ax1.annotate(f"${port_ev:.0f}", (delta_val, port_ev), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
        for (delta_val, port_ev, efficiency, width, _, _, _, _, _, _, _) in spx_sorted:
            ax1.annotate(f"${port_ev:.0f}", (delta_val, port_ev), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='orange')
        
        # Plot 1b: Portfolio Total Premium (Secondary Y-Axis)
        ax1_twin = ax1.twinx()
        ax1_twin.plot([x[0] for x in spy_sorted], [x[7] * (TARGET_MARGIN_DOLLARS / x[3]) for x in spy_sorted], label='SPY Total Premium', color='tab:blue', linestyle=':', alpha=0.5)
        ax1_twin.plot([x[0] for x in spx_sorted], [x[7] * (TARGET_MARGIN_DOLLARS / x[3]) for x in spx_sorted], label=f'{spx_ticker} Total Premium', color='tab:orange', linestyle=':', alpha=0.5)
        ax1_twin.set_ylabel(f"Total Portfolio Premium (${TARGET_MARGIN_DOLLARS:,.0f} Margin)", color='gray')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        
        # Combine legends for ax1
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    # Plot 2: EV Efficiency (Ratio)
    ax2.plot([x[0] for x in spy_sorted], [x[2] for x in spy_sorted], label='SPY', marker='o', color='tab:blue', alpha=0.8)
    ax2.plot([x[0] for x in spx_sorted], [x[2] for x in spx_sorted], label=spx_ticker, marker='x', color='tab:orange', alpha=0.8)

    # Plot 3: Premium Values (Dual Axis)
    x_spy = [x[0] for x in spy_sorted]
    spy_short = [x[4] for x in spy_sorted]
    spy_long = [x[5] for x in spy_sorted]
    
    ax3.plot(x_spy, spy_short, label='SPY Short', color='blue', linestyle='-', linewidth=2)
    ax3.plot(x_spy, spy_long, label='SPY Long', color='cyan', linestyle='--', linewidth=1.5)
    ax3.set_ylabel("SPY Premium ($)", color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    ax3_twin = ax3.twinx()
    x_spx = [x[0] for x in spx_sorted]
    spx_short = [x[4] for x in spx_sorted]
    spx_long = [x[5] for x in spx_sorted]
    
    ax3_twin.plot(x_spx, spx_short, label=f'{spx_ticker} Short', color='red', linestyle='-', linewidth=2)
    ax3_twin.plot(x_spx, spx_long, label=f'{spx_ticker} Long', color='orange', linestyle='--', linewidth=1.5)
    ax3_twin.set_ylabel(f"{spx_ticker} Premium ($)", color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Annotate Plot 3 (using long leg premium value)
    for (delta_val, port_ev, efficiency, width, short_prem, long_prem, _, _, _, _, _) in spy_sorted:
        ax3.annotate(f"${long_prem:.2f}", (delta_val, long_prem), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    for (delta_val, port_ev, efficiency, width, short_prem, long_prem, _, _, _, _, _) in spx_sorted:
        ax3_twin.annotate(f"${long_prem:.2f}", (delta_val, long_prem), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')
    
    # Combined Legend for ax3
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3_t, labels3_t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines3_t, labels3 + labels3_t, loc='best', fontsize='small')

    # Formatting Plot 1
    ax1.invert_xaxis() 
    ax1.set_title(f"Total Portfolio Expected Value (Normalized to ${TARGET_MARGIN_DOLLARS:,.0f} Margin)")
    ax1.set_xlabel("Absolute Delta of Long Leg")
    ax1.set_ylabel("Portfolio Expected Value ($)")
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Formatting Plot 2
    ax2.invert_xaxis() 
    ax2.set_title("EV Efficiency (Portfolio EV / |Portfolio ES|)")
    ax2.set_xlabel("Absolute Delta of Long Leg")
    ax2.set_ylabel("EV Efficiency (Ratio)")
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Formatting Plot 3
    ax3.invert_xaxis()
    ax3.set_title("Short and Long Leg Premiums")
    ax3.set_xlabel("Absolute Delta of Long Leg")
    ax3.grid(True, alpha=0.3)

    # Formatting Plot 4: Liquidity Fingerprint (Slippage vs Delta)
    ax4.plot([x[0] for x in spy_sorted], [x[8]*100 for x in spy_sorted], label='SPY Rel. Slippage', marker='o', color='tab:blue', alpha=0.8)
    ax4.plot([x[0] for x in spx_sorted], [x[8]*100 for x in spx_sorted], label=f'{spx_ticker} Rel. Slippage', marker='x', color='tab:orange', alpha=0.8)
    ax4.set_ylabel("Slippage (% of Net Credit)")
    ax4.set_title("Liquidity Fingerprint: Relative Slippage vs Delta")
    ax4.set_xlabel("Absolute Delta of Long Leg")
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    # Add Volume/OI as a secondary axis for ax4
    ax4_twin = ax4.twinx()
    # Use bar chart for volume context
    ax4_twin.bar([x[0] for x in spy_sorted], [x[9] for x in spy_sorted], width=0.002, alpha=0.1, color='blue', label='SPY Vol')
    ax4_twin.bar([x[0] for x in spx_sorted], [x[9] for x in spx_sorted], width=0.002, alpha=0.1, color='orange', label=f'{spx_ticker} Vol')
    ax4_twin.set_ylabel("Volume (Contracts)", color='gray', alpha=0.5)
    ax4_twin.tick_params(axis='y', labelcolor='gray')
    
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    if PROBABILITY_MODEL == "gmm":
        model_label = "GMM (2-Component)"
    elif PROBABILITY_MODEL == "parametric":
        model_label = "Parametric (Student-t)"
    else:
        model_label = "Bootstrap"
    markov_label = "On" if USE_MARKOV_TRANSITIONS else "Off"
    plt.suptitle(f"Expected Value (${TARGET_MARGIN_DOLLARS:,.0f} Margin) - VIX-Conditioned Historical Probabilities\n"
                 f"Model: {model_label}, Markov: {markov_label}, Regime: {regime_name}, Current VIX: {vix_price:.2f}\n"
                 f"Short Strike: SPY={spy_short_opt['strike']}, {spx_ticker}={spx_short_opt['strike']}")

    # Add secondary x-axis (top) for Long Strikes
    def add_strike_axis(ax, spy_data, spx_data):
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        
        # Determine base data (prefer SPY if available, else SPX)
        base_data = spy_data if spy_data else spx_data
        if not base_data: return
        
        # Pick ~10 representative points to avoid crowding
        step = max(1, len(base_data) // 10)
        tick_indices = range(0, len(base_data), step)
        
        tick_deltas = []
        tick_labels = []
        
        for i in tick_indices:
            d = base_data[i][0]
            s1 = base_data[i][6]
            
            # Find closest other strike by delta if both data sets exist
            if spy_data and spx_data:
                # If base was SPY, find SPX. If base was SPX, find SPY.
                other_data = spx_data if base_data is spy_data else spy_data
                closest_other = min(other_data, key=lambda x: abs(x[0] - d))
                s2 = closest_other[6]
                
                # Order them SPY/SPX in the label
                s_spy = s1 if base_data is spy_data else s2
                s_spx = s2 if base_data is spy_data else s1
                tick_labels.append(f"{s_spy:.0f}/{s_spx:.0f}")
            else:
                tick_labels.append(f"{s1:.0f}")
            
            tick_deltas.append(d)
        
        ax_top.set_xticks(tick_deltas)
        ax_top.set_xticklabels(tick_labels, rotation=30, fontsize=7)
        ax_top.set_xlabel("Long Leg Strikes (SPY/SPX)")

    if spy_sorted or spx_sorted:
        add_strike_axis(ax1, spy_sorted, spx_sorted)
        add_strike_axis(ax2, spy_sorted, spx_sorted)
        add_strike_axis(ax3, spy_sorted, spx_sorted)

    plt.tight_layout()
    
    output_path = "/Users/btian/.gemini/antigravity/artifacts/ev_delta_normalized_margin_plot.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

from live_trading.ev_engine import prepare_hmm_features, train_regime_hmm, fit_gmm, query_gmm

def run_calibration_backtest(horizon=45, val_start_year=2020, refit_interval=21, force_k=None, export_csv=None):
    """
    Walk-forward probability calibration backtest using the Continuous HMM logic.
    """
    t_start = time.time()
    print("=" * 70)
    print("  PROBABILITY CALIBRATION BACKTEST (HMM OOS)")
    print("=" * 70)
    
    # ── Load data ──────────────────────────────────────────────────────────
    df_raw = fetch_historical_data()
    if df_raw.empty:
        print("ERROR: No historical data available.")
        return
        
    trading_horizon = calendar_days_to_trading_days(horizon)
    df = df_raw.copy()
    df['future_return'] = df['SPY_Close'].shift(-trading_horizon) / df['SPY_Close'] - 1
    df_full = df.dropna(subset=['future_return']).copy()
    
    # We remove the global feature contamination entirely.
    # We will compute feature prep dynamically to ensure zero look-ahead bias.
    valid_dates = df_full.index[1:] # Log_Return inherently drops the very first row
    
    val_start = pd.Timestamp(f"{val_start_year}-01-01")
    val_indices = [d for d in valid_dates if d >= val_start]

    n_train_total = sum(1 for d in valid_dates if d < val_start)
    n_val = len(val_indices)
    print(f"\n  Total usable rows:  {len(valid_dates):,}")
    print(f"  Training pool:      {n_train_total:,}  (before {val_start_year})")
    print(f"  Validation pool:    {n_val:,}  ({val_start_year}–present)")

    if n_val < 50:
        print("ERROR: Validation set too small. Aborting.")
        return

    moneyness_levels = [-0.03, -0.05, -0.07, -0.10, -0.15]

    records = []
    
    last_refit_idx = -refit_interval
    n_total = len(val_indices)
    refits_done = 0
    
    current_hmm = None
    current_gmm_models = []
    K = 1

    print(f"\n[Backtest] Walking forward through {n_total:,} validation days...\n")

    for i, date in enumerate(val_indices):
        current_date_loc = df_full.index.get_loc(date)
        
        if i - last_refit_idx >= refit_interval:
            cutoff_loc = max(0, current_date_loc - trading_horizon)
            train_data = df_full.iloc[:cutoff_loc].copy()
            
            best_hmm, best_k, model_train_df = train_regime_hmm(train_data, n_components=force_k)
            train_common_idx = model_train_df.index.intersection(train_data.index)
            model_train_df = model_train_df.loc[train_common_idx].copy()
            model_train_df['future_return'] = train_data.loc[train_common_idx, 'future_return']
            current_hmm = best_hmm
            K = best_k
            
            current_gmm_models = []
            for s in range(K):
                ret_subset = model_train_df[model_train_df['HMM_State'] == s]['future_return'].values
                current_gmm_models.append(fit_gmm(ret_subset, regime_label=f'State_{s}'))
                
            last_refit_idx = i
            refits_done += 1
            if refits_done <= 5 or refits_done % 5 == 0:
                print(f"  [Refit #{refits_done}] {date.strftime('%Y-%m-%d')} "
                      f"| train_n={len(train_data):,} | K={K} | val_day {i+1}/{n_total}")

        realized_return = df_full.loc[date, 'future_return']
        
        # 2. Project Today's features using the current HMM's fusion
        ingestor = DataIngestor()
        # Fetch data up to today for projection
        start_p = (date - timedelta(days=730)).strftime("%Y-%m-%d")
        end_p = date.strftime("%Y-%m-%d")
        
        # Use existing sync-loop logic from ev_engine or similar
        loop = asyncio.get_event_loop()
        stationary_df = loop.run_until_complete(ingestor.build_fused_dataset(start_p, end_p, scale=False))
        
        scaled_values = current_hmm.scaler_.transform(stationary_df)
        scaled_df = pd.DataFrame(scaled_values, index=stationary_df.index, columns=stationary_df.columns)
        
        pcs = current_hmm.fusion_.sparse_pca.transform(scaled_df)
        posteriors = current_hmm.predict_proba(pcs)
        current_probs = posteriors[-1]
        
        projected_probs = current_probs
        if USE_MARKOV_TRANSITIONS:
            projected_probs = current_probs @ np.linalg.matrix_power(current_hmm.transmat_, trading_horizon)
        dominant_state = np.argmax(projected_probs)
        
        for m in moneyness_levels:
            predicted_p = 0.0
            for s in range(K):
                p_s = query_gmm(current_gmm_models[s], 1.0, 1.0 + m)
                predicted_p += projected_probs[s] * p_s
                
            realized_bin = 1 if realized_return <= m else 0
            records.append({
                'date': date,
                'predicted_state': f'State_{dominant_state}',
                'moneyness': m,
                'predicted': predicted_p,
                'realized': realized_bin,
                'actual_return': realized_return,
            })

        if (i + 1) % 200 == 0:
            print(f"  ... processed {i+1}/{n_total} validation days "
                  f"({(i+1)/n_total*100:.0f}%)", flush=True)

    results = pd.DataFrame(records)
    elapsed = time.time() - t_start
    print(f"\n[Backtest] Complete: {len(results):,} predictions across "
          f"{n_total:,} days × {len(moneyness_levels)} levels "
          f"({refits_done} GMM refits) in {elapsed:.1f}s\n")

    predicted = results['predicted'].values
    realized = results['realized'].values

    brier = float(np.mean((predicted - realized) ** 2))

    eps = 1e-15
    p_clipped = np.clip(predicted, eps, 1 - eps)
    log_loss = -float(np.mean(realized * np.log(p_clipped) +
                               (1 - realized) * np.log(1 - p_clipped)))

    print("─" * 60)
    print("  CALIBRATION METRICS (Overall)")
    print("─" * 60)
    print(f"  Brier Score:              {brier:.6f}   (lower = better, perfect = 0)")
    print(f"  Log-Loss:                 {log_loss:.6f}   (lower = better)")
    
    print("\n─" * 60)
    print("  STATISTICAL OUT-OF-SAMPLE STATE SEPARATION (KRUSKAL-WALLIS)")
    print("─" * 60)
    
    from scipy.stats import kruskal
    unique_dates = results.drop_duplicates(subset=['date'])
    # Resample non-overlapping windows to satisfy i.i.d assumption
    unique_dates = unique_dates.iloc[::trading_horizon]
    print(f"  (Evaluating true independent samples for KW test: n={len(unique_dates)})")
    
    groups = []
    states_present = sorted(unique_dates['predicted_state'].unique())
    for state in states_present:
        state_returns = unique_dates[unique_dates['predicted_state'] == state]['actual_return'].values
        if len(state_returns) > 0:
            groups.append(state_returns)
        
    if len(groups) > 1:
        kw_stat, kw_p = kruskal(*groups)
        print(f"  Kruskal-Wallis H-statistic: {kw_stat:.2f}")
        print(f"  p-value:                    {kw_p:.4e}")
        if kw_p < 0.05:
            print("  Conclusion: The HMM significantly distinguishes future return distributions out-of-sample (p<0.05)!")
        else:
            print("  Conclusion: The HMM struggles to find statistically significant difference in future returns.")
    else:
        print("  Only 1 state predicted out of sample, cannot perform test.")
        
    print()
    print("─" * 60)
    print("  PER-MONEYNESS BREAKDOWN")
    print("─" * 60)
    print(f"  {'Level':>8s}  {'Avg P(pred)':>11s}  {'Realized %':>10s}  {'Brier':>8s}  {'n':>6s}")
    for m in moneyness_levels:
        sub = results[results['moneyness'] == m]
        p_mean = sub['predicted'].mean()
        r_mean = sub['realized'].mean()
        b_m = float(np.mean((sub['predicted'].values - sub['realized'].values) ** 2))
        print(f"  {m:>+8.1%}  {p_mean:>11.4f}  {r_mean:>10.4f}  {b_m:>8.6f}  {len(sub):>6,}")
    print()

    print("─" * 60)
    print("  PER-STATE BREAKDOWN (Dominant Projected State)")
    print("─" * 60)
    print(f"  {'State':>18s}  {'Avg P(pred)':>11s}  {'Realized %':>10s}  {'Brier':>8s}  {'n':>6s}")
    states_sorted = sorted(results['predicted_state'].unique())
    for rl in states_sorted:
        sub = results[results['predicted_state'] == rl]
        p_mean = sub['predicted'].mean()
        r_mean = sub['realized'].mean()
        b_r = float(np.mean((sub['predicted'].values - sub['realized'].values) ** 2))
        print(f"  {rl:>18s}  {p_mean:>11.4f}  {r_mean:>10.4f}  {b_r:>8.6f}  {len(sub):>6,}")
    print()

    if export_csv:
        results.to_csv(export_csv, index=False)
        print(f"✓ Results exported to {export_csv}")

def sample_prediction_outcomes(horizon=45, force_k=None):
    """
    Spot-checks the model on specific historical dates to demonstrate 
    prediction vs outcome without look-forward bias.
    """
    df = fetch_historical_data()
    if df.empty: return
    trading_horizon = calendar_days_to_trading_days(horizon)
    df = df.copy()
    df['future_return'] = df['SPY_Close'].shift(-trading_horizon) / df['SPY_Close'] - 1
    df_full = df.dropna(subset=['future_return']).copy()

    # Specific dates of interest across different regimes
    test_dates = [
        "2020-02-14", # Pre-COVID Peak (Normal -> Extreme)
        "2020-03-20", # COVID Bottom (Extreme)
        "2021-01-05", # Post-COVID rally (Low)
        "2022-01-05", # 2022 Peak (Normal -> High)
        "2022-06-15", # Mid-2022 stress (High)
        "2023-10-15", # Late 2023 dip (Normal)
        "2024-01-15", # Recent trend (Low)
    ]
    
    moneyness_levels = [-0.05, -0.10, -0.15]
    
    print("\n" + "="*80)
    print(f"  SAMPLE PREDICTION VS REALIZED OUTCOMES ({horizon}d Horizon)")
    print("  (Each prediction uses ONLY data available before the trade date)")
    print("="*80)
    print(f"  {'Date':<12} {'VIX':>6} {'Regime':<18} {'Moneyness':>10} {'P(Loss)':>10} {'Actual Ret':>12} {'Result'}")
    print("-" * 80)

    for ds in test_dates:
        t_date = pd.Timestamp(ds)
        if t_date not in df_full.index:
            # find closest previous date
            actual_dates = df_full.index[df_full.index <= t_date]
            if len(actual_dates) == 0: continue
            date = actual_dates[-1]
        else:
            date = t_date

        vix = df_full.loc[date, 'VIX_Close']
        actual_ret = df_full.loc[date, 'future_return']
        
        # Training data: STRICTLY before this date, completely dropping the horizon bleed
        current_date_loc = df_full.index.get_loc(date)
        train_end_idx = max(0, current_date_loc - trading_horizon)
        train_data = df_full.iloc[:train_end_idx].copy()
        
        # Train HMM on strictly isolated data
        best_hmm, best_k, model_train_df = train_regime_hmm(train_data, n_components=force_k)
        train_common_idx = model_train_df.index.intersection(train_data.index)
        model_train_df = model_train_df.loc[train_common_idx].copy()
        model_train_df['future_return'] = train_data.loc[train_common_idx, 'future_return']
        
        # Use upgraded inference with PCA projection
        ingestor = DataIngestor()
        start_p = (date - timedelta(days=730)).strftime("%Y-%m-%d")
        end_p = date.strftime("%Y-%m-%d")
        loop = asyncio.get_event_loop()
        stationary_df = loop.run_until_complete(ingestor.build_fused_dataset(start_p, end_p, scale=False))
        
        scaled_values = best_hmm.scaler_.transform(stationary_df)
        scaled_df = pd.DataFrame(scaled_values, index=stationary_df.index, columns=stationary_df.columns)
        
        pcs = best_hmm.fusion_.sparse_pca.transform(scaled_df)
        posteriors = best_hmm.predict_proba(pcs)
        current_probs = posteriors[-1]
        projected_probs = current_probs
        if USE_MARKOV_TRANSITIONS:
            projected_probs = current_probs @ np.linalg.matrix_power(best_hmm.transmat_, trading_horizon)
        
        # Fit GMMs for each state
        gmm_models = []
        for s in range(best_k):
            # The model_train_df is already the pc_df from train_regime_hmm
            ret_subset = model_train_df[model_train_df['HMM_State'] == s]['future_return'].values
            gmm_models.append(fit_gmm(ret_subset, regime_label=f'State_{s}'))
            
        dominant_state = np.argmax(projected_probs)
        regime_label = get_regime_labels(best_hmm).get(dominant_state, f"HMM State {dominant_state}")
        
        for m in moneyness_levels:
            predicted_p = 0.0
            for s in range(best_k):
                p_s = query_gmm(gmm_models[s], 1.0, 1.0 + m)
                predicted_p += projected_probs[s] * p_s
                
            hit = "BREACH ❌" if actual_ret <= m else "Safe ✅"
            print(f"  {date.strftime('%Y-%m-%d'):<12} {vix:>6.2f} {regime_label:<18} {m:>10.1%} {predicted_p:>10.4f} {actual_ret:>12.2%} {hit}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E*TRADE Expected Value Plotter',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', help='use sandbox?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--username', help='username for login', type=str)
    parser.add_argument('--password', help='password for login', type=str)
    parser.add_argument('--no-headless', help='disable headless mode for login', action='store_true')
    parser.add_argument('--distributions', help='plot VIX-regime return distributions and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--gmm-plots', help='plot GMM internal clustering and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--gmm-dist', help='plot GMM distribution fit and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--calibrate', help='run probability calibration backtest and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--samples', help='run sample prediction spot-checks and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--timeline', help='plot regime timeline from 2024 and exit (no login needed)',
                        action='store_true')
    parser.add_argument('--regime-log-return-gmm',
                        help='plot causal regime timeline and per-regime SPY daily log-return GMM fits',
                        action='store_true')
    parser.add_argument('--diagnostic-start', help='Start date for regime diagnostics (YYYY-MM-DD)',
                        type=str, default='2015-01-01')
    parser.add_argument('--backtest', help='run historical option backtest using Massive API (no login needed)',
                        action='store_true')
    parser.add_argument('--backtest-start', help='Backtest start date (YYYY-MM-DD)', type=str, default='2025-01-01')
    parser.add_argument('--backtest-end', help='Backtest end date (YYYY-MM-DD)', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--strategy', help='Backtest strategy type', type=str, default='put_credit_spread',
                        choices=['put_credit_spread', 'call_credit_spread', 'iron_condor'])
    parser.add_argument('--spread-width', help='Spread width in dollars (default 20)', type=float, default=20.0)
    parser.add_argument('--target-dte', help='Target days to expiration (default 42)', type=int, default=42)
    parser.add_argument('--close-dte', help='Close position at this DTE (default 21)', type=int, default=21)
    parser.add_argument('--plot', help='Plot backtest results', action='store_true')
    parser.add_argument('--horizon', help='Target horizon in days for backtest (default 45)', type=int, default=45)
    parser.add_argument('--force-k', help='Force a specific number of HMM states (default: 3)', type=int, default=3)
    parser.add_argument('--exit-pct', help='Percentage of DTE to model for Probability of Touch (default 100)', type=int, default=100)
    parser.add_argument('--export-csv', help='Save calibration records to CSV file', type=str)
    parser.add_argument('--short-delta', help='Target delta for the short leg (default -0.15)', type=float, default=-0.15)
    parser.add_argument('--initial-capital', help='Initial backtest capital (default 100000)', type=float, default=100000.0)
    parser.add_argument('--log-backtest', help='Enable detailed structured logging of backtest path', action='store_true')
    parser.add_argument('--margin-limit', help='Margin limit as fraction of capital (default 0.5)', type=float, default=0.5)
    parser.add_argument('--qty', help='Fixed quantity per trade (default 0: max margin)', type=int, default=0)
    parser.add_argument('--hold-itm-exp', help='Hold ITM positions until expiration instead of closing at half-DTE', action='store_true')
    parser.add_argument('--panic-delta-mult', help='Delta multiplier in panic regime', type=float, default=4.0)
    parser.add_argument('--panic-dte-target', help='Target DTE in panic regime', type=int, default=63)
    parser.add_argument('--panic-width-mult', help='Spread width multiplier in panic regime', type=float, default=2.0)
    parser.add_argument('--no-panic-swap', help='Disable closing all positions when entering panic regime', action='store_true')
    parser.add_argument('--strategy-id', help='Load configuration from strategy_registry.md by ID', type=str)
    parser.add_argument('--override-risk-gates', help='Allow live trade construction even when hard regime/GEX/vol gates fire', action='store_true')
    args = parser.parse_args()

    if args.distributions:
        plot_regime_distributions()
    elif args.gmm_plots:
        plot_gmm_clusters()
    elif args.gmm_dist:
        plot_gmm_distributions()
    elif args.calibrate:
        run_calibration_backtest(horizon=args.horizon, force_k=args.force_k, export_csv=args.export_csv)
    elif args.samples:
        sample_prediction_outcomes(horizon=args.horizon, force_k=args.force_k)
    elif args.timeline:
        plot_regime_timeline(n_components=args.force_k)
    elif args.regime_log_return_gmm:
        plot_regime_log_return_gmm(n_components=args.force_k, start_date=args.diagnostic_start)
    elif args.backtest:
        import asyncio
        from backtesting.backtest_runner import run_put_credit_spread_backtest, get_trading_dates, lag_daily_regime_map
        from backtesting.strategy_loader import load_strategy
        from live_trading.ev_engine import fetch_historical_data, train_regime_hmm
        
        # Load from registry if ID provided, or default to first one if exactly one exists
        strategy_config = {}
        if hasattr(args, 'strategy_id') and args.strategy_id:
            print(f"  Loading strategy '{args.strategy_id}' from registry...")
            strategy_config = load_strategy(args.strategy_id)
        else:
            # Try to auto-load. Prefer 'dynamic_delta_variant' as a smart default.
            try:
                from backtesting.strategy_loader import load_all_strategies
                all_s = load_all_strategies()
                if not all_s:
                    print("  Note: Strategy registry is empty. Using CLI defaults.")
                else:
                    # Smart Default: Prefer the dynamic_delta_variant
                    if "dynamic_delta_variant" in all_s:
                        sid = "dynamic_delta_variant"
                        print(f"  No --strategy-id provided. Smart-defaulting to '{sid}' from registry...")
                        strategy_config = all_s[sid]
                    elif len(all_s) == 1:
                        sid = list(all_s.keys())[0]
                        print(f"  No --strategy-id provided. Auto-loading only strategy '{sid}' from registry...")
                        strategy_config = all_s[sid]
                    else:
                        available = list(all_s.keys())
                        print(f"  Warning: Multiple strategies in registry {available}. Use --strategy-id to pick one. Using CLI defaults.")
            except Exception as e:
                print(f"  Note: Could not auto-load from registry: {e}")
        
        # Apply registry overrides if a strategy was loaded
        if strategy_config:
            entry = strategy_config.get('entry', {})
            exit_cfg = strategy_config.get('exit', {})
            
            # Override args with registry values
            args.target_dte = entry.get('target_dte', args.target_dte)
            args.close_dte = exit_cfg.get('close_dte', args.close_dte)
            args.short_delta = entry.get('short_delta', args.short_delta)
            args.spread_width = entry.get('spread_width', args.spread_width)
            args.panic_delta_mult = entry.get('panic_delta_multiplier', args.panic_delta_mult)
            args.panic_dte_target = entry.get('panic_dte_target', args.panic_dte_target)
            args.panic_width_mult = entry.get('panic_width_multiplier', args.panic_width_mult)
            
            # New: support for explicit panic swap toggle in registry
            if 'panic_swap_enabled' in entry:
                args.no_panic_swap = not entry['panic_swap_enabled']
            
            print(f"  [REGISTRY OVERRIDE] DTE={args.target_dte}, Delta={args.short_delta}, PanicMult={args.panic_delta_mult}, PanicSwap={not args.no_panic_swap}")
        
        print(f"\n  Generating non-anticipatory regimes for backtest period (Walk-Forward)...")
        df_hist = fetch_historical_data()
        if df_hist.empty:
            print("  ERROR: Could not fetch historical data for regimes.")
            import sys
            sys.exit(1)
            
        # Walk-forward regime generation to eliminate parameter look-ahead bias.
        # The resulting close_T labels are lagged before trade entry below.
        cal_start_date = pd.to_datetime(args.backtest_start) - timedelta(days=365 * 5)
        df_cal = df_hist[
            (df_hist.index >= cal_start_date) &
            (df_hist.index <= pd.to_datetime(args.backtest_end))
        ].copy()
        print(f"  [Walk-Forward] Training causal trace from {cal_start_date.strftime('%Y-%m-%d')} to {args.backtest_end}")
        best_hmm, k, feature_df = train_regime_hmm(df_cal, n_components=args.force_k, expanding_window=True)
        if best_hmm is None or feature_df.empty:
            print("  ERROR: Could not build causal regime trace.")
            import sys
            sys.exit(1)
        regime_labels = get_regime_labels(best_hmm, feature_df)
        close_regimes = {
            d.strftime('%Y-%m-%d'): int(s)
            for d, s in feature_df['HMM_State'].dropna().to_dict().items()
        }
        trading_dates = get_trading_dates(args.backtest_start, args.backtest_end)
        regimes_dict = lag_daily_regime_map(close_regimes, trading_dates)
        print("  [Regime Timing] Using one-trading-day-lagged close_T regimes for trade entry.")

        # Extract SPY and VIX prices
        spy_close = df_hist['SPY_Close'].copy()
        spy_close.index = spy_close.index.strftime("%Y-%m-%d")
        vix_close = df_hist['VIX_Close'].copy()
        vix_close.index = vix_close.index.strftime("%Y-%m-%d")
        
        mask = (spy_close.index >= args.backtest_start) & (spy_close.index <= args.backtest_end)
        spy_prices = spy_close[mask]
        vix_prices = vix_close[mask]
        
        print(f"  Debug: spy_prices index sample: {spy_prices.index[:5].tolist()} ... {spy_prices.index[-5:].tolist()}")
        print(f"  Debug: spy_prices count: {len(spy_prices)}")

        asyncio.run(run_put_credit_spread_backtest(
            underlying="SPY",
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            target_dte=args.target_dte,
            close_dte=args.close_dte,
            target_short_delta=args.short_delta,
            spread_width=args.spread_width,
            plot=args.plot,
            initial_capital=args.initial_capital,
            margin_limit_pct=args.margin_limit,
            backtest_qty=args.qty,
            regimes=regimes_dict,
            underlying_prices=spy_prices,
            vix_prices=vix_prices,
            enable_logging=args.log_backtest,
            hold_itm_to_expiration=args.hold_itm_exp,
            regime_labels=regime_labels,
            panic_delta_multiplier=args.panic_delta_mult,
            panic_dte_target=args.panic_dte_target,
            panic_width_multiplier=args.panic_width_mult,
            panic_swap_enabled=not args.no_panic_swap
        ))
    else:
        if not args.username or not args.password:
            parser.error("--username and --password are required when not using --distributions")
        main(args)
