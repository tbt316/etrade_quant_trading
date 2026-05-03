import os
import sys
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add quant-review-0430 to sys.path
QUANT_REVIEW_DIR = os.path.join(os.getcwd(), "quant-review-0430")
if QUANT_REVIEW_DIR not in sys.path:
    sys.path.insert(0, QUANT_REVIEW_DIR)

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm, get_regime_labels
from live_trading.pca_fusion import PCAFusion

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.facecolor': '#121212',
        'figure.facecolor': '#0a0a0a',
        'grid.color': '#333333',
        'grid.alpha': 0.5,
        'axes.edgecolor': '#444444'
    })

def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def generate_audit_report(start_date, end_date):
    setup_plot_style()
    os.makedirs("audit_plots", exist_ok=True)
    ingestor = DataIngestor()
    
    print("Step 1: Fetching Raw Data...")
    fred_raw = ingestor.fetch_fred_data(start_date, end_date)
    yf_raw = ingestor.fetch_yf_data(start_date, end_date)
    
    print(f"  YFinance Data: {yf_raw.shape}")
    print(f"  FRED Data: {fred_raw.shape}")
    
    if yf_raw.empty:
        print("❌ Error: YFinance data is empty. Check internet or cache.")
        return

    # Combine data
    combined = pd.concat([yf_raw, fred_raw], axis=1).ffill()
    
    # Clean up empty columns (if any)
    missing_ratios = combined.isna().mean()
    bad_cols = missing_ratios[missing_ratios > 0.8].index
    if not bad_cols.empty:
        print(f"  Dropping sparse features (>80% NaN): {list(bad_cols)}")
        combined = combined.drop(columns=bad_cols)
    
    # Drop rows only where core features (SPY/VIX) are missing
    core_cols = ['SPY_Close', 'VIX_Close']
    existing_core = [c for c in core_cols if c in combined.columns]
    raw_df = combined.dropna(subset=existing_core)
    
    print(f"  Merged & Cleaned Data Columns: {raw_df.columns.tolist()}")
    
    if raw_df.empty:
        print("❌ Error: Merged dataset is empty after cleaning.")
        return
    
    # Plot 1: Raw Data
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    
    # Subplot 0: Price (SPY + BTC)
    axes[0].plot(raw_df.index, raw_df['SPY_Close'], color='#00e676', label='SPY Price', linewidth=2)
    axes[0].set_ylabel('SPY Price ($)', color='#00e676', fontweight='bold')
    axes[0].set_title('Raw Price Action: Equity vs Crypto Liquidity', fontsize=14)
    
    if 'BTC_Close' in raw_df.columns:
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(raw_df.index, raw_df['BTC_Close'], color='#ff9800', label='BTC-USD', linewidth=1.5, alpha=0.8)
        ax0_twin.set_ylabel('BTC Price ($)', color='#ff9800', fontweight='bold')
        ax0_twin.grid(False)
        # Combine legends
        lines, labels = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax0_twin.get_legend_handles_labels()
        axes[0].legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        print("⚠️ Warning: BTC_Close not found in raw_df!")
        axes[0].legend(loc='upper left')

    # Subplot 1: Volatility
    axes[1].plot(raw_df.index, raw_df['VIX_Close'], color='#ff5252', label='VIX')
    if 'VVIX_Close' in raw_df.columns:
        axes[1].plot(raw_df.index, raw_df['VVIX_Close'], color='#ff4081', label='VVIX', alpha=0.6)
    axes[1].set_ylabel('Volatility Index', fontweight='bold')
    axes[1].set_title('Volatility Surface', fontsize=14)
    axes[1].legend(loc='upper right')

    # Subplot 2: Macro Yields
    yield_cols = [c for c in raw_df.columns if 'Treasury' in c or 'Rate' in c]
    for col in yield_cols:
        axes[2].plot(raw_df.index, raw_df[col], label=col)
    axes[2].set_ylabel('Percent (%)', fontweight='bold')
    axes[2].set_title('Macroeconomic Yields (Rates)', fontsize=14)
    axes[2].legend(loc='upper right')

    # Subplot 3: Credit Spreads
    spread_cols = [c for c in raw_df.columns if 'Spread' in c or 'Curve' in c]
    for col in spread_cols:
        axes[3].plot(raw_df.index, raw_df[col], label=col)
    axes[3].set_ylabel('Spread Value', fontweight='bold')
    axes[3].set_title('Credit & Term Spreads', fontsize=14)
    axes[3].legend(loc='upper right')
    
    # Add timestamp to prove update
    fig.text(0.01, 0.01, f"Audit Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
             fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig("audit_plots/01_raw_data.png", dpi=150)
    plt.close(fig)
    
    print("Step 2: Processing Stationarity...")
    stationary_df = ingestor.ensure_stationarity(raw_df)
    
    # Plot 2: Stationary Data (Z-Scores for comparison)
    # Use expanding window scaling to avoid global leakage in visualization
    print("  Applying Causal Scaling for Heatmap...")
    scaled_stationary = ingestor.scale_features(stationary_df, expanding=True, warmup=min(252, len(stationary_df)-1))
    scaled_stationary = scaled_stationary.dropna()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(scaled_stationary.T, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Robust Scaled Value'})
    ax.set_title('Feature Heatmap after Fractional Differencing & Scaling')
    plt.savefig("audit_plots/02_stationary_heatmap.png")
    
    print("Step 3: PCA Fusion...")
    fusion = PCAFusion()
    pc_df = fusion.fit_transform(scaled_stationary)
    loadings = fusion.get_loadings_table()
    
    # Plot 3: PCA Loadings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
    ax1.set_title('Sparse PCA Loadings (Feature Contribution to Factors)')
    
    # Explained Variance
    from sklearn.decomposition import PCA
    std_pca = PCA().fit(scaled_stationary)
    ax2.bar(range(1, len(std_pca.explained_variance_ratio_) + 1), std_pca.explained_variance_ratio_, alpha=0.7)
    ax2.step(range(1, len(std_pca.explained_variance_ratio_) + 1), np.cumsum(std_pca.explained_variance_ratio_), where='mid', color='red')
    ax2.set_title('Scree Plot (Standard PCA Proxy for Variance Retention)')
    ax2.set_ylabel('Explained Variance Ratio')
    plt.savefig("audit_plots/03_pca_analysis.png")
    
    print("Step 4: HMM Training & State Centroids...")
    # Train HMM with expanding_window=True to eliminate look-ahead bias
    best_hmm, best_k, final_df = train_regime_hmm(stationary_df, expanding_window=True)
    # Labels are now in final_df['Regime_Label'] point-in-time
    
    # Plot 4: HMM State Clusters in PCA Space
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='HMM_State', palette='viridis', alpha=0.5, ax=ax)
    
    # Plot Centroids
    for i in range(best_k):
        # Calculate centroid based on FINAL model for cluster visualization
        centroid = np.sum(best_hmm.weights_[i][:, np.newaxis] * best_hmm.means_[i], axis=0)
        ax.scatter(centroid[0], centroid[1], marker='X', s=200, color='red', edgecolor='white')
        
        # Get label from the last known state archetype
        last_label = final_df[final_df['HMM_State'] == i]['Regime_Label'].iloc[-1] if not final_df[final_df['HMM_State'] == i].empty else f"State {i}"
        ax.annotate(last_label, (centroid[0], centroid[1]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    ax.set_title('HMM State Clusters in Latent Factor Space (PC1 vs PC2)')
    plt.savefig("audit_plots/04_hmm_clusters.png")
    
    print("Step 5: Final Regime Timeline...")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(final_df.index, final_df['PC1'], color='white', alpha=0.3, label='PC1 (Market Trend)')
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, best_k)]
    state_changes = final_df['HMM_State'].ne(final_df['HMM_State'].shift()).cumsum()
    groups = final_df.groupby(state_changes)
    for _, group in groups:
        state = int(group['HMM_State'].iloc[0])
        ax.axvspan(group.index[0], group.index[-1], color=colors[state], alpha=0.4)
        
    ax.set_title('Final Causal Regime Classification over PC1 Factor')
    plt.savefig("audit_plots/05_final_timeline.png")
    
    # Export Data for Review Package
    export_path = "quant-review-0430/data/regime_audit_2015_2020.csv"
    final_df.to_csv(export_path)
    print(f"✅ Audit data exported to: {export_path}")
    
    print("\n✅ Audit complete. Plots saved to audit_plots/")
    print(f"  • Final K: {best_k}")
    if 'PC1' in loadings.columns:
        print(f"  • Top PC1 Features: {loadings['PC1'].abs().sort_values(ascending=False).head(3).index.tolist()}")
    if 'PC2' in loadings.columns:
        print(f"  • Top PC2 Features: {loadings['PC2'].abs().sort_values(ascending=False).head(3).index.tolist()}")

if __name__ == "__main__":
    start = "2015-01-01"
    end = "2020-01-01"
    generate_audit_report(start, end)
