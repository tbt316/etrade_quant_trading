import os
import sys
import argparse
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Add quant-review-0430 to sys.path to access the remediated modules
QUANT_REVIEW_DIR = os.path.join(os.getcwd(), "quant-review-0430")
if QUANT_REVIEW_DIR not in sys.path:
    sys.path.insert(0, QUANT_REVIEW_DIR)

# Now import from the quant-review-0430 directory
from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm, get_regime_labels, build_regime_return_arrays, save_regime_cache
from live_trading.pca_fusion import PCAFusion

def setup_plot_style():
    """Sets a premium, dark-themed plotting style."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial'],
        'axes.facecolor': '#121212',
        'figure.facecolor': '#0a0a0a',
        'axes.grid': True,
        'grid.color': '#333333',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.edgecolor': '#444444',
        'axes.labelcolor': '#cccccc',
        'xtick.color': '#888888',
        'ytick.color': '#888888',
        'legend.facecolor': '#1e1e1e',
        'legend.edgecolor': '#333333',
        'legend.fontsize': 10
    })

def run_sync(coro):
    """Helper to run async coroutines from sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # This shouldn't happen in this script's structure
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return loop.run_until_complete(coro)

def run_regime_detection(start_date, end_date, n_components=None, save_results=False):

    print(f"\n🚀 [Regime Detector] Starting analysis for period: {start_date} to {end_date}")
    
    ingestor = DataIngestor()
    
    # 1. High-Dimensional Data Ingestion
    print("📡 Fetching feature set (Macro + Price + Vol)...")
    # build_fused_dataset is async, run it sync
    stationary_df = run_sync(ingestor.build_fused_dataset(start_date, end_date, scale=False))
    
    if stationary_df.empty:
        print("❌ Error: Failed to fetch data.")
        return
    
    # 2. Raw data for plotting (SPY and VIX)
    print("📊 Fetching raw price data for visualization...")
    raw_df = ingestor.fetch_yf_data(start_date, end_date)
    
    # 3. Training HMM with expanding window to eliminate look-ahead bias
    print(f"🧠 Training Gaussian HMM (Causal Walk-Forward)...")
    best_hmm, best_k, feature_df = train_regime_hmm(stationary_df, n_components=n_components, expanding_window=True)
    
    if best_hmm is None:
        print("❌ Error: HMM training failed.")
        return

    # 4. Calculate posterior probabilities (predict_proba)
    pc_cols = [c for c in feature_df.columns if c.startswith('PC')]
    features_pc = feature_df[pc_cols].values
    
    n_samples = len(features_pc)
    all_probs = np.zeros((n_samples, best_k))
    print(f"🔮 Calculating causal probabilities for {n_samples} samples...")
    for t in range(n_samples):
        all_probs[t] = best_hmm.predict_proba(features_pc[:t+1])[-1]
    
    for i in range(best_k):
        feature_df[f'prob_state_{i}'] = all_probs[:, i]
    
    feature_df['Dominant_State'] = np.argmax(all_probs, axis=1)
    
    # 5. Empirical Labeling (Safer than mathematical inversion)
    plot_df = feature_df.join(raw_df[['SPY_Close', 'VIX_Close']], how='inner')
    plot_df['SPY_Log_Return'] = np.log(plot_df['SPY_Close'] / plot_df['SPY_Close'].shift(1))
    
    regime_labels = {}
    print("🏷️  Calculating empirical regime labels...")
    for i in range(best_k):
        state_mask = plot_df['Dominant_State'] == i
        if not state_mask.any():
            continue
            
        median_vix = plot_df.loc[state_mask, 'VIX_Close'].median()
        # Annualized mean return, handle potential NaNs
        rets = plot_df.loc[state_mask, 'SPY_Log_Return'].dropna()
        mean_return = rets.mean() * 252 if not rets.empty else 0
        
        # REFINED THRESHOLDS: More aggressive Turmoil and realistic Expansion boundaries
        if median_vix > 25:
            name = "Market Turmoil"
        elif median_vix > 18:
            if mean_return < -0.05:
                name = "Cautious Decline"
            else:
                name = "High Vol Chop"
        elif mean_return > 0.05 and median_vix < 15:
            name = "Robust Expansion"
        elif mean_return > 0 and median_vix < 20:
            name = "Emerging Expansion"
        else:
            name = f"Regime {i}"
            
        regime_labels[i] = f"{name} ({i})"
        print(f"  • State {i}: VIX={median_vix:.1f}, Return={mean_return*100:.1f}% -> {name}")
    
    # 5. Plotting
    print("🎨 Generating Regime Analysis Dashboard...")
    setup_plot_style()
    
    fig, (ax_price, ax_prob) = plt.subplots(2, 1, figsize=(18, 12), 
                                            gridspec_kw={'height_ratios': [2, 1]}, 
                                            sharex=True)
    
    # FIX: Use integer index for plotting to eliminate weekend gaps
    plot_df = plot_df.reset_index().rename(columns={'index': 'Date'})
    plot_df['index_int'] = plot_df.index
    
    x_coords = plot_df['index_int']
    
    # Colormap and colors
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, best_k)]
    
    # Draw Regime Backgrounds
    state_changes = plot_df['Dominant_State'].ne(plot_df['Dominant_State'].shift()).cumsum()
    groups = plot_df.groupby(state_changes)
    
    added_to_legend = set()
    for _, group in groups:
        state = group['Dominant_State'].iloc[0]
        color = colors[state]
        label = regime_labels.get(state, f"Regime {state}")
        
        start_idx = group['index_int'].iloc[0]
        end_idx = group['index_int'].iloc[-1]
        
        if label not in added_to_legend:
            ax_price.axvspan(start_idx, end_idx, color=color, alpha=0.3, label=label)
            added_to_legend.add(label)
        else:
            ax_price.axvspan(start_idx, end_idx, color=color, alpha=0.3)
    
    ax_price.plot(x_coords, plot_df['SPY_Close'], color='#00e676', linewidth=2, label='SPY Price')
    ax_price.set_ylabel('SPY Price ($)', color='#00e676', fontsize=12, fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor='#00e676')
    
    ax_vix = ax_price.twinx()
    ax_vix.plot(x_coords, plot_df['VIX_Close'], color='#ff5252', linewidth=1, alpha=0.7, label='VIX Index')
    ax_vix.set_ylabel('VIX Index', color='#ff5252', fontsize=12, fontweight='bold')
    ax_vix.tick_params(axis='y', labelcolor='#ff5252')
    ax_vix.grid(False)
    
    ax_price.set_title(f'Market Regime Timeline Analysis (K={best_k})', fontsize=16, pad=20, fontweight='bold')
    ax_price.legend(loc='upper left', framealpha=0.8)
    
    # Format X-axis with Date Labels
    n_ticks = 10
    tick_indices = np.linspace(0, len(plot_df) - 1, n_ticks, dtype=int)
    tick_labels = [plot_df['Date'].iloc[i].strftime('%Y-%m') for i in tick_indices]
    ax_prob.set_xticks(tick_indices)
    ax_prob.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    prob_data = [plot_df[f'prob_state_{i}'].values for i in range(best_k)]
    labels = [regime_labels.get(i, f'State {i}') for i in range(best_k)]
    
    ax_prob.stackplot(x_coords, prob_data, labels=labels, colors=colors, alpha=0.7)
    ax_prob.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax_prob.set_ylim(0, 1)
    ax_prob.legend(loc='lower left', ncol=min(3, best_k), framealpha=0.8)
    ax_prob.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    output_fn = f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_fn, dpi=200, bbox_inches='tight')
    print(f"✅ Dashboard saved to: {output_fn}")
    
    # 6. Optional: Save to cache for use by etrade_cover_call_new.py
    if save_results:
        print("💾 Saving regime data to cache for live trading engine...")
        # build_regime_return_arrays does the return grouping and GMM fitting
        # and it will now use the current date to key the results
        regime_dict, model, daily_models = build_regime_return_arrays(int(datetime.now().timestamp()/86400), horizon=7, force_refit=True)
        if model:
            save_regime_cache(regime_dict, model, daily_models)
        else:
            print("❌ Failed to generate regime return arrays for saving.")

    print("\n" + "="*50)
    print("🔍 REGIME ENGINE VALIDITY CHECK")
    print("="*50)
    print(f"  • Date Range: {start_date} to {end_date}")
    print(f"  • Optimal K (BIC): {best_k}")
    print(f"  • Feature Set: {len(stationary_df.columns)} indicators")
    
    transmat = best_hmm.transmat_
    persistence = np.diag(transmat)
    print(f"  • Avg Regime Persistence: {np.mean(persistence):.2%}")
    for i, p in enumerate(persistence):
        label = regime_labels.get(i, f"State {i}")
        print(f"    - {label}: {p:.2%}")
    
    print("="*50)
    print("PRO TIP: Review the stacked probability chart to verify state stability.")
    print("Frequent 'flickering' between states suggests overfitting or noisy features.")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Regime Detection Review Script")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d"),
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--k", type=int, default=None,
                        help="Force number of HMM states (optional)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to cache for live trading agent")
    
    args = parser.parse_args()
    
    run_regime_detection(args.start, args.end, n_components=args.k, save_results=args.save)

