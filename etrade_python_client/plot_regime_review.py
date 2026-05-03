import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def setup_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.facecolor': '#121212',
        'figure.facecolor': '#0a0a0a',
        'grid.color': '#333333',
        'grid.alpha': 0.5,
    })

def plot_review():
    output_dir = "regime_review_2025"
    results_df = pd.read_csv(os.path.join(output_dir, "regime_results.csv"), index_col=0, parse_dates=True)
    levels_df = pd.read_csv(os.path.join(output_dir, "raw_price_levels.csv"), index_col=0, parse_dates=True)
    
    # Merge for plotting
    plot_df = results_df.join(levels_df[['SPY_Close', 'VIX_Close']], how='inner')
    
    setup_plot_style()
    fig, (ax_price, ax_prob) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Plot Price
    ax_price.plot(plot_df.index, plot_df['SPY_Close'], color='#00e676', linewidth=2, label='SPY Price')
    ax_price.set_ylabel('SPY Price ($)', color='#00e676')
    
    ax_vix = ax_price.twinx()
    ax_vix.plot(plot_df.index, plot_df['VIX_Close'], color='#ff5252', alpha=0.4, linewidth=1, label='VIX Index')
    ax_vix.set_ylabel('VIX Index', color='#ff5252')
    
    # Color Background by Regime
    labels = plot_df['Regime_Label'].unique()
    # Use a nice colormap
    cmap = plt.get_cmap('tab10')
    label_to_color = {label: cmap(i) for i, label in enumerate(labels)}
    
    # Handle contiguous regime blocks for fill_between
    plot_df['Regime_Int'] = pd.factorize(plot_df['Regime_Label'])[0]
    regime_changes = plot_df['Regime_Int'].ne(plot_df['Regime_Int'].shift()).cumsum()
    
    added_to_legend = set()
    for _, group in plot_df.groupby(regime_changes):
        label = group['Regime_Label'].iloc[0]
        color = label_to_color[label]
        if label not in added_to_legend:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.25, label=label)
            added_to_legend.add(label)
        else:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.25)
    
    ax_price.set_title("Market Regime Trace Review (2025-01-01 to 2025-06-01)", fontsize=18, fontweight='bold', pad=20)
    ax_price.legend(loc='upper left', framealpha=0.8)
    
    # Plot Probabilities
    prob_cols = [c for c in plot_df.columns if c.startswith('prob_state_')]
    if prob_cols:
        # Map state IDs to their semantic labels for the legend
        # We find the label corresponding to each state ID in the prob_cols
        state_id_to_label = {}
        for idx in range(len(labels)):
             state_id = int(labels[idx].split('(')[-1].strip(')'))
             state_id_to_label[state_id] = labels[idx]
        
        legend_labels = []
        for col in prob_cols:
            state_id = int(col.split('_')[-1])
            legend_labels.append(state_id_to_label.get(state_id, col))
            
        ax_prob.stackplot(plot_df.index, plot_df[prob_cols].values.T, labels=legend_labels, alpha=0.7)
        ax_prob.set_ylabel("Regime Probability", fontsize=12)
        ax_prob.set_ylim(0, 1)
        ax_prob.legend(loc='lower left', ncol=min(4, len(prob_cols)), framealpha=0.8)
        
    # Format dates
    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_trace_plot.png"), dpi=200)
    print(f"✅ Review plot saved to {output_dir}/regime_trace_plot.png")

if __name__ == "__main__":
    plot_review()
