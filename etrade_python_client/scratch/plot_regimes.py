import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from live_trading.data_ingestion import DataIngestor

def plot_regime_time_domain():
    print("📈 Generating Regime Time Domain Plot...")
    
    # 1. Load Regime Data
    df_regimes = pd.read_csv("scratch/regime_analysis_2020_2026_global.csv", index_col=0, parse_dates=True)
    
    # 2. Fetch SPY Price Data for the same period
    ingestor = DataIngestor()
    start_date = df_regimes.index.min().strftime("%Y-%m-%d")
    end_date = df_regimes.index.max().strftime("%Y-%m-%d")
    df_prices = ingestor.fetch_yf_data(start_date, end_date)
    
    if df_prices.empty:
        print("❌ Failed to fetch price data for plotting.")
        return

    # Align data
    common_idx = df_regimes.index.intersection(df_prices.index)
    df_regimes = df_regimes.loc[common_idx]
    df_prices = df_prices.loc[common_idx]

    # 3. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Define colors for regimes (0-4)
    # State 0 (Low Vol) -> Green
    # State 1 -> Light Green/Cyan
    # State 2 -> Yellow/Orange
    # State 3 (High Vol) -> Red
    # State 4 (Extreme Vol) -> Dark Red/Magenta
    colors = {0: '#2ecc71', 1: '#3498db', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c'}
    
    # --- Top Panel: SPY Price ---
    ax1.plot(df_prices.index, df_prices['SPY_Close'], color='black', alpha=0.7, label='SPY Close')
    ax1.set_title('SPY Price with Market Regimes (2020-2026)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('SPY Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Highlight backgrounds based on regime
    for state_id in sorted(df_regimes['HMM_State'].unique()):
        mask = df_regimes['HMM_State'] == state_id
        # We need to fill segments. Using fill_between is tricky for discontinuous blocks.
        # A better way is to iterate through changes.
        
    # Efficient way to draw background blocks
    state_changes = df_regimes['HMM_State'].diff().ne(0).cumsum()
    for _, group in df_regimes.groupby(state_changes):
        start_date = group.index[0]
        end_date = group.index[-1]
        state = int(group['HMM_State'].iloc[0])
        label = group['Regime_Label'].iloc[0]
        ax1.axvspan(start_date, end_date, color=colors.get(state, 'grey'), alpha=0.2)

    # --- Bottom Panel: HMM States ---
    ax2.step(df_regimes.index, df_regimes['HMM_State'], where='post', color='navy', linewidth=1.5)
    ax2.set_ylabel('HMM State', fontsize=12)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_ylim(-0.5, 4.5)
    ax2.grid(True, axis='y', alpha=0.5)
    
    # Format X-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=0)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=4, label=f"State {i}") for i, c in colors.items()]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plot_path = "scratch/regime_time_domain_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_regime_time_domain()
