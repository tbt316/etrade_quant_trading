import asyncio
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache
from backtesting.greeks_calculator import implied_volatility

# Configuration
TICKER = "WOLF"
EXPIRATIONS = ["2027-01-15", "2028-01-21"]
STRIKES = [15, 20, 25, 30, 35, 40, 45, 50]
LOOKBACK_DAYS = 180
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0

# Styling
plt.style.use('dark_background')
COLOR_SPOT = '#00e6ff'
COLOR_OPT = '#39ff14'
COLOR_IV = '#ff00ff'
COLOR_RV = '#ffff00'
COLOR_INT = '#4d4dff'
COLOR_EXT = '#808080'

async def main():
    # 1. Initialize
    cache = OptionDataCache("backtest_cache/option_data.db")
    client = MassiveAPIClient(cache)
    os.makedirs("audit_plots", exist_ok=True)

    # 2. Fetch Underlying
    print(f"Fetching underlying data for {TICKER}...")
    wolf_data = yf.download(TICKER, period="1y", interval="1d")
    if wolf_data.empty: return
    if isinstance(wolf_data.columns, pd.MultiIndex): wolf_data.columns = wolf_data.columns.get_level_values(0)
    wolf_data.index = pd.to_datetime(wolf_data.index).tz_localize(None)
    
    # Calculate 20D Realized Vol
    wolf_data['returns'] = np.log(wolf_data['Close'] / wolf_data['Close'].shift(1))
    wolf_data['rv_20d'] = wolf_data['returns'].rolling(window=20).std() * np.sqrt(252)

    # 3. Fetch Option Data
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    start_date_str = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    async with client:
        all_results = []
        for exp in EXPIRATIONS:
            for strike in STRIKES:
                yy, mm, dd = exp[2:4], exp[5:7], exp[8:10]
                strike_fmt = f"{int(strike * 1000):08d}"
                option_ticker = f"O:{TICKER}{yy}{mm}{dd}C{strike_fmt}"
                
                bars = await client.fetch_contract_daily_bars(option_ticker, start_date_str, end_date_str, TICKER, "call", float(strike), exp)
                if not bars: continue
                
                df_opt = pd.DataFrame(bars)
                df_opt['pricing_date'] = pd.to_datetime(df_opt['pricing_date']).dt.tz_localize(None)
                df_opt = df_opt.set_index('pricing_date')
                
                merged = df_opt.join(wolf_data[['Close', 'rv_20d']].rename(columns={'Close':'spot_price'}), how='inner')
                
                ivs = []
                for dt, row in merged.iterrows():
                    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                    T = max((exp_dt - dt.to_pydatetime()).days / 365.0, 1e-5)
                    iv = implied_volatility(row['close'], row['spot_price'], strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, "call")
                    ivs.append(iv)
                
                merged['iv'] = ivs
                merged['strike'] = strike
                merged['expiration'] = exp
                all_results.append(merged)

    if not all_results: return
    combined = pd.concat(all_results)
    
    # Select contract with MOST history
    target_exp, target_strike = combined.groupby(['expiration', 'strike']).size().idxmax()
    subset = combined[(combined['expiration'] == target_exp) & (combined['strike'] == target_strike)].copy()
    
    if not subset.empty:
        fig, (ax1, ax3, ax4) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Prices
        ax1.plot(subset.index, subset['spot_price'], color=COLOR_SPOT, linewidth=2, label=f'{TICKER} Spot')
        ax2 = ax1.twinx()
        ax2.plot(subset.index, subset['close'], color=COLOR_OPT, linewidth=2, label='Call Premium')
        ax1.set_ylabel('WOLF Price ($)', color=COLOR_SPOT, fontweight='bold')
        ax2.set_ylabel('Option Premium ($)', color=COLOR_OPT, fontweight='bold')
        ax1.grid(True, alpha=0.1)
        
        # Volatility
        ax3.plot(subset.index, subset['iv'], color=COLOR_IV, linewidth=2, label='Implied Vol (IV)')
        ax3.plot(subset.index, subset['rv_20d'], color=COLOR_RV, linestyle='--', alpha=0.7, label='Realized Vol (20D)')
        ax3.set_ylabel('Volatility', fontweight='bold')
        ax3.legend(loc='upper left', frameon=False)
        ax3.grid(True, alpha=0.1)
        
        # Intrinsic/Extrinsic
        subset['intrinsic'] = (subset['spot_price'] - target_strike).clip(lower=0)
        subset['extrinsic'] = subset['close'] - subset['intrinsic']
        ax4.stackplot(subset.index, subset['intrinsic'], subset['extrinsic'], 
                     labels=['Intrinsic', 'Extrinsic (Time)'], alpha=0.4, colors=[COLOR_INT, COLOR_EXT])
        ax4.set_ylabel('Value ($)', fontweight='bold')
        ax4.legend(loc='upper left', frameon=False)
        
        fig.suptitle(f"{TICKER} LEAP Audit: {target_exp} ${target_strike} Call", fontsize=18, fontweight='bold', color='white', y=0.98)
        plot_path = "/Users/btian/EtradePythonClient/etrade_python_client/audit_plots/wolf_leap_analysis.png"
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        
        # Summary
        print(f"\nEvaluation for {TICKER} {target_exp} ${target_strike} Call:")
        print(f"  Spot: ${subset['spot_price'].iloc[-1]:.2f}")
        print(f"  Premium: ${subset['close'].iloc[-1]:.2f}")
        print(f"  IV: {subset['iv'].iloc[-1]:.1%}")
        print(f"  RV (20D): {subset['rv_20d'].iloc[-1]:.1%}")
        if subset['rv_20d'].iloc[-1] > subset['iv'].iloc[-1]:
            print("  CONVICTION: Realized Volatility > Implied Volatility. LEAPs are fundamentally CHEAP.")
        else:
            print("  CAUTION: Implied Volatility > Realized Volatility. You are paying a premium for expected future vol.")

if __name__ == "__main__":
    asyncio.run(main())
