"""
Constant-Delta (20 Delta) OTM Call IV for RKLB — 5 Year History
==============================================================
Methodology:
  - Enumerate all monthly (3rd-Friday) option expirations for the past 5 years.
  - For each historical trading day, pick the expiration closest to 30 DTE.
  - Fetch OTM call strikes and find the one whose Delta is closest to 0.20.
  - Compute constant-delta (20d) IV series.
  - Overlay 20-day Realized Volatility and save plot to research_reports/.
"""

import asyncio
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date

from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache
from backtesting.greeks_calculator import implied_volatility, bs_call_delta

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "RKLB"
LOOKBACK_YEARS  = 5
TARGET_DTE      = 30       # constant-maturity target in calendar days
TARGET_DELTA    = 0.20     # 20 Delta OTM Call
RISK_FREE_RATE  = 0.045
DIVIDEND_YIELD  = 0.0
STRIKE_INCREMENT = 0.5     # RKLB typically has $0.50 strikes

OUTPUT_PATH = (
    "/Users/btian/EtradePythonClient/etrade_python_client"
    "/research_reports/rklb_20delta_call_iv.png"
)

plt.style.use('dark_background')


# ── Helper: enumerate 3rd-Friday monthly expirations ─────────────────────────
def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given month/year."""
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def all_monthly_exps(lookback_years: int) -> list[str]:
    """Return sorted list of 3rd-Friday expiration strings for past N years."""
    today = date.today()
    start = date(today.year - lookback_years, today.month, 1)
    exps = []
    y, m = start.year, start.month
    while date(y, m, 1) <= today:
        exp = third_friday(y, m)
        if exp > today:
            exps.append(exp.strftime("%Y-%m-%d"))
            if len(exps) > 0 and exp > today + timedelta(days=45):
                break
        else:
            exps.append(exp.strftime("%Y-%m-%d"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return sorted(exps)


# ── Helper: strike increment logic ───────────────────────────────────────────
def strike_increment(spot: float) -> float:
    if spot < 5: return 0.5
    if spot < 25: return 1.0
    return 2.5


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    # 1. Underlying data
    print(f"Downloading {TICKER} spot data ({LOOKBACK_YEARS}Y)...")
    raw = yf.download(TICKER, period=f"{LOOKBACK_YEARS + 1}y", interval="1d",
                      auto_adjust=True)
    if raw.empty:
        # RKLB was a SPAC, IPO was around 2021
        raw = yf.download(TICKER, start="2021-01-01", interval="1d", auto_adjust=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw['returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
    raw['rv_20d']  = raw['returns'].rolling(20).std() * np.sqrt(252)

    data = raw.copy()
    print(f"  Underlying rows: {len(data)}  ({data.index[0].date()} → {data.index[-1].date()})")

    # 2. Monthly expirations
    all_exps = all_monthly_exps(LOOKBACK_YEARS)
    print(f"  Monthly expirations to cover: {len(all_exps)}")

    # 3. Fetch option bars
    cache  = OptionDataCache("backtest_cache/option_data.db")
    client = MassiveAPIClient(cache)

    start_str = data.index[0].strftime("%Y-%m-%d")
    end_str   = data.index[-1].strftime("%Y-%m-%d")

    exp_bars: dict[str, dict[float, pd.DataFrame]] = {}

    async with client:
        for exp in all_exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            ref_date = exp_dt - timedelta(days=TARGET_DTE)
            ref_ts   = pd.Timestamp(ref_date)

            # Find spot around target window
            close_candidates = data.index[data.index <= ref_ts]
            if close_candidates.empty:
                ref_spot = float(data['Close'].iloc[0])
            else:
                ref_spot = float(data.loc[close_candidates[-1], 'Close'])
            
            # Build a strike ladder centered on the historically appropriate spot.
            inc = strike_increment(ref_spot)
            atm = round(ref_spot / inc) * inc
            candidate_strikes = [atm + i * inc for i in range(0, 16)]

            yy, mm, dd = exp[2:4], exp[5:7], exp[8:10]
            contracts_to_fetch = []
            for s in candidate_strikes:
                strike_fmt = f"{int(s * 1000):08d}"
                contracts_to_fetch.append({
                    "option_ticker": f"O:{TICKER}{yy}{mm}{dd}C{strike_fmt}",
                    "strike": float(s)
                })
            
            # Batch fetch
            await client.fetch_chain_ohlcv_batch(
                contracts_to_fetch, start_str, end_str,
                TICKER, "call", exp
            )

            # Post-process into exp_bars
            exp_bars[exp] = {}
            for c in contracts_to_fetch:
                ticker = c["option_ticker"]
                strike = c["strike"]
                bars = cache.get_ohlcv_range(ticker, start_str, end_str)
                if bars:
                    df = pd.DataFrame(bars)
                    df['pricing_date'] = pd.to_datetime(df['pricing_date']).dt.tz_localize(None)
                    df = df.set_index('pricing_date').sort_index()
                    exp_bars[exp][strike] = df

            fetched = len(exp_bars[exp])
            if fetched:
                print(f"  {exp}: {fetched} strikes with data")

    # 4. Build constant-delta (20d) IV series
    print("\nBuilding 20-Delta IV series...")
    cm_rows = []

    for dt, urow in data.iterrows():
        spot = float(urow['Close'])
        if np.isnan(spot): continue

        # Pick closest expiration to 30 DTE
        best_exp = None
        best_dte_diff = float('inf')
        for exp in all_exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_dt - dt.to_pydatetime()).days
            if dte < 5: continue
            diff = abs(dte - TARGET_DTE)
            if diff < best_dte_diff:
                best_dte_diff = diff
                best_exp = exp

        if best_exp is None: continue

        exp_dt = datetime.strptime(best_exp, "%Y-%m-%d")
        T = max((exp_dt - dt.to_pydatetime()).days / 365.0, 1e-5)

        strikes_avail = exp_bars.get(best_exp, {})
        best_strike = None
        best_delta_diff = float('inf')
        best_iv = None
        best_vol = 0.0

        for s, df in strikes_avail.items():
            if dt not in df.index: continue
            bar = df.loc[dt]
            mid = float(bar.get('close', np.nan))
            if np.isnan(mid) or mid <= 0: continue
            
            iv = implied_volatility(mid, spot, s, T, RISK_FREE_RATE, DIVIDEND_YIELD, "call")
            if iv is None or np.isnan(iv) or iv <= 0: continue
            
            delta = bs_call_delta(spot, s, T, RISK_FREE_RATE, iv, DIVIDEND_YIELD)
            delta_diff = abs(delta - TARGET_DELTA)
            
            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_strike = s
                best_iv = iv
                best_vol = float(bar.get('volume', 0))

        if best_iv is not None:
            cm_rows.append({
                'date': dt,
                'iv_20d': best_iv,
                'strike': best_strike,
                'spot': spot,
                'rv_20d': float(urow['rv_20d']),
                'volume': best_vol,
                'exp': best_exp
            })

    if not cm_rows:
        print("ERROR: No data found.")
        return

    cm_df = pd.DataFrame(cm_rows).set_index('date').sort_index()
    print(f"Series built: {len(cm_df)} rows")

    # 5. Stats
    curr_iv = cm_df['iv_20d'].iloc[-1]
    curr_rv = cm_df['rv_20d'].iloc[-1]
    iv_pct = (cm_df['iv_20d'] < curr_iv).mean()
    iv_mean = cm_df['iv_20d'].mean()

    print(f"\n── Summary (RKLB 20-Delta Call) ──")
    print(f"  Current IV: {curr_iv:.1%}")
    print(f"  Current RV: {curr_rv:.1%}")
    print(f"  IV Percentile: {iv_pct:.0%}")
    print(f"  Avg IV: {iv_mean:.1%}")

    # 6. Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios':[2,2,1]})
    fig.patch.set_facecolor('#0a0a0a')
    for ax in axes: ax.set_facecolor('#0d0d0d')

    # Panel 1: Spot
    axes[0].plot(data.index, data['Close'], color='#00e6ff', label='RKLB Spot')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].set_title('RKLB Spot Price')

    # Panel 2: 20-Delta IV vs RV
    axes[1].plot(cm_df.index, cm_df['iv_20d'], color='#ff00ff', label='20-Delta Call IV')
    axes[1].plot(cm_df.index, cm_df['rv_20d'], color='#ffff00', linestyle='--', label='20D Realized Vol')
    axes[1].fill_between(cm_df.index, cm_df['iv_20d'], cm_df['rv_20d'], where=(cm_df['iv_20d'] > cm_df['rv_20d']), color='red', alpha=0.1)
    axes[1].fill_between(cm_df.index, cm_df['iv_20d'], cm_df['rv_20d'], where=(cm_df['iv_20d'] <= cm_df['rv_20d']), color='green', alpha=0.1)
    axes[1].set_ylabel('Volatility')
    axes[1].legend()
    axes[1].set_title('RKLB 20-Delta Call IV vs Realized Vol')

    # Panel 3: Volume
    axes[2].bar(cm_df.index, cm_df['volume'], color='#ffa500', alpha=0.7)
    axes[2].set_ylabel('Volume')
    axes[2].set_title('Contract Volume')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Plot saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
