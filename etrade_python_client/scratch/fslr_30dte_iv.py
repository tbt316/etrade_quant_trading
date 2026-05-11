"""
Constant-Maturity 30-DTE ATM IV for First Solar (FSLR) — 5 Year History
=================================================================
Methodology:
  - Enumerate all monthly (3rd-Friday) option expirations for the past 5 years.
  - For each historical trading day, pick the expiration closest to 30 DTE.
  - Find the ATM strike (based on spot price that day) and fetch its closing bar.
  - Compute Black-Scholes IV from the bar price → builds a clean CM-30D IV series.
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
from backtesting.greeks_calculator import implied_volatility

# ── Config ────────────────────────────────────────────────────────────────────
TICKER          = "FSLR"
LOOKBACK_YEARS  = 5
TARGET_DTE      = 30       # constant-maturity target in calendar days
RISK_FREE_RATE  = 0.045
DIVIDEND_YIELD  = 0.0      # FSLR dividend is 0
STRIKE_BAND     = 0.25     # ±25% of spot to search for ATM strike

OUTPUT_PATH = (
    "/Users/btian/EtradePythonClient/etrade_python_client"
    "/research_reports/fslr_30dte_iv.png"
)

plt.style.use('dark_background')


# ── Helper: enumerate 3rd-Friday monthly expirations ─────────────────────────
def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given month/year (standard US option expiry)."""
    # First day of the month
    first = date(year, month, 1)
    # Weekday of the 1st (0=Mon, 4=Fri)
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
            # Include the next 2 upcoming expiries so we always have a valid
            # "closest to 30 DTE" target right up to today
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


# ── Helper: round to nearest standard strike increment ────────────────────────
def nearest_strike(spot: float, increment: float) -> float:
    """Round spot to the nearest standard strike increment."""
    return round(spot / increment) * increment


def strike_increment(spot: float) -> float:
    """Return the standard strike increment for a given spot price."""
    if spot < 30:
        return 0.5
    elif spot < 75:
        return 1.0
    elif spot < 150:
        return 2.5
    elif spot < 300:
        return 5.0
    elif spot < 600:
        return 10.0
    else:
        return 20.0


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    # 1. Underlying data
    print(f"Downloading {TICKER} spot data ({LOOKBACK_YEARS}Y)...")
    raw = yf.download(TICKER, period=f"{LOOKBACK_YEARS + 1}y", interval="1d",
                      auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw['returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
    raw['rv_20d']  = raw['returns'].rolling(20).std() * np.sqrt(252)

    # Trim to exactly LOOKBACK_YEARS
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=LOOKBACK_YEARS)
    data = raw[raw.index >= cutoff].copy()
    print(f"  Underlying rows: {len(data)}  ({data.index[0].date()} → {data.index[-1].date()})")

    # 2. Enumerate all monthly expirations for the lookback window
    all_exps = all_monthly_exps(LOOKBACK_YEARS)
    print(f"  Monthly expirations to cover: {len(all_exps)}  "
          f"({all_exps[0]} → {all_exps[-1]})")

    # 3. Fetch option bars
    cache  = OptionDataCache("backtest_cache/option_data.db")
    client = MassiveAPIClient(cache)

    start_str = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365 + 30)).strftime("%Y-%m-%d")
    end_str   = datetime.now().strftime("%Y-%m-%d")

    # exp → { strike → DataFrame of daily bars }
    exp_bars: dict[str, dict[float, pd.DataFrame]] = {}

    async with client:
        for exp in all_exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            # Estimate what spot was ~30 days before this expiry
            ref_date = exp_dt - timedelta(days=TARGET_DTE)
            ref_ts   = pd.Timestamp(ref_date)

            # Find closest available trading day to the 30-DTE reference date.
            # If the reference date is in the future (upcoming expiry), use today's spot.
            close_candidates = data.index[data.index <= ref_ts]
            if close_candidates.empty:
                ref_spot = float(data['Close'].iloc[-1])   # fallback: today's spot
            else:
                ref_spot = float(data.loc[close_candidates[-1], 'Close'])
                # If ref date is very recent but stock has moved sharply, prefer today's spot
                # for the strike ladder if this expiry is still live (dte > 0)
                days_until_exp = (exp_dt - datetime.now()).days
                if days_until_exp >= 0 and len(close_candidates) > 0:
                    today_spot = float(data['Close'].iloc[-1])
                    # Use the spot that's closer in time to today for the strike reference
                    if days_until_exp < 40:   # Within our target DTE window — use today's spot
                        ref_spot = today_spot

            # Build a strike ladder centered on the historically appropriate spot.
            inc = strike_increment(ref_spot)
            atm = nearest_strike(ref_spot, inc)
            candidate_strikes = sorted(
                set(round(atm + i * inc, 2) for i in range(-6, 7) if atm + i * inc > 0)
            )

            yy, mm, dd = exp[2:4], exp[5:7], exp[8:10]
            exp_bars[exp] = {}

            for strike in candidate_strikes:
                strike_fmt = f"{int(strike * 1000):08d}"
                opt_ticker = f"O:{TICKER}{yy}{mm}{dd}C{strike_fmt}"
                bars = await client.fetch_contract_daily_bars(
                    opt_ticker, start_str, end_str,
                    TICKER, "call", float(strike), exp
                )
                if bars:
                    df = pd.DataFrame(bars)
                    df['pricing_date'] = (
                        pd.to_datetime(df['pricing_date']).dt.tz_localize(None)
                    )
                    df = df.set_index('pricing_date').sort_index()
                    exp_bars[exp][strike] = df

            fetched = sum(len(v) > 0 for v in exp_bars[exp].values())
            if fetched:
                print(f"  {exp}: {fetched} strikes with data (ref spot≈${ref_spot:.1f})")

    # 4. Build constant-maturity 30-DTE IV series
    print("\nBuilding CM-30D IV series...")
    cm_rows = []

    for dt, urow in data.iterrows():
        spot = float(urow['Close'])
        if np.isnan(spot):
            continue

        # Pick the expiration whose DTE is closest to TARGET_DTE
        best_exp      = None
        best_dte_diff = float('inf')
        for exp in all_exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte    = (exp_dt - dt.to_pydatetime()).days
            if dte < 5:       # essentially expired
                continue
            diff = abs(dte - TARGET_DTE)
            if diff < best_dte_diff:
                best_dte_diff = diff
                best_exp      = exp

        if best_exp is None:
            continue

        exp_dt = datetime.strptime(best_exp, "%Y-%m-%d")
        T      = max((exp_dt - dt.to_pydatetime()).days / 365.0, 1e-5)

        # Find closest ATM strike with a bar on this date
        strikes_avail = exp_bars.get(best_exp, {})
        best_iv     = None
        best_vol    = 0.0
        best_strike = None

        for s, df in strikes_avail.items():
            if dt not in df.index:
                continue
            bar = df.loc[dt]
            mid = float(bar.get('close', np.nan))
            if np.isnan(mid) or mid <= 0:
                continue
            iv = implied_volatility(
                mid, spot, s, T,
                RISK_FREE_RATE, DIVIDEND_YIELD, "call"
            )
            if iv is None or np.isnan(iv) or iv <= 0 or iv > 5.0:
                continue
            # Prefer closest to ATM
            if best_iv is None or abs(s - spot) < abs(best_strike - spot):
                best_iv     = iv
                best_vol    = float(bar.get('volume', 0))
                best_strike = s

        if best_iv is not None:
            cm_rows.append({
                'date':   dt,
                'cm_iv':  best_iv,
                'dte':    (exp_dt - dt.to_pydatetime()).days,
                'exp':    best_exp,
                'strike': best_strike,
                'volume': best_vol,
                'rv_20d': float(urow['rv_20d']) if not np.isnan(urow['rv_20d']) else np.nan,
                'spot':   spot,
            })

    if not cm_rows:
        print("ERROR: No CM-IV data could be built. Check API data coverage.")
        return

    cm_df = pd.DataFrame(cm_rows).set_index('date').sort_index()
    print(f"CM-30D IV series: {len(cm_df)} rows  "
          f"({cm_df.index[0].date()} → {cm_df.index[-1].date()})")

    # 5. Summary stats
    current_iv  = float(cm_df['cm_iv'].iloc[-1])
    current_rv  = float(cm_df['rv_20d'].dropna().iloc[-1])
    iv_pct      = float((cm_df['cm_iv'] < current_iv).mean())
    iv_min      = cm_df['cm_iv'].min()
    iv_max      = cm_df['cm_iv'].max()
    iv_mean     = cm_df['cm_iv'].mean()
    iv_rv_spread = current_rv - current_iv

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Current 30D ATM IV  : {current_iv:.1%}")
    print(f"  Current 20D RV      : {current_rv:.1%}")
    print(f"  IV/RV Spread        : {iv_rv_spread:+.1%}  ({'RV>IV (cheap)' if iv_rv_spread > 0 else 'IV>RV (expensive)'})")
    print(f"  IV Percentile (5Y)  : {iv_pct:.0%}")
    print(f"  IV Range (5Y)       : {iv_min:.1%} – {iv_max:.1%}  (mean {iv_mean:.1%})")

    # 6. Plot
    COLORS = {
        'spot': '#00e6ff',
        'iv':   '#ff00ff',
        'rv':   '#ffff00',
        'vol':  '#ffa500',
    }

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1]})
    fig.patch.set_facecolor('#0a0a0a')
    for ax in axes:
        ax.set_facecolor('#0d0d0d')

    # ── Panel 1: Spot price ────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(data.index, data['Close'], color=COLORS['spot'], linewidth=1.5,
             label=f'{TICKER} Spot Price')
    ax0.set_ylabel('Price ($)', color=COLORS['spot'], fontweight='bold')
    ax0.legend(loc='upper left', frameon=False)
    ax0.grid(True, alpha=0.08)
    ax0.set_title(f'{TICKER} Spot Price (5-Year History)',
                  color='#aaaaaa', fontsize=12, pad=6)

    # ── Panel 2: CM-30D IV vs RV ──────────────────────────────────────────
    ax1 = axes[1]
    ax1.plot(cm_df.index, cm_df['cm_iv'], color=COLORS['iv'], linewidth=1.8,
             label='30-DTE ATM IV (Constant Maturity)', zorder=3)
    ax1.plot(cm_df.index, cm_df['rv_20d'], color=COLORS['rv'], linewidth=1.4,
             linestyle='--', alpha=0.85, label='20-Day Realized Vol', zorder=2)
    ax1.axhline(iv_mean, color=COLORS['iv'], linestyle=':', alpha=0.45,
                label=f'5Y Avg IV {iv_mean:.1%}')

    # Shade cheap vs expensive regimes
    iv_clean  = cm_df['cm_iv'].dropna()
    rv_aligned = cm_df['rv_20d'].reindex(iv_clean.index)
    ax1.fill_between(iv_clean.index, iv_clean, rv_aligned,
                     where=(iv_clean <= rv_aligned), alpha=0.18, color='#00cc44',
                     label='RV > IV  (cheap options)')
    ax1.fill_between(iv_clean.index, iv_clean, rv_aligned,
                     where=(iv_clean > rv_aligned), alpha=0.15, color='#cc2222',
                     label='IV > RV  (expensive options)')

    # Annotate current level
    ax1.annotate(
        f" Current IV: {current_iv:.1%}\n IV-Pct: {iv_pct:.0%} | RV: {current_rv:.1%}",
        xy=(cm_df.index[-1], current_iv),
        xytext=(-120, 18), textcoords='offset points',
        color='white', fontsize=9,
        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='#222222', alpha=0.85)
    )
    ax1.set_ylabel('Implied / Realized Vol (ann.)', fontweight='bold')
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.08)
    ax1.set_title(f'{TICKER} Constant-Maturity 30-DTE ATM IV vs 20D Realized Volatility',
                  color='#aaaaaa', fontsize=12, pad=6)

    # ── Panel 3: Volume (liquidity) ──────────────────────────────────────
    ax2 = axes[2]
    ax2.bar(cm_df.index, cm_df['volume'], color=COLORS['vol'], alpha=0.7,
            label='ATM 30-DTE Contract Volume', width=1.0)
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.08)
    ax2.set_title('Liquidity — Daily Volume of Nearest 30-DTE ATM Contract',
                  color='#aaaaaa', fontsize=11, pad=6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    fig.suptitle(
        f'{TICKER} — 30-DTE ATM Implied Volatility (Constant Maturity) | 5-Year Study',
        fontsize=15, fontweight='bold', color='white', y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close(fig)
    print(f"\nPlot saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
