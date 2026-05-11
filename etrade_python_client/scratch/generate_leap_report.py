import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import base64
from io import BytesIO

from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache
from backtesting.greeks_calculator import implied_volatility

# Configuration
TICKERS = ["WOLF", "GLW"]
LOOKBACK_DAYS = 1095  # 36 months
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = {"WOLF": 0.0, "GLW": 0.022}
TARGET_MATURITY_DAYS = 365  # Constant-maturity target

# Styling
plt.style.use('dark_background')
COLORS = {
    'spot': '#00e6ff',
    'opt':  '#39ff14',
    'iv':   '#ff00ff',
    'rv':   '#ffff00',
    'int':  '#4d4dff',
    'ext':  '#808080'
}

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

async def analyze_ticker(ticker, client, all_data):
    print(f"\n--- Analyzing {ticker} ---")

    # ── 1. Underlying data (full 3-year history) ──────────────────────────────
    data = all_data[ticker].copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['rv_20d']  = data['returns'].rolling(window=20).std() * np.sqrt(252)
    current_spot = float(data['Close'].iloc[-1])

    # ── 2. Discover ALL expirations from snapshot ─────────────────────────────
    snapshot = await client.fetch_option_snapshot(ticker)
    all_exps = sorted(set(
        r['details']['expiration_date'] for r in snapshot
        if r['details']['expiration_date'] > '2024-01-01'   # Jan 2025 onward
    ))
    # Keep only January expirations (standard LEAP cycle)
    jan_exps = [e for e in all_exps if e[5:7] == '01']
    if not jan_exps:
        jan_exps = all_exps[:3]  # fallback: take the first 3 available

    print(f"  LEAP expirations found: {jan_exps}")
    target_exp = max(jan_exps)   # most-distant = current "live" LEAP for today's entry

    # Current ATM strike (for the live entry contract summary)
    current_strikes = sorted(set(
        r['details']['strike_price'] for r in snapshot
        if r['details']['expiration_date'] == target_exp
    ))
    current_atm = min(current_strikes, key=lambda x: abs(x - current_spot))

    # ── 3. Fetch bars for each Jan expiry — ATM strike per expiry ─────────────
    start_date_str = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date_str   = datetime.now().strftime("%Y-%m-%d")

    # Map expiration → bars_df keyed by strike
    exp_data: dict[str, dict[float, pd.DataFrame]] = {}

    for exp in jan_exps:
        strikes_for_exp = sorted(set(
            r['details']['strike_price'] for r in snapshot
            if r['details']['expiration_date'] == exp
        ))
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        yy, mm, dd = exp[2:4], exp[5:7], exp[8:10]
        exp_data[exp] = {}

        # For each expiry, only fetch strikes that were near-ATM at some point in
        # the lookback window.  Use a ±40% band around current spot as a proxy.
        candidate_strikes = [s for s in strikes_for_exp
                             if 0.6 * current_spot <= s <= 1.6 * current_spot]
        # Always include the ATM for the current snapshot
        for s in candidate_strikes:
            strike_fmt  = f"{int(s * 1000):08d}"
            opt_ticker  = f"O:{ticker}{yy}{mm}{dd}C{strike_fmt}"
            bars = await client.fetch_contract_daily_bars(
                opt_ticker, start_date_str, end_date_str,
                ticker, "call", float(s), exp
            )
            if bars:
                df = pd.DataFrame(bars)
                df['pricing_date'] = pd.to_datetime(df['pricing_date']).dt.tz_localize(None)
                df = df.set_index('pricing_date').sort_index()
                exp_data[exp][s] = df

    # ── 4. Build constant-maturity 1-year IV series ───────────────────────────
    # For each trading day in the underlying history, pick the expiration whose
    # remaining days-to-expiry is closest to TARGET_MATURITY_DAYS, then find
    # the ATM strike for that expiry on that date, and compute IV.
    cm_iv_rows = []

    for dt, urow in data.iterrows():
        spot = float(urow['Close'])
        if np.isnan(spot):
            continue

        best_exp = None
        best_dte_diff = float('inf')
        for exp in jan_exps:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_dt - dt).days
            if dte < 10:           # expired or nearly expired — skip
                continue
            diff = abs(dte - TARGET_MATURITY_DAYS)
            if diff < best_dte_diff:
                best_dte_diff = diff
                best_exp = exp

        if best_exp is None:
            continue

        exp_dt = datetime.strptime(best_exp, "%Y-%m-%d")
        T = max((exp_dt - dt).days / 365.0, 1e-5)

        # Find the strike in that expiry with the best available bar on this date
        strikes_df = exp_data.get(best_exp, {})
        best_iv  = None
        best_vol = 0
        best_strike_used = None

        for s, df in strikes_df.items():
            if dt not in df.index:
                continue
            bar = df.loc[dt]
            mid = float(bar.get('close', bar.get('vwap', np.nan)))
            if np.isnan(mid) or mid <= 0:
                continue
            iv = implied_volatility(
                mid, spot, s, T,
                RISK_FREE_RATE, DIVIDEND_YIELD.get(ticker, 0), "call"
            )
            if iv is not None and not np.isnan(iv) and iv > 0:
                # Prefer the strike closest to ATM (lowest |moneyness|)
                if best_iv is None or abs(s - spot) < abs(best_strike_used - spot):
                    best_iv = iv
                    best_vol = float(bar.get('volume', 0))
                    best_strike_used = s

        if best_iv is not None:
            cm_iv_rows.append({
                'date':   dt,
                'cm_iv':  best_iv,
                'dte':    (exp_dt - dt).days,
                'exp':    best_exp,
                'strike': best_strike_used,
                'volume': best_vol,
                'rv_20d': float(urow['rv_20d']) if not np.isnan(urow['rv_20d']) else np.nan,
                'spot':   spot,
            })

    if not cm_iv_rows:
        print(f"  No constant-maturity IV data could be built for {ticker}")
        return None

    cm_df = pd.DataFrame(cm_iv_rows).set_index('date').sort_index()
    print(f"  Constant-maturity IV series: {len(cm_df)} rows, "
          f"{cm_df.index[0].date()} → {cm_df.index[-1].date()}")

    # ── 5. Current entry contract stats ───────────────────────────────────────
    current_iv  = float(cm_df['cm_iv'].iloc[-1])
    current_rv  = float(cm_df['rv_20d'].dropna().iloc[-1])
    iv_pct_rank = float((cm_df['cm_iv'] < current_iv).mean())  # IV Percentile

    # ── 6. Figure 1: Full underlying history ──────────────────────────────────
    fig1, (ax_spot, ax_rv) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )
    ax_spot.plot(data.index, data['Close'], color=COLORS['spot'], linewidth=2,
                 label=f'{ticker} Spot Price')
    ax_spot.set_ylabel('Price ($)', color=COLORS['spot'], fontweight='bold')
    ax_spot.legend(loc='upper left', frameon=False)
    ax_spot.grid(True, alpha=0.1)
    ax_spot.set_title('Underlying Price & Realized Volatility (3-Year History)',
                      fontsize=13, color='#aaaaaa', pad=8)

    ax_rv.plot(data.index, data['rv_20d'], color=COLORS['rv'], linewidth=1.5,
               label='Realized Vol 20D', alpha=0.85)
    ax_rv.set_ylabel('RV (ann.)', fontweight='bold')
    ax_rv.legend(loc='upper left', frameon=False)
    ax_rv.grid(True, alpha=0.1)
    ax_rv.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_rv.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_rv.xaxis.get_majorticklabels(), rotation=30, ha='right')

    fig1.suptitle(f"{ticker} — Underlying Price & Realized Volatility",
                  fontsize=16, fontweight='bold', color='white')
    fig1.tight_layout()
    plot_underlying_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # ── 7. Figure 2: Constant-maturity IV + liquidity ─────────────────────────
    fig2, (ax_vol, ax_decomp, ax_liq) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    ax_vol.plot(cm_df.index, cm_df['cm_iv'], color=COLORS['iv'], linewidth=2,
                label='1-Year Constant-Maturity IV')
    ax_vol.plot(cm_df.index, cm_df['rv_20d'], color=COLORS['rv'],
                linestyle='--', linewidth=1.5, alpha=0.85, label='Realized Vol 20D')
    ax_vol.axhline(cm_df['cm_iv'].mean(), color=COLORS['iv'], linestyle=':',
                   alpha=0.5, label=f"Avg CM-IV {cm_df['cm_iv'].mean():.1%}")
    ax_vol.fill_between(cm_df.index, cm_df['cm_iv'], cm_df['rv_20d'],
                        where=cm_df['cm_iv'] > cm_df['rv_20d'],
                        alpha=0.15, color='red', label='IV > RV (expensive)')
    ax_vol.fill_between(cm_df.index, cm_df['cm_iv'], cm_df['rv_20d'],
                        where=cm_df['cm_iv'] <= cm_df['rv_20d'],
                        alpha=0.15, color='green', label='RV > IV (cheap)')
    ax_vol.set_ylabel('Volatility (ann.)', fontweight='bold')
    ax_vol.legend(loc='upper left', frameon=False, fontsize=9)
    ax_vol.grid(True, alpha=0.1)
    ax_vol.set_title('Constant-Maturity 1-Year IV vs Realized Vol',
                     fontsize=13, color='#aaaaaa', pad=8)

    # Intrinsic / Extrinsic on current entry contract
    atm_df = exp_data.get(target_exp, {}).get(current_atm, pd.DataFrame())
    if not atm_df.empty:
        atm_merged = atm_df.join(data[['Close']].rename(columns={'Close': 'spot'}), how='inner')
        atm_merged['intrinsic'] = (atm_merged['spot'] - current_atm).clip(lower=0)
        atm_merged['extrinsic'] = (atm_merged['close'] - atm_merged['intrinsic']).clip(lower=0)
        ax_decomp.stackplot(atm_merged.index, atm_merged['intrinsic'], atm_merged['extrinsic'],
                            labels=[f'Intrinsic (K={current_atm})', 'Extrinsic'],
                            alpha=0.55, colors=[COLORS['int'], COLORS['ext']])
        ax_decomp.set_title(f'Premium Decomposition: {target_exp} ${current_atm} Call',
                            fontsize=12, color='#aaaaaa', pad=6)
    ax_decomp.set_ylabel('Value ($)', fontweight='bold')
    ax_decomp.legend(loc='upper left', frameon=False, fontsize=9)
    ax_decomp.grid(True, alpha=0.1)

    ax_liq.bar(cm_df.index, cm_df['volume'], color='#ffa500', alpha=0.75,
               label='ATM Contract Daily Volume', width=1.5)
    ax_liq.set_ylabel('Volume', fontweight='bold')
    ax_liq.set_title('Liquidity — Daily Volume of Nearest 1-Year ATM Contract',
                     fontsize=12, color='#aaaaaa', pad=6)
    ax_liq.legend(loc='upper left', frameon=False, fontsize=9)
    ax_liq.grid(True, alpha=0.1)
    ax_liq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax_liq.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_liq.xaxis.get_majorticklabels(), rotation=30, ha='right')

    fig2.suptitle(f"{ticker} — Constant-Maturity 1-Year LEAP IV Analysis",
                  fontsize=16, fontweight='bold', color='white')
    fig2.tight_layout()
    plot_option_b64 = fig_to_base64(fig2)
    plt.close(fig2)

    # ── 8. Text Analysis ──────────────────────────────────────────────────────
    conviction = "HIGH" if current_rv > current_iv else "NEUTRAL"
    iv_rv_spread = current_rv - current_iv
    if conviction == "HIGH":
        analysis_text = (
            f"<b>Volatility Edge Detected.</b> The constant-maturity 1-year IV is "
            f"<b>{current_iv:.1%}</b>, while the 20-day realized volatility is "
            f"<b>{current_rv:.1%}</b> — a spread of <b>+{iv_rv_spread:.1%}</b> in favor "
            f"of the option buyer. IV is currently at the <b>{iv_pct_rank:.0%} percentile</b> "
            f"of its 3-year range. LEAP premiums appear <b>cheap</b> relative to recent "
            f"realized price action, suggesting a high-conviction long entry."
        )
    else:
        analysis_text = (
            f"<b>Volatility Premium Detected.</b> The constant-maturity 1-year IV is "
            f"<b>{current_iv:.1%}</b>, exceeding the 20-day realized volatility of "
            f"<b>{current_rv:.1%}</b> by <b>{-iv_rv_spread:.1%}</b>. IV is at the "
            f"<b>{iv_pct_rank:.0%} percentile</b> of its 3-year range. The market is "
            f"pricing in more movement than the stock has recently delivered — options "
            f"are relatively <b>expensive</b>. Conviction is neutral."
        )

    return {
        'ticker':      ticker,
        'spot':        current_spot,
        'best_strike': current_atm,
        'exp':         target_exp,
        'iv':          current_iv,
        'rv':          current_rv,
        'iv_pct':      iv_pct_rank,
        'conviction':  conviction,
        'analysis':    analysis_text,
        'plot_underlying': plot_underlying_b64,
        'plot_option':     plot_option_b64,
    }

async def main():
    cache  = OptionDataCache("backtest_cache/option_data.db")
    client = MassiveAPIClient(cache)

    print(f"Fetching underlying data for {TICKERS}...")
    all_data = {}
    for t in TICKERS:
        df = yf.download(t, period="3y", interval="1d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        all_data[t] = df

    reports = []
    async with client:
        for ticker in TICKERS:
            res = await analyze_ticker(ticker, client, all_data)
            if res:
                reports.append(res)

    # ── HTML Report ───────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LEAP Quantitative Research Report</title>
    <meta charset="utf-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'Inter', sans-serif; margin: 0; padding: 40px; }}
        .report-card {{ background: #1a1a1a; border-radius: 12px; padding: 28px; margin-bottom: 48px; border: 1px solid #2a2a2a; }}
        h1 {{ color: #00e6ff; font-size: 30px; border-bottom: 2px solid #00e6ff; padding-bottom: 10px; margin-bottom: 6px; }}
        h2 {{ color: #39ff14; font-size: 22px; margin-bottom: 18px; }}
        h3 {{ color: #aaaaaa; font-size: 15px; font-weight: 600; margin: 24px 0 6px; text-transform: uppercase; letter-spacing: 1px; }}
        .meta {{ color: #555; font-size: 13px; margin-bottom: 30px; }}
        .stats {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat-box {{ background: #252525; padding: 16px 20px; border-radius: 10px; flex: 1; min-width: 140px; text-align: center; border: 1px solid #333; }}
        .stat-label {{ font-size: 11px; color: #777; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
        .conviction-HIGH {{ color: #39ff14; }}
        .conviction-NEUTRAL {{ color: #ffcc00; }}
        .analysis-text {{ background: #151f15; padding: 16px 20px; border-radius: 8px; margin: 20px 0; font-size: 14px; line-height: 1.7; border-left: 4px solid #00e6ff; }}
        img {{ width: 100%; border-radius: 8px; margin-top: 16px; display: block; }}
    </style>
</head>
<body>
    <h1>QUANTITATIVE LEAP RESEARCH REPORT</h1>
    <p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;|&nbsp;
       Methodology: Constant-Maturity 1-Year IV vs 20-Day Realized Volatility &nbsp;|&nbsp;
       Data Horizon: 36 Months</p>
"""

    for r in reports:
        html += f"""
    <div class="report-card">
        <h2>{r['ticker']} &nbsp;|&nbsp; Entry Target: {r['exp']} ${r['best_strike']} Call</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Spot Price</div>
                <div class="stat-value" style="color:#00e6ff">${r['spot']:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">1-Year CM IV</div>
                <div class="stat-value" style="color:#ff00ff">{r['iv']:.1%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Realized Vol (20D)</div>
                <div class="stat-value" style="color:#ffff00">{r['rv']:.1%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">IV Percentile (3Y)</div>
                <div class="stat-value">{r['iv_pct']:.0%}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Conviction</div>
                <div class="stat-value conviction-{r['conviction']}">{r['conviction']}</div>
            </div>
        </div>
        <div class="analysis-text">{r['analysis']}</div>
        <h3>&#9656; Underlying Price History &amp; Realized Volatility (3 Years)</h3>
        <img src="data:image/png;base64,{r['plot_underlying']}" />
        <h3>&#9656; Constant-Maturity IV, Premium Decomposition &amp; Liquidity</h3>
        <img src="data:image/png;base64,{r['plot_option']}" />
    </div>
"""

    html += "</body></html>"

    report_path = "/Users/btian/EtradePythonClient/etrade_python_client/research_reports/leap_evaluation_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())


