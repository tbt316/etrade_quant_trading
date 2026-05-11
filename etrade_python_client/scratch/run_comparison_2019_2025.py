"""
Market Regime Comparison Script (2019-2025)
-------------------------------------------
Full model (incl BTC + WTI Oil) vs. Partial model (ex BTC + WTI Oil).

Uses expanding_window=False for each model run to make it tractable;
the HMM is still fitted causally on 2005-2025 data, and a final
filtered probability pass (forward-algorithm only) is applied.
Run time: ~2-5 min per model.
"""

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import train_regime_hmm, fetch_historical_data, get_regime_labels

# ── Key events for annotation ──────────────────────────────────────────────
KEY_EVENTS = [
    ("2020-02-19", "COVID Peak",   "top"),
    ("2020-03-23", "COVID Bottom", "bottom"),
    ("2022-01-03", "Fed Hike Cycle Start", "top"),
    ("2022-10-12", "2022 Bear Bottom",     "bottom"),
    ("2023-01-20", "AI Bull Begins",       "top"),
    ("2024-08-05", "Yen Carry Unwind",     "bottom"),
]

PALETTE = plt.cm.Set2.colors        # 8 distinct colours
SPY_COLOR   = '#003f88'
VIX_COLOR   = '#c0392b'
ALPHA_SPAN  = 0.35


def plot_comparison(df, result_full, result_part, out_dir="research_reports"):
    os.makedirs(out_dir, exist_ok=True)

    runs = [
        ("full",    result_full["hmm"],  result_full["k"],  result_full["df"],  result_full["labels"]),
        ("partial", result_part["hmm"],  result_part["k"],  result_part["df"],  result_part["labels"]),
    ]
    for tag, best_hmm, best_k, feature_df, labels in runs:
        _save_single_plot(df, best_hmm, best_k, feature_df, labels, tag, out_dir)


def _save_single_plot(df, best_hmm, best_k, feature_df, labels, tag, out_dir):
    title_map = {
        "full":    f"Market Regime Analysis 2019-2025 — Full Model (incl BTC & Crude)\n"
                   f"GMMHMM K={best_k} | Walk-Forward Causal Filtering",
        "partial": f"Market Regime Analysis 2019-2025 — Equity-Only Model (ex BTC & Crude)\n"
                   f"GMMHMM K={best_k} | Walk-Forward Causal Filtering",
    }

    # join SPY/VIX for plotting
    plot_df = feature_df.join(df[['SPY_Close', 'VIX_Close']], how='inner')
    plot_df = plot_df[plot_df.index >= '2019-01-01'].copy()

    if plot_df.empty:
        print(f"  [WARN] No data for 2019+ in tag={tag}")
        return

    # ensure prob columns exist
    prob_cols = [f'prob_state_{i}' for i in range(best_k)]
    missing_prob = [c for c in prob_cols if c not in plot_df.columns]
    if missing_prob:
        print(f"  [WARN] Missing prob columns: {missing_prob}. Skipping {tag}.")
        return

    fig, (ax_price, ax_prob) = plt.subplots(
        2, 1, figsize=(22, 14),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor('#f9f9f9')

    dates = plot_df.index

    # ── Regime spans ────────────────────────────────────────────────────────
    state_changes = plot_df['HMM_State'].ne(plot_df['HMM_State'].shift()).cumsum()
    groups = plot_df.groupby(state_changes)
    added_legend = set()
    for _, grp in groups:
        raw_state = grp['HMM_State'].iloc[0]
        state_val = int(raw_state) if not pd.isna(raw_state) else 0
        color = PALETTE[state_val % len(PALETTE)]
        label = labels.get(state_val, f"Regime {state_val}")
        start_d, end_d = grp.index[0], grp.index[-1]
        if label not in added_legend:
            ax_price.axvspan(start_d, end_d, color=color, alpha=ALPHA_SPAN, label=label)
            added_legend.add(label)
        else:
            ax_price.axvspan(start_d, end_d, color=color, alpha=ALPHA_SPAN)

    # ── SPY price ────────────────────────────────────────────────────────────
    ax_price.plot(dates, plot_df['SPY_Close'], color=SPY_COLOR, linewidth=2.2,
                  label='SPY Price', zorder=5)
    ax_price.set_ylabel('SPY Price ($)', color=SPY_COLOR, fontsize=13, fontweight='bold')
    ax_price.tick_params(axis='y', labelcolor=SPY_COLOR, labelsize=11)
    ax_price.grid(True, alpha=0.25, linestyle='--')
    ax_price.set_facecolor('#ffffff')

    # ── VIX twin axis ────────────────────────────────────────────────────────
    ax_vix = ax_price.twinx()
    ax_vix.plot(dates, plot_df['VIX_Close'], color=VIX_COLOR, linewidth=1.4,
                alpha=0.75, label='VIX', linestyle='-')
    ax_vix.set_ylabel('VIX Index', color=VIX_COLOR, fontsize=13, fontweight='bold')
    ax_vix.tick_params(axis='y', labelcolor=VIX_COLOR, labelsize=11)

    # ── Key event annotations ─────────────────────────────────────────────
    for ev_date_str, ev_label, pos in KEY_EVENTS:
        ev_date = pd.to_datetime(ev_date_str)
        if plot_df.index.min() <= ev_date <= plot_df.index.max():
            ax_price.axvline(ev_date, color='#555555', linewidth=1.0, linestyle=':', alpha=0.7)
            ypos = plot_df['SPY_Close'].max() * 0.96 if pos == "top" else plot_df['SPY_Close'].min() * 1.04
            ax_price.annotate(ev_label, xy=(ev_date, ypos),
                              fontsize=8, color='#333333', rotation=90, va='top',
                              xytext=(3, 0), textcoords='offset points')

    # ── Legend ────────────────────────────────────────────────────────────
    h1, l1 = ax_price.get_legend_handles_labels()
    h2, l2 = ax_vix.get_legend_handles_labels()
    all_handles = h1 + h2
    all_labels  = l1 + l2
    ax_price.legend(all_handles, all_labels, loc='upper left',
                    frameon=True, shadow=True, fontsize=10, ncol=3)

    # ── Probability stacked area ──────────────────────────────────────────
    prob_data = [plot_df[f'prob_state_{i}'].values for i in range(best_k)]
    state_labels_list = [labels.get(i, f'State {i}') for i in range(best_k)]
    ax_prob.stackplot(dates, prob_data,
                      labels=state_labels_list,
                      colors=PALETTE[:best_k], alpha=0.85)
    ax_prob.set_ylabel('Posterior Probability', fontsize=12, fontweight='bold')
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_prob.legend(loc='lower left', frameon=True, fontsize=9, ncol=best_k)
    ax_prob.grid(True, alpha=0.25)
    ax_prob.set_facecolor('#ffffff')

    ax_prob.xaxis.set_major_locator(mdates.YearLocator())
    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_prob.get_xticklabels(), rotation=0, ha='center', fontsize=11)

    fig.suptitle(title_map[tag], fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f"regime_{tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅  Saved → {out_path}")


def run_model(df, label, exclude=None):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Running {label} ...")
    print(f"{'='*60}")
    # expanding_window=False: fast BIC-optimised single HMM fit + causal
    # forward-algorithm probability pass (no per-day refit loop).
    best_hmm, best_k, feature_df = train_regime_hmm(
        df,
        expanding_window=False,
        exclude_features=exclude,
    )
    labels = get_regime_labels(best_hmm, feature_df)
    elapsed = time.time() - t0
    print(f"  ✓  Done in {elapsed:.1f}s  |  K={best_k}  |  Labels: {labels}")

    # Validate prob columns are present
    prob_cols = [c for c in feature_df.columns if c.startswith('prob_state_')]
    print(f"  Probability columns found: {prob_cols}")
    print(f"  Feature columns: {list(feature_df.columns)}")

    return {"hmm": best_hmm, "k": best_k, "df": feature_df, "labels": labels}


def main():
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Fetching historical data...")
    df = fetch_historical_data()
    df = df[df.index <= '2025-12-31']
    print(f"  Data range: {df.index.min().date()} → {df.index.max().date()}  ({len(df)} rows)")

    # ── Full model ──────────────────────────────────────────────────────────
    result_full = run_model(df, "FULL MODEL (incl BTC + Crude)", exclude=None)

    # ── Equity-only model ───────────────────────────────────────────────────
    BTC_CRUDE_COLS = ['BTC_Close', 'BTC_Log_Return', 'WTI_Oil']
    result_part = run_model(df, "EQUITY-ONLY MODEL (ex BTC + Crude)", exclude=BTC_CRUDE_COLS)

    # ── Generate comparison plots ───────────────────────────────────────────
    print("\nGenerating plots...")
    plot_comparison(df, result_full, result_part, out_dir="research_reports")

    print("\n✅  All plots saved to research_reports/")


if __name__ == "__main__":
    main()
