"""
Causal market-regime diagnostics.

Produces:
  1. A time-domain plot of detected HMM regimes over SPY/VIX.
  2. Per-regime SPY daily log-return histograms with 2-component GMM fits.
  3. CSV outputs for the causal trace and GMM summary.

This script is diagnostic only; it does not run option trades.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition._pca")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
logging.getLogger("live_trading.pca_fusion").setLevel(logging.ERROR)


def _fetch_history(start_date, end_date):
    ingestor = DataIngestor()
    df = ingestor.fetch_yf_data(start_date, end_date)
    if df.empty:
        raise RuntimeError("Historical YFinance cache is empty or missing core SPY/VIX data.")
    missing = [c for c in ("SPY_Close", "VIX_Close") if c not in df.columns]
    if missing:
        raise RuntimeError(f"Historical data is missing required columns: {missing}")
    return df


def _state_label(frame, state):
    rows = frame[frame["HMM_State"] == state]
    if "Regime_Label" in rows.columns and not rows.empty and rows["Regime_Label"].notna().any():
        labels = rows["Regime_Label"].dropna()
        mode = labels.value_counts()
        if not mode.empty:
            return mode.index[0]
    return f"State {state}"


def build_regime_diagnostics(
    fetch_start="2015-01-01",
    analysis_start="2020-01-01",
    analysis_end=None,
    n_components=3,
    output_dir="research_reports/regime_diagnostics",
):
    os.makedirs(output_dir, exist_ok=True)
    end = analysis_end or datetime.now().strftime("%Y-%m-%d")

    print("=" * 78)
    print("CAUSAL MARKET REGIME DIAGNOSTICS")
    print("=" * 78)
    print(f"Fetch window:    {fetch_start} -> {end}")
    print(f"Analysis window: {analysis_start} -> {end}")
    print(f"HMM K:           {n_components if n_components else 'auto'}")
    print("Inference:       train_regime_hmm(..., expanding_window=True)")
    print("Signal timing:   close_T_for_next_session")
    print()

    df_raw = _fetch_history(fetch_start, end)
    df_raw = df_raw.loc[:end].copy()
    hmm_model, best_k, feature_df = train_regime_hmm(
        df_raw,
        n_components=n_components,
        expanding_window=True,
    )
    if hmm_model is None or feature_df.empty:
        raise RuntimeError("Regime model did not produce a usable causal trace.")

    common_idx = feature_df.index.intersection(df_raw.index)
    trace = feature_df.loc[common_idx].copy()
    trace["SPY_Close"] = df_raw.loc[common_idx, "SPY_Close"]
    trace["VIX_Close"] = df_raw.loc[common_idx, "VIX_Close"]
    trace["SPY_Log_Return"] = np.log(trace["SPY_Close"] / trace["SPY_Close"].shift(1))
    trace["Regime_Signal_Timestamp"] = trace.get(
        "Regime_Signal_Timestamp",
        pd.Series("close_T_for_next_session", index=trace.index),
    )
    if {"Detected_Regime_State", "Detected_Regime_Label"}.issubset(trace.columns):
        trace["Raw_HMM_State"] = trace["HMM_State"]
        trace["Raw_Regime_Label"] = trace["Regime_Label"]
        trace["HMM_State"] = trace["Detected_Regime_State"]
        trace["Regime_Label"] = trace["Detected_Regime_Label"]
    trace = trace.loc[pd.Timestamp(analysis_start):pd.Timestamp(end)].copy()
    trace = trace.dropna(subset=["SPY_Log_Return", "HMM_State"])
    if trace.empty:
        raise RuntimeError("No diagnostic rows remain after applying the analysis window.")
    effective_k = int(trace["HMM_State"].max()) + 1

    timeline_path = os.path.join(output_dir, "market_regime_timeline.png")
    gmm_path = os.path.join(output_dir, "spy_log_return_gmm_by_regime.png")
    trace_path = os.path.join(output_dir, "causal_regime_trace.csv")
    summary_path = os.path.join(output_dir, "spy_log_return_gmm_summary.csv")

    _plot_timeline(trace, effective_k, timeline_path)
    summary = _plot_gmm_fits(trace, effective_k, gmm_path)

    trace.to_csv(trace_path)
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print()
    print("Outputs")
    print(f"  Timeline: {timeline_path}")
    print(f"  GMM fits: {gmm_path}")
    print(f"  Trace:    {trace_path}")
    print(f"  Summary:  {summary_path}")

    return {
        "timeline_path": timeline_path,
        "gmm_path": gmm_path,
        "trace_path": trace_path,
        "summary_path": summary_path,
        "rows": len(trace),
        "best_k": best_k,
        "effective_k": effective_k,
    }


def _plot_timeline(trace, best_k, output_path):
    colors = plt.cm.Set3.colors
    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(24, 14),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    state_labels = {state: _state_label(trace, state) for state in range(best_k)}
    change_groups = trace["HMM_State"].ne(trace["HMM_State"].shift()).cumsum()
    added = set()
    for _, group in trace.groupby(change_groups):
        state = int(group["HMM_State"].iloc[0])
        label = state_labels.get(state, f"State {state}")
        color = colors[state % len(colors)]
        if label not in added:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.35, label=label)
            added.add(label)
        else:
            ax_price.axvspan(group.index[0], group.index[-1], color=color, alpha=0.35)

    ax_price.plot(trace.index, trace["SPY_Close"], color="navy", linewidth=2.0, label="SPY Close")
    ax_vix = ax_price.twinx()
    ax_vix.plot(trace.index, trace["VIX_Close"], color="darkred", linewidth=1.2, alpha=0.75, label="VIX")
    ax_price.set_ylabel("SPY Close")
    ax_vix.set_ylabel("VIX")
    ax_price.grid(True, alpha=0.25, linestyle="--")

    prob_cols = [f"prob_state_{i}" for i in range(best_k) if f"prob_state_{i}" in trace.columns]
    prob_labels = [state_labels.get(i, f"State {i}") for i in range(len(prob_cols))]
    ax_prob.stackplot(
        trace.index,
        [trace[c].values for c in prob_cols],
        labels=prob_labels,
        colors=colors[: len(prob_cols)],
        alpha=0.85,
    )
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Regime Probability")
    ax_prob.set_xlabel("Date")
    ax_prob.legend(loc="lower left", fontsize=9, ncol=max(1, min(best_k, 5)))
    ax_prob.grid(True, alpha=0.25)

    h1, l1 = ax_price.get_legend_handles_labels()
    h2, l2 = ax_vix.get_legend_handles_labels()
    ax_price.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=3)
    fig.suptitle(
        "Causal Market Regime Timeline\n"
        "GMMHMM walk-forward trace | Signal timestamp: close_T_for_next_session",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gmm_fits(trace, best_k, output_path):
    colors = plt.cm.Set3.colors
    cols = 2
    rows = max(1, int(np.ceil(best_k / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    x_all = trace["SPY_Log_Return"].dropna().values
    x_min, x_max = np.nanpercentile(x_all, [0.5, 99.5])
    x_grid = np.linspace(x_min, x_max, 500)
    summary = []

    print("Per-regime SPY daily log-return GMM fit")
    print("-" * 78)
    for state in range(best_k):
        ax = axes[state]
        state_df = trace[trace["HMM_State"] == state]
        values = state_df["SPY_Log_Return"].dropna().values
        label = _state_label(trace, state)
        n = len(values)
        color = colors[state % len(colors)]

        if n < 20:
            ax.set_title(f"{label}\ninsufficient observations: n={n}")
            ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center", va="center")
            print(f"State {state} | {label} | n={n:,} | insufficient observations")
            continue

        returns = values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(returns)
        density = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
        bic = float(gmm.bic(returns))

        ax.hist(
            values,
            bins=min(80, max(25, n // 20)),
            density=True,
            alpha=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            label="Empirical",
        )
        ax.plot(x_grid, density, color="black", linewidth=2.0, label="2-component GMM")

        for comp_idx, (weight, mean, covar) in enumerate(
            zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.reshape(-1))
        ):
            std = np.sqrt(max(float(covar), 1e-12))
            comp_density = weight * stats.norm.pdf(x_grid, loc=mean, scale=std)
            ax.plot(x_grid, comp_density, linestyle="--", linewidth=1.2, label=f"Comp {comp_idx + 1}")

        mean_ret = float(np.mean(values))
        std_ret = float(np.std(values, ddof=1))
        skew = float(stats.skew(values))
        kurt = float(stats.kurtosis(values))

        summary.append(
            {
                "state": state,
                "label": label,
                "n": n,
                "mean_log_return": mean_ret,
                "std_log_return": std_ret,
                "skew": skew,
                "excess_kurtosis": kurt,
                "gmm_bic": bic,
                "gmm_weights": json.dumps([float(x) for x in gmm.weights_]),
                "gmm_means": json.dumps([float(x) for x in gmm.means_.flatten()]),
                "gmm_stds": json.dumps(
                    [float(np.sqrt(max(x, 1e-12))) for x in gmm.covariances_.reshape(-1)]
                ),
            }
        )

        print(
            f"State {state} | {label} | n={n:,} | mean={mean_ret:.6f} "
            f"| std={std_ret:.6f} | skew={skew:.2f} | ex.kurt={kurt:.2f} | BIC={bic:.1f}"
        )

        ax.set_title(f"{label}\nn={n:,} | BIC={bic:.1f}")
        ax.set_xlabel("SPY daily log return")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[best_k:]:
        ax.set_visible(False)

    fig.suptitle("SPY Daily Log Return Distributions by Causal HMM Regime", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Causal market regime timeline and SPY log-return GMM diagnostics")
    parser.add_argument("--fetch-start", default="2015-01-01", help="First date fetched for model warmup")
    parser.add_argument("--analysis-start", default="2020-01-01", help="First date shown in outputs")
    parser.add_argument("--analysis-end", default=datetime.now().strftime("%Y-%m-%d"), help="Last date shown in outputs")
    parser.add_argument("--force-k", type=int, default=3, help="Fixed HMM state count; use 0 for auto BIC")
    parser.add_argument("--output-dir", default="research_reports/regime_diagnostics")
    args = parser.parse_args()

    build_regime_diagnostics(
        fetch_start=args.fetch_start,
        analysis_start=args.analysis_start,
        analysis_end=args.analysis_end,
        n_components=args.force_k or None,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
