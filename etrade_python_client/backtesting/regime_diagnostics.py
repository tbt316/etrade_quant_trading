"""
Causal market-regime diagnostics.

Produces:
  1. A time-domain plot of detected HMM regimes over SPY/VIX.
  2. Per-regime SPY daily log-return histograms with BIC-selected GMM fits.
  3. CSV outputs for the causal trace and GMM summary.

This script is diagnostic only; it does not run option trades.
"""

import argparse
import base64
import html
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm, select_gmm_by_bic

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


def _regime_columns(best_k):
    return [f"detected_prob_state_{i}" for i in range(3)], [
        "Expansion (0)",
        "Cautious Decline (1)",
        "Panic / Crisis (2)",
    ]


def _image_data_uri(path):
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_timeline_figure(trace, best_k):
    state_labels = {
        0: "Expansion (0)",
        1: "Cautious Decline (1)",
        2: "Panic / Crisis (2)",
    }
    colors = {
        0: "rgba(56, 189, 248, 0.18)",
        1: "rgba(245, 158, 11, 0.20)",
        2: "rgba(239, 68, 68, 0.20)",
    }
    line_colors = {0: "#0284c7", 1: "#d97706", 2: "#dc2626"}
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Scatter(
            x=trace.index,
            y=trace["SPY_Close"],
            name="SPY Close",
            mode="lines",
            line=dict(color="navy", width=1.4),
            opacity=0.75,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trace.index,
            y=trace["VIX_Close"],
            name="VIX",
            mode="lines",
            line=dict(color="darkred", width=1.1),
            opacity=0.8,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    prob_cols, prob_labels = _regime_columns(best_k)
    for idx, col in enumerate(prob_cols):
        if col not in trace.columns:
            continue
        state = idx if idx in state_labels else idx
        fig.add_trace(
            go.Scatter(
                x=trace.index,
                y=trace[col],
                name=prob_labels[idx] if idx < len(prob_labels) else col,
                mode="lines",
                line=dict(color=line_colors.get(state, "#475569"), width=1.4),
                stackgroup="regime_prob",
                fillcolor=colors.get(state, "rgba(100,116,139,0.18)"),
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
    for _, group in trace.groupby(trace["HMM_State"].ne(trace["HMM_State"].shift()).cumsum()):
        state = int(group["HMM_State"].iloc[0])
        fig.add_vrect(
            x0=group.index[0],
            x1=group.index[-1],
            fillcolor=colors.get(state, "rgba(100,116,139,0.18)"),
            opacity=1.0,
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )
    fig.update_layout(
        height=1050,
        autosize=True,
        margin=dict(l=60, r=60, t=60, b=50),
        template="plotly_white",
        hovermode="x",
        spikedistance=-1,
        hoverdistance=-1,
        showlegend=True,
        dragmode="zoom",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(255,255,255,0.7)"),
        title="Causal Market Regime Timeline",
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    fig.update_xaxes(title="Date", row=2, col=1)
    fig.update_yaxes(title="SPY Close", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title="VIX", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title="Regime Probability", range=[0, 1], row=2, col=1)
    return fig


def _df_to_html_table(df, max_rows=20):
    if df is None or df.empty:
        return "<p class=\"muted\">No rows.</p>"
    safe = df.head(max_rows).copy()
    return safe.to_html(index=False, classes="data-table", border=0, escape=True)


def _build_pca_diagnostics(hmm_model):
    fusion = getattr(hmm_model, "fusion_", None)
    pca = getattr(fusion, "sparse_pca", None) if fusion is not None else None
    feature_names = list(getattr(hmm_model, "feature_names_", []) or [])
    if pca is None or not hasattr(pca, "components_"):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    )
    evr = np.asarray(getattr(pca, "explained_variance_ratio_", []), dtype=float)
    pca_summary = []
    for idx, pc in enumerate(loadings.columns):
        abs_load = loadings[pc].abs().sort_values(ascending=False)
        top = abs_load.head(5)
        pca_summary.append(
            {
                "component": pc,
                "explained_variance_ratio": float(evr[idx]) if idx < len(evr) else np.nan,
                "cumulative_variance": float(np.nansum(evr[: idx + 1])) if len(evr) else np.nan,
                "top_features": ", ".join([f"{name} ({loadings.loc[name, pc]:.3f})" for name in top.index]),
            }
        )
    feature_influence = (
        loadings.abs()
        .mean(axis=1)
        .rename("mean_abs_loading")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_loading", ascending=False)
    )
    dominant_features = loadings.abs().max(axis=1).rename("max_abs_loading").reset_index().rename(columns={"index": "feature"})
    return (
        loadings.reset_index().rename(columns={"index": "feature"}),
        pd.DataFrame(pca_summary),
        feature_influence,
        dominant_features,
    )


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
    effective_k = 3
    pca_loadings_df, pca_summary_df, pca_feature_influence_df, pca_dominant_df = _build_pca_diagnostics(hmm_model)
    timeline_fig = _build_timeline_figure(trace, effective_k)

    timeline_path = os.path.join(output_dir, "market_regime_timeline.png")
    gmm_path = os.path.join(output_dir, "spy_log_return_gmm_by_regime.png")
    report_path = os.path.join(output_dir, "market_regime_diagnostics.html")
    trace_path = os.path.join(output_dir, "causal_regime_trace.csv")
    summary_path = os.path.join(output_dir, "spy_log_return_gmm_summary.csv")

    _plot_timeline(trace, effective_k, timeline_path)
    summary = _plot_gmm_fits(trace, effective_k, gmm_path)

    trace.to_csv(trace_path)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_path, index=False)
    _write_html_report(
        report_path=report_path,
        trace=trace,
        summary_df=summary_df,
        timeline_path=timeline_path,
        timeline_fig=timeline_fig,
        gmm_path=gmm_path,
        fetch_start=fetch_start,
        analysis_start=analysis_start,
        analysis_end=end,
        requested_k=n_components or 3,
        raw_k=best_k,
        pca_feature_names=getattr(hmm_model, "feature_names_", []),
        pca_loadings_df=pca_loadings_df,
        pca_summary_df=pca_summary_df,
        pca_feature_influence_df=pca_feature_influence_df,
        pca_dominant_df=pca_dominant_df,
    )

    print()
    print("Outputs")
    print(f"  Timeline: {timeline_path}")
    print(f"  GMM fits: {gmm_path}")
    print(f"  HTML:     {report_path}")
    print(f"  Trace:    {trace_path}")
    print(f"  Summary:  {summary_path}")

    return {
        "timeline_path": timeline_path,
        "gmm_path": gmm_path,
        "report_path": report_path,
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

    state_labels = {
        0: "Expansion (0)",
        1: "Cautious Decline (1)",
        2: "Panic / Crisis (2)",
    }
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

    detected_prob_cols, detected_prob_labels = _regime_columns(best_k)
    if all(c in trace.columns for c in detected_prob_cols):
        prob_cols = detected_prob_cols
        prob_labels = detected_prob_labels
    else:
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
        "Gaussian HMM walk-forward trace + causal stress overlay | Signal timestamp: close_T_for_next_session",
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
        selected, candidates = select_gmm_by_bic(values)
        gmm = selected["model"]
        selected_k = int(selected["k"])
        density = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
        bic = float(selected["bic"])

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
        ax.plot(x_grid, density, color="black", linewidth=2.0, label=f"BIC-selected GMM (k={selected_k})")

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
                "gmm_selected_k": selected_k,
                "gmm_bic": bic,
                "gmm_aic": float(selected["aic"]),
                "gmm_candidate_bic": json.dumps({int(c["k"]): float(c["bic"]) for c in candidates}),
                "gmm_weights": json.dumps([float(x) for x in gmm.weights_]),
                "gmm_means": json.dumps([float(x) for x in gmm.means_.flatten()]),
                "gmm_stds": json.dumps(
                    [float(np.sqrt(max(x, 1e-12))) for x in gmm.covariances_.reshape(-1)]
                ),
            }
        )

        print(
            f"State {state} | {label} | n={n:,} | mean={mean_ret:.6f} "
            f"| std={std_ret:.6f} | skew={skew:.2f} | ex.kurt={kurt:.2f} | GMM k={selected_k} | BIC={bic:.1f}"
        )

        ax.set_title(f"{label}\nn={n:,} | GMM k={selected_k} | BIC={bic:.1f}")
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


def _write_html_report(
    report_path,
    trace,
    summary_df,
    timeline_path,
    timeline_fig,
    gmm_path,
    fetch_start,
    analysis_start,
    analysis_end,
    requested_k,
    raw_k,
    pca_feature_names=None,
    pca_loadings_df=None,
    pca_summary_df=None,
    pca_feature_influence_df=None,
    pca_dominant_df=None,
):
    label_counts = (
        trace["Regime_Label"]
        .value_counts()
        .rename_axis("regime")
        .reset_index(name="rows")
    )
    raw_counts = pd.DataFrame()
    if "Raw_Regime_Label" in trace.columns:
        raw_counts = (
            trace["Raw_Regime_Label"]
            .value_counts()
            .rename_axis("raw_hmm_label")
            .reset_index(name="rows")
        )

    stress_cols = [
        "date",
        "Regime_Label",
        "Raw_Regime_Label",
        "Stress_Overlay",
        "Stress_21d_Drawdown",
        "Stress_5d_Log_Return",
        "SPY_Close",
        "VIX_Close",
        "SPY_Log_Return",
    ]
    trace_for_table = trace.reset_index().rename(columns={trace.index.name or "index": "date"})
    if "date" not in trace_for_table.columns:
        trace_for_table = trace.reset_index(names="date")
    stress_rows = trace_for_table[
        trace_for_table.get("Stress_Overlay", pd.Series("none", index=trace_for_table.index)).ne("none")
    ]
    stress_rows = stress_rows[[c for c in stress_cols if c in stress_rows.columns]]

    refit_cols = [
        "HMM_Refit_ID",
        "HMM_Refit_Date",
        "HMM_Refit_Method",
        "HMM_Refit_Fallback_Reason",
        "HMM_Prior_Alignment_Order",
        "HMM_Variance_Sort_Order",
        "HMM_State_Variances",
        "HMM_Semantic_Drift_Flag",
    ]
    refit_table = pd.DataFrame()
    if "HMM_Refit_ID" in trace_for_table.columns:
        refit_table = (
            trace_for_table[[c for c in refit_cols if c in trace_for_table.columns]]
            .drop_duplicates(subset=["HMM_Refit_ID"])
            .sort_values("HMM_Refit_ID")
        )
    fallback_table = pd.DataFrame()
    if not refit_table.empty and "HMM_Refit_Fallback_Reason" in refit_table.columns:
        fallback_table = refit_table[refit_table["HMM_Refit_Fallback_Reason"].notna()]
    drift_table = pd.DataFrame()
    if not refit_table.empty and "HMM_Semantic_Drift_Flag" in refit_table.columns:
        drift_table = refit_table[refit_table["HMM_Semantic_Drift_Flag"].fillna(False).astype(bool)]

    pca_feature_names = list(pca_feature_names or [])
    input_rows = []
    for col in pca_feature_names:
        if "Return" in col or "Log" in col:
            role = "PCA/HMM stationary return feature"
        elif "Vol" in col or "VIX" in col or "VVIX" in col:
            role = "PCA/HMM volatility or risk feature"
        elif "Rate" in col or "Yield" in col or "DGS" in col or "FED" in col:
            role = "PCA/HMM macro rate feature"
        else:
            role = "PCA/HMM fused stationary feature"
        input_rows.append({"data": col, "role": role, "timestamp": "close_T_for_next_session"})
    overlay_cols = [
        ("SPY_Close", "raw overlay and chart level"),
        ("VIX_Close", "raw overlay risk level"),
        ("SPY_Log_Return", "diagnostic daily return"),
        ("Stress_21d_Drawdown", "raw stress overlay"),
        ("Stress_5d_Log_Return", "raw stress overlay"),
        ("Stress_1d_Log_Return", "raw stress overlay"),
    ]
    for col, role in overlay_cols:
        if col in trace.columns:
            input_rows.append({"data": col, "role": role, "timestamp": "close_T_for_next_session"})
    input_table = pd.DataFrame(input_rows)

    timeline_uri = _image_data_uri(timeline_path)
    gmm_uri = _image_data_uri(gmm_path)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timeline_html = timeline_fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True},
        post_script="""
var gd = document.getElementById('{plot_id}');
var _rescaleTimer = null;
var _isRescaling = false;

gd.on('plotly_relayout', function(eventData) {
  if (_isRescaling) return;

  var xKey = null;
  for (var key in eventData) {
    if (key.indexOf('xaxis') === 0) {
      xKey = key.split('.')[0];
      break;
    }
  }
  if (!xKey) return;

  if (eventData[xKey + '.autorange'] === true) {
    _isRescaling = true;
    var reset = {};
    for (var layoutKey in gd.layout) {
      if (layoutKey.match(/^yaxis\\d*$/)) reset[layoutKey + '.autorange'] = true;
    }
    reset['yaxis3.range'] = [0, 1];
    Plotly.relayout(gd, reset).then(function() { _isRescaling = false; });
    return;
  }

  var range = gd.layout[xKey] && gd.layout[xKey].range;
  if (!range || !range[0]) return;

  if (_rescaleTimer) clearTimeout(_rescaleTimer);
  _rescaleTimer = setTimeout(function() {
    var update = {};
    var x0 = new Date(range[0]).getTime();
    var x1 = new Date(range[1]).getTime();
    if (isNaN(x0) || isNaN(x1)) return;

    var yRanges = {};
    for (var i = 0; i < gd.data.length; i++) {
      var trace = gd.data[i];
      if (!trace.x || !trace.y || trace.visible === false) continue;
      var yax = trace.yaxis || 'y';
      var yaxName = 'yaxis' + (yax === 'y' ? '' : yax.substring(1));
      if (yaxName === 'yaxis3') continue;

      var yMin = Infinity;
      var yMax = -Infinity;
      for (var j = 0; j < trace.x.length; j++) {
        var tx = new Date(trace.x[j]).getTime();
        if (tx >= x0 && tx <= x1) {
          var yv = trace.y[j];
          if (yv !== null && yv !== undefined && isFinite(yv)) {
            yMin = Math.min(yMin, yv);
            yMax = Math.max(yMax, yv);
          }
        }
      }
      if (yMin !== Infinity) {
        if (!yRanges[yaxName]) yRanges[yaxName] = {min: yMin, max: yMax};
        else {
          yRanges[yaxName].min = Math.min(yRanges[yaxName].min, yMin);
          yRanges[yaxName].max = Math.max(yRanges[yaxName].max, yMax);
        }
      }
    }

    for (var axName in yRanges) {
      var r = yRanges[axName];
      var span = r.max - r.min;
      if (span === 0) span = Math.abs(r.max) * 0.1 || 1;
      var pad = span * 0.05;
      update[axName + '.range'] = [r.min - pad, r.max + pad];
      update[axName + '.autorange'] = false;
    }
    update['yaxis3.range'] = [0, 1];

    if (Object.keys(update).length > 0) {
      _isRescaling = true;
      Plotly.relayout(gd, update).then(function() { _isRescaling = false; });
    }
  }, 100);
});
""",
    )

    flow_svg = """
<svg class="flow-svg" viewBox="0 0 1120 250" role="img" aria-label="Regime detection sequence">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#334155"></path>
    </marker>
  </defs>
  <g class="flow-node is-active" data-detail="data" tabindex="0" role="button" aria-label="Show Cached Market Data details">
    <rect x="20" y="70" width="150" height="80" rx="8"></rect>
    <text x="95" y="103">Cached Market</text><text x="95" y="125">Data</text>
  </g>
  <g class="flow-node" data-detail="features" tabindex="0" role="button" aria-label="Show Stationary Features details">
    <rect x="210" y="70" width="150" height="80" rx="8"></rect>
    <text x="285" y="103">Stationary</text><text x="285" y="125">Features</text>
  </g>
  <g class="flow-node" data-detail="scaling" tabindex="0" role="button" aria-label="Show Rolling Robust Scaling details">
    <rect x="400" y="70" width="150" height="80" rx="8"></rect>
    <text x="475" y="103">Rolling Robust</text><text x="475" y="125">Scaling</text>
  </g>
  <g class="flow-node" data-detail="pca" tabindex="0" role="button" aria-label="Show Causal Standard PCA details">
    <rect x="590" y="70" width="150" height="80" rx="8"></rect>
    <text x="665" y="103">Causal Standard</text><text x="665" y="125">PCA</text>
  </g>
  <g class="flow-node" data-detail="hmm" tabindex="0" role="button" aria-label="Show Walk-Forward Gaussian HMM details">
    <rect x="780" y="70" width="150" height="80" rx="8"></rect>
    <text x="855" y="103">Walk-Forward</text><text x="855" y="125">Gaussian HMM</text>
  </g>
  <g class="flow-node" data-detail="overlay" tabindex="0" role="button" aria-label="Show Three-Regime Overlay details">
    <rect x="970" y="70" width="130" height="80" rx="8"></rect>
    <text x="1035" y="103">3-Regime</text><text x="1035" y="125">Overlay</text>
  </g>
  <g stroke="#334155" stroke-width="2" marker-end="url(#arrow)">
    <line x1="170" y1="110" x2="210" y2="110"></line>
    <line x1="360" y1="110" x2="400" y2="110"></line>
    <line x1="550" y1="110" x2="590" y2="110"></line>
    <line x1="740" y1="110" x2="780" y2="110"></line>
    <line x1="930" y1="110" x2="970" y2="110"></line>
  </g>
  <g font-family="Inter, Arial, sans-serif" font-size="12" fill="#475569" text-anchor="middle">
    <text x="95" y="185">SPY, VIX and derived returns</text>
    <text x="285" y="185">no raw SPY level in HMM</text>
    <text x="475" y="185">past-only window</text>
    <text x="665" y="185">fit on data before T</text>
    <text x="855" y="185">refit every 21 samples</text>
    <text x="1035" y="185">Expansion / Decline / Crisis</text>
  </g>
</svg>
"""

    flow_detail_html = """
<div class="flow-details">
  <div class="flow-detail is-active" id="flow-detail-data">
    <h3>Cached Market Data</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Cached market data detail">
      <rect x="20" y="45" width="130" height="60"></rect><text x="85" y="80">SPY OHLCV</text>
      <rect x="190" y="45" width="130" height="60"></rect><text x="255" y="80">VIX/VVIX</text>
      <rect x="360" y="45" width="130" height="60"></rect><text x="425" y="80">Macro Data</text>
      <rect x="560" y="45" width="160" height="60"></rect><text x="640" y="80">Date-Aligned Frame</text>
      <line x1="150" y1="75" x2="190" y2="75"></line><line x1="320" y1="75" x2="360" y2="75"></line><line x1="490" y1="75" x2="560" y2="75"></line>
    </svg>
    <p class="muted">Loads cached Yahoo/FRED-style inputs for the fetch window, then trims to the requested analysis end date.</p>
  </div>
  <div class="flow-detail" id="flow-detail-features">
    <h3>Stationary Features</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Stationary feature detail">
      <rect x="20" y="45" width="140" height="60"></rect><text x="90" y="72">Raw Levels</text><text x="90" y="92">and Returns</text>
      <rect x="220" y="45" width="150" height="60"></rect><text x="295" y="72">Transform</text><text x="295" y="92">Stationary</text>
      <rect x="430" y="45" width="150" height="60"></rect><text x="505" y="72">Drop Excluded</text><text x="505" y="92">Features</text>
      <rect x="640" y="45" width="100" height="60"></rect><text x="690" y="80">Feature Matrix</text>
      <line x1="160" y1="75" x2="220" y2="75"></line><line x1="370" y1="75" x2="430" y2="75"></line><line x1="580" y1="75" x2="640" y2="75"></line>
    </svg>
    <p class="muted">Builds the fused dataset used by PCA/HMM. The Data Inputs table lists the actual feature names passed into PCA.</p>
  </div>
  <div class="flow-detail" id="flow-detail-scaling">
    <h3>Rolling Robust Scaling</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Rolling robust scaling detail">
      <rect x="20" y="45" width="160" height="60"></rect><text x="100" y="72">Rolling Window</text><text x="100" y="92">up to 5 years</text>
      <rect x="250" y="45" width="150" height="60"></rect><text x="325" y="72">Median</text><text x="325" y="92">and IQR</text>
      <rect x="470" y="45" width="140" height="60"></rect><text x="540" y="72">Scale Row T</text><text x="540" y="92">Causally</text>
      <line x1="180" y1="75" x2="250" y2="75"></line><line x1="400" y1="75" x2="470" y2="75"></line>
    </svg>
    <p class="muted">Uses only observations available at each timestamp. No global scaler is fit on the full backtest span.</p>
  </div>
  <div class="flow-detail" id="flow-detail-pca">
    <h3>Causal Standard PCA</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Causal PCA detail">
      <rect x="20" y="45" width="150" height="60"></rect><text x="95" y="72">Scaled</text><text x="95" y="92">Feature Matrix</text>
      <rect x="240" y="45" width="170" height="60"></rect><text x="325" y="72">Fit PCA on</text><text x="325" y="92">Rows before T</text>
      <rect x="500" y="45" width="160" height="60"></rect><text x="580" y="72">Project T into</text><text x="580" y="92">PC1 / PC2</text>
      <line x1="170" y1="75" x2="240" y2="75"></line><line x1="410" y1="75" x2="500" y2="75"></line>
    </svg>
    <p class="muted">Causal projections are computed in parallel chunks, with sign alignment to keep component orientation stable over time.</p>
  </div>
  <div class="flow-detail" id="flow-detail-hmm">
    <h3>Walk-Forward Gaussian HMM</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Walk-forward HMM detail">
      <rect x="20" y="45" width="130" height="60"></rect><text x="85" y="72">PC Trace</text>
      <rect x="210" y="45" width="150" height="60"></rect><text x="285" y="72">Warm-Start</text><text x="285" y="92">Prior HMM</text>
      <rect x="420" y="45" width="140" height="60"></rect><text x="490" y="72">Fit / Align</text><text x="490" y="92">States</text>
      <rect x="620" y="45" width="120" height="60"></rect><text x="680" y="72">Filtered</text><text x="680" y="92">Probabilities</text>
      <line x1="150" y1="75" x2="210" y2="75"></line><line x1="360" y1="75" x2="420" y2="75"></line><line x1="560" y1="75" x2="620" y2="75"></line>
    </svg>
    <p class="muted">Refits every 21 samples. The trace records refit method, state alignment, variance order, and semantic drift flags.</p>
  </div>
  <div class="flow-detail" id="flow-detail-overlay">
    <h3>Three-Regime Overlay</h3>
    <svg viewBox="0 0 760 150" role="img" aria-label="Three-regime overlay detail">
      <rect x="20" y="45" width="160" height="60"></rect><text x="100" y="72">Raw SPY/VIX</text><text x="100" y="92">Stress Checks</text>
      <rect x="260" y="45" width="140" height="60"></rect><text x="330" y="72">Expansion</text><text x="330" y="92">Default</text>
      <rect x="470" y="20" width="130" height="45"></rect><text x="535" y="48">Cautious</text>
      <rect x="470" y="85" width="130" height="45"></rect><text x="535" y="113">Panic</text>
      <line x1="180" y1="75" x2="260" y2="75"></line><line x1="400" y1="75" x2="470" y2="42"></line><line x1="400" y1="75" x2="470" y2="108"></line>
    </svg>
    <p class="muted">Final actionable labels require observable close-T stress. Raw HMM archetype labels remain audit-only.</p>
  </div>
</div>
"""

    css = """
body { margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f5f7fb; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }
h1 { font-size: 32px; margin: 0 0 8px; }
h2 { font-size: 22px; margin: 30px 0 12px; }
p { line-height: 1.55; }
.muted { color: #64748b; }
.panel { background: #fff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 20px; margin: 18px 0; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.metric { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }
.metric b { display: block; font-size: 22px; margin-top: 4px; }
.plot { width: 100%; border: 1px solid #dbe3ef; border-radius: 8px; background: white; }
.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.data-table th, .data-table td { border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }
.data-table th { background: #f1f5f9; color: #334155; }
.callout { border-left: 4px solid #2563eb; background: #eff6ff; padding: 12px 14px; border-radius: 6px; }
.sequence li { margin: 8px 0; line-height: 1.5; }
svg { width: 100%; height: auto; }
.flow-svg .flow-node { cursor: pointer; outline: none; }
.flow-svg .flow-node rect { fill: #f8fafc; stroke: #334155; stroke-width: 2; transition: fill 120ms ease, stroke 120ms ease; }
.flow-svg .flow-node text { font-family: Inter, Arial, sans-serif; font-size: 15px; fill: #0f172a; text-anchor: middle; pointer-events: none; }
.flow-svg .flow-node:hover rect, .flow-svg .flow-node:focus rect, .flow-svg .flow-node.is-active rect { fill: #e0f2fe; stroke: #0369a1; }
.flow-details { margin-top: 14px; }
.flow-detail { display: none; border: 1px solid #dbe3ef; border-radius: 8px; background: #f8fafc; padding: 14px; }
.flow-detail.is-active { display: block; }
.flow-detail h3 { margin: 0 0 10px; font-size: 17px; }
.flow-detail svg { max-height: 180px; }
.flow-detail rect { fill: #fff; stroke: #334155; stroke-width: 2; rx: 8; }
.flow-detail line { stroke: #334155; stroke-width: 2; marker-end: url(#arrow); }
.flow-detail text { font-family: Inter, Arial, sans-serif; font-size: 14px; fill: #0f172a; text-anchor: middle; }
code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
"""

    flow_script = """
<script>
document.querySelectorAll('.flow-node').forEach((node) => {
  const activate = () => {
    const target = node.getAttribute('data-detail');
    document.querySelectorAll('.flow-node').forEach((n) => n.classList.toggle('is-active', n === node));
    document.querySelectorAll('.flow-detail').forEach((panel) => {
      panel.classList.toggle('is-active', panel.id === `flow-detail-${target}`);
    });
  };
  node.addEventListener('click', activate);
  node.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      activate();
    }
  });
});
</script>
"""

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market Regime Diagnostics</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Market Regime Diagnostics</h1>
  <p class="muted">Generated {html.escape(generated_at)}. Signal timing: close_T_for_next_session.</p>

  <section class="panel grid">
    <div class="metric">Fetch window<b>{html.escape(fetch_start)} to {html.escape(analysis_end)}</b></div>
    <div class="metric">Analysis window<b>{html.escape(analysis_start)} to {html.escape(analysis_end)}</b></div>
    <div class="metric">HMM states<b>K = {int(requested_k)} raw, 3 displayed</b></div>
  </section>

  <section class="panel">
    <h2>Detection Flow</h2>
    {flow_svg}
    {flow_detail_html}
    <p class="callout">The report uses a fixed three-regime final taxonomy: <code>Expansion</code>, <code>Cautious Decline</code>, and <code>Panic / Crisis</code>. Raw HMM states and labels remain in the CSV as audit columns.</p>
  </section>

  <section class="panel">
    <h2>Data Inputs</h2>
    {_df_to_html_table(input_table)}
  </section>

  <section class="panel">
    <h2>PCA Structure</h2>
    <p class="muted">These tables show the fitted PCA loadings and which inputs actually dominated each component. Low mean absolute loading means the feature contributed weakly across the fitted components.</p>
    <h3>Component Summary</h3>
    {_df_to_html_table(pca_summary_df)}
    <h3>Feature Influence</h3>
    {_df_to_html_table(pca_feature_influence_df, max_rows=40)}
    <h3>Max Absolute Loading</h3>
    {_df_to_html_table(pca_dominant_df, max_rows=40)}
    <h3>Top Loadings</h3>
    {_df_to_html_table(pca_loadings_df, max_rows=30)}
  </section>

  <section class="panel">
    <h2>Training Sequence</h2>
    <ol class="sequence">
      <li>Load cached Yahoo/FRED-style market inputs for the fetch window.</li>
      <li>Create stationary features, including SPY log returns, realized volatility, VIX, VVIX/VIX-style ratios, and other available macro/market columns.</li>
      <li>Apply rolling robust scaling using only historical observations available at each timestamp.</li>
      <li>Fit standard PCA causally: each row T is projected using PCA loadings fit on observations before T.</li>
      <li>Refit a Gaussian HMM every 21 samples in walk-forward order with default <code>K=3</code>. Routine refits warm-start from the prior HMM, then record prior-state alignment and variance-sort metadata.</li>
      <li>Apply the causal stress overlay using raw close-T SPY drawdown/returns and VIX. This guards against fast drawdowns that a persistent HMM can under-react to.</li>
      <li>Fit a separate BIC-selected GMM to SPY daily log returns inside each final regime. Candidate counts include <code>k=1</code>, so a single Gaussian wins when extra mixture components do not improve penalized fit.</li>
    </ol>
  </section>

  <section class="panel">
    <h2>Regime Timeline</h2>
    {timeline_html}
  </section>

  <section class="panel">
    <h2>SPY Log-Return GMM Fits</h2>
    <img class="plot" src="{gmm_uri}" alt="SPY log return GMM fits by regime">
  </section>

  <section class="panel">
    <h2>Final Regime Counts</h2>
    {_df_to_html_table(label_counts)}
  </section>

  <section class="panel">
    <h2>Raw HMM Label Counts</h2>
    {_df_to_html_table(raw_counts)}
  </section>

  <section class="panel">
    <h2>HMM Refit Alignment Audit</h2>
    <p class="muted">One row per walk-forward refit. Raw HMM states are aligned and sorted before they are used for probabilities; final trading regimes still come from the three-regime overlay.</p>
    {_df_to_html_table(refit_table, max_rows=60)}
  </section>

  <section class="panel">
    <h2>Warm-Start Fallbacks</h2>
    {_df_to_html_table(fallback_table, max_rows=60)}
  </section>

  <section class="panel">
    <h2>Semantic Drift Flags</h2>
    <p class="muted">A flag means at least one state's fitted emission variance changed by more than the configured adjacent-refit tolerance. It is an audit warning, not an automatic model failure.</p>
    {_df_to_html_table(drift_table, max_rows=60)}
  </section>

  <section class="panel">
    <h2>GMM Fit Summary</h2>
    {_df_to_html_table(summary_df)}
  </section>

  <section class="panel">
    <h2>Stress Overlay Rows</h2>
    <p class="muted">First rows where the final 3-regime signal was set by raw market stress conditions.</p>
    {_df_to_html_table(stress_rows, max_rows=40)}
  </section>
</main>
{flow_script}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)


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
