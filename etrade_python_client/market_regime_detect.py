import argparse
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime, timedelta

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import (
    build_regime_return_arrays,
    save_regime_cache,
    train_regime_hmm,
)
from live_trading.market_sessions import (
    latest_completed_nyse_session,
    latest_nyse_session_before,
)

DEFAULT_CALIBRATION_YEARS = 8
REGIME_REVIEW_SCHEMA = "causal-regime-review.v1"


@dataclass(frozen=True)
class RegimeReviewWindow:
    """Explicit calibration and out-of-sample boundaries for one review."""

    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str

    def __post_init__(self):
        calibration_start = _canonical_date(
            self.calibration_start,
            "calibration_start",
        )
        calibration_end = _canonical_date(
            self.calibration_end,
            "calibration_end",
        )
        test_start = _canonical_date(self.test_start, "test_start")
        test_end = _canonical_date(self.test_end, "test_end")
        if calibration_start > calibration_end:
            raise ValueError(
                "calibration_start must not follow calibration_end"
            )
        if calibration_end >= test_start:
            raise ValueError(
                "calibration_end must be strictly before test_start"
            )
        if test_end < test_start:
            raise ValueError("test_end must not precede test_start")

    def manifest(self):
        return {
            "schema": REGIME_REVIEW_SCHEMA,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "inference_method": "walk_forward_refit",
            "regime_signal_timestamp": "close_T_for_next_session",
            "validity_status": "UNVERIFIED",
            "execution_eligible": False,
        }


def _canonical_date(value, field_name):
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a canonical ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a canonical ISO date")
    return pd.Timestamp(parsed)


def _resolve_review_window(
    start_date,
    end_date,
    *,
    calibration_start=None,
    calibration_end=None,
):
    """Resolve a causal calibration prefix strictly before the test range."""

    test_start = _canonical_date(start_date, "start_date")
    test_end = _canonical_date(end_date, "end_date")
    if test_end < test_start:
        raise ValueError("test_end must not precede test_start")

    if calibration_end is None:
        resolved_calibration_end = latest_nyse_session_before(test_start)
    else:
        requested_calibration_end = _canonical_date(
            calibration_end,
            "calibration_end",
        )
        if requested_calibration_end >= test_start:
            raise ValueError(
                "calibration_end must be strictly before test_start"
            )
        resolved_calibration_end = latest_nyse_session_before(
            requested_calibration_end + pd.Timedelta(days=1)
        )
    calibration_end_ts = pd.Timestamp(resolved_calibration_end)
    if calibration_end_ts >= test_start:
        raise ValueError(
            "calibration_end must be strictly before test_start"
        )

    if calibration_start is None:
        calibration_start_ts = test_start - pd.DateOffset(
            years=DEFAULT_CALIBRATION_YEARS
        )
    else:
        calibration_start_ts = _canonical_date(
            calibration_start,
            "calibration_start",
        )
    if calibration_start_ts > calibration_end_ts:
        raise ValueError(
            "calibration_start must not follow calibration_end"
        )

    return RegimeReviewWindow(
        calibration_start=calibration_start_ts.strftime("%Y-%m-%d"),
        calibration_end=calibration_end_ts.strftime("%Y-%m-%d"),
        test_start=test_start.strftime("%Y-%m-%d"),
        test_end=test_end.strftime("%Y-%m-%d"),
    )


def _train_causal_review_trace(frame, window, *, n_components=None):
    """Train on the declared prefix and return only requested OOS rows."""

    if not isinstance(window, RegimeReviewWindow):
        raise TypeError("window must be RegimeReviewWindow")
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or not frame.index.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        raise ValueError("review frame must be ordered and non-empty")
    frame_index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    calibration_end = pd.Timestamp(window.calibration_end)
    if calibration_end not in frame_index:
        raise ValueError("CALIBRATION_END_DATA_UNAVAILABLE")

    best_hmm, best_k, causal_trace = train_regime_hmm(
        frame,
        n_components=n_components,
        expanding_window=True,
        fit_end=window.calibration_end,
    )
    if best_hmm is None or causal_trace.empty:
        raise ValueError("CAUSAL_REGIME_TRACE_UNAVAILABLE")
    test_trace = causal_trace.loc[
        pd.Timestamp(window.test_start):pd.Timestamp(window.test_end)
    ].copy()
    if test_trace.empty:
        raise ValueError("OUT_OF_SAMPLE_REGIME_TRACE_UNAVAILABLE")
    test_trace.attrs["regime_review_manifest"] = window.manifest()
    return best_hmm, best_k, test_trace


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


def _build_completed_session_return_arrays(*, horizon, force_refit=False):
    """Build review buckets only through a fully closed NYSE session."""

    as_of_date = latest_completed_nyse_session()
    cache_key = datetime.fromisoformat(as_of_date).date().toordinal()
    return build_regime_return_arrays(
        cache_key,
        horizon=horizon,
        force_refit=force_refit,
        as_of_date=as_of_date,
    )


def run_regime_detection(
    start_date,
    end_date,
    n_components=None,
    save_results=False,
    *,
    calibration_start=None,
    calibration_end=None,
):
    window = _resolve_review_window(
        start_date,
        end_date,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
    )
    print(
        "\n🚀 [Regime Detector] Starting causal analysis "
        f"for test period: {window.test_start} to {window.test_end}"
    )
    print(
        "  Calibration prefix: "
        f"{window.calibration_start} to {window.calibration_end}"
    )
    
    ingestor = DataIngestor()
    
    # 1. High-Dimensional Data Ingestion
    print("📡 Fetching feature set (Macro + Price + Vol)...")
    stationary_df = run_sync(
        ingestor.build_fused_dataset(
            window.calibration_start,
            window.test_end,
            scale=False,
            fit_end=window.calibration_end,
        )
    )
    
    if stationary_df.empty:
        print("❌ Error: Failed to fetch data.")
        return

    # 2. Raw data for plotting (SPY and VIX)
    print("📊 Fetching raw price data for visualization...")
    raw_df = ingestor.fetch_yf_data(
        window.calibration_start,
        window.test_end,
    )
    if (
        raw_df.empty
        or not {"SPY_Close", "VIX_Close"}.issubset(raw_df.columns)
    ):
        raise ValueError("REGIME_REVIEW_MARKET_DATA_UNAVAILABLE")

    # 3. The engine owns causal posterior extraction. This caller supplies a
    # strictly pre-test fit boundary and consumes only the returned OOS trace.
    print("🧠 Training Gaussian HMM (Causal Walk-Forward)...")
    best_hmm, best_k, feature_df = _train_causal_review_trace(
        stationary_df,
        window,
        n_components=n_components,
    )
    probability_columns = [
        f"prob_state_{state}"
        for state in range(best_k)
    ]
    if (
        "HMM_State" not in feature_df.columns
        or "Regime_Label" not in feature_df.columns
        or any(
            column not in feature_df.columns
            for column in probability_columns
        )
    ):
        raise ValueError("CAUSAL_REGIME_PROVENANCE_INCOMPLETE")
    feature_df["Dominant_State"] = feature_df["HMM_State"].astype(int)

    # 4. Preserve the snapshot-time label carried by each causal row.
    plot_df = feature_df.join(raw_df[['SPY_Close', 'VIX_Close']], how='inner')
    if plot_df.empty:
        raise ValueError("REGIME_REVIEW_PLOT_DATA_UNAVAILABLE")
    
    regime_labels = {}
    print("🏷️  Reading snapshot-time regime labels...")
    for i in range(best_k):
        state_labels = plot_df.loc[
            plot_df["Dominant_State"] == i,
            "Regime_Label",
        ].dropna()
        if state_labels.empty:
            continue
        regime_labels[i] = state_labels.iloc[-1]
        print(f"  • State {i}: latest causal label={regime_labels[i]}")
    
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
    state_changes = (
        plot_df["Dominant_State"].ne(
            plot_df["Dominant_State"].shift()
        )
        | plot_df["Regime_Label"].ne(
            plot_df["Regime_Label"].shift()
        )
    ).cumsum()
    groups = plot_df.groupby(state_changes)
    
    added_to_legend = set()
    for _, group in groups:
        state = group['Dominant_State'].iloc[0]
        color = colors[state]
        label = group["Regime_Label"].iloc[0]
        
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
    
    ax_price.set_title(
        f"Causal OOS Market Regime Timeline "
        f"({window.test_start} to {window.test_end}, K={best_k})",
        fontsize=16,
        pad=20,
        fontweight='bold',
    )
    ax_price.legend(loc='upper left', framealpha=0.8)
    
    # Format X-axis with Date Labels
    n_ticks = 10
    tick_indices = np.linspace(0, len(plot_df) - 1, n_ticks, dtype=int)
    tick_labels = [plot_df['Date'].iloc[i].strftime('%Y-%m') for i in tick_indices]
    ax_prob.set_xticks(tick_indices)
    ax_prob.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    prob_data = [plot_df[f'prob_state_{i}'].values for i in range(best_k)]
    labels = [f"Raw HMM State {i}" for i in range(best_k)]
    
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
        regime_buckets, model, daily_models = (
            _build_completed_session_return_arrays(
                horizon=7,
                force_refit=True,
            )
        )
        if model:
            save_regime_cache(regime_buckets, model, daily_models)
        else:
            print("❌ Failed to generate regime return arrays for saving.")

    print("\n" + "="*50)
    print("🔍 REGIME ENGINE VALIDITY CHECK")
    print("="*50)
    print(
        "  • Calibration Range: "
        f"{window.calibration_start} to {window.calibration_end}"
    )
    print(
        f"  • OOS Test Range: {window.test_start} to {window.test_end}"
    )
    print("  • Inference Method: walk_forward_refit")
    print("  • Signal Timestamp: close_T_for_next_session")
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
    return {
        "artifact": output_fn,
        "regime_review_manifest": window.manifest(),
    }

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
    parser.add_argument(
        "--calibration-start",
        type=str,
        default=None,
        help=(
            "Calibration fetch start (YYYY-MM-DD). Defaults to eight years "
            "before --start."
        ),
    )
    parser.add_argument(
        "--calibration-end",
        type=str,
        default=None,
        help=(
            "Latest allowed calibration date (YYYY-MM-DD). It is resolved "
            "to an NYSE session strictly before --start."
        ),
    )
    
    args = parser.parse_args()
    
    run_regime_detection(
        args.start,
        args.end,
        n_components=args.k,
        save_results=args.save,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
    )
