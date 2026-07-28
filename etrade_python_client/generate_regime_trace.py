"""Generate a causal regime trace for a declared out-of-sample window."""

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import train_regime_hmm
from live_trading.market_sessions import latest_available_session_before


def run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return loop.run_until_complete(coro)


def generate_trace(
    *,
    fetch_start="2015-01-01",
    test_start="2025-01-02",
    test_end="2025-06-01",
    output_dir="regime_review_2025",
):
    """Persist engine-owned causal posteriors without final-model replay."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)
    if test_end_ts < test_start_ts:
        raise ValueError("INVALID_REGIME_TRACE_TEST_WINDOW")

    ingestor = DataIngestor()
    raw_levels = pd.concat(
        [
            ingestor.fetch_yf_data(fetch_start, test_end),
            ingestor.fetch_fred_data(fetch_start, test_end),
        ],
        axis=1,
    ).ffill()
    if raw_levels.empty:
        raise RuntimeError("REGIME_TRACE_MARKET_DATA_UNAVAILABLE")
    fit_end = latest_available_session_before(
        raw_levels.index,
        test_start_ts,
    )
    print(
        f"Calibration {fetch_start} to {fit_end}; "
        f"OOS test {test_start} to {test_end}."
    )

    stationary_df = run_sync(
        ingestor.build_fused_dataset(
            fetch_start,
            test_end,
            scale=False,
            fit_end=fit_end,
        )
    )
    if stationary_df.empty:
        raise RuntimeError("REGIME_TRACE_FEATURE_DATA_UNAVAILABLE")
    stationary_df.to_csv(output / "stationary_features_full.csv")
    raw_levels.to_csv(output / "raw_price_levels.csv")

    best_hmm, best_k, feature_df = train_regime_hmm(
        stationary_df,
        n_components=None,
        expanding_window=True,
        fit_end=fit_end,
    )
    if best_hmm is None or feature_df.empty:
        raise RuntimeError("CAUSAL_REGIME_TRACE_UNAVAILABLE")
    feature_df = feature_df.loc[test_start_ts:test_end_ts].copy()
    raw_test = raw_levels.loc[test_start_ts:test_end_ts].copy()
    if feature_df.empty:
        raise RuntimeError("OUT_OF_SAMPLE_REGIME_TRACE_UNAVAILABLE")

    feature_df.to_csv(output / "regime_results.csv")
    raw_test.to_csv(output / "raw_price_levels_test.csv")
    params = {
        "n_components": int(best_hmm.n_components),
        "startprob": best_hmm.startprob_.tolist(),
        "transmat": best_hmm.transmat_.tolist(),
        "feature_names": best_hmm.feature_names_,
        "raw_hmm_taxonomy_id": best_hmm.raw_hmm_taxonomy_id_,
        "model_training_end": best_hmm.model_training_end_,
    }
    for attribute, key in (
        ("means_", "means"),
        ("covars_", "covars"),
        ("weights_", "weights"),
    ):
        if hasattr(best_hmm, attribute):
            params[key] = getattr(best_hmm, attribute).tolist()
    with (output / "model_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as parameter_file:
        json.dump(params, parameter_file, indent=2, sort_keys=True)
        parameter_file.write("\n")

    manifest = {
        "schema": "causal-regime-trace.v1",
        "calibration_start": fetch_start,
        "calibration_end": fit_end,
        "test_start": test_start_ts.strftime("%Y-%m-%d"),
        "test_end": test_end_ts.strftime("%Y-%m-%d"),
        "inference_method": "walk_forward_refit",
        "regime_signal_timestamp": "close_T_for_next_session",
        "raw_hmm_taxonomy_scope": "per_row_refit",
        "fitted_state_count": int(best_k),
        "validity_status": "UNVERIFIED",
        "execution_eligible": False,
        "artifacts": [
            "stationary_features_full.csv",
            "raw_price_levels.csv",
            "raw_price_levels_test.csv",
            "regime_results.csv",
            "model_parameters.json",
        ],
    }
    with (output / "trace_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"Trace generation complete: {output}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate a causal out-of-sample regime trace."
    )
    parser.add_argument("--fetch-start", default="2015-01-01")
    parser.add_argument("--test-start", default="2025-01-02")
    parser.add_argument("--test-end", default="2025-06-01")
    parser.add_argument("--output-dir", default="regime_review_2025")
    args = parser.parse_args()
    generate_trace(
        fetch_start=args.fetch_start,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
