"""Generate a causal, explicitly out-of-sample regime review dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd

from live_trading.data_ingestion import DataIngestor
from live_trading.ev_engine import fetch_historical_data, train_regime_hmm
from live_trading.market_sessions import (
    latest_available_session_before,
    latest_completed_nyse_session,
)


DEFAULT_TEST_START = "2020-01-02"
DEFAULT_OUTPUT = Path("quant-review-0430/quant_review_data.csv")


def generate_quant_data(
    *,
    test_start=DEFAULT_TEST_START,
    test_end=None,
    output_path=DEFAULT_OUTPUT,
):
    """Write only causal OOS rows and an adjacent audit manifest."""

    end = test_end or latest_completed_nyse_session()
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(end)
    if (
        test_start_ts.tzinfo is not None
        or test_end_ts.tzinfo is not None
        or test_start_ts != test_start_ts.normalize()
        or test_end_ts != test_end_ts.normalize()
        or test_end_ts < test_start_ts
    ):
        raise ValueError("INVALID_QUANT_REVIEW_TEST_WINDOW")

    print("Fetching historical data...")
    history = fetch_historical_data()
    history = history.loc[history.index <= test_end_ts].copy()
    if history.empty:
        raise RuntimeError("HISTORICAL_REGIME_DATA_UNAVAILABLE")
    fit_end = latest_available_session_before(
        history.index,
        test_start_ts,
    )
    calibration_start = pd.Timestamp(history.index.min()).strftime(
        "%Y-%m-%d"
    )

    print(
        f"Training causal HMM: calibration {calibration_start} to "
        f"{fit_end}; OOS {test_start_ts.date()} to {test_end_ts.date()}."
    )
    best_hmm, best_k, feature_df = train_regime_hmm(
        history,
        n_components=3,
        expanding_window=True,
        fit_end=fit_end,
    )
    if best_hmm is None or feature_df.empty:
        raise RuntimeError("CAUSAL_REGIME_TRACE_UNAVAILABLE")
    feature_df = feature_df.loc[test_start_ts:test_end_ts].copy()
    if feature_df.empty:
        raise RuntimeError("OUT_OF_SAMPLE_REGIME_TRACE_UNAVAILABLE")

    ingestor = DataIngestor()
    raw_levels = pd.concat(
        [
            ingestor.fetch_yf_data(calibration_start, end),
            ingestor.fetch_fred_data(calibration_start, end),
        ],
        axis=1,
    ).ffill()
    final_df = raw_levels.join(feature_df, how="inner")
    if final_df.empty:
        raise RuntimeError("QUANT_REVIEW_JOIN_UNAVAILABLE")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output)
    manifest = {
        "schema": "causal-quant-review.v1",
        "calibration_start": calibration_start,
        "calibration_end": fit_end,
        "test_start": test_start_ts.strftime("%Y-%m-%d"),
        "test_end": test_end_ts.strftime("%Y-%m-%d"),
        "inference_method": "walk_forward_refit",
        "regime_signal_timestamp": "close_T_for_next_session",
        "raw_hmm_taxonomy_scope": "per_row_refit",
        "fitted_state_count": int(best_k),
        "validity_status": "UNVERIFIED",
        "execution_eligible": False,
    }
    manifest_path = output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"Data saved to {output}")
    print(f"Audit manifest saved to {manifest_path}")
    return output, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a causal out-of-sample regime dataset."
    )
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    generate_quant_data(
        test_start=args.test_start,
        test_end=args.test_end,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
