"""Reproduce the predeclared V2 regime calibration on legacy local data.

This command performs retrospective research only.  It never downloads data,
never mutates the source caches, and never marks a detector execution-eligible.
The optional artifact output is a deterministic, content-addressed JSON record
whose manifest explicitly identifies the source as unverified legacy data.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRATCH_DIR = Path(__file__).resolve().parent
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

from live_trading.regime_calibration import (
    CalibrationCandidate,
    CausalValidationFold,
    RegimeCalibrationPlan,
    run_research_calibration,
)
from live_trading.regime_detector_v2 import (
    RegimeDetectorConfig,
    detect_regimes_from_snapshot,
)
from regime_detector_v2_audit import _legacy_snapshot, _load_prices


SELECTION_YEARS = tuple(range(2016, 2025))
RETROSPECTIVE_TEST_YEAR = 2025
PROSPECTIVE_HOLDOUT_START = pd.Timestamp("2026-07-27").date()
FROZEN_PLAN_PATH = PROJECT_ROOT / "docs" / "regime_v2_calibration_plan.json"


def _session_on_or_after(
    index: pd.DatetimeIndex,
    value: str,
) -> pd.Timestamp:
    position = int(index.searchsorted(pd.Timestamp(value), side="left"))
    if position >= len(index):
        raise ValueError(f"No session exists on or after {value}")
    return pd.Timestamp(index[position])


def _session_on_or_before(
    index: pd.DatetimeIndex,
    value: str,
) -> pd.Timestamp:
    position = int(index.searchsorted(pd.Timestamp(value), side="right")) - 1
    if position < 0:
        raise ValueError(f"No session exists on or before {value}")
    return pd.Timestamp(index[position])


def _fold_for_year(
    index: pd.DatetimeIndex,
    year: int,
    *,
    max_outcome_resolution_lag: int,
    fold_id: str,
) -> CausalValidationFold:
    evaluation_start = _session_on_or_after(index, f"{year}-01-01")
    evaluation_end = _session_on_or_before(index, f"{year}-12-31")
    evaluation_position = int(index.get_loc(evaluation_start))
    reference_end_position = (
        evaluation_position - max_outcome_resolution_lag - 1
    )
    if reference_end_position < 0:
        raise ValueError(f"Insufficient purged reference history for {year}")
    return CausalValidationFold(
        fold_id=fold_id,
        outcome_reference_start=index[0].date(),
        outcome_reference_end=index[reference_end_position].date(),
        evaluation_start=evaluation_start.date(),
        evaluation_end=evaluation_end.date(),
    )


def build_preregistered_plan(
    prices: pd.DataFrame,
) -> RegimeCalibrationPlan:
    """Return the fixed B0/C1/C3/C4 candidate protocol."""

    baseline = RegimeDetectorConfig()
    candidates = (
        CalibrationCandidate("b0_baseline", baseline),
        CalibrationCandidate(
            "c1_slow_15d",
            replace(
                baseline,
                vix_slow_window=15,
                realized_vol_window=15,
            ),
        ),
        CalibrationCandidate(
            "c3_stress_entry_080",
            replace(baseline, stress_entry_score=0.80),
        ),
        CalibrationCandidate(
            "c4_stress_confirm_4d",
            replace(baseline, stress_entry_days=4),
        ),
    )
    max_outcome_resolution_lag = 21
    return RegimeCalibrationPlan(
        protocol_id="regime_v2_b0_c1_c3_c4_20260726",
        candidates=candidates,
        control_candidate_id="b0_baseline",
        selection_folds=tuple(
            _fold_for_year(
                prices.index,
                year,
                max_outcome_resolution_lag=max_outcome_resolution_lag,
                fold_id=f"selection_{year}",
            )
            for year in SELECTION_YEARS
        ),
        retrospective_test=_fold_for_year(
            prices.index,
            RETROSPECTIVE_TEST_YEAR,
            max_outcome_resolution_lag=max_outcome_resolution_lag,
            fold_id="retrospective_2025",
        ),
        prospective_holdout_start=PROSPECTIVE_HOLDOUT_START,
        outcome_horizons=(5, 20),
        tail_percentile=0.95,
        min_evaluation_rows=120,
        max_state_occupancy=0.95,
        minimum_evaluation_coverage=0.99,
        max_switches_per_252=24.0,
        isolated_shock_followup_sessions=5,
        minimum_isolated_shock_episodes=50,
        max_false_persistent_isolated_shock_rate=0.10,
        minimum_material_improvement=0.01,
        signal_lag_sessions=1,
    )


def load_frozen_plan(
    prices: pd.DataFrame,
    *,
    plan_path: Path = FROZEN_PLAN_PATH,
) -> RegimeCalibrationPlan:
    """Load the committed protocol and fail if code defaults drift from it."""

    try:
        payload = plan_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Frozen calibration plan is unavailable: {plan_path}"
        ) from exc
    frozen = RegimeCalibrationPlan.from_json(payload)
    generated = build_preregistered_plan(prices)
    if frozen != generated:
        raise RuntimeError(
            "Frozen calibration plan does not match the current "
            "predeclared protocol; review and commit a new protocol ID "
            "instead of silently changing this run"
        )
    return frozen


def _atomic_write(path: Path, payload: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _print_metric(prefix: str, metric) -> None:
    print(
        f"{prefix}: F1={metric.tail_f1:.4f}, "
        f"recall={metric.tail_recall:.4f}, "
        f"precision={metric.tail_precision:.4f}, "
        f"Spearman={metric.ordinal_risk_spearman:.4f}, "
        f"switches/252={metric.switches_per_252:.2f}, "
        f"max occupancy={metric.maximum_state_occupancy:.3f}, "
        f"guardrail={metric.guardrail_passed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic research-only V2 calibration."
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        help="Optional destination for canonical artifact JSON.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional destination for the canonical pre-registration plan.",
    )
    args = parser.parse_args()

    prices, quarantined, conflicts = _load_prices()
    plan = load_frozen_plan(prices)
    artifact = run_research_calibration(prices, plan)

    print("REGIME V2 CAUSAL CALIBRATION")
    print(f"Protocol: {plan.protocol_id}")
    print(f"Plan SHA-256: {plan.plan_sha256}")
    print(f"Artifact SHA-256: {artifact.artifact_sha256}")
    print(
        "Calibration input: "
        f"{artifact.data_manifest.first_session} through "
        f"{artifact.data_manifest.last_session}"
    )
    print(
        "Source provenance: legacy_normalized_unverified "
        "(retrospective research only)"
    )
    print(
        "Selection folds: "
        f"{plan.selection_folds[0].evaluation_start} through "
        f"{plan.selection_folds[-1].evaluation_end}; "
        "each reference is purged through the 21-session outcome resolution"
    )
    print(
        "Retrospective test: "
        f"{plan.retrospective_test.evaluation_start} through "
        f"{plan.retrospective_test.evaluation_end}"
    )
    print(f"Inference: {plan.inference_method}")
    print(f"Regime lag: {plan.signal_lag_sessions} trading session")
    print(
        "Return buckets: post-lag close outcomes T+1 through T+h+1, "
        "resolved before each evaluation decision"
    )
    print(f"Raw best: {artifact.raw_best_candidate_id}")
    print(f"Selected: {artifact.selected_candidate_id}")
    print(f"Selection reasons: {artifact.selection_reason_codes}")
    print(f"Promotion status: {artifact.promotion_status}")
    print(f"Execution eligible: {artifact.execution_eligible}")
    print(f"Prospective holdout starts: {plan.prospective_holdout_start}")
    print(f"Quarantined non-NYSE rows: {len(quarantined)}")
    print(f"Conflicting normalized cache rows: {len(conflicts)}")

    print("\nSelection summary")
    for summary in artifact.candidate_summaries:
        print(
            f"{summary.candidate_id}: F1={summary.tail_f1:.4f}, "
            f"recall={summary.tail_recall:.4f}, "
            f"precision={summary.tail_precision:.4f}, "
            f"mean Spearman={summary.mean_ordinal_risk_spearman:.4f}, "
            f"score={summary.selection_score:.4f}, "
            f"switches/252={summary.mean_switches_per_252:.2f}, "
            f"coverage={summary.evaluation_coverage:.3f}, "
            f"max occupancy={summary.maximum_state_occupancy:.3f}, "
            "isolated shock false-stress="
            f"{summary.false_persistent_on_isolated_shock}/"
            f"{summary.isolated_shocks}, "
            f"guardrail={summary.guardrail_passed}"
        )

    print("\nRetrospective 2025")
    for metric in artifact.retrospective_test_metrics:
        _print_metric(metric.candidate_id, metric)

    snapshot = _legacy_snapshot(prices)
    result = detect_regimes_from_snapshot(
        snapshot,
        artifact.selected_candidate.config,
    )
    periods = {
        "persistent_conflict_stress": ("2026-03-20", "2026-04-07"),
        "recent_six_weeks": ("2026-06-15", "2026-07-24"),
    }
    print("\nRetrospective 2026 case studies (not untouched OOS)")
    for name, (start, end) in periods.items():
        period = result.loc[start:end]
        print(
            f"{name} {start}..{end}: "
            f"background={period['Background_State'].value_counts().to_dict()}, "
            f"shock={period['Shock_State'].value_counts().to_dict()}, "
            f"VIX median/max="
            f"{period['VIX_Close'].median():.2f}/"
            f"{period['VIX_Close'].max():.2f}"
        )

    if args.artifact_output is not None:
        _atomic_write(args.artifact_output, artifact.to_json())
        print(f"\nArtifact written atomically: {args.artifact_output.resolve()}")
    if args.plan_output is not None:
        _atomic_write(args.plan_output, plan.to_json())
        print(f"Plan written atomically: {args.plan_output.resolve()}")


if __name__ == "__main__":
    main()
