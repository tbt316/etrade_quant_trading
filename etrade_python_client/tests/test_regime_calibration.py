import importlib.util
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from live_trading.regime_calibration import (
    CalibrationCandidate,
    CausalValidationFold,
    PROVENANCE_LEGACY_UNVERIFIED,
    PROVENANCE_PROVIDER_DECISION_TIME,
    RegimeCalibrationArtifact,
    RegimeCalibrationError,
    RegimeCalibrationPlan,
    build_calibration_data_manifest,
    build_forward_risk_outcomes,
    detect_regimes_with_calibration_artifact,
    evaluate_candidate_fold,
    run_research_calibration,
)
from live_trading.regime_detector_v2 import (
    RegimeDetectorConfig,
    detect_regimes,
    regime_detector_code_sha256,
    regime_detector_config_sha256,
)
from live_trading.regime_market_data import regime_market_schedule


NYSE = mcal.get_calendar("NYSE")


def _prices(rows=520):
    schedule = NYSE.schedule(start_date="2021-01-04", end_date="2025-12-31")
    index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()[:rows]
    position = np.arange(rows)
    stressed = position % 100 >= 60
    calm_returns = np.where(position % 2 == 0, 0.0015, -0.0007)
    stress_returns = np.where(position % 2 == 0, 0.013, -0.016)
    returns = np.where(stressed, stress_returns, calm_returns)
    spy = 400.0 * np.exp(np.cumsum(returns))
    vix = np.where(
        stressed,
        27.0 + 2.0 * np.sin(position / 3.0),
        16.0 + 0.8 * np.sin(position / 5.0),
    )
    return pd.DataFrame(
        {"SPY_Close": spy, "VIX_Close": vix},
        index=index,
    )


def _config(**overrides):
    values = {
        "calibration_window": 60,
        "min_calibration_history": 20,
        "vix_slow_window": 5,
        "realized_vol_window": 5,
        "drawdown_window": 10,
        "stress_entry_days": 2,
        "stress_exit_days": 3,
        "calm_entry_days": 2,
        "calm_exit_days": 2,
    }
    values.update(overrides)
    return RegimeDetectorConfig(**values)


def _fold(index, fold_id, reference_start, reference_end, start, end):
    return CausalValidationFold(
        fold_id=fold_id,
        outcome_reference_start=index[reference_start].date(),
        outcome_reference_end=index[reference_end].date(),
        evaluation_start=index[start].date(),
        evaluation_end=index[end].date(),
    )


def _plan(prices, *, candidates=None, signal_lag_sessions=1):
    index = prices.index
    baseline = CalibrationCandidate("baseline", _config())
    if candidates is None:
        candidates = (
            CalibrationCandidate("aaa_challenger", _config()),
            baseline,
        )
    return RegimeCalibrationPlan(
        protocol_id="synthetic_preregistered",
        candidates=tuple(candidates),
        control_candidate_id="baseline",
        selection_folds=(
            _fold(index, "fold_1", 50, 149, 160, 239),
            _fold(index, "fold_2", 50, 249, 260, 339),
        ),
        retrospective_test=_fold(
            index,
            "retrospective_1",
            50,
            349,
            360,
            439,
        ),
        prospective_holdout_start=index[460].date(),
        outcome_horizons=(2, 5),
        min_evaluation_rows=50,
        max_state_occupancy=0.99,
        minimum_isolated_shock_episodes=0,
        minimum_material_improvement=0.01,
        signal_lag_sessions=signal_lag_sessions,
    )


def _context(prices):
    schedule = regime_market_schedule(
        prices.index.min().date(),
        prices.index.max().date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return {
        "as_of": (
            pd.Timestamp(schedule.iloc[-1]["joint_finalization_at"])
            + pd.Timedelta(minutes=1)
        ),
        "spy_available_at": pd.Series(
            pd.DatetimeIndex(schedule["spy_event_at"]),
            index=sessions,
        ),
        "vix_available_at": pd.Series(
            pd.DatetimeIndex(schedule["vix_event_at"]),
            index=sessions,
        ),
        "source_provenance_verified": False,
    }


class RegimeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = _prices()
        cls.plan = _plan(cls.prices)
        cls.artifact = run_research_calibration(cls.prices, cls.plan)

    def test_detector_config_hash_and_code_hash_are_stable(self):
        config = _config()

        self.assertEqual(
            regime_detector_config_sha256(config),
            regime_detector_config_sha256(replace(config)),
        )
        self.assertEqual(len(regime_detector_code_sha256()), 64)
        with self.assertRaisesRegex(ValueError, "finite"):
            _config(score_halflife=float("nan"))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            _config(calibration_window=True)

    def test_forward_outcomes_begin_at_next_session_and_end_unresolved(self):
        prices = self.prices.iloc[:30]
        outcomes = build_forward_risk_outcomes(prices, (2,))
        expected_returns = np.diff(
            np.log(prices["SPY_Close"].iloc[1:4].to_numpy(dtype=float))
        )
        expected_rms = np.sqrt(np.mean(expected_returns**2) * 252.0)
        path = prices["SPY_Close"].iloc[1:4].to_numpy(dtype=float)
        expected_drawdown = np.max(
            1.0 - path / np.maximum.accumulate(path),
        )

        self.assertAlmostEqual(
            outcomes.iloc[0]["Forward_RMS_Vol_2"],
            expected_rms,
        )
        self.assertAlmostEqual(
            outcomes.iloc[0]["Forward_Max_Peak_Drawdown_2"],
            expected_drawdown,
        )
        self.assertEqual(
            outcomes.iloc[0]["Outcome_Resolved_At_2"],
            prices.index[3],
        )
        self.assertTrue(outcomes.iloc[-3:]["Forward_RMS_Vol_2"].isna().all())

    def test_plan_rejects_overlap_zero_lag_and_shock_tuning(self):
        with self.assertRaisesRegex(
            RegimeCalibrationError,
            "exactly one lagged session",
        ):
            _plan(self.prices, signal_lag_sessions=0)

        overlap = replace(
            self.plan.selection_folds[1],
            outcome_reference_end=self.prices.index[230].date(),
            evaluation_start=self.plan.selection_folds[0].evaluation_end,
        )
        with self.assertRaisesRegex(RegimeCalibrationError, "must not overlap"):
            replace(
                self.plan,
                selection_folds=(self.plan.selection_folds[0], overlap),
            )

        tuned_shock = CalibrationCandidate(
            "tuned_shock",
            _config(vix_shock_return=0.12),
        )
        with self.assertRaisesRegex(RegimeCalibrationError, "shock lane"):
            _plan(
                self.prices,
                candidates=(
                    CalibrationCandidate("baseline", _config()),
                    tuned_shock,
                ),
            )

    def test_fold_rejects_reference_labels_resolved_after_evaluation_starts(self):
        outcomes = build_forward_risk_outcomes(self.prices, (2, 5))
        trace = detect_regimes(
            self.prices,
            _config(),
            **_context(self.prices),
        )
        invalid = _fold(
            self.prices.index,
            "unpurged",
            50,
            159,
            160,
            239,
        )

        with self.assertRaisesRegex(RegimeCalibrationError, "not purged"):
            evaluate_candidate_fold(
                trace,
                outcomes,
                "baseline",
                invalid,
                self.plan,
            )

    def test_fold_guardrails_reject_chatter_missing_coverage_and_persistence(self):
        outcomes = build_forward_risk_outcomes(self.prices, (2, 5))
        trace = detect_regimes(
            self.prices,
            _config(),
            **_context(self.prices),
        )
        fold = self.plan.selection_folds[0]
        evaluation_index = trace.loc[
            fold.evaluation_start.isoformat() :
            fold.evaluation_end.isoformat()
        ].index

        chatter = trace.copy()
        chatter.loc[evaluation_index, "Background_State"] = np.where(
            np.arange(len(evaluation_index)) % 2 == 0,
            "calm",
            "persistent_stress",
        )
        chatter_metric = evaluate_candidate_fold(
            chatter,
            outcomes,
            "baseline",
            fold,
            self.plan,
        )
        self.assertGreater(
            chatter_metric.switches_per_252,
            self.plan.max_switches_per_252,
        )
        self.assertFalse(chatter_metric.guardrail_passed)

        incomplete = trace.copy()
        incomplete.loc[
            evaluation_index[:5],
            "Background_State",
        ] = "unavailable"
        coverage_metric = evaluate_candidate_fold(
            incomplete,
            outcomes,
            "baseline",
            fold,
            self.plan,
        )
        self.assertLess(
            coverage_metric.evaluation_coverage,
            self.plan.minimum_evaluation_coverage,
        )
        self.assertFalse(coverage_metric.guardrail_passed)

        false_persistent = trace.copy()
        false_persistent.loc[evaluation_index, "Background_State"] = "calm"
        false_persistent.loc[evaluation_index, "Shock_State"] = "none"
        event_position = 20
        event = evaluation_index[event_position]
        followup = evaluation_index[
            event_position + 1 :
            event_position
            + self.plan.isolated_shock_followup_sessions
            + 1
        ]
        false_persistent.loc[event, "Shock_State"] = "active"
        false_persistent.loc[followup, "Background_State"] = (
            "persistent_stress"
        )
        low_risk_outcomes = outcomes.copy()
        low_risk_outcomes.loc[
            evaluation_index,
            [
                "Forward_RMS_Vol_2",
                "Forward_RMS_Vol_5",
                "Forward_Max_Peak_Drawdown_5",
            ],
        ] = 0.0
        persistence_metric = evaluate_candidate_fold(
            false_persistent,
            low_risk_outcomes,
            "baseline",
            fold,
            replace(self.plan, minimum_isolated_shock_episodes=1),
        )
        self.assertEqual(persistence_metric.isolated_shocks, 1)
        self.assertEqual(
            persistence_metric.false_persistent_on_isolated_shock,
            1,
        )
        self.assertFalse(persistence_metric.guardrail_passed)

    def test_control_is_retained_when_challenger_gain_is_not_material(self):
        artifact = self.artifact

        self.assertEqual(artifact.raw_best_candidate_id, "aaa_challenger")
        self.assertEqual(artifact.selected_candidate_id, "baseline")
        self.assertIn(
            "challenger_improvement_not_material",
            artifact.selection_reason_codes,
        )
        self.assertFalse(artifact.execution_eligible)
        self.assertEqual(
            artifact.data_manifest.provenance_status,
            PROVENANCE_LEGACY_UNVERIFIED,
        )
        self.assertEqual(
            {item.candidate_id for item in artifact.retrospective_test_metrics},
            {"aaa_challenger", "baseline"},
        )

    def test_artifact_is_canonical_round_trippable_and_tamper_evident(self):
        payload = self.artifact.to_json()
        loaded = RegimeCalibrationArtifact.from_json(payload)

        self.assertEqual(loaded, self.artifact)
        self.assertEqual(loaded.artifact_sha256, self.artifact.artifact_sha256)

        envelope = json.loads(payload)
        envelope["artifact"]["selected_candidate_id"] = "aaa_challenger"
        tampered = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.assertRaises(RegimeCalibrationError):
            RegimeCalibrationArtifact.from_json(tampered)

        with self.assertRaisesRegex(RegimeCalibrationError, "canonical"):
            RegimeCalibrationArtifact.from_json(
                json.dumps(json.loads(payload), indent=2)
            )

    def test_checked_in_plan_is_the_cli_protocol_and_matches_artifact(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = (
            project_root / "scratch" / "regime_detector_v2_calibrate.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_regime_detector_v2_calibrate_test",
            script_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        frozen_plan = RegimeCalibrationPlan.from_json(
            (
                project_root / "docs" / "regime_v2_calibration_plan.json"
            ).read_text(encoding="utf-8")
        )
        frozen_artifact = RegimeCalibrationArtifact.from_json(
            (
                project_root
                / "research_reports"
                / "regime_v2_calibration_artifact.json"
            ).read_text(encoding="utf-8")
        )
        schedule = NYSE.schedule(
            start_date="2011-05-03",
            end_date="2026-07-24",
        )
        protocol_frame = pd.DataFrame(
            index=pd.DatetimeIndex(schedule.index)
            .tz_localize(None)
            .normalize()
        )

        self.assertEqual(
            module.load_frozen_plan(protocol_frame),
            frozen_plan,
        )
        self.assertEqual(frozen_artifact.plan, frozen_plan)
        self.assertFalse(frozen_artifact.execution_eligible)

    def test_stale_plan_provider_claim_and_unpinned_artifact_fail_closed(self):
        stale_plan = replace(
            self.plan,
            detector_code_sha256="0" * 64,
        )
        with self.assertRaisesRegex(RegimeCalibrationError, "code hash"):
            run_research_calibration(self.prices, stale_plan)

        stale_engine_plan = replace(
            self.plan,
            calibration_code_sha256="0" * 64,
        )
        with self.assertRaisesRegex(RegimeCalibrationError, "engine code"):
            run_research_calibration(self.prices, stale_engine_plan)

        with self.assertRaisesRegex(
            RegimeCalibrationError,
            "verified provider calibration is not implemented",
        ):
            build_calibration_data_manifest(
                self.prices.iloc[:50],
                provenance_status=PROVENANCE_PROVIDER_DECISION_TIME,
                snapshot_sha256="1" * 64,
                evidence_manifest_sha256="2" * 64,
                evidence_verification_kind="decision_time",
            )

        with self.assertRaisesRegex(RegimeCalibrationError, "pinned hash"):
            detect_regimes_with_calibration_artifact(
                self.prices,
                self.artifact,
                expected_artifact_sha256="f" * 64,
                **{
                    key: value
                    for key, value in _context(self.prices).items()
                    if key != "source_provenance_verified"
                },
            )

    def test_appending_future_sessions_cannot_change_artifact(self):
        cutoff_position = self.prices.index.get_loc(
            pd.Timestamp(self.artifact.evaluation_as_of_session)
        )
        exact_prefix = self.prices.iloc[: cutoff_position + 1]

        prefix_artifact = run_research_calibration(
            exact_prefix,
            self.plan,
        )

        self.assertEqual(prefix_artifact, self.artifact)
        self.assertEqual(
            prefix_artifact.artifact_sha256,
            self.artifact.artifact_sha256,
        )

    def test_runtime_requires_exact_calibrated_prefix_and_abstains(self):
        result = detect_regimes_with_calibration_artifact(
            self.prices,
            self.artifact,
            expected_artifact_sha256=self.artifact.artifact_sha256,
            **{
                key: value
                for key, value in _context(self.prices).items()
                if key != "source_provenance_verified"
            },
        )

        self.assertTrue(result["Calibration_Abstain"].all())
        self.assertFalse(result["Execution_Eligible"].any())
        self.assertFalse(result.attrs["execution_eligible"])
        self.assertIn(
            "prospective_holdout_incomplete",
            result.attrs["calibration_abstain_reasons"],
        )

        changed = self.prices.copy()
        changed.iloc[10, changed.columns.get_loc("SPY_Close")] *= 1.0001
        with self.assertRaisesRegex(
            RegimeCalibrationError,
            "does not match",
        ):
            detect_regimes_with_calibration_artifact(
                changed,
                self.artifact,
                expected_artifact_sha256=self.artifact.artifact_sha256,
                **{
                    key: value
                    for key, value in _context(changed).items()
                    if key != "source_provenance_verified"
                },
            )

    def test_sustained_stress_and_isolated_news_shock_remain_distinct(self):
        prices = self.prices.iloc[:260].copy()
        isolated_shock_position = 230
        isolated_shock_date = prices.index[isolated_shock_position]
        prices.iloc[
            isolated_shock_position,
            prices.columns.get_loc("VIX_Close"),
        ] = (
            prices.iloc[isolated_shock_position - 1]["VIX_Close"] * 1.12
        )
        result = detect_regimes(
            prices,
            _config(),
            **_context(prices),
        )

        self.assertEqual(
            result.iloc[190]["Background_State"],
            "persistent_stress",
        )
        self.assertEqual(
            result.loc[isolated_shock_date, "Shock_State"],
            "active",
        )
        self.assertNotEqual(
            result.loc[isolated_shock_date, "Background_State"],
            "persistent_stress",
        )
        following = result.iloc[
            isolated_shock_position : isolated_shock_position + 6
        ]
        self.assertFalse(
            following["Background_State"].eq("persistent_stress").any()
        )


if __name__ == "__main__":
    unittest.main()
