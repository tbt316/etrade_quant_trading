import dataclasses
import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np

import live_trading.regime_calibration as calibration_module
from live_trading.regime_calibration import (
    CalibrationCandidate,
    CausalValidationFold,
    RegimeCalibrationPlan,
    detect_regimes_with_calibration_artifact,
    regime_calibration_code_sha256,
    run_research_calibration,
)
from live_trading.regime_detector_v2 import (
    DETECTOR_VERSION,
    RegimeDetectorConfig,
    SIGNAL_TIMESTAMP,
    regime_detector_code_sha256,
)
from live_trading.regime_market_data import regime_market_schedule
from live_trading.regime_signal import (
    BackgroundState,
    R4_SHADOW_ABSTAIN_REASON,
    RegimeActionProjectionError,
    RegimeSignal,
    RegimeSignalError,
    ShockState,
    annotation_for_session,
    from_calibrated_v2_trace,
    next_tradable_session,
    to_action_projection,
)


CONFIG_HASH = "a" * 64
NYSE = mcal.get_calendar("NYSE")


def _trace(*, tradable_session="2025-04-04", execution_eligible=False):
    frame = pd.DataFrame(
        {
            "Background_State": ["elevated", "persistent_stress"],
            "Shock_State": ["active", "none"],
            "Signal_Available_At": [
                "2025-04-03T20:15:00Z",
                "2025-04-04T20:15:00Z",
            ],
            "Tradable_Session": [tradable_session, "2025-04-07"],
            "Detector_Version": [DETECTOR_VERSION, DETECTOR_VERSION],
            "Config_Hash": [CONFIG_HASH, CONFIG_HASH],
            "Regime_Signal_Timestamp": [SIGNAL_TIMESTAMP, SIGNAL_TIMESTAMP],
            "Reason_Codes": ["vix_daily_change_extreme", "persistent_score_confirmed"],
            "Data_Quality": [
                "exchange_sessions_and_source_times_valid",
                "exchange_sessions_and_source_times_valid",
            ],
            "Execution_Eligible": [execution_eligible, False],
        },
        index=pd.DatetimeIndex(["2025-04-03", "2025-04-04"]),
    )
    frame.attrs["detector_version"] = DETECTOR_VERSION
    frame.attrs["config_hash"] = CONFIG_HASH
    return frame


class _FakeArtifact:
    artifact_sha256 = "1" * 64
    execution_eligible = False
    selected_candidate_id = "baseline"
    promotion_status = "research_only"
    selected_candidate = SimpleNamespace(config_sha256=CONFIG_HASH)
    data_manifest = SimpleNamespace(
        provenance_status="legacy_normalized_unverified",
    )
    plan = SimpleNamespace(
        detector_version=DETECTOR_VERSION,
        detector_code_sha256=regime_detector_code_sha256(),
        calibration_code_sha256=regime_calibration_code_sha256(),
        plan_sha256="2" * 64,
        selection_folds=(
            SimpleNamespace(evaluation_end=date(2024, 12, 31)),
        ),
        retrospective_test=SimpleNamespace(
            evaluation_start=date(2025, 1, 2),
            evaluation_end=date(2025, 12, 31),
        ),
        inference_method="causal_prefix_filter",
        signal_lag_sessions=1,
    )


def _calibrated_trace():
    frame = _trace()
    frame["Calibration_Artifact_SHA256"] = _FakeArtifact.artifact_sha256
    frame["Calibration_Plan_SHA256"] = _FakeArtifact.plan.plan_sha256
    frame["Calibration_Profile"] = _FakeArtifact.selected_candidate_id
    frame["Calibration_Status"] = _FakeArtifact.promotion_status
    frame["Calibration_Provenance"] = (
        _FakeArtifact.data_manifest.provenance_status
    )
    frame["Calibration_Abstain"] = True
    frame["Calibration_Abstain_Reasons"] = "research_only"
    frame["Calibration_Execution_Eligible"] = False
    frame.attrs.update(
        {
            "detector_code_sha256": (
                _FakeArtifact.plan.detector_code_sha256
            ),
            "runtime_fingerprint_sha256": "3" * 64,
            "calibration_artifact_sha256": (
                _FakeArtifact.artifact_sha256
            ),
            "calibration_plan_sha256": _FakeArtifact.plan.plan_sha256,
            "calibration_profile": _FakeArtifact.selected_candidate_id,
            "calibration_status": _FakeArtifact.promotion_status,
            "calibration_provenance": (
                _FakeArtifact.data_manifest.provenance_status
            ),
            "calibration_execution_eligible": False,
            "calibration_abstain": True,
            "execution_eligible": False,
        }
    )
    return frame


class RegimeSignalTests(unittest.TestCase):
    def test_v2_adapter_preserves_two_axes_and_exact_lookup(self):
        annotations = from_calibrated_v2_trace(_trace())

        first = annotation_for_session(annotations, "2025-04-04")
        second = annotation_for_session(annotations, "2025-04-07")
        self.assertIsNotNone(first)
        self.assertEqual(first.background_state, BackgroundState.ELEVATED)
        self.assertEqual(first.shock_state, ShockState.ACTIVE)
        self.assertEqual(second.background_state, BackgroundState.PERSISTENT_STRESS)
        self.assertEqual(second.shock_state, ShockState.NONE)
        self.assertIsNone(annotation_for_session(annotations, "2025-04-05"))
        self.assertIsNone(annotation_for_session(annotations, "2025-04-08"))
        self.assertIn("calibration_lineage_missing", first.abstain_reasons)
        self.assertIn(R4_SHADOW_ABSTAIN_REASON, first.abstain_reasons)
        self.assertFalse(first.may_authorize_execution)

    def test_canonical_round_trip_hash_and_dashboard_payload(self):
        signal = annotation_for_session(from_calibrated_v2_trace(_trace()), "2025-04-04")
        encoded = signal.to_json()
        self.assertEqual(RegimeSignal.from_json(encoded), signal)
        payload = signal.to_dashboard_payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["background_state"], "elevated")
        self.assertEqual(payload["shock_state"], "active")
        self.assertFalse(payload["may_authorize_execution"])
        self.assertNotIn("state", {key for key in payload if key.startswith("hmm_")})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            signal.background_state = BackgroundState.CALM
        with self.assertRaises(RegimeActionProjectionError):
            to_action_projection(signal)

    def test_timing_and_execution_mismatches_fail_closed(self):
        with self.assertRaisesRegex(RegimeSignalError, "exact next NYSE"):
            from_calibrated_v2_trace(_trace(tradable_session="2025-04-07"))
        with self.assertRaisesRegex(RegimeSignalError, "execution-ineligible"):
            from_calibrated_v2_trace(_trace(execution_eligible=True))
        self.assertEqual(next_tradable_session("2025-04-03").isoformat(), "2025-04-04")
        self.assertEqual(next_tradable_session("2025-04-04").isoformat(), "2025-04-07")

    def test_hash_or_schema_tampering_fails_closed(self):
        signal = annotation_for_session(from_calibrated_v2_trace(_trace()), "2025-04-04")
        envelope = signal.to_json().replace(CONFIG_HASH, "b" * 64)
        with self.assertRaisesRegex(RegimeSignalError, "hash does not match"):
            RegimeSignal.from_json(envelope)
        persisted = json.loads(signal.to_json())
        persisted["signal"]["background_state"] = "calm"
        with self.assertRaisesRegex(RegimeSignalError, "hash does not match"):
            RegimeSignal.from_envelope(persisted)
        malformed = _trace()
        malformed.loc[malformed.index[0], "Detector_Version"] = "unexpected"
        with self.assertRaisesRegex(RegimeSignalError, "detector version"):
            from_calibrated_v2_trace(malformed)

    def test_calibrated_adapter_requires_pin_and_immutable_attributes(self):
        artifact = _FakeArtifact()
        trace = _calibrated_trace()

        with patch.object(
            calibration_module,
            "RegimeCalibrationArtifact",
            _FakeArtifact,
        ):
            with self.assertRaisesRegex(
                RegimeSignalError,
                "deployment-pinned",
            ):
                from_calibrated_v2_trace(trace, artifact=artifact)
            with self.assertRaisesRegex(
                RegimeSignalError,
                "deployment pin",
            ):
                from_calibrated_v2_trace(
                    trace,
                    artifact=artifact,
                    expected_artifact_sha256="f" * 64,
                )

            stripped = trace.copy()
            stripped.attrs = {}
            with self.assertRaisesRegex(
                RegimeSignalError,
                "immutable attributes",
            ):
                from_calibrated_v2_trace(
                    stripped,
                    artifact=artifact,
                    expected_artifact_sha256=artifact.artifact_sha256,
                )

            signal = annotation_for_session(
                from_calibrated_v2_trace(
                    trace,
                    artifact=artifact,
                    expected_artifact_sha256=artifact.artifact_sha256,
                ),
                "2025-04-04",
            )
            causal = signal.lineage.causal_record
            self.assertEqual(
                causal.calibration_end_session,
                date(2024, 12, 31),
            )
            self.assertEqual(causal.signal_lag_sessions, 1)
            self.assertEqual(
                causal.return_bucket_status.value,
                "not_used_shadow_annotation",
            )

    def test_real_calibration_detector_adapter_integration(self):
        schedule = NYSE.schedule(
            start_date="2022-01-03",
            end_date="2025-12-31",
        )
        index = (
            pd.DatetimeIndex(schedule.index)
            .tz_localize(None)
            .normalize()[:300]
        )
        position = np.arange(len(index))
        stressed = position % 80 >= 45
        returns = np.where(
            stressed,
            np.where(position % 2 == 0, 0.012, -0.015),
            np.where(position % 2 == 0, 0.0012, -0.0005),
        )
        prices = pd.DataFrame(
            {
                "SPY_Close": 400.0 * np.exp(np.cumsum(returns)),
                "VIX_Close": np.where(
                    stressed,
                    27.0 + np.sin(position / 3.0),
                    16.0 + 0.5 * np.sin(position / 5.0),
                ),
            },
            index=index,
        )
        config = RegimeDetectorConfig(
            calibration_window=60,
            min_calibration_history=20,
            vix_slow_window=5,
            realized_vol_window=5,
            drawdown_window=10,
            stress_entry_days=2,
            stress_exit_days=3,
            calm_entry_days=2,
            calm_exit_days=2,
        )

        def fold(fold_id, ref_end, start, end):
            return CausalValidationFold(
                fold_id=fold_id,
                outcome_reference_start=index[0].date(),
                outcome_reference_end=index[ref_end].date(),
                evaluation_start=index[start].date(),
                evaluation_end=index[end].date(),
            )

        plan = RegimeCalibrationPlan(
            protocol_id="signal_integration",
            candidates=(
                CalibrationCandidate("baseline", config),
            ),
            control_candidate_id="baseline",
            selection_folds=(
                fold("selection", 80, 100, 149),
            ),
            retrospective_test=fold(
                "retrospective",
                170,
                190,
                239,
            ),
            prospective_holdout_start=index[270].date(),
            outcome_horizons=(2, 5),
            min_evaluation_rows=20,
            max_state_occupancy=0.99,
            max_switches_per_252=100.0,
            minimum_isolated_shock_episodes=0,
            max_false_persistent_isolated_shock_rate=1.0,
        )
        artifact = run_research_calibration(prices, plan)
        market_schedule = regime_market_schedule(
            prices.index.min().date(),
            prices.index.max().date(),
        )
        sessions = (
            pd.DatetimeIndex(market_schedule.index)
            .tz_localize(None)
            .normalize()
        )
        trace = detect_regimes_with_calibration_artifact(
            prices,
            artifact,
            expected_artifact_sha256=artifact.artifact_sha256,
            as_of=(
                pd.Timestamp(
                    market_schedule.iloc[-1]["joint_finalization_at"]
                )
                + pd.Timedelta(minutes=1)
            ),
            spy_available_at=pd.Series(
                pd.DatetimeIndex(market_schedule["spy_event_at"]),
                index=sessions,
            ),
            vix_available_at=pd.Series(
                pd.DatetimeIndex(market_schedule["vix_event_at"]),
                index=sessions,
            ),
        )

        annotations = from_calibrated_v2_trace(
            trace.tail(1),
            artifact=artifact,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
        signal = next(iter(annotations.values()))
        self.assertEqual(
            signal.lineage.artifact_sha256,
            artifact.artifact_sha256,
        )
        self.assertEqual(
            signal.lineage.causal_record.calibration_end_session,
            plan.selection_folds[-1].evaluation_end,
        )
        self.assertFalse(signal.may_authorize_execution)
