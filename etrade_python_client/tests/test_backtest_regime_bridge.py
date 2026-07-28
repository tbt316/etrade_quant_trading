import unittest
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd

from backtesting.regime_bridge import (
    BacktestRegimeProtocol,
    ExactAssignmentTargetCache,
    ExactRegimeRunCache,
    LaggedFinalRiskMap,
    RegimeBridgeUnavailable,
    RegimeDecisionEvidence,
    ResolvedRawRegime,
    build_lagged_final_risk_map,
    make_regime_decision_evidence,
)
from live_trading.regime_taxonomy import (
    FinalRiskRegimeRef,
    RawHMMStateRef,
    RegimeReturnBuckets,
)


def _protocol():
    return BacktestRegimeProtocol(
        calibration_end="2025-01-03",
        test_start="2025-01-06",
        test_end="2025-01-31",
        inference_method="walk_forward_expanding",
        raw_hmm_components=2,
    )


def _model(*, taxonomy_id="a" * 64, state=1):
    probabilities = [0.1, 0.9] if state == 1 else [0.9, 0.1]
    return SimpleNamespace(
        n_components=2,
        raw_hmm_taxonomy_id_=taxonomy_id,
        model_training_end_="2025-01-03",
        causal_tail_as_of_="2025-01-10",
        causal_inference_mode_="fixed_snapshot_prefix_filter",
        causal_tail_probability_=probabilities,
    )


def _buckets(*, taxonomy_id="a" * 64):
    return RegimeReturnBuckets(
        taxonomy_id=taxonomy_id,
        model_training_end="2025-01-03",
        inference_as_of="2025-01-10",
        resolved_outcomes_through="2025-01-09",
        horizon_calendar_days=7,
        horizon_trading_days=5,
        buckets=(
            tuple(0.001 * value for value in range(1, 11)),
            tuple(-0.001 * value for value in range(1, 11)),
        ),
    )


class BacktestRegimeProtocolTests(unittest.TestCase):
    def test_protocol_requires_pre_oos_calibration_and_exact_lag(self):
        with self.assertRaisesRegex(ValueError, "calibration_end"):
            BacktestRegimeProtocol(
                calibration_end="2025-01-06",
                test_start="2025-01-06",
                test_end="2025-01-31",
                inference_method="walk_forward_expanding",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            BacktestRegimeProtocol(
                calibration_end="2025-01-03",
                test_start="2025-01-06",
                test_end="2025-01-31",
                inference_method="walk_forward_expanding",
                regime_lag_trading_sessions=2,
            )
        with self.assertRaisesRegex(ValueError, "timestamp"):
            BacktestRegimeProtocol(
                calibration_end="2025-01-03",
                test_start="2025-01-06",
                test_end="2025-01-31",
                inference_method="walk_forward_expanding",
                signal_timestamp="same_day_close",
            )
        with self.assertRaisesRegex(ValueError, "authorize execution"):
            BacktestRegimeProtocol(
                calibration_end="2025-01-03",
                test_start="2025-01-06",
                test_end="2025-01-31",
                inference_method="walk_forward_expanding",
                execution_eligible=True,
            )
        with self.assertRaisesRegex(ValueError, "raw_hmm_components"):
            BacktestRegimeProtocol(
                calibration_end="2025-01-03",
                test_start="2025-01-06",
                test_end="2025-01-31",
                inference_method="walk_forward_expanding",
                raw_hmm_components=1,
            )

    def test_protocol_digest_is_stable_and_manifest_bound(self):
        protocol = _protocol()
        self.assertEqual(protocol.sha256, _protocol().sha256)
        self.assertEqual(len(protocol.sha256), 64)
        self.assertEqual(protocol.manifest()["validity_status"], "UNVERIFIED")
        self.assertEqual(protocol.manifest()["raw_hmm_components"], 2)
        self.assertFalse(protocol.manifest()["execution_eligible"])


class FinalOverlayLagTests(unittest.TestCase):
    def test_friday_close_maps_only_to_monday_entry(self):
        trace = pd.DataFrame(
            {
                "Detected_Regime_State": [2],
                "Detected_Regime_Label": ["Panic / Crisis (2)"],
            },
            index=pd.to_datetime(["2025-01-03"]),
        )
        lagged = build_lagged_final_risk_map(trace, ["2025-01-06"])
        self.assertEqual(
            lagged.signal_session_by_entry["2025-01-06"],
            "2025-01-03",
        )
        self.assertEqual(
            lagged.by_entry_session["2025-01-06"],
            FinalRiskRegimeRef(2, "Panic / Crisis (2)"),
        )

    def test_missing_exact_prior_session_is_not_carried_forward(self):
        trace = pd.DataFrame(
            {
                "Detected_Regime_State": [0],
                "Detected_Regime_Label": ["Expansion (0)"],
            },
            index=pd.to_datetime(["2025-01-02"]),
        )
        lagged = build_lagged_final_risk_map(trace, ["2025-01-06"])
        self.assertNotIn("2025-01-06", lagged.by_entry_session)
        self.assertEqual(
            lagged.unavailable_code_by_entry["2025-01-06"],
            "FINAL_OVERLAY_PRIOR_SESSION_MISSING",
        )
        with self.assertRaises(TypeError):
            lagged.signal_session_by_entry["2025-01-06"] = "2025-01-02"

    def test_raw_hmm_columns_are_never_an_overlay_fallback(self):
        trace = pd.DataFrame(
            {
                "HMM_State": [1],
                "Regime_Label": ["raw state"],
            },
            index=pd.to_datetime(["2025-01-03"]),
        )
        with self.assertRaisesRegex(
            RegimeBridgeUnavailable,
            "FINAL_OVERLAY_COLUMNS_REQUIRED",
        ):
            build_lagged_final_risk_map(trace, ["2025-01-06"])

    def test_final_overlay_state_and_label_must_match_closed_taxonomy(self):
        trace = pd.DataFrame(
            {
                "Detected_Regime_State": [1],
                "Detected_Regime_Label": ["Panic / Crisis (2)"],
            },
            index=pd.to_datetime(["2025-01-03"]),
        )
        lagged = build_lagged_final_risk_map(trace, ["2025-01-06"])
        self.assertNotIn("2025-01-06", lagged.by_entry_session)
        self.assertEqual(
            lagged.unavailable_code_by_entry["2025-01-06"],
            "INVALID_FINAL_OVERLAY_VALUE",
        )

    def test_direct_lagged_map_construction_rejects_malformed_boundaries(self):
        with self.assertRaisesRegex(ValueError, "FinalRiskRegimeRef"):
            LaggedFinalRiskMap(
                by_entry_session={"2025-01-06": 2},
                signal_session_by_entry={"2025-01-06": "2025-01-03"},
                unavailable_code_by_entry={},
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            LaggedFinalRiskMap(
                by_entry_session={
                    "2025-01-06": FinalRiskRegimeRef(
                        0,
                        "Expansion (0)",
                    )
                },
                signal_session_by_entry={"2025-01-06": "2025-01-03"},
                unavailable_code_by_entry={
                    "2025-01-06": "FINAL_OVERLAY_UNAVAILABLE"
                },
            )
        with self.assertRaisesRegex(ValueError, "exactly every"):
            LaggedFinalRiskMap(
                by_entry_session={},
                signal_session_by_entry={"2025-01-06": "2025-01-03"},
                unavailable_code_by_entry={},
            )
        with self.assertRaisesRegex(ValueError, "stable codes"):
            LaggedFinalRiskMap(
                by_entry_session={},
                signal_session_by_entry={"2025-01-06": "2025-01-03"},
                unavailable_code_by_entry={"2025-01-06": "not stable"},
            )


class ExactRawRegimeCacheTests(unittest.TestCase):
    def test_success_is_cached_and_uses_raw_state_for_bucket_lookup(self):
        calls = []

        def builder(*args, **kwargs):
            calls.append((args, kwargs))
            return _buckets(), _model(state=1), []

        cache = ExactRegimeRunCache(_protocol())
        first = cache.resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=builder,
        )
        second = cache.resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=builder,
        )
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0][1]["as_of_date"],
            "2025-01-10",
        )
        self.assertEqual(first.value.state_ref, RawHMMStateRef("a" * 64, 1))
        self.assertEqual(first.value.return_bucket, _buckets().buckets[1])

        evidence = make_regime_decision_evidence(
            protocol=_protocol(),
            entry_session="2025-01-13",
            signal_as_of_session="2025-01-10",
            final_risk_regime=FinalRiskRegimeRef(2, "Panic / Crisis (2)"),
            raw_resolution=first,
            raw_context_required=True,
            assignment_probability=0.05,
        )
        self.assertEqual(evidence.final_risk_regime.state, 2)
        self.assertEqual(evidence.raw_hmm_state.state, 1)

    def test_failure_is_cached_with_stable_code(self):
        calls = []

        def builder(*_args, **_kwargs):
            calls.append(True)
            raise RegimeBridgeUnavailable("HISTORICAL_PREFIX_UNAVAILABLE")

        cache = ExactRegimeRunCache(_protocol())
        first = cache.resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=builder,
        )
        second = cache.resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=builder,
        )
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            first.unavailable_code,
            "HISTORICAL_PREFIX_UNAVAILABLE",
        )

    def test_as_of_and_horizon_are_part_of_the_exact_cache_key(self):
        calls = []

        def builder(*_args, **kwargs):
            calls.append((kwargs["as_of_date"], kwargs["horizon"]))
            raise RegimeBridgeUnavailable("HISTORICAL_PREFIX_UNAVAILABLE")

        cache = ExactRegimeRunCache(_protocol())
        for as_of, horizon in (
            ("2025-01-09", 7),
            ("2025-01-10", 7),
            ("2025-01-10", 14),
        ):
            cache.resolve(
                signal_as_of=as_of,
                horizon_calendar_days=horizon,
                n_components=2,
                builder=builder,
            )
        self.assertEqual(
            calls,
            [
                ("2025-01-09", 7),
                ("2025-01-10", 7),
                ("2025-01-10", 14),
            ],
        )

    def test_as_of_and_taxonomy_mismatches_fail_closed(self):
        cache = ExactRegimeRunCache(_protocol())

        wrong_as_of = cache.resolve(
            signal_as_of="2025-01-09",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        )
        self.assertEqual(
            wrong_as_of.unavailable_code,
            "REGIME_BUCKET_AS_OF_MISMATCH",
        )

        wrong_taxonomy = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(taxonomy_id="b" * 64),
                _model(taxonomy_id="a" * 64),
                [],
            ),
        )
        self.assertEqual(
            wrong_taxonomy.unavailable_code,
            "REGIME_TAXONOMY_MISMATCH",
        )
        component_mismatch = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=3,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        )
        self.assertEqual(
            component_mismatch.unavailable_code,
            "HMM_COMPONENT_PROTOCOL_MISMATCH",
        )

    def test_assignment_fit_failures_are_cached_by_exact_target(self):
        raw = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        ).value
        calls = []

        def failed_fit(*_args, **_kwargs):
            calls.append(True)
            raise RegimeBridgeUnavailable("GMM_FIT_UNAVAILABLE")

        cache = ExactAssignmentTargetCache()
        for target in (0.05, 0.05, 0.10):
            resolution = cache.resolve(
                raw=raw,
                target_assignment_probability=target,
                fit_model=failed_fit,
                query_probability=lambda *_args: 0.0,
                solve_root=lambda *_args: 0.0,
            )
            self.assertEqual(
                resolution.unavailable_code,
                "GMM_FIT_UNAVAILABLE",
            )
        self.assertEqual(len(calls), 2)


class RegimeDecisionEvidenceTests(unittest.TestCase):
    def test_evidence_is_serializable_and_keeps_full_provenance(self):
        resolution = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        )
        evidence = make_regime_decision_evidence(
            protocol=_protocol(),
            entry_session="2025-01-13",
            signal_as_of_session="2025-01-10",
            final_risk_regime=FinalRiskRegimeRef(0, "Expansion (0)"),
            raw_resolution=resolution,
            raw_context_required=True,
            assignment_probability=0.05,
        )
        serialized = asdict(evidence)
        self.assertEqual(
            serialized["raw_hmm_state"]["taxonomy_id"],
            "a" * 64,
        )
        self.assertEqual(serialized["return_bucket_as_of"], "2025-01-10")
        self.assertEqual(
            serialized["resolved_outcomes_through"],
            "2025-01-09",
        )
        self.assertEqual(serialized["validity_status"], "UNVERIFIED")
        self.assertFalse(serialized["execution_eligible"])

    def test_unavailable_raw_context_blocks_without_fake_probability(self):
        evidence = make_regime_decision_evidence(
            protocol=_protocol(),
            entry_session="2025-01-13",
            signal_as_of_session="2025-01-10",
            final_risk_regime=FinalRiskRegimeRef(0, "Expansion (0)"),
            raw_context_required=True,
        )
        self.assertEqual(evidence.validity_status, "UNAVAILABLE")
        self.assertEqual(
            evidence.unavailable_code,
            "RAW_REGIME_CONTEXT_UNAVAILABLE",
        )
        self.assertIsNone(evidence.assignment_probability)

    def test_raw_context_without_computed_probability_stays_unavailable(self):
        resolution = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        )
        evidence = make_regime_decision_evidence(
            protocol=_protocol(),
            entry_session="2025-01-13",
            signal_as_of_session="2025-01-10",
            final_risk_regime=FinalRiskRegimeRef(0, "Expansion (0)"),
            raw_resolution=resolution,
            raw_context_required=True,
        )
        self.assertEqual(evidence.validity_status, "UNAVAILABLE")
        self.assertEqual(
            evidence.unavailable_code,
            "ASSIGNMENT_PROBABILITY_UNAVAILABLE",
        )

    def test_same_session_signal_or_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            RegimeDecisionEvidence(
                entry_session="2025-01-10",
                signal_as_of_session="2025-01-10",
                protocol_sha256=_protocol().sha256,
                final_risk_regime=FinalRiskRegimeRef(
                    0,
                    "Expansion (0)",
                ),
            )
        with self.assertRaisesRegex(ValueError, "resolve before"):
            RegimeDecisionEvidence(
                entry_session="2025-01-13",
                signal_as_of_session="2025-01-10",
                protocol_sha256=_protocol().sha256,
                final_risk_regime=FinalRiskRegimeRef(
                    0,
                    "Expansion (0)",
                ),
                raw_hmm_state=RawHMMStateRef("a" * 64, 0),
                model_training_end="2025-01-03",
                raw_inference_method="fixed_snapshot_prefix_filter",
                return_bucket_as_of="2025-01-10",
                resolved_outcomes_through="2025-01-10",
                horizon_calendar_days=7,
                horizon_trading_days=5,
                return_bucket_manifest_sha256="b" * 64,
                raw_context_required=True,
            )

    def test_raw_manifest_is_defensively_copied_and_digest_bound(self):
        resolution = ExactRegimeRunCache(_protocol()).resolve(
            signal_as_of="2025-01-10",
            horizon_calendar_days=7,
            n_components=2,
            builder=lambda *_args, **_kwargs: (
                _buckets(),
                _model(),
                [],
            ),
        )
        raw = resolution.value
        original_count = raw.bucket_manifest["bucket_counts"][0]
        with self.assertRaises(TypeError):
            raw.bucket_manifest["state_count"] = 99
        with self.assertRaises(TypeError):
            raw.bucket_manifest["bucket_counts"][0] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            ResolvedRawRegime(
                state_ref=raw.state_ref,
                return_bucket=raw.return_bucket,
                model_training_end=raw.model_training_end,
                inference_method=raw.inference_method,
                bucket_as_of=raw.bucket_as_of,
                resolved_outcomes_through=raw.resolved_outcomes_through,
                horizon_calendar_days=raw.horizon_calendar_days,
                horizon_trading_days=raw.horizon_trading_days,
                bucket_manifest_sha256="b" * 64,
                bucket_manifest=dict(raw.bucket_manifest),
            )
        self.assertEqual(
            raw.bucket_manifest["bucket_counts"][0],
            original_count,
        )


if __name__ == "__main__":
    unittest.main()
