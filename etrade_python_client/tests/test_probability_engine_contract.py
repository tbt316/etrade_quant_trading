import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import live_trading.ev_engine as engine_module
from live_trading.ev_engine import (
    CausalModelPreparationManifest,
    MIN_REGIME_BUCKET_OBSERVATIONS,
    ProbabilityEngineResult,
    ProbabilityEngineUnavailable,
    get_probability_engine,
)
from live_trading.regime_taxonomy import RegimeReturnBuckets


class _FakeHMM:
    def __init__(self):
        self.n_components = 2
        self.transmat_ = np.array([[0.90, 0.10], [0.20, 0.80]])
        self.startprob_ = np.array([0.60, 0.40])
        self.means_ = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.covars_ = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ]
        )
        self.feature_names_ = ["SPY_Log_Return", "VIX_Change"]
        self.feature_manifest_ = SimpleNamespace(
            feature_hash="a" * 64,
            fit_end="2026-01-30",
            schema_version=engine_module.CAUSAL_FEATURE_SCHEMA_VERSION,
            modeling_calendar="NYSE",
            training_data_sha256="c" * 64,
            modeling_session_data_sha256="c" * 64,
        )
        self.model_preparation_manifest_ = CausalModelPreparationManifest(
            feature_hash="a" * 64,
            fit_end="2026-01-30",
            model_feature_names=tuple(self.feature_names_),
            pca_components=2,
            requested_hmm_components=2,
            inference_mode="fixed_out_of_sample",
            scaler_mode="fixed_prefix_robust",
            refit_interval_days=0,
        )
        self.model_training_end_ = "2026-01-30"
        self.causal_tail_probability_ = np.array([0.25, 0.75])
        self.causal_tail_as_of_ = "2026-07-24"
        self.causal_inference_mode_ = "fixed_snapshot_prefix_filter"
        engine_module._bind_raw_hmm_taxonomy(self)


def _buckets(model=None, *, taxonomy_id=None, second_bucket=None):
    model = model or _FakeHMM()
    count = MIN_REGIME_BUCKET_OBSERVATIONS
    return RegimeReturnBuckets(
        taxonomy_id=(
            taxonomy_id
            if taxonomy_id is not None
            else model.raw_hmm_taxonomy_id_
        ),
        model_training_end=model.model_training_end_,
        inference_as_of="2026-07-24",
        resolved_outcomes_through="2026-07-23",
        horizon_calendar_days=7,
        horizon_trading_days=5,
        buckets=(
            tuple(
                float(value)
                for value in np.linspace(-0.02, 0.02, count)
            ),
            tuple(
                float(value)
                for value in (
                    np.linspace(-0.04, 0.04, count)
                    if second_bucket is None
                    else second_bucket
                )
            ),
        ),
    )


class ProbabilityEngineContractTests(unittest.TestCase):
    def test_missing_inputs_raise_instead_of_returning_zero_probability(self):
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "TYPED_REGIME_BUCKETS_REQUIRED",
        ):
            get_probability_engine(
                500.0,
                20.0,
                {},
                horizon=7,
                hmm_model=_FakeHMM(),
            )
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "HMM_MODEL_UNAVAILABLE",
        ):
            get_probability_engine(
                500.0,
                20.0,
                _buckets(),
                horizon=7,
                hmm_model=None,
            )

    def test_legacy_dict_and_cross_model_taxonomy_are_rejected(self):
        model = _FakeHMM()
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "TYPED_REGIME_BUCKETS_REQUIRED",
        ):
            get_probability_engine(
                500.0,
                20.0,
                {
                    "State_0": np.zeros(10),
                    "State_1": np.zeros(10),
                },
                horizon=7,
                hmm_model=model,
            )

        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "REGIME_TAXONOMY_MISMATCH",
        ):
            get_probability_engine(
                500.0,
                20.0,
                _buckets(model, taxonomy_id="b" * 64),
                horizon=7,
                hmm_model=model,
            )

    def test_success_has_one_typed_non_executable_contract(self):
        def fake_fit(_bucket, regime_label=""):
            return {"regime_label": regime_label}

        def fake_query(model, _spot, _strike):
            return 0.20 if model["regime_label"] == "State_0" else 0.60

        with (
            patch.object(
                engine_module,
                "DataIngestor",
                side_effect=AssertionError(
                    "probability construction must not reacquire feature data"
                ),
            ),
            patch.object(
                engine_module,
                "validate_hmm_quality",
                return_value=(True, {"passed": True}),
            ),
            patch.object(engine_module, "fit_gmm", side_effect=fake_fit),
            patch.object(engine_module, "query_gmm", side_effect=fake_query),
        ):
            result = get_probability_engine(
                500.0,
                20.0,
                _buckets(),
                horizon=7,
                hmm_model=_FakeHMM(),
            )
            probability = result.probability(450.0)

        self.assertIs(type(result), ProbabilityEngineResult)
        self.assertEqual(result.validity_status, "UNVERIFIED")
        self.assertFalse(result.execution_eligible)
        self.assertEqual(len(result.projected_probabilities), 2)
        self.assertEqual(len(result.current_probabilities), 2)
        self.assertAlmostEqual(sum(result.projected_probabilities), 1.0)
        self.assertGreaterEqual(probability, 0.20)
        self.assertLessEqual(probability, 0.60)

    def test_statistically_collapsed_hmm_is_unavailable(self):
        with (
            patch.object(
                engine_module,
                "DataIngestor",
                side_effect=AssertionError(
                    "probability construction must not reacquire feature data"
                ),
            ),
            patch.object(
                engine_module,
                "validate_hmm_quality",
                return_value=(False, {"reason": "collapsed"}),
            ),
        ):
            with self.assertRaisesRegex(
                ProbabilityEngineUnavailable,
                "HMM_QUALITY_REJECTED",
            ):
                get_probability_engine(
                    500.0,
                    20.0,
                    _buckets(),
                    horizon=7,
                    hmm_model=_FakeHMM(),
                )

    def test_sparse_or_missing_state_bucket_is_unavailable(self):
        sparse = _buckets(second_bucket=(0.0,))
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "INSUFFICIENT_REGIME_BUCKET_STATE_1",
        ):
            get_probability_engine(
                500.0,
                20.0,
                sparse,
                horizon=7,
                hmm_model=_FakeHMM(),
            )

    def test_result_validates_probability_output_and_strike(self):
        result = ProbabilityEngineResult(
            probability_fn=lambda _strike: 1.5,
            regime_name="State_0",
            projected_probabilities=(1.0,),
            models=({"type": "fixture"},),
            current_probabilities=(1.0,),
            input_feature_hash="a" * 64,
            model_as_of_date="2026-07-24",
            inference_mode="fixed_snapshot_prefix_filter",
            raw_hmm_taxonomy_id="b" * 64,
            return_bucket_as_of="2026-07-24",
            return_outcomes_resolved_through="2026-07-23",
        )
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "INVALID_PROBABILITY_OUTPUT",
        ):
            result.probability(100.0)
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "INVALID_STRIKE",
        ):
            result.probability(float("nan"))

    def test_result_requires_exact_model_and_bucket_as_of(self):
        exact_result = ProbabilityEngineResult(
            probability_fn=lambda _strike: 0.5,
            regime_name="State_0",
            projected_probabilities=(1.0,),
            models=({"type": "fixture"},),
            current_probabilities=(1.0,),
            input_feature_hash="a" * 64,
            model_as_of_date="2026-07-24",
            inference_mode="fixed_snapshot_prefix_filter",
            raw_hmm_taxonomy_id="b" * 64,
            return_bucket_as_of="2026-07-24",
            return_outcomes_resolved_through="2026-07-23",
        )
        self.assertEqual(
            exact_result.return_bucket_as_of,
            exact_result.model_as_of_date,
        )

        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "INVALID_RETURN_RESOLUTION_DATE",
        ):
            ProbabilityEngineResult(
                probability_fn=lambda _strike: 0.5,
                regime_name="State_0",
                projected_probabilities=(1.0,),
                models=({"type": "fixture"},),
                current_probabilities=(1.0,),
                input_feature_hash="a" * 64,
                model_as_of_date="2026-07-24",
                inference_mode="fixed_snapshot_prefix_filter",
                raw_hmm_taxonomy_id="b" * 64,
                return_bucket_as_of="2026-07-25",
                return_outcomes_resolved_through="2026-07-24",
            )

    def test_missing_causal_tail_provenance_is_unavailable(self):
        model = _FakeHMM()
        model.causal_inference_mode_ = ""
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "MODEL_FEATURE_PROVENANCE_UNAVAILABLE",
        ):
            get_probability_engine(
                500.0,
                20.0,
                _buckets(),
                horizon=7,
                hmm_model=model,
            )

    def test_bucket_as_of_must_equal_model_causal_tail(self):
        model = _FakeHMM()
        mismatched = RegimeReturnBuckets(
            taxonomy_id=model.raw_hmm_taxonomy_id_,
            model_training_end=model.model_training_end_,
            inference_as_of="2026-07-23",
            resolved_outcomes_through="2026-07-22",
            horizon_calendar_days=7,
            horizon_trading_days=5,
            buckets=_buckets(model).buckets,
        )
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "REGIME_BUCKET_INFERENCE_AS_OF_MISMATCH",
        ):
            get_probability_engine(
                500.0,
                20.0,
                mismatched,
                horizon=7,
                hmm_model=model,
            )


if __name__ == "__main__":
    unittest.main()
