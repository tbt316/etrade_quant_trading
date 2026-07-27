import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import live_trading.ev_engine as engine_module
from live_trading.ev_engine import (
    MIN_REGIME_BUCKET_OBSERVATIONS,
    ProbabilityEngineResult,
    ProbabilityEngineUnavailable,
    get_probability_engine,
)


class _FakeScaler:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


class _FakePCA:
    def transform(self, frame):
        return np.asarray(frame, dtype=float)


class _FakeHMM:
    n_components = 2
    scaler_ = _FakeScaler()
    fusion_ = SimpleNamespace(sparse_pca=_FakePCA())
    transmat_ = np.array([[0.90, 0.10], [0.20, 0.80]])

    def predict_proba(self, values):
        return np.tile(np.array([[0.25, 0.75]]), (len(values), 1))


class _FakeIngestor:
    async def build_fused_dataset(self, *_args, **_kwargs):
        return pd.DataFrame(
            {
                "feature_a": np.linspace(-1.0, 1.0, 30),
                "feature_b": np.linspace(1.0, -1.0, 30),
            },
            index=pd.date_range("2026-01-02", periods=30, freq="B"),
        )


def _buckets():
    count = MIN_REGIME_BUCKET_OBSERVATIONS
    return {
        "State_0": np.linspace(-0.02, 0.02, count),
        "State_1": np.linspace(-0.04, 0.04, count),
    }


class ProbabilityEngineContractTests(unittest.TestCase):
    def test_missing_inputs_raise_instead_of_returning_zero_probability(self):
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "REGIME_BUCKETS_UNAVAILABLE",
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

    def test_success_has_one_typed_non_executable_contract(self):
        def fake_fit(_bucket, regime_label=""):
            return {"regime_label": regime_label}

        def fake_query(model, _spot, _strike):
            return 0.20 if model["regime_label"] == "State_0" else 0.60

        with (
            patch.object(engine_module, "DataIngestor", _FakeIngestor),
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
            patch.object(engine_module, "DataIngestor", _FakeIngestor),
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
        sparse = _buckets()
        sparse["State_1"] = np.array([0.0])
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


if __name__ == "__main__":
    unittest.main()
