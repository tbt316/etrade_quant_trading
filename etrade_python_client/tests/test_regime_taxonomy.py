import unittest

import numpy as np

from live_trading.regime_taxonomy import (
    FinalRiskRegimeRef,
    RawHMMStateRef,
    RegimeReturnBuckets,
    RegimeTaxonomyError,
    build_raw_hmm_taxonomy_id,
)


def _taxonomy_id(*, means=None):
    return build_raw_hmm_taxonomy_id(
        n_components=2,
        feature_manifest_sha256="a" * 64,
        training_end="2025-01-31",
        pipeline_version="causal-hmm.v1",
        state_labels=("calm", "volatile"),
        means=np.asarray([[0.0], [1.0]]) if means is None else means,
        covariances=np.asarray([[[1.0]], [[2.0]]]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        start_probabilities=np.asarray([0.7, 0.3]),
    )


class RawHMMTaxonomyTests(unittest.TestCase):
    def test_identity_is_deterministic_and_parameter_bound(self):
        first = _taxonomy_id()
        self.assertEqual(first, _taxonomy_id())
        self.assertNotEqual(
            first,
            _taxonomy_id(means=np.asarray([[0.0], [1.1]])),
        )
        with self.assertRaisesRegex(RegimeTaxonomyError, "parameter shapes"):
            _taxonomy_id(means=np.asarray([[0.0]]))
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "regular numeric array",
        ):
            _taxonomy_id(means=[[0.0], [1.0, 2.0]])

    def test_gmm_mixture_weights_are_part_of_the_identity(self):
        base = dict(
            n_components=2,
            feature_manifest_sha256="a" * 64,
            training_end="2025-01-31",
            pipeline_version="causal-hmm.v1",
            state_labels=("calm", "volatile"),
            means=np.asarray(
                [[[0.0], [0.5]], [[1.0], [1.5]]]
            ),
            covariances=np.asarray(
                [[[1.0], [1.2]], [[2.0], [2.2]]]
            ),
            transition_matrix=np.asarray([[0.9, 0.1], [0.2, 0.8]]),
            start_probabilities=np.asarray([0.7, 0.3]),
        )
        first = build_raw_hmm_taxonomy_id(
            **base,
            mixture_weights=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        )
        second = build_raw_hmm_taxonomy_id(
            **base,
            mixture_weights=np.asarray([[0.7, 0.3], [0.3, 0.7]]),
        )
        self.assertNotEqual(first, second)

    def test_raw_and_final_regime_types_are_not_interchangeable(self):
        taxonomy_id = _taxonomy_id()
        buckets = RegimeReturnBuckets(
            taxonomy_id=taxonomy_id,
            model_training_end="2025-01-31",
            inference_as_of="2025-02-04",
            resolved_outcomes_through="2025-02-03",
            horizon_calendar_days=7,
            horizon_trading_days=5,
            buckets=((0.01, -0.02), (0.03, -0.04)),
        )
        self.assertEqual(
            buckets.bucket_for(RawHMMStateRef(taxonomy_id, 1)),
            (0.03, -0.04),
        )
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "RawHMMStateRef",
        ):
            buckets.bucket_for(
                FinalRiskRegimeRef(
                    1,
                    "Cautious Decline (1)",
                )  # type: ignore[arg-type]
            )

    def test_final_overlay_uses_a_closed_state_label_mapping(self):
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "closed overlay taxonomy",
        ):
            FinalRiskRegimeRef(1, "Panic / Crisis (2)")
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "closed overlay taxonomy",
        ):
            FinalRiskRegimeRef(3, "New regime")

    def test_bucket_lookup_rejects_model_taxonomy_mismatch(self):
        buckets = RegimeReturnBuckets(
            taxonomy_id=_taxonomy_id(),
            model_training_end="2025-01-31",
            inference_as_of="2025-02-04",
            resolved_outcomes_through="2025-02-03",
            horizon_calendar_days=7,
            horizon_trading_days=5,
            buckets=((0.01,), (0.02,)),
        )
        with self.assertRaisesRegex(RegimeTaxonomyError, "taxonomy mismatch"):
            buckets.bucket_for(RawHMMStateRef("b" * 64, 0))

    def test_resolved_outcomes_must_precede_inference(self):
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "strictly before",
        ):
            RegimeReturnBuckets(
                taxonomy_id=_taxonomy_id(),
                model_training_end="2025-01-31",
                inference_as_of="2025-02-04",
                resolved_outcomes_through="2025-02-04",
                horizon_calendar_days=7,
                horizon_trading_days=5,
                buckets=((0.01,), (0.02,)),
            )

    def test_every_state_requires_a_resolved_return(self):
        with self.assertRaisesRegex(
            RegimeTaxonomyError,
            "every raw HMM state",
        ):
            RegimeReturnBuckets(
                taxonomy_id=_taxonomy_id(),
                model_training_end="2025-01-31",
                inference_as_of="2025-02-04",
                resolved_outcomes_through="2025-02-03",
                horizon_calendar_days=7,
                horizon_trading_days=5,
                buckets=((0.01,), ()),
            )

    def test_manifest_binds_counts_and_values_without_exposing_values(self):
        buckets = RegimeReturnBuckets(
            taxonomy_id=_taxonomy_id(),
            model_training_end="2025-01-31",
            inference_as_of="2025-02-04",
            resolved_outcomes_through="2025-02-03",
            horizon_calendar_days=7,
            horizon_trading_days=5,
            buckets=((0.01, -0.02), (0.03,)),
        )
        manifest = buckets.manifest()
        self.assertEqual(manifest["bucket_counts"], [2, 1])
        self.assertEqual(len(manifest["bucket_sha256"]), 2)
        self.assertNotIn("buckets", manifest)
        self.assertFalse(manifest["execution_eligible"])
        self.assertEqual(manifest["validity_status"], "UNVERIFIED")
        changed = RegimeReturnBuckets(
            taxonomy_id=_taxonomy_id(),
            model_training_end="2025-01-31",
            inference_as_of="2025-02-04",
            resolved_outcomes_through="2025-02-03",
            horizon_calendar_days=7,
            horizon_trading_days=5,
            buckets=((0.01, -0.03), (0.03,)),
        )
        self.assertEqual(
            changed.manifest()["bucket_counts"],
            manifest["bucket_counts"],
        )
        self.assertNotEqual(
            changed.manifest()["bucket_sha256"],
            manifest["bucket_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
