import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtesting.backtest_runner as runner
from backtesting.regime_bridge import (
    BacktestRegimeProtocol,
    ExactAssignmentTargetCache,
    ExactRegimeRunCache,
    RegimeBridgeUnavailable,
)
from live_trading.regime_taxonomy import (
    FinalRiskRegimeRef,
    RegimeReturnBuckets,
)


def _protocol():
    return BacktestRegimeProtocol(
        calibration_end="2025-01-03",
        test_start="2025-01-13",
        test_end="2025-01-31",
        inference_method="walk_forward_expanding",
    )


def _model(*, taxonomy_id="a" * 64, raw_state=1):
    return SimpleNamespace(
        n_components=3,
        raw_hmm_taxonomy_id_=taxonomy_id,
        model_training_end_="2025-01-03",
        causal_tail_as_of_="2025-01-10",
        causal_inference_mode_="fixed_snapshot_prefix_filter",
        causal_tail_probability_=(
            [0.05, 0.90, 0.05]
            if raw_state == 1
            else [0.90, 0.05, 0.05]
        ),
    )


def _buckets(*, taxonomy_id="a" * 64, bucket_size=10):
    return RegimeReturnBuckets(
        taxonomy_id=taxonomy_id,
        model_training_end="2025-01-03",
        inference_as_of="2025-01-10",
        resolved_outcomes_through="2025-01-09",
        horizon_calendar_days=7,
        horizon_trading_days=5,
        buckets=tuple(
            tuple(
                (state - 1) * 0.01 + value * 0.0001
                for value in range(1, bucket_size + 1)
            )
            for state in range(3)
        ),
    )


def _fit_and_query(captured_bucket):
    def fit_model(bucket, **_kwargs):
        captured_bucket.append(tuple(bucket))
        return object()

    def query_probability(_model, spot, strike):
        return math.log(strike / spot) + 0.5

    def solve_root(objective, _low, _high):
        root = -0.45
        objective(root)
        return root

    return fit_model, query_probability, solve_root


class RegimeAssignmentRunnerContractTests(unittest.TestCase):
    def test_exact_prior_as_of_and_raw_state_bucket_are_used(self):
        calls = []
        captured_bucket = []

        def builder(*_args, **kwargs):
            calls.append(kwargs)
            return _buckets(), _model(raw_state=1), []

        fit_model, query_probability, solve_root = _fit_and_query(
            captured_bucket
        )
        strike, evidence, manifest = (
            runner._resolve_regime_assignment_entry(
                protocol=_protocol(),
                regime_cache=ExactRegimeRunCache(_protocol()),
                target_cache=ExactAssignmentTargetCache(),
                entry_session="2025-01-13",
                signal_session="2025-01-10",
                final_risk_regime=FinalRiskRegimeRef(
                    2,
                    "Panic / Crisis (2)",
                ),
                horizon_calendar_days=7,
                spot=100.0,
                target_assignment_probability=0.05,
                bucket_builder=builder,
                fit_model=fit_model,
                query_probability=query_probability,
                solve_root=solve_root,
            )
        )

        self.assertEqual(calls[0]["as_of_date"], "2025-01-10")
        self.assertNotEqual(calls[0]["as_of_date"], "2025-01-13")
        self.assertEqual(captured_bucket, [_buckets().buckets[1]])
        self.assertEqual(evidence.raw_hmm_state.state, 1)
        self.assertEqual(evidence.final_risk_regime.state, 2)
        self.assertIsNotNone(strike)
        self.assertIsNotNone(manifest)
        self.assertEqual(evidence.validity_status, "UNVERIFIED")

    def test_bucket_failures_block_without_a_default_strike(self):
        cases = (
            (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RegimeBridgeUnavailable(
                        "HISTORICAL_PREFIX_UNAVAILABLE"
                    )
                ),
                "HISTORICAL_PREFIX_UNAVAILABLE",
            ),
            (
                lambda *_args, **_kwargs: (
                    _buckets(bucket_size=2),
                    _model(),
                    [],
                ),
                "INSUFFICIENT_REGIME_BUCKET_STATE_1",
            ),
            (
                lambda *_args, **_kwargs: (
                    _buckets(taxonomy_id="b" * 64),
                    _model(taxonomy_id="a" * 64),
                    [],
                ),
                "REGIME_TAXONOMY_MISMATCH",
            ),
        )
        for builder, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                fit_model, query_probability, solve_root = _fit_and_query([])
                strike, evidence, _manifest = (
                    runner._resolve_regime_assignment_entry(
                        protocol=_protocol(),
                        regime_cache=ExactRegimeRunCache(_protocol()),
                        target_cache=ExactAssignmentTargetCache(),
                        entry_session="2025-01-13",
                        signal_session="2025-01-10",
                        final_risk_regime=FinalRiskRegimeRef(
                            0,
                            "Expansion (0)",
                        ),
                        horizon_calendar_days=7,
                        spot=100.0,
                        target_assignment_probability=0.05,
                        bucket_builder=builder,
                        fit_model=fit_model,
                        query_probability=query_probability,
                        solve_root=solve_root,
                    )
                )
                self.assertIsNone(strike)
                self.assertEqual(
                    evidence.unavailable_code,
                    expected_code,
                )
                self.assertEqual(
                    evidence.validity_status,
                    "UNAVAILABLE",
                )
                self.assertIsNone(evidence.assignment_probability)

    def test_evidence_and_manifest_serialize_into_trade_result_and_path(self):
        fit_model, query_probability, solve_root = _fit_and_query([])
        strike, evidence, manifest = (
            runner._resolve_regime_assignment_entry(
                protocol=_protocol(),
                regime_cache=ExactRegimeRunCache(_protocol()),
                target_cache=ExactAssignmentTargetCache(),
                entry_session="2025-01-13",
                signal_session="2025-01-10",
                final_risk_regime=FinalRiskRegimeRef(
                    0,
                    "Expansion (0)",
                ),
                horizon_calendar_days=7,
                spot=100.0,
                target_assignment_probability=0.05,
                bucket_builder=lambda *_args, **_kwargs: (
                    _buckets(),
                    _model(),
                    [],
                ),
                fit_model=fit_model,
                query_probability=query_probability,
                solve_root=solve_root,
            )
        )
        trade = runner.SpreadTrade(
            entry_date="2025-01-13",
            short_strike=float(strike),
            regime_evidence=evidence,
        )
        result = runner.BacktestResult(
            trades=[trade],
            regime_protocol=_protocol(),
            regime_decision_history=[evidence],
            regime_return_bucket_manifests={
                evidence.return_bucket_manifest_sha256: manifest
            },
        )
        path_logger = runner.BacktestPathLogger("/tmp/not-written.json")
        path_logger.log_day(
            date_str="2025-01-13",
            spot=100.0,
            regime=0,
            regime_name="Expansion (0)",
            regime_v2=None,
            nlv=100000.0,
            cash=100000.0,
            margin=0.0,
            active_trades=[trade],
            events=[
                {
                    "type": "entry",
                    "regime_evidence": evidence.manifest(),
                }
            ],
        )
        serialized_trade = path_logger.log_data["path"][0][
            "active_trades"
        ][0]
        self.assertEqual(
            serialized_trade["regime_evidence"][
                "return_bucket_manifest_sha256"
            ],
            evidence.return_bucket_manifest_sha256,
        )
        self.assertIn(
            evidence.return_bucket_manifest_sha256,
            result.regime_return_bucket_manifests,
        )

    def test_legacy_untyped_bucket_and_fallback_paths_are_absent(self):
        source = inspect.getsource(runner)
        self.assertNotIn('.get(f"State_{current_regime}"', source)
        self.assertNotIn("result.gmm_cache", source)
        self.assertNotIn("np.array([0.0])", source)
        self.assertNotIn('result.regime_probabilities.loc[prev_td', source)


class _NoDataCache:
    def __init__(self, *_args, **_kwargs):
        pass

    def clear_daily_memory_cache(self):
        pass

    def close(self):
        pass


class _NoContractsClient:
    def __init__(self, *_args, **_kwargs):
        self.api_calls = 0
        self.cache_hits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def fetch_contracts_list(self, *_args, **_kwargs):
        return []

    async def fetch_chain_ohlcv_batch(self, *_args, **_kwargs):
        return 0


class BaselineRunnerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_does_not_create_or_require_regime_evidence(self):
        index = pd.bdate_range("2011-05-03", "2025-01-13")
        underlying = pd.Series(
            np.linspace(100.0, 500.0, len(index)),
            index=index.strftime("%Y-%m-%d"),
        )
        with (
            patch.object(runner, "OptionDataCache", _NoDataCache),
            patch.object(runner, "MassiveAPIClient", _NoContractsClient),
            patch.object(
                runner,
                "_fetch_underlying_prices",
                return_value=pd.Series(
                    [5.0],
                    index=["2025-01-13"],
                ),
            ),
        ):
            result = await runner.run_put_credit_spread_backtest(
                start_date="2025-01-13",
                end_date="2025-01-13",
                regimes={"2025-01-13": 2},
                regime_labels={2: "Panic / Crisis (2)"},
                underlying_prices=underlying,
                vix_prices=pd.Series(
                    [18.0],
                    index=["2025-01-13"],
                ),
                offline_only=True,
            )
        self.assertIsNone(result.regime_protocol)
        self.assertEqual(result.regime_decision_history, [])
        self.assertEqual(
            result.regime_initialization_unavailable_code,
            "",
        )
        self.assertEqual(result.regime_history, [("2025-01-13", 2)])

    async def test_untyped_external_map_cannot_authorize_regime_run(self):
        from backtesting.experiment_tracker import ExperimentTracker

        index = pd.bdate_range("2011-05-03", "2025-01-13")
        underlying = pd.Series(
            np.linspace(100.0, 500.0, len(index)),
            index=index.strftime("%Y-%m-%d"),
        )
        with (
            patch.object(runner, "OptionDataCache", _NoDataCache),
            patch.object(runner, "MassiveAPIClient", _NoContractsClient),
            patch.object(
                runner,
                "_fetch_underlying_prices",
                return_value=pd.Series(
                    [5.0],
                    index=["2025-01-13"],
                ),
            ),
            patch.object(runner, "plot_results_interactive"),
            patch.object(ExperimentTracker, "record_experiment"),
        ):
            result = await runner.run_put_credit_spread_backtest(
                start_date="2025-01-13",
                end_date="2025-01-13",
                regimes={"2025-01-13": 0},
                regime_labels={0: "Expansion (0)"},
                underlying_prices=underlying,
                vix_prices=pd.Series(
                    [18.0],
                    index=["2025-01-13"],
                ),
                strategy_config={"regime_aware": True},
                offline_only=True,
            )
        self.assertIsNone(result.regime_protocol)
        self.assertEqual(
            result.regime_initialization_unavailable_code,
            "UNTYPED_EXTERNAL_REGIME_MAP_REJECTED",
        )
        self.assertEqual(result.trades, [])


if __name__ == "__main__":
    unittest.main()
