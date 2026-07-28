import ast
import inspect
import pickle
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live_trading.ev_engine as engine
import live_trading.ev_plots as plots
import live_trading.etrade_cover_call_new as live_agent
import market_regime_detect as regime_review
from live_trading.ev_engine import (
    CausalModelPreparationManifest,
    ProbabilityEngineUnavailable,
    build_regime_return_arrays,
)
from live_trading.regime_taxonomy import (
    FINAL_RISK_NAMESPACE,
    RegimeReturnBuckets,
)


class _BoundFakeHMM:
    def __init__(self, training_end="2025-01-06"):
        self.n_components = 2
        self.startprob_ = np.array([0.6, 0.4])
        self.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.means_ = np.array([[0.0], [1.0]])
        self.covars_ = np.array([[[1.0]], [[2.0]]])
        self.feature_names_ = ["SPY_Log_Return"]
        self.feature_manifest_ = SimpleNamespace(
            feature_hash="a" * 64,
            fit_end=training_end,
            schema_version=engine.CAUSAL_FEATURE_SCHEMA_VERSION,
            modeling_calendar="NYSE",
            training_data_sha256="c" * 64,
            modeling_session_data_sha256="c" * 64,
        )
        self.model_preparation_manifest_ = CausalModelPreparationManifest(
            feature_hash="a" * 64,
            fit_end=training_end,
            model_feature_names=tuple(self.feature_names_),
            pca_components=1,
            requested_hmm_components=2,
            inference_mode="as_of_snapshot_fit",
            scaler_mode="fixed_prefix_robust",
            refit_interval_days=0,
        )
        self.model_training_end_ = training_end
        engine._bind_raw_hmm_taxonomy(self)
        self.causal_tail_as_of_ = "2025-01-10"


def _buckets(model, *, taxonomy_id=None):
    return RegimeReturnBuckets(
        taxonomy_id=taxonomy_id or model.raw_hmm_taxonomy_id_,
        model_training_end=model.model_training_end_,
        inference_as_of="2025-01-10",
        resolved_outcomes_through="2025-01-09",
        horizon_calendar_days=1,
        horizon_trading_days=1,
        buckets=((0.01, 0.02), (-0.01, -0.02)),
    )


class RegimeReturnBucketConstructionTests(unittest.TestCase):
    def test_bucket_builder_requires_an_explicit_as_of_date(self):
        with patch.object(
            engine,
            "fetch_historical_data",
            side_effect=AssertionError(
                "missing as-of must fail before market-data access"
            ),
        ):
            with self.assertRaisesRegex(
                ProbabilityEngineUnavailable,
                "EXPLICIT_BUCKET_AS_OF_DATE_REQUIRED",
            ):
                build_regime_return_arrays(
                    1,
                    horizon=7,
                    n_components=2,
                )

    def test_market_data_prefix_hash_is_value_and_order_bound(self):
        frame = pd.DataFrame(
            {
                "SPY_Close": [100.0, 101.0],
                "VIX_Close": [20.0, 19.5],
            },
            index=pd.bdate_range("2025-01-02", periods=2),
        )
        original = engine._market_data_prefix_sha256(frame)
        self.assertEqual(
            original,
            engine._market_data_prefix_sha256(frame.copy()),
        )
        changed = frame.copy()
        changed.iloc[0, 0] = 100.25
        self.assertNotEqual(
            original,
            engine._market_data_prefix_sha256(changed),
        )
        with self.assertRaisesRegex(ValueError, "INVALID_MARKET_DATA_PREFIX"):
            engine._market_data_prefix_sha256(frame.iloc[::-1])

    def test_bucket_as_of_rejects_weekends_and_nyse_holidays(self):
        history = pd.DataFrame(
            {
                "SPY_Close": np.linspace(100.0, 109.0, 10),
                "VIX_Close": np.linspace(20.0, 21.0, 10),
            },
            index=pd.to_datetime(
                [
                    "2025-01-13",
                    "2025-01-14",
                    "2025-01-15",
                    "2025-01-16",
                    "2025-01-17",
                    "2025-01-18",
                    "2025-01-19",
                    "2025-01-20",
                    "2025-01-21",
                    "2025-01-22",
                ]
            ),
        )
        for as_of_date in ("2025-01-18", "2025-01-20"):
            with self.subTest(as_of_date=as_of_date):
                with (
                    patch.object(
                        engine,
                        "fetch_historical_data",
                        return_value=history,
                    ),
                    patch.object(
                        engine,
                        "load_regime_snapshot",
                        side_effect=AssertionError(
                            "invalid as-of must fail before cache lookup"
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ProbabilityEngineUnavailable,
                        "BUCKET_AS_OF_MUST_BE_NYSE_SESSION",
                    ):
                        build_regime_return_arrays(
                            1,
                            horizon=1,
                            n_components=2,
                            as_of_date=as_of_date,
                        )

    def test_terminal_outcome_must_resolve_before_as_of(self):
        dates = pd.bdate_range("2025-01-02", periods=8)
        as_of = dates[6]
        anchor = dates[2]
        history = pd.DataFrame(
            {
                "SPY_Close": 100.0 + np.arange(len(dates)),
                "VIX_Close": 20.0 + np.arange(len(dates)) * 0.1,
            },
            index=dates,
        )
        model = _BoundFakeHMM(anchor.strftime("%Y-%m-%d"))

        def feature_frame(causal_df, *_args, **_kwargs):
            return pd.DataFrame(
                {
                    "HMM_State": np.arange(len(causal_df)) % 2,
                    "Log_Return": np.log(
                        causal_df["SPY_Close"]
                        / causal_df["SPY_Close"].shift(1)
                    ),
                },
                index=causal_df.index,
            )

        with (
            patch.object(engine, "fetch_historical_data", return_value=history),
            patch.object(
                engine,
                "_hmm_refit_anchor_date",
                return_value=anchor,
            ),
            patch.object(engine, "load_regime_snapshot", return_value=None),
            patch.object(engine, "load_regime_cache", return_value=None),
            patch.object(
                engine,
                "train_regime_hmm",
                return_value=(model, 2, pd.DataFrame()),
            ),
            patch.object(
                engine,
                "_build_causal_regime_feature_frame",
                side_effect=feature_frame,
            ),
            patch.object(engine, "save_regime_cache", return_value=True),
        ):
            buckets, returned_model, daily_models = (
                build_regime_return_arrays(
                    1,
                    horizon=1,
                    n_components=2,
                    as_of_date=as_of.strftime("%Y-%m-%d"),
                )
            )

        self.assertIs(type(buckets), RegimeReturnBuckets)
        self.assertIs(returned_model, model)
        self.assertEqual(len(daily_models), 2)
        self.assertEqual(
            buckets.resolved_outcomes_through,
            dates[4].strftime("%Y-%m-%d"),
        )

    def test_one_trading_session_horizon_skips_weekend_and_holiday(self):
        dates = pd.to_datetime(
            [
                "2025-01-13",
                "2025-01-14",
                "2025-01-15",
                "2025-01-16",
                "2025-01-17",
                "2025-01-18",
                "2025-01-19",
                "2025-01-20",
                "2025-01-21",
                "2025-01-22",
                "2025-01-23",
                "2025-01-24",
            ]
        )
        prices = [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            1000.0,
            1001.0,
            1002.0,
            105.0,
            106.0,
            107.0,
            108.0,
        ]
        history = pd.DataFrame(
            {
                "SPY_Close": prices,
                "VIX_Close": np.linspace(20.0, 21.1, len(dates)),
            },
            index=dates,
        )
        anchor = pd.Timestamp("2025-01-15")
        model = _BoundFakeHMM(anchor.strftime("%Y-%m-%d"))
        scored_indexes = []

        def feature_frame(causal_df, *_args, **_kwargs):
            scored_indexes.append(causal_df.index.copy())
            return pd.DataFrame(
                {
                    "HMM_State": np.arange(len(causal_df)) % 2,
                    "Log_Return": np.log(
                        causal_df["SPY_Close"]
                        / causal_df["SPY_Close"].shift(1)
                    ),
                },
                index=causal_df.index,
            )

        with (
            patch.object(engine, "fetch_historical_data", return_value=history),
            patch.object(
                engine,
                "_hmm_refit_anchor_date",
                return_value=anchor,
            ),
            patch.object(engine, "load_regime_snapshot", return_value=None),
            patch.object(engine, "load_regime_cache", return_value=None),
            patch.object(
                engine,
                "train_regime_hmm",
                return_value=(model, 2, pd.DataFrame()),
            ),
            patch.object(
                engine,
                "_build_causal_regime_feature_frame",
                side_effect=feature_frame,
            ),
            patch.object(engine, "save_regime_cache", return_value=True),
        ):
            buckets, _, _ = build_regime_return_arrays(
                1,
                horizon=1,
                n_components=2,
                force_refit=True,
                as_of_date="2025-01-24",
            )

        self.assertEqual(buckets.horizon_trading_days, 1)
        self.assertEqual(
            scored_indexes[0].strftime("%Y-%m-%d").tolist(),
            [
                "2025-01-13",
                "2025-01-14",
                "2025-01-15",
                "2025-01-16",
                "2025-01-17",
                "2025-01-21",
                "2025-01-22",
                "2025-01-23",
                "2025-01-24",
            ],
        )
        friday_to_tuesday = 105.0 / 104.0 - 1.0
        self.assertTrue(
            any(
                np.isclose(value, friday_to_tuesday)
                for bucket in buckets.buckets
                for value in bucket
            )
        )
        self.assertEqual(
            sum(len(bucket) for bucket in buckets.buckets),
            7,
        )
        self.assertEqual(
            buckets.taxonomy_id,
            model.raw_hmm_taxonomy_id_,
        )

        modeling_prices = history.loc[
            pd.to_datetime(
                [
                    "2025-01-13",
                    "2025-01-14",
                    "2025-01-15",
                    "2025-01-16",
                    "2025-01-17",
                    "2025-01-21",
                    "2025-01-22",
                    "2025-01-23",
                    "2025-01-24",
                ]
            ),
            "SPY_Close",
        ]
        expected = (
            modeling_prices.shift(-1) / modeling_prices - 1.0
        ).iloc[:7]
        actual = sorted(
            value
            for bucket in buckets.buckets
            for value in bucket
        )
        self.assertTrue(
            np.allclose(actual, sorted(expected.to_numpy(dtype=float)))
        )

    def test_missing_resolved_state_uses_stable_unavailable_code(self):
        dates = pd.bdate_range("2025-01-02", periods=8)
        as_of = dates[6]
        anchor = dates[2]
        history = pd.DataFrame(
            {
                "SPY_Close": 100.0 + np.arange(len(dates)),
                "VIX_Close": 20.0,
            },
            index=dates,
        )
        model = _BoundFakeHMM(anchor.strftime("%Y-%m-%d"))
        one_state_trace = pd.DataFrame(
            {
                "HMM_State": np.zeros(7, dtype=int),
                "Log_Return": np.linspace(-0.01, 0.01, 7),
            },
            index=dates[:7],
        )

        with (
            patch.object(engine, "fetch_historical_data", return_value=history),
            patch.object(
                engine,
                "_hmm_refit_anchor_date",
                return_value=anchor,
            ),
            patch.object(engine, "load_regime_snapshot", return_value=None),
            patch.object(engine, "load_regime_cache", return_value=None),
            patch.object(
                engine,
                "train_regime_hmm",
                return_value=(model, 2, pd.DataFrame()),
            ),
            patch.object(
                engine,
                "_build_causal_regime_feature_frame",
                return_value=one_state_trace,
            ),
        ):
            with self.assertRaisesRegex(
                ProbabilityEngineUnavailable,
                "INCOMPLETE_REGIME_RETURN_BUCKETS",
            ):
                build_regime_return_arrays(
                    1,
                    horizon=1,
                    n_components=2,
                    as_of_date=as_of.strftime("%Y-%m-%d"),
                )

    def test_snapshot_cache_rejects_legacy_and_tampered_taxonomies(self):
        model = _BoundFakeHMM()
        buckets = _buckets(model)
        metadata = {
            "as_of_date": "2025-01-10",
            "horizon_calendar_days": 1,
            "trading_horizon": 1,
            "requested_n_components": 2,
            "probability_model": "gmm",
            "model_class": "GaussianHMM",
        }
        daily_models = [{"type": "fixture"}, {"type": "fixture"}]

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                engine,
                "REGIME_SNAPSHOT_CACHE_DIR",
                temp_dir,
            ),
        ):
            path = engine.save_regime_snapshot(
                buckets,
                model,
                daily_models,
                metadata=metadata,
            )
            self.assertIsNotNone(path)
            self.assertIsNotNone(engine.load_regime_snapshot(metadata))

            legacy_payload = {
                "regime_dict": {
                    "State_0": np.array([0.01]),
                    "State_1": np.array([-0.01]),
                },
                "hmm_model": model,
                "daily_models": daily_models,
                "timestamp": "2025-01-10T00:00:00",
                "metadata": metadata,
            }
            with open(path, "wb") as cache_file:
                pickle.dump(legacy_payload, cache_file)
            self.assertIsNone(engine.load_regime_snapshot(metadata))

        valid_payload = {
            "regime_dict": _buckets(model, taxonomy_id="b" * 64),
            "hmm_model": model,
            "daily_models": daily_models,
            "timestamp": "2025-01-10T00:00:00",
            "metadata": {
                **metadata,
                "raw_hmm_taxonomy_id": model.raw_hmm_taxonomy_id_,
                "return_buckets_manifest": buckets.manifest(),
            },
        }
        self.assertIsNone(engine._validate_regime_cache_payload(valid_payload))

        changed_values = RegimeReturnBuckets(
            taxonomy_id=model.raw_hmm_taxonomy_id_,
            model_training_end=model.model_training_end_,
            inference_as_of="2025-01-10",
            resolved_outcomes_through="2025-01-09",
            horizon_calendar_days=1,
            horizon_trading_days=1,
            buckets=((0.01, 0.03), (-0.01, -0.02)),
        )
        self.assertIsNone(
            engine._validate_regime_cache_payload(
                {
                    "regime_dict": changed_values,
                    "hmm_model": model,
                    "daily_models": daily_models,
                    "metadata": {
                        **metadata,
                        "raw_hmm_taxonomy_id": model.raw_hmm_taxonomy_id_,
                        "return_buckets_manifest": buckets.manifest(),
                    },
                }
            )
        )

        model.means_[0, 0] = 0.25
        self.assertIsNone(
            engine._validate_regime_cache_payload(
                {
                    "regime_dict": buckets,
                    "hmm_model": model,
                    "daily_models": daily_models,
                    "metadata": {
                        **metadata,
                        "raw_hmm_taxonomy_id": buckets.taxonomy_id,
                        "return_buckets_manifest": buckets.manifest(),
                    },
                }
            )
        )

    def test_cache_binding_rejects_cross_date_model_tail(self):
        model = _BoundFakeHMM()
        buckets = _buckets(model)
        daily_models = [{"type": "fixture"}, {"type": "fixture"}]
        metadata = {
            "raw_hmm_taxonomy_id": buckets.taxonomy_id,
            "return_buckets_manifest": buckets.manifest(),
        }
        model.causal_tail_as_of_ = "2025-01-09"

        self.assertIsNone(
            engine._validate_regime_cache_payload(
                {
                    "regime_dict": buckets,
                    "hmm_model": model,
                    "daily_models": daily_models,
                    "metadata": metadata,
                }
            )
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                engine,
                "REGIME_SNAPSHOT_CACHE_DIR",
                temp_dir,
            ),
        ):
            self.assertIsNone(
                engine.save_regime_snapshot(
                    buckets,
                    model,
                    daily_models,
                    metadata=metadata,
                )
            )


class RegimePlotNamespaceTests(unittest.TestCase):
    def test_unsafe_forward_diagnostics_are_stable_tombstones(self):
        disabled = (
            (
                plots.plot_gmm_clusters,
                "UNSAFE_REGIME_GMM_CLUSTER_DIAGNOSTIC_DISABLED",
            ),
            (
                plots.run_calibration_backtest,
                "UNSAFE_REGIME_CALIBRATION_BACKTEST_DISABLED",
            ),
            (
                plots.sample_prediction_outcomes,
                "UNSAFE_REGIME_SAMPLE_OUTCOMES_DISABLED",
            ),
        )
        for function, code in disabled:
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, code):
                    function()

    def test_raw_hmm_label_cannot_change_live_risk_policy(self):
        source = inspect.getsource(plots.main)
        self.assertNotIn("STRATEGY_MAP", source)
        self.assertNotIn("is_turmoil", source)
        self.assertNotIn("vix_price = 20.0", source)

    def test_raw_hmm_columns_are_not_overwritten_by_overlay(self):
        original = pd.DataFrame(
            {
                "HMM_State": [0, 1],
                "Regime_Label": ["Raw quiet", "Raw volatile"],
                "Detected_Regime_State": [2, 0],
                "Detected_Regime_Label": [
                    "Panic / Crisis (2)",
                    "Expansion (0)",
                ],
            },
            index=pd.bdate_range("2025-01-02", periods=2),
        )

        prepared = plots._prepare_regime_plot_frame(original)

        self.assertEqual(prepared["HMM_State"].tolist(), [0, 1])
        self.assertEqual(
            prepared["Regime_Label"].tolist(),
            ["Raw quiet", "Raw volatile"],
        )
        self.assertEqual(
            prepared["Overlay_Regime_State"].tolist(),
            [2, 0],
        )
        self.assertNotIn("Overlay_Regime_State", original.columns)

    def test_calibration_cutoff_is_strictly_before_display_period(self):
        frame = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.bdate_range("2025-01-02", periods=3),
        )
        self.assertEqual(
            plots._calibration_fit_end(
                frame,
                "2025-01-06",
                context="fixture",
            ),
            "2025-01-03",
        )
        with self.assertRaisesRegex(ValueError, "strictly before"):
            plots._calibration_fit_end(
                frame,
                "2025-01-02",
                context="fixture",
            )

    def test_backtest_plot_caller_uses_only_exact_final_overlay(self):
        feature_df = pd.DataFrame(
            {
                "Detected_Regime_State": [2],
                "Detected_Regime_Label": ["Panic / Crisis (2)"],
                "HMM_State": [0],
                "Regime_Label": ["Raw state 0"],
            },
            index=pd.to_datetime(["2025-01-03"]),
        )
        lagged = plots._build_backtest_final_overlay_inputs(
            feature_df,
            ["2025-01-06"],
        )
        self.assertEqual(
            lagged.by_entry_session["2025-01-06"].state,
            2,
        )
        self.assertEqual(
            lagged.by_entry_session[
                "2025-01-06"
            ].namespace,
            FINAL_RISK_NAMESPACE,
        )

    def test_backtest_plot_cli_passes_typed_external_regime_contract(self):
        tree = ast.parse(inspect.getsource(plots))
        calls = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_put_credit_spread_backtest"
            )
        ]
        self.assertEqual(len(calls), 1)
        keywords = {
            keyword.arg
            for keyword in calls[0].keywords
        }
        self.assertIn("regime_protocol", keywords)
        self.assertIn("lagged_final_regimes", keywords)
        self.assertNotIn("regimes", keywords)
        self.assertNotIn("regime_labels", keywords)

    def test_every_expanding_plot_call_declares_fit_end(self):
        tree = ast.parse(inspect.getsource(plots))
        expanding_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Name)
                and node.func.id == "train_regime_hmm"
            ):
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            expanding = keywords.get("expanding_window")
            if (
                isinstance(expanding, ast.Constant)
                and expanding.value is True
            ):
                expanding_calls.append(keywords)

        self.assertGreaterEqual(len(expanding_calls), 3)
        self.assertTrue(
            all("fit_end" in keywords for keywords in expanding_calls)
        )


class StandaloneRegimeReviewCausalityTests(unittest.TestCase):
    def test_review_trace_uses_pretest_fit_end_and_records_ranges(self):
        window = regime_review.RegimeReviewWindow(
            calibration_start="2024-01-02",
            calibration_end="2024-12-31",
            test_start="2025-01-02",
            test_end="2025-01-06",
        )
        frame = pd.DataFrame(
            {"feature": np.arange(5, dtype=float)},
            index=pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-12-31",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
        )
        causal_trace = pd.DataFrame(
            {
                "HMM_State": [0, 1, 0, 1],
                "Regime_Label": ["quiet", "volatile", "quiet", "volatile"],
                "prob_state_0": [0.8, 0.2, 0.7, 0.3],
                "prob_state_1": [0.2, 0.8, 0.3, 0.7],
            },
            index=pd.to_datetime(
                [
                    "2024-12-31",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                ]
            ),
        )
        model = SimpleNamespace(n_components=2)

        with patch.object(
            regime_review,
            "train_regime_hmm",
            return_value=(model, 2, causal_trace),
        ) as train:
            returned_model, state_count, test_trace = (
                regime_review._train_causal_review_trace(
                    frame,
                    window,
                    n_components=2,
                )
            )

        self.assertIs(returned_model, model)
        self.assertEqual(state_count, 2)
        train.assert_called_once_with(
            frame,
            n_components=2,
            expanding_window=True,
            fit_end="2024-12-31",
        )
        self.assertEqual(
            test_trace.index.tolist(),
            pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06"]
            ).tolist(),
        )
        self.assertEqual(
            test_trace.attrs["regime_review_manifest"],
            window.manifest(),
        )
        self.assertLess(
            pd.Timestamp(window.calibration_end),
            pd.Timestamp(window.test_start),
        )


class LiveRegimeCausalityTests(unittest.TestCase):
    def test_open_session_uses_only_the_prior_completed_session(self):
        monday_open = datetime(
            2025,
            1,
            6,
            11,
            0,
            tzinfo=ZoneInfo("America/New_York"),
        )
        self.assertEqual(
            live_agent.latest_completed_nyse_session(monday_open),
            "2025-01-03",
        )
        with self.assertRaisesRegex(
            ProbabilityEngineUnavailable,
            "TIMEZONE_AWARE",
        ):
            live_agent.latest_completed_nyse_session(
                datetime(2025, 1, 6, 11, 0)
            )

    def test_live_builder_passes_completed_session_as_explicit_as_of(self):
        sentinel = (object(), object(), [])
        with (
            patch.object(
                live_agent,
                "latest_completed_nyse_session",
                return_value="2025-01-03",
            ),
            patch.object(
                live_agent,
                "build_regime_return_arrays",
                return_value=sentinel,
            ) as build,
        ):
            result = live_agent._build_live_regime_return_arrays(
                horizon=7,
                n_components=2,
            )

        self.assertIs(result, sentinel)
        build.assert_called_once_with(
            datetime(2025, 1, 3).date().toordinal(),
            horizon=7,
            n_components=2,
            as_of_date="2025-01-03",
        )

    def test_dashboard_status_scores_only_the_completed_session(self):
        history = pd.DataFrame(
            {
                "SPY_Close": [100.0, 101.0],
                "VIX_Close": [20.0, 21.0],
            },
            index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
        )
        scored = pd.DataFrame(
            {"HMM_State": [0]},
            index=pd.to_datetime(["2025-01-03"]),
        )
        sentinel = {"status": "completed-session-only"}
        with (
            patch.object(
                live_agent,
                "spy_regime_cache",
                {"timestamp": 0.0, "data": None},
            ),
            patch.object(
                live_agent,
                "latest_completed_nyse_session",
                return_value="2025-01-03",
            ),
            patch.object(
                live_agent,
                "fetch_historical_data",
                return_value=history,
            ),
            patch.object(
                live_agent,
                "_build_causal_regime_feature_frame",
                return_value=scored,
            ) as build,
            patch.object(
                live_agent,
                "_format_regime_row",
                return_value=sentinel,
            ),
        ):
            result = live_agent.calculate_spy_regime_status(object())

        self.assertIs(result, sentinel)
        causal_frame, _model, as_of = build.call_args.args
        self.assertEqual(
            causal_frame.index.strftime("%Y-%m-%d").tolist(),
            ["2025-01-03"],
        )
        self.assertEqual(as_of, pd.Timestamp("2025-01-03"))

    def test_standalone_review_passes_completed_session_as_explicit_as_of(
        self,
    ):
        sentinel = (object(), object(), [])
        with (
            patch.object(
                regime_review,
                "latest_completed_nyse_session",
                return_value="2025-01-03",
            ),
            patch.object(
                regime_review,
                "build_regime_return_arrays",
                return_value=sentinel,
            ) as build,
        ):
            result = (
                regime_review._build_completed_session_return_arrays(
                    horizon=7,
                    force_refit=True,
                )
            )

        self.assertIs(result, sentinel)
        build.assert_called_once_with(
            datetime(2025, 1, 3).date().toordinal(),
            horizon=7,
            force_refit=True,
            as_of_date="2025-01-03",
        )


if __name__ == "__main__":
    unittest.main()
