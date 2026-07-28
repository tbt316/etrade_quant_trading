import asyncio
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar

import live_trading.ev_engine as engine_module
from live_trading.data_ingestion import (
    CausalFeatureManifest,
    DataIngestor,
    RollingRobustScaler,
    StationarityDecision,
)
from live_trading.ev_engine import _select_pca_dimension, train_regime_hmm


def _nyse_index(start, rows):
    schedule = get_calendar("NYSE").schedule(
        start_date=start,
        end_date=(
            pd.Timestamp(start) + pd.Timedelta(days=rows * 2)
        ).date(),
    )
    return pd.DatetimeIndex(schedule.index[:rows]).tz_localize(None)


def _feature_frame(rows=180):
    rng = np.random.default_rng(20260727)
    index = _nyse_index("2025-01-02", rows)
    innovations = rng.normal(0.0004, 0.01, rows)
    price = 100.0 * np.exp(np.cumsum(innovations))
    stationary = np.empty(rows)
    stationary[0] = rng.normal()
    for i in range(1, rows):
        stationary[i] = 0.45 * stationary[i - 1] + rng.normal(scale=0.7)
    future_only = np.full(rows, np.nan)
    if rows > 120:
        future_only[120:] = rng.normal(size=rows - 120)
    return pd.DataFrame(
        {
            "SPY_Close": price,
            "Stationary_Signal": stationary,
            "Future_Only": future_only,
        },
        index=index,
    )


def _pure_ingestor():
    """Construct the pure preprocessing surface without opening cache state."""
    return DataIngestor.__new__(DataIngestor)


class CausalFeaturePipelineTests(unittest.TestCase):
    def test_rolling_scaler_prefix_is_invariant_to_appended_rows(self):
        frame = _feature_frame()[["SPY_Close", "Stationary_Signal"]]
        scaler = RollingRobustScaler(window=60)
        prefix = scaler.batch_rolling_transform(
            frame.iloc[:100],
            min_periods=20,
        )
        full = scaler.batch_rolling_transform(
            frame,
            min_periods=20,
        ).loc[prefix.index]
        pd.testing.assert_frame_equal(prefix, full)

    def test_selection_fractional_d_and_values_are_prefix_invariant(self):
        frame = _feature_frame()
        fit_end = frame.index[99]
        prefix = _pure_ingestor().prepare_causal_features(
            frame.iloc[:140],
            fit_end=fit_end,
            scaler_window=60,
        )
        full = _pure_ingestor().prepare_causal_features(
            frame,
            fit_end=fit_end,
            scaler_window=60,
        )

        self.assertEqual(prefix.manifest, full.manifest)
        self.assertNotIn("Future_Only", prefix.manifest.feature_names)
        pd.testing.assert_frame_equal(
            prefix.stationary,
            full.stationary.loc[prefix.stationary.index],
        )
        pd.testing.assert_frame_equal(
            prefix.scaled,
            full.scaled.loc[prefix.scaled.index],
        )

    def test_required_future_only_feature_fails_closed(self):
        frame = _feature_frame()
        with self.assertRaisesRegex(ValueError, "REQUIRED_FEATURES_UNFIT"):
            _pure_ingestor().prepare_causal_features(
                frame,
                fit_end=frame.index[99],
                required_columns=("SPY_Close", "Future_Only"),
            )

    def test_manifest_binds_exact_selected_training_prefix_values(self):
        frame = _feature_frame()
        fit_end = frame.index[99]
        original = _pure_ingestor().prepare_causal_features(
            frame,
            fit_end=fit_end,
            scaler_window=60,
        )
        revised = frame.copy()
        revised.loc[frame.index[50], "SPY_Close"] *= 1.01
        changed = _pure_ingestor().prepare_causal_features(
            revised,
            fit_end=fit_end,
            scaler_window=60,
        )

        self.assertNotEqual(
            original.manifest.training_data_sha256,
            changed.manifest.training_data_sha256,
        )
        self.assertNotEqual(
            original.manifest.feature_hash,
            changed.manifest.feature_hash,
        )

    def test_weekend_and_holiday_rows_do_not_change_modeling_prefix(self):
        frame = _feature_frame(90)
        fit_end = frame.index[69]
        extra = pd.DataFrame(
            {
                "SPY_Close": [9999.0, 8888.0],
                "Stationary_Signal": [99.0, -99.0],
                "Future_Only": [np.nan, np.nan],
            },
            index=pd.to_datetime(["2025-01-18", "2025-01-20"]),
        )
        contaminated = pd.concat([frame, extra]).sort_index()

        clean = _pure_ingestor().prepare_causal_features(
            frame,
            fit_end=fit_end,
            scaler_window=60,
        )
        filtered = _pure_ingestor().prepare_causal_features(
            contaminated,
            fit_end=fit_end,
            scaler_window=60,
        )

        pd.testing.assert_frame_equal(
            clean.stationary,
            filtered.stationary,
        )
        pd.testing.assert_frame_equal(clean.scaled, filtered.scaled)
        self.assertEqual(
            clean.manifest.modeling_session_data_sha256,
            filtered.manifest.modeling_session_data_sha256,
        )
        self.assertEqual(
            clean.manifest.feature_hash,
            filtered.manifest.feature_hash,
        )
        self.assertNotEqual(
            clean.manifest.source_training_data_sha256,
            filtered.manifest.source_training_data_sha256,
        )
        self.assertEqual(clean.manifest.modeling_calendar, "NYSE")

    def test_pca_dimension_selection_uses_only_fit_prefix(self):
        rng = np.random.default_rng(11)
        index = _nyse_index("2025-01-02", 160)
        prefix_values = rng.normal(size=(100, 4))
        future_values = rng.normal(scale=(1.0, 20.0, 40.0, 80.0), size=(60, 4))
        full = pd.DataFrame(
            np.vstack([prefix_values, future_values]),
            index=index,
            columns=["a", "b", "c", "d"],
        )
        fit_end = index[99]
        selected_prefix = _select_pca_dimension(
            full.iloc[:120],
            fit_end=fit_end,
        )
        selected_full = _select_pca_dimension(
            full,
            fit_end=fit_end,
        )
        self.assertEqual(selected_prefix, selected_full)

    def test_fit_cutoff_is_mandatory_before_data_acquisition(self):
        ingestor = _pure_ingestor()
        with self.assertRaisesRegex(
            ValueError,
            "EXPLICIT_FEATURE_FIT_END_REQUIRED",
        ):
            asyncio.run(
                ingestor.build_fused_dataset(
                    "2025-01-01",
                    "2025-12-31",
                    wait=False,
                    scale=False,
                )
            )

    def test_walk_forward_hmm_requires_explicit_feature_fit_cutoff(self):
        frame = _feature_frame(40)
        with patch.object(
            engine_module,
            "DataIngestor",
            side_effect=AssertionError("must fail before acquisition"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "EXPLICIT_FEATURE_FIT_END_REQUIRED",
            ):
                train_regime_hmm(
                    frame,
                    n_components=3,
                    expanding_window=True,
                )

    def test_fixed_oos_states_and_model_inputs_are_prefix_invariant(self):
        rng = np.random.default_rng(91)
        index = _nyse_index("2024-01-02", 150)
        state = np.repeat([0.0, 3.0], 75)
        stationary = pd.DataFrame(
            {
                "SPY_Log_Return": rng.normal(state * -0.003, 0.008),
                "VIX_Close": rng.normal(15.0 + state * 4.0, 0.8),
                "Stationary_Signal": rng.normal(state, 0.4),
            },
            index=index,
        )
        manifest = CausalFeatureManifest(
            fit_end=index[99].strftime("%Y-%m-%d"),
            feature_names=tuple(stationary.columns),
            stationarity=tuple(
                StationarityDecision(column, 0.0, 0.01)
                for column in stationary.columns
            ),
            missing_ratio_limit=0.80,
            scaler_window=1260,
            scaler_min_periods=20,
            availability="close_T_for_next_session",
            source_training_data_sha256="a" * 64,
            training_data_sha256="a" * 64,
            modeling_calendar="NYSE",
            modeling_session_data_sha256="a" * 64,
            feature_hash="b" * 64,
        )

        class _OfflineIngestor:
            def __init__(self):
                self.rolling_scaler = RollingRobustScaler()

            async def build_fused_dataset(self, start, end, **_kwargs):
                result = stationary.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
                result.attrs["causal_feature_manifest"] = manifest
                return result

            def scale_features(self, frame, rolling=False, window=1260, **_kwargs):
                if not rolling:
                    raise AssertionError("test expects causal rolling scaling")
                self.rolling_scaler = RollingRobustScaler(window=window)
                return self.rolling_scaler.batch_rolling_transform(
                    frame,
                    min_periods=20,
                )

        raw = pd.DataFrame(
            {
                "SPY_Close": 100.0 * np.exp(np.cumsum(stationary["SPY_Log_Return"])),
                "VIX_Close": stationary["VIX_Close"],
            },
            index=index,
        )
        fit_end = index[99]
        with patch.object(engine_module, "DataIngestor", _OfflineIngestor):
            prefix_model, _, prefix_trace = train_regime_hmm(
                raw.iloc[:130],
                n_components=2,
                expanding_window=False,
                pca_components=2,
                fit_end=fit_end,
            )
            full_model, _, full_trace = train_regime_hmm(
                raw,
                n_components=2,
                expanding_window=False,
                pca_components=2,
                fit_end=fit_end,
            )

        compare_columns = [
            "PC1",
            "PC2",
            "HMM_State",
            "Raw_HMM_Taxonomy_ID",
            "prob_state_0",
            "prob_state_1",
        ]
        pd.testing.assert_frame_equal(
            prefix_trace[compare_columns],
            full_trace.loc[prefix_trace.index, compare_columns],
        )
        self.assertEqual(
            prefix_trace.attrs["raw_hmm_taxonomy_scope"],
            "fixed_model",
        )
        self.assertEqual(
            prefix_trace["Raw_HMM_Taxonomy_ID"].nunique(),
            1,
        )

        class _ReplayIngestor(_OfflineIngestor):
            def scale_features(self, *_args, **_kwargs):
                raise AssertionError(
                    "cached fixed model must use its frozen scaler"
                )

        with patch.object(engine_module, "DataIngestor", _ReplayIngestor):
            replay_prefix = engine_module._build_causal_regime_feature_frame(
                raw.iloc[:130],
                prefix_model,
                raw.index[129],
                refit_date=fit_end,
            )
            replay_full = engine_module._build_causal_regime_feature_frame(
                raw,
                full_model,
                raw.index[-1],
                refit_date=fit_end,
            )
        pd.testing.assert_frame_equal(
            replay_prefix[compare_columns],
            replay_full.loc[replay_prefix.index, compare_columns],
        )

        with patch.object(engine_module, "DataIngestor", _OfflineIngestor):
            _, _, walk_prefix = train_regime_hmm(
                raw.iloc[:130],
                n_components=2,
                expanding_window=True,
                pca_components=2,
                fit_end=fit_end,
            )
            _, _, walk_full = train_regime_hmm(
                raw,
                n_components=2,
                expanding_window=True,
                pca_components=2,
                fit_end=fit_end,
            )
        pd.testing.assert_frame_equal(
            walk_prefix[compare_columns],
            walk_full.loc[walk_prefix.index, compare_columns],
        )
        self.assertEqual(
            walk_prefix.attrs["raw_hmm_taxonomy_scope"],
            "per_row_refit",
        )
        self.assertTrue(
            walk_prefix["Raw_HMM_Taxonomy_ID"].str.fullmatch(
                r"[0-9a-f]{64}"
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
