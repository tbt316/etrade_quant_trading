import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pandas as pd

import backtesting.backtest_runner as backtest_module
from backtesting.backtest_runner import (
    SpreadTrade,
    _forward_return_assignment_strike,
    lag_daily_regime_map,
    validate_regime_v2_annotations,
)
from live_trading.regime_detector_v2 import DETECTOR_VERSION, SIGNAL_TIMESTAMP
from live_trading.regime_signal import from_calibrated_v2_trace


def _annotations():
    trace = pd.DataFrame(
        {
            "Background_State": ["elevated"],
            "Shock_State": ["active"],
            "Signal_Available_At": ["2025-04-03T20:15:00Z"],
            "Tradable_Session": ["2025-04-04"],
            "Detector_Version": [DETECTOR_VERSION],
            "Config_Hash": ["a" * 64],
            "Regime_Signal_Timestamp": [SIGNAL_TIMESTAMP],
            "Reason_Codes": ["vix_daily_change_extreme"],
            "Data_Quality": [
                "exchange_sessions_and_source_times_valid"
            ],
            "Execution_Eligible": [False],
        },
        index=pd.DatetimeIndex(["2025-04-03"]),
    )
    trace.attrs["detector_version"] = DETECTOR_VERSION
    trace.attrs["config_hash"] = "a" * 64
    return from_calibrated_v2_trace(trace)


class RegimeBacktestParityTests(unittest.TestCase):
    def test_v2_annotations_are_typed_exact_and_do_not_change_legacy_map(self):
        legacy = {"2025-04-03": 2}
        sessions = ["2025-04-03", "2025-04-04", "2025-04-07"]
        before = lag_daily_regime_map(legacy, sessions)

        annotations = validate_regime_v2_annotations(_annotations())

        self.assertEqual(lag_daily_regime_map(legacy, sessions), before)
        self.assertEqual(set(annotations), {"2025-04-04"})
        self.assertEqual(
            annotations["2025-04-04"].composite_label,
            "elevated + active",
        )
        self.assertFalse(
            annotations["2025-04-04"].may_authorize_execution
        )
        with self.assertRaises(TypeError):
            validate_regime_v2_annotations({"2025-04-04": 2})
        with self.assertRaises(ValueError):
            validate_regime_v2_annotations(
                {"2025-04-07": annotations["2025-04-04"]}
            )

    def test_trade_audit_preserves_v2_without_numeric_coercion(self):
        signal = _annotations()["2025-04-04"]
        payload = signal.to_envelope()
        trade = SpreadTrade(
            entry_date="2025-04-04",
            entry_regime=2,
            entry_regime_name="Panic / Crisis",
            entry_regime_v2=payload,
        )

        serialized = asdict(trade)
        self.assertEqual(serialized["entry_regime"], 2)
        self.assertEqual(
            serialized["entry_regime_v2"]["signal"]["background_state"],
            "elevated",
        )
        self.assertEqual(
            serialized["entry_regime_v2"]["signal"]["shock_state"],
            "active",
        )
        self.assertFalse(
            serialized["entry_regime_v2"]["signal"][
                "may_authorize_execution"
            ]
        )
        self.assertEqual(
            serialized["entry_regime_v2"]["signal_sha256"],
            signal.signal_sha256,
        )

    def test_forward_bucket_requires_entry_date_and_prior_resolution(self):
        dates = pd.bdate_range("2025-01-02", periods=40)
        prices = pd.Series(
            100.0 * np.exp(np.linspace(0.0, 0.12, len(dates))),
            index=[item.date().isoformat() for item in dates],
        )
        entry = dates[30].date().isoformat()

        strike, diagnostics = _forward_return_assignment_strike(
            underlying_prices=prices,
            trading_dates=list(prices.index),
            trading_date_index={
                item: position for position, item in enumerate(prices.index)
            },
            as_of_date=entry,
            current_spot=float(prices.loc[entry]),
            option_horizon_days=7,
            target_assignment_prob=0.05,
        )

        self.assertIsNotNone(strike)
        self.assertTrue(
            diagnostics["outcomes_resolved_strictly_before_entry"]
        )
        self.assertLess(
            diagnostics["latest_resolution_session"],
            entry,
        )

        missing, reason = _forward_return_assignment_strike(
            underlying_prices=prices,
            trading_dates=list(prices.index),
            trading_date_index={},
            as_of_date="2025-12-31",
            current_spot=100.0,
            option_horizon_days=7,
            target_assignment_prob=0.05,
        )
        self.assertIsNone(missing)
        self.assertEqual(
            reason["reason"],
            "as_of_date_missing_from_underlying_prices",
        )


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


class RegimeBacktestRunnerParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_actions_are_identical_with_shadow_annotation(self):
        index = pd.bdate_range("2011-05-03", "2025-04-04")
        underlying = pd.Series(
            np.linspace(100.0, 500.0, len(index)),
            index=index.strftime("%Y-%m-%d"),
        )
        vix = pd.Series(
            [18.0, 18.5],
            index=["2025-04-03", "2025-04-04"],
        )

        async def run(regime_v2_annotations=None):
            with (
                patch.object(
                    backtest_module,
                    "OptionDataCache",
                    _NoDataCache,
                ),
                patch.object(
                    backtest_module,
                    "MassiveAPIClient",
                    _NoContractsClient,
                ),
                patch.object(
                    backtest_module,
                    "_fetch_underlying_prices",
                    return_value=pd.Series(
                        [5.0],
                        index=["2025-04-04"],
                    ),
                ),
            ):
                return await backtest_module.run_put_credit_spread_backtest(
                    start_date="2025-04-04",
                    end_date="2025-04-04",
                    underlying_prices=underlying,
                    vix_prices=vix,
                    regime_v2_annotations=regime_v2_annotations,
                    offline_only=True,
                )

        baseline = await run()
        annotated = await run(_annotations())

        for field_name in (
            "trades",
            "total_pnl",
            "capital_history",
            "cash_history",
            "nvl_history",
            "margin_history",
            "regime_history",
        ):
            self.assertEqual(
                getattr(annotated, field_name),
                getattr(baseline, field_name),
            )
        self.assertEqual(baseline.regime_v2_history, [("2025-04-04", None)])
        envelope = annotated.regime_v2_history[0][1]
        self.assertEqual(
            envelope["signal"]["background_state"],
            "elevated",
        )
        self.assertEqual(
            envelope["signal"]["shock_state"],
            "active",
        )
        self.assertIn("signal_sha256", envelope)


if __name__ == "__main__":
    unittest.main()
