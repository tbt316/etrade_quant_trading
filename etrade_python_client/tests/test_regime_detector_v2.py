import unittest

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from live_trading.regime_detector_v2 import (
    BACKGROUND_CALM,
    BACKGROUND_STRESS,
    BACKGROUND_UNAVAILABLE,
    DETECTOR_VERSION,
    RegimeDetectorConfig,
    SHOCK_ACTIVE,
    SHOCK_AFTERSHOCK,
    SHOCK_NONE,
    SHOCK_UNAVAILABLE,
    SIGNAL_TIMESTAMP,
    detect_regimes,
)

NYSE = mcal.get_calendar("NYSE")


def _config(**overrides):
    values = {
        "calibration_window": 80,
        "min_calibration_history": 40,
        "vix_slow_window": 5,
        "realized_vol_window": 5,
        "drawdown_window": 10,
    }
    values.update(overrides)
    return RegimeDetectorConfig(**values)


def _calm_prices(rows=120):
    schedule = NYSE.schedule(start_date="2025-01-02", end_date="2026-12-31")
    index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()[:rows]
    returns = np.tile([0.0010, -0.0005, 0.0008, -0.0003, 0.0006], rows // 5 + 1)[:rows]
    spy = 500.0 * np.exp(np.cumsum(returns))
    vix = np.tile([15.8, 16.0, 15.9, 16.1, 15.7], rows // 5 + 1)[:rows]
    return pd.DataFrame({"SPY_Close": spy, "VIX_Close": vix}, index=index)


def _context(prices):
    schedule = NYSE.schedule(start_date=prices.index.min(), end_date=prices.index.max())
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    spy_available_at = pd.Series(
        pd.DatetimeIndex(schedule["market_close"]) + pd.Timedelta(minutes=1),
        index=sessions,
    )
    vix_available_at = pd.Series(
        (
            sessions.tz_localize("America/New_York")
            + pd.Timedelta(hours=16, minutes=16)
        ).tz_convert("UTC"),
        index=sessions,
    )
    signal_available_at = max(
        spy_available_at.iloc[-1],
        vix_available_at.iloc[-1],
    )
    return {
        "spy_available_at": spy_available_at,
        "vix_available_at": vix_available_at,
        "as_of": signal_available_at + pd.Timedelta(minutes=1),
        "source_provenance_verified": True,
    }


def _detect(prices, config=None):
    return detect_regimes(prices, config or _config(), **_context(prices))


class RegimeDetectorV2Tests(unittest.TestCase):
    def test_isolated_vix_jump_is_shock_not_persistent_stress(self):
        prices = _calm_prices()
        shock_date = prices.index[-5]
        prices.loc[shock_date, "VIX_Close"] = prices["VIX_Close"].shift(1).loc[shock_date] * 1.12

        result = _detect(prices)

        self.assertEqual(result.loc[shock_date, "Shock_State"], SHOCK_ACTIVE)
        self.assertNotEqual(result.loc[shock_date, "Background_State"], BACKGROUND_STRESS)

    def test_persistent_high_volatility_enters_stress_after_confirmation(self):
        prices = _calm_prices(150)
        stress_dates = prices.index[-20:]
        stress_returns = np.tile([-0.018, 0.008], 10)
        start_spy = prices.loc[prices.index[-21], "SPY_Close"]
        prices.loc[stress_dates, "SPY_Close"] = start_spy * np.exp(np.cumsum(stress_returns))
        prices.loc[stress_dates, "VIX_Close"] = np.linspace(24.0, 30.0, len(stress_dates))

        result = _detect(prices)

        self.assertEqual(result.loc[stress_dates[-1], "Background_State"], BACKGROUND_STRESS)
        self.assertGreaterEqual(result.loc[stress_dates[-1], "Background_Regime_Age"], 1)
        self.assertTrue(result.loc[stress_dates[-1], "Absolute_Stress_Evidence"])

    def test_aftershock_state_decays_without_rewriting_background(self):
        prices = _calm_prices()
        shock_position = len(prices) - 5
        shock_date = prices.index[shock_position]
        prices.iloc[shock_position, prices.columns.get_loc("VIX_Close")] = (
            prices.iloc[shock_position - 1]["VIX_Close"] * 1.12
        )

        result = _detect(prices, _config(aftershock_days=2))

        self.assertEqual(result.loc[shock_date, "Shock_State"], SHOCK_ACTIVE)
        self.assertEqual(result.iloc[shock_position + 1]["Shock_State"], SHOCK_AFTERSHOCK)
        self.assertEqual(result.iloc[shock_position + 2]["Shock_State"], SHOCK_AFTERSHOCK)
        self.assertEqual(result.iloc[shock_position + 3]["Shock_State"], SHOCK_NONE)

    def test_appending_future_rows_does_not_change_prior_outputs(self):
        prices = _calm_prices(150)
        prefix = prices.iloc[:120]
        future_dates = prices.index[120:]
        prices.loc[future_dates, "VIX_Close"] = np.linspace(25.0, 45.0, len(future_dates))
        future_returns = np.tile([-0.025, 0.010], len(future_dates) // 2 + 1)[: len(future_dates)]
        start_spy = prices.loc[prices.index[119], "SPY_Close"]
        prices.loc[future_dates, "SPY_Close"] = start_spy * np.exp(np.cumsum(future_returns))

        prefix_result = _detect(prefix)
        full_result = _detect(prices).loc[prefix.index]
        prefix_result.attrs = {}
        full_result.attrs = {}

        pd.testing.assert_frame_equal(prefix_result, full_result)

    def test_invalid_required_data_fails_closed(self):
        prices = _calm_prices()
        prices.iloc[-1, prices.columns.get_loc("VIX_Close")] = np.nan

        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            detect_regimes(prices, _config(), **_context(prices))

    def test_empty_and_infinite_data_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            detect_regimes(
                _calm_prices().iloc[:0],
                _config(),
                as_of=pd.Timestamp("2025-01-03T22:00:00Z"),
                spy_available_at=pd.Series(dtype="object"),
                vix_available_at=pd.Series(dtype="object"),
                source_provenance_verified=True,
            )

        prices = _calm_prices()
        prices.iloc[-1, prices.columns.get_loc("SPY_Close")] = np.inf
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            detect_regimes(prices, _config(), **_context(prices))

    def test_weekend_row_fails_closed(self):
        prices = _calm_prices()
        weekend = prices.iloc[[-1]].copy()
        weekend_date = prices.index.max() + pd.Timedelta(days=1)
        while weekend_date.weekday() != 5:
            weekend_date += pd.Timedelta(days=1)
        weekend.index = pd.DatetimeIndex([weekend_date])
        prices = pd.concat([prices, weekend])

        with self.assertRaisesRegex(ValueError, "contiguous NYSE sessions"):
            detect_regimes(
                prices,
                _config(),
                as_of=weekend_date.tz_localize("UTC") + pd.Timedelta(days=2),
                spy_available_at=weekend_date.tz_localize("UTC"),
                vix_available_at=weekend_date.tz_localize("UTC"),
                source_provenance_verified=True,
            )

    def test_missing_interior_nyse_session_fails_closed(self):
        prices = _calm_prices()
        incomplete = prices.drop(prices.index[-10])

        with self.assertRaisesRegex(ValueError, "contiguous NYSE sessions"):
            detect_regimes(incomplete, _config(), **_context(prices))

    def test_signal_timestamp_is_explicit(self):
        prices = _calm_prices()
        result = _detect(prices)

        self.assertTrue(result["Regime_Signal_Timestamp"].eq(SIGNAL_TIMESTAMP).all())
        self.assertEqual(result.attrs["regime_signal_timestamp"], SIGNAL_TIMESTAMP)
        self.assertEqual(result.attrs["detector_version"], DETECTOR_VERSION)
        self.assertEqual(len(result.attrs["config_hash"]), 64)
        self.assertTrue(result.attrs["freshness_assessed"])
        self.assertEqual(
            result.attrs["latest_jointly_finalized_session"],
            prices.index.max().date().isoformat(),
        )

    def test_first_shock_state_is_unavailable_not_none(self):
        result = _detect(_calm_prices())

        self.assertEqual(result.iloc[0]["Shock_State"], SHOCK_UNAVAILABLE)
        self.assertTrue(pd.isna(result.iloc[0]["Shock_Active"]))

    def test_absolute_stress_evidence_is_never_labeled_calm(self):
        config = _config(
            stress_vix_floor=15.0,
            stress_realized_vol_floor=1.0,
            stress_drawdown_floor=1.0,
        )
        result = _detect(_calm_prices(150), config)
        valid = result["Background_State"] != BACKGROUND_UNAVAILABLE
        contradictory = (
            valid
            & result["Absolute_Stress_Evidence"]
            & (result["Background_State"] == BACKGROUND_CALM)
        )

        self.assertFalse(contradictory.any())

    def test_stale_latest_session_fails_closed(self):
        prices = _calm_prices()
        context = _context(prices)
        stale_context = context.copy()
        stale_context["spy_available_at"] = context["spy_available_at"].iloc[:-1]
        stale_context["vix_available_at"] = context["vix_available_at"].iloc[:-1]

        with self.assertRaisesRegex(ValueError, "latest jointly finalized SPY/VIX session"):
            detect_regimes(prices.iloc[:-1], _config(), **stale_context)

    def test_source_timestamps_before_finalization_fail_closed(self):
        prices = _calm_prices()
        context = _context(prices)
        premature_spy = context["spy_available_at"].copy()
        premature_spy.iloc[-1] -= pd.Timedelta(minutes=2)

        with self.assertRaisesRegex(ValueError, "before the official NYSE close"):
            detect_regimes(
                prices,
                _config(),
                as_of=context["as_of"],
                spy_available_at=premature_spy,
                vix_available_at=context["vix_available_at"],
                source_provenance_verified=True,
            )

        premature_vix = context["vix_available_at"].copy()
        premature_vix.iloc[-1] -= pd.Timedelta(minutes=2)
        with self.assertRaisesRegex(ValueError, "before the Cboe 4:15 p.m. ET cutoff"):
            detect_regimes(
                prices,
                _config(),
                as_of=context["as_of"],
                spy_available_at=context["spy_available_at"],
                vix_available_at=premature_vix,
                source_provenance_verified=True,
            )

    def test_latest_session_is_not_available_at_401_pm_eastern(self):
        prices = _calm_prices()
        latest = prices.index.max()
        schedule = NYSE.schedule(start_date=latest, end_date=latest)
        market_close = pd.Timestamp(schedule.iloc[-1]["market_close"])
        as_of = market_close + pd.Timedelta(minutes=1)
        context = _context(prices)
        spy_available_at = context["spy_available_at"].copy()
        vix_available_at = context["vix_available_at"].copy()
        spy_available_at.iloc[-1] = market_close + pd.Timedelta(seconds=30)
        vix_available_at.iloc[-1] = as_of

        with self.assertRaisesRegex(ValueError, "latest jointly finalized SPY/VIX session"):
            detect_regimes(
                prices,
                _config(),
                as_of=as_of,
                spy_available_at=spy_available_at,
                vix_available_at=vix_available_at,
                source_provenance_verified=True,
            )

    def test_unverified_source_provenance_is_explicit(self):
        prices = _calm_prices()
        context = _context(prices)
        context["source_provenance_verified"] = False

        result = detect_regimes(prices, _config(), **context)

        self.assertFalse(result.attrs["freshness_assessed"])
        self.assertFalse(result.attrs["source_provenance_verified"])
        self.assertTrue(
            result["Data_Quality"].str.contains("source_provenance_unverified").all()
        )

    def test_delayed_source_rolls_tradable_session_past_open(self):
        prices = _calm_prices()
        context = _context(prices)
        latest = prices.index.max()
        following = NYSE.schedule(
            start_date=latest + pd.Timedelta(days=1),
            end_date=latest + pd.Timedelta(days=14),
        )
        next_open = pd.Timestamp(following.iloc[0]["market_open"])
        delayed_arrival = next_open + pd.Timedelta(hours=2)
        spy_available_at = context["spy_available_at"].copy()
        vix_available_at = context["vix_available_at"].copy()
        spy_available_at.iloc[-1] = delayed_arrival
        vix_available_at.iloc[-1] = delayed_arrival

        result = detect_regimes(
            prices,
            _config(),
            as_of=delayed_arrival + pd.Timedelta(minutes=1),
            spy_available_at=spy_available_at,
            vix_available_at=vix_available_at,
            source_provenance_verified=True,
        )

        expected = pd.Timestamp(following.index[1]).tz_localize(None).normalize()
        self.assertEqual(result.iloc[-1]["Signal_Available_At"], delayed_arrival)
        self.assertEqual(result.iloc[-1]["Tradable_Session"], expected)

    def test_later_signal_inherits_delayed_historical_dependency(self):
        prices = _calm_prices()
        context = _context(prices)
        latest = prices.index.max()
        following = NYSE.schedule(
            start_date=latest + pd.Timedelta(days=1),
            end_date=latest + pd.Timedelta(days=14),
        )
        next_open = pd.Timestamp(following.iloc[0]["market_open"])
        delayed_arrival = next_open + pd.Timedelta(hours=2)
        spy_available_at = context["spy_available_at"].copy()
        vix_available_at = context["vix_available_at"].copy()
        spy_available_at.iloc[-2] = delayed_arrival
        vix_available_at.iloc[-2] = delayed_arrival

        result = detect_regimes(
            prices,
            _config(),
            as_of=delayed_arrival + pd.Timedelta(minutes=1),
            spy_available_at=spy_available_at,
            vix_available_at=vix_available_at,
            source_provenance_verified=True,
        )

        expected = pd.Timestamp(following.index[1]).tz_localize(None).normalize()
        self.assertEqual(result.iloc[-1]["Signal_Available_At"], delayed_arrival)
        self.assertEqual(result.iloc[-1]["Tradable_Session"], expected)

    def test_tradable_session_is_exact_next_nyse_session(self):
        prices = _calm_prices()
        result = _detect(prices)
        latest = prices.index.max()
        following = NYSE.schedule(
            start_date=latest + pd.Timedelta(days=1),
            end_date=latest + pd.Timedelta(days=14),
        )
        expected = pd.Timestamp(following.index[0]).tz_localize(None).normalize()

        self.assertEqual(result.iloc[-1]["Tradable_Session"], expected)
        self.assertIsNotNone(pd.Timestamp(result.iloc[-1]["Market_Close_At"]).tzinfo)
        self.assertIsNotNone(pd.Timestamp(result.iloc[-1]["VIX_Finalization_At"]).tzinfo)


if __name__ == "__main__":
    unittest.main()
