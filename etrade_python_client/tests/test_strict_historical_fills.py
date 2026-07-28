import ast
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backtesting.massive_api_client import (
    HistoricalFillEvidence,
    HistoricalFillEvidenceError,
    HistoricalFillSource,
    MassiveAPIClient,
    fill_bundle_diagnostics,
    validate_strict_nbbo_bundle,
)
from backtesting.option_data_cache import OptionDataCache
import backtesting.backtest_runner as backtest_runner


PRICING_DATE = "2025-01-02"
TICKERS = (
    "O:SPY250117P00500000",
    "O:SPY250117P00490000",
    "O:SPY250117P00480000",
)


def _observed(
    ticker: str,
    timestamp: str,
    *,
    bid: float = 1.0,
    ask: float = 1.2,
    pricing_date: str = PRICING_DATE,
) -> HistoricalFillEvidence:
    return HistoricalFillEvidence(
        option_ticker=ticker,
        pricing_date=pricing_date,
        source=HistoricalFillSource.OBSERVED_NBBO,
        bid=bid,
        ask=ask,
        mid=round((bid + ask) / 2.0, 4),
        event_timestamp_utc=timestamp,
        provider_route=f"/v3/quotes/{ticker}",
    )


class HistoricalFillEvidenceTests(unittest.TestCase):
    def test_observed_nbbo_rejects_missing_naive_nonfinite_and_crossed_values(self):
        invalid = (
            {"event_timestamp_utc": None},
            {"event_timestamp_utc": "2025-01-02T20:58:00"},
            {
                "event_timestamp_utc": "2025-01-02T20:58:00Z",
                "bid": math.nan,
                "ask": 1.2,
                "mid": math.nan,
            },
            {
                "event_timestamp_utc": "2025-01-02T20:58:00Z",
                "bid": 1.3,
                "ask": 1.2,
                "mid": 1.25,
            },
        )
        for overrides in invalid:
            values = {
                "option_ticker": TICKERS[0],
                "pricing_date": PRICING_DATE,
                "source": HistoricalFillSource.OBSERVED_NBBO,
                "bid": 1.0,
                "ask": 1.2,
                "mid": 1.1,
                "event_timestamp_utc": "2025-01-02T20:58:00Z",
                "provider_route": f"/v3/quotes/{TICKERS[0]}",
            }
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(HistoricalFillEvidenceError):
                    HistoricalFillEvidence(**values)

    def test_observed_nbbo_rejects_post_close_and_wrong_day_timestamps(self):
        for timestamp in (
            "2025-01-02T22:00:00Z",
            "2025-01-03T20:00:00Z",
        ):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(HistoricalFillEvidenceError):
                    _observed(TICKERS[0], timestamp)

    def test_strict_bundle_accepts_two_and_three_synchronized_legs(self):
        two_legs = (
            _observed(TICKERS[0], "2025-01-02T20:55:00Z"),
            _observed(TICKERS[1], "2025-01-02T20:59:59Z"),
        )
        self.assertEqual(
            validate_strict_nbbo_bundle(
                two_legs,
                TICKERS[:2],
                PRICING_DATE,
                5.0,
            ),
            two_legs,
        )

        three_legs = two_legs + (
            _observed(TICKERS[2], "2025-01-02T20:57:30Z"),
        )
        self.assertEqual(
            validate_strict_nbbo_bundle(
                three_legs,
                TICKERS,
                PRICING_DATE,
                5.0,
            ),
            three_legs,
        )
        diagnostics = fill_bundle_diagnostics(three_legs, 5.0)
        self.assertEqual(three_legs[0].bid_size, 0.0)
        self.assertEqual(three_legs[0].ask_size, 0.0)
        self.assertTrue(diagnostics["strict_nbbo_mark_validated"])
        self.assertFalse(diagnostics["strict_nbbo_authorized"])
        self.assertIn(
            "mark",
            diagnostics["strict_nbbo_authorized_deprecated"].lower(),
        )
        self.assertEqual(
            diagnostics["execution_assumption"],
            "HISTORICAL_MARK_NOT_EXECUTABLE_FILL",
        )
        self.assertTrue(diagnostics["temporally_synchronized"])
        self.assertTrue(diagnostics["all_observed_quotes_close_fresh"])

    def test_strict_bundle_rejects_pairwise_timestamp_skew_and_wrong_binding(self):
        skewed = (
            _observed(TICKERS[0], "2025-01-02T20:55:00Z"),
            _observed(TICKERS[1], "2025-01-02T20:57:01Z"),
        )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "configured delta",
        ):
            validate_strict_nbbo_bundle(
                skewed,
                TICKERS[:2],
                PRICING_DATE,
                2.0,
            )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "ticker mismatch",
        ):
            validate_strict_nbbo_bundle(
                tuple(reversed(skewed)),
                TICKERS[:2],
                PRICING_DATE,
                5.0,
            )

    def test_strict_bundle_rejects_in_session_quote_stale_at_close(self):
        stale = (
            _observed(TICKERS[0], "2025-01-02T14:31:00Z"),
            _observed(TICKERS[1], "2025-01-02T14:31:01Z"),
        )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "too old relative",
        ):
            validate_strict_nbbo_bundle(
                stale,
                TICKERS[:2],
                PRICING_DATE,
                5.0,
                5.0,
            )
        diagnostics = fill_bundle_diagnostics(stale, 5.0, 5.0)
        self.assertFalse(diagnostics["strict_nbbo_mark_validated"])
        self.assertFalse(diagnostics["all_observed_quotes_close_fresh"])
        self.assertIn(
            "OBSERVED_NBBO_TOO_OLD_AT_SESSION_CLOSE",
            diagnostics["strict_rejection_reasons"],
        )

    def test_strict_bundle_uses_exact_early_close(self):
        early_close_date = "2025-11-28"
        fresh = (
            _observed(
                TICKERS[0],
                "2025-11-28T17:55:00Z",
                pricing_date=early_close_date,
            ),
            _observed(
                TICKERS[1],
                "2025-11-28T17:59:00Z",
                pricing_date=early_close_date,
            ),
        )
        self.assertEqual(
            validate_strict_nbbo_bundle(
                fresh,
                TICKERS[:2],
                early_close_date,
                5.0,
                5.0,
            ),
            fresh,
        )
        stale = (
            _observed(
                TICKERS[0],
                "2025-11-28T17:54:59Z",
                pricing_date=early_close_date,
            ),
            fresh[1],
        )
        with self.assertRaisesRegex(HistoricalFillEvidenceError, "too old"):
            validate_strict_nbbo_bundle(
                stale,
                TICKERS[:2],
                early_close_date,
                5.0,
                5.0,
            )

    def test_bundle_shape_rejects_one_leg_duplicates_and_mixed_dates(self):
        one_leg = (
            _observed(TICKERS[0], "2025-01-02T20:58:00Z"),
        )
        one_diagnostics = fill_bundle_diagnostics(one_leg, 5.0, 5.0)
        self.assertFalse(one_diagnostics["bundle_shape_valid"])
        self.assertIn(
            "INVALID_LEG_COUNT",
            one_diagnostics["strict_rejection_reasons"],
        )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "exactly two or three",
        ):
            validate_strict_nbbo_bundle(
                one_leg,
                TICKERS[:1],
                PRICING_DATE,
                5.0,
                5.0,
            )

        duplicate = (one_leg[0], one_leg[0])
        duplicate_diagnostics = fill_bundle_diagnostics(
            duplicate,
            5.0,
            5.0,
        )
        self.assertFalse(duplicate_diagnostics["bundle_shape_valid"])
        self.assertIn(
            "DUPLICATE_OPTION_TICKER",
            duplicate_diagnostics["strict_rejection_reasons"],
        )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "unique",
        ):
            validate_strict_nbbo_bundle(
                duplicate,
                (TICKERS[0], TICKERS[0]),
                PRICING_DATE,
                5.0,
                5.0,
            )

        mixed_dates = (
            one_leg[0],
            _observed(
                TICKERS[1],
                "2025-01-03T20:58:00Z",
                pricing_date="2025-01-03",
            ),
        )
        mixed_diagnostics = fill_bundle_diagnostics(
            mixed_dates,
            5.0,
            5.0,
        )
        self.assertFalse(mixed_diagnostics["bundle_shape_valid"])
        self.assertIn(
            "MIXED_PRICING_DATES",
            mixed_diagnostics["strict_rejection_reasons"],
        )
        with self.assertRaisesRegex(
            HistoricalFillEvidenceError,
            "pricing-date mismatch",
        ):
            validate_strict_nbbo_bundle(
                mixed_dates,
                TICKERS[:2],
                PRICING_DATE,
                5.0,
                5.0,
            )


class MassiveFillAcquisitionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache = OptionDataCache(
            str(Path(self.temp_dir.name) / "option_data.db")
        )

    async def asyncTearDown(self):
        self.cache.close()
        self.temp_dir.cleanup()

    async def test_legacy_ambiguous_bid_ask_mid_cannot_authorize_strict_or_research(self):
        self.cache.upsert_full_record(
            {
                "underlying": "SPY",
                "option_ticker": TICKERS[0],
                "contract_type": "put",
                "strike": 500.0,
                "expiration": "2025-01-17",
                "pricing_date": PRICING_DATE,
                "bid": 1.0,
                "ask": 1.2,
                "mid": 1.1,
                "is_synchronized": 1,
            }
        )
        client = MassiveAPIClient(
            self.cache,
            api_key="",
            offline_only=True,
        )
        self.assertIsNone(
            await client.fetch_eod_quote(
                TICKERS[0],
                PRICING_DATE,
                strict_nbbo=True,
            )
        )
        self.assertIsNone(
            await client.fetch_eod_quote(
                TICKERS[0],
                PRICING_DATE,
                strict_nbbo=False,
            )
        )

    async def test_legacy_daily_close_is_explicit_research_fallback_only(self):
        self.cache.upsert_full_record(
            {
                "underlying": "SPY",
                "option_ticker": TICKERS[0],
                "contract_type": "put",
                "strike": 500.0,
                "expiration": "2025-01-17",
                "pricing_date": PRICING_DATE,
                "close": 1.15,
            }
        )
        client = MassiveAPIClient(
            self.cache,
            api_key="",
            offline_only=True,
        )
        evidence = await client.fetch_eod_quote(
            TICKERS[0],
            PRICING_DATE,
            strict_nbbo=False,
        )
        self.assertEqual(evidence.source, HistoricalFillSource.DAILY_CLOSE)
        self.assertFalse(evidence.strict_nbbo_eligible)
        self.assertIsNone(
            await client.fetch_eod_quote(
                TICKERS[0],
                PRICING_DATE,
                strict_nbbo=True,
            )
        )

    async def test_strict_mode_does_not_call_trade_fallback(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        client._get = AsyncMock(return_value={"results": []})
        client.fetch_latest_trade = AsyncMock(
            return_value={
                "price": 1.1,
                "size": 1,
                "timestamp": "2025-01-02T20:00:00Z",
                "source": HistoricalFillSource.TRADE_PRINT.value,
            }
        )
        self.assertIsNone(
            await client.fetch_eod_quote(
                TICKERS[0],
                PRICING_DATE,
                allow_trade_fallback=True,
                strict_nbbo=True,
            )
        )
        client.fetch_latest_trade.assert_not_awaited()

    async def test_provider_quote_parser_rejects_invalid_rows_and_accepts_locked_market(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        invalid_rows = (
            {
                "ticker": TICKERS[0],
                "bid_price": 1.0,
                "ask_price": 1.2,
            },
            {
                "ticker": TICKERS[0],
                "bid_price": 1.0,
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T20:58:00",
            },
            {
                "ticker": TICKERS[0],
                "bid_price": float("inf"),
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T20:58:00Z",
            },
            {
                "ticker": TICKERS[0],
                "bid_price": 1.3,
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T20:58:00Z",
            },
            {
                "ticker": TICKERS[0],
                "bid_price": 1.0,
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T22:00:00Z",
            },
            {
                "ticker": TICKERS[0],
                "bid_price": 1.0,
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T14:31:00Z",
            },
            {
                "ticker": TICKERS[1],
                "bid_price": 1.0,
                "ask_price": 1.2,
                "participant_timestamp": "2025-01-02T20:58:00Z",
            },
        )
        for row in invalid_rows:
            with self.subTest(row=row):
                client._get = AsyncMock(return_value={"results": [row]})
                self.assertIsNone(
                    await client.fetch_observed_nbbo(
                        TICKERS[0],
                        PRICING_DATE,
                    )
                )

        client._get = AsyncMock(
            return_value={
                "results": [
                    {
                        "ticker": TICKERS[0],
                        "bid_price": 1.2,
                        "ask_price": 1.2,
                        "participant_timestamp": (
                            "2025-01-02T20:58:00Z"
                        ),
                    }
                ]
            }
        )
        evidence = await client.fetch_observed_nbbo(
            TICKERS[0],
            PRICING_DATE,
        )
        self.assertEqual(evidence.bid, evidence.ask)
        self.assertEqual(
            evidence.source,
            HistoricalFillSource.OBSERVED_NBBO,
        )
        self.assertTrue(evidence.strict_nbbo_mark_eligible)
        self.assertFalse(evidence.strict_nbbo_eligible)
        request_params = client._get.await_args.args[1]
        self.assertEqual(
            request_params["timestamp.gte"],
            "2025-01-02T20:55:00Z",
        )
        self.assertEqual(
            request_params["timestamp.lte"],
            "2025-01-02T21:00:00Z",
        )

    async def test_research_mode_can_retain_stale_quote_but_diagnoses_it(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        client._get = AsyncMock(
            return_value={
                "results": [
                    {
                        "ticker": TICKERS[0],
                        "bid_price": 1.0,
                        "ask_price": 1.2,
                        "participant_timestamp": (
                            "2025-01-02T14:31:00Z"
                        ),
                    }
                ]
            }
        )
        evidence = await client.fetch_eod_quote(
            TICKERS[0],
            PRICING_DATE,
            allow_trade_fallback=False,
            strict_nbbo=False,
            max_quote_age_minutes=5.0,
        )
        self.assertEqual(evidence.source, HistoricalFillSource.OBSERVED_NBBO)
        diagnostics = fill_bundle_diagnostics((evidence,), 5.0, 5.0)
        self.assertFalse(diagnostics["strict_nbbo_mark_validated"])
        self.assertIn(
            "OBSERVED_NBBO_TOO_OLD_AT_SESSION_CLOSE",
            diagnostics["strict_rejection_reasons"],
        )

    async def test_strict_acquisition_uses_early_close_freshness_window(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        client._get = AsyncMock(
            return_value={
                "results": [
                    {
                        "ticker": TICKERS[0],
                        "bid_price": 1.0,
                        "ask_price": 1.2,
                        "participant_timestamp": (
                            "2025-11-28T17:59:00Z"
                        ),
                    }
                ]
            }
        )
        evidence = await client.fetch_observed_nbbo(
            TICKERS[0],
            "2025-11-28",
            max_quote_age_minutes=5.0,
        )
        self.assertIsNotNone(evidence)
        request_params = client._get.await_args.args[1]
        self.assertEqual(
            request_params["timestamp.gte"],
            "2025-11-28T17:55:00Z",
        )
        self.assertEqual(
            request_params["timestamp.lte"],
            "2025-11-28T18:00:00Z",
        )

    async def test_strict_multileg_fetch_rejects_skew_without_aggregate_fallback(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        by_ticker = {
            TICKERS[0]: _observed(
                TICKERS[0],
                "2025-01-02T20:55:00Z",
            ),
            TICKERS[1]: _observed(
                TICKERS[1],
                "2025-01-02T20:57:01Z",
            ),
        }
        client.fetch_observed_nbbo = AsyncMock(
            side_effect=lambda ticker, date, **kwargs: by_ticker[ticker]
        )
        client.fetch_synchronized_minute_aggregates = AsyncMock(
            return_value=(
                HistoricalFillEvidence(
                    option_ticker=TICKERS[0],
                    pricing_date=PRICING_DATE,
                    source=(
                        HistoricalFillSource.SYNCHRONIZED_MINUTE_AGGREGATE
                    ),
                    mid=1.0,
                    bid=1.0,
                    ask=1.0,
                    event_timestamp_utc="2025-01-02T20:00:00Z",
                ),
                HistoricalFillEvidence(
                    option_ticker=TICKERS[1],
                    pricing_date=PRICING_DATE,
                    source=(
                        HistoricalFillSource.SYNCHRONIZED_MINUTE_AGGREGATE
                    ),
                    mid=0.8,
                    bid=0.8,
                    ask=0.8,
                    event_timestamp_utc="2025-01-02T20:00:00Z",
                ),
            )
        )
        self.assertIsNone(
            await client.fetch_multileg_eod_marks(
                TICKERS[:2],
                PRICING_DATE,
                strict_nbbo=True,
                max_time_delta_minutes=2.0,
                max_quote_age_minutes=5.0,
            )
        )
        client.fetch_synchronized_minute_aggregates.assert_not_awaited()

    async def test_research_trade_fallback_has_explicit_non_nbbo_class(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        client.fetch_observed_nbbo = AsyncMock(return_value=None)
        client.fetch_latest_trade = AsyncMock(
            return_value={
                "price": 1.1,
                "size": 4,
                "timestamp": "2025-01-02T20:00:00Z",
                "source": HistoricalFillSource.TRADE_PRINT.value,
            }
        )
        evidence = await client.fetch_eod_quote(
            TICKERS[0],
            PRICING_DATE,
            strict_nbbo=False,
        )
        self.assertEqual(evidence.source, HistoricalFillSource.TRADE_PRINT)
        self.assertFalse(evidence.strict_nbbo_eligible)

    async def test_theoretical_fallback_is_option_type_aware(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        call_target = "O:SPY250117C00510000"
        call_reference = "O:SPY250117C00500000"
        with (
            patch(
                "backtesting.greeks_calculator.implied_volatility",
                return_value=0.2,
            ) as implied_vol,
            patch(
                "backtesting.greeks_calculator.bs_call_price",
                return_value=2.34,
            ) as call_price,
            patch(
                "backtesting.greeks_calculator.bs_put_price",
                return_value=9.99,
            ) as put_price,
        ):
            price = await client.fetch_theoretical_price(
                call_target,
                call_reference,
                5.0,
                500.0,
                PRICING_DATE,
                30 / 365,
                0.05,
                option_type="call",
            )
        self.assertEqual(price, 2.34)
        self.assertEqual(implied_vol.call_args.args[-1], "call")
        call_price.assert_called_once()
        put_price.assert_not_called()

        with (
            patch(
                "backtesting.greeks_calculator.implied_volatility",
                return_value=0.2,
            ) as implied_vol,
            patch(
                "backtesting.greeks_calculator.bs_call_price",
                return_value=9.99,
            ) as call_price,
            patch(
                "backtesting.greeks_calculator.bs_put_price",
                return_value=1.23,
            ) as put_price,
        ):
            price = await client.fetch_theoretical_price(
                TICKERS[1],
                TICKERS[0],
                5.0,
                500.0,
                PRICING_DATE,
                30 / 365,
                0.05,
                option_type="put",
            )
        self.assertEqual(price, 1.23)
        self.assertEqual(implied_vol.call_args.args[-1], "put")
        put_price.assert_called_once()
        call_price.assert_not_called()

        with self.assertRaisesRegex(ValueError, "option_type"):
            await client.fetch_theoretical_price(
                TICKERS[1],
                TICKERS[0],
                5.0,
                500.0,
                PRICING_DATE,
                30 / 365,
                0.05,
                option_type="future",
            )
        with self.assertRaisesRegex(ValueError, "contract symbols"):
            await client.fetch_theoretical_price(
                TICKERS[1],
                TICKERS[0],
                5.0,
                500.0,
                PRICING_DATE,
                30 / 365,
                0.05,
                option_type="call",
            )

    async def test_three_leg_synchronized_minute_fallback_checks_all_legs(self):
        client = MassiveAPIClient(self.cache, api_key="placeholder")
        client.offline_only = False
        timestamps = (
            1735848000000,
            1735848120000,
            1735848240000,
        )

        async def fake_get(url, params):
            ticker_index = next(
                index for index, ticker in enumerate(TICKERS) if ticker in url
            )
            return {
                "results": [
                    {
                        "t": timestamps[ticker_index],
                        "c": 1.0 + ticker_index * 0.1,
                    }
                ]
            }

        client._get = fake_get
        evidence = await client.fetch_synchronized_minute_aggregates(
            TICKERS,
            PRICING_DATE,
            max_time_delta_minutes=5.0,
        )
        self.assertEqual(len(evidence), 3)
        self.assertEqual(
            {item.source for item in evidence},
            {HistoricalFillSource.SYNCHRONIZED_MINUTE_AGGREGATE},
        )
        self.assertTrue(
            fill_bundle_diagnostics(
                evidence,
                5.0,
            )["temporally_synchronized"]
        )
        self.assertIsNone(
            await client.fetch_synchronized_minute_aggregates(
                TICKERS,
                PRICING_DATE,
                max_time_delta_minutes=1.0,
            )
        )


class BacktestFillBoundaryTests(unittest.TestCase):
    def test_entry_exit_mode_resolution_and_legacy_rejection(self):
        resolve = backtest_runner._resolve_historical_fill_modes
        self.assertEqual(
            resolve({"fill_mode": "strict_nbbo"}),
            ("strict_nbbo", "strict_nbbo"),
        )
        self.assertEqual(
            resolve(
                {
                    "entry_fill_mode": "strict_nbbo",
                    "exit_fill_mode": "research_fallback",
                }
            ),
            ("strict_nbbo", "research_fallback"),
        )
        with self.assertRaisesRegex(ValueError, "deprecated and ambiguous"):
            resolve({"require_entry_nbbo": False})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve(
                {
                    "fill_mode": "strict_nbbo",
                    "exit_fill_mode": "research_fallback",
                }
            )

    def test_programmatic_fill_threshold_precedence(self):
        resolve = backtest_runner._resolve_historical_fill_thresholds
        self.assertEqual(
            resolve(
                explicit_max_time_delta_minutes=None,
                explicit_max_quote_age_minutes=None,
                execution_config={
                    "max_time_delta_minutes": 2.0,
                    "max_quote_age_minutes": 3.0,
                },
                entry_config={"max_time_delta_minutes": 9.0},
            ),
            (2.0, 3.0),
        )
        self.assertEqual(
            resolve(
                explicit_max_time_delta_minutes=1.0,
                explicit_max_quote_age_minutes=4.0,
                execution_config={
                    "max_time_delta_minutes": 2.0,
                    "max_quote_age_minutes": 3.0,
                },
                entry_config={"max_time_delta_minutes": 9.0},
            ),
            (1.0, 4.0),
        )
        self.assertEqual(
            resolve(
                explicit_max_time_delta_minutes=None,
                explicit_max_quote_age_minutes=None,
                execution_config={},
                entry_config={"max_time_delta_minutes": 8.0},
            ),
            (8.0, 5.0),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve(
                explicit_max_time_delta_minutes=None,
                explicit_max_quote_age_minutes=None,
                execution_config={"max_quote_age_minutes": 0},
                entry_config={},
            )

    def test_entry_exit_roll_call_and_far_long_share_typed_multileg_boundary(self):
        source = inspect.getsource(
            backtest_runner.run_put_credit_spread_backtest
        )
        tree = ast.parse(source)
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        self.assertGreaterEqual(calls.count("fetch_multileg_eod_marks"), 3)
        self.assertNotIn("fetch_synchronized_ohlcv", calls)
        self.assertNotIn("fetch_eod_quote", calls)
        self.assertIn("far_long_ticker", source)
        self.assertIn("call_entry_evidence", source)
        self.assertIn("NO_VALIDATED_HISTORICAL_MARK", source)
        self.assertGreaterEqual(source.count("max_quote_age_minutes="), 6)
        self.assertGreaterEqual(source.count("option_type="), 2)
        self.assertIn("EXPIRATION_CLOSE_MARK_RESEARCH_ONLY", source)
        self.assertNotIn("EXPIRATION_INTRINSIC_SETTLEMENT", source)
        self.assertIn('"settlement_claimed": False', source)
        self.assertIn(
            "replacement_request_count_before_trade",
            source,
        )
        self.assertIn(
            'exit_reason = "expiration_close_mark_research_only"',
            source,
        )

    def test_trade_contract_persists_entry_and_exit_fill_diagnostics(self):
        trade = backtest_runner.SpreadTrade(
            entry_date=PRICING_DATE,
            entry_fill_mode="strict_nbbo",
            entry_fill_sources=[
                HistoricalFillSource.OBSERVED_NBBO.value,
                HistoricalFillSource.OBSERVED_NBBO.value,
            ],
            entry_fill_timestamps_utc=[
                "2025-01-02T20:00:00Z",
                "2025-01-02T20:00:01Z",
            ],
            entry_fill_temporally_synchronized=True,
            entry_fill_strict_nbbo_mark_validated=True,
            entry_fill_strict_nbbo_authorized=False,
            exit_fill_mode="research_fallback",
            exit_fill_sources=[
                HistoricalFillSource.DAILY_CLOSE.value,
                HistoricalFillSource.DAILY_CLOSE.value,
            ],
            exit_fill_temporally_synchronized=None,
            exit_fill_strict_nbbo_authorized=False,
        )
        serialized = backtest_runner.asdict(trade)
        self.assertTrue(
            serialized["entry_fill_strict_nbbo_mark_validated"]
        )
        self.assertFalse(serialized["entry_fill_strict_nbbo_authorized"])
        self.assertFalse(serialized["entry_execution_proven"])
        self.assertEqual(
            serialized["entry_pricing_assumption"],
            "HISTORICAL_MARK_NOT_EXECUTABLE_FILL",
        )
        self.assertEqual(
            serialized["entry_fill_sources"],
            [
                HistoricalFillSource.OBSERVED_NBBO.value,
                HistoricalFillSource.OBSERVED_NBBO.value,
            ],
        )
        self.assertFalse(serialized["exit_fill_strict_nbbo_authorized"])
        self.assertFalse(serialized["exit_execution_proven"])
        self.assertFalse(
            backtest_runner.BacktestResult().historical_execution_proven
        )
        self.assertEqual(
            serialized["exit_fill_sources"],
            [
                HistoricalFillSource.DAILY_CLOSE.value,
                HistoricalFillSource.DAILY_CLOSE.value,
            ],
        )


if __name__ == "__main__":
    unittest.main()
