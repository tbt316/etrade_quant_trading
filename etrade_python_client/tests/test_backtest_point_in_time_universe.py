import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import backtesting.backtest_runner as backtest_module
from backtesting.contract_universe import (
    CONTRACT_UNIVERSE_CAUSAL_STATUS,
    ContractUniverseError,
    acquisition_union_by_expiration,
    build_contract_universe_snapshot,
    filter_chain_for_snapshot,
)


TRADE_DATE = "2025-04-03"
EXPIRATION = "2025-05-16"
DECISION_TIME = "2025-04-03T20:00:00Z"


def _contract(ticker: str, strike: float, contract_type: str = "put") -> dict:
    return {
        "option_ticker": ticker,
        "strike": strike,
        "contract_type": contract_type,
    }


def _snapshot(
    contracts: list[dict],
    *,
    trade_date: str = TRADE_DATE,
    expiration: str = EXPIRATION,
    decision_time: str = DECISION_TIME,
    available_at: str | None = None,
):
    return build_contract_universe_snapshot(
        underlying="SPY",
        expiration=expiration,
        contract_type="put",
        as_of_date=trade_date,
        available_at=available_at or decision_time,
        decision_time=decision_time,
        contracts=contracts,
    )


def _chain_row(
    ticker: str,
    strike: float,
    *,
    pricing_date: str = TRADE_DATE,
    expiration: str = EXPIRATION,
) -> dict:
    return {
        "underlying": "SPY",
        "option_ticker": ticker,
        "contract_type": "put",
        "strike": strike,
        "expiration": expiration,
        "pricing_date": pricing_date,
        "close": 1.0,
    }


class ContractUniverseDomainTests(unittest.TestCase):
    def test_manifest_is_immutable_content_addressed_and_order_invariant(self):
        first = _contract("O:SPY250516P00095000", 95)
        second = _contract("O:SPY250516P00100000", 100)

        left = _snapshot([second, first])
        right = _snapshot([first, second])

        self.assertEqual(left, right)
        self.assertEqual(left.snapshot_sha256, right.snapshot_sha256)
        self.assertEqual(
            left.to_manifest()["snapshot_sha256"],
            left.snapshot_sha256,
        )
        self.assertEqual(
            left.causal_status,
            CONTRACT_UNIVERSE_CAUSAL_STATUS,
        )
        self.assertFalse(left.provider_available_at_verified)
        self.assertEqual(left.completeness_status, "UNVERIFIED")
        self.assertFalse(left.provider_page_evidence_verified)
        with self.assertRaises(AttributeError):
            left.available_at = "2025-04-04T20:00:00Z"

    def test_snapshot_rejects_availability_after_decision(self):
        with self.assertRaisesRegex(
            ContractUniverseError,
            "not available at the decision time",
        ):
            _snapshot(
                [_contract("O:SPY250516P00100000", 100)],
                available_at="2025-04-03T20:00:01Z",
            )

    def test_snapshot_cannot_be_reused_at_another_decision_time(self):
        snapshot = _snapshot(
            [_contract("O:SPY250516P00100000", 100)]
        )
        with self.assertRaisesRegex(
            ContractUniverseError,
            "different decision timestamp",
        ):
            snapshot.assert_usable_at("2025-04-04T20:00:00Z")

    def test_future_snapshot_and_future_contract_rows_do_not_change_prefix(self):
        prefix_snapshot = _snapshot(
            [
                _contract("O:SPY250516P00095000", 95),
                _contract("O:SPY250516P00100000", 100),
            ]
        )
        future_snapshot = _snapshot(
            [
                _contract("O:SPY250516P00050000", 50),
                _contract("O:SPY250516P00095000", 95),
                _contract("O:SPY250516P00100000", 100),
                _contract("O:SPY250516P00150000", 150),
            ],
            trade_date="2025-04-04",
            decision_time="2025-04-04T20:00:00Z",
        )

        prefix_union = acquisition_union_by_expiration(
            [prefix_snapshot],
            "put",
        )
        extended_union = acquisition_union_by_expiration(
            [prefix_snapshot, future_snapshot],
            "put",
        )
        self.assertNotEqual(prefix_union, extended_union)

        acquired_rows = [
            _chain_row("O:SPY250516P00050000", 50),
            _chain_row("O:SPY250516P00095000", 95),
            _chain_row("O:SPY250516P00100000", 100),
            _chain_row("O:SPY250516P00150000", 150),
        ]
        prefix_eligible = filter_chain_for_snapshot(
            acquired_rows,
            prefix_snapshot,
            DECISION_TIME,
        )
        extended_eligible = filter_chain_for_snapshot(
            list(reversed(acquired_rows)),
            prefix_snapshot,
            DECISION_TIME,
        )

        expected = [
            "O:SPY250516P00095000",
            "O:SPY250516P00100000",
        ]
        self.assertEqual(
            [row["option_ticker"] for row in prefix_eligible],
            expected,
        )
        self.assertEqual(
            [row["option_ticker"] for row in extended_eligible],
            expected,
        )

    def test_eligible_chain_requires_exact_decision_date_metadata(self):
        snapshot = _snapshot(
            [_contract("O:SPY250516P00100000", 100)]
        )
        with self.assertRaisesRegex(
            ContractUniverseError,
            "must match snapshot as_of_date",
        ):
            filter_chain_for_snapshot(
                [
                    _chain_row(
                        "O:SPY250516P00100000",
                        100,
                        pricing_date="2025-04-04",
                    )
                ],
                snapshot,
                DECISION_TIME,
            )


class PointInTimeSourceBoundaryTests(unittest.TestCase):
    def test_all_chain_cache_reads_are_centralized_behind_snapshot_filter(self):
        tree = ast.parse(
            Path(backtest_module.__file__).read_text(encoding="utf-8")
        )
        calls = []

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.functions = []

            def visit_FunctionDef(self, node):
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_Call(self, node):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get_chain_for_date"
                ):
                    calls.append((self.functions[-1], node.lineno))
                self.generic_visit(node)

        Visitor().visit(tree)
        self.assertEqual(
            [name for name, _ in calls],
            ["_point_in_time_chain"],
        )

    def test_runner_has_no_future_spot_window_strike_filter(self):
        source = inspect.getsource(
            backtest_module.run_put_credit_spread_backtest
        )
        self.assertNotIn("spot_window", source)
        self.assertNotIn("as_of = exp", source)


class _RecordingReferenceClient:
    def __init__(self, contracts=None):
        self.contracts = contracts or [
            _contract("O:SPY250516P00100000", 100)
        ]
        self.calls = []

    async def fetch_contracts_list(
        self,
        underlying,
        expiration,
        contract_type,
        as_of,
    ):
        self.calls.append(
            (underlying, expiration, contract_type, as_of)
        )
        return list(self.contracts)


class PointInTimeFetchBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_query_uses_trade_date_not_expiration(self):
        client = _RecordingReferenceClient()

        snapshot = (
            await backtest_module._fetch_point_in_time_contract_universe(
                client,
                underlying="SPY",
                expiration=EXPIRATION,
                contract_type="put",
                trade_date=TRADE_DATE,
                decision_time=DECISION_TIME,
            )
        )

        self.assertEqual(
            client.calls,
            [("SPY", EXPIRATION, "put", TRADE_DATE)],
        )
        self.assertEqual(snapshot.as_of_date, TRADE_DATE)
        self.assertEqual(snapshot.available_at, DECISION_TIME)
        self.assertEqual(snapshot.causal_status, "UNVERIFIED")


class _NoPriceCache:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_chain_for_date(self, *_args, **_kwargs):
        return []

    def clear_daily_memory_cache(self):
        pass

    def close(self):
        pass


class _UniverseRecordingClient:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.api_calls = 0
        self.cache_hits = 0
        self.reference_calls = []
        self.batches = []
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def fetch_contracts_list(
        self,
        underlying,
        expiration,
        contract_type,
        as_of,
    ):
        self.reference_calls.append(
            (underlying, expiration, contract_type, as_of)
        )
        yymmdd = expiration[2:4] + expiration[5:7] + expiration[8:10]
        flag = "P" if contract_type == "put" else "C"
        return [
            _contract(
                f"O:{underlying}{yymmdd}{flag}00050000",
                50,
                contract_type,
            ),
            _contract(
                f"O:{underlying}{yymmdd}{flag}00100000",
                100,
                contract_type,
            ),
        ]

    async def fetch_chain_ohlcv_batch(
        self,
        contracts,
        from_date,
        to_date,
        **kwargs,
    ):
        self.batches.append(
            {
                "contracts": tuple(
                    row["option_ticker"] for row in contracts
                ),
                "from_date": from_date,
                "to_date": to_date,
                **kwargs,
            }
        )
        return 0


class BacktestAcquisitionPrefixTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, underlying_prices):
        _UniverseRecordingClient.instances.clear()
        with (
            patch.object(
                backtest_module,
                "OptionDataCache",
                _NoPriceCache,
            ),
            patch.object(
                backtest_module,
                "MassiveAPIClient",
                _UniverseRecordingClient,
            ),
            patch.object(
                backtest_module,
                "_fetch_underlying_prices",
                return_value=pd.Series(
                    [5.0],
                    index=[TRADE_DATE],
                ),
            ),
        ):
            result = await backtest_module.run_put_credit_spread_backtest(
                start_date=TRADE_DATE,
                end_date=TRADE_DATE,
                underlying_prices=underlying_prices,
                vix_prices=pd.Series(
                    [18.0],
                    index=[TRADE_DATE],
                ),
                offline_only=True,
            )
        return result, _UniverseRecordingClient.instances[-1]

    async def test_future_underlying_rows_cannot_filter_acquisition_or_eligibility(self):
        prefix_prices = pd.Series(
            [100.0, 100.0],
            index=["2011-05-03", TRADE_DATE],
        )
        extended_prices = pd.concat(
            [
                prefix_prices,
                pd.Series(
                    [40.0],
                    index=["2025-04-10"],
                ),
            ]
        )

        prefix_result, prefix_client = await self._run(prefix_prices)
        extended_result, extended_client = await self._run(extended_prices)

        self.assertTrue(prefix_client.reference_calls)
        self.assertTrue(
            all(
                as_of == TRADE_DATE
                for _, _, _, as_of in prefix_client.reference_calls
            )
        )
        self.assertEqual(prefix_client.batches, extended_client.batches)
        self.assertTrue(
            all(len(batch["contracts"]) == 2 for batch in prefix_client.batches)
        )
        self.assertEqual(
            prefix_result.contract_universe_manifest_sha256,
            extended_result.contract_universe_manifest_sha256,
        )
        self.assertEqual(prefix_result.causal_validity, "UNVERIFIED")
        self.assertIn(
            "contract_reference_available_at_is_modeled_not_provider_observed",
            prefix_result.causal_validity_reasons,
        )
        self.assertIn(
            "contract_reference_pagination_completeness_unverified",
            prefix_result.causal_validity_reasons,
        )


if __name__ == "__main__":
    unittest.main()
