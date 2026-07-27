"""Regression coverage for the unconditional legacy execution tombstones."""

from __future__ import annotations

import copy
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import order.order_bo as order_bo
from live_trading.runtime_safety import (
    LegacyExecutionDisabled,
    RuntimeSafetyError,
    reject_legacy_execution,
)


class _ForbiddenBroker:
    def __init__(self):
        self.accesses = []

    def __getattr__(self, name):
        self.accesses.append(name)
        raise AssertionError(f"legacy code accessed broker session attribute {name}")


class _ReadResponse:
    status_code = 204
    text = ""
    headers = {}
    request = SimpleNamespace(headers={})


class _ReadSession:
    def __init__(self):
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _ReadResponse()


def _order_client(session):
    client = object.__new__(order_bo.Order)
    client.session = session
    client.account = {"accountIdKey": "account-key"}
    client.base_url = "https://example.invalid"
    client.consumer_key = "consumer-key"
    client.runtime_safety = None
    return client


class LegacyExecutionContainmentTests(unittest.TestCase):
    def test_rejection_has_no_environment_override(self):
        self.assertTrue(issubclass(LegacyExecutionDisabled, RuntimeSafetyError))
        with mock.patch.dict(
            os.environ,
            {
                "ALLOW_LEGACY_EXECUTION": "1",
                "ETRADE_ENABLE_LEGACY_EXECUTION": "true",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                LegacyExecutionDisabled,
                "order.order_bo.Order.place_order",
            ):
                reject_legacy_execution("order.order_bo.Order.place_order")

    def test_order_bo_mutations_reject_before_broker_input_sleep_or_payload_change(self):
        broker = _ForbiddenBroker()
        client = _order_client(broker)
        order = {
            "symbol": "BRK.B",
            "securityType": "OPTN",
            "orderAction": "SPREAD",
            "legs": [{"symbol": "BRK.B"}],
        }
        detail = {"limitPrice": 1.0, "Instrument": [{"Product": {"symbol": "SPY"}}]}
        previous = [{"symbol": "SPY", "client_order_id": "unchanged"}]
        mutable_inputs = {"order": order, "detail": detail, "previous": previous}
        before = copy.deepcopy(mutable_inputs)
        calls = (
            ("preview_order", (order,), {}),
            ("preview_order_old", (order,), {}),
            ("place_order", (order,), {}),
            ("place_order_old", (order,), {}),
            ("change_order_limit", (123, 0.95, detail, "SPREADS"), {}),
            ("wait_and_adjust_until_filled", (123,), {}),
            ("previous_order", (broker, client.account, previous), {}),
            ("options_selection", (["Continue"],), {}),
            ("user_select_order", (), {}),
            ("preview_order_menu", (broker, client.account, previous), {}),
            ("cancel_order", (), {}),
            ("view_orders", (), {}),
            ("filter_order", (), {}),
            ("place_option_order", ("SPY", 500.0, "2026-12-18", "PUT"), {}),
            ("cancel_all_order", (), {}),
            ("refresh_order_limit_old", (), {}),
            ("refresh_order_limit", (), {}),
        )

        with (
            mock.patch("builtins.input", side_effect=AssertionError("input reached")),
            mock.patch.object(
                order_bo.time,
                "sleep",
                side_effect=AssertionError("sleep reached"),
            ),
        ):
            for method_name, args, kwargs in calls:
                with self.subTest(method=method_name):
                    with self.assertRaisesRegex(
                        LegacyExecutionDisabled,
                        rf"order\.order_bo\.Order\.{method_name}",
                    ):
                        getattr(client, method_name)(*args, **kwargs)

        self.assertEqual(broker.accesses, [])
        self.assertEqual(mutable_inputs, before)

    def test_obsolete_interactive_order_class_rejects_during_construction(self):
        from order.order import Order as InteractiveOrder

        broker = _ForbiddenBroker()
        with self.assertRaisesRegex(
            LegacyExecutionDisabled,
            r"order\.order\.Order\.__init__",
        ):
            InteractiveOrder(broker, {"accountIdKey": "account-key"}, "https://example.invalid")
        self.assertEqual(broker.accesses, [])

    def test_live_trade_agent_execution_helpers_reject_before_order_access(self):
        from core_api.stock_trade_class import LiveTradeAgent

        agent = object.__new__(LiveTradeAgent)
        forbidden_order = _ForbiddenBroker()
        agent.order = forbidden_order
        tickers = ["SPY", "QQQ"]
        orders = [{"symbol": "SPY", "quantity": 1}]
        before = copy.deepcopy((tickers, orders))

        with self.assertRaisesRegex(
            LegacyExecutionDisabled,
            "LiveTradeAgent.check_short_availability",
        ):
            agent.check_short_availability(tickers)
        with self.assertRaisesRegex(
            LegacyExecutionDisabled,
            "LiveTradeAgent.place_order",
        ):
            agent.place_order(orders)

        self.assertEqual(forbidden_order.accesses, [])
        self.assertEqual((tickers, orders), before)

    def test_put_credit_margin_release_rejects_before_account_or_order_access(self):
        from live_trading.etrade_put_credit_spread import release_margin

        accounts = _ForbiddenBroker()
        etrade = _ForbiddenBroker()
        positions = [{"symbol": "SPY"}]
        cover_calls = {"SPY": 1}
        before = copy.deepcopy((positions, cover_calls))

        with self.assertRaisesRegex(
            LegacyExecutionDisabled,
            r"etrade_put_credit_spread\.release_margin",
        ):
            release_margin(accounts, etrade, positions, cover_calls)

        self.assertEqual(accounts.accesses, [])
        self.assertEqual(etrade.accesses, [])
        self.assertEqual((positions, cover_calls), before)

    def test_read_only_order_query_and_pure_price_helpers_remain_available(self):
        session = _ReadSession()
        client = _order_client(session)

        self.assertEqual(client.get_open_orders(), [])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0][1]["params"], {"status": "OPEN"})
        self.assertEqual(order_bo.Order._option_tick_size(2.95), 0.05)
        self.assertEqual(order_bo.Order._snap_option_limit_price(0.96, 1.0), 0.95)


if __name__ == "__main__":
    unittest.main()
