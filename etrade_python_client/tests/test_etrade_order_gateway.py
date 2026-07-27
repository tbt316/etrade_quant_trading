from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from rauth import OAuth1Session

import live_trading.etrade_order_gateway as gateway_module
from live_trading.etrade_broker_transport import (
    ETradeBrokerTransport,
    SelectedBrokerAccount,
    _ExchangeResult,
)
from live_trading.etrade_order_gateway import (
    BrokerCapacitySnapshot,
    BrokerOrderNotFound,
    BrokerOrderSnapshot,
    EtradeOrderGateway,
    GatewayReconciliationRequired,
    GatewayValidationError,
    RepriceOpeningCommand,
    SubmitOpeningCommand,
)
from live_trading.order_intent_ledger import (
    OrderIntentLedger,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    canonical_order_payload_hash,
)
from live_trading.runtime_safety import (
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
)


ACCOUNT_ID = "842468410"
ACCOUNT_KEY = "account/key"
INSTITUTION_TYPE = "BROKERAGE"
OWNER = "gateway-worker"


class Clock:
    def __init__(self):
        self.now = datetime.now(timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def vertical_payload(limit_price=1.25):
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": limit_price,
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2027,
                "expiryMonth": 1,
                "expiryDay": 15,
                "strikePrice": 620,
                "orderAction": "SELL_OPEN",
                "quantity": 1,
            },
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2027,
                "expiryMonth": 1,
                "expiryDay": 15,
                "strikePrice": 615,
                "orderAction": "BUY_OPEN",
                "quantity": 1,
            },
        ],
    }


def payload_bytes(limit_price=1.25):
    return json.dumps(
        vertical_payload(limit_price),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def order_payload_hash(limit_price=1.25):
    return canonical_order_payload_hash(
        vertical_payload(limit_price)
    )


def preview_result(
    *,
    preview_id="1020563279",
    status="OPEN",
    messages=None,
):
    order = {"status": status}
    if messages is not None:
        order["messages"] = {"Message": messages}
    body = {
        "PreviewOrderResponse": {
            "accountId": ACCOUNT_ID,
            "orderType": "SPREADS",
            "PreviewIds": [{"previewId": preview_id}],
            "Order": [order],
        }
    }
    return _ExchangeResult(
        "RESPONSE",
        http_status=200,
        raw_response=json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


def place_result(*, order_id="94", status="OPEN", messages=None):
    order = {"status": status}
    if messages is not None:
        order["messages"] = {"Message": messages}
    body = {
        "PlaceOrderResponse": {
            "accountId": ACCOUNT_ID,
            "orderType": "SPREADS",
            "OrderIds": [{"orderId": order_id}],
            "Order": [order],
        }
    }
    return _ExchangeResult(
        "RESPONSE",
        http_status=200,
        raw_response=json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


class ExchangeHarness:
    def __init__(self):
        self.outcomes = []
        self.calls = []

    def exchange(self, prepared, *, timeout_seconds):
        self.calls.append((prepared, timeout_seconds))
        if not self.outcomes:
            raise AssertionError("unexpected broker mutation")
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def add(self, *outcomes):
        self.outcomes.extend(outcomes)

    def count(self, suffix):
        return sum(call[0].url.endswith(suffix) for call in self.calls)


class FakeReader:
    def __init__(self, clock, account, environment="sandbox"):
        self.clock = clock
        self.account = account
        self.environment = environment
        self.capacity_digest = "a" * 64
        self.capacity_complete = True
        self.selected_hook = None
        self.query_behaviors = {}
        self.query_calls = []

    def assert_gateway_binding(self, runtime_safety):
        if self.environment != runtime_safety.environment:
            raise RuntimeSafetyError("reader environment mismatch")

    def selected_account(self):
        if self.selected_hook is not None:
            hook = self.selected_hook
            self.selected_hook = None
            hook()
        return self.account

    def read_capacity(self, account):
        return BrokerCapacitySnapshot(
            account=self.account,
            environment=self.environment,
            broker_buying_power=Decimal("2000"),
            observed_at=self.clock.now,
            portfolio_snapshot_digest=self.capacity_digest,
            positions_complete=self.capacity_complete,
            open_orders_complete=self.capacity_complete,
            source_response_digests=("1" * 64, "2" * 64),
        )

    def query_order(self, account, broker_order_id):
        self.query_calls.append((account, broker_order_id))
        behavior = self.query_behaviors.get(broker_order_id)
        if isinstance(behavior, BaseException):
            raise behavior
        if callable(behavior):
            return behavior()
        if behavior is None:
            return BrokerOrderNotFound(
                account=self.account,
                environment=self.environment,
                broker_order_id=broker_order_id,
                observed_at=self.clock.now,
                http_status=404,
                raw_response_digest="3" * 64,
                complete=True,
            )
        return behavior


class EtradeOrderGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        os.chmod(self.temporary.name, 0o700)
        self.path = (
            Path(self.temporary.name) / "runtime" / "orders.sqlite3"
        )
        self.clock = Clock()
        self.account = SelectedBrokerAccount(
            ACCOUNT_ID, ACCOUNT_KEY, INSTITUTION_TYPE
        )
        self.boundary = RuntimeSafetyBoundary(
            "sandbox",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(minutes=10),
        )
        self.ledger = OrderIntentLedger(
            self.path, clock=self.clock, run_id="gateway-run-a"
        )
        self.harness = ExchangeHarness()
        self.patcher = patch(
            "live_trading.etrade_broker_transport._isolated_exchange",
            side_effect=self.harness.exchange,
        )
        self.patcher.start()
        self.reader = FakeReader(self.clock, self.account)
        self.transport = self.make_transport(
            self.ledger, self.boundary
        )
        self.gateway = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=self.ledger,
            transport=self.transport,
            reader=self.reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        self.gateway.start()

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()

    def make_transport(self, ledger, boundary):
        return ETradeBrokerTransport(
            session=OAuth1Session(
                "consumer-key",
                "consumer-secret",
                access_token="access-token",
                access_token_secret="access-secret",
            ),
            ledger=ledger,
            runtime_safety=boundary,
            selected_account=self.account,
            clock=self.clock,
        )

    def command(
        self,
        *,
        key="order-1",
        decision="decision-1",
        lease_seconds=30,
    ):
        return SubmitOpeningCommand(
            strategy_id="put-credit-spread",
            decision_id=decision,
            idempotency_scope="decision",
            idempotency_key=key,
            payload_bytes=payload_bytes(),
            max_loss_amount=Decimal("400"),
            collateral_amount=Decimal("500"),
            quote_observed_at=self.clock.now,
            quote_digest="f" * 64,
            owner=OWNER,
            lease_seconds=lease_seconds,
        )

    def restart(self, *, reader=None):
        ledger = OrderIntentLedger(
            self.path, clock=self.clock, run_id="gateway-run-b"
        )
        reader = reader or FakeReader(self.clock, self.account)
        transport = self.make_transport(ledger, self.boundary)
        gateway = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=ledger,
            transport=transport,
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        return gateway, ledger, reader

    def submit_success(self):
        self.harness.add(preview_result(), place_result())
        return self.gateway.submit_opening(self.command())

    def test_real_transport_submission_is_idempotent_and_places_once(self):
        first = self.submit_success()
        replay = self.gateway.submit_opening(self.command())

        self.assertEqual(first.state, "SUBMITTED")
        self.assertEqual(replay.state, "SUBMITTED")
        self.assertFalse(replay.created)
        self.assertEqual(first.intent_id, replay.intent_id)
        self.assertEqual(self.harness.count("/orders/preview"), 1)
        self.assertEqual(self.harness.count("/orders/place"), 1)
        self.assertEqual(
            [
                receipt.transport_operation
                for receipt in self.ledger.transport_response_receipts(
                    first.intent_id
                )
            ],
            ["SUBMIT_PREVIEW", "SUBMIT_PLACE"],
        )

    def test_no_id_timeout_is_durable_and_cannot_be_auto_reconciled(self):
        self.harness.add(preview_result(), _ExchangeResult("TIMEOUT"))

        result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(result.reason_code, "TIMEOUT")
        with self.assertRaises(GatewayReconciliationRequired):
            self.gateway.submit_opening(self.command())
        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_known_executed_response_reconciles_filled_but_absorption_blocks(self):
        self.harness.add(
            preview_result(), place_result(status="EXECUTED")
        )
        unknown = self.gateway.submit_opening(self.command())
        self.assertEqual(unknown.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(unknown.broker_order_id, "94")
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="FILLED",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(ledger.get_intent(unknown.intent_id).state, "FILLED")
        self.assertEqual(
            ledger.unabsorbed_filled_reservation_count(
                ACCOUNT_ID, "sandbox"
            ),
            1,
        )

    def test_known_hold_reconciles_open_and_allows_next_opening(self):
        self.harness.add(
            preview_result(),
            place_result(
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                }
            ),
        )
        unknown = self.gateway.submit_opening(self.command())
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, reader = self.restart(reader=reader)
        restarted.start()
        self.assertEqual(ledger.get_intent(unknown.intent_id).state, "SUBMITTED")

        self.clock.advance(1)
        reader.capacity_digest = "5" * 64
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        result = restarted.submit_opening(
            self.command(key="order-2", decision="decision-2")
        )
        self.assertEqual(result.state, "SUBMITTED")

    def test_process_crash_after_place_claim_is_never_retried(self):
        self.harness.add(
            preview_result(), SystemExit("simulated crash")
        )

        with self.assertRaises(SystemExit):
            self.gateway.submit_opening(self.command())

        blockers = self.ledger.reconciliation_blockers(
            ACCOUNT_ID, "sandbox"
        )
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_preview_warning_fails_before_place_and_releases_reservation(self):
        self.harness.add(
            preview_result(
                messages={
                    "description": "possible duplicate",
                    "code": 1042,
                    "type": "WARNING",
                }
            )
        )

        result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "FAILED")
        self.assertEqual(result.reason_code, "PREVIEW_REVIEW_REQUIRED")
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            self.ledger.get_margin_reservation(result.intent_id).state,
            "RELEASED",
        )

    def test_lease_expiry_after_preview_resets_without_place(self):
        def expire_after_preview():
            self.reader.selected_hook = lambda: self.clock.advance(31)
            return preview_result()

        self.harness.add(expire_after_preview)

        with self.assertRaises(OrderIntentReconciliationRequired):
            self.gateway.submit_opening(self.command())

        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            self.ledger.reconciliation_blockers(
                ACCOUNT_ID, "sandbox"
            ),
            (),
        )

    def test_runtime_expiry_after_preview_fails_before_place(self):
        boundary = RuntimeSafetyBoundary(
            "production",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(seconds=1),
        )
        ledger = OrderIntentLedger(
            Path(self.temporary.name) / "expiring" / "orders.sqlite3",
            clock=self.clock,
            run_id="expiring-run",
        )
        reader = FakeReader(self.clock, self.account, "production")
        gateway = EtradeOrderGateway(
            runtime_safety=boundary,
            ledger=ledger,
            transport=self.make_transport(ledger, boundary),
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        gateway.start()

        def expire_after_preview():
            reader.selected_hook = lambda: self.clock.advance(2)
            return preview_result()

        self.harness.add(expire_after_preview)
        with self.assertRaises(RuntimeSafetyError):
            gateway.submit_opening(self.command())
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            ledger.active_reserved_margin(ACCOUNT_ID, "production"),
            Decimal("0"),
        )

    def test_runtime_expiry_after_place_claim_blocks_before_socket_io(self):
        boundary = RuntimeSafetyBoundary(
            "production",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(seconds=1),
        )
        ledger = OrderIntentLedger(
            Path(self.temporary.name) / "post-claim" / "orders.sqlite3",
            clock=self.clock,
            run_id="post-claim-run",
        )
        reader = FakeReader(self.clock, self.account, "production")
        gateway = EtradeOrderGateway(
            runtime_safety=boundary,
            ledger=ledger,
            transport=self.make_transport(ledger, boundary),
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        gateway.start()
        original = ledger.claim_transport_send

        def claim(request, authorization):
            original(request, authorization)
            if request.transport_operation == "SUBMIT_PLACE":
                self.clock.advance(2)

        self.harness.add(preview_result())
        with patch.object(
            ledger, "claim_transport_send", side_effect=claim
        ):
            with self.assertRaises(RuntimeSafetyError):
                gateway.submit_opening(self.command())

        blockers = ledger.reconciliation_blockers(
            ACCOUNT_ID, "production"
        )
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.harness.count("/orders/place"), 0)

    def test_response_persistence_failure_returns_durable_unknown(self):
        self.harness.add(preview_result(), place_result())
        original = self.ledger.record_transport_response

        def persist(request, response):
            if request.transport_operation == "SUBMIT_PLACE":
                raise RuntimeError("simulated durable write failure")
            return original(request, response)

        with patch.object(
            self.ledger,
            "record_transport_response",
            side_effect=persist,
        ):
            result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(
            result.reason_code,
            "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
        )
        self.assertEqual(
            self.ledger.get_intent(result.intent_id).pending_operation,
            "SUBMIT",
        )

    def test_amendment_response_persistence_failure_is_durable_unknown(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        original = self.ledger.record_transport_response

        def persist(request, response):
            if request.transport_operation == "AMEND_PLACE":
                raise RuntimeError("simulated durable write failure")
            return original(request, response)

        with patch.object(
            self.ledger,
            "record_transport_response",
            side_effect=persist,
        ):
            result = self.gateway.reprice_opening(
                RepriceOpeningCommand(
                    intent_id=submitted.intent_id,
                    idempotency_key="reprice-persistence-failure",
                    payload_bytes=payload_bytes(1.50),
                    owner=OWNER,
                )
            )

        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")
        self.assertEqual(
            result.reason_code,
            "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
        )
        self.assertEqual(
            self.ledger.get_intent(result.intent_id).pending_operation,
            "AMEND",
        )

    def test_successful_reprice_uses_real_change_transport(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        command = RepriceOpeningCommand(
            intent_id=submitted.intent_id,
            idempotency_key="reprice-1",
            payload_bytes=payload_bytes(1.50),
            owner=OWNER,
        )
        result = self.gateway.reprice_opening(command)
        replay = self.gateway.reprice_opening(command)

        self.assertEqual(result.state, "SUBMITTED")
        self.assertEqual(result.broker_order_id, "95")
        self.assertEqual(replay.reason_code, "IDEMPOTENT_REPLAY")
        self.assertFalse(replay.created)
        self.assertEqual(self.harness.count("/change/preview"), 1)
        self.assertEqual(self.harness.count("/change/place"), 1)
        current = self.ledger.get_intent(submitted.intent_id)
        self.assertIsNone(current.pending_operation)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="9" * 64,
            order_payload_hash=order_payload_hash(1.50),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.expected_order_payload_hash(submitted.intent_id),
            order_payload_hash(1.50),
        )

    def test_no_id_amendment_timeout_remains_hard_blocker(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            _ExchangeResult("TIMEOUT"),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-timeout",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")

        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])

    def test_process_crash_after_amendment_place_claim_is_not_retried(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            SystemExit("simulated amendment crash"),
        )

        with self.assertRaises(SystemExit):
            self.gateway.reprice_opening(
                RepriceOpeningCommand(
                    intent_id=submitted.intent_id,
                    idempotency_key="reprice-crash",
                    payload_bytes=payload_bytes(1.50),
                    owner=OWNER,
                )
            )

        self.assertEqual(
            self.ledger.get_intent(submitted.intent_id).pending_operation,
            "AMEND",
        )
        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])
        self.assertEqual(self.harness.count("/change/place"), 1)

    def test_known_amendment_hold_reconciles_replacement_open(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(
                order_id="95",
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                },
            ),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-hold",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="6" * 64,
            order_payload_hash=order_payload_hash(1.50),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        current = ledger.get_intent(submitted.intent_id)
        self.assertEqual(current.broker_order_id, "95")
        self.assertIsNone(current.pending_operation)

    def test_open_order_with_old_terms_does_not_clear_pending_amendment(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(
                order_id="95",
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                },
            ),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-old-terms",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="8" * 64,
            order_payload_hash=order_payload_hash(1.25),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(
            ledger.get_intent(result.intent_id).pending_operation,
            "AMEND",
        )

    def test_submitted_order_is_revalidated_on_restart(self):
        submitted = self.submit_success()
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="7" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(reader.query_calls[0][1], "94")
        self.assertEqual(
            ledger.get_intent(submitted.intent_id).last_reconciled_run,
            "gateway-run-b",
        )

    def test_not_found_known_order_keeps_gateway_read_only(self):
        self.submit_success()
        restarted, _, reader = self.restart()

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(reader.query_calls[0][1], "94")

    def test_incomplete_capacity_blocks_before_any_preview(self):
        self.reader.capacity_complete = False

        with self.assertRaises(GatewayValidationError):
            self.gateway.submit_opening(self.command())

        self.assertEqual(self.harness.calls, [])

    def test_gateway_owned_risk_ceiling_cannot_be_raised_by_a_command(self):
        limited = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=self.ledger,
            transport=self.transport,
            reader=self.reader,
            opening_risk_budget=Decimal("600"),
            clock=self.clock,
        )
        limited.start()
        self.harness.add(preview_result(), place_result())
        limited.submit_opening(self.command())

        with self.assertRaises(OrderIntentReservationError):
            limited.submit_opening(
                self.command(key="order-2", decision="decision-2")
            )

        self.assertNotIn(
            "risk_budget",
            SubmitOpeningCommand.__dataclass_fields__,
        )
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_strict_command_json_rejects_duplicates_and_nonfinite_values(self):
        for raw in (
            b'{"securityType":"OPTN","securityType":"OPTN"}',
            b'{"limitPrice":NaN}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(GatewayValidationError):
                    self.gateway.submit_opening(
                        replace(self.command(), payload_bytes=raw)
                    )
        self.assertEqual(self.harness.calls, [])

    def test_account_or_boundary_mismatch_is_rejected_before_mutation(self):
        other_reader = FakeReader(
            self.clock,
            SelectedBrokerAccount(
                "999999999", "other-key", INSTITUTION_TYPE
            ),
        )
        with self.assertRaises(RuntimeSafetyError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=self.ledger,
                transport=self.transport,
                reader=other_reader,
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            ).start()
        other_ledger = OrderIntentLedger(
            Path(self.temporary.name) / "other" / "orders.sqlite3",
            clock=self.clock,
            run_id="other-run",
        )
        with self.assertRaises(GatewayValidationError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=other_ledger,
                transport=self.transport,
                reader=self.reader,
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            )
        with self.assertRaises(GatewayValidationError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=self.ledger,
                transport=self.transport,
                reader=FakeReader(
                    self.clock, self.account, "production"
                ),
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            )
        self.assertEqual(self.harness.calls, [])

    def test_gateway_owns_no_duplicate_post_state_transitions(self):
        source = inspect.getsource(gateway_module)
        self.assertNotIn(".begin_submission(", source)
        self.assertNotIn(".begin_amendment(", source)
        self.assertNotIn(".record_post_acknowledgement(", source)
        self.assertNotIn(".record_amendment_acknowledgement(", source)

    def test_gateway_result_repr_redacts_broker_identifiers(self):
        result = self.submit_success()
        rendered = repr(result)
        self.assertNotIn("client_order_id=", rendered)
        self.assertNotIn("broker_order_id=", rendered)
        self.assertNotIn("preview_id=", rendered)


if __name__ == "__main__":
    unittest.main()
