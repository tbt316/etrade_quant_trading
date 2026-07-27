from __future__ import annotations

import os
import json
import hashlib
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

from live_trading.order_intent_ledger import (
    SCHEMA_VERSION,
    BrokerEvidence,
    AccountCapacityEvidence,
    OutboundAuthorization,
    TransportRequestEvidence,
    TransportResponseEvidence,
    OrderIntent,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLeaseConflict,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    OrderIntentTransitionError,
    OrderIntentValidationError,
    RiskEvidence,
    canonical_order_payload,
    normalize_order_payload,
    stable_client_order_id,
    wire_order_payload,
)


class AlwaysEqualAuthorization(OutboundAuthorization):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualBytes(bytes):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualInt(int):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysSmallDecimal(Decimal):
    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False


class BypassRiskEvidence(RiskEvidence):
    def validate(self, now):
        return None


class BypassCapacityEvidence(AccountCapacityEvidence):
    def validate(self, now):
        return None


class BypassBrokerEvidence(BrokerEvidence):
    def validate(self, now):
        return None


class EvilDateTime(datetime):
    def __sub__(self, other):
        return timedelta(0)

    def __gt__(self, other):
        return False


def hostile_authorizations(authorization):
    malicious_bytes = b'{"malicious":"substitution"}'
    return (
        (
            "authorization_subclass",
            AlwaysEqualAuthorization(
                intent_id=authorization.intent_id,
                operation=authorization.operation,
                owner=authorization.owner,
                fencing_token=authorization.fencing_token,
                client_order_id=authorization.client_order_id,
                payload_bytes=malicious_bytes,
                payload_digest="0" * 64,
            ),
        ),
        (
            "custom_str",
            OutboundAuthorization(
                intent_id=AlwaysEqualStr("malicious-intent"),
                operation=authorization.operation,
                owner=authorization.owner,
                fencing_token=authorization.fencing_token,
                client_order_id=authorization.client_order_id,
                payload_bytes=authorization.payload_bytes,
                payload_digest=authorization.payload_digest,
            ),
        ),
        (
            "custom_bytes",
            OutboundAuthorization(
                intent_id=authorization.intent_id,
                operation=authorization.operation,
                owner=authorization.owner,
                fencing_token=authorization.fencing_token,
                client_order_id=authorization.client_order_id,
                payload_bytes=AlwaysEqualBytes(malicious_bytes),
                payload_digest=hashlib.sha256(malicious_bytes).hexdigest(),
            ),
        ),
    )


class Clock:
    def __init__(self):
        self.now = datetime(2026, 7, 26, 16, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def raw_payload(strike=620):
    return {
        "securityType": "OPTN", "orderAction": "SPREAD",
        "priceType": "NET_CREDIT", "limitPrice": 1.25, "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {"symbol": "SPY", "callPut": "PUT", "expiryYear": 2026, "expiryMonth": 8, "expiryDay": 21, "strikePrice": strike, "orderAction": "SELL_OPEN", "quantity": 1},
            {"symbol": "SPY", "callPut": "PUT", "expiryYear": 2026, "expiryMonth": 8, "expiryDay": 21, "strikePrice": strike - 5, "orderAction": "BUY_OPEN", "quantity": 1},
        ],
    }


def closing_payload():
    payload = raw_payload()
    payload["legs"][0]["orderAction"] = "SELL_CLOSE"
    payload["legs"][1]["orderAction"] = "BUY_CLOSE"
    return payload


def make_intent(*, account="acct-1", environment="production", key="key-1", decision="decision-1", kind="OPENING", strike=620):
    order_payload = raw_payload(strike)
    if kind == "CLOSING":
        order_payload["legs"][0]["orderAction"] = "SELL_CLOSE"
        order_payload["legs"][1]["orderAction"] = "BUY_CLOSE"
    return OrderIntent.build(
        account_id=account, environment=environment, strategy_id="credit-spread", decision_id=decision,
        idempotency_scope="decision", idempotency_key=key, intent_kind=kind, order_payload=order_payload,
    )


def risk(record, clock, *, collateral="500", max_loss="400", quote_time=None):
    return RiskEvidence(
        decision_id=record.envelope.decision_id,
        max_loss_amount=Decimal(max_loss), collateral_amount=Decimal(collateral),
        quote_observed_at=quote_time or clock.now,
        quote_digest="a" * 64, portfolio_observed_at=quote_time or clock.now, portfolio_snapshot_digest="b" * 64,
    )


def capacity(clock, *, account="acct-1", environment="production", observed_at=None, buying_power="1000", risk_budget="1000", digest="b" * 64):
    return AccountCapacityEvidence(account_id=account, environment=environment, broker_buying_power=Decimal(buying_power), risk_budget=Decimal(risk_budget), observed_at=observed_at or clock.now, portfolio_snapshot_digest=digest)


def evidence(record, clock, *, operation="ORDER_QUERY", outcome="OPEN", broker_order_id="broker-1", client_order_id=None, observed_at=None):
    return BrokerEvidence(
        account_id=record.envelope.account_id, environment=record.envelope.environment,
        client_order_id=client_order_id or record.client_order_id, broker_order_id=broker_order_id,
        operation=operation, outcome=outcome, observed_at=observed_at or clock.now,
        http_status=200, raw_response_digest="c" * 64,
    )


def transport_request(
    authorization,
    *,
    operation="SUBMIT_PREVIEW",
    account_id="acct-1",
    account_id_key="account-key",
    institution_type="BROKERAGE",
    target_broker_order_id=None,
    preview_id=None,
):
    body = f"<{operation}/>".encode("ascii")
    if operation.startswith("AMEND"):
        route = (
            f"/v1/accounts/{account_id_key}/orders/"
            f"{target_broker_order_id}/change/"
            f"{'place' if operation.endswith('PLACE') else 'preview'}"
        )
        method = "PUT"
        authorization_operation = "AMEND"
    else:
        route = (
            f"/v1/accounts/{account_id_key}/orders/"
            f"{'place' if operation.endswith('PLACE') else 'preview'}"
        )
        method = "POST"
        authorization_operation = "SUBMIT"
    return TransportRequestEvidence(
        account_id=account_id,
        account_id_key=account_id_key,
        institution_type=institution_type,
        environment="production",
        intent_id=authorization.intent_id,
        owner=authorization.owner,
        authorization_operation=authorization_operation,
        fencing_token=authorization.fencing_token,
        transport_operation=operation,
        http_method=method,
        route=route,
        client_order_id=authorization.client_order_id,
        target_broker_order_id=target_broker_order_id,
        preview_id=preview_id,
        authorization_payload_digest=authorization.payload_digest,
        final_xml_bytes=body,
        final_xml_sha256=hashlib.sha256(body).hexdigest(),
    )


class OrderIntentLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.chmod(self.tmp.name, 0o700)
        self.path = Path(self.tmp.name) / "runtime" / "orders.sqlite3"
        self.clock = Clock()
        self.ledger = OrderIntentLedger(self.path, clock=self.clock, run_id="run-a")

    def tearDown(self):
        self.tmp.cleanup()

    def opening(self, *, key="key-1", account="acct-1"):
        record = self.ledger.create_intent(make_intent(key=key, account=account)).intent
        self.ledger.set_reservation_cap(capacity(self.clock, account=account))
        self.ledger.reserve_margin(record.intent_id, risk(record, self.clock))
        return record

    def begin_submission(self, record, owner, lease):
        authorization = self.ledger.prepare_submission_payload(record.intent_id, owner, lease.fencing_token)
        return self.ledger.begin_submission(record.intent_id, owner, lease.fencing_token, authorization)

    def begin_amendment(self, record, owner, lease):
        authorization = self.ledger.prepare_amendment_payload(record.intent_id, owner, lease.fencing_token)
        return self.ledger.begin_amendment(record.intent_id, owner, lease.fencing_token, authorization)

    def test_wire_payload_is_actual_broker_schema_and_hash_representation_is_separate(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        prepared = self.ledger.prepare_submission_payload(record.intent_id, "worker", lease.fencing_token)
        payload = json.loads(prepared.payload_bytes)
        self.assertIsInstance(payload["limitPrice"], float)
        self.assertIsInstance(payload["legs"][0]["strikePrice"], int)
        self.assertIsInstance(payload["legs"][0]["quantity"], int)
        self.assertEqual(payload["client_order_id"], record.client_order_id)
        self.assertNotIn("__number__", str(prepared))
        self.assertIn("__number__", record.envelope.canonical_payload)

    def test_outbound_and_transport_evidence_repr_redacts_order_identity(self):
        authorization = OutboundAuthorization(
            intent_id="intent-safe",
            operation="AMEND",
            owner="worker-safe",
            fencing_token=7,
            client_order_id="client-secret",
            payload_bytes=b'{"order":"payload-secret"}',
            payload_digest="a" * 64,
        )
        request = TransportRequestEvidence(
            account_id="account-secret",
            account_id_key="account-key-secret",
            institution_type="BROKERAGE",
            environment="production",
            intent_id="intent-safe",
            owner="worker-safe",
            authorization_operation="AMEND",
            fencing_token=7,
            transport_operation="AMEND_PLACE",
            http_method="PUT",
            route="/route/account-key-secret/target-secret",
            client_order_id="client-secret",
            target_broker_order_id="target-secret",
            preview_id="preview-secret",
            authorization_payload_digest="a" * 64,
            final_xml_bytes=b"<payload-secret/>",
            final_xml_sha256="b" * 64,
        )
        response = TransportResponseEvidence(
            disposition="UNKNOWN",
            http_status=200,
            broker_status="EXECUTED",
            broker_order_id="broker-secret",
            preview_id="preview-secret",
            message_codes=(),
            message_types=(),
            message_description_digests=(),
            raw_response_digest="c" * 64,
            observed_at=self.clock.now,
            unknown_reason="RECONCILIATION_REQUIRED",
        )

        rendered = repr((authorization, request, response))

        for secret in (
            "client-secret",
            "payload-secret",
            "account-secret",
            "account-key-secret",
            "target-secret",
            "preview-secret",
            "broker-secret",
        ):
            self.assertNotIn(secret, rendered)

    def test_generated_vertical_payload_is_normalized_without_hashing_ignored_fields(self):
        generated = raw_payload()
        generated.update({"client_order_id": 1234567890, "orderType": "SPREADS", "required_margin": 500})
        normalized = normalize_order_payload(generated)
        self.assertNotIn("client_order_id", normalized)
        self.assertNotIn("orderType", normalized)
        self.assertNotIn("required_margin", normalized)
        record = OrderIntent.build(
            account_id="acct", environment="production", strategy_id="s", decision_id="d",
            idempotency_scope="scope", idempotency_key="key", intent_kind="OPENING", order_payload=generated,
        )
        self.assertNotIn("symbol", json.loads(record.wire_payload))
        invalid = dict(normalized)
        invalid["symbol"] = "SPY"
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(
                account_id="acct", environment="production", strategy_id="s", decision_id="d",
                idempotency_scope="scope", idempotency_key="other", intent_kind="OPENING", order_payload=invalid,
            )

    def test_idempotency_scope_is_not_payload_hash_and_client_id_binds_account_environment(self):
        first = self.ledger.create_intent(make_intent()).intent
        retry = self.ledger.create_intent(make_intent())
        repeated = self.ledger.create_intent(make_intent(key="key-2", decision="decision-2")).intent
        other = self.ledger.create_intent(make_intent(account="acct-2", key="key-1")).intent
        self.assertFalse(retry.created)
        self.assertEqual(first.envelope.payload_hash, repeated.envelope.payload_hash)
        self.assertNotEqual(first.client_order_id, repeated.client_order_id)
        self.assertNotEqual(first.client_order_id, other.client_order_id)
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.create_intent(make_intent(strike=615))

    def test_environment_exposure_and_broker_schema_are_strictly_derived(self):
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="live", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="OPENING", order_payload=raw_payload())
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="CLOSING", order_payload=raw_payload())
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(
                account_id="acct",
                environment="production",
                strategy_id="s",
                decision_id="d",
                idempotency_scope="scope",
                idempotency_key="closing",
                intent_kind="CLOSING",
                order_payload=closing_payload(),
            )
        malformed = raw_payload()
        malformed["legs"][0]["quantity"] = 0
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="OPENING", order_payload=malformed)

    def test_equity_buy_and_sell_are_fail_closed_without_position_evidence(self):
        for action in ("BUY", "SELL"):
            payload = {
                "symbol": "SPY", "securityType": "EQ", "orderAction": action,
                "quantity": 100, "priceType": "MARKET", "orderTerm": "GOOD_FOR_DAY",
            }
            with self.assertRaises(OrderIntentValidationError):
                OrderIntent.build(
                    account_id="acct", environment="production", strategy_id="s", decision_id="d",
                    idempotency_scope="scope", idempotency_key=f"equity-{action}",
                    intent_kind="OPENING", order_payload=payload,
                )

    def test_opening_vertical_price_semantics_are_bounded_and_debit_reserve_is_derived(self):
        for price_type, limit_price in (("NET_DEBIT", 10.0), ("MARKET", None), ("LIMIT", 1.0)):
            payload = raw_payload()
            payload["priceType"] = price_type
            if limit_price is None:
                payload.pop("limitPrice")
            else:
                payload["limitPrice"] = limit_price
            with self.assertRaises(OrderIntentValidationError):
                OrderIntent.build(
                    account_id="acct", environment="production", strategy_id="s", decision_id="d",
                    idempotency_scope="scope", idempotency_key=f"invalid-{price_type}",
                    intent_kind="OPENING", order_payload=payload,
                )
        debit = raw_payload()
        debit["priceType"] = "NET_DEBIT"
        debit["limitPrice"] = 1.25
        debit["legs"][0]["orderAction"] = "BUY_OPEN"
        debit["legs"][1]["orderAction"] = "SELL_OPEN"
        record = self.ledger.create_intent(OrderIntent.build(
            account_id="acct-1", environment="production", strategy_id="credit-spread", decision_id="debit",
            idempotency_scope="decision", idempotency_key="debit", intent_kind="OPENING", order_payload=debit,
        )).intent
        self.ledger.set_reservation_cap(capacity(self.clock))
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, collateral="124.99", max_loss="124.99"))
        reserved = self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, collateral="125", max_loss="125"))
        self.assertEqual(reserved.amount, Decimal("125"))

    def test_opening_vertical_price_type_must_match_put_and_call_risk_orientation(self):
        short_put_as_debit = raw_payload()
        short_put_as_debit["priceType"] = "NET_DEBIT"
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="put-debit", intent_kind="OPENING", order_payload=short_put_as_debit)
        long_put_as_credit = raw_payload()
        long_put_as_credit["legs"][0]["orderAction"] = "BUY_OPEN"
        long_put_as_credit["legs"][1]["orderAction"] = "SELL_OPEN"
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="put-credit", intent_kind="OPENING", order_payload=long_put_as_credit)
        long_call_as_credit = raw_payload()
        for leg in long_call_as_credit["legs"]:
            leg["callPut"] = "CALL"
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="call-credit", intent_kind="OPENING", order_payload=long_call_as_credit)
        short_call_as_debit = raw_payload()
        for leg in short_call_as_debit["legs"]:
            leg["callPut"] = "CALL"
        short_call_as_debit["legs"][0]["strikePrice"] = 615
        short_call_as_debit["legs"][1]["strikePrice"] = 620
        short_call_as_debit["priceType"] = "NET_DEBIT"
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="call-debit", intent_kind="OPENING", order_payload=short_call_as_debit)
        malformed = raw_payload()
        malformed["legs"][1]["expiryDay"] = 31
        malformed["legs"][1]["expiryMonth"] = 2
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="OPENING", order_payload=malformed)

    def test_risk_evidence_is_required_positive_fresh_and_bound_to_decision(self):
        record = self.ledger.create_intent(make_intent()).intent
        self.ledger.set_reservation_cap(capacity(self.clock))
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, collateral="0", max_loss="0"))
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, collateral="1", max_loss="1"))
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, quote_time=self.clock.now - timedelta(minutes=6)))
        bad = risk(record, self.clock)
        object.__setattr__(bad, "decision_id", "other-decision")
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reserve_margin(record.intent_id, bad)
        reservation = self.ledger.reserve_margin(record.intent_id, risk(record, self.clock))
        self.assertEqual(reservation.max_loss_amount, Decimal("400"))
        self.assertEqual(reservation.quote_digest, "a" * 64)
        changed = risk(record, self.clock)
        object.__setattr__(changed, "quote_digest", "e" * 64)
        with self.assertRaises(OrderIntentTransitionError):
            self.ledger.reserve_margin(record.intent_id, changed)

    def test_evidence_rejects_numeric_identity_datetime_and_validate_subclasses(self):
        record = self.ledger.create_intent(make_intent()).intent
        normal_capacity = capacity(self.clock)
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.set_reservation_cap(
                BypassCapacityEvidence(**normal_capacity.__dict__)
            )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.set_reservation_cap(
                AccountCapacityEvidence(
                    account_id="acct-1",
                    environment="production",
                    broker_buying_power=AlwaysSmallDecimal("1"),
                    risk_budget=AlwaysSmallDecimal("1"),
                    observed_at=self.clock.now,
                    portfolio_snapshot_digest="b" * 64,
                )
            )
        self.ledger.set_reservation_cap(normal_capacity)

        normal_risk = risk(record, self.clock)
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(
                record.intent_id, BypassRiskEvidence(**normal_risk.__dict__)
            )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(
                record.intent_id,
                RiskEvidence(
                    decision_id=record.envelope.decision_id,
                    max_loss_amount=AlwaysSmallDecimal("1"),
                    collateral_amount=AlwaysSmallDecimal("1"),
                    quote_observed_at=self.clock.now,
                    quote_digest="a" * 64,
                    portfolio_observed_at=self.clock.now,
                    portfolio_snapshot_digest="b" * 64,
                ),
            )
        with localcontext() as decimal_context:
            decimal_context.prec = 1
            with self.assertRaises(OrderIntentReservationError):
                self.ledger.reserve_margin(
                    record.intent_id,
                    risk(record, self.clock, collateral="1", max_loss="1"),
                )

        self.ledger.reserve_margin(record.intent_id, normal_risk)
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", lease)
        normal_broker = evidence(record, self.clock, operation="SUBMIT_ACK")
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.record_post_acknowledgement(
                record.intent_id,
                "worker",
                lease.fencing_token,
                BypassBrokerEvidence(**normal_broker.__dict__),
            )
        hostile_identity = evidence(record, self.clock, operation="SUBMIT_ACK")
        object.__setattr__(
            hostile_identity,
            "client_order_id",
            AlwaysEqualStr(record.client_order_id),
        )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.record_post_acknowledgement(
                record.intent_id, "worker", lease.fencing_token, hostile_identity
            )
        hostile_time = evidence(record, self.clock, operation="SUBMIT_ACK")
        object.__setattr__(
            hostile_time,
            "observed_at",
            EvilDateTime(
                self.clock.now.year,
                self.clock.now.month,
                self.clock.now.day,
                self.clock.now.hour,
                self.clock.now.minute,
                tzinfo=timezone.utc,
            ),
        )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.record_post_acknowledgement(
                record.intent_id, "worker", lease.fencing_token, hostile_time
            )

    def test_claim_rejects_stale_persisted_capacity_and_risk_evidence(self):
        record = self.opening()
        self.clock.advance(301)
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)

    def test_atomic_concurrent_reservation_cap_and_same_account_claim_blocker(self):
        self.ledger.set_reservation_cap(capacity(self.clock))
        left = self.ledger.create_intent(make_intent(key="left")).intent
        right = self.ledger.create_intent(make_intent(key="right")).intent
        barrier = threading.Barrier(2)
        def reserve(record):
            barrier.wait()
            try:
                self.ledger.reserve_margin(record.intent_id, risk(record, self.clock, collateral="600", max_loss="400"))
                return True
            except OrderIntentReservationError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, (left, right)))
        self.assertEqual(sum(results), 1)
        winner = left if self.ledger.get_margin_reservation(left.intent_id) else right
        other = right if winner is left else left
        self.ledger.reserve_margin(
            other.intent_id,
            risk(other, self.clock, collateral="400", max_loss="400"),
        )
        lease = self.ledger.claim_submission(winner.intent_id, "worker-a", lease_seconds=30)
        self.begin_submission(winner, "worker-a", lease)
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.claim_submission(other.intent_id, "worker-b", lease_seconds=30)

    def test_capacity_snapshots_are_ordered_can_reach_zero_and_block_new_claims(self):
        record = self.ledger.create_intent(make_intent()).intent
        self.ledger.set_reservation_cap(capacity(self.clock))
        self.ledger.reserve_margin(record.intent_id, risk(record, self.clock))
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.set_reservation_cap(capacity(self.clock, buying_power="0", risk_budget="0"))
        self.clock.advance(1)
        self.assertEqual(self.ledger.set_reservation_cap(capacity(self.clock, buying_power="0", risk_budget="0")), Decimal("0"))
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        blocked = self.ledger.create_intent(make_intent(key="zero-cap", decision="zero-cap")).intent
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(blocked.intent_id, risk(blocked, self.clock))

    def test_identical_equal_time_capacity_is_idempotent_but_conflict_fails(self):
        snapshot = capacity(self.clock)

        self.assertEqual(
            self.ledger.set_reservation_cap(snapshot),
            Decimal("1000"),
        )
        self.assertEqual(
            self.ledger.set_reservation_cap(snapshot),
            Decimal("1000"),
        )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.set_reservation_cap(
                capacity(self.clock, digest="d" * 64)
            )

    def test_reservation_requires_the_capacity_snapshot_used_for_its_portfolio_risk(self):
        record = self.ledger.create_intent(make_intent()).intent
        self.ledger.set_reservation_cap(capacity(self.clock, digest="d" * 64))
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reserve_margin(record.intent_id, risk(record, self.clock))

    def test_expired_claim_without_place_attempt_returns_to_intent_with_new_fence(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=5)
        self.clock.advance(5)
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.prepare_submission_payload(record.intent_id, "worker", lease.fencing_token)
        self.assertEqual(self.ledger.get_intent(record.intent_id).state, "INTENT")
        replacement = self.ledger.claim_submission(
            record.intent_id, "other", lease_seconds=5
        )
        self.assertGreater(replacement.fencing_token, lease.fencing_token)

    def test_expired_preview_only_submission_returns_to_intent(self):
        record = self.opening()
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=5
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "worker", lease.fencing_token
        )
        self.ledger.claim_transport_send(
            transport_request(authorization), authorization
        )

        self.clock.advance(6)

        self.assertEqual(
            self.ledger.reconciliation_blockers("acct-1", "production"), ()
        )
        self.assertEqual(self.ledger.get_intent(record.intent_id).state, "INTENT")
        replacement = self.ledger.claim_submission(
            record.intent_id, "other", lease_seconds=5
        )
        self.assertGreater(replacement.fencing_token, lease.fencing_token)

    def test_place_must_reuse_exact_preview_account_route_identity(self):
        record = self.opening()
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "worker", lease.fencing_token
        )
        preview = transport_request(authorization)
        self.ledger.claim_transport_send(preview, authorization)
        self.ledger.record_transport_response(
            preview,
            TransportResponseEvidence(
                disposition="ACKNOWLEDGED",
                http_status=200,
                broker_status="OPEN",
                broker_order_id=None,
                preview_id="preview-1",
                message_codes=(),
                message_types=(),
                message_description_digests=(),
                raw_response_digest="c" * 64,
                observed_at=self.clock.now,
                unknown_reason=None,
            ),
        )
        rebound_place = transport_request(
            authorization,
            operation="SUBMIT_PLACE",
            account_id_key="other-account-key",
            institution_type="OTHER_BROKER",
            preview_id="preview-1",
        )

        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.claim_transport_send(
                rebound_place, authorization
            )

        self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")

    def test_preview_receipt_requires_independently_validated_ack(self):
        record = self.opening()
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "worker", lease.fencing_token
        )
        preview = transport_request(authorization)
        self.ledger.claim_transport_send(preview, authorization)
        invalid_responses = (
            TransportResponseEvidence(
                disposition="ACKNOWLEDGED",
                http_status=200,
                broker_status="REJECTED",
                broker_order_id=None,
                preview_id="preview-1",
                message_codes=(),
                message_types=(),
                message_description_digests=(),
                raw_response_digest="c" * 64,
                observed_at=self.clock.now,
                unknown_reason=None,
            ),
            TransportResponseEvidence(
                disposition="ACKNOWLEDGED",
                http_status=200,
                broker_status="OPEN",
                broker_order_id=None,
                preview_id="preview-1",
                message_codes=(1042,),
                message_types=("WARNING",),
                message_description_digests=("d" * 64,),
                raw_response_digest="c" * 64,
                observed_at=self.clock.now,
                unknown_reason=None,
            ),
        )

        for invalid in invalid_responses:
            with self.subTest(
                status=invalid.broker_status,
                messages=invalid.message_codes,
            ):
                with self.assertRaises(OrderIntentValidationError):
                    self.ledger.record_transport_response(
                        preview, invalid
                    )

        with sqlite3.connect(self.path) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM broker_preview_receipts"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM transport_response_receipts"
                ).fetchone()[0],
                0,
            )

    def test_expired_unattempted_amendment_releases_lease(self):
        record = self.opening()
        submit = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", submit)
        submitted = self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            submit.fencing_token,
            evidence(record, self.clock, operation="SUBMIT_ACK"),
        )
        amendment = self.ledger.acquire_amendment_lease(
            submitted.intent_id,
            "nudger",
            lease_seconds=5,
            idempotency_key="amend-unattempted",
            amendment_payload=raw_payload(),
        )

        self.clock.advance(6)

        self.assertEqual(
            self.ledger.reconciliation_blockers("acct-1", "production"), ()
        )
        replacement = self.ledger.acquire_amendment_lease(
            submitted.intent_id,
            "other",
            lease_seconds=5,
            idempotency_key="amend-unattempted",
            amendment_payload=raw_payload(),
        )
        self.assertGreater(replacement.fencing_token, amendment.fencing_token)

    def test_expired_preview_only_amendment_releases_lease(self):
        record = self.opening()
        submit = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", submit)
        submitted = self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            submit.fencing_token,
            evidence(record, self.clock, operation="SUBMIT_ACK"),
        )
        amendment = self.ledger.acquire_amendment_lease(
            submitted.intent_id,
            "nudger",
            lease_seconds=5,
            idempotency_key="amend-preview",
            amendment_payload=raw_payload(),
        )
        authorization = self.ledger.prepare_amendment_payload(
            submitted.intent_id, "nudger", amendment.fencing_token
        )
        self.ledger.claim_transport_send(
            transport_request(
                authorization,
                operation="AMEND_PREVIEW",
                target_broker_order_id=submitted.broker_order_id,
            ),
            authorization,
        )

        self.clock.advance(6)

        self.assertEqual(
            self.ledger.reconciliation_blockers("acct-1", "production"), ()
        )
        replacement = self.ledger.acquire_amendment_lease(
            submitted.intent_id,
            "other",
            lease_seconds=5,
            idempotency_key="amend-preview",
            amendment_payload=raw_payload(),
        )
        self.assertGreater(replacement.fencing_token, amendment.fencing_token)

    def test_submission_begin_rejects_economic_or_client_id_authorization_tampering(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        authorization = self.ledger.prepare_submission_payload(record.intent_id, "worker", lease.fencing_token)
        mutations = (
            ("limitPrice", lambda payload: payload.__setitem__("limitPrice", 1.50)),
            ("priceType", lambda payload: payload.__setitem__("priceType", "NET_DEBIT")),
            ("orderTerm", lambda payload: payload.__setitem__("orderTerm", "IMMEDIATE_OR_CANCEL")),
            ("spreadType", lambda payload: payload.__setitem__("spreadType", "CALENDAR")),
            ("symbol", lambda payload: payload["legs"][0].__setitem__("symbol", "QQQ")),
            ("callPut", lambda payload: payload["legs"][0].__setitem__("callPut", "CALL")),
            ("expiry", lambda payload: payload["legs"][0].__setitem__("expiryDay", 28)),
            ("strikePrice", lambda payload: payload["legs"][0].__setitem__("strikePrice", 625)),
            ("orderAction", lambda payload: payload["legs"][0].__setitem__("orderAction", "BUY_OPEN")),
            ("quantity", lambda payload: payload["legs"][0].__setitem__("quantity", 2)),
            ("client_order_id", lambda payload: payload.__setitem__("client_order_id", "9999999999")),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                payload = json.loads(authorization.payload_bytes)
                mutate(payload)
                payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                forged = OutboundAuthorization(
                    intent_id=authorization.intent_id,
                    operation=authorization.operation,
                    owner=authorization.owner,
                    fencing_token=authorization.fencing_token,
                    client_order_id=payload["client_order_id"],
                    payload_bytes=payload_bytes,
                    payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
                )
                with self.assertRaises(OrderIntentIntegrityError):
                    self.ledger.begin_submission(record.intent_id, "worker", lease.fencing_token, forged)
                self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")
        for attack, hostile in hostile_authorizations(authorization):
            with self.subTest(attack=attack):
                with self.assertRaises(OrderIntentValidationError):
                    self.ledger.begin_submission(record.intent_id, "worker", lease.fencing_token, hostile)
                self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")
        self.ledger.begin_submission(record.intent_id, "worker", lease.fencing_token, authorization)
        with sqlite3.connect(self.path) as conn:
            persisted = conn.execute(
                "SELECT payload_bytes, payload_digest FROM outbound_authorizations WHERE intent_id = ? AND operation = 'SUBMIT'",
                (record.intent_id,),
            ).fetchone()
        self.assertEqual(persisted, (authorization.payload_bytes, authorization.payload_digest))

    def test_owner_and_fence_primitives_are_exact_at_every_mutation_boundary(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "worker", lease.fencing_token
        )
        hostile_owner = AlwaysEqualStr("worker")
        hostile_fence = AlwaysEqualInt(lease.fencing_token)
        for operation in (
            lambda: self.ledger.renew_submission_lease(
                record.intent_id, "worker", hostile_fence, lease_seconds=30
            ),
            lambda: self.ledger.prepare_submission_payload(
                record.intent_id, hostile_owner, lease.fencing_token
            ),
            lambda: self.ledger.begin_submission(
                record.intent_id, "worker", hostile_fence, authorization
            ),
            lambda: self.ledger.mark_pre_post_failed(
                record.intent_id, "worker", hostile_fence
            ),
            lambda: self.ledger.mark_pre_post_failed(
                record.intent_id, "worker", True
            ),
        ):
            with self.assertRaises(OrderIntentValidationError):
                operation()
            self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")

        self.ledger.begin_submission(
            record.intent_id, "worker", lease.fencing_token, authorization
        )
        submit_evidence = evidence(record, self.clock, operation="SUBMIT_ACK")
        for owner, token in (
            (hostile_owner, lease.fencing_token),
            ("worker", hostile_fence),
        ):
            with self.assertRaises(OrderIntentValidationError):
                self.ledger.record_post_acknowledgement(
                    record.intent_id, owner, token, submit_evidence
                )
            with self.assertRaises(OrderIntentValidationError):
                self.ledger.record_post_unknown(
                    record.intent_id, owner, token, "POST_TIMEOUT"
                )
        self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            submit_evidence,
        )

        amendment = self.ledger.acquire_amendment_lease(
            record.intent_id,
            "nudger",
            lease_seconds=30,
            idempotency_key="primitive-fence",
            amendment_payload=raw_payload(),
        )
        amendment_authorization = self.ledger.prepare_amendment_payload(
            record.intent_id, "nudger", amendment.fencing_token
        )
        hostile_amendment_fence = AlwaysEqualInt(amendment.fencing_token)
        for operation in (
            lambda: self.ledger.prepare_amendment_payload(
                record.intent_id, "nudger", hostile_amendment_fence
            ),
            lambda: self.ledger.begin_amendment(
                record.intent_id,
                "nudger",
                hostile_amendment_fence,
                amendment_authorization,
            ),
            lambda: self.ledger.release_amendment_lease(
                record.intent_id, "nudger", hostile_amendment_fence
            ),
        ):
            with self.assertRaises(OrderIntentValidationError):
                operation()
        self.ledger.begin_amendment(
            record.intent_id,
            "nudger",
            amendment.fencing_token,
            amendment_authorization,
        )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.record_amendment_acknowledgement(
                record.intent_id,
                AlwaysEqualStr("nudger"),
                amendment.fencing_token,
                evidence(
                    record,
                    self.clock,
                    operation="AMEND_ACK",
                    client_order_id=amendment.client_order_id,
                    broker_order_id="primitive-amended",
                ),
            )

    def test_amendment_begin_rejects_economic_or_client_id_authorization_tampering(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-auth", amendment_payload=raw_payload())
        authorization = self.ledger.prepare_amendment_payload(record.intent_id, "nudger", amendment.fencing_token)
        for field, value in (("limitPrice", 1.50), ("client_order_id", "9999999999")):
            with self.subTest(field=field):
                payload = json.loads(authorization.payload_bytes)
                payload[field] = value
                payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                forged = OutboundAuthorization(
                    intent_id=authorization.intent_id,
                    operation=authorization.operation,
                    owner=authorization.owner,
                    fencing_token=authorization.fencing_token,
                    client_order_id=payload["client_order_id"],
                    payload_bytes=payload_bytes,
                    payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
                )
                with self.assertRaises(OrderIntentIntegrityError):
                    self.ledger.begin_amendment(record.intent_id, "nudger", amendment.fencing_token, forged)
        for attack, hostile in hostile_authorizations(authorization):
            with self.subTest(attack=attack):
                with self.assertRaises(OrderIntentValidationError):
                    self.ledger.begin_amendment(record.intent_id, "nudger", amendment.fencing_token, hostile)
        self.ledger.begin_amendment(record.intent_id, "nudger", amendment.fencing_token, authorization)
        with sqlite3.connect(self.path) as conn:
            persisted = conn.execute(
                "SELECT payload_bytes, payload_digest FROM outbound_authorizations WHERE intent_id = ? AND operation = 'AMEND'",
                (record.intent_id,),
            ).fetchone()
        self.assertEqual(persisted, (authorization.payload_bytes, authorization.payload_digest))

    def test_fabricated_stale_or_wrong_identity_broker_evidence_cannot_acknowledge(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", lease)
        stale = evidence(record, self.clock, operation="SUBMIT_ACK", observed_at=self.clock.now - timedelta(minutes=6))
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.record_post_acknowledgement(record.intent_id, "worker", lease.fencing_token, stale)
        wrong = evidence(record, self.clock, operation="SUBMIT_ACK", client_order_id="1000000000")
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.record_post_acknowledgement(record.intent_id, "worker", lease.fencing_token, wrong)
        ack = self.ledger.record_post_acknowledgement(record.intent_id, "worker", lease.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        self.assertEqual(ack.state, "SUBMITTED")

    def test_restart_reconciliation_keeps_terminal_reservation_until_r7b_position_evidence(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", lease)
        self.ledger.record_post_unknown(record.intent_id, "worker", lease.fencing_token, "POST_TIMEOUT")
        restarted = OrderIntentLedger(self.path, clock=self.clock, run_id="run-b")
        self.assertEqual([item.intent_id for item in restarted.reconciliation_blockers("acct-1", "production")], [record.intent_id])
        filled = restarted.reconcile_terminal(record.intent_id, "FILLED", evidence(record, self.clock, outcome="FILLED"))
        self.assertEqual(filled.state, "FILLED")
        self.assertEqual(restarted.get_margin_reservation(record.intent_id).state, "FILLED_PENDING_ABSORPTION")
        next_opening = restarted.create_intent(make_intent(key="next-opening", decision="next-decision")).intent
        restarted.reserve_margin(next_opening.intent_id, risk(next_opening, self.clock))
        with self.assertRaises(OrderIntentReservationError):
            restarted.claim_submission(next_opening.intent_id, "new-worker", lease_seconds=30)
        self.clock.advance(1)
        with self.assertRaises(OrderIntentReconciliationRequired):
            restarted.absorb_filled_reservation(record.intent_id, capacity(self.clock))
        with self.assertRaises(OrderIntentReconciliationRequired):
            restarted.absorb_filled_reservation(
                record.intent_id, capacity(self.clock, digest="d" * 64)
            )
        self.assertEqual(
            restarted.get_margin_reservation(record.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        event = next(item for item in restarted.events(record.intent_id) if item.reason_code == "BROKER_FILLED")
        self.assertEqual(event.raw_response_digest, "c" * 64)
        self.assertEqual((event.account_id, event.environment, event.client_order_id), ("acct-1", "production", record.client_order_id))
        self.assertEqual((event.evidence_operation, event.http_status, event.broker_status), ("ORDER_QUERY", 200, "FILLED"))

    def test_amendment_is_persisted_before_post_fenced_and_expiry_is_reconciliation_only(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        submitted = self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(submitted.intent_id, "nudger", lease_seconds=5, idempotency_key="amend-1", amendment_payload=raw_payload())
        begun = self.begin_amendment(submitted, "nudger", amendment)
        self.assertEqual(begun.client_order_id, amendment.client_order_id)
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.acquire_amendment_lease(submitted.intent_id, "other", lease_seconds=5, idempotency_key="amend-1", amendment_payload=raw_payload())
        self.clock.advance(6)
        self.assertEqual(self.ledger.reconciliation_blockers("acct-1", "production")[0].intent_id, submitted.intent_id)
        self.ledger.reconcile_terminal(submitted.intent_id, "CANCELLED", evidence(submitted, self.clock, operation="AMEND_QUERY", outcome="CANCELLED", client_order_id=amendment.client_order_id))
        with sqlite3.connect(self.path) as conn:
            completion = conn.execute("SELECT completion_state FROM amendment_history WHERE intent_id = ? AND idempotency_key = ?", (submitted.intent_id, "amend-1")).fetchone()
        self.assertEqual(completion, ("TERMINAL",))

    def test_amendment_acknowledgement_requires_its_own_client_id_and_fence(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-1", amendment_payload=raw_payload())
        prepared = self.ledger.prepare_amendment_payload(record.intent_id, "nudger", amendment.fencing_token)
        prepared_payload = json.loads(prepared.payload_bytes)
        self.assertEqual(prepared_payload["client_order_id"], amendment.client_order_id)
        self.assertIsInstance(prepared_payload["limitPrice"], float)
        self.ledger.begin_amendment(record.intent_id, "nudger", amendment.fencing_token, prepared)
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.record_amendment_acknowledgement(record.intent_id, "nudger", amendment.fencing_token, evidence(record, self.clock, operation="AMEND_ACK"))
        amended = self.ledger.record_amendment_acknowledgement(record.intent_id, "nudger", amendment.fencing_token, evidence(record, self.clock, operation="AMEND_ACK", client_order_id=amendment.client_order_id, broker_order_id="broker-2"))
        self.assertEqual(amended.broker_order_id, "broker-2")
        changed = raw_payload()
        changed["limitPrice"] = 1.2
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-1", amendment_payload=changed)
        with self.assertRaises(OrderIntentTransitionError):
            self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-1", amendment_payload=raw_payload())
        with sqlite3.connect(self.path) as conn:
            history = conn.execute("SELECT target_broker_order_id, client_order_id, payload_hash FROM amendment_history WHERE intent_id = ? AND idempotency_key = ?", (record.intent_id, "amend-1")).fetchone()
        self.assertEqual(history[:2], ("broker-1", amendment.client_order_id))
        self.assertEqual(len(history[2]), 64)

    def test_original_intent_rejects_known_client_id_collision_with_completed_amendment(self):
        source = self.opening(key="source")
        submit = self.ledger.claim_submission(source.intent_id, "worker", lease_seconds=30)
        self.begin_submission(source, "worker", submit)
        self.ledger.record_post_acknowledgement(source.intent_id, "worker", submit.fencing_token, evidence(source, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(source.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-13221", amendment_payload=raw_payload())
        self.begin_amendment(source, "nudger", amendment)
        self.ledger.record_amendment_acknowledgement(source.intent_id, "nudger", amendment.fencing_token, evidence(source, self.clock, operation="AMEND_ACK", client_order_id=amendment.client_order_id, broker_order_id="broker-amended"))
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.create_intent(make_intent(key="target-210261"))

    def test_amendment_is_reprice_only_and_in_doubt_lease_cannot_be_released(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        altered = raw_payload()
        altered["legs"][0]["strikePrice"] = 615
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-1", amendment_payload=altered)
        amendment = self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-1", amendment_payload=raw_payload())
        self.begin_amendment(record, "nudger", amendment)
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.release_amendment_lease(record.intent_id, "nudger", amendment.fencing_token)

    def test_amendment_fence_never_reuses_after_completed_amendment(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        first = self.ledger.acquire_amendment_lease(record.intent_id, "worker", lease_seconds=30, idempotency_key="amend-a", amendment_payload=raw_payload())
        self.begin_amendment(record, "worker", first)
        self.ledger.record_amendment_acknowledgement(record.intent_id, "worker", first.fencing_token, evidence(record, self.clock, operation="AMEND_ACK", client_order_id=first.client_order_id, broker_order_id="broker-a2"))
        replacement = raw_payload()
        replacement["limitPrice"] = 1.2
        second = self.ledger.acquire_amendment_lease(record.intent_id, "worker", lease_seconds=30, idempotency_key="amend-b", amendment_payload=replacement)
        self.assertGreater(second.fencing_token, first.fencing_token)
        with self.assertRaises(OrderIntentLeaseConflict):
            self.ledger.begin_amendment(record.intent_id, "worker", first.fencing_token, self.ledger.prepare_amendment_payload(record.intent_id, "worker", first.fencing_token))

    def test_stale_same_owner_release_cannot_delete_newer_amendment_lease(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        first = self.ledger.acquire_amendment_lease(record.intent_id, "worker", lease_seconds=30, idempotency_key="amend-a", amendment_payload=raw_payload())
        self.ledger.release_amendment_lease(record.intent_id, "worker", first.fencing_token)
        replacement = raw_payload()
        replacement["limitPrice"] = 1.2
        second = self.ledger.acquire_amendment_lease(record.intent_id, "worker", lease_seconds=30, idempotency_key="amend-b", amendment_payload=replacement)
        with self.assertRaises(OrderIntentLeaseConflict):
            self.ledger.release_amendment_lease(record.intent_id, "worker", first.fencing_token)
        prepared = self.ledger.prepare_amendment_payload(record.intent_id, "worker", second.fencing_token)
        self.assertEqual(json.loads(prepared.payload_bytes)["client_order_id"], second.client_order_id)

    def test_opening_reprice_cannot_increase_exposure_past_active_reservation(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        risk_increasing = raw_payload()
        risk_increasing["limitPrice"] = 0.10  # floor rises from $375 to $490.
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="increase-risk", amendment_payload=risk_increasing)
        risk_reducing = raw_payload()
        risk_reducing["limitPrice"] = 1.50  # floor falls to $350.
        lease = self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=30, idempotency_key="reduce-risk", amendment_payload=risk_reducing)
        self.assertEqual(lease.intent_id, record.intent_id)

    def test_amendment_lease_blocks_other_submissions_and_post_start_rechecks_risk(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=600)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(record.intent_id, "nudger", lease_seconds=600, idempotency_key="lease-block", amendment_payload=raw_payload())
        other = self.ledger.create_intent(make_intent(key="other-opening", decision="other-opening")).intent
        self.ledger.reserve_margin(
            other.intent_id,
            risk(other, self.clock, collateral="400", max_loss="400"),
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.claim_submission(other.intent_id, "other", lease_seconds=30)
        self.clock.advance(301)
        with self.assertRaises(OrderIntentReservationError):
            self.begin_amendment(record, "nudger", amendment)

    def test_begin_submission_rechecks_capacity_after_claim(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=600)
        self.clock.advance(1)
        self.ledger.set_reservation_cap(capacity(self.clock, buying_power="0", risk_budget="0"))
        with self.assertRaises(OrderIntentReservationError):
            self.begin_submission(record, "worker", submit)
        self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")

    def test_cancelled_and_expired_openings_hold_reservation_without_position_evidence(self):
        for terminal in ("CANCELLED", "EXPIRED"):
            with self.subTest(terminal=terminal):
                self.clock.advance(1)
                account = f"acct-{terminal.lower()}"
                record = self.opening(key=f"{terminal}-opening", account=account)
                submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
                self.begin_submission(record, "worker", submit)
                self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK", broker_order_id=f"{terminal}-broker"))
                self.ledger.reconcile_terminal(record.intent_id, terminal, evidence(record, self.clock, operation="ORDER_QUERY", outcome=terminal, broker_order_id=f"{terminal}-broker"))
                self.assertEqual(self.ledger.get_margin_reservation(record.intent_id).state, "FILLED_PENDING_ABSORPTION")
                blocked = self.ledger.create_intent(make_intent(account=account, key=f"{terminal}-blocked", decision=f"{terminal}-blocked")).intent
                self.ledger.reserve_margin(blocked.intent_id, risk(blocked, self.clock))
                with self.assertRaises(OrderIntentReservationError):
                    self.ledger.claim_submission(blocked.intent_id, "blocked", lease_seconds=30)
                self.clock.advance(1)
                with self.assertRaises(OrderIntentReconciliationRequired):
                    self.ledger.absorb_filled_reservation(
                        record.intent_id, capacity(self.clock, account=account)
                    )
                with self.assertRaises(OrderIntentReconciliationRequired):
                    self.ledger.absorb_filled_reservation(
                        record.intent_id,
                        capacity(self.clock, account=account, digest="d" * 64),
                    )
                with self.assertRaises(OrderIntentReservationError):
                    self.ledger.claim_submission(blocked.intent_id, "still-blocked", lease_seconds=30)

    def test_concurrent_amendment_race_has_one_winner(self):
        record = self.opening()
        submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", submit)
        self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK"))
        barrier = threading.Barrier(2)
        def acquire(owner):
            barrier.wait()
            try:
                return self.ledger.acquire_amendment_lease(record.intent_id, owner, lease_seconds=30, idempotency_key="amend-1", amendment_payload=raw_payload()).owner
            except OrderIntentLeaseConflict:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(acquire, ("nudger-a", "nudger-b")))
        self.assertEqual(sum(value is not None for value in outcomes), 1)

    def test_schema_8_migrates_without_losing_the_ledger(self):
        legacy_wire = wire_order_payload(closing_payload())
        legacy_canonical = canonical_order_payload(json.loads(legacy_wire))
        legacy = OrderIntent(
            account_id="legacy-account",
            environment="production",
            strategy_id="legacy-close",
            decision_id="legacy-decision",
            idempotency_scope="legacy",
            idempotency_key="legacy-closing",
            intent_kind="CLOSING",
            wire_payload=legacy_wire,
            canonical_payload=legacy_canonical,
            payload_hash=hashlib.sha256(
                b"etrade-order-payload.v2\0" + legacy_canonical.encode("utf-8")
            ).hexdigest(),
        )
        legacy_client_id = stable_client_order_id(legacy)
        claimed_legacy = OrderIntent(
            account_id=legacy.account_id,
            environment=legacy.environment,
            strategy_id=legacy.strategy_id,
            decision_id="legacy-claimed-decision",
            idempotency_scope=legacy.idempotency_scope,
            idempotency_key="legacy-claimed",
            intent_kind=legacy.intent_kind,
            wire_payload=legacy.wire_payload,
            canonical_payload=legacy.canonical_payload,
            payload_hash=legacy.payload_hash,
        )
        claimed_client_id = stable_client_order_id(claimed_legacy)
        now = int(self.clock.now.timestamp() * 1_000_000)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO order_intents (
                    intent_id, account_id, environment, strategy_id, decision_id,
                    idempotency_scope, idempotency_key, intent_kind, wire_payload,
                    canonical_payload, payload_hash, client_order_id, state,
                    broker_order_id, submission_fence, amendment_fence,
                    submission_lease_owner, submission_lease_expires_at,
                    pending_operation, pending_owner, pending_fence,
                    last_reconciled_run, created_at, updated_at
                ) VALUES (
                    'legacy-closing-intent', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'SUBMITTED', 'legacy-broker-order', 1, 0, NULL, NULL,
                    NULL, NULL, NULL, 'legacy-run', ?, ?
                )
                """,
                (
                    legacy.account_id,
                    legacy.environment,
                    legacy.strategy_id,
                    legacy.decision_id,
                    legacy.idempotency_scope,
                    legacy.idempotency_key,
                    legacy.intent_kind,
                    legacy.wire_payload,
                    legacy.canonical_payload,
                    legacy.payload_hash,
                    legacy_client_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO order_intents (
                    intent_id, account_id, environment, strategy_id, decision_id,
                    idempotency_scope, idempotency_key, intent_kind, wire_payload,
                    canonical_payload, payload_hash, client_order_id, state,
                    broker_order_id, submission_fence, amendment_fence,
                    submission_lease_owner, submission_lease_expires_at,
                    pending_operation, pending_owner, pending_fence,
                    last_reconciled_run, created_at, updated_at
                ) VALUES (
                    'legacy-claimed-intent', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'CLAIMED', NULL, 1, 0, 'legacy-worker', ?, NULL, NULL, NULL,
                    NULL, ?, ?
                )
                """,
                (
                    claimed_legacy.account_id,
                    claimed_legacy.environment,
                    claimed_legacy.strategy_id,
                    claimed_legacy.decision_id,
                    claimed_legacy.idempotency_scope,
                    claimed_legacy.idempotency_key,
                    claimed_legacy.intent_kind,
                    claimed_legacy.wire_payload,
                    claimed_legacy.canonical_payload,
                    claimed_legacy.payload_hash,
                    claimed_client_id,
                    now + 30_000_000,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE order_intents SET amendment_fence = 1 WHERE intent_id = 'legacy-closing-intent'"
            )
            conn.execute(
                """
                INSERT INTO amendment_leases (
                    intent_id, broker_order_id, client_order_id, idempotency_key,
                    wire_payload, canonical_payload, payload_hash, owner,
                    fencing_token, state, expires_at, updated_at
                ) VALUES (
                    'legacy-closing-intent', 'legacy-broker-order', '1234567890',
                    'legacy-amendment', ?, ?, ?, 'legacy-amender', 1, 'LEASED',
                    ?, ?
                )
                """,
                (
                    legacy.wire_payload,
                    legacy.canonical_payload,
                    legacy.payload_hash,
                    now + 30_000_000,
                    now,
                ),
            )
            conn.execute("DROP TRIGGER prevent_outbound_authorization_update")
            conn.execute("DROP TRIGGER prevent_outbound_authorization_delete")
            conn.execute("DROP TABLE outbound_authorizations")
            conn.execute("UPDATE ledger_metadata SET schema_version = 8 WHERE singleton = 1")
        migrated = OrderIntentLedger(self.path, clock=self.clock, run_id="run-migration")
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT schema_version FROM ledger_metadata").fetchone()[0], SCHEMA_VERSION)
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'outbound_authorizations'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'prevent_outbound_authorization_delete'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'transport_send_attempts'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'broker_preview_receipts'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'transport_response_receipts'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'prevent_transport_send_attempt_delete'").fetchone())
            self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'prevent_transport_response_receipt_delete'").fetchone())
        legacy_record = migrated.get_intent("legacy-closing-intent")
        self.assertEqual(legacy_record.envelope.intent_kind, "CLOSING")
        claimed_record = migrated.get_intent("legacy-claimed-intent")
        claimed_payload = json.loads(claimed_legacy.wire_payload)
        claimed_payload["client_order_id"] = claimed_client_id
        claimed_bytes = json.dumps(
            claimed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        claimed_authorization = OutboundAuthorization(
            intent_id=claimed_record.intent_id,
            operation="SUBMIT",
            owner="legacy-worker",
            fencing_token=1,
            client_order_id=claimed_client_id,
            payload_bytes=claimed_bytes,
            payload_digest=hashlib.sha256(claimed_bytes).hexdigest(),
        )
        with self.assertRaises(OrderIntentReservationError):
            migrated.renew_submission_lease(
                claimed_record.intent_id,
                "legacy-worker",
                1,
                lease_seconds=30,
            )
        with self.assertRaises(OrderIntentReservationError):
            migrated.prepare_submission_payload(
                claimed_record.intent_id, "legacy-worker", 1
            )
        with self.assertRaises(OrderIntentReservationError):
            migrated.begin_submission(
                claimed_record.intent_id,
                "legacy-worker",
                1,
                claimed_authorization,
            )

        amendment_payload = json.loads(legacy.wire_payload)
        amendment_payload["client_order_id"] = "1234567890"
        amendment_bytes = json.dumps(
            amendment_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        amendment_authorization = OutboundAuthorization(
            intent_id=legacy_record.intent_id,
            operation="AMEND",
            owner="legacy-amender",
            fencing_token=1,
            client_order_id="1234567890",
            payload_bytes=amendment_bytes,
            payload_digest=hashlib.sha256(amendment_bytes).hexdigest(),
        )
        with self.assertRaises(OrderIntentReservationError):
            migrated.prepare_amendment_payload(
                legacy_record.intent_id, "legacy-amender", 1
            )
        with self.assertRaises(OrderIntentReservationError):
            migrated.begin_amendment(
                legacy_record.intent_id,
                "legacy-amender",
                1,
                amendment_authorization,
            )
        migrated.release_amendment_lease(
            legacy_record.intent_id, "legacy-amender", 1
        )
        reconciled = migrated.mark_reconciled(
            legacy_record.intent_id,
            BrokerEvidence(
                account_id=legacy.account_id,
                environment=legacy.environment,
                client_order_id=legacy_client_id,
                broker_order_id="legacy-broker-order",
                operation="ORDER_QUERY",
                outcome="OPEN",
                observed_at=self.clock.now,
                http_status=200,
                raw_response_digest="f" * 64,
            ),
        )
        self.assertEqual(reconciled.state, "SUBMITTED")
        self.assertEqual(migrated.path, self.path)

    def test_owner_only_database_schema_and_identity_tamper_trigger(self):
        self.assertEqual(os.stat(self.path.parent).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        record = self.ledger.create_intent(make_intent()).intent
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT schema_version FROM ledger_metadata").fetchone()[0], SCHEMA_VERSION)
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("UPDATE order_intents SET wire_payload = '{}' WHERE intent_id = ?", (record.intent_id,))
            conn.execute("INSERT INTO broker_order_history (broker_order_id, intent_id, first_seen_at) VALUES (?, ?, ?)", ("historic-broker", record.intent_id, 0))
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM broker_order_history WHERE broker_order_id = ?", ("historic-broker",))
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            descriptor = os.open(sidecar, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(descriptor)
            os.chmod(sidecar, 0o644)
            with self.assertRaises(OrderIntentValidationError):
                self.ledger.get_intent(record.intent_id)
            sidecar.unlink()
        os.chmod(self.path, 0o644)
        with self.assertRaises(OrderIntentValidationError):
            OrderIntentLedger(self.path, clock=self.clock, run_id="run-b")

    def test_schema_9_adds_response_receipts_and_keeps_expiry_live(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "DROP TRIGGER prevent_transport_response_receipt_update"
            )
            conn.execute(
                "DROP TRIGGER prevent_transport_response_receipt_delete"
            )
            conn.execute("DROP TABLE transport_response_receipts")
            conn.execute(
                "UPDATE ledger_metadata SET schema_version = 9 WHERE singleton = 1"
            )

        migrated = OrderIntentLedger(
            self.path, clock=self.clock, run_id="run-schema-9-migration"
        )

        with sqlite3.connect(self.path) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertIsNotNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'transport_response_receipts'"
                ).fetchone()
            )
            self.assertIsNotNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'prevent_transport_response_receipt_delete'"
                ).fetchone()
            )
        record = migrated.create_intent(
            make_intent(key="migration-live", decision="migration-live")
        ).intent
        migrated.set_reservation_cap(capacity(self.clock))
        migrated.reserve_margin(record.intent_id, risk(record, self.clock))
        lease = migrated.claim_submission(
            record.intent_id, "worker", lease_seconds=5
        )
        self.clock.advance(6)
        with self.assertRaises(OrderIntentReconciliationRequired):
            migrated.prepare_submission_payload(
                record.intent_id, "worker", lease.fencing_token
            )
        self.assertEqual(migrated.get_intent(record.intent_id).state, "INTENT")


if __name__ == "__main__":
    unittest.main()
