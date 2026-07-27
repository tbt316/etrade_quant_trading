from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sqlite3
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any

from live_trading.order_intent_ledger import (
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    CancellationResponseEvidence,
    OrderIntent,
    OrderIntentLedger,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    OrderIntentValidationError,
    SCHEMA_VERSION,
)


_FIXTURE_PATH = Path(__file__).with_name("test_broker_read_ledger.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_closing_broker_read_fixture", _FIXTURE_PATH
)
if _FIXTURE_SPEC is None or _FIXTURE_SPEC.loader is None:
    raise RuntimeError("could not load broker-read test fixture")
_FIXTURE = importlib.util.module_from_spec(_FIXTURE_SPEC)
sys.modules[_FIXTURE_SPEC.name] = _FIXTURE
_FIXTURE_SPEC.loader.exec_module(_FIXTURE)


def _closing_payload() -> dict[str, Any]:
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_DEBIT",
        "limitPrice": 1.25,
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": 620,
                "orderAction": "BUY_CLOSE",
                "quantity": 1,
            },
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": 615,
                "orderAction": "SELL_CLOSE",
                "quantity": 1,
            },
        ],
    }


def _closing_intent(
    key: str, *, payload: dict[str, Any] | None = None
) -> OrderIntent:
    return OrderIntent.build(
        account_id=_FIXTURE.ACCOUNT_ID,
        environment=_FIXTURE.ENVIRONMENT,
        strategy_id="close-credit-spread",
        decision_id=f"close-{key}",
        idempotency_scope="closing-decision",
        idempotency_key=key,
        intent_kind="CLOSING",
        order_payload=payload or _closing_payload(),
    )


def _closing_order_raw(
    broker_order_id: str,
    *,
    outcome: str,
    replacement_linked: bool = False,
    partial_fill: bool = False,
) -> bytes:
    status = "EXECUTED" if outcome == "FILLED" else outcome
    document = json.loads(
        _FIXTURE._known_order_raw(
            broker_order_id,
            status=status,
        )
    )
    order = document["OrdersResponse"]["Order"][0]
    detail = order["OrderDetail"][0]
    detail["priceType"] = "NET_DEBIT"
    detail["limitPrice"] = "1.25"
    instruments = detail["Instrument"]
    instruments[0]["orderAction"] = "BUY_CLOSE"
    instruments[1]["orderAction"] = "SELL_CLOSE"
    if replacement_linked:
        order["replacedByOrderId"] = str(int(broker_order_id) + 1)
    if partial_fill:
        for instrument in instruments:
            instrument["filledQuantity"] = "0.5"
            instrument["cancelQuantity"] = "0.5"
    return _FIXTURE._canonical_json(document).encode("ascii")


class ClosingOrderLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _FIXTURE.BrokerReadLedgerTests("runTest")
        self.fixture.setUp()
        self.ledger = self.fixture.ledger
        self.clock = self.fixture.clock
        self.path = self.fixture.path

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _positions(self, broker_order_id: str = "9000"):
        return _FIXTURE._filled_vertical_positions(broker_order_id)

    def _capacity(self, *, positions=None, schema="etrade-capacity.v3"):
        return self.fixture._capacity_manifest(
            positions=self._positions() if positions is None else positions,
            schema=schema,
        )[0]

    def _create_close(
        self,
        key: str,
        *,
        positions=None,
        capacity=None,
    ):
        evidence = capacity or self._capacity(positions=positions)
        envelope = _closing_intent(key)
        result = self.ledger.create_closing_intent_from_read(
            envelope, evidence
        )
        return result.intent, envelope, evidence

    def _closing_order_manifest(
        self,
        *,
        broker_order_id: str,
        outcome: str,
        replacement_linked: bool = False,
        partial_fill: bool = False,
    ):
        raw = _closing_order_raw(
            broker_order_id,
            outcome=outcome,
            replacement_linked=replacement_linked,
            partial_fill=partial_fill,
        )
        binding_start = self.fixture._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_FIXTURE._account_list_raw(),
        )
        detail = self.fixture._record_response(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{_FIXTURE.ACCOUNT_ID_KEY}/orders/"
                f"{broker_order_id}.json"
            ),
            raw=raw,
            target_broker_order_id=broker_order_id,
        )
        parsed = self.fixture._persisted_parsed(detail.receipt_sha256)
        binding_end = self.fixture._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_FIXTURE._account_list_raw(),
            final_response=True,
        )
        result = {
            "schema": "etrade-order-query.v2",
            "broker_order_id": broker_order_id,
            "raw_status": parsed["raw_status"],
            "outcome": parsed["outcome"],
            "fill_summary": parsed["fill_summary"],
            "order_payload_hashes": parsed["order_payload_hashes"],
            "http_status": 200,
            "raw_response_digest": hashlib.sha256(raw).hexdigest(),
            "not_found": parsed["not_found"],
            "replacement_links": parsed["replacement_links"],
        }
        return self.ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="ORDER_QUERY",
                account_id=_FIXTURE.ACCOUNT_ID,
                account_id_key=_FIXTURE.ACCOUNT_ID_KEY,
                institution_type=_FIXTURE.INSTITUTION_TYPE,
                environment=_FIXTURE.ENVIRONMENT,
                origin=_FIXTURE.ORIGIN,
                target_broker_order_id=broker_order_id,
                observed_at=self.clock.now,
                completeness="COMPLETE",
                canonical_result_json=_FIXTURE._canonical_json(result),
            ),
            (
                BrokerReadManifestMember(
                    "binding.start", binding_start.receipt_sha256
                ),
                BrokerReadManifestMember(
                    "order.detail", detail.receipt_sha256
                ),
                BrokerReadManifestMember(
                    "binding.end", binding_end.receipt_sha256
                ),
            ),
        )

    def _submitted_close(self, key: str, broker_order_id: str):
        record, envelope, capacity = self._create_close(key)
        lease = self.ledger.claim_submission(
            record.intent_id, "close-worker", lease_seconds=60
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "close-worker", lease.fencing_token
        )
        self.ledger.begin_submission(
            record.intent_id,
            "close-worker",
            lease.fencing_token,
            authorization,
        )
        open_read = self._closing_order_manifest(
            broker_order_id=broker_order_id,
            outcome="OPEN",
        )
        evidence = self.ledger.broker_evidence_from_read(
            record.intent_id, open_read, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(evidence)
        submitted = self.ledger.reconcile_open(
            record.intent_id, evidence
        )
        self.assertEqual(submitted.state, "SUBMITTED")
        return record, envelope, capacity, open_read

    def test_closing_vertical_price_orientation_is_exact(self) -> None:
        envelope = _closing_intent("valid-orientation")
        self.assertEqual(envelope.intent_kind, "CLOSING")

        wrong_direction = _closing_payload()
        wrong_direction["priceType"] = "NET_CREDIT"
        with self.assertRaises(OrderIntentValidationError):
            _closing_intent("wrong-direction", payload=wrong_direction)

        zero_debit = _closing_payload()
        zero_debit["limitPrice"] = 0
        with self.assertRaises(OrderIntentValidationError):
            _closing_intent("zero-debit", payload=zero_debit)

        width_exceeded = _closing_payload()
        width_exceeded["limitPrice"] = 5.01
        with self.assertRaises(OrderIntentValidationError):
            _closing_intent("width-exceeded", payload=width_exceeded)

    def test_creation_requires_v3_and_denies_overlapping_contracts(self) -> None:
        v2 = self._capacity(
            positions=[], schema="etrade-capacity.v2"
        )
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.create_closing_intent_from_read(
                _closing_intent("v2-denied"), v2
            )

        self.clock.now += timedelta(seconds=1)
        first, _, _ = self._create_close("first")
        self.assertIsNotNone(
            self.ledger.get_closing_reservation(first.intent_id)
        )
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.create_closing_intent_from_read(
                _closing_intent("same-contract-second"),
                self._capacity(),
            )

        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM order_intents"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM closing_reservations"
                ).fetchone()[0],
                1,
            )

    def test_capacity_identity_and_available_lots_fail_closed(self) -> None:
        mutations = (
            ("adjusted", lambda positions: positions[0].__setitem__(
                "options_adjusted_flag", True
            )),
            ("multiplier", lambda positions: positions[0].__setitem__(
                "option_multiplier", "10"
            )),
            ("osi", lambda positions: positions[0].__setitem__(
                "osi_key", "SPY---260821P00621000"
            )),
            ("available", lambda positions: positions[0]["lots"][0].__setitem__(
                "available_quantity", "0"
            )),
        )
        for key, mutate in mutations:
            with self.subTest(key=key):
                positions = copy.deepcopy(self._positions())
                mutate(positions)
                with self.assertRaises(
                    (
                        AssertionError,
                        OrderIntentReservationError,
                        OrderIntentReconciliationRequired,
                    )
                ):
                    capacity = self._capacity(positions=positions)
                    self.ledger.create_closing_intent_from_read(
                        _closing_intent(f"invalid-{key}"), capacity
                    )
                self.clock.now += timedelta(seconds=1)

    def test_stale_unclaimed_restart_is_abandoned_not_submitted(self) -> None:
        record, envelope, old_capacity = self._create_close(
            "crash-before-claim"
        )
        self.clock.now += timedelta(seconds=301)
        restarted = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="closing-restart",
        )
        self.fixture.ledger = restarted
        self.ledger = restarted
        replay = restarted.find_intent(envelope)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.state, "INTENT")
        with self.assertRaises(OrderIntentReservationError):
            restarted.claim_submission(
                record.intent_id, "restart-worker", lease_seconds=60
            )

        failed = restarted.abandon_stale_closing_intent(
            record.intent_id
        )
        self.assertEqual(failed.state, "FAILED")
        self.assertEqual(
            restarted.abandon_stale_closing_intent(record.intent_id).state,
            "FAILED",
        )
        deterministic = restarted.create_closing_intent_from_read(
            envelope, old_capacity
        )
        self.assertFalse(deterministic.created)
        self.assertEqual(deterministic.intent.state, "FAILED")

        fresh_capacity = self._capacity()
        replacement = restarted.create_closing_intent_from_read(
            _closing_intent("crash-replacement"), fresh_capacity
        )
        self.assertTrue(replacement.created)

    def test_pre_post_failure_void_is_append_only(self) -> None:
        record, _, _ = self._create_close("pre-post-void")
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.abandon_stale_closing_intent(record.intent_id)
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=60
        )
        failed = self.ledger.mark_pre_post_failed(
            record.intent_id, "worker", lease.fencing_token
        )
        self.assertEqual(failed.state, "FAILED")
        self.assertIsNotNone(
            self.ledger.get_closing_reservation(record.intent_id)
        )
        with sqlite3.connect(self.path) as connection:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    """
                    UPDATE closing_reservation_voids
                    SET voided_at = voided_at + 1
                    WHERE intent_id = ?
                    """,
                    (record.intent_id,),
                )
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    """
                    DELETE FROM closing_reservation_voids
                    WHERE intent_id = ?
                    """,
                    (record.intent_id,),
                )

    def test_schema_13_migrates_closing_tables_and_triggers(self) -> None:
        closing_triggers = (
            "prevent_closing_reservation_update",
            "prevent_closing_reservation_delete",
            "prevent_closing_void_update",
            "prevent_closing_void_delete",
            "prevent_closing_absorption_update",
            "prevent_closing_absorption_delete",
            "validate_closing_reservation_insert",
            "validate_closing_void_insert",
            "validate_closing_absorption_insert",
        )
        with sqlite3.connect(self.path) as connection:
            for trigger in closing_triggers:
                connection.execute(f"DROP TRIGGER {trigger}")
            for table in (
                "closing_reservation_absorptions",
                "closing_reservation_voids",
                "closing_reservations",
            ):
                connection.execute(f"DROP TABLE {table}")
            connection.execute(
                "UPDATE ledger_metadata SET schema_version = 13"
            )

        migrated = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="closing-schema-migration",
        )
        self.fixture.ledger = migrated
        self.ledger = migrated
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                    """
                )
            }
            triggers = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'trigger'
                    """
                )
            }
        self.assertTrue(
            {
                "closing_reservations",
                "closing_reservation_voids",
                "closing_reservation_absorptions",
            }.issubset(tables)
        )
        self.assertTrue(set(closing_triggers).issubset(triggers))

    def test_full_fill_requires_newer_exact_position_absorption(self) -> None:
        record, _, _, _ = self._submitted_close(
            "full-fill", "9100"
        )
        self.clock.now += timedelta(seconds=1)
        terminal = self._closing_order_manifest(
            broker_order_id="9100",
            outcome="FILLED",
        )
        evidence = self.ledger.broker_evidence_from_read(
            record.intent_id, terminal, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(evidence)
        self.ledger.reconcile_terminal(
            record.intent_id, "FILLED", evidence
        )
        requirement = self.ledger.closing_absorption_requirement(
            record.intent_id, terminal
        )
        self.assertEqual(requirement.classification, "FULL_FILL")
        self.assertTrue(requirement.post_capacity_required)
        self.assertEqual(
            [item.intent_id for item in self.ledger.closing_reservation_blockers(
                _FIXTURE.ACCOUNT_ID, _FIXTURE.ENVIRONMENT
            )],
            [record.intent_id],
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.absorb_closing_reservation(
                record.intent_id, terminal
            )

        self.clock.now += timedelta(seconds=40)
        unchanged_positions = self._capacity()
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.absorb_closing_reservation(
                record.intent_id,
                terminal,
                post_capacity_evidence=unchanged_positions,
            )

        self.clock.now += timedelta(seconds=40)
        post_fill = self._capacity(positions=[])
        receipt = self.ledger.absorb_closing_reservation(
            record.intent_id,
            terminal,
            post_capacity_evidence=post_fill,
        )
        self.assertEqual(receipt.classification, "FULL_FILL")
        self.assertEqual(receipt.filled_quantity, 1)
        self.assertTrue(receipt.canonical_position_proof_json != "[]")
        self.assertEqual(
            self.ledger.closing_reservation_blockers(
                _FIXTURE.ACCOUNT_ID, _FIXTURE.ENVIRONMENT
            ),
            (),
        )

    def test_closing_cancellation_is_zero_fill_exact_and_absorbed(self) -> None:
        record, _, _, open_read = self._submitted_close(
            "cancel-close", "9200"
        )
        authorization = self.ledger.authorize_cancellation(
            record.intent_id,
            "cancel-close-key",
            "cancel-worker",
            60,
            open_read,
        )
        request = self.fixture._cancel_request(authorization)
        self.ledger.claim_cancellation_send(request, authorization)
        self.ledger.record_cancellation_response(
            request,
            CancellationResponseEvidence(
                disposition="REQUEST_ACCEPTED",
                http_status=200,
                message_codes=(5011,),
                message_types=("WARNING",),
                message_description_digests=("c" * 64,),
                raw_response_digest="d" * 64,
                observed_at=self.clock.now,
                unknown_reason=None,
            ),
        )

        self.clock.now += timedelta(seconds=1)
        replacement = self._closing_order_manifest(
            broker_order_id="9200",
            outcome="CANCELLED",
            replacement_linked=True,
        )
        self.assertEqual(
            self.ledger.classify_cancellation_read(
                record.intent_id, replacement
            ).outcome,
            "UNRESOLVED",
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.complete_cancellation(
                record.intent_id, replacement
            )

        self.clock.now += timedelta(seconds=1)
        with self.assertRaises(AssertionError):
            self._closing_order_manifest(
                broker_order_id="9200",
                outcome="CANCELLED",
                partial_fill=True,
            )

        self.clock.now += timedelta(seconds=1)
        terminal = self._closing_order_manifest(
            broker_order_id="9200",
            outcome="CANCELLED",
        )
        self.assertEqual(
            self.ledger.classify_cancellation_read(
                record.intent_id, terminal
            ).outcome,
            "CANCELLED",
        )
        evidence = self.ledger.broker_evidence_from_read(
            record.intent_id, terminal, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(evidence)
        self.ledger.reconcile_terminal(
            record.intent_id, "CANCELLED", evidence
        )
        completed = self.ledger.complete_cancellation(
            record.intent_id, terminal
        )
        self.assertEqual(completed.state, "TERMINAL")
        self.assertEqual(
            self.ledger.closing_absorption_requirement(
                record.intent_id, terminal
            ).classification,
            "ZERO_FILL",
        )
        receipt = self.ledger.absorb_closing_reservation(
            record.intent_id, terminal
        )
        self.assertEqual(receipt.classification, "ZERO_FILL")
        self.assertEqual(receipt.canonical_position_proof_json, "[]")
        self.assertEqual(
            self.ledger.closing_reservation_blockers(
                _FIXTURE.ACCOUNT_ID, _FIXTURE.ENVIRONMENT
            ),
            (),
        )
        self.assertEqual(
            self.ledger.get_cancellation(record.intent_id).state,
            "TERMINAL",
        )


if __name__ == "__main__":
    unittest.main()
