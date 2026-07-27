from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import struct
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from rauth import OAuth1Session
from requests.adapters import HTTPAdapter
from requests.exceptions import Timeout
from xml.etree import ElementTree as ET

from live_trading.etrade_broker_transport import (
    BrokerReply,
    ETradeBrokerTransport,
    ETradeBrokerTransportError,
    SelectedBrokerAccount,
    _EXCHANGE_RESULT_BUFFER_BYTES,
    _ExchangeResult,
    _exchange_worker,
    _isolated_exchange,
    _read_bounded_response,
    _read_exchange_result,
    _serialized_prepared_request,
    _transport_evidence,
    _transport_response_evidence,
    _write_exchange_result,
)
from live_trading.order_intent_ledger import (
    AccountCapacityEvidence,
    BrokerEvidence,
    OrderIntent,
    OrderIntentLedger,
    OrderIntentReconciliationRequired,
    OutboundAuthorization,
    RiskEvidence,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary


ACCOUNT_ID = "842468410"
ACCOUNT_KEY = "account/key"
INSTITUTION_TYPE = "BROKERAGE"
OWNER = "worker-1"


def vertical_payload(*, limit_price=1.25, symbol="SPY"):
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": limit_price,
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": symbol,
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": 620,
                "orderAction": "SELL_OPEN",
                "quantity": 1,
            },
            {
                "symbol": symbol,
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": 615,
                "orderAction": "BUY_OPEN",
                "quantity": 1,
            },
        ],
    }


class FakeResponse:
    def __init__(self, status_code=200, *, body=None, raw=None, headers=None):
        self.status_code = status_code
        self.content = (
            raw
            if raw is not None
            else json.dumps(
                body, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        self.raw = io.BytesIO(self.content)
        self.headers = headers or {}
        self.closed = False

    def close(self):
        self.closed = True
        self.raw.close()


def preview_response(*, preview_id="1020563279", account_id=ACCOUNT_ID):
    return FakeResponse(
        body={
            "PreviewOrderResponse": {
                "accountId": account_id,
                "orderType": "SPREADS",
                "PreviewIds": [{"previewId": preview_id}],
                "Order": [{"status": "OPEN"}],
            }
        }
    )


def place_response(*, order_id="94", account_id=ACCOUNT_ID, status="OPEN"):
    return FakeResponse(
        body={
            "PlaceOrderResponse": {
                "accountId": account_id,
                "orderType": "SPREADS",
                "OrderIds": [{"orderId": order_id}],
                "Order": [{"status": status}],
            }
        }
    )


class AdapterHarness:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def exchange(self, prepared, *, timeout_seconds):
        self.calls.append(
            (prepared, {"timeout_seconds": timeout_seconds})
        )
        if not self.outcomes:
            raise AssertionError("unexpected broker request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, (Timeout, TimeoutError)):
            return _ExchangeResult("TIMEOUT")
        if isinstance(outcome, BaseException):
            return _ExchangeResult("TRANSPORT_ERROR")
        try:
            status, raw = _read_bounded_response(outcome)
        except (Timeout, TimeoutError):
            return _ExchangeResult("TIMEOUT")
        except Exception:
            return _ExchangeResult("MALFORMED_RESPONSE")
        return _ExchangeResult(
            "RESPONSE", http_status=status, raw_response=raw
        )


@dataclass
class TransportCase:
    temporary: tempfile.TemporaryDirectory
    ledger: OrderIntentLedger
    transport: ETradeBrokerTransport
    adapter: AdapterHarness
    record: object
    authorization: OutboundAuthorization
    patcher: object


class ETradeBrokerTransportTests(unittest.TestCase):
    def setUp(self):
        self.cases = []

    def tearDown(self):
        for case in reversed(self.cases):
            case.patcher.stop()
            case.temporary.cleanup()

    def case(
        self,
        outcomes,
        *,
        payload=None,
        source_session=None,
        account_id=ACCOUNT_ID,
        account_key=ACCOUNT_KEY,
    ):
        temporary = tempfile.TemporaryDirectory()
        os.chmod(temporary.name, 0o700)
        ledger = OrderIntentLedger(
            Path(temporary.name) / "runtime" / "orders.sqlite3",
            clock=lambda: datetime.now(timezone.utc),
            run_id="transport-test",
        )
        payload = payload or vertical_payload()
        intent = OrderIntent.build(
            account_id=account_id,
            environment="production",
            strategy_id="credit-spread",
            decision_id="decision-1",
            idempotency_scope="decision",
            idempotency_key="key-1",
            intent_kind="OPENING",
            order_payload=payload,
        )
        record = ledger.create_intent(intent, intent_id="intent-1").intent
        observed_at = datetime.now(timezone.utc)
        ledger.set_reservation_cap(
            AccountCapacityEvidence(
                account_id=account_id,
                environment="production",
                broker_buying_power=Decimal("1000"),
                risk_budget=Decimal("1000"),
                observed_at=observed_at,
                portfolio_snapshot_digest="b" * 64,
            )
        )
        ledger.reserve_margin(
            record.intent_id,
            RiskEvidence(
                decision_id=record.envelope.decision_id,
                max_loss_amount=Decimal("400"),
                collateral_amount=Decimal("500"),
                quote_observed_at=observed_at,
                quote_digest="a" * 64,
                portfolio_observed_at=observed_at,
                portfolio_snapshot_digest="b" * 64,
            ),
        )
        lease = ledger.claim_submission(
            record.intent_id, OWNER, lease_seconds=60
        )
        authorization = ledger.prepare_submission_payload(
            record.intent_id, OWNER, lease.fencing_token
        )
        now = datetime.now(timezone.utc)
        runtime = RuntimeSafetyBoundary(
            environment="production",
            expected_account_id=account_id,
            expected_account_id_key=account_key,
            expected_institution_type=INSTITUTION_TYPE,
            arm_issued_at=now - timedelta(seconds=1),
            arm_expires_at=now + timedelta(minutes=10),
        )
        session = source_session or OAuth1Session(
            "consumer-key",
            "consumer-secret",
            access_token="access-token",
            access_token_secret="access-secret",
        )
        transport = ETradeBrokerTransport(
            session=session,
            ledger=ledger,
            runtime_safety=runtime,
            selected_account=SelectedBrokerAccount(
                account_id, account_key, INSTITUTION_TYPE
            ),
        )
        adapter = AdapterHarness(outcomes)
        patcher = patch(
            "live_trading.etrade_broker_transport._isolated_exchange",
            side_effect=adapter.exchange,
        )
        patcher.start()
        result = TransportCase(
            temporary,
            ledger,
            transport,
            adapter,
            record,
            authorization,
            patcher,
        )
        self.cases.append(result)
        return result

    def amendment_case(self, outcomes):
        case = self.case(outcomes)
        case.ledger.begin_submission(
            case.record.intent_id,
            OWNER,
            case.authorization.fencing_token,
            case.authorization,
        )
        case.ledger.record_post_acknowledgement(
            case.record.intent_id,
            OWNER,
            case.authorization.fencing_token,
            BrokerEvidence(
                account_id=ACCOUNT_ID,
                environment="production",
                client_order_id=case.record.client_order_id,
                broker_order_id="93",
                operation="SUBMIT_ACK",
                outcome="OPEN",
                observed_at=datetime.now(timezone.utc),
                http_status=200,
                raw_response_digest="c" * 64,
            ),
        )
        amended_payload = vertical_payload(limit_price=1.5)
        lease = case.ledger.acquire_amendment_lease(
            case.record.intent_id,
            OWNER,
            lease_seconds=60,
            idempotency_key="amend-1",
            amendment_payload=amended_payload,
        )
        case.authorization = case.ledger.prepare_amendment_payload(
            case.record.intent_id, OWNER, lease.fencing_token
        )
        return case

    @staticmethod
    def table_count(case, table):
        with sqlite3.connect(case.ledger.path) as connection:
            return connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

    def test_preview_then_place_is_durably_bound_and_sent_once(self):
        case = self.case([preview_response(), place_response()])

        preview = case.transport.preview(case.authorization)
        placed = case.transport.place(case.authorization, preview)

        self.assertEqual(preview.disposition, "ACKNOWLEDGED")
        self.assertEqual(placed.disposition, "ACKNOWLEDGED")
        self.assertEqual(placed.broker_order_id, "94")
        self.assertEqual(len(case.adapter.calls), 2)
        self.assertEqual(self.table_count(case, "transport_send_attempts"), 2)
        self.assertEqual(self.table_count(case, "broker_preview_receipts"), 1)
        self.assertEqual(
            self.table_count(case, "transport_response_receipts"), 2
        )
        self.assertEqual(
            [
                receipt.transport_operation
                for receipt in case.ledger.transport_response_receipts(
                    case.record.intent_id
                )
            ],
            ["SUBMIT_PREVIEW", "SUBMIT_PLACE"],
        )
        receipts = case.ledger.transport_response_receipts(
            case.record.intent_id
        )
        self.assertEqual(
            [receipt.client_order_id for receipt in receipts],
            [case.record.client_order_id, case.record.client_order_id],
        )
        self.assertEqual(
            [receipt.target_broker_order_id for receipt in receipts],
            [None, None],
        )
        prepared, kwargs = case.adapter.calls[1]
        self.assertEqual(
            prepared.url,
            "https://api.etrade.com/v1/accounts/account%2Fkey/orders/place",
        )
        self.assertEqual(bytes(prepared.body), placed.request.final_xml_bytes)
        root = ET.fromstring(placed.request.final_xml_bytes)
        self.assertEqual(root.findtext("./Order/stopPrice"), "0")
        self.assertEqual(
            [
                node.text
                for node in root.findall(
                    "./Order/Instrument/orderedQuantity"
                )
            ],
            ["1", "1"],
        )
        self.assertNotIn("?", prepared.url)
        self.assertNotIn("Cookie", prepared.headers)
        self.assertEqual(prepared.headers["Accept-Encoding"], "identity")
        self.assertEqual(kwargs["timeout_seconds"], 15.0)
        self.assertEqual(
            case.ledger.get_intent(case.record.intent_id).state,
            "SUBMITTED",
        )

        with self.assertRaises(Exception):
            case.transport.place(case.authorization, preview)
        self.assertEqual(len(case.adapter.calls), 2)

    def test_direct_place_without_stored_preview_is_blocked_before_io(self):
        case = self.case([place_response()])
        request = case.transport._build_request(
            authorization=case.authorization,
            expected_authorization_operation="SUBMIT",
            transport_operation="SUBMIT_PLACE",
            http_method="POST",
            target_broker_order_id=None,
            preview_id="1020563279",
        )

        with self.assertRaises(Exception):
            case.transport._execute(request, case.authorization)

        self.assertEqual(case.adapter.calls, [])
        self.assertEqual(self.table_count(case, "transport_send_attempts"), 0)
        self.assertEqual(
            case.ledger.get_intent(case.record.intent_id).state, "CLAIMED"
        )

    def test_forged_authorization_owner_is_rejected_by_ledger(self):
        case = self.case([preview_response()])
        forged = OutboundAuthorization(
            intent_id=case.authorization.intent_id,
            operation=case.authorization.operation,
            owner="not-the-lease-owner",
            fencing_token=case.authorization.fencing_token,
            client_order_id=case.authorization.client_order_id,
            payload_bytes=case.authorization.payload_bytes,
            payload_digest=case.authorization.payload_digest,
        )

        with self.assertRaises(Exception):
            case.transport.preview(forged)

        self.assertEqual(case.adapter.calls, [])
        self.assertEqual(self.table_count(case, "transport_send_attempts"), 0)

    def test_authorization_cannot_cross_account_identity(self):
        case = self.case([preview_response()])
        now = datetime.now(timezone.utc)
        other_runtime = RuntimeSafetyBoundary(
            "production",
            "999999999",
            "other-key",
            INSTITUTION_TYPE,
            now - timedelta(seconds=1),
            now + timedelta(minutes=10),
        )
        other = ETradeBrokerTransport(
            session=OAuth1Session(
                "consumer-key",
                "consumer-secret",
                access_token="access-token",
                access_token_secret="access-secret",
            ),
            ledger=case.ledger,
            runtime_safety=other_runtime,
            selected_account=SelectedBrokerAccount(
                "999999999", "other-key", INSTITUTION_TYPE
            ),
        )
        adapter = AdapterHarness([preview_response(account_id="999999999")])

        with self.assertRaises(Exception):
            other.preview(case.authorization)

        self.assertEqual(adapter.calls, [])

    def test_amendment_target_and_preview_receipt_are_ledger_bound(self):
        case = self.amendment_case([preview_response(), place_response(order_id="95")])

        preview = case.transport.preview_change("93", case.authorization)
        with self.assertRaises(Exception):
            case.transport.place_change("999", case.authorization, preview)
        self.assertEqual(len(case.adapter.calls), 1)

        placed = case.transport.place_change("93", case.authorization, preview)

        self.assertEqual(placed.broker_order_id, "95")
        self.assertEqual(
            placed.request.route,
            "/v1/accounts/account%2Fkey/orders/93/change/place",
        )
        self.assertEqual([call[0].method for call in case.adapter.calls], ["PUT", "PUT"])

    def test_source_session_ambient_state_cannot_reach_private_send(self):
        source = OAuth1Session(
            "consumer-key",
            "consumer-secret",
            access_token="access-token",
            access_token_secret="access-secret",
        )
        source.params = {"unexpected": "1"}
        source.proxies = {"https": "http://untrusted.invalid"}
        source.hooks["response"].append(lambda response, **_kwargs: response)
        source.cookies.set("session", "ambient")
        source.verify = False
        case = self.case([preview_response()], source_session=source)

        reply = case.transport.preview(case.authorization)

        self.assertEqual(reply.disposition, "ACKNOWLEDGED")
        prepared, kwargs = case.adapter.calls[0]
        self.assertNotIn("?", prepared.url)
        self.assertNotIn("Cookie", prepared.headers)
        self.assertEqual(kwargs["timeout_seconds"], 15.0)

    def test_isolated_worker_uses_one_pinned_no_retry_exchange(self):
        case = self.case([preview_response()])
        case.transport.preview(case.authorization)
        prepared = case.adapter.calls[0][0]

        output = bytearray(_EXCHANGE_RESULT_BUFFER_BYTES)
        response = preview_response()
        with patch.object(
            HTTPAdapter,
            "send",
            autospec=True,
            return_value=response,
        ) as send:
            _exchange_worker(
                output, _serialized_prepared_request(prepared)
            )

        self.assertEqual(send.call_count, 1)
        adapter = send.call_args.args[0]
        self.assertEqual(adapter.max_retries.total, 0)
        self.assertEqual(adapter.max_retries.connect, 0)
        self.assertEqual(adapter.max_retries.read, 0)
        self.assertEqual(adapter.max_retries.redirect, 0)
        kwargs = send.call_args.kwargs
        self.assertIs(kwargs["stream"], True)
        self.assertEqual(kwargs["timeout"], (3.05, 10.0))
        self.assertIs(kwargs["verify"], True)
        self.assertIsNone(kwargs["cert"])
        self.assertEqual(kwargs["proxies"], {})
        self.assertEqual(_read_exchange_result(output).kind, "RESPONSE")

    def test_timeout_is_unknown_and_durable_attempt_is_not_retried(self):
        case = self.case([Timeout("after possible send"), preview_response()])

        reply = case.transport.preview(case.authorization)

        self.assertEqual(
            (reply.disposition, reply.unknown_reason), ("UNKNOWN", "TIMEOUT")
        )
        self.assertEqual(len(case.adapter.calls), 1)
        self.assertEqual(
            self.table_count(case, "transport_response_receipts"), 1
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            case.transport.preview(case.authorization)
        self.assertEqual(len(case.adapter.calls), 1)

    def test_transport_error_is_unknown_and_persisted_once(self):
        case = self.case([RuntimeError("socket failure")])

        reply = case.transport.preview(case.authorization)

        self.assertEqual(
            (reply.disposition, reply.unknown_reason),
            ("UNKNOWN", "TRANSPORT_ERROR"),
        )
        self.assertEqual(len(case.adapter.calls), 1)
        self.assertEqual(
            self.table_count(case, "transport_response_receipts"), 1
        )

    def test_wrong_security_values_are_never_string_coerced(self):
        case = self.case([preview_response()])
        for payload in (
            vertical_payload(symbol=True),
            vertical_payload(symbol="spy"),
            vertical_payload(symbol="A/B"),
            vertical_payload(symbol="A" * 16),
        ):
            payload["client_order_id"] = case.authorization.client_order_id
            payload_bytes = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            forged = OutboundAuthorization(
                intent_id=case.authorization.intent_id,
                operation="SUBMIT",
                owner=OWNER,
                fencing_token=case.authorization.fencing_token,
                client_order_id=case.authorization.client_order_id,
                payload_bytes=payload_bytes,
                payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
            )
            with self.subTest(symbol=payload["legs"][0]["symbol"]):
                with self.assertRaises(ETradeBrokerTransportError):
                    case.transport.preview(forged)
        self.assertEqual(case.adapter.calls, [])

    def test_warning_or_info_message_requires_review_and_blocks_place(self):
        response = FakeResponse(
            body={
                "PreviewOrderResponse": {
                    "accountId": ACCOUNT_ID,
                    "orderType": "SPREADS",
                    "PreviewIds": [{"previewId": "1020563279"}],
                    "Order": [
                        {
                            "status": "OPEN",
                            "messages": {
                                "Message": [
                                    {
                                        "description": "possible duplicate order",
                                        "code": 1042,
                                        "type": "WARNING",
                                    }
                                ]
                            },
                        }
                    ],
                }
            }
        )
        case = self.case([response])

        reply = case.transport.preview(case.authorization)

        self.assertEqual(
            (reply.disposition, reply.unknown_reason),
            ("UNKNOWN", "REVIEW_REQUIRED"),
        )
        self.assertEqual(reply.preview_id, "1020563279")
        self.assertEqual(reply.broker_messages[0].code, 1042)
        self.assertNotIn("possible duplicate", repr(reply))
        self.assertEqual(self.table_count(case, "broker_preview_receipts"), 0)
        self.assertEqual(
            self.table_count(case, "transport_response_receipts"), 1
        )
        with self.assertRaises(ETradeBrokerTransportError):
            case.transport.place(case.authorization, reply)

    def test_place_hold_preserves_reconciliation_identifiers(self):
        held = FakeResponse(
            body={
                "PlaceOrderResponse": {
                    "accountId": ACCOUNT_ID,
                    "orderType": "SPREADS",
                    "OrderIds": [{"orderId": "94"}],
                    "Order": [
                        {
                            "status": "OPEN",
                            "messages": {
                                "Message": {
                                    "description": "manual review",
                                    "code": 4002,
                                    "type": "INFO_HOLD",
                                }
                            },
                        }
                    ],
                }
            }
        )
        case = self.case([preview_response(), held])
        preview = case.transport.preview(case.authorization)

        reply = case.transport.place(case.authorization, preview)

        self.assertEqual(
            (reply.disposition, reply.unknown_reason),
            ("UNKNOWN", "REVIEW_REQUIRED"),
        )
        self.assertEqual(reply.broker_order_id, "94")
        self.assertEqual(reply.broker_status, "OPEN")
        self.assertEqual(reply.broker_messages[0].message_type, "INFO_HOLD")
        receipt = case.ledger.transport_response_receipts(
            case.record.intent_id
        )[-1]
        self.assertEqual(receipt.transport_operation, "SUBMIT_PLACE")
        self.assertEqual(receipt.response.broker_order_id, "94")
        self.assertEqual(receipt.response.broker_status, "OPEN")
        self.assertEqual(receipt.response.message_codes, (4002,))

    def test_reviewed_place_success_warning_is_retained_and_acknowledged(self):
        successful_warning = FakeResponse(
            body={
                "PlaceOrderResponse": {
                    "accountId": ACCOUNT_ID,
                    "orderType": "SPREADS",
                    "OrderIds": [
                        {"orderId": "94", "cashMargin": "MARGIN"}
                    ],
                    "Order": [
                        {
                            "status": "OPEN",
                            "messages": {
                                "Message": {
                                    "description": "successfully entered",
                                    "code": "1026",
                                    "type": "WARNING",
                                }
                            },
                        }
                    ],
                }
            }
        )
        case = self.case([preview_response(), successful_warning])
        preview = case.transport.preview(case.authorization)

        reply = case.transport.place(case.authorization, preview)

        self.assertEqual(reply.disposition, "ACKNOWLEDGED")
        self.assertEqual(reply.broker_order_id, "94")
        self.assertEqual(reply.broker_messages[0].code, 1026)

    def test_preview_identifier_accepts_only_documented_cash_margin(self):
        valid = FakeResponse(
            body={
                "PreviewOrderResponse": {
                    "accountId": ACCOUNT_ID,
                    "orderType": "SPREADS",
                    "PreviewIds": [
                        {
                            "previewId": "1020563279",
                            "cashMargin": "MARGIN",
                        }
                    ],
                    "Order": [{"status": "OPEN"}],
                }
            }
        )
        case = self.case([valid])
        self.assertEqual(
            case.transport.preview(case.authorization).disposition,
            "ACKNOWLEDGED",
        )

        invalid = FakeResponse(
            body={
                "PreviewOrderResponse": {
                    "accountId": ACCOUNT_ID,
                    "orderType": "SPREADS",
                    "PreviewIds": [
                        {
                            "previewId": "1020563279",
                            "cashMargin": "BORROWED",
                        }
                    ],
                }
            }
        )
        case = self.case([invalid])
        self.assertEqual(
            case.transport.preview(case.authorization).disposition,
            "UNKNOWN",
        )

    def test_non_acknowledged_status_preserves_order_id(self):
        for status in ("REJECTED", "EXECUTED", "INDIVIDUAL_FILLS"):
            with self.subTest(status=status):
                case = self.case(
                    [preview_response(), place_response(status=status)]
                )
                preview = case.transport.preview(case.authorization)

                reply = case.transport.place(case.authorization, preview)

                self.assertEqual(
                    reply.unknown_reason,
                    "BROKER_STATUS_REQUIRES_RECONCILIATION",
                )
                self.assertEqual(reply.broker_order_id, "94")
                self.assertEqual(reply.broker_status, status)
                self.assertEqual(
                    case.ledger.get_intent(case.record.intent_id).state,
                    "SUBMISSION_UNKNOWN",
                )
                self.assertEqual(
                    self.table_count(
                        case, "transport_response_receipts"
                    ),
                    2,
                )

    def test_nonfinite_duplicate_and_overflow_responses_never_acknowledge(self):
        responses = (
            FakeResponse(
                raw=(
                    b'{"PreviewOrderResponse":{"accountId":"842468410",'
                    b'"orderType":"SPREADS","PreviewIds":[{"previewId":"1020563279"}],'
                    b'"totalOrderValue":NaN}}'
                )
            ),
            FakeResponse(
                raw=(
                    b'{"PreviewOrderResponse":{"accountId":"842468410",'
                    b'"accountId":"842468410","orderType":"SPREADS",'
                    b'"PreviewIds":[{"previewId":"1020563279"}]}}'
                )
            ),
            preview_response(preview_id="9999999999999999999"),
        )
        for response in responses:
            with self.subTest(body=response.content[:80]):
                case = self.case([response])
                reply = case.transport.preview(case.authorization)
                self.assertEqual(reply.disposition, "UNKNOWN")
                self.assertEqual(reply.unknown_reason, "MALFORMED_RESPONSE")
                self.assertTrue(response.closed)

    def test_account_mismatch_and_http_redirect_are_unknown_without_follow(self):
        for response, reason in (
            (preview_response(account_id="999999999"), "MALFORMED_RESPONSE"),
            (
                FakeResponse(
                    status_code=307,
                    raw=b"redirect",
                    headers={"Location": "https://attacker.invalid"},
                ),
                "HTTP_STATUS",
            ),
        ):
            with self.subTest(reason=reason):
                case = self.case([response])
                reply = case.transport.preview(case.authorization)
                self.assertEqual(reply.unknown_reason, reason)
                self.assertEqual(len(case.adapter.calls), 1)
                self.assertTrue(response.closed)

    def test_encoded_response_is_rejected_without_decoding(self):
        class GuardedRaw:
            def __init__(self):
                self.calls = []
                self.closed = False

            def read1(self, amount, *, decode_content):
                self.calls.append((amount, decode_content))
                raise AssertionError("encoded response must not be read")

            def close(self):
                self.closed = True

        response = preview_response()
        guarded = GuardedRaw()
        response.raw = guarded
        response.headers = {"Content-Encoding": "gzip"}
        case = self.case([response])

        reply = case.transport.preview(case.authorization)

        self.assertEqual(reply.unknown_reason, "MALFORMED_RESPONSE")
        self.assertEqual(guarded.calls, [])
        self.assertTrue(response.closed)

    def test_response_wall_deadline_closes_stream(self):
        response = preview_response()
        with patch(
            "live_trading.etrade_broker_transport.time.monotonic",
            side_effect=[0.0, 13.0],
        ):
            with self.assertRaises(TimeoutError):
                _read_bounded_response(response)
        self.assertTrue(response.closed)

    def test_parent_watchdog_covers_the_entire_exchange(self):
        case = self.case([preview_response()])
        case.transport.preview(case.authorization)
        prepared = case.adapter.calls[0][0]

        class Process:
            def __init__(self):
                self.started = False
                self.alive = False
                self.terminated = False
                self.join_timeouts = []

            def start(self):
                self.started = True
                self.alive = True

            def join(self, timeout):
                self.join_timeouts.append(timeout)

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.terminated = True
                self.alive = False

            def kill(self):
                self.alive = False

        class Context:
            def __init__(self):
                self.process = Process()

            def RawArray(self, typecode, size):
                self.typecode = typecode
                self.size = size
                return bytearray(size)

            def Process(self, *, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon
                return self.process

        context = Context()
        with patch(
            "live_trading.etrade_broker_transport.multiprocessing.get_context",
            return_value=context,
        ):
            result = _isolated_exchange(
                prepared, timeout_seconds=0.01
            )

        self.assertEqual(result.kind, "TIMEOUT")
        self.assertEqual(context.typecode, "B")
        self.assertEqual(context.size, _EXCHANGE_RESULT_BUFFER_BYTES)
        self.assertTrue(context.daemon)
        self.assertTrue(context.process.started)
        self.assertTrue(context.process.terminated)
        self.assertLessEqual(context.process.join_timeouts[0], 0.01)

    def test_fixed_exchange_frame_rejects_partial_writer_and_accepts_max_body(self):
        incomplete = bytearray(_EXCHANGE_RESULT_BUFFER_BYTES)
        declared = b"x" * 1024
        incomplete[:39] = struct.pack(
            "!BHI32s",
            1,
            200,
            len(declared),
            hashlib.sha256(declared).digest(),
        )
        with self.assertRaises(ETradeBrokerTransportError):
            _read_exchange_result(incomplete)

        maximum = b"y" * (64 * 1024 + 1)
        complete = bytearray(_EXCHANGE_RESULT_BUFFER_BYTES)
        _write_exchange_result(
            complete,
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=maximum,
            ),
        )
        decoded = _read_exchange_result(complete)
        self.assertEqual(decoded.http_status, 200)
        self.assertEqual(decoded.raw_response, maximum)

    def test_bound_request_and_reply_repr_are_redacted(self):
        case = self.case([preview_response()])

        reply = case.transport.preview(case.authorization)

        rendered = repr(reply)
        self.assertNotIn(ACCOUNT_ID, rendered)
        self.assertNotIn(ACCOUNT_KEY, rendered)
        self.assertNotIn(case.record.client_order_id, rendered)
        self.assertNotIn("PreviewOrderRequest", rendered)
        self.assertNotIn("consumer-secret", rendered)

        request_evidence = _transport_evidence(reply.request)
        response_evidence = _transport_response_evidence(reply)
        evidence_repr = repr((request_evidence, response_evidence))
        self.assertNotIn(ACCOUNT_ID, evidence_repr)
        self.assertNotIn(ACCOUNT_KEY, evidence_repr)
        self.assertNotIn(case.record.client_order_id, evidence_repr)
        self.assertNotIn(reply.preview_id, evidence_repr)
        self.assertNotIn("PreviewOrderRequest", evidence_repr)

    def test_exact_runtime_account_is_mandatory(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        os.chmod(temporary.name, 0o700)
        ledger = OrderIntentLedger(
            Path(temporary.name) / "runtime" / "orders.sqlite3"
        )
        runtime = RuntimeSafetyBoundary("sandbox", None, None, None)

        with self.assertRaises(ETradeBrokerTransportError):
            ETradeBrokerTransport(
                session=OAuth1Session(
                    "consumer-key",
                    "consumer-secret",
                    access_token="access-token",
                    access_token_secret="access-secret",
                ),
                ledger=ledger,
                runtime_safety=runtime,
                selected_account=SelectedBrokerAccount(
                    ACCOUNT_ID, ACCOUNT_KEY, INSTITUTION_TYPE
                ),
            )

    def test_transport_requires_exact_oauth_session(self):
        case = self.case([])
        with self.assertRaises(ETradeBrokerTransportError):
            ETradeBrokerTransport(
                session=object(),
                ledger=case.ledger,
                runtime_safety=case.transport._runtime_safety,
                selected_account=case.transport._selected_account,
            )


if __name__ == "__main__":
    unittest.main()
