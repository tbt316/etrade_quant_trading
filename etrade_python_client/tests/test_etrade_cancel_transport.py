from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from unittest.mock import patch

from rauth import OAuth1Session

from live_trading.etrade_broker_transport import (
    CancelBrokerReply,
    ETradeBrokerTransport,
    ETradeBrokerTransportError,
    SelectedBrokerAccount,
    _ExchangeResult,
)
from live_trading.etrade_broker_reader import (
    _PARSER_CODE_SHA256,
    _PARSER_CONFIG_SHA256,
    _PARSER_SCHEMA,
    _reparse_broker_read_response,
)
from live_trading.order_intent_ledger import (
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    BrokerReadResponseEvidence,
    CancellationAuthorization,
    OrderIntent,
    OrderIntentLedger,
    OrderIntentReconciliationRequired,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary


ACCOUNT_ID = "842468410"
ACCOUNT_KEY = "account/key"
INSTITUTION_TYPE = "BROKERAGE"
BROKER_ORDER_ID = "94"
OWNER = "cancel-worker"
IDEMPOTENCY_KEY = "cancel-intent-1-order-94"
ORIGIN = "https://api.etrade.com"
NOW = datetime(2026, 7, 27, 17, 0, tzinfo=timezone.utc)
EXPECTED_XML = (
    b"<?xml version='1.0' encoding='utf-8'?>\n"
    b"<CancelOrderRequest><orderId>94</orderId></CancelOrderRequest>"
)


def canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def vertical_payload() -> dict:
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
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
                "orderAction": "SELL_OPEN",
                "quantity": 1,
            },
            {
                "symbol": "SPY",
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


def accepted_xml(
    *,
    account_id: str = ACCOUNT_ID,
    order_id: str = BROKER_ORDER_ID,
    code: int = 5011,
    message_type: str = "WARNING",
) -> bytes:
    return (
        b"<CancelOrderResponse>"
        + f"<accountId>{account_id}</accountId>".encode("ascii")
        + f"<orderId>{order_id}</orderId>".encode("ascii")
        + b"<cancelTime>1785171600000</cancelTime>"
        + b"<Messages><Message>"
        + f"<code>{code}</code>".encode("ascii")
        + b"<description>200|Your request to cancel your order is being "
        b"processed.</description>"
        + f"<type>{message_type}</type>".encode("ascii")
        + b"</Message></Messages>"
        + b"</CancelOrderResponse>"
    )


def accepted_json(
    *,
    account_id: str = ACCOUNT_ID,
    order_id: str = BROKER_ORDER_ID,
    code: int = 5011,
    message_type: str = "WARNING",
) -> bytes:
    return json.dumps(
        {
            "CancelOrderResponse": {
                "accountId": account_id,
                "orderId": order_id,
                "cancelTime": 1785171600000,
                "messages": {
                    "Message": {
                        "code": code,
                        "description": (
                            "200|Your request to cancel your order is "
                            "being processed."
                        ),
                        "type": message_type,
                    }
                },
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def record_order_read_response(
    ledger: OrderIntentLedger,
    *,
    role: str,
    read_kind: str,
    route: str,
    raw: bytes,
    response_completed_at: datetime,
    target_broker_order_id: str | None = None,
):
    fixture = BrokerReadResponseEvidence(
        read_kind=read_kind,
        account_id=ACCOUNT_ID,
        account_id_key=ACCOUNT_KEY,
        institution_type=INSTITUTION_TYPE,
        environment="production",
        origin=ORIGIN,
        route=route,
        query_json="[]",
        authorization_sha256=hashlib.sha256(
            f"cancel-integration:{role}".encode("ascii")
        ).hexdigest(),
        target_broker_order_id=target_broker_order_id,
        request_started_at=(
            response_completed_at - timedelta(milliseconds=1)
        ),
        response_completed_at=response_completed_at,
        http_status=200,
        raw_response_bytes=raw,
        parser_schema=_PARSER_SCHEMA,
        parser_code_sha256=_PARSER_CODE_SHA256,
        parser_config_sha256=_PARSER_CONFIG_SHA256,
        canonical_parsed_json="null",
        completeness="INELIGIBLE",
    )
    canonical_parsed, completeness = (
        _reparse_broker_read_response(fixture)
    )
    if completeness != "COMPLETE":
        raise AssertionError("integration broker-read fixture is incomplete")
    receipt = ledger.record_broker_read_response(
        replace(
            fixture,
            canonical_parsed_json=canonical_parsed,
            completeness=completeness,
        )
    )
    return receipt, json.loads(canonical_parsed)


def authorize_real_cancellation(
    ledger: OrderIntentLedger,
    path: Path,
) -> CancellationAuthorization:
    intent = OrderIntent.build(
        account_id=ACCOUNT_ID,
        environment="production",
        strategy_id="credit-spread",
        decision_id="cancel-integration-decision",
        idempotency_scope="decision",
        idempotency_key="cancel-integration-intent",
        intent_kind="OPENING",
        order_payload=vertical_payload(),
    )
    record = ledger.create_intent(
        intent,
        intent_id="cancel-integration-intent",
    ).intent

    # Cancellation is the integration target.  Seed the unrelated, already
    # tested opening lifecycle at its public SUBMITTED boundary so the rest of
    # this fixture can exercise the real authorization/claim/receipt tables.
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE order_intents
            SET state = 'SUBMITTED', broker_order_id = ?
            WHERE intent_id = ?
            """,
            (BROKER_ORDER_ID, record.intent_id),
        )
        connection.commit()
    record = ledger.get_intent(record.intent_id)
    if record is None:
        raise AssertionError("integration intent was not retained")

    binding_raw = canonical_json(
        {
            "AccountListResponse": {
                "Accounts": {
                    "Account": [
                        {
                            "accountId": ACCOUNT_ID,
                            "accountIdKey": ACCOUNT_KEY,
                            "institutionType": INSTITUTION_TYPE,
                            "accountStatus": "ACTIVE",
                            "accountMode": "MARGIN",
                            "accountType": "INDIVIDUAL",
                        }
                    ]
                }
            }
        }
    ).encode("ascii")
    wire_payload = json.loads(record.envelope.wire_payload)
    order_raw = canonical_json(
        {
            "OrdersResponse": {
                "Order": [
                    {
                        "orderId": BROKER_ORDER_ID,
                        "orderType": "SPREADS",
                        "OrderDetail": [
                            {
                                "accountId": ACCOUNT_ID,
                                "orderNumber": BROKER_ORDER_ID,
                                "status": "OPEN",
                                "placedTime": str(
                                    int(
                                        (
                                            NOW
                                            - timedelta(minutes=1)
                                        ).timestamp()
                                        * 1_000
                                    )
                                ),
                                "priceType": wire_payload["priceType"],
                                "limitPrice": wire_payload["limitPrice"],
                                "orderTerm": "GOOD_FOR_DAY",
                                "marketSession": "REGULAR",
                                "allOrNone": False,
                                "stopPrice": 0,
                                "Instrument": [
                                    {
                                        "Product": {
                                            "symbol": leg["symbol"],
                                            "securityType": "OPTN",
                                            "callPut": leg["callPut"],
                                            "expiryYear": leg[
                                                "expiryYear"
                                            ],
                                            "expiryMonth": leg[
                                                "expiryMonth"
                                            ],
                                            "expiryDay": leg[
                                                "expiryDay"
                                            ],
                                            "strikePrice": leg[
                                                "strikePrice"
                                            ],
                                        },
                                        "quantityType": "QUANTITY",
                                        "orderedQuantity": leg[
                                            "quantity"
                                        ],
                                        "filledQuantity": 0,
                                        "cancelQuantity": 0,
                                        "orderAction": leg[
                                            "orderAction"
                                        ],
                                    }
                                    for leg in wire_payload["legs"]
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    ).encode("ascii")
    start, _ = record_order_read_response(
        ledger,
        role="binding.start",
        read_kind="ACCOUNT_LIST",
        route="/v1/accounts/list.json",
        raw=binding_raw,
        response_completed_at=NOW - timedelta(seconds=2),
    )
    detail, parsed_detail = record_order_read_response(
        ledger,
        role="order.detail",
        read_kind="ORDER_DETAIL",
        route=(
            f"/v1/accounts/{quote(ACCOUNT_KEY, safe='')}/orders/"
            f"{BROKER_ORDER_ID}.json"
        ),
        raw=order_raw,
        response_completed_at=NOW - timedelta(seconds=1),
        target_broker_order_id=BROKER_ORDER_ID,
    )
    end, _ = record_order_read_response(
        ledger,
        role="binding.end",
        read_kind="ACCOUNT_LIST",
        route="/v1/accounts/list.json",
        raw=binding_raw,
        response_completed_at=NOW,
    )
    expected_payload_hash = ledger.expected_order_payload_hash(
        record.intent_id
    )
    if expected_payload_hash not in parsed_detail[
        "order_payload_hashes"
    ]:
        raise AssertionError(
            "integration order does not match its durable intent"
        )
    result = {
        "schema": "etrade-order-query.v2",
        "broker_order_id": BROKER_ORDER_ID,
        "raw_status": parsed_detail["raw_status"],
        "outcome": parsed_detail["outcome"],
        "fill_summary": parsed_detail["fill_summary"],
        "order_payload_hashes": parsed_detail[
            "order_payload_hashes"
        ],
        "http_status": 200,
        "raw_response_digest": hashlib.sha256(order_raw).hexdigest(),
        "not_found": parsed_detail["not_found"],
        "replacement_links": parsed_detail["replacement_links"],
    }
    order_evidence = ledger.record_broker_read_manifest(
        BrokerReadManifestEvidence(
            evidence_kind="ORDER_QUERY",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            environment="production",
            origin=ORIGIN,
            target_broker_order_id=BROKER_ORDER_ID,
            observed_at=NOW,
            completeness="COMPLETE",
            canonical_result_json=canonical_json(result),
        ),
        (
            BrokerReadManifestMember(
                "binding.start",
                start.receipt_sha256,
            ),
            BrokerReadManifestMember(
                "order.detail",
                detail.receipt_sha256,
            ),
            BrokerReadManifestMember(
                "binding.end",
                end.receipt_sha256,
            ),
        ),
    )
    return ledger.authorize_cancellation(
        record.intent_id,
        IDEMPOTENCY_KEY,
        OWNER,
        60,
        order_evidence,
    )


class ExchangeHarness:
    def __init__(self, outcomes, events):
        self.outcomes = list(outcomes)
        self.events = events
        self.calls = []

    def exchange(
        self,
        prepared,
        *,
        timeout_seconds,
        max_response_bytes=64 * 1024,
    ):
        self.events.append("network")
        self.calls.append(
            (
                prepared,
                timeout_seconds,
                max_response_bytes,
            )
        )
        if not self.outcomes:
            raise AssertionError("unexpected cancellation exchange")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class CancelCase:
    temporary: tempfile.TemporaryDirectory
    ledger: OrderIntentLedger
    transport: ETradeBrokerTransport
    authorization: CancellationAuthorization
    harness: ExchangeHarness
    events: list
    claims: list
    responses: list
    current_time: list
    patchers: tuple


class ETradeCancelTransportTests(unittest.TestCase):
    def setUp(self):
        self.cases = []

    def tearDown(self):
        for case in reversed(self.cases):
            for patcher in reversed(case.patchers):
                patcher.stop()
            case.temporary.cleanup()

    def case(
        self,
        outcome=None,
        *,
        authorization=None,
        claim_side_effect=None,
        response_side_effect=None,
    ):
        if outcome is None:
            outcome = _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=accepted_xml(),
            )
        temporary = tempfile.TemporaryDirectory()
        os.chmod(temporary.name, 0o700)
        ledger = OrderIntentLedger(
            Path(temporary.name) / "runtime" / "orders.sqlite3",
            clock=lambda: NOW,
            run_id="cancel-transport-test",
        )
        current_time = [NOW]
        runtime = RuntimeSafetyBoundary(
            environment="production",
            expected_account_id=ACCOUNT_ID,
            expected_account_id_key=ACCOUNT_KEY,
            expected_institution_type=INSTITUTION_TYPE,
            arm_issued_at=NOW - timedelta(minutes=1),
            arm_expires_at=NOW + timedelta(minutes=10),
        )
        session = OAuth1Session(
            "consumer-secret-label",
            "consumer-secret-value",
            access_token="access-secret-label",
            access_token_secret="access-secret-value",
        )
        transport = ETradeBrokerTransport(
            session=session,
            ledger=ledger,
            runtime_safety=runtime,
            selected_account=SelectedBrokerAccount(
                ACCOUNT_ID,
                ACCOUNT_KEY,
                INSTITUTION_TYPE,
            ),
            clock=lambda: current_time[0],
        )
        authorization = authorization or CancellationAuthorization(
            intent_id="intent-1",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            environment="production",
            owner=OWNER,
            idempotency_key=IDEMPOTENCY_KEY,
            fencing_token=7,
            broker_order_id=BROKER_ORDER_ID,
            order_evidence_sha256="a" * 64,
            order_payload_hash="b" * 64,
            authorization_sha256="c" * 64,
            authorized_at=NOW - timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=5),
        )
        events = []
        claims = []
        responses = []

        def claim(evidence, received_authorization):
            events.append("claim")
            claims.append((evidence, received_authorization))
            if claim_side_effect is not None:
                return claim_side_effect(
                    evidence,
                    received_authorization,
                )
            return None

        def record(evidence, response):
            events.append("response")
            responses.append((evidence, response))
            if response_side_effect is not None:
                return response_side_effect(evidence, response)
            return None

        harness = ExchangeHarness([outcome], events)
        exchange_patcher = patch(
            "live_trading.etrade_broker_transport."
            "_isolated_mutation_exchange",
            side_effect=harness.exchange,
        )
        claim_patcher = patch.object(
            ledger,
            "claim_cancellation_send",
            side_effect=claim,
            create=True,
        )
        response_patcher = patch.object(
            ledger,
            "record_cancellation_response",
            side_effect=record,
            create=True,
        )
        for patcher in (
            exchange_patcher,
            claim_patcher,
            response_patcher,
        ):
            patcher.start()
        result = CancelCase(
            temporary=temporary,
            ledger=ledger,
            transport=transport,
            authorization=authorization,
            harness=harness,
            events=events,
            claims=claims,
            responses=responses,
            current_time=current_time,
            patchers=(
                exchange_patcher,
                claim_patcher,
                response_patcher,
            ),
        )
        self.cases.append(result)
        return result

    def test_exact_put_route_xml_and_durable_ordering(self):
        case = self.case()

        reply = case.transport.cancel(case.authorization)

        self.assertEqual(reply.disposition, "REQUEST_ACCEPTED")
        self.assertEqual(case.events, ["claim", "network", "response"])
        self.assertEqual(len(case.harness.calls), 1)
        prepared, timeout_seconds, max_response_bytes = (
            case.harness.calls[0]
        )
        self.assertEqual(prepared.method, "PUT")
        self.assertEqual(
            prepared.url,
            ORIGIN + "/v1/accounts/account%2Fkey/orders/cancel",
        )
        self.assertEqual(prepared.body, EXPECTED_XML)
        self.assertEqual(timeout_seconds, 15.0)
        self.assertEqual(max_response_bytes, 64 * 1024)
        self.assertNotIn("Cookie", prepared.headers)
        self.assertNotIn("Transfer-Encoding", prepared.headers)
        evidence, received_authorization = case.claims[0]
        self.assertIs(received_authorization, case.authorization)
        self.assertEqual(evidence.http_method, "PUT")
        self.assertEqual(
            evidence.route,
            "/v1/accounts/account%2Fkey/orders/cancel",
        )
        self.assertEqual(evidence.final_xml_bytes, EXPECTED_XML)
        self.assertEqual(
            evidence.final_xml_sha256,
            hashlib.sha256(EXPECTED_XML).hexdigest(),
        )
        response = case.responses[0][1]
        self.assertEqual(response.disposition, "REQUEST_ACCEPTED")
        self.assertEqual(response.message_codes, (5011,))
        self.assertEqual(response.message_types, ("WARNING",))

    def test_real_ledger_authorization_claim_and_receipt_are_one_shot(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.chmod(temporary, 0o700)
            path = Path(temporary) / "runtime" / "orders.sqlite3"
            ledger = OrderIntentLedger(
                path,
                clock=lambda: NOW,
                run_id="cancel-integration-test",
            )
            authorization = authorize_real_cancellation(ledger, path)
            runtime = RuntimeSafetyBoundary(
                environment="production",
                expected_account_id=ACCOUNT_ID,
                expected_account_id_key=ACCOUNT_KEY,
                expected_institution_type=INSTITUTION_TYPE,
                arm_issued_at=NOW - timedelta(minutes=1),
                arm_expires_at=NOW + timedelta(minutes=10),
            )
            transport = ETradeBrokerTransport(
                session=OAuth1Session(
                    "consumer-key",
                    "consumer-secret",
                    access_token="access-token",
                    access_token_secret="access-secret",
                ),
                ledger=ledger,
                runtime_safety=runtime,
                selected_account=SelectedBrokerAccount(
                    ACCOUNT_ID,
                    ACCOUNT_KEY,
                    INSTITUTION_TYPE,
                ),
                clock=lambda: NOW,
            )
            events = []
            harness = ExchangeHarness(
                [
                    _ExchangeResult(
                        "RESPONSE",
                        http_status=200,
                        raw_response=accepted_xml(),
                    )
                ],
                events,
            )
            with patch(
                "live_trading.etrade_broker_transport."
                "_isolated_mutation_exchange",
                side_effect=harness.exchange,
            ):
                reply = transport.cancel(authorization)
                self.assertEqual(
                    reply.disposition,
                    "REQUEST_ACCEPTED",
                )
                cancellation = ledger.get_cancellation(
                    authorization.intent_id
                )
                self.assertIsNotNone(cancellation)
                self.assertEqual(
                    cancellation.state,
                    "REQUEST_ACCEPTED",
                )
                with self.assertRaises(
                    OrderIntentReconciliationRequired
                ):
                    transport.cancel(authorization)

            self.assertEqual(events, ["network"])
            self.assertEqual(len(harness.calls), 1)
            with sqlite3.connect(path) as connection:
                send_count = connection.execute(
                    "SELECT COUNT(*) FROM cancel_send_attempts"
                ).fetchone()[0]
                response_count = connection.execute(
                    "SELECT COUNT(*) FROM cancel_response_receipts"
                ).fetchone()[0]
            self.assertEqual(send_count, 1)
            self.assertEqual(response_count, 1)

    def test_real_ledger_persistence_failure_is_not_resendable(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.chmod(temporary, 0o700)
            path = Path(temporary) / "runtime" / "orders.sqlite3"
            ledger = OrderIntentLedger(
                path,
                clock=lambda: NOW,
                run_id="cancel-persistence-failure",
            )
            authorization = authorize_real_cancellation(ledger, path)
            runtime = RuntimeSafetyBoundary(
                environment="production",
                expected_account_id=ACCOUNT_ID,
                expected_account_id_key=ACCOUNT_KEY,
                expected_institution_type=INSTITUTION_TYPE,
                arm_issued_at=NOW - timedelta(minutes=1),
                arm_expires_at=NOW + timedelta(minutes=10),
            )

            def transport_for(bound_ledger):
                return ETradeBrokerTransport(
                    session=OAuth1Session(
                        "consumer-key",
                        "consumer-secret",
                        access_token="access-token",
                        access_token_secret="access-secret",
                    ),
                    ledger=bound_ledger,
                    runtime_safety=runtime,
                    selected_account=SelectedBrokerAccount(
                        ACCOUNT_ID,
                        ACCOUNT_KEY,
                        INSTITUTION_TYPE,
                    ),
                    clock=lambda: NOW,
                )

            events = []
            harness = ExchangeHarness(
                [
                    _ExchangeResult(
                        "RESPONSE",
                        http_status=200,
                        raw_response=accepted_xml(),
                    )
                ],
                events,
            )
            with patch(
                "live_trading.etrade_broker_transport."
                "_isolated_mutation_exchange",
                side_effect=harness.exchange,
            ):
                with patch.object(
                    ledger,
                    "record_cancellation_response",
                    side_effect=RuntimeError("receipt disk failure"),
                ):
                    with self.assertRaisesRegex(
                        ETradeBrokerTransportError,
                        "could not be persisted",
                    ):
                        transport_for(ledger).cancel(authorization)

                cancellation = ledger.get_cancellation(
                    authorization.intent_id
                )
                self.assertIsNotNone(cancellation)
                self.assertEqual(cancellation.state, "SEND_UNKNOWN")
                with self.assertRaises(
                    OrderIntentReconciliationRequired
                ):
                    transport_for(ledger).cancel(authorization)

            self.assertEqual(events, ["network"])
            self.assertEqual(len(harness.calls), 1)
            with sqlite3.connect(path) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM cancel_send_attempts"
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) "
                        "FROM cancel_response_receipts"
                    ).fetchone()[0],
                    0,
                )

    def test_accepts_exact_xml_and_json_5011_acknowledgements(self):
        for raw in (accepted_xml(), accepted_json()):
            with self.subTest(prefix=raw[:1]):
                case = self.case(
                    _ExchangeResult(
                        "RESPONSE",
                        http_status=200,
                        raw_response=raw,
                    )
                )
                reply = case.transport.cancel(case.authorization)
                self.assertEqual(
                    reply.disposition,
                    "REQUEST_ACCEPTED",
                )
                self.assertEqual(
                    reply.broker_messages[0].code,
                    5011,
                )
                self.assertEqual(len(case.harness.calls), 1)
                self.assertEqual(
                    case.responses[0][1].raw_response_digest,
                    hashlib.sha256(raw).hexdigest(),
                )

    def test_identity_mismatch_malformed_non_200_and_timeout_are_unknown(self):
        valid_xml = accepted_xml()
        message_xml = valid_xml.split(
            b"<Messages>", 1
        )[1].split(b"</Messages>", 1)[0]
        duplicate_message_xml = valid_xml.replace(
            b"</Messages>",
            message_xml + b"</Messages>",
        )
        extra_json = json.loads(accepted_json().decode("utf-8"))
        extra_json["CancelOrderResponse"]["unexpected"] = True
        outcomes = (
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=accepted_xml(account_id="842468411"),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=accepted_json(order_id="95"),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=b"<CancelOrderResponse/>",
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=valid_xml.replace(
                    b"<CancelOrderResponse>",
                    b"<CancelOrderResponse>garbage",
                ),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=valid_xml.replace(
                    b"</accountId>",
                    b"</accountId>garbage",
                ),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=valid_xml.replace(
                    f"<accountId>{ACCOUNT_ID}</accountId>".encode(
                        "ascii"
                    ),
                    f"<accountId> {ACCOUNT_ID}</accountId>".encode(
                        "ascii"
                    ),
                ),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=duplicate_message_xml,
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=canonical_json(extra_json).encode("utf-8"),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=400,
                raw_response=accepted_xml(),
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=302,
                raw_response=b"redirect",
            ),
            _ExchangeResult(
                "RESPONSE",
                http_status=200,
                raw_response=b"x" * (64 * 1024 + 1),
            ),
            _ExchangeResult("TIMEOUT"),
            _ExchangeResult("TRANSPORT_ERROR"),
            _ExchangeResult("MALFORMED_RESPONSE"),
        )
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                case = self.case(outcome)
                reply = case.transport.cancel(case.authorization)
                self.assertEqual(reply.disposition, "UNKNOWN")
                self.assertIsNotNone(reply.unknown_reason)
                self.assertEqual(len(case.harness.calls), 1)
                self.assertEqual(
                    case.events,
                    ["claim", "network", "response"],
                )
                self.assertEqual(
                    case.responses[0][1].disposition,
                    "UNKNOWN",
                )

    def test_5011_must_be_the_only_accepted_warning(self):
        for raw in (
            accepted_xml(code=5003),
            accepted_json(message_type="ERROR"),
        ):
            with self.subTest(raw=raw):
                case = self.case(
                    _ExchangeResult(
                        "RESPONSE",
                        http_status=200,
                        raw_response=raw,
                    )
                )
                reply = case.transport.cancel(case.authorization)
                self.assertEqual(reply.disposition, "UNKNOWN")
                self.assertEqual(
                    reply.unknown_reason,
                    "REVIEW_REQUIRED",
                )

    def test_transport_exception_is_unknown_and_is_persisted(self):
        case = self.case(RuntimeError("socket failed"))

        reply = case.transport.cancel(case.authorization)

        self.assertEqual(reply.disposition, "UNKNOWN")
        self.assertEqual(reply.unknown_reason, "TRANSPORT_ERROR")
        self.assertEqual(case.events, ["claim", "network", "response"])
        self.assertEqual(len(case.harness.calls), 1)

    def test_duplicate_claim_failure_prevents_a_second_network_attempt(self):
        attempts = [0]

        def claim(_evidence, _authorization):
            attempts[0] += 1
            if attempts[0] > 1:
                raise RuntimeError("cancellation send already claimed")

        case = self.case(claim_side_effect=claim)
        first = case.transport.cancel(case.authorization)
        self.assertEqual(first.disposition, "REQUEST_ACCEPTED")

        with self.assertRaisesRegex(
            RuntimeError,
            "already claimed",
        ):
            case.transport.cancel(case.authorization)

        self.assertEqual(len(case.harness.calls), 1)
        self.assertEqual(case.events.count("claim"), 2)
        self.assertEqual(case.events.count("network"), 1)

    def test_response_persistence_failure_does_not_repeat_the_send(self):
        def fail_record(_evidence, _response):
            raise RuntimeError("disk failure")

        case = self.case(response_side_effect=fail_record)

        with self.assertRaisesRegex(
            ETradeBrokerTransportError,
            "could not be persisted",
        ):
            case.transport.cancel(case.authorization)

        self.assertEqual(case.events, ["claim", "network", "response"])
        self.assertEqual(len(case.harness.calls), 1)
        self.assertEqual(len(case.claims), 1)

    def test_authorization_account_and_runtime_are_revalidated(self):
        case = self.case()
        mismatched = replace(
            case.authorization,
            account_id="842468411",
        )

        with self.assertRaisesRegex(
            ETradeBrokerTransportError,
            "does not match the bound account",
        ):
            case.transport.cancel(mismatched)

        case.current_time[0] = NOW + timedelta(minutes=11)
        long_authorization = replace(
            case.authorization,
            expires_at=NOW + timedelta(minutes=30),
        )
        with self.assertRaises(Exception):
            case.transport.cancel(long_authorization)

        self.assertEqual(case.events, [])
        self.assertEqual(case.harness.calls, [])

    def test_runtime_expiry_after_claim_prevents_network_but_keeps_claim(self):
        def expire_after_claim(_evidence, _authorization):
            case.current_time[0] = NOW + timedelta(minutes=11)

        case = self.case(claim_side_effect=expire_after_claim)

        with self.assertRaises(Exception):
            case.transport.cancel(case.authorization)

        self.assertEqual(case.events, ["claim"])
        self.assertEqual(len(case.claims), 1)
        self.assertEqual(case.harness.calls, [])
        self.assertEqual(case.responses, [])

    def test_reply_repr_redacts_account_order_owner_payload_and_message(self):
        case = self.case()

        reply = case.transport.cancel(case.authorization)

        rendered = repr(reply)
        self.assertIsInstance(reply, CancelBrokerReply)
        for secret in (
            ACCOUNT_ID,
            ACCOUNT_KEY,
            BROKER_ORDER_ID,
            OWNER,
            IDEMPOTENCY_KEY,
            "CancelOrderRequest",
            "Your request to cancel",
            "consumer-secret-value",
            "access-secret-value",
        ):
            self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
