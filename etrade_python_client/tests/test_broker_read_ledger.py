from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from live_trading.etrade_broker_reader import (
    _PARSER_CODE_SHA256,
    _PARSER_CONFIG_SHA256,
    _PARSER_SCHEMA,
    _reparse_broker_read_response,
)
from live_trading.order_intent_ledger import (
    SCHEMA_VERSION,
    AccountCapacityEvidence,
    BrokerReadEvidenceRef,
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    BrokerReadResponseEvidence,
    OrderIntent,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLedgerError,
    OrderIntentValidationError,
    RiskEvidence,
    canonical_order_payload_hash,
)


ACCOUNT_ID = "842468410"
ACCOUNT_ID_KEY = "account-key-1"
INSTITUTION_TYPE = "BROKERAGE"
ENVIRONMENT = "production"
ORIGIN = "https://api.etrade.com"
_DERIVE_PARSED = object()


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _domain_json_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(
        domain + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _opening_payload() -> dict[str, Any]:
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


def _opening_intent(*, key: str = "decision-key") -> OrderIntent:
    return OrderIntent.build(
        account_id=ACCOUNT_ID,
        environment=ENVIRONMENT,
        strategy_id="credit-spread",
        decision_id=f"decision-{key}",
        idempotency_scope="decision",
        idempotency_key=key,
        intent_kind="OPENING",
        order_payload=_opening_payload(),
    )


def _account_list_raw() -> bytes:
    return _canonical_json(
        {
            "AccountListResponse": {
                "Accounts": {
                    "Account": [
                        {
                            "accountId": ACCOUNT_ID,
                            "accountIdKey": ACCOUNT_ID_KEY,
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


def _balance_raw(buying_power: str, as_of_date: str) -> bytes:
    return _canonical_json(
        {
            "BalanceResponse": {
                "accountId": ACCOUNT_ID,
                "institutionType": INSTITUTION_TYPE,
                "asOfDate": as_of_date,
                "Computed": {"marginBuyingPower": buying_power},
            }
        }
    ).encode("ascii")


def _portfolio_raw() -> bytes:
    return _canonical_json(
        {
            "PortfolioResponse": {
                "AccountPortfolio": [
                    {
                        "accountId": ACCOUNT_ID,
                        "totalNoOfPages": "1",
                        "Position": [],
                    }
                ]
            }
        }
    ).encode("ascii")


def _orders_raw() -> bytes:
    return _canonical_json(
        {"OrdersResponse": {"Order": []}}
    ).encode("ascii")


def _known_order_raw(
    broker_order_id: str,
    *,
    limit_price: str = "1.25",
    status: str = "OPEN",
) -> bytes:
    return _canonical_json(
        {
            "OrdersResponse": {
                "Order": [
                    {
                        "orderId": broker_order_id,
                        "orderType": "SPREADS",
                        "OrderDetail": [
                            {
                                "accountId": ACCOUNT_ID,
                                "orderNumber": broker_order_id,
                                "status": status,
                                "priceType": "NET_CREDIT",
                                "limitPrice": limit_price,
                                "orderTerm": "GOOD_FOR_DAY",
                                "marketSession": "REGULAR",
                                "allOrNone": False,
                                "stopPrice": "0",
                                "Instrument": [
                                    {
                                        "Product": {
                                            "symbol": "SPY",
                                            "securityType": "OPTN",
                                            "callPut": "PUT",
                                            "expiryYear": "2026",
                                            "expiryMonth": "8",
                                            "expiryDay": "21",
                                            "strikePrice": "620",
                                        },
                                        "orderAction": "SELL_OPEN",
                                        "quantityType": "QUANTITY",
                                        "orderedQuantity": "1",
                                        "filledQuantity": "0",
                                        "cancelQuantity": "0",
                                    },
                                    {
                                        "Product": {
                                            "symbol": "SPY",
                                            "securityType": "OPTN",
                                            "callPut": "PUT",
                                            "expiryYear": "2026",
                                            "expiryMonth": "8",
                                            "expiryDay": "21",
                                            "strikePrice": "615",
                                        },
                                        "orderAction": "BUY_OPEN",
                                        "quantityType": "QUANTITY",
                                        "orderedQuantity": "1",
                                        "filledQuantity": "0",
                                        "cancelQuantity": "0",
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    ).encode("ascii")


def _vertical_hashes(limit_price: str) -> tuple[str, ...]:
    payload = _opening_payload()
    payload["limitPrice"] = Decimal(limit_price)
    forward = canonical_order_payload_hash(payload)
    reverse_payload = dict(payload)
    reverse_payload["legs"] = list(reversed(payload["legs"]))
    reverse = canonical_order_payload_hash(reverse_payload)
    return tuple(sorted({forward, reverse}))


class BrokerReadLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.chmod(self.tmp.name, 0o700)
        self.path = Path(self.tmp.name) / "runtime" / "orders.sqlite3"
        self.clock = Clock()
        self.authorization_counter = 0
        self.ledger = OrderIntentLedger(
            self.path, clock=self.clock, run_id="reader-ledger-tests"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _record_response(
        self,
        *,
        read_kind: str,
        route: str,
        raw: bytes,
        parsed: Any = _DERIVE_PARSED,
        query: tuple[tuple[str, str], ...] = (),
        target_broker_order_id: str | None = None,
        completeness: str = "COMPLETE",
        account_id: str = ACCOUNT_ID,
        account_id_key: str = ACCOUNT_ID_KEY,
        final_response: bool = False,
    ):
        self.authorization_counter += 1
        authorization_sha256 = hashlib.sha256(
            (
                "broker-read-test-authorization:"
                f"{self.authorization_counter}"
            ).encode("ascii")
        ).hexdigest()
        response_completed_at = (
            self.clock.now
            if final_response
            else self.clock.now
            - timedelta(seconds=30)
            + timedelta(milliseconds=self.authorization_counter)
        )
        evidence = BrokerReadResponseEvidence(
            read_kind=read_kind,
            account_id=account_id,
            account_id_key=account_id_key,
            institution_type=INSTITUTION_TYPE,
            environment=ENVIRONMENT,
            origin=ORIGIN,
            route=route,
            query_json=_canonical_json(
                [list(pair) for pair in sorted(query)]
            ),
            authorization_sha256=authorization_sha256,
            target_broker_order_id=target_broker_order_id,
            request_started_at=response_completed_at
            - timedelta(microseconds=500),
            response_completed_at=response_completed_at,
            http_status=200,
            raw_response_bytes=raw,
            parser_schema=_PARSER_SCHEMA,
            parser_code_sha256=_PARSER_CODE_SHA256,
            parser_config_sha256=_PARSER_CONFIG_SHA256,
            canonical_parsed_json=(
                "null"
                if parsed is _DERIVE_PARSED
                else _canonical_json(parsed)
            ),
            completeness=completeness,
        )
        if parsed is _DERIVE_PARSED:
            reparsed_json, reparsed_completeness = (
                _reparse_broker_read_response(evidence)
            )
            if reparsed_completeness == "INELIGIBLE":
                raise AssertionError(
                    "valid broker-read fixture did not satisfy the concrete parser"
                )
            evidence = replace(
                evidence,
                canonical_parsed_json=reparsed_json,
                completeness=reparsed_completeness,
            )
        return self.ledger.record_broker_read_response(evidence)

    def _persisted_parsed(self, receipt_sha256: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT canonical_parsed_json
                FROM broker_read_receipts
                WHERE receipt_sha256 = ?
                """,
                (receipt_sha256,),
            ).fetchone()
        self.assertIsNotNone(row)
        parsed = json.loads(row[0])
        self.assertIs(type(parsed), dict)
        return parsed

    def _capacity_manifest(
        self, *, buying_power: str = "1000"
    ) -> tuple[BrokerReadEvidenceRef, str, tuple[str, ...]]:
        buying_power_as_of = str(
            int(
                (
                    self.clock.now - timedelta(seconds=30)
                ).timestamp()
                * 1_000
            )
        )
        sources = [
            (
                "binding.start",
                "ACCOUNT_LIST",
                "/v1/accounts/list.json",
                (),
                _account_list_raw(),
            ),
        ]
        for scan in ("a", "b"):
            sources.extend(
                (
                    (
                        f"scan_{scan}.balance",
                        "BALANCE",
                        f"/v1/accounts/{ACCOUNT_ID_KEY}/balance.json",
                        (
                            ("instType", "BROKERAGE"),
                            ("realTimeNAV", "true"),
                        ),
                        (
                            _balance_raw(
                                buying_power, buying_power_as_of
                            )
                        ),
                    ),
                    (
                        f"scan_{scan}.portfolio.0001",
                        "PORTFOLIO_PAGE",
                        f"/v1/accounts/{ACCOUNT_ID_KEY}/portfolio.json",
                        (
                            ("count", "50"),
                            ("lotsRequired", "false"),
                            ("marketSession", "REGULAR"),
                            ("pageNumber", "1"),
                            ("sortBy", "SYMBOL"),
                            ("sortOrder", "ASC"),
                            ("totalsRequired", "false"),
                            ("view", "COMPLETE"),
                        ),
                        _portfolio_raw(),
                    ),
                )
            )
            for lane in (
                "OPEN",
                "CANCEL_REQUESTED",
                "INDIVIDUAL_FILLS",
            ):
                sources.append(
                    (
                        f"scan_{scan}.orders.{lane}.0000",
                        "OPEN_ORDERS_PAGE",
                        f"/v1/accounts/{ACCOUNT_ID_KEY}/orders.json",
                        (("count", "100"), ("status", lane)),
                        _orders_raw(),
                    )
                )
        sources.append(
            (
                "binding.end",
                "ACCOUNT_LIST",
                "/v1/accounts/list.json",
                (),
                _account_list_raw(),
            )
        )
        members: list[BrokerReadManifestMember] = []
        receipt_hashes: list[str] = []
        for role, kind, route, query, raw in sources:
            receipt = self._record_response(
                read_kind=kind,
                route=route,
                query=query,
                raw=raw,
                final_response=role == "binding.end",
            )
            receipt_hashes.append(receipt.receipt_sha256)
            members.append(
                BrokerReadManifestMember(
                    role=role, receipt_sha256=receipt.receipt_sha256
                )
            )

        state = {
            "schema": "etrade-capacity.v1",
            "account_status": "ACTIVE",
            "account_mode": "MARGIN",
            "account_type": "INDIVIDUAL",
            "broker_buying_power": buying_power,
            "broker_buying_power_as_of": buying_power_as_of,
            "positions": [],
            "open_orders": [],
        }
        economic_state = dict(state)
        economic_state.pop("broker_buying_power_as_of")
        state_sha256 = _domain_json_sha256(
            b"etrade-capacity-state.v1\0", economic_state
        )
        result = dict(state)
        result["state_sha256"] = state_sha256
        evidence = self.ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="CAPACITY",
                account_id=ACCOUNT_ID,
                account_id_key=ACCOUNT_ID_KEY,
                institution_type=INSTITUTION_TYPE,
                environment=ENVIRONMENT,
                origin=ORIGIN,
                target_broker_order_id=None,
                observed_at=self.clock.now,
                completeness="COMPLETE",
                canonical_result_json=_canonical_json(result),
            ),
            tuple(members),
        )
        return evidence, state_sha256, tuple(receipt_hashes)

    def _order_query_manifest(
        self,
        *,
        broker_order_id: str,
        payload_hashes: tuple[str, ...],
        outcome: str = "OPEN",
    ) -> tuple[BrokerReadEvidenceRef, str]:
        default_hashes = _vertical_hashes("1.25")
        limit_price = (
            "1.25"
            if set(payload_hashes).intersection(default_hashes)
            else "1.30"
        )
        raw = _known_order_raw(
            broker_order_id,
            limit_price=limit_price,
            status=outcome,
        )
        binding_start = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
        )
        receipt = self._record_response(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{ACCOUNT_ID_KEY}/orders/"
                f"{broker_order_id}.json"
            ),
            raw=raw,
            target_broker_order_id=broker_order_id,
        )
        parsed = self._persisted_parsed(receipt.receipt_sha256)
        binding_end = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
            final_response=True,
        )
        result = {
            "schema": "etrade-order-query.v1",
            "broker_order_id": broker_order_id,
            "raw_status": parsed["raw_status"],
            "outcome": parsed["outcome"],
            "order_payload_hashes": parsed["order_payload_hashes"],
            "http_status": 200,
            "raw_response_digest": hashlib.sha256(raw).hexdigest(),
            "not_found": parsed["not_found"],
            "replacement_links": parsed["replacement_links"],
        }
        evidence = self.ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="ORDER_QUERY",
                account_id=ACCOUNT_ID,
                account_id_key=ACCOUNT_ID_KEY,
                institution_type=INSTITUTION_TYPE,
                environment=ENVIRONMENT,
                origin=ORIGIN,
                target_broker_order_id=broker_order_id,
                observed_at=self.clock.now,
                completeness="COMPLETE",
                canonical_result_json=_canonical_json(result),
            ),
            (
                BrokerReadManifestMember(
                    role="binding.start",
                    receipt_sha256=binding_start.receipt_sha256,
                ),
                BrokerReadManifestMember(
                    role="order.detail",
                    receipt_sha256=receipt.receipt_sha256,
                ),
                BrokerReadManifestMember(
                    role="binding.end",
                    receipt_sha256=binding_end.receipt_sha256,
                ),
            ),
        )
        return evidence, receipt.receipt_sha256

    def _capacity_decision(self):
        capacity, state_sha256, receipt_hashes = (
            self._capacity_manifest()
        )
        decision = self.ledger.set_reservation_cap_from_read(
            capacity, risk_budget=Decimal("750")
        )
        return capacity, decision, state_sha256, receipt_hashes

    def _create_reserved_unknown_intent(self):
        _, decision, state_sha256, _ = self._capacity_decision()
        record = self.ledger.create_intent(
            _opening_intent(key="query-provenance")
        ).intent
        reservation = self.ledger.reserve_margin(
            record.intent_id,
            RiskEvidence(
                decision_id=record.envelope.decision_id,
                max_loss_amount=Decimal("500"),
                collateral_amount=Decimal("500"),
                quote_observed_at=self.clock.now,
                quote_digest="c" * 64,
                portfolio_observed_at=decision.observed_at,
                portfolio_snapshot_digest=state_sha256,
                capacity_decision_sha256=decision.decision_sha256,
            ),
        )
        lease = self.ledger.claim_submission(
            record.intent_id, "worker-1", lease_seconds=30
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, "worker-1", lease.fencing_token
        )
        unknown = self.ledger.begin_submission(
            record.intent_id,
            "worker-1",
            lease.fencing_token,
            authorization,
        )
        self.assertEqual(unknown.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(
            reservation.capacity_decision_sha256,
            decision.decision_sha256,
        )
        return record, decision

    def test_self_attested_capacity_cannot_set_an_opening_cap(self) -> None:
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.set_reservation_cap(
                AccountCapacityEvidence(
                    account_id=ACCOUNT_ID,
                    environment=ENVIRONMENT,
                    broker_buying_power=Decimal("1000"),
                    risk_budget=Decimal("750"),
                    observed_at=self.clock.now,
                    portfolio_snapshot_digest="d" * 64,
                )
            )

    def test_complete_capacity_manifest_drives_decision_and_reservation(
        self,
    ) -> None:
        capacity, decision, state_sha256, receipt_hashes = (
            self._capacity_decision()
        )
        self.assertEqual(capacity.evidence_kind, "CAPACITY")
        self.assertEqual(decision.cap_amount, Decimal("750"))
        self.assertEqual(
            decision.broker_buying_power, Decimal("1000")
        )
        self.assertEqual(
            decision.portfolio_snapshot_digest, state_sha256
        )

        replay = self.ledger.set_reservation_cap_from_read(
            capacity, risk_budget=Decimal("750")
        )
        self.assertEqual(replay.decision_sha256, decision.decision_sha256)

        record = self.ledger.create_intent(
            _opening_intent(key="capacity-provenance")
        ).intent
        reservation = self.ledger.reserve_margin(
            record.intent_id,
            RiskEvidence(
                decision_id=record.envelope.decision_id,
                max_loss_amount=Decimal("500"),
                collateral_amount=Decimal("500"),
                quote_observed_at=self.clock.now,
                quote_digest="e" * 64,
                portfolio_observed_at=decision.observed_at,
                portfolio_snapshot_digest=state_sha256,
                capacity_decision_sha256=decision.decision_sha256,
            ),
        )
        self.assertEqual(
            reservation.capacity_decision_sha256,
            decision.decision_sha256,
        )
        self.assertEqual(
            reservation.portfolio_snapshot_digest, state_sha256
        )

        second = self.ledger.create_intent(
            _opening_intent(key="wrong-capacity-decision")
        ).intent
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reserve_margin(
                second.intent_id,
                RiskEvidence(
                    decision_id=second.envelope.decision_id,
                    max_loss_amount=Decimal("500"),
                    collateral_amount=Decimal("500"),
                    quote_observed_at=self.clock.now,
                    quote_digest="f" * 64,
                    portfolio_observed_at=decision.observed_at,
                    portfolio_snapshot_digest=state_sha256,
                    capacity_decision_sha256="0" * 64,
                ),
            )

        with sqlite3.connect(self.path) as conn:
            canonical_result_json = conn.execute(
                """
                SELECT canonical_result_json
                FROM broker_read_manifests
                WHERE evidence_sha256 = ?
                """,
                (capacity.evidence_sha256,),
            ).fetchone()[0]
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="CAPACITY",
                    account_id=ACCOUNT_ID,
                    account_id_key=ACCOUNT_ID_KEY,
                    institution_type=INSTITUTION_TYPE,
                    environment=ENVIRONMENT,
                    origin=ORIGIN,
                    target_broker_order_id=None,
                    observed_at=self.clock.now,
                    completeness="COMPLETE",
                    canonical_result_json=canonical_result_json,
                ),
                (
                    BrokerReadManifestMember(
                        role="binding.start",
                        receipt_sha256=receipt_hashes[0],
                    ),
                ),
            )

    def test_query_manifest_provenance_and_economic_mismatch_rejection(
        self,
    ) -> None:
        record, _ = self._create_reserved_unknown_intent()
        wrong_query, _ = self._order_query_manifest(
            broker_order_id="9000001",
            payload_hashes=("f" * 64,),
        )
        wrong_evidence = self.ledger.broker_evidence_from_read(
            record.intent_id, wrong_query, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(wrong_evidence)
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reconcile_open(
                record.intent_id, wrong_evidence
            )

        valid_query, _ = self._order_query_manifest(
            broker_order_id="9000001",
            payload_hashes=(record.envelope.payload_hash,),
        )
        evidence = self.ledger.broker_evidence_from_read(
            record.intent_id, valid_query, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(
            evidence.broker_read_evidence_sha256,
            valid_query.evidence_sha256,
        )
        self.assertEqual(
            tuple(
                payload_hash
                for payload_hash in evidence.order_payload_hashes
                if payload_hash == record.envelope.payload_hash
            ),
            (record.envelope.payload_hash,),
        )
        reconciled = self.ledger.reconcile_open(
            record.intent_id, evidence
        )
        self.assertEqual(reconciled.state, "SUBMITTED")
        self.assertEqual(
            self.ledger.events(record.intent_id)[-1]
            .broker_read_evidence_sha256,
            valid_query.evidence_sha256,
        )

    def test_query_manifest_rejects_receipt_for_a_different_order(
        self,
    ) -> None:
        binding_start = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
        )
        detail_raw = _known_order_raw("9000002")
        receipt = self._record_response(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{ACCOUNT_ID_KEY}/orders/"
                "9000002.json"
            ),
            raw=detail_raw,
            target_broker_order_id="9000002",
        )
        parsed = self._persisted_parsed(receipt.receipt_sha256)
        binding_end = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
            final_response=True,
        )
        result = {
            "schema": "etrade-order-query.v1",
            "broker_order_id": "9000003",
            "raw_status": parsed["raw_status"],
            "outcome": parsed["outcome"],
            "order_payload_hashes": parsed["order_payload_hashes"],
            "http_status": 200,
            "raw_response_digest": hashlib.sha256(
                detail_raw
            ).hexdigest(),
            "not_found": parsed["not_found"],
            "replacement_links": parsed["replacement_links"],
        }
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="ORDER_QUERY",
                    account_id=ACCOUNT_ID,
                    account_id_key=ACCOUNT_ID_KEY,
                    institution_type=INSTITUTION_TYPE,
                    environment=ENVIRONMENT,
                    origin=ORIGIN,
                    target_broker_order_id="9000003",
                    observed_at=self.clock.now,
                    completeness="COMPLETE",
                    canonical_result_json=_canonical_json(result),
                ),
                (
                    BrokerReadManifestMember(
                        role="binding.start",
                        receipt_sha256=binding_start.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        role="order.detail",
                        receipt_sha256=receipt.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        role="binding.end",
                        receipt_sha256=binding_end.receipt_sha256,
                    ),
                ),
            )

    def test_query_manifest_rejects_nonchronological_binding_bracket(
        self,
    ) -> None:
        order_id = "9000002"
        detail_raw = _known_order_raw(order_id)
        detail = self._record_response(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{ACCOUNT_ID_KEY}/orders/"
                f"{order_id}.json"
            ),
            raw=detail_raw,
            target_broker_order_id=order_id,
        )
        binding_start = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
        )
        binding_end = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=_account_list_raw(),
            final_response=True,
        )
        parsed = self._persisted_parsed(detail.receipt_sha256)
        result = {
            "schema": "etrade-order-query.v1",
            "broker_order_id": order_id,
            "raw_status": parsed["raw_status"],
            "outcome": parsed["outcome"],
            "order_payload_hashes": parsed["order_payload_hashes"],
            "http_status": 200,
            "raw_response_digest": hashlib.sha256(
                detail_raw
            ).hexdigest(),
            "not_found": parsed["not_found"],
            "replacement_links": parsed["replacement_links"],
        }

        with self.assertRaisesRegex(
            OrderIntentIntegrityError,
            "chronologically ordered",
        ):
            self.ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="ORDER_QUERY",
                    account_id=ACCOUNT_ID,
                    account_id_key=ACCOUNT_ID_KEY,
                    institution_type=INSTITUTION_TYPE,
                    environment=ENVIRONMENT,
                    origin=ORIGIN,
                    target_broker_order_id=order_id,
                    observed_at=self.clock.now,
                    completeness="COMPLETE",
                    canonical_result_json=_canonical_json(result),
                ),
                (
                    BrokerReadManifestMember(
                        role="binding.start",
                        receipt_sha256=binding_start.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        role="order.detail",
                        receipt_sha256=detail.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        role="binding.end",
                        receipt_sha256=binding_end.receipt_sha256,
                    ),
                ),
            )

    def test_raw_response_replay_rejects_inconsistent_parsed_json(
        self,
    ) -> None:
        inconsistent = {
            "account_id": ACCOUNT_ID,
            "account_id_key": ACCOUNT_ID_KEY,
            "institution_type": INSTITUTION_TYPE,
            "account_status": "CLOSED",
            "account_mode": "MARGIN",
            "account_type": "INDIVIDUAL",
        }
        with self.assertRaisesRegex(
            OrderIntentIntegrityError,
            "parser output does not match raw response bytes",
        ):
            self._record_response(
                read_kind="ACCOUNT_LIST",
                route="/v1/accounts/list.json",
                raw=_account_list_raw(),
                parsed=inconsistent,
            )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM broker_read_receipts"
                ).fetchone()[0],
                0,
            )

    def test_content_hashes_detect_raw_and_manifest_tampering(self) -> None:
        capacity, _, _, receipt_hashes = self._capacity_decision()
        query, _ = self._order_query_manifest(
            broker_order_id="9000004",
            payload_hashes=("1" * 64,),
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "DROP TRIGGER prevent_broker_read_receipt_update"
            )
            conn.execute(
                """
                UPDATE broker_read_receipts
                SET raw_response_bytes = zeroblob(raw_byte_length)
                WHERE receipt_sha256 = ?
                """,
                (receipt_hashes[0],),
            )
            conn.execute(
                "DROP TRIGGER prevent_broker_read_manifest_update"
            )
            conn.execute(
                """
                UPDATE broker_read_manifests
                SET canonical_result_json = ?
                WHERE evidence_sha256 = ?
                """,
                (_canonical_json({"tampered": True}), query.evidence_sha256),
            )

        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.broker_read_evidence(
                capacity.evidence_sha256
            )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.broker_read_evidence(query.evidence_sha256)

    def test_read_provenance_tables_are_append_only(self) -> None:
        capacity, decision, _, receipt_hashes = (
            self._capacity_decision()
        )
        mutations = (
            (
                "UPDATE broker_read_receipts SET completeness = 'INELIGIBLE' "
                "WHERE receipt_sha256 = ?",
                (receipt_hashes[0],),
            ),
            (
                "DELETE FROM broker_read_receipts WHERE receipt_sha256 = ?",
                (receipt_hashes[0],),
            ),
            (
                "UPDATE broker_read_manifests SET completeness = 'UNSTABLE' "
                "WHERE evidence_sha256 = ?",
                (capacity.evidence_sha256,),
            ),
            (
                "DELETE FROM broker_read_manifests WHERE evidence_sha256 = ?",
                (capacity.evidence_sha256,),
            ),
            (
                "UPDATE broker_read_manifest_members SET member_role = 'changed' "
                "WHERE evidence_sha256 = ? AND member_ordinal = 0",
                (capacity.evidence_sha256,),
            ),
            (
                "DELETE FROM broker_read_manifest_members "
                "WHERE evidence_sha256 = ? AND member_ordinal = 0",
                (capacity.evidence_sha256,),
            ),
            (
                "UPDATE capacity_decisions SET cap_amount = '1' "
                "WHERE capacity_decision_sha256 = ?",
                (decision.decision_sha256,),
            ),
            (
                "DELETE FROM capacity_decisions "
                "WHERE capacity_decision_sha256 = ?",
                (decision.decision_sha256,),
            ),
        )
        for statement, parameters in mutations:
            with self.subTest(statement=statement):
                with sqlite3.connect(self.path) as conn:
                    with self.assertRaises(sqlite3.DatabaseError):
                        conn.execute(statement, parameters)

    def test_fresh_schema_has_provenance_foreign_keys_and_triggers(
        self,
    ) -> None:
        required_foreign_keys = {
            ("broker_read_manifest_members", "broker_read_manifests"),
            ("broker_read_manifest_members", "broker_read_receipts"),
            ("capacity_decisions", "broker_read_manifests"),
            ("reservation_caps", "capacity_decisions"),
            ("margin_reservations", "capacity_decisions"),
            ("order_events", "broker_read_manifests"),
        }
        required_triggers = {
            "prevent_broker_read_receipt_update": "before update",
            "prevent_broker_read_receipt_delete": "before delete",
            "prevent_broker_read_manifest_update": "before update",
            "prevent_broker_read_manifest_delete": "before delete",
            "prevent_broker_read_member_update": "before update",
            "prevent_broker_read_member_delete": "before delete",
            "prevent_capacity_decision_update": "before update",
            "prevent_capacity_decision_delete": "before delete",
        }
        with sqlite3.connect(self.path) as connection:
            actual_foreign_keys = {
                (table, row[2])
                for table in {
                    table for table, _target in required_foreign_keys
                }
                for row in connection.execute(
                    f"PRAGMA foreign_key_list({table})"
                )
            }
            self.assertTrue(
                required_foreign_keys.issubset(actual_foreign_keys)
            )
            rows = connection.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type = 'trigger'
                """
            ).fetchall()
            triggers = {name: sql for name, sql in rows}
            self.assertTrue(required_triggers.keys() <= triggers.keys())
            for name, expected_action in required_triggers.items():
                normalized = " ".join(triggers[name].lower().split())
                self.assertIn(expected_action, normalized)
                self.assertIn("raise(abort", normalized)

    def test_malformed_provenance_schema_does_not_promote_metadata(
        self,
    ) -> None:
        malformed_table_path = (
            Path(self.tmp.name)
            / "malformed-table"
            / "orders.sqlite3"
        )
        malformed_table_path.parent.mkdir(mode=0o700)
        with sqlite3.connect(malformed_table_path) as connection:
            connection.executescript(
                """
                CREATE TABLE ledger_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                INSERT INTO ledger_metadata VALUES (1, 10);
                CREATE TABLE broker_read_receipts (
                    receipt_sha256 TEXT PRIMARY KEY
                );
                """
            )
        os.chmod(malformed_table_path, 0o600)
        with self.assertRaises(
            (sqlite3.DatabaseError, OrderIntentLedgerError)
        ):
            OrderIntentLedger(
                malformed_table_path,
                clock=self.clock,
                run_id="malformed-table",
            )
        with sqlite3.connect(malformed_table_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                10,
            )

        malformed_trigger_path = (
            Path(self.tmp.name)
            / "malformed-trigger"
            / "orders.sqlite3"
        )
        valid = OrderIntentLedger(
            malformed_trigger_path,
            clock=self.clock,
            run_id="valid-before-trigger-tamper",
        )
        self.assertEqual(valid.path, malformed_trigger_path)
        with sqlite3.connect(malformed_trigger_path) as connection:
            connection.executescript(
                """
                DROP TRIGGER prevent_broker_read_receipt_update;
                CREATE TRIGGER prevent_broker_read_receipt_update
                AFTER UPDATE ON broker_read_receipts
                BEGIN
                    SELECT 1;
                END;
                UPDATE ledger_metadata SET schema_version = 10
                WHERE singleton = 1;
                """
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                malformed_trigger_path,
                clock=self.clock,
                run_id="malformed-trigger",
            )
        with sqlite3.connect(malformed_trigger_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                10,
            )

    def test_schema_10_migration_adds_read_provenance_columns(self) -> None:
        migration_path = (
            Path(self.tmp.name) / "migration" / "orders.sqlite3"
        )
        migration_path.parent.mkdir(mode=0o700)
        with sqlite3.connect(migration_path) as conn:
            conn.executescript(
                """
                CREATE TABLE ledger_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                INSERT INTO ledger_metadata VALUES (1, 10);
                CREATE TABLE reservation_caps (
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    cap_amount TEXT NOT NULL,
                    broker_buying_power TEXT NOT NULL,
                    risk_budget TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    portfolio_snapshot_digest TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (account_id, environment)
                );
                CREATE TABLE margin_reservations (
                    intent_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    risk_decision_id TEXT NOT NULL,
                    max_loss_amount TEXT NOT NULL,
                    quote_observed_at INTEGER NOT NULL,
                    quote_digest TEXT NOT NULL,
                    portfolio_observed_at INTEGER NOT NULL,
                    portfolio_snapshot_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    released_reason_code TEXT,
                    created_at INTEGER NOT NULL,
                    released_at INTEGER
                );
                CREATE TABLE order_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    actor TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    broker_status TEXT,
                    broker_order_id TEXT,
                    observed_at INTEGER,
                    evidence_operation TEXT,
                    http_status INTEGER,
                    raw_response_digest TEXT,
                    created_at INTEGER NOT NULL
                );
                """
            )
        os.chmod(migration_path, 0o600)

        migrated = OrderIntentLedger(
            migration_path,
            clock=self.clock,
            run_id="schema-10-migration",
        )
        self.assertEqual(migrated.path, migration_path)
        with sqlite3.connect(migration_path) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertIn(
                "capacity_decision_sha256",
                {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(reservation_caps)"
                    )
                },
            )
            self.assertIn(
                "capacity_decision_sha256",
                {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(margin_reservations)"
                    )
                },
            )
            self.assertIn(
                "broker_read_evidence_sha256",
                {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(order_events)"
                    )
                },
            )
            for table in (
                "broker_read_receipts",
                "broker_read_manifests",
                "broker_read_manifest_members",
                "capacity_decisions",
            ):
                self.assertIsNotNone(
                    conn.execute(
                        """
                        SELECT 1 FROM sqlite_master
                        WHERE type = 'table' AND name = ?
                        """,
                        (table,),
                    ).fetchone()
                )


if __name__ == "__main__":
    unittest.main()
