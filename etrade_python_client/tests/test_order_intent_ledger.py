from __future__ import annotations

import os
import json
import hashlib
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any
from urllib.parse import quote

from live_trading.etrade_broker_reader import (
    _PARSER_CODE_SHA256,
    _PARSER_CONFIG_SHA256,
    _PARSER_SCHEMA,
    _reparse_broker_read_response,
)

from live_trading.order_intent_ledger import (
    SCHEMA_VERSION,
    BrokerEvidence,
    BrokerReadEvidenceRef,
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    BrokerReadResponseEvidence,
    AccountCapacityEvidence,
    OutboundAuthorization,
    TransportRequestEvidence,
    TransportResponseEvidence,
    OrderIntent,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLedgerError,
    OrderIntentLeaseConflict,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    OrderIntentTransitionError,
    OrderIntentValidationError,
    RiskEvidence,
    _capacity_policy_material,
    _capacity_policy_v1_material,
    _evaluate_opening_capacity_policy,
    _evaluate_opening_capacity_policy_version,
    canonical_order_payload,
    canonical_order_payload_hash,
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def domain_json_hash(domain: bytes, value: Any) -> str:
    return hashlib.sha256(
        domain + canonical_json(value).encode("utf-8")
    ).hexdigest()


def make_intent(*, account="1000000001", environment="production", key="key-1", decision="decision-1", kind="OPENING", strike=620):
    order_payload = raw_payload(strike)
    if kind == "CLOSING":
        order_payload["legs"][0]["orderAction"] = "SELL_CLOSE"
        order_payload["legs"][1]["orderAction"] = "BUY_CLOSE"
    return OrderIntent.build(
        account_id=account, environment=environment, strategy_id="credit-spread", decision_id=decision,
        idempotency_scope="decision", idempotency_key=key, intent_kind=kind, order_payload=order_payload,
    )


def evidence(record, clock, *, operation="SUBMIT_ACK", outcome="OPEN", broker_order_id="2000000001", client_order_id=None, observed_at=None):
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
    account_id="1000000001",
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
        self.read_authorization_counter = 0
        self.capacity_manifest_cache = {}
        self.latest_capacity_decisions = {}

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def account_key(account):
        suffix = hashlib.sha256(account.encode("ascii")).hexdigest()[:16]
        return f"account-key-{suffix}"

    def record_read(
        self,
        ledger,
        *,
        account,
        environment,
        observed_at,
        read_kind,
        route,
        parsed=None,
        raw,
        query=(),
        target_broker_order_id=None,
        final_response=False,
        return_parsed=False,
    ):
        self.read_authorization_counter += 1
        authorization_sha256 = hashlib.sha256(
            (
                "order-ledger-test-authorization:"
                f"{self.read_authorization_counter}"
            ).encode("ascii")
        ).hexdigest()
        response_completed_at = (
            observed_at
            if final_response
            else observed_at
            - timedelta(seconds=30)
            + timedelta(milliseconds=self.read_authorization_counter)
        )
        origin = (
            "https://api.etrade.com"
            if environment == "production"
            else "https://apisb.etrade.com"
        )
        fixture = BrokerReadResponseEvidence(
                read_kind=read_kind,
                account_id=account,
                account_id_key=self.account_key(account),
                institution_type="BROKERAGE",
                environment=environment,
                origin=origin,
                route=route,
                query_json=canonical_json(
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
                canonical_parsed_json=canonical_json(None),
                completeness="INELIGIBLE",
            )
        canonical_parsed_json, completeness = (
            _reparse_broker_read_response(fixture)
        )
        self.assertEqual(completeness, "COMPLETE")
        if parsed is not None:
            self.assertEqual(
                canonical_parsed_json,
                canonical_json(parsed),
            )
        receipt = ledger.record_broker_read_response(
            replace(
                fixture,
                canonical_parsed_json=canonical_parsed_json,
                completeness=completeness,
            )
        )
        if return_parsed:
            return receipt, json.loads(canonical_parsed_json)
        return receipt

    def set_capacity(
        self,
        *,
        ledger=None,
        account="1000000001",
        environment="production",
        observed_at=None,
        buying_power="1000",
        risk_budget="1000",
        state_marker="default",
    ):
        ledger = ledger or self.ledger
        observed_at = observed_at or self.clock.now
        cache_key = (
            str(ledger.path),
            account,
            environment,
            observed_at,
            str(buying_power),
            str(risk_budget),
            state_marker,
        )
        cached = self.capacity_manifest_cache.get(cache_key)
        if cached is None:
            account_key = self.account_key(account)
            origin = (
                "https://api.etrade.com"
                if environment == "production"
                else "https://apisb.etrade.com"
            )
            binding = {
                "account_id": account,
                "account_id_key": account_key,
                "institution_type": "BROKERAGE",
                "account_status": "ACTIVE",
                "account_mode": "MARGIN",
                "account_type": "INDIVIDUAL",
            }
            balance_as_of = str(
                int(
                    (
                        observed_at - timedelta(seconds=30)
                    ).timestamp()
                    * 1_000
                )
            )
            if state_marker == "default":
                positions = []
            else:
                position_id = str(
                    int(
                        hashlib.sha256(
                            state_marker.encode("ascii")
                        ).hexdigest()[:15],
                        16,
                    )
                    + 1
                )
                positions = [
                    {
                        "position_id": position_id,
                        "account_id": account,
                        "product": {
                            "symbol": "SPY",
                            "security_type": "EQ",
                            "call_put": None,
                            "expiry_year": None,
                            "expiry_month": None,
                            "expiry_day": None,
                            "strike_price": None,
                            "product_id": None,
                        },
                        "quantity": "1",
                        "position_type": "LONG",
                        "position_indicator": None,
                        "osi_key": None,
                        "option_multiplier": None,
                        "options_adjusted_flag": None,
                        "deliverables": None,
                        "lots": [],
                    }
                ]
            sources = [
                (
                    "binding.start",
                    "ACCOUNT_LIST",
                    "/v1/accounts/list.json",
                    (),
                    binding,
                )
            ]
            for scan in ("a", "b"):
                sources.extend(
                    (
                        (
                            f"scan_{scan}.balance",
                            "BALANCE",
                            f"/v1/accounts/{quote(account_key, safe='')}/balance.json",
                            (
                                ("instType", "BROKERAGE"),
                                ("realTimeNAV", "true"),
                            ),
                            {
                                "account_id": account,
                                "institution_type": "BROKERAGE",
                                "margin_buying_power": str(buying_power),
                                "as_of_date": balance_as_of,
                            },
                        ),
                        (
                            f"scan_{scan}.portfolio.0001",
                            "PORTFOLIO_PAGE",
                            f"/v1/accounts/{quote(account_key, safe='')}/portfolio.json",
                            (
                                ("count", "50"),
                                ("lotsRequired", "true"),
                                ("marketSession", "REGULAR"),
                                ("pageNumber", "1"),
                                ("sortBy", "SYMBOL"),
                                ("sortOrder", "ASC"),
                                ("totalsRequired", "false"),
                                ("view", "COMPLETE"),
                            ),
                            {
                                "page_number": 1,
                                "total_pages": 1,
                                "metadata_field": "totalNoOfPages",
                                "next_page": None,
                                "positions": positions,
                            },
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
                            f"/v1/accounts/{quote(account_key, safe='')}/orders.json",
                            (("count", "100"), ("status", lane)),
                            {
                                "status_lane": lane,
                                "orders": [],
                                "marker": None,
                            },
                        )
                    )
            sources.append(
                (
                    "binding.end",
                    "ACCOUNT_LIST",
                    "/v1/accounts/list.json",
                    (),
                    binding,
                )
            )
            members = []
            for role, read_kind, route, query, parsed in sources:
                if read_kind == "ACCOUNT_LIST":
                    raw_document = {
                        "AccountListResponse": {
                            "Accounts": {
                                "Account": [
                                    {
                                        "accountId": account,
                                        "accountIdKey": account_key,
                                        "institutionType": "BROKERAGE",
                                        "accountStatus": "ACTIVE",
                                        "accountMode": "MARGIN",
                                        "accountType": "INDIVIDUAL",
                                    }
                                ]
                            }
                        }
                    }
                elif read_kind == "BALANCE":
                    raw_document = {
                        "BalanceResponse": {
                            "accountId": account,
                            "institutionType": "BROKERAGE",
                            "asOfDate": balance_as_of,
                            "Computed": {
                                "marginBuyingPower": str(buying_power)
                            },
                        }
                    }
                elif read_kind == "PORTFOLIO_PAGE":
                    raw_document = {
                        "PortfolioResponse": {
                            "AccountPortfolio": [
                                {
                                    "accountId": account,
                                    "totalNoOfPages": 1,
                                    "Position": [
                                        {
                                            "positionId": position[
                                                "position_id"
                                            ],
                                            "accountId": account,
                                            "Product": {
                                                "symbol": "SPY",
                                                "securityType": "EQ",
                                            },
                                            "quantity": "1",
                                            "positionType": "LONG",
                                        }
                                        for position in positions
                                    ],
                                }
                            ]
                        }
                    }
                else:
                    self.assertEqual(read_kind, "OPEN_ORDERS_PAGE")
                    raw_document = {"OrdersResponse": {"Order": []}}
                raw = canonical_json(raw_document).encode("ascii")
                receipt = self.record_read(
                    ledger,
                    account=account,
                    environment=environment,
                    observed_at=observed_at,
                    read_kind=read_kind,
                    route=route,
                    query=query,
                    parsed=parsed,
                    raw=raw,
                    final_response=role == "binding.end",
                )
                members.append(
                    BrokerReadManifestMember(
                        role=role,
                        receipt_sha256=receipt.receipt_sha256,
                    )
                )
            economic_state = {
                "schema": "etrade-capacity.v3",
                "account_status": "ACTIVE",
                "account_mode": "MARGIN",
                "account_type": "INDIVIDUAL",
                "broker_buying_power": str(buying_power),
                "positions": positions,
                "open_orders": [],
            }
            result = dict(economic_state)
            result["broker_buying_power_as_of"] = balance_as_of
            result["state_sha256"] = domain_json_hash(
                b"etrade-capacity-state.v3\0", economic_state
            )
            reference = ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="CAPACITY",
                    account_id=account,
                    account_id_key=account_key,
                    institution_type="BROKERAGE",
                    environment=environment,
                    origin=origin,
                    target_broker_order_id=None,
                    observed_at=observed_at,
                    completeness="COMPLETE",
                    canonical_result_json=canonical_json(result),
                ),
                tuple(members),
            )
            self.capacity_manifest_cache[cache_key] = reference
        else:
            reference = cached
        decision = ledger.set_reservation_cap_from_read(
            reference,
            risk_budget=Decimal(str(risk_budget)),
            daily_risk_budget=Decimal(str(risk_budget)),
        )
        self.latest_capacity_decisions[
            (str(ledger.path), account, environment)
        ] = decision
        return decision

    def capacity_decision_for(
        self, record, *, ledger=None
    ):
        ledger = ledger or self.ledger
        key = (
            str(ledger.path),
            record.envelope.account_id,
            record.envelope.environment,
        )
        return self.latest_capacity_decisions[key]

    def risk(
        self,
        record,
        *,
        ledger=None,
        collateral="500",
        max_loss="400",
        quote_time=None,
        decision=None,
        portfolio_observed_at=None,
        portfolio_snapshot_digest=None,
        capacity_decision_sha256=None,
    ):
        decision = decision or self.capacity_decision_for(
            record, ledger=ledger
        )
        return RiskEvidence(
            decision_id=record.envelope.decision_id,
            max_loss_amount=Decimal(max_loss),
            collateral_amount=Decimal(collateral),
            quote_observed_at=quote_time or self.clock.now,
            quote_digest="a" * 64,
            portfolio_observed_at=(
                portfolio_observed_at or decision.observed_at
            ),
            portfolio_snapshot_digest=(
                portfolio_snapshot_digest
                or decision.portfolio_snapshot_digest
            ),
            capacity_decision_sha256=(
                capacity_decision_sha256 or decision.decision_sha256
            ),
        )

    def capacity_evidence(self, decision):
        return AccountCapacityEvidence(
            account_id=self._capacity_account(decision),
            environment=self._capacity_environment(decision),
            broker_buying_power=decision.broker_buying_power,
            risk_budget=decision.risk_budget,
            observed_at=decision.observed_at,
            portfolio_snapshot_digest=decision.portfolio_snapshot_digest,
            broker_read_evidence_sha256=decision.evidence_sha256,
        )

    def _capacity_account(self, decision):
        with sqlite3.connect(self.path) as conn:
            return conn.execute(
                """
                SELECT account_id FROM capacity_decisions
                WHERE capacity_decision_sha256 = ?
                """,
                (decision.decision_sha256,),
            ).fetchone()[0]

    def _capacity_environment(self, decision):
        with sqlite3.connect(self.path) as conn:
            return conn.execute(
                """
                SELECT environment FROM capacity_decisions
                WHERE capacity_decision_sha256 = ?
                """,
                (decision.decision_sha256,),
            ).fetchone()[0]

    def downgrade_to_schema_17(self):
        """Rebuild the two policy tables to their actual schema-17 shape."""

        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            decision = conn.execute(
                """
                SELECT * FROM capacity_decisions
                ORDER BY decided_at DESC
                LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(decision)
            legacy_material = {
                "evidence_sha256": decision["evidence_sha256"],
                "account_id": decision["account_id"],
                "environment": decision["environment"],
                "broker_buying_power":
                    decision["broker_buying_power"],
                "risk_budget": decision["risk_budget"],
                "cap_amount": decision["cap_amount"],
                "observed_at": int(decision["observed_at"]),
                "capacity_snapshot_sha256":
                    decision["capacity_snapshot_sha256"],
                "decided_at": int(decision["decided_at"]),
            }
            legacy_sha256 = domain_json_hash(
                b"etrade-capacity-decision.v1\0", legacy_material
            )
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN")
            trigger_names = [
                row["name"]
                for row in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'trigger'
                      AND (
                            sql LIKE '%capacity_decisions%'
                            OR sql LIKE '%reservation_caps%'
                            OR name =
                                'prevent_margin_reservation_identity_update'
                          )
                    """
                )
            ]
            for trigger_name in trigger_names:
                conn.execute(
                    f'DROP TRIGGER "{trigger_name}"'
                )
            conn.execute(
                """
                UPDATE margin_reservations
                SET capacity_decision_sha256 = ?
                WHERE capacity_decision_sha256 = ?
                """,
                (
                    legacy_sha256,
                    decision["capacity_decision_sha256"],
                ),
            )
            conn.execute(
                """
                CREATE TABLE reservation_caps_v17 (
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    cap_amount TEXT NOT NULL,
                    broker_buying_power TEXT NOT NULL,
                    risk_budget TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    portfolio_snapshot_digest TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    capacity_decision_sha256 TEXT
                        REFERENCES capacity_decisions(
                            capacity_decision_sha256
                        ),
                    PRIMARY KEY (account_id, environment),
                    CHECK (
                        CAST(cap_amount AS REAL) >= 0
                        AND CAST(broker_buying_power AS REAL) >= 0
                        AND CAST(risk_budget AS REAL) >= 0
                    ),
                    CHECK (length(portfolio_snapshot_digest) = 64)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO reservation_caps_v17 (
                    account_id, environment, cap_amount,
                    broker_buying_power, risk_budget, observed_at,
                    portfolio_snapshot_digest, updated_at,
                    capacity_decision_sha256
                )
                SELECT account_id, environment, cap_amount,
                       broker_buying_power, risk_budget, observed_at,
                       portfolio_snapshot_digest, updated_at, ?
                FROM reservation_caps
                """,
                (legacy_sha256,),
            )
            conn.execute(
                """
                CREATE TABLE capacity_decisions_v17 (
                    capacity_decision_sha256 TEXT NOT NULL PRIMARY KEY
                        CHECK (length(capacity_decision_sha256) = 64),
                    evidence_sha256 TEXT NOT NULL
                        REFERENCES broker_read_manifests(evidence_sha256),
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL
                        CHECK (
                            environment IN ('sandbox', 'production')
                        ),
                    broker_buying_power TEXT NOT NULL,
                    risk_budget TEXT NOT NULL,
                    cap_amount TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    capacity_snapshot_sha256 TEXT NOT NULL
                        CHECK (
                            length(capacity_snapshot_sha256) = 64
                        ),
                    decided_at INTEGER NOT NULL,
                    CHECK (
                        CAST(broker_buying_power AS REAL) >= 0
                        AND CAST(risk_budget AS REAL) >= 0
                        AND CAST(cap_amount AS REAL) >= 0
                    )
                )
                """
            )
            conn.execute(
                """
                INSERT INTO capacity_decisions_v17 (
                    capacity_decision_sha256, evidence_sha256,
                    account_id, environment, broker_buying_power,
                    risk_budget, cap_amount, observed_at,
                    capacity_snapshot_sha256, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (legacy_sha256, *legacy_material.values()),
            )
            conn.execute("DROP TABLE reservation_caps")
            conn.execute("DROP TABLE capacity_decisions")
            conn.execute(
                """
                ALTER TABLE capacity_decisions_v17
                RENAME TO capacity_decisions
                """
            )
            conn.execute(
                """
                ALTER TABLE reservation_caps_v17
                RENAME TO reservation_caps
                """
            )
            conn.execute(
                """
                DROP TRIGGER IF EXISTS
                    validate_margin_reservation_pre_post_release
                """
            )
            conn.execute(
                """
                CREATE TRIGGER
                    validate_margin_reservation_pre_post_release
                BEFORE UPDATE ON margin_reservations
                WHEN OLD.state = 'ACTIVE'
                 AND NEW.state = 'RELEASED'
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'schema-17 pre-post release guard'
                    );
                END
                """
            )
            conn.execute(
                """
                UPDATE ledger_metadata
                SET schema_version = 17
                WHERE singleton = 1
                """
            )
            conn.commit()
            self.assertEqual(
                conn.execute("PRAGMA foreign_key_check").fetchall(), []
            )
        return legacy_sha256

    def downgrade_to_schema_18_policy_v1(self):
        """Rewrite the active cap as an exact pre-fix schema-18 V1 record."""

        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            current = conn.execute(
                """
                SELECT * FROM capacity_decisions
                ORDER BY decided_at DESC
                LIMIT 1
                """
            ).fetchone()
            self.assertIsNotNone(current)
            self.assertEqual(
                current["risk_policy_version"],
                "OPENING_MAX_LOSS_V2",
            )
            self.assertEqual(current["external_position_risk"], "0")
            self.assertEqual(current["external_order_risk"], "0")
            self.assertEqual(current["represented_managed_risk"], "0")
            policy_inputs = json.loads(current["policy_inputs_json"])
            policy_inputs["schema"] = (
                "etrade-opening-capacity-policy-inputs.v1"
            )
            policy_inputs["policy"] = _capacity_policy_v1_material()
            policy_inputs_json = canonical_json(policy_inputs)
            policy_inputs_sha256 = domain_json_hash(
                b"etrade-capacity-policy-inputs.v1\0",
                policy_inputs,
            )
            risk_policy_sha256 = domain_json_hash(
                b"etrade-capacity-policy.v1\0",
                _capacity_policy_v1_material(),
            )
            decision_material = {
                "evidence_sha256": current["evidence_sha256"],
                "account_id": current["account_id"],
                "environment": current["environment"],
                "broker_buying_power":
                    current["broker_buying_power"],
                "risk_budget": current["risk_budget"],
                "daily_risk_budget": current["daily_risk_budget"],
                "daily_authorized_risk":
                    current["daily_authorized_risk"],
                "external_position_risk":
                    current["external_position_risk"],
                "external_order_risk":
                    current["external_order_risk"],
                "represented_managed_risk":
                    current["represented_managed_risk"],
                "daily_window_start":
                    int(current["daily_window_start"]),
                "daily_window_end":
                    int(current["daily_window_end"]),
                "risk_policy_version": "OPENING_MAX_LOSS_V1",
                "risk_policy_sha256": risk_policy_sha256,
                "policy_inputs_json": policy_inputs_json,
                "policy_inputs_sha256": policy_inputs_sha256,
                "policy_outcome": current["policy_outcome"],
                "policy_reason_code":
                    current["policy_reason_code"],
                "cap_amount": current["cap_amount"],
                "observed_at": int(current["observed_at"]),
                "capacity_snapshot_sha256":
                    current["capacity_snapshot_sha256"],
                "decided_at": int(current["decided_at"]),
            }
            decision_sha256 = domain_json_hash(
                b"etrade-capacity-decision.v2\0",
                decision_material,
            )
            conn.execute(
                """
                INSERT INTO capacity_decisions (
                    capacity_decision_sha256, evidence_sha256,
                    account_id, environment, broker_buying_power,
                    risk_budget, daily_risk_budget,
                    daily_authorized_risk, external_position_risk,
                    external_order_risk, represented_managed_risk,
                    daily_window_start, daily_window_end,
                    risk_policy_version, risk_policy_sha256,
                    policy_inputs_json, policy_inputs_sha256,
                    policy_outcome, policy_reason_code, cap_amount,
                    observed_at, capacity_snapshot_sha256, decided_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (decision_sha256, *decision_material.values()),
            )
            conn.execute(
                """
                UPDATE reservation_caps
                SET cap_amount = ?, broker_buying_power = ?,
                    risk_budget = ?, daily_risk_budget = ?,
                    daily_authorized_risk = ?,
                    external_position_risk = ?,
                    external_order_risk = ?,
                    represented_managed_risk = ?,
                    daily_window_start = ?, daily_window_end = ?,
                    risk_policy_version = ?, risk_policy_sha256 = ?,
                    policy_inputs_json = ?, policy_inputs_sha256 = ?,
                    policy_outcome = ?, policy_reason_code = ?,
                    observed_at = ?,
                    portfolio_snapshot_digest = ?,
                    capacity_decision_sha256 = ?
                WHERE account_id = ? AND environment = ?
                """,
                (
                    decision_material["cap_amount"],
                    decision_material["broker_buying_power"],
                    decision_material["risk_budget"],
                    decision_material["daily_risk_budget"],
                    decision_material["daily_authorized_risk"],
                    decision_material["external_position_risk"],
                    decision_material["external_order_risk"],
                    decision_material["represented_managed_risk"],
                    decision_material["daily_window_start"],
                    decision_material["daily_window_end"],
                    decision_material["risk_policy_version"],
                    decision_material["risk_policy_sha256"],
                    decision_material["policy_inputs_json"],
                    decision_material["policy_inputs_sha256"],
                    decision_material["policy_outcome"],
                    decision_material["policy_reason_code"],
                    decision_material["observed_at"],
                    decision_material["capacity_snapshot_sha256"],
                    decision_sha256,
                    decision_material["account_id"],
                    decision_material["environment"],
                ),
            )
            conn.execute(
                """
                DROP TRIGGER prevent_margin_reservation_identity_update
                """
            )
            conn.execute(
                """
                UPDATE margin_reservations
                SET capacity_decision_sha256 = ?
                WHERE capacity_decision_sha256 = ?
                """,
                (
                    decision_sha256,
                    current["capacity_decision_sha256"],
                ),
            )
            conn.execute(
                """
                UPDATE ledger_metadata
                SET schema_version = 18
                WHERE singleton = 1
                """
            )
            conn.commit()
            self.assertEqual(
                conn.execute("PRAGMA foreign_key_check").fetchall(), []
            )
        return decision_sha256

    def query_evidence(
        self,
        record,
        *,
        ledger=None,
        operation="ORDER_QUERY",
        outcome="OPEN",
        broker_order_id="2000000001",
        observed_at=None,
    ):
        ledger = ledger or self.ledger
        observed_at = observed_at or self.clock.now
        account = record.envelope.account_id
        environment = record.envelope.environment
        account_key = self.account_key(account)
        origin = (
            "https://api.etrade.com"
            if environment == "production"
            else "https://apisb.etrade.com"
        )
        binding = {
            "account_id": account,
            "account_id_key": account_key,
            "institution_type": "BROKERAGE",
            "account_status": "ACTIVE",
            "account_mode": "MARGIN",
            "account_type": "INDIVIDUAL",
        }
        payload_hashes = [
            ledger.expected_order_payload_hash(record.intent_id)
        ]
        binding_raw = canonical_json(
            {
                "AccountListResponse": {
                    "Accounts": {
                        "Account": [
                            {
                                "accountId": account,
                                "accountIdKey": account_key,
                                "institutionType": "BROKERAGE",
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
        raw_status = "EXECUTED" if outcome == "FILLED" else outcome
        placed_time = str(
            int((observed_at - timedelta(minutes=1)).timestamp() * 1_000)
        )
        executed_time = str(
            int((observed_at - timedelta(seconds=30)).timestamp() * 1_000)
        )
        raw = canonical_json(
            {
                "OrdersResponse": {
                    "Order": [
                        {
                            "orderId": broker_order_id,
                            "orderType": "SPREADS",
                            "OrderDetail": [
                                {
                                    "accountId": account,
                                    "orderNumber": broker_order_id,
                                    "status": raw_status,
                                    "placedTime": placed_time,
                                    **(
                                        {"executedTime": executed_time}
                                        if outcome == "FILLED"
                                        else {}
                                    ),
                                    "priceType": wire_payload[
                                        "priceType"
                                    ],
                                    "limitPrice": wire_payload[
                                        "limitPrice"
                                    ],
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
                                            "filledQuantity": (
                                                leg["quantity"]
                                                if outcome == "FILLED"
                                                else 0
                                            ),
                                            "cancelQuantity": (
                                                leg["quantity"]
                                                if outcome
                                                in {
                                                    "CANCELLED",
                                                    "REJECTED",
                                                    "EXPIRED",
                                                }
                                                else 0
                                            ),
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
        start = self.record_read(
            ledger,
            account=account,
            environment=environment,
            observed_at=observed_at,
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            parsed=binding,
            raw=binding_raw,
        )
        detail, detail_parsed = self.record_read(
            ledger,
            account=account,
            environment=environment,
            observed_at=observed_at,
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{quote(account_key, safe='')}/orders/"
                f"{quote(broker_order_id, safe='')}.json"
            ),
            raw=raw,
            target_broker_order_id=broker_order_id,
            return_parsed=True,
        )
        self.assertIn(
            payload_hashes[0],
            detail_parsed["order_payload_hashes"],
        )
        end = self.record_read(
            ledger,
            account=account,
            environment=environment,
            observed_at=observed_at,
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            parsed=binding,
            raw=binding_raw,
            final_response=True,
        )
        result = {
            "schema": "etrade-order-query.v2",
            "broker_order_id": broker_order_id,
            "raw_status": detail_parsed["raw_status"],
            "outcome": detail_parsed["outcome"],
            "fill_summary": detail_parsed["fill_summary"],
            "order_payload_hashes": detail_parsed[
                "order_payload_hashes"
            ],
            "http_status": 200,
            "raw_response_digest": hashlib.sha256(raw).hexdigest(),
            "not_found": detail_parsed["not_found"],
            "replacement_links": detail_parsed[
                "replacement_links"
            ],
        }
        reference = ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="ORDER_QUERY",
                account_id=account,
                account_id_key=account_key,
                institution_type="BROKERAGE",
                environment=environment,
                origin=origin,
                target_broker_order_id=broker_order_id,
                observed_at=observed_at,
                completeness="COMPLETE",
                canonical_result_json=canonical_json(result),
            ),
            (
                BrokerReadManifestMember(
                    "binding.start", start.receipt_sha256
                ),
                BrokerReadManifestMember(
                    "order.detail", detail.receipt_sha256
                ),
                BrokerReadManifestMember(
                    "binding.end", end.receipt_sha256
                ),
            ),
        )
        derived = ledger.broker_evidence_from_read(
            record.intent_id, reference, operation=operation
        )
        self.assertIsNotNone(derived)
        return derived

    def opening(self, *, key="key-1", account="1000000001"):
        record = self.ledger.create_intent(make_intent(key=key, account=account)).intent
        self.set_capacity(account=account)
        self.ledger.reserve_margin(record.intent_id, self.risk(record))
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

    def test_trace_free_opening_intent_can_be_safely_abandoned(self):
        record = self.ledger.create_intent(
            make_intent(key="trace-free", decision="trace-free")
        ).intent

        failed = self.ledger.abandon_trace_free_opening_intent(
            record.intent_id
        )
        replay = self.ledger.abandon_trace_free_opening_intent(
            record.intent_id
        )

        self.assertEqual(failed.state, "FAILED")
        self.assertEqual(replay, failed)
        self.assertIsNone(
            self.ledger.get_margin_reservation(record.intent_id)
        )
        self.assertEqual(
            [
                (
                    event.event_type,
                    event.from_state,
                    event.to_state,
                    event.reason_code,
                )
                for event in self.ledger.events(record.intent_id)
            ],
            [
                ("INTENT_CREATED", None, "INTENT", "INTENT_CREATED"),
                (
                    "INTENT_ABANDONED",
                    "INTENT",
                    "FAILED",
                    "PRE_POST_ABORTED",
                ),
            ],
        )
        restarted = OrderIntentLedger(
            self.path, clock=self.clock, run_id="trace-free-restart"
        )
        self.assertEqual(
            restarted.get_intent(record.intent_id).state,
            "FAILED",
        )

    def test_opening_intent_with_reservation_cannot_be_abandoned(self):
        record = self.opening(
            key="reserved-opening",
        )

        with self.assertRaises(OrderIntentTransitionError):
            self.ledger.abandon_trace_free_opening_intent(
                record.intent_id
            )

        self.assertEqual(
            self.ledger.get_intent(record.intent_id).state,
            "INTENT",
        )
        self.assertEqual(
            self.ledger.get_margin_reservation(record.intent_id).state,
            "ACTIVE",
        )

    def test_environment_exposure_and_broker_schema_are_strictly_derived(self):
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="live", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="OPENING", order_payload=raw_payload())
        with self.assertRaises(OrderIntentValidationError):
            OrderIntent.build(account_id="acct", environment="production", strategy_id="s", decision_id="d", idempotency_scope="scope", idempotency_key="key", intent_kind="CLOSING", order_payload=raw_payload())
        closing = OrderIntent.build(
            account_id="acct",
            environment="production",
            strategy_id="s",
            decision_id="d",
            idempotency_scope="scope",
            idempotency_key="closing",
            intent_kind="CLOSING",
            order_payload=closing_payload(),
        )
        self.assertEqual(closing.intent_kind, "CLOSING")
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.create_intent(closing)
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
            account_id="1000000001", environment="production", strategy_id="credit-spread", decision_id="debit",
            idempotency_scope="decision", idempotency_key="debit", intent_kind="OPENING", order_payload=debit,
        )).intent
        self.set_capacity()
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(record.intent_id, self.risk(record, collateral="124.99", max_loss="124.99"))
        reserved = self.ledger.reserve_margin(record.intent_id, self.risk(record, collateral="125", max_loss="125"))
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
        self.set_capacity()
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(record.intent_id, self.risk(record, collateral="0", max_loss="0"))
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(record.intent_id, self.risk(record, collateral="1", max_loss="1"))
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.reserve_margin(record.intent_id, self.risk(record, quote_time=self.clock.now - timedelta(minutes=6)))
        bad = self.risk(record)
        object.__setattr__(bad, "decision_id", "other-decision")
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reserve_margin(record.intent_id, bad)
        reservation = self.ledger.reserve_margin(record.intent_id, self.risk(record))
        self.assertEqual(reservation.max_loss_amount, Decimal("400"))
        self.assertEqual(reservation.quote_digest, "a" * 64)
        changed = self.risk(record)
        object.__setattr__(changed, "quote_digest", "e" * 64)
        with self.assertRaises(OrderIntentTransitionError):
            self.ledger.reserve_margin(record.intent_id, changed)

    def test_evidence_rejects_numeric_identity_datetime_and_validate_subclasses(self):
        record = self.ledger.create_intent(make_intent()).intent
        capacity_decision = self.set_capacity()
        normal_capacity = self.capacity_evidence(capacity_decision)
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.set_reservation_cap(
                BypassCapacityEvidence(**normal_capacity.__dict__)
            )
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.set_reservation_cap(
                AccountCapacityEvidence(
                    account_id="1000000001",
                    environment="production",
                    broker_buying_power=AlwaysSmallDecimal("1"),
                    risk_budget=AlwaysSmallDecimal("1"),
                    observed_at=self.clock.now,
                    portfolio_snapshot_digest=capacity_decision.portfolio_snapshot_digest,
                    broker_read_evidence_sha256=capacity_decision.evidence_sha256,
                )
            )
        self.ledger.set_reservation_cap(normal_capacity)

        normal_risk = self.risk(record)
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
                    portfolio_observed_at=capacity_decision.observed_at,
                    portfolio_snapshot_digest=capacity_decision.portfolio_snapshot_digest,
                    capacity_decision_sha256=capacity_decision.decision_sha256,
                ),
            )
        with localcontext() as decimal_context:
            decimal_context.prec = 1
            with self.assertRaises(OrderIntentReservationError):
                self.ledger.reserve_margin(
                    record.intent_id,
                    self.risk(record, collateral="1", max_loss="1"),
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
        self.set_capacity()
        left = self.ledger.create_intent(make_intent(key="left")).intent
        right = self.ledger.create_intent(make_intent(key="right")).intent
        barrier = threading.Barrier(2)
        def reserve(record):
            barrier.wait()
            try:
                self.ledger.reserve_margin(
                    record.intent_id,
                    self.risk(
                        record, collateral="600", max_loss="600"
                    ),
                )
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
            self.risk(other, collateral="400", max_loss="400"),
        )
        lease = self.ledger.claim_submission(winner.intent_id, "worker-a", lease_seconds=30)
        self.begin_submission(winner, "worker-a", lease)
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.claim_submission(other.intent_id, "worker-b", lease_seconds=30)

    def test_capacity_snapshots_are_ordered_can_reach_zero_and_block_new_claims(self):
        record = self.ledger.create_intent(make_intent()).intent
        self.set_capacity()
        self.ledger.reserve_margin(record.intent_id, self.risk(record))
        with self.assertRaises(OrderIntentIntegrityError):
            self.set_capacity(buying_power="0", risk_budget="0")
        self.clock.advance(1)
        self.assertEqual(
            self.set_capacity(
                buying_power="0", risk_budget="0"
            ).cap_amount,
            Decimal("0"),
        )
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        blocked = self.ledger.create_intent(make_intent(key="zero-cap", decision="zero-cap")).intent
        with self.assertRaises(OrderIntentReservationError):
            self.ledger.reserve_margin(blocked.intent_id, self.risk(blocked))

    def test_identical_equal_time_capacity_is_idempotent_but_conflict_fails(self):
        self.assertEqual(
            self.set_capacity().cap_amount,
            Decimal("1000"),
        )
        self.assertEqual(
            self.set_capacity().cap_amount,
            Decimal("1000"),
        )
        with self.assertRaises(OrderIntentIntegrityError):
            self.set_capacity(state_marker="different")

    def test_capacity_denies_managed_open_order_missing_from_snapshot(self):
        record = self.opening()
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            evidence(
                record,
                self.clock,
                operation="SUBMIT_ACK",
                broker_order_id="2000000101",
            ),
        )

        self.clock.advance(1)
        decision = self.set_capacity()

        self.assertEqual(decision.cap_amount, Decimal("0"))
        self.assertEqual(decision.policy_outcome, "DENY")
        self.assertEqual(
            decision.policy_reason_code,
            "UNSUPPORTED_MANAGED_RISK_STATE",
        )
        self.assertEqual(
            decision.represented_managed_risk, Decimal("0")
        )

    def test_capacity_denies_unsupported_managed_reservation_state(self):
        record = self.opening()
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            evidence(
                record,
                self.clock,
                operation="SUBMIT_ACK",
                broker_order_id="2000000102",
            ),
        )
        terminal = self.query_evidence(
            record,
            outcome="FILLED",
            broker_order_id="2000000102",
        )
        self.ledger.reconcile_terminal(
            record.intent_id, "FILLED", terminal
        )

        self.clock.advance(1)
        decision = self.set_capacity()

        self.assertEqual(decision.cap_amount, Decimal("0"))
        self.assertEqual(decision.policy_outcome, "DENY")
        self.assertEqual(
            decision.policy_reason_code,
            "UNSUPPORTED_MANAGED_RISK_STATE",
        )

    @staticmethod
    def capacity_policy_inputs(*, managed=()):
        return {
            "schema": "etrade-opening-capacity-policy-inputs.v2",
            "policy": _capacity_policy_material(),
            "daily_window": {"start": 1, "end": 2},
            "daily_authorizations": [],
            "managed_reservations": list(managed),
        }

    def test_capacity_denies_even_matching_managed_active_order(self):
        expected_legs = [
            {
                "symbol": "SPY",
                "call_put": "PUT",
                "expiry": "2026-08-21",
                "strike": "620",
                "signed_quantity": "-1",
            },
            {
                "symbol": "SPY",
                "call_put": "PUT",
                "expiry": "2026-08-21",
                "strike": "615",
                "signed_quantity": "1",
            },
        ]
        expected_legs.sort(key=canonical_json)
        managed = {
            "intent_id": "managed-intent",
            "broker_order_id": "2000000103",
            "role": "OPEN_ORDER",
            "reservation_amount": "400",
            "expected_payload_hash": canonical_order_payload_hash(
                raw_payload()
            ),
            "expected_legs": expected_legs,
        }
        matching_order = {
            "order_id": "2000000103",
            "order_type": "SPREADS",
            "replaces_order_id": None,
            "replaced_by_order_id": None,
            "details": [
                {
                    "account_id": "1000000001",
                    "status": "OPEN",
                    "price_type": "NET_CREDIT",
                    "limit_price": "1.25",
                    "order_term": "GOOD_FOR_DAY",
                    "market_session": "REGULAR",
                    "all_or_none": False,
                    "replaces_order_id": None,
                    "replaced_by_order_id": None,
                    "instruments": [
                        {
                            "product": {
                                "symbol": "SPY",
                                "security_type": "OPTN",
                                "call_put": "PUT",
                                "expiry_year": "2026",
                                "expiry_month": "8",
                                "expiry_day": "21",
                                "strike_price": "620",
                                "product_id": None,
                            },
                            "order_action": "SELL_OPEN",
                            "quantity_type": "QUANTITY",
                            "ordered_quantity": "1",
                            "filled_quantity": "0",
                            "cancel_quantity": "0",
                        },
                        {
                            "product": {
                                "symbol": "SPY",
                                "security_type": "OPTN",
                                "call_put": "PUT",
                                "expiry_year": "2026",
                                "expiry_month": "8",
                                "expiry_day": "21",
                                "strike_price": "615",
                                "product_id": None,
                            },
                            "order_action": "BUY_OPEN",
                            "quantity_type": "QUANTITY",
                            "ordered_quantity": "1",
                            "filled_quantity": "0",
                            "cancel_quantity": "0",
                        },
                    ],
                }
            ],
        }

        decision = _evaluate_opening_capacity_policy(
            {"positions": [], "open_orders": [matching_order]},
            self.capacity_policy_inputs(managed=(managed,)),
            broker_buying_power=Decimal("1000"),
            account_risk_budget=Decimal("1000"),
        )

        self.assertEqual(decision.cap_amount, Decimal("0"))
        self.assertEqual(decision.policy_outcome, "DENY")
        self.assertEqual(
            decision.policy_reason_code, "UNSUPPORTED_ACTIVE_ORDER"
        )

    def test_capacity_v2_does_not_add_managed_filled_risk_to_broker_power(
        self,
    ):
        broker_order_id = "2000000105"
        acquired = str(int(self.clock.now.timestamp() * 1_000) - 1)
        positions = []
        expected_legs = []
        for index, (strike, quantity, position_type) in enumerate(
            (("620", "-1", "SHORT"), ("615", "1", "LONG")),
            start=1,
        ):
            position_id = str(200 + index)
            positions.append(
                {
                    "position_id": position_id,
                    "account_id": "1000000001",
                    "product": {
                        "symbol": "SPY",
                        "security_type": "OPTN",
                        "call_put": "PUT",
                        "expiry_year": "2026",
                        "expiry_month": "8",
                        "expiry_day": "21",
                        "strike_price": strike,
                        "product_id": None,
                    },
                    "quantity": quantity,
                    "position_type": position_type,
                    "position_indicator": "TYPE1",
                    "osi_key": (
                        "SPY---260821P00620000"
                        if strike == "620"
                        else "SPY---260821P00615000"
                    ),
                    "option_multiplier": "100",
                    "options_adjusted_flag": False,
                    "deliverables": "100 shares of SPY",
                    "lots": [
                        {
                            "position_id": position_id,
                            "position_lot_id": str(2000 + index),
                            "order_no": broker_order_id,
                            "leg_no": str(index),
                            "original_quantity": quantity,
                            "remaining_quantity": quantity,
                            "available_quantity": quantity,
                            "acquired_date_epoch_ms": acquired,
                        }
                    ],
                }
            )
            expected_legs.append(
                {
                    "symbol": "SPY",
                    "call_put": "PUT",
                    "expiry": "2026-08-21",
                    "strike": strike,
                    "signed_quantity": quantity,
                }
            )
        expected_legs.sort(key=canonical_json)
        managed = {
            "intent_id": "managed-filled-intent",
            "broker_order_id": broker_order_id,
            "role": "POSITION",
            "reservation_amount": "375",
            "expected_payload_hash": canonical_order_payload_hash(
                raw_payload()
            ),
            "expected_legs": expected_legs,
        }

        decision = _evaluate_opening_capacity_policy(
            {"positions": positions, "open_orders": []},
            self.capacity_policy_inputs(managed=(managed,)),
            broker_buying_power=Decimal("625"),
            account_risk_budget=Decimal("1000"),
        )

        self.assertEqual(decision.policy_outcome, "ALLOW")
        self.assertEqual(decision.external_position_risk, Decimal("0"))
        self.assertEqual(decision.external_order_risk, Decimal("0"))
        self.assertEqual(
            decision.represented_managed_risk, Decimal("375")
        )
        self.assertEqual(decision.cap_amount, Decimal("625"))
        v1_inputs = self.capacity_policy_inputs(managed=(managed,))
        v1_inputs["schema"] = (
            "etrade-opening-capacity-policy-inputs.v1"
        )
        v1_inputs["policy"] = _capacity_policy_v1_material()
        replay_only_v1 = _evaluate_opening_capacity_policy_version(
            {"positions": positions, "open_orders": []},
            v1_inputs,
            broker_buying_power=Decimal("625"),
            account_risk_budget=Decimal("1000"),
            policy_version="OPENING_MAX_LOSS_V1",
        )
        self.assertEqual(replay_only_v1.cap_amount, Decimal("1000"))

    def test_fresh_capacity_decisions_use_policy_v2(self):
        decision = self.set_capacity(
            buying_power="625", risk_budget="1000"
        )

        self.assertEqual(
            decision.risk_policy_version, "OPENING_MAX_LOSS_V2"
        )
        self.assertEqual(decision.cap_amount, Decimal("625"))

    def test_capacity_requires_exact_standard_option_deliverables(self):
        def positions(deliverables):
            acquired = str(
                int(self.clock.now.timestamp() * 1_000) - 1
            )
            documents = []
            for index, (strike, quantity, position_type) in enumerate(
                (("620", "-1", "SHORT"), ("615", "1", "LONG")),
                start=1,
            ):
                position_id = str(100 + index)
                documents.append(
                    {
                        "position_id": position_id,
                        "account_id": "1000000001",
                        "product": {
                            "symbol": "SPY",
                            "security_type": "OPTN",
                            "call_put": "PUT",
                            "expiry_year": "2026",
                            "expiry_month": "8",
                            "expiry_day": "21",
                            "strike_price": strike,
                            "product_id": None,
                        },
                        "quantity": quantity,
                        "position_type": position_type,
                        "position_indicator": "TYPE1",
                        "osi_key": (
                            "SPY---260821P00620000"
                            if strike == "620"
                            else "SPY---260821P00615000"
                        ),
                        "option_multiplier": "100",
                        "options_adjusted_flag": False,
                        "deliverables": deliverables,
                        "lots": [
                            {
                                "position_id": position_id,
                                "position_lot_id": str(1000 + index),
                                "order_no": "2000000104",
                                "leg_no": str(index),
                                "original_quantity": quantity,
                                "remaining_quantity": quantity,
                                "available_quantity": quantity,
                                "acquired_date_epoch_ms": acquired,
                            }
                        ],
                    }
                )
            return documents

        for deliverables in (
            None,
            "100 shares",
            "100 shares of SPY",
        ):
            with self.subTest(allowed=deliverables):
                decision = _evaluate_opening_capacity_policy(
                    {
                        "positions": positions(deliverables),
                        "open_orders": [],
                    },
                    self.capacity_policy_inputs(),
                    broker_buying_power=Decimal("1000"),
                    account_risk_budget=Decimal("1000"),
                )
                self.assertEqual(decision.policy_outcome, "ALLOW")
                self.assertEqual(
                    decision.external_position_risk, Decimal("500")
                )
                self.assertEqual(decision.cap_amount, Decimal("500"))

        for deliverables in (
            "100 shares of QQQ",
            "50 shares of SPY",
            "100 Shares of SPY",
            "100 shares of SPY ",
        ):
            with self.subTest(denied=deliverables):
                decision = _evaluate_opening_capacity_policy(
                    {
                        "positions": positions(deliverables),
                        "open_orders": [],
                    },
                    self.capacity_policy_inputs(),
                    broker_buying_power=Decimal("1000"),
                    account_risk_budget=Decimal("1000"),
                )
                self.assertEqual(decision.cap_amount, Decimal("0"))
                self.assertEqual(decision.policy_outcome, "DENY")
                self.assertEqual(
                    decision.policy_reason_code,
                    "UNSUPPORTED_OPTION_POSITION",
                )

    def test_reservation_requires_the_capacity_snapshot_used_for_its_portfolio_risk(self):
        record = self.ledger.create_intent(make_intent()).intent
        decision = self.set_capacity(state_marker="different")
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.reserve_margin(
                record.intent_id,
                self.risk(
                    record,
                    portfolio_snapshot_digest="b" * 64,
                    capacity_decision_sha256=decision.decision_sha256,
                ),
            )

    def test_direct_sql_cannot_forge_an_understated_reservation(self):
        record = self.ledger.create_intent(
            make_intent(key="tiny-direct-reservation")
        ).intent
        decision = self.set_capacity()
        now = int(self.clock.now.timestamp() * 1_000_000)
        values = (
            record.intent_id,
            record.envelope.account_id,
            record.envelope.environment,
            "1",
            record.envelope.decision_id,
            "1",
            now,
            "a" * 64,
            int(decision.observed_at.timestamp() * 1_000_000),
            decision.portfolio_snapshot_digest,
            decision.decision_sha256,
            now,
        )
        statement = """
            INSERT INTO margin_reservations (
                intent_id, account_id, environment, amount,
                risk_decision_id, max_loss_amount,
                quote_observed_at, quote_digest,
                portfolio_observed_at, portfolio_snapshot_digest,
                capacity_decision_sha256, state,
                released_reason_code, created_at, released_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE',
                NULL, ?, NULL
            )
        """
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(statement, values)

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "DROP TRIGGER validate_margin_reservation_insert"
            )
            connection.execute(statement, values)

        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path,
                clock=self.clock,
                run_id="tiny-direct-reservation-restart",
            )

    def test_valid_pre_post_failure_releases_reserved_margin(self):
        record = self.opening(key="valid-pre-post-release")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        failed = self.ledger.mark_pre_post_failed(
            record.intent_id,
            "worker",
            lease.fencing_token,
        )
        reservation = self.ledger.get_margin_reservation(
            record.intent_id
        )
        self.assertEqual(failed.state, "FAILED")
        self.assertEqual(reservation.state, "RELEASED")
        self.assertEqual(
            reservation.released_reason_code, "PRE_POST_ABORTED"
        )
        self.assertEqual(
            [
                (event.event_type, event.reason_code)
                for event in self.ledger.events(record.intent_id)[-2:]
            ],
            [
                ("RESERVATION_RELEASED", "RESERVATION_RELEASED"),
                ("PRE_POST_FAILED", "PRE_POST_ABORTED"),
            ],
        )
        self.assertEqual(
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            ),
            Decimal("0"),
        )
        reopened = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="valid-pre-post-release-restart",
        )
        self.assertEqual(
            reopened.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            ),
            Decimal("0"),
        )

    def test_submitted_opening_cannot_forge_active_reservation_release(
        self,
    ):
        record = self.opening(key="forged-active-release")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            evidence(record, self.clock, operation="SUBMIT_ACK"),
        )
        released_at = int(self.clock.now.timestamp() * 1_000_000)
        statement = """
            UPDATE margin_reservations
            SET state = 'RELEASED',
                released_reason_code = 'RESERVATION_RELEASED',
                released_at = ?
            WHERE intent_id = ?
        """
        with sqlite3.connect(self.path) as connection:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    statement, (released_at, record.intent_id)
                )
        self.assertEqual(
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            ),
            Decimal("400"),
        )

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                DROP TRIGGER
                validate_margin_reservation_pre_post_release
                """
            )
            connection.execute(
                statement, (released_at, record.intent_id)
            )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path,
                clock=self.clock,
                run_id="forged-active-release-restart",
            )

    def test_post_started_evidence_blocks_forged_pre_post_release(
        self,
    ):
        record = self.opening(key="post-started-release-forgery")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        released_at = int(self.clock.now.timestamp() * 1_000_000)
        fail_intent = """
            UPDATE order_intents
            SET state = 'FAILED', pending_operation = NULL,
                pending_owner = NULL, pending_fence = NULL,
                last_reconciled_run = 'forged',
                updated_at = ?
            WHERE intent_id = ?
        """
        release = """
            UPDATE margin_reservations
            SET state = 'RELEASED',
                released_reason_code = 'PRE_POST_ABORTED',
                released_at = ?
            WHERE intent_id = ?
        """
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                fail_intent, (released_at, record.intent_id)
            )
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    release, (released_at, record.intent_id)
                )
            connection.rollback()

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                DROP TRIGGER
                validate_margin_reservation_pre_post_release
                """
            )
            connection.execute(
                fail_intent, (released_at, record.intent_id)
            )
            connection.execute(
                release, (released_at, record.intent_id)
            )
            for event_type, from_state, to_state, actor, reason in (
                (
                    "RESERVATION_RELEASED",
                    "FAILED",
                    "FAILED",
                    "system",
                    "RESERVATION_RELEASED",
                ),
                (
                    "PRE_POST_FAILED",
                    "CLAIMED",
                    "FAILED",
                    "worker",
                    "PRE_POST_ABORTED",
                ),
            ):
                connection.execute(
                    """
                    INSERT INTO order_events (
                        intent_id, account_id, environment,
                        client_order_id, event_type, from_state,
                        to_state, actor, reason_code, broker_status,
                        broker_order_id, observed_at,
                        evidence_operation, http_status,
                        raw_response_digest,
                        broker_read_evidence_sha256, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?
                    )
                    """,
                    (
                        record.intent_id,
                        record.envelope.account_id,
                        record.envelope.environment,
                        record.client_order_id,
                        event_type,
                        from_state,
                        to_state,
                        actor,
                        reason,
                        released_at,
                    ),
                )

        with self.assertRaisesRegex(
            OrderIntentIntegrityError, "durable POST evidence"
        ):
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path,
                clock=self.clock,
                run_id="post-started-release-forgery-restart",
            )

    def test_order_event_delete_trigger_missing_or_replaced_is_rejected(
        self,
    ):
        record = self.opening(key="post-started-delete-trigger")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "DROP TRIGGER prevent_order_event_delete"
            )

        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path,
                clock=self.clock,
                run_id="missing-order-event-delete-trigger",
            )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM order_events
                    WHERE intent_id = ? AND event_type = 'POST_STARTED'
                    """,
                    (record.intent_id,),
                ).fetchone()[0],
                1,
            )

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TRIGGER prevent_order_event_delete
                BEFORE DELETE ON order_events
                WHEN 0
                BEGIN
                    SELECT RAISE(ABORT, 'never runs');
                END;
                """
            )

        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path,
                clock=self.clock,
                run_id="reject-replaced-order-event-delete-trigger",
            )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM order_events
                    WHERE intent_id = ? AND event_type = 'POST_STARTED'
                    """,
                    (record.intent_id,),
                ).fetchone()[0],
                1,
            )

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
            self.ledger.reconciliation_blockers("1000000001", "production"), ()
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
            self.ledger.reconciliation_blockers("1000000001", "production"), ()
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
            self.ledger.reconciliation_blockers("1000000001", "production"), ()
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

    def test_restart_keeps_terminal_risk_until_exact_absorption_evidence(self):
        record = self.opening()
        lease = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
        self.begin_submission(record, "worker", lease)
        self.ledger.record_post_unknown(record.intent_id, "worker", lease.fencing_token, "POST_TIMEOUT")
        restarted = OrderIntentLedger(self.path, clock=self.clock, run_id="run-b")
        self.assertEqual([item.intent_id for item in restarted.reconciliation_blockers("1000000001", "production")], [record.intent_id])
        filled_evidence = self.query_evidence(
            record,
            ledger=restarted,
            outcome="FILLED",
            broker_order_id="2000000001",
        )
        filled = restarted.reconcile_terminal(
            record.intent_id, "FILLED", filled_evidence
        )
        self.assertEqual(filled.state, "FILLED")
        self.assertEqual(restarted.get_margin_reservation(record.intent_id).state, "FILLED_PENDING_ABSORPTION")
        next_opening = restarted.create_intent(make_intent(key="next-opening", decision="next-decision")).intent
        restarted.reserve_margin(
            next_opening.intent_id,
            self.risk(next_opening, ledger=restarted),
        )
        with self.assertRaises(OrderIntentReservationError):
            restarted.claim_submission(next_opening.intent_id, "new-worker", lease_seconds=30)
        self.clock.advance(1)
        refreshed_capacity = self.set_capacity(ledger=restarted)
        with self.assertRaises(OrderIntentReconciliationRequired):
            restarted.absorb_filled_reservation(
                record.intent_id,
                self.capacity_evidence(refreshed_capacity),
            )
        self.clock.advance(1)
        changed_capacity = self.set_capacity(
            ledger=restarted, state_marker="different"
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            restarted.absorb_filled_reservation(
                record.intent_id,
                self.capacity_evidence(changed_capacity),
            )
        self.assertEqual(
            restarted.get_margin_reservation(record.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        event = next(item for item in restarted.events(record.intent_id) if item.reason_code == "BROKER_FILLED")
        self.assertEqual(
            event.raw_response_digest,
            filled_evidence.raw_response_digest,
        )
        self.assertEqual((event.account_id, event.environment, event.client_order_id), ("1000000001", "production", record.client_order_id))
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
        self.assertEqual(self.ledger.reconciliation_blockers("1000000001", "production")[0].intent_id, submitted.intent_id)
        amendment_evidence = self.query_evidence(
            submitted,
            operation="AMEND_QUERY",
            outcome="CANCELLED",
            broker_order_id=submitted.broker_order_id,
        )
        self.ledger.reconcile_terminal(
            submitted.intent_id, "CANCELLED", amendment_evidence
        )
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
        self.assertEqual(history[:2], ("2000000001", amendment.client_order_id))
        self.assertEqual(len(history[2]), 64)

    def test_original_intent_rejects_known_client_id_collision_with_completed_amendment(self):
        source = self.opening(key="source")
        submit = self.ledger.claim_submission(source.intent_id, "worker", lease_seconds=30)
        self.begin_submission(source, "worker", submit)
        self.ledger.record_post_acknowledgement(source.intent_id, "worker", submit.fencing_token, evidence(source, self.clock, operation="SUBMIT_ACK"))
        amendment = self.ledger.acquire_amendment_lease(source.intent_id, "nudger", lease_seconds=30, idempotency_key="amend-68395", amendment_payload=raw_payload())
        self.begin_amendment(source, "nudger", amendment)
        self.ledger.record_amendment_acknowledgement(source.intent_id, "nudger", amendment.fencing_token, evidence(source, self.clock, operation="AMEND_ACK", client_order_id=amendment.client_order_id, broker_order_id="broker-amended"))
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.create_intent(make_intent(key="target-338459"))

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
            self.risk(other, collateral="400", max_loss="400"),
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
        self.set_capacity(buying_power="0", risk_budget="0")
        with self.assertRaises(OrderIntentReservationError):
            self.begin_submission(record, "worker", submit)
        self.assertEqual(self.ledger.get_intent(record.intent_id).state, "CLAIMED")

    def test_zero_fill_terminals_hold_risk_without_fresh_order_evidence(self):
        for terminal in ("CANCELLED", "EXPIRED"):
            with self.subTest(terminal=terminal):
                self.clock.advance(1)
                account = {
                    "CANCELLED": "1000000011",
                    "EXPIRED": "1000000012",
                }[terminal]
                broker_order_id = {
                    "CANCELLED": "2000000011",
                    "EXPIRED": "2000000012",
                }[terminal]
                record = self.opening(key=f"{terminal}-opening", account=account)
                submit = self.ledger.claim_submission(record.intent_id, "worker", lease_seconds=30)
                self.begin_submission(record, "worker", submit)
                self.ledger.record_post_acknowledgement(record.intent_id, "worker", submit.fencing_token, evidence(record, self.clock, operation="SUBMIT_ACK", broker_order_id=broker_order_id))
                terminal_evidence = self.query_evidence(
                    record,
                    outcome=terminal,
                    broker_order_id=broker_order_id,
                )
                self.ledger.reconcile_terminal(
                    record.intent_id, terminal, terminal_evidence
                )
                self.assertEqual(self.ledger.get_margin_reservation(record.intent_id).state, "FILLED_PENDING_ABSORPTION")
                blocked = self.ledger.create_intent(make_intent(account=account, key=f"{terminal}-blocked", decision=f"{terminal}-blocked")).intent
                self.ledger.reserve_margin(blocked.intent_id, self.risk(blocked))
                with self.assertRaises(OrderIntentReservationError):
                    self.ledger.claim_submission(blocked.intent_id, "blocked", lease_seconds=30)
                self.clock.advance(1)
                refreshed_capacity = self.set_capacity(account=account)
                with self.assertRaises(OrderIntentReconciliationRequired):
                    self.ledger.absorb_filled_reservation(
                        record.intent_id,
                        self.capacity_evidence(refreshed_capacity),
                    )
                self.clock.advance(1)
                changed_capacity = self.set_capacity(
                    account=account, state_marker="different"
                )
                with self.assertRaises(OrderIntentReconciliationRequired):
                    self.ledger.absorb_filled_reservation(
                        record.intent_id,
                        self.capacity_evidence(changed_capacity),
                    )
                with self.assertRaises(OrderIntentReservationError):
                    self.ledger.claim_submission(blocked.intent_id, "still-blocked", lease_seconds=30)

    def test_zero_fill_terminal_absorption_is_atomic_and_idempotent(self):
        record = self.opening(key="zero-fill-absorption")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        submitted = self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            evidence(
                record,
                self.clock,
                operation="SUBMIT_ACK",
                broker_order_id="2000000088",
            ),
        )
        terminal = self.query_evidence(
            submitted,
            outcome="CANCELLED",
            broker_order_id="2000000088",
        )
        self.ledger.reconcile_terminal(
            record.intent_id, "CANCELLED", terminal
        )
        with sqlite3.connect(self.path) as connection:
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    """
                    UPDATE margin_reservations
                    SET state = 'RELEASED',
                        released_reason_code = 'ZERO_FILL_CONFIRMED',
                        released_at = ?
                    WHERE intent_id = ?
                    """,
                    (
                        int(self.clock.now.timestamp() * 1_000_000),
                        record.intent_id,
                    ),
                )
        self.clock.advance(1)
        fresh_terminal = self.query_evidence(
            submitted,
            outcome="CANCELLED",
            broker_order_id="2000000088",
        )
        reference = BrokerReadEvidenceRef(
            fresh_terminal.broker_read_evidence_sha256,
            "ORDER_QUERY",
        )
        self.clock.advance(1)
        newest_terminal = self.query_evidence(
            submitted,
            outcome="CANCELLED",
            broker_order_id="2000000088",
        )
        newest_reference = BrokerReadEvidenceRef(
            newest_terminal.broker_read_evidence_sha256,
            "ORDER_QUERY",
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            self.ledger.terminal_absorption_requirement(
                record.intent_id, reference
            )
        reference = newest_reference

        requirement = self.ledger.terminal_absorption_requirement(
            record.intent_id, reference
        )
        self.assertEqual(requirement.classification, "ZERO_FILL")
        self.assertFalse(requirement.post_capacity_required)
        self.assertEqual(
            requirement.baseline_capacity_decision_sha256,
            self.ledger.get_margin_reservation(
                record.intent_id
            ).capacity_decision_sha256,
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TRIGGER abort_zero_absorption_event
                BEFORE INSERT ON order_events
                WHEN NEW.event_type = 'RESERVATION_RELEASED'
                BEGIN
                    SELECT RAISE(ABORT, 'simulated post-receipt failure');
                END
                """
            )
        with self.assertRaises(sqlite3.DatabaseError):
            self.ledger.absorb_terminal_reservation(
                record.intent_id, reference
            )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM reservation_absorptions
                    WHERE intent_id = ?
                    """,
                    (record.intent_id,),
                ).fetchone()[0],
                0,
            )
            connection.execute("DROP TRIGGER abort_zero_absorption_event")
        self.assertEqual(
            self.ledger.get_margin_reservation(record.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        receipt = self.ledger.absorb_terminal_reservation(
            record.intent_id, reference
        )
        replay = self.ledger.absorb_terminal_reservation(
            record.intent_id, reference
        )
        self.assertEqual(replay, receipt)
        self.assertEqual(receipt.absorbed_margin_amount, Decimal("0"))
        self.assertEqual(
            self.ledger.get_intent(record.intent_id).state,
            "CANCELLED",
        )
        self.assertEqual(
            self.ledger.get_margin_reservation(record.intent_id).state,
            "RELEASED",
        )
        self.assertEqual(
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            ),
            Decimal("0"),
        )
        final_event = self.ledger.events(record.intent_id)[-1]
        self.assertEqual(
            final_event.event_type, "RESERVATION_RELEASED"
        )
        self.assertEqual(
            final_event.reason_code, "RESERVATION_RELEASED"
        )
        conflicting = BrokerReadEvidenceRef(
            terminal.broker_read_evidence_sha256,
            "ORDER_QUERY",
        )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.absorb_terminal_reservation(
                record.intent_id, conflicting
            )

    def test_absorption_insert_trigger_rejects_zero_fill_for_filled_intent(
        self,
    ):
        record = self.opening(key="forged-zero-fill")
        lease = self.ledger.claim_submission(
            record.intent_id, "worker", lease_seconds=30
        )
        self.begin_submission(record, "worker", lease)
        submitted = self.ledger.record_post_acknowledgement(
            record.intent_id,
            "worker",
            lease.fencing_token,
            evidence(
                record,
                self.clock,
                operation="SUBMIT_ACK",
                broker_order_id="2000000089",
            ),
        )
        terminal = self.query_evidence(
            submitted,
            outcome="FILLED",
            broker_order_id="2000000089",
        )
        self.ledger.reconcile_terminal(
            record.intent_id, "FILLED", terminal
        )
        reservation = self.ledger.get_margin_reservation(record.intent_id)
        now = int(self.clock.now.timestamp() * 1_000_000)

        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.DatabaseError):
                connection.execute(
                    """
                    INSERT INTO reservation_absorptions (
                        absorption_sha256, intent_id, account_id,
                        environment, broker_order_id, terminal_state,
                        classification, terminal_order_evidence_sha256,
                        baseline_capacity_decision_sha256,
                        post_capacity_decision_sha256,
                        post_capacity_evidence_sha256, ordered_quantity,
                        filled_quantity, placed_time_epoch_ms,
                        executed_time_epoch_ms, canonical_lot_proof_json,
                        lot_proof_sha256, absorbed_margin_amount,
                        observed_at, recorded_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, 'CANCELLED', 'ZERO_FILL', ?, ?,
                        NULL, NULL, 1, 0, ?, NULL, '[]', ?, '0', ?, ?
                    )
                    """,
                    (
                        "f" * 64,
                        record.intent_id,
                        record.envelope.account_id,
                        record.envelope.environment,
                        "2000000089",
                        terminal.broker_read_evidence_sha256,
                        reservation.capacity_decision_sha256,
                        str(int(self.clock.now.timestamp() * 1_000)),
                        "e" * 64,
                        now,
                        now,
                    ),
                )

        self.assertEqual(
            self.ledger.get_margin_reservation(record.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        self.assertEqual(
            self.ledger.active_reserved_margin(
                record.envelope.account_id,
                record.envelope.environment,
            ),
            Decimal("400"),
        )

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

    def test_concurrent_first_initialization_rechecks_metadata_under_lock(self):
        parent = Path(self.tmp.name) / "parallel-initialize"
        parent.mkdir(mode=0o700)
        path = parent / "orders.sqlite3"
        barrier = threading.Barrier(2)

        def initialize(index):
            barrier.wait()
            return OrderIntentLedger(
                path,
                clock=self.clock,
                run_id=f"parallel-initialize-{index}",
            ).path

        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(initialize, (1, 2)))

        self.assertEqual(paths, [path, path])
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT schema_version FROM ledger_metadata"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )

    def test_schema_17_releases_only_never_claimed_opening_capacity(self):
        record = self.opening(key="schema-17-pristine")
        self.assertEqual(
            self.ledger.active_reserved_margin(
                "1000000001", "production"
            ),
            Decimal("400"),
        )
        self.downgrade_to_schema_17()

        migrated = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-18-upgrade",
        )

        self.assertEqual(
            migrated.get_intent(record.intent_id).state, "FAILED"
        )
        reservation = migrated.get_margin_reservation(
            record.intent_id
        )
        self.assertEqual(reservation.state, "RELEASED")
        self.assertEqual(
            reservation.released_reason_code,
            "SCHEMA_18_UNCLAIMED_LEGACY_RESERVATION_RELEASED",
        )
        self.assertEqual(
            migrated.active_reserved_margin(
                "1000000001", "production"
            ),
            Decimal("0"),
        )
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            self.assertEqual(
                conn.execute(
                    """
                    SELECT schema_version FROM ledger_metadata
                    WHERE singleton = 1
                    """
                ).fetchone()["schema_version"],
                SCHEMA_VERSION,
            )
            events = conn.execute(
                """
                SELECT event_type, from_state, to_state, actor,
                       reason_code
                FROM order_events
                WHERE intent_id = ?
                  AND event_type IN (
                        'RESERVATION_RELEASED','PRE_POST_FAILED'
                  )
                ORDER BY sequence
                """,
                (record.intent_id,),
            ).fetchall()
        self.assertEqual(
            [
                (
                    event["event_type"],
                    event["from_state"],
                    event["to_state"],
                    event["actor"],
                    event["reason_code"],
                )
                for event in events
            ],
            [
                (
                    "RESERVATION_RELEASED",
                    "FAILED",
                    "FAILED",
                    "schema-18-migration",
                    "RESERVATION_RELEASED",
                ),
                (
                    "PRE_POST_FAILED",
                    "INTENT",
                    "FAILED",
                    "schema-18-migration",
                    "PRE_POST_ABORTED",
                ),
            ],
        )
        reopened = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-18-reopen",
        )
        self.assertEqual(
            reopened.active_reserved_margin(
                "1000000001", "production"
            ),
            Decimal("0"),
        )

    def test_schema_17_submission_trace_is_fenced_and_never_reclaimable(
        self,
    ):
        record = self.opening(key="schema-17-prior-claim")
        self.ledger.claim_submission(
            record.intent_id, "legacy-worker", lease_seconds=5
        )
        self.clock.advance(6)
        self.assertEqual(
            self.ledger.reconciliation_blockers(
                "1000000001", "production"
            ),
            (),
        )
        self.assertEqual(
            self.ledger.get_intent(record.intent_id).state, "INTENT"
        )
        self.downgrade_to_schema_17()

        migrated = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-18-trace-upgrade",
        )

        self.assertEqual(
            migrated.get_intent(record.intent_id).state,
            "SUBMISSION_UNKNOWN",
        )
        self.assertEqual(
            migrated.get_margin_reservation(record.intent_id).state,
            "ACTIVE",
        )
        self.assertTrue(
            migrated.has_execution_blockers(
                "1000000001", "production"
            )
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            migrated.claim_submission(
                record.intent_id,
                "new-worker",
                lease_seconds=30,
            )

    def test_schema_17_policy_constraint_upgrade_matches_fresh_behavior(
        self,
    ):
        self.opening(key="schema-17-policy-constraints")
        self.downgrade_to_schema_17()
        OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-18-policy-upgrade",
        )
        fresh_parent = Path(self.tmp.name) / "fresh-schema-18"
        fresh_parent.mkdir(mode=0o700)
        fresh_path = fresh_parent / "orders.sqlite3"
        fresh = OrderIntentLedger(
            fresh_path,
            clock=self.clock,
            run_id="fresh-schema-18",
        )
        self.set_capacity(ledger=fresh)

        trigger_names = (
            "validate_capacity_decision_policy_insert",
            "validate_reservation_cap_policy_insert",
            "validate_reservation_cap_policy_update",
        )

        def trigger_sql(path):
            with sqlite3.connect(path) as conn:
                return {
                    name: " ".join(
                        conn.execute(
                            """
                            SELECT sql FROM sqlite_master
                            WHERE type = 'trigger' AND name = ?
                            """,
                            (name,),
                        ).fetchone()[0].split()
                    )
                    for name in trigger_names
                }

        self.assertEqual(
            trigger_sql(self.path), trigger_sql(fresh_path)
        )

        invalid_statements = (
            """
            UPDATE reservation_caps
            SET daily_risk_budget = '-1'
            """,
            """
            UPDATE reservation_caps
            SET risk_policy_sha256 = 'short'
            """,
            """
            UPDATE reservation_caps
            SET policy_outcome = 'MAYBE'
            """,
            """
            UPDATE reservation_caps
            SET daily_window_end = daily_window_start
            """,
            """
            INSERT INTO capacity_decisions (
                capacity_decision_sha256, evidence_sha256,
                account_id, environment, broker_buying_power,
                risk_budget, daily_risk_budget,
                daily_authorized_risk, external_position_risk,
                external_order_risk, represented_managed_risk,
                daily_window_start, daily_window_end,
                risk_policy_version, risk_policy_sha256,
                policy_inputs_json, policy_inputs_sha256,
                policy_outcome, policy_reason_code, cap_amount,
                observed_at, capacity_snapshot_sha256, decided_at
            )
            SELECT ?, evidence_sha256, account_id, environment,
                   broker_buying_power, risk_budget, '-1',
                   daily_authorized_risk, external_position_risk,
                   external_order_risk, represented_managed_risk,
                   daily_window_start, daily_window_end,
                   risk_policy_version, risk_policy_sha256,
                   policy_inputs_json, policy_inputs_sha256,
                   policy_outcome, policy_reason_code, cap_amount,
                   observed_at, capacity_snapshot_sha256, decided_at
            FROM capacity_decisions
            ORDER BY decided_at DESC
            LIMIT 1
            """,
        )
        for path in (self.path, fresh_path):
            for index, statement in enumerate(invalid_statements):
                with self.subTest(path=path.name, statement=index):
                    with sqlite3.connect(path) as conn:
                        conn.execute("PRAGMA foreign_keys = ON")
                        parameters = (
                            ("f" * 64,)
                            if "INSERT INTO capacity_decisions"
                            in statement
                            else ()
                        )
                        with self.assertRaises(sqlite3.DatabaseError):
                            conn.execute(statement, parameters)

    def test_schema_18_v1_pristine_reservation_is_released_safely(
        self,
    ):
        record = self.opening(key="schema-18-policy-v1-pristine")
        policy_v1_sha256 = self.downgrade_to_schema_18_policy_v1()

        migrated = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-19-policy-v2-upgrade",
        )

        self.assertEqual(
            migrated.get_intent(record.intent_id).state, "FAILED"
        )
        reservation = migrated.get_margin_reservation(record.intent_id)
        self.assertEqual(reservation.state, "RELEASED")
        self.assertEqual(
            reservation.released_reason_code,
            "SCHEMA_19_UNCLAIMED_POLICY_V1_RESERVATION_RELEASED",
        )
        self.assertEqual(
            migrated.active_reserved_margin(
                "1000000001", "production"
            ),
            Decimal("0"),
        )
        with migrated._connection() as conn:
            replayed, _manifest, _result = (
                migrated._verified_capacity_decision_row(
                    conn, policy_v1_sha256
                )
            )
        self.assertEqual(
            replayed["risk_policy_version"],
            "OPENING_MAX_LOSS_V1",
        )

    def test_schema_18_v1_submission_trace_is_fenced_and_cannot_authorize(
        self,
    ):
        record = self.opening(key="schema-18-policy-v1-traced")
        current_decision = self.capacity_decision_for(record)
        self.ledger.claim_submission(
            record.intent_id, "policy-v1-worker", lease_seconds=5
        )
        self.clock.advance(6)
        self.assertEqual(
            self.ledger.reconciliation_blockers(
                "1000000001", "production"
            ),
            (),
        )
        policy_v1_sha256 = self.downgrade_to_schema_18_policy_v1()

        migrated = OrderIntentLedger(
            self.path,
            clock=self.clock,
            run_id="schema-19-policy-v1-trace-upgrade",
        )

        self.assertEqual(
            migrated.get_intent(record.intent_id).state,
            "SUBMISSION_UNKNOWN",
        )
        self.assertEqual(
            migrated.get_margin_reservation(record.intent_id).state,
            "ACTIVE",
        )
        with self.assertRaises(OrderIntentReconciliationRequired):
            migrated.claim_submission(
                record.intent_id,
                "new-worker",
                lease_seconds=30,
            )

        fresh = migrated.create_intent(
            make_intent(
                key="schema-18-policy-v1-new-reserve",
                decision="schema-18-policy-v1-new-reserve",
            )
        ).intent
        with self.assertRaises(OrderIntentReservationError):
            migrated.reserve_margin(
                fresh.intent_id,
                self.risk(
                    fresh,
                    ledger=migrated,
                    decision=current_decision,
                    capacity_decision_sha256=policy_v1_sha256,
                ),
            )

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
        query_wire = wire_order_payload(raw_payload())
        query_canonical = canonical_order_payload(
            json.loads(query_wire)
        )
        query_legacy = OrderIntent(
            account_id="1000000099",
            environment="production",
            strategy_id="legacy-open",
            decision_id="legacy-query-decision",
            idempotency_scope="legacy",
            idempotency_key="legacy-query",
            intent_kind="OPENING",
            wire_payload=query_wire,
            canonical_payload=query_canonical,
            payload_hash=hashlib.sha256(
                b"etrade-order-payload.v2\0"
                + query_canonical.encode("utf-8")
            ).hexdigest(),
        )
        query_client_id = stable_client_order_id(query_legacy)
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
                    'legacy-query-intent', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'SUBMITTED', '2000000099', 1, 0, NULL, NULL,
                    NULL, NULL, NULL, 'legacy-run', ?, ?
                )
                """,
                (
                    query_legacy.account_id,
                    query_legacy.environment,
                    query_legacy.strategy_id,
                    query_legacy.decision_id,
                    query_legacy.idempotency_scope,
                    query_legacy.idempotency_key,
                    query_legacy.intent_kind,
                    query_legacy.wire_payload,
                    query_legacy.canonical_payload,
                    query_legacy.payload_hash,
                    query_client_id,
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
            legacy_reservation = conn.execute(
                """
                SELECT amount, max_loss_amount, risk_decision_id, state,
                       capacity_decision_sha256, quote_digest,
                       portfolio_snapshot_digest
                FROM margin_reservations
                WHERE intent_id = 'legacy-query-intent'
                """
            ).fetchone()
            self.assertEqual(
                legacy_reservation[:5],
                (
                    "375",
                    "375",
                    "legacy-opening-migration",
                    "ACTIVE",
                    None,
                ),
            )
            self.assertEqual(len(legacy_reservation[5]), 64)
            self.assertEqual(len(legacy_reservation[6]), 64)
            legacy_creation = conn.execute(
                """
                SELECT from_state, to_state, actor, reason_code
                FROM order_events
                WHERE intent_id = 'legacy-query-intent'
                  AND event_type = 'RESERVATION_CREATED'
                """
            ).fetchone()
            self.assertEqual(
                legacy_creation,
                (
                    "SUBMITTED",
                    "SUBMITTED",
                    "schema-migration",
                    "RESERVATION_CREATED",
                ),
            )
        self.assertEqual(
            migrated.active_reserved_margin(
                query_legacy.account_id, query_legacy.environment
            ),
            Decimal("375"),
        )
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
        query_record = migrated.get_intent("legacy-query-intent")
        legacy_query = self.query_evidence(
            query_record,
            ledger=migrated,
            broker_order_id="2000000099",
        )
        reconciled = migrated.mark_reconciled(
            query_record.intent_id, legacy_query
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
        self.set_capacity(ledger=migrated)
        migrated.reserve_margin(
            record.intent_id, self.risk(record, ledger=migrated)
        )
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
