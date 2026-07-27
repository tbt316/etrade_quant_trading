"""Durable, broker-agnostic order-intent boundary.

This module owns local safety state only.  It deliberately has no E*TRADE
imports and performs no network I/O.  A future gateway must call
``begin_submission`` immediately before a broker POST and write either a
definitive acknowledgement or a broker-backed reconciliation result afterwards.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping
from urllib.parse import quote


SCHEMA_VERSION = 12
_MIGRATABLE_SCHEMA_VERSIONS = frozenset({8, 9, 10, 11})
_BUSY_TIMEOUT_MS = 5_000
_EVIDENCE_MAX_AGE_SECONDS = 300
_DECIMAL_PRECISION = 50
_PAYLOAD_HASH_DOMAIN = b"etrade-order-payload.v2\0"
_CLIENT_ID_DOMAIN = b"etrade-client-order-id.v2\0"
_MAX_TRANSPORT_REQUEST_BYTES = 64 * 1024
_PREVIEW_RECEIPT_MAX_AGE_SECONDS = 180
_TRANSPORT_OPERATIONS = frozenset(
    {"SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"}
)
_BROKER_READ_KINDS = frozenset(
    {
        "ACCOUNT_LIST",
        "BALANCE",
        "PORTFOLIO_PAGE",
        "OPEN_ORDERS_PAGE",
        "ORDER_DETAIL",
    }
)
_BROKER_READ_COMPLETENESS = frozenset(
    {"COMPLETE", "HAS_NEXT", "INELIGIBLE"}
)
_BROKER_READ_EVIDENCE_KINDS = frozenset({"CAPACITY", "ORDER_QUERY"})
_BROKER_READ_EVIDENCE_COMPLETENESS = frozenset(
    {"COMPLETE", "INCOMPLETE", "UNSTABLE"}
)
_ETRADE_ORIGINS = {
    "sandbox": "https://apisb.etrade.com",
    "production": "https://api.etrade.com",
}
_MAX_BROKER_READ_BYTES = 2 * 1024 * 1024
_MAX_BROKER_READ_SPAN_SECONDS = 60
_ACTIVE_ORDER_READ_LANES = (
    "OPEN",
    "CANCEL_REQUESTED",
    "INDIVIDUAL_FILLS",
)
_READ_REQUEST_HASH_DOMAIN = b"etrade-read-request.v1\0"
_READ_PARSED_HASH_DOMAIN = b"etrade-read-parsed.v1\0"
_READ_RECEIPT_HASH_DOMAIN = b"etrade-read-receipt.v1\0"
_READ_MANIFEST_HASH_DOMAIN = b"etrade-read-manifest.v1\0"
_CAPACITY_DECISION_HASH_DOMAIN = b"etrade-capacity-decision.v1\0"
_RESERVATION_ABSORPTION_HASH_DOMAIN = (
    b"etrade-reservation-absorption.v1\0"
)
_LOT_PROOF_HASH_DOMAIN = b"etrade-reservation-lot-proof.v1\0"
_INTENT_KINDS = frozenset({"OPENING", "CLOSING"})
_ENVIRONMENTS = frozenset({"sandbox", "production"})
_TERMINAL_STATES = frozenset({"FILLED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED"})
_BROKER_LIVE_STATES = frozenset({"CLAIMED", "SUBMISSION_UNKNOWN", "SUBMITTED"})
_BROKER_TERMINAL_STATUS = {
    "FILLED": "FILLED",
    "CANCELLED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
}
_REASON_CODES = frozenset(
    {
        "INTENT_CREATED",
        "SUBMISSION_CLAIMED",
        "LEASE_RENEWED",
        "LEASE_EXPIRED_IN_DOUBT",
        "POST_STARTED",
        "POST_ACKNOWLEDGED",
        "POST_TIMEOUT",
        "POST_TRANSPORT_ERROR",
        "POST_RESPONSE_INVALID",
        "PRE_POST_ABORTED",
        "BROKER_OPEN_RECONCILED",
        "BROKER_FILLED",
        "BROKER_CANCELLED",
        "BROKER_REJECTED",
        "BROKER_EXPIRED",
        "RESERVATION_CREATED",
        "RESERVATION_RELEASED",
        "FILLED_ABSORBED",
        "AMENDMENT_LEASE_ACQUIRED",
        "AMENDMENT_LEASE_RELEASED",
    }
)
_POST_UNKNOWN_REASONS = frozenset({"POST_TIMEOUT", "POST_TRANSPORT_ERROR", "POST_RESPONSE_INVALID"})
_PRE_POST_FAILURE_REASONS = frozenset({"PRE_POST_ABORTED"})
_TOP_LEVEL_ORDER_FIELDS = frozenset(
    {
        "symbol",
        "quantity",
        "securityType",
        "priceType",
        "orderTerm",
        "limitPrice",
        "orderAction",
        "callPut",
        "expiryYear",
        "expiryMonth",
        "expiryDay",
        "strikePrice",
        "spreadType",
        "legs",
    }
)
_NORMALIZED_TRANSPORT_FIELDS = frozenset({"client_order_id", "orderType", "preview_id", "required_margin"})
_LEG_FIELDS = frozenset(
    {
        "symbol",
        "callPut",
        "expiryYear",
        "expiryMonth",
        "expiryDay",
        "strikePrice",
        "orderAction",
        "quantity",
    }
)

_REQUIRED_TRIGGER_DEFINITIONS = {
    "prevent_order_event_update": """
        CREATE TRIGGER prevent_order_event_update
        BEFORE UPDATE ON order_events
        BEGIN
            SELECT RAISE(ABORT, 'order events are append-only');
        END
    """,
    "prevent_order_event_delete": """
        CREATE TRIGGER prevent_order_event_delete
        BEFORE DELETE ON order_events
        BEGIN
            SELECT RAISE(ABORT, 'order events are append-only');
        END
    """,
    "prevent_amendment_history_update": """
        CREATE TRIGGER prevent_amendment_history_update
        BEFORE UPDATE ON amendment_history
        BEGIN
            SELECT RAISE(ABORT, 'amendment history is immutable');
        END
    """,
    "prevent_amendment_history_delete": """
        CREATE TRIGGER prevent_amendment_history_delete
        BEFORE DELETE ON amendment_history
        BEGIN
            SELECT RAISE(ABORT, 'amendment history is immutable');
        END
    """,
    "prevent_outbound_authorization_update": """
        CREATE TRIGGER prevent_outbound_authorization_update
        BEFORE UPDATE ON outbound_authorizations
        BEGIN
            SELECT RAISE(ABORT, 'outbound authorizations are immutable');
        END
    """,
    "prevent_outbound_authorization_delete": """
        CREATE TRIGGER prevent_outbound_authorization_delete
        BEFORE DELETE ON outbound_authorizations
        BEGIN
            SELECT RAISE(ABORT, 'outbound authorizations are immutable');
        END
    """,
    "prevent_transport_send_attempt_update": """
        CREATE TRIGGER prevent_transport_send_attempt_update
        BEFORE UPDATE ON transport_send_attempts
        BEGIN
            SELECT RAISE(ABORT, 'transport send attempts are immutable');
        END
    """,
    "prevent_transport_send_attempt_delete": """
        CREATE TRIGGER prevent_transport_send_attempt_delete
        BEFORE DELETE ON transport_send_attempts
        BEGIN
            SELECT RAISE(ABORT, 'transport send attempts are immutable');
        END
    """,
    "prevent_broker_preview_receipt_update": """
        CREATE TRIGGER prevent_broker_preview_receipt_update
        BEFORE UPDATE ON broker_preview_receipts
        BEGIN
            SELECT RAISE(ABORT, 'broker preview receipts are immutable');
        END
    """,
    "prevent_broker_preview_receipt_delete": """
        CREATE TRIGGER prevent_broker_preview_receipt_delete
        BEFORE DELETE ON broker_preview_receipts
        BEGIN
            SELECT RAISE(ABORT, 'broker preview receipts are immutable');
        END
    """,
    "prevent_transport_response_receipt_update": """
        CREATE TRIGGER prevent_transport_response_receipt_update
        BEFORE UPDATE ON transport_response_receipts
        BEGIN
            SELECT RAISE(ABORT, 'transport response receipts are immutable');
        END
    """,
    "prevent_transport_response_receipt_delete": """
        CREATE TRIGGER prevent_transport_response_receipt_delete
        BEFORE DELETE ON transport_response_receipts
        BEGIN
            SELECT RAISE(ABORT, 'transport response receipts are immutable');
        END
    """,
    "prevent_broker_order_history_update": """
        CREATE TRIGGER prevent_broker_order_history_update
        BEFORE UPDATE ON broker_order_history
        BEGIN
            SELECT RAISE(ABORT, 'broker order history is immutable');
        END
    """,
    "prevent_broker_order_history_delete": """
        CREATE TRIGGER prevent_broker_order_history_delete
        BEFORE DELETE ON broker_order_history
        BEGIN
            SELECT RAISE(ABORT, 'broker order history is immutable');
        END
    """,
    "prevent_intent_identity_mutation": """
        CREATE TRIGGER prevent_intent_identity_mutation
        BEFORE UPDATE ON order_intents
        WHEN OLD.account_id != NEW.account_id
          OR OLD.environment != NEW.environment
          OR OLD.strategy_id != NEW.strategy_id
          OR OLD.decision_id != NEW.decision_id
          OR OLD.idempotency_scope != NEW.idempotency_scope
          OR OLD.idempotency_key != NEW.idempotency_key
          OR OLD.intent_kind != NEW.intent_kind
          OR OLD.wire_payload != NEW.wire_payload
          OR OLD.canonical_payload != NEW.canonical_payload
          OR OLD.payload_hash != NEW.payload_hash
          OR OLD.client_order_id != NEW.client_order_id
        BEGIN
            SELECT RAISE(ABORT, 'order intent identity is immutable');
        END
    """,
    "prevent_terminal_rewrite": """
        CREATE TRIGGER prevent_terminal_rewrite
        BEFORE UPDATE ON order_intents
        WHEN OLD.state IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FAILED')
          AND NEW.state != OLD.state
        BEGIN
            SELECT RAISE(ABORT, 'terminal order intent cannot transition');
        END
    """,
    "prevent_broker_read_receipt_update": """
        CREATE TRIGGER prevent_broker_read_receipt_update
        BEFORE UPDATE ON broker_read_receipts
        BEGIN
            SELECT RAISE(ABORT, 'broker read receipts are append-only');
        END
    """,
    "prevent_broker_read_receipt_delete": """
        CREATE TRIGGER prevent_broker_read_receipt_delete
        BEFORE DELETE ON broker_read_receipts
        BEGIN
            SELECT RAISE(ABORT, 'broker read receipts are append-only');
        END
    """,
    "prevent_broker_read_manifest_update": """
        CREATE TRIGGER prevent_broker_read_manifest_update
        BEFORE UPDATE ON broker_read_manifests
        BEGIN
            SELECT RAISE(ABORT, 'broker read manifests are append-only');
        END
    """,
    "prevent_broker_read_manifest_delete": """
        CREATE TRIGGER prevent_broker_read_manifest_delete
        BEFORE DELETE ON broker_read_manifests
        BEGIN
            SELECT RAISE(ABORT, 'broker read manifests are append-only');
        END
    """,
    "prevent_broker_read_member_update": """
        CREATE TRIGGER prevent_broker_read_member_update
        BEFORE UPDATE ON broker_read_manifest_members
        BEGIN
            SELECT RAISE(ABORT, 'broker read manifest members are append-only');
        END
    """,
    "prevent_broker_read_member_delete": """
        CREATE TRIGGER prevent_broker_read_member_delete
        BEFORE DELETE ON broker_read_manifest_members
        BEGIN
            SELECT RAISE(ABORT, 'broker read manifest members are append-only');
        END
    """,
    "prevent_capacity_decision_update": """
        CREATE TRIGGER prevent_capacity_decision_update
        BEFORE UPDATE ON capacity_decisions
        BEGIN
            SELECT RAISE(ABORT, 'capacity decisions are append-only');
        END
    """,
    "prevent_capacity_decision_delete": """
        CREATE TRIGGER prevent_capacity_decision_delete
        BEFORE DELETE ON capacity_decisions
        BEGIN
            SELECT RAISE(ABORT, 'capacity decisions are append-only');
        END
    """,
    "prevent_reservation_absorption_update": """
        CREATE TRIGGER prevent_reservation_absorption_update
        BEFORE UPDATE ON reservation_absorptions
        BEGIN
            SELECT RAISE(ABORT, 'reservation absorptions are append-only');
        END
    """,
    "prevent_reservation_absorption_delete": """
        CREATE TRIGGER prevent_reservation_absorption_delete
        BEFORE DELETE ON reservation_absorptions
        BEGIN
            SELECT RAISE(ABORT, 'reservation absorptions are append-only');
        END
    """,
    "prevent_margin_reservation_delete": """
        CREATE TRIGGER prevent_margin_reservation_delete
        BEFORE DELETE ON margin_reservations
        BEGIN
            SELECT RAISE(ABORT, 'margin reservations are durable');
        END
    """,
    "prevent_margin_reservation_identity_update": """
        CREATE TRIGGER prevent_margin_reservation_identity_update
        BEFORE UPDATE ON margin_reservations
        WHEN OLD.intent_id IS NOT NEW.intent_id
           OR OLD.account_id IS NOT NEW.account_id
           OR OLD.environment IS NOT NEW.environment
           OR OLD.amount IS NOT NEW.amount
           OR OLD.risk_decision_id IS NOT NEW.risk_decision_id
           OR OLD.max_loss_amount IS NOT NEW.max_loss_amount
           OR OLD.quote_observed_at IS NOT NEW.quote_observed_at
           OR OLD.quote_digest IS NOT NEW.quote_digest
           OR OLD.portfolio_observed_at IS NOT NEW.portfolio_observed_at
           OR OLD.portfolio_snapshot_digest IS NOT NEW.portfolio_snapshot_digest
           OR OLD.capacity_decision_sha256 IS NOT NEW.capacity_decision_sha256
           OR OLD.created_at IS NOT NEW.created_at
        BEGIN
            SELECT RAISE(ABORT, 'margin reservation identity is immutable');
        END
    """,
    "prevent_margin_reservation_invalid_transition": """
        CREATE TRIGGER prevent_margin_reservation_invalid_transition
        BEFORE UPDATE ON margin_reservations
        WHEN NOT (
            OLD.state = NEW.state
            OR (
                OLD.state = 'ACTIVE'
                AND NEW.state IN ('FILLED_PENDING_ABSORPTION','RELEASED')
            )
            OR (
                OLD.state = 'FILLED_PENDING_ABSORPTION'
                AND NEW.state = 'RELEASED'
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid margin reservation transition');
        END
    """,
    "prevent_margin_reservation_release_rewrite": """
        CREATE TRIGGER prevent_margin_reservation_release_rewrite
        BEFORE UPDATE ON margin_reservations
        WHEN OLD.state = NEW.state
         AND (
            OLD.released_reason_code IS NOT NEW.released_reason_code
            OR OLD.released_at IS NOT NEW.released_at
         )
        BEGIN
            SELECT RAISE(ABORT, 'margin reservation release metadata is immutable');
        END
    """,
    "validate_margin_reservation_insert": """
        CREATE TRIGGER validate_margin_reservation_insert
        BEFORE INSERT ON margin_reservations
        WHEN NOT (
            EXISTS (
                SELECT 1
                FROM order_intents AS intent
                JOIN reservation_caps AS cap
                  ON cap.account_id = intent.account_id
                 AND cap.environment = intent.environment
                JOIN capacity_decisions AS decision
                  ON decision.capacity_decision_sha256 =
                        NEW.capacity_decision_sha256
                WHERE intent.intent_id = NEW.intent_id
                  AND intent.intent_kind = 'OPENING'
                  AND intent.state = 'INTENT'
                  AND intent.broker_order_id IS NULL
                  AND NEW.account_id = intent.account_id
                  AND NEW.environment = intent.environment
                  AND NEW.risk_decision_id = intent.decision_id
                  AND NEW.state = 'ACTIVE'
                  AND NEW.released_reason_code IS NULL
                  AND NEW.released_at IS NULL
                  AND NEW.created_at >= intent.created_at
                  AND cap.capacity_decision_sha256 =
                        NEW.capacity_decision_sha256
                  AND cap.cap_amount = decision.cap_amount
                  AND cap.broker_buying_power =
                        decision.broker_buying_power
                  AND cap.risk_budget = decision.risk_budget
                  AND cap.observed_at = decision.observed_at
                  AND cap.portfolio_snapshot_digest =
                        decision.capacity_snapshot_sha256
                  AND decision.account_id = NEW.account_id
                  AND decision.environment = NEW.environment
                  AND NEW.portfolio_observed_at =
                        decision.observed_at
                  AND NEW.portfolio_snapshot_digest =
                        decision.capacity_snapshot_sha256
                  AND etrade_decimal_gte(
                        NEW.amount,
                        etrade_opening_exposure_floor(
                            intent.wire_payload
                        )
                  ) = 1
                  AND etrade_decimal_gte(
                        NEW.max_loss_amount,
                        etrade_opening_exposure_floor(
                            intent.wire_payload
                        )
                  ) = 1
                  AND etrade_decimal_gte(
                        decision.cap_amount,
                        NEW.amount
                  ) = 1
            )
            OR EXISTS (
                SELECT 1
                FROM order_intents AS intent
                JOIN ledger_metadata AS metadata
                  ON metadata.singleton = 1
                WHERE metadata.schema_version IN (8, 9)
                  AND intent.intent_id = NEW.intent_id
                  AND intent.intent_kind = 'OPENING'
                  AND intent.state != 'FAILED'
                  AND NEW.account_id = intent.account_id
                  AND NEW.environment = intent.environment
                  AND NEW.risk_decision_id =
                        'legacy-opening-migration'
                  AND NEW.amount =
                        etrade_opening_exposure_floor(
                            intent.wire_payload
                        )
                  AND NEW.max_loss_amount = NEW.amount
                  AND NEW.quote_observed_at = intent.created_at
                  AND NEW.portfolio_observed_at =
                        intent.created_at
                  AND NEW.created_at = intent.created_at
                  AND NEW.quote_digest =
                        etrade_legacy_reservation_digest(
                            'quote', metadata.schema_version,
                            intent.intent_id, intent.payload_hash
                        )
                  AND NEW.portfolio_snapshot_digest =
                        etrade_legacy_reservation_digest(
                            'portfolio', metadata.schema_version,
                            intent.intent_id, intent.payload_hash
                        )
                  AND NEW.capacity_decision_sha256 IS NULL
                  AND NEW.released_reason_code IS NULL
                  AND NEW.released_at IS NULL
                  AND (
                        (
                            intent.state IN (
                                'FILLED','CANCELLED',
                                'REJECTED','EXPIRED'
                            )
                            AND NEW.state =
                                'FILLED_PENDING_ABSORPTION'
                        )
                        OR
                        (
                            intent.state IN (
                                'INTENT','CLAIMED',
                                'SUBMISSION_UNKNOWN','SUBMITTED'
                            )
                            AND NEW.state = 'ACTIVE'
                        )
                  )
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'margin reservation insert lacks exact risk provenance');
        END
    """,
    "validate_reservation_created_event_insert": """
        CREATE TRIGGER validate_reservation_created_event_insert
        BEFORE INSERT ON order_events
        WHEN NEW.event_type = 'RESERVATION_CREATED'
         AND NOT (
            EXISTS (
                SELECT 1
                FROM margin_reservations AS reservation
                JOIN order_intents AS intent
                  ON intent.intent_id = reservation.intent_id
                WHERE reservation.intent_id = NEW.intent_id
                  AND reservation.account_id = NEW.account_id
                  AND reservation.environment = NEW.environment
                  AND reservation.created_at = NEW.created_at
                  AND reservation.capacity_decision_sha256
                        IS NOT NULL
                  AND intent.client_order_id =
                        NEW.client_order_id
                  AND intent.state = 'INTENT'
                  AND NEW.from_state = 'INTENT'
                  AND NEW.to_state = 'INTENT'
                  AND NEW.actor = 'system'
                  AND NEW.reason_code =
                        'RESERVATION_CREATED'
                  AND NEW.broker_status IS NULL
                  AND NEW.broker_order_id IS NULL
                  AND NEW.observed_at IS NULL
                  AND NEW.evidence_operation IS NULL
                  AND NEW.http_status IS NULL
                  AND NEW.raw_response_digest IS NULL
                  AND NEW.broker_read_evidence_sha256 IS NULL
            )
            OR EXISTS (
                SELECT 1
                FROM margin_reservations AS reservation
                JOIN order_intents AS intent
                  ON intent.intent_id = reservation.intent_id
                JOIN ledger_metadata AS metadata
                  ON metadata.singleton = 1
                WHERE metadata.schema_version IN (8, 9)
                  AND reservation.intent_id = NEW.intent_id
                  AND reservation.account_id = NEW.account_id
                  AND reservation.environment = NEW.environment
                  AND reservation.created_at = NEW.created_at
                  AND reservation.risk_decision_id =
                        'legacy-opening-migration'
                  AND reservation.capacity_decision_sha256
                        IS NULL
                  AND intent.client_order_id =
                        NEW.client_order_id
                  AND NEW.from_state = intent.state
                  AND NEW.to_state = intent.state
                  AND NEW.actor = 'schema-migration'
                  AND NEW.reason_code =
                        'RESERVATION_CREATED'
                  AND NEW.broker_status IS NULL
                  AND NEW.broker_order_id IS NULL
                  AND NEW.observed_at IS NULL
                  AND NEW.evidence_operation IS NULL
                  AND NEW.http_status IS NULL
                  AND NEW.raw_response_digest IS NULL
                  AND NEW.broker_read_evidence_sha256 IS NULL
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'reservation creation event lacks exact risk provenance');
        END
    """,
    "validate_margin_reservation_pre_post_release": """
        CREATE TRIGGER validate_margin_reservation_pre_post_release
        BEFORE UPDATE ON margin_reservations
        WHEN OLD.state = 'ACTIVE'
         AND NEW.state = 'RELEASED'
         AND NOT EXISTS (
            SELECT 1
            FROM order_intents AS intent
            WHERE intent.intent_id = OLD.intent_id
              AND intent.intent_kind = 'OPENING'
              AND intent.state = 'FAILED'
              AND intent.broker_order_id IS NULL
              AND intent.updated_at = NEW.released_at
              AND NEW.released_reason_code =
                    'PRE_POST_ABORTED'
              AND EXISTS (
                    SELECT 1
                    FROM order_events AS claim
                    WHERE claim.intent_id = intent.intent_id
                      AND claim.event_type =
                            'SUBMISSION_CLAIMED'
                      AND claim.from_state = 'INTENT'
                      AND claim.to_state = 'CLAIMED'
                      AND claim.reason_code =
                            'SUBMISSION_CLAIMED'
                      AND claim.created_at <= NEW.released_at
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM order_events AS post
                    WHERE post.intent_id = intent.intent_id
                      AND post.event_type = 'POST_STARTED'
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM transport_send_attempts AS attempt
                    WHERE attempt.intent_id = intent.intent_id
                      AND attempt.transport_operation =
                            'SUBMIT_PLACE'
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM transport_response_receipts AS response
                    WHERE response.intent_id = intent.intent_id
                      AND response.transport_operation =
                            'SUBMIT_PLACE'
              )
         )
        BEGIN
            SELECT RAISE(ABORT, 'active reservation release lacks pre-post failure proof');
        END
    """,
    "validate_reservation_absorption_insert": """
        CREATE TRIGGER validate_reservation_absorption_insert
        BEFORE INSERT ON reservation_absorptions
        WHEN NOT EXISTS (
            SELECT 1
            FROM order_intents AS intent
            JOIN margin_reservations AS reservation
              ON reservation.intent_id = intent.intent_id
            JOIN broker_read_manifests AS terminal_manifest
              ON terminal_manifest.evidence_sha256 =
                    NEW.terminal_order_evidence_sha256
            WHERE intent.intent_id = NEW.intent_id
              AND intent.intent_kind = 'OPENING'
              AND intent.account_id = NEW.account_id
              AND intent.environment = NEW.environment
              AND intent.broker_order_id = NEW.broker_order_id
              AND intent.state = NEW.terminal_state
              AND reservation.account_id = NEW.account_id
              AND reservation.environment = NEW.environment
              AND reservation.state = 'FILLED_PENDING_ABSORPTION'
              AND reservation.capacity_decision_sha256
                    IS NEW.baseline_capacity_decision_sha256
              AND terminal_manifest.evidence_kind = 'ORDER_QUERY'
              AND terminal_manifest.completeness = 'COMPLETE'
              AND terminal_manifest.account_id = NEW.account_id
              AND terminal_manifest.environment = NEW.environment
              AND terminal_manifest.target_broker_order_id =
                    NEW.broker_order_id
              AND (
                    (
                        NEW.classification = 'ZERO_FILL'
                        AND intent.state IN (
                            'CANCELLED','REJECTED','EXPIRED'
                        )
                        AND NEW.absorbed_margin_amount = '0'
                        AND NEW.post_capacity_decision_sha256 IS NULL
                        AND NEW.post_capacity_evidence_sha256 IS NULL
                    )
                    OR
                    (
                        NEW.classification = 'FULL_FILL'
                        AND intent.state = 'FILLED'
                        AND NEW.absorbed_margin_amount = reservation.amount
                        AND EXISTS (
                            SELECT 1
                            FROM capacity_decisions AS post_decision
                            JOIN broker_read_manifests AS post_manifest
                              ON post_manifest.evidence_sha256 =
                                    NEW.post_capacity_evidence_sha256
                            WHERE post_decision.capacity_decision_sha256 =
                                    NEW.post_capacity_decision_sha256
                              AND post_decision.evidence_sha256 =
                                    NEW.post_capacity_evidence_sha256
                              AND post_decision.account_id = NEW.account_id
                              AND post_decision.environment = NEW.environment
                              AND post_manifest.evidence_kind = 'CAPACITY'
                              AND post_manifest.completeness = 'COMPLETE'
                              AND post_manifest.account_id = NEW.account_id
                              AND post_manifest.environment = NEW.environment
                              AND post_manifest.target_broker_order_id IS NULL
                        )
                    )
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'reservation absorption is not cross-bound to durable risk');
        END
    """,
    "require_terminal_absorption_receipt": """
        CREATE TRIGGER require_terminal_absorption_receipt
        BEFORE UPDATE ON margin_reservations
        WHEN OLD.state = 'FILLED_PENDING_ABSORPTION'
         AND NEW.state = 'RELEASED'
         AND NOT EXISTS (
            SELECT 1
            FROM reservation_absorptions
            WHERE intent_id = OLD.intent_id
              AND (
                    (
                        classification = 'ZERO_FILL'
                        AND NEW.released_reason_code =
                            'ZERO_FILL_CONFIRMED'
                    )
                    OR
                    (
                        classification = 'FULL_FILL'
                        AND NEW.released_reason_code =
                            'FULL_FILL_POSITION_ABSORBED'
                    )
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'terminal reservation release lacks absorption receipt');
        END
    """,
}


class OrderIntentLedgerError(RuntimeError):
    """Base class for order-intent failures."""


class OrderIntentValidationError(OrderIntentLedgerError):
    """Input is not safe for durable live-order state."""


class OrderIntentIntegrityError(OrderIntentLedgerError):
    """Persisted immutable identity has been tampered with or corrupted."""


class OrderIntentTransitionError(OrderIntentLedgerError):
    """The requested state transition is not allowed."""


class OrderIntentLeaseConflict(OrderIntentLedgerError):
    """Another worker owns the current submission or amendment lease."""


class OrderIntentReconciliationRequired(OrderIntentLedgerError):
    """The account/environment has unfinished broker work to reconcile."""


class OrderIntentBrokerTermsMismatch(
    OrderIntentIntegrityError, OrderIntentReconciliationRequired
):
    """Known broker order terms do not yet match the durable operation."""


class OrderIntentReservationError(OrderIntentLedgerError):
    """An opening intent lacks a valid margin reservation or capacity."""


@dataclass(frozen=True)
class OrderIntent:
    """Immutable identity envelope for one operator/strategy decision.

    ``canonical_payload`` is the durable, allowlisted economic payload.  It
    excludes all generated transport values; ``prepare_submission_payload`` is
    the only supported way to produce authorized submission bytes with the
    stored client order id.
    """

    account_id: str
    environment: str
    strategy_id: str
    decision_id: str
    idempotency_scope: str
    idempotency_key: str
    intent_kind: Literal["OPENING", "CLOSING"]
    wire_payload: str
    canonical_payload: str
    payload_hash: str

    @classmethod
    def build(
        cls,
        *,
        account_id: str,
        environment: str,
        strategy_id: str,
        decision_id: str,
        idempotency_scope: str,
        idempotency_key: str,
        intent_kind: Literal["OPENING", "CLOSING"],
        order_payload: Mapping[str, Any],
    ) -> "OrderIntent":
        _validate_identity("account_id", account_id)
        _validate_environment(environment)
        _validate_identity("strategy_id", strategy_id)
        _validate_identity("decision_id", decision_id)
        _validate_identity("idempotency_scope", idempotency_scope)
        _validate_identity("idempotency_key", idempotency_key)
        if type(intent_kind) is not str or intent_kind not in _INTENT_KINDS:
            raise OrderIntentValidationError("intent_kind must be OPENING or CLOSING")
        wire_payload = wire_order_payload(normalize_order_payload(order_payload))
        derived_kind = _derive_intent_kind(json.loads(wire_payload))
        if intent_kind != derived_kind:
            raise OrderIntentValidationError("intent_kind must match the exposure derived from order actions")
        if intent_kind == "CLOSING":
            raise OrderIntentValidationError(
                "closing orders require typed position and open-order capacity evidence"
            )
        canonical_payload = canonical_order_payload(json.loads(wire_payload))
        payload_hash = _payload_hash(canonical_payload)
        return cls(
            account_id=account_id,
            environment=environment,
            strategy_id=strategy_id,
            decision_id=decision_id,
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            intent_kind=intent_kind,
            wire_payload=wire_payload,
            canonical_payload=canonical_payload,
            payload_hash=payload_hash,
        )


@dataclass(frozen=True)
class IntentRecord:
    intent_id: str
    envelope: OrderIntent
    client_order_id: str
    state: str
    broker_order_id: str | None
    submission_fence: int
    submission_lease_owner: str | None
    submission_lease_expires_at: datetime | None
    pending_operation: Literal["SUBMIT", "AMEND"] | None
    pending_fence: int | None
    last_reconciled_run: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CreateIntentResult:
    intent: IntentRecord
    created: bool


@dataclass(frozen=True)
class SubmissionLease:
    intent_id: str
    owner: str
    fencing_token: int
    expires_at: datetime


@dataclass(frozen=True)
class AmendmentLease:
    intent_id: str
    broker_order_id: str
    client_order_id: str
    owner: str
    fencing_token: int
    expires_at: datetime


@dataclass(frozen=True)
class OutboundAuthorization:
    """Exact immutable broker-schema bytes approved for one fenced operation.

    The private transport must derive the final E*TRADE request body solely
    from ``payload_bytes`` plus the broker preview id. It must never reconstruct
    economic fields from a mutable caller-owned mapping.
    """

    intent_id: str
    operation: Literal["SUBMIT", "AMEND"]
    owner: str
    fencing_token: int
    client_order_id: str = field(repr=False)
    payload_bytes: bytes = field(repr=False)
    payload_digest: str


@dataclass(frozen=True)
class TransportRequestEvidence:
    """Exact broker request claimed durably before one network attempt."""

    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str
    environment: Literal["sandbox", "production"]
    intent_id: str
    owner: str
    authorization_operation: Literal["SUBMIT", "AMEND"]
    fencing_token: int
    transport_operation: Literal[
        "SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"
    ]
    http_method: Literal["POST", "PUT"]
    route: str = field(repr=False)
    client_order_id: str = field(repr=False)
    target_broker_order_id: str | None = field(repr=False)
    preview_id: str | None = field(repr=False)
    authorization_payload_digest: str
    final_xml_bytes: bytes = field(repr=False)
    final_xml_sha256: str


@dataclass(frozen=True)
class TransportResponseEvidence:
    """Parsed response bound to one exact transport send attempt."""

    disposition: Literal["ACKNOWLEDGED", "UNKNOWN"]
    http_status: int | None
    broker_status: str | None
    broker_order_id: str | None = field(repr=False)
    preview_id: str | None = field(repr=False)
    message_codes: tuple[int, ...]
    message_types: tuple[str, ...]
    message_description_digests: tuple[str, ...]
    raw_response_digest: str | None
    observed_at: datetime
    unknown_reason: str | None


@dataclass(frozen=True)
class TransportResponseReceipt:
    """Immutable typed response available to restart reconciliation."""

    intent_id: str
    authorization_operation: Literal["SUBMIT", "AMEND"]
    fencing_token: int
    transport_operation: Literal[
        "SUBMIT_PREVIEW", "SUBMIT_PLACE", "AMEND_PREVIEW", "AMEND_PLACE"
    ]
    client_order_id: str = field(repr=False)
    target_broker_order_id: str | None = field(repr=False)
    response: TransportResponseEvidence
    recorded_at: datetime


@dataclass(frozen=True, repr=False)
class BrokerReadResponseEvidence:
    """One exact, bounded E*TRADE GET response before semantic assembly."""

    read_kind: Literal[
        "ACCOUNT_LIST",
        "BALANCE",
        "PORTFOLIO_PAGE",
        "OPEN_ORDERS_PAGE",
        "ORDER_DETAIL",
    ]
    account_id: str
    account_id_key: str
    institution_type: str
    environment: Literal["sandbox", "production"]
    origin: str
    route: str
    query_json: str
    authorization_sha256: str
    target_broker_order_id: str | None
    request_started_at: datetime
    response_completed_at: datetime
    http_status: int
    raw_response_bytes: bytes
    parser_schema: str
    parser_code_sha256: str
    parser_config_sha256: str
    canonical_parsed_json: str
    completeness: Literal["COMPLETE", "HAS_NEXT", "INELIGIBLE"]


@dataclass(frozen=True)
class BrokerReadReceiptRef:
    receipt_sha256: str


@dataclass(frozen=True)
class BrokerReadManifestMember:
    role: str
    receipt_sha256: str


@dataclass(frozen=True, repr=False)
class BrokerReadManifestEvidence:
    """A typed result assembled only from durable raw-response receipts."""

    evidence_kind: Literal["CAPACITY", "ORDER_QUERY"]
    account_id: str
    account_id_key: str
    institution_type: str
    environment: Literal["sandbox", "production"]
    origin: str
    target_broker_order_id: str | None
    observed_at: datetime
    completeness: Literal["COMPLETE", "INCOMPLETE", "UNSTABLE"]
    canonical_result_json: str


@dataclass(frozen=True)
class BrokerReadEvidenceRef:
    evidence_sha256: str
    evidence_kind: Literal["CAPACITY", "ORDER_QUERY"]


@dataclass(frozen=True)
class CapacityDecisionReceipt:
    decision_sha256: str
    evidence_sha256: str
    cap_amount: Decimal
    broker_buying_power: Decimal
    risk_budget: Decimal
    observed_at: datetime
    portfolio_snapshot_digest: str


@dataclass(frozen=True)
class TerminalAbsorptionRequirement:
    intent_id: str
    classification: Literal["ZERO_FILL", "FULL_FILL"]
    terminal_state: Literal[
        "FILLED", "CANCELLED", "REJECTED", "EXPIRED"
    ]
    broker_order_id: str
    terminal_order_evidence_sha256: str
    baseline_capacity_decision_sha256: str | None
    ordered_quantity: int
    filled_quantity: int
    post_capacity_required: bool


@dataclass(frozen=True)
class ReservationAbsorptionReceipt:
    absorption_sha256: str
    intent_id: str
    account_id: str
    environment: Literal["sandbox", "production"]
    broker_order_id: str
    terminal_state: Literal[
        "FILLED", "CANCELLED", "REJECTED", "EXPIRED"
    ]
    classification: Literal["ZERO_FILL", "FULL_FILL"]
    terminal_order_evidence_sha256: str
    baseline_capacity_decision_sha256: str | None
    post_capacity_decision_sha256: str | None
    post_capacity_evidence_sha256: str | None
    ordered_quantity: int
    filled_quantity: int
    placed_time_epoch_ms: str
    executed_time_epoch_ms: str | None
    canonical_lot_proof_json: str
    lot_proof_sha256: str
    absorbed_margin_amount: Decimal
    observed_at: datetime
    recorded_at: datetime


@dataclass(frozen=True)
class MarginReservation:
    intent_id: str
    account_id: str
    environment: str
    amount: Decimal
    risk_decision_id: str
    max_loss_amount: Decimal
    quote_observed_at: datetime
    quote_digest: str
    portfolio_observed_at: datetime
    portfolio_snapshot_digest: str
    capacity_decision_sha256: str | None
    state: str
    released_reason_code: str | None
    created_at: datetime
    released_at: datetime | None


@dataclass(frozen=True)
class BrokerEvidence:
    """Typed, persisted broker evidence; callers cannot supply opaque receipts."""

    account_id: str
    environment: str
    client_order_id: str
    broker_order_id: str
    operation: Literal["SUBMIT_ACK", "ORDER_QUERY", "AMEND_ACK", "AMEND_QUERY"]
    outcome: Literal["OPEN", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"]
    observed_at: datetime
    http_status: int
    raw_response_digest: str
    broker_read_evidence_sha256: str | None = None
    order_payload_hashes: tuple[str, ...] = ()

    def validate(self, now: datetime) -> None:
        if type(self) is not BrokerEvidence:
            raise OrderIntentValidationError("broker evidence must use the exact BrokerEvidence type")
        if type(self.operation) is not str or type(self.outcome) is not str:
            raise OrderIntentValidationError("broker evidence operation/outcome must be exact strings")
        if self.operation not in {"SUBMIT_ACK", "ORDER_QUERY", "AMEND_ACK", "AMEND_QUERY"}:
            raise OrderIntentValidationError("unsupported broker evidence operation")
        if self.outcome not in {"OPEN", *_BROKER_TERMINAL_STATUS}:
            raise OrderIntentValidationError("unsupported broker evidence outcome")
        _validate_timestamp(self.observed_at)
        if self.observed_at > now + timedelta(seconds=5) or now - self.observed_at > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS):
            raise OrderIntentValidationError("broker evidence is stale or from the future")
        _validate_identity("account_id", self.account_id)
        _validate_identity("environment", self.environment)
        _validate_identity("client_order_id", self.client_order_id)
        _validate_identity("broker_order_id", self.broker_order_id)
        if type(self.http_status) is not int or self.http_status < 200 or self.http_status > 299:
            raise OrderIntentValidationError("valid broker evidence requires successful 2xx http_status")
        _validate_sha256("raw_response_digest", self.raw_response_digest)
        if self.operation in {"ORDER_QUERY", "AMEND_QUERY"}:
            _validate_sha256(
                "broker_read_evidence_sha256",
                self.broker_read_evidence_sha256,
            )
            if (
                type(self.order_payload_hashes) is not tuple
                or not self.order_payload_hashes
                or len(set(self.order_payload_hashes))
                != len(self.order_payload_hashes)
            ):
                raise OrderIntentValidationError(
                    "query evidence requires unique durable order payload hashes"
                )
            for payload_hash in self.order_payload_hashes:
                _validate_sha256("order_payload_hash", payload_hash)
        elif (
            self.broker_read_evidence_sha256 is not None
            or self.order_payload_hashes
        ):
            raise OrderIntentValidationError(
                "transport acknowledgements cannot claim broker-read provenance"
            )


@dataclass(frozen=True)
class RiskEvidence:
    decision_id: str
    max_loss_amount: Decimal
    collateral_amount: Decimal
    quote_observed_at: datetime
    quote_digest: str
    portfolio_observed_at: datetime
    portfolio_snapshot_digest: str
    capacity_decision_sha256: str

    def validate(self, now: datetime) -> None:
        if type(self) is not RiskEvidence:
            raise OrderIntentValidationError("risk evidence must use the exact RiskEvidence type")
        _validate_identity("decision_id", self.decision_id)
        for name, amount in (("max_loss_amount", self.max_loss_amount), ("collateral_amount", self.collateral_amount)):
            if type(amount) is not Decimal or not amount.is_finite() or amount <= 0:
                raise OrderIntentValidationError(f"{name} must be a positive finite Decimal")
        if self.collateral_amount < self.max_loss_amount:
            raise OrderIntentValidationError("collateral_amount cannot be less than max_loss_amount")
        _validate_timestamp(self.quote_observed_at)
        if self.quote_observed_at > now + timedelta(seconds=5) or now - self.quote_observed_at > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS):
            raise OrderIntentValidationError("risk evidence is stale or from the future")
        _validate_sha256("quote_digest", self.quote_digest)
        _validate_timestamp(self.portfolio_observed_at)
        if self.portfolio_observed_at > now + timedelta(seconds=5) or now - self.portfolio_observed_at > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS):
            raise OrderIntentValidationError("portfolio risk evidence is stale or from the future")
        _validate_sha256("portfolio_snapshot_digest", self.portfolio_snapshot_digest)
        _validate_sha256(
            "capacity_decision_sha256",
            self.capacity_decision_sha256,
        )


@dataclass(frozen=True)
class AccountCapacityEvidence:
    """Broker capacity snapshot whose digest covers positions and capacity fields."""

    account_id: str
    environment: str
    broker_buying_power: Decimal
    risk_budget: Decimal
    observed_at: datetime
    portfolio_snapshot_digest: str
    broker_read_evidence_sha256: str | None = None

    def validate(self, now: datetime) -> None:
        if type(self) is not AccountCapacityEvidence:
            raise OrderIntentValidationError(
                "capacity evidence must use the exact AccountCapacityEvidence type"
            )
        _validate_identity("account_id", self.account_id)
        _validate_environment(self.environment)
        for name, amount in (("broker_buying_power", self.broker_buying_power), ("risk_budget", self.risk_budget)):
            if type(amount) is not Decimal or not amount.is_finite() or amount < 0:
                raise OrderIntentValidationError(f"{name} must be a non-negative finite Decimal")
        _validate_timestamp(self.observed_at)
        if self.observed_at > now + timedelta(seconds=5) or now - self.observed_at > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS):
            raise OrderIntentValidationError("capacity evidence is stale or from the future")
        _validate_sha256("portfolio_snapshot_digest", self.portfolio_snapshot_digest)
        if self.broker_read_evidence_sha256 is not None:
            _validate_sha256(
                "broker_read_evidence_sha256",
                self.broker_read_evidence_sha256,
            )


@dataclass(frozen=True)
class IntentEvent:
    sequence: int
    intent_id: str
    account_id: str
    environment: str
    client_order_id: str
    event_type: str
    from_state: str | None
    to_state: str | None
    actor: str
    reason_code: str
    broker_status: str | None
    broker_order_id: str | None
    observed_at: datetime | None
    evidence_operation: str | None
    http_status: int | None
    raw_response_digest: str | None
    broker_read_evidence_sha256: str | None
    created_at: datetime


def canonical_order_payload(order_payload: Mapping[str, Any]) -> str:
    """Validate and canonicalize an allowlisted E*TRADE order payload."""
    if type(order_payload) is not dict or not order_payload:
        raise OrderIntentValidationError("order payload must be a non-empty mapping")
    _validate_mapping_keys(order_payload, _TOP_LEVEL_ORDER_FIELDS, "order payload")
    if "legs" in order_payload:
        legs = order_payload["legs"]
        if type(legs) not in {list, tuple} or len(legs) < 2:
            raise OrderIntentValidationError("spread order requires at least two legs")
        for leg in legs:
            if type(leg) is not dict:
                raise OrderIntentValidationError("each order leg must be a mapping")
            _validate_mapping_keys(leg, _LEG_FIELDS, "order leg")
    payload = _canonical_value(order_payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalize_order_payload(order_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip only generated transport/derived fields before immutable intent creation.

    The gateway must call this contract on output from ``generate_option_order``.
    Any other field that the XML builder would ignore remains an error during
    strict shape validation rather than silently influencing the intent hash.
    """
    if type(order_payload) is not dict:
        raise OrderIntentValidationError("order payload must be a mapping")
    return {key: value for key, value in order_payload.items() if key not in _NORMALIZED_TRANSPORT_FIELDS}


def canonical_order_payload_hash(order_payload: Mapping[str, Any]) -> str:
    """Return the immutable economic hash used for broker-order comparison."""

    normalized = normalize_order_payload(order_payload)
    wire_payload = wire_order_payload(normalized)
    canonical_payload = canonical_order_payload(json.loads(wire_payload))
    return _payload_hash(canonical_payload)


def wire_order_payload(order_payload: Mapping[str, Any]) -> str:
    """Return a strict broker-schema payload with primitive JSON values intact."""
    if type(order_payload) is not dict or not order_payload:
        raise OrderIntentValidationError("order payload must be a non-empty mapping")
    _validate_mapping_keys(order_payload, _TOP_LEVEL_ORDER_FIELDS, "order payload")
    if "legs" in order_payload:
        legs = order_payload["legs"]
        if type(legs) not in {list, tuple} or len(legs) < 2:
            raise OrderIntentValidationError("spread order requires at least two legs")
        for leg in legs:
            if type(leg) is not dict:
                raise OrderIntentValidationError("each order leg must be a mapping")
            _validate_mapping_keys(leg, _LEG_FIELDS, "order leg")
    _validate_broker_order_shape(order_payload)
    return json.dumps(_wire_value(order_payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_client_order_id(intent: OrderIntent) -> str:
    """Create a stable ten-digit id tied to account, environment, and scope."""
    material = "\x1f".join(
        (
            intent.account_id,
            intent.environment,
            intent.idempotency_scope,
            intent.idempotency_key,
        )
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(_CLIENT_ID_DOMAIN + material).digest()[:8], "big")
    return str(1_000_000_000 + value % 9_000_000_000)


def _stable_amendment_client_order_id(intent: sqlite3.Row, idempotency_key: str) -> str:
    material = "\x1f".join((intent["account_id"], intent["environment"], intent["idempotency_scope"], intent["idempotency_key"], "amend", idempotency_key)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(_CLIENT_ID_DOMAIN + material).digest()[:8], "big")
    return str(1_000_000_000 + value % 9_000_000_000)


def _execute_sql_script(conn: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement schema script without SQLite's implicit commit."""

    statement = ""
    for line in script.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise OrderIntentLedgerError(
            "ledger schema script ended with an incomplete statement"
        )


def _normalized_schema_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def _sqlite_opening_exposure_floor(wire_payload: Any) -> str | None:
    """SQLite fail-closed adapter for the immutable opening-risk floor."""

    try:
        payload = json.loads(wire_payload)
        if type(payload) is not dict:
            return None
        return _canonical_amount(_opening_exposure_floor(payload))
    except Exception:
        return None


def _sqlite_decimal_gte(left: Any, right: Any) -> int:
    """Compare canonical decimal text without SQLite's floating coercion."""

    try:
        return int(
            _canonical_signed_decimal_text(left, "left decimal")
            >= _canonical_signed_decimal_text(right, "right decimal")
        )
    except OrderIntentLedgerError:
        return 0


def _sqlite_legacy_reservation_digest(
    kind: Any,
    source_schema_version: Any,
    intent_id: Any,
    payload_hash: Any,
) -> str | None:
    if (
        kind not in {"quote", "portfolio"}
        or type(source_schema_version) is not int
        or source_schema_version not in {8, 9}
        or type(intent_id) is not str
        or type(payload_hash) is not str
    ):
        return None
    material = {
        "source_schema_version": source_schema_version,
        "intent_id": intent_id,
        "payload_hash": payload_hash,
    }
    domain = (
        b"etrade-legacy-reservation-quote.v1\0"
        if kind == "quote"
        else b"etrade-legacy-reservation-portfolio.v1\0"
    )
    return _domain_json_hash(domain, material)


class OrderIntentLedger:
    """SQLite state machine that refuses duplicate or unresolved live risk."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.run_id = run_id or uuid.uuid4().hex
        _validate_identity("run_id", self.run_id)
        self._securely_precreate_database()
        self._initialize_schema()

    def close(self) -> None:
        """Compatibility no-op; each operation owns a short SQLite connection."""

    def create_intent(self, envelope: OrderIntent, *, intent_id: str | None = None) -> CreateIntentResult:
        if type(envelope) is not OrderIntent:
            raise OrderIntentValidationError("create_intent requires an OrderIntent envelope")
        _validate_envelope(envelope)
        if envelope.intent_kind == "CLOSING":
            raise OrderIntentValidationError(
                "new closing intents require typed position and open-order capacity evidence"
            )
        requested_id = intent_id or uuid.uuid4().hex
        _validate_identity("intent_id", requested_id)
        client_order_id = stable_client_order_id(envelope)
        now = self._now_us()
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM order_intents
                WHERE account_id = ? AND environment = ?
                  AND idempotency_scope = ? AND idempotency_key = ?
                """,
                (
                    envelope.account_id,
                    envelope.environment,
                    envelope.idempotency_scope,
                    envelope.idempotency_key,
                ),
            ).fetchone()
            if existing is not None:
                record = self._intent_from_row(existing)
                if record.envelope != envelope:
                    raise OrderIntentIntegrityError(
                        "idempotency key is already bound to a different immutable envelope"
                    )
                return CreateIntentResult(record, created=False)
            by_id = conn.execute(
                "SELECT * FROM order_intents WHERE intent_id = ?", (requested_id,)
            ).fetchone()
            if by_id is not None:
                record = self._intent_from_row(by_id)
                if record.envelope != envelope:
                    raise OrderIntentIntegrityError("intent_id is already bound to another envelope")
                return CreateIntentResult(record, created=False)
            collision = conn.execute(
                "SELECT intent_id FROM order_intents WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
            if collision is not None:
                raise OrderIntentIntegrityError(
                    "stable client-order-id collision; do not submit ambiguous order intent"
                )
            amendment_collision = conn.execute(
                "SELECT intent_id FROM amendment_leases WHERE client_order_id = ? UNION ALL SELECT intent_id FROM amendment_history WHERE client_order_id = ? LIMIT 1",
                (client_order_id, client_order_id),
            ).fetchone()
            if amendment_collision is not None:
                raise OrderIntentIntegrityError(
                    "stable client-order-id collision with an amendment; do not submit ambiguous order intent"
                )
            conn.execute(
                """
                INSERT INTO order_intents (
                    intent_id, account_id, environment, strategy_id, decision_id,
                    idempotency_scope, idempotency_key, intent_kind, wire_payload, canonical_payload,
                    payload_hash, client_order_id, state, broker_order_id,
                    submission_fence, submission_lease_owner,
                    submission_lease_expires_at, last_reconciled_run, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'INTENT', NULL, 0, NULL, NULL, NULL, ?, ?)
                """,
                (
                    requested_id,
                    envelope.account_id,
                    envelope.environment,
                    envelope.strategy_id,
                    envelope.decision_id,
                    envelope.idempotency_scope,
                    envelope.idempotency_key,
                    envelope.intent_kind,
                    envelope.wire_payload,
                    envelope.canonical_payload,
                    envelope.payload_hash,
                    client_order_id,
                    now,
                    now,
                ),
            )
            self._append_event(
                conn, requested_id, "INTENT_CREATED", None, "INTENT", "system", "INTENT_CREATED", now
            )
            return CreateIntentResult(self._intent_from_row(self._require_intent(conn, requested_id)), True)

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        _validate_identity("intent_id", intent_id)
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (intent_id,)).fetchone()
        return self._intent_from_row(row) if row else None

    def expected_order_payload_hash(self, intent_id: str) -> str:
        """Return the exact durable order terms a broker query must prove."""

        _validate_identity("intent_id", intent_id)
        with self._connection() as conn:
            intent = self._require_intent(conn, intent_id)
            result = self._expected_order_payload_hash_conn(conn, intent)
        _validate_sha256("expected_order_payload_hash", result)
        return result

    def completed_amendment_payload_hash(
        self, intent_id: str, idempotency_key: str
    ) -> str | None:
        """Return immutable terms for a completed amendment replay."""

        _validate_identity("intent_id", intent_id)
        _validate_identity(
            "amendment_idempotency_key", idempotency_key
        )
        with self._connection() as conn:
            self._require_intent(conn, intent_id)
            row = conn.execute(
                """
                SELECT payload_hash FROM amendment_history
                WHERE intent_id = ? AND idempotency_key = ?
                """,
                (intent_id, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        result = row["payload_hash"]
        _validate_sha256("completed_amendment_payload_hash", result)
        return result

    def transport_response_receipts(
        self, intent_id: str
    ) -> tuple[TransportResponseReceipt, ...]:
        """Return durable parsed mutation responses for reconciliation."""

        _validate_identity("intent_id", intent_id)
        with self._connection() as conn:
            self._require_intent(conn, intent_id)
            rows = conn.execute(
                """
                SELECT response.*,
                    attempt.client_order_id AS request_client_order_id,
                    attempt.target_broker_order_id AS request_target_broker_order_id
                FROM transport_response_receipts AS response
                JOIN transport_send_attempts AS attempt
                  ON attempt.intent_id = response.intent_id
                 AND attempt.authorization_operation = response.authorization_operation
                 AND attempt.fencing_token = response.fencing_token
                 AND attempt.transport_operation = response.transport_operation
                WHERE response.intent_id = ?
                ORDER BY
                    CASE response.authorization_operation
                        WHEN 'SUBMIT' THEN 0 ELSE 1
                    END,
                    response.fencing_token,
                    response.recorded_at,
                    CASE
                        WHEN response.transport_operation LIKE '%_PREVIEW' THEN 0
                        ELSE 1
                    END,
                    response.transport_operation
                """,
                (intent_id,),
            ).fetchall()
        return tuple(_transport_response_receipt(row) for row in rows)

    def record_broker_read_response(
        self, evidence: BrokerReadResponseEvidence
    ) -> BrokerReadReceiptRef:
        """Persist one exact bounded GET response before returning it to a reader."""

        now = self._now_us()
        _validate_broker_read_response_evidence(evidence, _from_us(now))
        from live_trading.etrade_broker_reader import (
            _PARSER_CODE_SHA256,
            _PARSER_CONFIG_SHA256,
            _PARSER_SCHEMA,
            _reparse_broker_read_response,
        )

        if (
            evidence.parser_schema != _PARSER_SCHEMA
            or evidence.parser_code_sha256 != _PARSER_CODE_SHA256
            or evidence.parser_config_sha256 != _PARSER_CONFIG_SHA256
        ):
            raise OrderIntentIntegrityError(
                "broker read response did not use the installed parser"
            )
        reparsed_json, reparsed_completeness = (
            _reparse_broker_read_response(evidence)
        )
        if (
            not hmac.compare_digest(
                reparsed_json, evidence.canonical_parsed_json
            )
            or reparsed_completeness != evidence.completeness
        ):
            raise OrderIntentIntegrityError(
                "broker read parser output does not match raw response bytes"
            )
        request_material = {
            "read_kind": evidence.read_kind,
            "account_id": evidence.account_id,
            "account_id_key": evidence.account_id_key,
            "institution_type": evidence.institution_type,
            "environment": evidence.environment,
            "origin": evidence.origin,
            "http_method": "GET",
            "route": evidence.route,
            "query_json": evidence.query_json,
            "authorization_sha256": evidence.authorization_sha256,
            "target_broker_order_id": evidence.target_broker_order_id,
        }
        request_sha256 = _domain_json_hash(
            _READ_REQUEST_HASH_DOMAIN, request_material
        )
        raw_response_sha256 = hashlib.sha256(
            evidence.raw_response_bytes
        ).hexdigest()
        canonical_parsed_sha256 = _domain_bytes_hash(
            _READ_PARSED_HASH_DOMAIN,
            evidence.canonical_parsed_json.encode("utf-8"),
        )
        request_started_at = _to_us(evidence.request_started_at)
        response_completed_at = _to_us(evidence.response_completed_at)
        # The broker-observed completion time is deterministic, so a crash
        # after commit but before return can replay to the same content ID.
        recorded_at = response_completed_at
        receipt_material = {
            "request_sha256": request_sha256,
            "request_started_at": request_started_at,
            "response_completed_at": response_completed_at,
            "http_status": evidence.http_status,
            "raw_byte_length": len(evidence.raw_response_bytes),
            "raw_response_sha256": raw_response_sha256,
            "parser_schema": evidence.parser_schema,
            "parser_code_sha256": evidence.parser_code_sha256,
            "parser_config_sha256": evidence.parser_config_sha256,
            "canonical_parsed_sha256": canonical_parsed_sha256,
            "completeness": evidence.completeness,
            "recorded_at": recorded_at,
        }
        receipt_sha256 = _domain_json_hash(
            _READ_RECEIPT_HASH_DOMAIN, receipt_material
        )
        values = (
            receipt_sha256,
            evidence.read_kind,
            evidence.account_id,
            evidence.account_id_key,
            evidence.institution_type,
            evidence.environment,
            evidence.origin,
            "GET",
            evidence.route,
            evidence.query_json,
            evidence.authorization_sha256,
            request_sha256,
            evidence.target_broker_order_id,
            request_started_at,
            response_completed_at,
            evidence.http_status,
            evidence.raw_response_bytes,
            len(evidence.raw_response_bytes),
            raw_response_sha256,
            evidence.parser_schema,
            evidence.parser_code_sha256,
            evidence.parser_config_sha256,
            evidence.canonical_parsed_json,
            canonical_parsed_sha256,
            evidence.completeness,
            recorded_at,
        )
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM broker_read_receipts
                WHERE receipt_sha256 = ?
                """,
                (receipt_sha256,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO broker_read_receipts (
                        receipt_sha256, read_kind, account_id, account_id_key,
                        institution_type, environment, origin, http_method,
                        route, query_json, authorization_sha256,
                        request_sha256,
                        target_broker_order_id, request_started_at,
                        response_completed_at, http_status, raw_response_bytes,
                        raw_byte_length, raw_response_sha256, parser_schema,
                        parser_code_sha256, parser_config_sha256,
                        canonical_parsed_json, canonical_parsed_sha256,
                        completeness, recorded_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values,
                )
            else:
                columns = (
                    "receipt_sha256",
                    "read_kind",
                    "account_id",
                    "account_id_key",
                    "institution_type",
                    "environment",
                    "origin",
                    "http_method",
                    "route",
                    "query_json",
                    "authorization_sha256",
                    "request_sha256",
                    "target_broker_order_id",
                    "request_started_at",
                    "response_completed_at",
                    "http_status",
                    "raw_response_bytes",
                    "raw_byte_length",
                    "raw_response_sha256",
                    "parser_schema",
                    "parser_code_sha256",
                    "parser_config_sha256",
                    "canonical_parsed_json",
                    "canonical_parsed_sha256",
                    "completeness",
                    "recorded_at",
                )
                if tuple(existing[column] for column in columns) != values:
                    raise OrderIntentIntegrityError(
                        "broker read receipt content-address collision"
                    )
        return BrokerReadReceiptRef(receipt_sha256)

    def record_broker_read_manifest(
        self,
        evidence: BrokerReadManifestEvidence,
        members: tuple[BrokerReadManifestMember, ...],
    ) -> BrokerReadEvidenceRef:
        """Persist one immutable typed result and its ordered response lineage."""

        now = self._now_us()
        _validate_broker_read_manifest_evidence(evidence, _from_us(now))
        _validate_broker_read_manifest_members(members)
        canonical_result_sha256 = _domain_bytes_hash(
            _READ_PARSED_HASH_DOMAIN,
            evidence.canonical_result_json.encode("utf-8"),
        )
        observed_at = _to_us(evidence.observed_at)
        created_at = observed_at
        member_material = [
            {
                "ordinal": ordinal,
                "role": member.role,
                "receipt_sha256": member.receipt_sha256,
            }
            for ordinal, member in enumerate(members)
        ]
        manifest_material = {
            "evidence_kind": evidence.evidence_kind,
            "account_id": evidence.account_id,
            "account_id_key": evidence.account_id_key,
            "institution_type": evidence.institution_type,
            "environment": evidence.environment,
            "origin": evidence.origin,
            "target_broker_order_id": evidence.target_broker_order_id,
            "observed_at": observed_at,
            "completeness": evidence.completeness,
            "canonical_result_sha256": canonical_result_sha256,
            "members": member_material,
            "created_at": created_at,
        }
        evidence_sha256 = _domain_json_hash(
            _READ_MANIFEST_HASH_DOMAIN, manifest_material
        )
        manifest_values = (
            evidence_sha256,
            evidence.evidence_kind,
            evidence.account_id,
            evidence.account_id_key,
            evidence.institution_type,
            evidence.environment,
            evidence.origin,
            evidence.target_broker_order_id,
            observed_at,
            evidence.completeness,
            evidence.canonical_result_json,
            canonical_result_sha256,
            created_at,
        )
        with self._transaction() as conn:
            receipt_rows = []
            for member in members:
                receipt = conn.execute(
                    """
                    SELECT * FROM broker_read_receipts
                    WHERE receipt_sha256 = ?
                    """,
                    (member.receipt_sha256,),
                ).fetchone()
                if receipt is None:
                    raise OrderIntentIntegrityError(
                        "broker read manifest references an unknown receipt"
                    )
                _verify_broker_read_receipt_row(receipt)
                if (
                    receipt["account_id"] != evidence.account_id
                    or receipt["account_id_key"] != evidence.account_id_key
                    or receipt["institution_type"]
                    != evidence.institution_type
                    or receipt["environment"] != evidence.environment
                    or receipt["origin"] != evidence.origin
                    or int(receipt["response_completed_at"]) > observed_at
                ):
                    raise OrderIntentIntegrityError(
                        "broker read manifest member binding is inconsistent"
                    )
                receipt_rows.append(receipt)
            if evidence.completeness == "COMPLETE" and any(
                row["completeness"] == "INELIGIBLE"
                for row in receipt_rows
            ):
                raise OrderIntentIntegrityError(
                    "complete broker read evidence has an ineligible member"
                )
            result = _load_canonical_json_object(
                evidence.canonical_result_json,
                "broker read manifest result",
            )
            _validate_broker_read_manifest_semantics(
                evidence_kind=evidence.evidence_kind,
                account_id=evidence.account_id,
                account_id_key=evidence.account_id_key,
                institution_type=evidence.institution_type,
                target_broker_order_id=evidence.target_broker_order_id,
                observed_at=observed_at,
                completeness=evidence.completeness,
                result=result,
                member_roles=tuple(member.role for member in members),
                receipt_rows=tuple(receipt_rows),
            )
            existing = conn.execute(
                """
                SELECT * FROM broker_read_manifests
                WHERE evidence_sha256 = ?
                """,
                (evidence_sha256,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO broker_read_manifests (
                        evidence_sha256, evidence_kind, account_id,
                        account_id_key, institution_type, environment, origin,
                        target_broker_order_id, observed_at, completeness,
                        canonical_result_json, canonical_result_sha256,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    manifest_values,
                )
                conn.executemany(
                    """
                    INSERT INTO broker_read_manifest_members (
                        evidence_sha256, member_ordinal, member_role,
                        receipt_sha256
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            evidence_sha256,
                            ordinal,
                            member.role,
                            member.receipt_sha256,
                        )
                        for ordinal, member in enumerate(members)
                    ),
                )
            else:
                columns = (
                    "evidence_sha256",
                    "evidence_kind",
                    "account_id",
                    "account_id_key",
                    "institution_type",
                    "environment",
                    "origin",
                    "target_broker_order_id",
                    "observed_at",
                    "completeness",
                    "canonical_result_json",
                    "canonical_result_sha256",
                    "created_at",
                )
                if tuple(existing[column] for column in columns) != manifest_values:
                    raise OrderIntentIntegrityError(
                        "broker read manifest content-address collision"
                    )
                existing_members = conn.execute(
                    """
                    SELECT member_role, receipt_sha256
                    FROM broker_read_manifest_members
                    WHERE evidence_sha256 = ?
                    ORDER BY member_ordinal
                    """,
                    (evidence_sha256,),
                ).fetchall()
                if tuple(
                    (row["member_role"], row["receipt_sha256"])
                    for row in existing_members
                ) != tuple(
                    (member.role, member.receipt_sha256)
                    for member in members
                ):
                    raise OrderIntentIntegrityError(
                        "broker read manifest membership conflicts"
                    )
        return BrokerReadEvidenceRef(
            evidence_sha256, evidence.evidence_kind
        )

    def broker_read_evidence(
        self, evidence_sha256: str
    ) -> BrokerReadEvidenceRef:
        """Verify and return a redacted reference to immutable read evidence."""

        _validate_sha256("evidence_sha256", evidence_sha256)
        with self._connection() as conn:
            row, _ = self._verified_broker_read_manifest(
                conn, evidence_sha256
            )
        return BrokerReadEvidenceRef(
            evidence_sha256, row["evidence_kind"]
        )

    def set_reservation_cap_from_read(
        self,
        evidence: BrokerReadEvidenceRef,
        *,
        risk_budget: Decimal,
    ) -> CapacityDecisionReceipt:
        """Derive and persist a cap only from a complete durable capacity read."""

        if (
            type(evidence) is not BrokerReadEvidenceRef
            or evidence.evidence_kind != "CAPACITY"
        ):
            raise OrderIntentValidationError(
                "capacity requires an exact CAPACITY evidence reference"
            )
        _validate_sha256("capacity evidence", evidence.evidence_sha256)
        if (
            type(risk_budget) is not Decimal
            or not risk_budget.is_finite()
            or risk_budget < 0
        ):
            raise OrderIntentValidationError(
                "risk_budget must be a non-negative finite Decimal"
            )
        now = self._now_us()
        with self._transaction() as conn:
            manifest, result = self._verified_broker_read_manifest(
                conn, evidence.evidence_sha256
            )
            if (
                manifest["evidence_kind"] != "CAPACITY"
                or manifest["completeness"] != "COMPLETE"
                or manifest["target_broker_order_id"] is not None
            ):
                raise OrderIntentIntegrityError(
                    "capacity evidence is not a complete capacity manifest"
                )
            _validate_capacity_manifest_result(result)
            observed_at = int(manifest["observed_at"])
            if (
                observed_at > now + 5_000_000
                or now - observed_at
                > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000
            ):
                raise OrderIntentValidationError(
                    "capacity evidence is stale or from the future"
                )
            broker_buying_power = Decimal(
                result["broker_buying_power"]
            )
            canonical_risk_budget = _canonical_amount(risk_budget)
            cap_amount = _canonical_amount(
                min(broker_buying_power, risk_budget)
            )
            snapshot_digest = result["state_sha256"]
            decided_at = observed_at
            decision_material = {
                "evidence_sha256": evidence.evidence_sha256,
                "account_id": manifest["account_id"],
                "environment": manifest["environment"],
                "broker_buying_power": _canonical_amount(
                    broker_buying_power
                ),
                "risk_budget": canonical_risk_budget,
                "cap_amount": cap_amount,
                "observed_at": observed_at,
                "capacity_snapshot_sha256": snapshot_digest,
                "decided_at": decided_at,
            }
            decision_sha256 = _domain_json_hash(
                _CAPACITY_DECISION_HASH_DOMAIN, decision_material
            )
            decision_values = (
                decision_sha256,
                evidence.evidence_sha256,
                manifest["account_id"],
                manifest["environment"],
                _canonical_amount(broker_buying_power),
                canonical_risk_budget,
                cap_amount,
                observed_at,
                snapshot_digest,
                decided_at,
            )
            existing_decision = conn.execute(
                """
                SELECT * FROM capacity_decisions
                WHERE capacity_decision_sha256 = ?
                """,
                (decision_sha256,),
            ).fetchone()
            if existing_decision is None:
                conn.execute(
                    """
                    INSERT INTO capacity_decisions (
                        capacity_decision_sha256, evidence_sha256,
                        account_id, environment, broker_buying_power,
                        risk_budget, cap_amount, observed_at,
                        capacity_snapshot_sha256, decided_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    decision_values,
                )
            else:
                columns = (
                    "capacity_decision_sha256",
                    "evidence_sha256",
                    "account_id",
                    "environment",
                    "broker_buying_power",
                    "risk_budget",
                    "cap_amount",
                    "observed_at",
                    "capacity_snapshot_sha256",
                    "decided_at",
                )
                if tuple(
                    existing_decision[column] for column in columns
                ) != decision_values:
                    raise OrderIntentIntegrityError(
                        "capacity decision content-address collision"
                    )
            existing_cap = conn.execute(
                """
                SELECT * FROM reservation_caps
                WHERE account_id = ? AND environment = ?
                """,
                (manifest["account_id"], manifest["environment"]),
            ).fetchone()
            if (
                existing_cap is not None
                and observed_at < int(existing_cap["observed_at"])
            ):
                raise OrderIntentIntegrityError(
                    "capacity evidence must not predate the persisted account snapshot"
                )
            if (
                existing_cap is not None
                and observed_at == int(existing_cap["observed_at"])
                and existing_cap["capacity_decision_sha256"]
                not in {None, decision_sha256}
            ):
                raise OrderIntentIntegrityError(
                    "equal-time capacity evidence conflicts with the persisted snapshot"
                )
            conn.execute(
                """
                INSERT INTO reservation_caps (
                    account_id, environment, cap_amount,
                    broker_buying_power, risk_budget, observed_at,
                    portfolio_snapshot_digest, updated_at,
                    capacity_decision_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, environment) DO UPDATE SET
                    cap_amount = excluded.cap_amount,
                    broker_buying_power = excluded.broker_buying_power,
                    risk_budget = excluded.risk_budget,
                    observed_at = excluded.observed_at,
                    portfolio_snapshot_digest =
                        excluded.portfolio_snapshot_digest,
                    updated_at = excluded.updated_at,
                    capacity_decision_sha256 =
                        excluded.capacity_decision_sha256
                """,
                (
                    manifest["account_id"],
                    manifest["environment"],
                    cap_amount,
                    _canonical_amount(broker_buying_power),
                    canonical_risk_budget,
                    observed_at,
                    snapshot_digest,
                    now,
                    decision_sha256,
                ),
            )
        return CapacityDecisionReceipt(
            decision_sha256=decision_sha256,
            evidence_sha256=evidence.evidence_sha256,
            cap_amount=Decimal(cap_amount),
            broker_buying_power=broker_buying_power,
            risk_budget=Decimal(canonical_risk_budget),
            observed_at=_from_us(observed_at),
            portfolio_snapshot_digest=snapshot_digest,
        )

    def broker_evidence_from_read(
        self,
        intent_id: str,
        evidence: BrokerReadEvidenceRef,
        *,
        operation: Literal["ORDER_QUERY", "AMEND_QUERY"],
    ) -> BrokerEvidence | None:
        """Derive reconciliation facts from a durable known-order manifest."""

        _validate_identity("intent_id", intent_id)
        if (
            type(evidence) is not BrokerReadEvidenceRef
            or evidence.evidence_kind != "ORDER_QUERY"
        ):
            raise OrderIntentValidationError(
                "order reconciliation requires exact ORDER_QUERY evidence"
            )
        if type(operation) is not str or operation not in {
            "ORDER_QUERY",
            "AMEND_QUERY",
        }:
            raise OrderIntentValidationError(
                "read evidence operation is invalid"
            )
        now = self._now_us()
        with self._connection() as conn:
            intent = self._require_intent(conn, intent_id)
            manifest, result = self._verified_broker_read_manifest(
                conn, evidence.evidence_sha256
            )
            _validate_order_query_manifest_result(result)
            if (
                manifest["evidence_kind"] != "ORDER_QUERY"
                or manifest["account_id"] != intent["account_id"]
                or manifest["environment"] != intent["environment"]
                or manifest["target_broker_order_id"]
                != result["broker_order_id"]
            ):
                raise OrderIntentIntegrityError(
                    "order read evidence does not match the durable intent"
                )
            if (
                manifest["completeness"] != "COMPLETE"
                or result["not_found"] is True
                or result["outcome"] == "UNRESOLVED"
            ):
                return None
            observed_at = int(manifest["observed_at"])
            if (
                observed_at > now + 5_000_000
                or now - observed_at
                > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000
            ):
                raise OrderIntentValidationError(
                    "order read evidence is stale or from the future"
                )
            expected_client_id = intent["client_order_id"]
            if operation == "AMEND_QUERY":
                amendment = conn.execute(
                    """
                    SELECT client_order_id FROM amendment_leases
                    WHERE intent_id = ?
                    """,
                    (intent_id,),
                ).fetchone()
                if amendment is None:
                    raise OrderIntentIntegrityError(
                        "amendment query lacks a durable amendment operation"
                    )
                expected_client_id = amendment["client_order_id"]
        return BrokerEvidence(
            account_id=manifest["account_id"],
            environment=manifest["environment"],
            client_order_id=expected_client_id,
            broker_order_id=result["broker_order_id"],
            operation=operation,
            outcome=result["outcome"],
            observed_at=_from_us(observed_at),
            http_status=result["http_status"],
            raw_response_digest=result["raw_response_digest"],
            broker_read_evidence_sha256=evidence.evidence_sha256,
            order_payload_hashes=tuple(result["order_payload_hashes"]),
        )

    def set_reservation_cap(self, evidence: AccountCapacityEvidence) -> Decimal:
        """Compatibility wrapper that still requires durable broker evidence."""

        if type(evidence) is not AccountCapacityEvidence:
            raise OrderIntentValidationError("set_reservation_cap requires typed AccountCapacityEvidence")
        AccountCapacityEvidence.validate(
            evidence, _from_us(self._now_us())
        )
        if evidence.broker_read_evidence_sha256 is None:
            raise OrderIntentValidationError(
                "capacity evidence must reference a durable broker-read manifest"
            )
        decision = self.set_reservation_cap_from_read(
            BrokerReadEvidenceRef(
                evidence.broker_read_evidence_sha256, "CAPACITY"
            ),
            risk_budget=evidence.risk_budget,
        )
        if (
            decision.broker_buying_power != evidence.broker_buying_power
            or decision.observed_at != evidence.observed_at
            or decision.portfolio_snapshot_digest
            != evidence.portfolio_snapshot_digest
        ):
            raise OrderIntentIntegrityError(
                "capacity wrapper conflicts with durable read evidence"
            )
        return decision.cap_amount

    def reserve_margin(self, intent_id: str, evidence: RiskEvidence) -> MarginReservation:
        _validate_identity("intent_id", intent_id)
        now = self._now_us()
        if type(evidence) is not RiskEvidence:
            raise OrderIntentValidationError("reserve_margin requires typed RiskEvidence")
        RiskEvidence.validate(evidence, _from_us(now))
        value = _canonical_amount(evidence.collateral_amount)
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["intent_kind"] != "OPENING" or intent["state"] != "INTENT":
                raise OrderIntentTransitionError("only a new opening intent may reserve margin")
            if evidence.decision_id != intent["decision_id"]:
                raise OrderIntentIntegrityError("risk evidence decision id does not match intent")
            exposure_floor = _opening_exposure_floor(json.loads(intent["wire_payload"]))
            if evidence.max_loss_amount < exposure_floor or evidence.collateral_amount < exposure_floor:
                raise OrderIntentReservationError(
                    "risk evidence is below the immutable opening exposure floor"
                )
            cap = conn.execute(
                """
                SELECT *
                FROM reservation_caps
                WHERE account_id = ? AND environment = ?
                """,
                (intent["account_id"], intent["environment"]),
            ).fetchone()
            if cap is None:
                raise OrderIntentReservationError("opening reservations require an account/environment cap")
            if (
                cap["capacity_decision_sha256"] is None
                or evidence.capacity_decision_sha256
                != cap["capacity_decision_sha256"]
            ):
                raise OrderIntentIntegrityError(
                    "reservation must name the exact durable capacity decision"
                )
            self._verified_reservation_cap_row(conn, cap)
            if (
                int(cap["observed_at"]) != _to_us(evidence.portfolio_observed_at)
                or cap["portfolio_snapshot_digest"] != evidence.portfolio_snapshot_digest
            ):
                raise OrderIntentIntegrityError(
                    "reservation risk evidence must use the exact portfolio snapshot that set its capacity cap"
                )
            existing = conn.execute(
                "SELECT * FROM margin_reservations WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if existing is not None:
                reservation = self._reservation_from_row(existing)
                if reservation.state == "ACTIVE" and (
                    reservation.amount == Decimal(value) and reservation.risk_decision_id == evidence.decision_id
                    and reservation.max_loss_amount == evidence.max_loss_amount and reservation.quote_observed_at == evidence.quote_observed_at
                    and reservation.quote_digest == evidence.quote_digest and reservation.portfolio_observed_at == evidence.portfolio_observed_at
                    and reservation.portfolio_snapshot_digest == evidence.portfolio_snapshot_digest
                    and reservation.capacity_decision_sha256
                    == evidence.capacity_decision_sha256
                ):
                    return reservation
                raise OrderIntentTransitionError("reservation already exists and is immutable")
            active = self._active_reservation_total(conn, intent["account_id"], intent["environment"])
            with localcontext() as decimal_context:
                decimal_context.prec = _DECIMAL_PRECISION
                exceeds_cap = active + Decimal(value) > Decimal(cap["cap_amount"])
            if exceeds_cap:
                raise OrderIntentReservationError("margin reservation exceeds account/environment cap")
            conn.execute(
                """
                INSERT INTO margin_reservations (
                    intent_id, account_id, environment, amount, risk_decision_id, max_loss_amount,
                    quote_observed_at, quote_digest, portfolio_observed_at,
                    portfolio_snapshot_digest, capacity_decision_sha256,
                    state, released_reason_code, created_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, NULL)
                """,
                (
                    intent_id,
                    intent["account_id"],
                    intent["environment"],
                    value,
                    evidence.decision_id,
                    _canonical_amount(evidence.max_loss_amount),
                    _to_us(evidence.quote_observed_at),
                    evidence.quote_digest,
                    _to_us(evidence.portfolio_observed_at),
                    evidence.portfolio_snapshot_digest,
                    evidence.capacity_decision_sha256,
                    now,
                ),
            )
            self._append_event(conn, intent_id, "RESERVATION_CREATED", "INTENT", "INTENT", "system", "RESERVATION_CREATED", now)
            return self._reservation_from_row(
                conn.execute("SELECT * FROM margin_reservations WHERE intent_id = ?", (intent_id,)).fetchone()
            )

    def get_margin_reservation(self, intent_id: str) -> MarginReservation | None:
        _validate_identity("intent_id", intent_id)
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM margin_reservations WHERE intent_id = ?", (intent_id,)).fetchone()
        return self._reservation_from_row(row) if row else None

    def active_reserved_margin(self, account_id: str, environment: str) -> Decimal:
        _validate_identity("account_id", account_id)
        _validate_identity("environment", environment)
        with self._connection() as conn:
            return self._active_reservation_total(conn, account_id, environment)

    def unabsorbed_filled_reservation_count(
        self, account_id: str, environment: str
    ) -> int:
        """Return terminal opening risk not yet proven in positions."""

        _validate_identity("account_id", account_id)
        _validate_environment(environment)
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM margin_reservations
                WHERE account_id = ? AND environment = ?
                  AND state = 'FILLED_PENDING_ABSORPTION'
                """,
                (account_id, environment),
            ).fetchone()
        return int(row["total"])

    def pending_terminal_reservations(
        self, account_id: str, environment: str
    ) -> tuple[IntentRecord, ...]:
        """Return terminal opening intents whose risk is not yet absorbed."""

        _validate_identity("account_id", account_id)
        _validate_environment(environment)
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT order_intents.*,
                       margin_reservations.account_id
                           AS reservation_account_id,
                       margin_reservations.environment
                           AS reservation_environment
                FROM margin_reservations
                JOIN order_intents USING (intent_id)
                WHERE margin_reservations.state =
                          'FILLED_PENDING_ABSORPTION'
                  AND (
                        (
                            margin_reservations.account_id = ?
                            AND margin_reservations.environment = ?
                        )
                        OR
                        (
                            order_intents.account_id = ?
                            AND order_intents.environment = ?
                        )
                  )
                ORDER BY order_intents.created_at, order_intents.intent_id
                """,
                (
                    account_id,
                    environment,
                    account_id,
                    environment,
                ),
            ).fetchall()
            for row in rows:
                if (
                    row["account_id"] != account_id
                    or row["environment"] != environment
                    or row["reservation_account_id"] != account_id
                    or row["reservation_environment"] != environment
                    or row["intent_kind"] != "OPENING"
                    or row["state"]
                    not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
                    or row["broker_order_id"] is None
                    or conn.execute(
                        """
                        SELECT 1 FROM reservation_absorptions
                        WHERE intent_id = ?
                        """,
                        (row["intent_id"],),
                    ).fetchone()
                    is not None
                ):
                    raise OrderIntentIntegrityError(
                        "pending terminal reservation has inconsistent durable state"
                    )
            return tuple(self._intent_from_row(row) for row in rows)

    def claim_submission(self, intent_id: str, owner: str, *, lease_seconds: float) -> SubmissionLease:
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        now = self._now_us()
        expires_at = self._lease_expiry_us(lease_seconds, now)
        lease: SubmissionLease | None = None
        failure: Exception | None = None
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            self._expire_claimed_leases(conn, intent["account_id"], intent["environment"], now)
            self._expire_amendment_leases(conn, intent["account_id"], intent["environment"], now)
            intent = self._require_intent(conn, intent_id)
            if intent["state"] != "INTENT":
                if intent["state"] in _BROKER_LIVE_STATES:
                    failure = OrderIntentReconciliationRequired(
                        f"intent {intent_id} is no longer safe to submit without reconciliation"
                    )
                else:
                    failure = OrderIntentTransitionError(f"cannot claim submission from {intent['state']}")
            elif intent["intent_kind"] == "CLOSING":
                failure = OrderIntentReservationError(
                    "closing submission requires typed position and open-order capacity evidence"
                )
            elif self._blocker_rows(conn, intent["account_id"], intent["environment"]) or self._amendment_blocker_rows(conn, intent["account_id"], intent["environment"]):
                failure = OrderIntentReconciliationRequired("account/environment has unresolved broker intents")
            else:
                if intent["intent_kind"] == "OPENING":
                    self._require_active_opening_reservation(conn, intent, now)
                fence = int(intent["submission_fence"]) + 1
                conn.execute(
                    """
                    UPDATE order_intents
                    SET state = 'CLAIMED', submission_fence = ?, submission_lease_owner = ?,
                        submission_lease_expires_at = ?, last_reconciled_run = NULL, updated_at = ?
                    WHERE intent_id = ?
                    """,
                    (fence, owner, expires_at, now, intent_id),
                )
                self._append_event(conn, intent_id, "SUBMISSION_CLAIMED", "INTENT", "CLAIMED", owner, "SUBMISSION_CLAIMED", now)
                lease = SubmissionLease(intent_id, owner, fence, _from_us(expires_at))
        if failure is not None:
            raise failure
        assert lease is not None
        return lease

    def renew_submission_lease(
        self, intent_id: str, owner: str, fencing_token: int, *, lease_seconds: float
    ) -> SubmissionLease:
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        expires_at = self._lease_expiry_us(lease_seconds, now)
        lease: SubmissionLease | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["intent_kind"] == "CLOSING":
                raise OrderIntentReservationError(
                    "closing submission renewal requires typed position and open-order capacity evidence"
                )
            if self._claim_expired(intent, now):
                self._mark_claim_in_doubt(conn, intent, now)
                expired = True
            else:
                self._require_submission_fence(intent, owner, fencing_token)
                if intent["intent_kind"] == "OPENING":
                    self._require_active_opening_reservation(conn, intent, now)
                if self._amendment_blocker_rows(conn, intent["account_id"], intent["environment"]):
                    raise OrderIntentReconciliationRequired("account/environment has an unresolved amendment")
                conn.execute(
                    "UPDATE order_intents SET submission_lease_expires_at = ?, updated_at = ? WHERE intent_id = ?",
                    (expires_at, now, intent_id),
                )
                self._append_event(conn, intent_id, "LEASE_RENEWED", "CLAIMED", "CLAIMED", owner, "LEASE_RENEWED", now)
                lease = SubmissionLease(intent_id, owner, fencing_token, _from_us(expires_at))
        if expired:
            raise OrderIntentReconciliationRequired("expired claimed lease is in-doubt and cannot be renewed")
        assert lease is not None
        return lease

    def prepare_submission_payload(self, intent_id: str, owner: str, fencing_token: int) -> OutboundAuthorization:
        """Return immutable, exact bytes authorized for the fenced submission."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        authorization: OutboundAuthorization | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["intent_kind"] == "CLOSING":
                raise OrderIntentReservationError(
                    "closing payload preparation requires typed position and open-order capacity evidence"
                )
            if self._claim_expired(intent, now):
                self._mark_claim_in_doubt(conn, intent, now)
                expired = True
            else:
                self._require_submission_fence(intent, owner, fencing_token)
                record = self._intent_from_row(intent)
                payload = json.loads(record.envelope.wire_payload)
                if not isinstance(payload, dict):
                    raise OrderIntentIntegrityError("stored wire payload is not an object")
                if canonical_order_payload(payload) != record.envelope.canonical_payload:
                    raise OrderIntentIntegrityError("stored wire payload no longer matches canonical payload")
                payload["client_order_id"] = record.client_order_id
                authorization = _outbound_authorization(
                    intent_id=intent_id,
                    operation="SUBMIT",
                    owner=owner,
                    fencing_token=fencing_token,
                    client_order_id=record.client_order_id,
                    payload=payload,
                )
        if expired:
            raise OrderIntentReconciliationRequired("expired claimed lease is in-doubt; do not POST")
        assert authorization is not None
        return authorization

    def claim_transport_send(
        self,
        evidence: TransportRequestEvidence,
        authorization: OutboundAuthorization,
    ) -> None:
        """Persist one exact broker send and transition place calls in-doubt.

        The unique attempt row is the durable exactly-once fence. A process
        crash after this method returns is reconciled and never retransmitted.
        """

        _validate_transport_request_evidence(evidence)
        if type(authorization) is not OutboundAuthorization:
            raise OrderIntentValidationError(
                "transport send requires exact outbound authorization"
            )
        now = self._now_us()
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, evidence.intent_id)
            if (
                intent["account_id"] != evidence.account_id
                or intent["environment"] != evidence.environment
                or (
                    evidence.authorization_operation == "SUBMIT"
                    and intent["client_order_id"] != evidence.client_order_id
                )
            ):
                raise OrderIntentIntegrityError(
                    "transport account or client identity does not match intent"
                )
            if conn.execute(
                """
                SELECT 1 FROM transport_send_attempts
                WHERE intent_id = ? AND authorization_operation = ?
                  AND fencing_token = ? AND transport_operation = ?
                """,
                (
                    evidence.intent_id,
                    evidence.authorization_operation,
                    evidence.fencing_token,
                    evidence.transport_operation,
                ),
            ).fetchone() is not None:
                raise OrderIntentReconciliationRequired(
                    "exact transport stage already has a durable send attempt"
                )

            if evidence.authorization_operation == "SUBMIT":
                if evidence.target_broker_order_id is not None:
                    raise OrderIntentIntegrityError(
                        "submission transport cannot target a broker order"
                    )
                if self._claim_expired(intent, now):
                    self._mark_claim_in_doubt(conn, intent, now)
                    expired = True
                else:
                    self._require_submission_fence(
                        intent, evidence.owner, evidence.fencing_token
                    )
                    payload = json.loads(intent["wire_payload"])
                    if type(payload) is not dict:
                        raise OrderIntentIntegrityError(
                            "stored submission payload is invalid"
                        )
                    payload["client_order_id"] = intent["client_order_id"]
                    validated = self._require_outbound_authorization(
                        authorization,
                        intent_id=evidence.intent_id,
                        operation="SUBMIT",
                        owner=evidence.owner,
                        fencing_token=evidence.fencing_token,
                        client_order_id=evidence.client_order_id,
                        payload=payload,
                    )
                    self._require_active_opening_reservation(conn, intent, now)
                    if self._amendment_blocker_rows(
                        conn,
                        intent["account_id"],
                        intent["environment"],
                    ):
                        raise OrderIntentReconciliationRequired(
                            "account/environment has an unresolved amendment"
                        )
                    self._record_outbound_authorization(conn, validated, now)
                    if evidence.transport_operation == "SUBMIT_PLACE":
                        self._require_preview_receipt(conn, evidence, now)
                        conn.execute(
                            """
                            UPDATE order_intents
                            SET state = 'SUBMISSION_UNKNOWN',
                                submission_lease_owner = NULL,
                                submission_lease_expires_at = NULL,
                                pending_operation = 'SUBMIT', pending_owner = ?,
                                pending_fence = ?, last_reconciled_run = NULL,
                                updated_at = ?
                            WHERE intent_id = ?
                            """,
                            (
                                evidence.owner,
                                evidence.fencing_token,
                                now,
                                evidence.intent_id,
                            ),
                        )
                        self._append_event(
                            conn,
                            evidence.intent_id,
                            "POST_STARTED",
                            "CLAIMED",
                            "SUBMISSION_UNKNOWN",
                            evidence.owner,
                            "POST_STARTED",
                            now,
                        )
            else:
                amendment = conn.execute(
                    "SELECT * FROM amendment_leases WHERE intent_id = ?",
                    (evidence.intent_id,),
                ).fetchone()
                if (
                    amendment is None
                    or amendment["state"] != "LEASED"
                    or amendment["owner"] != evidence.owner
                    or int(amendment["fencing_token"]) != evidence.fencing_token
                ):
                    raise OrderIntentLeaseConflict(
                        "amendment transport is not owned by this fence"
                    )
                if int(amendment["expires_at"]) <= now:
                    self._mark_amendment_in_doubt(conn, intent, amendment, now)
                    expired = True
                elif (
                    evidence.target_broker_order_id != amendment["broker_order_id"]
                    or evidence.client_order_id != amendment["client_order_id"]
                ):
                    raise OrderIntentIntegrityError(
                        "amendment transport target is not lease-bound"
                    )
                else:
                    payload = json.loads(amendment["wire_payload"])
                    if type(payload) is not dict:
                        raise OrderIntentIntegrityError(
                            "stored amendment payload is invalid"
                        )
                    payload["client_order_id"] = amendment["client_order_id"]
                    validated = self._require_outbound_authorization(
                        authorization,
                        intent_id=evidence.intent_id,
                        operation="AMEND",
                        owner=evidence.owner,
                        fencing_token=evidence.fencing_token,
                        client_order_id=evidence.client_order_id,
                        payload=payload,
                    )
                    if self._blocker_rows(
                        conn,
                        intent["account_id"],
                        intent["environment"],
                        exclude_intent_id=evidence.intent_id,
                    ) or self._amendment_blocker_rows(
                        conn,
                        intent["account_id"],
                        intent["environment"],
                        exclude_intent_id=evidence.intent_id,
                    ):
                        raise OrderIntentReconciliationRequired(
                            "account/environment has unresolved broker work"
                        )
                    self._require_opening_amendment_reservation(
                        conn, intent, amendment["wire_payload"], now
                    )
                    self._record_outbound_authorization(conn, validated, now)
                    if evidence.transport_operation == "AMEND_PLACE":
                        self._require_preview_receipt(conn, evidence, now)
                        conn.execute(
                            "UPDATE amendment_leases SET state = 'IN_DOUBT', updated_at = ? WHERE intent_id = ?",
                            (now, evidence.intent_id),
                        )
                        conn.execute(
                            """
                            UPDATE order_intents
                            SET pending_operation = 'AMEND', pending_owner = ?,
                                pending_fence = ?, last_reconciled_run = NULL,
                                updated_at = ?
                            WHERE intent_id = ?
                            """,
                            (
                                evidence.owner,
                                evidence.fencing_token,
                                now,
                                evidence.intent_id,
                            ),
                        )
                        self._append_event(
                            conn,
                            evidence.intent_id,
                            "AMENDMENT_STARTED",
                            "SUBMITTED",
                            "SUBMITTED",
                            evidence.owner,
                            "POST_STARTED",
                            now,
                            broker_order_id=intent["broker_order_id"],
                        )

            if not expired:
                conn.execute(
                    """
                    INSERT INTO transport_send_attempts (
                        intent_id, authorization_operation, fencing_token,
                        transport_operation, owner, account_id, account_id_key,
                        institution_type, environment, http_method, route,
                        client_order_id, target_broker_order_id, preview_id,
                        authorization_payload_digest, final_xml_bytes,
                        final_xml_sha256, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence.intent_id,
                        evidence.authorization_operation,
                        evidence.fencing_token,
                        evidence.transport_operation,
                        evidence.owner,
                        evidence.account_id,
                        evidence.account_id_key,
                        evidence.institution_type,
                        evidence.environment,
                        evidence.http_method,
                        evidence.route,
                        evidence.client_order_id,
                        evidence.target_broker_order_id,
                        evidence.preview_id,
                        evidence.authorization_payload_digest,
                        evidence.final_xml_bytes,
                        evidence.final_xml_sha256,
                        now,
                    ),
                )
        if expired:
            raise OrderIntentReconciliationRequired(
                "expired transport lease is in-doubt and cannot be sent"
            )

    def record_transport_response(
        self,
        request: TransportRequestEvidence,
        response: TransportResponseEvidence,
    ) -> IntentRecord:
        """Persist the first parsed response and atomically apply an ACK."""

        _validate_transport_request_evidence(request)
        _validate_transport_response_evidence(response)
        _validate_acknowledged_transport_response(request, response)
        now = self._now_us()
        observed = _to_us(response.observed_at)
        messages_json = json.dumps(
            [
                {
                    "code": code,
                    "type": message_type,
                    "description_sha256": description_digest,
                }
                for code, message_type, description_digest in zip(
                    response.message_codes,
                    response.message_types,
                    response.message_description_digests,
                    strict=True,
                )
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._transaction() as conn:
            attempt = conn.execute(
                """
                SELECT * FROM transport_send_attempts
                WHERE intent_id = ? AND authorization_operation = ?
                  AND fencing_token = ? AND transport_operation = ?
                """,
                (
                    request.intent_id,
                    request.authorization_operation,
                    request.fencing_token,
                    request.transport_operation,
                ),
            ).fetchone()
            if attempt is None or any(
                attempt[field] != expected
                for field, expected in (
                    ("owner", request.owner),
                    ("account_id", request.account_id),
                    ("account_id_key", request.account_id_key),
                    ("institution_type", request.institution_type),
                    ("environment", request.environment),
                    ("http_method", request.http_method),
                    ("route", request.route),
                    ("client_order_id", request.client_order_id),
                    ("target_broker_order_id", request.target_broker_order_id),
                    ("preview_id", request.preview_id),
                    (
                        "authorization_payload_digest",
                        request.authorization_payload_digest,
                    ),
                    ("final_xml_sha256", request.final_xml_sha256),
                )
            ):
                raise OrderIntentIntegrityError(
                    "transport response does not match its durable send attempt"
                )
            if observed < int(attempt["claimed_at"]) or observed > now + 5_000_000:
                raise OrderIntentIntegrityError(
                    "transport response time is not causally bound to its send"
                )
            conn.execute(
                """
                INSERT INTO transport_response_receipts (
                    intent_id, authorization_operation, fencing_token,
                    transport_operation, disposition, http_status,
                    broker_status, broker_order_id, preview_id, messages_json,
                    raw_response_digest, observed_at, unknown_reason, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.intent_id,
                    request.authorization_operation,
                    request.fencing_token,
                    request.transport_operation,
                    response.disposition,
                    response.http_status,
                    response.broker_status,
                    response.broker_order_id,
                    response.preview_id,
                    messages_json,
                    response.raw_response_digest,
                    observed,
                    response.unknown_reason,
                    now,
                ),
            )
            if (
                response.disposition == "ACKNOWLEDGED"
                and request.transport_operation.endswith("PREVIEW")
            ):
                conn.execute(
                    """
                    INSERT INTO broker_preview_receipts (
                        intent_id, authorization_operation, fencing_token,
                        account_id, environment, client_order_id,
                        target_broker_order_id, authorization_payload_digest,
                        preview_id, raw_response_digest, observed_at, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.intent_id,
                        request.authorization_operation,
                        request.fencing_token,
                        request.account_id,
                        request.environment,
                        request.client_order_id,
                        request.target_broker_order_id,
                        request.authorization_payload_digest,
                        response.preview_id,
                        response.raw_response_digest,
                        observed,
                        now,
                    ),
                )
            if (
                response.disposition == "ACKNOWLEDGED"
                and request.transport_operation.endswith("PLACE")
            ):
                self._apply_transport_ack(conn, request, response, now)
            return self._intent_from_row(
                self._require_intent(conn, request.intent_id)
            )

    def begin_submission(
        self, intent_id: str, owner: str, fencing_token: int, authorization: OutboundAuthorization
    ) -> IntentRecord:
        """Durably authorize exact bytes and enter in-doubt state before POST."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        record: IntentRecord | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["intent_kind"] == "CLOSING":
                raise OrderIntentReservationError(
                    "closing submission requires typed position and open-order capacity evidence"
                )
            if self._claim_expired(intent, now):
                self._mark_claim_in_doubt(conn, intent, now)
                expired = True
            else:
                self._require_submission_fence(intent, owner, fencing_token)
                record = self._intent_from_row(intent)
                payload = json.loads(record.envelope.wire_payload)
                if not isinstance(payload, dict):
                    raise OrderIntentIntegrityError("stored wire payload is not an object")
                payload["client_order_id"] = record.client_order_id
                validated_authorization = self._require_outbound_authorization(
                    authorization,
                    intent_id=intent_id,
                    operation="SUBMIT",
                    owner=owner,
                    fencing_token=fencing_token,
                    client_order_id=record.client_order_id,
                    payload=payload,
                )
                if intent["intent_kind"] == "OPENING":
                    self._require_active_opening_reservation(conn, intent, now)
                if self._amendment_blocker_rows(conn, intent["account_id"], intent["environment"]):
                    raise OrderIntentReconciliationRequired("account/environment has an unresolved amendment")
                self._record_outbound_authorization(conn, validated_authorization, now)
                conn.execute(
                    """
                    UPDATE order_intents
                    SET state = 'SUBMISSION_UNKNOWN', submission_lease_owner = NULL,
                        submission_lease_expires_at = NULL, pending_operation = 'SUBMIT',
                        pending_owner = ?, pending_fence = ?, last_reconciled_run = NULL, updated_at = ?
                    WHERE intent_id = ?
                    """,
                    (owner, fencing_token, now, intent_id),
                )
                self._append_event(conn, intent_id, "POST_STARTED", "CLAIMED", "SUBMISSION_UNKNOWN", owner, "POST_STARTED", now)
                record = self._intent_from_row(self._require_intent(conn, intent_id))
        if expired:
            raise OrderIntentReconciliationRequired("expired claimed lease is in-doubt; do not POST")
        assert record is not None
        return record

    def record_post_acknowledgement(self, intent_id: str, owner: str, fencing_token: int, evidence: BrokerEvidence) -> IntentRecord:
        """Persist a definitive response to the just-started POST, without a retry."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        if type(evidence) is not BrokerEvidence:
            raise OrderIntentValidationError("broker acknowledgement requires exact BrokerEvidence")
        BrokerEvidence.validate(evidence, _from_us(now))
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"SUBMIT_ACK"}, allowed_outcomes={"OPEN"})
            if intent["state"] != "SUBMISSION_UNKNOWN" or intent["pending_operation"] != "SUBMIT" or intent["pending_owner"] != owner or int(intent["pending_fence"] or -1) != fencing_token:
                raise OrderIntentTransitionError("broker acknowledgement requires SUBMISSION_UNKNOWN")
            if evidence.http_status != 200:
                raise OrderIntentValidationError("definitive submit acknowledgement requires HTTP 200")
            self._bind_broker_order_history(conn, evidence.broker_order_id, intent_id, now)
            conn.execute(
                """
                UPDATE order_intents
                SET state = 'SUBMITTED', broker_order_id = ?, pending_operation = NULL, pending_owner = NULL,
                    pending_fence = NULL, last_reconciled_run = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (evidence.broker_order_id, self.run_id, now, intent_id),
            )
            self._append_reconciliation_event(conn, intent_id, "SUBMISSION_UNKNOWN", "SUBMITTED", "POST_ACKNOWLEDGED", "POST_ACKNOWLEDGED", evidence, now)
            return self._intent_from_row(self._require_intent(conn, intent_id))

    def record_post_unknown(self, intent_id: str, owner: str, fencing_token: int, reason_code: str) -> IntentRecord:
        """Append the precise failed POST outcome; state remains reconciliation-only."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        if type(reason_code) is not str or reason_code not in _POST_UNKNOWN_REASONS:
            raise OrderIntentValidationError("reason_code must be an allowlisted ambiguous POST failure")
        now = self._now_us()
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["state"] != "SUBMISSION_UNKNOWN" or intent["pending_operation"] != "SUBMIT" or intent["pending_owner"] != owner or int(intent["pending_fence"] or -1) != fencing_token:
                raise OrderIntentTransitionError("ambiguous POST result requires SUBMISSION_UNKNOWN")
            self._append_event(conn, intent_id, "POST_UNKNOWN", "SUBMISSION_UNKNOWN", "SUBMISSION_UNKNOWN", "broker-post", reason_code, now)
            return self._intent_from_row(intent)

    def mark_pre_post_failed(self, intent_id: str, owner: str, fencing_token: int, reason_code: str = "PRE_POST_ABORTED") -> IntentRecord:
        """Fail only before POST; after ``begin_submission`` this operation is forbidden."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        if type(reason_code) is not str or reason_code not in _PRE_POST_FAILURE_REASONS:
            raise OrderIntentValidationError("reason_code must be an allowlisted pre-POST failure")
        now = self._now_us()
        record: IntentRecord | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if self._claim_expired(intent, now):
                self._mark_claim_in_doubt(conn, intent, now)
                expired = True
            else:
                self._require_submission_fence(intent, owner, fencing_token)
                conn.execute(
                    """
                    UPDATE order_intents
                    SET state = 'FAILED', submission_lease_owner = NULL, submission_lease_expires_at = NULL,
                        last_reconciled_run = ?, updated_at = ? WHERE intent_id = ?
                    """,
                    (self.run_id, now, intent_id),
                )
                self._release_reservation(conn, intent_id, "PRE_POST_ABORTED", now)
                self._append_event(conn, intent_id, "PRE_POST_FAILED", "CLAIMED", "FAILED", owner, reason_code, now)
                record = self._intent_from_row(self._require_intent(conn, intent_id))
        if expired:
            raise OrderIntentReconciliationRequired("expired claim may have posted; it cannot be FAILED")
        assert record is not None
        return record

    def reconcile_open(self, intent_id: str, evidence: BrokerEvidence) -> IntentRecord:
        now = self._now_us()
        if type(evidence) is not BrokerEvidence:
            raise OrderIntentValidationError("open reconciliation requires exact BrokerEvidence")
        BrokerEvidence.validate(evidence, _from_us(now))
        if evidence.outcome != "OPEN" or evidence.operation not in {"ORDER_QUERY", "AMEND_QUERY"}:
            raise OrderIntentValidationError("open reconciliation requires status OPEN")
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["state"] == "SUBMISSION_UNKNOWN":
                self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"ORDER_QUERY"}, allowed_outcomes={"OPEN"})
            elif intent["state"] == "SUBMITTED" and intent["pending_operation"] == "AMEND":
                self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"AMEND_QUERY"}, allowed_outcomes={"OPEN"})
            else:
                raise OrderIntentTransitionError("open reconciliation must match the pending submission or amendment operation")
            amendment = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            self._bind_broker_order_history(conn, evidence.broker_order_id, intent_id, now)
            conn.execute(
                """
                UPDATE order_intents
                SET state = 'SUBMITTED', broker_order_id = ?, pending_operation = NULL, pending_owner = NULL,
                    pending_fence = NULL, last_reconciled_run = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (evidence.broker_order_id, self.run_id, now, intent_id),
            )
            if intent["pending_operation"] == "AMEND":
                if amendment is None:
                    raise OrderIntentIntegrityError("pending amendment is missing its durable operation")
                self._archive_amendment(conn, amendment, "RECONCILED_OPEN", now)
            conn.execute("DELETE FROM amendment_leases WHERE intent_id = ?", (intent_id,))
            self._append_reconciliation_event(conn, intent_id, intent["state"], "SUBMITTED", "BROKER_OPEN_RECONCILED", "BROKER_OPEN_RECONCILED", evidence, now)
            return self._intent_from_row(self._require_intent(conn, intent_id))

    def reconcile_terminal(
        self, intent_id: str, terminal_state: Literal["FILLED", "CANCELLED", "REJECTED", "EXPIRED"], evidence: BrokerEvidence
    ) -> IntentRecord:
        now = self._now_us()
        if type(evidence) is not BrokerEvidence:
            raise OrderIntentValidationError("terminal reconciliation requires exact BrokerEvidence")
        BrokerEvidence.validate(evidence, _from_us(now))
        if type(terminal_state) is not str or terminal_state not in _BROKER_TERMINAL_STATUS or evidence.outcome != _BROKER_TERMINAL_STATUS[terminal_state] or evidence.operation not in {"ORDER_QUERY", "AMEND_QUERY"}:
            raise OrderIntentValidationError("terminal state must match attested broker status")
        reason_code = f"BROKER_{terminal_state}"
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["state"] not in {"SUBMISSION_UNKNOWN", "SUBMITTED"}:
                raise OrderIntentTransitionError("terminal broker reconciliation requires unknown or submitted intent")
            if intent["state"] == "SUBMISSION_UNKNOWN":
                self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"ORDER_QUERY"}, allowed_outcomes={evidence.outcome})
            elif intent["pending_operation"] == "AMEND":
                self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"AMEND_QUERY"}, allowed_outcomes={evidence.outcome})
            elif intent["pending_operation"] is None:
                self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"ORDER_QUERY"}, allowed_outcomes={evidence.outcome})
            else:
                raise OrderIntentTransitionError("terminal reconciliation does not match pending operation")
            existing_id = intent["broker_order_id"]
            if existing_id is not None and existing_id != evidence.broker_order_id and intent["pending_operation"] != "AMEND":
                raise OrderIntentIntegrityError("attestation broker order id conflicts with durable intent")
            amendment = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            self._bind_broker_order_history(conn, evidence.broker_order_id, intent_id, now)
            conn.execute(
                """
                UPDATE order_intents
                SET state = ?, broker_order_id = ?, submission_lease_owner = NULL,
                    submission_lease_expires_at = NULL, pending_operation = NULL, pending_owner = NULL,
                    pending_fence = NULL, last_reconciled_run = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (terminal_state, evidence.broker_order_id, self.run_id, now, intent_id),
            )
            if intent["intent_kind"] == "OPENING":
                conn.execute("UPDATE margin_reservations SET state = 'FILLED_PENDING_ABSORPTION' WHERE intent_id = ? AND state = 'ACTIVE'", (intent_id,))
            else:
                self._release_reservation(conn, intent_id, reason_code, now)
            if intent["pending_operation"] == "AMEND":
                if amendment is None:
                    raise OrderIntentIntegrityError("pending amendment is missing its durable operation")
                self._archive_amendment(conn, amendment, "TERMINAL", now)
            conn.execute("DELETE FROM amendment_leases WHERE intent_id = ?", (intent_id,))
            self._append_reconciliation_event(conn, intent_id, intent["state"], terminal_state, "BROKER_TERMINAL_RECONCILED", reason_code, evidence, now)
            return self._intent_from_row(self._require_intent(conn, intent_id))

    def terminal_absorption_requirement(
        self,
        intent_id: str,
        terminal_order_evidence: BrokerReadEvidenceRef,
    ) -> TerminalAbsorptionRequirement:
        """Classify an exact fresh terminal read without changing risk state."""

        _validate_identity("intent_id", intent_id)
        if (
            type(terminal_order_evidence) is not BrokerReadEvidenceRef
            or terminal_order_evidence.evidence_kind != "ORDER_QUERY"
        ):
            raise OrderIntentValidationError(
                "terminal absorption requires exact ORDER_QUERY evidence"
            )
        now = self._now_us()
        with self._connection() as conn:
            requirement, _, _, _, _ = (
                self._terminal_absorption_requirement_conn(
                    conn,
                    intent_id,
                    terminal_order_evidence,
                    now,
                )
            )
        return requirement

    def absorb_terminal_reservation(
        self,
        intent_id: str,
        terminal_order_evidence: BrokerReadEvidenceRef,
        *,
        post_capacity_decision: CapacityDecisionReceipt | None = None,
    ) -> ReservationAbsorptionReceipt:
        """Release terminal reservation state only from exact durable proof."""

        _validate_identity("intent_id", intent_id)
        if (
            type(terminal_order_evidence) is not BrokerReadEvidenceRef
            or terminal_order_evidence.evidence_kind != "ORDER_QUERY"
        ):
            raise OrderIntentValidationError(
                "terminal absorption requires exact ORDER_QUERY evidence"
            )
        if (
            post_capacity_decision is not None
            and type(post_capacity_decision) is not CapacityDecisionReceipt
        ):
            raise OrderIntentValidationError(
                "post-capacity proof must use the exact decision receipt type"
            )
        now = self._now_us()
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM reservation_absorptions
                WHERE intent_id = ?
                """,
                (intent_id,),
            ).fetchone()
            if existing is not None:
                receipt = self._reservation_absorption_from_row(
                    conn, existing
                )
                requested_decision = (
                    None
                    if post_capacity_decision is None
                    else post_capacity_decision.decision_sha256
                )
                if (
                    receipt.terminal_order_evidence_sha256
                    != terminal_order_evidence.evidence_sha256
                    or receipt.post_capacity_decision_sha256
                    != requested_decision
                ):
                    raise OrderIntentIntegrityError(
                        "terminal reservation already has a conflicting absorption proof"
                    )
                return receipt

            (
                requirement,
                intent,
                reservation,
                terminal_manifest,
                terminal_result,
            ) = self._terminal_absorption_requirement_conn(
                conn,
                intent_id,
                terminal_order_evidence,
                now,
            )
            post_decision_sha256: str | None = None
            post_evidence_sha256: str | None = None
            canonical_lot_proof: list[dict[str, Any]] = []
            absorbed_margin_amount = "0"
            observed_at = int(terminal_manifest["observed_at"])
            if requirement.classification == "ZERO_FILL":
                if post_capacity_decision is not None:
                    raise OrderIntentIntegrityError(
                        "zero-fill absorption cannot be rebound to capacity evidence"
                    )
            else:
                if post_capacity_decision is None:
                    raise OrderIntentReconciliationRequired(
                        "full-fill absorption requires a newer exact capacity decision"
                    )
                (
                    post_manifest,
                    post_result,
                ) = self._verified_capacity_decision_receipt(
                    conn, post_capacity_decision, now
                )
                self._require_latest_complete_manifest_head(
                    conn, post_manifest, recorded_at_boundary=now
                )
                if (
                    post_manifest["account_id"] != intent["account_id"]
                    or post_manifest["environment"]
                    != intent["environment"]
                    or int(post_manifest["observed_at"])
                    <= int(terminal_manifest["observed_at"])
                ):
                    raise OrderIntentReconciliationRequired(
                        "post-fill capacity evidence must be newer and bound to the same account"
                    )
                if self._manifest_request_started_at(
                    conn, post_capacity_decision.evidence_sha256
                ) <= int(terminal_manifest["observed_at"]):
                    raise OrderIntentReconciliationRequired(
                        "post-fill capacity read began before terminal order evidence"
                    )
                canonical_lot_proof = _terminal_position_lot_proof(
                    post_result,
                    broker_order_id=requirement.broker_order_id,
                    fill_summary=terminal_result["fill_summary"],
                )
                post_decision_sha256 = (
                    post_capacity_decision.decision_sha256
                )
                post_evidence_sha256 = (
                    post_capacity_decision.evidence_sha256
                )
                absorbed_margin_amount = reservation["amount"]
                observed_at = int(post_manifest["observed_at"])

            lot_proof_json = _canonical_read_json(
                canonical_lot_proof
            )
            lot_proof_sha256 = _domain_bytes_hash(
                _LOT_PROOF_HASH_DOMAIN,
                lot_proof_json.encode("utf-8"),
            )
            fill_summary = terminal_result["fill_summary"]
            absorption_material = {
                "intent_id": intent_id,
                "account_id": intent["account_id"],
                "environment": intent["environment"],
                "broker_order_id": requirement.broker_order_id,
                "terminal_state": requirement.terminal_state,
                "classification": requirement.classification,
                "terminal_order_evidence_sha256":
                    terminal_order_evidence.evidence_sha256,
                "baseline_capacity_decision_sha256":
                    reservation["capacity_decision_sha256"],
                "post_capacity_decision_sha256":
                    post_decision_sha256,
                "post_capacity_evidence_sha256":
                    post_evidence_sha256,
                "ordered_quantity": requirement.ordered_quantity,
                "filled_quantity": requirement.filled_quantity,
                "placed_time_epoch_ms":
                    fill_summary["placed_time_epoch_ms"],
                "executed_time_epoch_ms":
                    fill_summary["executed_time_epoch_ms"],
                "canonical_lot_proof_json": lot_proof_json,
                "lot_proof_sha256": lot_proof_sha256,
                "absorbed_margin_amount": absorbed_margin_amount,
                "observed_at": observed_at,
                "recorded_at": now,
            }
            absorption_sha256 = _domain_json_hash(
                _RESERVATION_ABSORPTION_HASH_DOMAIN,
                absorption_material,
            )
            conn.execute(
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
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    absorption_sha256,
                    *absorption_material.values(),
                ),
            )
            release_reason = (
                "ZERO_FILL_CONFIRMED"
                if requirement.classification == "ZERO_FILL"
                else "FULL_FILL_POSITION_ABSORBED"
            )
            conn.execute(
                """
                UPDATE margin_reservations
                SET state = 'RELEASED', released_reason_code = ?,
                    released_at = ?
                WHERE intent_id = ?
                  AND state = 'FILLED_PENDING_ABSORPTION'
                """,
                (release_reason, now, intent_id),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise OrderIntentIntegrityError(
                    "terminal reservation release was not atomic"
                )
            self._append_event(
                conn,
                intent_id,
                (
                    "RESERVATION_RELEASED"
                    if requirement.classification == "ZERO_FILL"
                    else "FILLED_ABSORBED"
                ),
                intent["state"],
                intent["state"],
                "broker-evidence",
                (
                    "RESERVATION_RELEASED"
                    if requirement.classification == "ZERO_FILL"
                    else "FILLED_ABSORBED"
                ),
                now,
                broker_status=intent["state"],
                broker_order_id=requirement.broker_order_id,
                observed_at=int(terminal_manifest["observed_at"]),
                evidence_operation="ORDER_QUERY",
                broker_read_evidence_sha256=(
                    terminal_order_evidence.evidence_sha256
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM reservation_absorptions
                WHERE intent_id = ?
                """,
                (intent_id,),
            ).fetchone()
            return self._reservation_absorption_from_row(conn, row)

    def absorb_filled_reservation(
        self, intent_id: str, evidence: AccountCapacityEvidence
    ) -> MarginReservation:
        """Retained fail-closed compatibility shim for pre-R7e callers."""

        _validate_identity("intent_id", intent_id)
        if type(evidence) is not AccountCapacityEvidence:
            raise OrderIntentValidationError(
                "absorb_filled_reservation requires typed AccountCapacityEvidence"
            )
        AccountCapacityEvidence.validate(
            evidence, _from_us(self._now_us())
        )
        raise OrderIntentReconciliationRequired(
            "use durable terminal order and schema-v2 capacity evidence"
        )

    def reconciliation_blockers(self, account_id: str, environment: str) -> tuple[IntentRecord, ...]:
        _validate_identity("account_id", account_id)
        _validate_identity("environment", environment)
        now = self._now_us()
        with self._transaction() as conn:
            self._expire_claimed_leases(conn, account_id, environment, now)
            self._expire_amendment_leases(conn, account_id, environment, now)
            rows = self._blocker_rows(conn, account_id, environment)
            return tuple(self._intent_from_row(row) for row in rows)

    def mark_reconciled(self, intent_id: str, evidence: BrokerEvidence) -> IntentRecord:
        """Acknowledge that a known submitted order was observed OPEN this run."""
        now = self._now_us()
        if type(evidence) is not BrokerEvidence:
            raise OrderIntentValidationError("submitted reconciliation requires exact BrokerEvidence")
        BrokerEvidence.validate(evidence, _from_us(now))
        if evidence.outcome != "OPEN" or evidence.operation not in {"ORDER_QUERY", "AMEND_QUERY"}:
            raise OrderIntentValidationError("submitted acknowledgement requires OPEN attestation")
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"ORDER_QUERY", "AMEND_QUERY"}, allowed_outcomes={"OPEN"})
            if intent["pending_operation"] is not None:
                raise OrderIntentTransitionError("pending submission or amendment must resolve through its matching operation")
            if intent["state"] != "SUBMITTED" or intent["broker_order_id"] != evidence.broker_order_id:
                raise OrderIntentIntegrityError("attestation does not match a submitted durable order")
            conn.execute("UPDATE order_intents SET last_reconciled_run = ?, updated_at = ? WHERE intent_id = ?", (self.run_id, now, intent_id))
            self._append_reconciliation_event(conn, intent_id, "SUBMITTED", "SUBMITTED", "BROKER_OPEN_RECONCILED", "BROKER_OPEN_RECONCILED", evidence, now)
            return self._intent_from_row(self._require_intent(conn, intent_id))

    def acquire_amendment_lease(self, intent_id: str, owner: str, *, lease_seconds: float, idempotency_key: str, amendment_payload: Mapping[str, Any]) -> AmendmentLease:
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_identity("amendment_idempotency_key", idempotency_key)
        amendment_wire_payload = wire_order_payload(normalize_order_payload(amendment_payload))
        amendment_canonical_payload = canonical_order_payload(json.loads(amendment_wire_payload))
        amendment_payload_hash = _payload_hash(amendment_canonical_payload)
        now = self._now_us()
        expires_at = self._lease_expiry_us(lease_seconds, now)
        lease: AmendmentLease | None = None
        failure: Exception | None = None
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            self._expire_claimed_leases(conn, intent["account_id"], intent["environment"], now)
            self._expire_amendment_leases(conn, intent["account_id"], intent["environment"], now)
            if self._blocker_rows(conn, intent["account_id"], intent["environment"], exclude_intent_id=intent_id) or self._amendment_blocker_rows(conn, intent["account_id"], intent["environment"], exclude_intent_id=intent_id):
                failure = OrderIntentReconciliationRequired("account/environment has unresolved broker intents")
            else:
                intent = self._require_intent(conn, intent_id)
                if intent["intent_kind"] == "CLOSING":
                    failure = OrderIntentReservationError(
                        "closing amendment requires typed position and open-order capacity evidence"
                    )
                elif intent["state"] != "SUBMITTED" or not intent["broker_order_id"] or intent["last_reconciled_run"] != self.run_id:
                    failure = OrderIntentReconciliationRequired("amendment needs a current reconciled submitted broker order")
                else:
                    _validate_reprice_only_amendment(intent["wire_payload"], amendment_wire_payload)
                    if intent["intent_kind"] == "OPENING":
                        self._require_opening_amendment_reservation(conn, intent, amendment_wire_payload, now)
                    completed = conn.execute(
                        "SELECT * FROM amendment_history WHERE intent_id = ? AND idempotency_key = ?",
                        (intent_id, idempotency_key),
                    ).fetchone()
                    if completed is not None:
                        if (
                            completed["canonical_payload"] != amendment_canonical_payload
                            or completed["payload_hash"] != amendment_payload_hash
                        ):
                            failure = OrderIntentIntegrityError(
                                "completed amendment idempotency key cannot be rebound to another payload"
                            )
                        else:
                            failure = OrderIntentTransitionError(
                                "completed amendment idempotency key cannot be posted again"
                            )
                    else:
                        existing = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
                        if existing is not None and existing["owner"] != owner and int(existing["expires_at"]) > now:
                            failure = OrderIntentLeaseConflict("another worker owns the amendment lease")
                        else:
                            if existing is not None and (existing["idempotency_key"] != idempotency_key or existing["canonical_payload"] != amendment_canonical_payload or existing["broker_order_id"] != intent["broker_order_id"]):
                                failure = OrderIntentIntegrityError("amendment retry must use the same target, idempotency key, and payload")
                            if failure is not None:
                                pass
                            else:
                                amendment_fence = int(intent["amendment_fence"]) + 1
                                amendment_client_id = _stable_amendment_client_order_id(intent, idempotency_key)
                                self._ensure_amendment_client_id_is_unambiguous(conn, amendment_client_id, intent_id, idempotency_key)
                                conn.execute(
                                    "UPDATE order_intents SET amendment_fence = ?, updated_at = ? WHERE intent_id = ?",
                                    (amendment_fence, now, intent_id),
                                )
                                conn.execute(
                                    """
                                    INSERT INTO amendment_leases (intent_id, broker_order_id, client_order_id, idempotency_key, wire_payload, canonical_payload, payload_hash, owner, fencing_token, state, expires_at, updated_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'LEASED', ?, ?)
                                    ON CONFLICT(intent_id) DO UPDATE SET owner = excluded.owner, fencing_token = excluded.fencing_token, state = 'LEASED', expires_at = excluded.expires_at, updated_at = excluded.updated_at
                                    """,
                                    (intent_id, intent["broker_order_id"], amendment_client_id, idempotency_key, amendment_wire_payload, amendment_canonical_payload, amendment_payload_hash, owner, amendment_fence, expires_at, now),
                                )
                                self._append_event(conn, intent_id, "AMENDMENT_LEASE_ACQUIRED", "SUBMITTED", "SUBMITTED", owner, "AMENDMENT_LEASE_ACQUIRED", now, broker_order_id=intent["broker_order_id"])
                                lease = AmendmentLease(intent_id, intent["broker_order_id"], amendment_client_id, owner, amendment_fence, _from_us(expires_at))
        if failure is not None:
            raise failure
        assert lease is not None
        return lease

    def release_amendment_lease(self, intent_id: str, owner: str, fencing_token: int) -> None:
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        payload: dict[str, Any] | None = None
        expired = False
        with self._transaction() as conn:
            existing = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            if existing is None:
                return
            if existing["owner"] != owner or int(existing["fencing_token"]) != fencing_token:
                raise OrderIntentLeaseConflict("amendment lease owner or fencing token does not match")
            if existing["state"] != "LEASED":
                raise OrderIntentReconciliationRequired("an in-doubt amendment must be reconciled, not released")
            conn.execute("DELETE FROM amendment_leases WHERE intent_id = ?", (intent_id,))
            intent = self._require_intent(conn, intent_id)
            self._append_event(conn, intent_id, "AMENDMENT_LEASE_RELEASED", "SUBMITTED", "SUBMITTED", owner, "AMENDMENT_LEASE_RELEASED", now, broker_order_id=intent["broker_order_id"])

    def begin_amendment(
        self, intent_id: str, owner: str, fencing_token: int, authorization: OutboundAuthorization
    ) -> AmendmentLease:
        """Durably authorize exact bytes before the broker change-order POST."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        lease: AmendmentLease | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            amendment = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            if intent["intent_kind"] == "CLOSING":
                raise OrderIntentReservationError(
                    "closing amendment requires typed position and open-order capacity evidence"
                )
            if amendment is None or amendment["state"] != "LEASED" or amendment["owner"] != owner or int(amendment["fencing_token"]) != fencing_token:
                raise OrderIntentLeaseConflict("amendment owner or fencing token does not match")
            if int(amendment["expires_at"]) <= now:
                self._mark_amendment_in_doubt(conn, intent, amendment, now)
                expired = True
            else:
                payload = json.loads(amendment["wire_payload"])
                if not isinstance(payload, dict):
                    raise OrderIntentIntegrityError("stored amendment wire payload is not an object")
                payload["client_order_id"] = amendment["client_order_id"]
                validated_authorization = self._require_outbound_authorization(
                    authorization,
                    intent_id=intent_id,
                    operation="AMEND",
                    owner=owner,
                    fencing_token=fencing_token,
                    client_order_id=amendment["client_order_id"],
                    payload=payload,
                )
                if self._blocker_rows(conn, intent["account_id"], intent["environment"], exclude_intent_id=intent_id) or self._amendment_blocker_rows(conn, intent["account_id"], intent["environment"], exclude_intent_id=intent_id):
                    raise OrderIntentReconciliationRequired("account/environment has unresolved broker work")
                if intent["intent_kind"] == "OPENING":
                    self._require_opening_amendment_reservation(conn, intent, amendment["wire_payload"], now)
                self._record_outbound_authorization(conn, validated_authorization, now)
                conn.execute("UPDATE amendment_leases SET state = 'IN_DOUBT', updated_at = ? WHERE intent_id = ?", (now, intent_id))
                conn.execute("UPDATE order_intents SET pending_operation = 'AMEND', pending_owner = ?, pending_fence = ?, last_reconciled_run = NULL, updated_at = ? WHERE intent_id = ?", (owner, fencing_token, now, intent_id))
                self._append_event(conn, intent_id, "AMENDMENT_STARTED", "SUBMITTED", "SUBMITTED", owner, "POST_STARTED", now, broker_order_id=intent["broker_order_id"])
                lease = AmendmentLease(intent_id, amendment["broker_order_id"], amendment["client_order_id"], owner, fencing_token, _from_us(amendment["expires_at"]))
        if expired:
            raise OrderIntentReconciliationRequired("expired amendment lease is in-doubt and cannot be reissued")
        assert lease is not None
        return lease

    def prepare_amendment_payload(self, intent_id: str, owner: str, fencing_token: int) -> OutboundAuthorization:
        """Return immutable, exact bytes authorized for the fenced amendment."""
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        authorization: OutboundAuthorization | None = None
        expired = False
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            if intent["intent_kind"] == "CLOSING":
                raise OrderIntentReservationError(
                    "closing amendment payload requires typed position and open-order capacity evidence"
                )
            amendment = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            if amendment is None or amendment["state"] != "LEASED" or amendment["owner"] != owner or int(amendment["fencing_token"]) != fencing_token:
                raise OrderIntentLeaseConflict("amendment payload is not leased by this fenced owner")
            if int(amendment["expires_at"]) <= now:
                intent = self._require_intent(conn, intent_id)
                self._mark_amendment_in_doubt(conn, intent, amendment, now)
                expired = True
            else:
                payload = json.loads(amendment["wire_payload"])
                if wire_order_payload(payload) != amendment["wire_payload"] or canonical_order_payload(payload) != amendment["canonical_payload"] or _payload_hash(amendment["canonical_payload"]) != amendment["payload_hash"]:
                    raise OrderIntentIntegrityError("persisted amendment payload identity does not verify")
                payload["client_order_id"] = amendment["client_order_id"]
                authorization = _outbound_authorization(
                    intent_id=intent_id,
                    operation="AMEND",
                    owner=owner,
                    fencing_token=fencing_token,
                    client_order_id=amendment["client_order_id"],
                    payload=payload,
                )
        if expired:
            raise OrderIntentReconciliationRequired("expired amendment cannot be prepared or reissued")
        assert authorization is not None
        return authorization

    def record_amendment_acknowledgement(self, intent_id: str, owner: str, fencing_token: int, evidence: BrokerEvidence) -> IntentRecord:
        _validate_identity("intent_id", intent_id)
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        now = self._now_us()
        if type(evidence) is not BrokerEvidence:
            raise OrderIntentValidationError("amendment acknowledgement requires exact BrokerEvidence")
        BrokerEvidence.validate(evidence, _from_us(now))
        with self._transaction() as conn:
            intent = self._require_intent(conn, intent_id)
            amendment = conn.execute("SELECT * FROM amendment_leases WHERE intent_id = ?", (intent_id,)).fetchone()
            self._validate_broker_evidence(conn, intent, evidence, allowed_operations={"AMEND_ACK"}, allowed_outcomes={"OPEN"})
            if amendment is None or amendment["state"] != "IN_DOUBT" or amendment["owner"] != owner or int(amendment["fencing_token"]) != fencing_token or intent["pending_operation"] != "AMEND" or intent["pending_owner"] != owner or int(intent["pending_fence"] or -1) != fencing_token:
                raise OrderIntentLeaseConflict("amendment acknowledgement is not fenced to this in-doubt amendment")
            if evidence.http_status != 200:
                raise OrderIntentValidationError("definitive amendment acknowledgement requires HTTP 200")
            self._bind_broker_order_history(conn, evidence.broker_order_id, intent_id, now)
            conn.execute("UPDATE order_intents SET broker_order_id = ?, pending_operation = NULL, pending_owner = NULL, pending_fence = NULL, last_reconciled_run = ?, updated_at = ? WHERE intent_id = ?", (evidence.broker_order_id, self.run_id, now, intent_id))
            self._archive_amendment(conn, amendment, "ACKNOWLEDGED", now)
            conn.execute("DELETE FROM amendment_leases WHERE intent_id = ?", (intent_id,))
            self._append_reconciliation_event(conn, intent_id, "SUBMITTED", "SUBMITTED", "AMENDMENT_ACKNOWLEDGED", "POST_ACKNOWLEDGED", evidence, now)
            return self._intent_from_row(self._require_intent(conn, intent_id))

    def events(self, intent_id: str) -> tuple[IntentEvent, ...]:
        _validate_identity("intent_id", intent_id)
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM order_events WHERE intent_id = ? ORDER BY sequence", (intent_id,)).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def _securely_precreate_database(self) -> None:
        # sqlite3 accepts only a pathname, not a caller-owned file descriptor.
        # We therefore fail closed: a private 0700 parent is created first and
        # every pathname (including SQLite sidecars) is opened with O_NOFOLLOW
        # and descriptor-validated before SQLite is allowed to use it.
        self._ensure_secure_parent()
        if self.path.exists() or self.path.is_symlink():
            self._validate_database_file()
            return
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            self._validate_database_file()
            return
        try:
            os.fchmod(descriptor, 0o600)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise OrderIntentValidationError("new ledger database is not owner-only")
        finally:
            os.close(descriptor)

    def _ensure_secure_parent(self) -> None:
        try:
            info = self.path.parent.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(self.path.parent, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise OrderIntentValidationError("could not create ledger parent directory") from exc
            info = self.path.parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OrderIntentValidationError("ledger parent must be a real directory")
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise OrderIntentValidationError("ledger parent must be owned by this user and owner-only")

    def _validate_database_file(self) -> None:
        self._validate_private_file(self.path, "ledger database")

    @staticmethod
    def _validate_private_file(path: Path, label: str) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise OrderIntentValidationError(f"could not safely open {label}") from exc
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise OrderIntentValidationError(f"{label} must be regular, current-user, and owner-only")
        finally:
            os.close(descriptor)

    def _secure_sqlite_sidecars(self) -> None:
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = Path(f"{self.path}{suffix}")
            if not sidecar.exists() and not sidecar.is_symlink():
                continue
            self._validate_private_file(sidecar, f"SQLite {suffix[1:]} sidecar")
            try:
                os.chmod(sidecar, 0o600)
            except OSError as exc:
                raise OrderIntentValidationError(f"could not secure SQLite {suffix[1:]} sidecar") from exc
            self._validate_private_file(sidecar, f"SQLite {suffix[1:]} sidecar")

    def _initialize_schema(self) -> None:
        with self._connection() as conn:
            metadata_before_lock = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ledger_metadata'"
            ).fetchone()
            if metadata_before_lock is None:
                mode = conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
                if str(mode).lower() != "delete":
                    raise OrderIntentLedgerError("ledger requires SQLite DELETE journaling in its private directory")
            conn.execute("BEGIN EXCLUSIVE")
            # Another process may have initialized the securely precreated file
            # while this constructor waited for the exclusive lock. Re-read the
            # catalog under that lock rather than acting on stale pre-lock state.
            metadata_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ledger_metadata'"
            ).fetchone()
            if metadata_table is None:
                conn.execute("CREATE TABLE ledger_metadata (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), schema_version INTEGER NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO ledger_metadata (singleton, schema_version) VALUES (1, ?)", (SCHEMA_VERSION,))
            metadata = conn.execute("SELECT schema_version FROM ledger_metadata WHERE singleton = 1").fetchone()
            current_schema_version = int(metadata["schema_version"])
            if current_schema_version != SCHEMA_VERSION and current_schema_version not in _MIGRATABLE_SCHEMA_VERSIONS:
                raise OrderIntentLedgerError(f"unsupported ledger schema {metadata['schema_version']}; expected {SCHEMA_VERSION}")
            self._ensure_broker_read_schema(conn)
            _execute_sql_script(
                conn,
                """
                CREATE TABLE IF NOT EXISTS order_intents (
                    intent_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    idempotency_scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    intent_kind TEXT NOT NULL CHECK (intent_kind IN ('OPENING', 'CLOSING')),
                    wire_payload TEXT NOT NULL,
                    canonical_payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    client_order_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (state IN ('INTENT', 'CLAIMED', 'SUBMISSION_UNKNOWN', 'SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FAILED')),
                    broker_order_id TEXT UNIQUE,
                    submission_fence INTEGER NOT NULL CHECK (submission_fence >= 0),
                    amendment_fence INTEGER NOT NULL DEFAULT 0 CHECK (amendment_fence >= 0),
                    submission_lease_owner TEXT,
                    submission_lease_expires_at INTEGER,
                    pending_operation TEXT CHECK (pending_operation IN ('SUBMIT', 'AMEND') OR pending_operation IS NULL),
                    pending_owner TEXT,
                    pending_fence INTEGER,
                    last_reconciled_run TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE (account_id, environment, idempotency_scope, idempotency_key),
                    CHECK ((state = 'CLAIMED') = (submission_lease_owner IS NOT NULL AND submission_lease_expires_at IS NOT NULL AND submission_fence > 0)),
                    CHECK (state NOT IN ('SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED') OR broker_order_id IS NOT NULL),
                    CHECK (state != 'FAILED' OR broker_order_id IS NULL)
                );
                CREATE TABLE IF NOT EXISTS reservation_caps (
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    cap_amount TEXT NOT NULL,
                    broker_buying_power TEXT NOT NULL,
                    risk_budget TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    portfolio_snapshot_digest TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    capacity_decision_sha256 TEXT REFERENCES capacity_decisions(capacity_decision_sha256),
                    PRIMARY KEY (account_id, environment),
                    CHECK (CAST(cap_amount AS REAL) >= 0 AND CAST(broker_buying_power AS REAL) >= 0 AND CAST(risk_budget AS REAL) >= 0),
                    CHECK (length(portfolio_snapshot_digest) = 64)
                );
                CREATE TABLE IF NOT EXISTS margin_reservations (
                    intent_id TEXT PRIMARY KEY REFERENCES order_intents(intent_id),
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    risk_decision_id TEXT NOT NULL,
                    max_loss_amount TEXT NOT NULL,
                    quote_observed_at INTEGER NOT NULL,
                    quote_digest TEXT NOT NULL,
                    portfolio_observed_at INTEGER NOT NULL,
                    portfolio_snapshot_digest TEXT NOT NULL,
                    capacity_decision_sha256 TEXT REFERENCES capacity_decisions(capacity_decision_sha256),
                    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'FILLED_PENDING_ABSORPTION', 'RELEASED')),
                    released_reason_code TEXT,
                    created_at INTEGER NOT NULL,
                    released_at INTEGER,
                    CHECK (CAST(amount AS REAL) > 0 AND CAST(max_loss_amount AS REAL) > 0),
                    CHECK (length(quote_digest) = 64 AND length(portfolio_snapshot_digest) = 64),
                    CHECK ((state IN ('ACTIVE', 'FILLED_PENDING_ABSORPTION')) = (released_reason_code IS NULL AND released_at IS NULL))
                );
                CREATE TABLE IF NOT EXISTS reservation_absorptions (
                    absorption_sha256 TEXT PRIMARY KEY
                        CHECK (length(absorption_sha256) = 64),
                    intent_id TEXT NOT NULL UNIQUE
                        REFERENCES order_intents(intent_id),
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL
                        CHECK (environment IN ('sandbox', 'production')),
                    broker_order_id TEXT NOT NULL,
                    terminal_state TEXT NOT NULL CHECK (
                        terminal_state IN (
                            'FILLED','CANCELLED','REJECTED','EXPIRED'
                        )
                    ),
                    classification TEXT NOT NULL
                        CHECK (classification IN ('ZERO_FILL','FULL_FILL')),
                    terminal_order_evidence_sha256 TEXT NOT NULL
                        REFERENCES broker_read_manifests(evidence_sha256),
                    baseline_capacity_decision_sha256 TEXT
                        REFERENCES capacity_decisions(
                            capacity_decision_sha256
                        ),
                    post_capacity_decision_sha256 TEXT
                        REFERENCES capacity_decisions(
                            capacity_decision_sha256
                        ),
                    post_capacity_evidence_sha256 TEXT
                        REFERENCES broker_read_manifests(evidence_sha256),
                    ordered_quantity INTEGER NOT NULL
                        CHECK (ordered_quantity > 0),
                    filled_quantity INTEGER NOT NULL CHECK (
                        filled_quantity >= 0
                        AND filled_quantity <= ordered_quantity
                    ),
                    placed_time_epoch_ms TEXT NOT NULL,
                    executed_time_epoch_ms TEXT,
                    canonical_lot_proof_json TEXT NOT NULL,
                    lot_proof_sha256 TEXT NOT NULL
                        CHECK (length(lot_proof_sha256) = 64),
                    absorbed_margin_amount TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    recorded_at INTEGER NOT NULL,
                    CHECK (
                        (
                            classification = 'ZERO_FILL'
                            AND terminal_state IN (
                                'CANCELLED','REJECTED','EXPIRED'
                            )
                            AND filled_quantity = 0
                            AND post_capacity_decision_sha256 IS NULL
                            AND post_capacity_evidence_sha256 IS NULL
                            AND executed_time_epoch_ms IS NULL
                            AND canonical_lot_proof_json = '[]'
                            AND absorbed_margin_amount = '0'
                        )
                        OR
                        (
                            classification = 'FULL_FILL'
                            AND terminal_state = 'FILLED'
                            AND filled_quantity = ordered_quantity
                            AND baseline_capacity_decision_sha256 IS NOT NULL
                            AND post_capacity_decision_sha256 IS NOT NULL
                            AND post_capacity_evidence_sha256 IS NOT NULL
                            AND executed_time_epoch_ms IS NOT NULL
                            AND canonical_lot_proof_json != '[]'
                            AND CAST(absorbed_margin_amount AS REAL) > 0
                        )
                    )
                );
                CREATE TABLE IF NOT EXISTS amendment_leases (
                    intent_id TEXT PRIMARY KEY REFERENCES order_intents(intent_id),
                    broker_order_id TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    wire_payload TEXT NOT NULL,
                    canonical_payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('LEASED', 'IN_DOUBT')),
                    expires_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS amendment_history (
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    idempotency_key TEXT NOT NULL,
                    target_broker_order_id TEXT NOT NULL,
                    client_order_id TEXT NOT NULL UNIQUE,
                    wire_payload TEXT NOT NULL,
                    canonical_payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    completion_state TEXT NOT NULL CHECK (completion_state IN ('ACKNOWLEDGED', 'RECONCILED_OPEN', 'TERMINAL')),
                    completed_at INTEGER NOT NULL,
                    PRIMARY KEY (intent_id, idempotency_key),
                    CHECK (length(payload_hash) = 64)
                );
                CREATE TABLE IF NOT EXISTS outbound_authorizations (
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    operation TEXT NOT NULL CHECK (operation IN ('SUBMIT', 'AMEND')),
                    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
                    client_order_id TEXT NOT NULL,
                    payload_bytes BLOB NOT NULL,
                    payload_digest TEXT NOT NULL,
                    authorized_at INTEGER NOT NULL,
                    PRIMARY KEY (intent_id, operation, fencing_token),
                    CHECK (length(payload_bytes) > 0 AND length(payload_digest) = 64)
                );
                CREATE TABLE IF NOT EXISTS transport_send_attempts (
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    authorization_operation TEXT NOT NULL CHECK (authorization_operation IN ('SUBMIT', 'AMEND')),
                    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
                    transport_operation TEXT NOT NULL CHECK (transport_operation IN ('SUBMIT_PREVIEW', 'SUBMIT_PLACE', 'AMEND_PREVIEW', 'AMEND_PLACE')),
                    owner TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    account_id_key TEXT NOT NULL,
                    institution_type TEXT NOT NULL,
                    environment TEXT NOT NULL CHECK (environment IN ('sandbox', 'production')),
                    http_method TEXT NOT NULL CHECK (http_method IN ('POST', 'PUT')),
                    route TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    target_broker_order_id TEXT,
                    preview_id TEXT,
                    authorization_payload_digest TEXT NOT NULL,
                    final_xml_bytes BLOB NOT NULL,
                    final_xml_sha256 TEXT NOT NULL,
                    claimed_at INTEGER NOT NULL,
                    PRIMARY KEY (intent_id, authorization_operation, fencing_token, transport_operation),
                    CHECK (length(authorization_payload_digest) = 64),
                    CHECK (length(final_xml_bytes) > 0 AND length(final_xml_sha256) = 64),
                    CHECK ((transport_operation LIKE 'AMEND_%') = (target_broker_order_id IS NOT NULL)),
                    CHECK ((transport_operation LIKE '%_PLACE') = (preview_id IS NOT NULL))
                );
                CREATE TABLE IF NOT EXISTS broker_preview_receipts (
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    authorization_operation TEXT NOT NULL CHECK (authorization_operation IN ('SUBMIT', 'AMEND')),
                    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL CHECK (environment IN ('sandbox', 'production')),
                    client_order_id TEXT NOT NULL,
                    target_broker_order_id TEXT,
                    authorization_payload_digest TEXT NOT NULL,
                    preview_id TEXT NOT NULL,
                    raw_response_digest TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    recorded_at INTEGER NOT NULL,
                    PRIMARY KEY (intent_id, authorization_operation, fencing_token),
                    UNIQUE (account_id, environment, preview_id),
                    CHECK (length(authorization_payload_digest) = 64),
                    CHECK (length(raw_response_digest) = 64)
                );
                CREATE TABLE IF NOT EXISTS transport_response_receipts (
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    authorization_operation TEXT NOT NULL CHECK (authorization_operation IN ('SUBMIT', 'AMEND')),
                    fencing_token INTEGER NOT NULL CHECK (fencing_token > 0),
                    transport_operation TEXT NOT NULL CHECK (transport_operation IN ('SUBMIT_PREVIEW', 'SUBMIT_PLACE', 'AMEND_PREVIEW', 'AMEND_PLACE')),
                    disposition TEXT NOT NULL CHECK (disposition IN ('ACKNOWLEDGED', 'UNKNOWN')),
                    http_status INTEGER,
                    broker_status TEXT,
                    broker_order_id TEXT,
                    preview_id TEXT,
                    messages_json TEXT NOT NULL,
                    raw_response_digest TEXT,
                    observed_at INTEGER NOT NULL,
                    unknown_reason TEXT,
                    recorded_at INTEGER NOT NULL,
                    PRIMARY KEY (intent_id, authorization_operation, fencing_token, transport_operation),
                    CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
                    CHECK (raw_response_digest IS NULL OR length(raw_response_digest) = 64),
                    CHECK ((disposition = 'UNKNOWN') = (unknown_reason IS NOT NULL))
                );
                CREATE TABLE IF NOT EXISTS order_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    account_id TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    client_order_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    actor TEXT NOT NULL,
                    reason_code TEXT NOT NULL CHECK (reason_code IN (
                        'INTENT_CREATED','SUBMISSION_CLAIMED','LEASE_RENEWED','LEASE_EXPIRED_IN_DOUBT',
                        'POST_STARTED','POST_ACKNOWLEDGED','POST_TIMEOUT','POST_TRANSPORT_ERROR','POST_RESPONSE_INVALID',
                        'PRE_POST_ABORTED','BROKER_OPEN_RECONCILED','BROKER_FILLED','BROKER_CANCELLED','BROKER_REJECTED','BROKER_EXPIRED',
                        'RESERVATION_CREATED','RESERVATION_RELEASED','FILLED_ABSORBED','AMENDMENT_LEASE_ACQUIRED','AMENDMENT_LEASE_RELEASED'
                    )),
                    broker_status TEXT,
                    broker_order_id TEXT,
                    observed_at INTEGER,
                    evidence_operation TEXT,
                    http_status INTEGER,
                    raw_response_digest TEXT,
                    broker_read_evidence_sha256 TEXT REFERENCES broker_read_manifests(evidence_sha256),
                    created_at INTEGER NOT NULL,
                    CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
                    CHECK (raw_response_digest IS NULL OR length(raw_response_digest) = 64)
                );
                CREATE INDEX IF NOT EXISTS idx_intents_account_environment_state ON order_intents(account_id, environment, state);
                CREATE INDEX IF NOT EXISTS idx_reservations_account_environment ON margin_reservations(account_id, environment, state);
                CREATE INDEX IF NOT EXISTS idx_reservation_absorptions_account
                ON reservation_absorptions(
                    account_id, environment, classification
                );
                CREATE TABLE IF NOT EXISTS broker_order_history (
                    broker_order_id TEXT PRIMARY KEY,
                    intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
                    first_seen_at INTEGER NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS prevent_order_event_update BEFORE UPDATE ON order_events BEGIN SELECT RAISE(ABORT, 'order events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_order_event_delete BEFORE DELETE ON order_events BEGIN SELECT RAISE(ABORT, 'order events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_amendment_history_update BEFORE UPDATE ON amendment_history BEGIN SELECT RAISE(ABORT, 'amendment history is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_amendment_history_delete BEFORE DELETE ON amendment_history BEGIN SELECT RAISE(ABORT, 'amendment history is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_outbound_authorization_update BEFORE UPDATE ON outbound_authorizations BEGIN SELECT RAISE(ABORT, 'outbound authorizations are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_outbound_authorization_delete BEFORE DELETE ON outbound_authorizations BEGIN SELECT RAISE(ABORT, 'outbound authorizations are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_transport_send_attempt_update BEFORE UPDATE ON transport_send_attempts BEGIN SELECT RAISE(ABORT, 'transport send attempts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_transport_send_attempt_delete BEFORE DELETE ON transport_send_attempts BEGIN SELECT RAISE(ABORT, 'transport send attempts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_broker_preview_receipt_update BEFORE UPDATE ON broker_preview_receipts BEGIN SELECT RAISE(ABORT, 'broker preview receipts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_broker_preview_receipt_delete BEFORE DELETE ON broker_preview_receipts BEGIN SELECT RAISE(ABORT, 'broker preview receipts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_transport_response_receipt_update BEFORE UPDATE ON transport_response_receipts BEGIN SELECT RAISE(ABORT, 'transport response receipts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_transport_response_receipt_delete BEFORE DELETE ON transport_response_receipts BEGIN SELECT RAISE(ABORT, 'transport response receipts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_broker_order_history_update BEFORE UPDATE ON broker_order_history BEGIN SELECT RAISE(ABORT, 'broker order history is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_broker_order_history_delete BEFORE DELETE ON broker_order_history BEGIN SELECT RAISE(ABORT, 'broker order history is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_intent_identity_mutation BEFORE UPDATE ON order_intents
                WHEN OLD.account_id != NEW.account_id OR OLD.environment != NEW.environment OR OLD.strategy_id != NEW.strategy_id
                   OR OLD.decision_id != NEW.decision_id OR OLD.idempotency_scope != NEW.idempotency_scope
                   OR OLD.idempotency_key != NEW.idempotency_key OR OLD.intent_kind != NEW.intent_kind
                   OR OLD.wire_payload != NEW.wire_payload OR OLD.canonical_payload != NEW.canonical_payload OR OLD.payload_hash != NEW.payload_hash
                   OR OLD.client_order_id != NEW.client_order_id
                BEGIN SELECT RAISE(ABORT, 'order intent identity is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS prevent_terminal_rewrite BEFORE UPDATE ON order_intents
                WHEN OLD.state IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'FAILED') AND NEW.state != OLD.state
                BEGIN SELECT RAISE(ABORT, 'terminal order intent cannot transition'); END;
                """
            )
            for definition in _REQUIRED_TRIGGER_DEFINITIONS.values():
                conn.execute(
                    definition.replace(
                        "CREATE TRIGGER ",
                        "CREATE TRIGGER IF NOT EXISTS ",
                        1,
                    )
                )
            self._migrate_legacy_opening_reservations(
                conn, current_schema_version
            )
            if current_schema_version != SCHEMA_VERSION:
                conn.execute("UPDATE ledger_metadata SET schema_version = ? WHERE singleton = 1", (SCHEMA_VERSION,))
            self._verify_schema_structure(conn)
            conn.execute("COMMIT")
            self._secure_sqlite_sidecars()

    def _migrate_legacy_opening_reservations(
        self,
        conn: sqlite3.Connection,
        source_schema_version: int,
    ) -> None:
        """Retain conservative risk for pre-reservation opening intents."""

        if source_schema_version not in {8, 9}:
            return
        rows = conn.execute(
            """
            SELECT intent.*
            FROM order_intents AS intent
            LEFT JOIN margin_reservations AS reservation
              ON reservation.intent_id = intent.intent_id
            WHERE intent.intent_kind = 'OPENING'
              AND intent.state != 'FAILED'
              AND reservation.intent_id IS NULL
            ORDER BY intent.intent_id
            """
        ).fetchall()
        for intent in rows:
            exposure = _canonical_amount(
                _opening_exposure_floor(
                    json.loads(intent["wire_payload"])
                )
            )
            migration_material = {
                "source_schema_version": source_schema_version,
                "intent_id": intent["intent_id"],
                "payload_hash": intent["payload_hash"],
            }
            quote_digest = _domain_json_hash(
                b"etrade-legacy-reservation-quote.v1\0",
                migration_material,
            )
            portfolio_digest = _domain_json_hash(
                b"etrade-legacy-reservation-portfolio.v1\0",
                migration_material,
            )
            reservation_state = (
                "FILLED_PENDING_ABSORPTION"
                if intent["state"]
                in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
                else "ACTIVE"
            )
            observed_at = int(intent["created_at"])
            conn.execute(
                """
                INSERT INTO margin_reservations (
                    intent_id, account_id, environment, amount,
                    risk_decision_id, max_loss_amount,
                    quote_observed_at, quote_digest,
                    portfolio_observed_at,
                    portfolio_snapshot_digest,
                    capacity_decision_sha256, state,
                    released_reason_code, created_at, released_at
                ) VALUES (
                    ?, ?, ?, ?, 'legacy-opening-migration', ?, ?, ?, ?,
                    ?, NULL, ?, NULL, ?, NULL
                )
                """,
                (
                    intent["intent_id"],
                    intent["account_id"],
                    intent["environment"],
                    exposure,
                    exposure,
                    observed_at,
                    quote_digest,
                    observed_at,
                    portfolio_digest,
                    reservation_state,
                    observed_at,
                ),
            )
            self._append_event(
                conn,
                intent["intent_id"],
                "RESERVATION_CREATED",
                intent["state"],
                intent["state"],
                "schema-migration",
                "RESERVATION_CREATED",
                observed_at,
            )

    @staticmethod
    def _ensure_broker_read_schema(conn: sqlite3.Connection) -> None:
        """Create schema-11 provenance tables before dependent column upgrades."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS broker_read_receipts (
                receipt_sha256 TEXT NOT NULL PRIMARY KEY
                    CHECK (length(receipt_sha256) = 64),
                read_kind TEXT NOT NULL CHECK (read_kind IN (
                    'ACCOUNT_LIST','BALANCE','PORTFOLIO_PAGE',
                    'OPEN_ORDERS_PAGE','ORDER_DETAIL'
                )),
                account_id TEXT NOT NULL,
                account_id_key TEXT NOT NULL,
                institution_type TEXT NOT NULL,
                environment TEXT NOT NULL
                    CHECK (environment IN ('sandbox', 'production')),
                origin TEXT NOT NULL,
                http_method TEXT NOT NULL CHECK (http_method = 'GET'),
                route TEXT NOT NULL,
                query_json TEXT NOT NULL,
                authorization_sha256 TEXT NOT NULL
                    CHECK (length(authorization_sha256) = 64),
                request_sha256 TEXT NOT NULL
                    CHECK (length(request_sha256) = 64),
                target_broker_order_id TEXT,
                request_started_at INTEGER NOT NULL,
                response_completed_at INTEGER NOT NULL,
                http_status INTEGER NOT NULL
                    CHECK (http_status BETWEEN 100 AND 599),
                raw_response_bytes BLOB NOT NULL,
                raw_byte_length INTEGER NOT NULL
                    CHECK (
                        raw_byte_length = length(raw_response_bytes)
                        AND raw_byte_length <= 2097152
                    ),
                raw_response_sha256 TEXT NOT NULL
                    CHECK (length(raw_response_sha256) = 64),
                parser_schema TEXT NOT NULL,
                parser_code_sha256 TEXT NOT NULL
                    CHECK (length(parser_code_sha256) = 64),
                parser_config_sha256 TEXT NOT NULL
                    CHECK (length(parser_config_sha256) = 64),
                canonical_parsed_json TEXT NOT NULL,
                canonical_parsed_sha256 TEXT NOT NULL
                    CHECK (length(canonical_parsed_sha256) = 64),
                completeness TEXT NOT NULL CHECK (
                    completeness IN ('COMPLETE','HAS_NEXT','INELIGIBLE')
                ),
                recorded_at INTEGER NOT NULL,
                CHECK (response_completed_at >= request_started_at),
                CHECK (
                    (environment = 'production'
                     AND origin = 'https://api.etrade.com')
                    OR
                    (environment = 'sandbox'
                     AND origin = 'https://apisb.etrade.com')
                ),
                CHECK (
                    (read_kind = 'ORDER_DETAIL')
                    = (target_broker_order_id IS NOT NULL)
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS broker_read_manifests (
                evidence_sha256 TEXT NOT NULL PRIMARY KEY
                    CHECK (length(evidence_sha256) = 64),
                evidence_kind TEXT NOT NULL
                    CHECK (evidence_kind IN ('CAPACITY','ORDER_QUERY')),
                account_id TEXT NOT NULL,
                account_id_key TEXT NOT NULL,
                institution_type TEXT NOT NULL,
                environment TEXT NOT NULL
                    CHECK (environment IN ('sandbox', 'production')),
                origin TEXT NOT NULL,
                target_broker_order_id TEXT,
                observed_at INTEGER NOT NULL,
                completeness TEXT NOT NULL CHECK (
                    completeness IN ('COMPLETE','INCOMPLETE','UNSTABLE')
                ),
                canonical_result_json TEXT NOT NULL,
                canonical_result_sha256 TEXT NOT NULL
                    CHECK (length(canonical_result_sha256) = 64),
                created_at INTEGER NOT NULL,
                CHECK (
                    (evidence_kind = 'ORDER_QUERY')
                    = (target_broker_order_id IS NOT NULL)
                ),
                CHECK (
                    (environment = 'production'
                     AND origin = 'https://api.etrade.com')
                    OR
                    (environment = 'sandbox'
                     AND origin = 'https://apisb.etrade.com')
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS broker_read_manifest_members (
                evidence_sha256 TEXT NOT NULL
                    REFERENCES broker_read_manifests(evidence_sha256),
                member_ordinal INTEGER NOT NULL
                    CHECK (member_ordinal >= 0),
                member_role TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL
                    REFERENCES broker_read_receipts(receipt_sha256),
                PRIMARY KEY (evidence_sha256, member_ordinal),
                UNIQUE (evidence_sha256, member_role),
                UNIQUE (evidence_sha256, receipt_sha256)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS capacity_decisions (
                capacity_decision_sha256 TEXT NOT NULL PRIMARY KEY
                    CHECK (length(capacity_decision_sha256) = 64),
                evidence_sha256 TEXT NOT NULL
                    REFERENCES broker_read_manifests(evidence_sha256),
                account_id TEXT NOT NULL,
                environment TEXT NOT NULL
                    CHECK (environment IN ('sandbox', 'production')),
                broker_buying_power TEXT NOT NULL,
                risk_budget TEXT NOT NULL,
                cap_amount TEXT NOT NULL,
                observed_at INTEGER NOT NULL,
                capacity_snapshot_sha256 TEXT NOT NULL
                    CHECK (length(capacity_snapshot_sha256) = 64),
                decided_at INTEGER NOT NULL,
                CHECK (
                    CAST(broker_buying_power AS REAL) >= 0
                    AND CAST(risk_budget AS REAL) >= 0
                    AND CAST(cap_amount AS REAL) >= 0
                )
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_broker_read_receipts_binding
            ON broker_read_receipts (
                account_id, environment, response_completed_at
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_broker_read_manifests_binding
            ON broker_read_manifests (
                account_id, environment, evidence_kind, observed_at
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_broker_read_manifest_heads
            ON broker_read_manifests (
                account_id, environment, evidence_kind,
                target_broker_order_id, completeness,
                observed_at DESC, created_at
            )
            """,
        )
        for statement in statements:
            conn.execute(statement)
        column_upgrades = (
            (
                "broker_read_receipts",
                "authorization_sha256",
                "TEXT",
            ),
            (
                "reservation_caps",
                "capacity_decision_sha256",
                "TEXT REFERENCES capacity_decisions(capacity_decision_sha256)",
            ),
            (
                "margin_reservations",
                "capacity_decision_sha256",
                "TEXT REFERENCES capacity_decisions(capacity_decision_sha256)",
            ),
            (
                "order_events",
                "broker_read_evidence_sha256",
                "TEXT REFERENCES broker_read_manifests(evidence_sha256)",
            ),
        )
        for table, column, definition in column_upgrades:
            table_exists = conn.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table,),
            ).fetchone()
            if table_exists is None:
                continue
            columns = {
                row["name"]
                for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )

    def _verify_schema_structure(self, conn: sqlite3.Connection) -> None:
        if (
            str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            != "delete"
            or int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1
        ):
            raise OrderIntentLedgerError(
                "ledger SQLite safety pragmas are not active"
            )
        required_columns = {
            "broker_read_receipts": {
                "receipt_sha256",
                "read_kind",
                "account_id",
                "account_id_key",
                "institution_type",
                "environment",
                "origin",
                "http_method",
                "route",
                "query_json",
                "authorization_sha256",
                "request_sha256",
                "target_broker_order_id",
                "request_started_at",
                "response_completed_at",
                "http_status",
                "raw_response_bytes",
                "raw_byte_length",
                "raw_response_sha256",
                "parser_schema",
                "parser_code_sha256",
                "parser_config_sha256",
                "canonical_parsed_json",
                "canonical_parsed_sha256",
                "completeness",
                "recorded_at",
            },
            "broker_read_manifests": {
                "evidence_sha256",
                "evidence_kind",
                "account_id",
                "account_id_key",
                "institution_type",
                "environment",
                "origin",
                "target_broker_order_id",
                "observed_at",
                "completeness",
                "canonical_result_json",
                "canonical_result_sha256",
                "created_at",
            },
            "broker_read_manifest_members": {
                "evidence_sha256",
                "member_ordinal",
                "member_role",
                "receipt_sha256",
            },
            "capacity_decisions": {
                "capacity_decision_sha256",
                "evidence_sha256",
                "account_id",
                "environment",
                "broker_buying_power",
                "risk_budget",
                "cap_amount",
                "observed_at",
                "capacity_snapshot_sha256",
                "decided_at",
            },
            "reservation_caps": {"capacity_decision_sha256"},
            "margin_reservations": {"capacity_decision_sha256"},
            "order_events": {"broker_read_evidence_sha256"},
            "reservation_absorptions": {
                "absorption_sha256",
                "intent_id",
                "account_id",
                "environment",
                "broker_order_id",
                "terminal_state",
                "classification",
                "terminal_order_evidence_sha256",
                "baseline_capacity_decision_sha256",
                "post_capacity_decision_sha256",
                "post_capacity_evidence_sha256",
                "ordered_quantity",
                "filled_quantity",
                "placed_time_epoch_ms",
                "executed_time_epoch_ms",
                "canonical_lot_proof_json",
                "lot_proof_sha256",
                "absorbed_margin_amount",
                "observed_at",
                "recorded_at",
            },
        }
        table_columns: dict[str, dict[str, sqlite3.Row]] = {}
        for table, required in required_columns.items():
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            columns = {row["name"]: row for row in rows}
            if not required.issubset(columns):
                raise OrderIntentLedgerError(
                    f"ledger table {table} is structurally incomplete"
                )
            table_columns[table] = columns
        for column in (
            "receipt_sha256",
            "read_kind",
            "account_id",
            "account_id_key",
            "institution_type",
            "environment",
            "origin",
            "http_method",
            "route",
            "query_json",
            "authorization_sha256",
            "request_sha256",
            "request_started_at",
            "response_completed_at",
            "http_status",
            "raw_response_bytes",
            "raw_byte_length",
            "raw_response_sha256",
            "parser_schema",
            "parser_code_sha256",
            "parser_config_sha256",
            "canonical_parsed_json",
            "canonical_parsed_sha256",
            "completeness",
            "recorded_at",
        ):
            if int(
                table_columns["broker_read_receipts"][column]["notnull"]
            ) != 1:
                raise OrderIntentLedgerError(
                    "broker read receipt schema permits missing provenance"
                )
        required_triggers = set(_REQUIRED_TRIGGER_DEFINITIONS)
        trigger_rows = conn.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'trigger'
            """
        ).fetchall()
        triggers = {row["name"]: row["sql"] for row in trigger_rows}
        if not required_triggers.issubset(triggers):
            raise OrderIntentLedgerError(
                "ledger append-only provenance triggers are incomplete"
            )
        for name, definition in _REQUIRED_TRIGGER_DEFINITIONS.items():
            if _normalized_schema_sql(str(triggers[name])) != (
                _normalized_schema_sql(definition)
            ):
                raise OrderIntentLedgerError(
                    "ledger provenance trigger definition is invalid"
                )
        required_foreign_keys = {
            ("broker_read_manifest_members", "broker_read_manifests"),
            ("broker_read_manifest_members", "broker_read_receipts"),
            ("capacity_decisions", "broker_read_manifests"),
            ("reservation_caps", "capacity_decisions"),
            ("margin_reservations", "capacity_decisions"),
            ("order_events", "broker_read_manifests"),
            ("reservation_absorptions", "order_intents"),
            ("reservation_absorptions", "broker_read_manifests"),
            ("reservation_absorptions", "capacity_decisions"),
        }
        actual_foreign_keys = {
            (table, row["table"])
            for table in required_columns
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        }
        if not required_foreign_keys.issubset(actual_foreign_keys):
            raise OrderIntentLedgerError(
                "ledger provenance foreign keys are incomplete"
            )
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise OrderIntentLedgerError(
                "ledger contains foreign-key violations"
            )
        quick_check = conn.execute("PRAGMA quick_check").fetchall()
        if [row[0] for row in quick_check] != ["ok"]:
            raise OrderIntentLedgerError(
                "ledger structural integrity check failed"
            )
        metadata = conn.execute(
            """
            SELECT singleton, schema_version FROM ledger_metadata
            ORDER BY singleton
            """
        ).fetchall()
        if (
            len(metadata) != 1
            or int(metadata[0]["singleton"]) != 1
            or int(metadata[0]["schema_version"]) != SCHEMA_VERSION
        ):
            raise OrderIntentLedgerError(
                "ledger schema metadata is inconsistent"
            )
        self._verify_durable_risk_state(conn)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_secure_parent()
        self._validate_database_file()
        conn = sqlite3.connect(str(self.path), timeout=_BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.create_function(
                "etrade_opening_exposure_floor",
                1,
                _sqlite_opening_exposure_floor,
                deterministic=True,
            )
            conn.create_function(
                "etrade_decimal_gte",
                2,
                _sqlite_decimal_gte,
                deterministic=True,
            )
            conn.create_function(
                "etrade_legacy_reservation_digest",
                4,
                _sqlite_legacy_reservation_digest,
                deterministic=True,
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA synchronous = FULL")
            self._secure_sqlite_sidecars()
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                self._secure_sqlite_sidecars()
                conn.execute("COMMIT")
                self._secure_sqlite_sidecars()

    def _expire_claimed_leases(self, conn: sqlite3.Connection, account_id: str, environment: str, now: int) -> None:
        rows = conn.execute(
            """
            SELECT * FROM order_intents WHERE account_id = ? AND environment = ?
              AND state = 'CLAIMED' AND submission_lease_expires_at <= ?
            """,
            (account_id, environment, now),
        ).fetchall()
        for row in rows:
            self._mark_claim_in_doubt(conn, row, now)

    def _expire_amendment_leases(self, conn: sqlite3.Connection, account_id: str, environment: str, now: int) -> None:
        rows = conn.execute(
            """
            SELECT amendment_leases.*, order_intents.account_id, order_intents.environment
            FROM amendment_leases JOIN order_intents USING(intent_id)
            WHERE order_intents.account_id = ? AND order_intents.environment = ?
              AND amendment_leases.state = 'LEASED' AND amendment_leases.expires_at <= ?
            """,
            (account_id, environment, now),
        ).fetchall()
        for amendment in rows:
            intent = self._require_intent(conn, amendment["intent_id"])
            self._mark_amendment_in_doubt(conn, intent, amendment, now)

    def _mark_claim_in_doubt(self, conn: sqlite3.Connection, intent: sqlite3.Row, now: int) -> None:
        if intent["state"] != "CLAIMED":
            return
        place_attempt = conn.execute(
            """
            SELECT 1 FROM transport_send_attempts
            WHERE intent_id = ? AND authorization_operation = 'SUBMIT'
              AND fencing_token = ? AND transport_operation = 'SUBMIT_PLACE'
            """,
            (intent["intent_id"], intent["submission_fence"]),
        ).fetchone()
        if place_attempt is None:
            conn.execute(
                """
                UPDATE order_intents SET state = 'INTENT',
                    submission_lease_owner = NULL,
                    submission_lease_expires_at = NULL,
                    pending_operation = NULL, pending_owner = NULL,
                    pending_fence = NULL, last_reconciled_run = NULL,
                    updated_at = ?
                WHERE intent_id = ?
                """,
                (now, intent["intent_id"]),
            )
            self._append_event(
                conn,
                intent["intent_id"],
                "PREVIEW_ONLY_EXPIRED",
                "CLAIMED",
                "INTENT",
                "system",
                "LEASE_EXPIRED_IN_DOUBT",
                now,
            )
            return
        conn.execute(
            """
            UPDATE order_intents SET state = 'SUBMISSION_UNKNOWN', submission_lease_owner = NULL,
                submission_lease_expires_at = NULL, last_reconciled_run = NULL, updated_at = ?
            WHERE intent_id = ?
            """,
            (now, intent["intent_id"]),
        )
        self._append_event(conn, intent["intent_id"], "LEASE_EXPIRED_IN_DOUBT", "CLAIMED", "SUBMISSION_UNKNOWN", "system", "LEASE_EXPIRED_IN_DOUBT", now)

    def _mark_amendment_in_doubt(self, conn: sqlite3.Connection, intent: sqlite3.Row, amendment: sqlite3.Row, now: int) -> None:
        if amendment["state"] != "LEASED":
            return
        place_attempt = conn.execute(
            """
            SELECT 1 FROM transport_send_attempts
            WHERE intent_id = ? AND authorization_operation = 'AMEND'
              AND fencing_token = ? AND transport_operation = 'AMEND_PLACE'
            """,
            (intent["intent_id"], amendment["fencing_token"]),
        ).fetchone()
        if place_attempt is None:
            conn.execute(
                "DELETE FROM amendment_leases WHERE intent_id = ?",
                (intent["intent_id"],),
            )
            self._append_event(
                conn,
                intent["intent_id"],
                "AMENDMENT_PREVIEW_ONLY_EXPIRED",
                "SUBMITTED",
                "SUBMITTED",
                "system",
                "LEASE_EXPIRED_IN_DOUBT",
                now,
                broker_order_id=intent["broker_order_id"],
            )
            return
        conn.execute("UPDATE amendment_leases SET state = 'IN_DOUBT', updated_at = ? WHERE intent_id = ?", (now, intent["intent_id"]))
        conn.execute("UPDATE order_intents SET pending_operation = 'AMEND', pending_owner = ?, pending_fence = ?, last_reconciled_run = NULL, updated_at = ? WHERE intent_id = ?", (amendment["owner"], amendment["fencing_token"], now, intent["intent_id"]))
        self._append_event(conn, intent["intent_id"], "AMENDMENT_LEASE_EXPIRED_IN_DOUBT", "SUBMITTED", "SUBMITTED", "system", "LEASE_EXPIRED_IN_DOUBT", now, broker_order_id=intent["broker_order_id"])

    def _blocker_rows(self, conn: sqlite3.Connection, account_id: str, environment: str, *, exclude_intent_id: str | None = None) -> list[sqlite3.Row]:
        params: list[Any] = [account_id, environment, self.run_id]
        exclusion = ""
        if exclude_intent_id is not None:
            exclusion = " AND intent_id != ?"
            params.append(exclude_intent_id)
        return conn.execute(
            f"""
            SELECT * FROM order_intents WHERE account_id = ? AND environment = ?
              AND state IN ('CLAIMED', 'SUBMISSION_UNKNOWN', 'SUBMITTED')
              AND (last_reconciled_run IS NULL OR last_reconciled_run != ?){exclusion}
            ORDER BY created_at, intent_id
            """,
            params,
        ).fetchall()

    @staticmethod
    def _amendment_blocker_rows(conn: sqlite3.Connection, account_id: str, environment: str, *, exclude_intent_id: str | None = None) -> list[sqlite3.Row]:
        params: list[Any] = [account_id, environment]
        exclusion = ""
        if exclude_intent_id is not None:
            exclusion = " AND amendment_leases.intent_id != ?"
            params.append(exclude_intent_id)
        return conn.execute(
            f"""
            SELECT amendment_leases.* FROM amendment_leases
            JOIN order_intents USING(intent_id)
            WHERE order_intents.account_id = ? AND order_intents.environment = ?
              AND amendment_leases.state IN ('LEASED', 'IN_DOUBT'){exclusion}
            ORDER BY amendment_leases.updated_at, amendment_leases.intent_id
            """,
            params,
        ).fetchall()

    def _require_opening_amendment_reservation(self, conn: sqlite3.Connection, intent: sqlite3.Row, amendment_wire_payload: str, now: int) -> None:
        amended_floor = _opening_exposure_floor(json.loads(amendment_wire_payload))
        self._require_active_opening_reservation(conn, intent, now)
        reservation = conn.execute(
            "SELECT amount, max_loss_amount FROM margin_reservations WHERE intent_id = ?",
            (intent["intent_id"],),
        ).fetchone()
        if reservation is None or Decimal(reservation["amount"]) < amended_floor or Decimal(reservation["max_loss_amount"]) < amended_floor:
            raise OrderIntentReservationError(
                "opening amendment increases immutable exposure beyond its active reservation"
            )

    def _require_active_opening_reservation(self, conn: sqlite3.Connection, intent: sqlite3.Row, now: int) -> None:
        reservation = conn.execute("SELECT * FROM margin_reservations WHERE intent_id = ?", (intent["intent_id"],)).fetchone()
        cap = conn.execute(
            """
            SELECT cap_amount, observed_at, portfolio_snapshot_digest,
                   capacity_decision_sha256
            FROM reservation_caps
            WHERE account_id = ? AND environment = ?
            """,
            (intent["account_id"], intent["environment"]),
        ).fetchone()
        if (
            reservation is None
            or reservation["state"] != "ACTIVE"
            or cap is None
            or reservation["capacity_decision_sha256"] is None
            or cap["capacity_decision_sha256"] is None
            or reservation["capacity_decision_sha256"]
            != cap["capacity_decision_sha256"]
        ):
            raise OrderIntentReservationError("opening submission requires an active capped margin reservation")
        if now - int(reservation["quote_observed_at"]) > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000 or now - int(reservation["portfolio_observed_at"]) > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000 or now - int(cap["observed_at"]) > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000:
            raise OrderIntentReservationError("opening submission requires fresh quote, portfolio, and capacity evidence")
        if int(cap["observed_at"]) < int(reservation["portfolio_observed_at"]):
            raise OrderIntentReservationError("capacity snapshot predates reserved portfolio evidence")
        if int(cap["observed_at"]) == int(reservation["portfolio_observed_at"]) and cap["portfolio_snapshot_digest"] != reservation["portfolio_snapshot_digest"]:
            raise OrderIntentIntegrityError("capacity snapshot digest conflicts with reserved portfolio evidence")
        pending_fill = conn.execute("SELECT 1 FROM margin_reservations WHERE account_id = ? AND environment = ? AND state = 'FILLED_PENDING_ABSORPTION'", (intent["account_id"], intent["environment"])).fetchone()
        if pending_fill is not None:
            raise OrderIntentReservationError("filled risk reservation must be absorbed by a newer portfolio snapshot before new openings")
        if self._active_reservation_total(conn, intent["account_id"], intent["environment"]) > Decimal(cap["cap_amount"]):
            raise OrderIntentReservationError("active reservations exceed account/environment cap")

    @staticmethod
    def _require_preview_receipt(
        conn: sqlite3.Connection,
        evidence: TransportRequestEvidence,
        now: int,
    ) -> None:
        preview_operation = evidence.transport_operation.replace("PLACE", "PREVIEW")
        receipt = conn.execute(
            """
            SELECT * FROM broker_preview_receipts
            WHERE intent_id = ? AND authorization_operation = ?
              AND fencing_token = ?
            """,
            (
                evidence.intent_id,
                evidence.authorization_operation,
                evidence.fencing_token,
            ),
        ).fetchone()
        if receipt is None or any(
            receipt[field] != expected
            for field, expected in (
                ("account_id", evidence.account_id),
                ("environment", evidence.environment),
                ("client_order_id", evidence.client_order_id),
                ("target_broker_order_id", evidence.target_broker_order_id),
                (
                    "authorization_payload_digest",
                    evidence.authorization_payload_digest,
                ),
                ("preview_id", evidence.preview_id),
            )
        ):
            raise OrderIntentIntegrityError(
                "place request lacks its exact broker preview receipt"
            )
        preview_attempt = conn.execute(
            """
            SELECT * FROM transport_send_attempts
            WHERE intent_id = ? AND authorization_operation = ?
              AND fencing_token = ? AND transport_operation = ?
            """,
            (
                evidence.intent_id,
                evidence.authorization_operation,
                evidence.fencing_token,
                preview_operation,
            ),
        ).fetchone()
        expected_preview_route = (
            evidence.route.removesuffix("place") + "preview"
        )
        if preview_attempt is None or any(
            preview_attempt[field] != expected
            for field, expected in (
                ("owner", evidence.owner),
                ("account_id", evidence.account_id),
                ("account_id_key", evidence.account_id_key),
                ("institution_type", evidence.institution_type),
                ("environment", evidence.environment),
                ("http_method", evidence.http_method),
                ("route", expected_preview_route),
                ("client_order_id", evidence.client_order_id),
                ("target_broker_order_id", evidence.target_broker_order_id),
                ("preview_id", None),
                (
                    "authorization_payload_digest",
                    evidence.authorization_payload_digest,
                ),
            )
        ):
            raise OrderIntentIntegrityError(
                "place request lacks its exact durable preview send attempt"
            )
        observed_at = int(receipt["observed_at"])
        if (
            observed_at > now + 5_000_000
            or now - observed_at > _PREVIEW_RECEIPT_MAX_AGE_SECONDS * 1_000_000
        ):
            raise OrderIntentReconciliationRequired(
                "broker preview receipt is stale; do not place"
            )

    def _apply_transport_ack(
        self,
        conn: sqlite3.Connection,
        request: TransportRequestEvidence,
        response: TransportResponseEvidence,
        now: int,
    ) -> None:
        if (
            response.http_status != 200
            or response.broker_order_id is None
            or response.raw_response_digest is None
            or response.broker_status not in {None, "OPEN"}
            or response.unknown_reason is not None
            or (
                response.message_codes
                and any(
                    code != 1026 or message_type != "WARNING"
                    for code, message_type in zip(
                        response.message_codes,
                        response.message_types,
                        strict=True,
                    )
                )
            )
        ):
            raise OrderIntentValidationError(
                "transport acknowledgement is not definitive"
            )
        operation = (
            "SUBMIT_ACK"
            if request.authorization_operation == "SUBMIT"
            else "AMEND_ACK"
        )
        broker_evidence = BrokerEvidence(
            account_id=request.account_id,
            environment=request.environment,
            client_order_id=request.client_order_id,
            broker_order_id=response.broker_order_id,
            operation=operation,
            outcome="OPEN",
            observed_at=response.observed_at,
            http_status=response.http_status,
            raw_response_digest=response.raw_response_digest,
        )
        BrokerEvidence.validate(broker_evidence, _from_us(now))
        intent = self._require_intent(conn, request.intent_id)
        if request.authorization_operation == "SUBMIT":
            self._validate_broker_evidence(
                conn,
                intent,
                broker_evidence,
                allowed_operations={"SUBMIT_ACK"},
                allowed_outcomes={"OPEN"},
            )
            if (
                intent["state"] != "SUBMISSION_UNKNOWN"
                or intent["pending_operation"] != "SUBMIT"
                or intent["pending_owner"] != request.owner
                or int(intent["pending_fence"] or -1) != request.fencing_token
            ):
                raise OrderIntentTransitionError(
                    "transport acknowledgement requires its pending submission"
                )
            self._bind_broker_order_history(
                conn, response.broker_order_id, request.intent_id, now
            )
            conn.execute(
                """
                UPDATE order_intents
                SET state = 'SUBMITTED', broker_order_id = ?,
                    pending_operation = NULL, pending_owner = NULL,
                    pending_fence = NULL, last_reconciled_run = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (
                    response.broker_order_id,
                    self.run_id,
                    now,
                    request.intent_id,
                ),
            )
            self._append_reconciliation_event(
                conn,
                request.intent_id,
                "SUBMISSION_UNKNOWN",
                "SUBMITTED",
                "POST_ACKNOWLEDGED",
                "POST_ACKNOWLEDGED",
                broker_evidence,
                now,
            )
            return
        amendment = conn.execute(
            "SELECT * FROM amendment_leases WHERE intent_id = ?",
            (request.intent_id,),
        ).fetchone()
        self._validate_broker_evidence(
            conn,
            intent,
            broker_evidence,
            allowed_operations={"AMEND_ACK"},
            allowed_outcomes={"OPEN"},
        )
        if (
            amendment is None
            or amendment["state"] != "IN_DOUBT"
            or amendment["owner"] != request.owner
            or int(amendment["fencing_token"]) != request.fencing_token
            or intent["pending_operation"] != "AMEND"
            or intent["pending_owner"] != request.owner
            or int(intent["pending_fence"] or -1) != request.fencing_token
        ):
            raise OrderIntentLeaseConflict(
                "transport acknowledgement is not fenced to its amendment"
            )
        self._bind_broker_order_history(
            conn, response.broker_order_id, request.intent_id, now
        )
        conn.execute(
            """
            UPDATE order_intents
            SET broker_order_id = ?, pending_operation = NULL,
                pending_owner = NULL, pending_fence = NULL,
                last_reconciled_run = ?, updated_at = ?
            WHERE intent_id = ?
            """,
            (
                response.broker_order_id,
                self.run_id,
                now,
                request.intent_id,
            ),
        )
        self._archive_amendment(conn, amendment, "ACKNOWLEDGED", now)
        conn.execute(
            "DELETE FROM amendment_leases WHERE intent_id = ?",
            (request.intent_id,),
        )
        self._append_reconciliation_event(
            conn,
            request.intent_id,
            "SUBMITTED",
            "SUBMITTED",
            "AMENDMENT_ACKNOWLEDGED",
            "POST_ACKNOWLEDGED",
            broker_evidence,
            now,
        )

    @staticmethod
    def _require_outbound_authorization(
        authorization: OutboundAuthorization,
        *,
        intent_id: str,
        operation: Literal["SUBMIT", "AMEND"],
        owner: str,
        fencing_token: int,
        client_order_id: str,
        payload: Mapping[str, Any],
    ) -> OutboundAuthorization:
        if type(authorization) is not OutboundAuthorization:
            raise OrderIntentValidationError("begin operation requires an immutable outbound authorization")
        primitive_fields = (
            (authorization.intent_id, str),
            (authorization.operation, str),
            (authorization.owner, str),
            (authorization.fencing_token, int),
            (authorization.client_order_id, str),
            (authorization.payload_bytes, bytes),
            (authorization.payload_digest, str),
        )
        if any(type(value) is not expected_type for value, expected_type in primitive_fields):
            raise OrderIntentValidationError("outbound authorization fields must use exact primitive types")
        computed_digest = hashlib.sha256(authorization.payload_bytes).hexdigest()
        if not hmac.compare_digest(authorization.payload_digest, computed_digest):
            raise OrderIntentIntegrityError("outbound authorization payload digest does not verify")
        expected = _outbound_authorization(
            intent_id=intent_id,
            operation=operation,
            owner=owner,
            fencing_token=fencing_token,
            client_order_id=client_order_id,
            payload=payload,
        )
        string_fields_match = all(
            hmac.compare_digest(actual, wanted)
            for actual, wanted in (
                (authorization.intent_id, expected.intent_id),
                (authorization.operation, expected.operation),
                (authorization.owner, expected.owner),
                (authorization.client_order_id, expected.client_order_id),
                (authorization.payload_digest, expected.payload_digest),
            )
        )
        if (
            not string_fields_match
            or authorization.fencing_token != expected.fencing_token
            or not hmac.compare_digest(authorization.payload_bytes, expected.payload_bytes)
        ):
            raise OrderIntentIntegrityError("outbound authorization does not exactly match the durable fenced payload")
        return expected

    @staticmethod
    def _record_outbound_authorization(
        conn: sqlite3.Connection, authorization: OutboundAuthorization, now: int
    ) -> None:
        existing = conn.execute(
            """
            SELECT * FROM outbound_authorizations
            WHERE intent_id = ? AND operation = ? AND fencing_token = ?
            """,
            (
                authorization.intent_id,
                authorization.operation,
                authorization.fencing_token,
            ),
        ).fetchone()
        if existing is not None:
            if any(
                existing[field] != expected
                for field, expected in (
                    ("client_order_id", authorization.client_order_id),
                    ("payload_bytes", authorization.payload_bytes),
                    ("payload_digest", authorization.payload_digest),
                )
            ):
                raise OrderIntentIntegrityError(
                    "durable outbound authorization conflicts with this fence"
                )
            return
        conn.execute(
            """
            INSERT INTO outbound_authorizations
                (intent_id, operation, fencing_token, client_order_id, payload_bytes, payload_digest, authorized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                authorization.intent_id,
                authorization.operation,
                authorization.fencing_token,
                authorization.client_order_id,
                authorization.payload_bytes,
                authorization.payload_digest,
                now,
            ),
        )

    @staticmethod
    def _require_submission_fence(intent: sqlite3.Row, owner: str, fencing_token: int) -> None:
        _validate_identity("owner", owner)
        _validate_fencing_token(fencing_token)
        if intent["state"] != "CLAIMED":
            raise OrderIntentTransitionError("submission operation requires a claimed intent")
        if intent["submission_lease_owner"] != owner or int(intent["submission_fence"]) != fencing_token:
            raise OrderIntentLeaseConflict("submission lease owner or fencing token does not match")

    @staticmethod
    def _claim_expired(intent: sqlite3.Row, now: int) -> bool:
        return intent["state"] == "CLAIMED" and int(intent["submission_lease_expires_at"]) <= now

    @staticmethod
    def _ensure_broker_order_is_unambiguous(conn: sqlite3.Connection, broker_order_id: str, intent_id: str) -> None:
        existing = conn.execute("SELECT intent_id FROM broker_order_history WHERE broker_order_id = ?", (broker_order_id,)).fetchone()
        if existing is not None and existing["intent_id"] != intent_id:
            raise OrderIntentIntegrityError("broker order id is already bound to another durable intent")

    @staticmethod
    def _bind_broker_order_history(conn: sqlite3.Connection, broker_order_id: str, intent_id: str, now: int) -> None:
        OrderIntentLedger._ensure_broker_order_is_unambiguous(conn, broker_order_id, intent_id)
        conn.execute("INSERT OR IGNORE INTO broker_order_history (broker_order_id, intent_id, first_seen_at) VALUES (?, ?, ?)", (broker_order_id, intent_id, now))

    @staticmethod
    def _archive_amendment(conn: sqlite3.Connection, amendment: sqlite3.Row, completion_state: str, now: int) -> None:
        if completion_state not in {"ACKNOWLEDGED", "RECONCILED_OPEN", "TERMINAL"}:
            raise OrderIntentValidationError("invalid amendment completion state")
        existing = conn.execute(
            "SELECT * FROM amendment_history WHERE intent_id = ? AND idempotency_key = ?",
            (amendment["intent_id"], amendment["idempotency_key"]),
        ).fetchone()
        expected = (
            amendment["broker_order_id"], amendment["client_order_id"], amendment["wire_payload"],
            amendment["canonical_payload"], amendment["payload_hash"],
        )
        if existing is not None:
            actual = (
                existing["target_broker_order_id"], existing["client_order_id"], existing["wire_payload"],
                existing["canonical_payload"], existing["payload_hash"],
            )
            if actual != expected:
                raise OrderIntentIntegrityError("amendment history identity conflict")
            return
        conn.execute(
            """
            INSERT INTO amendment_history (
                intent_id, idempotency_key, target_broker_order_id, client_order_id,
                wire_payload, canonical_payload, payload_hash, completion_state, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (amendment["intent_id"], amendment["idempotency_key"], amendment["broker_order_id"], amendment["client_order_id"], amendment["wire_payload"], amendment["canonical_payload"], amendment["payload_hash"], completion_state, now),
        )

    @staticmethod
    def _ensure_amendment_client_id_is_unambiguous(conn: sqlite3.Connection, client_order_id: str, intent_id: str, idempotency_key: str) -> None:
        original = conn.execute("SELECT intent_id FROM order_intents WHERE client_order_id = ?", (client_order_id,)).fetchone()
        if original is not None:
            raise OrderIntentIntegrityError("amendment client order id collides with an original intent")
        active = conn.execute("SELECT intent_id, idempotency_key FROM amendment_leases WHERE client_order_id = ?", (client_order_id,)).fetchone()
        if active is not None and (active["intent_id"], active["idempotency_key"]) != (intent_id, idempotency_key):
            raise OrderIntentIntegrityError("amendment client order id collides with another active amendment")
        completed = conn.execute("SELECT intent_id, idempotency_key FROM amendment_history WHERE client_order_id = ?", (client_order_id,)).fetchone()
        if completed is not None and (completed["intent_id"], completed["idempotency_key"]) != (intent_id, idempotency_key):
            raise OrderIntentIntegrityError("amendment client order id collides with completed amendment history")

    def _validate_broker_evidence(self, conn: sqlite3.Connection, intent: sqlite3.Row, evidence: BrokerEvidence, *, allowed_operations: set[str], allowed_outcomes: set[str]) -> None:
        if evidence.operation not in allowed_operations or evidence.outcome not in allowed_outcomes:
            raise OrderIntentValidationError("broker evidence operation/outcome is not valid for this transition")
        if evidence.account_id != intent["account_id"] or evidence.environment != intent["environment"]:
            raise OrderIntentIntegrityError("broker evidence account/environment does not match intent")
        expected_client_id = intent["client_order_id"]
        if evidence.operation.startswith("AMEND"):
            amendment = conn.execute("SELECT client_order_id FROM amendment_leases WHERE intent_id = ?", (intent["intent_id"],)).fetchone()
            if amendment is None:
                raise OrderIntentIntegrityError("amendment evidence has no durable amendment operation")
            expected_client_id = amendment["client_order_id"]
        if evidence.client_order_id != expected_client_id:
            raise OrderIntentIntegrityError("broker evidence client order id does not match durable intent")
        if evidence.operation in {"ORDER_QUERY", "AMEND_QUERY"}:
            if evidence.broker_read_evidence_sha256 is None:
                raise OrderIntentIntegrityError(
                    "query evidence lacks durable broker-read provenance"
                )
            manifest, result = self._verified_broker_read_manifest(
                conn, evidence.broker_read_evidence_sha256
            )
            _validate_order_query_manifest_result(result)
            if (
                manifest["evidence_kind"] != "ORDER_QUERY"
                or manifest["completeness"] != "COMPLETE"
                or manifest["account_id"] != evidence.account_id
                or manifest["environment"] != evidence.environment
                or manifest["target_broker_order_id"]
                != evidence.broker_order_id
                or int(manifest["observed_at"])
                != _to_us(evidence.observed_at)
                or result["broker_order_id"]
                != evidence.broker_order_id
                or result["outcome"] != evidence.outcome
                or result["http_status"] != evidence.http_status
                or result["raw_response_digest"]
                != evidence.raw_response_digest
                or tuple(result["order_payload_hashes"])
                != evidence.order_payload_hashes
                or result["not_found"] is not False
            ):
                raise OrderIntentIntegrityError(
                    "broker evidence conflicts with its durable read manifest"
                )
            expected_payload_hash = self._expected_order_payload_hash_conn(
                conn, intent
            )
            if expected_payload_hash not in evidence.order_payload_hashes:
                raise OrderIntentBrokerTermsMismatch(
                    "broker order terms do not match the durable intent"
                )

    @staticmethod
    def _expected_order_payload_hash_conn(
        conn: sqlite3.Connection, intent: sqlite3.Row
    ) -> str:
        intent_id = intent["intent_id"]
        if intent["pending_operation"] == "AMEND":
            amendment = conn.execute(
                "SELECT * FROM amendment_leases WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if (
                amendment is None
                or amendment["state"] != "IN_DOUBT"
                or int(amendment["fencing_token"])
                != int(intent["pending_fence"] or -1)
            ):
                raise OrderIntentIntegrityError(
                    "pending amendment lacks matching durable order terms"
                )
            result = amendment["payload_hash"]
        elif intent["pending_operation"] == "SUBMIT":
            result = intent["payload_hash"]
        else:
            completed = conn.execute(
                """
                SELECT payload_hash, completed_at
                FROM amendment_history
                WHERE intent_id = ?
                ORDER BY completed_at DESC
                """,
                (intent_id,),
            ).fetchall()
            if not completed:
                result = intent["payload_hash"]
            else:
                latest_at = int(completed[0]["completed_at"])
                latest_hashes = {
                    row["payload_hash"]
                    for row in completed
                    if int(row["completed_at"]) == latest_at
                }
                if len(latest_hashes) != 1:
                    raise OrderIntentIntegrityError(
                        "latest durable amendment terms are ambiguous"
                    )
                result = next(iter(latest_hashes))
        _validate_sha256("expected_order_payload_hash", result)
        return result

    def _active_reservation_total(
        self,
        conn: sqlite3.Connection,
        account_id: str,
        environment: str,
    ) -> Decimal:
        self._verify_durable_risk_state(
            conn, account_id=account_id, environment=environment
        )
        rows = conn.execute(
            """
            SELECT reservation.amount, reservation.state,
                   reservation.released_reason_code
            FROM margin_reservations AS reservation
            JOIN order_intents AS intent USING (intent_id)
            WHERE (
                    (
                        reservation.account_id = ?
                        AND reservation.environment = ?
                    )
                    OR
                    (
                        intent.account_id = ?
                        AND intent.environment = ?
                    )
                  )
              AND (
                    reservation.state IN (
                        'ACTIVE','FILLED_PENDING_ABSORPTION'
                    )
                    OR (
                        reservation.state = 'RELEASED'
                        AND reservation.released_reason_code =
                            'FULL_FILL_POSITION_ABSORBED'
                    )
                  )
            """,
            (account_id, environment, account_id, environment),
        ).fetchall()
        with localcontext() as decimal_context:
            decimal_context.prec = _DECIMAL_PRECISION
            return sum(
                (Decimal(row["amount"]) for row in rows), Decimal("0")
            )

    def _verify_reservation_creation_provenance(
        self,
        conn: sqlite3.Connection,
        reservation: sqlite3.Row,
        amount: Decimal,
    ) -> None:
        """Rebuild the immutable opening floor and its creation event."""

        try:
            wire_payload = json.loads(
                reservation["intent_wire_payload"]
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise OrderIntentIntegrityError(
                "reservation intent payload is not valid JSON"
            ) from exc
        if type(wire_payload) is not dict:
            raise OrderIntentIntegrityError(
                "reservation intent payload is not an object"
            )
        try:
            exposure_floor = _opening_exposure_floor(wire_payload)
        except OrderIntentLedgerError as exc:
            raise OrderIntentIntegrityError(
                "reservation intent no longer has a defensible exposure floor"
            ) from exc
        max_loss = _canonical_signed_decimal_text(
            reservation["max_loss_amount"],
            "margin reservation max loss",
        )
        if amount < exposure_floor or max_loss < exposure_floor:
            raise OrderIntentIntegrityError(
                "margin reservation understates its immutable opening exposure"
            )

        creation_events = conn.execute(
            """
            SELECT * FROM order_events
            WHERE intent_id = ?
              AND event_type = 'RESERVATION_CREATED'
            ORDER BY sequence
            """,
            (reservation["intent_id"],),
        ).fetchall()
        if len(creation_events) != 1:
            raise OrderIntentIntegrityError(
                "margin reservation lacks one creation event"
            )
        event = creation_events[0]
        if (
            event["account_id"] != reservation["account_id"]
            or event["environment"] != reservation["environment"]
            or event["client_order_id"]
            != reservation["intent_client_order_id"]
            or event["reason_code"] != "RESERVATION_CREATED"
            or int(event["created_at"])
            != int(reservation["created_at"])
            or any(
                event[field] is not None
                for field in (
                    "broker_status",
                    "broker_order_id",
                    "observed_at",
                    "evidence_operation",
                    "http_status",
                    "raw_response_digest",
                    "broker_read_evidence_sha256",
                )
            )
        ):
            raise OrderIntentIntegrityError(
                "margin reservation creation event is disconnected"
            )

        decision_sha256 = reservation["capacity_decision_sha256"]
        if decision_sha256 is None:
            matching_versions = []
            for source_version in (8, 9):
                material = {
                    "source_schema_version": source_version,
                    "intent_id": reservation["intent_id"],
                    "payload_hash": reservation["intent_payload_hash"],
                }
                if (
                    reservation["quote_digest"]
                    == _domain_json_hash(
                        b"etrade-legacy-reservation-quote.v1\0",
                        material,
                    )
                    and reservation["portfolio_snapshot_digest"]
                    == _domain_json_hash(
                        b"etrade-legacy-reservation-portfolio.v1\0",
                        material,
                    )
                ):
                    matching_versions.append(source_version)
            synthetic_migration = (
                matching_versions in ([8], [9])
                and reservation["risk_decision_id"]
                == "legacy-opening-migration"
                and amount == exposure_floor
                and max_loss == exposure_floor
                and int(reservation["quote_observed_at"])
                == int(reservation["intent_created_at"])
                and int(reservation["portfolio_observed_at"])
                == int(reservation["intent_created_at"])
                and int(reservation["created_at"])
                == int(reservation["intent_created_at"])
                and event["actor"] == "schema-migration"
                and event["from_state"] == event["to_state"]
                and event["from_state"]
                in {
                    "INTENT",
                    "CLAIMED",
                    "SUBMISSION_UNKNOWN",
                    "SUBMITTED",
                    "FILLED",
                    "CANCELLED",
                    "REJECTED",
                    "EXPIRED",
                }
            )
            historical_uncapped = (
                reservation["risk_decision_id"]
                == reservation["intent_decision_id"]
                and int(reservation["created_at"])
                >= int(reservation["intent_created_at"])
                and event["actor"] == "system"
                and event["from_state"] == "INTENT"
                and event["to_state"] == "INTENT"
            )
            if not synthetic_migration and not historical_uncapped:
                raise OrderIntentIntegrityError(
                    "uncapped reservation lacks exact legacy migration provenance"
                )
        elif (
            reservation["risk_decision_id"]
            != reservation["intent_decision_id"]
            or int(reservation["created_at"])
            < int(reservation["intent_created_at"])
            or event["actor"] != "system"
            or event["from_state"] != "INTENT"
            or event["to_state"] != "INTENT"
        ):
            raise OrderIntentIntegrityError(
                "margin reservation creation provenance changed"
            )

        state = reservation["state"]
        intent_state = reservation["intent_state"]
        if (
            state == "ACTIVE"
            and intent_state
            not in {
                "INTENT",
                "CLAIMED",
                "SUBMISSION_UNKNOWN",
                "SUBMITTED",
            }
        ) or (
            state == "FILLED_PENDING_ABSORPTION"
            and intent_state
            not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
        ):
            raise OrderIntentIntegrityError(
                "margin reservation lifecycle diverges from its intent"
            )

    @staticmethod
    def _verify_pre_post_release_provenance(
        conn: sqlite3.Connection,
        reservation: sqlite3.Row,
    ) -> None:
        if (
            reservation["intent_kind"] != "OPENING"
            or reservation["intent_state"] != "FAILED"
            or reservation["intent_broker_order_id"] is not None
            or reservation["released_reason_code"]
            != "PRE_POST_ABORTED"
            or reservation["released_at"] is None
            or int(reservation["released_at"])
            != int(reservation["intent_updated_at"])
        ):
            raise OrderIntentIntegrityError(
                "non-absorption reservation release is not a pre-post failure"
            )
        posted = conn.execute(
            """
            SELECT 1
            FROM order_events
            WHERE intent_id = ? AND event_type = 'POST_STARTED'
            UNION ALL
            SELECT 1
            FROM transport_send_attempts
            WHERE intent_id = ?
              AND transport_operation = 'SUBMIT_PLACE'
            UNION ALL
            SELECT 1
            FROM transport_response_receipts
            WHERE intent_id = ?
              AND transport_operation = 'SUBMIT_PLACE'
            LIMIT 1
            """,
            (
                reservation["intent_id"],
                reservation["intent_id"],
                reservation["intent_id"],
            ),
        ).fetchone()
        if posted is not None:
            raise OrderIntentIntegrityError(
                "pre-post reservation release has durable POST evidence"
            )
        events = conn.execute(
            """
            SELECT * FROM order_events
            WHERE intent_id = ?
              AND event_type IN (
                    'SUBMISSION_CLAIMED',
                    'RESERVATION_RELEASED',
                    'PRE_POST_FAILED'
              )
            ORDER BY sequence
            """,
            (reservation["intent_id"],),
        ).fetchall()
        claims = [
            event
            for event in events
            if event["event_type"] == "SUBMISSION_CLAIMED"
        ]
        releases = [
            event
            for event in events
            if event["event_type"] == "RESERVATION_RELEASED"
        ]
        failures = [
            event
            for event in events
            if event["event_type"] == "PRE_POST_FAILED"
        ]
        if len(claims) != 1 or len(releases) != 1 or len(failures) != 1:
            raise OrderIntentIntegrityError(
                "pre-post reservation release event chain is incomplete"
            )
        claim, release, failure = claims[0], releases[0], failures[0]
        released_at = int(reservation["released_at"])
        identity = (
            reservation["account_id"],
            reservation["environment"],
            reservation["intent_client_order_id"],
        )
        empty_evidence_fields = (
            "broker_status",
            "broker_order_id",
            "observed_at",
            "evidence_operation",
            "http_status",
            "raw_response_digest",
            "broker_read_evidence_sha256",
        )
        if (
            (
                claim["account_id"],
                claim["environment"],
                claim["client_order_id"],
            )
            != identity
            or claim["from_state"] != "INTENT"
            or claim["to_state"] != "CLAIMED"
            or claim["reason_code"] != "SUBMISSION_CLAIMED"
            or (
                release["account_id"],
                release["environment"],
                release["client_order_id"],
            )
            != identity
            or release["from_state"] != "FAILED"
            or release["to_state"] != "FAILED"
            or release["actor"] != "system"
            or release["reason_code"] != "RESERVATION_RELEASED"
            or int(release["created_at"]) != released_at
            or (
                failure["account_id"],
                failure["environment"],
                failure["client_order_id"],
            )
            != identity
            or failure["from_state"] != "CLAIMED"
            or failure["to_state"] != "FAILED"
            or failure["actor"] != claim["actor"]
            or failure["reason_code"] != "PRE_POST_ABORTED"
            or int(failure["created_at"]) != released_at
            or int(release["sequence"]) + 1
            != int(failure["sequence"])
            or int(claim["sequence"]) >= int(release["sequence"])
            or any(
                event[field] is not None
                for event in (claim, release, failure)
                for field in empty_evidence_fields
            )
        ):
            raise OrderIntentIntegrityError(
                "pre-post reservation release event semantics changed"
            )

    def _verify_durable_risk_state(
        self,
        conn: sqlite3.Connection,
        *,
        account_id: str | None = None,
        environment: str | None = None,
    ) -> None:
        """Fail closed if durable reservation and absorption state diverge."""

        if (account_id is None) != (environment is None):
            raise OrderIntentIntegrityError(
                "durable risk verification scope is incomplete"
            )
        parameters: tuple[str, ...] = ()
        reservation_scope = ""
        intent_scope = ""
        absorption_scope = ""
        cap_scope = ""
        decision_cache: dict[
            str, tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]
        ] = {}
        absorption_cache: dict[
            str, ReservationAbsorptionReceipt
        ] = {}
        if account_id is not None and environment is not None:
            parameters = (
                account_id,
                environment,
                account_id,
                environment,
            )
            reservation_scope = """
                WHERE (
                    (
                        reservation.account_id = ?
                        AND reservation.environment = ?
                    )
                    OR
                    (
                        intent.account_id = ?
                        AND intent.environment = ?
                    )
                )
            """
            intent_scope = """
                AND intent.account_id = ? AND intent.environment = ?
            """
            absorption_scope = """
                WHERE (
                    (
                        absorption.account_id = ?
                        AND absorption.environment = ?
                    )
                    OR
                    (
                        intent.account_id = ?
                        AND intent.environment = ?
                    )
                )
            """
            cap_scope = """
                WHERE (
                    (
                        cap.account_id = ?
                        AND cap.environment = ?
                    )
                    OR
                    (
                        decision.account_id = ?
                        AND decision.environment = ?
                    )
                )
            """

        missing_reservation = conn.execute(
            f"""
            SELECT intent.intent_id
            FROM order_intents AS intent
            LEFT JOIN margin_reservations AS reservation
              ON reservation.intent_id = intent.intent_id
            WHERE intent.intent_kind = 'OPENING'
              AND intent.state != 'FAILED'
              AND reservation.intent_id IS NULL
              AND (
                    intent.state != 'INTENT'
                    OR EXISTS (
                        SELECT 1
                        FROM order_events AS event
                        WHERE event.intent_id = intent.intent_id
                          AND event.event_type = 'RESERVATION_CREATED'
                    )
                  )
              {intent_scope}
            LIMIT 1
            """,
            (() if account_id is None else (account_id, environment)),
        ).fetchone()
        if missing_reservation is not None:
            raise OrderIntentIntegrityError(
                "opening intent lost its durable margin reservation"
            )

        reservations = conn.execute(
            f"""
            SELECT reservation.*, intent.account_id AS intent_account_id,
                   intent.environment AS intent_environment,
                   intent.intent_kind AS intent_kind,
                   intent.state AS intent_state,
                   intent.decision_id AS intent_decision_id,
                   intent.wire_payload AS intent_wire_payload,
                   intent.payload_hash AS intent_payload_hash,
                   intent.client_order_id AS intent_client_order_id,
                   intent.broker_order_id AS intent_broker_order_id,
                   intent.created_at AS intent_created_at,
                   intent.updated_at AS intent_updated_at
            FROM margin_reservations AS reservation
            JOIN order_intents AS intent USING (intent_id)
            {reservation_scope}
            ORDER BY reservation.intent_id
            """,
            parameters,
        ).fetchall()
        for reservation in reservations:
            if (
                reservation["intent_kind"] != "OPENING"
                or reservation["account_id"]
                != reservation["intent_account_id"]
                or reservation["environment"]
                != reservation["intent_environment"]
            ):
                raise OrderIntentIntegrityError(
                    "margin reservation identity diverges from its opening intent"
                )
            amount = _canonical_signed_decimal_text(
                reservation["amount"], "margin reservation amount"
            )
            if amount <= 0:
                raise OrderIntentIntegrityError(
                    "margin reservation amount is not positive"
                )
            self._verify_reservation_creation_provenance(
                conn, reservation, amount
            )
            decision_sha256 = reservation["capacity_decision_sha256"]
            if decision_sha256 is not None:
                decision, _manifest, _result = (
                    self._cached_capacity_decision_row(
                        conn, decision_sha256, decision_cache
                    )
                )
                if (
                    decision["account_id"] != reservation["account_id"]
                    or decision["environment"] != reservation["environment"]
                    or int(decision["observed_at"])
                    != int(reservation["portfolio_observed_at"])
                    or decision["capacity_snapshot_sha256"]
                    != reservation["portfolio_snapshot_digest"]
                ):
                    raise OrderIntentIntegrityError(
                        "margin reservation baseline decision chain diverges"
                    )
            absorption = conn.execute(
                """
                SELECT * FROM reservation_absorptions
                WHERE intent_id = ?
                """,
                (reservation["intent_id"],),
            ).fetchone()
            absorption_reason = reservation["released_reason_code"] in {
                "ZERO_FILL_CONFIRMED",
                "FULL_FILL_POSITION_ABSORBED",
            }
            if reservation["state"] == "RELEASED" and absorption_reason:
                if absorption is None:
                    raise OrderIntentIntegrityError(
                        "released terminal reservation lost its absorption receipt"
                    )
                absorption_sha256 = absorption["absorption_sha256"]
                receipt = absorption_cache.get(absorption_sha256)
                if receipt is None:
                    receipt = self._reservation_absorption_from_row(
                        conn,
                        absorption,
                        decision_cache=decision_cache,
                    )
                    absorption_cache[absorption_sha256] = receipt
                expected_classification = (
                    "ZERO_FILL"
                    if reservation["released_reason_code"]
                    == "ZERO_FILL_CONFIRMED"
                    else "FULL_FILL"
                )
                if receipt.classification != expected_classification:
                    raise OrderIntentIntegrityError(
                        "released terminal reservation has the wrong absorption class"
                    )
            elif absorption is not None:
                raise OrderIntentIntegrityError(
                    "absorption receipt is disconnected from released terminal risk"
                )
            elif reservation["state"] == "RELEASED":
                self._verify_pre_post_release_provenance(
                    conn, reservation
                )
            if (
                reservation["intent_state"] == "FILLED"
                and reservation["state"] == "RELEASED"
                and reservation["released_reason_code"]
                != "FULL_FILL_POSITION_ABSORBED"
            ):
                raise OrderIntentIntegrityError(
                    "filled opening reservation was released without retained risk"
                )

        absorptions = conn.execute(
            f"""
            SELECT absorption.*
            FROM reservation_absorptions AS absorption
            JOIN order_intents AS intent USING (intent_id)
            {absorption_scope}
            ORDER BY absorption.intent_id
            """,
            parameters,
        ).fetchall()
        for absorption in absorptions:
            absorption_sha256 = absorption["absorption_sha256"]
            if absorption_sha256 not in absorption_cache:
                absorption_cache[absorption_sha256] = (
                    self._reservation_absorption_from_row(
                        conn,
                        absorption,
                        decision_cache=decision_cache,
                    )
                )

        caps = conn.execute(
            f"""
            SELECT cap.*
            FROM reservation_caps AS cap
            LEFT JOIN capacity_decisions AS decision
              ON decision.capacity_decision_sha256 =
                    cap.capacity_decision_sha256
            {cap_scope}
            ORDER BY cap.account_id, cap.environment
            """,
            parameters,
        ).fetchall()
        for cap in caps:
            if cap["capacity_decision_sha256"] is not None:
                self._verified_reservation_cap_row(
                    conn, cap, decision_cache=decision_cache
                )

    def _cached_capacity_decision_row(
        self,
        conn: sqlite3.Connection,
        decision_sha256: str,
        cache: dict[
            str, tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]
        ] | None,
    ) -> tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]:
        if cache is None:
            return self._verified_capacity_decision_row(
                conn, decision_sha256
            )
        cached = cache.get(decision_sha256)
        if cached is None:
            cached = self._verified_capacity_decision_row(
                conn, decision_sha256
            )
            cache[decision_sha256] = cached
        return cached

    def _verified_reservation_cap_row(
        self,
        conn: sqlite3.Connection,
        cap: sqlite3.Row,
        *,
        decision_cache: dict[
            str, tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]
        ] | None = None,
    ) -> sqlite3.Row:
        decision_sha256 = cap["capacity_decision_sha256"]
        if decision_sha256 is None:
            raise OrderIntentIntegrityError(
                "reservation cap lacks a durable capacity decision"
            )
        decision, _manifest, _result = (
            self._cached_capacity_decision_row(
                conn, decision_sha256, decision_cache
            )
        )
        if (
            cap["account_id"] != decision["account_id"]
            or cap["environment"] != decision["environment"]
            or cap["cap_amount"] != decision["cap_amount"]
            or cap["broker_buying_power"]
            != decision["broker_buying_power"]
            or cap["risk_budget"] != decision["risk_budget"]
            or int(cap["observed_at"]) != int(decision["observed_at"])
            or cap["portfolio_snapshot_digest"]
            != decision["capacity_snapshot_sha256"]
        ):
            raise OrderIntentIntegrityError(
                "reservation cap conflicts with its content-addressed decision"
            )
        return decision

    def _release_reservation(self, conn: sqlite3.Connection, intent_id: str, reason_code: str, now: int) -> None:
        reservation = conn.execute("SELECT * FROM margin_reservations WHERE intent_id = ?", (intent_id,)).fetchone()
        if reservation is None or reservation["state"] != "ACTIVE":
            return
        conn.execute("UPDATE margin_reservations SET state = 'RELEASED', released_reason_code = ?, released_at = ? WHERE intent_id = ?", (reason_code, now, intent_id))
        intent = self._require_intent(conn, intent_id)
        self._append_event(conn, intent_id, "RESERVATION_RELEASED", intent["state"], intent["state"], "system", "RESERVATION_RELEASED", now, broker_order_id=intent["broker_order_id"])

    def _append_reconciliation_event(self, conn: sqlite3.Connection, intent_id: str, from_state: str, to_state: str, event_type: str, reason_code: str, evidence: BrokerEvidence, now: int) -> None:
        self._append_event(conn, intent_id, event_type, from_state, to_state, "broker-evidence", reason_code, now, broker_status=evidence.outcome, broker_order_id=evidence.broker_order_id, observed_at=_to_us(evidence.observed_at), evidence_operation=evidence.operation, http_status=evidence.http_status, raw_response_digest=evidence.raw_response_digest, broker_read_evidence_sha256=evidence.broker_read_evidence_sha256)

    def _append_event(self, conn: sqlite3.Connection, intent_id: str, event_type: str, from_state: str | None, to_state: str | None, actor: str, reason_code: str, now: int, *, broker_status: str | None = None, broker_order_id: str | None = None, observed_at: int | None = None, evidence_operation: str | None = None, http_status: int | None = None, raw_response_digest: str | None = None, broker_read_evidence_sha256: str | None = None) -> None:
        if reason_code not in _REASON_CODES:
            raise OrderIntentValidationError("reason_code is not allowlisted")
        _validate_identity("actor", actor)
        identity = conn.execute("SELECT account_id, environment, client_order_id FROM order_intents WHERE intent_id = ?", (intent_id,)).fetchone()
        if identity is None:
            raise OrderIntentValidationError("cannot append event for unknown intent")
        conn.execute(
            """
            INSERT INTO order_events (
                intent_id, account_id, environment, client_order_id,
                event_type, from_state, to_state, actor, reason_code,
                broker_status, broker_order_id, observed_at,
                evidence_operation, http_status, raw_response_digest,
                broker_read_evidence_sha256, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (intent_id, identity["account_id"], identity["environment"], identity["client_order_id"], event_type, from_state, to_state, actor, reason_code, broker_status, broker_order_id, observed_at, evidence_operation, http_status, raw_response_digest, broker_read_evidence_sha256, now),
        )

    @staticmethod
    def _verified_broker_read_manifest(
        conn: sqlite3.Connection, evidence_sha256: str
    ) -> tuple[sqlite3.Row, dict[str, Any]]:
        _validate_sha256("evidence_sha256", evidence_sha256)
        manifest = conn.execute(
            """
            SELECT * FROM broker_read_manifests
            WHERE evidence_sha256 = ?
            """,
            (evidence_sha256,),
        ).fetchone()
        if manifest is None:
            raise OrderIntentIntegrityError(
                "unknown durable broker read evidence"
            )
        members = conn.execute(
            """
            SELECT member_ordinal, member_role, receipt_sha256
            FROM broker_read_manifest_members
            WHERE evidence_sha256 = ?
            ORDER BY member_ordinal
            """,
            (evidence_sha256,),
        ).fetchall()
        if not members or tuple(
            int(row["member_ordinal"]) for row in members
        ) != tuple(range(len(members))):
            raise OrderIntentIntegrityError(
                "broker read manifest member sequence is incomplete"
            )
        member_material = []
        receipt_rows: list[sqlite3.Row] = []
        seen_roles: set[str] = set()
        seen_receipts: set[str] = set()
        for member in members:
            role = member["member_role"]
            receipt_sha256 = member["receipt_sha256"]
            if (
                type(role) is not str
                or not role
                or role in seen_roles
                or receipt_sha256 in seen_receipts
            ):
                raise OrderIntentIntegrityError(
                    "broker read manifest membership is ambiguous"
                )
            receipt = conn.execute(
                """
                SELECT * FROM broker_read_receipts
                WHERE receipt_sha256 = ?
                """,
                (receipt_sha256,),
            ).fetchone()
            if receipt is None:
                raise OrderIntentIntegrityError(
                    "broker read manifest receipt is missing"
                )
            _verify_broker_read_receipt_row(receipt)
            if (
                receipt["account_id"] != manifest["account_id"]
                or receipt["account_id_key"]
                != manifest["account_id_key"]
                or receipt["institution_type"]
                != manifest["institution_type"]
                or receipt["environment"] != manifest["environment"]
                or receipt["origin"] != manifest["origin"]
                or int(receipt["response_completed_at"])
                > int(manifest["observed_at"])
            ):
                raise OrderIntentIntegrityError(
                    "broker read receipt binding changed after persistence"
                )
            seen_roles.add(role)
            seen_receipts.add(receipt_sha256)
            receipt_rows.append(receipt)
            member_material.append(
                {
                    "ordinal": int(member["member_ordinal"]),
                    "role": role,
                    "receipt_sha256": receipt_sha256,
                }
            )
        canonical_result_json = manifest["canonical_result_json"]
        result = _load_canonical_json_object(
            canonical_result_json, "broker read manifest result"
        )
        result_sha256 = _domain_bytes_hash(
            _READ_PARSED_HASH_DOMAIN,
            canonical_result_json.encode("utf-8"),
        )
        if result_sha256 != manifest["canonical_result_sha256"]:
            raise OrderIntentIntegrityError(
                "broker read manifest parsed digest does not verify"
            )
        _validate_broker_read_manifest_semantics(
            evidence_kind=manifest["evidence_kind"],
            account_id=manifest["account_id"],
            account_id_key=manifest["account_id_key"],
            institution_type=manifest["institution_type"],
            target_broker_order_id=manifest["target_broker_order_id"],
            observed_at=int(manifest["observed_at"]),
            completeness=manifest["completeness"],
            result=result,
            member_roles=tuple(
                member["member_role"] for member in members
            ),
            receipt_rows=tuple(receipt_rows),
        )
        manifest_material = {
            "evidence_kind": manifest["evidence_kind"],
            "account_id": manifest["account_id"],
            "account_id_key": manifest["account_id_key"],
            "institution_type": manifest["institution_type"],
            "environment": manifest["environment"],
            "origin": manifest["origin"],
            "target_broker_order_id": manifest[
                "target_broker_order_id"
            ],
            "observed_at": int(manifest["observed_at"]),
            "completeness": manifest["completeness"],
            "canonical_result_sha256": result_sha256,
            "members": member_material,
            "created_at": int(manifest["created_at"]),
        }
        expected_evidence_sha256 = _domain_json_hash(
            _READ_MANIFEST_HASH_DOMAIN, manifest_material
        )
        if not hmac.compare_digest(
            expected_evidence_sha256, evidence_sha256
        ):
            raise OrderIntentIntegrityError(
                "broker read manifest content hash does not verify"
            )
        return manifest, result

    def _require_latest_complete_manifest_head(
        self,
        conn: sqlite3.Connection,
        selected: sqlite3.Row,
        *,
        recorded_at_boundary: int,
    ) -> None:
        """Require selected evidence to be the unique semantic head at a time."""

        if int(selected["created_at"]) > recorded_at_boundary:
            raise OrderIntentReconciliationRequired(
                "selected broker evidence did not exist at the decision boundary"
            )
        candidates = conn.execute(
            """
            SELECT evidence_sha256
            FROM broker_read_manifests
            WHERE evidence_kind = ?
              AND account_id = ?
              AND environment = ?
              AND completeness = 'COMPLETE'
              AND created_at <= ?
              AND observed_at >= ?
              AND (
                    (
                        ? IS NULL
                        AND target_broker_order_id IS NULL
                    )
                    OR target_broker_order_id = ?
                  )
            ORDER BY observed_at DESC, evidence_sha256
            """,
            (
                selected["evidence_kind"],
                selected["account_id"],
                selected["environment"],
                recorded_at_boundary,
                int(selected["observed_at"]),
                selected["target_broker_order_id"],
                selected["target_broker_order_id"],
            ),
        ).fetchall()
        if not candidates:
            raise OrderIntentReconciliationRequired(
                "selected broker evidence has no complete durable head"
            )
        selected_observed_at = int(selected["observed_at"])
        selected_result_sha256 = selected["canonical_result_sha256"]
        for candidate_ref in candidates:
            candidate, _result = self._verified_broker_read_manifest(
                conn, candidate_ref["evidence_sha256"]
            )
            if (
                candidate["account_id_key"]
                != selected["account_id_key"]
                or candidate["institution_type"]
                != selected["institution_type"]
                or candidate["origin"] != selected["origin"]
            ):
                raise OrderIntentReconciliationRequired(
                    "complete broker evidence heads disagree on account binding"
                )
            candidate_observed_at = int(candidate["observed_at"])
            if candidate_observed_at > selected_observed_at:
                raise OrderIntentReconciliationRequired(
                    "selected broker evidence has been superseded"
                )
            if (
                candidate_observed_at == selected_observed_at
                and candidate["canonical_result_sha256"]
                != selected_result_sha256
            ):
                raise OrderIntentReconciliationRequired(
                    "equal-time complete broker evidence heads conflict"
                )

    @staticmethod
    def _manifest_request_started_at(
        conn: sqlite3.Connection, evidence_sha256: str
    ) -> int:
        row = conn.execute(
            """
            SELECT MIN(receipt.request_started_at) AS first_started_at,
                   COUNT(*) AS receipt_count
            FROM broker_read_manifest_members AS member
            JOIN broker_read_receipts AS receipt
              ON receipt.receipt_sha256 = member.receipt_sha256
            WHERE member.evidence_sha256 = ?
            """,
            (evidence_sha256,),
        ).fetchone()
        if row is None or int(row["receipt_count"]) <= 0:
            raise OrderIntentIntegrityError(
                "broker read manifest lost its request sequence"
            )
        return int(row["first_started_at"])

    def _require_no_later_order_contradiction(
        self,
        conn: sqlite3.Connection,
        selected_manifest: sqlite3.Row,
        selected_result: dict[str, Any],
        *,
        recorded_at_boundary: int,
    ) -> None:
        semantic_keys = (
            "schema",
            "broker_order_id",
            "raw_status",
            "outcome",
            "order_payload_hashes",
            "not_found",
            "replacement_links",
            "fill_summary",
        )
        selected_semantics = {
            key: selected_result.get(key) for key in semantic_keys
        }
        rows = conn.execute(
            """
            SELECT evidence_sha256
            FROM broker_read_manifests
            WHERE evidence_kind = 'ORDER_QUERY'
              AND account_id = ?
              AND environment = ?
              AND target_broker_order_id = ?
              AND completeness = 'COMPLETE'
              AND created_at > ?
              AND observed_at >= ?
            ORDER BY observed_at, evidence_sha256
            """,
            (
                selected_manifest["account_id"],
                selected_manifest["environment"],
                selected_manifest["target_broker_order_id"],
                recorded_at_boundary,
                int(selected_manifest["observed_at"]),
            ),
        ).fetchall()
        for row in rows:
            candidate, result = self._verified_broker_read_manifest(
                conn, row["evidence_sha256"]
            )
            if (
                candidate["account_id_key"]
                != selected_manifest["account_id_key"]
                or candidate["institution_type"]
                != selected_manifest["institution_type"]
                or candidate["origin"] != selected_manifest["origin"]
                or {
                    key: result.get(key) for key in semantic_keys
                }
                != selected_semantics
            ):
                raise OrderIntentIntegrityError(
                    "later complete order evidence contradicts absorbed risk"
                )

    def _terminal_absorption_requirement_conn(
        self,
        conn: sqlite3.Connection,
        intent_id: str,
        terminal_order_evidence: BrokerReadEvidenceRef,
        now: int,
    ) -> tuple[
        TerminalAbsorptionRequirement,
        sqlite3.Row,
        sqlite3.Row,
        sqlite3.Row,
        dict[str, Any],
    ]:
        intent = self._require_intent(conn, intent_id)
        reservation = conn.execute(
            """
            SELECT * FROM margin_reservations WHERE intent_id = ?
            """,
            (intent_id,),
        ).fetchone()
        if (
            intent["intent_kind"] != "OPENING"
            or intent["state"]
            not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
            or intent["broker_order_id"] is None
            or reservation is None
        ):
            raise OrderIntentReconciliationRequired(
                "only a terminal opening intent with durable risk can be absorbed"
            )
        if reservation["state"] != "FILLED_PENDING_ABSORPTION":
            raise OrderIntentReconciliationRequired(
                "terminal reservation is not pending absorption"
            )
        if (
            reservation["account_id"] != intent["account_id"]
            or reservation["environment"] != intent["environment"]
        ):
            raise OrderIntentIntegrityError(
                "terminal reservation identity conflicts with its intent"
            )
        manifest, result = self._verified_broker_read_manifest(
            conn, terminal_order_evidence.evidence_sha256
        )
        if (
            manifest["evidence_kind"] != "ORDER_QUERY"
            or manifest["completeness"] != "COMPLETE"
            or manifest["account_id"] != intent["account_id"]
            or manifest["environment"] != intent["environment"]
            or manifest["target_broker_order_id"]
            != intent["broker_order_id"]
            or result.get("schema") != "etrade-order-query.v2"
            or result.get("broker_order_id")
            != intent["broker_order_id"]
            or result.get("not_found") is not False
            or result.get("outcome") != intent["state"]
        ):
            raise OrderIntentReconciliationRequired(
                "terminal evidence is not an exact complete schema-v2 read of this intent"
            )
        observed_at = int(manifest["observed_at"])
        if (
            observed_at > now + 5_000_000
            or now - observed_at
            > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000
        ):
            raise OrderIntentReconciliationRequired(
                "terminal absorption evidence is stale or from the future"
            )
        self._require_latest_complete_manifest_head(
            conn, manifest, recorded_at_boundary=now
        )
        replacement_links = result.get("replacement_links")
        if (
            type(replacement_links) is not dict
            or replacement_links
            != {
                "replaces_order_id": None,
                "replaced_by_order_id": None,
            }
        ):
            raise OrderIntentReconciliationRequired(
                "replacement-linked terminal orders cannot be absorbed"
            )
        expected_payload_hash = self._expected_order_payload_hash_conn(
            conn, intent
        )
        if expected_payload_hash not in result["order_payload_hashes"]:
            raise OrderIntentBrokerTermsMismatch(
                "terminal absorption order terms do not match the durable intent"
            )
        (
            classification,
            ordered_quantity,
            filled_quantity,
        ) = _terminal_fill_classification(
            intent,
            result,
            manifest_observed_at=observed_at,
        )
        if (
            classification == "FULL_FILL"
            and reservation["capacity_decision_sha256"] is None
        ):
            raise OrderIntentReconciliationRequired(
                "full-fill absorption lacks its baseline capacity decision"
            )
        requirement = TerminalAbsorptionRequirement(
            intent_id=intent_id,
            classification=classification,
            terminal_state=intent["state"],
            broker_order_id=intent["broker_order_id"],
            terminal_order_evidence_sha256=(
                terminal_order_evidence.evidence_sha256
            ),
            baseline_capacity_decision_sha256=(
                reservation["capacity_decision_sha256"]
            ),
            ordered_quantity=ordered_quantity,
            filled_quantity=filled_quantity,
            post_capacity_required=classification == "FULL_FILL",
        )
        return requirement, intent, reservation, manifest, result

    def _verified_capacity_decision_row(
        self,
        conn: sqlite3.Connection,
        decision_sha256: str,
    ) -> tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]:
        """Rebuild one historical capacity decision from its immutable source."""

        _validate_sha256("capacity decision", decision_sha256)
        row = conn.execute(
            """
            SELECT * FROM capacity_decisions
            WHERE capacity_decision_sha256 = ?
            """,
            (decision_sha256,),
        ).fetchone()
        if row is None:
            raise OrderIntentIntegrityError(
                "capacity decision is not durable"
            )
        decision_material = {
            "evidence_sha256": row["evidence_sha256"],
            "account_id": row["account_id"],
            "environment": row["environment"],
            "broker_buying_power": row["broker_buying_power"],
            "risk_budget": row["risk_budget"],
            "cap_amount": row["cap_amount"],
            "observed_at": int(row["observed_at"]),
            "capacity_snapshot_sha256":
                row["capacity_snapshot_sha256"],
            "decided_at": int(row["decided_at"]),
        }
        expected_decision_sha256 = _domain_json_hash(
            _CAPACITY_DECISION_HASH_DOMAIN, decision_material
        )
        if not hmac.compare_digest(
            expected_decision_sha256, decision_sha256
        ):
            raise OrderIntentIntegrityError(
                "capacity decision digest does not verify"
            )
        manifest, result = self._verified_broker_read_manifest(
            conn, row["evidence_sha256"]
        )
        _validate_capacity_manifest_result(result)
        buying_power = _canonical_signed_decimal_text(
            row["broker_buying_power"], "capacity decision buying power"
        )
        risk_budget = _canonical_signed_decimal_text(
            row["risk_budget"], "capacity decision risk budget"
        )
        cap_amount = _canonical_signed_decimal_text(
            row["cap_amount"], "capacity decision cap"
        )
        if (
            buying_power < 0
            or risk_budget < 0
            or cap_amount != min(buying_power, risk_budget)
            or manifest["evidence_kind"] != "CAPACITY"
            or manifest["completeness"] != "COMPLETE"
            or manifest["target_broker_order_id"] is not None
            or manifest["account_id"] != row["account_id"]
            or manifest["environment"] != row["environment"]
            or int(manifest["observed_at"]) != int(row["observed_at"])
            or int(row["decided_at"]) != int(row["observed_at"])
            or result["broker_buying_power"]
            != row["broker_buying_power"]
            or result["state_sha256"]
            != row["capacity_snapshot_sha256"]
        ):
            raise OrderIntentIntegrityError(
                "capacity decision is disconnected from its durable manifest"
            )
        return row, manifest, result

    def _verified_capacity_decision_receipt(
        self,
        conn: sqlite3.Connection,
        receipt: CapacityDecisionReceipt,
        now: int,
    ) -> tuple[sqlite3.Row, dict[str, Any]]:
        _validate_sha256(
            "post capacity decision", receipt.decision_sha256
        )
        _validate_sha256(
            "post capacity evidence", receipt.evidence_sha256
        )
        _validate_timestamp(receipt.observed_at)
        _validate_sha256(
            "post capacity snapshot",
            receipt.portfolio_snapshot_digest,
        )
        for name in (
            "cap_amount",
            "broker_buying_power",
            "risk_budget",
        ):
            value = getattr(receipt, name)
            if type(value) is not Decimal or not value.is_finite() or value < 0:
                raise OrderIntentValidationError(
                    f"{name} must be an exact non-negative Decimal"
                )
        row, manifest, result = self._verified_capacity_decision_row(
            conn, receipt.decision_sha256
        )
        expected_values = (
            receipt.evidence_sha256,
            _canonical_amount(receipt.broker_buying_power),
            _canonical_amount(receipt.risk_budget),
            _canonical_amount(receipt.cap_amount),
            _to_us(receipt.observed_at),
            receipt.portfolio_snapshot_digest,
        )
        actual_values = (
            row["evidence_sha256"],
            row["broker_buying_power"],
            row["risk_budget"],
            row["cap_amount"],
            int(row["observed_at"]),
            row["capacity_snapshot_sha256"],
        )
        if actual_values != expected_values:
            raise OrderIntentIntegrityError(
                "post capacity receipt conflicts with its durable decision"
            )
        observed_at = int(manifest["observed_at"])
        if (
            manifest["evidence_kind"] != "CAPACITY"
            or manifest["completeness"] != "COMPLETE"
            or manifest["target_broker_order_id"] is not None
            or result.get("schema") != "etrade-capacity.v2"
            or result.get("state_sha256")
            != receipt.portfolio_snapshot_digest
            or observed_at != _to_us(receipt.observed_at)
            or observed_at != int(row["observed_at"])
            or observed_at > now + 5_000_000
            or now - observed_at
            > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000
        ):
            raise OrderIntentReconciliationRequired(
                "post capacity decision lacks fresh complete schema-v2 evidence"
            )
        return manifest, result

    def _reservation_absorption_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        decision_cache: dict[
            str, tuple[sqlite3.Row, sqlite3.Row, dict[str, Any]]
        ] | None = None,
    ) -> ReservationAbsorptionReceipt:
        lot_proof = _load_canonical_json(
            row["canonical_lot_proof_json"],
            "reservation lot proof",
        )
        if type(lot_proof) is not list:
            raise OrderIntentIntegrityError(
                "reservation lot proof must be an array"
            )
        expected_lot_sha256 = _domain_bytes_hash(
            _LOT_PROOF_HASH_DOMAIN,
            row["canonical_lot_proof_json"].encode("utf-8"),
        )
        if not hmac.compare_digest(
            expected_lot_sha256, row["lot_proof_sha256"]
        ):
            raise OrderIntentIntegrityError(
                "reservation lot proof digest does not verify"
            )
        material = {
            "intent_id": row["intent_id"],
            "account_id": row["account_id"],
            "environment": row["environment"],
            "broker_order_id": row["broker_order_id"],
            "terminal_state": row["terminal_state"],
            "classification": row["classification"],
            "terminal_order_evidence_sha256":
                row["terminal_order_evidence_sha256"],
            "baseline_capacity_decision_sha256":
                row["baseline_capacity_decision_sha256"],
            "post_capacity_decision_sha256":
                row["post_capacity_decision_sha256"],
            "post_capacity_evidence_sha256":
                row["post_capacity_evidence_sha256"],
            "ordered_quantity": int(row["ordered_quantity"]),
            "filled_quantity": int(row["filled_quantity"]),
            "placed_time_epoch_ms": row["placed_time_epoch_ms"],
            "executed_time_epoch_ms": row["executed_time_epoch_ms"],
            "canonical_lot_proof_json":
                row["canonical_lot_proof_json"],
            "lot_proof_sha256": row["lot_proof_sha256"],
            "absorbed_margin_amount":
                row["absorbed_margin_amount"],
            "observed_at": int(row["observed_at"]),
            "recorded_at": int(row["recorded_at"]),
        }
        expected_absorption_sha256 = _domain_json_hash(
            _RESERVATION_ABSORPTION_HASH_DOMAIN, material
        )
        if not hmac.compare_digest(
            expected_absorption_sha256, row["absorption_sha256"]
        ):
            raise OrderIntentIntegrityError(
                "reservation absorption digest does not verify"
            )
        intent = self._require_intent(conn, row["intent_id"])
        reservation = conn.execute(
            """
            SELECT * FROM margin_reservations WHERE intent_id = ?
            """,
            (row["intent_id"],),
        ).fetchone()
        expected_reason = (
            "ZERO_FILL_CONFIRMED"
            if row["classification"] == "ZERO_FILL"
            else "FULL_FILL_POSITION_ABSORBED"
        )
        if (
            intent["account_id"] != row["account_id"]
            or intent["environment"] != row["environment"]
            or intent["broker_order_id"] != row["broker_order_id"]
            or intent["state"] != row["terminal_state"]
            or reservation is None
            or reservation["state"] != "RELEASED"
            or reservation["released_reason_code"] != expected_reason
            or reservation["capacity_decision_sha256"]
            != row["baseline_capacity_decision_sha256"]
            or (
                row["classification"] == "FULL_FILL"
                and reservation["amount"]
                != row["absorbed_margin_amount"]
            )
        ):
            raise OrderIntentIntegrityError(
                "reservation absorption is disconnected from durable intent risk"
            )
        baseline_sha256 = row["baseline_capacity_decision_sha256"]
        if baseline_sha256 is not None:
            baseline, _baseline_manifest, _baseline_result = (
                self._cached_capacity_decision_row(
                    conn, baseline_sha256, decision_cache
                )
            )
            if (
                baseline["account_id"] != row["account_id"]
                or baseline["environment"] != row["environment"]
                or baseline["capacity_snapshot_sha256"]
                != reservation["portfolio_snapshot_digest"]
                or int(baseline["observed_at"])
                != int(reservation["portfolio_observed_at"])
            ):
                raise OrderIntentIntegrityError(
                    "reservation absorption baseline capacity chain changed"
                )
        terminal_manifest, terminal_result = (
            self._verified_broker_read_manifest(
                conn, row["terminal_order_evidence_sha256"]
            )
        )
        try:
            self._require_latest_complete_manifest_head(
                conn,
                terminal_manifest,
                recorded_at_boundary=int(row["recorded_at"]),
            )
        except OrderIntentReconciliationRequired as exc:
            raise OrderIntentIntegrityError(
                "reservation absorption did not use the terminal evidence head"
            ) from exc
        if (
            terminal_manifest["evidence_kind"] != "ORDER_QUERY"
            or terminal_manifest["completeness"] != "COMPLETE"
            or terminal_result.get("schema")
            != "etrade-order-query.v2"
            or terminal_result.get("not_found") is not False
            or terminal_result.get("outcome") != intent["state"]
            or terminal_result.get("broker_order_id")
            != row["broker_order_id"]
            or terminal_result.get("replacement_links")
            != {
                "replaces_order_id": None,
                "replaced_by_order_id": None,
            }
            or self._expected_order_payload_hash_conn(conn, intent)
            not in terminal_result.get("order_payload_hashes", [])
        ):
            raise OrderIntentIntegrityError(
                "reservation absorption terminal evidence no longer proves the intent"
            )
        try:
            (
                terminal_classification,
                terminal_ordered_quantity,
                terminal_filled_quantity,
            ) = _terminal_fill_classification(
                intent,
                terminal_result,
                manifest_observed_at=int(
                    terminal_manifest["observed_at"]
                ),
            )
        except OrderIntentReconciliationRequired as exc:
            raise OrderIntentIntegrityError(
                "reservation absorption terminal proof no longer verifies"
            ) from exc
        terminal_summary = terminal_result["fill_summary"]
        if (
            terminal_classification != row["classification"]
            or terminal_ordered_quantity
            != int(row["ordered_quantity"])
            or terminal_filled_quantity
            != int(row["filled_quantity"])
            or terminal_summary["placed_time_epoch_ms"]
            != row["placed_time_epoch_ms"]
            or terminal_summary["executed_time_epoch_ms"]
            != row["executed_time_epoch_ms"]
        ):
            raise OrderIntentIntegrityError(
                "reservation absorption terminal classification changed"
            )
        if (
            terminal_manifest["account_id"] != row["account_id"]
            or terminal_manifest["environment"] != row["environment"]
            or terminal_manifest["target_broker_order_id"]
            != row["broker_order_id"]
        ):
            raise OrderIntentIntegrityError(
                "reservation absorption terminal evidence binding changed"
            )
        self._require_no_later_order_contradiction(
            conn,
            terminal_manifest,
            terminal_result,
            recorded_at_boundary=int(row["recorded_at"]),
        )
        amount = _canonical_signed_decimal_text(
            row["absorbed_margin_amount"],
            "absorbed margin amount",
        )
        if amount < 0:
            raise OrderIntentIntegrityError(
                "absorbed margin amount cannot be negative"
            )
        if row["classification"] == "ZERO_FILL":
            if (
                lot_proof
                or int(row["observed_at"])
                != int(terminal_manifest["observed_at"])
                or amount != 0
            ):
                raise OrderIntentIntegrityError(
                    "zero-fill absorption contains unsupported risk proof"
                )
        else:
            decision, post_manifest, post_result = (
                self._cached_capacity_decision_row(
                    conn,
                    row["post_capacity_decision_sha256"],
                    decision_cache,
                )
            )
            if (
                decision["evidence_sha256"]
                != row["post_capacity_evidence_sha256"]
                or decision["account_id"] != row["account_id"]
                or decision["environment"] != row["environment"]
            ):
                raise OrderIntentIntegrityError(
                    "reservation absorption post-capacity chain changed"
                )
            try:
                self._require_latest_complete_manifest_head(
                    conn,
                    post_manifest,
                    recorded_at_boundary=int(row["recorded_at"]),
                )
            except OrderIntentReconciliationRequired as exc:
                raise OrderIntentIntegrityError(
                    "reservation absorption did not use the capacity evidence head"
                ) from exc
            if (
                post_manifest["evidence_kind"] != "CAPACITY"
                or post_manifest["completeness"] != "COMPLETE"
                or post_manifest["account_id"] != row["account_id"]
                or post_manifest["environment"] != row["environment"]
                or post_result.get("schema") != "etrade-capacity.v2"
                or int(post_manifest["observed_at"])
                <= int(terminal_manifest["observed_at"])
                or int(post_manifest["observed_at"])
                != int(row["observed_at"])
                or post_result.get("state_sha256")
                != decision["capacity_snapshot_sha256"]
            ):
                raise OrderIntentIntegrityError(
                    "reservation absorption post-capacity evidence changed"
                )
            if self._manifest_request_started_at(
                conn, row["post_capacity_evidence_sha256"]
            ) <= int(terminal_manifest["observed_at"]):
                raise OrderIntentIntegrityError(
                    "reservation absorption capacity read overlapped terminal evidence"
                )
            try:
                rebuilt_lot_proof = _terminal_position_lot_proof(
                    post_result,
                    broker_order_id=row["broker_order_id"],
                    fill_summary=terminal_summary,
                )
            except OrderIntentReconciliationRequired as exc:
                raise OrderIntentIntegrityError(
                    "reservation absorption lot proof no longer verifies"
                ) from exc
            if rebuilt_lot_proof != lot_proof:
                raise OrderIntentIntegrityError(
                    "reservation absorption lot proof is not reproducible"
                )
        return ReservationAbsorptionReceipt(
            absorption_sha256=row["absorption_sha256"],
            intent_id=row["intent_id"],
            account_id=row["account_id"],
            environment=row["environment"],
            broker_order_id=row["broker_order_id"],
            terminal_state=row["terminal_state"],
            classification=row["classification"],
            terminal_order_evidence_sha256=(
                row["terminal_order_evidence_sha256"]
            ),
            baseline_capacity_decision_sha256=(
                row["baseline_capacity_decision_sha256"]
            ),
            post_capacity_decision_sha256=(
                row["post_capacity_decision_sha256"]
            ),
            post_capacity_evidence_sha256=(
                row["post_capacity_evidence_sha256"]
            ),
            ordered_quantity=int(row["ordered_quantity"]),
            filled_quantity=int(row["filled_quantity"]),
            placed_time_epoch_ms=row["placed_time_epoch_ms"],
            executed_time_epoch_ms=row["executed_time_epoch_ms"],
            canonical_lot_proof_json=(
                row["canonical_lot_proof_json"]
            ),
            lot_proof_sha256=row["lot_proof_sha256"],
            absorbed_margin_amount=Decimal(
                row["absorbed_margin_amount"]
            ),
            observed_at=_from_us(row["observed_at"]),
            recorded_at=_from_us(row["recorded_at"]),
        )

    def _require_intent(self, conn: sqlite3.Connection, intent_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM order_intents WHERE intent_id = ?", (intent_id,)).fetchone()
        if row is None:
            raise OrderIntentValidationError("unknown intent_id")
        return row

    def _now_us(self) -> int:
        value = self._clock()
        _validate_timestamp(value)
        return _to_us(value)

    @staticmethod
    def _lease_expiry_us(lease_seconds: float, now: int) -> int:
        if type(lease_seconds) not in {int, float}:
            raise OrderIntentValidationError("lease_seconds must be an exact int or float")
        seconds = float(lease_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise OrderIntentValidationError("lease_seconds must be positive and finite")
        return now + int(seconds * 1_000_000)

    @staticmethod
    def _intent_from_row(row: sqlite3.Row) -> IntentRecord:
        envelope = OrderIntent(
            account_id=row["account_id"], environment=row["environment"], strategy_id=row["strategy_id"],
            decision_id=row["decision_id"], idempotency_scope=row["idempotency_scope"],
            idempotency_key=row["idempotency_key"], intent_kind=row["intent_kind"], wire_payload=row["wire_payload"],
            canonical_payload=row["canonical_payload"], payload_hash=row["payload_hash"],
        )
        _validate_envelope(envelope)
        expected_client_id = stable_client_order_id(envelope)
        if row["client_order_id"] != expected_client_id:
            raise OrderIntentIntegrityError("stored client order id does not match immutable scope identity")
        return IntentRecord(
            intent_id=row["intent_id"], envelope=envelope, client_order_id=row["client_order_id"], state=row["state"],
            broker_order_id=row["broker_order_id"], submission_fence=int(row["submission_fence"]),
            submission_lease_owner=row["submission_lease_owner"],
            submission_lease_expires_at=_from_us(row["submission_lease_expires_at"]) if row["submission_lease_expires_at"] is not None else None,
            pending_operation=row["pending_operation"],
            pending_fence=(
                int(row["pending_fence"])
                if row["pending_fence"] is not None
                else None
            ),
            last_reconciled_run=row["last_reconciled_run"], created_at=_from_us(row["created_at"]), updated_at=_from_us(row["updated_at"]),
        )

    @staticmethod
    def _reservation_from_row(row: sqlite3.Row) -> MarginReservation:
        return MarginReservation(intent_id=row["intent_id"], account_id=row["account_id"], environment=row["environment"], amount=Decimal(row["amount"]), risk_decision_id=row["risk_decision_id"], max_loss_amount=Decimal(row["max_loss_amount"]), quote_observed_at=_from_us(row["quote_observed_at"]), quote_digest=row["quote_digest"], portfolio_observed_at=_from_us(row["portfolio_observed_at"]), portfolio_snapshot_digest=row["portfolio_snapshot_digest"], capacity_decision_sha256=row["capacity_decision_sha256"], state=row["state"], released_reason_code=row["released_reason_code"], created_at=_from_us(row["created_at"]), released_at=_from_us(row["released_at"]) if row["released_at"] is not None else None)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> IntentEvent:
        return IntentEvent(sequence=int(row["sequence"]), intent_id=row["intent_id"], account_id=row["account_id"], environment=row["environment"], client_order_id=row["client_order_id"], event_type=row["event_type"], from_state=row["from_state"], to_state=row["to_state"], actor=row["actor"], reason_code=row["reason_code"], broker_status=row["broker_status"], broker_order_id=row["broker_order_id"], observed_at=_from_us(row["observed_at"]) if row["observed_at"] is not None else None, evidence_operation=row["evidence_operation"], http_status=row["http_status"], raw_response_digest=row["raw_response_digest"], broker_read_evidence_sha256=row["broker_read_evidence_sha256"], created_at=_from_us(row["created_at"]))


def _transport_response_receipt(
    row: sqlite3.Row,
) -> TransportResponseReceipt:
    try:
        messages = json.loads(row["messages_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise OrderIntentIntegrityError(
            "stored transport response messages are invalid"
        ) from exc
    if (
        type(messages) is not list
        or any(
            type(message) is not dict
            or set(message)
            != {"code", "type", "description_sha256"}
            for message in messages
        )
        or json.dumps(
            messages, sort_keys=True, separators=(",", ":")
        )
        != row["messages_json"]
    ):
        raise OrderIntentIntegrityError(
            "stored transport response messages are not canonical"
        )
    response = TransportResponseEvidence(
        disposition=row["disposition"],
        http_status=row["http_status"],
        broker_status=row["broker_status"],
        broker_order_id=row["broker_order_id"],
        preview_id=row["preview_id"],
        message_codes=tuple(message["code"] for message in messages),
        message_types=tuple(message["type"] for message in messages),
        message_description_digests=tuple(
            message["description_sha256"] for message in messages
        ),
        raw_response_digest=row["raw_response_digest"],
        observed_at=_from_us(row["observed_at"]),
        unknown_reason=row["unknown_reason"],
    )
    _validate_transport_response_evidence(response)
    intent_id = row["intent_id"]
    authorization_operation = row["authorization_operation"]
    fencing_token = int(row["fencing_token"])
    transport_operation = row["transport_operation"]
    client_order_id = row["request_client_order_id"]
    target_broker_order_id = row["request_target_broker_order_id"]
    _validate_identity("intent_id", intent_id)
    _validate_identity("client_order_id", client_order_id)
    if target_broker_order_id is not None:
        _validate_identity(
            "target_broker_order_id", target_broker_order_id
        )
    _validate_fencing_token(fencing_token)
    if (
        authorization_operation not in {"SUBMIT", "AMEND"}
        or transport_operation not in _TRANSPORT_OPERATIONS
        or transport_operation.split("_", 1)[0]
        != authorization_operation
    ):
        raise OrderIntentIntegrityError(
            "stored transport response operation is invalid"
        )
    return TransportResponseReceipt(
        intent_id=intent_id,
        authorization_operation=authorization_operation,
        fencing_token=fencing_token,
        transport_operation=transport_operation,
        client_order_id=client_order_id,
        target_broker_order_id=target_broker_order_id,
        response=response,
        recorded_at=_from_us(row["recorded_at"]),
    )


def _validate_envelope(envelope: OrderIntent) -> None:
    if type(envelope) is not OrderIntent:
        raise OrderIntentValidationError("intent envelope must use the exact OrderIntent type")
    for name in ("account_id", "strategy_id", "decision_id", "idempotency_scope", "idempotency_key"):
        _validate_identity(name, getattr(envelope, name))
    _validate_environment(envelope.environment)
    if type(envelope.intent_kind) is not str or envelope.intent_kind not in _INTENT_KINDS:
        raise OrderIntentValidationError("invalid intent kind")
    for name in ("wire_payload", "canonical_payload", "payload_hash"):
        if type(getattr(envelope, name)) is not str:
            raise OrderIntentValidationError(f"{name} must be an exact string")
    _validate_sha256("payload_hash", envelope.payload_hash)
    if _payload_hash(envelope.canonical_payload) != envelope.payload_hash:
        raise OrderIntentIntegrityError("payload hash does not match canonical payload")
    try:
        wire_payload = json.loads(envelope.wire_payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OrderIntentIntegrityError("wire payload is invalid JSON") from exc
    if wire_order_payload(wire_payload) != envelope.wire_payload or canonical_order_payload(wire_payload) != envelope.canonical_payload or _derive_intent_kind(wire_payload) != envelope.intent_kind:
        raise OrderIntentIntegrityError("wire payload does not satisfy strict canonical identity")


def _validate_transport_request_evidence(
    evidence: TransportRequestEvidence,
) -> None:
    if type(evidence) is not TransportRequestEvidence:
        raise OrderIntentValidationError(
            "transport request evidence must use its exact immutable type"
        )
    for name in (
        "account_id",
        "account_id_key",
        "institution_type",
        "intent_id",
        "owner",
        "client_order_id",
    ):
        _validate_identity(name, getattr(evidence, name))
    _validate_environment(evidence.environment)
    _validate_fencing_token(evidence.fencing_token)
    if (
        type(evidence.authorization_operation) is not str
        or evidence.authorization_operation not in {"SUBMIT", "AMEND"}
        or type(evidence.transport_operation) is not str
        or evidence.transport_operation not in _TRANSPORT_OPERATIONS
        or type(evidence.http_method) is not str
        or evidence.http_method not in {"POST", "PUT"}
    ):
        raise OrderIntentValidationError(
            "transport operation metadata is invalid"
        )
    expected_authorization = (
        "SUBMIT"
        if evidence.transport_operation.startswith("SUBMIT")
        else "AMEND"
    )
    expected_method = (
        "POST"
        if evidence.transport_operation.startswith("SUBMIT")
        else "PUT"
    )
    if (
        evidence.authorization_operation != expected_authorization
        or evidence.http_method != expected_method
    ):
        raise OrderIntentIntegrityError(
            "transport operation is inconsistent with authorization"
        )
    is_amend = expected_authorization == "AMEND"
    is_place = evidence.transport_operation.endswith("PLACE")
    if is_amend != (evidence.target_broker_order_id is not None):
        raise OrderIntentIntegrityError("transport target binding is inconsistent")
    if is_place != (evidence.preview_id is not None):
        raise OrderIntentIntegrityError("transport preview binding is inconsistent")
    if evidence.target_broker_order_id is not None:
        _validate_identity(
            "target_broker_order_id", evidence.target_broker_order_id
        )
    if evidence.preview_id is not None:
        _validate_identity("preview_id", evidence.preview_id)
    _validate_sha256(
        "authorization_payload_digest",
        evidence.authorization_payload_digest,
    )
    _validate_sha256("final_xml_sha256", evidence.final_xml_sha256)
    if (
        type(evidence.final_xml_bytes) is not bytes
        or not evidence.final_xml_bytes
        or len(evidence.final_xml_bytes) > _MAX_TRANSPORT_REQUEST_BYTES
        or not hmac.compare_digest(
            hashlib.sha256(evidence.final_xml_bytes).hexdigest(),
            evidence.final_xml_sha256,
        )
    ):
        raise OrderIntentIntegrityError(
            "transport request body does not verify"
        )
    encoded_account = quote(evidence.account_id_key, safe="")
    if is_amend:
        encoded_target = quote(evidence.target_broker_order_id or "", safe="")
        action = "place" if is_place else "preview"
        expected_route = (
            f"/v1/accounts/{encoded_account}/orders/"
            f"{encoded_target}/change/{action}"
        )
    else:
        action = "place" if is_place else "preview"
        expected_route = f"/v1/accounts/{encoded_account}/orders/{action}"
    if type(evidence.route) is not str or evidence.route != expected_route:
        raise OrderIntentIntegrityError("transport route binding is invalid")


def _validate_transport_response_evidence(
    evidence: TransportResponseEvidence,
) -> None:
    if type(evidence) is not TransportResponseEvidence:
        raise OrderIntentValidationError(
            "transport response evidence must use its exact immutable type"
        )
    if (
        type(evidence.disposition) is not str
        or evidence.disposition not in {"ACKNOWLEDGED", "UNKNOWN"}
        or (
            evidence.http_status is not None
            and (
                type(evidence.http_status) is not int
                or not 100 <= evidence.http_status <= 599
            )
        )
    ):
        raise OrderIntentValidationError(
            "transport response disposition is invalid"
        )
    for name, value in (
        ("broker_status", evidence.broker_status),
        ("broker_order_id", evidence.broker_order_id),
        ("preview_id", evidence.preview_id),
        ("unknown_reason", evidence.unknown_reason),
    ):
        if value is not None:
            _validate_identity(name, value)
    tuples = (
        evidence.message_codes,
        evidence.message_types,
        evidence.message_description_digests,
    )
    if (
        any(type(value) is not tuple for value in tuples)
        or len({len(value) for value in tuples}) != 1
        or len(evidence.message_codes) > 64
        or any(
            type(code) is not int or not 0 <= code <= 2_147_483_647
            for code in evidence.message_codes
        )
        or any(
            type(message_type) is not str
            or message_type not in {"WARNING", "INFO", "INFO_HOLD", "ERROR"}
            for message_type in evidence.message_types
        )
    ):
        raise OrderIntentValidationError(
            "transport response messages are invalid"
        )
    for digest in evidence.message_description_digests:
        _validate_sha256("message_description_digest", digest)
    if evidence.raw_response_digest is not None:
        _validate_sha256(
            "raw_response_digest", evidence.raw_response_digest
        )
    _validate_timestamp(evidence.observed_at)
    if (evidence.disposition == "UNKNOWN") != (
        evidence.unknown_reason is not None
    ):
        raise OrderIntentValidationError(
            "transport response unknown reason is inconsistent"
        )


def _validate_acknowledged_transport_response(
    request: TransportRequestEvidence,
    response: TransportResponseEvidence,
) -> None:
    if response.disposition != "ACKNOWLEDGED":
        return
    if (
        response.http_status != 200
        or response.raw_response_digest is None
        or response.broker_status not in {None, "OPEN"}
    ):
        raise OrderIntentValidationError(
            "transport acknowledgement lacks definitive broker evidence"
        )
    if request.transport_operation.endswith("PREVIEW"):
        if (
            response.preview_id is None
            or response.broker_order_id is not None
            or response.message_codes
        ):
            raise OrderIntentValidationError(
                "preview acknowledgement is not safe for placement"
            )
        return
    if (
        response.broker_order_id is None
        or response.preview_id != request.preview_id
        or any(
            code != 1026 or message_type != "WARNING"
            for code, message_type in zip(
                response.message_codes,
                response.message_types,
                strict=True,
            )
        )
    ):
        raise OrderIntentValidationError(
            "place acknowledgement is not definitive"
        )


def _payload_hash(canonical_payload: str) -> str:
    if type(canonical_payload) is not str:
        raise OrderIntentValidationError("canonical payload must be an exact string")
    return hashlib.sha256(_PAYLOAD_HASH_DOMAIN + canonical_payload.encode("utf-8")).hexdigest()


def _outbound_authorization(
    *,
    intent_id: str,
    operation: Literal["SUBMIT", "AMEND"],
    owner: str,
    fencing_token: int,
    client_order_id: str,
    payload: Mapping[str, Any],
) -> OutboundAuthorization:
    """Construct deterministic bytes that the gateway must submit unchanged."""
    _validate_identity("intent_id", intent_id)
    _validate_identity("owner", owner)
    _validate_identity("client_order_id", client_order_id)
    if operation not in {"SUBMIT", "AMEND"}:
        raise OrderIntentValidationError("unsupported outbound authorization operation")
    if type(fencing_token) is not int or fencing_token <= 0:
        raise OrderIntentValidationError("fencing_token must be a positive integer")
    try:
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OrderIntentIntegrityError("durable outbound payload cannot be serialized") from exc
    return OutboundAuthorization(
        intent_id=intent_id,
        operation=operation,
        owner=owner,
        fencing_token=fencing_token,
        client_order_id=client_order_id,
        payload_bytes=payload_bytes,
        payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _domain_bytes_hash(domain: bytes, payload: bytes) -> str:
    if type(domain) is not bytes or not domain.endswith(b"\0"):
        raise OrderIntentValidationError("hash domain is invalid")
    if type(payload) is not bytes:
        raise OrderIntentValidationError("hash payload must be exact bytes")
    return hashlib.sha256(domain + payload).hexdigest()


def _domain_json_hash(domain: bytes, value: Any) -> str:
    return _domain_bytes_hash(
        domain, _canonical_read_json(value).encode("utf-8")
    )


def _canonical_read_json(value: Any) -> str:
    _validate_read_json_tree(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _load_canonical_json(value: str, label: str) -> Any:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > _MAX_BROKER_READ_BYTES
    ):
        raise OrderIntentValidationError(
            f"{label} must be bounded canonical JSON"
        )

    def strict_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if type(key) is not str or key in result:
                raise OrderIntentValidationError(
                    f"{label} contains duplicate or invalid keys"
                )
            result[key] = item
        return result

    def reject_constant(_constant: str) -> Any:
        raise OrderIntentValidationError(
            f"{label} contains a non-finite number"
        )

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except OrderIntentValidationError:
        raise
    except Exception as exc:
        raise OrderIntentValidationError(
            f"{label} is not valid JSON"
        ) from exc
    _validate_read_json_tree(parsed)
    if _canonical_read_json(parsed) != value:
        raise OrderIntentValidationError(
            f"{label} is not in canonical form"
        )
    return parsed


def _load_canonical_json_object(
    value: str, label: str
) -> dict[str, Any]:
    parsed = _load_canonical_json(value, label)
    if type(parsed) is not dict:
        raise OrderIntentValidationError(
            f"{label} must be a JSON object"
        )
    return parsed


def _validate_read_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 50_000 or depth > 48:
            raise OrderIntentValidationError(
                "canonical broker read JSON is too complex"
            )
        item_type = type(item)
        if item is None or item_type in {str, bool, int}:
            if item_type is str and len(item) > _MAX_BROKER_READ_BYTES:
                raise OrderIntentValidationError(
                    "canonical broker read string is too large"
                )
            continue
        if item_type is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if item_type is dict:
            if any(type(key) is not str for key in item):
                raise OrderIntentValidationError(
                    "canonical broker read keys must be strings"
                )
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise OrderIntentValidationError(
            "canonical broker read JSON uses an unsupported scalar"
        )


def _validate_broker_read_response_evidence(
    evidence: BrokerReadResponseEvidence, now: datetime
) -> None:
    if type(evidence) is not BrokerReadResponseEvidence:
        raise OrderIntentValidationError(
            "broker read response must use its exact evidence type"
        )
    if (
        type(evidence.read_kind) is not str
        or evidence.read_kind not in _BROKER_READ_KINDS
        or type(evidence.completeness) is not str
        or evidence.completeness not in _BROKER_READ_COMPLETENESS
    ):
        raise OrderIntentValidationError(
            "broker read response kind/completeness is invalid"
        )
    _validate_identity("read account id", evidence.account_id)
    _validate_identity("read account key", evidence.account_id_key)
    _validate_identity(
        "read institution type", evidence.institution_type
    )
    _validate_environment(evidence.environment)
    if evidence.origin != _ETRADE_ORIGINS[evidence.environment]:
        raise OrderIntentValidationError(
            "broker read origin is not pinned to its environment"
        )
    if (
        type(evidence.route) is not str
        or not evidence.route.startswith("/v1/accounts/")
        or len(evidence.route) > 1024
        or "?" in evidence.route
        or "#" in evidence.route
        or any(ord(char) < 33 or ord(char) > 126 for char in evidence.route)
    ):
        raise OrderIntentValidationError("broker read route is invalid")
    query = _load_canonical_json(
        evidence.query_json, "broker read query"
    )
    if (
        type(query) is not list
        or any(
            type(pair) is not list
            or len(pair) != 2
            or any(type(item) is not str for item in pair)
            for pair in query
        )
        or query != sorted(query)
        or len({pair[0] for pair in query}) != len(query)
    ):
        raise OrderIntentValidationError(
            "broker read query must be a canonical unique string pair list"
        )
    _validate_sha256(
        "broker read authorization digest",
        evidence.authorization_sha256,
    )
    if evidence.read_kind == "ORDER_DETAIL":
        _validate_identity(
            "target broker order id",
            evidence.target_broker_order_id,
        )
    elif evidence.target_broker_order_id is not None:
        raise OrderIntentValidationError(
            "only an order-detail read may name a broker order id"
        )
    _validate_timestamp(evidence.request_started_at)
    _validate_timestamp(evidence.response_completed_at)
    _validate_timestamp(now)
    if (
        evidence.response_completed_at < evidence.request_started_at
        or evidence.response_completed_at > now + timedelta(seconds=5)
        or now - evidence.response_completed_at
        > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS)
    ):
        raise OrderIntentValidationError(
            "broker read response time is invalid or stale"
        )
    if (
        type(evidence.http_status) is not int
        or not 100 <= evidence.http_status <= 599
        or type(evidence.raw_response_bytes) is not bytes
        or len(evidence.raw_response_bytes) > _MAX_BROKER_READ_BYTES
    ):
        raise OrderIntentValidationError(
            "broker read response status/body is invalid"
        )
    _validate_identity("parser schema", evidence.parser_schema)
    _validate_sha256("parser code digest", evidence.parser_code_sha256)
    _validate_sha256(
        "parser config digest", evidence.parser_config_sha256
    )
    _load_canonical_json(
        evidence.canonical_parsed_json, "parsed broker response"
    )


def _validate_broker_read_manifest_evidence(
    evidence: BrokerReadManifestEvidence, now: datetime
) -> None:
    if type(evidence) is not BrokerReadManifestEvidence:
        raise OrderIntentValidationError(
            "broker read manifest must use its exact evidence type"
        )
    if (
        type(evidence.evidence_kind) is not str
        or evidence.evidence_kind not in _BROKER_READ_EVIDENCE_KINDS
        or type(evidence.completeness) is not str
        or evidence.completeness
        not in _BROKER_READ_EVIDENCE_COMPLETENESS
    ):
        raise OrderIntentValidationError(
            "broker read manifest kind/completeness is invalid"
        )
    _validate_identity("manifest account id", evidence.account_id)
    _validate_identity("manifest account key", evidence.account_id_key)
    _validate_identity(
        "manifest institution type", evidence.institution_type
    )
    _validate_environment(evidence.environment)
    if evidence.origin != _ETRADE_ORIGINS[evidence.environment]:
        raise OrderIntentValidationError(
            "broker read manifest origin is invalid"
        )
    if evidence.evidence_kind == "ORDER_QUERY":
        _validate_identity(
            "manifest broker order id",
            evidence.target_broker_order_id,
        )
    elif evidence.target_broker_order_id is not None:
        raise OrderIntentValidationError(
            "capacity manifests cannot target an order"
        )
    _validate_timestamp(evidence.observed_at)
    _validate_timestamp(now)
    if (
        evidence.observed_at > now + timedelta(seconds=5)
        or now - evidence.observed_at
        > timedelta(seconds=_EVIDENCE_MAX_AGE_SECONDS)
    ):
        raise OrderIntentValidationError(
            "broker read manifest is stale or from the future"
        )
    _load_canonical_json_object(
        evidence.canonical_result_json, "broker read manifest result"
    )


def _validate_broker_read_manifest_members(
    members: tuple[BrokerReadManifestMember, ...],
) -> None:
    if (
        type(members) is not tuple
        or not members
        or len(members) > 2_048
        or any(type(member) is not BrokerReadManifestMember for member in members)
    ):
        raise OrderIntentValidationError(
            "broker read manifest members are invalid"
        )
    roles: set[str] = set()
    receipts: set[str] = set()
    for member in members:
        _validate_identity("broker read member role", member.role)
        _validate_sha256(
            "broker read member receipt", member.receipt_sha256
        )
        if member.role in roles or member.receipt_sha256 in receipts:
            raise OrderIntentValidationError(
                "broker read manifest members must be unique"
            )
        roles.add(member.role)
        receipts.add(member.receipt_sha256)


def _validate_broker_read_manifest_semantics(
    *,
    evidence_kind: str,
    account_id: str,
    account_id_key: str,
    institution_type: str,
    target_broker_order_id: str | None,
    observed_at: int,
    completeness: str,
    result: dict[str, Any],
    member_roles: tuple[str, ...],
    receipt_rows: tuple[sqlite3.Row, ...],
) -> None:
    """Rebuild complete manifest results from their ordered durable receipts."""

    if len(member_roles) != len(receipt_rows) or not receipt_rows:
        raise OrderIntentIntegrityError(
            "broker read manifest receipt lineage is incomplete"
        )
    if any(
        int(row["response_completed_at"]) > observed_at
        or observed_at - int(row["request_started_at"])
        > _MAX_BROKER_READ_SPAN_SECONDS * 1_000_000
        for row in receipt_rows
    ):
        raise OrderIntentIntegrityError(
            "broker read manifest exceeds its bounded observation window"
        )
    if any(
        int(previous["response_completed_at"])
        > int(current["request_started_at"])
        for previous, current in zip(receipt_rows, receipt_rows[1:])
    ):
        raise OrderIntentIntegrityError(
            "broker read manifest receipts are not chronologically ordered"
        )
    if int(receipt_rows[-1]["response_completed_at"]) != observed_at:
        raise OrderIntentIntegrityError(
            "broker read manifest timestamp is not its final response"
        )
    authorization_digests = tuple(
        row["authorization_sha256"] for row in receipt_rows
    )
    if (
        any(type(value) is not str for value in authorization_digests)
        or len(set(authorization_digests)) != len(authorization_digests)
    ):
        raise OrderIntentIntegrityError(
            "broker read manifest does not prove distinct OAuth exchanges"
        )

    parser_provenance = {
        (
            row["parser_schema"],
            row["parser_code_sha256"],
            row["parser_config_sha256"],
        )
        for row in receipt_rows
    }
    if len(parser_provenance) != 1:
        raise OrderIntentIntegrityError(
            "broker read manifest mixes parser versions"
        )
    if completeness != "COMPLETE":
        return
    if any(row["completeness"] == "INELIGIBLE" for row in receipt_rows):
        raise OrderIntentIntegrityError(
            "complete broker read manifest contains ineligible evidence"
        )
    if evidence_kind == "ORDER_QUERY":
        _validate_order_query_receipt_lineage(
            account_id=account_id,
            account_id_key=account_id_key,
            institution_type=institution_type,
            target_broker_order_id=target_broker_order_id,
            result=result,
            member_roles=member_roles,
            receipt_rows=receipt_rows,
        )
        return
    if evidence_kind == "CAPACITY":
        _validate_capacity_receipt_lineage(
            account_id=account_id,
            account_id_key=account_id_key,
            institution_type=institution_type,
            result=result,
            member_roles=member_roles,
            receipt_rows=receipt_rows,
        )
        return
    raise OrderIntentIntegrityError(
        "broker read manifest evidence kind is unsupported"
    )


def _validate_order_query_receipt_lineage(
    *,
    account_id: str,
    account_id_key: str,
    institution_type: str,
    target_broker_order_id: str | None,
    result: dict[str, Any],
    member_roles: tuple[str, ...],
    receipt_rows: tuple[sqlite3.Row, ...],
) -> None:
    if (
        target_broker_order_id is None
        or member_roles
        != ("binding.start", "order.detail", "binding.end")
        or tuple(row["read_kind"] for row in receipt_rows)
        != ("ACCOUNT_LIST", "ORDER_DETAIL", "ACCOUNT_LIST")
    ):
        raise OrderIntentIntegrityError(
            "order query manifest lacks its exact read sequence"
        )
    binding_start = _verified_binding_receipt(
        receipt_rows[0],
        account_id=account_id,
        account_id_key=account_id_key,
        institution_type=institution_type,
    )
    binding_end = _verified_binding_receipt(
        receipt_rows[2],
        account_id=account_id,
        account_id_key=account_id_key,
        institution_type=institution_type,
    )
    if binding_start != binding_end:
        raise OrderIntentIntegrityError(
            "order query account binding changed during observation"
        )
    detail = receipt_rows[1]
    encoded_account = quote(account_id_key, safe="")
    encoded_order = quote(target_broker_order_id, safe="")
    if (
        detail["route"]
        != f"/v1/accounts/{encoded_account}/orders/{encoded_order}.json"
        or _receipt_query(detail)
        or detail["target_broker_order_id"] != target_broker_order_id
        or detail["completeness"] != "COMPLETE"
        or receipt_rows[0]["completeness"] != "COMPLETE"
        or receipt_rows[2]["completeness"] != "COMPLETE"
    ):
        raise OrderIntentIntegrityError(
            "order query receipt is not bound to the exact known order"
        )
    parsed = _receipt_parsed_object(detail)
    expected_parsed_keys = {
        "not_found",
        "raw_status",
        "outcome",
        "order_payload_hashes",
        "replacement_links",
        "normalized_order",
    }
    schema = result.get("schema")
    if schema == "etrade-order-query.v2":
        expected_parsed_keys.add("fill_summary")
    elif schema != "etrade-order-query.v1":
        raise OrderIntentIntegrityError(
            "order query receipt uses an unsupported result schema"
        )
    if set(parsed) != expected_parsed_keys:
        raise OrderIntentIntegrityError(
            "order query receipt parser result has an unexpected shape"
        )
    expected_result = {
        "schema": schema,
        "broker_order_id": target_broker_order_id,
        "raw_status": parsed["raw_status"],
        "outcome": parsed["outcome"],
        "order_payload_hashes": parsed["order_payload_hashes"],
        "http_status": int(detail["http_status"]),
        "raw_response_digest": detail["raw_response_sha256"],
        "not_found": parsed["not_found"],
        "replacement_links": parsed["replacement_links"],
    }
    if schema == "etrade-order-query.v2":
        expected_result["fill_summary"] = parsed["fill_summary"]
    if result != expected_result:
        raise OrderIntentIntegrityError(
            "order query manifest result is disconnected from its receipt"
        )
    _validate_order_query_manifest_result(result)


def _validate_capacity_receipt_lineage(
    *,
    account_id: str,
    account_id_key: str,
    institution_type: str,
    result: dict[str, Any],
    member_roles: tuple[str, ...],
    receipt_rows: tuple[sqlite3.Row, ...],
) -> None:
    schema = result.get("schema")
    if schema not in {"etrade-capacity.v1", "etrade-capacity.v2"}:
        raise OrderIntentIntegrityError(
            "capacity manifest uses an unsupported result schema"
        )
    lots_required = schema == "etrade-capacity.v2"
    if member_roles[0] != "binding.start" or member_roles[-1] != "binding.end":
        raise OrderIntentIntegrityError(
            "capacity manifest lacks account-binding brackets"
        )
    binding_start = _verified_binding_receipt(
        receipt_rows[0],
        account_id=account_id,
        account_id_key=account_id_key,
        institution_type=institution_type,
    )
    binding_end = _verified_binding_receipt(
        receipt_rows[-1],
        account_id=account_id,
        account_id_key=account_id_key,
        institution_type=institution_type,
    )
    if binding_start != binding_end:
        raise OrderIntentIntegrityError(
            "capacity account binding changed during observation"
        )
    cursor = 1
    scans: list[dict[str, Any]] = []
    for scan_name in ("scan_a", "scan_b"):
        if (
            cursor >= len(receipt_rows) - 1
            or member_roles[cursor] != f"{scan_name}.balance"
        ):
            raise OrderIntentIntegrityError(
                "capacity manifest lacks both balance scans"
            )
        balance_row = receipt_rows[cursor]
        _validate_capacity_route(
            balance_row,
            account_id_key=account_id_key,
            expected_kind="BALANCE",
        )
        if (
            _receipt_query(balance_row)
            != {
                "instType": institution_type,
                "realTimeNAV": "true",
            }
            or balance_row["completeness"] != "COMPLETE"
        ):
            raise OrderIntentIntegrityError(
                "capacity balance request is not the reviewed real-time read"
            )
        balance = _receipt_parsed_object(balance_row)
        if (
            set(balance)
            != {
                "account_id",
                "institution_type",
                "margin_buying_power",
                "as_of_date",
            }
            or balance["account_id"] != account_id
            or balance["institution_type"] not in {None, institution_type}
            or type(balance["margin_buying_power"]) is not str
            or type(balance["as_of_date"]) is not str
            or len(balance["as_of_date"]) != 13
            or not balance["as_of_date"].isdigit()
        ):
            raise OrderIntentIntegrityError(
                "capacity balance parser result is invalid"
            )
        balance_as_of_us = int(balance["as_of_date"]) * 1_000
        balance_completed_us = int(
            balance_row["response_completed_at"]
        )
        if (
            balance_as_of_us > balance_completed_us + 5_000_000
            or balance_completed_us - balance_as_of_us
            > _EVIDENCE_MAX_AGE_SECONDS * 1_000_000
        ):
            raise OrderIntentIntegrityError(
                "capacity balance effective time is stale or future"
            )
        cursor += 1

        portfolio_rows: list[sqlite3.Row] = []
        portfolio_roles: list[str] = []
        portfolio_prefix = f"{scan_name}.portfolio."
        while (
            cursor < len(receipt_rows) - 1
            and member_roles[cursor].startswith(portfolio_prefix)
        ):
            portfolio_roles.append(member_roles[cursor])
            portfolio_rows.append(receipt_rows[cursor])
            cursor += 1
        if not portfolio_rows:
            raise OrderIntentIntegrityError(
                "capacity scan lacks a complete portfolio traversal"
            )
        positions = _rebuild_portfolio_scan(
            portfolio_rows,
            portfolio_roles,
            account_id_key=account_id_key,
            lots_required=lots_required,
        )

        orders: list[dict[str, Any]] = []
        seen_order_ids: set[str] = set()
        for lane in _ACTIVE_ORDER_READ_LANES:
            lane_rows: list[sqlite3.Row] = []
            lane_roles: list[str] = []
            lane_prefix = f"{scan_name}.orders.{lane}."
            while (
                cursor < len(receipt_rows) - 1
                and member_roles[cursor].startswith(lane_prefix)
            ):
                lane_roles.append(member_roles[cursor])
                lane_rows.append(receipt_rows[cursor])
                cursor += 1
            if not lane_rows:
                raise OrderIntentIntegrityError(
                    "capacity scan lacks an active-order status traversal"
                )
            for order in _rebuild_order_lane(
                lane_rows,
                lane_roles,
                account_id_key=account_id_key,
                lane=lane,
            ):
                order_id = order.get("order_id")
                if type(order_id) is not str or order_id in seen_order_ids:
                    raise OrderIntentIntegrityError(
                        "capacity scan contains ambiguous active orders"
                    )
                seen_order_ids.add(order_id)
                orders.append(order)
        orders.sort(key=lambda item: item["order_id"])
        scans.append(
            {
                "schema": schema,
                "account_status": binding_start["account_status"],
                "account_mode": binding_start["account_mode"],
                "account_type": binding_start["account_type"],
                "broker_buying_power": balance["margin_buying_power"],
                "broker_buying_power_as_of": balance["as_of_date"],
                "positions": positions,
                "open_orders": orders,
            }
        )
    if cursor != len(receipt_rows) - 1:
        raise OrderIntentIntegrityError(
            "capacity manifest contains an unrecognized receipt role"
        )
    economic_scans = []
    for scan in scans:
        economic = dict(scan)
        economic.pop("broker_buying_power_as_of")
        economic_scans.append(economic)
    if economic_scans[0] != economic_scans[1]:
        raise OrderIntentIntegrityError(
            "capacity manifest scans are not semantically stable"
        )
    if int(scans[1]["broker_buying_power_as_of"]) < int(
        scans[0]["broker_buying_power_as_of"]
    ):
        raise OrderIntentIntegrityError(
            "capacity balance effective time moved backward"
        )
    rebuilt = dict(scans[1])
    rebuilt["state_sha256"] = _domain_json_hash(
        (
            b"etrade-capacity-state.v2\0"
            if lots_required
            else b"etrade-capacity-state.v1\0"
        ),
        economic_scans[1],
    )
    if result != rebuilt:
        raise OrderIntentIntegrityError(
            "capacity manifest result is disconnected from its receipts"
        )
    _validate_capacity_manifest_result(
        result, expected_account_id=account_id
    )


def _verified_binding_receipt(
    row: sqlite3.Row,
    *,
    account_id: str,
    account_id_key: str,
    institution_type: str,
) -> dict[str, Any]:
    if (
        row["read_kind"] != "ACCOUNT_LIST"
        or row["route"] != "/v1/accounts/list.json"
        or _receipt_query(row)
        or row["target_broker_order_id"] is not None
        or row["http_status"] != 200
        or row["completeness"] != "COMPLETE"
    ):
        raise OrderIntentIntegrityError(
            "account binding receipt is not an exact account-list read"
        )
    parsed = _receipt_parsed_object(row)
    if (
        set(parsed)
        != {
            "account_id",
            "account_id_key",
            "institution_type",
            "account_status",
            "account_mode",
            "account_type",
        }
        or parsed["account_id"] != account_id
        or parsed["account_id_key"] != account_id_key
        or parsed["institution_type"] != institution_type
        or parsed["account_status"] != "ACTIVE"
        or parsed["account_mode"] != "MARGIN"
        or type(parsed["account_type"]) is not str
        or not parsed["account_type"]
    ):
        raise OrderIntentIntegrityError(
            "account binding receipt is not the configured active margin account"
        )
    return parsed


def _validate_capacity_route(
    row: sqlite3.Row,
    *,
    account_id_key: str,
    expected_kind: str,
) -> None:
    suffixes = {
        "BALANCE": "balance.json",
        "PORTFOLIO_PAGE": "portfolio.json",
        "OPEN_ORDERS_PAGE": "orders.json",
    }
    suffix = suffixes.get(expected_kind)
    if (
        suffix is None
        or row["read_kind"] != expected_kind
        or row["route"]
        != f"/v1/accounts/{quote(account_id_key, safe='')}/{suffix}"
        or row["target_broker_order_id"] is not None
        or row["http_status"] not in {200, 204}
    ):
        raise OrderIntentIntegrityError(
            "capacity receipt route or account binding is invalid"
        )


def _rebuild_portfolio_scan(
    rows: list[sqlite3.Row],
    roles: list[str],
    *,
    account_id_key: str,
    lots_required: bool,
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_next: int | None = None
    for ordinal, (row, role) in enumerate(zip(rows, roles), start=1):
        expected_role = role.rsplit(".", 1)[0] + f".{ordinal:04d}"
        if role != expected_role:
            raise OrderIntentIntegrityError(
                "portfolio manifest roles are not contiguous"
            )
        _validate_capacity_route(
            row,
            account_id_key=account_id_key,
            expected_kind="PORTFOLIO_PAGE",
        )
        expected_query = {
            "count": "50",
            "lotsRequired": "true" if lots_required else "false",
            "marketSession": "REGULAR",
            "pageNumber": str(ordinal),
            "sortBy": "SYMBOL",
            "sortOrder": "ASC",
            "totalsRequired": "false",
            "view": "COMPLETE",
        }
        if _receipt_query(row) != expected_query:
            raise OrderIntentIntegrityError(
                "portfolio page query is not the fixed complete traversal"
            )
        parsed = _receipt_parsed_object(row)
        if (
            set(parsed)
            != {
                "page_number",
                "total_pages",
                "metadata_field",
                "next_page",
                "positions",
            }
            or parsed["page_number"] != ordinal
            or type(parsed["positions"]) is not list
            or (
                ordinal > 1
                and previous_next != ordinal
            )
        ):
            raise OrderIntentIntegrityError(
                "portfolio parser result is not contiguous"
            )
        is_last = ordinal == len(rows)
        if (
            row["completeness"]
            != ("COMPLETE" if is_last else "HAS_NEXT")
            or (
                not is_last
                and parsed["next_page"] != ordinal + 1
            )
            or (is_last and parsed["next_page"] is not None)
        ):
            raise OrderIntentIntegrityError(
                "portfolio receipt completeness conflicts with pagination"
            )
        previous_next = parsed["next_page"]
        for position in parsed["positions"]:
            if type(position) is not dict:
                raise OrderIntentIntegrityError(
                    "portfolio position result is not an object"
                )
            position_id = position.get("position_id")
            if type(position_id) is not str or position_id in seen_ids:
                raise OrderIntentIntegrityError(
                    "portfolio position identity is ambiguous"
                )
            seen_ids.add(position_id)
            positions.append(position)
    positions.sort(key=lambda item: item["position_id"])
    return positions


def _rebuild_order_lane(
    rows: list[sqlite3.Row],
    roles: list[str],
    *,
    account_id_key: str,
    lane: str,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    previous_marker: str | None = None
    for ordinal, (row, role) in enumerate(zip(rows, roles)):
        expected_role = role.rsplit(".", 1)[0] + f".{ordinal:04d}"
        if role != expected_role:
            raise OrderIntentIntegrityError(
                "order manifest roles are not contiguous"
            )
        _validate_capacity_route(
            row,
            account_id_key=account_id_key,
            expected_kind="OPEN_ORDERS_PAGE",
        )
        query = _receipt_query(row)
        expected_query = {"count": "100", "status": lane}
        if ordinal:
            expected_query["marker"] = previous_marker
        if query != expected_query:
            raise OrderIntentIntegrityError(
                "active-order marker traversal is incomplete"
            )
        parsed = _receipt_parsed_object(row)
        if (
            set(parsed) != {"status_lane", "orders", "marker"}
            or parsed["status_lane"] != lane
            or type(parsed["orders"]) is not list
        ):
            raise OrderIntentIntegrityError(
                "active-order parser result is invalid"
            )
        is_last = ordinal == len(rows) - 1
        if (
            row["completeness"]
            != ("COMPLETE" if is_last else "HAS_NEXT")
            or (is_last and parsed["marker"] is not None)
            or (
                not is_last
                and (
                    type(parsed["marker"]) is not str
                    or not parsed["marker"]
                )
            )
        ):
            raise OrderIntentIntegrityError(
                "active-order receipt completeness conflicts with markers"
            )
        previous_marker = parsed["marker"]
        orders.extend(parsed["orders"])
    return orders


def _receipt_query(row: sqlite3.Row) -> dict[str, str]:
    parsed = _load_canonical_json(
        row["query_json"], "broker read receipt query"
    )
    if (
        type(parsed) is not list
        or any(
            type(pair) is not list
            or len(pair) != 2
            or any(type(item) is not str for item in pair)
            for pair in parsed
        )
    ):
        raise OrderIntentIntegrityError(
            "broker read receipt query shape is invalid"
        )
    return {pair[0]: pair[1] for pair in parsed}


def _receipt_parsed_object(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return _load_canonical_json_object(
            row["canonical_parsed_json"],
            "broker read receipt parser result",
        )
    except OrderIntentValidationError as exc:
        raise OrderIntentIntegrityError(
            "broker read receipt parser result is corrupt"
        ) from exc


def _verify_broker_read_receipt_row(row: sqlite3.Row) -> None:
    raw = row["raw_response_bytes"]
    if (
        type(raw) is not bytes
        or len(raw) != int(row["raw_byte_length"])
        or len(raw) > _MAX_BROKER_READ_BYTES
    ):
        raise OrderIntentIntegrityError(
            "broker read receipt raw bytes are corrupt"
        )
    raw_digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(raw_digest, row["raw_response_sha256"]):
        raise OrderIntentIntegrityError(
            "broker read receipt raw digest does not verify"
        )
    _load_canonical_json(row["query_json"], "persisted broker query")
    _load_canonical_json(
        row["canonical_parsed_json"], "persisted parsed broker response"
    )
    parsed_digest = _domain_bytes_hash(
        _READ_PARSED_HASH_DOMAIN,
        row["canonical_parsed_json"].encode("utf-8"),
    )
    if parsed_digest != row["canonical_parsed_sha256"]:
        raise OrderIntentIntegrityError(
            "broker read receipt parsed digest does not verify"
        )
    request_material = {
        "read_kind": row["read_kind"],
        "account_id": row["account_id"],
        "account_id_key": row["account_id_key"],
        "institution_type": row["institution_type"],
        "environment": row["environment"],
        "origin": row["origin"],
        "http_method": row["http_method"],
        "route": row["route"],
        "query_json": row["query_json"],
        "authorization_sha256": row["authorization_sha256"],
        "target_broker_order_id": row["target_broker_order_id"],
    }
    request_digest = _domain_json_hash(
        _READ_REQUEST_HASH_DOMAIN, request_material
    )
    if request_digest != row["request_sha256"]:
        raise OrderIntentIntegrityError(
            "broker read request digest does not verify"
        )
    receipt_material = {
        "request_sha256": request_digest,
        "request_started_at": int(row["request_started_at"]),
        "response_completed_at": int(row["response_completed_at"]),
        "http_status": int(row["http_status"]),
        "raw_byte_length": int(row["raw_byte_length"]),
        "raw_response_sha256": raw_digest,
        "parser_schema": row["parser_schema"],
        "parser_code_sha256": row["parser_code_sha256"],
        "parser_config_sha256": row["parser_config_sha256"],
        "canonical_parsed_sha256": parsed_digest,
        "completeness": row["completeness"],
        "recorded_at": int(row["recorded_at"]),
    }
    receipt_digest = _domain_json_hash(
        _READ_RECEIPT_HASH_DOMAIN, receipt_material
    )
    if not hmac.compare_digest(receipt_digest, row["receipt_sha256"]):
        raise OrderIntentIntegrityError(
            "broker read receipt content hash does not verify"
        )


def _canonical_signed_decimal_text(
    value: Any, label: str
) -> Decimal:
    if type(value) is not str or not value:
        raise OrderIntentIntegrityError(
            f"{label} must be a canonical decimal string"
        )
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise OrderIntentIntegrityError(
            f"{label} must be a finite decimal"
        ) from exc
    if not parsed.is_finite():
        raise OrderIntentIntegrityError(
            f"{label} must be a finite decimal"
        )
    normalized = format(parsed, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    if normalized != value:
        raise OrderIntentIntegrityError(
            f"{label} is not canonical"
        )
    return parsed


def _canonical_unsigned_integer_text(
    value: Any,
    label: str,
    *,
    positive: bool,
) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdigit()
        or (len(value) > 1 and value[0] == "0")
    ):
        raise OrderIntentIntegrityError(
            f"{label} must be a canonical integer string"
        )
    parsed = int(value)
    if (positive and parsed <= 0) or (not positive and parsed < 0):
        raise OrderIntentIntegrityError(
            f"{label} is outside the allowed range"
        )
    return parsed


def _validate_normalized_product(
    product: Any, label: str
) -> None:
    expected = {
        "symbol",
        "security_type",
        "call_put",
        "expiry_year",
        "expiry_month",
        "expiry_day",
        "strike_price",
        "product_id",
    }
    if type(product) is not dict or set(product) != expected:
        raise OrderIntentIntegrityError(
            f"{label} product shape is invalid"
        )
    if (
        type(product["symbol"]) is not str
        or not product["symbol"]
        or product["security_type"] not in {"EQ", "OPTN"}
        or (
            product["product_id"] is not None
            and (
                type(product["product_id"]) is not dict
                or set(product["product_id"])
                != {"symbol", "type_code"}
                or any(
                    value is not None
                    and (type(value) is not str or not value)
                    for value in product["product_id"].values()
                )
            )
        )
    ):
        raise OrderIntentIntegrityError(
            f"{label} product identity is invalid"
        )
    if product["security_type"] == "OPTN":
        if (
            product["call_put"] not in {"PUT", "CALL"}
            or any(
                type(product[name]) is not str
                for name in (
                    "expiry_year",
                    "expiry_month",
                    "expiry_day",
                    "strike_price",
                )
            )
        ):
            raise OrderIntentIntegrityError(
                f"{label} option identity is incomplete"
            )
        for name in ("expiry_year", "expiry_month", "expiry_day"):
            _canonical_unsigned_integer_text(
                product[name], f"{label} {name}", positive=True
            )
        strike = _canonical_signed_decimal_text(
            product["strike_price"], f"{label} strike"
        )
        if strike <= 0:
            raise OrderIntentIntegrityError(
                f"{label} strike must be positive"
            )


def _validate_fill_summary_shape(summary: Any) -> None:
    if (
        type(summary) is not dict
        or set(summary)
        != {
            "classification",
            "placed_time_epoch_ms",
            "executed_time_epoch_ms",
            "legs",
        }
        or summary["classification"]
        not in {
            "OPEN",
            "FULL_FILL",
            "ZERO_FILL_TERMINAL",
            "UNRESOLVED",
        }
        or type(summary["legs"]) is not list
    ):
        raise OrderIntentIntegrityError(
            "order fill summary shape is invalid"
        )
    for name in (
        "placed_time_epoch_ms",
        "executed_time_epoch_ms",
    ):
        value = summary[name]
        if value is not None:
            parsed = _canonical_unsigned_integer_text(
                value, f"fill summary {name}", positive=True
            )
            if len(value) != 13 or parsed > 9_223_372_036_854_775_807:
                raise OrderIntentIntegrityError(
                    f"fill summary {name} is invalid"
                )
    legs = summary["legs"]
    if not legs:
        if summary["classification"] != "UNRESOLVED":
            raise OrderIntentIntegrityError(
                "resolved fill summary must contain exact legs"
            )
        return
    if len(legs) != 2:
        raise OrderIntentIntegrityError(
            "fill summary must contain exactly two legs"
        )
    numbers: set[int] = set()
    quantities: list[tuple[Decimal, Decimal, Decimal]] = []
    for leg in legs:
        if (
            type(leg) is not dict
            or set(leg)
            != {
                "leg_number",
                "product",
                "order_action",
                "ordered_quantity",
                "filled_quantity",
                "cancel_quantity",
            }
            or type(leg["leg_number"]) is not int
            or leg["leg_number"] not in {1, 2}
            or leg["order_action"] not in {
                "BUY_OPEN",
                "SELL_OPEN",
            }
        ):
            raise OrderIntentIntegrityError(
                "fill summary leg shape is invalid"
            )
        numbers.add(leg["leg_number"])
        _validate_normalized_product(
            leg["product"], "fill summary leg"
        )
        ordered = _canonical_signed_decimal_text(
            leg["ordered_quantity"], "ordered quantity"
        )
        filled = _canonical_signed_decimal_text(
            leg["filled_quantity"], "filled quantity"
        )
        cancelled = _canonical_signed_decimal_text(
            leg["cancel_quantity"], "cancel quantity"
        )
        if (
            ordered <= 0
            or ordered != ordered.to_integral_value()
            or filled < 0
            or cancelled < 0
            or filled + cancelled > ordered
        ):
            raise OrderIntentIntegrityError(
                "fill summary quantities are inconsistent"
            )
        quantities.append((ordered, filled, cancelled))
    if numbers != {1, 2}:
        raise OrderIntentIntegrityError(
            "fill summary leg numbers are ambiguous"
        )
    classification = summary["classification"]
    executed = summary["executed_time_epoch_ms"]
    if classification == "OPEN" and (
        executed is not None
        or any(filled != 0 or cancelled != 0 for _, filled, cancelled in quantities)
    ):
        raise OrderIntentIntegrityError(
            "open fill summary is inconsistent"
        )
    if classification == "FULL_FILL" and (
        executed is None
        or any(
            filled != ordered or cancelled != 0
            for ordered, filled, cancelled in quantities
        )
    ):
        raise OrderIntentIntegrityError(
            "full-fill summary is inconsistent"
        )
    if classification == "ZERO_FILL_TERMINAL" and (
        executed is not None
        or any(
            filled != 0 or cancelled != ordered
            for ordered, filled, cancelled in quantities
        )
    ):
        raise OrderIntentIntegrityError(
            "zero-fill terminal summary is inconsistent"
        )


def _validate_capacity_v2_positions(
    positions: list[Any],
) -> None:
    position_ids: list[str] = []
    lot_ids: set[str] = set()
    for position in positions:
        if (
            type(position) is not dict
            or set(position)
            != {
                "position_id",
                "account_id",
                "product",
                "quantity",
                "position_type",
                "position_indicator",
                "osi_key",
                "lots",
            }
        ):
            raise OrderIntentIntegrityError(
                "schema-v2 position shape is invalid"
            )
        _canonical_unsigned_integer_text(
            position["position_id"],
            "position id",
            positive=True,
        )
        _canonical_unsigned_integer_text(
            position["account_id"],
            "position account id",
            positive=True,
        )
        _canonical_signed_decimal_text(
            position["quantity"], "position quantity"
        )
        _validate_normalized_product(
            position["product"], "position"
        )
        for name in (
            "position_type",
            "position_indicator",
            "osi_key",
        ):
            value = position[name]
            if value is not None and (
                type(value) is not str or not value
            ):
                raise OrderIntentIntegrityError(
                    f"position {name} is invalid"
                )
        if type(position["lots"]) is not list:
            raise OrderIntentIntegrityError(
                "position lots must be an array"
            )
        if position["lots"] != sorted(
            position["lots"], key=_canonical_read_json
        ):
            raise OrderIntentIntegrityError(
                "position lots are not canonically ordered"
            )
        position_ids.append(position["position_id"])
        for lot in position["lots"]:
            if (
                type(lot) is not dict
                or set(lot)
                != {
                    "position_id",
                    "position_lot_id",
                    "order_no",
                    "leg_no",
                    "original_quantity",
                    "remaining_quantity",
                    "available_quantity",
                    "acquired_date_epoch_ms",
                }
                or lot["position_id"] != position["position_id"]
            ):
                raise OrderIntentIntegrityError(
                    "position lot shape or parent identity is invalid"
                )
            _canonical_unsigned_integer_text(
                lot["position_lot_id"],
                "position lot id",
                positive=True,
            )
            if lot["position_lot_id"] in lot_ids:
                raise OrderIntentIntegrityError(
                    "capacity positions contain a duplicate lot id"
                )
            lot_ids.add(lot["position_lot_id"])
            if lot["order_no"] is not None:
                _canonical_unsigned_integer_text(
                    lot["order_no"],
                    "position lot order number",
                    positive=False,
                )
            if lot["leg_no"] is not None:
                leg_no = _canonical_unsigned_integer_text(
                    lot["leg_no"],
                    "position lot leg number",
                    positive=False,
                )
                if leg_no > 2_147_483_647:
                    raise OrderIntentIntegrityError(
                        "position lot leg number exceeds signed int32"
                    )
            for name in (
                "original_quantity",
                "remaining_quantity",
                "available_quantity",
            ):
                _canonical_signed_decimal_text(
                    lot[name], f"position lot {name}"
                )
            acquired = lot["acquired_date_epoch_ms"]
            if (
                type(acquired) is not str
                or not acquired
                or acquired in {"+0", "-0"}
            ):
                raise OrderIntentIntegrityError(
                    "position lot acquired date is invalid"
                )
            acquired_digits = (
                acquired[1:] if acquired.startswith("-") else acquired
            )
            if (
                not acquired_digits.isascii()
                or not acquired_digits.isdigit()
                or (
                    len(acquired_digits) > 1
                    and acquired_digits[0] == "0"
                )
                or not -(2**63) <= int(acquired) <= 2**63 - 1
            ):
                raise OrderIntentIntegrityError(
                    "position lot acquired date is invalid"
                )
    if position_ids != sorted(position_ids) or len(position_ids) != len(
        set(position_ids)
    ):
        raise OrderIntentIntegrityError(
            "capacity position identities are ambiguous"
        )


def _terminal_fill_classification(
    intent: sqlite3.Row,
    result: dict[str, Any],
    *,
    manifest_observed_at: int,
) -> tuple[Literal["ZERO_FILL", "FULL_FILL"], int, int]:
    summary = result["fill_summary"]
    _validate_fill_summary_shape(summary)
    terminal_state = intent["state"]
    if terminal_state == "FILLED":
        expected_summary = "FULL_FILL"
        expected_raw_status = "EXECUTED"
        classification: Literal["ZERO_FILL", "FULL_FILL"] = "FULL_FILL"
    else:
        expected_summary = "ZERO_FILL_TERMINAL"
        expected_raw_status = terminal_state
        classification = "ZERO_FILL"
    if (
        summary["classification"] != expected_summary
        or result["raw_status"] != expected_raw_status
    ):
        raise OrderIntentReconciliationRequired(
            "terminal status and exact fill classification disagree"
        )
    placed = summary["placed_time_epoch_ms"]
    if placed is None:
        raise OrderIntentReconciliationRequired(
            "terminal fill evidence lacks a placement timestamp"
        )
    placed_ms = int(placed)
    if placed_ms * 1_000 > manifest_observed_at + 5_000_000:
        raise OrderIntentReconciliationRequired(
            "terminal placement timestamp is in the future"
        )
    executed = summary["executed_time_epoch_ms"]
    if classification == "FULL_FILL":
        if (
            executed is None
            or int(executed) < placed_ms
            or int(executed) * 1_000
            > manifest_observed_at + 5_000_000
        ):
            raise OrderIntentReconciliationRequired(
                "full-fill execution timestamp is missing or inconsistent"
            )
    elif executed is not None:
        raise OrderIntentReconciliationRequired(
            "zero-fill terminal evidence contains an execution timestamp"
        )
    try:
        payload = json.loads(intent["wire_payload"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise OrderIntentIntegrityError(
            "durable intent payload cannot be decoded"
        ) from exc
    intent_legs = payload.get("legs")
    if type(intent_legs) is not list or len(intent_legs) != 2:
        raise OrderIntentIntegrityError(
            "terminal opening intent is not an exact vertical"
        )
    unmatched = list(intent_legs)
    ordered_values: set[int] = set()
    filled_values: set[int] = set()
    for fill_leg in summary["legs"]:
        product = fill_leg["product"]
        matches = [
            leg
            for leg in unmatched
            if (
                leg["symbol"] == product["symbol"]
                and product["security_type"] == "OPTN"
                and leg["callPut"] == product["call_put"]
                and str(leg["expiryYear"]) == product["expiry_year"]
                and str(leg["expiryMonth"]) == product["expiry_month"]
                and str(leg["expiryDay"]) == product["expiry_day"]
                and _canonical_amount(leg["strikePrice"])
                == product["strike_price"]
                and leg["orderAction"]
                == fill_leg["order_action"]
            )
        ]
        if len(matches) != 1:
            raise OrderIntentReconciliationRequired(
                "terminal fill legs do not match immutable intent products/actions"
            )
        intent_leg = matches[0]
        unmatched.remove(intent_leg)
        ordered = _canonical_signed_decimal_text(
            fill_leg["ordered_quantity"], "terminal ordered quantity"
        )
        filled = _canonical_signed_decimal_text(
            fill_leg["filled_quantity"], "terminal filled quantity"
        )
        if (
            ordered != Decimal(intent_leg["quantity"])
            or ordered != ordered.to_integral_value()
            or filled != filled.to_integral_value()
        ):
            raise OrderIntentReconciliationRequired(
                "terminal fill quantities do not match immutable intent"
            )
        ordered_values.add(int(ordered))
        filled_values.add(int(filled))
    if unmatched or len(ordered_values) != 1 or len(filled_values) != 1:
        raise OrderIntentReconciliationRequired(
            "terminal fill leg quantities are ambiguous"
        )
    ordered_quantity = next(iter(ordered_values))
    filled_quantity = next(iter(filled_values))
    if (
        classification == "ZERO_FILL"
        and filled_quantity != 0
    ) or (
        classification == "FULL_FILL"
        and filled_quantity != ordered_quantity
    ):
        raise OrderIntentReconciliationRequired(
            "terminal fill quantity is partial or unresolved"
        )
    return classification, ordered_quantity, filled_quantity


def _terminal_position_lot_proof(
    capacity_result: dict[str, Any],
    *,
    broker_order_id: str,
    fill_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    _validate_capacity_manifest_result(capacity_result)
    if capacity_result["schema"] != "etrade-capacity.v2":
        raise OrderIntentReconciliationRequired(
            "full-fill absorption requires schema-v2 position lots"
        )
    if any(
        type(order) is dict
        and order.get("order_id") == broker_order_id
        for order in capacity_result["open_orders"]
    ):
        raise OrderIntentReconciliationRequired(
            "terminal order remains present in active orders"
        )
    fill_legs = {
        leg["leg_number"]: leg for leg in fill_summary["legs"]
    }
    matched_by_leg: dict[int, list[dict[str, Any]]] = {
        1: [],
        2: [],
    }
    proof: list[dict[str, Any]] = []
    for position in capacity_result["positions"]:
        target_lots = [
            lot
            for lot in position["lots"]
            if lot["order_no"] == broker_order_id
        ]
        if not target_lots:
            continue
        parent_quantity = _canonical_signed_decimal_text(
            position["quantity"], "target parent position quantity"
        )
        matched_parent_remaining = Decimal("0")
        for lot in target_lots:
            if lot["leg_no"] is None:
                raise OrderIntentReconciliationRequired(
                    "target position lot lacks an exact leg number"
                )
            leg_number = _canonical_unsigned_integer_text(
                lot["leg_no"],
                "target position lot leg number",
                positive=True,
            )
            if leg_number not in {1, 2}:
                raise OrderIntentReconciliationRequired(
                    "target position lot has an unsupported leg number"
                )
            fill_leg = fill_legs[leg_number]
            action = fill_leg["order_action"]
            expected_type = (
                "LONG" if action == "BUY_OPEN" else "SHORT"
            )
            expected_sign = Decimal("1") if action == "BUY_OPEN" else Decimal("-1")
            if (
                not _absorption_products_match(
                    position["product"], fill_leg["product"]
                )
                or position["position_type"] != expected_type
                or parent_quantity * expected_sign <= 0
            ):
                raise OrderIntentReconciliationRequired(
                    "target lot product/action direction is inconsistent"
                )
            original = _canonical_signed_decimal_text(
                lot["original_quantity"], "target lot original quantity"
            )
            remaining = _canonical_signed_decimal_text(
                lot["remaining_quantity"], "target lot remaining quantity"
            )
            available = _canonical_signed_decimal_text(
                lot["available_quantity"], "target lot available quantity"
            )
            if (
                original * expected_sign <= 0
                or remaining * expected_sign <= 0
                or available * expected_sign <= 0
                or abs(available) > abs(remaining)
                or abs(remaining) > abs(original)
            ):
                raise OrderIntentReconciliationRequired(
                    "target lot quantities do not prove intact opening exposure"
                )
            matched_parent_remaining += abs(remaining)
            item = {
                "leg_number": leg_number,
                "order_action": action,
                "filled_quantity": fill_leg["filled_quantity"],
                "position_id": position["position_id"],
                "position_type": position["position_type"],
                "position_quantity": position["quantity"],
                "product": position["product"],
                "lot": lot,
            }
            matched_by_leg[leg_number].append(item)
            proof.append(item)
        if abs(parent_quantity) < matched_parent_remaining:
            raise OrderIntentReconciliationRequired(
                "target lots exceed their parent position quantity"
            )
    if {number for number, items in matched_by_leg.items() if items} != {
        1,
        2,
    }:
        raise OrderIntentReconciliationRequired(
            "post-fill positions lack exact lots for both spread legs"
        )
    for leg_number, items in matched_by_leg.items():
        expected = _canonical_signed_decimal_text(
            fill_legs[leg_number]["filled_quantity"],
            "full-fill quantity",
        )
        for quantity_name in (
            "original_quantity",
            "remaining_quantity",
            "available_quantity",
        ):
            actual = sum(
                (
                    abs(
                        _canonical_signed_decimal_text(
                            item["lot"][quantity_name],
                            f"target lot {quantity_name}",
                        )
                    )
                    for item in items
                ),
                Decimal("0"),
            )
            if actual != expected:
                raise OrderIntentReconciliationRequired(
                    "post-fill target lot quantities do not equal exact fills"
                )
    proof.sort(key=_canonical_read_json)
    return proof


def _absorption_products_match(
    position_product: dict[str, Any],
    fill_product: dict[str, Any],
) -> bool:
    economic_keys = (
        "symbol",
        "security_type",
        "call_put",
        "expiry_year",
        "expiry_month",
        "expiry_day",
        "strike_price",
    )
    if any(
        position_product[key] != fill_product[key]
        for key in economic_keys
    ):
        return False
    position_id = position_product["product_id"]
    fill_id = fill_product["product_id"]
    return (
        position_id is None
        or fill_id is None
        or position_id == fill_id
    )


def _validate_capacity_manifest_result(
    result: dict[str, Any],
    *,
    expected_account_id: str | None = None,
) -> None:
    expected = {
        "schema",
        "account_status",
        "account_mode",
        "account_type",
        "broker_buying_power",
        "broker_buying_power_as_of",
        "positions",
        "open_orders",
        "state_sha256",
    }
    if set(result) != expected:
        raise OrderIntentIntegrityError(
            "capacity manifest result shape is invalid"
        )
    schema = result["schema"]
    if (
        schema not in {"etrade-capacity.v1", "etrade-capacity.v2"}
        or result["account_status"] != "ACTIVE"
        or result["account_mode"] != "MARGIN"
        or type(result["account_type"]) is not str
        or not result["account_type"]
        or type(result["positions"]) is not list
        or type(result["open_orders"]) is not list
        or type(result["broker_buying_power_as_of"]) is not str
        or len(result["broker_buying_power_as_of"]) != 13
        or not result["broker_buying_power_as_of"].isdigit()
    ):
        raise OrderIntentIntegrityError(
            "capacity manifest is not eligible for margin opening risk"
        )
    buying_power = result["broker_buying_power"]
    if (
        type(buying_power) is not str
        or _canonical_amount(buying_power) != buying_power
    ):
        raise OrderIntentIntegrityError(
            "capacity manifest buying power is not canonical"
        )
    _validate_sha256("capacity state digest", result["state_sha256"])
    if schema == "etrade-capacity.v2":
        _validate_capacity_v2_positions(result["positions"])
        if expected_account_id is not None and any(
            position["account_id"] != expected_account_id
            for position in result["positions"]
        ):
            raise OrderIntentIntegrityError(
                "capacity position belongs to a different account"
            )
    state = {
        key: result[key]
        for key in expected
        if key not in {"state_sha256", "broker_buying_power_as_of"}
    }
    expected_digest = _domain_json_hash(
        (
            b"etrade-capacity-state.v2\0"
            if schema == "etrade-capacity.v2"
            else b"etrade-capacity-state.v1\0"
        ),
        state,
    )
    if not hmac.compare_digest(expected_digest, result["state_sha256"]):
        raise OrderIntentIntegrityError(
            "capacity manifest state digest does not verify"
        )


def _validate_order_query_manifest_result(
    result: dict[str, Any],
) -> None:
    expected = {
        "schema",
        "broker_order_id",
        "raw_status",
        "outcome",
        "order_payload_hashes",
        "http_status",
        "raw_response_digest",
        "not_found",
        "replacement_links",
    }
    schema = result.get("schema")
    if schema == "etrade-order-query.v2":
        expected.add("fill_summary")
    if set(result) != expected:
        raise OrderIntentIntegrityError(
            "order query manifest result shape is invalid"
        )
    if (
        schema not in {
            "etrade-order-query.v1",
            "etrade-order-query.v2",
        }
        or type(result["broker_order_id"]) is not str
        or type(result["raw_status"]) is not str
        or result["outcome"]
        not in {
            "OPEN",
            "FILLED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
            "UNRESOLVED",
        }
        or type(result["order_payload_hashes"]) is not list
        or type(result["http_status"]) is not int
        or not 100 <= result["http_status"] <= 599
        or type(result["not_found"]) is not bool
        or type(result["replacement_links"]) is not dict
    ):
        raise OrderIntentIntegrityError(
            "order query manifest contains invalid typed values"
        )
    _validate_identity("manifest broker order id", result["broker_order_id"])
    _validate_sha256(
        "order query raw response digest",
        result["raw_response_digest"],
    )
    hashes = result["order_payload_hashes"]
    if (
        len(hashes) > 2
        or len(set(hashes)) != len(hashes)
        or hashes != sorted(hashes)
    ):
        raise OrderIntentIntegrityError(
            "order query payload hashes are ambiguous"
        )
    for payload_hash in hashes:
        _validate_sha256("order query payload hash", payload_hash)
    replacement_links = result["replacement_links"]
    if schema == "etrade-order-query.v2":
        _validate_fill_summary_shape(result["fill_summary"])
        classification = result["fill_summary"]["classification"]
        if result["not_found"]:
            expected_outcome = "UNRESOLVED"
        elif classification == "OPEN":
            expected_outcome = "OPEN"
        elif classification == "FULL_FILL":
            expected_outcome = "FILLED"
        elif classification == "ZERO_FILL_TERMINAL":
            expected_outcome = result["raw_status"]
            if expected_outcome not in {
                "CANCELLED",
                "REJECTED",
                "EXPIRED",
            }:
                raise OrderIntentIntegrityError(
                    "zero-fill classification lacks a terminal broker status"
                )
        else:
            expected_outcome = "UNRESOLVED"
        if result["outcome"] != expected_outcome:
            raise OrderIntentIntegrityError(
                "order outcome conflicts with its exact fill classification"
            )
    if result["not_found"]:
        if replacement_links:
            raise OrderIntentIntegrityError(
                "negative order evidence cannot contain replacement links"
            )
    else:
        expected_replacement_keys = {
            "replaces_order_id",
            "replaced_by_order_id",
        }
        if set(replacement_links) != expected_replacement_keys:
            raise OrderIntentIntegrityError(
                "order replacement-link shape is invalid"
            )
        for name, broker_order_id in replacement_links.items():
            if broker_order_id is not None:
                _validate_identity(name, broker_order_id)
        if (
            any(
                broker_order_id is not None
                for broker_order_id in replacement_links.values()
            )
            and result["outcome"] != "UNRESOLVED"
        ):
            raise OrderIntentIntegrityError(
                "replacement-linked order evidence must stay unresolved"
            )
    if result["not_found"]:
        if (
            result["http_status"] != 404
            or result["outcome"] != "UNRESOLVED"
            or result["raw_status"] != "NOT_FOUND"
            or hashes
        ):
            raise OrderIntentIntegrityError(
                "negative order evidence is inconsistent"
            )
    elif (
        result["http_status"] != 200
        or (
            result["outcome"] != "UNRESOLVED"
            and not hashes
        )
    ):
        raise OrderIntentIntegrityError(
            "resolved order evidence is incomplete"
        )


def _validate_mapping_keys(value: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    keys = list(value.keys())
    if any(type(key) is not str for key in keys):
        raise OrderIntentValidationError(f"{context} keys must all be strings")
    unexpected = set(keys) - allowed
    if unexpected:
        raise OrderIntentValidationError(f"{context} contains disallowed fields: {', '.join(sorted(unexpected))}")


def _validate_broker_order_shape(payload: Mapping[str, Any]) -> None:
    security_type = payload.get("securityType")
    if security_type not in {"EQ", "OPTN"}:
        raise OrderIntentValidationError("securityType must be EQ or OPTN")
    if payload.get("priceType") not in {"MARKET", "LIMIT", "NET_CREDIT", "NET_DEBIT"}:
        raise OrderIntentValidationError("unsupported priceType")
    if payload.get("orderTerm") != "GOOD_FOR_DAY":
        raise OrderIntentValidationError("only GOOD_FOR_DAY orderTerm is supported")
    has_legs = "legs" in payload
    if security_type == "EQ":
        raise OrderIntentValidationError(
            "equity orders are unsupported until position-aware exposure evidence is implemented"
        )
    elif has_legs:
        if payload.get("orderAction") != "SPREAD" or payload.get("spreadType") != "VERTICAL":
            raise OrderIntentValidationError("only VERTICAL option spreads are supported")
        legs = payload["legs"]
        if len(legs) != 2:
            raise OrderIntentValidationError("vertical spread must contain exactly two legs")
        ignored_cross_shape = {"symbol", "quantity", "callPut", "expiryYear", "expiryMonth", "expiryDay", "strikePrice"} & set(payload)
        if ignored_cross_shape:
            raise OrderIntentValidationError("spread payload must derive identity exclusively from legs")
        first = legs[0]
        reference = (first.get("symbol"), first.get("expiryYear"), first.get("expiryMonth"), first.get("expiryDay"), first.get("callPut"), first.get("quantity"))
        for leg in legs:
            required = {"symbol", "callPut", "expiryYear", "expiryMonth", "expiryDay", "strikePrice", "orderAction", "quantity"}
            if not required.issubset(leg) or leg.get("callPut") not in {"PUT", "CALL"} or leg.get("orderAction") not in {"BUY_OPEN", "SELL_OPEN", "BUY_CLOSE", "SELL_CLOSE"}:
                raise OrderIntentValidationError("invalid spread leg")
            _validate_expiry(leg["expiryYear"], leg["expiryMonth"], leg["expiryDay"])
            _require_positive_number(leg["strikePrice"], "strikePrice")
            _require_positive_integral(leg["quantity"], "quantity")
            if (leg.get("symbol"), leg.get("expiryYear"), leg.get("expiryMonth"), leg.get("expiryDay"), leg.get("callPut"), leg.get("quantity")) != reference:
                raise OrderIntentValidationError("vertical spread legs must share symbol, expiry, callPut, and quantity")
        strikes = {leg["strikePrice"] for leg in legs}
        sides = {leg["orderAction"].split("_")[0] for leg in legs}
        exposure = {leg["orderAction"].split("_")[1] for leg in legs}
        if len(strikes) != 2 or sides != {"BUY", "SELL"} or len(exposure) != 1:
            raise OrderIntentValidationError("vertical spread requires distinct strikes, one buy/one sell, and uniform OPEN/CLOSE exposure")
        if exposure == {"OPEN"}:
            if payload["priceType"] not in {"NET_CREDIT", "NET_DEBIT"} or "limitPrice" not in payload:
                raise OrderIntentValidationError("opening verticals require bounded NET_CREDIT or NET_DEBIT pricing")
            with localcontext() as decimal_context:
                decimal_context.prec = _DECIMAL_PRECISION
                width = abs(Decimal(str(legs[0]["strikePrice"])) - Decimal(str(legs[1]["strikePrice"])))
                limit_price = Decimal(str(payload["limitPrice"]))
            sell_leg = next(leg for leg in legs if leg["orderAction"] == "SELL_OPEN")
            buy_leg = next(leg for leg in legs if leg["orderAction"] == "BUY_OPEN")
            short_risk = (sell_leg["callPut"] == "PUT" and sell_leg["strikePrice"] > buy_leg["strikePrice"]) or (sell_leg["callPut"] == "CALL" and sell_leg["strikePrice"] < buy_leg["strikePrice"])
            expected_price_type = "NET_CREDIT" if short_risk else "NET_DEBIT"
            if payload["priceType"] != expected_price_type:
                raise OrderIntentValidationError("opening vertical price type does not match leg risk orientation")
            if payload["priceType"] == "NET_CREDIT" and not (Decimal("0") <= limit_price < width):
                raise OrderIntentValidationError("opening vertical credit must be non-negative and strictly less than strike width")
            if payload["priceType"] == "NET_DEBIT" and not (Decimal("0") < limit_price <= width):
                raise OrderIntentValidationError("opening vertical debit must be positive and no greater than strike width")
    else:
        required = {"symbol", "quantity", "orderAction", "callPut", "expiryYear", "expiryMonth", "expiryDay", "strikePrice", "priceType", "orderTerm"}
        if not required.issubset(payload) or payload.get("orderAction") not in {"BUY_OPEN", "SELL_OPEN", "BUY_CLOSE", "SELL_CLOSE"} or payload.get("callPut") not in {"PUT", "CALL"} or type(payload.get("symbol")) is not str or not payload["symbol"].strip() or "spreadType" in payload:
            raise OrderIntentValidationError("invalid single-option order shape")
        _require_positive_integral(payload["quantity"], "quantity")
        _validate_expiry(payload["expiryYear"], payload["expiryMonth"], payload["expiryDay"])
        _require_positive_number(payload["strikePrice"], "strikePrice")
    if payload["priceType"] != "MARKET":
        if "limitPrice" not in payload:
            raise OrderIntentValidationError("non-market orders require limitPrice")
        if payload["priceType"] == "NET_CREDIT":
            _require_nonnegative_number(payload["limitPrice"], "limitPrice")
        else:
            _require_positive_number(payload["limitPrice"], "limitPrice")


def _derive_intent_kind(payload: Mapping[str, Any]) -> Literal["OPENING", "CLOSING"]:
    actions = [leg["orderAction"] for leg in payload.get("legs", [])] or [payload["orderAction"]]
    normalized = {"BUY_OPEN" if action == "BUY" else "SELL_CLOSE" if action == "SELL" else action for action in actions}
    opening = {action.endswith("_OPEN") for action in normalized}
    if len(opening) != 1:
        raise OrderIntentValidationError("mixed opening and closing exposure is unsupported")
    return "OPENING" if True in opening else "CLOSING"


def _validate_reprice_only_amendment(original_wire_payload: str, amendment_wire_payload: str) -> None:
    """Permit a durable broker amendment to change only the limit price."""
    original = json.loads(original_wire_payload)
    amendment = json.loads(amendment_wire_payload)
    if original.get("priceType") == "MARKET" or amendment.get("priceType") == "MARKET":
        raise OrderIntentValidationError("market orders cannot be repriced")
    original.pop("limitPrice", None)
    amendment.pop("limitPrice", None)
    if original != amendment:
        raise OrderIntentValidationError("amendment may only change immutable order intent limitPrice")


def _opening_exposure_floor(payload: Mapping[str, Any]) -> Decimal:
    """Return the minimum defensible reserve for an opening vertical spread.

    Other opening order forms lack position-aware collateral semantics in this
    ledger and are intentionally fail-closed rather than accepting a caller
    supplied number that could understate live exposure.
    """
    if payload.get("orderAction") != "SPREAD" or payload.get("spreadType") != "VERTICAL" or "legs" not in payload:
        raise OrderIntentReservationError("opening reservations currently require a validated vertical spread")
    legs = payload["legs"]
    if _derive_intent_kind(payload) != "OPENING" or len(legs) != 2:
        raise OrderIntentReservationError("opening exposure floor requires exactly two opening vertical legs")
    quantity = legs[0]["quantity"]
    with localcontext() as decimal_context:
        decimal_context.prec = _DECIMAL_PRECISION
        width = abs(Decimal(str(legs[0]["strikePrice"])) - Decimal(str(legs[1]["strikePrice"])))
        gross = width * Decimal("100") * Decimal(quantity)
        limit_price = Decimal(str(payload["limitPrice"]))
        if payload["priceType"] == "NET_CREDIT":
            if not (Decimal("0") <= limit_price < width):
                raise OrderIntentValidationError("vertical credit must be non-negative and strictly less than strike width")
            return gross - limit_price * Decimal("100") * Decimal(quantity)
        if payload["priceType"] == "NET_DEBIT":
            if not (Decimal("0") < limit_price <= width):
                raise OrderIntentValidationError("vertical debit must be positive and no greater than strike width")
            return limit_price * Decimal("100") * Decimal(quantity)
    raise OrderIntentReservationError("opening vertical exposure requires NET_CREDIT or NET_DEBIT pricing")


def _require_positive_integral(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise OrderIntentValidationError(f"{name} must be a positive integer")


def _require_positive_number(value: Any, name: str) -> None:
    if type(value) not in {int, float, Decimal}:
        raise OrderIntentValidationError(f"{name} must be a positive finite number")
    if (type(value) is Decimal and (not value.is_finite() or value <= 0)) or (
        type(value) in {int, float} and (not math.isfinite(value) or value <= 0)
    ):
        raise OrderIntentValidationError(f"{name} must be a positive finite number")


def _require_nonnegative_number(value: Any, name: str) -> None:
    if type(value) not in {int, float, Decimal}:
        raise OrderIntentValidationError(f"{name} must be a non-negative finite number")
    if (type(value) is Decimal and (not value.is_finite() or value < 0)) or (
        type(value) in {int, float} and (not math.isfinite(value) or value < 0)
    ):
        raise OrderIntentValidationError(f"{name} must be a non-negative finite number")


def _validate_expiry(year: Any, month: Any, day: Any) -> None:
    if any(type(value) is not int for value in (year, month, day)):
        raise OrderIntentValidationError("expiry must use integral year/month/day")
    try:
        date(year, month, day)
    except ValueError as exc:
        raise OrderIntentValidationError("expiry is not a valid calendar date") from exc


def _canonical_value(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, bool}:
        return value
    if value_type is int:
        return {"__number__": str(value)}
    if value_type is Decimal:
        return {"__number__": _canonical_amount(value)}
    if value_type is float:
        if not math.isfinite(value):
            raise OrderIntentValidationError("payload cannot contain non-finite float")
        return {"__number__": _canonical_amount(value)}
    if value_type is dict:
        keys = list(value.keys())
        if any(type(key) is not str for key in keys):
            raise OrderIntentValidationError("payload keys must all be strings")
        return {key: _canonical_value(value[key]) for key in sorted(keys)}
    if value_type in {list, tuple}:
        return [_canonical_value(item) for item in value]
    raise OrderIntentValidationError(f"unsupported payload value type {type(value).__name__}")


def _wire_value(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return value
    if value_type is Decimal:
        if not value.is_finite():
            raise OrderIntentValidationError("payload cannot contain non-finite Decimal")
        return float(value)
    if value_type is float:
        if not math.isfinite(value):
            raise OrderIntentValidationError("payload cannot contain non-finite float")
        return value
    if value_type is dict:
        keys = list(value.keys())
        if any(type(key) is not str for key in keys):
            raise OrderIntentValidationError("payload keys must all be strings")
        return {key: _wire_value(value[key]) for key in sorted(keys)}
    if value_type in {list, tuple}:
        return [_wire_value(item) for item in value]
    raise OrderIntentValidationError(f"unsupported payload value type {type(value).__name__}")


def _canonical_amount(value: Decimal | int | float | str) -> str:
    if type(value) not in {Decimal, int, float, str}:
        raise OrderIntentValidationError("amount must use an exact decimal-compatible primitive type")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise OrderIntentValidationError("amount must be a finite decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise OrderIntentValidationError("amount must be non-negative and finite")
    normalized = format(parsed, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _validate_identity(name: str, value: str) -> None:
    if type(value) is not str or not value or len(value) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise OrderIntentValidationError(f"{name} must be a printable non-empty ASCII string up to 256 characters")


def _validate_fencing_token(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise OrderIntentValidationError("fencing_token must be a positive exact integer")


def _validate_environment(value: str) -> None:
    if type(value) is not str or value not in _ENVIRONMENTS:
        raise OrderIntentValidationError("environment must be sandbox or production")


def _validate_sha256(name: str, value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise OrderIntentValidationError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_timestamp(value: datetime) -> None:
    if type(value) is not datetime or value.tzinfo is not timezone.utc:
        raise OrderIntentValidationError("timestamp must be an exact UTC datetime")


def _to_us(value: datetime) -> int:
    return int(value.astimezone(timezone.utc).timestamp() * 1_000_000)


def _from_us(value: int) -> datetime:
    return datetime.fromtimestamp(int(value) / 1_000_000, tz=timezone.utc)
