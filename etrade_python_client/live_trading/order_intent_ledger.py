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


SCHEMA_VERSION = 10
_MIGRATABLE_SCHEMA_VERSIONS = frozenset({8, 9})
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


@dataclass(frozen=True)
class RiskEvidence:
    decision_id: str
    max_loss_amount: Decimal
    collateral_amount: Decimal
    quote_observed_at: datetime
    quote_digest: str
    portfolio_observed_at: datetime
    portfolio_snapshot_digest: str

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


@dataclass(frozen=True)
class AccountCapacityEvidence:
    """Broker capacity snapshot whose digest covers positions and capacity fields."""

    account_id: str
    environment: str
    broker_buying_power: Decimal
    risk_budget: Decimal
    observed_at: datetime
    portfolio_snapshot_digest: str

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

    def set_reservation_cap(self, evidence: AccountCapacityEvidence) -> Decimal:
        if type(evidence) is not AccountCapacityEvidence:
            raise OrderIntentValidationError("set_reservation_cap requires typed AccountCapacityEvidence")
        now = self._now_us()
        AccountCapacityEvidence.validate(evidence, _from_us(now))
        cap = _canonical_amount(min(evidence.broker_buying_power, evidence.risk_budget))
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM reservation_caps
                WHERE account_id = ? AND environment = ?
                """,
                (evidence.account_id, evidence.environment),
            ).fetchone()
            observed_at = _to_us(evidence.observed_at)
            if existing is not None and observed_at == int(existing["observed_at"]):
                expected = (
                    cap,
                    _canonical_amount(evidence.broker_buying_power),
                    _canonical_amount(evidence.risk_budget),
                    evidence.portfolio_snapshot_digest,
                )
                actual = (
                    existing["cap_amount"],
                    existing["broker_buying_power"],
                    existing["risk_budget"],
                    existing["portfolio_snapshot_digest"],
                )
                if actual == expected:
                    return Decimal(existing["cap_amount"])
                raise OrderIntentIntegrityError(
                    "equal-time capacity evidence conflicts with the persisted snapshot"
                )
            if existing is not None and observed_at < int(existing["observed_at"]):
                raise OrderIntentIntegrityError(
                    "capacity evidence must not predate the persisted account snapshot"
                )
            conn.execute(
                """
                INSERT INTO reservation_caps (account_id, environment, cap_amount, broker_buying_power, risk_budget, observed_at, portfolio_snapshot_digest, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, environment) DO UPDATE SET
                    cap_amount = excluded.cap_amount, broker_buying_power = excluded.broker_buying_power,
                    risk_budget = excluded.risk_budget, observed_at = excluded.observed_at,
                    portfolio_snapshot_digest = excluded.portfolio_snapshot_digest, updated_at = excluded.updated_at
                """,
                (evidence.account_id, evidence.environment, cap, _canonical_amount(evidence.broker_buying_power), _canonical_amount(evidence.risk_budget), observed_at, evidence.portfolio_snapshot_digest, now),
            )
        return Decimal(cap)

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
                "SELECT cap_amount, observed_at, portfolio_snapshot_digest FROM reservation_caps WHERE account_id = ? AND environment = ?",
                (intent["account_id"], intent["environment"]),
            ).fetchone()
            if cap is None:
                raise OrderIntentReservationError("opening reservations require an account/environment cap")
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
                    quote_observed_at, quote_digest, portfolio_observed_at, portfolio_snapshot_digest, state, released_reason_code, created_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', NULL, ?, NULL)
                """,
                (intent_id, intent["account_id"], intent["environment"], value, evidence.decision_id, _canonical_amount(evidence.max_loss_amount), _to_us(evidence.quote_observed_at), evidence.quote_digest, _to_us(evidence.portfolio_observed_at), evidence.portfolio_snapshot_digest, now),
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

    def absorb_filled_reservation(
        self, intent_id: str, evidence: AccountCapacityEvidence
    ) -> MarginReservation:
        """Fail closed until R7b can prove a terminal order's position effect.

        A newer or merely different account digest cannot prove that the
        snapshot incorporated this specific order, especially after a partial
        fill. Keeping the reservation blocks new exposure without guessing.
        """
        _validate_identity("intent_id", intent_id)
        if type(evidence) is not AccountCapacityEvidence:
            raise OrderIntentValidationError("absorb_filled_reservation requires typed AccountCapacityEvidence")
        AccountCapacityEvidence.validate(evidence, _from_us(self._now_us()))
        raise OrderIntentReconciliationRequired(
            "terminal reservations require R7b position-level absorption evidence"
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
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
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
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
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
            metadata_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'ledger_metadata'"
            ).fetchone()
            if metadata_table is None:
                mode = conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
                if str(mode).lower() != "delete":
                    raise OrderIntentLedgerError("ledger requires SQLite DELETE journaling in its private directory")
                conn.execute("CREATE TABLE ledger_metadata (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), schema_version INTEGER NOT NULL)")
            conn.execute("INSERT OR IGNORE INTO ledger_metadata (singleton, schema_version) VALUES (1, ?)", (SCHEMA_VERSION,))
            metadata = conn.execute("SELECT schema_version FROM ledger_metadata WHERE singleton = 1").fetchone()
            current_schema_version = int(metadata["schema_version"])
            if current_schema_version != SCHEMA_VERSION and current_schema_version not in _MIGRATABLE_SCHEMA_VERSIONS:
                raise OrderIntentLedgerError(f"unsupported ledger schema {metadata['schema_version']}; expected {SCHEMA_VERSION}")
            conn.executescript(
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
                    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'FILLED_PENDING_ABSORPTION', 'RELEASED')),
                    released_reason_code TEXT,
                    created_at INTEGER NOT NULL,
                    released_at INTEGER,
                    CHECK (CAST(amount AS REAL) > 0 AND CAST(max_loss_amount AS REAL) > 0),
                    CHECK (length(quote_digest) = 64 AND length(portfolio_snapshot_digest) = 64),
                    CHECK ((state IN ('ACTIVE', 'FILLED_PENDING_ABSORPTION')) = (released_reason_code IS NULL AND released_at IS NULL))
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
                    created_at INTEGER NOT NULL,
                    CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
                    CHECK (raw_response_digest IS NULL OR length(raw_response_digest) = 64)
                );
                CREATE INDEX IF NOT EXISTS idx_intents_account_environment_state ON order_intents(account_id, environment, state);
                CREATE INDEX IF NOT EXISTS idx_reservations_account_environment ON margin_reservations(account_id, environment, state);
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
            if current_schema_version != SCHEMA_VERSION:
                conn.execute("UPDATE ledger_metadata SET schema_version = ? WHERE singleton = 1", (SCHEMA_VERSION,))
            self._secure_sqlite_sidecars()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_secure_parent()
        self._validate_database_file()
        conn = sqlite3.connect(str(self.path), timeout=_BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
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
        cap = conn.execute("SELECT cap_amount, observed_at, portfolio_snapshot_digest FROM reservation_caps WHERE account_id = ? AND environment = ?", (intent["account_id"], intent["environment"])).fetchone()
        if reservation is None or reservation["state"] != "ACTIVE" or cap is None:
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

    @staticmethod
    def _validate_broker_evidence(conn: sqlite3.Connection, intent: sqlite3.Row, evidence: BrokerEvidence, *, allowed_operations: set[str], allowed_outcomes: set[str]) -> None:
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

    @staticmethod
    def _active_reservation_total(conn: sqlite3.Connection, account_id: str, environment: str) -> Decimal:
        rows = conn.execute("SELECT amount FROM margin_reservations WHERE account_id = ? AND environment = ? AND state IN ('ACTIVE', 'FILLED_PENDING_ABSORPTION')", (account_id, environment)).fetchall()
        with localcontext() as decimal_context:
            decimal_context.prec = _DECIMAL_PRECISION
            return sum((Decimal(row["amount"]) for row in rows), Decimal("0"))

    def _release_reservation(self, conn: sqlite3.Connection, intent_id: str, reason_code: str, now: int) -> None:
        reservation = conn.execute("SELECT * FROM margin_reservations WHERE intent_id = ?", (intent_id,)).fetchone()
        if reservation is None or reservation["state"] != "ACTIVE":
            return
        conn.execute("UPDATE margin_reservations SET state = 'RELEASED', released_reason_code = ?, released_at = ? WHERE intent_id = ?", (reason_code, now, intent_id))
        intent = self._require_intent(conn, intent_id)
        self._append_event(conn, intent_id, "RESERVATION_RELEASED", intent["state"], intent["state"], "system", "RESERVATION_RELEASED", now, broker_order_id=intent["broker_order_id"])

    def _append_reconciliation_event(self, conn: sqlite3.Connection, intent_id: str, from_state: str, to_state: str, event_type: str, reason_code: str, evidence: BrokerEvidence, now: int) -> None:
        self._append_event(conn, intent_id, event_type, from_state, to_state, "broker-evidence", reason_code, now, broker_status=evidence.outcome, broker_order_id=evidence.broker_order_id, observed_at=_to_us(evidence.observed_at), evidence_operation=evidence.operation, http_status=evidence.http_status, raw_response_digest=evidence.raw_response_digest)

    def _append_event(self, conn: sqlite3.Connection, intent_id: str, event_type: str, from_state: str | None, to_state: str | None, actor: str, reason_code: str, now: int, *, broker_status: str | None = None, broker_order_id: str | None = None, observed_at: int | None = None, evidence_operation: str | None = None, http_status: int | None = None, raw_response_digest: str | None = None) -> None:
        if reason_code not in _REASON_CODES:
            raise OrderIntentValidationError("reason_code is not allowlisted")
        _validate_identity("actor", actor)
        identity = conn.execute("SELECT account_id, environment, client_order_id FROM order_intents WHERE intent_id = ?", (intent_id,)).fetchone()
        if identity is None:
            raise OrderIntentValidationError("cannot append event for unknown intent")
        conn.execute(
            """
            INSERT INTO order_events (intent_id, account_id, environment, client_order_id, event_type, from_state, to_state, actor, reason_code, broker_status, broker_order_id, observed_at, evidence_operation, http_status, raw_response_digest, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (intent_id, identity["account_id"], identity["environment"], identity["client_order_id"], event_type, from_state, to_state, actor, reason_code, broker_status, broker_order_id, observed_at, evidence_operation, http_status, raw_response_digest, now),
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
        return MarginReservation(intent_id=row["intent_id"], account_id=row["account_id"], environment=row["environment"], amount=Decimal(row["amount"]), risk_decision_id=row["risk_decision_id"], max_loss_amount=Decimal(row["max_loss_amount"]), quote_observed_at=_from_us(row["quote_observed_at"]), quote_digest=row["quote_digest"], portfolio_observed_at=_from_us(row["portfolio_observed_at"]), portfolio_snapshot_digest=row["portfolio_snapshot_digest"], state=row["state"], released_reason_code=row["released_reason_code"], created_at=_from_us(row["created_at"]), released_at=_from_us(row["released_at"]) if row["released_at"] is not None else None)

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> IntentEvent:
        return IntentEvent(sequence=int(row["sequence"]), intent_id=row["intent_id"], account_id=row["account_id"], environment=row["environment"], client_order_id=row["client_order_id"], event_type=row["event_type"], from_state=row["from_state"], to_state=row["to_state"], actor=row["actor"], reason_code=row["reason_code"], broker_status=row["broker_status"], broker_order_id=row["broker_order_id"], observed_at=_from_us(row["observed_at"]) if row["observed_at"] is not None else None, evidence_operation=row["evidence_operation"], http_status=row["http_status"], raw_response_digest=row["raw_response_digest"], created_at=_from_us(row["created_at"]))


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
