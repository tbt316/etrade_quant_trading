"""Durable coordinator for the hardened E*TRADE mutation transport.

The gateway owns orchestration only.  The transport owns exact XML, the
single broker exchange, and durable response persistence.  The ledger owns
all state transitions and replay fences.  Broker reads are injected through
an explicitly read-only protocol and can never authorize a mutation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from live_trading.etrade_broker_transport import (
    BrokerReply,
    ETradeBrokerTransport,
    ETradeBrokerTransportError,
    SelectedBrokerAccount,
)
from live_trading.order_intent_ledger import (
    AccountCapacityEvidence,
    BrokerEvidence,
    IntentRecord,
    OrderIntent,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLedgerError,
    OrderIntentReconciliationRequired,
    OrderIntentTransitionError,
    RiskEvidence,
    TransportResponseReceipt,
    canonical_order_payload_hash,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary, RuntimeSafetyError


_MAX_COMMAND_BYTES = 32 * 1024
_MAX_EVIDENCE_AGE_SECONDS = 300
_MAX_LEASE_SECONDS = 300


class EtradeOrderGatewayError(RuntimeError):
    """Base error for the durable order coordinator."""


class GatewayValidationError(EtradeOrderGatewayError):
    """A command or typed read response violated the closed contract."""


class GatewayReconciliationRequired(EtradeOrderGatewayError):
    """New mutations are blocked by unresolved durable broker work."""


@dataclass(frozen=True)
class BrokerCapacitySnapshot:
    """Complete account risk snapshot produced by a read-only adapter."""

    account: SelectedBrokerAccount
    environment: Literal["sandbox", "production"]
    broker_buying_power: Decimal
    observed_at: datetime
    portfolio_snapshot_digest: str
    positions_complete: bool
    open_orders_complete: bool
    source_response_digests: tuple[str, ...]


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    """Exact normalized status for one known broker order id."""

    account: SelectedBrokerAccount
    environment: Literal["sandbox", "production"]
    broker_order_id: str = field(repr=False)
    outcome: Literal[
        "OPEN",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
        "UNRESOLVED",
    ]
    observed_at: datetime
    http_status: int
    raw_response_digest: str
    order_payload_hash: str
    complete: bool


@dataclass(frozen=True)
class BrokerOrderNotFound:
    """Complete negative lookup for one known broker order id."""

    account: SelectedBrokerAccount
    environment: Literal["sandbox", "production"]
    broker_order_id: str = field(repr=False)
    observed_at: datetime
    http_status: int
    raw_response_digest: str
    complete: bool


@runtime_checkable
class EtradeBrokerReader(Protocol):
    """Read-only evidence required by the order coordinator."""

    def assert_gateway_binding(
        self, runtime_safety: RuntimeSafetyBoundary
    ) -> None: ...

    def selected_account(self) -> SelectedBrokerAccount: ...

    def read_capacity(
        self, account: SelectedBrokerAccount
    ) -> BrokerCapacitySnapshot: ...

    def query_order(
        self,
        account: SelectedBrokerAccount,
        broker_order_id: str,
    ) -> BrokerOrderSnapshot | BrokerOrderNotFound: ...


@dataclass(frozen=True)
class SubmitOpeningCommand:
    strategy_id: str
    decision_id: str
    idempotency_scope: str
    idempotency_key: str
    payload_bytes: bytes = field(repr=False)
    max_loss_amount: Decimal
    collateral_amount: Decimal
    quote_observed_at: datetime
    quote_digest: str
    owner: str
    lease_seconds: int = 30


@dataclass(frozen=True)
class RepriceOpeningCommand:
    intent_id: str
    idempotency_key: str
    payload_bytes: bytes = field(repr=False)
    owner: str
    lease_seconds: int = 30


@dataclass(frozen=True)
class GatewayMutationResult:
    intent_id: str
    client_order_id: str = field(repr=False)
    state: Literal[
        "SUBMITTED", "SUBMISSION_UNKNOWN", "AMENDMENT_UNKNOWN", "FAILED"
    ]
    created: bool
    broker_order_id: str | None = field(repr=False)
    preview_id: str | None = field(repr=False)
    reason_code: str | None


@dataclass(frozen=True)
class _ReconciliationContext:
    operation: Literal["ORDER_QUERY", "AMEND_QUERY"]
    client_order_id: str = field(repr=False)
    broker_order_id: str = field(repr=False)
    expected_order_payload_hash: str


class EtradeOrderGateway:
    """Compose runtime arming, durable state, typed reads, and one transport."""

    def __init__(
        self,
        *,
        runtime_safety: RuntimeSafetyBoundary,
        ledger: OrderIntentLedger,
        transport: ETradeBrokerTransport,
        reader: EtradeBrokerReader,
        opening_risk_budget: Decimal,
        clock=None,
    ) -> None:
        if type(runtime_safety) is not RuntimeSafetyBoundary:
            raise GatewayValidationError(
                "runtime_safety must be an exact RuntimeSafetyBoundary"
            )
        if type(ledger) is not OrderIntentLedger:
            raise GatewayValidationError(
                "ledger must be an exact OrderIntentLedger"
            )
        if type(transport) is not ETradeBrokerTransport:
            raise GatewayValidationError(
                "gateway requires the exact hardened ETradeBrokerTransport"
            )
        if not isinstance(reader, EtradeBrokerReader):
            raise GatewayValidationError(
                "reader does not implement the read-only broker protocol"
            )
        try:
            transport.assert_gateway_binding(ledger, runtime_safety)
            reader.assert_gateway_binding(runtime_safety)
        except (ETradeBrokerTransportError, RuntimeSafetyError) as exc:
            raise GatewayValidationError(
                "gateway adapters and runtime safety boundary do not match"
            ) from exc
        _exact_decimal(
            opening_risk_budget,
            "opening_risk_budget",
            allow_zero=True,
        )
        self.runtime_safety = runtime_safety
        self.ledger = ledger
        self.transport = transport
        self.reader = reader
        self._opening_risk_budget = opening_risk_budget
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._started = False
        self._account: SelectedBrokerAccount | None = None

    def start(self) -> None:
        """Reconcile every resolvable prior order before enabling mutation."""

        self._started = False
        account = self._checked_account()
        self._account = account
        blockers = self.ledger.reconciliation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        for blocker in blockers:
            self._reconcile_once(blocker, account)
        remaining = self.ledger.reconciliation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        unabsorbed = self.ledger.unabsorbed_filled_reservation_count(
            account.account_id, self.runtime_safety.environment
        )
        if remaining or unabsorbed:
            raise GatewayReconciliationRequired(
                "gateway remains read-only: "
                f"{len(remaining)} broker operation(s), "
                f"{unabsorbed} unabsorbed fill reservation(s)"
            )
        self._checked_account()
        self._started = True

    def submit_opening(
        self, command: SubmitOpeningCommand
    ) -> GatewayMutationResult:
        """Submit one opening vertical through preview and one fenced place."""

        self._require_started()
        command = _validate_submit_command(command)
        account = self._checked_account()
        self._require_no_blockers(account)
        envelope = OrderIntent.build(
            account_id=account.account_id,
            environment=self.runtime_safety.environment,
            strategy_id=command.strategy_id,
            decision_id=command.decision_id,
            idempotency_scope=command.idempotency_scope,
            idempotency_key=command.idempotency_key,
            intent_kind="OPENING",
            order_payload=_decode_payload(command.payload_bytes),
        )
        created = self.ledger.create_intent(envelope)
        record = created.intent
        existing = self._existing_submission_result(
            record, created.created
        )
        if existing is not None:
            return existing
        if self.ledger.get_margin_reservation(record.intent_id) is None:
            capacity = self._read_capacity(account)
            self.ledger.set_reservation_cap(capacity)
            self.ledger.reserve_margin(
                record.intent_id,
                RiskEvidence(
                    decision_id=command.decision_id,
                    max_loss_amount=command.max_loss_amount,
                    collateral_amount=command.collateral_amount,
                    quote_observed_at=command.quote_observed_at,
                    quote_digest=command.quote_digest,
                    portfolio_observed_at=capacity.observed_at,
                    portfolio_snapshot_digest=(
                        capacity.portfolio_snapshot_digest
                    ),
                ),
            )
        lease = self.ledger.claim_submission(
            record.intent_id,
            command.owner,
            lease_seconds=command.lease_seconds,
        )
        authorization = self.ledger.prepare_submission_payload(
            record.intent_id, command.owner, lease.fencing_token
        )
        try:
            preview = self.transport.preview(authorization)
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, command.owner, lease.fencing_token
            )
            raise
        if preview.disposition == "UNKNOWN":
            failed = self.ledger.mark_pre_post_failed(
                record.intent_id,
                command.owner,
                lease.fencing_token,
            )
            return self._result(
                failed,
                created.created,
                "FAILED",
                preview.broker_order_id,
                preview.preview_id,
                f"PREVIEW_{preview.unknown_reason}",
            )
        try:
            self._checked_account()
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, command.owner, lease.fencing_token
            )
            raise
        try:
            placed = self.transport.place(authorization, preview)
        except ETradeBrokerTransportError:
            current = self._require_intent(record.intent_id)
            if current.state == "SUBMISSION_UNKNOWN":
                return self._result(
                    current,
                    created.created,
                    "SUBMISSION_UNKNOWN",
                    self._known_place_order_id(current, "SUBMIT_PLACE"),
                    preview.preview_id,
                    "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
                )
            self._fail_unplaced_submission(
                record.intent_id, command.owner, lease.fencing_token
            )
            raise
        current = self._require_intent(record.intent_id)
        if placed.disposition == "ACKNOWLEDGED":
            if (
                current.state != "SUBMITTED"
                or current.broker_order_id != placed.broker_order_id
            ):
                raise GatewayReconciliationRequired(
                    "transport acknowledgement was not applied durably"
                )
            return self._result(
                current,
                created.created,
                "SUBMITTED",
                current.broker_order_id,
                preview.preview_id,
                None,
            )
        if current.state != "SUBMISSION_UNKNOWN":
            raise GatewayReconciliationRequired(
                "ambiguous place response lacks its durable pending state"
            )
        return self._result(
            current,
            created.created,
            "SUBMISSION_UNKNOWN",
            placed.broker_order_id,
            preview.preview_id,
            placed.unknown_reason,
        )

    def reprice_opening(
        self, command: RepriceOpeningCommand
    ) -> GatewayMutationResult:
        """Issue one previewed, fenced limit-price amendment."""

        self._require_started()
        command = _validate_reprice_command(command)
        record = self._require_intent(command.intent_id)
        if (
            record.envelope.intent_kind != "OPENING"
            or record.state != "SUBMITTED"
        ):
            raise GatewayValidationError(
                "only a submitted opening intent may be repriced"
            )
        account = self._checked_account()
        self._require_no_blockers(account)
        if (
            record.envelope.account_id != account.account_id
            or record.envelope.environment
            != self.runtime_safety.environment
        ):
            raise RuntimeSafetyError(
                "durable intent does not match the armed account"
            )
        amendment_payload = _decode_payload(command.payload_bytes)
        amendment_hash = canonical_order_payload_hash(
            amendment_payload
        )
        replay = self._completed_amendment_result(
            record, command.idempotency_key, amendment_hash
        )
        if replay is not None:
            return replay
        try:
            lease = self.ledger.acquire_amendment_lease(
                record.intent_id,
                command.owner,
                lease_seconds=command.lease_seconds,
                idempotency_key=command.idempotency_key,
                amendment_payload=amendment_payload,
            )
        except OrderIntentTransitionError:
            current = self._require_intent(record.intent_id)
            replay = self._completed_amendment_result(
                current, command.idempotency_key, amendment_hash
            )
            if replay is not None:
                return replay
            raise
        authorization = self.ledger.prepare_amendment_payload(
            record.intent_id, command.owner, lease.fencing_token
        )
        try:
            preview = self.transport.preview_change(
                lease.broker_order_id, authorization
            )
        except Exception:
            self.ledger.release_amendment_lease(
                record.intent_id, command.owner, lease.fencing_token
            )
            raise
        if preview.disposition == "UNKNOWN":
            self.ledger.release_amendment_lease(
                record.intent_id, command.owner, lease.fencing_token
            )
            return self._result(
                self._require_intent(record.intent_id),
                False,
                "SUBMITTED",
                record.broker_order_id,
                preview.preview_id,
                f"PREVIEW_{preview.unknown_reason}",
            )
        try:
            self._checked_account()
            placed = self.transport.place_change(
                lease.broker_order_id, authorization, preview
            )
        except ETradeBrokerTransportError:
            current = self._require_intent(record.intent_id)
            if current.pending_operation == "AMEND":
                return self._result(
                    current,
                    False,
                    "AMENDMENT_UNKNOWN",
                    self._known_place_order_id(current, "AMEND_PLACE"),
                    preview.preview_id,
                    "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
                )
            self.ledger.release_amendment_lease(
                record.intent_id,
                command.owner,
                lease.fencing_token,
            )
            raise
        except RuntimeSafetyError:
            current = self._require_intent(record.intent_id)
            if current.pending_operation is None:
                self.ledger.release_amendment_lease(
                    record.intent_id,
                    command.owner,
                    lease.fencing_token,
                )
            raise
        current = self._require_intent(record.intent_id)
        if placed.disposition == "ACKNOWLEDGED":
            if (
                current.state != "SUBMITTED"
                or current.pending_operation is not None
                or current.broker_order_id != placed.broker_order_id
            ):
                raise GatewayReconciliationRequired(
                    "amendment acknowledgement was not applied durably"
                )
            return self._result(
                current,
                False,
                "SUBMITTED",
                current.broker_order_id,
                preview.preview_id,
                None,
            )
        if current.pending_operation != "AMEND":
            raise GatewayReconciliationRequired(
                "ambiguous amendment lacks its durable pending state"
            )
        return self._result(
            current,
            False,
            "AMENDMENT_UNKNOWN",
            placed.broker_order_id,
            preview.preview_id,
            placed.unknown_reason,
        )

    def _checked_account(self) -> SelectedBrokerAccount:
        initial_now = self._now()
        self.runtime_safety.assert_current(initial_now)
        self.transport.assert_gateway_binding(
            self.ledger, self.runtime_safety
        )
        transport_account = self.transport.selected_account()
        self.reader.assert_gateway_binding(self.runtime_safety)
        reader_account = self.reader.selected_account()
        _validate_account(transport_account)
        _validate_account(reader_account)
        _require_same_account(transport_account, reader_account)
        final_now = self._now()
        self.runtime_safety.assert_current(final_now)
        self.runtime_safety.verify_account(
            transport_account.runtime_mapping(), now=final_now
        )
        if self._account is not None:
            _require_same_account(self._account, transport_account)
        return transport_account

    def _read_capacity(
        self,
        account: SelectedBrokerAccount,
    ) -> AccountCapacityEvidence:
        checked = self._checked_account()
        _require_same_account(account, checked)
        snapshot = self.reader.read_capacity(checked)
        if type(snapshot) is not BrokerCapacitySnapshot:
            raise GatewayValidationError(
                "capacity reader returned an invalid typed response"
            )
        _validate_account(snapshot.account)
        _require_same_account(checked, snapshot.account)
        _require_environment(
            self.runtime_safety.environment, snapshot.environment
        )
        _exact_decimal(
            snapshot.broker_buying_power,
            "broker_buying_power",
            allow_zero=True,
        )
        _fresh_time(snapshot.observed_at, self._now(), "capacity observed_at")
        _exact_sha256(
            snapshot.portfolio_snapshot_digest,
            "portfolio_snapshot_digest",
        )
        if (
            snapshot.positions_complete is not True
            or snapshot.open_orders_complete is not True
            or type(snapshot.source_response_digests) is not tuple
            or not snapshot.source_response_digests
        ):
            raise GatewayValidationError(
                "capacity snapshot must prove complete positions and open orders"
            )
        for digest in snapshot.source_response_digests:
            _exact_sha256(digest, "capacity source response digest")
        self._checked_account()
        return AccountCapacityEvidence(
            account_id=checked.account_id,
            environment=self.runtime_safety.environment,
            broker_buying_power=snapshot.broker_buying_power,
            risk_budget=self._opening_risk_budget,
            observed_at=snapshot.observed_at,
            portfolio_snapshot_digest=snapshot.portfolio_snapshot_digest,
        )

    def _reconcile_once(
        self,
        record: IntentRecord,
        account: SelectedBrokerAccount,
    ) -> None:
        context = self._reconciliation_context(record)
        if context is None:
            return
        try:
            reply = self.reader.query_order(
                account, context.broker_order_id
            )
        except Exception:
            return
        if type(reply) not in {BrokerOrderSnapshot, BrokerOrderNotFound}:
            return
        try:
            _validate_order_read(
                reply,
                account,
                context.broker_order_id,
                self.runtime_safety.environment,
                self._now(),
            )
        except (GatewayValidationError, RuntimeSafetyError):
            return
        if type(reply) is BrokerOrderNotFound:
            return
        if reply.outcome == "UNRESOLVED":
            return
        if (
            reply.order_payload_hash
            != context.expected_order_payload_hash
        ):
            return
        evidence = BrokerEvidence(
            account_id=account.account_id,
            environment=self.runtime_safety.environment,
            client_order_id=context.client_order_id,
            broker_order_id=reply.broker_order_id,
            operation=context.operation,
            outcome=reply.outcome,
            observed_at=reply.observed_at,
            http_status=reply.http_status,
            raw_response_digest=reply.raw_response_digest,
        )
        try:
            if reply.outcome == "OPEN":
                if record.pending_operation is not None:
                    self.ledger.reconcile_open(record.intent_id, evidence)
                else:
                    self.ledger.mark_reconciled(record.intent_id, evidence)
            else:
                self.ledger.reconcile_terminal(
                    record.intent_id, reply.outcome, evidence
                )
        except OrderIntentLedgerError:
            return

    def _reconciliation_context(
        self, record: IntentRecord
    ) -> _ReconciliationContext | None:
        if record.state == "CLAIMED":
            return None
        if record.pending_operation is None:
            if record.state != "SUBMITTED" or record.broker_order_id is None:
                return None
            return _ReconciliationContext(
                "ORDER_QUERY",
                record.client_order_id,
                record.broker_order_id,
                self.ledger.expected_order_payload_hash(record.intent_id),
            )
        operation = (
            "SUBMIT_PLACE"
            if record.pending_operation == "SUBMIT"
            else "AMEND_PLACE"
        )
        receipt = self._pending_place_receipt(record, operation)
        if receipt is None or receipt.response.broker_order_id is None:
            # E*TRADE does not echo clientOrderId in order responses. Without
            # a broker id there is no causal lookup key and guessing is unsafe.
            return None
        return _ReconciliationContext(
            "ORDER_QUERY"
            if record.pending_operation == "SUBMIT"
            else "AMEND_QUERY",
            receipt.client_order_id,
            receipt.response.broker_order_id,
            self.ledger.expected_order_payload_hash(record.intent_id),
        )

    def _pending_place_receipt(
        self,
        record: IntentRecord,
        operation: Literal["SUBMIT_PLACE", "AMEND_PLACE"],
    ) -> TransportResponseReceipt | None:
        matches = [
            receipt
            for receipt in self.ledger.transport_response_receipts(
                record.intent_id
            )
            if receipt.transport_operation == operation
            and receipt.fencing_token == record.pending_fence
        ]
        return matches[-1] if matches else None

    def _known_place_order_id(
        self,
        record: IntentRecord,
        operation: Literal["SUBMIT_PLACE", "AMEND_PLACE"],
    ) -> str | None:
        receipt = self._pending_place_receipt(record, operation)
        return receipt.response.broker_order_id if receipt else None

    def _require_no_blockers(
        self, account: SelectedBrokerAccount
    ) -> None:
        blockers = self.ledger.reconciliation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        if blockers:
            raise GatewayReconciliationRequired(
                f"{len(blockers)} durable broker operation(s) block mutation"
            )
        if self.ledger.unabsorbed_filled_reservation_count(
            account.account_id, self.runtime_safety.environment
        ):
            raise GatewayReconciliationRequired(
                "unabsorbed filled risk blocks mutation"
            )

    def _require_started(self) -> None:
        if not self._started:
            raise GatewayReconciliationRequired(
                "gateway.start() must complete before mutation"
            )

    def _require_intent(self, intent_id: str) -> IntentRecord:
        record = self.ledger.get_intent(intent_id)
        if record is None:
            raise GatewayValidationError("unknown intent_id")
        return record

    def _completed_amendment_result(
        self,
        record: IntentRecord,
        idempotency_key: str,
        requested_payload_hash: str,
    ) -> GatewayMutationResult | None:
        completed_hash = self.ledger.completed_amendment_payload_hash(
            record.intent_id, idempotency_key
        )
        if completed_hash is None:
            return None
        if completed_hash != requested_payload_hash:
            raise OrderIntentIntegrityError(
                "completed amendment idempotency key cannot be rebound"
            )
        if (
            record.state != "SUBMITTED"
            or record.pending_operation is not None
            or self.ledger.expected_order_payload_hash(record.intent_id)
            != completed_hash
        ):
            raise GatewayValidationError(
                "completed amendment has been superseded or is unresolved"
            )
        return self._result(
            record,
            False,
            "SUBMITTED",
            record.broker_order_id,
            None,
            "IDEMPOTENT_REPLAY",
        )

    def _fail_unplaced_submission(
        self, intent_id: str, owner: str, fencing_token: int
    ) -> None:
        current = self._require_intent(intent_id)
        if current.state == "CLAIMED":
            self.ledger.mark_pre_post_failed(
                intent_id, owner, fencing_token
            )

    def _now(self) -> datetime:
        value = self._clock()
        _exact_utc_time(value, "gateway clock")
        return value

    @staticmethod
    def _existing_submission_result(
        record: IntentRecord, created: bool
    ) -> GatewayMutationResult | None:
        if created or record.state == "INTENT":
            return None
        if record.state == "SUBMITTED":
            return EtradeOrderGateway._result(
                record,
                False,
                "SUBMITTED",
                record.broker_order_id,
                None,
                "IDEMPOTENT_REPLAY",
            )
        if record.state == "FAILED":
            return EtradeOrderGateway._result(
                record,
                False,
                "FAILED",
                None,
                None,
                "IDEMPOTENT_REPLAY",
            )
        raise GatewayReconciliationRequired(
            "existing intent has unresolved durable broker state"
        )

    @staticmethod
    def _result(
        record: IntentRecord,
        created: bool,
        state: Literal[
            "SUBMITTED",
            "SUBMISSION_UNKNOWN",
            "AMENDMENT_UNKNOWN",
            "FAILED",
        ],
        broker_order_id: str | None,
        preview_id: str | None,
        reason_code: str | None,
    ) -> GatewayMutationResult:
        return GatewayMutationResult(
            intent_id=record.intent_id,
            client_order_id=record.client_order_id,
            state=state,
            created=created,
            broker_order_id=broker_order_id,
            preview_id=preview_id,
            reason_code=reason_code,
        )


def _validate_submit_command(
    command: SubmitOpeningCommand,
) -> SubmitOpeningCommand:
    if type(command) is not SubmitOpeningCommand:
        raise GatewayValidationError(
            "submit command must use its exact immutable type"
        )
    for name in (
        "strategy_id",
        "decision_id",
        "idempotency_scope",
        "idempotency_key",
        "owner",
    ):
        _exact_text(getattr(command, name), name)
    _exact_payload_bytes(command.payload_bytes)
    _exact_decimal(command.max_loss_amount, "max_loss_amount")
    _exact_decimal(command.collateral_amount, "collateral_amount")
    _exact_utc_time(command.quote_observed_at, "quote_observed_at")
    _exact_sha256(command.quote_digest, "quote_digest")
    _lease_seconds(command.lease_seconds)
    return command


def _validate_reprice_command(
    command: RepriceOpeningCommand,
) -> RepriceOpeningCommand:
    if type(command) is not RepriceOpeningCommand:
        raise GatewayValidationError(
            "reprice command must use its exact immutable type"
        )
    _exact_text(command.intent_id, "intent_id")
    _exact_text(command.idempotency_key, "idempotency_key")
    _exact_text(command.owner, "owner")
    _exact_payload_bytes(command.payload_bytes)
    _lease_seconds(command.lease_seconds)
    return command


def _decode_payload(payload_bytes: bytes) -> dict[str, Any]:
    _exact_payload_bytes(payload_bytes)
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayValidationError(
            "payload_bytes must contain one strict JSON object"
        ) from exc
    if type(payload) is not dict:
        raise GatewayValidationError(
            "payload_bytes must contain one strict JSON object"
        )
    return payload


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise GatewayValidationError(
                "payload JSON contains duplicate or invalid keys"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise GatewayValidationError(
        f"payload JSON constant {value} is forbidden"
    )


def _validate_account(account: SelectedBrokerAccount) -> None:
    if type(account) is not SelectedBrokerAccount:
        raise GatewayValidationError(
            "broker account must use exact SelectedBrokerAccount"
        )


def _require_same_account(
    expected: SelectedBrokerAccount,
    actual: SelectedBrokerAccount,
) -> None:
    if actual != expected:
        raise RuntimeSafetyError(
            "broker account identity changed during the operation"
        )


def _require_environment(expected: str, actual: str) -> None:
    if (
        type(expected) is not str
        or expected not in {"sandbox", "production"}
        or type(actual) is not str
        or actual != expected
    ):
        raise RuntimeSafetyError(
            "broker environment changed during the operation"
        )


def _validate_order_read(
    reply: BrokerOrderSnapshot | BrokerOrderNotFound,
    account: SelectedBrokerAccount,
    broker_order_id: str,
    environment: str,
    now: datetime,
) -> None:
    _validate_account(reply.account)
    _require_same_account(account, reply.account)
    _require_environment(environment, reply.environment)
    _exact_broker_id(reply.broker_order_id)
    if reply.broker_order_id != broker_order_id:
        raise GatewayValidationError(
            "order read returned a different broker order id"
        )
    _fresh_time(reply.observed_at, now, "order observed_at")
    if type(reply.http_status) is not int or reply.http_status not in {
        200,
        404,
    }:
        raise GatewayValidationError(
            "order read has an unsupported HTTP status"
        )
    _exact_sha256(reply.raw_response_digest, "order raw_response_digest")
    if reply.complete is not True:
        raise GatewayValidationError("order read must be complete")
    if type(reply) is BrokerOrderSnapshot:
        _exact_sha256(
            reply.order_payload_hash, "order_payload_hash"
        )
        if reply.http_status != 200 or reply.outcome not in {
            "OPEN",
            "FILLED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
            "UNRESOLVED",
        }:
            raise GatewayValidationError(
                "order snapshot outcome is invalid"
            )


def _fresh_time(value: datetime, now: datetime, name: str) -> None:
    _exact_utc_time(value, name)
    if value > now or (now - value).total_seconds() > _MAX_EVIDENCE_AGE_SECONDS:
        raise GatewayValidationError(f"{name} is stale or future-dated")


def _exact_payload_bytes(value: Any) -> None:
    if (
        type(value) is not bytes
        or not value
        or len(value) > _MAX_COMMAND_BYTES
    ):
        raise GatewayValidationError(
            "payload_bytes must be bounded non-empty exact bytes"
        )


def _exact_text(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise GatewayValidationError(
            f"{name} must be a printable ASCII identifier"
        )


def _exact_broker_id(value: Any) -> None:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdigit()
        or value[0] == "0"
        or len(value) > 19
    ):
        raise GatewayValidationError("broker order id is invalid")


def _exact_sha256(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GatewayValidationError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def _exact_decimal(
    value: Any,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if type(value) is not Decimal or not value.is_finite():
        raise GatewayValidationError(
            f"{name} must be an exact finite Decimal"
        )
    if value < 0 or (value == 0 and not allow_zero):
        raise GatewayValidationError(f"{name} is outside the allowed range")


def _exact_utc_time(value: Any, name: str) -> None:
    if (
        type(value) is not datetime
        or value.tzinfo is not timezone.utc
    ):
        raise GatewayValidationError(
            f"{name} must use the exact UTC timezone"
        )


def _lease_seconds(value: Any) -> None:
    if (
        type(value) is not int
        or not 20 <= value <= _MAX_LEASE_SECONDS
    ):
        raise GatewayValidationError(
            "lease_seconds must be an exact integer between 20 and 300"
        )
