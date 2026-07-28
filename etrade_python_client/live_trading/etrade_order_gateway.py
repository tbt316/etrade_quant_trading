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
from typing import Any, Literal

from live_trading.etrade_broker_transport import (
    BrokerReply,
    CancelBrokerReply,
    ETradeBrokerTransport,
    ETradeBrokerTransportError,
    SelectedBrokerAccount,
)
from live_trading.etrade_broker_reader import (
    ETradeBrokerReader,
    ETradeBrokerReaderError,
    ETradeBrokerReaderIntegrityError,
    ETradeBrokerReaderUnavailable,
)
from live_trading.order_intent_ledger import (
    BrokerReadEvidenceRef,
    BrokerEvidence,
    CancellationObservation,
    CancellationRecord,
    CapacityDecisionReceipt,
    ClosingAbsorptionReceipt,
    ClosingAbsorptionRequirement,
    ClosingReservation,
    IntentRecord,
    MarginReservation,
    OrderIntent,
    OrderIntentBrokerTermsMismatch,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLedgerError,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    OrderIntentTransitionError,
    ReservationAbsorptionReceipt,
    RiskEvidence,
    TerminalAbsorptionRequirement,
    TransportResponseReceipt,
    canonical_order_payload_hash,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary, RuntimeSafetyError


_MAX_COMMAND_BYTES = 32 * 1024
_MAX_LEASE_SECONDS = 300


class EtradeOrderGatewayError(RuntimeError):
    """Base error for the durable order coordinator."""


class GatewayValidationError(EtradeOrderGatewayError):
    """A command or typed read response violated the closed contract."""


class GatewayReconciliationRequired(EtradeOrderGatewayError):
    """New mutations are blocked by unresolved durable broker work."""


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
    quote_valid_until: datetime
    quote_digest: str
    owner: str
    lease_seconds: int = 30


@dataclass(frozen=True)
class SubmitClosingCommand:
    strategy_id: str
    decision_id: str
    idempotency_scope: str
    idempotency_key: str
    payload_bytes: bytes = field(repr=False)
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
class CancelOpeningCommand:
    intent_id: str
    idempotency_key: str
    owner: str
    lease_seconds: int = 30


@dataclass(frozen=True)
class CancelClosingCommand:
    intent_id: str
    idempotency_key: str
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
class GatewayCancellationResult:
    intent_id: str
    state: Literal[
        "LEASED", "SEND_UNKNOWN", "REQUEST_ACCEPTED", "TERMINAL"
    ]
    broker_order_id: str = field(repr=False)
    reason_code: str | None


@dataclass(frozen=True)
class _ReconciliationContext:
    operation: Literal["ORDER_QUERY", "AMEND_QUERY"]
    client_order_id: str = field(repr=False)
    broker_order_id: str = field(repr=False)


@dataclass(frozen=True)
class _TerminalAbsorptionCandidate:
    record: IntentRecord
    terminal_read: BrokerReadEvidenceRef
    requirement: TerminalAbsorptionRequirement


@dataclass(frozen=True)
class _ClosingAbsorptionCandidate:
    record: IntentRecord
    terminal_read: BrokerReadEvidenceRef
    requirement: ClosingAbsorptionRequirement


class EtradeOrderGateway:
    """Compose runtime arming, durable state, typed reads, and one transport."""

    __slots__ = (
        "_runtime_safety",
        "_ledger",
        "_transport",
        "_reader",
        "_opening_risk_budget",
        "_daily_opening_risk_budget",
        "_clock",
        "_started",
        "_account",
    )

    def __init__(
        self,
        *,
        runtime_safety: RuntimeSafetyBoundary,
        ledger: OrderIntentLedger,
        transport: ETradeBrokerTransport,
        reader: ETradeBrokerReader,
        opening_risk_budget: Decimal,
        daily_opening_risk_budget: Decimal | None = None,
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
        if type(reader) is not ETradeBrokerReader:
            raise GatewayValidationError(
                "gateway requires the exact hardened ETradeBrokerReader"
            )
        try:
            transport.assert_gateway_binding(ledger, runtime_safety)
            reader.assert_gateway_binding(ledger, runtime_safety)
        except (
            ETradeBrokerTransportError,
            ETradeBrokerReaderError,
            RuntimeSafetyError,
        ) as exc:
            raise GatewayValidationError(
                "gateway adapters and runtime safety boundary do not match"
            ) from exc
        _exact_decimal(
            opening_risk_budget,
            "opening_risk_budget",
            allow_zero=True,
        )
        if daily_opening_risk_budget is None:
            daily_opening_risk_budget = opening_risk_budget
        _exact_decimal(
            daily_opening_risk_budget,
            "daily_opening_risk_budget",
            allow_zero=True,
        )
        self._runtime_safety = runtime_safety
        self._ledger = ledger
        self._transport = transport
        self._reader = reader
        self._opening_risk_budget = opening_risk_budget
        self._daily_opening_risk_budget = daily_opening_risk_budget
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._started = False
        self._account: SelectedBrokerAccount | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name
            not in {"_started", "_account"}
            and hasattr(self, name)
        ):
            raise AttributeError(
                "gateway safety dependencies are immutable"
            )
        object.__setattr__(self, name, value)

    @property
    def runtime_safety(self) -> RuntimeSafetyBoundary:
        return self._runtime_safety

    @property
    def ledger(self) -> OrderIntentLedger:
        return self._ledger

    @property
    def reader(self) -> ETradeBrokerReader:
        return self._reader

    @property
    def execution_ready(self) -> bool:
        """Return whether mutation is currently safe without changing state."""

        if not self._started:
            return False
        try:
            account = self._checked_account()
            if self._account is None or account != self._account:
                return False
            return not self.ledger.has_execution_blockers(
                account.account_id,
                self.runtime_safety.environment,
            )
        except Exception:
            return False

    def recent_opening_intents(
        self,
        *,
        idempotency_scope: str,
        limit: int,
    ) -> tuple[IntentRecord, ...]:
        """Read exact durable opening history without broker I/O or a live arm."""

        account = self._history_account()
        return self.ledger.recent_intents(
            account_id=account.account_id,
            environment=self.runtime_safety.environment,
            intent_kind="OPENING",
            idempotency_scope=idempotency_scope,
            limit=limit,
        )

    def start(self) -> None:
        """Reconcile every resolvable prior order before enabling mutation."""

        self._started = False
        account = self._checked_account()
        self._account = account
        cancellations = self.ledger.cancellation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        for cancellation in cancellations:
            self._reconcile_cancellation_once(cancellation, account)
        remaining_cancellations = self.ledger.cancellation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        if remaining_cancellations:
            raise GatewayReconciliationRequired(
                "gateway remains read-only: "
                f"{len(remaining_cancellations)} cancellation(s) "
                "lack terminal broker proof"
            )
        blockers = self.ledger.reconciliation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        for blocker in blockers:
            self._reconcile_once(blocker, account)
        pending_absorptions = self.ledger.pending_terminal_reservations(
            account.account_id, self.runtime_safety.environment
        )
        absorption_candidates = []
        for pending in pending_absorptions:
            candidate = self._prepare_terminal_absorption(
                pending, account
            )
            if candidate is not None:
                absorption_candidates.append(candidate)
        post_capacity = None
        if any(
            candidate.requirement.classification == "FULL_FILL"
            for candidate in absorption_candidates
        ):
            try:
                post_capacity = self._read_capacity(account)
            except GatewayReconciliationRequired:
                post_capacity = None
        for candidate in absorption_candidates:
            self._apply_terminal_absorption(
                candidate,
                account,
                post_capacity=(
                    post_capacity
                    if candidate.requirement.classification == "FULL_FILL"
                    else None
                ),
            )
        closing_candidates = []
        for pending in self.ledger.closing_reservation_blockers(
            account.account_id, self.runtime_safety.environment
        ):
            if pending.state not in {
                "FILLED",
                "CANCELLED",
                "REJECTED",
                "EXPIRED",
            }:
                continue
            candidate = self._prepare_closing_absorption(
                pending, account
            )
            if candidate is not None:
                closing_candidates.append(candidate)
        closing_post_capacity = None
        if any(
            candidate.requirement.classification == "FULL_FILL"
            for candidate in closing_candidates
        ):
            try:
                closing_post_capacity = self._read_capacity_evidence(
                    account
                )
            except GatewayReconciliationRequired:
                closing_post_capacity = None
        for candidate in closing_candidates:
            self._apply_closing_absorption(
                candidate,
                account,
                post_capacity=(
                    closing_post_capacity
                    if candidate.requirement.classification == "FULL_FILL"
                    else None
                ),
            )
        remaining = self.ledger.reconciliation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        unabsorbed = self.ledger.pending_terminal_reservations(
            account.account_id, self.runtime_safety.environment
        )
        closing_unabsorbed = self.ledger.closing_reservation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        if remaining or unabsorbed or closing_unabsorbed:
            raise GatewayReconciliationRequired(
                "gateway remains read-only: "
                f"{len(remaining)} broker operation(s), "
                f"{len(unabsorbed)} pending opening absorption(s), "
                f"{len(closing_unabsorbed)} pending closing reservation(s)"
            )
        self._checked_account()
        self._started = True

    def submit_opening(
        self, command: SubmitOpeningCommand
    ) -> GatewayMutationResult:
        """Submit one opening vertical through preview and one fenced place."""

        self._require_started()
        command = _validate_submit_command(command)
        if self._now() >= command.quote_valid_until:
            raise GatewayValidationError(
                "opening quote expired before durable submission began"
            )
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
            try:
                capacity = self._read_capacity(account)
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
                        capacity_decision_sha256=(
                            capacity.decision_sha256
                        ),
                    ),
                )
            except Exception:
                try:
                    self.ledger.abandon_trace_free_opening_intent(
                        record.intent_id
                    )
                except Exception:
                    # The original capacity/reservation failure is the caller's
                    # actionable error.  A refusal here means durable evidence
                    # exists, so the intent correctly remains a blocker.
                    pass
                raise
        return self._submit_intent(
            record,
            created=created.created,
            owner=command.owner,
            lease_seconds=command.lease_seconds,
            quote_valid_until=command.quote_valid_until,
        )

    def submit_closing(
        self, command: SubmitClosingCommand
    ) -> GatewayMutationResult:
        """Reserve and submit one exact closing vertical."""

        self._require_started()
        command = _validate_closing_command(command)
        account = self._checked_account()
        envelope = OrderIntent.build(
            account_id=account.account_id,
            environment=self.runtime_safety.environment,
            strategy_id=command.strategy_id,
            decision_id=command.decision_id,
            idempotency_scope=command.idempotency_scope,
            idempotency_key=command.idempotency_key,
            intent_kind="CLOSING",
            order_payload=_decode_payload(command.payload_bytes),
        )
        try:
            record = self.ledger.find_intent(envelope)
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "closing idempotency lookup failed durable validation"
            ) from exc
        if record is not None:
            existing = self._existing_submission_result(record, False)
            if existing is not None:
                return existing
        self._require_no_blockers(account)
        created = False
        if record is None:
            capacity_evidence = self._read_capacity_evidence(account)
            try:
                creation = self.ledger.create_closing_intent_from_read(
                    envelope, capacity_evidence
                )
            except OrderIntentReconciliationRequired as exc:
                raise GatewayReconciliationRequired(
                    "closing capacity is blocked by unresolved broker work"
                ) from exc
            except OrderIntentLedgerError as exc:
                raise GatewayValidationError(
                    "closing capacity could not authorize an exact reservation"
                ) from exc
            record = creation.intent
            created = creation.created
            existing = self._existing_submission_result(record, created)
            if existing is not None:
                return existing
        try:
            return self._submit_intent(
                record,
                created=created,
                owner=command.owner,
                lease_seconds=command.lease_seconds,
            )
        except OrderIntentReservationError as exc:
            current = self._require_intent(record.intent_id)
            if current.state != "INTENT":
                raise
            try:
                abandoned = self.ledger.abandon_stale_closing_intent(
                    current.intent_id
                )
            except OrderIntentReservationError:
                raise exc
            return self._result(
                abandoned,
                created,
                "FAILED",
                None,
                None,
                "STALE_CAPACITY_ABANDONED",
            )

    def _submit_intent(
        self,
        record: IntentRecord,
        *,
        created: bool,
        owner: str,
        lease_seconds: int,
        quote_valid_until: datetime | None = None,
    ) -> GatewayMutationResult:
        """Run the one reviewed preview/place path for an already-reserved intent."""

        lease = self.ledger.claim_submission(
            record.intent_id,
            owner,
            lease_seconds=lease_seconds,
        )
        if (
            quote_valid_until is not None
            and self._now() >= quote_valid_until
        ):
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise GatewayValidationError(
                "opening quote expired before broker preview"
            )
        try:
            authorization = self.ledger.prepare_submission_payload(
                record.intent_id, owner, lease.fencing_token
            )
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise
        try:
            preview = self._transport.preview(
                authorization,
                not_after=quote_valid_until,
            )
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise
        if preview.disposition == "UNKNOWN":
            failed = self.ledger.mark_pre_post_failed(
                record.intent_id,
                owner,
                lease.fencing_token,
            )
            return self._result(
                failed,
                created,
                "FAILED",
                preview.broker_order_id,
                preview.preview_id,
                f"PREVIEW_{preview.unknown_reason}",
            )
        try:
            self._checked_account()
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise
        if (
            quote_valid_until is not None
            and self._now() >= quote_valid_until
        ):
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise GatewayValidationError(
                "opening quote expired before broker placement"
            )
        try:
            placed = self._transport.place(
                authorization,
                preview,
                not_after=quote_valid_until,
            )
        except ETradeBrokerTransportError:
            current = self._require_intent(record.intent_id)
            if current.state == "SUBMISSION_UNKNOWN":
                return self._result(
                    current,
                    created,
                    "SUBMISSION_UNKNOWN",
                    self._known_place_order_id(current, "SUBMIT_PLACE"),
                    preview.preview_id,
                    "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
                )
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
            )
            raise
        except Exception:
            self._fail_unplaced_submission(
                record.intent_id, owner, lease.fencing_token
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
                created,
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
            created,
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
            preview = self._transport.preview_change(
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
            placed = self._transport.place_change(
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

    def cancel_opening(
        self, command: CancelOpeningCommand
    ) -> GatewayCancellationResult:
        """Cancel one opening order through the durable one-shot protocol."""

        command = _validate_cancel_command(command)
        return self._cancel_order(
            intent_id=command.intent_id,
            idempotency_key=command.idempotency_key,
            owner=command.owner,
            lease_seconds=command.lease_seconds,
            expected_kind="OPENING",
        )

    def cancel_closing(
        self, command: CancelClosingCommand
    ) -> GatewayCancellationResult:
        """Cancel one closing order without releasing its position claim."""

        command = _validate_closing_cancel_command(command)
        return self._cancel_order(
            intent_id=command.intent_id,
            idempotency_key=command.idempotency_key,
            owner=command.owner,
            lease_seconds=command.lease_seconds,
            expected_kind="CLOSING",
        )

    def _cancel_order(
        self,
        *,
        intent_id: str,
        idempotency_key: str,
        owner: str,
        lease_seconds: int,
        expected_kind: Literal["OPENING", "CLOSING"],
    ) -> GatewayCancellationResult:
        """Request once; only a later exact read may prove terminal."""

        existing = self.ledger.get_cancellation(intent_id)
        if (
            existing is not None
            and existing.idempotency_key != idempotency_key
        ):
            raise GatewayValidationError(
                "cancellation idempotency key cannot be rebound"
            )
        if existing is not None and existing.state != "LEASED":
            return self._cancellation_result(
                existing, "IDEMPOTENT_REPLAY"
        )
        if existing is None:
            self._require_started()
        record = self._require_intent(intent_id)
        if (
            record.envelope.intent_kind != expected_kind
            or record.state != "SUBMITTED"
            or record.pending_operation is not None
            or record.broker_order_id is None
        ):
            raise GatewayValidationError(
                f"only a known submitted {expected_kind.lower()} "
                "intent may be cancelled"
            )
        account = self._checked_account()
        if (
            record.envelope.account_id != account.account_id
            or record.envelope.environment
            != self.runtime_safety.environment
        ):
            raise RuntimeSafetyError(
                "durable cancel intent does not match the armed account"
            )
        if existing is None:
            self._require_no_blockers(account)
        order_read = self._query_known_order(
            account, record.broker_order_id
        )
        if existing is not None:
            observation = self.ledger.classify_cancellation_read(
                record.intent_id, order_read
            )
            if observation.outcome in {
                "FILLED",
                "CANCELLED",
                "REJECTED",
                "EXPIRED",
            }:
                self._apply_cancellation_terminal(
                    record, observation, order_read
                )
                resolved = self.ledger.complete_cancellation(
                    record.intent_id, order_read
                )
                return self._cancellation_result(
                    resolved, "BROKER_TERMINAL"
                )
        authorization = self.ledger.authorize_cancellation(
            record.intent_id,
            idempotency_key,
            owner,
            lease_seconds,
            order_read,
        )
        self._checked_account()
        reply: CancelBrokerReply
        try:
            reply = self._transport.cancel(authorization)
        except ETradeBrokerTransportError:
            current = self.ledger.get_cancellation(record.intent_id)
            if current is not None and current.state in {
                "SEND_UNKNOWN",
                "REQUEST_ACCEPTED",
            }:
                return self._cancellation_result(
                    current,
                    "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
                )
            raise
        finally:
            # After entering the transport path, the gateway must reconcile a
            # direct broker read before enabling any further mutation.
            self._started = False
        current = self.ledger.get_cancellation(record.intent_id)
        if current is None:
            raise GatewayReconciliationRequired(
                "cancel response lacks durable cancellation state"
            )
        expected_state = (
            "REQUEST_ACCEPTED"
            if reply.disposition == "REQUEST_ACCEPTED"
            else "SEND_UNKNOWN"
        )
        if current.state != expected_state:
            raise GatewayReconciliationRequired(
                "cancel response and durable state disagree"
            )
        return self._cancellation_result(
            current,
            (
                None
                if reply.disposition == "REQUEST_ACCEPTED"
                else reply.unknown_reason
            ),
        )

    def _checked_account(self) -> SelectedBrokerAccount:
        if (
            type(self._runtime_safety) is not RuntimeSafetyBoundary
            or type(self._ledger) is not OrderIntentLedger
            or type(self._transport) is not ETradeBrokerTransport
            or type(self._reader) is not ETradeBrokerReader
        ):
            raise GatewayValidationError(
                "gateway safety dependency identity changed after construction"
            )
        initial_now = self._now()
        self.runtime_safety.assert_current(initial_now)
        self._transport.assert_gateway_binding(
            self.ledger, self.runtime_safety
        )
        transport_account = self._transport.selected_account()
        self.reader.assert_gateway_binding(
            self.ledger, self.runtime_safety
        )
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

    def _history_account(self) -> SelectedBrokerAccount:
        """Verify immutable adapter/runtime identity without arm freshness."""

        if (
            type(self._runtime_safety) is not RuntimeSafetyBoundary
            or type(self._ledger) is not OrderIntentLedger
            or type(self._transport) is not ETradeBrokerTransport
            or type(self._reader) is not ETradeBrokerReader
        ):
            raise GatewayValidationError(
                "gateway history dependency identity changed"
            )
        transport_account = self._transport.selected_account()
        reader_account = self._reader.selected_account()
        _validate_account(transport_account)
        _validate_account(reader_account)
        _require_same_account(transport_account, reader_account)
        if (
            transport_account.account_id
            != self.runtime_safety.expected_account_id
            or transport_account.account_id_key
            != self.runtime_safety.expected_account_id_key
            or transport_account.institution_type
            != self.runtime_safety.expected_institution_type
        ):
            raise GatewayValidationError(
                "gateway history account does not match runtime binding"
            )
        if self._account is not None:
            _require_same_account(self._account, transport_account)
        return transport_account

    def _query_known_order(
        self,
        account: SelectedBrokerAccount,
        broker_order_id: str,
    ) -> BrokerReadEvidenceRef:
        checked = self._checked_account()
        _require_same_account(account, checked)
        _exact_broker_id(broker_order_id)
        try:
            evidence = self.reader.query_order(
                checked, broker_order_id
            )
        except ETradeBrokerReaderUnavailable as exc:
            raise GatewayReconciliationRequired(
                "known broker order could not be read completely"
            ) from exc
        except ETradeBrokerReaderIntegrityError as exc:
            raise GatewayValidationError(
                "known broker order violated the durable read contract"
            ) from exc
        except ETradeBrokerReaderError as exc:
            raise GatewayReconciliationRequired(
                "known broker order could not be read safely"
            ) from exc
        if (
            type(evidence) is not BrokerReadEvidenceRef
            or evidence.evidence_kind != "ORDER_QUERY"
        ):
            raise GatewayValidationError(
                "order reader returned invalid durable evidence"
            )
        self._checked_account()
        return evidence

    def _reconcile_cancellation_once(
        self,
        cancellation: CancellationRecord,
        account: SelectedBrokerAccount,
    ) -> None:
        if (
            type(cancellation) is not CancellationRecord
            or cancellation.account_id != account.account_id
            or cancellation.environment
            != self.runtime_safety.environment
            or cancellation.state == "TERMINAL"
        ):
            raise GatewayValidationError(
                "cancellation blocker has invalid durable identity"
            )
        order_read = self._query_known_order(
            account, cancellation.broker_order_id
        )
        try:
            observation = self.ledger.classify_cancellation_read(
                cancellation.intent_id, order_read
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "cancellation read failed durable validation"
            ) from exc
        if observation.outcome not in {
            "FILLED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
        }:
            return
        record = self._require_intent(cancellation.intent_id)
        self._apply_cancellation_terminal(
            record, observation, order_read
        )
        try:
            completed = self.ledger.complete_cancellation(
                cancellation.intent_id, order_read
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal cancellation could not be completed durably"
            ) from exc
        if completed.state != "TERMINAL":
            raise GatewayValidationError(
                "terminal cancellation completion was not durable"
            )

    def _apply_cancellation_terminal(
        self,
        record: IntentRecord,
        observation: CancellationObservation,
        order_read: BrokerReadEvidenceRef,
    ) -> None:
        if (
            type(observation) is not CancellationObservation
            or observation.intent_id != record.intent_id
            or observation.broker_order_id != record.broker_order_id
            or observation.evidence_sha256
            != order_read.evidence_sha256
            or observation.outcome
            not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
        ):
            raise GatewayValidationError(
                "terminal cancellation observation is not intent-bound"
            )
        try:
            broker_evidence = self.ledger.broker_evidence_from_read(
                record.intent_id,
                order_read,
                operation="ORDER_QUERY",
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal cancellation evidence failed validation"
            ) from exc
        if (
            type(broker_evidence) is not BrokerEvidence
            or broker_evidence.outcome != observation.outcome
        ):
            raise GatewayReconciliationRequired(
                "terminal cancellation lacks exact broker evidence"
            )
        current = self._require_intent(record.intent_id)
        if current.state == observation.outcome:
            return
        if current.state != "SUBMITTED":
            raise GatewayReconciliationRequired(
                "terminal cancellation conflicts with durable intent state"
            )
        try:
            self.ledger.reconcile_terminal(
                record.intent_id,
                observation.outcome,
                broker_evidence,
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal cancellation could not reconcile the intent"
            ) from exc

    def _read_capacity(
        self,
        account: SelectedBrokerAccount,
    ) -> CapacityDecisionReceipt:
        evidence = self._read_capacity_evidence(account)
        try:
            decision = self.ledger.set_reservation_cap_from_read(
                evidence,
                risk_budget=self._opening_risk_budget,
                daily_risk_budget=self._daily_opening_risk_budget,
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "durable capacity evidence could not authorize a cap"
            ) from exc
        self._checked_account()
        return decision

    def _read_capacity_evidence(
        self,
        account: SelectedBrokerAccount,
    ) -> BrokerReadEvidenceRef:
        checked = self._checked_account()
        _require_same_account(account, checked)
        try:
            evidence = self.reader.read_capacity(checked)
        except ETradeBrokerReaderUnavailable as exc:
            raise GatewayReconciliationRequired(
                "broker capacity did not yield a stable complete snapshot"
            ) from exc
        except ETradeBrokerReaderIntegrityError as exc:
            raise GatewayValidationError(
                "broker capacity evidence violated the durable read contract"
            ) from exc
        except ETradeBrokerReaderError as exc:
            raise GatewayReconciliationRequired(
                "broker capacity could not be read safely"
            ) from exc
        if (
            type(evidence) is not BrokerReadEvidenceRef
            or evidence.evidence_kind != "CAPACITY"
        ):
            raise GatewayValidationError(
                "capacity reader returned an invalid durable evidence reference"
            )
        try:
            verified = self.ledger.broker_read_evidence(
                evidence.evidence_sha256
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "capacity evidence reference failed durable verification"
            ) from exc
        if verified != evidence:
            raise GatewayValidationError(
                "capacity evidence reference failed durable verification"
            )
        self._checked_account()
        return evidence

    def _reconcile_once(
        self,
        record: IntentRecord,
        account: SelectedBrokerAccount,
    ) -> None:
        context = self._reconciliation_context(record)
        if context is None:
            return
        try:
            read_evidence = self.reader.query_order(
                account, context.broker_order_id
            )
        except ETradeBrokerReaderUnavailable:
            return
        except ETradeBrokerReaderIntegrityError as exc:
            raise GatewayValidationError(
                "durable broker read integrity failed during reconciliation"
            ) from exc
        except ETradeBrokerReaderError as exc:
            raise GatewayReconciliationRequired(
                "broker reader could not safely reconcile the known order"
            ) from exc
        if (
            type(read_evidence) is not BrokerReadEvidenceRef
            or read_evidence.evidence_kind != "ORDER_QUERY"
        ):
            raise GatewayValidationError(
                "order reader returned an invalid durable evidence reference"
            )
        try:
            evidence = self.ledger.broker_evidence_from_read(
                record.intent_id,
                read_evidence,
                operation=context.operation,
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "durable broker order evidence failed ledger validation"
            ) from exc
        if evidence is None:
            return
        try:
            if evidence.outcome == "OPEN":
                if record.pending_operation is not None:
                    self.ledger.reconcile_open(record.intent_id, evidence)
                else:
                    self.ledger.mark_reconciled(record.intent_id, evidence)
            else:
                self.ledger.reconcile_terminal(
                    record.intent_id, evidence.outcome, evidence
                )
        except OrderIntentBrokerTermsMismatch as exc:
            raise GatewayReconciliationRequired(
                "known broker order terms do not match the pending operation"
            ) from exc
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "broker evidence could not be applied atomically"
            ) from exc

    def _prepare_terminal_absorption(
        self,
        record: IntentRecord,
        account: SelectedBrokerAccount,
    ) -> _TerminalAbsorptionCandidate | None:
        """Collect one exact terminal read before any shared capacity scan."""

        if (
            record.envelope.account_id != account.account_id
            or record.envelope.environment
            != self.runtime_safety.environment
            or record.broker_order_id is None
        ):
            raise GatewayValidationError(
                "pending terminal reservation has invalid durable identity"
            )
        checked = self._checked_account()
        _require_same_account(account, checked)
        try:
            terminal_read = self.reader.query_order(
                checked, record.broker_order_id
            )
        except ETradeBrokerReaderUnavailable:
            return None
        except ETradeBrokerReaderIntegrityError as exc:
            raise GatewayValidationError(
                "terminal absorption order read violated the durable contract"
            ) from exc
        except ETradeBrokerReaderError:
            return None
        if (
            type(terminal_read) is not BrokerReadEvidenceRef
            or terminal_read.evidence_kind != "ORDER_QUERY"
        ):
            raise GatewayValidationError(
                "terminal absorption requires exact durable order evidence"
            )
        try:
            requirement = self.ledger.terminal_absorption_requirement(
                record.intent_id, terminal_read
            )
        except OrderIntentReconciliationRequired:
            return None
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal absorption requirement failed durable validation"
            ) from exc
        if (
            type(requirement) is not TerminalAbsorptionRequirement
            or requirement.intent_id != record.intent_id
            or requirement.broker_order_id != record.broker_order_id
            or requirement.terminal_state != record.state
            or requirement.terminal_order_evidence_sha256
            != terminal_read.evidence_sha256
        ):
            raise GatewayValidationError(
                "terminal absorption requirement is not bound to the intent"
            )
        try:
            reservation = self.ledger.get_margin_reservation(
                record.intent_id
            )
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal absorption reservation failed durable validation"
            ) from exc
        if (
            type(reservation) is not MarginReservation
            or reservation.intent_id != record.intent_id
            or reservation.account_id != account.account_id
            or reservation.environment
            != self.runtime_safety.environment
            or reservation.state != "FILLED_PENDING_ABSORPTION"
            or reservation.capacity_decision_sha256
            != requirement.baseline_capacity_decision_sha256
        ):
            raise GatewayValidationError(
                "terminal absorption requirement is disconnected from risk"
            )
        if requirement.post_capacity_required != (
            requirement.classification == "FULL_FILL"
        ):
            raise GatewayValidationError(
                "terminal absorption requirement has inconsistent capacity needs"
            )
        if requirement.classification not in {"ZERO_FILL", "FULL_FILL"}:
            raise GatewayValidationError(
                "terminal absorption classification is unsupported"
            )
        self._checked_account()
        return _TerminalAbsorptionCandidate(
            record=record,
            terminal_read=terminal_read,
            requirement=requirement,
        )

    def _apply_terminal_absorption(
        self,
        candidate: _TerminalAbsorptionCandidate,
        account: SelectedBrokerAccount,
        *,
        post_capacity: CapacityDecisionReceipt | None,
    ) -> bool:
        """Apply one candidate using a capacity snapshot newer than the batch."""

        if type(candidate) is not _TerminalAbsorptionCandidate:
            raise GatewayValidationError(
                "terminal absorption candidate has an invalid type"
            )
        record = candidate.record
        terminal_read = candidate.terminal_read
        requirement = candidate.requirement
        if requirement.classification == "FULL_FILL":
            if post_capacity is None:
                return False
            if type(post_capacity) is not CapacityDecisionReceipt:
                raise GatewayValidationError(
                    "full-fill absorption requires exact capacity evidence"
                )
        elif (
            requirement.classification == "ZERO_FILL"
            and post_capacity is not None
        ):
            raise GatewayValidationError(
                "zero-fill absorption cannot use capacity evidence"
            )
        checked = self._checked_account()
        _require_same_account(account, checked)
        try:
            receipt = self.ledger.absorb_terminal_reservation(
                record.intent_id,
                terminal_read,
                post_capacity_decision=post_capacity,
            )
        except OrderIntentReconciliationRequired:
            return False
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "terminal reservation could not be absorbed atomically"
            ) from exc
        if (
            type(receipt) is not ReservationAbsorptionReceipt
            or receipt.intent_id != record.intent_id
            or receipt.broker_order_id != record.broker_order_id
            or receipt.terminal_state != record.state
            or receipt.classification != requirement.classification
            or receipt.terminal_order_evidence_sha256
            != terminal_read.evidence_sha256
            or (
                post_capacity is None
                and receipt.post_capacity_decision_sha256 is not None
            )
            or (
                post_capacity is not None
                and receipt.post_capacity_decision_sha256
                != post_capacity.decision_sha256
            )
        ):
            raise GatewayValidationError(
                "terminal absorption receipt is not bound to the intent"
            )
        self._checked_account()
        return True

    def _prepare_closing_absorption(
        self,
        record: IntentRecord,
        account: SelectedBrokerAccount,
    ) -> _ClosingAbsorptionCandidate | None:
        """Collect exact terminal evidence for one durable closing claim."""

        if (
            record.envelope.intent_kind != "CLOSING"
            or record.envelope.account_id != account.account_id
            or record.envelope.environment
            != self.runtime_safety.environment
            or record.state
            not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}
            or record.broker_order_id is None
        ):
            raise GatewayValidationError(
                "pending closing reservation has invalid durable identity"
            )
        checked = self._checked_account()
        _require_same_account(account, checked)
        try:
            terminal_read = self.reader.query_order(
                checked, record.broker_order_id
            )
        except ETradeBrokerReaderUnavailable:
            return None
        except ETradeBrokerReaderIntegrityError as exc:
            raise GatewayValidationError(
                "closing absorption order read violated the durable contract"
            ) from exc
        except ETradeBrokerReaderError:
            return None
        if (
            type(terminal_read) is not BrokerReadEvidenceRef
            or terminal_read.evidence_kind != "ORDER_QUERY"
        ):
            raise GatewayValidationError(
                "closing absorption requires exact durable order evidence"
            )
        try:
            requirement = self.ledger.closing_absorption_requirement(
                record.intent_id, terminal_read
            )
            reservation = self.ledger.get_closing_reservation(
                record.intent_id
            )
        except OrderIntentReconciliationRequired:
            return None
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "closing absorption requirement failed durable validation"
            ) from exc
        if (
            type(requirement) is not ClosingAbsorptionRequirement
            or type(reservation) is not ClosingReservation
            or requirement.intent_id != record.intent_id
            or requirement.broker_order_id != record.broker_order_id
            or requirement.terminal_state != record.state
            or requirement.terminal_order_evidence_sha256
            != terminal_read.evidence_sha256
            or requirement.baseline_capacity_evidence_sha256
            != reservation.capacity_evidence_sha256
            or reservation.intent_id != record.intent_id
            or reservation.account_id != account.account_id
            or reservation.environment
            != self.runtime_safety.environment
            or requirement.post_capacity_required
            != (requirement.classification == "FULL_FILL")
            or requirement.classification not in {"ZERO_FILL", "FULL_FILL"}
        ):
            raise GatewayValidationError(
                "closing absorption requirement is not bound to its reservation"
            )
        self._checked_account()
        return _ClosingAbsorptionCandidate(
            record=record,
            terminal_read=terminal_read,
            requirement=requirement,
        )

    def _apply_closing_absorption(
        self,
        candidate: _ClosingAbsorptionCandidate,
        account: SelectedBrokerAccount,
        *,
        post_capacity: BrokerReadEvidenceRef | None,
    ) -> bool:
        """Release a terminal close only through its append-only proof."""

        if type(candidate) is not _ClosingAbsorptionCandidate:
            raise GatewayValidationError(
                "closing absorption candidate has an invalid type"
            )
        requirement = candidate.requirement
        if requirement.classification == "FULL_FILL":
            if (
                type(post_capacity) is not BrokerReadEvidenceRef
                or post_capacity.evidence_kind != "CAPACITY"
            ):
                return False
        elif post_capacity is not None:
            raise GatewayValidationError(
                "zero-fill closing absorption cannot use capacity evidence"
            )
        checked = self._checked_account()
        _require_same_account(account, checked)
        try:
            receipt = self.ledger.absorb_closing_reservation(
                candidate.record.intent_id,
                candidate.terminal_read,
                post_capacity_evidence=post_capacity,
            )
        except OrderIntentReconciliationRequired:
            return False
        except OrderIntentLedgerError as exc:
            raise GatewayValidationError(
                "closing reservation could not be absorbed atomically"
            ) from exc
        expected_post = (
            None
            if post_capacity is None
            else post_capacity.evidence_sha256
        )
        if (
            type(receipt) is not ClosingAbsorptionReceipt
            or receipt.intent_id != candidate.record.intent_id
            or receipt.account_id != account.account_id
            or receipt.environment != self.runtime_safety.environment
            or receipt.broker_order_id
            != candidate.record.broker_order_id
            or receipt.terminal_state != candidate.record.state
            or receipt.classification != requirement.classification
            or receipt.terminal_order_evidence_sha256
            != candidate.terminal_read.evidence_sha256
            or receipt.baseline_capacity_evidence_sha256
            != requirement.baseline_capacity_evidence_sha256
            or receipt.post_capacity_evidence_sha256 != expected_post
            or receipt.ordered_quantity != requirement.ordered_quantity
            or receipt.filled_quantity != requirement.filled_quantity
        ):
            raise GatewayValidationError(
                "closing absorption receipt is not bound to the intent"
            )
        self._checked_account()
        return True

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
        cancellations = self.ledger.cancellation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        if cancellations:
            raise GatewayReconciliationRequired(
                f"{len(cancellations)} unresolved cancellation(s) "
                "block mutation"
            )
        closing = self.ledger.closing_reservation_blockers(
            account.account_id, self.runtime_safety.environment
        )
        if closing:
            raise GatewayReconciliationRequired(
                f"{len(closing)} unresolved closing reservation(s) "
                "block mutation"
            )
        if self.ledger.unabsorbed_filled_reservation_count(
            account.account_id, self.runtime_safety.environment
        ):
            raise GatewayReconciliationRequired(
                "unabsorbed terminal risk blocks mutation"
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

    @staticmethod
    def _cancellation_result(
        record: CancellationRecord,
        reason_code: str | None,
    ) -> GatewayCancellationResult:
        if type(record) is not CancellationRecord:
            raise GatewayValidationError(
                "cancellation result requires exact durable state"
            )
        return GatewayCancellationResult(
            intent_id=record.intent_id,
            state=record.state,
            broker_order_id=record.broker_order_id,
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
    _exact_utc_time(command.quote_valid_until, "quote_valid_until")
    if command.quote_valid_until <= command.quote_observed_at:
        raise GatewayValidationError(
            "quote_valid_until must be after quote_observed_at"
        )
    _exact_sha256(command.quote_digest, "quote_digest")
    _lease_seconds(command.lease_seconds)
    return command


def _validate_closing_command(
    command: SubmitClosingCommand,
) -> SubmitClosingCommand:
    if type(command) is not SubmitClosingCommand:
        raise GatewayValidationError(
            "closing command must use its exact immutable type"
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


def _validate_cancel_command(
    command: CancelOpeningCommand,
) -> CancelOpeningCommand:
    if type(command) is not CancelOpeningCommand:
        raise GatewayValidationError(
            "cancel command must use its exact immutable type"
        )
    _exact_text(command.intent_id, "intent_id")
    _exact_text(command.idempotency_key, "idempotency_key")
    _exact_text(command.owner, "owner")
    _lease_seconds(command.lease_seconds)
    return command


def _validate_closing_cancel_command(
    command: CancelClosingCommand,
) -> CancelClosingCommand:
    if type(command) is not CancelClosingCommand:
        raise GatewayValidationError(
            "closing cancel command must use its exact immutable type"
        )
    _exact_text(command.intent_id, "intent_id")
    _exact_text(command.idempotency_key, "idempotency_key")
    _exact_text(command.owner, "owner")
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
