"""Supervised dashboard opening for exact two-leg credit spreads.

The browser never supplies authoritative order economics.  A server-side
preview is converted into a short-lived, account-bound HMAC proposal.  A
confirmed proposal is then translated into one durable gateway command with a
proposal-bound idempotency key and a separate browser correlation id.

This module deliberately has no broker session or mutation transport.  The
reviewed execution composition root injects the durable gateway.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Protocol

from live_trading.etrade_broker_reader import OpeningQuoteRead
from live_trading.etrade_broker_transport import SelectedBrokerAccount
from live_trading.etrade_order_gateway import (
    GatewayMutationResult,
    SubmitOpeningCommand,
)
from live_trading.market_sessions import (
    MarketSessionUnavailable,
    require_open_nyse_session,
)
from live_trading.order_domain import OptionContractId, OrderDomainError
from live_trading.order_intent_ledger import IntentRecord, OrderIntent
from live_trading.runtime_config import RuntimeConfig
from live_trading.runtime_safety import (
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
)


PROPOSAL_SCHEMA = "dashboard-manual-credit-spread.v1"
IDEMPOTENCY_SCOPE = "dashboard-manual-open.v1"
_ALLOWED_TICKERS = frozenset({"SPY", "SPX"})
_ALLOWED_SIDES = frozenset({"PUT", "CALL"})
_PROPOSAL_FIELDS = frozenset(
    {
        "schema",
        "proposal_id",
        "runtime_config_sha256",
        "account_binding_sha256",
        "account_id",
        "environment",
        "ticker",
        "broker_symbol",
        "side",
        "expiration",
        "sell_strike",
        "buy_strike",
        "sell_osi_key",
        "buy_osi_key",
        "quote_receipt_sha256",
        "quote_snapshot_sha256",
        "sell_bid_cents",
        "sell_ask_cents",
        "buy_bid_cents",
        "buy_ask_cents",
        "sell_quote_observed_at",
        "buy_quote_observed_at",
        "limit_credit",
        "observed_at",
        "expires_at",
    }
)
_MAX_TOKEN_BYTES = 8 * 1024
_MAX_IDENTIFIER_LENGTH = 256
_MAX_QUOTE_SKEW_SECONDS = 5
_SESSION_CLOSE_BUFFER_SECONDS = 15
_OPERATOR_STATUS_BY_DURABLE_STATE = {
    "INTENT": "DO_NOT_RETRY",
    "CLAIMED": "DO_NOT_RETRY",
    "SUBMISSION_UNKNOWN": "DO_NOT_RETRY",
    "SUBMITTED": "BROKER_ACKNOWLEDGED",
    "FILLED": "FILLED",
    "CANCELLED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
    "FAILED": "NOT_SENT",
}


class ManualOpenError(RuntimeError):
    """Base error for the supervised manual-opening contract."""

    code = "MANUAL_OPEN_ERROR"


class ManualOpenUnavailable(ManualOpenError):
    """The runtime is not configured or armed for manual opening."""

    code = "MANUAL_OPEN_UNAVAILABLE"


class ManualOpenValidationError(ManualOpenError):
    """A preview, proposal, or confirmation violated the closed contract."""

    code = "MANUAL_OPEN_INVALID"


class ManualOpeningGateway(Protocol):
    """Narrow mutation capability accepted by the application service."""

    @property
    def execution_ready(self) -> bool: ...

    def submit_opening(
        self, command: SubmitOpeningCommand
    ) -> GatewayMutationResult: ...

    def recent_opening_intents(
        self,
        *,
        idempotency_scope: str,
        limit: int,
    ) -> tuple[IntentRecord, ...]: ...


class ManualOpeningQuoteReader(Protocol):
    """Narrow origin-pinned read capability accepted by this service."""

    def read_opening_quotes(
        self,
        account: SelectedBrokerAccount,
        contracts: tuple[OptionContractId, OptionContractId],
    ) -> OpeningQuoteRead: ...


@dataclass(frozen=True, slots=True)
class ManualSpreadPreview:
    """Candidate spread identity selected by the read-only dashboard scanner.

    ``limit_credit`` and ``observed_at`` remain for compatibility with the
    scanner response but never authorize execution.  The service replaces
    both with a durable exact two-leg E*TRADE quote read.
    """

    ticker: str
    broker_symbol: str
    side: str
    expiration: date
    sell_strike: Decimal
    buy_strike: Decimal
    sell_osi_key: str
    buy_osi_key: str
    limit_credit: Decimal
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class _VerifiedManualSpread:
    ticker: str
    broker_symbol: str
    side: str
    expiration: date
    sell_strike: Decimal
    buy_strike: Decimal
    sell_osi_key: str
    buy_osi_key: str
    quote_receipt_sha256: str
    quote_snapshot_sha256: str
    sell_bid_cents: int
    sell_ask_cents: int
    buy_bid_cents: int
    buy_ask_cents: int
    sell_quote_observed_at: datetime
    buy_quote_observed_at: datetime
    limit_credit: Decimal
    observed_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class ManualOpenProposal:
    """Safe dashboard projection of a signed executable proposal."""

    proposal_token: str
    proposal_id: str
    expires_at: datetime
    account_id: str
    environment: str
    runtime_config_sha256: str
    ticker: str
    broker_symbol: str
    side: str
    expiration: date
    sell_strike: Decimal
    buy_strike: Decimal
    sell_osi_key: str
    buy_osi_key: str
    quote_receipt_sha256: str
    quote_snapshot_sha256: str
    sell_quote_observed_at: datetime
    buy_quote_observed_at: datetime
    quote_observed_at: datetime
    limit_credit: Decimal
    max_quantity: int

    def __repr__(self) -> str:
        return (
            "ManualOpenProposal("
            f"proposal_id={self.proposal_id!r}, "
            f"ticker={self.ticker!r}, side={self.side!r}, "
            "proposal_token=[REDACTED])"
        )

    def dashboard_payload(self) -> dict[str, Any]:
        return {
            "proposal_token": self.proposal_token,
            "proposal_id": self.proposal_id,
            "proposal_expires_at": self.expires_at.isoformat(),
            "execution_account_id": self.account_id,
            "execution_environment": self.environment,
            "runtime_config_sha256": self.runtime_config_sha256,
            "ticker": self.ticker,
            "broker_symbol": self.broker_symbol,
            "side": self.side,
            "expiration": self.expiration.isoformat(),
            "sell_strike": _decimal_text(self.sell_strike),
            "buy_strike": _decimal_text(self.buy_strike),
            "sell_osi_key": self.sell_osi_key,
            "buy_osi_key": self.buy_osi_key,
            "premium": _decimal_text(self.limit_credit),
            "limit_credit": _decimal_text(self.limit_credit),
            "quote_receipt_sha256": self.quote_receipt_sha256,
            "quote_snapshot_sha256": self.quote_snapshot_sha256,
            "sell_quote_observed_at":
                self.sell_quote_observed_at.isoformat(),
            "buy_quote_observed_at":
                self.buy_quote_observed_at.isoformat(),
            "quote_observed_at": self.quote_observed_at.isoformat(),
            "max_quantity": self.max_quantity,
            "execution_enabled": True,
        }


@dataclass(frozen=True, slots=True)
class ManualOpenSubmission:
    """Redacted result returned to the authenticated dashboard."""

    request_id: str
    intent_id: str
    state: str
    broker_order_id: str | None
    reason_code: str | None
    created: bool

    def dashboard_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "state": self.state,
            "broker_order_id": self.broker_order_id,
            "reason_code": self.reason_code,
            "created": self.created,
        }


@dataclass(frozen=True, slots=True)
class ManualOpenStatus:
    """Redacted durable status for one supervised manual-opening intent."""

    proposal_id: str
    intent_id: str
    status: str
    durable_state: str
    broker_order_id: str | None
    reason_code: str | None
    created_at: datetime
    updated_at: datetime
    ticker: str
    side: str
    expiration: date
    sell_strike: Decimal
    buy_strike: Decimal
    limit_credit: Decimal
    quantity: int

    def dashboard_payload(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "intent_id": self.intent_id,
            "status": self.status,
            "durable_state": self.durable_state,
            "broker_order_id": self.broker_order_id,
            "reason_code": self.reason_code,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ticker": self.ticker,
            "side": self.side,
            "expiration": self.expiration.isoformat(),
            "sell_strike": _decimal_text(self.sell_strike),
            "buy_strike": _decimal_text(self.buy_strike),
            "limit_credit": _decimal_text(self.limit_credit),
            "quantity": self.quantity,
        }


class ManualOpenService:
    """Bind signed dashboard proposals to the durable order gateway."""

    def __init__(
        self,
        *,
        gateway: ManualOpeningGateway,
        quote_reader: ManualOpeningQuoteReader,
        runtime_config: RuntimeConfig,
        runtime_safety: RuntimeSafetyBoundary,
        proposal_secret: str,
        owner: str = "dashboard-manual-open",
        clock=None,
    ) -> None:
        if type(runtime_config) is not RuntimeConfig:
            raise ManualOpenUnavailable(
                "manual opening requires an exact runtime configuration"
            )
        if type(runtime_safety) is not RuntimeSafetyBoundary:
            raise ManualOpenUnavailable(
                "manual opening requires an exact runtime safety boundary"
            )
        if (
            runtime_config.schema_version < 2
            or not runtime_config.execution.broker_mutations_enabled
        ):
            raise ManualOpenUnavailable(
                "runtime configuration does not enable supervised manual opening"
            )
        if runtime_config.mode not in {"sandbox", "live"}:
            raise ManualOpenUnavailable(
                "manual opening is supported only in sandbox or live mode"
            )
        if runtime_config.broker_environment != runtime_safety.environment:
            raise ManualOpenUnavailable(
                "runtime configuration and armed broker environment differ"
            )
        selected = runtime_config.selected_account
        if selected is None:
            raise ManualOpenUnavailable(
                "manual opening requires one exact selected account"
            )
        expected_identity = (
            runtime_safety.expected_account_id,
            runtime_safety.expected_account_id_key,
            runtime_safety.expected_institution_type,
        )
        if expected_identity != (
            selected.account_id,
            selected.account_id_key,
            selected.institution_type,
        ):
            raise ManualOpenUnavailable(
                "runtime configuration and armed account identity differ"
            )
        if (
            not runtime_config.strategy.enabled
            or not runtime_config.strategy.symbols
            or not set(runtime_config.strategy.symbols).issubset(
                _ALLOWED_TICKERS
            )
        ):
            raise ManualOpenUnavailable(
                "manual opening requires an enabled SPY/SPX-only strategy"
            )
        if runtime_config.model.required_for_entry:
            raise ManualOpenUnavailable(
                "manual opening cannot satisfy a required model authorization"
            )
        risk = runtime_config.risk
        if (
            risk.max_order_contracts <= 0
            or risk.max_order_loss_cents <= 0
            or risk.max_account_open_risk_cents <= 0
        ):
            raise ManualOpenUnavailable(
                "manual opening requires positive immutable risk limits"
            )
        if (
            type(proposal_secret) is not str
            or len(proposal_secret) < 32
            or len(proposal_secret) > 4096
        ):
            raise ManualOpenUnavailable(
                "manual opening proposal secret is unavailable"
            )
        _identifier(owner, "owner")
        if (
            not callable(getattr(gateway, "submit_opening", None))
            or not callable(
                getattr(gateway, "recent_opening_intents", None)
            )
            or type(getattr(gateway, "execution_ready", None)) is not bool
        ):
            raise ManualOpenUnavailable(
                "durable opening gateway is unavailable"
            )
        if not callable(
            getattr(quote_reader, "read_opening_quotes", None)
        ):
            raise ManualOpenUnavailable(
                "durable opening quote reader is unavailable"
            )

        self._gateway = gateway
        self._quote_reader = quote_reader
        self._config = runtime_config
        self._runtime_safety = runtime_safety
        self._proposal_secret = proposal_secret.encode("utf-8")
        self._owner = owner
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._submit_lock = threading.Lock()
        self._account_id = selected.account_id
        self._selected_account = SelectedBrokerAccount(
            selected.account_id,
            selected.account_id_key,
            selected.institution_type,
        )
        self._account_binding_sha256 = _account_binding_sha256(
            selected.account_id,
            selected.account_id_key,
            selected.institution_type,
        )

    @property
    def account_id(self) -> str:
        """Exact configured account used by the injected durable gateway."""

        return self._account_id

    @property
    def environment(self) -> str:
        """Non-secret broker environment bound into every proposal."""

        return self._runtime_safety.environment

    @property
    def runtime_config_sha256(self) -> str:
        """Non-secret immutable runtime configuration binding."""

        return self._config.source_sha256

    @property
    def execution_enabled(self) -> bool:
        try:
            self._require_execution_ready(self._now())
        except ManualOpenError:
            return False
        return True

    def recent_submissions(
        self, *, limit: int = 10
    ) -> tuple[ManualOpenStatus, ...]:
        """Return a redacted projection of recent durable manual intents."""

        if (
            type(limit) is not int
            or limit < 1
            or limit > 25
        ):
            raise ManualOpenValidationError(
                "recent manual-opening limit must be between 1 and 25"
            )
        try:
            records = self._gateway.recent_opening_intents(
                idempotency_scope=IDEMPOTENCY_SCOPE,
                limit=limit,
            )
        except Exception as exc:
            raise ManualOpenUnavailable(
                "durable manual-opening status is unavailable"
            ) from exc
        if type(records) is not tuple:
            raise ManualOpenUnavailable(
                "durable opening gateway returned invalid status"
            )

        statuses: list[ManualOpenStatus] = []
        for record in records:
            if type(record) is not IntentRecord:
                raise ManualOpenUnavailable(
                    "durable opening gateway returned invalid status"
                )
            envelope = record.envelope
            if type(envelope) is not OrderIntent:
                raise ManualOpenUnavailable(
                    "durable opening gateway returned invalid status"
                )
            if (
                envelope.account_id != self._account_id
                or envelope.environment
                != self._runtime_safety.environment
                or envelope.idempotency_scope != IDEMPOTENCY_SCOPE
                or envelope.intent_kind != "OPENING"
            ):
                raise ManualOpenUnavailable(
                    "durable opening gateway returned out-of-scope status"
                )
            statuses.append(_manual_open_status(record))
        return tuple(statuses)

    def issue_proposal(
        self, preview: ManualSpreadPreview
    ) -> ManualOpenProposal:
        now = self._now()
        session_close = self._require_execution_ready(now)
        submission_deadline = (
            session_close
            - timedelta(seconds=_SESSION_CLOSE_BUFFER_SECONDS)
        )
        if submission_deadline <= now:
            raise ManualOpenUnavailable(
                "manual opening is too close to the NYSE session close"
            )
        checked = self._refresh_preview(preview, now=now)
        expires_at = min(
            checked.observed_at
            + timedelta(
                seconds=self._config.risk.max_quote_age_seconds
            ),
            now
            + timedelta(
                seconds=self._config.risk.max_quote_age_seconds
            ),
            submission_deadline,
        )
        if expires_at <= now:
            raise ManualOpenValidationError(
                "manual opening preview is already stale"
            )
        proposal_id = str(uuid.uuid4())
        material = {
            "schema": PROPOSAL_SCHEMA,
            "proposal_id": proposal_id,
            "runtime_config_sha256": self._config.source_sha256,
            "account_binding_sha256": self._account_binding_sha256,
            "account_id": self._account_id,
            "environment": self._runtime_safety.environment,
            "ticker": checked.ticker,
            "broker_symbol": checked.broker_symbol,
            "side": checked.side,
            "expiration": checked.expiration.isoformat(),
            "sell_strike": _decimal_text(checked.sell_strike),
            "buy_strike": _decimal_text(checked.buy_strike),
            "sell_osi_key": checked.sell_osi_key,
            "buy_osi_key": checked.buy_osi_key,
            "quote_receipt_sha256":
                checked.quote_receipt_sha256,
            "quote_snapshot_sha256":
                checked.quote_snapshot_sha256,
            "sell_bid_cents": checked.sell_bid_cents,
            "sell_ask_cents": checked.sell_ask_cents,
            "buy_bid_cents": checked.buy_bid_cents,
            "buy_ask_cents": checked.buy_ask_cents,
            "sell_quote_observed_at":
                checked.sell_quote_observed_at.isoformat(),
            "buy_quote_observed_at":
                checked.buy_quote_observed_at.isoformat(),
            "limit_credit": _decimal_text(checked.limit_credit),
            "observed_at": checked.observed_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        payload = _canonical_json_bytes(material)
        signature = hmac.new(
            self._proposal_secret, payload, hashlib.sha256
        ).digest()
        token = f"{_b64encode(payload)}.{_b64encode(signature)}"
        return ManualOpenProposal(
            proposal_token=token,
            proposal_id=proposal_id,
            expires_at=expires_at,
            account_id=self._account_id,
            environment=self._runtime_safety.environment,
            runtime_config_sha256=self._config.source_sha256,
            ticker=checked.ticker,
            broker_symbol=checked.broker_symbol,
            side=checked.side,
            expiration=checked.expiration,
            sell_strike=checked.sell_strike,
            buy_strike=checked.buy_strike,
            sell_osi_key=checked.sell_osi_key,
            buy_osi_key=checked.buy_osi_key,
            quote_receipt_sha256=checked.quote_receipt_sha256,
            quote_snapshot_sha256=checked.quote_snapshot_sha256,
            sell_quote_observed_at=checked.sell_quote_observed_at,
            buy_quote_observed_at=checked.buy_quote_observed_at,
            quote_observed_at=checked.observed_at,
            limit_credit=checked.limit_credit,
            max_quantity=self._config.risk.max_order_contracts,
        )

    def submit(
        self,
        *,
        proposal_token: str,
        quantity: int,
        request_id: str,
    ) -> ManualOpenSubmission:
        """Submit once; the signed proposal id is the durable retry key."""

        with self._submit_lock:
            now = self._now()
            self._require_execution_ready(now)
            (
                preview,
                proposal_payload,
                proposal_id,
                proposal_expires_at,
            ) = self._verify_proposal(proposal_token, now=now)
            canonical_request_id = _request_id(request_id)
            if (
                type(quantity) is not int
                or not 1
                <= quantity
                <= self._config.risk.max_order_contracts
            ):
                raise ManualOpenValidationError(
                    "quantity exceeds the configured per-order limit"
                )
            payload = _opening_payload(preview, quantity)
            max_loss, collateral, max_loss_cents = _opening_exposure(
                preview, quantity
            )
            if max_loss_cents > self._config.risk.max_order_loss_cents:
                raise ManualOpenValidationError(
                    "order maximum loss exceeds the configured limit"
                )
            decision_id = (
                "manual-"
                + hashlib.sha256(
                    proposal_payload
                    + b"\0"
                    + str(quantity).encode("ascii")
                ).hexdigest()[:48]
            )
            payload_bytes = _canonical_json_bytes(payload)
            result = self._gateway.submit_opening(
                SubmitOpeningCommand(
                    strategy_id=self._config.strategy.strategy_id,
                    decision_id=decision_id,
                    idempotency_scope=IDEMPOTENCY_SCOPE,
                    # A proposal represents one operator-reviewed economic
                    # decision.  Binding durable idempotency to the signed
                    # proposal prevents a new browser request id from turning
                    # one confirmation into a second order.
                    idempotency_key=proposal_id,
                    payload_bytes=payload_bytes,
                    max_loss_amount=max_loss,
                    collateral_amount=collateral,
                    quote_observed_at=preview.observed_at,
                    quote_valid_until=proposal_expires_at,
                    quote_digest=preview.quote_receipt_sha256,
                    owner=self._owner,
                )
            )
            if type(result) is not GatewayMutationResult:
                raise ManualOpenUnavailable(
                    "durable opening gateway returned an invalid result"
                )
            return ManualOpenSubmission(
                request_id=canonical_request_id,
                intent_id=result.intent_id,
                state=result.state,
                broker_order_id=result.broker_order_id,
                reason_code=result.reason_code,
                created=result.created,
            )

    def _verify_proposal(
        self,
        token: str,
        *,
        now: datetime,
    ) -> tuple[_VerifiedManualSpread, bytes, str, datetime]:
        if (
            type(token) is not str
            or not token
            or len(token.encode("utf-8")) > _MAX_TOKEN_BYTES
            or token.count(".") != 1
        ):
            raise ManualOpenValidationError(
                "manual opening proposal token is invalid"
            )
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _b64decode(encoded_payload)
        signature = _b64decode(encoded_signature)
        expected = hmac.new(
            self._proposal_secret, payload, hashlib.sha256
        ).digest()
        if len(signature) != len(expected) or not hmac.compare_digest(
            signature, expected
        ):
            raise ManualOpenValidationError(
                "manual opening proposal signature is invalid"
            )
        material = _strict_json_object(payload)
        if set(material) != _PROPOSAL_FIELDS:
            raise ManualOpenValidationError(
                "manual opening proposal schema is invalid"
            )
        if (
            material["schema"] != PROPOSAL_SCHEMA
            or material["runtime_config_sha256"]
            != self._config.source_sha256
            or material["account_binding_sha256"]
            != self._account_binding_sha256
            or material["account_id"] != self._account_id
            or material["environment"]
            != self._runtime_safety.environment
        ):
            raise ManualOpenValidationError(
                "manual opening proposal is bound to another runtime"
            )
        proposal_id = _request_id(material["proposal_id"])
        observed_at = _utc_datetime(
            material["observed_at"], "proposal observed_at"
        )
        expires_at = _utc_datetime(
            material["expires_at"], "proposal expires_at"
        )
        if (
            expires_at <= now
            or expires_at
            > observed_at
            + timedelta(
                seconds=self._config.risk.max_quote_age_seconds
            )
        ):
            raise ManualOpenValidationError(
                "manual opening proposal is stale"
            )
        preview = _VerifiedManualSpread(
            ticker=material["ticker"],
            broker_symbol=material["broker_symbol"],
            side=material["side"],
            expiration=_iso_date(
                material["expiration"], "proposal expiration"
            ),
            sell_strike=_positive_decimal(
                material["sell_strike"], "proposal sell strike"
            ),
            buy_strike=_positive_decimal(
                material["buy_strike"], "proposal buy strike"
            ),
            sell_osi_key=material["sell_osi_key"],
            buy_osi_key=material["buy_osi_key"],
            quote_receipt_sha256=_sha256_text(
                material["quote_receipt_sha256"],
                "proposal quote receipt",
            ),
            quote_snapshot_sha256=_sha256_text(
                material["quote_snapshot_sha256"],
                "proposal quote snapshot",
            ),
            sell_bid_cents=_quote_cents(
                material["sell_bid_cents"], "proposal sell bid"
            ),
            sell_ask_cents=_quote_cents(
                material["sell_ask_cents"], "proposal sell ask"
            ),
            buy_bid_cents=_quote_cents(
                material["buy_bid_cents"], "proposal buy bid"
            ),
            buy_ask_cents=_quote_cents(
                material["buy_ask_cents"], "proposal buy ask"
            ),
            sell_quote_observed_at=_utc_datetime(
                material["sell_quote_observed_at"],
                "proposal sell quote observed_at",
            ),
            buy_quote_observed_at=_utc_datetime(
                material["buy_quote_observed_at"],
                "proposal buy quote observed_at",
            ),
            limit_credit=_positive_decimal(
                material["limit_credit"], "proposal limit credit"
            ),
            observed_at=observed_at,
        )
        return (
            self._validate_verified_preview(preview, now=now),
            payload,
            proposal_id,
            expires_at,
        )

    def _refresh_preview(
        self,
        preview: ManualSpreadPreview,
        *,
        now: datetime,
    ) -> _VerifiedManualSpread:
        if type(preview) is not ManualSpreadPreview:
            raise ManualOpenValidationError(
                "manual opening preview has an invalid type"
            )
        ticker = _ascii_choice(
            preview.ticker, _ALLOWED_TICKERS, "ticker"
        )
        if ticker not in self._config.strategy.symbols:
            raise ManualOpenValidationError(
                "ticker is not enabled by runtime configuration"
            )
        side = _ascii_choice(preview.side, _ALLOWED_SIDES, "side")
        broker_symbol = _broker_symbol(
            preview.broker_symbol, ticker
        )
        if type(preview.expiration) is not date:
            raise ManualOpenValidationError(
                "expiration must be an exact date"
            )
        if preview.expiration < now.date():
            raise ManualOpenValidationError(
                "expiration is already past"
            )
        sell_strike = _positive_decimal(
            preview.sell_strike, "sell strike"
        )
        buy_strike = _positive_decimal(
            preview.buy_strike, "buy strike"
        )
        if (
            side == "PUT" and sell_strike <= buy_strike
        ) or (
            side == "CALL" and sell_strike >= buy_strike
        ):
            raise ManualOpenValidationError(
                "credit-spread strike orientation is invalid"
            )
        sell_osi_key = _standard_osi_key(
            preview.sell_osi_key,
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            strike=sell_strike,
            label="sell",
        )
        buy_osi_key = _standard_osi_key(
            preview.buy_osi_key,
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            strike=buy_strike,
            label="buy",
        )
        if sell_osi_key == buy_osi_key:
            raise ManualOpenValidationError(
                "spread legs must retain distinct OSI identities"
            )
        _json_number(sell_strike, "sell strike")
        _json_number(buy_strike, "buy strike")
        try:
            sell_contract = OptionContractId(
                symbol=broker_symbol,
                expiry=preview.expiration,
                call_put=side,
                strike=sell_strike,
                osi_key=sell_osi_key,
                multiplier=Decimal("100"),
                adjusted=False,
                deliverables=None,
            )
            buy_contract = OptionContractId(
                symbol=broker_symbol,
                expiry=preview.expiration,
                call_put=side,
                strike=buy_strike,
                osi_key=buy_osi_key,
                multiplier=Decimal("100"),
                adjusted=False,
                deliverables=None,
            )
        except OrderDomainError as exc:
            raise ManualOpenValidationError(
                "spread contract identity is inconsistent"
            ) from exc
        try:
            quote_read = self._quote_reader.read_opening_quotes(
                self._selected_account,
                (sell_contract, buy_contract),
            )
        except Exception as exc:
            raise ManualOpenUnavailable(
                "exact two-leg E*TRADE quote evidence is unavailable"
            ) from exc
        if type(quote_read) is not OpeningQuoteRead:
            raise ManualOpenUnavailable(
                "opening quote reader returned an invalid result"
            )
        if quote_read.snapshot.complete is not True:
            raise ManualOpenUnavailable(
                "opening quote evidence is incomplete"
            )
        by_contract = {
            quote.contract: quote
            for quote in quote_read.snapshot.quotes
        }
        if set(by_contract) != {sell_contract, buy_contract}:
            raise ManualOpenUnavailable(
                "opening quote evidence changed spread identity"
            )
        sell_quote = by_contract[sell_contract]
        buy_quote = by_contract[buy_contract]
        midpoint_credit_cents = (
            sell_quote.bid_cents
            + sell_quote.ask_cents
            - buy_quote.bid_cents
            - buy_quote.ask_cents
        ) // 2
        if (
            sell_quote.bid_cents <= 0
            or buy_quote.bid_cents <= 0
            or midpoint_credit_cents <= 0
        ):
            raise ManualOpenValidationError(
                "two-leg E*TRADE quote does not support a positive credit"
            )
        verified = _VerifiedManualSpread(
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            sell_strike=sell_strike,
            buy_strike=buy_strike,
            sell_osi_key=sell_osi_key,
            buy_osi_key=buy_osi_key,
            quote_receipt_sha256=
                quote_read.receipt.receipt_sha256,
            quote_snapshot_sha256=
                quote_read.snapshot.snapshot_sha256,
            sell_bid_cents=sell_quote.bid_cents,
            sell_ask_cents=sell_quote.ask_cents,
            buy_bid_cents=buy_quote.bid_cents,
            buy_ask_cents=buy_quote.ask_cents,
            sell_quote_observed_at=sell_quote.observed_at,
            buy_quote_observed_at=buy_quote.observed_at,
            limit_credit=(
                Decimal(midpoint_credit_cents) / Decimal("100")
            ),
            observed_at=min(
                sell_quote.observed_at,
                buy_quote.observed_at,
            ),
        )
        return self._validate_verified_preview(verified, now=now)

    def _validate_verified_preview(
        self,
        preview: _VerifiedManualSpread,
        *,
        now: datetime,
    ) -> _VerifiedManualSpread:
        if type(preview) is not _VerifiedManualSpread:
            raise ManualOpenValidationError(
                "verified manual opening preview has an invalid type"
            )
        ticker = _ascii_choice(
            preview.ticker, _ALLOWED_TICKERS, "ticker"
        )
        if ticker not in self._config.strategy.symbols:
            raise ManualOpenValidationError(
                "ticker is not enabled by runtime configuration"
            )
        side = _ascii_choice(preview.side, _ALLOWED_SIDES, "side")
        broker_symbol = _broker_symbol(
            preview.broker_symbol, ticker
        )
        if (
            type(preview.expiration) is not date
            or preview.expiration < now.date()
        ):
            raise ManualOpenValidationError(
                "expiration is invalid or already past"
            )
        sell_strike = _positive_decimal(
            preview.sell_strike, "sell strike"
        )
        buy_strike = _positive_decimal(
            preview.buy_strike, "buy strike"
        )
        if (
            side == "PUT" and sell_strike <= buy_strike
        ) or (
            side == "CALL" and sell_strike >= buy_strike
        ):
            raise ManualOpenValidationError(
                "credit-spread strike orientation is invalid"
            )
        sell_osi_key = _standard_osi_key(
            preview.sell_osi_key,
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            strike=sell_strike,
            label="sell",
        )
        buy_osi_key = _standard_osi_key(
            preview.buy_osi_key,
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            strike=buy_strike,
            label="buy",
        )
        if sell_osi_key == buy_osi_key:
            raise ManualOpenValidationError(
                "spread legs must retain distinct OSI identities"
            )
        quote_receipt_sha256 = _sha256_text(
            preview.quote_receipt_sha256, "quote receipt"
        )
        quote_snapshot_sha256 = _sha256_text(
            preview.quote_snapshot_sha256, "quote snapshot"
        )
        sell_bid_cents = _quote_cents(
            preview.sell_bid_cents, "sell bid"
        )
        sell_ask_cents = _quote_cents(
            preview.sell_ask_cents, "sell ask"
        )
        buy_bid_cents = _quote_cents(
            preview.buy_bid_cents, "buy bid"
        )
        buy_ask_cents = _quote_cents(
            preview.buy_ask_cents, "buy ask"
        )
        if (
            sell_bid_cents <= 0
            or buy_bid_cents <= 0
            or sell_ask_cents < sell_bid_cents
            or buy_ask_cents < buy_bid_cents
        ):
            raise ManualOpenValidationError(
                "opening quote NBBO is unusable"
            )
        sell_observed_at = _exact_utc(
            preview.sell_quote_observed_at,
            "sell quote observed_at",
        )
        buy_observed_at = _exact_utc(
            preview.buy_quote_observed_at,
            "buy quote observed_at",
        )
        observed_at = _exact_utc(
            preview.observed_at, "preview observed_at"
        )
        if observed_at != min(sell_observed_at, buy_observed_at):
            raise ManualOpenValidationError(
                "opening quote observation time is not the oldest leg"
            )
        if (
            abs(
                (
                    sell_observed_at - buy_observed_at
                ).total_seconds()
            )
            > _MAX_QUOTE_SKEW_SECONDS
            or any(
                timestamp > now + timedelta(seconds=5)
                or now - timestamp
                > timedelta(
                    seconds=self._config.risk.max_quote_age_seconds
                )
                for timestamp in (
                    sell_observed_at,
                    buy_observed_at,
                )
            )
        ):
            raise ManualOpenValidationError(
                "opening quote legs are stale, future, or time-skewed"
            )
        expected_credit_cents = (
            sell_bid_cents
            + sell_ask_cents
            - buy_bid_cents
            - buy_ask_cents
        ) // 2
        credit = _positive_decimal(
            preview.limit_credit, "limit credit"
        )
        if (
            expected_credit_cents <= 0
            or credit
            != Decimal(expected_credit_cents) / Decimal("100")
            or credit >= abs(sell_strike - buy_strike)
        ):
            raise ManualOpenValidationError(
                "limit credit is not the exact two-leg quote midpoint"
            )
        _json_number(sell_strike, "sell strike")
        _json_number(buy_strike, "buy strike")
        _json_number(credit, "limit credit")
        return _VerifiedManualSpread(
            ticker=ticker,
            broker_symbol=broker_symbol,
            side=side,
            expiration=preview.expiration,
            sell_strike=sell_strike,
            buy_strike=buy_strike,
            sell_osi_key=sell_osi_key,
            buy_osi_key=buy_osi_key,
            quote_receipt_sha256=quote_receipt_sha256,
            quote_snapshot_sha256=quote_snapshot_sha256,
            sell_bid_cents=sell_bid_cents,
            sell_ask_cents=sell_ask_cents,
            buy_bid_cents=buy_bid_cents,
            buy_ask_cents=buy_ask_cents,
            sell_quote_observed_at=sell_observed_at,
            buy_quote_observed_at=buy_observed_at,
            limit_credit=credit,
            observed_at=observed_at,
        )

    def _now(self) -> datetime:
        return _exact_utc(self._clock(), "clock")

    def _require_current_runtime(self, now: datetime) -> None:
        try:
            self._runtime_safety.assert_current(now)
        except RuntimeSafetyError as exc:
            raise ManualOpenUnavailable(
                "manual opening runtime arm is not current"
            ) from exc

    def _require_execution_ready(self, now: datetime) -> datetime:
        """Require both mutation readiness and readable durable history."""

        self._require_current_runtime(now)
        try:
            session_close = require_open_nyse_session(now)
        except MarketSessionUnavailable as exc:
            raise ManualOpenUnavailable(
                "manual opening requires the open NYSE regular session"
            ) from exc
        if self._gateway.execution_ready is not True:
            raise ManualOpenUnavailable(
                "durable opening gateway is not reconciled"
            )
        # The dedicated manual-open scope is the recovery source after a lost
        # HTTP response.  If it cannot be read and projected exactly, issuing
        # or submitting another proposal could hide an unresolved order.
        self.recent_submissions(limit=25)
        return session_close


def _opening_payload(
    preview: _VerifiedManualSpread,
    quantity: int,
) -> dict[str, Any]:
    expiration = preview.expiration
    common = {
        "symbol": preview.broker_symbol,
        "callPut": preview.side,
        "expiryYear": expiration.year,
        "expiryMonth": expiration.month,
        "expiryDay": expiration.day,
        "quantity": quantity,
    }
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": _json_number(
            preview.limit_credit, "limit credit"
        ),
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                **common,
                "strikePrice": _json_number(
                    preview.sell_strike, "sell strike"
                ),
                "orderAction": "SELL_OPEN",
            },
            {
                **common,
                "strikePrice": _json_number(
                    preview.buy_strike, "buy strike"
                ),
                "orderAction": "BUY_OPEN",
            },
        ],
    }


def _opening_exposure(
    preview: _VerifiedManualSpread,
    quantity: int,
) -> tuple[Decimal, Decimal, int]:
    with localcontext() as context:
        context.prec = 38
        width = abs(preview.sell_strike - preview.buy_strike)
        max_loss = (
            width - preview.limit_credit
        ) * Decimal("100") * quantity
        collateral = max_loss
        cents = max_loss * Decimal("100")
    if (
        max_loss <= 0
        or collateral <= 0
        or cents != cents.to_integral_value()
    ):
        raise ManualOpenValidationError(
            "credit spread does not have exact positive risk"
        )
    return max_loss, collateral, int(cents)


def _manual_open_status(record: IntentRecord) -> ManualOpenStatus:
    try:
        proposal_id = _request_id(record.envelope.idempotency_key)
        intent_id = _identifier(record.intent_id, "intent id")
        broker_order_id = (
            None
            if record.broker_order_id is None
            else _identifier(
                record.broker_order_id, "broker order id"
            )
        )
        created_at = _exact_utc(
            record.created_at, "intent created_at"
        )
        updated_at = _exact_utc(
            record.updated_at, "intent updated_at"
        )
        if updated_at < created_at:
            raise ManualOpenValidationError(
                "intent timestamps are inconsistent"
            )
        (
            ticker,
            side,
            expiration,
            sell_strike,
            buy_strike,
            limit_credit,
            quantity,
        ) = _manual_open_payload_summary(
            record.envelope.wire_payload
        )
    except ManualOpenValidationError as exc:
        raise ManualOpenUnavailable(
            "durable manual-opening status is invalid"
        ) from exc

    durable_state = record.state
    if (
        type(durable_state) is not str
        or durable_state not in _OPERATOR_STATUS_BY_DURABLE_STATE
    ):
        raise ManualOpenUnavailable(
            "durable manual-opening state is invalid"
        )
    if durable_state == "FAILED" and broker_order_id is not None:
        raise ManualOpenUnavailable(
            "failed manual-opening intent cannot have a broker order id"
        )
    return ManualOpenStatus(
        proposal_id=proposal_id,
        intent_id=intent_id,
        status=_OPERATOR_STATUS_BY_DURABLE_STATE[durable_state],
        durable_state=durable_state,
        broker_order_id=broker_order_id,
        reason_code=None,
        created_at=created_at,
        updated_at=updated_at,
        ticker=ticker,
        side=side,
        expiration=expiration,
        sell_strike=sell_strike,
        buy_strike=buy_strike,
        limit_credit=limit_credit,
        quantity=quantity,
    )


def _manual_open_payload_summary(
    wire_payload: str,
) -> tuple[
    str,
    str,
    date,
    Decimal,
    Decimal,
    Decimal,
    int,
]:
    if (
        type(wire_payload) is not str
        or not wire_payload
        or not wire_payload.isascii()
    ):
        raise ManualOpenValidationError(
            "durable manual-opening payload is invalid"
        )
    payload = _strict_json_object(wire_payload.encode("ascii"))
    if set(payload) != {
        "securityType",
        "orderAction",
        "priceType",
        "limitPrice",
        "orderTerm",
        "spreadType",
        "legs",
    }:
        raise ManualOpenValidationError(
            "durable manual-opening payload shape is invalid"
        )
    if (
        payload["securityType"] != "OPTN"
        or payload["orderAction"] != "SPREAD"
        or payload["priceType"] != "NET_CREDIT"
        or payload["orderTerm"] != "GOOD_FOR_DAY"
        or payload["spreadType"] != "VERTICAL"
    ):
        raise ManualOpenValidationError(
            "durable manual-opening payload is not a credit vertical"
        )
    legs = payload["legs"]
    if type(legs) is not list or len(legs) != 2:
        raise ManualOpenValidationError(
            "durable manual-opening payload requires exactly two legs"
        )
    leg_fields = {
        "symbol",
        "callPut",
        "expiryYear",
        "expiryMonth",
        "expiryDay",
        "quantity",
        "strikePrice",
        "orderAction",
    }
    sell_leg, buy_leg = legs
    if (
        type(sell_leg) is not dict
        or type(buy_leg) is not dict
        or set(sell_leg) != leg_fields
        or set(buy_leg) != leg_fields
        or sell_leg["orderAction"] != "SELL_OPEN"
        or buy_leg["orderAction"] != "BUY_OPEN"
    ):
        raise ManualOpenValidationError(
            "durable manual-opening leg shape is invalid"
        )
    common_fields = (
        "symbol",
        "callPut",
        "expiryYear",
        "expiryMonth",
        "expiryDay",
        "quantity",
    )
    if any(
        sell_leg[field] != buy_leg[field]
        or type(sell_leg[field]) is not type(buy_leg[field])
        for field in common_fields
    ):
        raise ManualOpenValidationError(
            "durable manual-opening legs are inconsistent"
        )

    symbol = sell_leg["symbol"]
    if symbol == "SPY":
        ticker = "SPY"
    elif symbol in {"SPX", "SPXW"}:
        ticker = "SPX"
    else:
        raise ManualOpenValidationError(
            "durable manual-opening symbol is unsupported"
        )
    side = _ascii_choice(
        sell_leg["callPut"], _ALLOWED_SIDES, "side"
    )
    expiry_parts = (
        sell_leg["expiryYear"],
        sell_leg["expiryMonth"],
        sell_leg["expiryDay"],
    )
    if any(
        type(value) is not int
        for value in expiry_parts
    ):
        raise ManualOpenValidationError(
            "durable manual-opening expiration is invalid"
        )
    try:
        expiration = date(*expiry_parts)
    except ValueError as exc:
        raise ManualOpenValidationError(
            "durable manual-opening expiration is invalid"
        ) from exc
    quantity = sell_leg["quantity"]
    if type(quantity) is not int or quantity <= 0:
        raise ManualOpenValidationError(
            "durable manual-opening quantity is invalid"
        )

    sell_strike = _exact_wire_decimal(
        sell_leg["strikePrice"], "sell strike"
    )
    buy_strike = _exact_wire_decimal(
        buy_leg["strikePrice"], "buy strike"
    )
    limit_credit = _exact_wire_decimal(
        payload["limitPrice"], "limit credit"
    )
    if (
        side == "PUT" and sell_strike <= buy_strike
    ) or (
        side == "CALL" and sell_strike >= buy_strike
    ):
        raise ManualOpenValidationError(
            "durable manual-opening strike orientation is invalid"
        )
    if limit_credit >= abs(sell_strike - buy_strike):
        raise ManualOpenValidationError(
            "durable manual-opening credit is invalid"
        )
    return (
        ticker,
        side,
        expiration,
        sell_strike,
        buy_strike,
        limit_credit,
        quantity,
    )


def _exact_wire_decimal(value: Any, label: str) -> Decimal:
    result = _positive_decimal(value, label)
    encoded = _json_number(result, label)
    if type(encoded) is not type(value) or encoded != value:
        raise ManualOpenValidationError(
            f"{label} is not canonical in the durable payload"
        )
    return result


def _account_binding_sha256(
    account_id: str,
    account_id_key: str,
    institution_type: str,
) -> str:
    material = "\0".join(
        (account_id, account_id_key, institution_type)
    ).encode("utf-8")
    return hashlib.sha256(
        b"manual-open-account-binding.v1\0" + material
    ).hexdigest()


def _sha256_text(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManualOpenValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _quote_cents(value: Any, label: str) -> int:
    if (
        type(value) is not int
        or value < 0
        or value > 9_223_372_036_854_775_807
    ):
        raise ManualOpenValidationError(
            f"{label} must be exact non-negative cents"
        )
    return value


def _request_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
    ):
        raise ManualOpenValidationError(
            "request id must be a canonical UUID"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ManualOpenValidationError(
            "request id must be a canonical UUID"
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ManualOpenValidationError(
            "request id must be a canonical UUIDv4"
        )
    return value


def _identifier(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ManualOpenValidationError(
            f"{label} must be a printable ASCII identifier"
        )
    return value


def _ascii_choice(
    value: Any,
    choices: frozenset[str],
    label: str,
) -> str:
    if type(value) is not str or value not in choices:
        raise ManualOpenValidationError(
            f"{label} is not supported"
        )
    return value


def _broker_symbol(value: Any, ticker: str) -> str:
    allowed = {"SPY"} if ticker == "SPY" else {"SPX", "SPXW"}
    if type(value) is not str or value not in allowed:
        raise ManualOpenValidationError(
            "broker option symbol does not match the requested ticker"
        )
    return value


def _standard_osi_key(
    value: Any,
    *,
    ticker: str,
    broker_symbol: str,
    side: str,
    expiration: date,
    strike: Decimal,
    label: str,
) -> str:
    """Retain one exact unadjusted SPY/SPX OSI contract identity."""

    if (
        type(value) is not str
        or len(value) != 21
        or not value.isascii()
    ):
        raise ManualOpenValidationError(
            f"{label} leg OSI identity is malformed"
        )
    expected_symbols = (
        {"SPY"} if ticker == "SPY" else {"SPX", "SPXW"}
    )
    if broker_symbol not in expected_symbols:
        raise ManualOpenValidationError(
            f"{label} leg OSI identity is unsupported"
        )
    expected_roots = {
        broker_symbol.ljust(6, "-"),
        broker_symbol.ljust(6, " "),
    }
    expiry_text = value[6:12]
    option_type = value[12]
    strike_text = value[13:]
    if (
        value[:6] not in expected_roots
        or not expiry_text.isdigit()
        or option_type not in {"C", "P"}
        or not strike_text.isdigit()
    ):
        raise ManualOpenValidationError(
            f"{label} leg OSI identity is not a standard "
            f"unadjusted {ticker} contract"
        )
    try:
        osi_expiration = date(
            2000 + int(expiry_text[:2]),
            int(expiry_text[2:4]),
            int(expiry_text[4:6]),
        )
    except ValueError as exc:
        raise ManualOpenValidationError(
            f"{label} leg OSI identity is malformed"
        ) from exc
    osi_side = "CALL" if option_type == "C" else "PUT"
    osi_strike = Decimal(strike_text) / Decimal("1000")
    if (
        osi_expiration != expiration
        or osi_side != side
        or osi_strike != strike
    ):
        raise ManualOpenValidationError(
            f"{label} leg OSI identity does not match preview economics"
        )
    return value


def _positive_decimal(value: Any, label: str) -> Decimal:
    if type(value) is bool:
        raise ManualOpenValidationError(
            f"{label} must be a positive decimal"
        )
    try:
        result = value if type(value) is Decimal else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ManualOpenValidationError(
            f"{label} must be a positive decimal"
        ) from exc
    if (
        type(result) is not Decimal
        or not result.is_finite()
        or result <= 0
        or result.adjusted() > 12
        or result.adjusted() < -6
    ):
        raise ManualOpenValidationError(
            f"{label} must be a positive decimal"
        )
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _json_number(value: Decimal, label: str) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    result = float(value)
    if Decimal(str(result)) != value:
        raise ManualOpenValidationError(
            f"{label} cannot be represented exactly in the broker payload"
        )
    return result


def _exact_utc(value: Any, label: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is not timezone.utc:
        raise ManualOpenValidationError(
            f"{label} must use exact UTC"
        )
    return value


def _utc_datetime(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ManualOpenValidationError(f"{label} is invalid")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManualOpenValidationError(
            f"{label} is invalid"
        ) from exc
    if result.tzinfo is None:
        raise ManualOpenValidationError(
            f"{label} must include a timezone"
        )
    result = result.astimezone(timezone.utc)
    if result.isoformat() != value:
        raise ManualOpenValidationError(
            f"{label} is not canonical UTC"
        )
    return result


def _iso_date(value: Any, label: str) -> date:
    if type(value) is not str:
        raise ManualOpenValidationError(f"{label} is invalid")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ManualOpenValidationError(
            f"{label} is invalid"
        ) from exc
    if result.isoformat() != value:
        raise ManualOpenValidationError(
            f"{label} is not canonical"
        )
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ManualOpenValidationError(
            "manual opening data is not canonical JSON"
        ) from exc


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise ManualOpenValidationError(
            "manual opening proposal payload is invalid"
        )

    def object_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise ManualOpenValidationError(
                    "manual opening proposal contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ManualOpenValidationError(
                    "manual opening proposal contains a non-finite number"
                )
            ),
        )
    except ManualOpenValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManualOpenValidationError(
            "manual opening proposal payload is invalid"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_json_bytes(value) != payload
    ):
        raise ManualOpenValidationError(
            "manual opening proposal payload is not canonical"
        )
    return value


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    if (
        type(value) is not str
        or not value
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in value
        )
    ):
        raise ManualOpenValidationError(
            "manual opening proposal encoding is invalid"
        )
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise ManualOpenValidationError(
            "manual opening proposal encoding is invalid"
        ) from exc
    if _b64encode(decoded) != value:
        raise ManualOpenValidationError(
            "manual opening proposal encoding is not canonical"
        )
    return decoded


__all__ = [
    "ManualOpenError",
    "ManualOpenProposal",
    "ManualOpenService",
    "ManualOpenStatus",
    "ManualOpenSubmission",
    "ManualOpenUnavailable",
    "ManualOpenValidationError",
    "ManualSpreadPreview",
]
