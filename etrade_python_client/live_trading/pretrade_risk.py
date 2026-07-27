"""Pure, fail-closed pre-trade risk policy for opening option verticals.

This module performs no I/O, reads no clock, and owns no execution capability.
It accepts immutable evidence, derives exposure from exact order economics, and
returns a deterministic content-addressed decision.  A model or regime overlay
can only add a block or lower a limit; it can never make a base-policy denial
eligible.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from live_trading.order_domain import OptionContractId


PRETRADE_RISK_SCHEMA_VERSION = 1
_DECISION_DOMAIN = "etrade-pretrade-risk.v1"
_MAX_CENTS = (1 << 63) - 1
_MAX_QUANTITY = 1_000_000
_MAX_TEXT = 128
_MAX_ARM_LIFETIME_SECONDS = 15 * 60
_ENVIRONMENTS = frozenset({"sandbox", "production"})
_OPEN_ACTIONS = frozenset({"BUY_OPEN", "SELL_OPEN"})
_PRICE_TYPES = frozenset({"NET_CREDIT", "NET_DEBIT"})
_OVERLAY_KINDS = frozenset({"MODEL", "REGIME"})
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,14}\Z")
_STRATEGY = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PretradeRiskValidationError(ValueError):
    """Raised when a caller does not supply the closed typed contract."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class OpeningLeg:
    contract: OptionContractId
    action: str

    def __post_init__(self) -> None:
        if type(self.contract) is not OptionContractId:
            _invalid("INVALID_OPENING_CONTRACT")
        if type(self.action) is not str or self.action not in _OPEN_ACTIONS:
            _invalid("INVALID_OPENING_ACTION")

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (self.contract.canonical_material, self.action)


@dataclass(frozen=True, slots=True, repr=False)
class OpeningSpreadRequest:
    strategy_decision_id: str
    strategy_id: str
    environment: str
    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str
    trade_session: date
    legs: tuple[OpeningLeg, ...]
    quantity: int
    price_type: str
    limit_price_cents: int

    def __post_init__(self) -> None:
        _validate_identity(
            self.strategy_decision_id, "INVALID_STRATEGY_DECISION_ID"
        )
        _validate_strategy(self.strategy_id)
        _validate_environment(self.environment)
        _validate_account_fields(
            self.account_id, self.account_id_key, self.institution_type
        )
        _validate_date(self.trade_session, "INVALID_TRADE_SESSION")
        if type(self.legs) is not tuple or len(self.legs) != 2:
            _invalid("INVALID_OPENING_LEGS")
        for leg in self.legs:
            if type(leg) is not OpeningLeg:
                _invalid("INVALID_OPENING_LEG")
        _validate_int(
            self.quantity,
            "INVALID_ORDER_QUANTITY",
            minimum=1,
            maximum=_MAX_QUANTITY,
        )
        if (
            type(self.price_type) is not str
            or self.price_type not in _PRICE_TYPES
        ):
            _invalid("INVALID_NET_PRICE_TYPE")
        _validate_int(
            self.limit_price_cents,
            "INVALID_LIMIT_PRICE",
            minimum=1,
            maximum=_MAX_CENTS,
        )

    def __repr__(self) -> str:
        return "OpeningSpreadRequest([REDACTED])"

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.strategy_decision_id,
            self.strategy_id,
            self.environment,
            self.account_id,
            self.account_id_key,
            self.institution_type,
            self.trade_session.isoformat(),
            tuple(sorted(leg.canonical_material for leg in self.legs)),
            self.quantity,
            self.price_type,
            self.limit_price_cents,
        )


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Immutable ceilings; cents are integer USD cents.

    Order notional is one underlying-equivalent notional at the larger strike,
    not a sum of both hedged legs.
    """

    allowed_environments: tuple[str, ...]
    allowed_strategies: tuple[str, ...]
    allowed_symbols: tuple[str, ...]
    max_order_contracts: int
    max_order_notional_cents: int
    max_order_loss_cents: int
    max_order_collateral_cents: int
    max_account_open_risk_cents: int
    max_symbol_open_risk_cents: int
    min_remaining_buying_power_cents: int
    max_abs_portfolio_delta: Decimal
    max_abs_symbol_delta: Decimal
    max_daily_loss_cents: int
    max_daily_orders: int
    max_daily_new_risk_cents: int
    max_quote_age_seconds: int
    max_portfolio_age_seconds: int
    max_authority_age_seconds: int
    max_overlay_age_seconds: int
    max_option_ask_cents: int
    max_bid_ask_width_cents: int
    max_fee_cents_per_contract_per_leg: int
    min_open_interest: int
    min_volume: int
    require_model_provenance: bool
    require_regime_provenance: bool

    def __post_init__(self) -> None:
        _validate_closed_text_set(
            self.allowed_environments,
            "INVALID_ALLOWED_ENVIRONMENTS",
            _validate_environment,
        )
        _validate_closed_text_set(
            self.allowed_strategies,
            "INVALID_ALLOWED_STRATEGIES",
            _validate_strategy,
        )
        _validate_closed_text_set(
            self.allowed_symbols,
            "INVALID_ALLOWED_SYMBOLS",
            _validate_symbol,
        )
        for name, value, minimum, maximum in (
            (
                "max_order_contracts",
                self.max_order_contracts,
                1,
                _MAX_QUANTITY,
            ),
            (
                "max_order_notional_cents",
                self.max_order_notional_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_order_loss_cents",
                self.max_order_loss_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_order_collateral_cents",
                self.max_order_collateral_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_account_open_risk_cents",
                self.max_account_open_risk_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_symbol_open_risk_cents",
                self.max_symbol_open_risk_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "min_remaining_buying_power_cents",
                self.min_remaining_buying_power_cents,
                0,
                _MAX_CENTS,
            ),
            (
                "max_daily_loss_cents",
                self.max_daily_loss_cents,
                1,
                _MAX_CENTS,
            ),
            ("max_daily_orders", self.max_daily_orders, 1, _MAX_QUANTITY),
            (
                "max_daily_new_risk_cents",
                self.max_daily_new_risk_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_quote_age_seconds",
                self.max_quote_age_seconds,
                1,
                86_400,
            ),
            (
                "max_portfolio_age_seconds",
                self.max_portfolio_age_seconds,
                1,
                86_400,
            ),
            (
                "max_authority_age_seconds",
                self.max_authority_age_seconds,
                1,
                86_400,
            ),
            (
                "max_overlay_age_seconds",
                self.max_overlay_age_seconds,
                1,
                604_800,
            ),
            (
                "max_option_ask_cents",
                self.max_option_ask_cents,
                1,
                _MAX_CENTS,
            ),
            (
                "max_bid_ask_width_cents",
                self.max_bid_ask_width_cents,
                0,
                _MAX_CENTS,
            ),
            (
                "max_fee_cents_per_contract_per_leg",
                self.max_fee_cents_per_contract_per_leg,
                1,
                _MAX_CENTS,
            ),
            ("min_open_interest", self.min_open_interest, 0, _MAX_CENTS),
            ("min_volume", self.min_volume, 0, _MAX_CENTS),
        ):
            _validate_int(
                value,
                f"INVALID_{name.upper()}",
                minimum=minimum,
                maximum=maximum,
            )
        _validate_positive_decimal(
            self.max_abs_portfolio_delta,
            "INVALID_MAX_ABS_PORTFOLIO_DELTA",
        )
        _validate_positive_decimal(
            self.max_abs_symbol_delta,
            "INVALID_MAX_ABS_SYMBOL_DELTA",
        )
        _validate_bool(
            self.require_model_provenance,
            "INVALID_MODEL_PROVENANCE_REQUIREMENT",
        )
        _validate_bool(
            self.require_regime_provenance,
            "INVALID_REGIME_PROVENANCE_REQUIREMENT",
        )
        if self.max_order_loss_cents > self.max_account_open_risk_cents:
            _invalid("ORDER_LOSS_EXCEEDS_ACCOUNT_LIMIT")
        if self.max_order_loss_cents > self.max_symbol_open_risk_cents:
            _invalid("ORDER_LOSS_EXCEEDS_SYMBOL_LIMIT")
        if self.max_order_loss_cents > self.max_daily_new_risk_cents:
            _invalid("ORDER_LOSS_EXCEEDS_DAILY_RISK_LIMIT")

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.allowed_environments)),
            tuple(sorted(self.allowed_strategies)),
            tuple(sorted(self.allowed_symbols)),
            self.max_order_contracts,
            self.max_order_notional_cents,
            self.max_order_loss_cents,
            self.max_order_collateral_cents,
            self.max_account_open_risk_cents,
            self.max_symbol_open_risk_cents,
            self.min_remaining_buying_power_cents,
            _decimal_text(self.max_abs_portfolio_delta),
            _decimal_text(self.max_abs_symbol_delta),
            self.max_daily_loss_cents,
            self.max_daily_orders,
            self.max_daily_new_risk_cents,
            self.max_quote_age_seconds,
            self.max_portfolio_age_seconds,
            self.max_authority_age_seconds,
            self.max_overlay_age_seconds,
            self.max_option_ask_cents,
            self.max_bid_ask_width_cents,
            self.max_fee_cents_per_contract_per_leg,
            self.min_open_interest,
            self.min_volume,
            self.require_model_provenance,
            self.require_regime_provenance,
        )


@dataclass(frozen=True, slots=True, repr=False)
class AuthorityEvidence:
    """Independent runtime-arm, kill-switch, and market-session evidence."""

    environment: str
    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str
    session_date: date
    observed_at: datetime
    arm_issued_at: datetime
    arm_expires_at: datetime
    session_opens_at: datetime
    session_closes_at: datetime
    complete: bool
    mutations_enabled: bool
    operator_armed: bool
    kill_switch_clear: bool
    session_open: bool
    runtime_config_sha256: str
    arm_sha256: str
    session_sha256: str

    def __post_init__(self) -> None:
        _validate_environment(self.environment)
        _validate_account_fields(
            self.account_id, self.account_id_key, self.institution_type
        )
        _validate_date(self.session_date, "INVALID_SESSION_DATE")
        _validate_datetime(self.observed_at, "INVALID_AUTHORITY_TIMESTAMP")
        _validate_datetime(self.arm_issued_at, "INVALID_ARM_ISSUED_AT")
        _validate_datetime(self.arm_expires_at, "INVALID_ARM_EXPIRY")
        _validate_datetime(self.session_opens_at, "INVALID_SESSION_OPEN")
        _validate_datetime(self.session_closes_at, "INVALID_SESSION_CLOSE")
        for name, value in (
            ("complete", self.complete),
            ("mutations_enabled", self.mutations_enabled),
            ("operator_armed", self.operator_armed),
            ("kill_switch_clear", self.kill_switch_clear),
            ("session_open", self.session_open),
        ):
            _validate_bool(value, f"INVALID_{name.upper()}")
        _validate_sha256(
            self.runtime_config_sha256, "INVALID_RUNTIME_CONFIG_DIGEST"
        )
        _validate_sha256(self.arm_sha256, "INVALID_ARM_DIGEST")
        _validate_sha256(self.session_sha256, "INVALID_SESSION_DIGEST")

    def __repr__(self) -> str:
        return "AuthorityEvidence([REDACTED])"

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.environment,
            self.account_id,
            self.account_id_key,
            self.institution_type,
            self.session_date.isoformat(),
            _datetime_text(self.observed_at),
            _datetime_text(self.arm_issued_at),
            _datetime_text(self.arm_expires_at),
            _datetime_text(self.session_opens_at),
            _datetime_text(self.session_closes_at),
            self.complete,
            self.mutations_enabled,
            self.operator_armed,
            self.kill_switch_clear,
            self.session_open,
            self.runtime_config_sha256,
            self.arm_sha256,
            self.session_sha256,
        )


@dataclass(frozen=True, slots=True)
class ContractQuote:
    contract: OptionContractId
    bid_cents: int
    ask_cents: int
    delta: Decimal
    open_interest: int
    volume: int
    observed_at: datetime
    source_sha256: str

    def __post_init__(self) -> None:
        if type(self.contract) is not OptionContractId:
            _invalid("INVALID_QUOTE_CONTRACT")
        _validate_int(
            self.bid_cents,
            "INVALID_QUOTE_BID",
            minimum=0,
            maximum=_MAX_CENTS,
        )
        _validate_int(
            self.ask_cents,
            "INVALID_QUOTE_ASK",
            minimum=1,
            maximum=_MAX_CENTS,
        )
        if self.ask_cents < self.bid_cents:
            _invalid("CROSSED_OPTION_QUOTE")
        _validate_bounded_decimal(
            self.delta,
            "INVALID_OPTION_DELTA",
            minimum=Decimal("-1"),
            maximum=Decimal("1"),
        )
        _validate_int(
            self.open_interest,
            "INVALID_OPEN_INTEREST",
            minimum=0,
            maximum=_MAX_CENTS,
        )
        _validate_int(
            self.volume,
            "INVALID_VOLUME",
            minimum=0,
            maximum=_MAX_CENTS,
        )
        _validate_datetime(self.observed_at, "INVALID_QUOTE_TIMESTAMP")
        _validate_sha256(self.source_sha256, "INVALID_QUOTE_SOURCE_DIGEST")

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.contract.canonical_material,
            self.bid_cents,
            self.ask_cents,
            _decimal_text(self.delta),
            self.open_interest,
            self.volume,
            _datetime_text(self.observed_at),
            self.source_sha256,
        )


@dataclass(frozen=True, slots=True)
class QuoteSnapshotEvidence:
    complete: bool
    snapshot_sha256: str
    quotes: tuple[ContractQuote, ...]

    def __post_init__(self) -> None:
        _validate_bool(self.complete, "INVALID_QUOTE_COMPLETENESS")
        _validate_sha256(self.snapshot_sha256, "INVALID_QUOTE_SNAPSHOT_DIGEST")
        if type(self.quotes) is not tuple:
            _invalid("INVALID_QUOTE_SET")
        for quote in self.quotes:
            if type(quote) is not ContractQuote:
                _invalid("INVALID_QUOTE")

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.complete,
            self.snapshot_sha256,
            tuple(sorted(quote.canonical_material for quote in self.quotes)),
        )


@dataclass(frozen=True, slots=True, repr=False)
class PortfolioRiskEvidence:
    """Complete committed-risk snapshot for one account and target symbol.

    ``open_risk_cents`` and ``symbol_open_risk_cents`` include positions,
    active opening orders, durable reservations, and absorbed filled risk.
    ``daily_pnl_cents`` is the complete marked daily P&L (realized,
    unrealized, fees, and commissions).  Daily counts include durably claimed
    submissions rather than broker acknowledgements only.
    ``open_order_contracts`` includes every active order leg, regardless of
    opening or closing direction.
    """

    environment: str
    account_id: str = field(repr=False)
    account_id_key: str = field(repr=False)
    institution_type: str
    session_date: date
    symbol: str
    observed_at: datetime
    complete: bool
    stable: bool
    buying_power_cents: int
    open_risk_cents: int
    symbol_open_risk_cents: int
    portfolio_delta: Decimal
    symbol_delta: Decimal
    daily_pnl_cents: int
    daily_order_count: int
    daily_new_risk_cents: int
    position_contracts: tuple[OptionContractId, ...]
    open_order_contracts: tuple[OptionContractId, ...]
    snapshot_sha256: str
    broker_read_evidence_sha256: str

    def __post_init__(self) -> None:
        _validate_environment(self.environment)
        _validate_account_fields(
            self.account_id, self.account_id_key, self.institution_type
        )
        _validate_date(self.session_date, "INVALID_PORTFOLIO_SESSION")
        _validate_symbol(self.symbol)
        _validate_datetime(self.observed_at, "INVALID_PORTFOLIO_TIMESTAMP")
        _validate_bool(self.complete, "INVALID_PORTFOLIO_COMPLETENESS")
        _validate_bool(self.stable, "INVALID_PORTFOLIO_STABILITY")
        for name, value in (
            ("buying_power_cents", self.buying_power_cents),
            ("open_risk_cents", self.open_risk_cents),
            ("symbol_open_risk_cents", self.symbol_open_risk_cents),
            ("daily_order_count", self.daily_order_count),
            ("daily_new_risk_cents", self.daily_new_risk_cents),
        ):
            _validate_int(
                value,
                f"INVALID_{name.upper()}",
                minimum=0,
                maximum=_MAX_CENTS,
            )
        _validate_int(
            self.daily_pnl_cents,
            "INVALID_DAILY_PNL",
            minimum=-_MAX_CENTS,
            maximum=_MAX_CENTS,
        )
        _validate_finite_decimal(
            self.portfolio_delta, "INVALID_PORTFOLIO_DELTA"
        )
        _validate_finite_decimal(self.symbol_delta, "INVALID_SYMBOL_DELTA")
        _validate_contract_set(
            self.position_contracts, "INVALID_POSITION_CONTRACTS"
        )
        _validate_contract_set(
            self.open_order_contracts, "INVALID_OPEN_ORDER_CONTRACTS"
        )
        _validate_sha256(self.snapshot_sha256, "INVALID_PORTFOLIO_DIGEST")
        _validate_sha256(
            self.broker_read_evidence_sha256,
            "INVALID_BROKER_READ_EVIDENCE_DIGEST",
        )

    def __repr__(self) -> str:
        return "PortfolioRiskEvidence([REDACTED])"

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.environment,
            self.account_id,
            self.account_id_key,
            self.institution_type,
            self.session_date.isoformat(),
            self.symbol,
            _datetime_text(self.observed_at),
            self.complete,
            self.stable,
            self.buying_power_cents,
            self.open_risk_cents,
            self.symbol_open_risk_cents,
            _decimal_text(self.portfolio_delta),
            _decimal_text(self.symbol_delta),
            self.daily_pnl_cents,
            self.daily_order_count,
            self.daily_new_risk_cents,
            tuple(
                sorted(
                    contract.canonical_material
                    for contract in self.position_contracts
                )
            ),
            tuple(
                sorted(
                    contract.canonical_material
                    for contract in self.open_order_contracts
                )
            ),
            self.snapshot_sha256,
            self.broker_read_evidence_sha256,
        )


@dataclass(frozen=True, slots=True)
class RiskOverlayEvidence:
    """Provenance plus optional reductions; there is no allow override."""

    kind: str
    effective_session: date
    observed_at: datetime
    complete: bool
    approved_for_risk: bool
    block_new_risk: bool
    max_contracts: int | None
    max_new_risk_cents: int | None
    provenance_sha256: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _OVERLAY_KINDS:
            _invalid("INVALID_OVERLAY_KIND")
        _validate_date(
            self.effective_session, "INVALID_OVERLAY_EFFECTIVE_SESSION"
        )
        _validate_datetime(self.observed_at, "INVALID_OVERLAY_TIMESTAMP")
        _validate_bool(self.complete, "INVALID_OVERLAY_COMPLETENESS")
        _validate_bool(
            self.approved_for_risk, "INVALID_OVERLAY_APPROVAL_FLAG"
        )
        _validate_bool(self.block_new_risk, "INVALID_OVERLAY_BLOCK_FLAG")
        for name, value, maximum in (
            ("max_contracts", self.max_contracts, _MAX_QUANTITY),
            ("max_new_risk_cents", self.max_new_risk_cents, _MAX_CENTS),
        ):
            if value is not None:
                _validate_int(
                    value,
                    f"INVALID_OVERLAY_{name.upper()}",
                    minimum=0,
                    maximum=maximum,
                )
        _validate_sha256(
            self.provenance_sha256, "INVALID_OVERLAY_PROVENANCE_DIGEST"
        )

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.effective_session.isoformat(),
            _datetime_text(self.observed_at),
            self.complete,
            self.approved_for_risk,
            self.block_new_risk,
            self.max_contracts,
            self.max_new_risk_cents,
            self.provenance_sha256,
        )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Self-verifying result bound to policy, request, and evidence hashes."""

    schema_version: int
    decision_sha256: str
    allowed: bool
    reason_codes: tuple[str, ...]
    evaluated_at: datetime
    request_sha256: str
    policy_sha256: str
    evidence_sha256: str
    order_notional_cents: int
    max_loss_cents: int
    collateral_cents: int
    fee_cents: int
    projected_portfolio_delta: Decimal
    projected_symbol_delta: Decimal

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PRETRADE_RISK_SCHEMA_VERSION
        ):
            _invalid("INVALID_RISK_DECISION_SCHEMA")
        _validate_sha256(
            self.decision_sha256, "INVALID_RISK_DECISION_DIGEST"
        )
        _validate_bool(self.allowed, "INVALID_RISK_DECISION_ALLOW_FLAG")
        if (
            type(self.reason_codes) is not tuple
            or not self.reason_codes
            or self.reason_codes != tuple(sorted(set(self.reason_codes)))
        ):
            _invalid("INVALID_RISK_DECISION_REASONS")
        for reason in self.reason_codes:
            if (
                type(reason) is not str
                or not reason
                or len(reason) > _MAX_TEXT
                or not reason.replace("_", "").isalnum()
                or reason != reason.upper()
            ):
                _invalid("INVALID_RISK_DECISION_REASONS")
        if self.allowed != (self.reason_codes == ("RISK_ALLOWED",)):
            _invalid("INCONSISTENT_RISK_DECISION")
        _validate_datetime(
            self.evaluated_at, "INVALID_RISK_DECISION_TIMESTAMP"
        )
        _validate_sha256(self.request_sha256, "INVALID_REQUEST_DIGEST")
        _validate_sha256(self.policy_sha256, "INVALID_POLICY_DIGEST")
        _validate_sha256(self.evidence_sha256, "INVALID_EVIDENCE_DIGEST")
        for name, value in (
            ("order_notional_cents", self.order_notional_cents),
            ("max_loss_cents", self.max_loss_cents),
            ("collateral_cents", self.collateral_cents),
            ("fee_cents", self.fee_cents),
        ):
            _validate_int(
                value,
                f"INVALID_{name.upper()}",
                minimum=0,
                maximum=_MAX_CENTS,
            )
        _validate_finite_decimal(
            self.projected_portfolio_delta,
            "INVALID_PROJECTED_PORTFOLIO_DELTA",
        )
        _validate_finite_decimal(
            self.projected_symbol_delta,
            "INVALID_PROJECTED_SYMBOL_DELTA",
        )
        material = (
            _DECISION_DOMAIN,
            self.schema_version,
            self.request_sha256,
            self.policy_sha256,
            self.evidence_sha256,
            _datetime_text(self.evaluated_at),
            self.allowed,
            self.reason_codes,
            self.order_notional_cents,
            self.max_loss_cents,
            self.collateral_cents,
            self.fee_cents,
            _decimal_text(self.projected_portfolio_delta),
            _decimal_text(self.projected_symbol_delta),
        )
        if self.decision_sha256 != _digest(material):
            _invalid("RISK_DECISION_DIGEST_MISMATCH")


@dataclass(frozen=True, slots=True, repr=False)
class OpeningRiskAuthorization:
    """Exact replayable inputs for one content-addressed risk decision.

    The decision alone proves internal consistency, but it cannot prove which
    typed inputs produced its request, policy, and evidence hashes.  This
    immutable bundle keeps those inputs together so a durable execution
    boundary can independently re-run the pure policy before reserving or
    sending an order.
    """

    spread: OpeningSpreadRequest = field(repr=False)
    limits: RiskLimits = field(repr=False)
    authority: AuthorityEvidence = field(repr=False)
    quotes: QuoteSnapshotEvidence = field(repr=False)
    portfolio: PortfolioRiskEvidence = field(repr=False)
    overlays: tuple[RiskOverlayEvidence, ...] = field(repr=False)
    decision: RiskDecision

    def __post_init__(self) -> None:
        _require_exact(
            self.spread, OpeningSpreadRequest, "INVALID_OPENING_REQUEST"
        )
        _require_exact(self.limits, RiskLimits, "INVALID_RISK_LIMITS")
        _require_exact(
            self.authority,
            AuthorityEvidence,
            "INVALID_AUTHORITY_EVIDENCE",
        )
        _require_exact(
            self.quotes,
            QuoteSnapshotEvidence,
            "INVALID_QUOTE_EVIDENCE",
        )
        _require_exact(
            self.portfolio,
            PortfolioRiskEvidence,
            "INVALID_PORTFOLIO_EVIDENCE",
        )
        if type(self.overlays) is not tuple:
            _invalid("INVALID_RISK_OVERLAYS")
        for overlay in self.overlays:
            _require_exact(
                overlay, RiskOverlayEvidence, "INVALID_RISK_OVERLAY"
            )
        _require_exact(
            self.decision, RiskDecision, "INVALID_RISK_DECISION"
        )
        expected = evaluate_pretrade(
            self.spread,
            self.limits,
            self.authority,
            self.quotes,
            self.portfolio,
            self.overlays,
            evaluated_at=self.decision.evaluated_at,
        )
        if self.decision != expected:
            _invalid("RISK_AUTHORIZATION_DECISION_MISMATCH")

    def __repr__(self) -> str:
        return (
            "OpeningRiskAuthorization("
            f"decision_sha256={self.decision.decision_sha256!r},"
            "[EVIDENCE REDACTED])"
        )

    @property
    def request_sha256(self) -> str:
        return opening_request_sha256(self.spread)

    @property
    def policy_sha256(self) -> str:
        return risk_policy_sha256(self.limits)

    @property
    def authority_sha256(self) -> str:
        return authority_evidence_sha256(self.authority)

    @property
    def quote_evidence_sha256(self) -> str:
        return quote_evidence_sha256(self.quotes)

    @property
    def portfolio_evidence_sha256(self) -> str:
        return portfolio_evidence_sha256(self.portfolio)

    @property
    def overlays_sha256(self) -> str:
        return overlay_evidence_sha256(self.overlays)

    @property
    def evidence_sha256(self) -> str:
        return combined_evidence_sha256(
            self.authority,
            self.quotes,
            self.portfolio,
            self.overlays,
        )

    @property
    def valid_until(self) -> datetime:
        """Return the conservative instant after which this bundle is stale."""

        deadlines = [
            self.authority.observed_at
            + _seconds(self.limits.max_authority_age_seconds),
            self.authority.arm_expires_at,
            self.authority.session_closes_at,
            self.portfolio.observed_at
            + _seconds(self.limits.max_portfolio_age_seconds),
        ]
        deadlines.extend(
            quote.observed_at
            + _seconds(self.limits.max_quote_age_seconds)
            for quote in self.quotes.quotes
        )
        deadlines.extend(
            overlay.observed_at
            + _seconds(self.limits.max_overlay_age_seconds)
            for overlay in self.overlays
        )
        return min(deadlines)


@dataclass(frozen=True, slots=True)
class _DerivedRisk:
    order_notional_cents: int = 0
    max_loss_cents: int = 0
    collateral_cents: int = 0
    fee_cents: int = 0
    width_cents: int = 0
    multiplier: int = 0


def evaluate_pretrade(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    authority: AuthorityEvidence,
    quotes: QuoteSnapshotEvidence,
    portfolio: PortfolioRiskEvidence,
    overlays: tuple[RiskOverlayEvidence, ...],
    *,
    evaluated_at: datetime,
) -> RiskDecision:
    """Evaluate one opening vertical without consulting ambient state."""

    _require_exact(request, OpeningSpreadRequest, "INVALID_OPENING_REQUEST")
    _require_exact(limits, RiskLimits, "INVALID_RISK_LIMITS")
    _require_exact(authority, AuthorityEvidence, "INVALID_AUTHORITY_EVIDENCE")
    _require_exact(quotes, QuoteSnapshotEvidence, "INVALID_QUOTE_EVIDENCE")
    _require_exact(
        portfolio, PortfolioRiskEvidence, "INVALID_PORTFOLIO_EVIDENCE"
    )
    _validate_datetime(evaluated_at, "INVALID_EVALUATION_TIMESTAMP")
    if type(overlays) is not tuple:
        _invalid("INVALID_RISK_OVERLAYS")
    for overlay in overlays:
        _require_exact(overlay, RiskOverlayEvidence, "INVALID_RISK_OVERLAY")

    reasons: set[str] = set()
    derived = _derive_order_risk(request, limits, reasons)
    _evaluate_request_policy(request, limits, derived, reasons)
    _evaluate_authority(
        request, limits, authority, evaluated_at, reasons
    )
    quote_by_contract = _evaluate_quotes(
        request, limits, quotes, evaluated_at, reasons
    )
    new_delta = _derive_delta(
        request, quote_by_contract, derived.multiplier, reasons
    )
    projected_portfolio_delta, projected_symbol_delta = _evaluate_portfolio(
        request,
        limits,
        portfolio,
        evaluated_at,
        derived,
        new_delta,
        reasons,
    )
    _evaluate_overlays(
        request, limits, overlays, evaluated_at, derived, reasons
    )

    reason_codes = tuple(sorted(reasons)) if reasons else ("RISK_ALLOWED",)
    allowed = not reasons
    request_sha256 = opening_request_sha256(request)
    policy_sha256 = risk_policy_sha256(limits)
    evidence_sha256 = combined_evidence_sha256(
        authority,
        quotes,
        portfolio,
        overlays,
    )
    decision_material = (
        _DECISION_DOMAIN,
        PRETRADE_RISK_SCHEMA_VERSION,
        request_sha256,
        policy_sha256,
        evidence_sha256,
        _datetime_text(evaluated_at),
        allowed,
        reason_codes,
        derived.order_notional_cents,
        derived.max_loss_cents,
        derived.collateral_cents,
        derived.fee_cents,
        _decimal_text(projected_portfolio_delta),
        _decimal_text(projected_symbol_delta),
    )
    return RiskDecision(
        schema_version=PRETRADE_RISK_SCHEMA_VERSION,
        decision_sha256=_digest(decision_material),
        allowed=allowed,
        reason_codes=reason_codes,
        evaluated_at=evaluated_at,
        request_sha256=request_sha256,
        policy_sha256=policy_sha256,
        evidence_sha256=evidence_sha256,
        order_notional_cents=derived.order_notional_cents,
        max_loss_cents=derived.max_loss_cents,
        collateral_cents=derived.collateral_cents,
        fee_cents=derived.fee_cents,
        projected_portfolio_delta=projected_portfolio_delta,
        projected_symbol_delta=projected_symbol_delta,
    )


def validate_opening_risk_authorization(
    authorization: OpeningRiskAuthorization,
    *,
    at: datetime,
) -> None:
    """Require an exact allowed decision whose evidence remains usable at ``at``."""

    _require_exact(
        authorization,
        OpeningRiskAuthorization,
        "INVALID_RISK_AUTHORIZATION",
    )
    _validate_datetime(at, "INVALID_AUTHORIZATION_CHECK_TIMESTAMP")
    decision = authorization.decision
    if not decision.allowed:
        _invalid("RISK_DECISION_DENIED")
    if decision.evaluated_at > at:
        _invalid("RISK_AUTHORIZATION_FROM_FUTURE")
    exact = evaluate_pretrade(
        authorization.spread,
        authorization.limits,
        authorization.authority,
        authorization.quotes,
        authorization.portfolio,
        authorization.overlays,
        evaluated_at=decision.evaluated_at,
    )
    if decision != exact:
        _invalid("RISK_AUTHORIZATION_DECISION_MISMATCH")
    current = evaluate_pretrade(
        authorization.spread,
        authorization.limits,
        authorization.authority,
        authorization.quotes,
        authorization.portfolio,
        authorization.overlays,
        evaluated_at=at,
    )
    if not current.allowed or at >= authorization.valid_until:
        _invalid("RISK_AUTHORIZATION_NOT_CURRENT")


def opening_request_sha256(request: OpeningSpreadRequest) -> str:
    _require_exact(request, OpeningSpreadRequest, "INVALID_OPENING_REQUEST")
    return _digest(("request", request.canonical_material))


def risk_policy_sha256(limits: RiskLimits) -> str:
    _require_exact(limits, RiskLimits, "INVALID_RISK_LIMITS")
    return _digest(("policy", limits.canonical_material))


def authority_evidence_sha256(evidence: AuthorityEvidence) -> str:
    _require_exact(
        evidence, AuthorityEvidence, "INVALID_AUTHORITY_EVIDENCE"
    )
    return _digest(("authority", evidence.canonical_material))


def quote_evidence_sha256(evidence: QuoteSnapshotEvidence) -> str:
    _require_exact(
        evidence, QuoteSnapshotEvidence, "INVALID_QUOTE_EVIDENCE"
    )
    return _digest(("quotes", evidence.canonical_material))


def portfolio_evidence_sha256(evidence: PortfolioRiskEvidence) -> str:
    _require_exact(
        evidence, PortfolioRiskEvidence, "INVALID_PORTFOLIO_EVIDENCE"
    )
    return _digest(("portfolio", evidence.canonical_material))


def overlay_evidence_sha256(
    overlays: tuple[RiskOverlayEvidence, ...],
) -> str:
    if type(overlays) is not tuple:
        _invalid("INVALID_RISK_OVERLAYS")
    for overlay in overlays:
        _require_exact(
            overlay, RiskOverlayEvidence, "INVALID_RISK_OVERLAY"
        )
    return _digest(
        (
            "overlays",
            tuple(
                sorted(
                    overlay.canonical_material for overlay in overlays
                )
            ),
        )
    )


def combined_evidence_sha256(
    authority: AuthorityEvidence,
    quotes: QuoteSnapshotEvidence,
    portfolio: PortfolioRiskEvidence,
    overlays: tuple[RiskOverlayEvidence, ...],
) -> str:
    _require_exact(
        authority, AuthorityEvidence, "INVALID_AUTHORITY_EVIDENCE"
    )
    _require_exact(
        quotes, QuoteSnapshotEvidence, "INVALID_QUOTE_EVIDENCE"
    )
    _require_exact(
        portfolio, PortfolioRiskEvidence, "INVALID_PORTFOLIO_EVIDENCE"
    )
    if type(overlays) is not tuple:
        _invalid("INVALID_RISK_OVERLAYS")
    for overlay in overlays:
        _require_exact(
            overlay, RiskOverlayEvidence, "INVALID_RISK_OVERLAY"
        )
    return _digest(
        (
            "evidence",
            authority.canonical_material,
            quotes.canonical_material,
            portfolio.canonical_material,
            tuple(
                sorted(
                    overlay.canonical_material for overlay in overlays
                )
            ),
        )
    )


def _derive_order_risk(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    reasons: set[str],
) -> _DerivedRisk:
    raw_fee_cents = (
        limits.max_fee_cents_per_contract_per_leg
        * len(request.legs)
        * request.quantity
    )
    if raw_fee_cents > _MAX_CENTS:
        reasons.add("DERIVED_VALUE_OUT_OF_RANGE")
    fee_cents = min(raw_fee_cents, _MAX_CENTS)
    first, second = request.legs
    first_contract = first.contract
    second_contract = second.contract
    if first_contract == second_contract:
        reasons.add("DUPLICATE_OPENING_CONTRACT")
    if (
        first_contract.symbol != second_contract.symbol
        or first_contract.expiry != second_contract.expiry
        or first_contract.call_put != second_contract.call_put
        or first_contract.strike == second_contract.strike
    ):
        reasons.add("NOT_A_VERTICAL_SPREAD")
    actions = {first.action, second.action}
    if actions != _OPEN_ACTIONS:
        reasons.add("OPENING_ACTIONS_NOT_OPPOSITE")
    if first_contract.multiplier != second_contract.multiplier:
        reasons.add("CONTRACT_MULTIPLIER_MISMATCH")
    if (
        first_contract.expiry <= request.trade_session
        or second_contract.expiry <= request.trade_session
    ):
        reasons.add("CONTRACT_NOT_LIVE")
    for contract in (first_contract, second_contract):
        eligibility = contract.close_eligibility_code()
        if eligibility is not None:
            reasons.add(eligibility)

    try:
        first_strike_cents = _dollars_to_cents(first_contract.strike)
        second_strike_cents = _dollars_to_cents(second_contract.strike)
        multiplier = _decimal_to_positive_int(first_contract.multiplier)
    except PretradeRiskValidationError:
        reasons.add("UNSUPPORTED_CONTRACT_ECONOMICS")
        return _DerivedRisk(fee_cents=fee_cents)
    width_cents = abs(first_strike_cents - second_strike_cents)
    if width_cents <= 0:
        reasons.add("INVALID_VERTICAL_WIDTH")
        return _DerivedRisk(
            fee_cents=fee_cents,
            multiplier=multiplier,
        )

    buy_leg = next(
        (leg for leg in request.legs if leg.action == "BUY_OPEN"), None
    )
    sell_leg = next(
        (leg for leg in request.legs if leg.action == "SELL_OPEN"), None
    )
    if buy_leg is None or sell_leg is None:
        return _DerivedRisk(
            fee_cents=fee_cents,
            width_cents=width_cents,
            multiplier=multiplier,
        )
    short_risk = (
        sell_leg.contract.call_put == "PUT"
        and sell_leg.contract.strike > buy_leg.contract.strike
    ) or (
        sell_leg.contract.call_put == "CALL"
        and sell_leg.contract.strike < buy_leg.contract.strike
    )
    expected_price_type = "NET_CREDIT" if short_risk else "NET_DEBIT"
    if request.price_type != expected_price_type:
        reasons.add("NET_PRICE_TYPE_MISMATCH")

    gross_width = width_cents * multiplier * request.quantity
    premium = request.limit_price_cents * multiplier * request.quantity
    notional = (
        max(first_strike_cents, second_strike_cents)
        * multiplier
        * request.quantity
    )
    if request.price_type == "NET_CREDIT":
        economic_collateral = gross_width
        economic_max_loss = gross_width - premium
    else:
        economic_collateral = premium
        economic_max_loss = premium
    collateral = economic_collateral + raw_fee_cents
    max_loss = economic_max_loss + raw_fee_cents
    if (
        request.limit_price_cents >= width_cents
        or economic_max_loss <= 0
    ):
        reasons.add("INVALID_NET_PRICE_FOR_WIDTH")
    if max(notional, max_loss, collateral) > _MAX_CENTS:
        reasons.add("DERIVED_VALUE_OUT_OF_RANGE")
    return _DerivedRisk(
        order_notional_cents=min(max(notional, 0), _MAX_CENTS),
        max_loss_cents=min(max(max_loss, 0), _MAX_CENTS),
        collateral_cents=min(max(collateral, 0), _MAX_CENTS),
        fee_cents=fee_cents,
        width_cents=width_cents,
        multiplier=multiplier,
    )


def _evaluate_request_policy(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    derived: _DerivedRisk,
    reasons: set[str],
) -> None:
    first_symbol = request.legs[0].contract.symbol
    if request.environment not in limits.allowed_environments:
        reasons.add("ENVIRONMENT_NOT_ALLOWED")
    if request.strategy_id not in limits.allowed_strategies:
        reasons.add("STRATEGY_NOT_ALLOWED")
    if first_symbol not in limits.allowed_symbols:
        reasons.add("SYMBOL_NOT_ALLOWED")
    if request.quantity > limits.max_order_contracts:
        reasons.add("ORDER_QUANTITY_LIMIT_EXCEEDED")
    if derived.order_notional_cents > limits.max_order_notional_cents:
        reasons.add("ORDER_NOTIONAL_LIMIT_EXCEEDED")
    if derived.max_loss_cents > limits.max_order_loss_cents:
        reasons.add("ORDER_LOSS_LIMIT_EXCEEDED")
    if derived.collateral_cents > limits.max_order_collateral_cents:
        reasons.add("ORDER_COLLATERAL_LIMIT_EXCEEDED")


def _evaluate_authority(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    authority: AuthorityEvidence,
    evaluated_at: datetime,
    reasons: set[str],
) -> None:
    if not authority.complete:
        reasons.add("AUTHORITY_EVIDENCE_INCOMPLETE")
    if not _identity_matches(request, authority):
        reasons.add("AUTHORITY_IDENTITY_MISMATCH")
    if authority.session_date != request.trade_session:
        reasons.add("AUTHORITY_SESSION_MISMATCH")
    if _is_stale_or_future(
        authority.observed_at,
        evaluated_at,
        limits.max_authority_age_seconds,
    ):
        reasons.add("AUTHORITY_EVIDENCE_STALE")
    if not authority.mutations_enabled:
        reasons.add("BROKER_MUTATIONS_DISABLED")
    if not authority.operator_armed:
        reasons.add("OPERATOR_NOT_ARMED")
    if not authority.kill_switch_clear:
        reasons.add("KILL_SWITCH_ACTIVE")
    if not authority.session_open:
        reasons.add("MARKET_SESSION_CLOSED")
    session_window_valid = (
        authority.session_opens_at < authority.session_closes_at
        and authority.session_opens_at.date() == authority.session_date
        and authority.session_closes_at.date() == authority.session_date
    )
    if not session_window_valid:
        reasons.add("SESSION_WINDOW_INVALID")
    else:
        if not (
            authority.session_opens_at
            <= evaluated_at
            < authority.session_closes_at
        ):
            reasons.add("MARKET_SESSION_CLOSED")
        if authority.session_open and not (
            authority.session_opens_at
            <= authority.observed_at
            < authority.session_closes_at
        ):
            reasons.add("SESSION_EVIDENCE_INCOHERENT")
    arm_lifetime = authority.arm_expires_at - authority.arm_issued_at
    if (
        authority.arm_issued_at > authority.observed_at
        or arm_lifetime.total_seconds() <= 0
        or arm_lifetime.total_seconds() > _MAX_ARM_LIFETIME_SECONDS
    ):
        reasons.add("ARM_WINDOW_INVALID")
    if authority.arm_expires_at <= evaluated_at:
        reasons.add("ARM_EXPIRED")


def _evaluate_quotes(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    evidence: QuoteSnapshotEvidence,
    evaluated_at: datetime,
    reasons: set[str],
) -> dict[OptionContractId, ContractQuote]:
    if not evidence.complete:
        reasons.add("QUOTE_EVIDENCE_INCOMPLETE")
    requested_contracts = {leg.contract for leg in request.legs}
    quote_by_contract: dict[OptionContractId, ContractQuote] = {}
    duplicate = False
    for quote in evidence.quotes:
        if quote.contract in quote_by_contract:
            duplicate = True
        quote_by_contract[quote.contract] = quote
    if duplicate:
        reasons.add("AMBIGUOUS_QUOTE_EVIDENCE")
    if set(quote_by_contract) != requested_contracts:
        reasons.add("QUOTE_CONTRACT_SET_MISMATCH")
    for contract in requested_contracts:
        quote = quote_by_contract.get(contract)
        if quote is None:
            continue
        if _is_stale_or_future(
            quote.observed_at,
            evaluated_at,
            limits.max_quote_age_seconds,
        ):
            reasons.add("QUOTE_EVIDENCE_STALE")
        if quote.ask_cents > limits.max_option_ask_cents:
            reasons.add("OPTION_PRICE_LIMIT_EXCEEDED")
        if quote.bid_cents == 0:
            reasons.add("ZERO_BID_OPTION_QUOTE")
        if (
            quote.ask_cents - quote.bid_cents
            > limits.max_bid_ask_width_cents
        ):
            reasons.add("BID_ASK_WIDTH_LIMIT_EXCEEDED")
        if quote.open_interest < limits.min_open_interest:
            reasons.add("OPEN_INTEREST_TOO_LOW")
        if quote.volume < limits.min_volume:
            reasons.add("OPTION_VOLUME_TOO_LOW")

    buy_leg = next(
        (leg for leg in request.legs if leg.action == "BUY_OPEN"), None
    )
    sell_leg = next(
        (leg for leg in request.legs if leg.action == "SELL_OPEN"), None
    )
    if (
        buy_leg is not None
        and sell_leg is not None
        and buy_leg.contract in quote_by_contract
        and sell_leg.contract in quote_by_contract
    ):
        buy_quote = quote_by_contract[buy_leg.contract]
        sell_quote = quote_by_contract[sell_leg.contract]
        if request.price_type == "NET_CREDIT":
            natural = sell_quote.bid_cents - buy_quote.ask_cents
            if natural <= 0:
                reasons.add("NO_EXECUTABLE_NATURAL_CREDIT")
            elif request.limit_price_cents < natural:
                reasons.add("LIMIT_PRICE_OUTSIDE_NBBO")
        else:
            natural = buy_quote.ask_cents - sell_quote.bid_cents
            if natural <= 0:
                reasons.add("NO_EXECUTABLE_NATURAL_DEBIT")
            elif request.limit_price_cents > natural:
                reasons.add("LIMIT_PRICE_OUTSIDE_NBBO")
    return quote_by_contract


def _derive_delta(
    request: OpeningSpreadRequest,
    quote_by_contract: dict[OptionContractId, ContractQuote],
    multiplier: int,
    reasons: set[str],
) -> Decimal:
    if multiplier <= 0:
        return Decimal("0")
    result = Decimal("0")
    for leg in request.legs:
        quote = quote_by_contract.get(leg.contract)
        if quote is None:
            return Decimal("0")
        direction = Decimal("1") if leg.action == "BUY_OPEN" else Decimal("-1")
        result += (
            direction
            * quote.delta
            * Decimal(multiplier)
            * Decimal(request.quantity)
        )
    if not result.is_finite():
        reasons.add("DERIVED_DELTA_INVALID")
        return Decimal("0")
    return result


def _evaluate_portfolio(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    portfolio: PortfolioRiskEvidence,
    evaluated_at: datetime,
    derived: _DerivedRisk,
    new_delta: Decimal,
    reasons: set[str],
) -> tuple[Decimal, Decimal]:
    if not portfolio.complete:
        reasons.add("PORTFOLIO_EVIDENCE_INCOMPLETE")
    if not portfolio.stable:
        reasons.add("PORTFOLIO_EVIDENCE_UNSTABLE")
    if not _identity_matches(request, portfolio):
        reasons.add("PORTFOLIO_IDENTITY_MISMATCH")
    requested_symbol = request.legs[0].contract.symbol
    if (
        portfolio.session_date != request.trade_session
        or portfolio.symbol != requested_symbol
    ):
        reasons.add("PORTFOLIO_SCOPE_MISMATCH")
    if _is_stale_or_future(
        portfolio.observed_at,
        evaluated_at,
        limits.max_portfolio_age_seconds,
    ):
        reasons.add("PORTFOLIO_EVIDENCE_STALE")

    requested_contracts = {leg.contract for leg in request.legs}
    if requested_contracts.intersection(portfolio.position_contracts):
        reasons.add("POSITION_CONTRACT_CONFLICT")
    if requested_contracts.intersection(portfolio.open_order_contracts):
        reasons.add("OPEN_ORDER_CONTRACT_CONFLICT")
    if (
        portfolio.buying_power_cents - derived.collateral_cents
        < limits.min_remaining_buying_power_cents
    ):
        reasons.add("INSUFFICIENT_BUYING_POWER")
    if (
        portfolio.open_risk_cents + derived.max_loss_cents
        > limits.max_account_open_risk_cents
    ):
        reasons.add("ACCOUNT_OPEN_RISK_LIMIT_EXCEEDED")
    if (
        portfolio.symbol_open_risk_cents + derived.max_loss_cents
        > limits.max_symbol_open_risk_cents
    ):
        reasons.add("SYMBOL_CONCENTRATION_LIMIT_EXCEEDED")
    if portfolio.daily_pnl_cents <= -limits.max_daily_loss_cents:
        reasons.add("DAILY_LOSS_LIMIT_REACHED")
    if portfolio.daily_order_count + 1 > limits.max_daily_orders:
        reasons.add("DAILY_ORDER_LIMIT_EXCEEDED")
    if (
        portfolio.daily_new_risk_cents + derived.max_loss_cents
        > limits.max_daily_new_risk_cents
    ):
        reasons.add("DAILY_NEW_RISK_LIMIT_EXCEEDED")

    projected_portfolio_delta = portfolio.portfolio_delta + new_delta
    projected_symbol_delta = portfolio.symbol_delta + new_delta
    if abs(projected_portfolio_delta) > limits.max_abs_portfolio_delta:
        reasons.add("PORTFOLIO_DELTA_LIMIT_EXCEEDED")
    if abs(projected_symbol_delta) > limits.max_abs_symbol_delta:
        reasons.add("SYMBOL_DELTA_LIMIT_EXCEEDED")
    return projected_portfolio_delta, projected_symbol_delta


def _evaluate_overlays(
    request: OpeningSpreadRequest,
    limits: RiskLimits,
    overlays: tuple[RiskOverlayEvidence, ...],
    evaluated_at: datetime,
    derived: _DerivedRisk,
    reasons: set[str],
) -> None:
    grouped: dict[str, list[RiskOverlayEvidence]] = {
        "MODEL": [],
        "REGIME": [],
    }
    for overlay in overlays:
        grouped[overlay.kind].append(overlay)
    for kind, required in (
        ("MODEL", limits.require_model_provenance),
        ("REGIME", limits.require_regime_provenance),
    ):
        candidates = grouped[kind]
        if required and not candidates:
            reasons.add(f"{kind}_PROVENANCE_REQUIRED")
        if len(candidates) > 1:
            reasons.add(f"AMBIGUOUS_{kind}_PROVENANCE")
        for overlay in candidates:
            if not overlay.complete or not overlay.approved_for_risk:
                reasons.add(f"{kind}_PROVENANCE_INVALID")
            if overlay.effective_session != request.trade_session:
                reasons.add(f"{kind}_SESSION_MISMATCH")
            if _is_stale_or_future(
                overlay.observed_at,
                evaluated_at,
                limits.max_overlay_age_seconds,
            ):
                reasons.add(f"{kind}_PROVENANCE_STALE")
            if overlay.block_new_risk:
                reasons.add(f"{kind}_OVERLAY_BLOCK")
            if (
                overlay.max_contracts is not None
                and overlay.max_contracts > limits.max_order_contracts
            ) or (
                overlay.max_new_risk_cents is not None
                and overlay.max_new_risk_cents
                > limits.max_order_loss_cents
            ):
                reasons.add(f"{kind}_OVERLAY_RELAXATION_ATTEMPT")
            if (
                overlay.max_contracts is not None
                and request.quantity > overlay.max_contracts
            ):
                reasons.add(f"{kind}_CONTRACT_LIMIT_EXCEEDED")
            if (
                overlay.max_new_risk_cents is not None
                and derived.max_loss_cents
                > overlay.max_new_risk_cents
            ):
                reasons.add(f"{kind}_RISK_LIMIT_EXCEEDED")


def _identity_matches(request: OpeningSpreadRequest, evidence: Any) -> bool:
    return (
        evidence.environment == request.environment
        and evidence.account_id == request.account_id
        and evidence.account_id_key == request.account_id_key
        and evidence.institution_type == request.institution_type
    )


def _is_stale_or_future(
    observed_at: datetime,
    evaluated_at: datetime,
    maximum_age_seconds: int,
) -> bool:
    age = evaluated_at - observed_at
    return age.total_seconds() < 0 or age.total_seconds() > maximum_age_seconds


def _validate_contract_set(
    value: Any,
    code: str,
) -> None:
    if type(value) is not tuple:
        _invalid(code)
    seen: set[OptionContractId] = set()
    for contract in value:
        if type(contract) is not OptionContractId or contract in seen:
            _invalid(code)
        seen.add(contract)


def _validate_closed_text_set(
    value: Any,
    code: str,
    validator: Any,
) -> None:
    if type(value) is not tuple or not value:
        _invalid(code)
    seen: set[str] = set()
    for item in value:
        try:
            validator(item)
        except PretradeRiskValidationError as exc:
            raise PretradeRiskValidationError(code) from exc
        if item in seen:
            _invalid(code)
        seen.add(item)


def _validate_account_fields(
    account_id: Any,
    account_id_key: Any,
    institution_type: Any,
) -> None:
    _validate_identity(account_id, "INVALID_ACCOUNT_ID")
    if (
        type(account_id_key) is not str
        or not account_id_key
        or len(account_id_key) > _MAX_TEXT
        or not account_id_key.isascii()
        or any(
            ord(character) < 33 or ord(character) > 126
            for character in account_id_key
        )
    ):
        _invalid("INVALID_ACCOUNT_ID_KEY")
    if (
        type(institution_type) is not str
        or not institution_type
        or len(institution_type) > 64
        or not institution_type.isascii()
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -"
            for character in institution_type
        )
    ):
        _invalid("INVALID_INSTITUTION_TYPE")


def _validate_identity(value: Any, code: str) -> None:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        _invalid(code)


def _validate_strategy(value: Any) -> None:
    if type(value) is not str or _STRATEGY.fullmatch(value) is None:
        _invalid("INVALID_STRATEGY_ID")


def _validate_symbol(value: Any) -> None:
    if type(value) is not str or _SYMBOL.fullmatch(value) is None:
        _invalid("INVALID_SYMBOL")


def _validate_environment(value: Any) -> None:
    if type(value) is not str or value not in _ENVIRONMENTS:
        _invalid("INVALID_ENVIRONMENT")


def _validate_sha256(value: Any, code: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _invalid(code)


def _validate_date(value: Any, code: str) -> None:
    if type(value) is not date:
        _invalid(code)


def _validate_datetime(value: Any, code: str) -> None:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        _invalid(code)


def _validate_bool(value: Any, code: str) -> None:
    if type(value) is not bool:
        _invalid(code)


def _validate_int(
    value: Any,
    code: str,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if type(value) is not int or value < minimum or value > maximum:
        _invalid(code)


def _validate_finite_decimal(value: Any, code: str) -> None:
    if type(value) is not Decimal or not value.is_finite():
        _invalid(code)


def _validate_positive_decimal(value: Any, code: str) -> None:
    _validate_finite_decimal(value, code)
    if value <= 0:
        _invalid(code)


def _validate_bounded_decimal(
    value: Any,
    code: str,
    *,
    minimum: Decimal,
    maximum: Decimal,
) -> None:
    _validate_finite_decimal(value, code)
    if value < minimum or value > maximum:
        _invalid(code)


def _decimal_to_positive_int(value: Decimal) -> int:
    if (
        type(value) is not Decimal
        or not value.is_finite()
        or value <= 0
        or value != value.to_integral_value()
        or value > _MAX_CENTS
    ):
        _invalid("UNSUPPORTED_CONTRACT_MULTIPLIER")
    return int(value)


def _dollars_to_cents(value: Decimal) -> int:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        _invalid("INVALID_STRIKE")
    cents = value * Decimal("100")
    if cents != cents.to_integral_value() or cents > _MAX_CENTS:
        _invalid("UNSUPPORTED_STRIKE_INCREMENT")
    return int(cents)


def _require_exact(value: Any, expected: type[Any], code: str) -> None:
    if type(value) is not expected:
        _invalid(code)


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _datetime_text(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _seconds(value: int) -> timedelta:
    return timedelta(seconds=value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _invalid(code: str) -> None:
    raise PretradeRiskValidationError(code)


__all__ = [
    "AuthorityEvidence",
    "ContractQuote",
    "OpeningRiskAuthorization",
    "OpeningLeg",
    "OpeningSpreadRequest",
    "PRETRADE_RISK_SCHEMA_VERSION",
    "PortfolioRiskEvidence",
    "PretradeRiskValidationError",
    "QuoteSnapshotEvidence",
    "RiskDecision",
    "RiskLimits",
    "RiskOverlayEvidence",
    "authority_evidence_sha256",
    "combined_evidence_sha256",
    "evaluate_pretrade",
    "opening_request_sha256",
    "overlay_evidence_sha256",
    "portfolio_evidence_sha256",
    "quote_evidence_sha256",
    "risk_policy_sha256",
    "validate_opening_risk_authorization",
]
