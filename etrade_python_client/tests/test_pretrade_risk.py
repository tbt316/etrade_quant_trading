from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from live_trading.order_domain import OptionContractId
from live_trading.pretrade_risk import (
    AuthorityEvidence,
    ContractQuote,
    OpeningLeg,
    OpeningRiskAuthorization,
    OpeningSpreadRequest,
    PortfolioRiskEvidence,
    PretradeRiskValidationError,
    QuoteSnapshotEvidence,
    RiskLimits,
    RiskOverlayEvidence,
    authority_evidence_sha256,
    combined_evidence_sha256,
    evaluate_pretrade,
    opening_request_sha256,
    overlay_evidence_sha256,
    portfolio_evidence_sha256,
    quote_evidence_sha256,
    risk_policy_sha256,
    validate_opening_risk_authorization,
)


NOW = datetime(2026, 7, 27, 17, 0, tzinfo=timezone.utc)
SESSION = date(2026, 7, 27)
EXPIRY = date(2026, 8, 21)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _osi(strike: int, option_type: str = "P") -> str:
    return f"{'SPY':<6}260821{option_type}{strike * 1000:08d}"


def _contract(
    strike: int,
    *,
    call_put: str = "PUT",
    multiplier: str = "100",
    adjusted: bool = False,
    deliverables: str | None = None,
) -> OptionContractId:
    return OptionContractId(
        symbol="SPY",
        expiry=EXPIRY,
        call_put=call_put,
        strike=Decimal(strike),
        osi_key=_osi(strike, "P" if call_put == "PUT" else "C"),
        multiplier=Decimal(multiplier),
        adjusted=adjusted,
        deliverables=deliverables,
    )


def _request(
    *,
    quantity: int = 2,
    price_cents: int = 100,
    environment: str = "sandbox",
    strategy: str = "put-credit.v1",
    legs: tuple[OpeningLeg, ...] | None = None,
) -> OpeningSpreadRequest:
    selected_legs = legs or (
        OpeningLeg(_contract(650), "SELL_OPEN"),
        OpeningLeg(_contract(640), "BUY_OPEN"),
    )
    return OpeningSpreadRequest(
        strategy_decision_id="strategy-decision-1",
        strategy_id=strategy,
        environment=environment,
        account_id="account-display",
        account_id_key="account-key",
        institution_type="BROKERAGE",
        trade_session=SESSION,
        legs=selected_legs,
        quantity=quantity,
        price_type="NET_CREDIT",
        limit_price_cents=price_cents,
    )


def _limits(**changes: object) -> RiskLimits:
    values: dict[str, object] = {
        "allowed_environments": ("sandbox",),
        "allowed_strategies": ("put-credit.v1",),
        "allowed_symbols": ("SPY",),
        "max_order_contracts": 5,
        "max_order_notional_cents": 20_000_000,
        "max_order_loss_cents": 500_000,
        "max_order_collateral_cents": 500_000,
        "max_account_open_risk_cents": 2_000_000,
        "max_symbol_open_risk_cents": 1_000_000,
        "min_remaining_buying_power_cents": 100_000,
        "max_abs_portfolio_delta": Decimal("500"),
        "max_abs_symbol_delta": Decimal("250"),
        "max_daily_loss_cents": 100_000,
        "max_daily_orders": 10,
        "max_daily_new_risk_cents": 1_000_000,
        "max_quote_age_seconds": 30,
        "max_portfolio_age_seconds": 60,
        "max_authority_age_seconds": 60,
        "max_overlay_age_seconds": 300,
        "max_option_ask_cents": 1_000,
        "max_bid_ask_width_cents": 10,
        "max_fee_cents_per_contract_per_leg": 75,
        "min_open_interest": 100,
        "min_volume": 10,
        "require_model_provenance": False,
        "require_regime_provenance": False,
    }
    values.update(changes)
    return RiskLimits(**values)


def _authority(**changes: object) -> AuthorityEvidence:
    values: dict[str, object] = {
        "environment": "sandbox",
        "account_id": "account-display",
        "account_id_key": "account-key",
        "institution_type": "BROKERAGE",
        "session_date": SESSION,
        "observed_at": NOW - timedelta(seconds=10),
        "arm_issued_at": NOW - timedelta(minutes=1),
        "arm_expires_at": NOW + timedelta(minutes=10),
        "session_opens_at": NOW - timedelta(hours=3),
        "session_closes_at": NOW + timedelta(hours=3),
        "complete": True,
        "mutations_enabled": True,
        "operator_armed": True,
        "kill_switch_clear": True,
        "session_open": True,
        "runtime_config_sha256": HASH_A,
        "arm_sha256": HASH_B,
        "session_sha256": HASH_C,
    }
    values.update(changes)
    return AuthorityEvidence(**values)


def _quote(
    contract: OptionContractId,
    *,
    bid: int,
    ask: int,
    delta: str,
    observed_at: datetime | None = None,
    open_interest: int = 500,
    volume: int = 50,
    digest: str = HASH_D,
) -> ContractQuote:
    return ContractQuote(
        contract=contract,
        bid_cents=bid,
        ask_cents=ask,
        delta=Decimal(delta),
        open_interest=open_interest,
        volume=volume,
        observed_at=observed_at or NOW - timedelta(seconds=5),
        source_sha256=digest,
    )


def _quotes(
    request: OpeningSpreadRequest,
    *,
    complete: bool = True,
    observed_at: datetime | None = None,
    reverse: bool = False,
) -> QuoteSnapshotEvidence:
    by_action = {leg.action: leg.contract for leg in request.legs}
    values = (
        _quote(
            by_action["SELL_OPEN"],
            bid=220,
            ask=230,
            delta="-0.20",
            observed_at=observed_at,
        ),
        _quote(
            by_action["BUY_OPEN"],
            bid=110,
            ask=120,
            delta="-0.10",
            observed_at=observed_at,
            digest=HASH_C,
        ),
    )
    return QuoteSnapshotEvidence(
        complete=complete,
        snapshot_sha256=HASH_B,
        quotes=tuple(reversed(values)) if reverse else values,
    )


def _portfolio(**changes: object) -> PortfolioRiskEvidence:
    values: dict[str, object] = {
        "environment": "sandbox",
        "account_id": "account-display",
        "account_id_key": "account-key",
        "institution_type": "BROKERAGE",
        "session_date": SESSION,
        "symbol": "SPY",
        "observed_at": NOW - timedelta(seconds=10),
        "complete": True,
        "stable": True,
        "buying_power_cents": 2_000_000,
        "open_risk_cents": 100_000,
        "symbol_open_risk_cents": 50_000,
        "portfolio_delta": Decimal("0"),
        "symbol_delta": Decimal("0"),
        "daily_pnl_cents": 0,
        "daily_order_count": 0,
        "daily_new_risk_cents": 10_000,
        "position_contracts": (),
        "open_order_contracts": (),
        "snapshot_sha256": HASH_A,
        "broker_read_evidence_sha256": HASH_D,
    }
    values.update(changes)
    return PortfolioRiskEvidence(**values)


def _overlay(
    kind: str,
    **changes: object,
) -> RiskOverlayEvidence:
    values: dict[str, object] = {
        "kind": kind,
        "effective_session": SESSION,
        "observed_at": NOW - timedelta(seconds=15),
        "complete": True,
        "approved_for_risk": True,
        "block_new_risk": False,
        "max_contracts": None,
        "max_new_risk_cents": None,
        "provenance_sha256": HASH_A if kind == "MODEL" else HASH_B,
    }
    values.update(changes)
    return RiskOverlayEvidence(**values)


def _evaluate(
    *,
    request: OpeningSpreadRequest | None = None,
    limits: RiskLimits | None = None,
    authority: AuthorityEvidence | None = None,
    quotes: QuoteSnapshotEvidence | None = None,
    portfolio: PortfolioRiskEvidence | None = None,
    overlays: tuple[RiskOverlayEvidence, ...] = (),
    evaluated_at: datetime = NOW,
):
    selected_request = request or _request()
    return evaluate_pretrade(
        selected_request,
        limits or _limits(),
        authority or _authority(),
        quotes or _quotes(selected_request),
        portfolio or _portfolio(),
        overlays,
        evaluated_at=evaluated_at,
    )


def _authorization(
    *,
    request: OpeningSpreadRequest | None = None,
    limits: RiskLimits | None = None,
    authority: AuthorityEvidence | None = None,
    quotes: QuoteSnapshotEvidence | None = None,
    portfolio: PortfolioRiskEvidence | None = None,
    overlays: tuple[RiskOverlayEvidence, ...] = (),
    evaluated_at: datetime = NOW,
) -> OpeningRiskAuthorization:
    selected_request = request or _request()
    selected_limits = limits or _limits()
    selected_authority = authority or _authority()
    selected_quotes = quotes or _quotes(selected_request)
    selected_portfolio = portfolio or _portfolio()
    decision = evaluate_pretrade(
        selected_request,
        selected_limits,
        selected_authority,
        selected_quotes,
        selected_portfolio,
        overlays,
        evaluated_at=evaluated_at,
    )
    return OpeningRiskAuthorization(
        selected_request,
        selected_limits,
        selected_authority,
        selected_quotes,
        selected_portfolio,
        overlays,
        decision,
    )


def test_allows_exact_vertical_and_derives_economics_and_delta() -> None:
    decision = _evaluate()

    assert decision.allowed is True
    assert decision.reason_codes == ("RISK_ALLOWED",)
    assert decision.order_notional_cents == 13_000_000
    assert decision.fee_cents == 300
    assert decision.collateral_cents == 200_300
    assert decision.max_loss_cents == 180_300
    assert decision.projected_portfolio_delta == Decimal("20")
    assert decision.projected_symbol_delta == Decimal("20")
    assert len(decision.decision_sha256) == 64


def test_account_bound_evidence_repr_is_redacted() -> None:
    for value in (_request(), _authority(), _portfolio()):
        rendered = repr(value)
        assert rendered.endswith("([REDACTED])")
        assert "account-display" not in rendered
        assert "account-key" not in rendered
        field_metadata = {
            item.name: item.repr for item in fields(value)
        }
        assert field_metadata["account_id"] is False
        assert field_metadata["account_id_key"] is False


def test_decision_is_frozen_and_content_addressed_deterministically() -> None:
    request = _request()
    model = _overlay("MODEL", max_contracts=4)
    regime = _overlay("REGIME", max_new_risk_cents=200_000)

    first = _evaluate(
        request=request,
        quotes=_quotes(request),
        overlays=(model, regime),
    )
    second = _evaluate(
        request=request,
        quotes=_quotes(request, reverse=True),
        overlays=(regime, model),
    )

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.allowed = False  # type: ignore[misc]
    with pytest.raises(
        PretradeRiskValidationError, match="INCONSISTENT_RISK_DECISION"
    ):
        replace(first, allowed=False)
    with pytest.raises(
        PretradeRiskValidationError, match="RISK_DECISION_DIGEST_MISMATCH"
    ):
        replace(first, decision_sha256=HASH_D)


def test_authorization_replays_exact_decision_and_exposes_component_hashes() -> None:
    model = _overlay("MODEL", max_contracts=4)
    regime = _overlay("REGIME", max_new_risk_cents=200_000)
    authorization = _authorization(overlays=(regime, model))

    assert authorization.decision.allowed is True
    assert authorization.request_sha256 == opening_request_sha256(
        authorization.spread
    )
    assert authorization.policy_sha256 == risk_policy_sha256(
        authorization.limits
    )
    assert authorization.authority_sha256 == authority_evidence_sha256(
        authorization.authority
    )
    assert (
        authorization.quote_evidence_sha256
        == quote_evidence_sha256(authorization.quotes)
    )
    assert (
        authorization.portfolio_evidence_sha256
        == portfolio_evidence_sha256(authorization.portfolio)
    )
    assert authorization.overlays_sha256 == overlay_evidence_sha256(
        authorization.overlays
    )
    assert authorization.evidence_sha256 == combined_evidence_sha256(
        authorization.authority,
        authorization.quotes,
        authorization.portfolio,
        authorization.overlays,
    )
    assert authorization.decision.request_sha256 == (
        authorization.request_sha256
    )
    assert authorization.decision.policy_sha256 == (
        authorization.policy_sha256
    )
    assert authorization.decision.evidence_sha256 == (
        authorization.evidence_sha256
    )
    validate_opening_risk_authorization(authorization, at=NOW)


def test_authorization_rejects_denied_stale_and_post_construction_tamper() -> None:
    denied = _authorization(
        limits=_limits(allowed_symbols=("QQQ",))
    )
    assert denied.decision.allowed is False
    with pytest.raises(
        PretradeRiskValidationError, match="RISK_DECISION_DENIED"
    ):
        validate_opening_risk_authorization(denied, at=NOW)

    allowed = _authorization()
    with pytest.raises(
        PretradeRiskValidationError,
        match="RISK_AUTHORIZATION_NOT_CURRENT",
    ):
        validate_opening_risk_authorization(
            allowed, at=NOW + timedelta(seconds=31)
        )

    future = _authorization(
        evaluated_at=NOW + timedelta(seconds=20)
    )
    with pytest.raises(
        PretradeRiskValidationError,
        match="RISK_AUTHORIZATION_FROM_FUTURE",
    ):
        validate_opening_risk_authorization(future, at=NOW)

    other = _authorization(request=_request(price_cents=101))
    object.__setattr__(allowed, "decision", other.decision)
    with pytest.raises(
        PretradeRiskValidationError,
        match="RISK_AUTHORIZATION_DECISION_MISMATCH",
    ):
        validate_opening_risk_authorization(allowed, at=NOW)


def test_any_material_evidence_change_changes_decision_hash() -> None:
    baseline = _evaluate()
    changed = _evaluate(
        portfolio=_portfolio(snapshot_sha256=HASH_B)
    )

    assert baseline.allowed is changed.allowed is True
    assert baseline.evidence_sha256 != changed.evidence_sha256
    assert baseline.decision_sha256 != changed.decision_sha256


def test_fee_ceiling_is_bound_and_monotonically_increases_risk() -> None:
    baseline = _evaluate(
        limits=_limits(
            max_order_loss_cents=180_300,
        )
    )
    higher_fee = _evaluate(
        limits=_limits(
            max_order_loss_cents=180_300,
            max_fee_cents_per_contract_per_leg=76,
        )
    )

    assert baseline.allowed
    assert baseline.fee_cents == 300
    assert higher_fee.fee_cents == 304
    assert higher_fee.max_loss_cents == 180_304
    assert "ORDER_LOSS_LIMIT_EXCEEDED" in higher_fee.reason_codes
    assert baseline.policy_sha256 != higher_fee.policy_sha256
    assert baseline.decision_sha256 != higher_fee.decision_sha256


def test_debit_vertical_uses_limit_debit_as_maximum_loss() -> None:
    buy_contract = _contract(650)
    sell_contract = _contract(640)
    request = replace(
        _request(
            legs=(
                OpeningLeg(buy_contract, "BUY_OPEN"),
                OpeningLeg(sell_contract, "SELL_OPEN"),
            )
        ),
        price_type="NET_DEBIT",
    )
    quotes = QuoteSnapshotEvidence(
        complete=True,
        snapshot_sha256=HASH_B,
        quotes=(
            _quote(
                buy_contract,
                bid=220,
                ask=230,
                delta="-0.20",
            ),
            _quote(
                sell_contract,
                bid=110,
                ask=120,
                delta="-0.10",
                digest=HASH_C,
            ),
        ),
    )

    decision = _evaluate(request=request, quotes=quotes)

    assert decision.allowed
    assert decision.fee_cents == 300
    assert decision.max_loss_cents == 20_300
    assert decision.collateral_cents == 20_300
    assert decision.projected_portfolio_delta == Decimal("-20")


@pytest.mark.parametrize(
    ("request_change", "limits_change", "code"),
    [
        (
            {"environment": "production"},
            {"allowed_environments": ("sandbox",)},
            "ENVIRONMENT_NOT_ALLOWED",
        ),
        (
            {"strategy": "other-strategy.v1"},
            {"allowed_strategies": ("put-credit.v1",)},
            "STRATEGY_NOT_ALLOWED",
        ),
        (
            {},
            {"allowed_symbols": ("QQQ",)},
            "SYMBOL_NOT_ALLOWED",
        ),
    ],
)
def test_environment_strategy_and_symbol_allowlists(
    request_change: dict[str, object],
    limits_change: dict[str, object],
    code: str,
) -> None:
    request = _request(**request_change)
    authority = _authority(environment=request.environment)
    portfolio = _portfolio(environment=request.environment)

    decision = _evaluate(
        request=request,
        limits=_limits(**limits_change),
        authority=authority,
        portfolio=portfolio,
    )

    assert not decision.allowed
    assert code in decision.reason_codes


def test_non_vertical_and_wrong_price_orientation_fail_closed() -> None:
    non_vertical = _request(
        legs=(
            OpeningLeg(_contract(650, call_put="CALL"), "SELL_OPEN"),
            OpeningLeg(_contract(640), "BUY_OPEN"),
        )
    )
    wrong_orientation = replace(
        _request(
            legs=(
                OpeningLeg(_contract(650), "BUY_OPEN"),
                OpeningLeg(_contract(640), "SELL_OPEN"),
            )
        ),
        price_type="NET_CREDIT",
    )

    assert "NOT_A_VERTICAL_SPREAD" in _evaluate(
        request=non_vertical,
        quotes=QuoteSnapshotEvidence(
            complete=True,
            snapshot_sha256=HASH_A,
            quotes=(),
        ),
    ).reason_codes
    assert "NET_PRICE_TYPE_MISMATCH" in _evaluate(
        request=wrong_orientation
    ).reason_codes


@pytest.mark.parametrize(
    ("field", "boundary", "denied_value", "code"),
    [
        ("max_order_contracts", 2, 1, "ORDER_QUANTITY_LIMIT_EXCEEDED"),
        (
            "max_order_notional_cents",
            13_000_000,
            12_999_999,
            "ORDER_NOTIONAL_LIMIT_EXCEEDED",
        ),
        (
            "max_order_loss_cents",
            180_300,
            180_299,
            "ORDER_LOSS_LIMIT_EXCEEDED",
        ),
        (
            "max_order_collateral_cents",
            200_300,
            200_299,
            "ORDER_COLLATERAL_LIMIT_EXCEEDED",
        ),
    ],
)
def test_order_limits_allow_equality_and_deny_one_unit_tighter(
    field: str,
    boundary: int,
    denied_value: int,
    code: str,
) -> None:
    boundary_changes = {
        field: boundary,
        "max_account_open_risk_cents": 2_000_000,
        "max_symbol_open_risk_cents": 1_000_000,
        "max_daily_new_risk_cents": 1_000_000,
    }
    assert _evaluate(limits=_limits(**boundary_changes)).allowed

    denied_changes = dict(boundary_changes)
    denied_changes[field] = denied_value
    decision = _evaluate(limits=_limits(**denied_changes))
    assert not decision.allowed
    assert code in decision.reason_codes


@pytest.mark.parametrize(
    ("limits_change", "allowed_change", "denied_change", "code"),
    [
        (
            {"min_remaining_buying_power_cents": 100_000},
            {"buying_power_cents": 300_300},
            {"buying_power_cents": 300_299},
            "INSUFFICIENT_BUYING_POWER",
        ),
        (
            {
                "max_order_loss_cents": 180_300,
                "max_account_open_risk_cents": 280_300,
            },
            {"open_risk_cents": 100_000},
            {"open_risk_cents": 100_001},
            "ACCOUNT_OPEN_RISK_LIMIT_EXCEEDED",
        ),
        (
            {
                "max_order_loss_cents": 180_300,
                "max_symbol_open_risk_cents": 230_300,
            },
            {"symbol_open_risk_cents": 50_000},
            {"symbol_open_risk_cents": 50_001},
            "SYMBOL_CONCENTRATION_LIMIT_EXCEEDED",
        ),
        (
            {"max_daily_orders": 1},
            {"daily_order_count": 0},
            {"daily_order_count": 1},
            "DAILY_ORDER_LIMIT_EXCEEDED",
        ),
        (
            {
                "max_order_loss_cents": 180_300,
                "max_daily_new_risk_cents": 190_300,
            },
            {"daily_new_risk_cents": 10_000},
            {"daily_new_risk_cents": 10_001},
            "DAILY_NEW_RISK_LIMIT_EXCEEDED",
        ),
    ],
)
def test_account_budget_boundaries_are_monotonic(
    limits_change: dict[str, object],
    allowed_change: dict[str, object],
    denied_change: dict[str, object],
    code: str,
) -> None:
    assert _evaluate(
        limits=_limits(**limits_change),
        portfolio=_portfolio(**allowed_change),
    ).allowed

    denied = _evaluate(
        limits=_limits(**limits_change),
        portfolio=_portfolio(**denied_change),
    )
    assert not denied.allowed
    assert code in denied.reason_codes


def test_daily_loss_boundary_blocks_when_limit_is_reached() -> None:
    before = _evaluate(
        portfolio=_portfolio(daily_pnl_cents=-99_999)
    )
    reached = _evaluate(
        portfolio=_portfolio(daily_pnl_cents=-100_000)
    )

    assert before.allowed
    assert "DAILY_LOSS_LIMIT_REACHED" in reached.reason_codes


def test_delta_boundaries_are_exact_and_monotonic() -> None:
    allowed = _evaluate(
        limits=_limits(
            max_abs_portfolio_delta=Decimal("20"),
            max_abs_symbol_delta=Decimal("20"),
        )
    )
    denied = _evaluate(
        limits=_limits(
            max_abs_portfolio_delta=Decimal("19.999"),
            max_abs_symbol_delta=Decimal("19.999"),
        )
    )

    assert allowed.allowed
    assert denied.reason_codes == (
        "PORTFOLIO_DELTA_LIMIT_EXCEEDED",
        "SYMBOL_DELTA_LIMIT_EXCEEDED",
    )


def test_quantity_growth_cannot_return_to_allowed_after_limit_breach() -> None:
    outcomes = [
        _evaluate(
            request=_request(quantity=quantity),
            limits=_limits(
                max_order_contracts=3,
                max_order_loss_cents=500_000,
                max_account_open_risk_cents=2_000_000,
                max_symbol_open_risk_cents=1_000_000,
                max_daily_new_risk_cents=1_000_000,
            ),
        ).allowed
        for quantity in range(1, 6)
    ]

    assert outcomes == [True, True, True, False, False]


@pytest.mark.parametrize(
    ("authority_change", "code"),
    [
        ({"complete": False}, "AUTHORITY_EVIDENCE_INCOMPLETE"),
        ({"environment": "production"}, "AUTHORITY_IDENTITY_MISMATCH"),
        ({"account_id_key": "other-key"}, "AUTHORITY_IDENTITY_MISMATCH"),
        (
            {"session_date": date(2026, 7, 28)},
            "AUTHORITY_SESSION_MISMATCH",
        ),
        (
            {"observed_at": NOW - timedelta(seconds=61)},
            "AUTHORITY_EVIDENCE_STALE",
        ),
        (
            {"observed_at": NOW + timedelta(microseconds=1)},
            "AUTHORITY_EVIDENCE_STALE",
        ),
        ({"mutations_enabled": False}, "BROKER_MUTATIONS_DISABLED"),
        ({"operator_armed": False}, "OPERATOR_NOT_ARMED"),
        ({"kill_switch_clear": False}, "KILL_SWITCH_ACTIVE"),
        ({"session_open": False}, "MARKET_SESSION_CLOSED"),
        (
            {
                "arm_issued_at": NOW - timedelta(minutes=16),
                "arm_expires_at": NOW + timedelta(seconds=1),
            },
            "ARM_WINDOW_INVALID",
        ),
        (
            {"arm_issued_at": NOW},
            "ARM_WINDOW_INVALID",
        ),
        ({"arm_expires_at": NOW}, "ARM_EXPIRED"),
        ({"session_closes_at": NOW}, "MARKET_SESSION_CLOSED"),
    ],
)
def test_authority_dependency_matrix_fails_closed(
    authority_change: dict[str, object],
    code: str,
) -> None:
    decision = _evaluate(authority=_authority(**authority_change))

    assert not decision.allowed
    assert code in decision.reason_codes


def test_true_session_boolean_cannot_authorize_before_exact_open() -> None:
    before_open = NOW - timedelta(hours=4)
    request = _request()
    authority = _authority(
        observed_at=before_open - timedelta(seconds=1),
        arm_issued_at=before_open - timedelta(minutes=1),
        arm_expires_at=before_open + timedelta(minutes=10),
        session_open=True,
    )

    decision = _evaluate(
        request=request,
        authority=authority,
        quotes=_quotes(
            request,
            observed_at=before_open - timedelta(seconds=1),
        ),
        portfolio=_portfolio(
            observed_at=before_open - timedelta(seconds=1),
        ),
        evaluated_at=before_open,
    )

    assert not decision.allowed
    assert "MARKET_SESSION_CLOSED" in decision.reason_codes
    assert "SESSION_EVIDENCE_INCOHERENT" in decision.reason_codes


@pytest.mark.parametrize(
    "authority_change",
    [
        {
            "session_opens_at": NOW + timedelta(hours=3),
            "session_closes_at": NOW + timedelta(hours=2),
        },
        {
            "session_closes_at": datetime(
                2026, 7, 28, 1, 0, tzinfo=timezone.utc
            ),
        },
    ],
)
def test_session_window_ordering_and_utc_date_must_be_coherent(
    authority_change: dict[str, object],
) -> None:
    decision = _evaluate(
        authority=_authority(**authority_change)
    )

    assert not decision.allowed
    assert "SESSION_WINDOW_INVALID" in decision.reason_codes


@pytest.mark.parametrize(
    ("quote_change", "code"),
    [
        ({"complete": False}, "QUOTE_EVIDENCE_INCOMPLETE"),
        (
            {"observed_at": NOW - timedelta(seconds=31)},
            "QUOTE_EVIDENCE_STALE",
        ),
        (
            {"observed_at": NOW + timedelta(microseconds=1)},
            "QUOTE_EVIDENCE_STALE",
        ),
    ],
)
def test_quote_dependency_matrix_fails_closed(
    quote_change: dict[str, object],
    code: str,
) -> None:
    request = _request()
    decision = _evaluate(
        request=request,
        quotes=_quotes(request, **quote_change),
    )

    assert not decision.allowed
    assert code in decision.reason_codes


def test_evidence_age_boundaries_are_inclusive() -> None:
    request = _request()
    decision = _evaluate(
        request=request,
        authority=_authority(observed_at=NOW - timedelta(seconds=60)),
        quotes=_quotes(
            request, observed_at=NOW - timedelta(seconds=30)
        ),
        portfolio=_portfolio(
            observed_at=NOW - timedelta(seconds=60)
        ),
        overlays=(
            _overlay(
                "MODEL",
                observed_at=NOW - timedelta(seconds=300),
            ),
        ),
    )

    assert decision.allowed


def test_quote_completeness_requires_exact_nonambiguous_contract_set() -> None:
    request = _request()
    full = _quotes(request)
    missing = replace(full, quotes=full.quotes[:1])
    duplicate = replace(
        full, quotes=(full.quotes[0], full.quotes[0])
    )

    assert "QUOTE_CONTRACT_SET_MISMATCH" in _evaluate(
        request=request, quotes=missing
    ).reason_codes
    duplicate_decision = _evaluate(request=request, quotes=duplicate)
    assert "AMBIGUOUS_QUOTE_EVIDENCE" in duplicate_decision.reason_codes
    assert "QUOTE_CONTRACT_SET_MISMATCH" in duplicate_decision.reason_codes


@pytest.mark.parametrize(
    ("quote_mutator", "limits_change", "code"),
    [
        (
            lambda quote: replace(quote, ask_cents=231),
            {"max_bid_ask_width_cents": 10},
            "BID_ASK_WIDTH_LIMIT_EXCEEDED",
        ),
        (
            lambda quote: replace(quote, ask_cents=1_001),
            {"max_option_ask_cents": 1_000},
            "OPTION_PRICE_LIMIT_EXCEEDED",
        ),
        (
            lambda quote: replace(quote, open_interest=99),
            {"min_open_interest": 100},
            "OPEN_INTEREST_TOO_LOW",
        ),
        (
            lambda quote: replace(quote, volume=9),
            {"min_volume": 10},
            "OPTION_VOLUME_TOO_LOW",
        ),
    ],
)
def test_quote_price_width_and_liquidity_limits(
    quote_mutator,
    limits_change: dict[str, object],
    code: str,
) -> None:
    request = _request()
    evidence = _quotes(request)
    changed = replace(
        evidence,
        quotes=(quote_mutator(evidence.quotes[0]), evidence.quotes[1]),
    )

    decision = _evaluate(
        request=request,
        quotes=changed,
        limits=_limits(**limits_change),
    )
    assert code in decision.reason_codes


def test_zero_bid_leg_is_ineligible_despite_other_liquidity_fields() -> None:
    request = _request()
    evidence = _quotes(request)
    zero_bid_long = replace(evidence.quotes[1], bid_cents=0)
    changed = replace(
        evidence,
        quotes=(evidence.quotes[0], zero_bid_long),
    )

    decision = _evaluate(
        request=request,
        quotes=changed,
        limits=_limits(max_bid_ask_width_cents=120),
    )

    assert not decision.allowed
    assert "ZERO_BID_OPTION_QUOTE" in decision.reason_codes


def test_credit_limit_cannot_give_away_current_natural_price() -> None:
    decision = _evaluate(request=_request(price_cents=99))

    assert not decision.allowed
    assert "LIMIT_PRICE_OUTSIDE_NBBO" in decision.reason_codes


@pytest.mark.parametrize(
    ("portfolio_change", "code"),
    [
        ({"complete": False}, "PORTFOLIO_EVIDENCE_INCOMPLETE"),
        ({"stable": False}, "PORTFOLIO_EVIDENCE_UNSTABLE"),
        ({"account_id": "other-account"}, "PORTFOLIO_IDENTITY_MISMATCH"),
        (
            {"session_date": date(2026, 7, 28)},
            "PORTFOLIO_SCOPE_MISMATCH",
        ),
        ({"symbol": "QQQ"}, "PORTFOLIO_SCOPE_MISMATCH"),
        (
            {"observed_at": NOW - timedelta(seconds=61)},
            "PORTFOLIO_EVIDENCE_STALE",
        ),
        (
            {"observed_at": NOW + timedelta(microseconds=1)},
            "PORTFOLIO_EVIDENCE_STALE",
        ),
    ],
)
def test_portfolio_dependency_matrix_fails_closed(
    portfolio_change: dict[str, object],
    code: str,
) -> None:
    decision = _evaluate(portfolio=_portfolio(**portfolio_change))

    assert not decision.allowed
    assert code in decision.reason_codes


def test_exact_position_and_open_order_contract_conflicts_block() -> None:
    request = _request()
    contract = request.legs[0].contract

    position = _evaluate(
        request=request,
        portfolio=_portfolio(position_contracts=(contract,)),
    )
    open_order = _evaluate(
        request=request,
        portfolio=_portfolio(open_order_contracts=(contract,)),
    )

    assert "POSITION_CONTRACT_CONFLICT" in position.reason_codes
    assert "OPEN_ORDER_CONTRACT_CONFLICT" in open_order.reason_codes


def test_adjusted_and_nonstandard_contracts_deny() -> None:
    adjusted_request = _request(
        legs=(
            OpeningLeg(_contract(650, adjusted=True), "SELL_OPEN"),
            OpeningLeg(_contract(640, adjusted=True), "BUY_OPEN"),
        )
    )
    multiplier_request = _request(
        legs=(
            OpeningLeg(_contract(650, multiplier="10"), "SELL_OPEN"),
            OpeningLeg(_contract(640, multiplier="10"), "BUY_OPEN"),
        )
    )

    assert "ADJUSTED_OPTION_CONTRACT" in _evaluate(
        request=adjusted_request
    ).reason_codes
    assert "NONSTANDARD_OPTION_MULTIPLIER" in _evaluate(
        request=multiplier_request
    ).reason_codes


def test_required_model_and_regime_provenance_are_independent() -> None:
    limits = _limits(
        require_model_provenance=True,
        require_regime_provenance=True,
    )
    missing = _evaluate(limits=limits)
    model_only = _evaluate(limits=limits, overlays=(_overlay("MODEL"),))
    both = _evaluate(
        limits=limits,
        overlays=(_overlay("MODEL"), _overlay("REGIME")),
    )

    assert missing.reason_codes == (
        "MODEL_PROVENANCE_REQUIRED",
        "REGIME_PROVENANCE_REQUIRED",
    )
    assert model_only.reason_codes == ("REGIME_PROVENANCE_REQUIRED",)
    assert both.allowed


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"complete": False}, "MODEL_PROVENANCE_INVALID"),
        ({"approved_for_risk": False}, "MODEL_PROVENANCE_INVALID"),
        (
            {"effective_session": date(2026, 7, 28)},
            "MODEL_SESSION_MISMATCH",
        ),
        (
            {"observed_at": NOW - timedelta(seconds=301)},
            "MODEL_PROVENANCE_STALE",
        ),
        ({"block_new_risk": True}, "MODEL_OVERLAY_BLOCK"),
        ({"max_contracts": 1}, "MODEL_CONTRACT_LIMIT_EXCEEDED"),
        (
            {"max_new_risk_cents": 180_299},
            "MODEL_RISK_LIMIT_EXCEEDED",
        ),
    ],
)
def test_overlay_dependency_matrix_only_adds_constraints(
    change: dict[str, object],
    code: str,
) -> None:
    decision = _evaluate(overlays=(_overlay("MODEL", **change),))

    assert not decision.allowed
    assert code in decision.reason_codes


def test_overlay_cannot_raise_a_base_limit_or_rescue_a_denial() -> None:
    base_denied = _limits(max_order_contracts=1)
    attempted_relaxation = _overlay("MODEL", max_contracts=2)

    decision = _evaluate(
        limits=base_denied,
        overlays=(attempted_relaxation,),
    )

    assert "ORDER_QUANTITY_LIMIT_EXCEEDED" in decision.reason_codes
    assert "MODEL_OVERLAY_RELAXATION_ATTEMPT" in decision.reason_codes


def test_duplicate_overlay_kind_is_ambiguous_and_denied() -> None:
    decision = _evaluate(
        overlays=(_overlay("REGIME"), _overlay("REGIME"))
    )

    assert decision.reason_codes == ("AMBIGUOUS_REGIME_PROVENANCE",)


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (
            lambda: _request(quantity=True),  # type: ignore[arg-type]
            "INVALID_ORDER_QUANTITY",
        ),
        (
            lambda: _limits(allowed_symbols=["SPY"]),  # type: ignore[list-item]
            "INVALID_ALLOWED_SYMBOLS",
        ),
        (
            lambda: _limits(max_fee_cents_per_contract_per_leg=0),
            "INVALID_MAX_FEE_CENTS_PER_CONTRACT_PER_LEG",
        ),
        (
            lambda: _quote(
                _contract(650),
                bid=1,
                ask=2,
                delta="NaN",
            ),
            "INVALID_OPTION_DELTA",
        ),
        (
            lambda: _portfolio(portfolio_delta=Decimal("Infinity")),
            "INVALID_PORTFOLIO_DELTA",
        ),
        (
            lambda: _authority(
                observed_at=datetime(2026, 7, 27, 17, 0)
            ),
            "INVALID_AUTHORITY_TIMESTAMP",
        ),
        (
            lambda: RiskOverlayEvidence(
                kind="UNKNOWN",
                effective_session=SESSION,
                observed_at=NOW,
                complete=True,
                approved_for_risk=True,
                block_new_risk=False,
                max_contracts=None,
                max_new_risk_cents=None,
                provenance_sha256=HASH_A,
            ),
            "INVALID_OVERLAY_KIND",
        ),
    ],
)
def test_malformed_and_nonfinite_fields_are_rejected_before_evaluation(
    factory,
    code: str,
) -> None:
    with pytest.raises(PretradeRiskValidationError) as caught:
        factory()

    assert caught.value.code == code


def test_exact_top_level_types_reject_subclasses() -> None:
    class LimitsSubclass(RiskLimits):
        pass

    base = _limits()
    subclass = LimitsSubclass(
        **{
            field: getattr(base, field)
            for field in base.__dataclass_fields__
        }
    )

    with pytest.raises(
        PretradeRiskValidationError, match="INVALID_RISK_LIMITS"
    ):
        _evaluate(limits=subclass)


def test_reason_codes_are_unique_sorted_and_stable() -> None:
    request = _request(quantity=5)
    decision = _evaluate(
        request=request,
        limits=_limits(max_order_contracts=2),
        authority=_authority(
            complete=False,
            kill_switch_clear=False,
        ),
        quotes=_quotes(request, complete=False),
        portfolio=_portfolio(complete=False, stable=False),
        overlays=(_overlay("REGIME", block_new_risk=True),),
    )

    assert decision.reason_codes == tuple(sorted(set(decision.reason_codes)))
    assert "ORDER_QUANTITY_LIMIT_EXCEEDED" in decision.reason_codes
    assert "KILL_SWITCH_ACTIVE" in decision.reason_codes
    assert "REGIME_OVERLAY_BLOCK" in decision.reason_codes
