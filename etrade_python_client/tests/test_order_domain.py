from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from live_trading.order_domain import (
    ActiveClosingOrder,
    ClosingCapacity,
    ClosingLeg,
    DurableClosingReservation,
    OptionContractId,
    OrderDomainError,
    PositionLotCapacity,
    evaluate_order,
)


def _osi(
    strike: int,
    *,
    root: str = "SPY",
    expiry: str = "260821",
    option_type: str = "P",
) -> str:
    return f"{root:<6}{expiry}{option_type}{strike * 1000:08d}"


def _contract(
    strike: int,
    *,
    root: str = "SPY",
    adjusted: bool = False,
    multiplier: str = "100",
    deliverables: str | None = None,
) -> OptionContractId:
    return OptionContractId.from_normalized(
        {
            "symbol": "SPY",
            "security_type": "OPTN",
            "call_put": "PUT",
            "expiry_year": "2026",
            "expiry_month": "8",
            "expiry_day": "21",
            "strike_price": str(strike),
            "product_id": None,
        },
        osi_key=_osi(strike, root=root),
        option_multiplier=multiplier,
        options_adjusted_flag=adjusted,
        deliverables=deliverables,
    )


def _capacity(
    contract: OptionContractId,
    position: int,
    *,
    lots: tuple[int | Decimal, ...] | None = None,
    active: tuple[ActiveClosingOrder, ...] = (),
    reservations: tuple[DurableClosingReservation, ...] = (),
) -> ClosingCapacity:
    lot_values = (position,) if lots is None else lots
    return ClosingCapacity(
        contract=contract,
        position_quantity=Decimal(position),
        lots=tuple(
            PositionLotCapacity(
                contract=contract,
                position_lot_id=str(index),
                available_quantity=(
                    value if type(value) is Decimal else Decimal(value)
                ),
            )
            for index, value in enumerate(lot_values, start=1)
        ),
        active_closes=active,
        durable_reservations=reservations,
    )


def _credit_vertical(
    quantity: str = "2",
) -> tuple[
    tuple[ClosingLeg, ClosingLeg],
    tuple[ClosingCapacity, ClosingCapacity],
]:
    short_contract = _contract(650)
    long_contract = _contract(640)
    legs = (
        ClosingLeg(
            short_contract, "BUY_CLOSE", Decimal(quantity)
        ),
        ClosingLeg(
            long_contract, "SELL_CLOSE", Decimal(quantity)
        ),
    )
    capacities = (
        _capacity(short_contract, -2),
        _capacity(long_contract, 2),
    )
    return legs, capacities


def test_parses_exact_standard_and_weekly_osi_identity() -> None:
    standard = _contract(
        650, deliverables="100 shares of SPY"
    )
    weekly = _contract(640, root="SPYW")

    assert standard.osi_key == "SPY   260821P00650000"
    assert standard.strike == Decimal("650")
    assert standard.close_eligibility_code() is None
    assert weekly.osi_key == "SPYW  260821P00640000"
    assert weekly.close_eligibility_code() is None
    assert hash(standard.canonical_material)


def test_accepts_etrade_hyphen_padded_osi_identity() -> None:
    contract = OptionContractId.from_normalized(
        {
            "symbol": "SPY",
            "security_type": "OPTN",
            "call_put": "PUT",
            "expiry_year": "2026",
            "expiry_month": "8",
            "expiry_day": "21",
            "strike_price": "650",
            "product_id": None,
        },
        osi_key="SPY---260821P00650000",
        option_multiplier="100",
        options_adjusted_flag=False,
        deliverables="100 shares of SPY",
    )

    assert contract.close_eligibility_code() is None


@pytest.mark.parametrize(
    "osi_key, code",
    [
        ("SPY  260821P00650000", "MALFORMED_OSI"),
        (" SPY  260821P00650000", "MALFORMED_OSI"),
        ("SPY   261332P00650000", "MALFORMED_OSI"),
        ("SPY   260821X00650000", "MALFORMED_OSI"),
    ],
)
def test_rejects_malformed_osi(osi_key: str, code: str) -> None:
    with pytest.raises(OrderDomainError, match=code) as caught:
        OptionContractId.from_normalized(
            {
                "symbol": "SPY",
                "security_type": "OPTN",
                "call_put": "PUT",
                "expiry_year": "2026",
                "expiry_month": "8",
                "expiry_day": "21",
                "strike_price": "650",
            },
            osi_key=osi_key,
            option_multiplier="100",
            options_adjusted_flag=False,
            deliverables=None,
        )

    assert caught.value.code == code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "QQQ"),
        ("call_put", "CALL"),
        ("expiry_day", "22"),
        ("strike_price", "651"),
    ],
)
def test_rejects_product_to_osi_identity_mismatch(
    field: str, value: str
) -> None:
    product = {
        "symbol": "SPY",
        "security_type": "OPTN",
        "call_put": "PUT",
        "expiry_year": "2026",
        "expiry_month": "8",
        "expiry_day": "21",
        "strike_price": "650",
    }
    product[field] = value

    with pytest.raises(
        OrderDomainError, match="CONTRACT_IDENTITY_MISMATCH"
    ):
        OptionContractId.from_normalized(
            product,
            osi_key=_osi(650),
            option_multiplier="100",
            options_adjusted_flag=False,
            deliverables=None,
        )


def test_rejects_contradictory_normalized_product_id() -> None:
    product = {
        "symbol": "SPY",
        "security_type": "OPTN",
        "call_put": "PUT",
        "expiry_year": "2026",
        "expiry_month": "8",
        "expiry_day": "21",
        "strike_price": "650",
        "product_id": {
            "symbol": "QQQ",
            "type_code": "OPTN",
        },
    }

    with pytest.raises(
        OrderDomainError, match="CONTRACT_IDENTITY_MISMATCH"
    ):
        OptionContractId.from_normalized(
            product,
            osi_key=_osi(650),
            option_multiplier="100",
            options_adjusted_flag=False,
            deliverables=None,
        )


def test_allows_credit_vertical_close_and_projects_toward_zero() -> None:
    legs, capacities = _credit_vertical()

    decision = evaluate_order(legs, capacities)

    assert decision.allowed is True
    assert decision.reason_code == "CLOSE_ALLOWED"
    assert decision.requested_quantity == Decimal("2")
    assert {
        projection.position_after
        for projection in decision.projections
    } == {Decimal("0")}
    assert hash(decision.canonical_material)


def test_allows_inverse_long_vertical_close() -> None:
    long_contract = _contract(650)
    short_contract = _contract(640)
    legs = (
        ClosingLeg(
            long_contract, "SELL_CLOSE", Decimal("3")
        ),
        ClosingLeg(
            short_contract, "BUY_CLOSE", Decimal("3")
        ),
    )
    capacities = (
        _capacity(long_contract, 3),
        _capacity(short_contract, -3),
    )

    decision = evaluate_order(legs, capacities)

    assert decision.allowed is True
    assert tuple(
        projection.position_after
        for projection in decision.projections
    ) == (Decimal("0"), Decimal("0"))


def test_denies_oversubscription_after_active_close() -> None:
    legs, capacities = _credit_vertical()
    short_contract = legs[0].contract
    capacities = (
        _capacity(
            short_contract,
            -2,
            active=(
                ActiveClosingOrder(
                    short_contract,
                    "10001",
                    "BUY_CLOSE",
                    Decimal("1"),
                ),
            ),
        ),
        capacities[1],
    )

    decision = evaluate_order(legs, capacities)

    assert decision.allowed is False
    assert decision.reason_code == "INSUFFICIENT_CLOSING_CAPACITY"
    short_projection = next(
        projection
        for projection in decision.projections
        if projection.contract == short_contract
    )
    assert short_projection.available_quantity == Decimal("1")


def test_does_not_double_count_reservation_bound_to_active_order() -> None:
    short_contract = _contract(650)
    long_contract = _contract(640)
    legs = (
        ClosingLeg(
            short_contract, "BUY_CLOSE", Decimal("2")
        ),
        ClosingLeg(
            long_contract, "SELL_CLOSE", Decimal("2")
        ),
    )
    capacities = (
        _capacity(
            short_contract,
            -5,
            active=(
                ActiveClosingOrder(
                    short_contract,
                    "10001",
                    "BUY_CLOSE",
                    Decimal("2"),
                ),
            ),
            reservations=(
                DurableClosingReservation(
                    short_contract,
                    "reservation-active",
                    "BUY_CLOSE",
                    Decimal("4"),
                    broker_order_id="10001",
                ),
                DurableClosingReservation(
                    short_contract,
                    "reservation-unsent",
                    "BUY_CLOSE",
                    Decimal("1"),
                ),
            ),
        ),
        _capacity(long_contract, 5),
    )

    decision = evaluate_order(legs, capacities)

    assert decision.allowed is True
    short_projection = next(
        projection
        for projection in decision.projections
        if projection.contract == short_contract
    )
    assert short_projection.active_close_quantity == Decimal("2")
    assert (
        short_projection.unrepresented_reservation_quantity
        == Decimal("1")
    )
    assert short_projection.available_quantity == Decimal("2")


@pytest.mark.parametrize(
    ("contract", "reason_code"),
    [
        (
            lambda: _contract(650, adjusted=True),
            "ADJUSTED_OPTION_CONTRACT",
        ),
        (
            lambda: _contract(650, multiplier="10"),
            "NONSTANDARD_OPTION_MULTIPLIER",
        ),
        (
            lambda: _contract(
                650, deliverables="50 shares + cash"
            ),
            "UNSUPPORTED_OPTION_DELIVERABLES",
        ),
        (
            lambda: _contract(
                650, deliverables="100 shares of QQQ"
            ),
            "UNSUPPORTED_OPTION_DELIVERABLES",
        ),
    ],
)
def test_denies_nonstandard_contracts(
    contract, reason_code: str
) -> None:
    first = contract()
    second = _contract(640)
    legs = (
        ClosingLeg(first, "BUY_CLOSE", Decimal("1")),
        ClosingLeg(second, "SELL_CLOSE", Decimal("1")),
    )

    decision = evaluate_order(
        legs,
        (_capacity(first, -1), _capacity(second, 1)),
    )

    assert decision.allowed is False
    assert decision.reason_code == reason_code


def test_denies_sign_action_mismatch() -> None:
    legs, capacities = _credit_vertical()
    capacities = (
        _capacity(legs[0].contract, 2),
        capacities[1],
    )

    decision = evaluate_order(legs, capacities)

    assert decision.allowed is False
    assert decision.reason_code == "POSITION_ACTION_MISMATCH"


def test_denies_capacity_and_lot_identity_mismatches() -> None:
    legs, capacities = _credit_vertical()
    other = _contract(630)
    wrong_capacity = evaluate_order(
        legs,
        (capacities[0], _capacity(other, 2)),
    )
    bad_lot_capacity = ClosingCapacity(
        contract=legs[0].contract,
        position_quantity=Decimal("-2"),
        lots=(
            PositionLotCapacity(
                contract=other,
                position_lot_id="999",
                available_quantity=Decimal("-2"),
            ),
        ),
    )
    wrong_lot = evaluate_order(
        legs,
        (bad_lot_capacity, capacities[1]),
    )

    assert (
        wrong_capacity.reason_code
        == "CAPACITY_IDENTITY_MISMATCH"
    )
    assert wrong_lot.reason_code == "CAPACITY_IDENTITY_MISMATCH"


def test_denies_duplicate_lot_or_active_reservation_evidence() -> None:
    legs, capacities = _credit_vertical()
    contract = legs[0].contract
    duplicate_lots = ClosingCapacity(
        contract=contract,
        position_quantity=Decimal("-2"),
        lots=(
            PositionLotCapacity(
                contract, "1", Decimal("-1")
            ),
            PositionLotCapacity(
                contract, "1", Decimal("-1")
            ),
        ),
    )
    duplicate_active_binding = _capacity(
        contract,
        -2,
        active=(
            ActiveClosingOrder(
                contract, "10001", "BUY_CLOSE", Decimal("1")
            ),
        ),
        reservations=(
            DurableClosingReservation(
                contract,
                "reservation-1",
                "BUY_CLOSE",
                Decimal("1"),
                broker_order_id="10001",
            ),
            DurableClosingReservation(
                contract,
                "reservation-2",
                "BUY_CLOSE",
                Decimal("1"),
                broker_order_id="10001",
            ),
        ),
    )

    assert (
        evaluate_order(
            legs, (duplicate_lots, capacities[1])
        ).reason_code
        == "AMBIGUOUS_POSITION_LOT"
    )
    assert (
        evaluate_order(
            legs, (duplicate_active_binding, capacities[1])
        ).reason_code
        == "AMBIGUOUS_ACTIVE_RESERVATION"
    )


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda legs, capacities: (
                (
                    ClosingLeg(
                        legs[0].contract,
                        legs[0].action,
                        Decimal("1.5"),
                    ),
                    legs[1],
                ),
                capacities,
            ),
            "INVALID_CLOSE_QUANTITY",
        ),
        (
            lambda legs, capacities: (
                (
                    ClosingLeg(
                        legs[0].contract,
                        legs[0].action,
                        Decimal("NaN"),
                    ),
                    legs[1],
                ),
                capacities,
            ),
            "INVALID_CLOSE_QUANTITY",
        ),
        (
            lambda legs, capacities: (
                legs,
                (
                    ClosingCapacity(
                        capacities[0].contract,
                        Decimal("-1.5"),
                        capacities[0].lots,
                    ),
                    capacities[1],
                ),
            ),
            "INVALID_POSITION_QUANTITY",
        ),
        (
            lambda legs, capacities: (
                legs,
                (
                    ClosingCapacity(
                        capacities[0].contract,
                        capacities[0].position_quantity,
                        (
                            PositionLotCapacity(
                                capacities[0].contract,
                                "1",
                                Decimal("-1.5"),
                            ),
                        ),
                    ),
                    capacities[1],
                ),
            ),
            "INVALID_LOT_QUANTITY",
        ),
        (
            lambda legs, capacities: (
                legs,
                (
                    _capacity(
                        capacities[0].contract,
                        -2,
                        active=(
                            ActiveClosingOrder(
                                capacities[0].contract,
                                "10001",
                                "BUY_CLOSE",
                                Decimal("1.5"),
                            ),
                        ),
                    ),
                    capacities[1],
                ),
            ),
            "INVALID_ACTIVE_CLOSE_QUANTITY",
        ),
        (
            lambda legs, capacities: (
                legs,
                (
                    _capacity(
                        capacities[0].contract,
                        -2,
                        reservations=(
                            DurableClosingReservation(
                                capacities[0].contract,
                                "reservation-1",
                                "BUY_CLOSE",
                                Decimal("1.5"),
                            ),
                        ),
                    ),
                    capacities[1],
                ),
            ),
            "INVALID_RESERVATION_QUANTITY",
        ),
    ],
)
def test_denies_noninteger_quantities(
    mutate, reason_code: str
) -> None:
    legs, capacities = _credit_vertical()
    changed_legs, changed_capacities = mutate(legs, capacities)

    decision = evaluate_order(
        changed_legs, changed_capacities
    )

    assert decision.allowed is False
    assert decision.reason_code == reason_code


def test_denies_duplicate_or_partial_vertical_legs() -> None:
    legs, capacities = _credit_vertical()
    duplicate = (
        legs[0],
        ClosingLeg(
            legs[0].contract, "SELL_CLOSE", Decimal("2")
        ),
    )
    unequal = (
        legs[0],
        ClosingLeg(
            legs[1].contract, "SELL_CLOSE", Decimal("1")
        ),
    )

    assert (
        evaluate_order(duplicate, capacities).reason_code
        == "DUPLICATE_CLOSE_LEGS"
    )
    assert (
        evaluate_order(unequal, capacities).reason_code
        == "UNEQUAL_CLOSE_QUANTITY"
    )
    assert (
        evaluate_order((legs[0],), (capacities[0],)).reason_code
        == "INCOMPLETE_VERTICAL_CLOSE"
    )


def test_canonical_decision_is_order_independent_and_immutable() -> None:
    legs, capacities = _credit_vertical()

    forward = evaluate_order(legs, capacities)
    reverse = evaluate_order(
        tuple(reversed(legs)), tuple(reversed(capacities))
    )

    assert forward.canonical_material == reverse.canonical_material
    with pytest.raises(FrozenInstanceError):
        forward.allowed = False  # type: ignore[misc]
