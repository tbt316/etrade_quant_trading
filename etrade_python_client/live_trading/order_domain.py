"""Pure option-closing identity and capacity policy.

The types in this module are immutable value objects.  They perform no I/O,
read no clock, and depend on no process state.  Callers remain responsible for
proving that the broker evidence supplied here is fresh and durable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any


_BUY_CLOSE = "BUY_CLOSE"
_SELL_CLOSE = "SELL_CLOSE"
_CLOSE_ACTIONS = frozenset({_BUY_CLOSE, _SELL_CLOSE})
_DECISION_DOMAIN = "etrade-closing-risk.v1"
_DECIMAL_PRECISION = 50
_MAX_QUANTITY = Decimal("2147483647")
_MAX_DECIMAL_ADJUSTED_EXPONENT = 18


class OrderDomainError(ValueError):
    """A deterministic, fail-closed domain validation error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class OptionContractId:
    """Exact option identity, including the broker's OCC/OSI evidence."""

    symbol: str
    expiry: date
    call_put: str
    strike: Decimal
    osi_key: str
    multiplier: Decimal
    adjusted: bool
    deliverables: str | None

    def __post_init__(self) -> None:
        if (
            type(self.symbol) is not str
            or not 1 <= len(self.symbol) <= 6
            or not self.symbol.isascii()
            or not self.symbol[0].isalpha()
            or any(
                character
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
                for character in self.symbol
            )
            or self.symbol != self.symbol.upper()
        ):
            raise OrderDomainError("INVALID_OPTION_SYMBOL")
        if type(self.expiry) is not date:
            raise OrderDomainError("INVALID_OPTION_EXPIRY")
        if (
            type(self.call_put) is not str
            or self.call_put not in {"CALL", "PUT"}
        ):
            raise OrderDomainError("INVALID_CALL_PUT")
        if not _is_finite_positive_decimal(self.strike):
            raise OrderDomainError("INVALID_STRIKE")
        if not _is_finite_positive_decimal(self.multiplier):
            raise OrderDomainError("INVALID_MULTIPLIER")
        if type(self.adjusted) is not bool:
            raise OrderDomainError("INVALID_ADJUSTED_FLAG")
        if self.deliverables is not None and (
            type(self.deliverables) is not str
            or not self.deliverables
            or len(self.deliverables) > 512
            or any(
                ord(character) < 32 or ord(character) > 126
                for character in self.deliverables
            )
        ):
            raise OrderDomainError("INVALID_DELIVERABLES")

        root, expiry, call_put, strike = _parse_osi(self.osi_key)
        normalized_symbol = self.symbol.replace(".", "").replace("-", "")
        normalized_root = root.replace(".", "").replace("-", "")
        allowed_roots = {normalized_symbol}
        if len(normalized_symbol) <= 5:
            allowed_roots.add(f"{normalized_symbol}W")
        if (
            normalized_root not in allowed_roots
            or expiry != self.expiry
            or call_put != self.call_put
            or strike != self.strike
        ):
            raise OrderDomainError("CONTRACT_IDENTITY_MISMATCH")

    @classmethod
    def from_normalized(
        cls,
        product: dict[str, Any],
        *,
        osi_key: str,
        option_multiplier: str,
        options_adjusted_flag: bool,
        deliverables: str | None,
    ) -> "OptionContractId":
        """Parse one strict normalized broker-reader position identity."""

        if type(product) is not dict:
            raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
        if product.get("security_type") != "OPTN":
            raise OrderDomainError("NOT_OPTION_CONTRACT")
        try:
            symbol = product["symbol"]
            call_put = product["call_put"]
            year = _normalized_positive_int(product["expiry_year"])
            month = _normalized_positive_int(product["expiry_month"])
            day = _normalized_positive_int(product["expiry_day"])
            strike = _normalized_decimal(product["strike_price"])
            multiplier = _normalized_decimal(option_multiplier)
        except (KeyError, TypeError, ValueError, OrderDomainError) as exc:
            if isinstance(exc, OrderDomainError):
                raise
            raise OrderDomainError("INVALID_NORMALIZED_PRODUCT") from exc
        product_id = product.get("product_id")
        if product_id is not None:
            if type(product_id) is not dict:
                raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
            product_id_symbol = product_id.get("symbol")
            product_id_type = product_id.get("type_code")
            if (
                product_id_symbol is not None
                and product_id_symbol != symbol
                or product_id_type is not None
                and product_id_type != "OPTN"
            ):
                raise OrderDomainError("CONTRACT_IDENTITY_MISMATCH")
        try:
            expiry = date(year, month, day)
        except ValueError as exc:
            raise OrderDomainError("INVALID_OPTION_EXPIRY") from exc
        return cls(
            symbol=symbol,
            expiry=expiry,
            call_put=call_put,
            strike=strike,
            osi_key=osi_key,
            multiplier=multiplier,
            adjusted=options_adjusted_flag,
            deliverables=deliverables,
        )

    @property
    def canonical_material(self) -> tuple[str, ...]:
        return (
            self.symbol,
            self.expiry.isoformat(),
            self.call_put,
            _decimal_text(self.strike),
            self.osi_key,
            _decimal_text(self.multiplier),
            "true" if self.adjusted else "false",
            self.deliverables or "",
        )

    def close_eligibility_code(self) -> str | None:
        if self.multiplier != Decimal("100"):
            return "NONSTANDARD_OPTION_MULTIPLIER"
        if self.adjusted:
            return "ADJUSTED_OPTION_CONTRACT"
        if self.deliverables not in {
            None,
            "100 shares",
            f"100 shares of {self.symbol}",
        }:
            return "UNSUPPORTED_OPTION_DELIVERABLES"
        return None


@dataclass(frozen=True, slots=True)
class ClosingLeg:
    contract: OptionContractId
    action: str
    quantity: Decimal


@dataclass(frozen=True, slots=True)
class PositionLotCapacity:
    contract: OptionContractId
    position_lot_id: str
    available_quantity: Decimal


@dataclass(frozen=True, slots=True)
class ActiveClosingOrder:
    contract: OptionContractId
    broker_order_id: str
    action: str
    remaining_quantity: Decimal


@dataclass(frozen=True, slots=True)
class DurableClosingReservation:
    contract: OptionContractId
    reservation_id: str
    action: str
    quantity: Decimal
    broker_order_id: str | None = None


@dataclass(frozen=True, slots=True)
class ClosingCapacity:
    contract: OptionContractId
    position_quantity: Decimal
    lots: tuple[PositionLotCapacity, ...]
    active_closes: tuple[ActiveClosingOrder, ...] = ()
    durable_reservations: tuple[DurableClosingReservation, ...] = ()


@dataclass(frozen=True, slots=True)
class PositionEffectProjection:
    contract: OptionContractId
    action: str
    requested_quantity: Decimal
    position_before: Decimal
    position_after: Decimal
    position_capacity: Decimal
    lot_capacity: Decimal
    active_close_quantity: Decimal
    unrepresented_reservation_quantity: Decimal
    available_quantity: Decimal
    capacity_evidence_material: tuple[Any, ...]

    @property
    def canonical_material(self) -> tuple[Any, ...]:
        return (
            self.contract.canonical_material,
            self.action,
            _decimal_text(self.requested_quantity),
            _decimal_text(self.position_before),
            _decimal_text(self.position_after),
            _decimal_text(self.position_capacity),
            _decimal_text(self.lot_capacity),
            _decimal_text(self.active_close_quantity),
            _decimal_text(self.unrepresented_reservation_quantity),
            _decimal_text(self.available_quantity),
            self.capacity_evidence_material,
        )


@dataclass(frozen=True, slots=True)
class ClosingRiskDecision:
    allowed: bool
    reason_code: str
    requested_quantity: Decimal | None
    projections: tuple[PositionEffectProjection, ...]
    canonical_material: tuple[Any, ...]


def project_position_effect(
    leg: ClosingLeg,
    capacity: ClosingCapacity,
) -> PositionEffectProjection:
    """Project one close leg after all existing close claims."""

    _validate_leg(leg)
    if type(capacity) is not ClosingCapacity:
        raise OrderDomainError("INVALID_CLOSING_CAPACITY")
    if capacity.contract != leg.contract:
        raise OrderDomainError("CAPACITY_IDENTITY_MISMATCH")
    if type(capacity.position_quantity) is not Decimal or not _is_integer(
        capacity.position_quantity
    ):
        raise OrderDomainError("INVALID_POSITION_QUANTITY")
    if capacity.position_quantity == 0:
        raise OrderDomainError("NO_POSITION_TO_CLOSE")
    expected_action = closing_action_for_position(
        capacity.position_quantity
    )
    if leg.action != expected_action:
        raise OrderDomainError("POSITION_ACTION_MISMATCH")

    if type(capacity.lots) is not tuple:
        raise OrderDomainError("INVALID_POSITION_LOTS")
    lot_ids: set[str] = set()
    lot_material: list[tuple[str, str]] = []
    lot_capacity = Decimal("0")
    position_is_positive = capacity.position_quantity > 0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        for lot in capacity.lots:
            if (
                type(lot) is not PositionLotCapacity
                or lot.contract != capacity.contract
            ):
                raise OrderDomainError("CAPACITY_IDENTITY_MISMATCH")
            _validate_broker_id(
                lot.position_lot_id, "INVALID_POSITION_LOT_ID"
            )
            if lot.position_lot_id in lot_ids:
                raise OrderDomainError("AMBIGUOUS_POSITION_LOT")
            lot_ids.add(lot.position_lot_id)
            if (
                type(lot.available_quantity) is not Decimal
                or not _is_integer(lot.available_quantity)
            ):
                raise OrderDomainError("INVALID_LOT_QUANTITY")
            if lot.available_quantity != 0 and (
                (lot.available_quantity > 0) != position_is_positive
            ):
                raise OrderDomainError("LOT_SIGN_MISMATCH")
            lot_capacity += abs(lot.available_quantity)
            lot_material.append(
                (
                    lot.position_lot_id,
                    _decimal_text(lot.available_quantity),
                )
            )

    if type(capacity.active_closes) is not tuple:
        raise OrderDomainError("INVALID_ACTIVE_CLOSES")
    active_ids: set[str] = set()
    active_material: list[tuple[str, str, str]] = []
    active_quantity = Decimal("0")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        for active in capacity.active_closes:
            if (
                type(active) is not ActiveClosingOrder
                or active.contract != capacity.contract
            ):
                raise OrderDomainError("CAPACITY_IDENTITY_MISMATCH")
            _validate_broker_id(
                active.broker_order_id, "INVALID_BROKER_ORDER_ID"
            )
            if active.broker_order_id in active_ids:
                raise OrderDomainError("AMBIGUOUS_ACTIVE_CLOSE")
            active_ids.add(active.broker_order_id)
            if active.action != expected_action:
                raise OrderDomainError("ACTIVE_CLOSE_ACTION_MISMATCH")
            if not _is_positive_integer(active.remaining_quantity):
                raise OrderDomainError(
                    "INVALID_ACTIVE_CLOSE_QUANTITY"
                )
            active_quantity += active.remaining_quantity
            active_material.append(
                (
                    active.broker_order_id,
                    active.action,
                    _decimal_text(active.remaining_quantity),
                )
            )

    if type(capacity.durable_reservations) is not tuple:
        raise OrderDomainError("INVALID_DURABLE_RESERVATIONS")
    reservation_ids: set[str] = set()
    represented_active_ids: set[str] = set()
    reservation_material: list[tuple[str, str, str, str]] = []
    unrepresented_reservations = Decimal("0")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        for reservation in capacity.durable_reservations:
            if (
                type(reservation) is not DurableClosingReservation
                or reservation.contract != capacity.contract
            ):
                raise OrderDomainError("CAPACITY_IDENTITY_MISMATCH")
            _validate_id(
                reservation.reservation_id,
                "INVALID_RESERVATION_ID",
            )
            if reservation.reservation_id in reservation_ids:
                raise OrderDomainError(
                    "AMBIGUOUS_DURABLE_RESERVATION"
                )
            reservation_ids.add(reservation.reservation_id)
            if reservation.action != expected_action:
                raise OrderDomainError("RESERVATION_ACTION_MISMATCH")
            if not _is_positive_integer(reservation.quantity):
                raise OrderDomainError(
                    "INVALID_RESERVATION_QUANTITY"
                )
            broker_order_id = reservation.broker_order_id
            if broker_order_id is not None:
                _validate_broker_id(
                    broker_order_id, "INVALID_BROKER_ORDER_ID"
                )
            if broker_order_id in active_ids:
                if broker_order_id in represented_active_ids:
                    raise OrderDomainError(
                        "AMBIGUOUS_ACTIVE_RESERVATION"
                    )
                represented_active_ids.add(broker_order_id)
            else:
                unrepresented_reservations += reservation.quantity
            reservation_material.append(
                (
                    reservation.reservation_id,
                    reservation.action,
                    _decimal_text(reservation.quantity),
                    broker_order_id or "",
                )
            )

        position_capacity = abs(capacity.position_quantity)
        gross_capacity = min(position_capacity, lot_capacity)
        available = max(
            Decimal("0"),
            gross_capacity
            - active_quantity
            - unrepresented_reservations,
        )
        if leg.quantity > available:
            position_after = capacity.position_quantity
        elif leg.action == _BUY_CLOSE:
            position_after = (
                capacity.position_quantity + leg.quantity
            )
        else:
            position_after = (
                capacity.position_quantity - leg.quantity
            )
    evidence_material = (
        capacity.contract.canonical_material,
        _decimal_text(capacity.position_quantity),
        tuple(sorted(lot_material)),
        tuple(sorted(active_material)),
        tuple(sorted(reservation_material)),
    )
    return PositionEffectProjection(
        contract=leg.contract,
        action=leg.action,
        requested_quantity=leg.quantity,
        position_before=capacity.position_quantity,
        position_after=position_after,
        position_capacity=position_capacity,
        lot_capacity=lot_capacity,
        active_close_quantity=active_quantity,
        unrepresented_reservation_quantity=unrepresented_reservations,
        available_quantity=available,
        capacity_evidence_material=evidence_material,
    )


def evaluate_order(
    legs: tuple[ClosingLeg, ...],
    capacities: tuple[ClosingCapacity, ...],
) -> ClosingRiskDecision:
    """Evaluate an exact two-leg vertical close and fail closed."""

    if type(legs) is not tuple or len(legs) != 2:
        return _deny("INCOMPLETE_VERTICAL_CLOSE")
    try:
        for leg in legs:
            _validate_leg(leg)
    except OrderDomainError as exc:
        return _deny(exc.code)

    leg_material = tuple(
        sorted(_leg_material(leg) for leg in legs)
    )
    first, second = legs
    if first.contract == second.contract:
        return _deny("DUPLICATE_CLOSE_LEGS", evidence=leg_material)
    if (
        first.contract.symbol != second.contract.symbol
        or first.contract.expiry != second.contract.expiry
        or first.contract.call_put != second.contract.call_put
        or first.contract.strike == second.contract.strike
    ):
        return _deny("NOT_A_VERTICAL_CLOSE", evidence=leg_material)
    if first.quantity != second.quantity:
        return _deny("UNEQUAL_CLOSE_QUANTITY", evidence=leg_material)
    if first.action == second.action:
        return _deny(
            "CLOSE_ACTIONS_NOT_OPPOSITE",
            requested_quantity=first.quantity,
            evidence=leg_material,
        )
    for leg in legs:
        eligibility = leg.contract.close_eligibility_code()
        if eligibility is not None:
            return _deny(
                eligibility,
                requested_quantity=first.quantity,
                evidence=leg_material,
            )

    if type(capacities) is not tuple or len(capacities) != 2:
        return _deny(
            "CAPACITY_SET_MISMATCH",
            requested_quantity=first.quantity,
            evidence=leg_material,
        )
    capacity_by_contract: dict[OptionContractId, ClosingCapacity] = {}
    for capacity in capacities:
        if (
            type(capacity) is not ClosingCapacity
            or type(capacity.contract) is not OptionContractId
            or capacity.contract in capacity_by_contract
        ):
            return _deny(
                "CAPACITY_SET_MISMATCH",
                requested_quantity=first.quantity,
                evidence=leg_material,
            )
        capacity_by_contract[capacity.contract] = capacity
    if set(capacity_by_contract) != {
        first.contract,
        second.contract,
    }:
        return _deny(
            "CAPACITY_IDENTITY_MISMATCH",
            requested_quantity=first.quantity,
            evidence=leg_material,
        )

    projections: list[PositionEffectProjection] = []
    try:
        for leg in legs:
            projections.append(
                project_position_effect(
                    leg, capacity_by_contract[leg.contract]
                )
            )
    except OrderDomainError as exc:
        return _deny(
            exc.code,
            requested_quantity=first.quantity,
            evidence=leg_material,
        )
    ordered_projections = tuple(
        sorted(
            projections,
            key=lambda projection: projection.contract.canonical_material,
        )
    )
    if any(
        projection.requested_quantity
        > projection.available_quantity
        for projection in ordered_projections
    ):
        return _deny(
            "INSUFFICIENT_CLOSING_CAPACITY",
            requested_quantity=first.quantity,
            projections=ordered_projections,
            evidence=leg_material,
        )
    material = (
        _DECISION_DOMAIN,
        "ALLOW",
        "CLOSE_ALLOWED",
        _decimal_text(first.quantity),
        leg_material,
        tuple(
            projection.canonical_material
            for projection in ordered_projections
        ),
    )
    return ClosingRiskDecision(
        allowed=True,
        reason_code="CLOSE_ALLOWED",
        requested_quantity=first.quantity,
        projections=ordered_projections,
        canonical_material=material,
    )


def closing_action_for_position(position_quantity: Decimal) -> str:
    """Return the only action that can reduce the signed position."""

    if type(position_quantity) is not Decimal or not _is_integer(
        position_quantity
    ):
        raise OrderDomainError("INVALID_POSITION_QUANTITY")
    if position_quantity < 0:
        return _BUY_CLOSE
    if position_quantity > 0:
        return _SELL_CLOSE
    raise OrderDomainError("NO_POSITION_TO_CLOSE")


def _validate_leg(leg: ClosingLeg) -> None:
    if type(leg) is not ClosingLeg or type(leg.contract) is not OptionContractId:
        raise OrderDomainError("INVALID_CLOSE_LEG")
    if type(leg.action) is not str or leg.action not in _CLOSE_ACTIONS:
        raise OrderDomainError("INVALID_CLOSE_ACTION")
    if not _is_positive_integer(leg.quantity):
        raise OrderDomainError("INVALID_CLOSE_QUANTITY")


def _deny(
    reason_code: str,
    *,
    requested_quantity: Decimal | None = None,
    projections: tuple[PositionEffectProjection, ...] = (),
    evidence: tuple[Any, ...] = (),
) -> ClosingRiskDecision:
    material = (
        _DECISION_DOMAIN,
        "DENY",
        reason_code,
        (
            ""
            if requested_quantity is None
            else _decimal_text(requested_quantity)
        ),
        evidence,
        tuple(
            projection.canonical_material
            for projection in projections
        ),
    )
    return ClosingRiskDecision(
        allowed=False,
        reason_code=reason_code,
        requested_quantity=requested_quantity,
        projections=projections,
        canonical_material=material,
    )


def _leg_material(leg: ClosingLeg) -> tuple[Any, ...]:
    return (
        leg.contract.canonical_material,
        leg.action,
        _decimal_text(leg.quantity),
    )


def _parse_osi(value: Any) -> tuple[str, date, str, Decimal]:
    if (
        type(value) is not str
        or len(value) != 21
        or not value.isascii()
    ):
        raise OrderDomainError("MALFORMED_OSI")
    padded_root = value[:6]
    root = padded_root.rstrip("- ")
    padding = padded_root[len(root):]
    if (
        not root
        or (not padding and len(root) != 6)
        or len(set(padding)) > 1
        or any(character not in "- " for character in padding)
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
            for character in root
        )
        or root != root.upper()
    ):
        raise OrderDomainError("MALFORMED_OSI")
    expiry_text = value[6:12]
    option_type = value[12]
    strike_text = value[13:]
    if (
        not expiry_text.isascii()
        or not expiry_text.isdigit()
        or option_type not in {"C", "P"}
        or not strike_text.isascii()
        or not strike_text.isdigit()
    ):
        raise OrderDomainError("MALFORMED_OSI")
    try:
        expiry = date(
            2000 + int(expiry_text[:2]),
            int(expiry_text[2:4]),
            int(expiry_text[4:6]),
        )
    except ValueError as exc:
        raise OrderDomainError("MALFORMED_OSI") from exc
    return (
        root,
        expiry,
        "CALL" if option_type == "C" else "PUT",
        Decimal(
            f"{strike_text[:-3]}.{strike_text[-3:]}"
        ),
    )


def _normalized_positive_int(value: Any) -> int:
    if (
        type(value) is not str
        or not value
        or len(value) > 4
        or not value.isascii()
        or not value.isdigit()
        or value.startswith("0")
    ):
        raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
    result = int(value)
    if result <= 0 or str(result) != value:
        raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
    return result


def _normalized_decimal(value: Any) -> Decimal:
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or not value.isascii()
    ):
        raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise OrderDomainError("INVALID_NORMALIZED_PRODUCT") from exc
    if (
        not result.is_finite()
        or result <= 0
        or abs(result.adjusted()) > _MAX_DECIMAL_ADJUSTED_EXPONENT
        or _decimal_text(result) != value
    ):
        raise OrderDomainError("INVALID_NORMALIZED_PRODUCT")
    return result


def _validate_id(value: Any, code: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or any(
            ord(character) < 33 or ord(character) > 126
            for character in value
        )
    ):
        raise OrderDomainError(code)


def _validate_broker_id(value: Any, code: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 19
        or not value.isascii()
        or not value.isdigit()
        or value.startswith("0")
        or int(value) > 9_223_372_036_854_775_807
    ):
        raise OrderDomainError(code)


def _is_finite_positive_decimal(value: Any) -> bool:
    return (
        type(value) is Decimal
        and value.is_finite()
        and value > 0
        and abs(value.adjusted()) <= _MAX_DECIMAL_ADJUSTED_EXPONENT
    )


def _is_integer(value: Decimal) -> bool:
    return (
        value.is_finite()
        and abs(value) <= _MAX_QUANTITY
        and abs(value.as_tuple().exponent)
        <= _MAX_DECIMAL_ADJUSTED_EXPONENT
        and value == value.to_integral_value()
    )


def _is_positive_integer(value: Any) -> bool:
    return (
        type(value) is Decimal
        and _is_integer(value)
        and value > 0
    )


def _decimal_text(value: Decimal) -> str:
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


__all__ = [
    "ActiveClosingOrder",
    "ClosingCapacity",
    "ClosingLeg",
    "ClosingRiskDecision",
    "DurableClosingReservation",
    "OptionContractId",
    "OrderDomainError",
    "PositionEffectProjection",
    "PositionLotCapacity",
    "closing_action_for_position",
    "evaluate_order",
    "project_position_effect",
]
