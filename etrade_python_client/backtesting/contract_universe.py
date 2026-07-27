"""Point-in-time option-contract universe contracts for deterministic replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_UNIVERSE_SCHEMA_VERSION = 1
MASSIVE_AS_OF_AVAILABILITY_POLICY = (
    "massive_reference_as_of_date_granular_modeled_close_v1"
)
DATE_GRANULAR_AVAILABILITY_BASIS = (
    "MODELED_NOT_PROVIDER_OBSERVED"
)
CONTRACT_UNIVERSE_CAUSAL_STATUS = "UNVERIFIED"
CONTRACT_UNIVERSE_COMPLETENESS_STATUS = "UNVERIFIED"


class ContractUniverseError(ValueError):
    """Raised when a contract universe violates its declared causal boundary."""


def _utc_timestamp(value: str, field_name: str) -> datetime:
    if type(value) is not str or not value:
        raise ContractUniverseError(f"{field_name} must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractUniverseError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractUniverseError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso_date(value: str, field_name: str) -> str:
    if type(value) is not str:
        raise ContractUniverseError(f"{field_name} must be an ISO date")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ContractUniverseError(f"{field_name} must be an ISO date") from exc
    return parsed.date().isoformat()


def _canonical_strike(value: float) -> str:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ContractUniverseError("contract strike must be numeric") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ContractUniverseError("contract strike must be finite and positive")
    return format(decimal_value.normalize(), "f")


@dataclass(frozen=True, order=True)
class OptionContractIdentity:
    """Immutable identity admitted by one historical reference snapshot."""

    option_ticker: str
    strike: float
    contract_type: str

    def __post_init__(self) -> None:
        if type(self.option_ticker) is not str or not self.option_ticker:
            raise ContractUniverseError("option_ticker must be a non-empty string")
        if self.contract_type not in {"put", "call"}:
            raise ContractUniverseError("contract_type must be put or call")
        if type(self.strike) not in {int, float} or type(self.strike) is bool:
            raise ContractUniverseError("contract strike must be an exact number")
        strike = float(self.strike)
        if not math.isfinite(strike) or strike <= 0:
            raise ContractUniverseError("contract strike must be finite and positive")
        object.__setattr__(self, "strike", strike)

    def canonical_record(self) -> dict[str, str]:
        return {
            "option_ticker": self.option_ticker,
            "strike": _canonical_strike(self.strike),
            "contract_type": self.contract_type,
        }

    def as_legacy_record(self) -> dict[str, Any]:
        return {
            "option_ticker": self.option_ticker,
            "strike": self.strike,
            "contract_type": self.contract_type,
        }


@dataclass(frozen=True)
class ContractUniverseSnapshot:
    """Content-addressed identities under an explicit, unverified time model."""

    underlying: str
    expiration: str
    contract_type: str
    as_of_date: str
    available_at: str
    decision_time: str
    source: str
    availability_policy: str
    availability_basis: str
    provider_available_at_verified: bool
    causal_status: str
    completeness_status: str
    provider_page_evidence_verified: bool
    contracts: tuple[OptionContractIdentity, ...]
    schema_version: int = CONTRACT_UNIVERSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ContractUniverseError("schema_version must be an exact integer")
        if self.schema_version != CONTRACT_UNIVERSE_SCHEMA_VERSION:
            raise ContractUniverseError("unsupported contract-universe schema")
        if type(self.underlying) is not str or not self.underlying:
            raise ContractUniverseError("underlying must be a non-empty string")
        if self.contract_type not in {"put", "call"}:
            raise ContractUniverseError("contract_type must be put or call")
        _iso_date(self.expiration, "expiration")
        _iso_date(self.as_of_date, "as_of_date")
        if self.expiration <= self.as_of_date:
            raise ContractUniverseError("expiration must be after as_of_date")
        available_at = _utc_timestamp(self.available_at, "available_at")
        decision_time = _utc_timestamp(self.decision_time, "decision_time")
        if available_at > decision_time:
            raise ContractUniverseError(
                "contract universe was not available at the decision time"
            )
        if self.as_of_date != decision_time.date().isoformat():
            raise ContractUniverseError(
                "as_of_date must match the UTC date of decision_time"
            )
        if type(self.source) is not str or not self.source:
            raise ContractUniverseError("source must be a non-empty string")
        if self.availability_policy != MASSIVE_AS_OF_AVAILABILITY_POLICY:
            raise ContractUniverseError("unsupported availability policy")
        if self.availability_basis != DATE_GRANULAR_AVAILABILITY_BASIS:
            raise ContractUniverseError("unsupported availability basis")
        if type(self.provider_available_at_verified) is not bool:
            raise ContractUniverseError(
                "provider_available_at_verified must be an exact boolean"
            )
        if self.provider_available_at_verified:
            raise ContractUniverseError(
                "date-granular Massive as_of data has no verified intraday "
                "available_at evidence"
            )
        if self.causal_status != CONTRACT_UNIVERSE_CAUSAL_STATUS:
            raise ContractUniverseError(
                "date-granular contract snapshots must remain UNVERIFIED"
            )
        if (
            self.completeness_status
            != CONTRACT_UNIVERSE_COMPLETENESS_STATUS
        ):
            raise ContractUniverseError(
                "contract-universe completeness must remain UNVERIFIED"
            )
        if type(self.provider_page_evidence_verified) is not bool:
            raise ContractUniverseError(
                "provider_page_evidence_verified must be an exact boolean"
            )
        if self.provider_page_evidence_verified:
            raise ContractUniverseError(
                "raw adapter provides no durable complete-pagination evidence"
            )
        if type(self.contracts) is not tuple:
            raise ContractUniverseError("contracts must be an immutable tuple")
        if not self.contracts:
            raise ContractUniverseError("contract universe cannot be empty")

        tickers: set[str] = set()
        prior: OptionContractIdentity | None = None
        for contract in self.contracts:
            if type(contract) is not OptionContractIdentity:
                raise ContractUniverseError(
                    "contracts must contain exact OptionContractIdentity values"
                )
            if contract.contract_type != self.contract_type:
                raise ContractUniverseError("contract type disagrees with snapshot")
            if contract.option_ticker in tickers:
                raise ContractUniverseError("duplicate option_ticker in snapshot")
            if prior is not None and prior > contract:
                raise ContractUniverseError("contracts must be canonically sorted")
            tickers.add(contract.option_ticker)
            prior = contract

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "underlying": self.underlying,
            "expiration": self.expiration,
            "contract_type": self.contract_type,
            "as_of_date": self.as_of_date,
            "available_at": self.available_at,
            "decision_time": self.decision_time,
            "source": self.source,
            "availability_policy": self.availability_policy,
            "availability_basis": self.availability_basis,
            "provider_available_at_verified": (
                self.provider_available_at_verified
            ),
            "causal_status": self.causal_status,
            "completeness_status": self.completeness_status,
            "provider_page_evidence_verified": (
                self.provider_page_evidence_verified
            ),
            "contracts": [
                contract.canonical_record() for contract in self.contracts
            ],
        }

    @property
    def snapshot_sha256(self) -> str:
        encoded = json.dumps(
            self._canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        payload = self._canonical_payload()
        payload["snapshot_sha256"] = self.snapshot_sha256
        return payload

    def assert_usable_at(self, decision_time: str) -> None:
        requested = _utc_timestamp(decision_time, "decision_time")
        sealed = _utc_timestamp(self.decision_time, "snapshot decision_time")
        available = _utc_timestamp(self.available_at, "available_at")
        if requested != sealed:
            raise ContractUniverseError(
                "snapshot is bound to a different decision timestamp"
            )
        if available > requested:
            raise ContractUniverseError(
                "contract universe was not available at the decision time"
            )


def build_contract_universe_snapshot(
    *,
    underlying: str,
    expiration: str,
    contract_type: str,
    as_of_date: str,
    available_at: str,
    decision_time: str,
    contracts: Sequence[Mapping[str, Any]],
    source: str = "massive_reference_options_contracts",
) -> ContractUniverseSnapshot:
    """Normalize one exact-as-of result into an immutable, status-bearing manifest."""

    if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
        raise ContractUniverseError("contracts must be a sequence of mappings")
    normalized: list[OptionContractIdentity] = []
    for row in contracts:
        if not isinstance(row, Mapping):
            raise ContractUniverseError("contract rows must be mappings")
        normalized.append(
            OptionContractIdentity(
                option_ticker=row.get("option_ticker"),
                strike=row.get("strike"),
                contract_type=row.get("contract_type", contract_type),
            )
        )
    normalized.sort()
    return ContractUniverseSnapshot(
        underlying=underlying,
        expiration=expiration,
        contract_type=contract_type,
        as_of_date=as_of_date,
        available_at=available_at,
        decision_time=decision_time,
        source=source,
        availability_policy=MASSIVE_AS_OF_AVAILABILITY_POLICY,
        availability_basis=DATE_GRANULAR_AVAILABILITY_BASIS,
        provider_available_at_verified=False,
        causal_status=CONTRACT_UNIVERSE_CAUSAL_STATUS,
        completeness_status=CONTRACT_UNIVERSE_COMPLETENESS_STATUS,
        provider_page_evidence_verified=False,
        contracts=tuple(normalized),
    )


def acquisition_union_by_expiration(
    snapshots: Iterable[ContractUniverseSnapshot],
    contract_type: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return a deterministic over-fetch union; never use it for eligibility."""

    if contract_type not in {"put", "call"}:
        raise ContractUniverseError("contract_type must be put or call")
    by_expiration: dict[str, dict[str, OptionContractIdentity]] = {}
    for snapshot in snapshots:
        if type(snapshot) is not ContractUniverseSnapshot:
            raise ContractUniverseError("snapshot collection contains invalid value")
        if snapshot.contract_type != contract_type:
            continue
        ticker_map = by_expiration.setdefault(snapshot.expiration, {})
        for contract in snapshot.contracts:
            prior = ticker_map.get(contract.option_ticker)
            if prior is not None and prior != contract:
                raise ContractUniverseError(
                    "contract identity changed across point-in-time snapshots"
                )
            ticker_map[contract.option_ticker] = contract
    return {
        expiration: [
            ticker_map[ticker].as_legacy_record()
            for ticker in sorted(ticker_map)
        ]
        for expiration, ticker_map in sorted(by_expiration.items())
    }


def filter_chain_for_snapshot(
    chain_rows: Sequence[Mapping[str, Any]],
    snapshot: ContractUniverseSnapshot,
    decision_time: str,
) -> list[dict[str, Any]]:
    """Admit only exact-date rows whose identity existed in the PIT snapshot."""

    if type(snapshot) is not ContractUniverseSnapshot:
        raise ContractUniverseError("snapshot must be exact ContractUniverseSnapshot")
    snapshot.assert_usable_at(decision_time)
    if isinstance(chain_rows, (str, bytes)) or not isinstance(chain_rows, Sequence):
        raise ContractUniverseError("chain_rows must be a sequence of mappings")

    identities = {
        contract.option_ticker: contract for contract in snapshot.contracts
    }
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in chain_rows:
        if not isinstance(raw_row, Mapping):
            raise ContractUniverseError("chain row must be a mapping")
        ticker = raw_row.get("option_ticker")
        contract = identities.get(ticker)
        if contract is None:
            continue
        if ticker in seen:
            raise ContractUniverseError(
                "duplicate eligible contract row at decision time"
            )
        if raw_row.get("pricing_date") != snapshot.as_of_date:
            raise ContractUniverseError(
                "eligible chain row must match snapshot as_of_date"
            )
        if raw_row.get("expiration") != snapshot.expiration:
            raise ContractUniverseError(
                "eligible chain row expiration disagrees with snapshot"
            )
        if raw_row.get("contract_type") != snapshot.contract_type:
            raise ContractUniverseError(
                "eligible chain row type disagrees with snapshot"
            )
        if _canonical_strike(raw_row.get("strike")) != _canonical_strike(
            contract.strike
        ):
            raise ContractUniverseError(
                "eligible chain row strike disagrees with snapshot"
            )
        seen.add(ticker)
        filtered.append(dict(raw_row))
    return sorted(
        filtered,
        key=lambda row: (float(row["strike"]), str(row["option_ticker"])),
    )
