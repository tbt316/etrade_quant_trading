"""Point-in-time option-contract universe contracts for deterministic replay."""

from __future__ import annotations

import hashlib
import hmac
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
CONTRACT_UNIVERSE_TERMINAL_CHAIN_STATUS = (
    "INTERNAL_TERMINAL_CHAIN_REPLAYED"
)
CONTRACT_REFERENCE_REPLAY_SCOPE = (
    "INTERNAL_METADATA_ONLY_RAW_RESPONSE_NOT_RETAINED"
)
CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION = 1
CONTRACT_REFERENCE_ENVIRONMENT = "historical_research"
CONTRACT_REFERENCE_SOURCE = "massive_v3_reference_options_contracts"
CONTRACT_REFERENCE_MAX_PAGES_PER_ROOT = 100
CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE = 1000
CONTRACT_REFERENCE_DIGEST_DOMAIN = (
    "etrade_backtest_contract_reference_snapshot_v1"
)


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


def parse_option_ticker_identity(
    option_ticker: str,
) -> tuple[str, str, str, str]:
    """Parse an OCC-style option ticker from its fixed-width suffix."""

    if type(option_ticker) is not str or not option_ticker:
        raise ContractUniverseError("option_ticker must be a non-empty string")
    clean = (
        option_ticker[2:]
        if option_ticker.startswith("O:")
        else option_ticker
    )
    if len(clean) <= 15:
        raise ContractUniverseError("option_ticker has an invalid OCC suffix")
    root = clean[:-15]
    date_text = clean[-15:-9]
    flag = clean[-9]
    strike_digits = clean[-8:]
    if (
        not root
        or not date_text.isdigit()
        or flag not in {"P", "C"}
        or not strike_digits.isdigit()
    ):
        raise ContractUniverseError("option_ticker has an invalid OCC suffix")
    try:
        expiration = datetime.strptime(date_text, "%y%m%d").date().isoformat()
    except ValueError as exc:
        raise ContractUniverseError(
            "option_ticker has an invalid expiration"
        ) from exc
    strike_decimal = Decimal(int(strike_digits)) / Decimal(1000)
    if strike_decimal <= 0:
        raise ContractUniverseError(
            "option_ticker strike must be positive"
        )
    return (
        root,
        expiration,
        "put" if flag == "P" else "call",
        format(strike_decimal.normalize(), "f"),
    )


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
        _, _, ticker_type, ticker_strike = parse_option_ticker_identity(
            self.option_ticker
        )
        if ticker_type != self.contract_type:
            raise ContractUniverseError(
                "option_ticker type disagrees with contract_type"
            )
        if ticker_strike != _canonical_strike(strike):
            raise ContractUniverseError(
                "option_ticker strike disagrees with contract strike"
            )
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
class ContractReferencePageEvidence:
    """Credential-free identity and pagination linkage for one response page."""

    root_ordinal: int
    root_ticker: str
    page_ordinal: int
    request_sha256: str
    response_sha256: str
    result_count: int
    next_request_sha256: str | None
    is_terminal: bool

    def __post_init__(self) -> None:
        if type(self.root_ordinal) is not int or self.root_ordinal < 0:
            raise ContractUniverseError("page root_ordinal is invalid")
        if type(self.root_ticker) is not str or not self.root_ticker:
            raise ContractUniverseError("page root_ticker is invalid")
        if type(self.page_ordinal) is not int or self.page_ordinal < 1:
            raise ContractUniverseError("page page_ordinal is invalid")
        for field_name, value in (
            ("request_sha256", self.request_sha256),
            ("response_sha256", self.response_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ContractUniverseError(f"page {field_name} is invalid")
        if (
            type(self.result_count) is not int
            or self.result_count < 0
            or self.result_count > CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE
        ):
            raise ContractUniverseError("page result_count is invalid")
        if type(self.is_terminal) is not bool:
            raise ContractUniverseError("page is_terminal must be exact bool")
        if self.is_terminal:
            if self.next_request_sha256 is not None:
                raise ContractUniverseError(
                    "terminal page cannot carry next request identity"
                )
        elif (
            type(self.next_request_sha256) is not str
            or len(self.next_request_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.next_request_sha256
            )
        ):
            raise ContractUniverseError(
                "non-terminal page requires next request identity"
            )

    def canonical_record(self) -> dict[str, Any]:
        return {
            "root_ordinal": self.root_ordinal,
            "root_ticker": self.root_ticker,
            "page_ordinal": self.page_ordinal,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "result_count": self.result_count,
            "next_request_sha256": self.next_request_sha256,
            "is_terminal": self.is_terminal,
        }


def contract_reference_snapshot_sha256(
    *,
    schema_version: int,
    environment: str,
    source: str,
    underlying: str,
    expiration: str,
    contract_type: str,
    as_of_date: str,
    expected_roots: tuple[str, ...],
    pages: tuple[ContractReferencePageEvidence, ...],
    contracts: tuple[OptionContractIdentity, ...],
) -> str:
    """Return the domain-separated digest for complete reference evidence."""

    payload = {
        "domain": CONTRACT_REFERENCE_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "environment": environment,
        "source": source,
        "underlying": underlying,
        "expiration": expiration,
        "contract_type": contract_type,
        "as_of_date": as_of_date,
        "expected_roots": list(expected_roots),
        "pages": [page.canonical_record() for page in pages],
        "contracts": [
            contract.canonical_record() for contract in contracts
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConfirmedContractReferenceSnapshot:
    """Rows backed by durable terminal-chain metadata and declared hashes.

    This type does not claim provider entitlement, intraday availability, or
    independent parser replay from raw response bytes; raw bytes are
    intentionally not retained.
    """

    underlying: str
    expiration: str
    contract_type: str
    as_of_date: str
    attempt_id: str
    snapshot_sha256: str
    confirmed_at: str
    expected_roots: tuple[str, ...]
    pages: tuple[ContractReferencePageEvidence, ...]
    contracts: tuple[OptionContractIdentity, ...]
    environment: str = CONTRACT_REFERENCE_ENVIRONMENT
    source: str = CONTRACT_REFERENCE_SOURCE
    schema_version: int = CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ContractUniverseError("reference schema_version must be exact int")
        if self.schema_version != CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION:
            raise ContractUniverseError("unsupported contract-reference schema")
        if self.environment != CONTRACT_REFERENCE_ENVIRONMENT:
            raise ContractUniverseError("unexpected contract-reference environment")
        if self.source != CONTRACT_REFERENCE_SOURCE:
            raise ContractUniverseError("unexpected contract-reference source")
        if type(self.underlying) is not str or not self.underlying:
            raise ContractUniverseError("reference underlying must be non-empty")
        _iso_date(self.expiration, "reference expiration")
        _iso_date(self.as_of_date, "reference as_of_date")
        if self.expiration <= self.as_of_date:
            raise ContractUniverseError(
                "reference expiration must be after as_of_date"
            )
        if self.contract_type not in {"put", "call"}:
            raise ContractUniverseError("reference contract_type must be put or call")
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise ContractUniverseError("reference attempt_id must be non-empty")
        if (
            type(self.snapshot_sha256) is not str
            or len(self.snapshot_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.snapshot_sha256)
        ):
            raise ContractUniverseError(
                "reference snapshot_sha256 must be lowercase SHA-256"
            )
        _utc_timestamp(self.confirmed_at, "reference confirmed_at")
        if (
            type(self.expected_roots) is not tuple
            or not self.expected_roots
            or any(type(root) is not str or not root for root in self.expected_roots)
            or len(set(self.expected_roots)) != len(self.expected_roots)
        ):
            raise ContractUniverseError(
                "reference expected_roots must be a unique non-empty tuple"
            )
        canonical_roots = (
            ("SPX", "SPXW")
            if self.underlying == "SPX"
            else (self.underlying,)
        )
        if self.expected_roots != canonical_roots:
            raise ContractUniverseError(
                "reference expected_roots disagree with underlying"
            )
        if type(self.pages) is not tuple:
            raise ContractUniverseError("reference pages must be immutable tuple")
        if any(
            type(page) is not ContractReferencePageEvidence
            for page in self.pages
        ):
            raise ContractUniverseError(
                "reference pages must contain exact page evidence"
            )
        if self.pages != tuple(
            sorted(
                self.pages,
                key=lambda page: (
                    page.root_ordinal,
                    page.page_ordinal,
                ),
            )
        ):
            raise ContractUniverseError(
                "reference pages must be canonically ordered"
            )
        expected_page_count = 0
        seen_requests: set[str] = set()
        for root_ordinal, root_ticker in enumerate(self.expected_roots):
            root_pages = [
                page
                for page in self.pages
                if page.root_ordinal == root_ordinal
            ]
            if not root_pages:
                raise ContractUniverseError(
                    "reference evidence lacks an expected root"
                )
            if len(root_pages) > CONTRACT_REFERENCE_MAX_PAGES_PER_ROOT:
                raise ContractUniverseError(
                    "reference evidence exceeds the page bound"
                )
            for page_ordinal, page in enumerate(root_pages, start=1):
                if page.request_sha256 in seen_requests:
                    raise ContractUniverseError(
                        "reference request identity repeats"
                    )
                seen_requests.add(page.request_sha256)
                if (
                    page.root_ticker != root_ticker
                    or page.page_ordinal != page_ordinal
                ):
                    raise ContractUniverseError(
                        "reference page chain is not contiguous"
                    )
                is_last = page_ordinal == len(root_pages)
                if is_last:
                    if not page.is_terminal:
                        raise ContractUniverseError(
                            "reference root lacks terminal page"
                        )
                else:
                    next_page = root_pages[page_ordinal]
                    if (
                        page.is_terminal
                        or page.next_request_sha256
                        != next_page.request_sha256
                    ):
                        raise ContractUniverseError(
                            "reference page chain linkage is invalid"
                        )
            expected_page_count += len(root_pages)
        if expected_page_count != len(self.pages):
            raise ContractUniverseError(
                "reference pages contain an unexpected root"
            )
        if type(self.contracts) is not tuple or not self.contracts:
            raise ContractUniverseError(
                "confirmed reference snapshot must contain contracts"
            )
        prior: OptionContractIdentity | None = None
        seen: set[str] = set()
        root_contract_counts = {
            root: 0 for root in self.expected_roots
        }
        for contract in self.contracts:
            if type(contract) is not OptionContractIdentity:
                raise ContractUniverseError(
                    "reference contracts must be exact identities"
                )
            if contract.contract_type != self.contract_type:
                raise ContractUniverseError(
                    "reference contract type disagrees with snapshot"
                )
            if contract.option_ticker in seen:
                raise ContractUniverseError(
                    "duplicate ticker in confirmed reference snapshot"
                )
            if prior is not None and prior > contract:
                raise ContractUniverseError(
                    "reference contracts must be canonically sorted"
                )
            root, expiration, _, _ = parse_option_ticker_identity(
                contract.option_ticker
            )
            if (
                root not in root_contract_counts
                or expiration != self.expiration
            ):
                raise ContractUniverseError(
                    "reference contract identity disagrees with query tuple"
                )
            root_contract_counts[root] += 1
            seen.add(contract.option_ticker)
            prior = contract
        if sum(page.result_count for page in self.pages) != len(self.contracts):
            raise ContractUniverseError(
                "reference page counts disagree with contracts"
            )
        for root_ordinal, root_ticker in enumerate(self.expected_roots):
            root_result_count = sum(
                page.result_count
                for page in self.pages
                if page.root_ordinal == root_ordinal
            )
            if root_result_count != root_contract_counts[root_ticker]:
                raise ContractUniverseError(
                    "reference root page counts disagree with contracts"
                )
        recomputed = contract_reference_snapshot_sha256(
            schema_version=self.schema_version,
            environment=self.environment,
            source=self.source,
            underlying=self.underlying,
            expiration=self.expiration,
            contract_type=self.contract_type,
            as_of_date=self.as_of_date,
            expected_roots=self.expected_roots,
            pages=self.pages,
            contracts=self.contracts,
        )
        if not hmac.compare_digest(recomputed, self.snapshot_sha256):
            raise ContractUniverseError(
                "reference snapshot digest verification failed"
            )

    def as_legacy_records(self) -> list[dict[str, Any]]:
        return [contract.as_legacy_record() for contract in self.contracts]

    @property
    def page_count(self) -> int:
        return len(self.pages)


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
    reference_terminal_chain_replayed: bool
    reference_snapshot_sha256: str | None
    reference_attempt_id: str | None
    reference_page_count: int | None
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
        if self.completeness_status not in {
            CONTRACT_UNIVERSE_COMPLETENESS_STATUS,
            CONTRACT_UNIVERSE_TERMINAL_CHAIN_STATUS,
        }:
            raise ContractUniverseError("unsupported completeness status")
        if type(self.reference_terminal_chain_replayed) is not bool:
            raise ContractUniverseError(
                "reference_terminal_chain_replayed must be an exact boolean"
            )
        if (
            self.completeness_status
            == CONTRACT_UNIVERSE_COMPLETENESS_STATUS
        ):
            if self.reference_terminal_chain_replayed:
                raise ContractUniverseError(
                    "unverified completeness cannot claim terminal replay"
                )
            if any(
                value is not None
                for value in (
                    self.reference_snapshot_sha256,
                    self.reference_attempt_id,
                    self.reference_page_count,
                )
            ):
                raise ContractUniverseError(
                    "unverified completeness cannot carry confirmed references"
                )
        else:
            if not self.reference_terminal_chain_replayed:
                raise ContractUniverseError(
                    "terminal-chain status requires internal replay"
                )
            if (
                type(self.reference_snapshot_sha256) is not str
                or len(self.reference_snapshot_sha256) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.reference_snapshot_sha256
                )
                or type(self.reference_attempt_id) is not str
                or not self.reference_attempt_id
                or type(self.reference_page_count) is not int
                or self.reference_page_count < 1
            ):
                raise ContractUniverseError(
                    "terminal-chain replay requires exact reference lineage"
                )
        if (
            self.reference_terminal_chain_replayed
            and self.completeness_status
            != CONTRACT_UNIVERSE_TERMINAL_CHAIN_STATUS
        ):
            raise ContractUniverseError(
                "terminal-chain replay and completeness status disagree"
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
            "reference_terminal_chain_replayed": (
                self.reference_terminal_chain_replayed
            ),
            "reference_evidence_replay_scope": (
                CONTRACT_REFERENCE_REPLAY_SCOPE
                if self.reference_terminal_chain_replayed
                else "NONE"
            ),
            "reference_snapshot_sha256": self.reference_snapshot_sha256,
            "reference_attempt_id": self.reference_attempt_id,
            "reference_page_count": self.reference_page_count,
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
    contracts: Sequence[Mapping[str, Any]] | None = None,
    reference_evidence: ConfirmedContractReferenceSnapshot | None = None,
    source: str = CONTRACT_REFERENCE_SOURCE,
) -> ContractUniverseSnapshot:
    """Normalize one exact-as-of result into an immutable, status-bearing manifest."""

    if reference_evidence is not None:
        if type(reference_evidence) is not ConfirmedContractReferenceSnapshot:
            raise ContractUniverseError(
                "reference_evidence must be exact confirmed evidence"
            )
        if contracts is not None:
            raise ContractUniverseError(
                "contracts and reference_evidence are mutually exclusive"
            )
        if (
            reference_evidence.underlying != underlying
            or reference_evidence.expiration != expiration
            or reference_evidence.contract_type != contract_type
            or reference_evidence.as_of_date != as_of_date
        ):
            raise ContractUniverseError(
                "reference evidence disagrees with requested snapshot tuple"
            )
        normalized = list(reference_evidence.contracts)
        completeness_status = CONTRACT_UNIVERSE_TERMINAL_CHAIN_STATUS
        reference_terminal_chain_replayed = True
        reference_snapshot_sha256 = reference_evidence.snapshot_sha256
        reference_attempt_id = reference_evidence.attempt_id
        reference_page_count = reference_evidence.page_count
    else:
        if isinstance(contracts, (str, bytes)) or not isinstance(
            contracts, Sequence
        ):
            raise ContractUniverseError("contracts must be a sequence of mappings")
        normalized = []
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
        completeness_status = CONTRACT_UNIVERSE_COMPLETENESS_STATUS
        reference_terminal_chain_replayed = False
        reference_snapshot_sha256 = None
        reference_attempt_id = None
        reference_page_count = None
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
        completeness_status=completeness_status,
        reference_terminal_chain_replayed=(
            reference_terminal_chain_replayed
        ),
        reference_snapshot_sha256=reference_snapshot_sha256,
        reference_attempt_id=reference_attempt_id,
        reference_page_count=reference_page_count,
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
