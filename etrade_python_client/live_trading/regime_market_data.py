"""Immutable, causal SPY/VIX input contract for regime detection.

The legacy market-data caches store source-free scalar values.  This module
defines the stricter boundary required by the V2 detector: each close carries
an explicit session, source identity, payload-checksum lineage, event time, source
availability time, and local ingestion time.

The contract is deliberately independent of networking and broker code.  It
can validate and replay a stored snapshot offline, but it cannot upgrade a
normalized cache or an assumed historical timestamp into verified provenance.
Until a provider adapter durably links raw bytes and a parser receipt, even a
structurally complete provider-response manifest remains unverified.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


REGIME_DATA_SCHEMA_VERSION = "regime_market_data.v2"
CALENDAR_POLICY_VERSION = "nyse+cboe_index_options.v1"
SOURCE_POLICY_VERSION = "regime_source_identity.v1"

PAYLOAD_PROVIDER_RESPONSE = "provider_response"
PAYLOAD_NORMALIZED_ONLY = "normalized_only"

AVAILABILITY_PROVIDER_TIMESTAMP = "provider_timestamp"
AVAILABILITY_LIVE_RETRIEVAL = "live_retrieval"
AVAILABILITY_HISTORICAL_ASSUMPTION = "historical_finalization_assumption"

REQUIRED_INSTRUMENTS = ("SPY", "VIX")

_PAYLOAD_KINDS = {
    PAYLOAD_PROVIDER_RESPONSE,
    PAYLOAD_NORMALIZED_ONLY,
}
_AVAILABILITY_BASES = {
    AVAILABILITY_PROVIDER_TIMESTAMP,
    AVAILABILITY_LIVE_RETRIEVAL,
    AVAILABILITY_HISTORICAL_ASSUMPTION,
}
_OBSERVED_AVAILABILITY_BASES = {
    AVAILABILITY_PROVIDER_TIMESTAMP,
    AVAILABILITY_LIVE_RETRIEVAL,
}
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

NYSE = mcal.get_calendar("NYSE")
CBOE_INDEX_OPTIONS = mcal.get_calendar("CBOE_Index_Options")


class RegimeMarketDataError(ValueError):
    """A fail-closed market-data contract error with a stable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise RegimeMarketDataError(code, message)


def _nonempty_text(value: Any, field_name: str, *, lower: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        _fail("EMPTY_FIELD", f"{field_name} cannot be empty")
    if len(text) > 128:
        _fail("FIELD_TOO_LONG", f"{field_name} exceeds 128 characters")
    return text.lower() if lower else text


def _session_date(value: Any) -> date:
    """Parse an explicit session date without deriving it from a timestamp."""

    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        raise TypeError("session must be a date, not a datetime/timestamp")
    if isinstance(value, date):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise RegimeMarketDataError(
                "INVALID_SESSION",
                f"session is not a valid calendar date: {value}",
            ) from exc
    raise TypeError("session must be datetime.date or an ISO YYYY-MM-DD string")


def _aware_utc(value: Any, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise RegimeMarketDataError(
            "INVALID_TIMESTAMP",
            f"{field_name} is not a valid timestamp",
        ) from exc
    if timestamp.tzinfo is None:
        _fail("NAIVE_TIMESTAMP", f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _positive_close(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("close must be a real number, not a boolean")
    try:
        close = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("close must be numeric") from exc
    if not np.isfinite(close) or close <= 0:
        _fail("INVALID_CLOSE", "close must be finite and strictly positive")
    return close


def _sha256_text(value: Any, field_name: str) -> str:
    text = _nonempty_text(value, field_name, lower=True)
    if not _SHA256.fullmatch(text) or text == "0" * 64:
        _fail("INVALID_SHA256", f"{field_name} must be a non-zero SHA-256 digest")
    return text


def _iso_utc(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class SourceIdentity:
    """Semantic identity of one provider field.

    Construction normalizes textual fields but intentionally does not confer
    trust.  A snapshot accepts only identities in the explicit registry below.
    """

    provider: str
    dataset: str
    provider_symbol: str
    canonical_instrument: str
    field: str
    adjustment: str
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _nonempty_text(self.provider, "provider", lower=True),
        )
        object.__setattr__(
            self,
            "dataset",
            _nonempty_text(self.dataset, "dataset", lower=True),
        )
        object.__setattr__(
            self,
            "provider_symbol",
            _nonempty_text(self.provider_symbol, "provider_symbol").upper(),
        )
        object.__setattr__(
            self,
            "canonical_instrument",
            _nonempty_text(
                self.canonical_instrument,
                "canonical_instrument",
            ).upper(),
        )
        object.__setattr__(
            self,
            "field",
            _nonempty_text(self.field, "field", lower=True),
        )
        object.__setattr__(
            self,
            "adjustment",
            _nonempty_text(self.adjustment, "adjustment", lower=True),
        )
        object.__setattr__(
            self,
            "unit",
            _nonempty_text(self.unit, "unit", lower=True),
        )
        if self.canonical_instrument not in REQUIRED_INSTRUMENTS:
            _fail(
                "UNSUPPORTED_INSTRUMENT",
                f"unsupported canonical instrument: {self.canonical_instrument}",
            )

    @property
    def registry_key(self) -> tuple[str, ...]:
        return (
            self.provider,
            self.dataset,
            self.provider_symbol,
            self.canonical_instrument,
            self.field,
            self.adjustment,
            self.unit,
        )

    @property
    def source_field_key(self) -> tuple[str, ...]:
        return (
            self.provider,
            self.dataset,
            self.provider_symbol,
            self.field,
            self.adjustment,
            self.unit,
        )

    @property
    def is_recognized(self) -> bool:
        return self.registry_key in _SOURCE_IDENTITY_POLICY

    @property
    def is_provenance_capable(self) -> bool:
        return bool(_SOURCE_IDENTITY_POLICY.get(self.registry_key, False))

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "provider_symbol": self.provider_symbol,
            "canonical_instrument": self.canonical_instrument,
            "field": self.field,
            "adjustment": self.adjustment,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceIdentity":
        return cls(
            provider=payload["provider"],
            dataset=payload["dataset"],
            provider_symbol=payload["provider_symbol"],
            canonical_instrument=payload["canonical_instrument"],
            field=payload["field"],
            adjustment=payload["adjustment"],
            unit=payload["unit"],
        )


# The Boolean says whether the identity could support raw provenance once a
# provider adapter persists bytes and a parser receipt. It does not itself
# confer verified status. Local artifacts are shadow-only.
_SOURCE_IDENTITY_POLICY: dict[tuple[str, ...], bool] = {
    (
        "yahoo",
        "daily_prices",
        "SPY",
        "SPY",
        "close",
        "unadjusted",
        "usd",
    ): True,
    (
        "massive",
        "aggregates",
        "SPY",
        "SPY",
        "close",
        "unadjusted",
        "usd",
    ): True,
    (
        "cboe",
        "vix_daily",
        "VIX",
        "VIX",
        "close",
        "none",
        "index_points",
    ): True,
    (
        "yahoo",
        "daily_prices",
        "^VIX",
        "VIX",
        "close",
        "none",
        "index_points",
    ): True,
    (
        "local_artifact",
        "derived_cache",
        "SPY",
        "SPY",
        "close",
        "unknown",
        "usd",
    ): False,
    (
        "local_artifact",
        "derived_cache",
        "VIX",
        "VIX",
        "close",
        "unknown",
        "index_points",
    ): False,
}


def _current_source_policy_sha256() -> str:
    entries = [
        {
            "identity": list(identity),
            "raw_provenance_capable": capable,
        }
        for identity, capable in sorted(_SOURCE_IDENTITY_POLICY.items())
    ]
    payload = {
        "source_policy_version": SOURCE_POLICY_VERSION,
        "entries": entries,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def regime_market_schedule(start: Any, end: Any) -> pd.DataFrame:
    """Return the exact paired NYSE and Cboe close clock.

    SPY uses the NYSE market close.  VIX uses the Cboe Index Options market
    close, which correctly moves from 4:15 p.m. ET to 1:15 p.m. ET on the
    exchange's shortened sessions.
    """

    start_date = _session_date(start)
    end_date = _session_date(end)
    if start_date > end_date:
        _fail("INVALID_RANGE", "schedule start cannot be later than end")

    nyse = NYSE.schedule(start_date=start_date, end_date=end_date)
    cboe = CBOE_INDEX_OPTIONS.schedule(start_date=start_date, end_date=end_date)
    if nyse.empty:
        return pd.DataFrame(
            {
                "spy_event_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "vix_event_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "joint_finalization_at": pd.Series(dtype="datetime64[ns, UTC]"),
            },
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        )

    nyse_index = pd.DatetimeIndex(nyse.index).tz_localize(None).normalize()
    cboe_index = pd.DatetimeIndex(cboe.index).tz_localize(None).normalize()
    missing_cboe = nyse_index.difference(cboe_index)
    if len(missing_cboe):
        preview = [item.date().isoformat() for item in missing_cboe[:5]]
        _fail(
            "CALENDAR_SESSION_MISMATCH",
            f"Cboe calendar is missing NYSE sessions: {preview}",
        )

    nyse_closes = pd.Series(
        pd.DatetimeIndex(nyse["market_close"]).tz_convert("UTC"),
        index=nyse_index,
        name="spy_event_at",
    )
    cboe_closes = pd.Series(
        pd.DatetimeIndex(cboe["market_close"]).tz_convert("UTC"),
        index=cboe_index,
        name="vix_event_at",
    ).reindex(nyse_index)
    schedule = pd.concat([nyse_closes, cboe_closes], axis=1)
    schedule["joint_finalization_at"] = schedule.max(axis=1)
    return schedule


def _schedule_sha256(schedule: pd.DataFrame) -> str:
    rows = [
        {
            "session": session.date().isoformat(),
            "spy_event_at": _iso_utc(pd.Timestamp(row.spy_event_at)),
            "vix_event_at": _iso_utc(pd.Timestamp(row.vix_event_at)),
        }
        for session, row in schedule.iterrows()
    ]
    payload = {
        "calendar_policy_version": CALENDAR_POLICY_VERSION,
        "sessions": rows,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def regime_source_policy_sha256() -> str:
    """Return the active source-identity policy digest."""

    return _current_source_policy_sha256()


def regime_schedule_sha256(start: Any, end: Any) -> str:
    """Return the active exchange-clock digest for an inclusive date range."""

    return _schedule_sha256(regime_market_schedule(start, end))


@dataclass(frozen=True)
class MarketObservation:
    """One immutable finalized close and its source/clock lineage."""

    session: date
    identity: SourceIdentity
    close: float
    event_at: pd.Timestamp
    available_at: pd.Timestamp
    ingested_at: pd.Timestamp
    request_id: str
    raw_payload_sha256: str
    payload_kind: str
    availability_basis: str
    is_final: bool = True
    source_policy_version: str = field(init=False)
    source_policy_sha256: str = field(init=False)
    source_identity_recognized: bool = field(init=False)
    source_identity_provenance_capable: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", _session_date(self.session))
        if not isinstance(self.identity, SourceIdentity):
            raise TypeError("identity must be a SourceIdentity")
        object.__setattr__(
            self,
            "source_policy_version",
            SOURCE_POLICY_VERSION,
        )
        object.__setattr__(
            self,
            "source_policy_sha256",
            _current_source_policy_sha256(),
        )
        object.__setattr__(
            self,
            "source_identity_recognized",
            self.identity.is_recognized,
        )
        object.__setattr__(
            self,
            "source_identity_provenance_capable",
            self.identity.is_provenance_capable,
        )
        object.__setattr__(self, "close", _positive_close(self.close))
        object.__setattr__(
            self,
            "event_at",
            _aware_utc(self.event_at, "event_at"),
        )
        object.__setattr__(
            self,
            "available_at",
            _aware_utc(self.available_at, "available_at"),
        )
        object.__setattr__(
            self,
            "ingested_at",
            _aware_utc(self.ingested_at, "ingested_at"),
        )
        request_id = _nonempty_text(self.request_id, "request_id")
        if not _SAFE_REQUEST_ID.fullmatch(request_id):
            _fail(
                "INVALID_REQUEST_ID",
                "request_id must be an opaque identifier without whitespace or query data",
            )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "raw_payload_sha256",
            _sha256_text(self.raw_payload_sha256, "raw_payload_sha256"),
        )
        payload_kind = _nonempty_text(
            self.payload_kind,
            "payload_kind",
            lower=True,
        )
        if payload_kind not in _PAYLOAD_KINDS:
            _fail("UNSUPPORTED_PAYLOAD_KIND", f"unsupported payload kind: {payload_kind}")
        object.__setattr__(self, "payload_kind", payload_kind)
        availability_basis = _nonempty_text(
            self.availability_basis,
            "availability_basis",
            lower=True,
        )
        if availability_basis not in _AVAILABILITY_BASES:
            _fail(
                "UNSUPPORTED_AVAILABILITY_BASIS",
                f"unsupported availability basis: {availability_basis}",
            )
        object.__setattr__(self, "availability_basis", availability_basis)
        if not isinstance(self.is_final, (bool, np.bool_)):
            raise TypeError("is_final must be a boolean")
        object.__setattr__(self, "is_final", bool(self.is_final))

        if not self.event_at <= self.available_at <= self.ingested_at:
            _fail(
                "INVALID_TIMESTAMP_ORDER",
                "timestamps must satisfy event_at <= available_at <= ingested_at",
            )

    @property
    def instrument(self) -> str:
        return self.identity.canonical_instrument

    @property
    def effective_available_at(self) -> pd.Timestamp:
        return max(self.event_at, self.available_at, self.ingested_at)

    @property
    def provenance_evidence_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.source_identity_recognized:
            failures.append("source_identity_unrecognized")
        elif not self.source_identity_provenance_capable:
            failures.append("source_identity_shadow_only")
        if self.payload_kind != PAYLOAD_PROVIDER_RESPONSE:
            failures.append("raw_provider_payload_unavailable")
        if self.availability_basis not in _OBSERVED_AVAILABILITY_BASES:
            failures.append("availability_historical_assumption")
        if not self.is_final:
            failures.append("observation_not_final")
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session.isoformat(),
            "identity": self.identity.to_dict(),
            "close": self.close,
            "event_at": _iso_utc(self.event_at),
            "available_at": _iso_utc(self.available_at),
            "ingested_at": _iso_utc(self.ingested_at),
            "request_id": self.request_id,
            "raw_payload_sha256": self.raw_payload_sha256,
            "payload_kind": self.payload_kind,
            "availability_basis": self.availability_basis,
            "is_final": self.is_final,
            "source_policy": {
                "version": self.source_policy_version,
                "sha256": self.source_policy_sha256,
                "identity_recognized": self.source_identity_recognized,
                "identity_provenance_capable": (
                    self.source_identity_provenance_capable
                ),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketObservation":
        if "is_final" not in payload:
            _fail(
                "MISSING_IS_FINAL",
                "serialized observations must declare finality explicitly",
            )
        observation = cls(
            session=payload["session"],
            identity=SourceIdentity.from_dict(payload["identity"]),
            close=payload["close"],
            event_at=payload["event_at"],
            available_at=payload["available_at"],
            ingested_at=payload["ingested_at"],
            request_id=payload["request_id"],
            raw_payload_sha256=payload["raw_payload_sha256"],
            payload_kind=payload["payload_kind"],
            availability_basis=payload["availability_basis"],
            is_final=payload["is_final"],
        )
        stored_policy = payload.get("source_policy")
        expected_policy = {
            "version": observation.source_policy_version,
            "sha256": observation.source_policy_sha256,
            "identity_recognized": observation.source_identity_recognized,
            "identity_provenance_capable": (
                observation.source_identity_provenance_capable
            ),
        }
        if stored_policy != expected_policy:
            _fail(
                "SOURCE_POLICY_MISMATCH",
                "stored observation source-policy verdict does not match "
                "the current versioned policy",
            )
        return observation


@dataclass(frozen=True)
class RegimeMarketDataSnapshot:
    """Exact, immutable SPY/VIX history used for one detector publication."""

    as_of: pd.Timestamp
    observations: tuple[MarketObservation, ...]
    schema_version: str = REGIME_DATA_SCHEMA_VERSION
    source_policy_version: str = field(init=False)
    source_policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        as_of = _aware_utc(self.as_of, "as_of")
        object.__setattr__(self, "as_of", as_of)
        if self.schema_version != REGIME_DATA_SCHEMA_VERSION:
            _fail(
                "UNSUPPORTED_SCHEMA",
                f"unsupported regime data schema: {self.schema_version}",
            )

        try:
            observations = tuple(self.observations)
        except TypeError as exc:
            raise TypeError("observations must be an iterable") from exc
        if not observations:
            _fail("EMPTY_SNAPSHOT", "snapshot has no observations")
        if not all(isinstance(item, MarketObservation) for item in observations):
            raise TypeError("every observation must be a MarketObservation")
        policy_versions = {item.source_policy_version for item in observations}
        policy_hashes = {item.source_policy_sha256 for item in observations}
        if len(policy_versions) != 1 or len(policy_hashes) != 1:
            _fail(
                "MIXED_SOURCE_POLICY",
                "all observations must use one source-policy version and hash",
            )
        object.__setattr__(
            self,
            "source_policy_version",
            policy_versions.pop(),
        )
        object.__setattr__(
            self,
            "source_policy_sha256",
            policy_hashes.pop(),
        )

        observations = tuple(
            sorted(
                observations,
                key=lambda item: (
                    item.session,
                    item.instrument,
                    item.identity.registry_key,
                    item.request_id,
                ),
            )
        )
        object.__setattr__(self, "observations", observations)

        duplicate_keys: set[tuple[str, date]] = set()
        seen_keys: set[tuple[str, date]] = set()
        for item in observations:
            key = (item.instrument, item.session)
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)
            if item.ingested_at > as_of:
                _fail(
                    "OBSERVATION_AFTER_AS_OF",
                    f"{item.instrument} {item.session} was ingested after as_of",
                )
            if not item.is_final:
                _fail(
                    "OBSERVATION_NOT_FINAL",
                    f"{item.instrument} {item.session} is not final",
                )
            if not item.source_identity_recognized:
                _fail(
                    "SOURCE_IDENTITY_MISMATCH",
                    f"unrecognized source identity for {item.instrument}: "
                    f"{item.identity.registry_key}",
                )
        if duplicate_keys:
            preview = sorted(
                f"{instrument}:{session.isoformat()}"
                for instrument, session in duplicate_keys
            )
            _fail("DUPLICATE_OBSERVATION", f"duplicate observation keys: {preview[:5]}")

        source_fields: dict[tuple[str, ...], str] = {}
        for item in observations:
            previous = source_fields.setdefault(
                item.identity.source_field_key,
                item.instrument,
            )
            if previous != item.instrument:
                _fail(
                    "SOURCE_IDENTITY_COLLISION",
                    "one provider field cannot represent multiple canonical instruments",
                )

        sessions = sorted({item.session for item in observations})
        schedule = regime_market_schedule(sessions[0], sessions[-1])
        expected_sessions = {
            session.date()
            for session in pd.DatetimeIndex(schedule.index)
        }
        actual_sessions = set(sessions)
        if actual_sessions != expected_sessions:
            missing = sorted(expected_sessions - actual_sessions)
            unexpected = sorted(actual_sessions - expected_sessions)
            _fail(
                "SESSION_COVERAGE_MISMATCH",
                "snapshot must cover contiguous NYSE sessions; "
                f"missing={[item.isoformat() for item in missing[:5]]}, "
                f"unexpected={[item.isoformat() for item in unexpected[:5]]}",
            )

        for item in observations:
            event_column = (
                "spy_event_at"
                if item.instrument == "SPY"
                else "vix_event_at"
            )
            expected_event = pd.Timestamp(
                schedule.loc[pd.Timestamp(item.session), event_column]
            )
            if item.event_at != expected_event:
                _fail(
                    "EVENT_TIME_MISMATCH",
                    f"{item.instrument} event_at must equal "
                    f"{expected_event.isoformat()} for session "
                    f"{item.session.isoformat()}",
                )

        by_session: dict[date, set[str]] = {}
        for item in observations:
            by_session.setdefault(item.session, set()).add(item.instrument)
        incomplete = {
            session: sorted(set(REQUIRED_INSTRUMENTS) - instruments)
            for session, instruments in by_session.items()
            if instruments != set(REQUIRED_INSTRUMENTS)
        }
        if incomplete:
            preview = {
                session.isoformat(): missing
                for session, missing in sorted(incomplete.items())[:5]
            }
            _fail(
                "MISSING_REQUIRED_OBSERVATION",
                f"each session requires independent SPY and VIX observations: {preview}",
            )

    @property
    def schedule_sha256(self) -> str:
        schedule = regime_market_schedule(
            self.observations[0].session,
            self.observations[-1].session,
        )
        return _schedule_sha256(schedule)

    @property
    def provenance_failures(self) -> tuple[str, ...]:
        failures = {
            f"{item.instrument}:{item.session.isoformat()}:{failure}"
            for item in self.observations
            for failure in item.provenance_evidence_failures
        }
        failures.add("SNAPSHOT:durable_raw_payload_unverified")
        return tuple(sorted(failures))

    @property
    def provenance_verified(self) -> bool:
        return not self.provenance_failures

    @property
    def provenance_evidence_complete(self) -> bool:
        return not any(
            item.provenance_evidence_failures
            for item in self.observations
        )

    def _canonical_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calendar_policy_version": CALENDAR_POLICY_VERSION,
            "source_policy_version": self.source_policy_version,
            "source_policy_sha256": self.source_policy_sha256,
            "schedule_sha256": self.schedule_sha256,
            "as_of": _iso_utc(self.as_of),
            "observations": [item.to_dict() for item in self.observations],
        }

    @property
    def snapshot_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._canonical_body()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._canonical_body(),
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeMarketDataSnapshot":
        if not isinstance(payload, Mapping):
            raise TypeError("snapshot payload must be a mapping")
        stored_digest = _sha256_text(
            payload.get("snapshot_sha256"),
            "snapshot_sha256",
        )
        if payload.get("calendar_policy_version") != CALENDAR_POLICY_VERSION:
            _fail(
                "UNSUPPORTED_CALENDAR_POLICY",
                f"unsupported calendar policy: {payload.get('calendar_policy_version')}",
            )
        snapshot = cls(
            as_of=payload["as_of"],
            observations=tuple(
                MarketObservation.from_dict(item)
                for item in payload["observations"]
            ),
            schema_version=payload["schema_version"],
        )
        if (
            payload.get("source_policy_version")
            != snapshot.source_policy_version
            or payload.get("source_policy_sha256")
            != snapshot.source_policy_sha256
        ):
            _fail(
                "SOURCE_POLICY_MISMATCH",
                "stored snapshot source policy does not match its observations",
            )
        if payload.get("schedule_sha256") != snapshot.schedule_sha256:
            _fail(
                "SCHEDULE_CHECKSUM_MISMATCH",
                "stored calendar schedule checksum does not match observations",
            )
        if stored_digest != snapshot.snapshot_sha256:
            _fail(
                "SNAPSHOT_CHECKSUM_MISMATCH",
                "stored snapshot checksum does not match canonical contents",
            )
        return snapshot

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def detector_inputs(self) -> dict[str, Any]:
        """Return the exact low-level arguments accepted by the V2 detector."""

        sessions = sorted({item.session for item in self.observations})
        index = pd.DatetimeIndex(sessions)
        observations = {
            (item.instrument, item.session): item
            for item in self.observations
        }
        prices = pd.DataFrame(
            {
                "SPY_Close": [
                    observations[("SPY", session)].close
                    for session in sessions
                ],
                "VIX_Close": [
                    observations[("VIX", session)].close
                    for session in sessions
                ],
            },
            index=index,
        )
        spy_available_at = pd.Series(
            [
                observations[("SPY", session)].effective_available_at
                for session in sessions
            ],
            index=index,
            name="spy_available_at",
        )
        vix_available_at = pd.Series(
            [
                observations[("VIX", session)].effective_available_at
                for session in sessions
            ],
            index=index,
            name="vix_available_at",
        )
        return {
            "prices": prices,
            "as_of": self.as_of,
            "spy_available_at": spy_available_at,
            "vix_available_at": vix_available_at,
            "source_provenance_verified": self.provenance_verified,
        }

    def source_metadata_frame(self) -> pd.DataFrame:
        """Return per-session provenance columns for detector/audit output."""

        rows: dict[pd.Timestamp, dict[str, Any]] = {}
        for item in self.observations:
            prefix = item.instrument
            row = rows.setdefault(pd.Timestamp(item.session), {})
            row[f"{prefix}_Provider"] = item.identity.provider
            row[f"{prefix}_Dataset"] = item.identity.dataset
            row[f"{prefix}_Provider_Symbol"] = item.identity.provider_symbol
            row[f"{prefix}_Event_At"] = item.event_at
            row[f"{prefix}_Available_At"] = item.available_at
            row[f"{prefix}_Ingested_At"] = item.ingested_at
            row[f"{prefix}_Request_ID"] = item.request_id
            row[f"{prefix}_Raw_Payload_SHA256"] = item.raw_payload_sha256
            row[f"{prefix}_Payload_Kind"] = item.payload_kind
            row[f"{prefix}_Availability_Basis"] = item.availability_basis
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def load_snapshot_json(text: str | bytes) -> RegimeMarketDataSnapshot:
    """Load and verify a canonical snapshot JSON document."""

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RegimeMarketDataError(
                "INVALID_SNAPSHOT_ENCODING",
                "snapshot must be UTF-8",
            ) from exc
    if not isinstance(text, str):
        raise TypeError("snapshot JSON must be str or bytes")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegimeMarketDataError(
            "INVALID_SNAPSHOT_JSON",
            "snapshot is not valid JSON",
        ) from exc
    return RegimeMarketDataSnapshot.from_dict(payload)
