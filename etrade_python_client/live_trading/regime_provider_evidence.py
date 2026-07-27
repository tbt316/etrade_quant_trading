"""Strict, replayable provider parsers for the V2 SPY/VIX evidence path.

The module deliberately has no network client.  Its boundary is a retained
plain-byte HTTP entity plus a declarative parser configuration.  That makes a
receipt deterministic to reparse and prevents credentials from entering the
evidence database.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import urlsplit

import pandas as pd

from live_trading.regime_market_data import (
    AVAILABILITY_LIVE_RETRIEVAL,
    PAYLOAD_PROVIDER_RESPONSE,
    MarketObservation,
    SourceIdentity,
    regime_market_schedule,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_NAME = re.compile(
    r"(?:authorization|api[-_]?key|token|secret|cookie|password)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(r"(?:bearer\s+|basic\s+|api[-_]?key)", re.IGNORECASE)
_CONFIG_HASH_DOMAIN = b"regime-provider-parser-config.v1\0"
_RESPONSE_HASH_DOMAIN = b"regime-provider-response.v1\0"
_MASSIVE_IDENTITY = SourceIdentity(
    provider="massive",
    dataset="aggregates",
    provider_symbol="SPY",
    canonical_instrument="SPY",
    field="close",
    adjustment="unadjusted",
    unit="usd",
)
_CBOE_IDENTITY = SourceIdentity(
    provider="cboe",
    dataset="vix_daily",
    provider_symbol="VIX",
    canonical_instrument="VIX",
    field="close",
    adjustment="none",
    unit="index_points",
)
_FINALITY_POLICY = (
    "observed_after_exchange_close_not_provider_attested.v1"
)


class ProviderEvidenceError(ValueError):
    """Raised when retained provider bytes cannot support a safe observation."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_sha256(domain: bytes, payload: str) -> str:
    return hashlib.sha256(domain + payload.encode("utf-8")).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderEvidenceError(
                f"Massive response contains a duplicate JSON field: {key}"
            )
        result[key] = value
    return result


def _utc_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ProviderEvidenceError(f"{field_name} is not a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ProviderEvidenceError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _session(value: Any, field_name: str = "session") -> date:
    if isinstance(value, datetime):
        raise ProviderEvidenceError(f"{field_name} must be a calendar date")
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            raise ProviderEvidenceError(f"{field_name} must be a calendar date")
        value = value.date()
    if not isinstance(value, date):
        raise ProviderEvidenceError(f"{field_name} must be a calendar date")
    return value


def _safe_pairs(value: Iterable[tuple[str, str]], field_name: str) -> tuple[tuple[str, str], ...]:
    try:
        pairs = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of key/value pairs") from exc
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{field_name} entries must be two-item pairs")
        key, raw_value = item
        if not isinstance(key, str) or not isinstance(raw_value, str):
            raise TypeError(f"{field_name} keys and values must be strings")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            raise ProviderEvidenceError(f"{field_name} cannot contain empty keys or values")
        if len(key) > 128 or len(raw_value) > 512:
            raise ProviderEvidenceError(f"{field_name} contains an overlong value")
        if _SENSITIVE_NAME.search(key) or _SENSITIVE_VALUE.search(raw_value):
            raise ProviderEvidenceError(f"{field_name} cannot contain credentials or secrets")
        lowered = key.lower()
        if lowered in seen:
            raise ProviderEvidenceError(f"{field_name} cannot contain duplicate keys")
        seen.add(lowered)
        normalized.append((key, raw_value))
    return tuple(sorted(normalized, key=lambda pair: (pair[0].lower(), pair[0], pair[1])))


def _safe_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("endpoint must be a non-empty URL string")
    endpoint = value.strip()
    parsed = urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ProviderEvidenceError("endpoint contains an invalid port") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderEvidenceError(
            "endpoint must use the default HTTPS authority without "
            "credentials, query, or fragment"
        )
    if len(endpoint) > 1_024:
        raise ProviderEvidenceError("endpoint is too long")
    return endpoint


def _identity_from_payload(payload: Mapping[str, Any]) -> SourceIdentity:
    if not isinstance(payload, Mapping):
        raise TypeError("source_identity must be an object")
    return SourceIdentity.from_dict(payload)


@dataclass(frozen=True)
class RawProviderResponse:
    """One complete, plain-byte response with credential-safe metadata.

    ``body`` is the exact decoded entity consumed by the parser.  It must not
    be re-encoded, pretty-printed, or otherwise normalized before capture.
    """

    provider: str
    endpoint: str
    requested_sessions: tuple[date, ...]
    request_started_at: Any
    completed_at: Any
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    request_parameters: tuple[tuple[str, str], ...] = ()
    body_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise TypeError("provider must be a non-empty string")
        object.__setattr__(self, "provider", self.provider.strip().lower())
        object.__setattr__(self, "endpoint", _safe_endpoint(self.endpoint))
        try:
            sessions = tuple(_session(item, "requested_sessions item") for item in self.requested_sessions)
        except TypeError as exc:
            raise TypeError("requested_sessions must be an iterable of dates") from exc
        if not sessions or len(set(sessions)) != len(sessions) or sessions != tuple(sorted(sessions)):
            raise ProviderEvidenceError("requested_sessions must be non-empty, unique, and sorted")
        object.__setattr__(self, "requested_sessions", sessions)
        started = _utc_timestamp(self.request_started_at, "request_started_at")
        completed = _utc_timestamp(self.completed_at, "completed_at")
        if completed < started:
            raise ProviderEvidenceError("completed_at cannot precede request_started_at")
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "completed_at", completed)
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise TypeError("status_code must be an integer")
        if not 100 <= self.status_code <= 599:
            raise ProviderEvidenceError("status_code must be an HTTP status code")
        object.__setattr__(self, "headers", _safe_pairs(self.headers, "headers"))
        object.__setattr__(self, "request_parameters", _safe_pairs(self.request_parameters, "request_parameters"))
        if not isinstance(self.body, bytes):
            raise TypeError("body must be exact plain entity bytes")
        if not isinstance(self.body_complete, bool):
            raise TypeError("body_complete must be a boolean")
        if not self.body_complete:
            raise ProviderEvidenceError("partial response bodies cannot become evidence")

    @property
    def body_sha256(self) -> str:
        return _sha256_bytes(self.body)

    @property
    def media_type(self) -> str:
        for key, value in self.headers:
            if key.lower() == "content-type":
                return value.split(";", 1)[0].strip().lower() or "application/octet-stream"
        return "application/octet-stream"

    @property
    def content_encoding(self) -> str:
        """Encoding of the retained parser-input bytes."""

        return "identity"

    @property
    def transport_content_encoding(self) -> str:
        """Original HTTP Content-Encoding before transport decoding."""

        for key, value in self.headers:
            if key.lower() == "content-encoding":
                return value.lower()
        return "identity"

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "regime_raw_provider_response.v1",
            "provider": self.provider,
            "endpoint": self.endpoint,
            "requested_sessions": [item.isoformat() for item in self.requested_sessions],
            "request_parameters": [list(item) for item in self.request_parameters],
            "request_started_at": self.request_started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "status_code": self.status_code,
            "headers": [list(item) for item in self.headers],
            "media_type": self.media_type,
            "content_encoding": self.content_encoding,
            "transport_content_encoding": self.transport_content_encoding,
            "body_representation": "decoded_http_entity_parser_input",
            "body_sha256": self.body_sha256,
            "byte_length": len(self.body),
            "body_complete": self.body_complete,
        }

    @property
    def response_sha256(self) -> str:
        return _domain_sha256(_RESPONSE_HASH_DOMAIN, _canonical_json(self.to_metadata_dict()))


@dataclass(frozen=True)
class ProviderParseConfig:
    """Fully serialized parser policy; no implicit current-time inputs."""

    parser_id: str
    requested_sessions: tuple[date, ...]
    source_identity: SourceIdentity
    request_parameters: tuple[tuple[str, str], ...] = ()
    finality_policy: str = _FINALITY_POLICY
    parser_options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parser_id, str) or not self.parser_id.strip():
            raise TypeError("parser_id must be a non-empty string")
        object.__setattr__(self, "parser_id", self.parser_id.strip())
        try:
            sessions = tuple(_session(item, "requested_sessions item") for item in self.requested_sessions)
        except TypeError as exc:
            raise TypeError("requested_sessions must be an iterable of dates") from exc
        if not sessions or len(set(sessions)) != len(sessions) or sessions != tuple(sorted(sessions)):
            raise ProviderEvidenceError("requested_sessions must be non-empty, unique, and sorted")
        object.__setattr__(self, "requested_sessions", sessions)
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be a SourceIdentity")
        object.__setattr__(self, "request_parameters", _safe_pairs(self.request_parameters, "request_parameters"))
        object.__setattr__(self, "parser_options", _safe_pairs(self.parser_options, "parser_options"))
        if self.finality_policy != _FINALITY_POLICY:
            raise ProviderEvidenceError("unsupported finality_policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "regime_provider_parse_config.v1",
            "parser_id": self.parser_id,
            "requested_sessions": [item.isoformat() for item in self.requested_sessions],
            "source_identity": self.source_identity.to_dict(),
            "request_parameters": [list(item) for item in self.request_parameters],
            "finality_policy": self.finality_policy,
            "parser_options": [list(item) for item in self.parser_options],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def config_sha256(self) -> str:
        return _domain_sha256(_CONFIG_HASH_DOMAIN, self.to_json())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProviderParseConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("parser config must be an object")
        if payload.get("schema_version") != "regime_provider_parse_config.v1":
            raise ProviderEvidenceError("unsupported parser configuration schema")
        expected = {
            "schema_version", "parser_id", "requested_sessions", "source_identity",
            "request_parameters", "finality_policy", "parser_options",
        }
        if set(payload) != expected:
            raise ProviderEvidenceError("parser configuration fields do not match the schema")
        try:
            sessions = tuple(date.fromisoformat(item) for item in payload["requested_sessions"])
        except (TypeError, ValueError) as exc:
            raise ProviderEvidenceError("parser configuration has invalid sessions") from exc
        return cls(
            parser_id=payload["parser_id"],
            requested_sessions=sessions,
            source_identity=_identity_from_payload(payload["source_identity"]),
            request_parameters=tuple(tuple(item) for item in payload["request_parameters"]),
            finality_policy=payload["finality_policy"],
            parser_options=tuple(tuple(item) for item in payload["parser_options"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ProviderParseConfig":
        if not isinstance(payload, str):
            raise TypeError("parser config JSON must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderEvidenceError("parser configuration is not JSON") from exc
        if _canonical_json(decoded) != payload:
            raise ProviderEvidenceError("parser configuration JSON is not canonical")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class ProviderSourceLocator:
    """Stable source location of one parsed close inside the retained body."""

    kind: str
    value: str

    def __post_init__(self) -> None:
        if self.kind not in {"json_pointer", "csv_row"}:
            raise ProviderEvidenceError("unsupported source locator kind")
        if not isinstance(self.value, str) or not self.value.strip() or len(self.value) > 256:
            raise ProviderEvidenceError("source locator value must be a short non-empty string")
        object.__setattr__(self, "value", self.value.strip())

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True)
class ParsedProviderOutput:
    observation: MarketObservation
    source_locator: ProviderSourceLocator

    def __post_init__(self) -> None:
        if not isinstance(self.observation, MarketObservation):
            raise TypeError("observation must be a MarketObservation")
        if not isinstance(self.source_locator, ProviderSourceLocator):
            raise TypeError("source_locator must be a ProviderSourceLocator")


@dataclass(frozen=True)
class ParsedProviderBatch:
    parser_id: str
    parser_version: str
    parser_code_sha256: str
    config_sha256: str
    response_sha256: str
    outputs: tuple[ParsedProviderOutput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parser_id, str) or not self.parser_id:
            raise ProviderEvidenceError("parser_id must be a non-empty string")
        if not isinstance(self.parser_version, str) or not self.parser_version:
            raise ProviderEvidenceError("parser_version must be a non-empty string")
        for field_name in ("parser_code_sha256", "config_sha256", "response_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ProviderEvidenceError(f"{field_name} must be a SHA-256 digest")
        try:
            outputs = tuple(self.outputs)
        except TypeError as exc:
            raise TypeError("outputs must be an iterable") from exc
        if not outputs:
            raise ProviderEvidenceError("parsed provider batch cannot be empty")
        if not all(isinstance(item, ParsedProviderOutput) for item in outputs):
            raise TypeError("outputs must contain ParsedProviderOutput values")
        object.__setattr__(self, "outputs", outputs)


class ProviderParser(Protocol):
    parser_id: str
    parser_version: str

    @property
    def code_sha256(self) -> str: ...

    def parse(
        self,
        response: RawProviderResponse,
        config: ProviderParseConfig,
        *,
        ingested_at: Any,
    ) -> ParsedProviderBatch: ...


def _require_exact_identity(actual: SourceIdentity, expected: SourceIdentity) -> None:
    if actual.registry_key != expected.registry_key:
        raise ProviderEvidenceError("parser source_identity does not match the provider contract")


def _require_final(response: RawProviderResponse, session: date, instrument: str) -> pd.Timestamp:
    """Apply the versioned after-close acceptance policy.

    Providers do not attest that these historical endpoints are immutable.
    ``is_final`` therefore means accepted after the official exchange close
    under ``_FINALITY_POLICY``; later corrections remain append-only revisions.
    """

    schedule = regime_market_schedule(session, session)
    column = "spy_event_at" if instrument == "SPY" else "vix_event_at"
    event_at = pd.Timestamp(schedule.loc[pd.Timestamp(session), column])
    if response.completed_at < event_at:
        raise ProviderEvidenceError(
            f"{instrument} response completed before the official exchange close for {session.isoformat()}"
        )
    return event_at


def _require_nyse_sessions(sessions: tuple[date, ...]) -> None:
    """Reject holiday/weekend rows before they can obtain a false close clock."""

    for session in sessions:
        if regime_market_schedule(session, session).empty:
            raise ProviderEvidenceError(
                f"requested session is not a NYSE trading session: {session.isoformat()}"
            )


def _observation(
    *,
    session: date,
    identity: SourceIdentity,
    close: float,
    event_at: pd.Timestamp,
    response: RawProviderResponse,
    ingested_at: Any,
) -> MarketObservation:
    ingested = _utc_timestamp(ingested_at, "ingested_at")
    if ingested < response.completed_at:
        raise ProviderEvidenceError("ingested_at cannot precede response completion")
    return MarketObservation(
        session=session,
        identity=identity,
        close=close,
        event_at=event_at,
        available_at=response.completed_at,
        ingested_at=ingested,
        request_id=f"{response.provider}-{response.response_sha256[:24]}",
        raw_payload_sha256=response.body_sha256,
        payload_kind=PAYLOAD_PROVIDER_RESPONSE,
        availability_basis=AVAILABILITY_LIVE_RETRIEVAL,
        is_final=True,
    )


def _source_code_sha256() -> str:
    """Bind a receipt to every behavior-affecting helper in this module."""

    return _sha256_bytes(inspect.getsource(sys.modules[__name__]).encode("utf-8"))


class MassiveDailyTickerSummaryParser:
    parser_id = "massive.daily_ticker_summary"
    parser_version = "v1"

    @property
    def code_sha256(self) -> str:
        return _source_code_sha256()

    def parse(
        self,
        response: RawProviderResponse,
        config: ProviderParseConfig,
        *,
        ingested_at: Any,
    ) -> ParsedProviderBatch:
        if response.provider != "massive":
            raise ProviderEvidenceError("Massive parser requires a massive response")
        if not 200 <= response.status_code < 300:
            raise ProviderEvidenceError("Massive parser requires a successful HTTP response")
        if config.parser_id != self.parser_id:
            raise ProviderEvidenceError("Massive parser configuration has the wrong parser_id")
        _require_exact_identity(config.source_identity, _MASSIVE_IDENTITY)
        if len(config.requested_sessions) != 1 or response.requested_sessions != config.requested_sessions:
            raise ProviderEvidenceError("Massive Daily Ticker Summary requires exactly one matching session")
        _require_nyse_sessions(config.requested_sessions)
        if config.request_parameters != (("adjusted", "false"),):
            raise ProviderEvidenceError("Massive parser requires adjusted=false")
        if config.parser_options:
            raise ProviderEvidenceError("Massive parser does not support parser_options")
        if response.request_parameters != config.request_parameters:
            raise ProviderEvidenceError("Massive response parameters do not match parser configuration")
        session = config.requested_sessions[0]
        expected_path = f"/v1/open-close/SPY/{session.isoformat()}"
        endpoint = urlsplit(response.endpoint)
        if endpoint.hostname != "api.massive.com" or endpoint.path != expected_path:
            raise ProviderEvidenceError("Massive response endpoint does not match the requested session")
        if response.media_type not in {"application/json", "text/json"}:
            raise ProviderEvidenceError("Massive response must declare JSON content")
        try:
            decoded = json.loads(
                response.body.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderEvidenceError("Massive response body is not valid UTF-8 JSON") from exc
        if not isinstance(decoded, dict):
            raise ProviderEvidenceError("Massive response body must be a JSON object")
        if decoded.get("status") != "OK":
            raise ProviderEvidenceError("Massive response status must be OK")
        if decoded.get("symbol") != "SPY" or decoded.get("from") != session.isoformat():
            raise ProviderEvidenceError("Massive response symbol or session does not match the request")
        if "close" not in decoded:
            raise ProviderEvidenceError("Massive response has no regular-session close")
        raw_close = decoded["close"]
        if isinstance(raw_close, bool) or not isinstance(raw_close, (int, float)):
            raise ProviderEvidenceError("Massive close must be numeric")
        close = float(raw_close)
        if not math.isfinite(close) or close <= 0:
            raise ProviderEvidenceError("Massive close must be finite and positive")
        event_at = _require_final(response, session, "SPY")
        output = ParsedProviderOutput(
            observation=_observation(
                session=session,
                identity=_MASSIVE_IDENTITY,
                close=close,
                event_at=event_at,
                response=response,
                ingested_at=ingested_at,
            ),
            source_locator=ProviderSourceLocator("json_pointer", "/close"),
        )
        return ParsedProviderBatch(
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parser_code_sha256=self.code_sha256,
            config_sha256=config.config_sha256,
            response_sha256=response.response_sha256,
            outputs=(output,),
        )


class CboeVixDailyCsvParser:
    parser_id = "cboe.vix_daily_csv"
    parser_version = "v1"

    @property
    def code_sha256(self) -> str:
        return _source_code_sha256()

    def parse(
        self,
        response: RawProviderResponse,
        config: ProviderParseConfig,
        *,
        ingested_at: Any,
    ) -> ParsedProviderBatch:
        if response.provider != "cboe":
            raise ProviderEvidenceError("Cboe parser requires a cboe response")
        if not 200 <= response.status_code < 300:
            raise ProviderEvidenceError("Cboe parser requires a successful HTTP response")
        if config.parser_id != self.parser_id:
            raise ProviderEvidenceError("Cboe parser configuration has the wrong parser_id")
        _require_exact_identity(config.source_identity, _CBOE_IDENTITY)
        if response.requested_sessions != config.requested_sessions:
            raise ProviderEvidenceError("Cboe response sessions do not match parser configuration")
        _require_nyse_sessions(config.requested_sessions)
        if response.request_parameters != config.request_parameters:
            raise ProviderEvidenceError("Cboe response parameters do not match parser configuration")
        if config.request_parameters:
            raise ProviderEvidenceError("Cboe parser does not allow request parameters")
        if config.parser_options:
            raise ProviderEvidenceError("Cboe parser does not support parser_options")
        endpoint = urlsplit(response.endpoint)
        if (
            endpoint.hostname != "cdn.cboe.com"
            or endpoint.path != "/api/global/us_indices/daily_prices/VIX_History.csv"
        ):
            raise ProviderEvidenceError("Cboe response endpoint is not the official VIX history CSV")
        if response.media_type not in {"text/csv", "application/csv", "application/octet-stream"}:
            raise ProviderEvidenceError("Cboe response must declare CSV content")
        try:
            text = response.body.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames != ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
                raise ProviderEvidenceError(
                    "Cboe CSV header does not match the official VIX history schema"
                )
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ProviderEvidenceError("Cboe response body is not valid UTF-8 CSV") from exc
        if not rows:
            raise ProviderEvidenceError("Cboe response has no data rows")
        requested = set(config.requested_sessions)
        selected: dict[date, tuple[int, float]] = {}
        for ordinal, row in enumerate(rows, start=2):
            if None in row:
                raise ProviderEvidenceError(
                    f"Cboe CSV row {ordinal} has unexpected extra fields"
                )
            try:
                session = datetime.strptime(str(row["DATE"]).strip(), "%m/%d/%Y").date()
            except ValueError as exc:
                raise ProviderEvidenceError(f"Cboe CSV row {ordinal} has an invalid DATE") from exc
            if session not in requested:
                continue
            if session in selected:
                raise ProviderEvidenceError(f"Cboe CSV has a duplicate requested session: {session.isoformat()}")
            try:
                close = float(str(row["CLOSE"]).strip())
            except (TypeError, ValueError) as exc:
                raise ProviderEvidenceError(f"Cboe CSV row {ordinal} has a non-numeric CLOSE") from exc
            if not math.isfinite(close) or close <= 0:
                raise ProviderEvidenceError(f"Cboe CSV row {ordinal} has a non-positive or non-finite CLOSE")
            selected[session] = (ordinal, close)
        missing = [item.isoformat() for item in config.requested_sessions if item not in selected]
        if missing:
            raise ProviderEvidenceError(f"Cboe CSV is missing requested sessions: {missing}")
        outputs: list[ParsedProviderOutput] = []
        for session in config.requested_sessions:
            ordinal, close = selected[session]
            event_at = _require_final(response, session, "VIX")
            outputs.append(
                ParsedProviderOutput(
                    observation=_observation(
                        session=session,
                        identity=_CBOE_IDENTITY,
                        close=close,
                        event_at=event_at,
                        response=response,
                        ingested_at=ingested_at,
                    ),
                    source_locator=ProviderSourceLocator("csv_row", str(ordinal)),
                )
            )
        return ParsedProviderBatch(
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parser_code_sha256=self.code_sha256,
            config_sha256=config.config_sha256,
            response_sha256=response.response_sha256,
            outputs=tuple(outputs),
        )


MASSIVE_DAILY_TICKER_SUMMARY_PARSER = MassiveDailyTickerSummaryParser()
CBOE_VIX_DAILY_CSV_PARSER = CboeVixDailyCsvParser()
_REGISTERED_PARSERS: dict[tuple[str, str], ProviderParser] = {
    (
        MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parser_id,
        MASSIVE_DAILY_TICKER_SUMMARY_PARSER.parser_version,
    ): MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
    (
        CBOE_VIX_DAILY_CSV_PARSER.parser_id,
        CBOE_VIX_DAILY_CSV_PARSER.parser_version,
    ): CBOE_VIX_DAILY_CSV_PARSER,
}


def get_registered_provider_parser(parser_id: str, parser_version: str) -> ProviderParser:
    try:
        return _REGISTERED_PARSERS[(parser_id, parser_version)]
    except KeyError as exc:
        raise KeyError(f"No registered provider parser: {parser_id}@{parser_version}") from exc


@dataclass(frozen=True)
class ProviderParseJob:
    fetch_sha256: str
    parser: ProviderParser
    config: ProviderParseConfig

    def __post_init__(self) -> None:
        if not isinstance(self.fetch_sha256, str) or not _SHA256.fullmatch(self.fetch_sha256):
            raise ProviderEvidenceError("fetch_sha256 must be a SHA-256 digest")
        if not isinstance(self.config, ProviderParseConfig):
            raise TypeError("config must be a ProviderParseConfig")
        registered = get_registered_provider_parser(self.parser.parser_id, self.parser.parser_version)
        if self.parser is not registered:
            raise ProviderEvidenceError("parser must be a registered singleton")
        if self.config.parser_id != self.parser.parser_id:
            raise ProviderEvidenceError("parser config does not belong to the parser")


__all__ = [
    "CBOE_VIX_DAILY_CSV_PARSER",
    "MASSIVE_DAILY_TICKER_SUMMARY_PARSER",
    "CboeVixDailyCsvParser",
    "MassiveDailyTickerSummaryParser",
    "ParsedProviderBatch",
    "ParsedProviderOutput",
    "ProviderEvidenceError",
    "ProviderParseConfig",
    "ProviderParseJob",
    "ProviderParser",
    "ProviderSourceLocator",
    "RawProviderResponse",
    "get_registered_provider_parser",
]
