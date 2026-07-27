"""Typed, non-authoritative V2 regime signals.

This is the narrow compatibility boundary between the causal V2 detector and
read-only consumers such as dashboards and backtest annotations.  It has no
numeric regime state, posterior, return bucket, or order-facing projection.
In R4 every signal is advisory by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np

from live_trading.regime_detector_v2 import (
    BACKGROUND_CALM,
    BACKGROUND_ELEVATED,
    BACKGROUND_STRESS,
    BACKGROUND_UNAVAILABLE,
    DETECTOR_VERSION,
    SHOCK_ACTIVE,
    SHOCK_AFTERSHOCK,
    SHOCK_NONE,
    SHOCK_UNAVAILABLE,
    SIGNAL_TIMESTAMP,
    regime_detector_code_sha256,
)
from live_trading.regime_market_data import regime_market_schedule


REGIME_SIGNAL_SCHEMA_VERSION = "regime_signal.v1"
REGIME_SIGNAL_HASH_DOMAIN = b"regime-signal.v1\0"
R4_SHADOW_ABSTAIN_REASON = "r4_shadow_non_authoritative"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REASON = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
NYSE = mcal.get_calendar("NYSE")


class RegimeSignalError(ValueError):
    """Raised when a V2 trace cannot become a safe advisory signal."""


class RegimeActionProjectionError(RuntimeError):
    """Raised for any attempted use of a V2 shadow signal in order logic."""


class BackgroundState(str, Enum):
    UNAVAILABLE = BACKGROUND_UNAVAILABLE
    CALM = BACKGROUND_CALM
    ELEVATED = BACKGROUND_ELEVATED
    PERSISTENT_STRESS = BACKGROUND_STRESS


class ShockState(str, Enum):
    UNAVAILABLE = SHOCK_UNAVAILABLE
    NONE = SHOCK_NONE
    ACTIVE = SHOCK_ACTIVE
    AFTERSHOCK = SHOCK_AFTERSHOCK


class SignalSourceFamily(str, Enum):
    V2_SHADOW = "v2_shadow"


class SignalAvailability(str, Enum):
    ADVISORY = "advisory"
    UNAVAILABLE = "unavailable"


class ReturnBucketStatus(str, Enum):
    """R4 annotation consumers deliberately do not use return buckets."""

    NOT_USED_SHADOW_ANNOTATION = "not_used_shadow_annotation"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        REGIME_SIGNAL_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _date(value: Any, field_name: str) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None or value != value.normalize():
            raise RegimeSignalError(f"{field_name} must be a calendar date")
        value = value.date()
    elif isinstance(value, datetime):
        raise RegimeSignalError(f"{field_name} must be a calendar date")
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise RegimeSignalError(f"{field_name} must be an ISO date") from exc
    if not isinstance(value, date):
        raise RegimeSignalError(f"{field_name} must be a calendar date")
    return value


def _utc_datetime(value: Any, field_name: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise RegimeSignalError(f"{field_name} must be a timestamp") from exc
    if timestamp.tzinfo is None:
        raise RegimeSignalError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RegimeSignalError(f"{field_name} must be a SHA-256 digest")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RegimeSignalError(f"{field_name} must be a nonempty string or null")
    return value


def _reason_codes(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        values = value.split("|")
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        raise RegimeSignalError(f"{field_name} must be pipe-delimited text or a list")
    normalized = []
    for item in values:
        if not isinstance(item, str) or not _REASON.fullmatch(item):
            raise RegimeSignalError(f"{field_name} contains an invalid reason code")
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _is_nyse_session(session: date) -> bool:
    return not NYSE.schedule(start_date=session, end_date=session).empty


def next_tradable_session(session: date) -> date:
    """Return the exact next NYSE session after a close-dated signal."""

    session = _date(session, "signal_session")
    if not _is_nyse_session(session):
        raise RegimeSignalError("signal_session must be an NYSE session")
    schedule = NYSE.schedule(
        start_date=session,
        end_date=session + pd.Timedelta(days=14),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    following = sessions[sessions > pd.Timestamp(session)]
    if len(following) < 1:
        raise RegimeSignalError("could not resolve the next NYSE session")
    return following[0].date()


def _exact_row_value(row: pd.Series, column: str) -> Any:
    if column not in row.index:
        raise RegimeSignalError(f"V2 trace is missing required column: {column}")
    value = row[column]
    if value is None or pd.isna(value):
        raise RegimeSignalError(f"V2 trace has no value for required column: {column}")
    return value


def _exact_bool(row: pd.Series, column: str) -> bool:
    value = _exact_row_value(row, column)
    if not isinstance(value, (bool, np.bool_)):
        raise RegimeSignalError(f"V2 trace column must be boolean: {column}")
    return bool(value)


@dataclass(frozen=True)
class CausalRecord:
    """Causal facts required to prevent annotation-to-backtest leakage."""

    calibration_end_session: date | None
    retrospective_test_start: date | None
    retrospective_test_end: date | None
    inference_method: str | None
    signal_lag_sessions: int | None
    return_bucket_status: ReturnBucketStatus = (
        ReturnBucketStatus.NOT_USED_SHADOW_ANNOTATION
    )

    def __post_init__(self) -> None:
        for field_name in (
            "calibration_end_session",
            "retrospective_test_start",
            "retrospective_test_end",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = _date(value, field_name)
                if not _is_nyse_session(value):
                    raise RegimeSignalError(f"{field_name} must be an NYSE session")
            object.__setattr__(self, field_name, value)
        if (self.retrospective_test_start is None) != (self.retrospective_test_end is None):
            raise RegimeSignalError("retrospective test range must be complete or absent")
        if (
            self.retrospective_test_start is not None
            and self.retrospective_test_start > self.retrospective_test_end
        ):
            raise RegimeSignalError("retrospective test range is inverted")
        object.__setattr__(
            self,
            "inference_method",
            _optional_text(self.inference_method, "inference_method"),
        )
        if self.signal_lag_sessions is not None:
            if (
                isinstance(self.signal_lag_sessions, bool)
                or not isinstance(self.signal_lag_sessions, int)
                or self.signal_lag_sessions != 1
            ):
                raise RegimeSignalError("R4 signals require exactly one lagged session")
        try:
            object.__setattr__(
                self,
                "return_bucket_status",
                ReturnBucketStatus(self.return_bucket_status),
            )
        except ValueError as exc:
            raise RegimeSignalError("unsupported return bucket status") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_end_session": (
                None if self.calibration_end_session is None
                else self.calibration_end_session.isoformat()
            ),
            "retrospective_test_start": (
                None if self.retrospective_test_start is None
                else self.retrospective_test_start.isoformat()
            ),
            "retrospective_test_end": (
                None if self.retrospective_test_end is None
                else self.retrospective_test_end.isoformat()
            ),
            "inference_method": self.inference_method,
            "signal_lag_sessions": self.signal_lag_sessions,
            "return_bucket_status": self.return_bucket_status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalRecord":
        expected = {
            "calibration_end_session", "retrospective_test_start",
            "retrospective_test_end", "inference_method", "signal_lag_sessions",
            "return_bucket_status",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RegimeSignalError("causal record fields do not match the schema")
        return cls(
            calibration_end_session=payload["calibration_end_session"],
            retrospective_test_start=payload["retrospective_test_start"],
            retrospective_test_end=payload["retrospective_test_end"],
            inference_method=payload["inference_method"],
            signal_lag_sessions=payload["signal_lag_sessions"],
            return_bucket_status=ReturnBucketStatus(payload["return_bucket_status"]),
        )


@dataclass(frozen=True)
class RegimeLineage:
    """Immutable identifiers and causal record required to audit one V2 signal."""

    artifact_sha256: str | None
    plan_sha256: str | None
    config_sha256: str
    detector_version: str
    detector_code_sha256: str | None
    calibration_code_sha256: str | None
    calibration_profile: str | None
    calibration_status: str | None
    calibration_provenance: str | None
    snapshot_sha256: str | None
    source_provenance_status: str | None
    evidence_manifest_sha256: str | None
    evidence_verification_kind: str | None
    evidence_decision_time_eligible: bool
    calendar_schedule_sha256: str | None
    source_policy_sha256: str | None
    runtime_fingerprint_sha256: str | None
    causal_record: CausalRecord

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_sha256",
            "plan_sha256",
            "config_sha256",
            "detector_code_sha256",
            "calibration_code_sha256",
            "snapshot_sha256",
            "evidence_manifest_sha256",
            "calendar_schedule_sha256",
            "source_policy_sha256",
            "runtime_fingerprint_sha256",
        ):
            value = _sha256(getattr(self, field_name), field_name)
            if field_name == "config_sha256" and value is None:
                raise RegimeSignalError("config_sha256 is required")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "detector_version",
            "calibration_profile",
            "calibration_status",
            "calibration_provenance",
            "source_provenance_status",
            "evidence_verification_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )
        if self.detector_version is None:
            raise RegimeSignalError("detector_version is required")
        if self.source_provenance_status not in {
            None,
            "unverified",
            "verified",
        }:
            raise RegimeSignalError(
                "source_provenance_status is outside the closed contract"
            )
        if self.evidence_verification_kind not in {
            None,
            "decision_time",
            "verified_replay",
        }:
            raise RegimeSignalError(
                "evidence_verification_kind is outside the closed contract"
            )
        if not isinstance(self.evidence_decision_time_eligible, bool):
            raise RegimeSignalError(
                "evidence_decision_time_eligible must be boolean"
            )
        if self.evidence_manifest_sha256 is not None and (
            self.snapshot_sha256 is None
            or self.evidence_verification_kind is None
        ):
            raise RegimeSignalError(
                "evidence lineage requires a snapshot and verification kind"
            )
        if self.source_provenance_status == "verified" and (
            self.snapshot_sha256 is None
            or self.evidence_manifest_sha256 is None
        ):
            raise RegimeSignalError(
                "verified source provenance requires durable evidence"
            )
        if self.source_provenance_status == "unverified" and (
            self.evidence_manifest_sha256 is not None
            or self.evidence_verification_kind is not None
            or self.evidence_decision_time_eligible
        ):
            raise RegimeSignalError(
                "unverified source provenance cannot claim evidence"
            )
        if self.evidence_decision_time_eligible and (
            self.evidence_manifest_sha256 is None
            or self.evidence_verification_kind != "decision_time"
        ):
            raise RegimeSignalError(
                "decision-time eligibility requires decision-time evidence"
            )
        if not isinstance(self.causal_record, CausalRecord):
            raise TypeError("causal_record must be a CausalRecord")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "plan_sha256": self.plan_sha256,
            "config_sha256": self.config_sha256,
            "detector_version": self.detector_version,
            "detector_code_sha256": self.detector_code_sha256,
            "calibration_code_sha256": self.calibration_code_sha256,
            "calibration_profile": self.calibration_profile,
            "calibration_status": self.calibration_status,
            "calibration_provenance": self.calibration_provenance,
            "snapshot_sha256": self.snapshot_sha256,
            "source_provenance_status": self.source_provenance_status,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "evidence_verification_kind": self.evidence_verification_kind,
            "evidence_decision_time_eligible": (
                self.evidence_decision_time_eligible
            ),
            "calendar_schedule_sha256": self.calendar_schedule_sha256,
            "source_policy_sha256": self.source_policy_sha256,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "causal_record": self.causal_record.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeLineage":
        expected = {
            "artifact_sha256", "plan_sha256", "config_sha256",
            "detector_version", "detector_code_sha256", "calibration_code_sha256",
            "calibration_profile", "calibration_status", "calibration_provenance",
            "snapshot_sha256", "source_provenance_status",
            "evidence_manifest_sha256", "evidence_verification_kind",
            "evidence_decision_time_eligible", "calendar_schedule_sha256",
            "source_policy_sha256",
            "runtime_fingerprint_sha256", "causal_record",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RegimeSignalError("regime lineage fields do not match the schema")
        values = dict(payload)
        values["causal_record"] = CausalRecord.from_dict(values["causal_record"])
        return cls(**values)


@dataclass(frozen=True)
class RegimeSignal:
    """Lossless V2 background-plus-shock signal for advisory consumers only."""

    as_of_session: date
    available_at: datetime
    effective_session: date
    background_state: BackgroundState
    shock_state: ShockState
    availability: SignalAvailability
    signal_timestamp: str
    reason_codes: tuple[str, ...]
    abstain_reasons: tuple[str, ...]
    data_quality: tuple[str, ...]
    lineage: RegimeLineage
    schema_version: str = REGIME_SIGNAL_SCHEMA_VERSION
    source_family: SignalSourceFamily = SignalSourceFamily.V2_SHADOW

    def __post_init__(self) -> None:
        if self.schema_version != REGIME_SIGNAL_SCHEMA_VERSION:
            raise RegimeSignalError("unsupported regime signal schema")
        if self.source_family is not SignalSourceFamily.V2_SHADOW:
            raise RegimeSignalError("R4 only supports V2 shadow signals")
        session = _date(self.as_of_session, "as_of_session")
        effective = _date(self.effective_session, "effective_session")
        if not _is_nyse_session(session):
            raise RegimeSignalError("as_of_session must be an NYSE session")
        expected_effective = next_tradable_session(session)
        if effective != expected_effective:
            raise RegimeSignalError(
                "effective_session must equal the exact next NYSE tradable session"
            )
        object.__setattr__(self, "as_of_session", session)
        object.__setattr__(self, "effective_session", effective)
        object.__setattr__(self, "available_at", _utc_datetime(self.available_at, "available_at"))
        try:
            object.__setattr__(self, "background_state", BackgroundState(self.background_state))
            object.__setattr__(self, "shock_state", ShockState(self.shock_state))
            object.__setattr__(self, "availability", SignalAvailability(self.availability))
        except ValueError as exc:
            raise RegimeSignalError("regime state is outside the closed V2 taxonomy") from exc
        if self.signal_timestamp != SIGNAL_TIMESTAMP:
            raise RegimeSignalError("signal_timestamp does not match the V2 close policy")
        if not isinstance(self.lineage, RegimeLineage):
            raise TypeError("lineage must be a RegimeLineage")
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes, "reason_codes"))
        abstain = _reason_codes(self.abstain_reasons, "abstain_reasons")
        quality = _reason_codes(self.data_quality, "data_quality")
        if R4_SHADOW_ABSTAIN_REASON not in abstain:
            abstain = (*abstain, R4_SHADOW_ABSTAIN_REASON)
        object.__setattr__(self, "abstain_reasons", abstain)
        object.__setattr__(self, "data_quality", quality)
        expected_availability = (
            SignalAvailability.UNAVAILABLE
            if (
                self.background_state is BackgroundState.UNAVAILABLE
                or self.shock_state is ShockState.UNAVAILABLE
            )
            else SignalAvailability.ADVISORY
        )
        if self.availability is not expected_availability:
            raise RegimeSignalError("availability does not match V2 state availability")

    @property
    def may_authorize_execution(self) -> bool:
        """Hard R4 guardrail: advisory signals can never authorize execution."""

        return False

    @property
    def composite_label(self) -> str:
        return f"{self.background_state.value} + {self.shock_state.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_family": self.source_family.value,
            "as_of_session": self.as_of_session.isoformat(),
            "available_at": _utc_iso(self.available_at),
            "effective_session": self.effective_session.isoformat(),
            "background_state": self.background_state.value,
            "shock_state": self.shock_state.value,
            "availability": self.availability.value,
            "signal_timestamp": self.signal_timestamp,
            "reason_codes": list(self.reason_codes),
            "abstain_reasons": list(self.abstain_reasons),
            "data_quality": list(self.data_quality),
            "lineage": self.lineage.to_dict(),
            "may_authorize_execution": False,
        }

    @property
    def signal_sha256(self) -> str:
        return _hash(self.to_dict())

    def to_envelope(self) -> dict[str, Any]:
        """Return the primitive, tamper-evident persistence envelope."""

        return {
            "signal": self.to_dict(),
            "signal_sha256": self.signal_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_envelope())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeSignal":
        expected = {
            "schema_version", "source_family", "as_of_session", "available_at",
            "effective_session", "background_state", "shock_state", "availability",
            "signal_timestamp", "reason_codes", "abstain_reasons", "data_quality",
            "lineage", "may_authorize_execution",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise RegimeSignalError("regime signal fields do not match the schema")
        if payload["may_authorize_execution"] is not False:
            raise RegimeSignalError("R4 regime signals cannot authorize execution")
        return cls(
            schema_version=payload["schema_version"],
            source_family=SignalSourceFamily(payload["source_family"]),
            as_of_session=payload["as_of_session"],
            available_at=payload["available_at"],
            effective_session=payload["effective_session"],
            background_state=BackgroundState(payload["background_state"]),
            shock_state=ShockState(payload["shock_state"]),
            availability=SignalAvailability(payload["availability"]),
            signal_timestamp=payload["signal_timestamp"],
            reason_codes=tuple(payload["reason_codes"]),
            abstain_reasons=tuple(payload["abstain_reasons"]),
            data_quality=tuple(payload["data_quality"]),
            lineage=RegimeLineage.from_dict(payload["lineage"]),
        )

    @classmethod
    def from_envelope(
        cls,
        envelope: Mapping[str, Any],
    ) -> "RegimeSignal":
        if not isinstance(envelope, Mapping) or set(envelope) != {"signal", "signal_sha256"}:
            raise RegimeSignalError("regime signal envelope fields do not match the schema")
        signal = cls.from_dict(envelope["signal"])
        if not isinstance(envelope["signal_sha256"], str) or signal.signal_sha256 != envelope["signal_sha256"]:
            raise RegimeSignalError("regime signal hash does not match")
        return signal

    @classmethod
    def from_json(cls, payload: str) -> "RegimeSignal":
        if not isinstance(payload, str):
            raise TypeError("regime signal JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RegimeSignalError("regime signal is not valid JSON") from exc
        if _canonical_json(envelope) != payload:
            raise RegimeSignalError("regime signal JSON must be canonical")
        return cls.from_envelope(envelope)

    def to_dashboard_payload(self) -> dict[str, Any]:
        """Return an explicit V2-only payload with no action-facing fields."""

        return {
            "available": self.availability is SignalAvailability.ADVISORY,
            "source_family": self.source_family.value,
            "schema_version": self.schema_version,
            "as_of_session": self.as_of_session.isoformat(),
            "effective_session": self.effective_session.isoformat(),
            "available_at": _utc_iso(self.available_at),
            "background_state": self.background_state.value,
            "shock_state": self.shock_state.value,
            "composite_label": self.composite_label,
            "reason_codes": list(self.reason_codes),
            "abstain_reasons": list(self.abstain_reasons),
            "data_quality": list(self.data_quality),
            "lineage": self.lineage.to_dict(),
            "may_authorize_execution": False,
            "signal_sha256": self.signal_sha256,
        }

    def to_action_projection(self) -> None:
        raise RegimeActionProjectionError(
            "V2 shadow signals have no action projection in R4"
        )


def _trace_lineage(trace: pd.DataFrame, row: pd.Series, artifact: Any | None) -> RegimeLineage:
    def optional_column(name: str) -> Any:
        if name not in row.index or pd.isna(row[name]):
            return None
        return row[name]

    detector_code = trace.attrs.get("detector_code_sha256")
    runtime_fingerprint = trace.attrs.get("runtime_fingerprint_sha256")
    lineage = RegimeLineage(
        artifact_sha256=optional_column("Calibration_Artifact_SHA256"),
        plan_sha256=optional_column("Calibration_Plan_SHA256"),
        config_sha256=str(_exact_row_value(row, "Config_Hash")),
        detector_version=str(_exact_row_value(row, "Detector_Version")),
        detector_code_sha256=detector_code,
        calibration_code_sha256=None,
        calibration_profile=optional_column("Calibration_Profile"),
        calibration_status=optional_column("Calibration_Status"),
        calibration_provenance=optional_column("Calibration_Provenance"),
        snapshot_sha256=optional_column("Input_Snapshot_SHA256"),
        source_provenance_status=optional_column("Input_Provenance_Status"),
        evidence_manifest_sha256=optional_column(
            "Evidence_Manifest_SHA256"
        ),
        evidence_verification_kind=optional_column(
            "Evidence_Verification_Kind"
        ),
        evidence_decision_time_eligible=(
            _exact_bool(row, "Evidence_Decision_Time_Eligible")
            if "Evidence_Decision_Time_Eligible" in row.index
            else False
        ),
        calendar_schedule_sha256=optional_column(
            "Calendar_Schedule_SHA256"
        ),
        source_policy_sha256=optional_column("Source_Policy_SHA256"),
        runtime_fingerprint_sha256=runtime_fingerprint,
        causal_record=CausalRecord(
            calibration_end_session=None,
            retrospective_test_start=None,
            retrospective_test_end=None,
            inference_method=None,
            signal_lag_sessions=None,
        ),
    )
    attr_version = trace.attrs.get("detector_version")
    attr_config = trace.attrs.get("config_hash")
    if attr_version is not None and attr_version != lineage.detector_version:
        raise RegimeSignalError("trace detector version does not match its rows")
    if attr_config is not None and attr_config != lineage.config_sha256:
        raise RegimeSignalError("trace config hash does not match its rows")
    for attr_name, expected in (
        ("input_snapshot_sha256", lineage.snapshot_sha256),
        (
            "calendar_schedule_sha256",
            lineage.calendar_schedule_sha256,
        ),
        ("source_policy_sha256", lineage.source_policy_sha256),
        (
            "evidence_manifest_sha256",
            lineage.evidence_manifest_sha256,
        ),
        (
            "evidence_verification_kind",
            lineage.evidence_verification_kind,
        ),
        (
            "evidence_decision_time_eligible",
            lineage.evidence_decision_time_eligible,
        ),
    ):
        actual = trace.attrs.get(attr_name)
        if actual is not None and actual != expected:
            raise RegimeSignalError(
                f"trace {attr_name} does not match its rows"
            )
    source_verified = trace.attrs.get("source_provenance_verified")
    if source_verified is not None:
        if not isinstance(source_verified, (bool, np.bool_)):
            raise RegimeSignalError(
                "trace source_provenance_verified must be boolean"
            )
        expected_verified = (
            lineage.source_provenance_status == "verified"
        )
        if bool(source_verified) != expected_verified:
            raise RegimeSignalError(
                "trace source provenance does not match its rows"
            )
    if lineage.detector_version != DETECTOR_VERSION:
        raise RegimeSignalError("trace detector version is stale")
    if (
        lineage.detector_code_sha256 is not None
        and lineage.detector_code_sha256 != regime_detector_code_sha256()
    ):
        raise RegimeSignalError("trace detector code hash is stale")

    if artifact is None:
        return lineage

    from live_trading.regime_calibration import (
        RegimeCalibrationArtifact,
        regime_calibration_code_sha256,
    )

    if not isinstance(artifact, RegimeCalibrationArtifact):
        raise TypeError("artifact must be a RegimeCalibrationArtifact")
    for attr_name, expected in (
        ("calibration_artifact_sha256", lineage.artifact_sha256),
        ("calibration_plan_sha256", lineage.plan_sha256),
        ("calibration_profile", lineage.calibration_profile),
        ("calibration_status", lineage.calibration_status),
        ("calibration_provenance", lineage.calibration_provenance),
    ):
        actual = trace.attrs.get(attr_name)
        if actual is not None and actual != expected:
            raise RegimeSignalError(f"trace {attr_name} does not match its rows")
    if artifact.execution_eligible is not False:
        raise RegimeSignalError("calibration artifact cannot authorize execution")
    if artifact.plan.detector_version != DETECTOR_VERSION:
        raise RegimeSignalError("calibration artifact detector version is stale")
    if artifact.plan.detector_code_sha256 != regime_detector_code_sha256():
        raise RegimeSignalError("calibration artifact detector code hash is stale")
    if artifact.plan.calibration_code_sha256 != regime_calibration_code_sha256():
        raise RegimeSignalError("calibration artifact calibration code hash is stale")
    if lineage.artifact_sha256 != artifact.artifact_sha256:
        raise RegimeSignalError("trace artifact hash does not match the artifact")
    if lineage.plan_sha256 != artifact.plan.plan_sha256:
        raise RegimeSignalError("trace plan hash does not match the artifact")
    if lineage.config_sha256 != artifact.selected_candidate.config_sha256:
        raise RegimeSignalError("trace config hash does not match the artifact")
    if lineage.calibration_profile != artifact.selected_candidate_id:
        raise RegimeSignalError("trace calibration profile does not match the artifact")
    if lineage.calibration_status != artifact.promotion_status:
        raise RegimeSignalError("trace calibration status does not match the artifact")
    if lineage.calibration_provenance != artifact.data_manifest.provenance_status:
        raise RegimeSignalError("trace calibration provenance does not match the artifact")
    if (
        lineage.detector_code_sha256 is not None
        and lineage.detector_code_sha256 != artifact.plan.detector_code_sha256
    ):
        raise RegimeSignalError("trace detector code hash does not match the artifact")
    return RegimeLineage(
        **{
            **lineage.to_dict(),
            "detector_code_sha256": artifact.plan.detector_code_sha256,
            "calibration_code_sha256": artifact.plan.calibration_code_sha256,
            "causal_record": CausalRecord(
                calibration_end_session=(
                    artifact.plan.selection_folds[-1].evaluation_end
                ),
                retrospective_test_start=(
                    artifact.plan.retrospective_test.evaluation_start
                ),
                retrospective_test_end=(
                    artifact.plan.retrospective_test.evaluation_end
                ),
                inference_method=artifact.plan.inference_method,
                signal_lag_sessions=artifact.plan.signal_lag_sessions,
            ),
        }
    )


def _trace_signal(trace: pd.DataFrame, session: date, row: pd.Series, artifact: Any | None) -> RegimeSignal:
    background = BackgroundState(str(_exact_row_value(row, "Background_State")))
    shock = ShockState(str(_exact_row_value(row, "Shock_State")))
    if "Composite_Regime" in row.index:
        expected_composite = f"{background.value}+{shock.value}"
        if _exact_row_value(row, "Composite_Regime") != expected_composite:
            raise RegimeSignalError(
                "trace composite regime does not match its two axes"
            )
    if _exact_row_value(row, "Regime_Signal_Timestamp") != SIGNAL_TIMESTAMP:
        raise RegimeSignalError("trace signal timestamp does not match the V2 close policy")
    if _exact_bool(row, "Execution_Eligible"):
        raise RegimeSignalError("V2 trace must remain execution-ineligible")
    if "Calibration_Execution_Eligible" in row.index:
        if _exact_bool(row, "Calibration_Execution_Eligible"):
            raise RegimeSignalError("calibrated V2 trace must remain execution-ineligible")
    if "Calibration_Abstain" in row.index:
        if not _exact_bool(row, "Calibration_Abstain"):
            raise RegimeSignalError("calibrated V2 trace must explicitly abstain")
    effective = _date(_exact_row_value(row, "Tradable_Session"), "Tradable_Session")
    expected = next_tradable_session(session)
    if effective != expected:
        raise RegimeSignalError("Tradable_Session is not the exact next NYSE session")
    available_at = _utc_datetime(
        _exact_row_value(row, "Signal_Available_At"),
        "Signal_Available_At",
    )
    finalization = pd.Timestamp(
        regime_market_schedule(session, session).iloc[0]["joint_finalization_at"]
    ).to_pydatetime()
    if available_at < finalization:
        raise RegimeSignalError("Signal_Available_At precedes joint close finalization")
    next_open = pd.Timestamp(
        NYSE.schedule(start_date=effective, end_date=effective).iloc[0]["market_open"]
    ).to_pydatetime()
    if available_at >= next_open:
        raise RegimeSignalError("Signal_Available_At is too late for the next session")
    abstain = list(_reason_codes(row.get("Calibration_Abstain_Reasons"), "Calibration_Abstain_Reasons"))
    if artifact is None:
        missing_reason = (
            "calibration_artifact_unverified"
            if "Calibration_Artifact_SHA256" in row.index
            else "calibration_lineage_missing"
        )
        if missing_reason not in abstain:
            abstain.append(missing_reason)
    if "execution_ineligible" not in abstain:
        abstain.append("execution_ineligible")
    return RegimeSignal(
        as_of_session=session,
        available_at=available_at,
        effective_session=effective,
        background_state=background,
        shock_state=shock,
        availability=(
            SignalAvailability.UNAVAILABLE
            if background is BackgroundState.UNAVAILABLE or shock is ShockState.UNAVAILABLE
            else SignalAvailability.ADVISORY
        ),
        signal_timestamp=SIGNAL_TIMESTAMP,
        reason_codes=_reason_codes(_exact_row_value(row, "Reason_Codes"), "Reason_Codes"),
        abstain_reasons=tuple(abstain),
        data_quality=_reason_codes(_exact_row_value(row, "Data_Quality"), "Data_Quality"),
        lineage=_trace_lineage(trace, row, artifact),
    )


def from_calibrated_v2_trace(
    trace: pd.DataFrame,
    *,
    artifact: Any | None = None,
    expected_artifact_sha256: str | None = None,
) -> dict[str, RegimeSignal]:
    """Adapt an authentic V2 trace into exact effective-session annotations.

    The returned dictionary is keyed only by the trace's explicit
    ``Tradable_Session`` values.  It does not forward-fill, infer, or map a
    V2 state into an HMM integer.  Any schema, lineage, or timing discrepancy
    raises :class:`RegimeSignalError` rather than producing an action-capable
    fallback.
    """

    if not isinstance(trace, pd.DataFrame) or trace.empty:
        raise RegimeSignalError("V2 trace must be a nonempty pandas DataFrame")
    if artifact is None and expected_artifact_sha256 is not None:
        raise RegimeSignalError(
            "expected_artifact_sha256 requires a calibration artifact"
        )
    if artifact is not None:
        from live_trading.regime_calibration import RegimeCalibrationArtifact

        if not isinstance(artifact, RegimeCalibrationArtifact):
            raise TypeError("artifact must be a RegimeCalibrationArtifact")
        expected_hash = _sha256(
            expected_artifact_sha256,
            "expected_artifact_sha256",
        )
        if expected_hash is None:
            raise RegimeSignalError(
                "calibrated V2 traces require a deployment-pinned artifact hash"
            )
        if artifact.artifact_sha256 != expected_hash:
            raise RegimeSignalError(
                "calibration artifact does not match the deployment pin"
            )
        required_calibration_columns = {
            "Calibration_Artifact_SHA256",
            "Calibration_Plan_SHA256",
            "Calibration_Profile",
            "Calibration_Status",
            "Calibration_Provenance",
            "Calibration_Abstain",
            "Calibration_Abstain_Reasons",
            "Calibration_Execution_Eligible",
        }
        missing = sorted(required_calibration_columns.difference(trace.columns))
        if missing:
            raise RegimeSignalError(
                "calibrated V2 trace is missing lineage columns: "
                f"{missing}"
            )
        required_attrs = {
            "detector_version",
            "config_hash",
            "detector_code_sha256",
            "runtime_fingerprint_sha256",
            "calibration_artifact_sha256",
            "calibration_plan_sha256",
            "calibration_profile",
            "calibration_status",
            "calibration_provenance",
            "calibration_execution_eligible",
            "calibration_abstain",
            "execution_eligible",
        }
        missing_attrs = sorted(required_attrs.difference(trace.attrs))
        if missing_attrs:
            raise RegimeSignalError(
                "calibrated V2 trace is missing immutable attributes: "
                f"{missing_attrs}"
            )
        if (
            trace.attrs["calibration_execution_eligible"] is not False
            or trace.attrs["calibration_abstain"] is not True
            or trace.attrs["execution_eligible"] is not False
        ):
            raise RegimeSignalError(
                "calibrated V2 trace must remain abstaining and "
                "execution-ineligible"
            )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(trace.index, errors="raise"))
    except Exception as exc:
        raise RegimeSignalError("V2 trace index must contain session dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RegimeSignalError("V2 trace sessions must be unique and sorted")
    if any(not _is_nyse_session(item.date()) for item in index):
        raise RegimeSignalError("V2 trace sessions must be NYSE sessions")

    frame = trace.copy()
    frame.index = index
    annotations: dict[str, RegimeSignal] = {}
    for timestamp, row in frame.iterrows():
        signal = _trace_signal(frame, timestamp.date(), row, artifact)
        key = signal.effective_session.isoformat()
        if key in annotations:
            raise RegimeSignalError("V2 trace produced duplicate effective sessions")
        annotations[key] = signal
    return annotations


def annotation_for_session(
    annotations: Mapping[str, RegimeSignal],
    session: date | str,
) -> RegimeSignal | None:
    """Return an annotation only for its exact effective date; never fill."""

    if not isinstance(annotations, Mapping):
        raise TypeError("annotations must be a mapping")
    target = _date(session, "session").isoformat()
    signal = annotations.get(target)
    if signal is None:
        return None
    if not isinstance(signal, RegimeSignal):
        raise TypeError("annotation mapping contains a non-RegimeSignal value")
    if signal.effective_session.isoformat() != target:
        raise RegimeSignalError("annotation key does not match effective_session")
    return signal


def to_dashboard_payload(signal: RegimeSignal) -> dict[str, Any]:
    if not isinstance(signal, RegimeSignal):
        raise TypeError("signal must be a RegimeSignal")
    return signal.to_dashboard_payload()


def to_action_projection(signal: RegimeSignal) -> None:
    if not isinstance(signal, RegimeSignal):
        raise TypeError("signal must be a RegimeSignal")
    return signal.to_action_projection()


__all__ = [
    "BackgroundState",
    "CausalRecord",
    "R4_SHADOW_ABSTAIN_REASON",
    "REGIME_SIGNAL_SCHEMA_VERSION",
    "RegimeActionProjectionError",
    "RegimeLineage",
    "RegimeSignal",
    "RegimeSignalError",
    "ReturnBucketStatus",
    "ShockState",
    "SignalAvailability",
    "SignalSourceFamily",
    "annotation_for_session",
    "from_calibrated_v2_trace",
    "next_tradable_session",
    "to_action_projection",
    "to_dashboard_payload",
]
