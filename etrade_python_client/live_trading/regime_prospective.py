"""Immutable prospective-review contracts for Regime V2.

This module is intentionally disconnected from provider transports, brokerage
accounts, order gateways, and execution ledgers.  It defines only:

* a frozen prospective-review protocol;
* durable entitlement and evidence receipts;
* content-addressed journal records;
* a structural promotion gate whose PASS status cannot authorize execution.

The statistical review remains an independent, immutable report.  This module
verifies that the report is bound to the exact frozen evidence; it does not
silently recompute or retune the detector.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import pandas as pd
import pandas_market_calendars as mcal

from live_trading.regime_market_data import regime_market_schedule
from live_trading.regime_signal import (
    BackgroundState,
    RegimeSignal,
    RegimeSignalError,
    ShockState,
    SignalAvailability,
)


PROSPECTIVE_PROTOCOL_SCHEMA_VERSION = "regime_prospective_review.v1"
ENTITLEMENT_RECEIPT_SCHEMA_VERSION = "regime_entitlement_receipt.v1"
JOURNAL_ENTRY_SCHEMA_VERSION = "regime_prospective_journal_entry.v1"
OUTCOME_RECORD_SCHEMA_VERSION = "regime_prospective_outcome.v1"
CALIBRATION_EVIDENCE_SCHEMA_VERSION = "regime_verified_calibration.v1"
REVIEW_RECEIPT_SCHEMA_VERSION = "regime_prospective_review_receipt.v1"

MISSING_SIGNAL_RULE = "block_if_any_expected_session_missing"
MISSING_OUTCOME_RULE = "incomplete_until_resolution_then_block"
LATE_BACKFILL_RULE = "forbidden_after_effective_session_open"
STATISTICAL_REVIEW_RULE = "independent_content_addressed_report_required"
ACTIVATION_BLOCKED = "blocked_preregistration"
ACTIVATION_ACTIVE = "activated_before_holdout"

# Updated only after an operator reviews and commits a canonical protocol.
# The currently pinned protocol is a blocked preregistration, not an active
# prospective study.
DEPLOYMENT_PROTOCOL_SHA256 = (
    "78f3480f6b803681be9e29f38b75c4823168cfa64e8d8400922a62b01c73aad3"
)

# R6 defines structural evidence contracts but does not yet authenticate
# external issuers/reviewers or execute the frozen statistical decision rule.
# Keeping this false makes self-attested PASS impossible.
PROMOTION_AUTHENTICATION_IMPLEMENTED = False

_PROTOCOL_HASH_DOMAIN = b"regime-prospective-protocol.v1\0"
_ENTITLEMENT_HASH_DOMAIN = b"regime-entitlement-receipt.v1\0"
_JOURNAL_HASH_DOMAIN = b"regime-prospective-journal-entry.v1\0"
_OUTCOME_HASH_DOMAIN = b"regime-prospective-outcome.v1\0"
_OUTCOME_SET_HASH_DOMAIN = b"regime-prospective-outcome-set.v1\0"
_CALIBRATION_HASH_DOMAIN = b"regime-verified-calibration.v1\0"
_REVIEW_HASH_DOMAIN = b"regime-prospective-review-receipt.v1\0"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,255}$")
NYSE = mcal.get_calendar("NYSE")


class ProspectiveReviewError(ValueError):
    """Raised when prospective evidence violates the frozen contract."""


class PromotionGateStatus(str, Enum):
    """Closed promotion result; even PASS is not an execution permission."""

    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    PASS = "PASS"

    @property
    def may_authorize_execution(self) -> bool:
        return False


class EntitlementStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class ReviewVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _domain_hash(domain: bytes, payload: Any) -> str:
    return hashlib.sha256(
        domain + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _strict_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ProspectiveReviewError(
            f"{label} fields do not match the schema"
        )


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProspectiveReviewError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field_name)


def _token(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ProspectiveReviewError(f"{field_name} is not a valid token")
    return value


def _tokens(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ProspectiveReviewError(
            f"{field_name} must be an iterable of tokens"
        )
    try:
        result = tuple(_token(item, field_name) for item in values)
    except TypeError as exc:
        raise ProspectiveReviewError(
            f"{field_name} must be an iterable of tokens"
        ) from exc
    if not result or result != tuple(sorted(set(result))):
        raise ProspectiveReviewError(
            f"{field_name} must be nonempty, sorted, and unique"
        )
    return result


def _optional_tokens(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ProspectiveReviewError(
            f"{field_name} must be an iterable of tokens"
        )
    try:
        result = tuple(_token(item, field_name) for item in values)
    except TypeError as exc:
        raise ProspectiveReviewError(
            f"{field_name} must be an iterable of tokens"
        ) from exc
    if result != tuple(sorted(set(result))):
        raise ProspectiveReviewError(
            f"{field_name} must be sorted and unique"
        )
    return result


def _calendar_date(value: Any, field_name: str) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None or value != value.normalize():
            raise ProspectiveReviewError(
                f"{field_name} must be a calendar date"
            )
        value = value.date()
    elif isinstance(value, datetime):
        raise ProspectiveReviewError(f"{field_name} must be a calendar date")
    elif isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise ProspectiveReviewError(
                f"{field_name} must be an ISO date"
            ) from exc
    if not isinstance(value, date):
        raise ProspectiveReviewError(f"{field_name} must be a calendar date")
    return value


def _market_session(value: Any, field_name: str) -> date:
    session = _calendar_date(value, field_name)
    if NYSE.schedule(start_date=session, end_date=session).empty:
        raise ProspectiveReviewError(f"{field_name} must be an NYSE session")
    return session


def _utc_datetime(value: Any, field_name: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ProspectiveReviewError(
            f"{field_name} must be a timestamp"
        ) from exc
    if timestamp.tzinfo is None:
        raise ProspectiveReviewError(
            f"{field_name} must be timezone-aware"
        )
    return timestamp.tz_convert("UTC").to_pydatetime()


def _optional_utc_datetime(
    value: Any,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _utc_datetime(value, field_name)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _trusted_utc_now() -> datetime:
    """Production-owned review clock; patched only in isolated unit tests."""

    return datetime.now(timezone.utc)


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProspectiveReviewError(
            f"{field_name} must be a positive integer"
        )
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProspectiveReviewError(
            f"{field_name} must be a nonnegative integer"
        )
    return value


def _market_sessions(start: date, end: date) -> tuple[date, ...]:
    schedule = NYSE.schedule(start_date=start, end_date=end)
    return tuple(pd.Timestamp(item).date() for item in schedule.index)


def _shift_market_session(session: date, count: int) -> date:
    count = _positive_integer(count, "session_shift")
    end = session + pd.Timedelta(days=7 * count + 21)
    sessions = _market_sessions(session, end)
    following = tuple(item for item in sessions if item > session)
    if len(following) < count:
        raise ProspectiveReviewError(
            "could not resolve the required future NYSE session"
        )
    return following[count - 1]


def _session_finalization(session: date) -> datetime:
    schedule = regime_market_schedule(session, session)
    return pd.Timestamp(
        schedule.iloc[0]["joint_finalization_at"]
    ).to_pydatetime()


def _session_open(session: date) -> datetime:
    schedule = NYSE.schedule(start_date=session, end_date=session)
    return pd.Timestamp(schedule.iloc[0]["market_open"]).to_pydatetime()


@dataclass(frozen=True)
class ProspectiveReviewProtocol:
    """Immutable pre-registration for the untouched prospective window."""

    protocol_id: str
    calibration_plan_sha256: str
    research_calibration_artifact_sha256: str
    selected_config_sha256: str
    detector_version: str
    detector_code_sha256: str
    calibration_code_sha256: str
    runtime_fingerprint_sha256: str
    holdout_start_session: date
    observation_end_session: date
    final_outcome_resolution_session: date
    review_not_before_session: date
    activation_deadline: datetime
    activation_status: str
    activated_at: datetime | None
    activation_receipt_sha256: str | None
    activation_reason_codes: tuple[str, ...]
    outcome_horizons: tuple[int, ...]
    signal_lag_sessions: int
    minimum_signal_rows: int
    minimum_resolved_rows: int
    minimum_outcome_coverage: float
    maximum_missing_signal_sessions: int
    required_entitlement_scopes: tuple[str, ...]
    maximum_entitlement_check_age_seconds: int
    missing_signal_rule: str = MISSING_SIGNAL_RULE
    missing_outcome_rule: str = MISSING_OUTCOME_RULE
    late_backfill_rule: str = LATE_BACKFILL_RULE
    statistical_review_rule: str = STATISTICAL_REVIEW_RULE
    no_retuning: bool = True
    execution_eligible: bool = False
    schema_version: str = PROSPECTIVE_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROSPECTIVE_PROTOCOL_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported prospective protocol schema"
            )
        object.__setattr__(
            self,
            "protocol_id",
            _token(self.protocol_id, "protocol_id"),
        )
        for field_name in (
            "calibration_plan_sha256",
            "research_calibration_artifact_sha256",
            "selected_config_sha256",
            "detector_code_sha256",
            "calibration_code_sha256",
            "runtime_fingerprint_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "detector_version",
            _token(self.detector_version, "detector_version"),
        )
        start = _market_session(
            self.holdout_start_session,
            "holdout_start_session",
        )
        end = _market_session(
            self.observation_end_session,
            "observation_end_session",
        )
        resolution = _market_session(
            self.final_outcome_resolution_session,
            "final_outcome_resolution_session",
        )
        review = _market_session(
            self.review_not_before_session,
            "review_not_before_session",
        )
        if not start <= end < resolution or review < resolution:
            raise ProspectiveReviewError(
                "prospective window dates are inconsistent"
            )
        activation_deadline = _utc_datetime(
            self.activation_deadline,
            "activation_deadline",
        )
        if activation_deadline >= _session_open(start):
            raise ProspectiveReviewError(
                "activation deadline must precede the first holdout open"
            )
        if self.activation_status not in {
            ACTIVATION_BLOCKED,
            ACTIVATION_ACTIVE,
        }:
            raise ProspectiveReviewError(
                "unsupported prospective activation status"
            )
        activation_receipt = _optional_sha256(
            self.activation_receipt_sha256,
            "activation_receipt_sha256",
        )
        activated_at = _optional_utc_datetime(
            self.activated_at,
            "activated_at",
        )
        activation_reasons = _optional_tokens(
            self.activation_reason_codes,
            "activation_reason_codes",
        )
        if self.activation_status == ACTIVATION_ACTIVE:
            if (
                activation_receipt is None
                or activated_at is None
                or activated_at > activation_deadline
                or activation_reasons
            ):
                raise ProspectiveReviewError(
                    "active study requires timely receipt and no block reasons"
                )
        elif (
            activation_receipt is not None
            or activated_at is not None
            or not activation_reasons
        ):
            raise ProspectiveReviewError(
                "blocked preregistration requires explicit block reasons"
            )
        if isinstance(self.outcome_horizons, (str, bytes)):
            raise ProspectiveReviewError(
                "outcome_horizons must be iterable"
            )
        try:
            horizons = tuple(self.outcome_horizons)
        except TypeError as exc:
            raise ProspectiveReviewError(
                "outcome_horizons must be iterable"
            ) from exc
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons)))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 1
                for item in horizons
            )
        ):
            raise ProspectiveReviewError(
                "outcome_horizons must be sorted unique positive integers"
            )
        lag = _positive_integer(
            self.signal_lag_sessions,
            "signal_lag_sessions",
        )
        if lag != 1:
            raise ProspectiveReviewError(
                "prospective review requires exactly one lagged session"
            )
        expected_resolution = _shift_market_session(
            end,
            lag + max(horizons),
        )
        if resolution != expected_resolution:
            raise ProspectiveReviewError(
                "final outcome resolution does not match the frozen horizon"
            )
        minimum_rows = _positive_integer(
            self.minimum_signal_rows,
            "minimum_signal_rows",
        )
        resolved_rows = _positive_integer(
            self.minimum_resolved_rows,
            "minimum_resolved_rows",
        )
        expected_rows = len(_market_sessions(start, end))
        if minimum_rows > expected_rows or resolved_rows > minimum_rows:
            raise ProspectiveReviewError(
                "row requirements exceed the frozen observation window"
            )
        missing = _nonnegative_integer(
            self.maximum_missing_signal_sessions,
            "maximum_missing_signal_sessions",
        )
        if missing > expected_rows - minimum_rows:
            raise ProspectiveReviewError(
                "missing-session allowance conflicts with minimum rows"
            )
        coverage = self.minimum_outcome_coverage
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(coverage)
            or not 0.0 < coverage <= 1.0
        ):
            raise ProspectiveReviewError(
                "minimum_outcome_coverage must be in (0, 1]"
            )
        check_age = _positive_integer(
            self.maximum_entitlement_check_age_seconds,
            "maximum_entitlement_check_age_seconds",
        )
        fixed_rules = {
            "missing_signal_rule": MISSING_SIGNAL_RULE,
            "missing_outcome_rule": MISSING_OUTCOME_RULE,
            "late_backfill_rule": LATE_BACKFILL_RULE,
            "statistical_review_rule": STATISTICAL_REVIEW_RULE,
        }
        for field_name, expected in fixed_rules.items():
            if getattr(self, field_name) != expected:
                raise ProspectiveReviewError(
                    f"unsupported {field_name}"
                )
        if self.no_retuning is not True:
            raise ProspectiveReviewError(
                "the prospective protocol must covenant no retuning"
            )
        if self.execution_eligible is not False:
            raise ProspectiveReviewError(
                "the prospective protocol cannot authorize execution"
            )
        object.__setattr__(self, "holdout_start_session", start)
        object.__setattr__(self, "observation_end_session", end)
        object.__setattr__(
            self,
            "final_outcome_resolution_session",
            resolution,
        )
        object.__setattr__(self, "review_not_before_session", review)
        object.__setattr__(
            self,
            "activation_deadline",
            activation_deadline,
        )
        object.__setattr__(
            self,
            "activation_receipt_sha256",
            activation_receipt,
        )
        object.__setattr__(self, "activated_at", activated_at)
        object.__setattr__(
            self,
            "activation_reason_codes",
            activation_reasons,
        )
        object.__setattr__(self, "outcome_horizons", horizons)
        object.__setattr__(self, "signal_lag_sessions", lag)
        object.__setattr__(self, "minimum_signal_rows", minimum_rows)
        object.__setattr__(self, "minimum_resolved_rows", resolved_rows)
        object.__setattr__(
            self,
            "minimum_outcome_coverage",
            float(coverage),
        )
        object.__setattr__(
            self,
            "maximum_missing_signal_sessions",
            missing,
        )
        object.__setattr__(
            self,
            "required_entitlement_scopes",
            _tokens(
                self.required_entitlement_scopes,
                "required_entitlement_scopes",
            ),
        )
        object.__setattr__(
            self,
            "maximum_entitlement_check_age_seconds",
            check_age,
        )

    @property
    def expected_signal_sessions(self) -> tuple[date, ...]:
        return _market_sessions(
            self.holdout_start_session,
            self.observation_end_session,
        )

    @property
    def accumulation_eligible(self) -> bool:
        return self.activation_status == ACTIVATION_ACTIVE

    def resolution_session_for(
        self,
        signal_session: date | str,
        horizon: int,
    ) -> date:
        session = _market_session(signal_session, "signal_session")
        if horizon not in self.outcome_horizons:
            raise ProspectiveReviewError(
                "horizon is not in the frozen protocol"
            )
        return _shift_market_session(
            session,
            self.signal_lag_sessions + horizon,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "calibration_plan_sha256": self.calibration_plan_sha256,
            "research_calibration_artifact_sha256": (
                self.research_calibration_artifact_sha256
            ),
            "selected_config_sha256": self.selected_config_sha256,
            "detector_version": self.detector_version,
            "detector_code_sha256": self.detector_code_sha256,
            "calibration_code_sha256": self.calibration_code_sha256,
            "runtime_fingerprint_sha256": (
                self.runtime_fingerprint_sha256
            ),
            "holdout_start_session": self.holdout_start_session.isoformat(),
            "observation_end_session": (
                self.observation_end_session.isoformat()
            ),
            "final_outcome_resolution_session": (
                self.final_outcome_resolution_session.isoformat()
            ),
            "review_not_before_session": (
                self.review_not_before_session.isoformat()
            ),
            "activation_deadline": _utc_iso(self.activation_deadline),
            "activation_status": self.activation_status,
            "activated_at": (
                None
                if self.activated_at is None
                else _utc_iso(self.activated_at)
            ),
            "activation_receipt_sha256": (
                self.activation_receipt_sha256
            ),
            "activation_reason_codes": list(
                self.activation_reason_codes
            ),
            "outcome_horizons": list(self.outcome_horizons),
            "signal_lag_sessions": self.signal_lag_sessions,
            "minimum_signal_rows": self.minimum_signal_rows,
            "minimum_resolved_rows": self.minimum_resolved_rows,
            "minimum_outcome_coverage": self.minimum_outcome_coverage,
            "maximum_missing_signal_sessions": (
                self.maximum_missing_signal_sessions
            ),
            "required_entitlement_scopes": list(
                self.required_entitlement_scopes
            ),
            "maximum_entitlement_check_age_seconds": (
                self.maximum_entitlement_check_age_seconds
            ),
            "missing_signal_rule": self.missing_signal_rule,
            "missing_outcome_rule": self.missing_outcome_rule,
            "late_backfill_rule": self.late_backfill_rule,
            "statistical_review_rule": self.statistical_review_rule,
            "no_retuning": self.no_retuning,
            "execution_eligible": self.execution_eligible,
        }

    @property
    def protocol_sha256(self) -> str:
        return _domain_hash(_PROTOCOL_HASH_DOMAIN, self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProspectiveReviewProtocol":
        expected = {
            "schema_version",
            "protocol_id",
            "calibration_plan_sha256",
            "research_calibration_artifact_sha256",
            "selected_config_sha256",
            "detector_version",
            "detector_code_sha256",
            "calibration_code_sha256",
            "runtime_fingerprint_sha256",
            "holdout_start_session",
            "observation_end_session",
            "final_outcome_resolution_session",
            "review_not_before_session",
            "activation_deadline",
            "activation_status",
            "activated_at",
            "activation_receipt_sha256",
            "activation_reason_codes",
            "outcome_horizons",
            "signal_lag_sessions",
            "minimum_signal_rows",
            "minimum_resolved_rows",
            "minimum_outcome_coverage",
            "maximum_missing_signal_sessions",
            "required_entitlement_scopes",
            "maximum_entitlement_check_age_seconds",
            "missing_signal_rule",
            "missing_outcome_rule",
            "late_backfill_rule",
            "statistical_review_rule",
            "no_retuning",
            "execution_eligible",
        }
        _strict_fields(payload, expected, "prospective_review_protocol")
        values = dict(payload)
        values["outcome_horizons"] = tuple(values["outcome_horizons"])
        values["activation_reason_codes"] = tuple(
            values["activation_reason_codes"]
        )
        values["required_entitlement_scopes"] = tuple(
            values["required_entitlement_scopes"]
        )
        return cls(**values)

    @classmethod
    def from_json(cls, payload: str) -> "ProspectiveReviewProtocol":
        if not isinstance(payload, str):
            raise TypeError("prospective protocol JSON must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "prospective protocol is not valid JSON"
            ) from exc
        if _canonical_json(decoded) != payload:
            raise ProspectiveReviewError(
                "prospective protocol JSON must be canonical"
            )
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class ProviderEntitlementReceipt:
    """Secret-free durable receipt issued by the external rights validator."""

    record_id: str
    subject_sha256: str
    entitlement_evidence_sha256: str
    terms_sha256: str
    retention_policy_sha256: str
    deletion_policy_sha256: str
    reviewer_sha256: str
    scopes: tuple[str, ...]
    valid_from: datetime
    valid_until: datetime
    checked_at: datetime
    status: EntitlementStatus = EntitlementStatus.ACTIVE
    schema_version: str = ENTITLEMENT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENTITLEMENT_RECEIPT_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported entitlement receipt schema"
            )
        object.__setattr__(
            self,
            "record_id",
            _token(self.record_id, "record_id"),
        )
        for field_name in (
            "subject_sha256",
            "entitlement_evidence_sha256",
            "terms_sha256",
            "retention_policy_sha256",
            "deletion_policy_sha256",
            "reviewer_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "scopes",
            _tokens(self.scopes, "scopes"),
        )
        valid_from = _utc_datetime(self.valid_from, "valid_from")
        valid_until = _utc_datetime(self.valid_until, "valid_until")
        checked_at = _utc_datetime(self.checked_at, "checked_at")
        if not valid_from <= checked_at < valid_until:
            raise ProspectiveReviewError(
                "entitlement receipt validity times are inconsistent"
            )
        try:
            status = EntitlementStatus(self.status)
        except ValueError as exc:
            raise ProspectiveReviewError(
                "unsupported entitlement status"
            ) from exc
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "checked_at", checked_at)
        object.__setattr__(self, "status", status)

    def covers(
        self,
        *,
        required_scopes: Sequence[str],
        at: datetime,
        maximum_check_age_seconds: int,
    ) -> bool:
        instant = _utc_datetime(at, "entitlement_use_at")
        age = (instant - self.checked_at).total_seconds()
        return (
            self.status is EntitlementStatus.ACTIVE
            and self.valid_from <= instant < self.valid_until
            and 0 <= age <= maximum_check_age_seconds
            and set(required_scopes).issubset(self.scopes)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "subject_sha256": self.subject_sha256,
            "entitlement_evidence_sha256": (
                self.entitlement_evidence_sha256
            ),
            "terms_sha256": self.terms_sha256,
            "retention_policy_sha256": self.retention_policy_sha256,
            "deletion_policy_sha256": self.deletion_policy_sha256,
            "reviewer_sha256": self.reviewer_sha256,
            "scopes": list(self.scopes),
            "valid_from": _utc_iso(self.valid_from),
            "valid_until": _utc_iso(self.valid_until),
            "checked_at": _utc_iso(self.checked_at),
            "status": self.status.value,
        }

    @property
    def receipt_sha256(self) -> str:
        return _domain_hash(_ENTITLEMENT_HASH_DOMAIN, self.to_dict())

    def to_envelope(self) -> dict[str, Any]:
        return {
            "receipt": self.to_dict(),
            "receipt_sha256": self.receipt_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_envelope())

    @classmethod
    def from_envelope(
        cls,
        envelope: Mapping[str, Any],
    ) -> "ProviderEntitlementReceipt":
        _strict_fields(
            envelope,
            {"receipt", "receipt_sha256"},
            "entitlement_receipt_envelope",
        )
        body = envelope["receipt"]
        expected = {
            "schema_version",
            "record_id",
            "subject_sha256",
            "entitlement_evidence_sha256",
            "terms_sha256",
            "retention_policy_sha256",
            "deletion_policy_sha256",
            "reviewer_sha256",
            "scopes",
            "valid_from",
            "valid_until",
            "checked_at",
            "status",
        }
        _strict_fields(body, expected, "entitlement_receipt")
        values = dict(body)
        values["scopes"] = tuple(values["scopes"])
        receipt = cls(**values)
        if receipt.receipt_sha256 != _sha256(
            envelope["receipt_sha256"],
            "receipt_sha256",
        ):
            raise ProspectiveReviewError(
                "entitlement receipt hash does not match"
            )
        return receipt

    @classmethod
    def from_json(cls, payload: str) -> "ProviderEntitlementReceipt":
        if not isinstance(payload, str):
            raise TypeError("entitlement receipt JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "entitlement receipt is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise ProspectiveReviewError(
                "entitlement receipt JSON must be canonical"
            )
        return cls.from_envelope(envelope)


@dataclass(frozen=True)
class ProspectiveJournalEntry:
    """One immutable, hash-chained prospective signal observation."""

    protocol_sha256: str
    sequence_number: int
    previous_entry_sha256: str | None
    recorded_at: datetime
    signal: RegimeSignal
    entitlement_receipt: ProviderEntitlementReceipt
    schema_version: str = JOURNAL_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOURNAL_ENTRY_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported prospective journal entry schema"
            )
        object.__setattr__(
            self,
            "protocol_sha256",
            _sha256(self.protocol_sha256, "protocol_sha256"),
        )
        sequence = _positive_integer(
            self.sequence_number,
            "sequence_number",
        )
        previous = self.previous_entry_sha256
        if sequence == 1:
            if previous is not None:
                raise ProspectiveReviewError(
                    "the first journal entry cannot have a predecessor"
                )
        else:
            previous = _sha256(previous, "previous_entry_sha256")
        if type(self.signal) is not RegimeSignal:
            raise TypeError("signal must be an exact RegimeSignal instance")
        if type(self.entitlement_receipt) is not ProviderEntitlementReceipt:
            raise TypeError(
                "entitlement_receipt must be an exact receipt instance"
            )
        if self.signal.may_authorize_execution:
            raise ProspectiveReviewError(
                "execution-capable signals cannot enter the journal"
            )
        try:
            sealed_signal = RegimeSignal.from_json(self.signal.to_json())
        except (TypeError, RegimeSignalError) as exc:
            raise ProspectiveReviewError(
                "journal signal is not a sealed RegimeSignal"
            ) from exc
        if sealed_signal != self.signal:
            raise ProspectiveReviewError(
                "journal signal does not round-trip exactly"
            )
        object.__setattr__(self, "sequence_number", sequence)
        object.__setattr__(self, "previous_entry_sha256", previous)
        object.__setattr__(
            self,
            "recorded_at",
            _utc_datetime(self.recorded_at, "recorded_at"),
        )

    @classmethod
    def create(
        cls,
        *,
        protocol: ProspectiveReviewProtocol,
        sequence_number: int,
        previous_entry_sha256: str | None,
        recorded_at: datetime,
        signal: RegimeSignal,
        entitlement_receipt: ProviderEntitlementReceipt,
    ) -> "ProspectiveJournalEntry":
        if type(protocol) is not ProspectiveReviewProtocol:
            raise TypeError(
                "protocol must be an exact ProspectiveReviewProtocol"
            )
        entry = cls(
            protocol_sha256=protocol.protocol_sha256,
            sequence_number=sequence_number,
            previous_entry_sha256=previous_entry_sha256,
            recorded_at=recorded_at,
            signal=signal,
            entitlement_receipt=entitlement_receipt,
        )
        entry.validate_against(protocol)
        return entry

    def validate_against(
        self,
        protocol: ProspectiveReviewProtocol,
    ) -> None:
        if type(protocol) is not ProspectiveReviewProtocol:
            raise TypeError(
                "protocol must be an exact ProspectiveReviewProtocol"
            )
        if self.protocol_sha256 != protocol.protocol_sha256:
            raise ProspectiveReviewError(
                "journal entry protocol hash does not match"
            )
        if not protocol.accumulation_eligible:
            raise ProspectiveReviewError(
                "prospective protocol was not activated before holdout"
            )
        signal = self.signal
        if signal.as_of_session not in protocol.expected_signal_sessions:
            raise ProspectiveReviewError(
                "signal is outside the prospective observation window"
            )
        if self.recorded_at < signal.available_at:
            raise ProspectiveReviewError(
                "journal entry predates signal availability"
            )
        if signal.available_at < _session_finalization(
            signal.as_of_session
        ):
            raise ProspectiveReviewError(
                "signal availability precedes joint close finalization"
            )
        if signal.available_at >= _session_open(
            signal.effective_session
        ):
            raise ProspectiveReviewError(
                "signal availability misses the effective-session open"
            )
        if self.recorded_at >= _session_open(signal.effective_session):
            raise ProspectiveReviewError(
                "late or retrospective journal backfill is forbidden"
            )
        lineage = signal.lineage
        exact_lineage = {
            "artifact_sha256": (
                protocol.research_calibration_artifact_sha256
            ),
            "plan_sha256": protocol.calibration_plan_sha256,
            "config_sha256": protocol.selected_config_sha256,
            "detector_version": protocol.detector_version,
            "detector_code_sha256": protocol.detector_code_sha256,
            "calibration_code_sha256": protocol.calibration_code_sha256,
            "runtime_fingerprint_sha256": (
                protocol.runtime_fingerprint_sha256
            ),
        }
        for field_name, expected in exact_lineage.items():
            if getattr(lineage, field_name) != expected:
                raise ProspectiveReviewError(
                    f"signal {field_name} does not match the protocol"
                )
        if (
            lineage.source_provenance_status != "verified"
            or lineage.snapshot_sha256 is None
            or lineage.evidence_manifest_sha256 is None
            or lineage.evidence_verification_kind != "decision_time"
            or not lineage.evidence_decision_time_eligible
            or lineage.calendar_schedule_sha256 is None
            or lineage.source_policy_sha256 is None
        ):
            raise ProspectiveReviewError(
                "journal signal lacks verified decision-time evidence"
            )
        receipt = self.entitlement_receipt
        if not receipt.covers(
            required_scopes=protocol.required_entitlement_scopes,
            at=signal.available_at,
            maximum_check_age_seconds=(
                protocol.maximum_entitlement_check_age_seconds
            ),
        ):
            raise ProspectiveReviewError(
                "entitlement receipt does not cover signal creation"
            )
        if not receipt.covers(
            required_scopes=protocol.required_entitlement_scopes,
            at=self.recorded_at,
            maximum_check_age_seconds=(
                protocol.maximum_entitlement_check_age_seconds
            ),
        ):
            raise ProspectiveReviewError(
                "entitlement receipt does not cover journal publication"
            )

    def to_dict(self) -> dict[str, Any]:
        lineage = self.signal.lineage
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "sequence_number": self.sequence_number,
            "previous_entry_sha256": self.previous_entry_sha256,
            "recorded_at": _utc_iso(self.recorded_at),
            "signal": self.signal.to_envelope(),
            "signal_sha256": self.signal.signal_sha256,
            "snapshot_sha256": lineage.snapshot_sha256,
            "evidence_manifest_sha256": (
                lineage.evidence_manifest_sha256
            ),
            "detector_version": lineage.detector_version,
            "detector_code_sha256": lineage.detector_code_sha256,
            "config_sha256": lineage.config_sha256,
            "calibration_artifact_sha256": lineage.artifact_sha256,
            "calibration_plan_sha256": lineage.plan_sha256,
            "calibration_code_sha256": lineage.calibration_code_sha256,
            "runtime_fingerprint_sha256": (
                lineage.runtime_fingerprint_sha256
            ),
            "entitlement_receipt": (
                self.entitlement_receipt.to_envelope()
            ),
            "entitlement_receipt_sha256": (
                self.entitlement_receipt.receipt_sha256
            ),
            "execution_eligible": False,
        }

    @property
    def entry_sha256(self) -> str:
        return _domain_hash(_JOURNAL_HASH_DOMAIN, self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(
            {
                "entry": self.to_dict(),
                "entry_sha256": self.entry_sha256,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "ProspectiveJournalEntry":
        if not isinstance(payload, str):
            raise TypeError("journal entry JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "journal entry is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise ProspectiveReviewError(
                "journal entry JSON must be canonical"
            )
        _strict_fields(
            envelope,
            {"entry", "entry_sha256"},
            "journal_entry_envelope",
        )
        body = envelope["entry"]
        expected = {
            "schema_version",
            "protocol_sha256",
            "sequence_number",
            "previous_entry_sha256",
            "recorded_at",
            "signal",
            "signal_sha256",
            "snapshot_sha256",
            "evidence_manifest_sha256",
            "detector_version",
            "detector_code_sha256",
            "config_sha256",
            "calibration_artifact_sha256",
            "calibration_plan_sha256",
            "calibration_code_sha256",
            "runtime_fingerprint_sha256",
            "entitlement_receipt",
            "entitlement_receipt_sha256",
            "execution_eligible",
        }
        _strict_fields(body, expected, "journal_entry")
        if body["execution_eligible"] is not False:
            raise ProspectiveReviewError(
                "journal entries cannot authorize execution"
            )
        try:
            signal = RegimeSignal.from_envelope(body["signal"])
        except (TypeError, RegimeSignalError) as exc:
            raise ProspectiveReviewError(
                "journal signal envelope is invalid"
            ) from exc
        receipt = ProviderEntitlementReceipt.from_envelope(
            body["entitlement_receipt"]
        )
        entry = cls(
            schema_version=body["schema_version"],
            protocol_sha256=body["protocol_sha256"],
            sequence_number=body["sequence_number"],
            previous_entry_sha256=body["previous_entry_sha256"],
            recorded_at=body["recorded_at"],
            signal=signal,
            entitlement_receipt=receipt,
        )
        expected_bindings = {
            "signal_sha256": signal.signal_sha256,
            "snapshot_sha256": signal.lineage.snapshot_sha256,
            "evidence_manifest_sha256": (
                signal.lineage.evidence_manifest_sha256
            ),
            "detector_version": signal.lineage.detector_version,
            "detector_code_sha256": (
                signal.lineage.detector_code_sha256
            ),
            "config_sha256": signal.lineage.config_sha256,
            "calibration_artifact_sha256": (
                signal.lineage.artifact_sha256
            ),
            "calibration_plan_sha256": signal.lineage.plan_sha256,
            "calibration_code_sha256": (
                signal.lineage.calibration_code_sha256
            ),
            "runtime_fingerprint_sha256": (
                signal.lineage.runtime_fingerprint_sha256
            ),
            "entitlement_receipt_sha256": receipt.receipt_sha256,
        }
        for field_name, expected_value in expected_bindings.items():
            if body[field_name] != expected_value:
                raise ProspectiveReviewError(
                    f"journal {field_name} binding does not match"
                )
        if entry.entry_sha256 != _sha256(
            envelope["entry_sha256"],
            "entry_sha256",
        ):
            raise ProspectiveReviewError(
                "journal entry hash does not match"
            )
        return entry


def validate_journal_chain(
    protocol: ProspectiveReviewProtocol,
    entries: Sequence[ProspectiveJournalEntry],
) -> tuple[ProspectiveJournalEntry, ...]:
    if type(protocol) is not ProspectiveReviewProtocol:
        raise TypeError(
            "protocol must be an exact ProspectiveReviewProtocol"
        )
    try:
        chain = tuple(entries)
    except TypeError as exc:
        raise ProspectiveReviewError(
            "journal entries must be iterable"
        ) from exc
    previous_hash: str | None = None
    previous_session: date | None = None
    for expected_sequence, entry in enumerate(chain, start=1):
        if type(entry) is not ProspectiveJournalEntry:
            raise TypeError(
                "journal must contain exact ProspectiveJournalEntry values"
            )
        entry.validate_against(protocol)
        if (
            entry.sequence_number != expected_sequence
            or entry.previous_entry_sha256 != previous_hash
        ):
            raise ProspectiveReviewError(
                "prospective journal hash chain is not contiguous"
            )
        if (
            previous_session is not None
            and entry.signal.as_of_session <= previous_session
        ):
            raise ProspectiveReviewError(
                "journal signal sessions must be strictly increasing"
            )
        previous_hash = entry.entry_sha256
        previous_session = entry.signal.as_of_session
    return chain


@dataclass(frozen=True)
class ProspectiveOutcomeRecord:
    """Content identity and resolution clock for one journaled signal."""

    protocol_sha256: str
    journal_entry_sha256: str
    signal_as_of_session: date
    resolved_horizons: tuple[int, ...]
    final_resolution_session: date
    outcome_payload_sha256: str
    outcome_evidence_sha256: str
    computed_at: datetime
    schema_version: str = OUTCOME_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OUTCOME_RECORD_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported prospective outcome schema"
            )
        for field_name in (
            "protocol_sha256",
            "journal_entry_sha256",
            "outcome_payload_sha256",
            "outcome_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "signal_as_of_session",
            _market_session(
                self.signal_as_of_session,
                "signal_as_of_session",
            ),
        )
        object.__setattr__(
            self,
            "final_resolution_session",
            _market_session(
                self.final_resolution_session,
                "final_resolution_session",
            ),
        )
        if isinstance(self.resolved_horizons, (str, bytes)):
            raise ProspectiveReviewError(
                "resolved_horizons must be iterable"
            )
        try:
            horizons = tuple(self.resolved_horizons)
        except TypeError as exc:
            raise ProspectiveReviewError(
                "resolved_horizons must be iterable"
            ) from exc
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons)))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 1
                for item in horizons
            )
        ):
            raise ProspectiveReviewError(
                "resolved_horizons must be sorted unique positive integers"
            )
        object.__setattr__(self, "resolved_horizons", horizons)
        object.__setattr__(
            self,
            "computed_at",
            _utc_datetime(self.computed_at, "computed_at"),
        )

    def validate_against(
        self,
        protocol: ProspectiveReviewProtocol,
        entry: ProspectiveJournalEntry,
    ) -> None:
        if self.protocol_sha256 != protocol.protocol_sha256:
            raise ProspectiveReviewError(
                "outcome protocol hash does not match"
            )
        if (
            self.journal_entry_sha256 != entry.entry_sha256
            or self.signal_as_of_session != entry.signal.as_of_session
        ):
            raise ProspectiveReviewError(
                "outcome does not bind the journal entry"
            )
        if self.resolved_horizons != protocol.outcome_horizons:
            raise ProspectiveReviewError(
                "outcome horizons do not match the frozen protocol"
            )
        expected_resolution = protocol.resolution_session_for(
            self.signal_as_of_session,
            max(self.resolved_horizons),
        )
        if self.final_resolution_session != expected_resolution:
            raise ProspectiveReviewError(
                "outcome resolution session does not match"
            )
        if self.computed_at < _session_finalization(expected_resolution):
            raise ProspectiveReviewError(
                "outcome was computed before its horizon resolved"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "journal_entry_sha256": self.journal_entry_sha256,
            "signal_as_of_session": self.signal_as_of_session.isoformat(),
            "resolved_horizons": list(self.resolved_horizons),
            "final_resolution_session": (
                self.final_resolution_session.isoformat()
            ),
            "outcome_payload_sha256": self.outcome_payload_sha256,
            "outcome_evidence_sha256": self.outcome_evidence_sha256,
            "computed_at": _utc_iso(self.computed_at),
            "execution_eligible": False,
        }

    @property
    def outcome_sha256(self) -> str:
        return _domain_hash(_OUTCOME_HASH_DOMAIN, self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(
            {
                "outcome": self.to_dict(),
                "outcome_sha256": self.outcome_sha256,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "ProspectiveOutcomeRecord":
        if not isinstance(payload, str):
            raise TypeError("prospective outcome JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "prospective outcome is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise ProspectiveReviewError(
                "prospective outcome JSON must be canonical"
            )
        _strict_fields(
            envelope,
            {"outcome", "outcome_sha256"},
            "prospective_outcome_envelope",
        )
        body = envelope["outcome"]
        expected = {
            "schema_version",
            "protocol_sha256",
            "journal_entry_sha256",
            "signal_as_of_session",
            "resolved_horizons",
            "final_resolution_session",
            "outcome_payload_sha256",
            "outcome_evidence_sha256",
            "computed_at",
            "execution_eligible",
        }
        _strict_fields(body, expected, "prospective_outcome")
        if body["execution_eligible"] is not False:
            raise ProspectiveReviewError(
                "prospective outcomes cannot authorize execution"
            )
        outcome = cls(
            schema_version=body["schema_version"],
            protocol_sha256=body["protocol_sha256"],
            journal_entry_sha256=body["journal_entry_sha256"],
            signal_as_of_session=body["signal_as_of_session"],
            resolved_horizons=tuple(body["resolved_horizons"]),
            final_resolution_session=body[
                "final_resolution_session"
            ],
            outcome_payload_sha256=body["outcome_payload_sha256"],
            outcome_evidence_sha256=body[
                "outcome_evidence_sha256"
            ],
            computed_at=body["computed_at"],
        )
        if outcome.outcome_sha256 != _sha256(
            envelope["outcome_sha256"],
            "outcome_sha256",
        ):
            raise ProspectiveReviewError(
                "prospective outcome hash does not match"
            )
        return outcome


def outcome_set_sha256(
    outcomes: Sequence[ProspectiveOutcomeRecord],
) -> str:
    records = tuple(outcomes)
    if any(type(item) is not ProspectiveOutcomeRecord for item in records):
        raise TypeError(
            "outcomes must contain exact ProspectiveOutcomeRecord values"
        )
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.signal_as_of_session,
                item.journal_entry_sha256,
            ),
        )
    )
    return _domain_hash(
        _OUTCOME_SET_HASH_DOMAIN,
        [item.outcome_sha256 for item in ordered],
    )


@dataclass(frozen=True)
class VerifiedCalibrationEvidence:
    """Receipt for a provider-verified replay of the frozen calibration."""

    calibration_plan_sha256: str
    research_calibration_artifact_sha256: str
    verified_replay_artifact_sha256: str
    selected_config_sha256: str
    detector_version: str
    detector_code_sha256: str
    calibration_code_sha256: str
    runtime_fingerprint_sha256: str
    verified_data_manifest_sha256: str
    evidence_manifest_sha256: str
    entitlement_receipt: ProviderEntitlementReceipt
    verified_at: datetime
    provenance_status: str = "verified"
    verification_kind: str = "verified_replay"
    selected_candidate_unchanged: bool = True
    no_retuning_confirmed: bool = True
    execution_eligible: bool = False
    schema_version: str = CALIBRATION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_EVIDENCE_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported verified calibration schema"
            )
        for field_name in (
            "calibration_plan_sha256",
            "research_calibration_artifact_sha256",
            "verified_replay_artifact_sha256",
            "selected_config_sha256",
            "detector_code_sha256",
            "calibration_code_sha256",
            "runtime_fingerprint_sha256",
            "verified_data_manifest_sha256",
            "evidence_manifest_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        if type(self.entitlement_receipt) is not ProviderEntitlementReceipt:
            raise TypeError(
                "entitlement_receipt must be an exact receipt instance"
            )
        object.__setattr__(
            self,
            "detector_version",
            _token(self.detector_version, "detector_version"),
        )
        object.__setattr__(
            self,
            "verified_at",
            _utc_datetime(self.verified_at, "verified_at"),
        )
        if (
            self.provenance_status != "verified"
            or self.verification_kind != "verified_replay"
            or self.selected_candidate_unchanged is not True
            or self.no_retuning_confirmed is not True
            or self.execution_eligible is not False
        ):
            raise ProspectiveReviewError(
                "verified calibration must preserve the frozen candidate"
            )

    def validate_against(
        self,
        protocol: ProspectiveReviewProtocol,
    ) -> None:
        exact = {
            "calibration_plan_sha256": (
                protocol.calibration_plan_sha256
            ),
            "research_calibration_artifact_sha256": (
                protocol.research_calibration_artifact_sha256
            ),
            "selected_config_sha256": (
                protocol.selected_config_sha256
            ),
            "detector_version": protocol.detector_version,
            "detector_code_sha256": protocol.detector_code_sha256,
            "calibration_code_sha256": (
                protocol.calibration_code_sha256
            ),
            "runtime_fingerprint_sha256": (
                protocol.runtime_fingerprint_sha256
            ),
        }
        for field_name, expected in exact.items():
            if getattr(self, field_name) != expected:
                raise ProspectiveReviewError(
                    f"verified calibration {field_name} does not match"
                )
        if not self.entitlement_receipt.covers(
            required_scopes=protocol.required_entitlement_scopes,
            at=self.verified_at,
            maximum_check_age_seconds=(
                protocol.maximum_entitlement_check_age_seconds
            ),
        ):
            raise ProspectiveReviewError(
                "verified calibration lacks current entitlement scope"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_plan_sha256": (
                self.calibration_plan_sha256
            ),
            "research_calibration_artifact_sha256": (
                self.research_calibration_artifact_sha256
            ),
            "verified_replay_artifact_sha256": (
                self.verified_replay_artifact_sha256
            ),
            "selected_config_sha256": self.selected_config_sha256,
            "detector_version": self.detector_version,
            "detector_code_sha256": self.detector_code_sha256,
            "calibration_code_sha256": self.calibration_code_sha256,
            "runtime_fingerprint_sha256": (
                self.runtime_fingerprint_sha256
            ),
            "verified_data_manifest_sha256": (
                self.verified_data_manifest_sha256
            ),
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "entitlement_receipt": (
                self.entitlement_receipt.to_envelope()
            ),
            "entitlement_receipt_sha256": (
                self.entitlement_receipt.receipt_sha256
            ),
            "verified_at": _utc_iso(self.verified_at),
            "provenance_status": self.provenance_status,
            "verification_kind": self.verification_kind,
            "selected_candidate_unchanged": (
                self.selected_candidate_unchanged
            ),
            "no_retuning_confirmed": self.no_retuning_confirmed,
            "execution_eligible": self.execution_eligible,
        }

    @property
    def evidence_sha256(self) -> str:
        return _domain_hash(_CALIBRATION_HASH_DOMAIN, self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(
            {
                "evidence": self.to_dict(),
                "evidence_sha256": self.evidence_sha256,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "VerifiedCalibrationEvidence":
        if not isinstance(payload, str):
            raise TypeError("verified calibration JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "verified calibration is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise ProspectiveReviewError(
                "verified calibration JSON must be canonical"
            )
        _strict_fields(
            envelope,
            {"evidence", "evidence_sha256"},
            "verified_calibration_envelope",
        )
        body = envelope["evidence"]
        expected = {
            "schema_version",
            "calibration_plan_sha256",
            "research_calibration_artifact_sha256",
            "verified_replay_artifact_sha256",
            "selected_config_sha256",
            "detector_version",
            "detector_code_sha256",
            "calibration_code_sha256",
            "runtime_fingerprint_sha256",
            "verified_data_manifest_sha256",
            "evidence_manifest_sha256",
            "entitlement_receipt",
            "entitlement_receipt_sha256",
            "verified_at",
            "provenance_status",
            "verification_kind",
            "selected_candidate_unchanged",
            "no_retuning_confirmed",
            "execution_eligible",
        }
        _strict_fields(body, expected, "verified_calibration")
        receipt = ProviderEntitlementReceipt.from_envelope(
            body["entitlement_receipt"]
        )
        if body["entitlement_receipt_sha256"] != receipt.receipt_sha256:
            raise ProspectiveReviewError(
                "verified calibration entitlement hash does not match"
            )
        evidence = cls(
            schema_version=body["schema_version"],
            calibration_plan_sha256=body["calibration_plan_sha256"],
            research_calibration_artifact_sha256=body[
                "research_calibration_artifact_sha256"
            ],
            verified_replay_artifact_sha256=body[
                "verified_replay_artifact_sha256"
            ],
            selected_config_sha256=body["selected_config_sha256"],
            detector_version=body["detector_version"],
            detector_code_sha256=body["detector_code_sha256"],
            calibration_code_sha256=body["calibration_code_sha256"],
            runtime_fingerprint_sha256=body[
                "runtime_fingerprint_sha256"
            ],
            verified_data_manifest_sha256=body[
                "verified_data_manifest_sha256"
            ],
            evidence_manifest_sha256=body[
                "evidence_manifest_sha256"
            ],
            entitlement_receipt=receipt,
            verified_at=body["verified_at"],
            provenance_status=body["provenance_status"],
            verification_kind=body["verification_kind"],
            selected_candidate_unchanged=body[
                "selected_candidate_unchanged"
            ],
            no_retuning_confirmed=body["no_retuning_confirmed"],
            execution_eligible=body["execution_eligible"],
        )
        if evidence.evidence_sha256 != _sha256(
            envelope["evidence_sha256"],
            "evidence_sha256",
        ):
            raise ProspectiveReviewError(
                "verified calibration hash does not match"
            )
        return evidence


@dataclass(frozen=True)
class ProspectiveReviewReceipt:
    """Independent statistical review bound to the exact evidence set."""

    protocol_sha256: str
    journal_head_sha256: str
    outcome_set_sha256: str
    calibration_evidence_sha256: str
    statistical_report_sha256: str
    reviewer_sha256: str
    reviewed_at: datetime
    verdict: ReviewVerdict
    no_retuning_confirmed: bool = True
    execution_eligible: bool = False
    schema_version: str = REVIEW_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_RECEIPT_SCHEMA_VERSION:
            raise ProspectiveReviewError(
                "unsupported prospective review receipt schema"
            )
        for field_name in (
            "protocol_sha256",
            "journal_head_sha256",
            "outcome_set_sha256",
            "calibration_evidence_sha256",
            "statistical_report_sha256",
            "reviewer_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "reviewed_at",
            _utc_datetime(self.reviewed_at, "reviewed_at"),
        )
        try:
            verdict = ReviewVerdict(self.verdict)
        except ValueError as exc:
            raise ProspectiveReviewError(
                "unsupported statistical review verdict"
            ) from exc
        if (
            self.no_retuning_confirmed is not True
            or self.execution_eligible is not False
        ):
            raise ProspectiveReviewError(
                "review receipt must preserve no-retuning and abstention"
            )
        object.__setattr__(self, "verdict", verdict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_sha256": self.protocol_sha256,
            "journal_head_sha256": self.journal_head_sha256,
            "outcome_set_sha256": self.outcome_set_sha256,
            "calibration_evidence_sha256": (
                self.calibration_evidence_sha256
            ),
            "statistical_report_sha256": (
                self.statistical_report_sha256
            ),
            "reviewer_sha256": self.reviewer_sha256,
            "reviewed_at": _utc_iso(self.reviewed_at),
            "verdict": self.verdict.value,
            "no_retuning_confirmed": self.no_retuning_confirmed,
            "execution_eligible": self.execution_eligible,
        }

    @property
    def receipt_sha256(self) -> str:
        return _domain_hash(_REVIEW_HASH_DOMAIN, self.to_dict())

    def to_json(self) -> str:
        return _canonical_json(
            {
                "receipt": self.to_dict(),
                "receipt_sha256": self.receipt_sha256,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "ProspectiveReviewReceipt":
        if not isinstance(payload, str):
            raise TypeError("prospective review JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProspectiveReviewError(
                "prospective review is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise ProspectiveReviewError(
                "prospective review JSON must be canonical"
            )
        _strict_fields(
            envelope,
            {"receipt", "receipt_sha256"},
            "prospective_review_envelope",
        )
        body = envelope["receipt"]
        expected = {
            "schema_version",
            "protocol_sha256",
            "journal_head_sha256",
            "outcome_set_sha256",
            "calibration_evidence_sha256",
            "statistical_report_sha256",
            "reviewer_sha256",
            "reviewed_at",
            "verdict",
            "no_retuning_confirmed",
            "execution_eligible",
        }
        _strict_fields(body, expected, "prospective_review")
        receipt = cls(
            schema_version=body["schema_version"],
            protocol_sha256=body["protocol_sha256"],
            journal_head_sha256=body["journal_head_sha256"],
            outcome_set_sha256=body["outcome_set_sha256"],
            calibration_evidence_sha256=body[
                "calibration_evidence_sha256"
            ],
            statistical_report_sha256=body[
                "statistical_report_sha256"
            ],
            reviewer_sha256=body["reviewer_sha256"],
            reviewed_at=body["reviewed_at"],
            verdict=body["verdict"],
            no_retuning_confirmed=body["no_retuning_confirmed"],
            execution_eligible=body["execution_eligible"],
        )
        if receipt.receipt_sha256 != _sha256(
            envelope["receipt_sha256"],
            "receipt_sha256",
        ):
            raise ProspectiveReviewError(
                "prospective review hash does not match"
            )
        return receipt


class RegimePromotionGate:
    """Pure structural gate; PASS never projects into an execution action."""

    @staticmethod
    def evaluate(
        *,
        protocol: ProspectiveReviewProtocol,
        entries: Sequence[ProspectiveJournalEntry],
        outcomes: Sequence[ProspectiveOutcomeRecord],
        calibration_evidence: VerifiedCalibrationEvidence | None,
        review_receipt: ProspectiveReviewReceipt | None,
    ) -> PromotionGateStatus:
        try:
            return RegimePromotionGate._evaluate(
                protocol=protocol,
                entries=entries,
                outcomes=outcomes,
                calibration_evidence=calibration_evidence,
                review_receipt=review_receipt,
            )
        except (ProspectiveReviewError, TypeError, ValueError):
            return PromotionGateStatus.BLOCKED

    @staticmethod
    def _evaluate(
        *,
        protocol: ProspectiveReviewProtocol,
        entries: Sequence[ProspectiveJournalEntry],
        outcomes: Sequence[ProspectiveOutcomeRecord],
        calibration_evidence: VerifiedCalibrationEvidence | None,
        review_receipt: ProspectiveReviewReceipt | None,
    ) -> PromotionGateStatus:
        if type(protocol) is not ProspectiveReviewProtocol:
            raise TypeError(
                "protocol must be an exact ProspectiveReviewProtocol"
            )
        if protocol.protocol_sha256 != DEPLOYMENT_PROTOCOL_SHA256:
            return PromotionGateStatus.BLOCKED
        if not protocol.accumulation_eligible:
            return PromotionGateStatus.BLOCKED
        if not PROMOTION_AUTHENTICATION_IMPLEMENTED:
            return PromotionGateStatus.BLOCKED
        current_time = _trusted_utc_now()
        current_date = current_time.date()
        chain = validate_journal_chain(protocol, entries)
        expected_sessions = set(protocol.expected_signal_sessions)
        observed_sessions = {
            entry.signal.as_of_session
            for entry in chain
            if (
                entry.signal.availability is SignalAvailability.ADVISORY
                and entry.signal.background_state
                is not BackgroundState.UNAVAILABLE
                and entry.signal.shock_state is not ShockState.UNAVAILABLE
            )
        }
        missing_sessions = expected_sessions.difference(observed_sessions)

        try:
            outcome_records = tuple(outcomes)
        except TypeError as exc:
            raise ProspectiveReviewError(
                "outcomes must be iterable"
            ) from exc
        by_entry = {entry.entry_sha256: entry for entry in chain}
        seen_outcomes: set[str] = set()
        for outcome in outcome_records:
            if type(outcome) is not ProspectiveOutcomeRecord:
                raise TypeError(
                    "outcomes must contain ProspectiveOutcomeRecord values"
                )
            if outcome.journal_entry_sha256 in seen_outcomes:
                raise ProspectiveReviewError(
                    "duplicate outcome for one journal entry"
                )
            entry = by_entry.get(outcome.journal_entry_sha256)
            if entry is None:
                raise ProspectiveReviewError(
                    "outcome refers to an unknown journal entry"
                )
            outcome.validate_against(protocol, entry)
            seen_outcomes.add(outcome.journal_entry_sha256)

        complete_outcomes = len(outcome_records)
        coverage = (
            complete_outcomes / len(protocol.expected_signal_sessions)
        )
        window_is_resolved = current_time >= _session_finalization(
            protocol.final_outcome_resolution_session
        )
        review_may_begin = (
            current_date > protocol.review_not_before_session
            or (
                current_date == protocol.review_not_before_session
                and current_time
                >= _session_finalization(
                    protocol.review_not_before_session
                )
            )
        )
        structurally_complete = (
            len(observed_sessions) >= protocol.minimum_signal_rows
            and len(missing_sessions)
            <= protocol.maximum_missing_signal_sessions
            and complete_outcomes >= protocol.minimum_resolved_rows
            and coverage >= protocol.minimum_outcome_coverage
        )
        if not structurally_complete:
            return (
                PromotionGateStatus.BLOCKED
                if window_is_resolved and review_may_begin
                else PromotionGateStatus.INCOMPLETE
            )
        if not review_may_begin:
            return PromotionGateStatus.INCOMPLETE
        if calibration_evidence is None:
            return PromotionGateStatus.BLOCKED
        if type(calibration_evidence) is not VerifiedCalibrationEvidence:
            raise TypeError(
                "calibration_evidence must be verified calibration evidence"
            )
        calibration_evidence.validate_against(protocol)
        if review_receipt is None:
            return PromotionGateStatus.INCOMPLETE
        if type(review_receipt) is not ProspectiveReviewReceipt:
            raise TypeError(
                "review_receipt must be a ProspectiveReviewReceipt"
            )
        if not chain:
            return PromotionGateStatus.BLOCKED
        expected_receipt_values = {
            "protocol_sha256": protocol.protocol_sha256,
            "journal_head_sha256": chain[-1].entry_sha256,
            "outcome_set_sha256": outcome_set_sha256(outcome_records),
            "calibration_evidence_sha256": (
                calibration_evidence.evidence_sha256
            ),
        }
        for field_name, expected in expected_receipt_values.items():
            if getattr(review_receipt, field_name) != expected:
                return PromotionGateStatus.BLOCKED
        if (
            review_receipt.reviewed_at.date()
            < protocol.review_not_before_session
            or review_receipt.reviewed_at > current_time
            or calibration_evidence.verified_at > review_receipt.reviewed_at
            or any(
                outcome.computed_at > review_receipt.reviewed_at
                for outcome in outcome_records
            )
        ):
            return PromotionGateStatus.BLOCKED
        if review_receipt.verdict is ReviewVerdict.FAIL:
            return PromotionGateStatus.BLOCKED
        return PromotionGateStatus.PASS


__all__ = [
    "ACTIVATION_ACTIVE",
    "ACTIVATION_BLOCKED",
    "CALIBRATION_EVIDENCE_SCHEMA_VERSION",
    "DEPLOYMENT_PROTOCOL_SHA256",
    "ENTITLEMENT_RECEIPT_SCHEMA_VERSION",
    "EntitlementStatus",
    "JOURNAL_ENTRY_SCHEMA_VERSION",
    "LATE_BACKFILL_RULE",
    "MISSING_OUTCOME_RULE",
    "MISSING_SIGNAL_RULE",
    "OUTCOME_RECORD_SCHEMA_VERSION",
    "PROSPECTIVE_PROTOCOL_SCHEMA_VERSION",
    "PROMOTION_AUTHENTICATION_IMPLEMENTED",
    "PromotionGateStatus",
    "ProspectiveJournalEntry",
    "ProspectiveOutcomeRecord",
    "ProspectiveReviewError",
    "ProspectiveReviewProtocol",
    "ProspectiveReviewReceipt",
    "ProviderEntitlementReceipt",
    "REVIEW_RECEIPT_SCHEMA_VERSION",
    "RegimePromotionGate",
    "ReviewVerdict",
    "STATISTICAL_REVIEW_RULE",
    "VerifiedCalibrationEvidence",
    "outcome_set_sha256",
    "validate_journal_chain",
]
