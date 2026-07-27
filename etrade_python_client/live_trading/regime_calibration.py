"""Causal, research-only calibration for the transparent V2 detector.

The detector has no observable ground-truth regime label.  This module
therefore evaluates a small, predeclared candidate set against independent
forward risk outcomes: future realized volatility and drawdown.  Every
reference outcome must be fully resolved before its validation fold begins.

Calibration artifacts are deterministic, content-addressed, and explicitly
non-executable.  Legacy normalized prices can support retrospective research,
but only decision-time provider evidence can satisfy the provenance portion of
a future promotion gate.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from live_trading.regime_detector_v2 import (
    BACKGROUND_CALM,
    BACKGROUND_ELEVATED,
    BACKGROUND_STRESS,
    BACKGROUND_UNAVAILABLE,
    DETECTOR_VERSION,
    SHOCK_ACTIVE,
    SHOCK_UNAVAILABLE,
    SIGNAL_TIMESTAMP,
    RegimeDetectorConfig,
    detect_regimes,
    regime_detector_code_sha256,
    regime_detector_config_sha256,
)
from live_trading.regime_market_data import (
    CALENDAR_POLICY_VERSION,
    SOURCE_POLICY_VERSION,
    regime_market_schedule,
    regime_schedule_sha256,
    regime_source_policy_sha256,
)


CALIBRATION_SCHEMA_VERSION = "regime_v2_calibration.v1"
CALIBRATION_METHOD_VERSION = "purged_walk_forward_tail_risk.v1"
OUTCOME_DEFINITION_VERSION = "post_lag_forward_rms_vol_and_peak_drawdown.v1"
SELECTION_RULE_VERSION = "material_improvement_over_control.v1"
STATE_SEED_POLICY = "replay_from_manifest_start_elevated.v1"
INFERENCE_METHOD = "causal_prefix_filter"
PROVENANCE_LEGACY_UNVERIFIED = "legacy_normalized_unverified"
PROVENANCE_PROVIDER_DECISION_TIME = "provider_evidence_decision_time"
PROMOTION_RESEARCH_ONLY = "research_only"
_ARTIFACT_HASH_DOMAIN = b"regime-v2-calibration-artifact.v1\0"
_DATA_HASH_DOMAIN = b"regime-v2-calibration-data.v1\0"
_PLAN_HASH_DOMAIN = b"regime-v2-calibration-plan.v1\0"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BACKGROUND_ORDER = {
    BACKGROUND_CALM: 0.0,
    BACKGROUND_ELEVATED: 1.0,
    BACKGROUND_STRESS: 2.0,
}


class RegimeCalibrationError(ValueError):
    """Raised when a calibration contract or causal boundary is invalid."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _domain_sha256(domain: bytes, payload: str) -> str:
    return hashlib.sha256(domain + payload.encode("utf-8")).hexdigest()


def regime_calibration_code_sha256() -> str:
    """Return the exact calibration-engine source identity."""

    source = inspect.getsource(sys.modules[__name__]).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _strict_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise RegimeCalibrationError(f"{name} fields do not match the schema")


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RegimeCalibrationError(
            f"{field_name} must be a lowercase stable identifier"
        )
    return value


def _calendar_date(value: Any, field_name: str) -> date:
    if isinstance(value, datetime):
        raise RegimeCalibrationError(f"{field_name} must be a calendar date")
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            raise RegimeCalibrationError(f"{field_name} must be a calendar date")
        value = value.date()
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise RegimeCalibrationError(
                f"{field_name} must be an ISO calendar date"
            ) from exc
    if not isinstance(value, date):
        raise RegimeCalibrationError(f"{field_name} must be a calendar date")
    return value


def _market_session(value: Any, field_name: str) -> date:
    session = _calendar_date(value, field_name)
    if regime_market_schedule(session, session).empty:
        raise RegimeCalibrationError(
            f"{field_name} must be an NYSE/Cboe joint market session"
        )
    return session


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RegimeCalibrationError(f"{field_name} must be a SHA-256 digest")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegimeCalibrationError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RegimeCalibrationError(f"{field_name} must be finite")
    return normalized


def _config_from_dict(payload: Mapping[str, Any]) -> RegimeDetectorConfig:
    expected = {item.name for item in fields(RegimeDetectorConfig)}
    _strict_fields(payload, expected, "detector_config")
    try:
        return RegimeDetectorConfig(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise RegimeCalibrationError("detector_config is invalid") from exc


@dataclass(frozen=True)
class CalibrationCandidate:
    """One predeclared detector configuration."""

    candidate_id: str
    config: RegimeDetectorConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        if not isinstance(self.config, RegimeDetectorConfig):
            raise TypeError("config must be a RegimeDetectorConfig")

    @property
    def config_sha256(self) -> str:
        return regime_detector_config_sha256(self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config": asdict(self.config),
            "config_sha256": self.config_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationCandidate":
        _strict_fields(
            payload,
            {"candidate_id", "config", "config_sha256"},
            "calibration_candidate",
        )
        candidate = cls(
            candidate_id=payload["candidate_id"],
            config=_config_from_dict(payload["config"]),
        )
        if candidate.config_sha256 != _sha256(
            payload["config_sha256"],
            "config_sha256",
        ):
            raise RegimeCalibrationError("candidate config hash does not match")
        return candidate


@dataclass(frozen=True)
class CausalValidationFold:
    """A purged reference window followed by a disjoint evaluation window."""

    fold_id: str
    outcome_reference_start: date
    outcome_reference_end: date
    evaluation_start: date
    evaluation_end: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _identifier(self.fold_id, "fold_id"))
        for field_name in (
            "outcome_reference_start",
            "outcome_reference_end",
            "evaluation_start",
            "evaluation_end",
        ):
            object.__setattr__(
                self,
                field_name,
                _market_session(getattr(self, field_name), field_name),
            )
        if not (
            self.outcome_reference_start
            <= self.outcome_reference_end
            < self.evaluation_start
            <= self.evaluation_end
        ):
            raise RegimeCalibrationError(
                "fold dates must order reference_start <= reference_end "
                "< evaluation_start <= evaluation_end"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "fold_id": self.fold_id,
            "outcome_reference_start": self.outcome_reference_start.isoformat(),
            "outcome_reference_end": self.outcome_reference_end.isoformat(),
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalValidationFold":
        expected = {
            "fold_id",
            "outcome_reference_start",
            "outcome_reference_end",
            "evaluation_start",
            "evaluation_end",
        }
        _strict_fields(payload, expected, "causal_validation_fold")
        return cls(**dict(payload))


@dataclass(frozen=True)
class RegimeCalibrationPlan:
    """Immutable pre-registration for candidate selection and acceptance."""

    protocol_id: str
    candidates: tuple[CalibrationCandidate, ...]
    control_candidate_id: str
    selection_folds: tuple[CausalValidationFold, ...]
    retrospective_test: CausalValidationFold
    prospective_holdout_start: date
    outcome_horizons: tuple[int, ...] = (5, 20)
    tail_percentile: float = 0.95
    min_evaluation_rows: int = 120
    minimum_evaluation_coverage: float = 0.99
    max_state_occupancy: float = 0.95
    max_switches_per_252: float = 24.0
    isolated_shock_followup_sessions: int = 5
    minimum_isolated_shock_episodes: int = 50
    max_false_persistent_isolated_shock_rate: float = 0.10
    minimum_material_improvement: float = 0.01
    signal_lag_sessions: int = 1
    detector_version: str = DETECTOR_VERSION
    detector_code_sha256: str = ""
    calibration_code_sha256: str = ""
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    method_version: str = CALIBRATION_METHOD_VERSION
    outcome_definition_version: str = OUTCOME_DEFINITION_VERSION
    selection_rule_version: str = SELECTION_RULE_VERSION
    inference_method: str = INFERENCE_METHOD
    signal_timestamp: str = SIGNAL_TIMESTAMP
    state_seed_policy: str = STATE_SEED_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _identifier(self.protocol_id, "protocol_id"),
        )
        try:
            candidates = tuple(self.candidates)
            folds = tuple(self.selection_folds)
            horizons = tuple(self.outcome_horizons)
        except TypeError as exc:
            raise TypeError(
                "candidates, selection_folds, and outcome_horizons "
                "must be iterable"
            ) from exc
        if not candidates or not all(
            isinstance(item, CalibrationCandidate) for item in candidates
        ):
            raise RegimeCalibrationError(
                "candidates must contain CalibrationCandidate values"
            )
        candidate_ids = [item.candidate_id for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RegimeCalibrationError("candidate IDs must be unique")
        shock_contracts = {
            (
                item.config.vix_shock_return,
                item.config.spy_shock_log_return,
                item.config.aftershock_days,
            )
            for item in candidates
        }
        if len(shock_contracts) != 1:
            raise RegimeCalibrationError(
                "candidate selection cannot tune the independent shock lane"
            )
        control = _identifier(
            self.control_candidate_id,
            "control_candidate_id",
        )
        if control not in candidate_ids:
            raise RegimeCalibrationError(
                "control_candidate_id is not in candidates"
            )
        if not folds or not all(
            isinstance(item, CausalValidationFold) for item in folds
        ):
            raise RegimeCalibrationError(
                "selection_folds must contain CausalValidationFold values"
            )
        fold_ids = [item.fold_id for item in folds]
        if len(fold_ids) != len(set(fold_ids)):
            raise RegimeCalibrationError("selection fold IDs must be unique")
        if folds != tuple(
            sorted(folds, key=lambda item: item.evaluation_start)
        ):
            raise RegimeCalibrationError(
                "selection folds must be ordered by evaluation_start"
            )
        if any(
            previous.evaluation_end >= current.evaluation_start
            for previous, current in zip(folds, folds[1:])
        ):
            raise RegimeCalibrationError(
                "selection evaluation windows must not overlap"
            )
        if not isinstance(self.retrospective_test, CausalValidationFold):
            raise TypeError(
                "retrospective_test must be a CausalValidationFold"
            )
        if max(item.evaluation_end for item in folds) >= (
            self.retrospective_test.evaluation_start
        ):
            raise RegimeCalibrationError(
                "selection folds must end before the retrospective test"
            )
        prospective = _calendar_date(
            self.prospective_holdout_start,
            "prospective_holdout_start",
        )
        prospective = _market_session(
            prospective,
            "prospective_holdout_start",
        )
        if prospective <= self.retrospective_test.evaluation_end:
            raise RegimeCalibrationError(
                "prospective holdout must begin after the retrospective test"
            )
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in horizons
            )
        ):
            raise RegimeCalibrationError(
                "outcome_horizons must be unique sorted positive integers"
            )
        if (
            isinstance(self.min_evaluation_rows, bool)
            or not isinstance(self.min_evaluation_rows, int)
            or self.min_evaluation_rows < 1
        ):
            raise RegimeCalibrationError(
                "min_evaluation_rows must be a positive integer"
            )
        if (
            isinstance(self.isolated_shock_followup_sessions, bool)
            or not isinstance(self.isolated_shock_followup_sessions, int)
            or self.isolated_shock_followup_sessions < 1
        ):
            raise RegimeCalibrationError(
                "isolated_shock_followup_sessions must be positive"
            )
        if (
            isinstance(self.minimum_isolated_shock_episodes, bool)
            or not isinstance(self.minimum_isolated_shock_episodes, int)
            or self.minimum_isolated_shock_episodes < 0
        ):
            raise RegimeCalibrationError(
                "minimum_isolated_shock_episodes must be nonnegative"
            )
        if (
            isinstance(self.signal_lag_sessions, bool)
            or not isinstance(self.signal_lag_sessions, int)
            or self.signal_lag_sessions != 1
        ):
            raise RegimeCalibrationError(
                "this outcome definition requires exactly one lagged session"
            )
        holdout_bridge = regime_market_schedule(
            self.retrospective_test.evaluation_end,
            prospective,
        )
        if (
            len(holdout_bridge)
            <= self.signal_lag_sessions + max(horizons)
        ):
            raise RegimeCalibrationError(
                "prospective holdout must begin after retrospective "
                "outcomes are fully resolved"
            )
        tail = _finite_float(self.tail_percentile, "tail_percentile")
        occupancy = _finite_float(
            self.max_state_occupancy,
            "max_state_occupancy",
        )
        coverage = _finite_float(
            self.minimum_evaluation_coverage,
            "minimum_evaluation_coverage",
        )
        switches = _finite_float(
            self.max_switches_per_252,
            "max_switches_per_252",
        )
        false_persistence = _finite_float(
            self.max_false_persistent_isolated_shock_rate,
            "max_false_persistent_isolated_shock_rate",
        )
        improvement = _finite_float(
            self.minimum_material_improvement,
            "minimum_material_improvement",
        )
        if not 0.5 < tail < 1.0:
            raise RegimeCalibrationError(
                "tail_percentile must be between 0.5 and 1"
            )
        if not 0.0 < occupancy < 1.0:
            raise RegimeCalibrationError(
                "max_state_occupancy must be between zero and one"
            )
        if not 0.0 < coverage <= 1.0:
            raise RegimeCalibrationError(
                "minimum_evaluation_coverage must be in (0, 1]"
            )
        if switches <= 0:
            raise RegimeCalibrationError(
                "max_switches_per_252 must be positive"
            )
        if not 0.0 <= false_persistence <= 1.0:
            raise RegimeCalibrationError(
                "max_false_persistent_isolated_shock_rate must be in [0, 1]"
            )
        if improvement < 0:
            raise RegimeCalibrationError(
                "minimum_material_improvement cannot be negative"
            )
        if self.detector_version != DETECTOR_VERSION:
            raise RegimeCalibrationError(
                "plan detector_version does not match the active detector"
            )
        code_sha = self.detector_code_sha256 or regime_detector_code_sha256()
        _sha256(code_sha, "detector_code_sha256")
        calibration_sha = (
            self.calibration_code_sha256
            or regime_calibration_code_sha256()
        )
        _sha256(calibration_sha, "calibration_code_sha256")
        fixed_values = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "method_version": CALIBRATION_METHOD_VERSION,
            "outcome_definition_version": OUTCOME_DEFINITION_VERSION,
            "selection_rule_version": SELECTION_RULE_VERSION,
            "inference_method": INFERENCE_METHOD,
            "signal_timestamp": SIGNAL_TIMESTAMP,
            "state_seed_policy": STATE_SEED_POLICY,
        }
        for field_name, expected in fixed_values.items():
            if getattr(self, field_name) != expected:
                raise RegimeCalibrationError(
                    f"unsupported {field_name}: {getattr(self, field_name)}"
                )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "control_candidate_id", control)
        object.__setattr__(self, "selection_folds", folds)
        object.__setattr__(self, "prospective_holdout_start", prospective)
        object.__setattr__(self, "outcome_horizons", horizons)
        object.__setattr__(self, "tail_percentile", tail)
        object.__setattr__(self, "max_state_occupancy", occupancy)
        object.__setattr__(
            self,
            "minimum_evaluation_coverage",
            coverage,
        )
        object.__setattr__(self, "max_switches_per_252", switches)
        object.__setattr__(
            self,
            "max_false_persistent_isolated_shock_rate",
            false_persistence,
        )
        object.__setattr__(
            self,
            "minimum_material_improvement",
            improvement,
        )
        object.__setattr__(self, "detector_code_sha256", code_sha)
        object.__setattr__(
            self,
            "calibration_code_sha256",
            calibration_sha,
        )

    @property
    def max_outcome_horizon(self) -> int:
        return max(self.outcome_horizons)

    @property
    def max_outcome_resolution_lag(self) -> int:
        return self.signal_lag_sessions + self.max_outcome_horizon

    def candidate(self, candidate_id: str) -> CalibrationCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"Unknown calibration candidate: {candidate_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "outcome_definition_version": self.outcome_definition_version,
            "selection_rule_version": self.selection_rule_version,
            "protocol_id": self.protocol_id,
            "detector_version": self.detector_version,
            "detector_code_sha256": self.detector_code_sha256,
            "calibration_code_sha256": self.calibration_code_sha256,
            "signal_timestamp": self.signal_timestamp,
            "signal_lag_sessions": self.signal_lag_sessions,
            "inference_method": self.inference_method,
            "state_seed_policy": self.state_seed_policy,
            "outcome_horizons": list(self.outcome_horizons),
            "tail_percentile": self.tail_percentile,
            "min_evaluation_rows": self.min_evaluation_rows,
            "minimum_evaluation_coverage": (
                self.minimum_evaluation_coverage
            ),
            "max_state_occupancy": self.max_state_occupancy,
            "max_switches_per_252": self.max_switches_per_252,
            "isolated_shock_followup_sessions": (
                self.isolated_shock_followup_sessions
            ),
            "minimum_isolated_shock_episodes": (
                self.minimum_isolated_shock_episodes
            ),
            "max_false_persistent_isolated_shock_rate": (
                self.max_false_persistent_isolated_shock_rate
            ),
            "minimum_material_improvement": (
                self.minimum_material_improvement
            ),
            "candidates": [item.to_dict() for item in self.candidates],
            "control_candidate_id": self.control_candidate_id,
            "selection_folds": [
                item.to_dict() for item in self.selection_folds
            ],
            "retrospective_test": self.retrospective_test.to_dict(),
            "prospective_holdout_start": (
                self.prospective_holdout_start.isoformat()
            ),
        }

    @property
    def plan_sha256(self) -> str:
        return _domain_sha256(_PLAN_HASH_DOMAIN, _canonical_json(self.to_dict()))

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegimeCalibrationPlan":
        expected = {
            "schema_version",
            "method_version",
            "outcome_definition_version",
            "selection_rule_version",
            "protocol_id",
            "detector_version",
            "detector_code_sha256",
            "calibration_code_sha256",
            "signal_timestamp",
            "signal_lag_sessions",
            "inference_method",
            "state_seed_policy",
            "outcome_horizons",
            "tail_percentile",
            "min_evaluation_rows",
            "minimum_evaluation_coverage",
            "max_state_occupancy",
            "max_switches_per_252",
            "isolated_shock_followup_sessions",
            "minimum_isolated_shock_episodes",
            "max_false_persistent_isolated_shock_rate",
            "minimum_material_improvement",
            "candidates",
            "control_candidate_id",
            "selection_folds",
            "retrospective_test",
            "prospective_holdout_start",
        }
        _strict_fields(payload, expected, "regime_calibration_plan")
        return cls(
            protocol_id=payload["protocol_id"],
            candidates=tuple(
                CalibrationCandidate.from_dict(item)
                for item in payload["candidates"]
            ),
            control_candidate_id=payload["control_candidate_id"],
            selection_folds=tuple(
                CausalValidationFold.from_dict(item)
                for item in payload["selection_folds"]
            ),
            retrospective_test=CausalValidationFold.from_dict(
                payload["retrospective_test"]
            ),
            prospective_holdout_start=payload["prospective_holdout_start"],
            outcome_horizons=tuple(payload["outcome_horizons"]),
            tail_percentile=payload["tail_percentile"],
            min_evaluation_rows=payload["min_evaluation_rows"],
            minimum_evaluation_coverage=payload[
                "minimum_evaluation_coverage"
            ],
            max_state_occupancy=payload["max_state_occupancy"],
            max_switches_per_252=payload["max_switches_per_252"],
            isolated_shock_followup_sessions=payload[
                "isolated_shock_followup_sessions"
            ],
            minimum_isolated_shock_episodes=payload[
                "minimum_isolated_shock_episodes"
            ],
            max_false_persistent_isolated_shock_rate=payload[
                "max_false_persistent_isolated_shock_rate"
            ],
            minimum_material_improvement=(
                payload["minimum_material_improvement"]
            ),
            signal_lag_sessions=payload["signal_lag_sessions"],
            detector_version=payload["detector_version"],
            detector_code_sha256=payload["detector_code_sha256"],
            calibration_code_sha256=payload["calibration_code_sha256"],
            schema_version=payload["schema_version"],
            method_version=payload["method_version"],
            outcome_definition_version=payload[
                "outcome_definition_version"
            ],
            selection_rule_version=payload["selection_rule_version"],
            inference_method=payload["inference_method"],
            signal_timestamp=payload["signal_timestamp"],
            state_seed_policy=payload["state_seed_policy"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "RegimeCalibrationPlan":
        if not isinstance(payload, str):
            raise TypeError("calibration plan JSON must be a string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RegimeCalibrationError(
                "calibration plan is not valid JSON"
            ) from exc
        if _canonical_json(decoded) != payload:
            raise RegimeCalibrationError(
                "calibration plan JSON must be canonical"
            )
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class CalibrationDataManifest:
    """Content identity and provenance status of the exact evaluated prefix."""

    provenance_status: str
    first_session: date
    last_session: date
    session_count: int
    prices_sha256: str
    calendar_policy_version: str
    schedule_sha256: str
    source_policy_version: str
    source_policy_sha256: str
    snapshot_sha256: str | None = None
    evidence_manifest_sha256: str | None = None
    evidence_verification_kind: str | None = None

    def __post_init__(self) -> None:
        if self.provenance_status != PROVENANCE_LEGACY_UNVERIFIED:
            raise RegimeCalibrationError(
                "R3 accepts only explicitly unverified legacy calibration "
                "data; verified provider calibration is not implemented"
            )
        first = _calendar_date(self.first_session, "first_session")
        last = _calendar_date(self.last_session, "last_session")
        if last < first:
            raise RegimeCalibrationError(
                "last_session cannot precede first_session"
            )
        if (
            isinstance(self.session_count, bool)
            or not isinstance(self.session_count, int)
            or self.session_count < 1
        ):
            raise RegimeCalibrationError(
                "session_count must be a positive integer"
            )
        _sha256(self.prices_sha256, "prices_sha256")
        if self.calendar_policy_version != CALENDAR_POLICY_VERSION:
            raise RegimeCalibrationError(
                "calibration calendar policy is stale"
            )
        schedule_sha = _sha256(self.schedule_sha256, "schedule_sha256")
        if schedule_sha != regime_schedule_sha256(first, last):
            raise RegimeCalibrationError(
                "calibration schedule hash does not match the active clock"
            )
        if self.source_policy_version != SOURCE_POLICY_VERSION:
            raise RegimeCalibrationError(
                "calibration source policy is stale"
            )
        source_policy_sha = _sha256(
            self.source_policy_sha256,
            "source_policy_sha256",
        )
        if source_policy_sha != regime_source_policy_sha256():
            raise RegimeCalibrationError(
                "calibration source policy hash is stale"
            )
        if any(
            value is not None
            for value in (
                self.snapshot_sha256,
                self.evidence_manifest_sha256,
                self.evidence_verification_kind,
            )
        ):
            raise RegimeCalibrationError(
                "legacy calibration data cannot claim provider evidence"
            )
        object.__setattr__(self, "first_session", first)
        object.__setattr__(self, "last_session", last)

    @property
    def decision_time_provenance(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_status": self.provenance_status,
            "first_session": self.first_session.isoformat(),
            "last_session": self.last_session.isoformat(),
            "session_count": self.session_count,
            "prices_sha256": self.prices_sha256,
            "calendar_policy_version": self.calendar_policy_version,
            "schedule_sha256": self.schedule_sha256,
            "source_policy_version": self.source_policy_version,
            "source_policy_sha256": self.source_policy_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "evidence_verification_kind": self.evidence_verification_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationDataManifest":
        expected = {
            "provenance_status",
            "first_session",
            "last_session",
            "session_count",
            "prices_sha256",
            "calendar_policy_version",
            "schedule_sha256",
            "source_policy_version",
            "source_policy_sha256",
            "snapshot_sha256",
            "evidence_manifest_sha256",
            "evidence_verification_kind",
        }
        _strict_fields(payload, expected, "calibration_data_manifest")
        return cls(**dict(payload))


def _validated_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    required = ["SPY_Close", "VIX_Close"]
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise RegimeCalibrationError(
            f"calibration prices are missing columns: {missing}"
        )
    frame = prices[required].copy()
    if frame.empty:
        raise RegimeCalibrationError("calibration prices are empty")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    except Exception as exc:
        raise RegimeCalibrationError(
            "calibration price index is invalid"
        ) from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RegimeCalibrationError(
            "calibration sessions must be unique and sorted"
        )
    frame.index = index
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RegimeCalibrationError(
            "calibration prices must be finite and positive"
        )
    schedule = regime_market_schedule(
        frame.index.min().date(),
        frame.index.max().date(),
    )
    expected = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    if not frame.index.equals(expected):
        raise RegimeCalibrationError(
            "calibration prices must exactly cover contiguous NYSE sessions"
        )
    return frame


def _prices_sha256(prices: pd.DataFrame) -> str:
    rows = [
        {
            "session": session.date().isoformat(),
            "spy_close": format(float(row["SPY_Close"]), ".17g"),
            "vix_close": format(float(row["VIX_Close"]), ".17g"),
        }
        for session, row in prices.iterrows()
    ]
    return _domain_sha256(_DATA_HASH_DOMAIN, _canonical_json(rows))


def build_calibration_data_manifest(
    prices: pd.DataFrame,
    *,
    provenance_status: str,
    snapshot_sha256: str | None = None,
    evidence_manifest_sha256: str | None = None,
    evidence_verification_kind: str | None = None,
) -> CalibrationDataManifest:
    """Build a content manifest without trusting filenames or mtimes."""

    frame = _validated_prices(prices)
    return CalibrationDataManifest(
        provenance_status=provenance_status,
        first_session=frame.index.min().date(),
        last_session=frame.index.max().date(),
        session_count=len(frame),
        prices_sha256=_prices_sha256(frame),
        calendar_policy_version=CALENDAR_POLICY_VERSION,
        schedule_sha256=regime_schedule_sha256(
            frame.index.min().date(),
            frame.index.max().date(),
        ),
        source_policy_version=SOURCE_POLICY_VERSION,
        source_policy_sha256=regime_source_policy_sha256(),
        snapshot_sha256=snapshot_sha256,
        evidence_manifest_sha256=evidence_manifest_sha256,
        evidence_verification_kind=evidence_verification_kind,
    )


def build_forward_risk_outcomes(
    prices: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Create conservative post-lag labels from close T+1 through T+h+1.

    A close-T signal is first tradable during session T+1.  Daily closes cannot
    reconstruct that session's open-to-close return, so the research label
    begins at the T+1 close rather than including an untradeable close-T to
    close-T+1 move.
    """

    frame = _validated_prices(prices)
    normalized_horizons = tuple(horizons)
    if (
        not normalized_horizons
        or normalized_horizons
        != tuple(sorted(set(normalized_horizons)))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in normalized_horizons
        )
    ):
        raise RegimeCalibrationError(
            "horizons must be unique sorted positive integers"
        )
    spy = frame["SPY_Close"].to_numpy(dtype=float)
    log_spy = np.log(spy)
    output = pd.DataFrame(index=frame.index)
    for horizon in normalized_horizons:
        rms = np.full(len(frame), np.nan, dtype=float)
        drawdown = np.full(len(frame), np.nan, dtype=float)
        resolved = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
        if len(frame) > horizon + 1:
            log_windows = np.lib.stride_tricks.sliding_window_view(
                log_spy,
                horizon + 2,
            )
            future_returns = np.diff(log_windows[:, 1:], axis=1)
            rms[: len(log_windows)] = np.sqrt(
                np.mean(np.square(future_returns), axis=1) * 252.0
            )
            price_windows = np.lib.stride_tricks.sliding_window_view(
                spy,
                horizon + 2,
            )
            post_lag_paths = price_windows[:, 1:]
            running_peaks = np.maximum.accumulate(post_lag_paths, axis=1)
            drawdown[: len(price_windows)] = np.max(
                1.0 - post_lag_paths / running_peaks,
                axis=1,
            )
            resolved[: len(log_windows)] = frame.index[
                horizon + 1 :
            ].to_numpy(
                dtype="datetime64[ns]"
            )
        output[f"Forward_RMS_Vol_{horizon}"] = rms
        output[f"Forward_Max_Peak_Drawdown_{horizon}"] = drawdown
        output[f"Outcome_Resolved_At_{horizon}"] = pd.DatetimeIndex(
            resolved
        )
    return output


def _empirical_percentiles(
    reference: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) < 1:
        raise RegimeCalibrationError(
            "outcome reference contains no finite observations"
        )
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2.0 * len(reference))


@dataclass(frozen=True)
class FoldMetrics:
    """Finite, serializable diagnostics for one candidate/fold pair."""

    fold_id: str
    candidate_id: str
    planned_evaluation_rows: int
    evaluation_rows: int
    evaluation_coverage: float
    tail_events: int
    true_positive: int
    false_positive: int
    false_negative: int
    tail_recall: float
    tail_precision: float
    tail_f1: float
    ordinal_risk_spearman: float
    switches_per_252: float
    maximum_state_occupancy: float
    isolated_shocks: int
    false_persistent_on_isolated_shock: int
    false_persistent_isolated_shock_rate: float
    guardrail_passed: bool
    guardrail_failure_codes: tuple[str, ...]
    state_occupancy: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _identifier(self.fold_id, "fold_id"))
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        integer_fields = (
            "planned_evaluation_rows",
            "evaluation_rows",
            "tail_events",
            "true_positive",
            "false_positive",
            "false_negative",
            "isolated_shocks",
            "false_persistent_on_isolated_shock",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RegimeCalibrationError(
                    f"{field_name} must be a nonnegative integer"
                )
        for field_name in (
            "tail_recall",
            "tail_precision",
            "tail_f1",
            "ordinal_risk_spearman",
            "switches_per_252",
            "maximum_state_occupancy",
            "evaluation_coverage",
            "false_persistent_isolated_shock_rate",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )
        if self.evaluation_rows > self.planned_evaluation_rows:
            raise RegimeCalibrationError(
                "evaluation_rows cannot exceed planned_evaluation_rows"
            )
        if not 0.0 <= self.evaluation_coverage <= 1.0:
            raise RegimeCalibrationError(
                "evaluation_coverage must be in [0, 1]"
            )
        if not 0.0 <= self.false_persistent_isolated_shock_rate <= 1.0:
            raise RegimeCalibrationError(
                "false persistent shock rate must be in [0, 1]"
            )
        if not isinstance(self.guardrail_passed, bool):
            raise TypeError("guardrail_passed must be a boolean")
        failure_codes = tuple(
            _identifier(item, "guardrail_failure_code")
            for item in self.guardrail_failure_codes
        )
        if self.guardrail_passed == bool(failure_codes):
            raise RegimeCalibrationError(
                "guardrail status and failure codes are inconsistent"
            )
        occupancy = tuple(self.state_occupancy)
        if any(
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or item[0] not in _BACKGROUND_ORDER
            for item in occupancy
        ):
            raise RegimeCalibrationError("state_occupancy is invalid")
        normalized = tuple(
            (str(state), _finite_float(value, f"occupancy[{state}]"))
            for state, value in occupancy
        )
        if normalized and not math.isclose(
            sum(value for _, value in normalized),
            1.0,
            abs_tol=1e-9,
        ):
            raise RegimeCalibrationError(
                "state occupancy proportions must sum to one"
            )
        object.__setattr__(self, "state_occupancy", normalized)
        object.__setattr__(
            self,
            "guardrail_failure_codes",
            failure_codes,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state_occupancy"] = [list(item) for item in self.state_occupancy]
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoldMetrics":
        expected = {item.name for item in fields(cls)}
        _strict_fields(payload, expected, "fold_metrics")
        values = dict(payload)
        values["state_occupancy"] = tuple(
            tuple(item) for item in values["state_occupancy"]
        )
        return cls(**values)


def _fold_risk_frame(
    outcomes: pd.DataFrame,
    fold: CausalValidationFold,
    horizons: tuple[int, ...],
    tail_percentile: float,
) -> pd.DataFrame:
    max_horizon = max(horizons)
    reference = outcomes.loc[
        fold.outcome_reference_start.isoformat() :
        fold.outcome_reference_end.isoformat()
    ].copy()
    evaluation = outcomes.loc[
        fold.evaluation_start.isoformat() :
        fold.evaluation_end.isoformat()
    ].copy()
    resolution_column = f"Outcome_Resolved_At_{max_horizon}"
    measure_columns = [
        *(f"Forward_RMS_Vol_{horizon}" for horizon in horizons),
        f"Forward_Max_Peak_Drawdown_{max_horizon}",
    ]
    if reference.empty or evaluation.empty:
        raise RegimeCalibrationError(
            f"fold {fold.fold_id} has no reference or evaluation rows"
        )
    if reference[measure_columns + [resolution_column]].isna().any().any():
        raise RegimeCalibrationError(
            f"fold {fold.fold_id} reference outcomes are unresolved"
        )
    if evaluation[measure_columns].isna().any().any():
        raise RegimeCalibrationError(
            f"fold {fold.fold_id} evaluation outcomes are unresolved"
        )
    latest_reference_resolution = reference[resolution_column].max()
    if latest_reference_resolution >= pd.Timestamp(fold.evaluation_start):
        raise RegimeCalibrationError(
            f"fold {fold.fold_id} reference outcomes are not purged"
        )
    risk = pd.DataFrame(index=evaluation.index)
    evaluation_scores = []
    for column in measure_columns:
        reference_values = reference[column].to_numpy(dtype=float)
        evaluation_scores.append(
            _empirical_percentiles(
                reference_values,
                evaluation[column].to_numpy(dtype=float),
            )
        )
    evaluation_composite = np.max(
        np.column_stack(evaluation_scores),
        axis=1,
    )
    risk["Forward_Risk_Score"] = evaluation_composite
    risk["Forward_Tail_Event"] = evaluation_composite >= tail_percentile
    risk["Tail_Threshold"] = float(tail_percentile)
    return risk


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_candidate_fold(
    detector_result: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidate_id: str,
    fold: CausalValidationFold,
    plan: RegimeCalibrationPlan,
) -> FoldMetrics:
    """Evaluate one already-causal detector trace on one purged fold."""

    risk = _fold_risk_frame(
        outcomes,
        fold,
        plan.outcome_horizons,
        plan.tail_percentile,
    )
    aligned = risk.join(detector_result, how="left")
    required_detector_columns = {
        "Background_State",
        "Shock_State",
    }
    if not required_detector_columns.issubset(aligned.columns):
        raise RegimeCalibrationError(
            "detector result is missing required regime columns"
        )
    available = (
        aligned["Background_State"].notna()
        & aligned["Shock_State"].notna()
        & (aligned["Background_State"] != BACKGROUND_UNAVAILABLE)
        & (aligned["Shock_State"] != SHOCK_UNAVAILABLE)
    )
    planned_rows = len(aligned)
    evaluation_coverage = float(available.sum() / planned_rows)
    frame = aligned.loc[available].copy()
    if frame.empty:
        raise RegimeCalibrationError(
            f"fold {fold.fold_id} has no detector output to evaluate"
        )
    tail = frame["Forward_Tail_Event"].astype(bool)
    alert = (
        (frame["Background_State"] == BACKGROUND_STRESS)
        | (frame["Shock_State"] == SHOCK_ACTIVE)
    )
    true_positive = int((alert & tail).sum())
    false_positive = int((alert & ~tail).sum())
    false_negative = int((~alert & tail).sum())
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    ordinal = frame["Background_State"].map(_BACKGROUND_ORDER).astype(float)
    risk_score = frame["Forward_Risk_Score"].astype(float)
    if ordinal.nunique() < 2 or risk_score.nunique() < 2:
        spearman = 0.0
    else:
        spearman_value = ordinal.corr(risk_score, method="spearman")
        spearman = (
            float(spearman_value) if pd.notna(spearman_value) else 0.0
        )
    state_counts = frame["Background_State"].value_counts()
    occupancy = tuple(
        sorted(
            (
                state,
                float(count / len(frame)),
            )
            for state, count in state_counts.items()
            if state in _BACKGROUND_ORDER
        )
    )
    maximum_occupancy = max((value for _, value in occupancy), default=1.0)
    transitions = int(
        frame["Background_State"]
        .ne(frame["Background_State"].shift(1))
        .iloc[1:]
        .sum()
    )
    switches_per_252 = float(transitions * 252.0 / len(frame))
    previous_background = detector_result["Background_State"].shift(1).reindex(
        aligned.index
    )
    previous_shock = detector_result["Shock_State"].shift(1).reindex(
        aligned.index
    )
    isolated_shocks = 0
    false_persistent = 0
    followup = plan.isolated_shock_followup_sessions
    for position in range(0, len(aligned) - followup):
        current = aligned.iloc[position]
        if (
            not bool(available.iloc[position])
            or current["Shock_State"] != SHOCK_ACTIVE
            or previous_shock.iloc[position] == SHOCK_ACTIVE
            or previous_background.iloc[position] == BACKGROUND_STRESS
        ):
            continue
        episode = aligned.iloc[position : position + followup + 1]
        if (
            not available.iloc[position : position + followup + 1].all()
            or episode["Forward_Tail_Event"].astype(bool).any()
            or (episode["Shock_State"].iloc[1:] == SHOCK_ACTIVE).any()
        ):
            continue
        isolated_shocks += 1
        if (episode["Background_State"] == BACKGROUND_STRESS).any():
            false_persistent += 1
    false_persistent_rate = _safe_ratio(
        false_persistent,
        isolated_shocks,
    )
    guardrail_failures = []
    if len(frame) < plan.min_evaluation_rows:
        guardrail_failures.append("insufficient_evaluation_rows")
    if evaluation_coverage < plan.minimum_evaluation_coverage:
        guardrail_failures.append("incomplete_evaluation_coverage")
    if switches_per_252 > plan.max_switches_per_252:
        guardrail_failures.append("excessive_state_switching")
    if (
        plan.minimum_isolated_shock_episodes > 0
        and isolated_shocks >= plan.minimum_isolated_shock_episodes
        and false_persistent_rate
        > plan.max_false_persistent_isolated_shock_rate
    ):
        guardrail_failures.append("excessive_false_persistence")
    return FoldMetrics(
        fold_id=fold.fold_id,
        candidate_id=candidate_id,
        planned_evaluation_rows=planned_rows,
        evaluation_rows=len(frame),
        evaluation_coverage=evaluation_coverage,
        tail_events=int(tail.sum()),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        tail_recall=recall,
        tail_precision=precision,
        tail_f1=f1,
        ordinal_risk_spearman=spearman,
        switches_per_252=switches_per_252,
        maximum_state_occupancy=maximum_occupancy,
        isolated_shocks=isolated_shocks,
        false_persistent_on_isolated_shock=false_persistent,
        false_persistent_isolated_shock_rate=false_persistent_rate,
        guardrail_passed=not guardrail_failures,
        guardrail_failure_codes=tuple(guardrail_failures),
        state_occupancy=occupancy,
    )


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    config_sha256: str
    fold_metrics: tuple[FoldMetrics, ...]
    planned_evaluation_rows: int
    evaluation_rows: int
    evaluation_coverage: float
    tail_events: int
    tail_recall: float
    tail_precision: float
    tail_f1: float
    mean_ordinal_risk_spearman: float
    mean_switches_per_252: float
    maximum_state_occupancy: float
    isolated_shocks: int
    false_persistent_on_isolated_shock: int
    false_persistent_isolated_shock_rate: float
    selection_score: float
    guardrail_passed: bool
    guardrail_failure_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        _sha256(self.config_sha256, "config_sha256")
        metrics = tuple(self.fold_metrics)
        if not metrics or any(
            item.candidate_id != self.candidate_id for item in metrics
        ):
            raise RegimeCalibrationError(
                "candidate summary fold metrics do not match candidate"
            )
        object.__setattr__(self, "fold_metrics", metrics)
        for field_name in (
            "planned_evaluation_rows",
            "evaluation_rows",
            "tail_events",
            "isolated_shocks",
            "false_persistent_on_isolated_shock",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RegimeCalibrationError(
                    f"{field_name} must be a nonnegative integer"
                )
        for field_name in (
            "tail_recall",
            "tail_precision",
            "tail_f1",
            "mean_ordinal_risk_spearman",
            "mean_switches_per_252",
            "maximum_state_occupancy",
            "evaluation_coverage",
            "false_persistent_isolated_shock_rate",
            "selection_score",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )
        if self.evaluation_rows > self.planned_evaluation_rows:
            raise RegimeCalibrationError(
                "evaluation_rows cannot exceed planned_evaluation_rows"
            )
        if not 0.0 <= self.evaluation_coverage <= 1.0:
            raise RegimeCalibrationError(
                "evaluation_coverage must be in [0, 1]"
            )
        if not 0.0 <= self.maximum_state_occupancy <= 1.0:
            raise RegimeCalibrationError(
                "maximum_state_occupancy must be in [0, 1]"
            )
        if not 0.0 <= self.false_persistent_isolated_shock_rate <= 1.0:
            raise RegimeCalibrationError(
                "false persistent shock rate must be in [0, 1]"
            )
        if not isinstance(self.guardrail_passed, bool):
            raise TypeError("guardrail_passed must be a boolean")
        failure_codes = tuple(
            _identifier(item, "guardrail_failure_code")
            for item in self.guardrail_failure_codes
        )
        if self.guardrail_passed == bool(failure_codes):
            raise RegimeCalibrationError(
                "guardrail status and failure codes are inconsistent"
            )
        object.__setattr__(
            self,
            "guardrail_failure_codes",
            failure_codes,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fold_metrics"] = [
            item.to_dict() for item in self.fold_metrics
        ]
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateSummary":
        expected = {item.name for item in fields(cls)}
        _strict_fields(payload, expected, "candidate_summary")
        values = dict(payload)
        values["fold_metrics"] = tuple(
            FoldMetrics.from_dict(item) for item in values["fold_metrics"]
        )
        return cls(**values)


def _summarize_candidate(
    candidate: CalibrationCandidate,
    metrics: tuple[FoldMetrics, ...],
    plan: RegimeCalibrationPlan,
) -> CandidateSummary:
    true_positive = sum(item.true_positive for item in metrics)
    false_positive = sum(item.false_positive for item in metrics)
    false_negative = sum(item.false_negative for item in metrics)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    mean_spearman = float(
        np.mean([item.ordinal_risk_spearman for item in metrics])
    )
    score = f1 + 0.25 * max(mean_spearman, 0.0)
    planned_rows = sum(item.planned_evaluation_rows for item in metrics)
    evaluation_rows = sum(item.evaluation_rows for item in metrics)
    aggregate_state_counts = {
        state: sum(
            dict(item.state_occupancy).get(state, 0.0)
            * item.evaluation_rows
            for item in metrics
        )
        for state in _BACKGROUND_ORDER
    }
    observed_states = {
        state
        for state, count in aggregate_state_counts.items()
        if count > 0
    }
    maximum_occupancy = max(
        (
            count / evaluation_rows
            for count in aggregate_state_counts.values()
        ),
        default=1.0,
    )
    isolated_shocks = sum(item.isolated_shocks for item in metrics)
    false_persistent = sum(
        item.false_persistent_on_isolated_shock for item in metrics
    )
    false_persistent_rate = _safe_ratio(
        false_persistent,
        isolated_shocks,
    )
    guardrail_failures = []
    if not all(item.guardrail_passed for item in metrics):
        guardrail_failures.append("fold_guardrail_failed")
    if maximum_occupancy > plan.max_state_occupancy:
        guardrail_failures.append("excessive_aggregate_state_occupancy")
    if len(observed_states) < 2:
        guardrail_failures.append("insufficient_aggregate_state_diversity")
    if plan.minimum_isolated_shock_episodes > 0:
        if isolated_shocks < plan.minimum_isolated_shock_episodes:
            guardrail_failures.append(
                "insufficient_isolated_shock_evidence"
            )
        elif (
            false_persistent_rate
            > plan.max_false_persistent_isolated_shock_rate
        ):
            guardrail_failures.append(
                "excessive_aggregate_false_persistence"
            )
    return CandidateSummary(
        candidate_id=candidate.candidate_id,
        config_sha256=candidate.config_sha256,
        fold_metrics=metrics,
        planned_evaluation_rows=planned_rows,
        evaluation_rows=evaluation_rows,
        evaluation_coverage=_safe_ratio(evaluation_rows, planned_rows),
        tail_events=sum(item.tail_events for item in metrics),
        tail_recall=recall,
        tail_precision=precision,
        tail_f1=f1,
        mean_ordinal_risk_spearman=mean_spearman,
        mean_switches_per_252=float(
            np.mean([item.switches_per_252 for item in metrics])
        ),
        maximum_state_occupancy=maximum_occupancy,
        isolated_shocks=isolated_shocks,
        false_persistent_on_isolated_shock=false_persistent,
        false_persistent_isolated_shock_rate=false_persistent_rate,
        selection_score=score,
        guardrail_passed=not guardrail_failures,
        guardrail_failure_codes=tuple(guardrail_failures),
    )


def _research_detector_context(prices: pd.DataFrame) -> dict[str, Any]:
    schedule = regime_market_schedule(
        prices.index.min().date(),
        prices.index.max().date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return {
        "as_of": (
            pd.Timestamp(schedule.iloc[-1]["joint_finalization_at"])
            + pd.Timedelta(minutes=1)
        ),
        "spy_available_at": pd.Series(
            pd.DatetimeIndex(schedule["spy_event_at"]),
            index=sessions,
        ),
        "vix_available_at": pd.Series(
            pd.DatetimeIndex(schedule["vix_event_at"]),
            index=sessions,
        ),
        "source_provenance_verified": False,
    }


def _required_data_end(
    index: pd.DatetimeIndex,
    plan: RegimeCalibrationPlan,
) -> pd.Timestamp:
    target = pd.Timestamp(plan.retrospective_test.evaluation_end)
    positions = np.flatnonzero(index == target)
    if len(positions) != 1:
        raise RegimeCalibrationError(
            "retrospective test end is not present in calibration data"
        )
    resolution_position = (
        int(positions[0]) + plan.max_outcome_resolution_lag
    )
    if resolution_position >= len(index):
        raise RegimeCalibrationError(
            "retrospective test outcomes are not fully resolved"
        )
    return pd.Timestamp(index[resolution_position])


@dataclass(frozen=True)
class RegimeCalibrationArtifact:
    """Deterministic result of one research calibration plan."""

    plan: RegimeCalibrationPlan
    data_manifest: CalibrationDataManifest
    evaluation_as_of_session: date
    raw_best_candidate_id: str
    selected_candidate_id: str
    selection_reason_codes: tuple[str, ...]
    candidate_summaries: tuple[CandidateSummary, ...]
    retrospective_test_metrics: tuple[FoldMetrics, ...]
    promotion_status: str = PROMOTION_RESEARCH_ONLY
    execution_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RegimeCalibrationPlan):
            raise TypeError("plan must be a RegimeCalibrationPlan")
        if not isinstance(self.data_manifest, CalibrationDataManifest):
            raise TypeError(
                "data_manifest must be a CalibrationDataManifest"
            )
        evaluation_as_of = _calendar_date(
            self.evaluation_as_of_session,
            "evaluation_as_of_session",
        )
        summaries = tuple(self.candidate_summaries)
        tests = tuple(self.retrospective_test_metrics)
        summary_ids = {item.candidate_id for item in summaries}
        expected_candidate_order = tuple(
            item.candidate_id for item in self.plan.candidates
        )
        expected_ids = set(expected_candidate_order)
        if (
            len(summaries) != len(expected_ids)
            or summary_ids != expected_ids
            or tuple(item.candidate_id for item in summaries)
            != expected_candidate_order
        ):
            raise RegimeCalibrationError(
                "artifact summaries do not cover the predeclared candidates"
            )
        expected_selection_folds = tuple(
            item.fold_id for item in self.plan.selection_folds
        )
        for summary in summaries:
            candidate = self.plan.candidate(summary.candidate_id)
            if summary.config_sha256 != candidate.config_sha256:
                raise RegimeCalibrationError(
                    "artifact summary config does not match its candidate"
                )
            if tuple(
                item.fold_id for item in summary.fold_metrics
            ) != expected_selection_folds:
                raise RegimeCalibrationError(
                    "artifact summary folds do not match the plan"
                )
            if summary != _summarize_candidate(
                candidate,
                summary.fold_metrics,
                self.plan,
            ):
                raise RegimeCalibrationError(
                    "artifact summary aggregates do not match fold metrics"
                )
        raw_best = _identifier(
            self.raw_best_candidate_id,
            "raw_best_candidate_id",
        )
        selected = _identifier(
            self.selected_candidate_id,
            "selected_candidate_id",
        )
        if raw_best not in summary_ids or selected not in summary_ids:
            raise RegimeCalibrationError(
                "artifact selected candidate is not in its plan"
            )
        reason_codes = tuple(
            _identifier(item, "selection_reason_code")
            for item in self.selection_reason_codes
        )
        if not reason_codes:
            raise RegimeCalibrationError(
                "artifact must contain selection reason codes"
            )
        if (
            len(tests) != len(expected_ids)
            or {item.candidate_id for item in tests} != expected_ids
            or tuple(item.candidate_id for item in tests)
            != expected_candidate_order
        ):
            raise RegimeCalibrationError(
                "retrospective metrics must cover every predeclared candidate"
            )
        if any(
            item.fold_id != self.plan.retrospective_test.fold_id
            for item in tests
        ):
            raise RegimeCalibrationError(
                "retrospective metrics do not match the planned test fold"
            )
        expected_raw, expected_selected, expected_reasons = _choose_candidate(
            self.plan,
            summaries,
        )
        if (
            raw_best != expected_raw
            or selected != expected_selected
            or reason_codes != expected_reasons
        ):
            raise RegimeCalibrationError(
                "artifact selection does not match the frozen selection rule"
            )
        if evaluation_as_of != self.data_manifest.last_session:
            raise RegimeCalibrationError(
                "evaluation_as_of_session must equal the manifest end"
            )
        if self.plan.prospective_holdout_start <= evaluation_as_of:
            raise RegimeCalibrationError(
                "prospective holdout must begin after artifact evaluation"
            )
        if self.promotion_status != PROMOTION_RESEARCH_ONLY:
            raise RegimeCalibrationError(
                "R3 calibration artifacts must remain research_only"
            )
        if self.execution_eligible is not False:
            raise RegimeCalibrationError(
                "calibration artifacts cannot authorize execution"
            )
        object.__setattr__(
            self,
            "evaluation_as_of_session",
            evaluation_as_of,
        )
        object.__setattr__(self, "raw_best_candidate_id", raw_best)
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(self, "selection_reason_codes", reason_codes)
        object.__setattr__(self, "candidate_summaries", summaries)
        object.__setattr__(self, "retrospective_test_metrics", tests)

    @property
    def selected_candidate(self) -> CalibrationCandidate:
        return self.plan.candidate(self.selected_candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "plan": self.plan.to_dict(),
            "plan_sha256": self.plan.plan_sha256,
            "data_manifest": self.data_manifest.to_dict(),
            "evaluation_as_of_session": (
                self.evaluation_as_of_session.isoformat()
            ),
            "raw_best_candidate_id": self.raw_best_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_config_sha256": (
                self.selected_candidate.config_sha256
            ),
            "selection_reason_codes": list(self.selection_reason_codes),
            "candidate_summaries": [
                item.to_dict() for item in self.candidate_summaries
            ],
            "retrospective_test_metrics": [
                item.to_dict() for item in self.retrospective_test_metrics
            ],
            "promotion_status": self.promotion_status,
            "execution_eligible": self.execution_eligible,
        }

    @property
    def artifact_sha256(self) -> str:
        return _domain_sha256(
            _ARTIFACT_HASH_DOMAIN,
            _canonical_json(self.to_dict()),
        )

    def to_json(self) -> str:
        return _canonical_json(
            {
                "artifact": self.to_dict(),
                "artifact_sha256": self.artifact_sha256,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "RegimeCalibrationArtifact":
        if not isinstance(payload, str):
            raise TypeError("calibration artifact JSON must be a string")
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RegimeCalibrationError(
                "calibration artifact is not valid JSON"
            ) from exc
        if _canonical_json(envelope) != payload:
            raise RegimeCalibrationError(
                "calibration artifact JSON must be canonical"
            )
        _strict_fields(
            envelope,
            {"artifact", "artifact_sha256"},
            "calibration_artifact_envelope",
        )
        body = envelope["artifact"]
        expected = {
            "schema_version",
            "plan",
            "plan_sha256",
            "data_manifest",
            "evaluation_as_of_session",
            "raw_best_candidate_id",
            "selected_candidate_id",
            "selected_config_sha256",
            "selection_reason_codes",
            "candidate_summaries",
            "retrospective_test_metrics",
            "promotion_status",
            "execution_eligible",
        }
        _strict_fields(body, expected, "calibration_artifact")
        if body["schema_version"] != CALIBRATION_SCHEMA_VERSION:
            raise RegimeCalibrationError(
                "unsupported calibration artifact schema"
            )
        plan = RegimeCalibrationPlan.from_dict(body["plan"])
        if plan.plan_sha256 != _sha256(
            body["plan_sha256"],
            "plan_sha256",
        ):
            raise RegimeCalibrationError("artifact plan hash does not match")
        artifact = cls(
            plan=plan,
            data_manifest=CalibrationDataManifest.from_dict(
                body["data_manifest"]
            ),
            evaluation_as_of_session=body["evaluation_as_of_session"],
            raw_best_candidate_id=body["raw_best_candidate_id"],
            selected_candidate_id=body["selected_candidate_id"],
            selection_reason_codes=tuple(body["selection_reason_codes"]),
            candidate_summaries=tuple(
                CandidateSummary.from_dict(item)
                for item in body["candidate_summaries"]
            ),
            retrospective_test_metrics=tuple(
                FoldMetrics.from_dict(item)
                for item in body["retrospective_test_metrics"]
            ),
            promotion_status=body["promotion_status"],
            execution_eligible=body["execution_eligible"],
        )
        if artifact.selected_candidate.config_sha256 != _sha256(
            body["selected_config_sha256"],
            "selected_config_sha256",
        ):
            raise RegimeCalibrationError(
                "artifact selected config hash does not match"
            )
        if artifact.artifact_sha256 != _sha256(
            envelope["artifact_sha256"],
            "artifact_sha256",
        ):
            raise RegimeCalibrationError(
                "calibration artifact hash does not match"
            )
        return artifact


def _choose_candidate(
    plan: RegimeCalibrationPlan,
    summaries: tuple[CandidateSummary, ...],
) -> tuple[str, str, tuple[str, ...]]:
    eligible = [item for item in summaries if item.guardrail_passed]
    control = next(
        item
        for item in summaries
        if item.candidate_id == plan.control_candidate_id
    )
    if not control.guardrail_passed:
        failed_folds = {
            item.fold_id: item.guardrail_failure_codes
            for item in control.fold_metrics
            if not item.guardrail_passed
        }
        raise RegimeCalibrationError(
            "predeclared control failed calibration guardrails: "
            f"summary={control.guardrail_failure_codes}, "
            f"folds={failed_folds}"
        )
    if not eligible:
        raise RegimeCalibrationError(
            "every calibration candidate failed guardrails"
        )
    raw_best = sorted(
        eligible,
        key=lambda item: (-item.selection_score, item.candidate_id),
    )[0]
    if raw_best.candidate_id == control.candidate_id:
        return (
            raw_best.candidate_id,
            control.candidate_id,
            ("control_ranked_first",),
        )
    improvement = raw_best.selection_score - control.selection_score
    if improvement < plan.minimum_material_improvement:
        return (
            raw_best.candidate_id,
            control.candidate_id,
            ("challenger_improvement_not_material", "control_retained"),
        )
    return (
        raw_best.candidate_id,
        raw_best.candidate_id,
        ("challenger_materially_improved",),
    )


def run_research_calibration(
    prices: pd.DataFrame,
    plan: RegimeCalibrationPlan,
    *,
    provenance_status: str = PROVENANCE_LEGACY_UNVERIFIED,
) -> RegimeCalibrationArtifact:
    """Run a deterministic purged calibration on an exact price prefix.

    This entry point deliberately rejects claims of provider-verified input.
    A future production calibration path must start from an R2 snapshot and its
    decision-time evidence manifest rather than a caller-provided DataFrame.
    """

    if not isinstance(plan, RegimeCalibrationPlan):
        raise TypeError("plan must be a RegimeCalibrationPlan")
    if plan.detector_code_sha256 != regime_detector_code_sha256():
        raise RegimeCalibrationError(
            "calibration plan detector code hash is stale"
        )
    if plan.calibration_code_sha256 != regime_calibration_code_sha256():
        raise RegimeCalibrationError(
            "calibration plan engine code hash is stale"
        )
    if provenance_status != PROVENANCE_LEGACY_UNVERIFIED:
        raise RegimeCalibrationError(
            "research calibration accepts only explicitly unverified data"
        )
    frame = _validated_prices(prices)
    cutoff = _required_data_end(frame.index, plan)
    scoped = frame.loc[:cutoff].copy()
    outcomes = build_forward_risk_outcomes(scoped, plan.outcome_horizons)
    context = _research_detector_context(scoped)
    traces: dict[str, pd.DataFrame] = {}
    summaries: list[CandidateSummary] = []
    for candidate in plan.candidates:
        trace = detect_regimes(scoped, candidate.config, **context)
        traces[candidate.candidate_id] = trace
        fold_metrics = tuple(
            evaluate_candidate_fold(
                trace,
                outcomes,
                candidate.candidate_id,
                fold,
                plan,
            )
            for fold in plan.selection_folds
        )
        summaries.append(
            _summarize_candidate(candidate, fold_metrics, plan)
        )
    summary_tuple = tuple(summaries)
    raw_best, selected, reason_codes = _choose_candidate(
        plan,
        summary_tuple,
    )
    retrospective_ids = tuple(
        candidate.candidate_id for candidate in plan.candidates
    )
    retrospective_metrics = tuple(
        evaluate_candidate_fold(
            traces[candidate_id],
            outcomes,
            candidate_id,
            plan.retrospective_test,
            plan,
        )
        for candidate_id in retrospective_ids
    )
    return RegimeCalibrationArtifact(
        plan=plan,
        data_manifest=build_calibration_data_manifest(
            scoped,
            provenance_status=provenance_status,
        ),
        evaluation_as_of_session=cutoff.date(),
        raw_best_candidate_id=raw_best,
        selected_candidate_id=selected,
        selection_reason_codes=reason_codes,
        candidate_summaries=summary_tuple,
        retrospective_test_metrics=retrospective_metrics,
    )


def detect_regimes_with_calibration_artifact(
    prices: pd.DataFrame,
    artifact: RegimeCalibrationArtifact,
    *,
    expected_artifact_sha256: str,
    as_of,
    spy_available_at,
    vix_available_at,
) -> pd.DataFrame:
    """Run the selected research profile and attach its immutable lineage."""

    if not isinstance(artifact, RegimeCalibrationArtifact):
        raise TypeError("artifact must be a RegimeCalibrationArtifact")
    expected_hash = _sha256(
        expected_artifact_sha256,
        "expected_artifact_sha256",
    )
    if artifact.artifact_sha256 != expected_hash:
        raise RegimeCalibrationError(
            "calibration artifact does not match the deployment-pinned hash"
        )
    if artifact.plan.detector_version != DETECTOR_VERSION:
        raise RegimeCalibrationError(
            "calibration artifact detector version is stale"
        )
    if artifact.plan.detector_code_sha256 != regime_detector_code_sha256():
        raise RegimeCalibrationError(
            "calibration artifact detector code hash is stale"
        )
    frame = _validated_prices(prices)
    manifest_end = pd.Timestamp(artifact.data_manifest.last_session)
    if (
        frame.index.min().date() != artifact.data_manifest.first_session
        or manifest_end not in frame.index
    ):
        raise RegimeCalibrationError(
            "runtime history does not contain the calibrated data prefix"
        )
    runtime_prefix_manifest = build_calibration_data_manifest(
        frame.loc[:manifest_end],
        provenance_status=artifact.data_manifest.provenance_status,
        snapshot_sha256=artifact.data_manifest.snapshot_sha256,
        evidence_manifest_sha256=(
            artifact.data_manifest.evidence_manifest_sha256
        ),
        evidence_verification_kind=(
            artifact.data_manifest.evidence_verification_kind
        ),
    )
    if runtime_prefix_manifest != artifact.data_manifest:
        raise RegimeCalibrationError(
            "runtime history does not match the calibrated data manifest"
        )
    result = detect_regimes(
        frame,
        artifact.selected_candidate.config,
        as_of=as_of,
        spy_available_at=spy_available_at,
        vix_available_at=vix_available_at,
        source_provenance_verified=False,
    )
    result["Calibration_Artifact_SHA256"] = artifact.artifact_sha256
    result["Calibration_Plan_SHA256"] = artifact.plan.plan_sha256
    result["Calibration_Profile"] = artifact.selected_candidate_id
    result["Calibration_Status"] = artifact.promotion_status
    result["Calibration_Provenance"] = (
        artifact.data_manifest.provenance_status
    )
    result["Calibration_Prospective_Holdout_Start"] = (
        artifact.plan.prospective_holdout_start.isoformat()
    )
    result["Calibration_Abstain"] = True
    result["Calibration_Abstain_Reasons"] = (
        "legacy_calibration_provenance_unverified"
        "|prospective_holdout_incomplete"
        "|research_only"
    )
    result["Calibration_Execution_Eligible"] = False
    result["Execution_Eligible"] = False
    result.attrs["calibration_artifact_sha256"] = artifact.artifact_sha256
    result.attrs["calibration_plan_sha256"] = artifact.plan.plan_sha256
    result.attrs["calibration_profile"] = artifact.selected_candidate_id
    result.attrs["calibration_status"] = artifact.promotion_status
    result.attrs["calibration_provenance"] = (
        artifact.data_manifest.provenance_status
    )
    result.attrs["calibration_execution_eligible"] = False
    result.attrs["calibration_abstain"] = True
    result.attrs["calibration_abstain_reasons"] = (
        "legacy_calibration_provenance_unverified",
        "prospective_holdout_incomplete",
        "research_only",
    )
    result.attrs["execution_eligible"] = False
    result.attrs["threshold_status"] = (
        "retrospective_research_profile_unverified"
    )
    return result


__all__ = [
    "CALIBRATION_METHOD_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationCandidate",
    "CalibrationDataManifest",
    "CandidateSummary",
    "CausalValidationFold",
    "FoldMetrics",
    "PROVENANCE_LEGACY_UNVERIFIED",
    "PROVENANCE_PROVIDER_DECISION_TIME",
    "PROMOTION_RESEARCH_ONLY",
    "RegimeCalibrationArtifact",
    "RegimeCalibrationError",
    "RegimeCalibrationPlan",
    "build_calibration_data_manifest",
    "build_forward_risk_outcomes",
    "detect_regimes_with_calibration_artifact",
    "evaluate_candidate_fold",
    "regime_calibration_code_sha256",
    "run_research_calibration",
]
