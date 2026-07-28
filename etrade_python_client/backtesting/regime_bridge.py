"""Causal, typed regime inputs for historical backtests.

The final stress overlay and raw HMM states are deliberately separate:

* final overlay states may only veto or reduce risk;
* raw HMM states may only select a return bucket from their exact taxonomy.

Nothing in this module is live-order capable.  All records remain explicitly
uncertified until the repository's prospective validation gate is complete.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from live_trading.market_sessions import prior_nyse_session_map
from live_trading.regime_taxonomy import (
    FINAL_RISK_LABELS,
    FINAL_RISK_NAMESPACE,
    FinalRiskRegimeRef,
    RawHMMStateRef,
    RegimeReturnBuckets,
    RegimeTaxonomyError,
)


PROTOCOL_SCHEMA = "backtest-regime-protocol.v1"
DECISION_EVIDENCE_SCHEMA = "backtest-regime-decision-evidence.v1"
SIGNAL_TIMESTAMP = "close_T_for_next_session"
INFERENCE_METHODS = {
    "walk_forward_expanding",
    "external_exact_lagged_overlay",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RegimeBridgeUnavailable(RuntimeError):
    """A causal regime input could not be established."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _iso_date(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a canonical ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must be a canonical ISO date")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _calendar_days_to_trading_days(calendar_days: int) -> int:
    return max(1, int(round(float(calendar_days) * 252.0 / 365.0)))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class BacktestRegimeProtocol:
    """Immutable causal contract for one regime-aware backtest run."""

    calibration_end: str
    test_start: str
    test_end: str
    inference_method: str
    raw_hmm_components: int = 3
    signal_timestamp: str = SIGNAL_TIMESTAMP
    regime_lag_trading_sessions: int = 1
    validity_status: str = "UNVERIFIED"
    execution_eligible: bool = False
    schema: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        calibration_end = _iso_date(self.calibration_end, "calibration_end")
        test_start = _iso_date(self.test_start, "test_start")
        test_end = _iso_date(self.test_end, "test_end")
        if not calibration_end < test_start <= test_end:
            raise ValueError(
                "calibration_end must precede the ordered test date range"
            )
        if self.inference_method not in INFERENCE_METHODS:
            raise ValueError("unsupported regime inference_method")
        if (
            type(self.raw_hmm_components) is not int
            or self.raw_hmm_components < 2
        ):
            raise ValueError("raw_hmm_components must be an integer of at least two")
        if self.signal_timestamp != SIGNAL_TIMESTAMP:
            raise ValueError("regime signal timestamp must be close_T_for_next_session")
        if self.regime_lag_trading_sessions != 1:
            raise ValueError("regime values must be lagged exactly one NYSE session")
        if self.validity_status != "UNVERIFIED":
            raise ValueError("backtest regime protocol must remain UNVERIFIED")
        if self.execution_eligible is not False:
            raise ValueError("backtest regime protocol cannot authorize execution")
        if self.schema != PROTOCOL_SCHEMA:
            raise ValueError("unsupported backtest regime protocol schema")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RegimeDecisionEvidence:
    """Immutable evidence used for one trade-entry decision."""

    entry_session: str
    signal_as_of_session: str
    protocol_sha256: str
    final_risk_regime: Optional[FinalRiskRegimeRef]
    raw_hmm_state: Optional[RawHMMStateRef] = None
    model_training_end: str = ""
    raw_inference_method: str = ""
    return_bucket_as_of: str = ""
    resolved_outcomes_through: str = ""
    horizon_calendar_days: Optional[int] = None
    horizon_trading_days: Optional[int] = None
    return_bucket_manifest_sha256: str = ""
    assignment_probability: Optional[float] = None
    raw_context_required: bool = False
    validity_status: str = "UNVERIFIED"
    unavailable_code: str = ""
    execution_eligible: bool = False
    schema: str = DECISION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        entry = _iso_date(self.entry_session, "entry_session")
        signal = _iso_date(self.signal_as_of_session, "signal_as_of_session")
        if not signal < entry:
            raise ValueError("regime signal must precede the entry session")
        if _SHA256.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256")
        if (
            self.final_risk_regime is not None
            and type(self.final_risk_regime) is not FinalRiskRegimeRef
        ):
            raise ValueError("final_risk_regime must be a FinalRiskRegimeRef")
        if self.raw_hmm_state is not None:
            if type(self.raw_hmm_state) is not RawHMMStateRef:
                raise ValueError("raw_hmm_state must be a RawHMMStateRef")
            training_end = _iso_date(
                self.model_training_end,
                "model_training_end",
            )
            bucket_as_of = _iso_date(
                self.return_bucket_as_of,
                "return_bucket_as_of",
            )
            resolved_through = _iso_date(
                self.resolved_outcomes_through,
                "resolved_outcomes_through",
            )
            if not training_end <= signal:
                raise ValueError("raw HMM training cannot follow its signal")
            if bucket_as_of != signal:
                raise ValueError("return bucket as-of must equal the signal session")
            if not resolved_through < signal:
                raise ValueError(
                    "return outcomes must resolve before the signal session"
                )
            if not self.raw_inference_method:
                raise ValueError("raw inference method is required")
            if (
                type(self.horizon_calendar_days) is not int
                or self.horizon_calendar_days < 1
                or type(self.horizon_trading_days) is not int
                or self.horizon_trading_days < 1
            ):
                raise ValueError("raw return horizons must be positive integers")
            if _SHA256.fullmatch(self.return_bucket_manifest_sha256) is None:
                raise ValueError(
                    "return_bucket_manifest_sha256 must be a lowercase SHA-256"
                )
        elif any(
            (
                self.model_training_end,
                self.raw_inference_method,
                self.return_bucket_as_of,
                self.resolved_outcomes_through,
                self.return_bucket_manifest_sha256,
            )
        ) or self.horizon_calendar_days is not None or self.horizon_trading_days is not None:
            raise ValueError("raw-HMM provenance cannot exist without a raw state")
        if self.assignment_probability is not None:
            if self.raw_hmm_state is None:
                raise ValueError("assignment probability requires raw-HMM evidence")
            if (
                type(self.assignment_probability) not in {int, float}
                or not math.isfinite(float(self.assignment_probability))
                or not 0.0 <= float(self.assignment_probability) <= 1.0
            ):
                raise ValueError("assignment_probability must be finite in [0, 1]")
        unavailable = bool(self.unavailable_code) or (
            self.final_risk_regime is None
            or (
                self.raw_context_required
                and (
                    self.raw_hmm_state is None
                    or self.assignment_probability is None
                )
            )
        )
        expected_status = "UNAVAILABLE" if unavailable else "UNVERIFIED"
        if self.validity_status != expected_status:
            raise ValueError(
                f"regime decision validity_status must be {expected_status}"
            )
        structurally_unavailable = (
            self.final_risk_regime is None
            or (
                self.raw_context_required
                and (
                    self.raw_hmm_state is None
                    or self.assignment_probability is None
                )
            )
        )
        if structurally_unavailable and not self.unavailable_code:
            raise ValueError(
                "unavailable regime evidence requires a stable code"
            )
        if self.execution_eligible is not False:
            raise ValueError("regime decision evidence cannot authorize execution")
        if self.schema != DECISION_EVIDENCE_SCHEMA:
            raise ValueError("unsupported regime decision evidence schema")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LaggedFinalRiskMap:
    """Exact T-1 final overlay evidence for requested entry sessions."""

    by_entry_session: Mapping[str, FinalRiskRegimeRef]
    signal_session_by_entry: Mapping[str, str]
    unavailable_code_by_entry: Mapping[str, str]

    def __post_init__(self) -> None:
        available = dict(self.by_entry_session)
        signals = dict(self.signal_session_by_entry)
        unavailable = dict(self.unavailable_code_by_entry)
        if set(available).intersection(unavailable):
            raise ValueError(
                "available and unavailable final-regime entries must be disjoint"
            )
        if set(signals) != set(available).union(unavailable):
            raise ValueError(
                "signal sessions must cover exactly every requested entry"
            )
        for entry_session, signal_session in signals.items():
            entry = _iso_date(entry_session, "entry_session")
            signal = _iso_date(signal_session, "signal_session")
            if not signal < entry:
                raise ValueError("final-regime signal must precede entry")
        for entry_session, value in available.items():
            _iso_date(entry_session, "entry_session")
            if type(value) is not FinalRiskRegimeRef:
                raise ValueError(
                    "final regime values must be FinalRiskRegimeRef"
                )
        for entry_session, code in unavailable.items():
            _iso_date(entry_session, "entry_session")
            if (
                type(code) is not str
                or re.fullmatch(r"[A-Z0-9_]{3,96}", code) is None
            ):
                raise ValueError(
                    "unavailable final-regime entries require stable codes"
                )
        object.__setattr__(
            self,
            "by_entry_session",
            MappingProxyType(available),
        )
        object.__setattr__(
            self,
            "signal_session_by_entry",
            MappingProxyType(signals),
        )
        object.__setattr__(
            self,
            "unavailable_code_by_entry",
            MappingProxyType(unavailable),
        )


def build_lagged_final_risk_map(
    feature_df: pd.DataFrame,
    entry_sessions: list[str],
) -> LaggedFinalRiskMap:
    """Map each entry session to the final overlay on its exact prior session.

    Missing rows stay missing.  There is deliberately no forward fill or raw
    HMM fallback.
    """

    required = {"Detected_Regime_State", "Detected_Regime_Label"}
    if feature_df is None or feature_df.empty:
        raise RegimeBridgeUnavailable("FINAL_OVERLAY_TRACE_UNAVAILABLE")
    if not required.issubset(feature_df.columns):
        raise RegimeBridgeUnavailable("FINAL_OVERLAY_COLUMNS_REQUIRED")
    try:
        frame = feature_df.loc[
            :,
            ["Detected_Regime_State", "Detected_Regime_Label"],
        ].copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
    except (TypeError, ValueError) as error:
        raise RegimeBridgeUnavailable("INVALID_FINAL_OVERLAY_INDEX") from error
    if (
        frame.index.tz is not None
        or not frame.index.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        raise RegimeBridgeUnavailable("INVALID_FINAL_OVERLAY_INDEX")
    try:
        prior_sessions = prior_nyse_session_map(entry_sessions)
    except Exception as error:
        code = getattr(error, "code", "PRIOR_NYSE_SESSION_UNAVAILABLE")
        raise RegimeBridgeUnavailable(code) from error

    exact_rows = {
        timestamp.strftime("%Y-%m-%d"): row
        for timestamp, row in frame.iterrows()
    }
    effective: dict[str, FinalRiskRegimeRef] = {}
    signal_by_entry: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    for entry_session in entry_sessions:
        signal_session = prior_sessions[entry_session]
        signal_by_entry[entry_session] = signal_session
        row = exact_rows.get(signal_session)
        if row is None:
            unavailable[entry_session] = "FINAL_OVERLAY_PRIOR_SESSION_MISSING"
            continue
        raw_state = row["Detected_Regime_State"]
        raw_label = row["Detected_Regime_Label"]
        try:
            numeric_state = float(raw_state)
            if (
                not math.isfinite(numeric_state)
                or numeric_state != math.floor(numeric_state)
            ):
                raise ValueError
            state = int(numeric_state)
            if type(raw_label) is not str:
                raise ValueError
            effective[entry_session] = FinalRiskRegimeRef(
                state=state,
                label=raw_label,
            )
        except (TypeError, ValueError, RegimeTaxonomyError):
            unavailable[entry_session] = "INVALID_FINAL_OVERLAY_VALUE"
    return LaggedFinalRiskMap(
        by_entry_session=effective,
        signal_session_by_entry=signal_by_entry,
        unavailable_code_by_entry=unavailable,
    )


@dataclass(frozen=True, slots=True)
class ResolvedRawRegime:
    """Validated raw-state bucket context for one signal date and horizon."""

    state_ref: RawHMMStateRef
    return_bucket: tuple[float, ...]
    model_training_end: str
    inference_method: str
    bucket_as_of: str
    resolved_outcomes_through: str
    horizon_calendar_days: int
    horizon_trading_days: int
    bucket_manifest_sha256: str
    bucket_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.state_ref) is not RawHMMStateRef:
            raise ValueError("state_ref must be a RawHMMStateRef")
        if type(self.return_bucket) is not tuple or not self.return_bucket:
            raise ValueError("return_bucket must be a non-empty tuple")
        manifest = dict(self.bucket_manifest)
        if _SHA256.fullmatch(self.bucket_manifest_sha256) is None:
            raise ValueError("bucket manifest SHA-256 is invalid")
        if _canonical_sha256(manifest) != self.bucket_manifest_sha256:
            raise ValueError("bucket manifest does not match its SHA-256")
        object.__setattr__(
            self,
            "bucket_manifest",
            _deep_freeze(manifest),
        )


@dataclass(frozen=True, slots=True)
class RawRegimeResolution:
    value: Optional[ResolvedRawRegime] = None
    unavailable_code: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) == (not self.unavailable_code):
            raise ValueError("raw regime resolution must be success or failure")


@dataclass(frozen=True, slots=True)
class _RawCacheKey:
    protocol_sha256: str
    signal_as_of: str
    horizon_calendar_days: int
    horizon_trading_days: int
    n_components: int


class ExactRegimeRunCache:
    """Run-local cache whose exact key retains successes and failures."""

    def __init__(
        self,
        protocol: BacktestRegimeProtocol,
        *,
        minimum_bucket_observations: int = 10,
    ) -> None:
        if type(protocol) is not BacktestRegimeProtocol:
            raise TypeError("protocol must be a BacktestRegimeProtocol")
        if (
            type(minimum_bucket_observations) is not int
            or minimum_bucket_observations < 1
        ):
            raise ValueError("minimum_bucket_observations must be positive")
        self.protocol = protocol
        self.minimum_bucket_observations = minimum_bucket_observations
        self._cache: dict[_RawCacheKey, RawRegimeResolution] = {}

    def resolve(
        self,
        *,
        signal_as_of: str,
        horizon_calendar_days: int,
        n_components: int,
        builder: Callable[..., Any],
    ) -> RawRegimeResolution:
        _iso_date(signal_as_of, "signal_as_of")
        if (
            type(horizon_calendar_days) is not int
            or horizon_calendar_days < 1
        ):
            return RawRegimeResolution(
                unavailable_code="INVALID_REGIME_BUCKET_HORIZON"
            )
        if type(n_components) is not int or n_components < 2:
            return RawRegimeResolution(
                unavailable_code="INVALID_HMM_COMPONENT_COUNT"
            )
        if n_components != self.protocol.raw_hmm_components:
            return RawRegimeResolution(
                unavailable_code="HMM_COMPONENT_PROTOCOL_MISMATCH"
            )
        trading_horizon = _calendar_days_to_trading_days(
            horizon_calendar_days
        )
        key = _RawCacheKey(
            protocol_sha256=self.protocol.sha256,
            signal_as_of=signal_as_of,
            horizon_calendar_days=horizon_calendar_days,
            horizon_trading_days=trading_horizon,
            n_components=n_components,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            built = builder(
                date.fromisoformat(signal_as_of).toordinal(),
                horizon=horizon_calendar_days,
                n_components=n_components,
                force_refit=False,
                as_of_date=signal_as_of,
            )
            resolved = self._validate_built_result(
                built,
                signal_as_of=signal_as_of,
                horizon_calendar_days=horizon_calendar_days,
                horizon_trading_days=trading_horizon,
                n_components=n_components,
            )
            resolution = RawRegimeResolution(value=resolved)
        except Exception as error:
            resolution = RawRegimeResolution(
                unavailable_code=_stable_unavailable_code(error)
            )
        self._cache[key] = resolution
        return resolution

    def _validate_built_result(
        self,
        built: Any,
        *,
        signal_as_of: str,
        horizon_calendar_days: int,
        horizon_trading_days: int,
        n_components: int,
    ) -> ResolvedRawRegime:
        if not isinstance(built, tuple) or len(built) != 3:
            raise RegimeBridgeUnavailable("INVALID_REGIME_BUCKET_RESULT")
        buckets, model, _daily_models = built
        if type(buckets) is not RegimeReturnBuckets:
            raise RegimeBridgeUnavailable("TYPED_REGIME_BUCKETS_REQUIRED")
        if model is None:
            raise RegimeBridgeUnavailable("HMM_MODEL_UNAVAILABLE")
        taxonomy_id = getattr(model, "raw_hmm_taxonomy_id_", "")
        if _SHA256.fullmatch(str(taxonomy_id)) is None:
            raise RegimeBridgeUnavailable("MODEL_TAXONOMY_UNAVAILABLE")
        if buckets.taxonomy_id != taxonomy_id:
            raise RegimeBridgeUnavailable("REGIME_TAXONOMY_MISMATCH")
        model_components = getattr(model, "n_components", None)
        if (
            type(model_components) is not int
            or model_components != n_components
            or buckets.state_count != model_components
        ):
            raise RegimeBridgeUnavailable("REGIME_BUCKET_MODEL_BINDING_MISMATCH")
        training_end = getattr(model, "model_training_end_", "")
        if (
            buckets.model_training_end != training_end
            or not training_end
            or training_end > signal_as_of
        ):
            raise RegimeBridgeUnavailable("REGIME_BUCKET_MODEL_BINDING_MISMATCH")
        if (
            buckets.inference_as_of != signal_as_of
            or getattr(model, "causal_tail_as_of_", "") != signal_as_of
        ):
            raise RegimeBridgeUnavailable("REGIME_BUCKET_AS_OF_MISMATCH")
        if (
            buckets.horizon_calendar_days != horizon_calendar_days
            or buckets.horizon_trading_days != horizon_trading_days
        ):
            raise RegimeBridgeUnavailable("REGIME_BUCKET_HORIZON_MISMATCH")
        if (
            buckets.validity_status != "UNVERIFIED"
            or buckets.execution_eligible
        ):
            raise RegimeBridgeUnavailable("UNCERTIFIED_BUCKET_CONTRACT_INVALID")
        inference_method = getattr(model, "causal_inference_mode_", "")
        if type(inference_method) is not str or not inference_method:
            raise RegimeBridgeUnavailable("RAW_INFERENCE_METHOD_UNAVAILABLE")
        try:
            probabilities = np.asarray(
                getattr(model, "causal_tail_probability_"),
                dtype=float,
            ).reshape(-1)
        except (AttributeError, TypeError, ValueError) as error:
            raise RegimeBridgeUnavailable(
                "CAUSAL_TAIL_PROBABILITY_UNAVAILABLE"
            ) from error
        if (
            len(probabilities) != model_components
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.isclose(float(probabilities.sum()), 1.0)
        ):
            raise RegimeBridgeUnavailable("INVALID_CAUSAL_TAIL_PROBABILITY")
        state_ref = RawHMMStateRef(
            taxonomy_id=taxonomy_id,
            state=int(np.argmax(probabilities)),
        )
        try:
            return_bucket = buckets.bucket_for(state_ref)
        except RegimeTaxonomyError as error:
            raise RegimeBridgeUnavailable("REGIME_BUCKET_LOOKUP_FAILED") from error
        if len(return_bucket) < self.minimum_bucket_observations:
            raise RegimeBridgeUnavailable(
                f"INSUFFICIENT_REGIME_BUCKET_STATE_{state_ref.state}"
            )
        manifest = dict(buckets.manifest())
        manifest_sha256 = _canonical_sha256(manifest)
        return ResolvedRawRegime(
            state_ref=state_ref,
            return_bucket=return_bucket,
            model_training_end=training_end,
            inference_method=inference_method,
            bucket_as_of=buckets.inference_as_of,
            resolved_outcomes_through=buckets.resolved_outcomes_through,
            horizon_calendar_days=buckets.horizon_calendar_days,
            horizon_trading_days=buckets.horizon_trading_days,
            bucket_manifest_sha256=manifest_sha256,
            bucket_manifest=manifest,
        )


@dataclass(frozen=True, slots=True)
class AssignmentTarget:
    target_log_return: float
    fitted_model: Any


@dataclass(frozen=True, slots=True)
class AssignmentTargetResolution:
    value: Optional[AssignmentTarget] = None
    unavailable_code: str = ""

    def __post_init__(self) -> None:
        if (self.value is None) == (not self.unavailable_code):
            raise ValueError("assignment target must be success or failure")


@dataclass(frozen=True, slots=True)
class _AssignmentCacheKey:
    bucket_manifest_sha256: str
    taxonomy_id: str
    raw_state: int
    target_assignment_probability: float


class ExactAssignmentTargetCache:
    """Exact run-local GMM/root cache, including failed fits and roots."""

    def __init__(self) -> None:
        self._cache: dict[
            _AssignmentCacheKey,
            AssignmentTargetResolution,
        ] = {}

    def resolve(
        self,
        *,
        raw: ResolvedRawRegime,
        target_assignment_probability: float,
        fit_model: Callable[..., Any],
        query_probability: Callable[..., float],
        solve_root: Callable[..., float],
    ) -> AssignmentTargetResolution:
        if type(raw) is not ResolvedRawRegime:
            raise TypeError("raw must be a ResolvedRawRegime")
        target = float(target_assignment_probability)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            return AssignmentTargetResolution(
                unavailable_code="INVALID_TARGET_ASSIGNMENT_PROBABILITY"
            )
        key = _AssignmentCacheKey(
            bucket_manifest_sha256=raw.bucket_manifest_sha256,
            taxonomy_id=raw.state_ref.taxonomy_id,
            raw_state=raw.state_ref.state,
            target_assignment_probability=target,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            model = fit_model(
                raw.return_bucket,
                regime_label=(
                    f"RawHMM_{raw.state_ref.taxonomy_id[:12]}_"
                    f"State_{raw.state_ref.state}"
                ),
            )

            def objective(log_return: float) -> float:
                probability = float(
                    query_probability(model, 1.0, math.exp(log_return))
                )
                if not math.isfinite(probability):
                    raise RegimeBridgeUnavailable(
                        "INVALID_ASSIGNMENT_PROBABILITY"
                    )
                return probability - target

            root = float(solve_root(objective, -1.0, 0.5))
            if not math.isfinite(root) or not -1.0 <= root <= 0.5:
                raise RegimeBridgeUnavailable(
                    "INVALID_ASSIGNMENT_TARGET_ROOT"
                )
            fitted_probability = float(
                query_probability(model, 1.0, math.exp(root))
            )
            if (
                not math.isfinite(fitted_probability)
                or not 0.0 <= fitted_probability <= 1.0
                or not math.isclose(
                    fitted_probability,
                    target,
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )
            ):
                raise RegimeBridgeUnavailable(
                    "ASSIGNMENT_TARGET_NOT_ATTAINED"
                )
            resolution = AssignmentTargetResolution(
                value=AssignmentTarget(
                    target_log_return=root,
                    fitted_model=model,
                )
            )
        except Exception as error:
            resolution = AssignmentTargetResolution(
                unavailable_code=_stable_unavailable_code(error)
            )
        self._cache[key] = resolution
        return resolution


def make_regime_decision_evidence(
    *,
    protocol: BacktestRegimeProtocol,
    entry_session: str,
    signal_as_of_session: str,
    final_risk_regime: Optional[FinalRiskRegimeRef],
    raw_resolution: Optional[RawRegimeResolution] = None,
    raw_context_required: bool = False,
    assignment_probability: Optional[float] = None,
    unavailable_code: str = "",
) -> RegimeDecisionEvidence:
    """Combine independently typed final-overlay and raw-HMM evidence."""

    raw = raw_resolution.value if raw_resolution is not None else None
    if not unavailable_code:
        if final_risk_regime is None:
            unavailable_code = "FINAL_REGIME_UNAVAILABLE"
        elif raw_context_required and raw is None:
            unavailable_code = (
                raw_resolution.unavailable_code
                if raw_resolution is not None
                else "RAW_REGIME_CONTEXT_UNAVAILABLE"
            )
        elif raw_context_required and assignment_probability is None:
            unavailable_code = "ASSIGNMENT_PROBABILITY_UNAVAILABLE"
    unavailable = bool(unavailable_code)
    return RegimeDecisionEvidence(
        entry_session=entry_session,
        signal_as_of_session=signal_as_of_session,
        protocol_sha256=protocol.sha256,
        final_risk_regime=final_risk_regime,
        raw_hmm_state=raw.state_ref if raw is not None else None,
        model_training_end=raw.model_training_end if raw is not None else "",
        raw_inference_method=raw.inference_method if raw is not None else "",
        return_bucket_as_of=raw.bucket_as_of if raw is not None else "",
        resolved_outcomes_through=(
            raw.resolved_outcomes_through if raw is not None else ""
        ),
        horizon_calendar_days=(
            raw.horizon_calendar_days if raw is not None else None
        ),
        horizon_trading_days=(
            raw.horizon_trading_days if raw is not None else None
        ),
        return_bucket_manifest_sha256=(
            raw.bucket_manifest_sha256 if raw is not None else ""
        ),
        assignment_probability=assignment_probability,
        raw_context_required=raw_context_required,
        validity_status="UNAVAILABLE" if unavailable else "UNVERIFIED",
        unavailable_code=unavailable_code,
    )


def _stable_unavailable_code(error: Exception) -> str:
    code = getattr(error, "code", "")
    if type(code) is str and re.fullmatch(r"[A-Z0-9_]{3,96}", code):
        return code
    if isinstance(error, RegimeTaxonomyError):
        return "REGIME_TAXONOMY_ERROR"
    return "REGIME_CONTEXT_BUILD_FAILED"
