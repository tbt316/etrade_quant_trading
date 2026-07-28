"""Typed identities for raw HMM states and final risk overlays.

Raw HMM state numbers are meaningful only inside the exact fitted model that
defined their ordering.  Final stress overlays belong to a different
namespace.  This module makes crossing those namespaces an explicit error.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np


TAXONOMY_SCHEMA = "raw-hmm-taxonomy.v1"
RETURN_BUCKET_SCHEMA = "raw-hmm-return-buckets.v1"
FINAL_RISK_NAMESPACE = "causal-stress-overlay.v1"
FINAL_RISK_LABELS = {
    0: "Expansion (0)",
    1: "Cautious Decline (1)",
    2: "Panic / Crisis (2)",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RegimeTaxonomyError(ValueError):
    """A regime value is not bound to its exact semantic namespace."""


def _iso_date(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise RegimeTaxonomyError(f"{field_name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise RegimeTaxonomyError(f"{field_name} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise RegimeTaxonomyError(f"{field_name} must be a canonical ISO date")
    return value


def _sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RegimeTaxonomyError(f"{field_name} must be a lowercase SHA-256")
    return value


def _exact_state(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise RegimeTaxonomyError(f"{field_name} must be a non-negative integer")
    return value


def _array_digest(value: object, field_name: str) -> str:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RegimeTaxonomyError(f"{field_name} must be numeric") from error
    if array.size == 0 or not np.isfinite(array).all():
        raise RegimeTaxonomyError(f"{field_name} must be finite and non-empty")
    canonical = np.ascontiguousarray(array.astype(">f8", copy=False))
    shape = json.dumps(
        list(canonical.shape),
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(field_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(shape)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_array(value: object, field_name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RegimeTaxonomyError(
            f"{field_name} must be a regular numeric array"
        ) from error
    if array.size == 0 or not np.isfinite(array).all():
        raise RegimeTaxonomyError(
            f"{field_name} must be finite and non-empty"
        )
    return array


def build_raw_hmm_taxonomy_id(
    *,
    n_components: int,
    feature_manifest_sha256: str,
    training_end: str,
    pipeline_version: str,
    state_labels: Sequence[str],
    means: object,
    covariances: object,
    transition_matrix: object,
    start_probabilities: object,
    mixture_weights: object | None = None,
) -> str:
    """Hash the exact fitted parameters and ordered state semantics."""

    if type(n_components) is not int or n_components < 2:
        raise RegimeTaxonomyError("n_components must be an integer of at least two")
    feature_hash = _sha256(
        feature_manifest_sha256,
        "feature_manifest_sha256",
    )
    cutoff = _iso_date(training_end, "training_end")
    if (
        type(pipeline_version) is not str
        or not pipeline_version
        or len(pipeline_version) > 128
    ):
        raise RegimeTaxonomyError("pipeline_version must be a bounded string")
    if (
        not isinstance(state_labels, Sequence)
        or isinstance(state_labels, (str, bytes))
        or len(state_labels) != n_components
        or any(type(label) is not str or not label for label in state_labels)
    ):
        raise RegimeTaxonomyError(
            "state_labels must bind every ordered HMM state"
        )
    parameter_arrays = {
        "means": _numeric_array(means, "means"),
        "covariances": _numeric_array(
            covariances,
            "covariances",
        ),
        "transition_matrix": _numeric_array(
            transition_matrix,
            "transition_matrix",
        ),
        "start_probabilities": _numeric_array(
            start_probabilities,
            "start_probabilities",
        ),
    }
    if (
        parameter_arrays["means"].ndim < 2
        or parameter_arrays["means"].shape[0] != n_components
        or parameter_arrays["covariances"].ndim < 2
        or parameter_arrays["covariances"].shape[0] != n_components
        or parameter_arrays["transition_matrix"].shape
        != (n_components, n_components)
        or parameter_arrays["start_probabilities"].shape != (n_components,)
    ):
        raise RegimeTaxonomyError(
            "HMM parameter shapes do not match n_components"
    )
    if mixture_weights is not None:
        weights = _numeric_array(
            mixture_weights,
            "mixture_weights",
        )
        if (
            weights.ndim != 2
            or weights.shape[0] != n_components
            or parameter_arrays["means"].ndim < 3
            or parameter_arrays["means"].shape[:2] != weights.shape
            or parameter_arrays["covariances"].shape[:2] != weights.shape
        ):
            raise RegimeTaxonomyError(
                "HMM mixture parameter shapes do not match"
            )
        parameter_arrays["mixture_weights"] = weights
    elif parameter_arrays["means"].ndim != 2:
        raise RegimeTaxonomyError(
            "Gaussian HMM means must be a two-dimensional array"
        )

    payload = {
        "schema": TAXONOMY_SCHEMA,
        "n_components": n_components,
        "feature_manifest_sha256": feature_hash,
        "training_end": cutoff,
        "pipeline_version": pipeline_version,
        "state_labels": list(state_labels),
        "parameter_digests": {
            "means": _array_digest(parameter_arrays["means"], "means"),
            "covariances": _array_digest(
                parameter_arrays["covariances"],
                "covariances",
            ),
            "transition_matrix": _array_digest(
                parameter_arrays["transition_matrix"],
                "transition_matrix",
            ),
            "start_probabilities": _array_digest(
                parameter_arrays["start_probabilities"],
                "start_probabilities",
            ),
        },
    }
    if "mixture_weights" in parameter_arrays:
        payload["parameter_digests"]["mixture_weights"] = _array_digest(
            parameter_arrays["mixture_weights"],
            "mixture_weights",
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class RawHMMStateRef:
    """One raw state bound to the exact HMM taxonomy that produced it."""

    taxonomy_id: str
    state: int

    def __post_init__(self) -> None:
        _sha256(self.taxonomy_id, "taxonomy_id")
        _exact_state(self.state, "state")


@dataclass(frozen=True, slots=True)
class FinalRiskRegimeRef:
    """A final overlay state that must never index raw HMM return buckets."""

    state: int
    label: str
    namespace: str = FINAL_RISK_NAMESPACE

    def __post_init__(self) -> None:
        _exact_state(self.state, "state")
        if self.namespace != FINAL_RISK_NAMESPACE:
            raise RegimeTaxonomyError("unsupported final-risk namespace")
        if FINAL_RISK_LABELS.get(self.state) != self.label:
            raise RegimeTaxonomyError(
                "final-risk state and label must match the closed overlay "
                "taxonomy"
            )


@dataclass(frozen=True, slots=True)
class RegimeReturnBuckets:
    """Resolved-only returns grouped by one exact raw HMM taxonomy."""

    taxonomy_id: str
    model_training_end: str
    inference_as_of: str
    resolved_outcomes_through: str
    horizon_calendar_days: int
    horizon_trading_days: int
    buckets: tuple[tuple[float, ...], ...]
    validity_status: str = "UNVERIFIED"
    execution_eligible: bool = False
    schema: str = RETURN_BUCKET_SCHEMA

    def __post_init__(self) -> None:
        _sha256(self.taxonomy_id, "taxonomy_id")
        training_end = _iso_date(self.model_training_end, "model_training_end")
        inference_as_of = _iso_date(self.inference_as_of, "inference_as_of")
        resolved_through = _iso_date(
            self.resolved_outcomes_through,
            "resolved_outcomes_through",
        )
        if not training_end <= inference_as_of:
            raise RegimeTaxonomyError(
                "model training end cannot follow inference as-of"
            )
        if not resolved_through < inference_as_of:
            raise RegimeTaxonomyError(
                "return outcomes must resolve strictly before inference"
            )
        if (
            type(self.horizon_calendar_days) is not int
            or self.horizon_calendar_days < 1
            or type(self.horizon_trading_days) is not int
            or self.horizon_trading_days < 1
        ):
            raise RegimeTaxonomyError("return horizons must be positive integers")
        if (
            type(self.buckets) is not tuple
            or len(self.buckets) < 2
            or any(type(bucket) is not tuple for bucket in self.buckets)
        ):
            raise RegimeTaxonomyError(
                "buckets must be an ordered tuple for every raw state"
            )
        for bucket in self.buckets:
            if not bucket:
                raise RegimeTaxonomyError(
                    "every raw HMM state requires a non-empty return bucket"
                )
            for value in bucket:
                if (
                    type(value) not in {int, float}
                    or type(value) is bool
                    or not math.isfinite(float(value))
                    or float(value) <= -1.0
                ):
                    raise RegimeTaxonomyError(
                        "return buckets must contain finite fractional returns"
                    )
        if self.validity_status not in {"UNVERIFIED", "INVALID", "VALID"}:
            raise RegimeTaxonomyError("unsupported validity status")
        if type(self.execution_eligible) is not bool:
            raise RegimeTaxonomyError("execution_eligible must be bool")
        if self.validity_status != "VALID" and self.execution_eligible:
            raise RegimeTaxonomyError(
                "an uncertified return bucket cannot authorize execution"
            )
        if self.schema != RETURN_BUCKET_SCHEMA:
            raise RegimeTaxonomyError("unsupported return-bucket schema")

    @property
    def state_count(self) -> int:
        return len(self.buckets)

    def bucket_for(self, state: RawHMMStateRef) -> tuple[float, ...]:
        if type(state) is not RawHMMStateRef:
            raise RegimeTaxonomyError(
                "raw HMM return buckets require an exact RawHMMStateRef"
            )
        if state.taxonomy_id != self.taxonomy_id:
            raise RegimeTaxonomyError("raw HMM taxonomy mismatch")
        if state.state >= self.state_count:
            raise RegimeTaxonomyError("raw HMM state is outside the taxonomy")
        return self.buckets[state.state]

    def manifest(self) -> Mapping[str, Any]:
        """Return a value-free audit manifest for reports and run records."""

        bucket_hashes = []
        for bucket in self.buckets:
            canonical = np.asarray(bucket, dtype=">f8").tobytes(order="C")
            bucket_hashes.append(hashlib.sha256(canonical).hexdigest())
        return {
            "schema": self.schema,
            "taxonomy_id": self.taxonomy_id,
            "model_training_end": self.model_training_end,
            "inference_as_of": self.inference_as_of,
            "resolved_outcomes_through": self.resolved_outcomes_through,
            "horizon_calendar_days": self.horizon_calendar_days,
            "horizon_trading_days": self.horizon_trading_days,
            "state_count": self.state_count,
            "bucket_counts": [len(bucket) for bucket in self.buckets],
            "bucket_sha256": bucket_hashes,
            "validity_status": self.validity_status,
            "execution_eligible": self.execution_eligible,
        }
