"""Pure counterfactual Regime V2 risk policy.

The policy computes an advisory cap relative to a caller-supplied baseline.  It
does not construct orders, submit orders, mutate account state, or authorize
execution.  Unavailable or abstaining signals always produce a zero cap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from live_trading.regime_signal import (
    BackgroundState,
    RegimeSignal,
    ShockState,
    SignalAvailability,
)


REGIME_RISK_POLICY_SCHEMA_VERSION = "regime_counterfactual_risk_policy.v1"
_POLICY_HASH_DOMAIN = b"regime-counterfactual-risk-policy.v1\0"
_MAX_BPS = 10_000


class RegimeRiskPolicyError(ValueError):
    """Raised when a counterfactual policy is non-monotonic or malformed."""


def _basis_points(value: Any, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_BPS
    ):
        raise RegimeRiskPolicyError(
            f"{field_name} must be integer basis points in [0, 10000]"
        )
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class RegimeRiskDecision:
    """One non-authoritative counterfactual exposure calculation."""

    background_state: BackgroundState
    shock_state: ShockState
    baseline_short_put_exposure_bps: int
    regime_cap_bps: int
    counterfactual_short_put_exposure_bps: int
    block_new_short_puts: bool
    reason_codes: tuple[str, ...]
    policy_sha256: str
    counterfactual_only: bool = True
    may_authorize_execution: bool = False

    def __post_init__(self) -> None:
        try:
            background = BackgroundState(self.background_state)
            shock = ShockState(self.shock_state)
        except ValueError as exc:
            raise RegimeRiskPolicyError(
                "risk decision state is outside the V2 taxonomy"
            ) from exc
        baseline = _basis_points(
            self.baseline_short_put_exposure_bps,
            "baseline_short_put_exposure_bps",
        )
        cap = _basis_points(self.regime_cap_bps, "regime_cap_bps")
        counterfactual = _basis_points(
            self.counterfactual_short_put_exposure_bps,
            "counterfactual_short_put_exposure_bps",
        )
        if counterfactual != baseline * cap // _MAX_BPS:
            raise RegimeRiskPolicyError(
                "counterfactual exposure does not match the policy cap"
            )
        if self.block_new_short_puts is not (cap == 0):
            raise RegimeRiskPolicyError(
                "block flag does not match the zero-exposure cap"
            )
        try:
            reasons = tuple(self.reason_codes)
        except TypeError as exc:
            raise RegimeRiskPolicyError(
                "reason_codes must be iterable"
            ) from exc
        if (
            not reasons
            or len(reasons) != len(set(reasons))
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise RegimeRiskPolicyError(
                "reason_codes must be nonempty unique strings"
            )
        if (
            not isinstance(self.policy_sha256, str)
            or len(self.policy_sha256) != 64
            or any(item not in "0123456789abcdef" for item in self.policy_sha256)
        ):
            raise RegimeRiskPolicyError(
                "policy_sha256 must be a lowercase SHA-256 digest"
            )
        if (
            self.counterfactual_only is not True
            or self.may_authorize_execution is not False
        ):
            raise RegimeRiskPolicyError(
                "regime risk decisions cannot authorize execution"
            )
        object.__setattr__(self, "background_state", background)
        object.__setattr__(self, "shock_state", shock)
        object.__setattr__(
            self,
            "baseline_short_put_exposure_bps",
            baseline,
        )
        object.__setattr__(self, "regime_cap_bps", cap)
        object.__setattr__(
            self,
            "counterfactual_short_put_exposure_bps",
            counterfactual,
        )
        object.__setattr__(self, "reason_codes", reasons)


@dataclass(frozen=True)
class RegimeRiskPolicy:
    """Monotonic two-axis cap applied only to a baseline counterfactual."""

    calm_cap_bps: int = 10_000
    elevated_cap_bps: int = 6_000
    persistent_stress_cap_bps: int = 0
    no_shock_cap_bps: int = 10_000
    aftershock_cap_bps: int = 5_000
    active_shock_cap_bps: int = 0
    unavailable_cap_bps: int = 0
    abstain_cap_bps: int = 0
    counterfactual_only: bool = True
    execution_eligible: bool = False
    schema_version: str = REGIME_RISK_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGIME_RISK_POLICY_SCHEMA_VERSION:
            raise RegimeRiskPolicyError(
                "unsupported regime risk policy schema"
            )
        for field_name in (
            "calm_cap_bps",
            "elevated_cap_bps",
            "persistent_stress_cap_bps",
            "no_shock_cap_bps",
            "aftershock_cap_bps",
            "active_shock_cap_bps",
            "unavailable_cap_bps",
            "abstain_cap_bps",
        ):
            object.__setattr__(
                self,
                field_name,
                _basis_points(getattr(self, field_name), field_name),
            )
        if not (
            self.calm_cap_bps
            >= self.elevated_cap_bps
            >= self.persistent_stress_cap_bps
        ):
            raise RegimeRiskPolicyError(
                "background caps must decrease as stress worsens"
            )
        if not (
            self.no_shock_cap_bps
            >= self.aftershock_cap_bps
            >= self.active_shock_cap_bps
        ):
            raise RegimeRiskPolicyError(
                "shock caps must decrease as shock severity worsens"
            )
        if (
            self.persistent_stress_cap_bps != 0
            or self.active_shock_cap_bps != 0
            or self.unavailable_cap_bps != 0
            or self.abstain_cap_bps != 0
        ):
            raise RegimeRiskPolicyError(
                "stress, active shock, unavailable, and abstain must block"
            )
        if (
            self.counterfactual_only is not True
            or self.execution_eligible is not False
        ):
            raise RegimeRiskPolicyError(
                "regime risk policy cannot authorize execution"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calm_cap_bps": self.calm_cap_bps,
            "elevated_cap_bps": self.elevated_cap_bps,
            "persistent_stress_cap_bps": (
                self.persistent_stress_cap_bps
            ),
            "no_shock_cap_bps": self.no_shock_cap_bps,
            "aftershock_cap_bps": self.aftershock_cap_bps,
            "active_shock_cap_bps": self.active_shock_cap_bps,
            "unavailable_cap_bps": self.unavailable_cap_bps,
            "abstain_cap_bps": self.abstain_cap_bps,
            "counterfactual_only": self.counterfactual_only,
            "execution_eligible": self.execution_eligible,
        }

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(
            _POLICY_HASH_DOMAIN
            + _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def _background_cap(self, state: BackgroundState) -> int:
        return {
            BackgroundState.CALM: self.calm_cap_bps,
            BackgroundState.ELEVATED: self.elevated_cap_bps,
            BackgroundState.PERSISTENT_STRESS: (
                self.persistent_stress_cap_bps
            ),
            BackgroundState.UNAVAILABLE: self.unavailable_cap_bps,
        }[state]

    def _shock_cap(self, state: ShockState) -> int:
        return {
            ShockState.NONE: self.no_shock_cap_bps,
            ShockState.AFTERSHOCK: self.aftershock_cap_bps,
            ShockState.ACTIVE: self.active_shock_cap_bps,
            ShockState.UNAVAILABLE: self.unavailable_cap_bps,
        }[state]

    def evaluate(
        self,
        *,
        background_state: BackgroundState | str,
        shock_state: ShockState | str,
        baseline_short_put_exposure_bps: int,
        signal_available: bool,
        abstain_reasons: Iterable[str] = (),
    ) -> RegimeRiskDecision:
        """Return a pure counterfactual; no caller action is implied."""

        try:
            background = BackgroundState(background_state)
            shock = ShockState(shock_state)
        except ValueError as exc:
            raise RegimeRiskPolicyError(
                "policy input state is outside the V2 taxonomy"
            ) from exc
        baseline = _basis_points(
            baseline_short_put_exposure_bps,
            "baseline_short_put_exposure_bps",
        )
        if not isinstance(signal_available, bool):
            raise RegimeRiskPolicyError(
                "signal_available must be boolean"
            )
        if isinstance(abstain_reasons, (str, bytes)):
            raise RegimeRiskPolicyError(
                "abstain_reasons must be iterable"
            )
        try:
            abstains = tuple(abstain_reasons)
        except TypeError as exc:
            raise RegimeRiskPolicyError(
                "abstain_reasons must be iterable"
            ) from exc
        if any(not isinstance(item, str) or not item for item in abstains):
            raise RegimeRiskPolicyError(
                "abstain_reasons must contain nonempty strings"
            )

        reasons: list[str] = []
        if not signal_available:
            cap = self.unavailable_cap_bps
            reasons.append("signal_unavailable")
        elif abstains:
            cap = self.abstain_cap_bps
            reasons.append("signal_abstained")
        elif (
            background is BackgroundState.UNAVAILABLE
            or shock is ShockState.UNAVAILABLE
        ):
            cap = self.unavailable_cap_bps
            reasons.append("state_unavailable")
        else:
            background_cap = self._background_cap(background)
            shock_cap = self._shock_cap(shock)
            cap = min(background_cap, shock_cap)
            reasons.extend(
                (
                    f"background_{background.value}",
                    f"shock_{shock.value}",
                )
            )
        if cap == 0 and "new_short_puts_blocked" not in reasons:
            reasons.append("new_short_puts_blocked")
        return RegimeRiskDecision(
            background_state=background,
            shock_state=shock,
            baseline_short_put_exposure_bps=baseline,
            regime_cap_bps=cap,
            counterfactual_short_put_exposure_bps=(
                baseline * cap // _MAX_BPS
            ),
            block_new_short_puts=cap == 0,
            reason_codes=tuple(reasons),
            policy_sha256=self.policy_sha256,
        )

    def evaluate_signal(
        self,
        signal: RegimeSignal,
        *,
        baseline_short_put_exposure_bps: int,
    ) -> RegimeRiskDecision:
        if type(signal) is not RegimeSignal:
            raise TypeError("signal must be an exact RegimeSignal instance")
        return self.evaluate(
            background_state=signal.background_state,
            shock_state=signal.shock_state,
            baseline_short_put_exposure_bps=(
                baseline_short_put_exposure_bps
            ),
            signal_available=(
                signal.availability is SignalAvailability.ADVISORY
            ),
            abstain_reasons=signal.abstain_reasons,
        )


__all__ = [
    "REGIME_RISK_POLICY_SCHEMA_VERSION",
    "RegimeRiskDecision",
    "RegimeRiskPolicy",
    "RegimeRiskPolicyError",
]
