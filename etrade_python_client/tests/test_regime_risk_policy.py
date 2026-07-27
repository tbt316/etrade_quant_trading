import dataclasses
import unittest

import pandas as pd

from live_trading.regime_detector_v2 import DETECTOR_VERSION, SIGNAL_TIMESTAMP
from live_trading.regime_risk_policy import (
    RegimeRiskPolicy,
    RegimeRiskPolicyError,
)
from live_trading.regime_signal import (
    BackgroundState,
    ShockState,
    from_calibrated_v2_trace,
)


class RegimeRiskPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = RegimeRiskPolicy()

    def _exposure(
        self,
        background: BackgroundState,
        shock: ShockState,
    ) -> int:
        return self.policy.evaluate(
            background_state=background,
            shock_state=shock,
            baseline_short_put_exposure_bps=10_000,
            signal_available=True,
        ).counterfactual_short_put_exposure_bps

    def test_every_worsening_background_or_shock_is_nonincreasing(self):
        backgrounds = (
            BackgroundState.CALM,
            BackgroundState.ELEVATED,
            BackgroundState.PERSISTENT_STRESS,
        )
        shocks = (
            ShockState.NONE,
            ShockState.AFTERSHOCK,
            ShockState.ACTIVE,
        )

        for shock in shocks:
            exposures = [
                self._exposure(background, shock)
                for background in backgrounds
            ]
            self.assertEqual(exposures, sorted(exposures, reverse=True))
        for background in backgrounds:
            exposures = [
                self._exposure(background, shock)
                for shock in shocks
            ]
            self.assertEqual(exposures, sorted(exposures, reverse=True))

    def test_active_shock_stress_unavailable_and_abstain_block(self):
        cases = (
            {
                "background_state": BackgroundState.CALM,
                "shock_state": ShockState.ACTIVE,
                "signal_available": True,
                "abstain_reasons": (),
            },
            {
                "background_state": BackgroundState.PERSISTENT_STRESS,
                "shock_state": ShockState.NONE,
                "signal_available": True,
                "abstain_reasons": (),
            },
            {
                "background_state": BackgroundState.UNAVAILABLE,
                "shock_state": ShockState.UNAVAILABLE,
                "signal_available": False,
                "abstain_reasons": (),
            },
            {
                "background_state": BackgroundState.CALM,
                "shock_state": ShockState.NONE,
                "signal_available": True,
                "abstain_reasons": ("prospective_holdout_incomplete",),
            },
        )

        for case in cases:
            decision = self.policy.evaluate(
                baseline_short_put_exposure_bps=8_000,
                **case,
            )
            self.assertEqual(decision.regime_cap_bps, 0)
            self.assertEqual(
                decision.counterfactual_short_put_exposure_bps,
                0,
            )
            self.assertTrue(decision.block_new_short_puts)
            self.assertTrue(decision.counterfactual_only)
            self.assertFalse(decision.may_authorize_execution)

    def test_baseline_is_only_reduced_and_decision_is_deterministic(self):
        first = self.policy.evaluate(
            background_state=BackgroundState.ELEVATED,
            shock_state=ShockState.AFTERSHOCK,
            baseline_short_put_exposure_bps=7_500,
            signal_available=True,
        )
        second = self.policy.evaluate(
            background_state="elevated",
            shock_state="aftershock",
            baseline_short_put_exposure_bps=7_500,
            signal_available=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.regime_cap_bps, 5_000)
        self.assertEqual(
            first.counterfactual_short_put_exposure_bps,
            3_750,
        )
        self.assertLessEqual(
            first.counterfactual_short_put_exposure_bps,
            first.baseline_short_put_exposure_bps,
        )
        self.assertEqual(first.policy_sha256, self.policy.policy_sha256)

    def test_current_typed_shadow_signal_remains_blocked_by_r4_abstention(self):
        trace = pd.DataFrame(
            {
                "Background_State": ["calm"],
                "Shock_State": ["none"],
                "Signal_Available_At": ["2025-04-03T20:15:00Z"],
                "Tradable_Session": ["2025-04-04"],
                "Detector_Version": [DETECTOR_VERSION],
                "Config_Hash": ["a" * 64],
                "Regime_Signal_Timestamp": [SIGNAL_TIMESTAMP],
                "Reason_Codes": ["no_stress_evidence"],
                "Data_Quality": [
                    "exchange_sessions_and_source_times_valid"
                ],
                "Execution_Eligible": [False],
            },
            index=pd.DatetimeIndex(["2025-04-03"]),
        )
        trace.attrs["detector_version"] = DETECTOR_VERSION
        trace.attrs["config_hash"] = "a" * 64
        signal = from_calibrated_v2_trace(trace)["2025-04-04"]

        decision = self.policy.evaluate_signal(
            signal,
            baseline_short_put_exposure_bps=10_000,
        )

        self.assertIn("signal_abstained", decision.reason_codes)
        self.assertEqual(decision.regime_cap_bps, 0)
        self.assertFalse(decision.may_authorize_execution)

    def test_nonmonotonic_or_permissive_hard_block_config_is_rejected(self):
        with self.assertRaisesRegex(
            RegimeRiskPolicyError,
            "background caps",
        ):
            dataclasses.replace(
                self.policy,
                calm_cap_bps=4_000,
                elevated_cap_bps=6_000,
            )
        with self.assertRaisesRegex(
            RegimeRiskPolicyError,
            "must block",
        ):
            dataclasses.replace(
                self.policy,
                active_shock_cap_bps=1,
            )
        with self.assertRaisesRegex(
            RegimeRiskPolicyError,
            "cannot authorize",
        ):
            dataclasses.replace(
                self.policy,
                execution_eligible=True,
            )


if __name__ == "__main__":
    unittest.main()
