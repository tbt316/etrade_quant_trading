"""Tests for the bounded, fail-closed V2 shadow publisher."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from live_trading.regime_detector_v2 import (
    DETECTOR_VERSION,
    SIGNAL_TIMESTAMP,
    regime_detector_code_sha256,
)
from live_trading.regime_shadow_publish import (
    RegimeShadowPublishError,
    main,
    publish_shadow_signal,
)
from live_trading.regime_shadow_store import RegimeShadowStore
from live_trading.regime_signal import from_calibrated_v2_trace


SESSION = date(2025, 3, 10)
AVAILABLE_AT = datetime(2025, 3, 10, 20, 20, tzinfo=timezone.utc)
SNAPSHOT_SHA256 = "1" * 64
EVIDENCE_SHA256 = "2" * 64
CALENDAR_SHA256 = "3" * 64
POLICY_SHA256 = "4" * 64
CONFIG_SHA256 = "5" * 64


class _Gateway:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def refresh(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _EvidenceStore:
    def __init__(self, latest):
        self.latest = latest
        self.calls = []

    def latest_snapshot(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.latest


def _snapshot(sha256=SNAPSHOT_SHA256):
    return SimpleNamespace(snapshot_sha256=sha256)


def _refresh(snapshot, *, published=True, published_sha256=SNAPSHOT_SHA256):
    return SimpleNamespace(
        snapshot=snapshot,
        published=published,
        published_snapshot_sha256=published_sha256 if published else None,
    )


def _trace(*, background="elevated", shock="active"):
    frame = pd.DataFrame(
        {
            "Background_State": [background],
            "Shock_State": [shock],
            "Signal_Available_At": [AVAILABLE_AT.isoformat()],
            "Tradable_Session": ["2025-03-11"],
            "Detector_Version": [DETECTOR_VERSION],
            "Config_Hash": [CONFIG_SHA256],
            "Regime_Signal_Timestamp": [SIGNAL_TIMESTAMP],
            "Reason_Codes": ["vix_daily_change_extreme"],
            "Data_Quality": ["exchange_sessions_and_source_times_valid"],
            "Execution_Eligible": [False],
            "Input_Snapshot_SHA256": [SNAPSHOT_SHA256],
            "Input_Provenance_Status": ["verified"],
            "Evidence_Manifest_SHA256": [EVIDENCE_SHA256],
            "Evidence_Verification_Kind": ["decision_time"],
            "Evidence_Decision_Time_Eligible": [True],
            "Calendar_Schedule_SHA256": [CALENDAR_SHA256],
            "Source_Policy_SHA256": [POLICY_SHA256],
        },
        index=pd.DatetimeIndex([SESSION]),
    )
    frame.attrs.update(
        {
            "detector_version": DETECTOR_VERSION,
            "config_hash": CONFIG_SHA256,
            "detector_code_sha256": regime_detector_code_sha256(),
            "runtime_fingerprint_sha256": "6" * 64,
            "input_snapshot_sha256": SNAPSHOT_SHA256,
            "calendar_schedule_sha256": CALENDAR_SHA256,
            "source_policy_sha256": POLICY_SHA256,
            "evidence_manifest_sha256": EVIDENCE_SHA256,
            "evidence_verification_kind": "decision_time",
            "evidence_decision_time_eligible": True,
            "source_provenance_verified": True,
        }
    )
    return frame


class RegimeShadowPublishTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "regime_v2_shadow.json"
        self.shadow_store = RegimeShadowStore(self.path)
        self.snapshot = _snapshot()
        self.evidence_store = _EvidenceStore(self.snapshot)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _publish(self, gateway, trace):
        with mock.patch(
            "live_trading.regime_shadow_publish.detect_regimes_from_verified_snapshot",
            return_value=trace,
        ) as detector:
            signal = publish_shadow_signal(
                gateway=gateway,
                evidence_store=self.evidence_store,
                shadow_store=self.shadow_store,
                refresh_start=SESSION,
                session=SESSION,
                snapshot_start=SESSION,
                entitlement_authorized=lambda: True,
            )
        detector.assert_called_once_with(self.snapshot, self.evidence_store, None)
        return signal

    def test_publishes_only_exact_new_verified_decision_time_signal(self):
        gateway = _Gateway(_refresh(self.snapshot))
        signal = self._publish(gateway, _trace())

        self.assertEqual(signal.as_of_session, SESSION)
        self.assertEqual(signal.effective_session.isoformat(), "2025-03-11")
        self.assertEqual(signal.lineage.snapshot_sha256, SNAPSHOT_SHA256)
        self.assertEqual(signal.lineage.evidence_manifest_sha256, EVIDENCE_SHA256)
        self.assertFalse(signal.may_authorize_execution)
        self.assertEqual(
            self.shadow_store.load(now=AVAILABLE_AT + timedelta(minutes=1)),
            signal,
        )
        self.assertEqual(
            gateway.calls,
            [((SESSION, SESSION), {"channel": "shadow", "snapshot_start": SESSION})],
        )
        self.assertEqual(
            self.evidence_store.calls,
            [(("shadow",), {"require_verified": True})],
        )

    def test_failed_refresh_and_unavailable_result_preserve_prior_file(self):
        prior = from_calibrated_v2_trace(_trace())["2025-03-11"]
        self.shadow_store.publish(prior)
        original = self.path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(
            RegimeShadowPublishError, "verified_refresh_not_published"
        ):
            publish_shadow_signal(
                gateway=_Gateway(_refresh(None, published=False)),
                evidence_store=self.evidence_store,
                shadow_store=self.shadow_store,
                refresh_start=SESSION,
                session=SESSION,
                snapshot_start=SESSION,
                entitlement_authorized=lambda: True,
            )
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

        with mock.patch(
            "live_trading.regime_shadow_publish.detect_regimes_from_verified_snapshot",
            return_value=_trace(background="unavailable"),
        ):
            with self.assertRaisesRegex(
                RegimeShadowPublishError, "insufficient_history"
            ):
                publish_shadow_signal(
                    gateway=_Gateway(_refresh(self.snapshot)),
                    evidence_store=self.evidence_store,
                    shadow_store=self.shadow_store,
                    refresh_start=SESSION,
                    session=SESSION,
                    snapshot_start=SESSION,
                    entitlement_authorized=lambda: True,
                )
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_mismatched_latest_snapshot_preserves_prior_file(self):
        prior = from_calibrated_v2_trace(_trace())["2025-03-11"]
        self.shadow_store.publish(prior)
        original = self.path.read_text(encoding="utf-8")
        mismatched_store = _EvidenceStore(_snapshot("9" * 64))

        with self.assertRaisesRegex(
            RegimeShadowPublishError, "published_snapshot_mismatch"
        ):
            publish_shadow_signal(
                gateway=_Gateway(_refresh(self.snapshot)),
                evidence_store=mismatched_store,
                shadow_store=self.shadow_store,
                refresh_start=SESSION,
                session=SESSION,
                snapshot_start=SESSION,
                entitlement_authorized=lambda: True,
            )
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_direct_calls_require_entitlement_before_all_mutations(self):
        gateway = _Gateway(_refresh(self.snapshot))
        for capability in (None, lambda: False, lambda: (_ for _ in ()).throw(RuntimeError)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(
                    RegimeShadowPublishError, "entitlement_invalid"
                ):
                    publish_shadow_signal(
                        gateway=gateway,
                        evidence_store=self.evidence_store,
                        shadow_store=self.shadow_store,
                        refresh_start=SESSION,
                        session=SESSION,
                        snapshot_start=SESSION,
                        entitlement_authorized=capability,
                    )
        self.assertEqual(gateway.calls, [])
        self.assertEqual(self.evidence_store.calls, [])
        self.assertFalse(self.path.exists())

    def test_cli_entitlement_failure_precedes_credentials_and_transport(self):
        args = [
            "--evidence-db", str(Path(self.temp_dir.name) / "evidence.sqlite3"),
            "--shadow-signal-file", str(self.path),
            "--refresh-start", "2025-03-10",
            "--session", "2025-03-10",
            "--snapshot-start", "2025-03-10",
            "--entitlement-id", "opaque-record-id",
            "--entitlement-validator", "external:validate",
        ]
        with mock.patch(
            "live_trading.regime_shadow_publish._load_entitlement_validator",
            return_value=lambda _: False,
        ), mock.patch(
            "live_trading.regime_shadow_publish.RequestsProviderTransport"
        ) as transport:
            self.assertEqual(main(args), 1)
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
