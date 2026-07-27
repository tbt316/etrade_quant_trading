import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import live_trading.regime_shadow_store as shadow_store_module
from live_trading.regime_detector_v2 import DETECTOR_VERSION, SIGNAL_TIMESTAMP
from live_trading.regime_market_data import regime_market_schedule
from live_trading.regime_shadow_store import (
    RegimeShadowStore,
    RegimeShadowStoreError,
)
from live_trading.regime_signal import from_calibrated_v2_trace


CONFIG_HASH = "a" * 64


def _signal(
    *,
    as_of_session="2025-04-03",
    available_at="2025-04-03T20:15:00Z",
    effective_session="2025-04-04",
):
    trace = pd.DataFrame(
        {
            "Background_State": ["elevated"],
            "Shock_State": ["active"],
            "Signal_Available_At": [available_at],
            "Tradable_Session": [effective_session],
            "Detector_Version": [DETECTOR_VERSION],
            "Config_Hash": [CONFIG_HASH],
            "Regime_Signal_Timestamp": [SIGNAL_TIMESTAMP],
            "Reason_Codes": ["vix_daily_change_extreme"],
            "Data_Quality": ["exchange_sessions_and_source_times_valid"],
            "Execution_Eligible": [False],
        },
        index=pd.DatetimeIndex([as_of_session]),
    )
    trace.attrs["detector_version"] = DETECTOR_VERSION
    trace.attrs["config_hash"] = CONFIG_HASH
    return from_calibrated_v2_trace(trace)[effective_session]


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class RegimeShadowStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "regime_v2_shadow.json"
        self.store = RegimeShadowStore(self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_publish_is_atomic_owner_only_and_dashboard_payload_is_redacted(self):
        signal = _signal()
        now = _utc("2025-04-03T20:16:00Z")
        self.store.publish(signal)

        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(self.store.load(now=now), signal)
        payload = self.store.dashboard_payload(now=now)
        self.assertEqual(
            set(payload),
            {
                "available", "status", "source_family", "as_of_session",
                "effective_session", "background_state", "shock_state",
                "composite_label", "availability", "reason_codes",
                "abstain_reasons", "stale", "may_authorize_execution",
            },
        )
        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"], "advisory")
        self.assertEqual(payload["background_state"], "elevated")
        self.assertEqual(payload["shock_state"], "active")
        self.assertFalse(payload["stale"])
        self.assertFalse(payload["may_authorize_execution"])
        self.assertNotIn("lineage", payload)
        self.assertNotIn("signal_sha256", payload)
        self.assertNotIn("available_at", payload)
        self.assertNotIn("data_quality", payload)
        self.assertFalse(any(item.name.startswith(".regime_v2_shadow") for item in self.path.parent.iterdir()))

    def test_missing_invalid_and_not_yet_available_are_publicly_redacted(self):
        missing = self.store.dashboard_payload(now=_utc("2025-04-03T20:16:00Z"))
        self.assertEqual(missing["reason_codes"], ["missing"])
        self.assertFalse(missing["available"])
        self.assertFalse(missing["may_authorize_execution"])

        self.store.publish(_signal())
        self.path.write_text("{}", encoding="utf-8")
        os.chmod(self.path, 0o600)
        invalid = self.store.dashboard_payload(now=_utc("2025-04-03T20:16:00Z"))
        self.assertEqual(invalid["reason_codes"], ["invalid"])
        self.assertNotIn("shadow_signal_schema_invalid", repr(invalid))

        self.store.publish(_signal())
        future = self.store.dashboard_payload(now=_utc("2025-04-03T20:14:00Z"))
        self.assertEqual(future["reason_codes"], ["not_yet_available"])

    def test_session_causality_keeps_weekend_valid_and_stales_at_effective_finalization(self):
        signal = _signal(
            as_of_session="2025-04-04",
            available_at="2025-04-04T20:15:00Z",
            effective_session="2025-04-07",
        )
        self.store.publish(signal)
        saturday = _utc("2025-04-05T18:00:00Z")
        self.assertEqual(self.store.load(now=saturday), signal)
        finalization = pd.Timestamp(
            regime_market_schedule("2025-04-07", "2025-04-07").iloc[0]["joint_finalization_at"]
        ).to_pydatetime()
        with self.assertRaisesRegex(RegimeShadowStoreError, "shadow_signal_stale"):
            self.store.load(now=finalization)
        stale = self.store.dashboard_payload(now=finalization)
        self.assertEqual(stale["reason_codes"], ["stale"])
        self.assertTrue(stale["stale"])

    def test_file_and_parent_permission_or_symlink_fail_closed(self):
        self.store.publish(_signal())
        os.chmod(self.path, 0o644)
        with self.assertRaisesRegex(RegimeShadowStoreError, "shadow_signal_path_unsafe"):
            self.store.load(now=_utc("2025-04-03T20:16:00Z"))

        os.chmod(self.path, 0o600)
        os.chmod(self.path.parent, 0o755)
        with self.assertRaisesRegex(RegimeShadowStoreError, "shadow_store_parent_unsafe"):
            self.store.load(now=_utc("2025-04-03T20:16:00Z"))
        os.chmod(self.path.parent, 0o700)

        real_parent = Path(self.temp_dir.name) / "real"
        real_parent.mkdir(mode=0o700)
        linked_parent = Path(self.temp_dir.name) / "linked"
        os.symlink(real_parent, linked_parent)
        linked = RegimeShadowStore(linked_parent / "regime_v2_shadow.json")
        unavailable = linked.dashboard_payload(now=_utc("2025-04-03T20:16:00Z"))
        self.assertEqual(unavailable["reason_codes"], ["invalid"])

    def test_lstat_open_fstat_identity_check_rejects_replacement_race(self):
        self.store.publish(_signal())
        replacement = self.path.parent / "replacement.json"
        replacement.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        os.chmod(replacement, 0o600)
        original_open = os.open
        replaced = False

        def race_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal replaced
            if path == self.path.name and dir_fd is not None and not replaced:
                replaced = True
                os.replace(replacement, self.path)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with patch.object(shadow_store_module.os, "open", side_effect=race_open):
            with self.assertRaisesRegex(RegimeShadowStoreError, "shadow_signal_path_unsafe"):
                self.store.load(now=_utc("2025-04-03T20:16:00Z"))

    def test_fifo_hardlink_and_blocking_open_fail_closed(self):
        os.mkfifo(self.path, mode=0o600)
        payload = self.store.dashboard_payload(
            now=_utc("2025-04-03T20:16:00Z")
        )
        self.assertEqual(payload["reason_codes"], ["invalid"])

        self.path.unlink()
        target = self.path.parent / "hardlink-target.json"
        target.write_text("{}", encoding="utf-8")
        os.chmod(target, 0o600)
        os.link(target, self.path)
        with self.assertRaisesRegex(
            RegimeShadowStoreError,
            "shadow_signal_path_unsafe",
        ):
            self.store.load(now=_utc("2025-04-03T20:16:00Z"))

        self.path.unlink()
        target.unlink()
        self.store.publish(_signal())
        original_open = os.open
        observed_flags = []

        def inspect_open(path, flags, mode=0o777, *, dir_fd=None):
            if path == self.path.name and dir_fd is not None:
                observed_flags.append(flags)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with patch.object(
            shadow_store_module.os,
            "open",
            side_effect=inspect_open,
        ):
            self.store.load(now=_utc("2025-04-03T20:16:00Z"))
        self.assertTrue(observed_flags)
        self.assertTrue(
            all(
                flags & getattr(os, "O_NONBLOCK", 0)
                for flags in observed_flags
            )
        )

    def test_publish_rejects_non_signal_or_execution_capable_subclass(self):
        with self.assertRaises(TypeError):
            self.store.publish(object())

        class UnsafeSignal(type(_signal())):
            @property
            def may_authorize_execution(self):
                return True

        unsafe = UnsafeSignal(**_signal().__dict__)
        with self.assertRaises(TypeError):
            self.store.publish(unsafe)


if __name__ == "__main__":
    unittest.main()
