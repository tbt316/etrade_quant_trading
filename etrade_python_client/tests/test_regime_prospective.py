import dataclasses
import json
import os
import stat
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import live_trading.regime_prospective_journal as journal_module
from live_trading.regime_calibration import (
    RegimeCalibrationArtifact,
    RegimeCalibrationPlan,
)
from live_trading.regime_detector_v2 import (
    SIGNAL_TIMESTAMP,
    regime_detector_runtime_fingerprint,
)
from live_trading.regime_prospective import (
    EntitlementStatus,
    PromotionGateStatus,
    ProspectiveJournalEntry,
    ProspectiveOutcomeRecord,
    ProspectiveReviewError,
    ProspectiveReviewProtocol,
    ProspectiveReviewReceipt,
    ProviderEntitlementReceipt,
    RegimePromotionGate,
    ReviewVerdict,
    VerifiedCalibrationEvidence,
    outcome_set_sha256,
)
from live_trading.regime_prospective_journal import (
    ProspectiveJournalStore,
    ProspectiveJournalStoreError,
)
from live_trading.regime_signal import (
    BackgroundState,
    CausalRecord,
    RegimeLineage,
    RegimeSignal,
    ShockState,
    SignalAvailability,
)


PLAN_HASH = "a" * 64
ARTIFACT_HASH = "b" * 64
CONFIG_HASH = "c" * 64
DETECTOR_CODE_HASH = "d" * 64
CALIBRATION_CODE_HASH = "e" * 64
RUNTIME_HASH = "f" * 64
SCOPES = (
    "cboe:vix_eod:derived_regime:non_display:retain",
    "massive:spy_daily:derived_regime:non_display:retain",
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _protocol() -> ProspectiveReviewProtocol:
    return ProspectiveReviewProtocol(
        protocol_id="regime_v2_test_window",
        calibration_plan_sha256=PLAN_HASH,
        research_calibration_artifact_sha256=ARTIFACT_HASH,
        selected_config_sha256=CONFIG_HASH,
        detector_version="regime_v2_shadow_test",
        detector_code_sha256=DETECTOR_CODE_HASH,
        calibration_code_sha256=CALIBRATION_CODE_HASH,
        runtime_fingerprint_sha256=RUNTIME_HASH,
        holdout_start_session="2025-04-03",
        observation_end_session="2025-04-04",
        final_outcome_resolution_session="2025-04-08",
        review_not_before_session="2025-04-08",
        activation_deadline=_utc("2025-04-03T13:00:00Z"),
        activation_status="activated_before_holdout",
        activated_at=_utc("2025-04-02T20:00:00Z"),
        activation_receipt_sha256="6" * 64,
        activation_reason_codes=(),
        outcome_horizons=(1,),
        signal_lag_sessions=1,
        minimum_signal_rows=2,
        minimum_resolved_rows=2,
        minimum_outcome_coverage=1.0,
        maximum_missing_signal_sessions=0,
        required_entitlement_scopes=SCOPES,
        maximum_entitlement_check_age_seconds=3600,
    )


def _signal(
    session: str,
    *,
    available_at: str,
    effective_session: str,
    background: BackgroundState = BackgroundState.CALM,
    shock: ShockState = ShockState.NONE,
) -> RegimeSignal:
    lineage = RegimeLineage(
        artifact_sha256=ARTIFACT_HASH,
        plan_sha256=PLAN_HASH,
        config_sha256=CONFIG_HASH,
        detector_version="regime_v2_shadow_test",
        detector_code_sha256=DETECTOR_CODE_HASH,
        calibration_code_sha256=CALIBRATION_CODE_HASH,
        calibration_profile="frozen_baseline",
        calibration_status="research_only",
        calibration_provenance="legacy_normalized_unverified",
        snapshot_sha256="1" * 64,
        source_provenance_status="verified",
        evidence_manifest_sha256="2" * 64,
        evidence_verification_kind="decision_time",
        evidence_decision_time_eligible=True,
        calendar_schedule_sha256="3" * 64,
        source_policy_sha256="4" * 64,
        runtime_fingerprint_sha256=RUNTIME_HASH,
        causal_record=CausalRecord(
            calibration_end_session=date(2024, 12, 31),
            retrospective_test_start=date(2025, 1, 2),
            retrospective_test_end=date(2025, 3, 31),
            inference_method="causal_prefix_filter",
            signal_lag_sessions=1,
        ),
    )
    return RegimeSignal(
        as_of_session=session,
        available_at=available_at,
        effective_session=effective_session,
        background_state=background,
        shock_state=shock,
        availability=SignalAvailability.ADVISORY,
        signal_timestamp=SIGNAL_TIMESTAMP,
        reason_codes=("no_stress_evidence",),
        abstain_reasons=("research_only",),
        data_quality=("verified_decision_time_evidence",),
        lineage=lineage,
    )


def _receipt(*, checked_at: str) -> ProviderEntitlementReceipt:
    return ProviderEntitlementReceipt(
        record_id="external_entitlement_record_42",
        subject_sha256="5" * 64,
        entitlement_evidence_sha256="6" * 64,
        terms_sha256="7" * 64,
        retention_policy_sha256="8" * 64,
        deletion_policy_sha256="9" * 64,
        reviewer_sha256="0" * 64,
        scopes=SCOPES,
        valid_from=_utc("2025-04-01T00:00:00Z"),
        valid_until=_utc("2025-04-10T00:00:00Z"),
        checked_at=_utc(checked_at),
    )


def _entry(
    *,
    protocol: ProspectiveReviewProtocol,
    sequence: int,
    previous: str | None,
    session: str,
    available_at: str,
    recorded_at: str,
    effective_session: str,
) -> ProspectiveJournalEntry:
    return ProspectiveJournalEntry.create(
        protocol=protocol,
        sequence_number=sequence,
        previous_entry_sha256=previous,
        recorded_at=_utc(recorded_at),
        signal=_signal(
            session,
            available_at=available_at,
            effective_session=effective_session,
        ),
        entitlement_receipt=_receipt(checked_at=available_at),
    )


def _entries(
    protocol: ProspectiveReviewProtocol,
) -> tuple[ProspectiveJournalEntry, ProspectiveJournalEntry]:
    first = _entry(
        protocol=protocol,
        sequence=1,
        previous=None,
        session="2025-04-03",
        available_at="2025-04-03T20:15:00Z",
        recorded_at="2025-04-03T20:20:00Z",
        effective_session="2025-04-04",
    )
    second = _entry(
        protocol=protocol,
        sequence=2,
        previous=first.entry_sha256,
        session="2025-04-04",
        available_at="2025-04-04T20:15:00Z",
        recorded_at="2025-04-04T20:20:00Z",
        effective_session="2025-04-07",
    )
    return first, second


def _outcomes(
    protocol: ProspectiveReviewProtocol,
    entries: tuple[ProspectiveJournalEntry, ...],
) -> tuple[ProspectiveOutcomeRecord, ...]:
    computed_times = (
        "2025-04-07T20:20:00Z",
        "2025-04-08T20:20:00Z",
    )
    records = []
    for position, (entry, computed_at) in enumerate(
        zip(entries, computed_times),
        start=1,
    ):
        records.append(
            ProspectiveOutcomeRecord(
                protocol_sha256=protocol.protocol_sha256,
                journal_entry_sha256=entry.entry_sha256,
                signal_as_of_session=entry.signal.as_of_session,
                resolved_horizons=protocol.outcome_horizons,
                final_resolution_session=(
                    protocol.resolution_session_for(
                        entry.signal.as_of_session,
                        max(protocol.outcome_horizons),
                    )
                ),
                outcome_payload_sha256=f"{position}" * 64,
                outcome_evidence_sha256=f"{position + 2}" * 64,
                computed_at=_utc(computed_at),
            )
        )
    return tuple(records)


def _calibration_evidence(
    protocol: ProspectiveReviewProtocol,
) -> VerifiedCalibrationEvidence:
    return VerifiedCalibrationEvidence(
        calibration_plan_sha256=protocol.calibration_plan_sha256,
        research_calibration_artifact_sha256=(
            protocol.research_calibration_artifact_sha256
        ),
        verified_replay_artifact_sha256="7" * 64,
        selected_config_sha256=protocol.selected_config_sha256,
        detector_version=protocol.detector_version,
        detector_code_sha256=protocol.detector_code_sha256,
        calibration_code_sha256=protocol.calibration_code_sha256,
        runtime_fingerprint_sha256=(
            protocol.runtime_fingerprint_sha256
        ),
        verified_data_manifest_sha256="8" * 64,
        evidence_manifest_sha256="9" * 64,
        entitlement_receipt=_receipt(
            checked_at="2025-04-08T20:25:00Z"
        ),
        verified_at=_utc("2025-04-08T20:25:00Z"),
    )


def _review_receipt(
    *,
    protocol: ProspectiveReviewProtocol,
    entries: tuple[ProspectiveJournalEntry, ...],
    outcomes: tuple[ProspectiveOutcomeRecord, ...],
    calibration: VerifiedCalibrationEvidence,
    verdict: ReviewVerdict = ReviewVerdict.PASS,
) -> ProspectiveReviewReceipt:
    return ProspectiveReviewReceipt(
        protocol_sha256=protocol.protocol_sha256,
        journal_head_sha256=entries[-1].entry_sha256,
        outcome_set_sha256=outcome_set_sha256(outcomes),
        calibration_evidence_sha256=calibration.evidence_sha256,
        statistical_report_sha256="0" * 64,
        reviewer_sha256="1" * 64,
        reviewed_at=_utc("2025-04-08T21:00:00Z"),
        verdict=verdict,
    )


class ProspectiveProtocolTests(unittest.TestCase):
    def test_protocol_is_canonical_immutable_and_resolution_frozen(self):
        protocol = _protocol()

        self.assertEqual(
            ProspectiveReviewProtocol.from_json(protocol.to_json()),
            protocol,
        )
        self.assertEqual(
            protocol.expected_signal_sessions,
            (date(2025, 4, 3), date(2025, 4, 4)),
        )
        self.assertEqual(
            protocol.resolution_session_for("2025-04-04", 1),
            date(2025, 4, 8),
        )
        self.assertFalse(protocol.execution_eligible)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            protocol.minimum_signal_rows = 1
        with self.assertRaisesRegex(
            ProspectiveReviewError,
            "canonical",
        ):
            ProspectiveReviewProtocol.from_json(protocol.to_json() + "\n")
        with self.assertRaisesRegex(
            ProspectiveReviewError,
            "no retuning",
        ):
            dataclasses.replace(protocol, no_retuning=False)
        with self.assertRaisesRegex(
            ProspectiveReviewError,
            "first holdout open",
        ):
            dataclasses.replace(
                protocol,
                activation_deadline=_utc("2025-04-03T13:30:00Z"),
            )

    def test_committed_protocol_pins_current_frozen_research_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        protocol = ProspectiveReviewProtocol.from_json(
            (
                root
                / "docs"
                / "regime_v2_prospective_review_protocol.json"
            ).read_text(encoding="utf-8")
        )
        plan = RegimeCalibrationPlan.from_json(
            (
                root / "docs" / "regime_v2_calibration_plan.json"
            ).read_text(encoding="utf-8")
        )
        artifact = RegimeCalibrationArtifact.from_json(
            (
                root
                / "research_reports"
                / "regime_v2_calibration_artifact.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(protocol.calibration_plan_sha256, plan.plan_sha256)
        self.assertEqual(
            protocol.research_calibration_artifact_sha256,
            artifact.artifact_sha256,
        )
        self.assertEqual(
            protocol.selected_config_sha256,
            artifact.selected_candidate.config_sha256,
        )
        self.assertEqual(
            protocol.runtime_fingerprint_sha256,
            regime_detector_runtime_fingerprint(),
        )
        self.assertEqual(len(protocol.expected_signal_sessions), 120)
        self.assertEqual(
            protocol.final_outcome_resolution_session,
            date(2027, 2, 16),
        )
        self.assertEqual(
            protocol.protocol_sha256,
            "78f3480f6b803681be9e29f38b75c4823168cfa64e8d8400922a62b01c73aad3",
        )


class ProspectiveJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "prospective"
        self.directory.mkdir(mode=0o700)
        os.chmod(self.directory, 0o700)
        self.protocol = _protocol()
        self.store = ProspectiveJournalStore(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def _append(
        self,
        *,
        store: ProspectiveJournalStore,
        signal: RegimeSignal,
        receipt: ProviderEntitlementReceipt,
        now: str,
    ) -> ProspectiveJournalEntry:
        with patch.object(
            journal_module,
            "_trusted_utc_now",
            return_value=_utc(now),
        ):
            return store.append(
                protocol=self.protocol,
                signal=signal,
                entitlement_receipt=receipt,
            )

    def test_append_restart_and_second_append_verify_full_chain(self):
        first_signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T20:15:00Z",
            effective_session="2025-04-04",
        )
        first = self._append(
            store=self.store,
            signal=first_signal,
            receipt=_receipt(
                checked_at="2025-04-03T20:15:00Z"
            ),
            now="2025-04-03T20:20:00Z",
        )
        restarted = ProspectiveJournalStore(self.directory)
        self.assertEqual(
            restarted.load_entries(protocol=self.protocol),
            (first,),
        )

        second_signal = _signal(
            "2025-04-04",
            available_at="2025-04-04T20:15:00Z",
            effective_session="2025-04-07",
        )
        second = self._append(
            store=restarted,
            signal=second_signal,
            receipt=_receipt(
                checked_at="2025-04-04T20:15:00Z"
            ),
            now="2025-04-04T20:20:00Z",
        )
        self.assertEqual(second.previous_entry_sha256, first.entry_sha256)
        self.assertEqual(
            ProspectiveJournalStore(self.directory).load_entries(
                protocol=self.protocol
            ),
            (first, second),
        )
        entry_files = sorted(self.directory.glob("*.json"))
        self.assertEqual(len(entry_files), 2)
        for path in entry_files:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertIn(
                ProspectiveJournalEntry.from_json(
                    path.read_text(encoding="utf-8")
                ).entry_sha256,
                path.name,
            )

    def test_tamper_unknown_file_and_exact_type_fail_closed(self):
        signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T20:15:00Z",
            effective_session="2025-04-04",
        )
        self._append(
            store=self.store,
            signal=signal,
            receipt=_receipt(
                checked_at="2025-04-03T20:15:00Z"
            ),
            now="2025-04-03T20:20:00Z",
        )
        entry_path = next(self.directory.glob("*.json"))
        decoded = json.loads(entry_path.read_text(encoding="utf-8"))
        decoded["entry"]["recorded_at"] = "2025-04-03T20:21:00Z"
        entry_path.write_text(
            json.dumps(decoded, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(entry_path, 0o600)
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_entry_invalid",
        ):
            ProspectiveJournalStore(self.directory).load_entries(
                protocol=self.protocol
            )

        entry_path.unlink()
        unexpected = self.directory / "notes.txt"
        unexpected.write_text("not part of the journal", encoding="utf-8")
        os.chmod(unexpected, 0o600)
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "unknown_file",
        ):
            self.store.load_entries(protocol=self.protocol)

        unexpected.unlink()
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_entry_rejected",
        ):
            self._append(
                store=self.store,
                signal=object(),
                receipt=_receipt(
                    checked_at="2025-04-03T20:15:00Z"
                ),
                now="2025-04-03T20:20:00Z",
            )

    def test_scope_expiry_late_backfill_and_permissions_fail_closed(self):
        signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T20:15:00Z",
            effective_session="2025-04-04",
        )
        expired = dataclasses.replace(
            _receipt(checked_at="2025-04-03T20:15:00Z"),
            valid_until=_utc("2025-04-03T20:16:00Z"),
        )
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_entry_rejected",
        ):
            self._append(
                store=self.store,
                signal=signal,
                receipt=expired,
                now="2025-04-03T20:20:00Z",
            )
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_entry_rejected",
        ):
            self._append(
                store=self.store,
                signal=signal,
                receipt=_receipt(
                    checked_at="2025-04-03T20:15:00Z"
                ),
                now="2025-04-04T14:00:00Z",
            )

        early_signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T14:00:00Z",
            effective_session="2025-04-04",
        )
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_entry_rejected",
        ):
            self._append(
                store=self.store,
                signal=early_signal,
                receipt=_receipt(
                    checked_at="2025-04-03T14:00:00Z"
                ),
                now="2025-04-03T14:01:00Z",
            )

        os.chmod(self.directory, 0o755)
        with self.assertRaisesRegex(
            ProspectiveJournalStoreError,
            "journal_directory_unsafe",
        ):
            self.store.load_entries(protocol=self.protocol)

    def test_exclusive_create_collision_never_deletes_preexisting_file(self):
        signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T20:15:00Z",
            effective_session="2025-04-04",
        )
        receipt = _receipt(checked_at="2025-04-03T20:15:00Z")
        recorded_at = _utc("2025-04-03T20:20:00Z")
        entry = ProspectiveJournalEntry.create(
            protocol=self.protocol,
            sequence_number=1,
            previous_entry_sha256=None,
            recorded_at=recorded_at,
            signal=signal,
            entitlement_receipt=receipt,
        )
        filename = f"{entry.sequence_number:012d}-{entry.entry_sha256}.json"
        original_open = os.open
        original_link = os.link
        collided = False

        def collide(
            source,
            destination,
            *,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=True,
        ):
            nonlocal collided
            if destination == filename and not collided:
                collided = True
                descriptor = original_open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dst_dir_fd,
                )
                os.write(descriptor, b"preexisting")
                os.close(descriptor)
            return original_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        with patch.object(
            journal_module,
            "_trusted_utc_now",
            return_value=recorded_at,
        ):
            with patch.object(
                journal_module.os,
                "link",
                side_effect=collide,
            ):
                with self.assertRaisesRegex(
                    ProspectiveJournalStoreError,
                    "journal_append_failed",
                ):
                    self.store.append(
                        protocol=self.protocol,
                        signal=signal,
                        entitlement_receipt=receipt,
                    )

        collision_path = self.directory / filename
        self.assertTrue(collision_path.exists())
        self.assertEqual(collision_path.read_bytes(), b"preexisting")

    def test_interrupted_link_cleanup_recovers_committed_entry_on_restart(self):
        signal = _signal(
            "2025-04-03",
            available_at="2025-04-03T20:15:00Z",
            effective_session="2025-04-04",
        )
        receipt = _receipt(checked_at="2025-04-03T20:15:00Z")
        original_unlink = os.unlink

        def interrupt_pending_cleanup(path, *, dir_fd=None):
            if str(path).startswith(".pending-"):
                raise OSError("simulated crash before pending unlink")
            return original_unlink(path, dir_fd=dir_fd)

        with patch.object(
            journal_module,
            "_trusted_utc_now",
            return_value=_utc("2025-04-03T20:20:00Z"),
        ):
            with patch.object(
                journal_module.os,
                "unlink",
                side_effect=interrupt_pending_cleanup,
            ):
                with self.assertRaisesRegex(
                    ProspectiveJournalStoreError,
                    "journal_append_failed",
                ):
                    self.store.append(
                        protocol=self.protocol,
                        signal=signal,
                        entitlement_receipt=receipt,
                    )

        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)
        self.assertEqual(
            len(
                [
                    path
                    for path in self.directory.iterdir()
                    if path.name.startswith(".pending-")
                ]
            ),
            1,
        )
        recovered = ProspectiveJournalStore(self.directory).load_entries(
            protocol=self.protocol
        )
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].signal, signal)
        self.assertFalse(
            any(
                path.name.startswith(".pending-")
                for path in self.directory.iterdir()
            )
        )


class RegimePromotionGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = _protocol()
        self.entries = _entries(self.protocol)
        self.outcomes = _outcomes(self.protocol, self.entries)
        self.calibration = _calibration_evidence(self.protocol)

    def test_unpinned_protocol_and_self_attested_pass_are_blocked(self):
        self_attested_review = _review_receipt(
            protocol=self.protocol,
            entries=self.entries,
            outcomes=self.outcomes,
            calibration=self.calibration,
        )
        self.assertIs(
            RegimePromotionGate.evaluate(
                protocol=self.protocol,
                entries=self.entries,
                outcomes=self.outcomes,
                calibration_evidence=self.calibration,
                review_receipt=self_attested_review,
            ),
            PromotionGateStatus.BLOCKED,
        )

    def test_committed_july_protocol_is_explicitly_blocked(self):
        root = Path(__file__).resolve().parents[1]
        protocol = ProspectiveReviewProtocol.from_json(
            (
                root
                / "docs"
                / "regime_v2_prospective_review_protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(protocol.accumulation_eligible)
        self.assertIn(
            "provider_entitlement_unresolved",
            protocol.activation_reason_codes,
        )
        self.assertIs(
            RegimePromotionGate.evaluate(
                protocol=protocol,
                entries=(),
                outcomes=(),
                calibration_evidence=None,
                review_receipt=None,
            ),
            PromotionGateStatus.BLOCKED,
        )

    def test_no_gate_status_can_authorize_execution(self):
        for status in PromotionGateStatus:
            self.assertFalse(status.may_authorize_execution)

    def test_all_promotion_receipts_round_trip_and_tamper_fails(self):
        receipt = _review_receipt(
            protocol=self.protocol,
            entries=self.entries,
            outcomes=self.outcomes,
            calibration=self.calibration,
        )

        self.assertEqual(
            ProviderEntitlementReceipt.from_json(
                self.calibration.entitlement_receipt.to_json()
            ),
            self.calibration.entitlement_receipt,
        )
        self.assertEqual(
            ProspectiveOutcomeRecord.from_json(
                self.outcomes[0].to_json()
            ),
            self.outcomes[0],
        )
        self.assertEqual(
            VerifiedCalibrationEvidence.from_json(
                self.calibration.to_json()
            ),
            self.calibration,
        )
        self.assertEqual(
            ProspectiveReviewReceipt.from_json(receipt.to_json()),
            receipt,
        )

        tampered = json.loads(receipt.to_json())
        tampered["receipt"]["statistical_report_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            ProspectiveReviewError,
            "hash does not match",
        ):
            ProspectiveReviewReceipt.from_json(
                json.dumps(
                    tampered,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )


if __name__ == "__main__":
    unittest.main()
