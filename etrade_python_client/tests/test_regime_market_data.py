import dataclasses
import hashlib
import json
import unittest
from datetime import date

import pandas as pd

import live_trading.regime_market_data as regime_market_data
from live_trading.regime_market_data import (
    AVAILABILITY_HISTORICAL_ASSUMPTION,
    AVAILABILITY_LIVE_RETRIEVAL,
    AVAILABILITY_PROVIDER_TIMESTAMP,
    CALENDAR_POLICY_VERSION,
    PAYLOAD_NORMALIZED_ONLY,
    PAYLOAD_PROVIDER_RESPONSE,
    MarketObservation,
    RegimeMarketDataSnapshot,
    SourceIdentity,
    load_snapshot_json,
    regime_market_schedule,
)
from live_trading.regime_detector_v2 import detect_regimes_from_snapshot


SPY_IDENTITY = SourceIdentity(
    provider="yahoo",
    dataset="daily_prices",
    provider_symbol="SPY",
    canonical_instrument="SPY",
    field="close",
    adjustment="unadjusted",
    unit="USD",
)
VIX_IDENTITY = SourceIdentity(
    provider="cboe",
    dataset="vix_daily",
    provider_symbol="VIX",
    canonical_instrument="VIX",
    field="close",
    adjustment="none",
    unit="index_points",
)
SESSIONS = (
    date(2025, 3, 6),
    date(2025, 3, 7),
    date(2025, 3, 10),
)


def _event_times(session):
    schedule = regime_market_schedule(session, session)
    row = schedule.loc[pd.Timestamp(session)]
    return pd.Timestamp(row["spy_event_at"]), pd.Timestamp(row["vix_event_at"])


def _observation(session, identity, close, **overrides):
    spy_event_at, vix_event_at = _event_times(session)
    event_at = (
        spy_event_at
        if identity.canonical_instrument == "SPY"
        else vix_event_at
    )
    payload = (
        f"{identity.provider}|{identity.dataset}|{identity.provider_symbol}|"
        f"{session.isoformat()}|{close}"
    ).encode("utf-8")
    values = {
        "session": session,
        "identity": identity,
        "close": float(close),
        "event_at": event_at,
        "available_at": event_at + pd.Timedelta(minutes=1),
        "ingested_at": event_at + pd.Timedelta(minutes=2),
        "request_id": f"{identity.provider}:{session.isoformat()}",
        "raw_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_kind": PAYLOAD_PROVIDER_RESPONSE,
        "availability_basis": AVAILABILITY_PROVIDER_TIMESTAMP,
        "is_final": True,
    }
    values.update(overrides)
    return MarketObservation(**values)


def _observations():
    observations = []
    for position, session in enumerate(SESSIONS):
        observations.extend(
            [
                _observation(session, SPY_IDENTITY, 500.0 + position),
                _observation(session, VIX_IDENTITY, 16.0 + position),
            ]
        )
    return observations


def _snapshot(observations=None, as_of=None):
    observations = list(observations if observations is not None else _observations())
    if as_of is None:
        as_of = max(item.ingested_at for item in observations) + pd.Timedelta(minutes=1)
    return RegimeMarketDataSnapshot(
        as_of=as_of,
        observations=tuple(observations),
    )


class RegimeMarketDataTests(unittest.TestCase):
    def test_schedule_has_fixed_dst_and_early_close_boundaries(self):
        dst_schedule = regime_market_schedule(date(2025, 3, 7), date(2025, 3, 10))

        self.assertEqual(
            pd.Timestamp(dst_schedule.loc[pd.Timestamp("2025-03-07"), "spy_event_at"]),
            pd.Timestamp("2025-03-07T21:00:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(dst_schedule.loc[pd.Timestamp("2025-03-07"), "vix_event_at"]),
            pd.Timestamp("2025-03-07T21:15:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(dst_schedule.loc[pd.Timestamp("2025-03-10"), "spy_event_at"]),
            pd.Timestamp("2025-03-10T20:00:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(dst_schedule.loc[pd.Timestamp("2025-03-10"), "vix_event_at"]),
            pd.Timestamp("2025-03-10T20:15:00Z"),
        )

        early_close = regime_market_schedule(date(2025, 11, 28), date(2025, 11, 28))
        row = early_close.loc[pd.Timestamp("2025-11-28")]
        self.assertEqual(
            pd.Timestamp(row["spy_event_at"]),
            pd.Timestamp("2025-11-28T18:00:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(row["vix_event_at"]),
            pd.Timestamp("2025-11-28T18:15:00Z"),
        )
        self.assertEqual(
            pd.Timestamp(row["joint_finalization_at"]),
            pd.Timestamp("2025-11-28T18:15:00Z"),
        )

    def test_records_are_frozen_and_session_must_be_a_date(self):
        observation = _observations()[0]
        snapshot = _snapshot()

        with self.assertRaises(dataclasses.FrozenInstanceError):
            observation.close = 999.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.as_of = pd.Timestamp("2025-03-11T00:00:00Z")
        with self.assertRaises((TypeError, ValueError)):
            dataclasses.replace(
                observation,
                session=pd.Timestamp("2025-03-06T16:00:00-05:00"),
            )

    def test_provider_symbol_swaps_and_vvix_identity_are_rejected(self):
        invalid_identities = (
            dataclasses.replace(SPY_IDENTITY, provider_symbol="^VIX"),
            dataclasses.replace(VIX_IDENTITY, provider_symbol="VVIX"),
            dataclasses.replace(VIX_IDENTITY, canonical_instrument="SPY"),
        )

        for identity in invalid_identities:
            with self.subTest(identity=identity):
                with self.assertRaises((TypeError, ValueError)):
                    observations = _observations()
                    observations[0] = dataclasses.replace(
                        observations[0],
                        identity=identity,
                    )
                    _snapshot(observations)

    def test_valid_snapshot_returns_exact_detector_inputs(self):
        snapshot = _snapshot()
        inputs = snapshot.detector_inputs()

        self.assertTrue(snapshot.schema_version)
        self.assertEqual(len(snapshot.schedule_sha256), 64)
        self.assertEqual(len(snapshot.snapshot_sha256), 64)
        self.assertTrue(snapshot.provenance_evidence_complete)
        self.assertFalse(snapshot.provenance_verified)
        self.assertIn(
            "SNAPSHOT:durable_raw_payload_unverified",
            snapshot.provenance_failures,
        )
        self.assertFalse(inputs["source_provenance_verified"])

        prices = inputs["prices"]
        self.assertEqual(list(prices.columns), ["SPY_Close", "VIX_Close"])
        self.assertEqual(
            list(prices.index),
            [pd.Timestamp(session) for session in SESSIONS],
        )
        self.assertEqual(prices["SPY_Close"].tolist(), [500.0, 501.0, 502.0])
        self.assertEqual(prices["VIX_Close"].tolist(), [16.0, 17.0, 18.0])

        expected = {
            (item.identity.canonical_instrument, item.session): max(
                item.event_at,
                item.available_at,
                item.ingested_at,
            )
            for item in snapshot.observations
        }
        for session in SESSIONS:
            index = pd.Timestamp(session)
            self.assertEqual(
                inputs["spy_available_at"].loc[index],
                expected[("SPY", session)],
            )
            self.assertEqual(
                inputs["vix_available_at"].loc[index],
                expected[("VIX", session)],
            )

    def test_missing_required_leg_is_not_forward_filled(self):
        observations = [
            item
            for item in _observations()
            if not (
                item.session == SESSIONS[1]
                and item.identity.canonical_instrument == "VIX"
            )
        ]

        with self.assertRaises((TypeError, ValueError)):
            _snapshot(observations)

    def test_duplicate_required_leg_is_rejected(self):
        observations = _observations()
        observations.append(observations[-1])

        with self.assertRaises((TypeError, ValueError)):
            _snapshot(observations)

    def test_naive_timestamps_are_rejected(self):
        valid = _observations()[0]
        timestamp_fields = ("event_at", "available_at", "ingested_at")

        for field_name in timestamp_fields:
            with self.subTest(field=field_name):
                naive = pd.Timestamp(getattr(valid, field_name)).tz_localize(None)
                with self.assertRaises((TypeError, ValueError)):
                    dataclasses.replace(valid, **{field_name: naive})

        with self.assertRaises((TypeError, ValueError)):
            _snapshot(as_of=pd.Timestamp("2025-03-10T21:00:00"))

    def test_timestamp_ordering_and_as_of_are_enforced(self):
        valid = _observations()[0]
        invalid_changes = (
            {"available_at": valid.event_at - pd.Timedelta(seconds=1)},
            {"ingested_at": valid.available_at - pd.Timedelta(seconds=1)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    dataclasses.replace(valid, **changes)

        observations = _observations()
        too_early_as_of = max(item.ingested_at for item in observations) - pd.Timedelta(seconds=1)
        with self.assertRaises((TypeError, ValueError)):
            _snapshot(observations, as_of=too_early_as_of)

    def test_exchange_event_time_mismatch_is_rejected(self):
        observations = _observations()
        valid = observations[0]
        observations[0] = dataclasses.replace(
            valid,
            event_at=valid.event_at + pd.Timedelta(minutes=1),
        )

        with self.assertRaisesRegex(ValueError, "EVENT_TIME_MISMATCH"):
            _snapshot(observations)

    def test_delayed_ingestion_controls_effective_availability(self):
        observations = _observations()
        target = next(
            item
            for item in observations
            if item.session == SESSIONS[-1]
            and item.identity.canonical_instrument == "SPY"
        )
        delayed_ingestion = pd.Timestamp("2025-03-11T15:00:00Z")
        replacement = dataclasses.replace(
            target,
            availability_basis=AVAILABILITY_LIVE_RETRIEVAL,
            ingested_at=delayed_ingestion,
        )
        observations[observations.index(target)] = replacement
        snapshot = _snapshot(
            observations,
            as_of=delayed_ingestion + pd.Timedelta(minutes=1),
        )

        inputs = snapshot.detector_inputs()
        self.assertEqual(
            inputs["spy_available_at"].loc[pd.Timestamp(SESSIONS[-1])],
            delayed_ingestion,
        )
        self.assertLess(target.available_at, delayed_ingestion)

    def test_snapshot_hash_and_json_are_independent_of_input_order(self):
        observations = _observations()
        forward = _snapshot(observations)
        reversed_snapshot = _snapshot(list(reversed(observations)), as_of=forward.as_of)

        self.assertEqual(forward.schedule_sha256, reversed_snapshot.schedule_sha256)
        self.assertEqual(forward.snapshot_sha256, reversed_snapshot.snapshot_sha256)
        self.assertEqual(forward.to_json(), reversed_snapshot.to_json())

    def test_material_observation_change_changes_snapshot_hash(self):
        observations = _observations()
        original = _snapshot(observations)
        observations[0] = dataclasses.replace(
            observations[0],
            close=observations[0].close + 0.01,
        )
        changed = _snapshot(observations, as_of=original.as_of)

        self.assertNotEqual(original.snapshot_sha256, changed.snapshot_sha256)

    def test_serialization_round_trip_preserves_canonical_snapshot(self):
        snapshot = _snapshot()
        restored = load_snapshot_json(snapshot.to_json())

        self.assertEqual(restored.snapshot_sha256, snapshot.snapshot_sha256)
        self.assertEqual(restored.schedule_sha256, snapshot.schedule_sha256)
        self.assertEqual(restored.provenance_verified, snapshot.provenance_verified)
        self.assertEqual(restored.to_json(), snapshot.to_json())
        self.assertEqual(restored.detector_inputs()["prices"].to_dict(), snapshot.detector_inputs()["prices"].to_dict())

    def test_snapshot_binds_the_source_policy_and_frozen_verdict(self):
        snapshot = _snapshot()
        serialized = snapshot.to_json()
        original_hash = snapshot.snapshot_sha256
        original_evidence_status = snapshot.provenance_evidence_complete
        registry_key = SPY_IDENTITY.registry_key
        original_policy = regime_market_data._SOURCE_IDENTITY_POLICY.pop(
            registry_key
        )
        try:
            self.assertEqual(snapshot.snapshot_sha256, original_hash)
            self.assertEqual(
                snapshot.provenance_evidence_complete,
                original_evidence_status,
            )
            with self.assertRaisesRegex(ValueError, "SOURCE_POLICY_MISMATCH"):
                load_snapshot_json(serialized)
        finally:
            regime_market_data._SOURCE_IDENTITY_POLICY[registry_key] = (
                original_policy
            )

    def test_serialized_snapshot_tampering_fails_closed(self):
        snapshot = _snapshot()
        payload = json.loads(snapshot.to_json())
        payload["observations"][0]["close"] += 1.0
        tampered = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        with self.assertRaises((TypeError, ValueError)):
            load_snapshot_json(tampered)

    def test_serialized_observation_cannot_omit_finality(self):
        payload = json.loads(_snapshot().to_json())
        del payload["observations"][0]["is_final"]
        missing_finality = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self.assertRaisesRegex(ValueError, "MISSING_IS_FINAL"):
            load_snapshot_json(missing_finality)

    def test_normalized_and_historical_inputs_are_explicitly_unverified(self):
        variants = (
            {
                "payload_kind": PAYLOAD_NORMALIZED_ONLY,
                "availability_basis": AVAILABILITY_PROVIDER_TIMESTAMP,
            },
            {
                "payload_kind": PAYLOAD_PROVIDER_RESPONSE,
                "availability_basis": AVAILABILITY_HISTORICAL_ASSUMPTION,
            },
        )

        for changes in variants:
            with self.subTest(changes=changes):
                observations = [
                    dataclasses.replace(item, **changes)
                    for item in _observations()
                ]
                snapshot = _snapshot(observations)
                inputs = snapshot.detector_inputs()

                self.assertFalse(snapshot.provenance_verified)
                self.assertTrue(snapshot.provenance_failures)
                self.assertFalse(inputs["source_provenance_verified"])

    def test_complete_snapshot_stays_unverified_and_execution_ineligible(self):
        snapshot = _snapshot()

        result = detect_regimes_from_snapshot(snapshot)

        self.assertFalse(result.attrs["source_provenance_verified"])
        self.assertTrue(result["Input_Provenance_Status"].eq("unverified").all())
        self.assertTrue(
            result["Input_Provenance_Evidence"]
            .eq("complete_but_not_durably_verified")
            .all()
        )
        self.assertTrue(result["Detector_Stage"].eq("shadow").all())
        self.assertFalse(result["Execution_Eligible"].any())
        self.assertEqual(result.attrs["detector_stage"], "shadow")
        self.assertFalse(result.attrs["execution_eligible"])
        self.assertEqual(
            result.attrs["calendar_policy_version"],
            CALENDAR_POLICY_VERSION,
        )
        self.assertTrue(
            result["Source_Policy_SHA256"]
            .eq(snapshot.source_policy_sha256)
            .all()
        )
        self.assertEqual(
            result.attrs["exchange_calendars"],
            ("NYSE", "CBOE_Index_Options"),
        )
        self.assertTrue(
            result.attrs["input_provenance_evidence_complete"]
        )
        self.assertIn(
            "durable_raw_payload_unverified",
            result.attrs["input_provenance_failure_codes"],
        )
        self.assertEqual(
            result.attrs["input_snapshot_sha256"],
            snapshot.snapshot_sha256,
        )

    def test_delayed_snapshot_ingestion_advances_tradable_session(self):
        baseline = detect_regimes_from_snapshot(_snapshot())
        observations = _observations()
        target = next(
            item
            for item in observations
            if item.session == SESSIONS[-1]
            and item.identity.canonical_instrument == "SPY"
        )
        delayed_ingestion = pd.Timestamp("2025-03-11T15:00:00Z")
        observations[observations.index(target)] = dataclasses.replace(
            target,
            availability_basis=AVAILABILITY_LIVE_RETRIEVAL,
            ingested_at=delayed_ingestion,
        )
        delayed_snapshot = _snapshot(
            observations,
            as_of=delayed_ingestion + pd.Timedelta(minutes=1),
        )

        delayed = detect_regimes_from_snapshot(delayed_snapshot)
        latest_session = pd.Timestamp(SESSIONS[-1])

        self.assertEqual(
            baseline.loc[latest_session, "Tradable_Session"],
            pd.Timestamp("2025-03-11"),
        )
        self.assertEqual(
            delayed.loc[latest_session, "Signal_Available_At"],
            delayed_ingestion,
        )
        self.assertEqual(
            delayed.loc[latest_session, "Tradable_Session"],
            pd.Timestamp("2025-03-12"),
        )

    def test_ingestion_at_next_open_rolls_to_following_session(self):
        observations = _observations()
        target = next(
            item
            for item in observations
            if item.session == SESSIONS[-1]
            and item.identity.canonical_instrument == "SPY"
        )
        next_open = pd.Timestamp("2025-03-11T13:30:00Z")
        observations[observations.index(target)] = dataclasses.replace(
            target,
            availability_basis=AVAILABILITY_LIVE_RETRIEVAL,
            ingested_at=next_open,
        )
        snapshot = _snapshot(
            observations,
            as_of=next_open + pd.Timedelta(minutes=1),
        )

        result = detect_regimes_from_snapshot(snapshot)

        self.assertEqual(
            result.loc[pd.Timestamp(SESSIONS[-1]), "Signal_Available_At"],
            next_open,
        )
        self.assertEqual(
            result.loc[pd.Timestamp(SESSIONS[-1]), "Tradable_Session"],
            pd.Timestamp("2025-03-12"),
        )

    def test_normalized_snapshot_provenance_remains_false_in_detector(self):
        observations = [
            dataclasses.replace(item, payload_kind=PAYLOAD_NORMALIZED_ONLY)
            for item in _observations()
        ]
        snapshot = _snapshot(observations)

        result = detect_regimes_from_snapshot(snapshot)

        self.assertFalse(result.attrs["source_provenance_verified"])
        self.assertTrue(result["Input_Provenance_Status"].eq("unverified").all())
        self.assertTrue(
            result["Data_Quality"]
            .str.contains("source_provenance_unverified")
            .all()
        )
        self.assertIn(
            "raw_provider_payload_unavailable",
            result.attrs["input_provenance_failure_codes"],
        )
        self.assertFalse(result["Execution_Eligible"].any())


if __name__ == "__main__":
    unittest.main()
