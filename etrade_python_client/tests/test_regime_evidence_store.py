import dataclasses
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from live_trading.regime_evidence_store import (
    SCHEMA_VERSION,
    RegimeEvidenceIntegrityError,
    RegimeEvidenceStore,
    RegimeEvidenceStoreError,
)
from live_trading.regime_market_data import (
    AVAILABILITY_PROVIDER_TIMESTAMP,
    PAYLOAD_PROVIDER_RESPONSE,
    MarketObservation,
    RegimeMarketDataSnapshot,
    SourceIdentity,
    regime_market_schedule,
)


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


def _observation(session, identity, close, *, ingested_at=None):
    schedule = regime_market_schedule(session, session)
    row = schedule.loc[pd.Timestamp(session)]
    event_at = pd.Timestamp(
        row[
            "spy_event_at"
            if identity.canonical_instrument == "SPY"
            else "vix_event_at"
        ]
    )
    ingestion_time = (
        pd.Timestamp(ingested_at)
        if ingested_at is not None
        else event_at + pd.Timedelta(minutes=2)
    )
    raw_payload = (
        f"{identity.provider}|{identity.dataset}|{identity.provider_symbol}|"
        f"{session.isoformat()}|{close}"
    ).encode("utf-8")
    return MarketObservation(
        session=session,
        identity=identity,
        close=float(close),
        event_at=event_at,
        available_at=event_at + pd.Timedelta(minutes=1),
        ingested_at=ingestion_time,
        request_id=(
            f"{identity.provider}:{session.isoformat()}:{close}:"
            f"{ingestion_time.strftime('%Y%m%dT%H%M%SZ')}"
        ),
        raw_payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
        payload_kind=PAYLOAD_PROVIDER_RESPONSE,
        availability_basis=AVAILABILITY_PROVIDER_TIMESTAMP,
        is_final=True,
    )


def _all_observations(close_offset=0.0):
    observations = []
    for position, session in enumerate(SESSIONS):
        observations.extend(
            (
                _observation(
                    session,
                    SPY_IDENTITY,
                    500.0 + position + close_offset,
                ),
                _observation(
                    session,
                    VIX_IDENTITY,
                    16.0 + position + close_offset,
                ),
            )
        )
    return tuple(observations)


def _snapshot(*, as_of=None, close_offset=0.0):
    observations = _all_observations(close_offset)
    snapshot_as_of = (
        pd.Timestamp(as_of)
        if as_of is not None
        else max(item.ingested_at for item in observations)
        + pd.Timedelta(minutes=1)
    )
    return RegimeMarketDataSnapshot(
        as_of=snapshot_as_of,
        observations=observations,
    )


class RegimeEvidenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "regime_evidence.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record_spy_success(self, store, close_offset=0.0):
        attempted_at = pd.Timestamp("2025-03-10T20:30:00Z")
        attempt_id = store.begin_attempt(
            "yahoo",
            "SPY",
            SESSIONS[0],
            SESSIONS[-1],
            attempted_at,
        )
        observations = tuple(
            _observation(
                session,
                SPY_IDENTITY,
                500.0 + position + close_offset,
                ingested_at=attempted_at + pd.Timedelta(seconds=30),
            )
            for position, session in enumerate(SESSIONS)
        )
        store.record_success(
            attempt_id,
            attempted_at + pd.Timedelta(minutes=1),
            observations,
        )
        return attempt_id

    def test_schema_quick_check_and_file_permissions(self):
        with RegimeEvidenceStore(self.db_path) as store:
            self.assertTrue(store.quick_check())
            user_version = store._connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            self.assertEqual(
                store._connection.execute("PRAGMA journal_mode").fetchone()[0],
                "wal",
            )
            self.assertEqual(
                store._connection.execute("PRAGMA synchronous").fetchone()[0],
                2,
            )
            self.assertEqual(
                store._connection.execute("PRAGMA foreign_keys").fetchone()[0],
                1,
            )
            self.assertEqual(
                store._connection.execute("PRAGMA busy_timeout").fetchone()[0],
                5_000,
            )
            tables = {
                row[0]
                for row in store._connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                )
            }

        self.assertEqual(user_version, SCHEMA_VERSION)
        self.assertTrue(
            {
                "source_attempts",
                "source_state",
                "market_observations",
                "regime_input_snapshots",
                "regime_snapshot_publications",
            }.issubset(tables)
        )
        self.assertEqual(os.stat(self.db_path).st_mode & 0o777, 0o600)

    def test_failure_preserves_last_success_observations_and_snapshot(self):
        with RegimeEvidenceStore(self.db_path) as store:
            successful_attempt = self._record_spy_success(store)
            snapshot_sha256 = store.publish_snapshot(_snapshot())
            observation_count = store._connection.execute(
                "SELECT COUNT(*) FROM market_observations"
            ).fetchone()[0]

            failed_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[-1],
                pd.Timestamp("2025-03-10T21:00:00Z"),
            )
            store.record_failure(
                failed_attempt,
                pd.Timestamp("2025-03-10T21:01:00Z"),
                "upstream_timeout",
                "The provider did not complete the request",
            )

            health = store.source_health("yahoo", "SPY")
            self.assertEqual(health["last_attempt_id"], failed_attempt)
            self.assertEqual(health["last_attempt_status"], "failure")
            self.assertEqual(health["last_success_id"], successful_attempt)
            self.assertEqual(health["last_error_code"], "upstream_timeout")
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                observation_count,
            )
            self.assertEqual(
                store.latest_snapshot().snapshot_sha256,
                snapshot_sha256,
            )

    def test_tampered_snapshot_fails_closed(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot = _snapshot()
            snapshot_sha256 = store.publish_snapshot(snapshot)
            payload = json.loads(snapshot.to_json())
            payload["observations"][0]["close"] += 1.0
            tampered_json = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            store._connection.execute(
                """
                UPDATE regime_input_snapshots
                SET snapshot_json = ?
                WHERE snapshot_sha256 = ?
                """,
                (tampered_json, snapshot_sha256),
            )

            with self.assertRaises(RegimeEvidenceIntegrityError):
                store.load_snapshot(snapshot_sha256)
            with self.assertRaises(RegimeEvidenceIntegrityError):
                store.latest_snapshot()

    def test_snapshot_publication_is_idempotent(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot = _snapshot()
            first = store.publish_snapshot(snapshot)
            second = store.publish_snapshot(snapshot)
            shadow = store.publish_snapshot(snapshot, channel="shadow")

            self.assertEqual(first, second)
            self.assertEqual(first, shadow)
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM regime_input_snapshots"
                ).fetchone()[0],
                1,
            )
            publications = store._connection.execute(
                """
                SELECT channel
                FROM regime_snapshot_publications
                ORDER BY publication_sequence
                """
            ).fetchall()
            self.assertEqual(
                [row["channel"] for row in publications],
                ["research", "shadow"],
            )
            self.assertEqual(
                store.load_snapshot(first).to_json(),
                snapshot.to_json(),
            )
            self.assertEqual(
                store.latest_snapshot("shadow").snapshot_sha256,
                first,
            )

    def test_later_published_older_backfill_does_not_regress_channel(self):
        older = _snapshot(close_offset=1.0)
        newer = _snapshot(
            as_of=older.as_of + pd.Timedelta(days=1),
            close_offset=2.0,
        )
        with RegimeEvidenceStore(self.db_path) as store:
            newer_sha256 = store.publish_snapshot(newer)
            store.publish_snapshot(older)

            self.assertEqual(
                store.latest_snapshot().snapshot_sha256,
                newer_sha256,
            )

    def test_latest_snapshot_orders_fractional_seconds_correctly(self):
        older = _snapshot(
            as_of="2025-03-10T22:00:00.000000100Z",
            close_offset=1.0,
        )
        newer = _snapshot(
            as_of="2025-03-10T22:00:00.000000900Z",
            close_offset=2.0,
        )
        with RegimeEvidenceStore(self.db_path) as store:
            newer_sha256 = store.publish_snapshot(newer)
            store.publish_snapshot(older)

            self.assertEqual(
                store.latest_snapshot().snapshot_sha256,
                newer_sha256,
            )

    def test_equal_as_of_uses_latest_channel_publication(self):
        first = _snapshot(close_offset=1.0)
        second = _snapshot(
            as_of=first.as_of,
            close_offset=2.0,
        )
        with RegimeEvidenceStore(self.db_path) as store:
            store.publish_snapshot(first)
            second_sha256 = store.publish_snapshot(second)

            self.assertEqual(
                store.latest_snapshot().snapshot_sha256,
                second_sha256,
            )

    def test_latest_snapshot_is_channel_scoped(self):
        research = _snapshot(close_offset=1.0)
        shadow = _snapshot(close_offset=2.0)
        with RegimeEvidenceStore(self.db_path) as store:
            research_sha256 = store.publish_snapshot(
                research,
                channel="research",
            )
            shadow_sha256 = store.publish_snapshot(
                shadow,
                channel="shadow",
            )

            self.assertEqual(
                store.latest_snapshot("research").snapshot_sha256,
                research_sha256,
            )
            self.assertEqual(
                store.latest_snapshot("shadow").snapshot_sha256,
                shadow_sha256,
            )

    def test_snapshot_channel_must_be_research_or_shadow(self):
        with RegimeEvidenceStore(self.db_path) as store:
            with self.assertRaisesRegex(ValueError, "channel must be one of"):
                store.publish_snapshot(_snapshot(), channel="production")
            with self.assertRaisesRegex(ValueError, "channel must be one of"):
                store.latest_snapshot("production")

            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM regime_input_snapshots"
                ).fetchone()[0],
                0,
            )

    def test_corrected_observation_appends_a_revision(self):
        with RegimeEvidenceStore(self.db_path) as store:
            first_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:03:00Z"),
            )
            store.record_success(
                first_attempt,
                pd.Timestamp("2025-03-06T21:04:00Z"),
                (
                    _observation(
                        SESSIONS[0],
                        SPY_IDENTITY,
                        500.0,
                        ingested_at=pd.Timestamp("2025-03-06T21:03:30Z"),
                    ),
                ),
            )
            second_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )
            store.record_success(
                second_attempt,
                pd.Timestamp("2025-03-06T22:01:00Z"),
                (
                    _observation(
                        SESSIONS[0],
                        SPY_IDENTITY,
                        500.5,
                        ingested_at=pd.Timestamp("2025-03-06T22:00:30Z"),
                    ),
                ),
            )

            revisions = store._connection.execute(
                """
                SELECT revision, close_value
                FROM market_observations
                WHERE provider = 'yahoo' AND instrument = 'SPY'
                  AND session_date = ?
                ORDER BY revision
                """,
                (SESSIONS[0].isoformat(),),
            ).fetchall()

            self.assertEqual(
                [(row["revision"], row["close_value"]) for row in revisions],
                [(1, "500.0"), (2, "500.5")],
            )
            self.assertEqual(
                store.source_health("yahoo", "SPY")["last_success_id"],
                second_attempt,
            )

    def test_old_observation_replay_cannot_be_a_fresh_success(self):
        observation = _observation(
            SESSIONS[0],
            SPY_IDENTITY,
            500.0,
            ingested_at=pd.Timestamp("2025-03-06T21:03:30Z"),
        )
        with RegimeEvidenceStore(self.db_path) as store:
            first_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:03:00Z"),
            )
            store.record_success(
                first_attempt,
                pd.Timestamp("2025-03-06T21:04:00Z"),
                (observation,),
            )
            second_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "observation.ingested_at cannot be before attempted_at",
            ):
                store.record_success(
                    second_attempt,
                    pd.Timestamp("2025-03-06T22:01:00Z"),
                    (observation,),
                )

            health = store.source_health("yahoo", "SPY")
            self.assertEqual(health["last_attempt_status"], "in_progress")
            self.assertEqual(
                health["last_success_id"],
                first_attempt,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                1,
            )

    def test_current_refetch_appends_a_revision_and_advances_success(self):
        with RegimeEvidenceStore(self.db_path) as store:
            first_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:03:00Z"),
            )
            store.record_success(
                first_attempt,
                pd.Timestamp("2025-03-06T21:04:00Z"),
                (
                    _observation(
                        SESSIONS[0],
                        SPY_IDENTITY,
                        500.0,
                        ingested_at=pd.Timestamp("2025-03-06T21:03:30Z"),
                    ),
                ),
            )
            second_attempt = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )
            store.record_success(
                second_attempt,
                pd.Timestamp("2025-03-06T22:01:00Z"),
                (
                    _observation(
                        SESSIONS[0],
                        SPY_IDENTITY,
                        500.0,
                        ingested_at=pd.Timestamp("2025-03-06T22:00:30Z"),
                    ),
                ),
            )

            revisions = store._connection.execute(
                """
                SELECT revision, attempt_id, ingested_at
                FROM market_observations
                WHERE provider = 'yahoo' AND instrument = 'SPY'
                  AND session_date = ?
                ORDER BY revision
                """,
                (SESSIONS[0].isoformat(),),
            ).fetchall()

            self.assertEqual(
                [
                    (row["revision"], row["attempt_id"], row["ingested_at"])
                    for row in revisions
                ],
                [
                    (1, first_attempt, "2025-03-06T21:03:30.000000000Z"),
                    (2, second_attempt, "2025-03-06T22:00:30.000000000Z"),
                ],
            )
            self.assertEqual(
                store.source_health("yahoo", "SPY")["last_success_id"],
                second_attempt,
            )

    def test_partial_success_rolls_back_without_advancing_attempt(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[-1],
                pd.Timestamp("2025-03-10T20:30:00Z"),
            )
            partial_observations = tuple(
                _observation(session, SPY_IDENTITY, 500.0 + position)
                for position, session in enumerate(SESSIONS[:-1])
            )

            with self.assertRaisesRegex(
                ValueError,
                "exactly cover requested NYSE sessions",
            ):
                store.record_success(
                    attempt_id,
                    pd.Timestamp("2025-03-10T20:31:00Z"),
                    partial_observations,
                )

            health = store.source_health("yahoo", "SPY")
            self.assertEqual(health["last_attempt_status"], "in_progress")
            self.assertIsNone(health["last_success_id"])
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )

    def test_semantically_invalid_observations_cannot_advance_success(self):
        unrecognized_identity = dataclasses.replace(
            SPY_IDENTITY,
            dataset="unregistered_daily_prices",
        )
        base = _observation(
            SESSIONS[0],
            SPY_IDENTITY,
            500.0,
            ingested_at=pd.Timestamp("2025-03-06T22:00:30Z"),
        )
        variants = (
            (
                dataclasses.replace(base, is_final=False),
                "observation must be final",
            ),
            (
                dataclasses.replace(base, identity=unrecognized_identity),
                "source identity is not recognized",
            ),
            (
                dataclasses.replace(
                    base,
                    event_at=base.event_at + pd.Timedelta(minutes=1),
                ),
                "event_at does not match the exchange close",
            ),
        )

        with RegimeEvidenceStore(self.db_path) as store:
            for observation, error in variants:
                with self.subTest(error=error):
                    attempt_id = store.begin_attempt(
                        "yahoo",
                        "SPY",
                        SESSIONS[0],
                        SESSIONS[0],
                        pd.Timestamp("2025-03-06T22:00:00Z"),
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        store.record_success(
                            attempt_id,
                            pd.Timestamp("2025-03-06T22:01:00Z"),
                            (observation,),
                        )
                    health = store.source_health("yahoo", "SPY")
                    self.assertEqual(
                        health["last_attempt_status"],
                        "in_progress",
                    )
                    self.assertIsNone(health["last_success_id"])
                    self.assertEqual(
                        store._connection.execute(
                            "SELECT COUNT(*) FROM market_observations"
                        ).fetchone()[0],
                        0,
                    )

    def test_duplicate_success_sessions_roll_back(self):
        observation = _observation(SESSIONS[0], SPY_IDENTITY, 500.0)
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:03:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate sessions",
            ):
                store.record_success(
                    attempt_id,
                    pd.Timestamp("2025-03-06T21:04:00Z"),
                    (observation, observation),
                )

            self.assertEqual(
                store.source_health("yahoo", "SPY")["last_attempt_status"],
                "in_progress",
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )

    def test_unexpected_success_session_rolls_back(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:03:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "unexpected=",
            ):
                store.record_success(
                    attempt_id,
                    pd.Timestamp("2025-03-07T22:00:00Z"),
                    (
                        _observation(SESSIONS[0], SPY_IDENTITY, 500.0),
                        _observation(SESSIONS[1], SPY_IDENTITY, 501.0),
                    ),
                )

            self.assertEqual(
                store.source_health("yahoo", "SPY")["last_attempt_status"],
                "in_progress",
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )

    def test_success_completion_cannot_precede_observation_ingestion(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T21:00:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "completed_at cannot be before observation.ingested_at",
            ):
                store.record_success(
                    attempt_id,
                    pd.Timestamp("2025-03-06T21:01:00Z"),
                    (_observation(SESSIONS[0], SPY_IDENTITY, 500.0),),
                )

            health = store.source_health("yahoo", "SPY")
            self.assertEqual(health["last_attempt_status"], "in_progress")
            self.assertIsNone(health["last_success_id"])
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )

    def test_success_completion_cannot_precede_attempt(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "completed_at cannot be before attempted_at",
            ):
                store.record_success(
                    attempt_id,
                    pd.Timestamp("2025-03-06T21:59:59Z"),
                    (_observation(SESSIONS[0], SPY_IDENTITY, 500.0),),
                )

            self.assertEqual(
                store.source_health("yahoo", "SPY")["last_attempt_status"],
                "in_progress",
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )

    def test_failure_completion_cannot_precede_attempt(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "yahoo",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "completed_at cannot be before attempted_at",
            ):
                store.record_failure(
                    attempt_id,
                    pd.Timestamp("2025-03-06T21:59:59Z"),
                    "clock_error",
                    "Completion timestamp moved backward",
                )

            health = store.source_health("yahoo", "SPY")
            self.assertEqual(health["last_attempt_status"], "in_progress")
            self.assertIsNone(health["last_success_id"])
            self.assertIsNone(health["last_error_code"])

    def test_unsupported_legacy_schema_fails_closed(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """
            CREATE TABLE regime_input_snapshots (
                snapshot_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_sha256    TEXT NOT NULL UNIQUE,
                snapshot_json      TEXT NOT NULL,
                published_at       TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "No migration is defined",
        ):
            RegimeEvidenceStore(self.db_path)

    def test_newer_schema_version_fails_closed(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        connection.close()

        with self.assertRaises(RegimeEvidenceStoreError):
            RegimeEvidenceStore(self.db_path)


if __name__ == "__main__":
    unittest.main()
