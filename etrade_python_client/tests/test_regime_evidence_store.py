import dataclasses
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

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
from live_trading.regime_provider_evidence import (
    CBOE_VIX_DAILY_CSV_PARSER,
    MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
    ProviderParseConfig,
    ProviderParseJob,
    RawProviderResponse,
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
MASSIVE_SPY_IDENTITY = SourceIdentity(
    provider="massive",
    dataset="aggregates",
    provider_symbol="SPY",
    canonical_instrument="SPY",
    field="close",
    adjustment="unadjusted",
    unit="usd",
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


def _provider_times(session):
    joint_close = pd.Timestamp(
        regime_market_schedule(session, session).iloc[0][
            "joint_finalization_at"
        ]
    )
    request_started_at = joint_close + pd.Timedelta(minutes=1)
    completed_at = request_started_at + pd.Timedelta(seconds=10)
    return request_started_at, completed_at


def _massive_response(
    session,
    *,
    close=500.25,
    request_started_at=None,
    completed_at=None,
):
    default_started, default_completed = _provider_times(session)
    return RawProviderResponse(
        provider="massive",
        endpoint=(
            "https://api.massive.com/v1/open-close/SPY/"
            f"{session.isoformat()}"
        ),
        requested_sessions=(session,),
        request_parameters=(("adjusted", "false"),),
        request_started_at=(
            default_started
            if request_started_at is None
            else request_started_at
        ),
        completed_at=(
            default_completed
            if completed_at is None
            else completed_at
        ),
        status_code=200,
        headers=(("Content-Type", "application/json"),),
        body=json.dumps(
            {
                "status": "OK",
                "symbol": "SPY",
                "from": session.isoformat(),
                "close": close,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )


def _cboe_response(session, *, close=16.5):
    request_started_at, completed_at = _provider_times(session)
    return RawProviderResponse(
        provider="cboe",
        endpoint=(
            "https://cdn.cboe.com/api/global/us_indices/"
            "daily_prices/VIX_History.csv"
        ),
        requested_sessions=(session,),
        request_started_at=request_started_at,
        completed_at=completed_at,
        status_code=200,
        headers=(("Content-Type", "text/csv"),),
        body=(
            "DATE,OPEN,HIGH,LOW,CLOSE\r\n"
            f"{session.strftime('%m/%d/%Y')},"
            f"16.0,17.0,15.5,{close}\r\n"
        ).encode("utf-8"),
    )


def _parse_job(fetch_sha256, parser, response, identity):
    return ProviderParseJob(
        fetch_sha256=fetch_sha256,
        parser=parser,
        config=ProviderParseConfig(
            parser_id=parser.parser_id,
            requested_sessions=response.requested_sessions,
            source_identity=identity,
            request_parameters=response.request_parameters,
        ),
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

    def _record_captured_response(
        self,
        store,
        response,
        parser,
        identity,
    ):
        attempt_id = store.begin_attempt(
            response.provider,
            identity.canonical_instrument,
            response.requested_sessions[0],
            response.requested_sessions[-1],
            response.request_started_at - pd.Timedelta(seconds=5),
        )
        fetch_sha256 = store.capture_response(attempt_id, response)
        completed_at = response.completed_at + pd.Timedelta(seconds=10)
        observations = store.record_success_from_captures(
            attempt_id,
            completed_at,
            (
                _parse_job(
                    fetch_sha256,
                    parser,
                    response,
                    identity,
                ),
            ),
        )
        return attempt_id, fetch_sha256, observations

    def _record_verified_pair(self, store):
        session = SESSIONS[0]
        spy = self._record_captured_response(
            store,
            _massive_response(session),
            MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
            MASSIVE_SPY_IDENTITY,
        )
        vix = self._record_captured_response(
            store,
            _cboe_response(session),
            CBOE_VIX_DAILY_CSV_PARSER,
            VIX_IDENTITY,
        )
        observations = spy[2] + vix[2]
        snapshot = RegimeMarketDataSnapshot(
            as_of=max(item.ingested_at for item in observations)
            + pd.Timedelta(seconds=1),
            observations=observations,
        )
        return snapshot, spy, vix

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
                "raw_payload_blobs",
                "fetch_receipts",
                "parser_receipts",
                "parser_receipt_outputs",
                "regime_input_snapshots",
                "snapshot_evidence_manifests",
                "snapshot_evidence_observations",
                "regime_snapshot_publications",
            }.issubset(tables)
        )
        self.assertEqual(os.stat(self.db_path).st_mode & 0o777, 0o600)
        self.assertEqual(
            os.stat(self.db_path.parent).st_mode & 0o077,
            0,
        )

    def test_store_rejects_permissive_parent_and_hard_linked_database(self):
        permissive_parent = Path(self.temp_dir.name) / "shared"
        permissive_parent.mkdir(mode=0o755)
        os.chmod(permissive_parent, 0o755)
        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "parent permissions must be owner-only",
        ):
            RegimeEvidenceStore(permissive_parent / "evidence.sqlite3")

        with RegimeEvidenceStore(self.db_path):
            pass
        linked_path = Path(self.temp_dir.name) / "linked.sqlite3"
        os.link(self.db_path, linked_path)
        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "single-link",
        ):
            RegimeEvidenceStore(linked_path)

    def test_store_rejects_symlinked_parent_and_companion_files(self):
        private_parent = Path(self.temp_dir.name) / "private"
        private_parent.mkdir(mode=0o700)
        os.chmod(private_parent, 0o700)
        parent_alias = Path(self.temp_dir.name) / "private-alias"
        parent_alias.symlink_to(private_parent, target_is_directory=True)
        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "parent path cannot contain symbolic links",
        ):
            RegimeEvidenceStore(parent_alias / "evidence.sqlite3")

        companion_db = Path(self.temp_dir.name) / "companion.sqlite3"
        with RegimeEvidenceStore(companion_db):
            pass
        companion_target = Path(self.temp_dir.name) / "companion-target"
        companion_target.touch(mode=0o600)
        Path(f"{companion_db}-wal").symlink_to(companion_target)
        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "securely open evidence file",
        ):
            RegimeEvidenceStore(companion_db)

        fifo_db = Path(self.temp_dir.name) / "fifo.sqlite3"
        os.mkfifo(f"{fifo_db}-wal", mode=0o600)
        with self.assertRaisesRegex(
            RegimeEvidenceStoreError,
            "regular, single-link",
        ):
            RegimeEvidenceStore(fifo_db)

    def test_existing_foreign_key_corruption_fails_on_open(self):
        with RegimeEvidenceStore(self.db_path):
            pass
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO regime_snapshot_publications (
                channel,
                snapshot_sha256,
                evidence_sha256,
                published_at
            ) VALUES (
                'research',
                ?,
                NULL,
                '2025-03-10T22:00:00.000000000Z'
            )
            """,
            ("f" * 64,),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            RegimeEvidenceIntegrityError,
            "foreign-key violations",
        ):
            RegimeEvidenceStore(self.db_path)

    def test_duplicate_body_bytes_share_blob_but_keep_distinct_fetches(self):
        session = SESSIONS[0]
        first_response = _massive_response(session)
        second_response = _massive_response(
            session,
            request_started_at=(
                first_response.request_started_at
                + pd.Timedelta(seconds=30)
            ),
            completed_at=(
                first_response.completed_at
                + pd.Timedelta(seconds=30)
            ),
        )
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "massive",
                "SPY",
                session,
                session,
                first_response.request_started_at
                - pd.Timedelta(seconds=5),
            )
            first_fetch = store.capture_response(
                attempt_id,
                first_response,
            )
            second_fetch = store.capture_response(
                attempt_id,
                second_response,
            )

            self.assertNotEqual(first_fetch, second_fetch)
            self.assertEqual(first_response.body, second_response.body)
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM raw_payload_blobs"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM fetch_receipts"
                ).fetchone()[0],
                2,
            )

    def test_capture_recomputes_body_digest_instead_of_trusting_caller(self):
        class LyingRawProviderResponse(RawProviderResponse):
            @property
            def body_sha256(self):
                return "0" * 64

        valid = _massive_response(SESSIONS[0])
        response = LyingRawProviderResponse(
            provider=valid.provider,
            endpoint=valid.endpoint,
            requested_sessions=valid.requested_sessions,
            request_parameters=valid.request_parameters,
            request_started_at=valid.request_started_at,
            completed_at=valid.completed_at,
            status_code=valid.status_code,
            headers=valid.headers,
            body=valid.body,
        )
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "massive",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                response.request_started_at - pd.Timedelta(seconds=5),
            )

            with self.assertRaisesRegex(
                RegimeEvidenceIntegrityError,
                "checksum does not match its exact bytes",
            ):
                store.capture_response(attempt_id, response)

            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM raw_payload_blobs"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM fetch_receipts"
                ).fetchone()[0],
                0,
            )

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

    def test_failure_diagnostics_reject_credential_bearing_text(self):
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "massive",
                "SPY",
                SESSIONS[0],
                SESSIONS[0],
                pd.Timestamp("2025-03-06T22:00:00Z"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "cannot contain credential-bearing text",
            ):
                store.record_failure(
                    attempt_id,
                    pd.Timestamp("2025-03-06T22:01:00Z"),
                    "upstream_unauthorized",
                    "Authorization: Bearer must-not-be-retained",
                )

            health = store.source_health("massive", "SPY")
            self.assertEqual(health["last_attempt_status"], "in_progress")
            stored = store._connection.execute(
                """
                SELECT error_code, error_message
                FROM source_attempts
                WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            self.assertIsNone(stored["error_code"])
            self.assertIsNone(stored["error_message"])

    def test_verified_publication_requires_replayable_store_evidence(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot, _, _ = self._record_verified_pair(store)

            unverified_sha256 = store.publish_snapshot(snapshot)
            self.assertIsNone(
                store.latest_snapshot(require_verified=True)
            )

            report = store.verify_snapshot(snapshot)
            verified_sha256 = store.publish_verified_snapshot(snapshot)

            self.assertTrue(report.verified)
            self.assertEqual(report.failures, ())
            self.assertEqual(
                report.verification_kind,
                "decision_time",
            )
            self.assertEqual(
                unverified_sha256,
                verified_sha256,
            )
            self.assertEqual(
                store.latest_snapshot(
                    require_verified=True
                ).snapshot_sha256,
                snapshot.snapshot_sha256,
            )
            publications = store._connection.execute(
                """
                SELECT evidence_sha256
                FROM regime_snapshot_publications
                WHERE snapshot_sha256 = ?
                ORDER BY publication_sequence
                """,
                (snapshot.snapshot_sha256,),
            ).fetchall()
            self.assertEqual(len(publications), 2)
            self.assertIsNone(publications[0]["evidence_sha256"])
            self.assertEqual(
                publications[1]["evidence_sha256"],
                report.evidence_sha256,
            )

    def test_tampered_raw_blob_fails_closed_on_reparse(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot, spy, _ = self._record_verified_pair(store)
            store.publish_verified_snapshot(snapshot)
            fetch_sha256 = spy[1]
            payload = store._connection.execute(
                """
                SELECT blob.payload_bytes, blob.payload_sha256
                FROM fetch_receipts AS fetch
                JOIN raw_payload_blobs AS blob
                  ON blob.payload_sha256 = fetch.payload_sha256
                WHERE fetch.fetch_sha256 = ?
                """,
                (fetch_sha256,),
            ).fetchone()
            original = bytes(payload["payload_bytes"])
            tampered = (
                b"X" + original[1:]
                if original[:1] != b"X"
                else b"Y" + original[1:]
            )
            store._connection.execute(
                """
                UPDATE raw_payload_blobs
                SET payload_bytes = ?
                WHERE payload_sha256 = ?
                """,
                (
                    sqlite3.Binary(tampered),
                    payload["payload_sha256"],
                ),
            )

            with self.assertRaisesRegex(
                RegimeEvidenceIntegrityError,
                "BLOB failed length or SHA-256",
            ):
                store.verify_snapshot(snapshot)
            with self.assertRaises(RegimeEvidenceIntegrityError):
                store.latest_snapshot(require_verified=True)

    def test_tampered_parser_receipt_fails_closed(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot, _, _ = self._record_verified_pair(store)
            store.publish_verified_snapshot(snapshot)
            store._connection.execute(
                """
                UPDATE parser_receipts
                SET receipt_json = '{}'
                WHERE receipt_sha256 = (
                    SELECT receipt_sha256
                    FROM parser_receipts
                    ORDER BY receipt_sha256
                    LIMIT 1
                )
                """
            )

            with self.assertRaisesRegex(
                RegimeEvidenceIntegrityError,
                "field was modified: receipt_json",
            ):
                store.verify_snapshot(snapshot)
            with self.assertRaises(RegimeEvidenceIntegrityError):
                store.latest_snapshot(require_verified=True)

    def test_partial_captured_parse_rolls_back_derived_state(self):
        response = _massive_response(SESSIONS[0])
        with RegimeEvidenceStore(self.db_path) as store:
            attempt_id = store.begin_attempt(
                "massive",
                "SPY",
                SESSIONS[0],
                SESSIONS[-1],
                response.request_started_at - pd.Timedelta(seconds=5),
            )
            fetch_sha256 = store.capture_response(attempt_id, response)

            with self.assertRaisesRegex(
                ValueError,
                "exactly cover requested NYSE sessions",
            ):
                store.record_success_from_captures(
                    attempt_id,
                    response.completed_at + pd.Timedelta(seconds=10),
                    (
                        _parse_job(
                            fetch_sha256,
                            MASSIVE_DAILY_TICKER_SUMMARY_PARSER,
                            response,
                            MASSIVE_SPY_IDENTITY,
                        ),
                    ),
                )

            self.assertEqual(
                store.source_health(
                    "massive",
                    "SPY",
                )["last_attempt_status"],
                "in_progress",
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM parser_receipts"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM market_observations"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                store._connection.execute(
                    "SELECT COUNT(*) FROM fetch_receipts"
                ).fetchone()[0],
                1,
            )

    def test_verification_rejects_cross_attempt_observation_linkage(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot, spy, vix = self._record_verified_pair(store)
            spy_observation = spy[2][0]
            serialized = store._serialize_observation(spy_observation)
            store._connection.execute(
                """
                UPDATE market_observations
                SET attempt_id = ?
                WHERE observation_sha256 = ?
                """,
                (
                    vix[0],
                    serialized["observation_sha256"],
                ),
            )

            with self.assertRaisesRegex(
                RegimeEvidenceIntegrityError,
                "same source attempt",
            ):
                store.verify_snapshot(snapshot)

    def test_verified_publication_holds_write_lock_through_reparse(self):
        with RegimeEvidenceStore(self.db_path) as store:
            snapshot, spy, _ = self._record_verified_pair(store)
            original_verify = store._verify_snapshot_locked
            lock_failures = []

            def verify_and_probe_lock(value):
                report = original_verify(value)
                contender = sqlite3.connect(
                    self.db_path,
                    timeout=0,
                    isolation_level=None,
                )
                contender.execute("PRAGMA busy_timeout = 0")
                try:
                    contender.execute(
                        """
                        UPDATE source_attempts
                        SET error_message = 'concurrent-write'
                        WHERE attempt_id = ?
                        """,
                        (spy[0],),
                    )
                except sqlite3.OperationalError as exc:
                    lock_failures.append(str(exc))
                finally:
                    contender.close()
                return report

            with mock.patch.object(
                store,
                "_verify_snapshot_locked",
                side_effect=verify_and_probe_lock,
            ):
                store.publish_verified_snapshot(snapshot)

            self.assertEqual(len(lock_failures), 1)
            self.assertIn("locked", lock_failures[0].lower())
            self.assertEqual(
                store._connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM regime_snapshot_publications
                    WHERE evidence_sha256 IS NOT NULL
                    """
                ).fetchone()[0],
                1,
            )

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

    def test_v2_publication_migrates_as_explicitly_unverified(self):
        snapshot = _snapshot()
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """
            CREATE TABLE regime_input_snapshots (
                snapshot_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_sha256    TEXT NOT NULL UNIQUE,
                snapshot_as_of     TEXT NOT NULL,
                snapshot_json      TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE regime_snapshot_publications (
                publication_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
                channel               TEXT NOT NULL
                                      CHECK (
                                          channel IN ('research', 'shadow')
                                      ),
                snapshot_sha256       TEXT NOT NULL,
                published_at          TEXT NOT NULL,
                UNIQUE (channel, snapshot_sha256),
                FOREIGN KEY (snapshot_sha256)
                    REFERENCES regime_input_snapshots(snapshot_sha256)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_regime_snapshot_publications_channel
            ON regime_snapshot_publications(
                channel,
                publication_sequence
            )
            """
        )
        connection.execute(
            """
            INSERT INTO regime_input_snapshots (
                snapshot_sha256,
                snapshot_as_of,
                snapshot_json
            ) VALUES (?, ?, ?)
            """,
            (
                snapshot.snapshot_sha256,
                snapshot.as_of.isoformat(
                    timespec="nanoseconds"
                ).replace("+00:00", "Z"),
                snapshot.to_json(),
            ),
        )
        connection.execute(
            """
            INSERT INTO regime_snapshot_publications (
                channel,
                snapshot_sha256,
                published_at
            ) VALUES ('research', ?, '2025-03-10T22:00:00.000000000Z')
            """,
            (snapshot.snapshot_sha256,),
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        connection.close()

        with RegimeEvidenceStore(self.db_path) as store:
            publication = store._connection.execute(
                """
                SELECT evidence_sha256
                FROM regime_snapshot_publications
                WHERE channel = 'research'
                """
            ).fetchone()

            self.assertEqual(
                store._connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertIsNone(publication["evidence_sha256"])
            self.assertEqual(
                store.latest_snapshot().snapshot_sha256,
                snapshot.snapshot_sha256,
            )
            self.assertIsNone(
                store.latest_snapshot(require_verified=True)
            )

    def test_newer_schema_version_fails_closed(self):
        connection = sqlite3.connect(self.db_path)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        connection.close()

        with self.assertRaises(RegimeEvidenceStoreError):
            RegimeEvidenceStore(self.db_path)


if __name__ == "__main__":
    unittest.main()
