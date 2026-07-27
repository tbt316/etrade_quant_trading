import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from unittest.mock import AsyncMock, patch

import backtesting.massive_api_client as massive_client_module
from backtesting.contract_universe import (
    ConfirmedContractReferenceSnapshot,
    ContractReferencePageEvidence,
    ContractUniverseError,
    OptionContractIdentity,
)
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import (
    ContractReferenceCacheError,
    OptionDataCache,
)


UNDERLYING = "SPY"
EXPIRATION = "2025-05-16"
CONTRACT_TYPE = "put"
AS_OF = "2025-04-03"


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract(strike: int, root: str = UNDERLYING) -> dict:
    return {
        "option_ticker": f"O:{root}250516P{strike * 1000:08d}",
        "strike": float(strike),
        "contract_type": CONTRACT_TYPE,
    }


def _provider_contract(strike: int, root: str = UNDERLYING) -> dict:
    return {
        "ticker": f"O:{root}250516P{strike * 1000:08d}",
        "strike_price": float(strike),
        "contract_type": CONTRACT_TYPE,
        "expiration_date": EXPIRATION,
    }


def _begin(
    cache: OptionDataCache,
    *,
    underlying: str = UNDERLYING,
    as_of_date: str = AS_OF,
) -> str:
    roots = ("SPX", "SPXW") if underlying == "SPX" else (underlying,)
    return cache.begin_contract_reference_attempt(
        underlying=underlying,
        expiration=EXPIRATION,
        contract_type=CONTRACT_TYPE,
        as_of_date=as_of_date,
        expected_roots=roots,
    )


def _record_single_page(
    cache: OptionDataCache,
    attempt_id: str,
    *,
    result_count: int,
    response_label: str = "response",
    root_ticker: str = UNDERLYING,
    root_ordinal: int = 0,
) -> None:
    cache.record_contract_reference_page(
        attempt_id=attempt_id,
        root_ordinal=root_ordinal,
        root_ticker=root_ticker,
        page_ordinal=1,
        request_sha256=_sha256(
            f"{root_ticker}|request"
        ),
        response_sha256=_sha256(response_label),
        result_count=result_count,
        next_request_sha256=None,
        is_terminal=True,
    )


class ContractReferenceCacheTruthTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "option_data.db")
        self.cache = OptionDataCache(self.db_path)

    def tearDown(self):
        self.cache.close()
        self.temp_dir.cleanup()

    def _confirmed(
        self,
        contracts: list[dict],
        *,
        response_label: str = "response",
        as_of_date: str = AS_OF,
    ):
        attempt_id = _begin(self.cache, as_of_date=as_of_date)
        _record_single_page(
            self.cache,
            attempt_id,
            result_count=len(contracts),
            response_label=response_label,
        )
        snapshot = self.cache.confirm_contract_reference_attempt(
            attempt_id,
            contracts,
        )
        self.assertIsNotNone(snapshot)
        return attempt_id, snapshot

    def test_legacy_rows_and_complete_fetch_log_never_authorize_after_restart(self):
        self.cache.close()
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                root = Path(self.temp_dir.name) / (
                    "sharded" if sharded else "legacy"
                )
                root.mkdir()
                if sharded:
                    (root / "option_data_shards").mkdir()
                db_path = str(root / "option_data.db")
                cache = OptionDataCache(db_path)
                cache.save_contracts(
                    UNDERLYING,
                    EXPIRATION,
                    CONTRACT_TYPE,
                    AS_OF,
                    [_contract(100)],
                )
                cache.mark_fetch_complete(
                    UNDERLYING,
                    AS_OF,
                    EXPIRATION,
                    CONTRACT_TYPE,
                    "contracts",
                )
                cache.close()

                reopened = OptionDataCache(db_path)
                self.assertIsNone(
                    reopened.get_cached_contracts(
                        UNDERLYING,
                        EXPIRATION,
                        CONTRACT_TYPE,
                        AS_OF,
                    )
                )
                self.assertIsNone(
                    reopened.get_confirmed_contract_reference(
                        underlying=UNDERLYING,
                        expiration=EXPIRATION,
                        contract_type=CONTRACT_TYPE,
                        as_of_date=AS_OF,
                    )
                )
                reopened.close()
        self.cache = OptionDataCache(self.db_path)

    def test_started_attempt_and_pages_survive_crash_but_do_not_authorize(self):
        attempt_id = _begin(self.cache)
        _record_single_page(
            self.cache,
            attempt_id,
            result_count=1,
        )
        self.cache.close()
        self.cache = OptionDataCache(self.db_path)

        attempt = self.cache.get_contract_reference_attempt(attempt_id)
        self.assertEqual(attempt["status"], "STARTED")
        self.assertEqual(
            [event["event_type"] for event in
             self.cache.get_contract_reference_attempt_events(attempt_id)],
            ["STARTED"],
        )
        self.assertEqual(
            len(self.cache.get_contract_reference_pages(attempt_id)),
            1,
        )
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

    def test_confirmed_snapshot_replays_after_restart(self):
        attempt_id, original = self._confirmed(
            [_contract(95), _contract(100)]
        )
        self.cache.close()
        self.cache = OptionDataCache(self.db_path)

        replayed = self.cache.get_confirmed_contract_reference(
            underlying=UNDERLYING,
            expiration=EXPIRATION,
            contract_type=CONTRACT_TYPE,
            as_of_date=AS_OF,
        )
        self.assertEqual(replayed, original)
        self.assertEqual(
            [event["event_type"] for event in
             self.cache.get_contract_reference_attempt_events(attempt_id)],
            ["STARTED", "CONFIRMED"],
        )
        persisted = json.dumps(
            self.cache.get_contract_reference_pages(attempt_id)
            + self.cache.get_contract_reference_attempt_events(attempt_id),
            sort_keys=True,
        )
        self.assertNotIn("http", persisted.lower())
        self.assertNotIn("apikey", persisted.lower())

    def test_page_or_contract_corruption_fails_closed(self):
        attempt_id, _ = self._confirmed([_contract(100)])
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            "DROP TRIGGER trg_contract_reference_pages_no_update"
        )
        raw.execute(
            """
            UPDATE contract_reference_pages
            SET response_sha256=?
            WHERE attempt_id=?
            """,
            (_sha256("tampered"), attempt_id),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

        other_as_of = "2025-04-04"
        _, second = self._confirmed(
            [_contract(105)],
            as_of_date=other_as_of,
        )
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_snapshot_contracts_no_update
            """
        )
        raw.execute(
            """
            UPDATE contract_reference_snapshot_contracts
            SET strike_text='106'
            WHERE snapshot_sha256=?
            """,
            (second.snapshot_sha256,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=other_as_of,
            )
        )

    def test_complete_empty_is_auditable_but_never_authorizes(self):
        attempt_id = _begin(self.cache)
        _record_single_page(
            self.cache,
            attempt_id,
            result_count=0,
        )
        self.cache.finish_contract_reference_empty(attempt_id)

        attempt = self.cache.get_contract_reference_attempt(attempt_id)
        self.assertEqual(attempt["status"], "CONFIRMED_EMPTY")
        self.assertEqual(
            [event["event_type"] for event in
             self.cache.get_contract_reference_attempt_events(attempt_id)],
            ["STARTED", "CONFIRMED_EMPTY"],
        )
        self.assertIsNone(
            self.cache.get_cached_contracts(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )

    def test_same_key_identical_attempt_is_idempotent(self):
        first_id, first = self._confirmed([_contract(100)])
        second_id = _begin(self.cache)
        _record_single_page(
            self.cache,
            second_id,
            result_count=1,
        )
        second = self.cache.confirm_contract_reference_attempt(
            second_id,
            [_contract(100)],
        )

        self.assertEqual(second, first)
        self.assertEqual(
            self.cache.get_contract_reference_attempt(first_id)["status"],
            "CONFIRMED",
        )
        self.assertEqual(
            self.cache.get_contract_reference_attempt(second_id)["status"],
            "DUPLICATE_CONFIRMED",
        )

    def test_simultaneous_same_key_writers_publish_one_identical_head(self):
        barrier = threading.Barrier(2)

        def writer():
            cache = OptionDataCache(self.db_path)
            try:
                attempt_id = _begin(cache)
                _record_single_page(
                    cache,
                    attempt_id,
                    result_count=1,
                )
                barrier.wait(timeout=10)
                snapshot = cache.confirm_contract_reference_attempt(
                    attempt_id,
                    [_contract(100)],
                )
                status = cache.get_contract_reference_attempt(
                    attempt_id
                )["status"]
                return status, snapshot.snapshot_sha256
            finally:
                cache.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: writer(), range(2)))

        self.assertEqual(
            sorted(status for status, _ in results),
            ["CONFIRMED", "DUPLICATE_CONFIRMED"],
        )
        self.assertEqual(
            len({digest for _, digest in results}),
            1,
        )

    def test_same_key_different_attempt_conflicts_without_moving_head(self):
        _, first = self._confirmed([_contract(100)])
        second_id = _begin(self.cache)
        _record_single_page(
            self.cache,
            second_id,
            result_count=1,
            response_label="different-response",
        )
        second = self.cache.confirm_contract_reference_attempt(
            second_id,
            [_contract(105)],
        )

        self.assertIsNone(second)
        self.assertEqual(
            self.cache.get_contract_reference_attempt(second_id)["status"],
            "CONFLICT",
        )
        replayed = self.cache.get_confirmed_contract_reference(
            underlying=UNDERLYING,
            expiration=EXPIRATION,
            contract_type=CONTRACT_TYPE,
            as_of_date=AS_OF,
        )
        self.assertEqual(replayed, first)

    def test_future_snapshot_cannot_change_prior_snapshot(self):
        _, prior = self._confirmed(
            [_contract(95), _contract(100)],
            as_of_date=AS_OF,
        )
        _, future = self._confirmed(
            [_contract(90), _contract(95), _contract(100), _contract(105)],
            response_label="future-response",
            as_of_date="2025-04-04",
        )

        replayed_prior = self.cache.get_confirmed_contract_reference(
            underlying=UNDERLYING,
            expiration=EXPIRATION,
            contract_type=CONTRACT_TYPE,
            as_of_date=AS_OF,
        )
        self.assertEqual(replayed_prior, prior)
        self.assertNotEqual(
            replayed_prior.snapshot_sha256,
            future.snapshot_sha256,
        )
        self.assertEqual(
            [row.option_ticker for row in replayed_prior.contracts],
            [row.option_ticker for row in prior.contracts],
        )

    def test_schema_environment_and_event_tampering_fail_closed(self):
        attempt_id, snapshot = self._confirmed([_contract(100)])
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_attempt_events_no_update
            """
        )
        raw.execute(
            """
            UPDATE contract_reference_attempt_events
            SET event_type='FAILED'
            WHERE attempt_id=? AND event_ordinal=2
            """,
            (attempt_id,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

        with self.assertRaises(ContractUniverseError):
            ConfirmedContractReferenceSnapshot(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
                attempt_id=attempt_id,
                snapshot_sha256=_sha256("forged"),
                confirmed_at=snapshot.confirmed_at,
                expected_roots=snapshot.expected_roots,
                pages=snapshot.pages,
                contracts=snapshot.contracts,
            )

    def test_owned_rows_are_immutable_and_attempt_transition_is_guarded(self):
        attempt_id, snapshot = self._confirmed([_contract(100)])
        conn = self.cache._meta()

        immutable_statements = [
            (
                """
                UPDATE contract_reference_attempt_events
                SET event_type=event_type
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
            (
                """
                DELETE FROM contract_reference_attempt_events
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
            (
                """
                UPDATE contract_reference_pages
                SET response_sha256=response_sha256
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
            (
                """
                DELETE FROM contract_reference_pages
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
            (
                """
                UPDATE contract_reference_snapshots
                SET source=source
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                DELETE FROM contract_reference_snapshots
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                UPDATE contract_reference_snapshot_contracts
                SET strike_text=strike_text
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                DELETE FROM contract_reference_snapshot_contracts
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                UPDATE contract_reference_heads
                SET confirmed_at=confirmed_at
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                DELETE FROM contract_reference_heads
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                INSERT OR REPLACE INTO contract_reference_heads
                SELECT * FROM contract_reference_heads
                WHERE snapshot_sha256=?
                """,
                (snapshot.snapshot_sha256,),
            ),
            (
                """
                DELETE FROM contract_reference_attempts
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
            (
                """
                UPDATE contract_reference_attempts
                SET underlying='QQQ'
                WHERE attempt_id=?
                """,
                (attempt_id,),
            ),
        ]
        for sql, params in immutable_statements:
            with self.subTest(sql=" ".join(sql.split())[:70]):
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(sql, params)
                conn.rollback()

        started_id = _begin(self.cache, as_of_date="2025-04-04")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT OR REPLACE INTO contract_reference_attempts
                SELECT * FROM contract_reference_attempts
                WHERE attempt_id=?
                """,
                (started_id,),
            )
        conn.rollback()
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                """
                UPDATE contract_reference_attempts
                SET status='CONFIRMED',
                    finished_at='2025-04-04T23:59:59Z'
                WHERE attempt_id=?
                """,
                (started_id,),
            )
        conn.rollback()
        self.assertEqual(
            self.cache.get_contract_reference_attempt(started_id)["status"],
            "STARTED",
        )

    def test_schema_attestation_rejects_malformed_table_or_trigger(self):
        malformed_path = str(
            Path(self.temp_dir.name) / "malformed_schema.db"
        )
        raw = sqlite3.connect(malformed_path)
        raw.execute(
            """
            CREATE TABLE contract_reference_attempts (
                attempt_id TEXT PRIMARY KEY
            )
            """
        )
        raw.commit()
        raw.close()
        with self.assertRaisesRegex(
            ContractReferenceCacheError,
            "table schema mismatch",
        ):
            OptionDataCache(malformed_path)

        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_pages_no_update
            """
        )
        raw.execute(
            """
            CREATE TRIGGER trg_contract_reference_pages_no_update
            BEFORE UPDATE ON contract_reference_pages
            BEGIN
                SELECT 1;
            END
            """
        )
        raw.commit()
        raw.close()
        with self.assertRaisesRegex(
            ContractReferenceCacheError,
            "unexpected contract-reference trigger",
        ):
            OptionDataCache(self.db_path)

    def test_absent_owned_triggers_are_safely_installed_on_restart(self):
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        trigger_names = [
            row[0]
            for row in raw.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='trigger'
                  AND name LIKE 'trg_contract_reference_%'
                """
            ).fetchall()
        ]
        self.assertEqual(len(trigger_names), 18)
        for trigger_name in trigger_names:
            raw.execute(f'DROP TRIGGER "{trigger_name}"')
        raw.commit()
        raw.close()

        self.cache = OptionDataCache(self.db_path)
        reinstalled = self.cache._meta().execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='trigger'
              AND name LIKE 'trg_contract_reference_%'
            """
        ).fetchone()[0]
        self.assertEqual(reinstalled, 18)

    def test_timestamp_and_terminal_identity_tampering_fail_replay(self):
        attempt_id, snapshot = self._confirmed([_contract(100)])
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_pages_no_update
            """
        )
        raw.execute(
            """
            UPDATE contract_reference_pages
            SET recorded_at='2099-01-01T00:00:00Z'
            WHERE attempt_id=?
            """,
            (attempt_id,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

        other_as_of = "2025-04-04"
        _, other = self._confirmed(
            [_contract(105)],
            response_label="other",
            as_of_date=other_as_of,
        )
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            "DROP TRIGGER trg_contract_reference_heads_no_update"
        )
        raw.execute(
            """
            UPDATE contract_reference_heads
            SET confirmed_at='2025-04-04T00:00:00Z'
            WHERE snapshot_sha256=?
            """,
            (other.snapshot_sha256,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=other_as_of,
            )
        )

    def test_event_timestamp_tampering_fails_replay(self):
        attempt_id, _ = self._confirmed([_contract(100)])
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_attempt_events_no_update
            """
        )
        raw.execute(
            """
            UPDATE contract_reference_attempt_events
            SET recorded_at='2099-01-01T00:00:00Z'
            WHERE attempt_id=? AND event_ordinal=2
            """,
            (attempt_id,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

    def test_snapshot_terminal_identity_tampering_fails_replay(self):
        _, snapshot = self._confirmed([_contract(100)])
        self.cache.close()
        raw = sqlite3.connect(self.db_path)
        raw.execute(
            """
            DROP TRIGGER
            trg_contract_reference_snapshots_no_update
            """
        )
        raw.execute(
            """
            UPDATE contract_reference_snapshots
            SET confirmed_at='2025-04-03T00:00:00Z'
            WHERE snapshot_sha256=?
            """,
            (snapshot.snapshot_sha256,),
        )
        raw.commit()
        raw.close()
        self.cache = OptionDataCache(self.db_path)
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

    def test_page_bound_and_root_binding_are_executable_guards(self):
        with self.assertRaises(ContractReferenceCacheError):
            self.cache.begin_contract_reference_attempt(
                underlying="SPX",
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
                expected_roots=("SPX",),
            )

        spx_attempt = _begin(self.cache, underlying="SPX")
        with self.assertRaises(ContractReferenceCacheError):
            _record_single_page(
                self.cache,
                spx_attempt,
                result_count=0,
                root_ticker="SPXW",
                root_ordinal=1,
            )

        attempt_id = _begin(self.cache)
        with self.assertRaises(ContractReferenceCacheError):
            self.cache.record_contract_reference_page(
                attempt_id=attempt_id,
                root_ordinal=0,
                root_ticker=UNDERLYING,
                page_ordinal=101,
                request_sha256=_sha256("request"),
                response_sha256=_sha256("response"),
                result_count=0,
                next_request_sha256=None,
                is_terminal=True,
            )

        bound_attempt = _begin(self.cache, as_of_date="2025-04-04")
        _record_single_page(
            self.cache,
            bound_attempt,
            result_count=1,
            response_label="wrong-root",
        )
        with self.assertRaises(ContractReferenceCacheError):
            self.cache.confirm_contract_reference_attempt(
                bound_attempt,
                [_contract(100, root="QQQ")],
            )
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date="2025-04-04",
            )
        )


class ContractReferenceClientTruthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "option_data.db")
        self.cache = OptionDataCache(self.db_path)

    def tearDown(self):
        self.cache.close()
        self.temp_dir.cleanup()

    def _client(self, *, offline_only: bool = False) -> MassiveAPIClient:
        return MassiveAPIClient(
            self.cache,
            api_key="test-secret-key",
            offline_only=offline_only,
        )

    def _only_attempt(self) -> dict:
        row = self.cache._meta().execute(
            "SELECT * FROM contract_reference_attempts"
        ).fetchone()
        return dict(row)

    async def test_two_page_success_is_confirmed_and_cache_replayed(self):
        next_url = (
            f"{massive_client_module.BASE_URL}"
            "/v3/reference/options/contracts?cursor=page-two"
            "&apiKey=test-secret-key"
        )
        client = self._client()
        client._get = AsyncMock(
            side_effect=[
                {
                    "results": [_provider_contract(95)],
                    "next_url": next_url,
                },
                {
                    "results": [_provider_contract(100)],
                },
            ]
        )

        snapshot = await client.fetch_contracts_snapshot(
            UNDERLYING,
            EXPIRATION,
            CONTRACT_TYPE,
            AS_OF,
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.page_count, 2)
        self.assertEqual(len(snapshot.contracts), 2)
        attempt = self._only_attempt()
        self.assertEqual(attempt["status"], "CONFIRMED")
        persisted = json.dumps(
            self.cache.get_contract_reference_pages(
                attempt["attempt_id"]
            ),
            sort_keys=True,
        )
        self.assertNotIn("secret", persisted)
        self.assertNotIn("http", persisted.lower())
        self.assertNotIn(
            "secret",
            client._get.await_args_list[1].args[0],
        )

        replayed = await client.fetch_contracts_snapshot(
            UNDERLYING,
            EXPIRATION,
            CONTRACT_TYPE,
            AS_OF,
        )
        self.assertEqual(replayed, snapshot)
        self.assertEqual(client._get.await_count, 2)
        self.assertEqual(client.cache_hits, 1)

    async def test_offline_attempt_fails_without_authorizing_empty(self):
        client = self._client(offline_only=True)
        snapshot = await client.fetch_contracts_snapshot(
            UNDERLYING,
            EXPIRATION,
            CONTRACT_TYPE,
            AS_OF,
        )
        self.assertIsNone(snapshot)
        attempt = self._only_attempt()
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(attempt["failure_code"], "OFFLINE_ONLY")
        self.assertIsNone(
            self.cache.get_cached_contracts(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )

    async def test_partial_pagination_failure_preserves_page_evidence_only(self):
        client = self._client()
        client._get = AsyncMock(
            side_effect=[
                {
                    "results": [_provider_contract(95)],
                    "next_url": (
                        f"{massive_client_module.BASE_URL}"
                        "/v3/reference/options/contracts?cursor=page-two"
                    ),
                },
                None,
            ]
        )
        self.assertIsNone(
            await client.fetch_contracts_snapshot(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )
        attempt = self._only_attempt()
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(attempt["failure_code"], "REQUEST_FAILED")
        self.assertEqual(
            len(self.cache.get_contract_reference_pages(
                attempt["attempt_id"]
            )),
            1,
        )
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying=UNDERLYING,
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )

    async def test_pagination_loop_and_page_limit_fail_closed(self):
        initial_query = urlencode(
            {
                "underlying_ticker": UNDERLYING,
                "contract_type": CONTRACT_TYPE,
                "expiration_date": EXPIRATION,
                "as_of": AS_OF,
                "limit": 1000,
                "order": "asc",
                "sort": "strike_price",
            }
        )
        loop_url = (
            f"{massive_client_module.BASE_URL}"
            f"/v3/reference/options/contracts?{initial_query}"
        )
        client = self._client()
        client._get = AsyncMock(
            return_value={
                "results": [_provider_contract(100)],
                "next_url": loop_url,
            }
        )
        self.assertIsNone(
            await client.fetch_contracts_snapshot(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )
        self.assertEqual(
            self._only_attempt()["failure_code"],
            "PAGINATION_LOOP",
        )

        self.cache.close()
        self.db_path = str(Path(self.temp_dir.name) / "page_limit.db")
        self.cache = OptionDataCache(self.db_path)
        limited = self._client()
        limited._get = AsyncMock(
            return_value={
                "results": [_provider_contract(100)],
                "next_url": (
                    f"{massive_client_module.BASE_URL}"
                    "/v3/reference/options/contracts?cursor=page-two"
                ),
            }
        )
        with patch.object(
            massive_client_module,
            "MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT",
            1,
        ):
            self.assertIsNone(
                await limited.fetch_contracts_snapshot(
                    UNDERLYING,
                    EXPIRATION,
                    CONTRACT_TYPE,
                    AS_OF,
                )
            )
        self.assertEqual(
            self._only_attempt()["failure_code"],
            "PAGE_LIMIT_EXCEEDED",
        )

    async def test_empty_and_malformed_results_never_authorize(self):
        empty_client = self._client()
        empty_client._get = AsyncMock(return_value={"results": []})
        self.assertIsNone(
            await empty_client.fetch_contracts_snapshot(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )
        self.assertEqual(
            self._only_attempt()["status"],
            "CONFIRMED_EMPTY",
        )

        self.cache.close()
        self.db_path = str(Path(self.temp_dir.name) / "malformed.db")
        self.cache = OptionDataCache(self.db_path)
        malformed_client = self._client()
        malformed_client._get = AsyncMock(
            return_value={
                "results": [{"ticker": _contract(100)["option_ticker"]}],
            }
        )
        self.assertIsNone(
            await malformed_client.fetch_contracts_snapshot(
                UNDERLYING,
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )
        attempt = self._only_attempt()
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(
            attempt["failure_code"],
            "MALFORMED_CONTRACT_ROW",
        )
        self.assertEqual(
            len(self.cache.get_contract_reference_pages(
                attempt["attempt_id"]
            )),
            1,
        )

    async def test_spx_second_root_failure_cannot_publish_partial_union(self):
        client = self._client()
        client._get = AsyncMock(
            side_effect=[
                {"results": [_provider_contract(5000, root="SPX")]},
                None,
            ]
        )
        self.assertIsNone(
            await client.fetch_contracts_snapshot(
                "SPX",
                EXPIRATION,
                CONTRACT_TYPE,
                AS_OF,
            )
        )
        attempt = self._only_attempt()
        self.assertEqual(attempt["status"], "FAILED")
        self.assertEqual(attempt["failure_code"], "REQUEST_FAILED")
        self.assertEqual(
            len(self.cache.get_contract_reference_pages(
                attempt["attempt_id"]
            )),
            1,
        )
        self.assertIsNone(
            self.cache.get_confirmed_contract_reference(
                underlying="SPX",
                expiration=EXPIRATION,
                contract_type=CONTRACT_TYPE,
                as_of_date=AS_OF,
            )
        )


if __name__ == "__main__":
    unittest.main()
