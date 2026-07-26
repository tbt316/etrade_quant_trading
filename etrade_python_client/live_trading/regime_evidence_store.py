"""Durable storage for canonical regime-market-data evidence.

The store deliberately has a small surface:

* source attempts and their last-attempt/last-success health;
* append-only market-observation revisions; and
* immutable, content-addressed regime input snapshots with channel-scoped
  publications.

It does not fetch data or run the detector.  Those concerns stay in the market
data gateway and the pure regime detector respectively. It stores payload
checksums, not raw provider bytes, so publication here does not confer verified
provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from live_trading.regime_market_data import (
    REGIME_DATA_SCHEMA_VERSION,
    MarketObservation,
    RegimeMarketDataSnapshot,
    load_snapshot_json,
    regime_market_schedule,
)


SCHEMA_VERSION = 2
_BUSY_TIMEOUT_MS = 5_000
_SNAPSHOT_CHANNELS = frozenset({"research", "shadow"})


class RegimeEvidenceStoreError(RuntimeError):
    """Base error for evidence-store failures."""


class RegimeEvidenceIntegrityError(RegimeEvidenceStoreError):
    """Raised when persisted evidence does not match its recorded identity."""


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS source_attempts (
        attempt_id         TEXT PRIMARY KEY,
        provider           TEXT NOT NULL,
        instrument         TEXT NOT NULL,
        requested_start    TEXT NOT NULL,
        requested_end      TEXT NOT NULL,
        attempted_at       TEXT NOT NULL,
        completed_at       TEXT,
        status             TEXT NOT NULL
                           CHECK (status IN ('in_progress', 'success', 'failure')),
        observation_count  INTEGER NOT NULL DEFAULT 0
                           CHECK (observation_count >= 0),
        error_code         TEXT,
        error_message      TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_source_attempts_source_time
    ON source_attempts(provider, instrument, attempted_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS source_state (
        provider             TEXT NOT NULL,
        instrument           TEXT NOT NULL,
        last_attempt_id      TEXT NOT NULL,
        last_attempt_at      TEXT NOT NULL,
        last_attempt_status  TEXT NOT NULL
                             CHECK (last_attempt_status IN
                                    ('in_progress', 'success', 'failure')),
        last_success_id      TEXT,
        last_success_at      TEXT,
        PRIMARY KEY (provider, instrument),
        FOREIGN KEY (last_attempt_id)
            REFERENCES source_attempts(attempt_id),
        FOREIGN KEY (last_success_id)
            REFERENCES source_attempts(attempt_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_observations (
        observation_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_sha256    TEXT NOT NULL,
        attempt_id             TEXT NOT NULL,
        provider               TEXT NOT NULL,
        instrument             TEXT NOT NULL,
        session_date           TEXT NOT NULL,
        revision               INTEGER NOT NULL CHECK (revision >= 1),
        close_value            TEXT NOT NULL,
        dataset                TEXT NOT NULL,
        provider_symbol        TEXT NOT NULL,
        price_field            TEXT NOT NULL,
        adjustment             TEXT NOT NULL,
        unit                   TEXT NOT NULL,
        source_identity_json   TEXT NOT NULL,
        event_at               TEXT NOT NULL,
        available_at           TEXT NOT NULL,
        ingested_at            TEXT NOT NULL,
        request_id             TEXT NOT NULL,
        raw_payload_sha256      TEXT NOT NULL,
        payload_kind           TEXT NOT NULL,
        availability_basis     TEXT NOT NULL,
        is_final               INTEGER NOT NULL CHECK (is_final IN (0, 1)),
        observation_schema_version TEXT NOT NULL,
        observation_json       TEXT NOT NULL,
        UNIQUE (provider, instrument, session_date, revision),
        FOREIGN KEY (attempt_id)
            REFERENCES source_attempts(attempt_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_market_observations_lookup
    ON market_observations(provider, instrument, session_date, revision)
    """,
    """
    CREATE TABLE IF NOT EXISTS regime_input_snapshots (
        snapshot_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_sha256    TEXT NOT NULL UNIQUE,
        snapshot_as_of     TEXT NOT NULL,
        snapshot_json      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS regime_snapshot_publications (
        publication_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
        channel               TEXT NOT NULL
                              CHECK (channel IN ('research', 'shadow')),
        snapshot_sha256       TEXT NOT NULL,
        published_at          TEXT NOT NULL,
        UNIQUE (channel, snapshot_sha256),
        FOREIGN KEY (snapshot_sha256)
            REFERENCES regime_input_snapshots(snapshot_sha256)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_regime_snapshot_publications_channel
    ON regime_snapshot_publications(channel, publication_sequence)
    """,
)


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _snapshot_channel(value: Any) -> str:
    channel = _required_text(value, "channel").lower()
    if channel not in _SNAPSHOT_CHANNELS:
        raise ValueError(
            f"channel must be one of {sorted(_SNAPSHOT_CHANNELS)}"
        )
    return channel


def _date_iso(value: Any, field_name: str) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _utc_iso(value: Any, field_name: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    return (
        timestamp.tz_convert("UTC")
        .isoformat(timespec="nanoseconds")
        .replace("+00:00", "Z")
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _nested_source_identity(payload: dict[str, Any]) -> dict[str, Any]:
    identity = payload.get(
        "identity",
        payload.get("source_identity", payload.get("source")),
    )
    if not isinstance(identity, dict):
        raise ValueError("MarketObservation must contain a source identity")
    return identity


class RegimeEvidenceStore:
    """SQLite-backed evidence store for one-host, one-writer operation."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

        try:
            connection = sqlite3.connect(
                str(self.path),
                timeout=_BUSY_TIMEOUT_MS / 1_000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            self._conn = connection
            self._enforce_permissions()
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RegimeEvidenceStoreError(
                    f"Could not enable SQLite WAL mode: {journal_mode}"
                )
            connection.execute("PRAGMA synchronous = FULL")
            self._initialize_schema()
            self._enforce_permissions()
        except Exception:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            raise

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RegimeEvidenceStoreError("Evidence store is closed")
        return self._conn

    def __enter__(self) -> "RegimeEvidenceStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is None:
            return
        self._enforce_permissions()
        self._conn.close()
        self._conn = None

    def _enforce_permissions(self) -> None:
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if candidate.exists():
                os.chmod(candidate, 0o600)

    def _initialize_schema(self) -> None:
        connection = self._connection
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RegimeEvidenceStoreError(
                f"Evidence schema {version} is newer than supported version "
                f"{SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise RegimeEvidenceStoreError(
                f"No migration is defined from evidence schema {version}"
            )

        with self._immediate_transaction():
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            self._enforce_permissions()

    def begin_attempt(
        self,
        provider: str,
        instrument: str,
        requested_start,
        requested_end,
        attempted_at,
    ) -> str:
        provider_text = _required_text(provider, "provider")
        instrument_text = _required_text(instrument, "instrument")
        start_text = _date_iso(requested_start, "requested_start")
        end_text = _date_iso(requested_end, "requested_end")
        if end_text < start_text:
            raise ValueError("requested_end cannot be before requested_start")
        attempted_text = _utc_iso(attempted_at, "attempted_at")
        attempt_id = uuid.uuid4().hex

        with self._immediate_transaction():
            self._connection.execute(
                """
                INSERT INTO source_attempts (
                    attempt_id, provider, instrument, requested_start,
                    requested_end, attempted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'in_progress')
                """,
                (
                    attempt_id,
                    provider_text,
                    instrument_text,
                    start_text,
                    end_text,
                    attempted_text,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO source_state (
                    provider, instrument, last_attempt_id, last_attempt_at,
                    last_attempt_status
                ) VALUES (?, ?, ?, ?, 'in_progress')
                ON CONFLICT(provider, instrument) DO UPDATE SET
                    last_attempt_id = excluded.last_attempt_id,
                    last_attempt_at = excluded.last_attempt_at,
                    last_attempt_status = excluded.last_attempt_status
                """,
                (provider_text, instrument_text, attempt_id, attempted_text),
            )
        return attempt_id

    def _attempt_row(self, attempt_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM source_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown source attempt: {attempt_id}")
        return row

    @staticmethod
    def _validate_completion_time(
        attempt: sqlite3.Row,
        completed_at: str,
    ) -> None:
        attempted = pd.Timestamp(attempt["attempted_at"])
        completed = pd.Timestamp(completed_at)
        if completed < attempted:
            raise ValueError("completed_at cannot be before attempted_at")

    @staticmethod
    def _validate_success_observations(
        attempt: sqlite3.Row,
        completed_at: str,
        observations: list[dict[str, Any]],
    ) -> None:
        actual_sessions = [item["session_date"] for item in observations]
        seen_sessions: set[str] = set()
        duplicate_sessions: set[str] = set()
        for session in actual_sessions:
            if session in seen_sessions:
                duplicate_sessions.add(session)
            seen_sessions.add(session)
        if duplicate_sessions:
            raise ValueError(
                "observations contain duplicate sessions: "
                f"{sorted(duplicate_sessions)}"
            )

        schedule = regime_market_schedule(
            attempt["requested_start"],
            attempt["requested_end"],
        )
        expected_sessions = {
            session.date().isoformat()
            for session in schedule.index
        }
        actual_session_set = set(actual_sessions)
        missing = sorted(expected_sessions - actual_session_set)
        unexpected = sorted(actual_session_set - expected_sessions)
        if missing or unexpected:
            raise ValueError(
                "observations do not exactly cover requested NYSE sessions; "
                f"missing={missing}, unexpected={unexpected}"
            )

        attempted = pd.Timestamp(attempt["attempted_at"])
        completed = pd.Timestamp(completed_at)
        for item in observations:
            if not item["source_identity_recognized"]:
                raise ValueError(
                    "observation source identity is not recognized"
                )
            if not item["is_final"]:
                raise ValueError("observation must be final")
            event_column = (
                "spy_event_at"
                if item["instrument"] == "SPY"
                else "vix_event_at"
            )
            expected_event = pd.Timestamp(
                schedule.loc[
                    pd.Timestamp(item["session_date"]),
                    event_column,
                ]
            )
            if pd.Timestamp(item["event_at"]) != expected_event:
                raise ValueError(
                    "observation event_at does not match the exchange close"
                )
            ingested = pd.Timestamp(item["ingested_at"])
            if ingested < attempted:
                raise ValueError(
                    "observation.ingested_at cannot be before attempted_at"
                )
            if ingested > completed:
                raise ValueError(
                    "completed_at cannot be before observation.ingested_at"
                )

    def record_failure(
        self,
        attempt_id: str,
        completed_at,
        error_code: str,
        error_message: str,
    ) -> None:
        completed_text = _utc_iso(completed_at, "completed_at")
        error_code_text = _required_text(error_code, "error_code")[:100]
        error_message_text = _required_text(error_message, "error_message")[:1_000]

        with self._immediate_transaction():
            attempt = self._attempt_row(attempt_id)
            if attempt["status"] != "in_progress":
                raise RegimeEvidenceStoreError(
                    f"Attempt {attempt_id} is already {attempt['status']}"
                )
            self._validate_completion_time(attempt, completed_text)
            self._connection.execute(
                """
                UPDATE source_attempts
                SET completed_at = ?, status = 'failure',
                    error_code = ?, error_message = ?
                WHERE attempt_id = ?
                """,
                (
                    completed_text,
                    error_code_text,
                    error_message_text,
                    attempt_id,
                ),
            )
            self._connection.execute(
                """
                UPDATE source_state
                SET last_attempt_status = 'failure'
                WHERE provider = ? AND instrument = ?
                  AND last_attempt_id = ?
                """,
                (attempt["provider"], attempt["instrument"], attempt_id),
            )

    def record_success(
        self,
        attempt_id: str,
        completed_at,
        observations: tuple[MarketObservation, ...],
    ) -> None:
        completed_text = _utc_iso(completed_at, "completed_at")
        if not isinstance(observations, tuple) or not observations:
            raise ValueError("observations must be a non-empty tuple")

        serialized = [
            self._serialize_observation(observation)
            for observation in observations
        ]

        with self._immediate_transaction():
            attempt = self._attempt_row(attempt_id)
            if attempt["status"] != "in_progress":
                raise RegimeEvidenceStoreError(
                    f"Attempt {attempt_id} is already {attempt['status']}"
                )
            self._validate_completion_time(attempt, completed_text)
            for item in serialized:
                if item["provider"] != attempt["provider"]:
                    raise ValueError(
                        "Observation provider does not match the source attempt"
                    )
                if item["instrument"] != attempt["instrument"]:
                    raise ValueError(
                        "Observation instrument does not match the source attempt"
                    )
            self._validate_success_observations(
                attempt,
                completed_text,
                serialized,
            )
            for item in serialized:
                existing = self._connection.execute(
                    """
                    SELECT revision
                    FROM market_observations
                    WHERE provider = ? AND instrument = ?
                      AND session_date = ? AND observation_sha256 = ?
                    LIMIT 1
                    """,
                    (
                        item["provider"],
                        item["instrument"],
                        item["session_date"],
                        item["observation_sha256"],
                    ),
                ).fetchone()
                if existing is not None:
                    continue

                revision = int(
                    self._connection.execute(
                        """
                        SELECT COALESCE(MAX(revision), 0) + 1
                        FROM market_observations
                        WHERE provider = ? AND instrument = ? AND session_date = ?
                        """,
                        (
                            item["provider"],
                            item["instrument"],
                            item["session_date"],
                        ),
                    ).fetchone()[0]
                )
                self._connection.execute(
                    """
                    INSERT INTO market_observations (
                        observation_sha256, attempt_id, provider, instrument,
                        session_date, revision, close_value, dataset,
                        provider_symbol, price_field, adjustment, unit,
                        source_identity_json, event_at, available_at,
                        ingested_at, request_id, raw_payload_sha256,
                        payload_kind, availability_basis, is_final,
                        observation_schema_version, observation_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        item["observation_sha256"],
                        attempt_id,
                        item["provider"],
                        item["instrument"],
                        item["session_date"],
                        revision,
                        item["close_value"],
                        item["dataset"],
                        item["provider_symbol"],
                        item["price_field"],
                        item["adjustment"],
                        item["unit"],
                        item["source_identity_json"],
                        item["event_at"],
                        item["available_at"],
                        item["ingested_at"],
                        item["request_id"],
                        item["raw_payload_sha256"],
                        item["payload_kind"],
                        item["availability_basis"],
                        item["is_final"],
                        item["observation_schema_version"],
                        item["observation_json"],
                    ),
                )

            self._connection.execute(
                """
                UPDATE source_attempts
                SET completed_at = ?, status = 'success',
                    observation_count = ?, error_code = NULL,
                    error_message = NULL
                WHERE attempt_id = ?
                """,
                (completed_text, len(serialized), attempt_id),
            )
            self._connection.execute(
                """
                UPDATE source_state
                SET
                    last_attempt_status = CASE
                        WHEN last_attempt_id = ? THEN 'success'
                        ELSE last_attempt_status
                    END,
                    last_success_id = CASE
                        WHEN last_success_at IS NULL OR last_success_at <= ?
                        THEN ? ELSE last_success_id
                    END,
                    last_success_at = CASE
                        WHEN last_success_at IS NULL OR last_success_at <= ?
                        THEN ? ELSE last_success_at
                    END
                WHERE provider = ? AND instrument = ?
                """,
                (
                    attempt_id,
                    completed_text,
                    attempt_id,
                    completed_text,
                    completed_text,
                    attempt["provider"],
                    attempt["instrument"],
                ),
            )

    def _serialize_observation(
        self,
        observation: MarketObservation,
    ) -> dict[str, Any]:
        if not isinstance(observation, MarketObservation):
            raise TypeError("observations must contain MarketObservation values")

        payload = observation.to_dict()
        if not isinstance(payload, dict):
            raise TypeError("MarketObservation.to_dict() must return a dictionary")
        identity = _nested_source_identity(payload)
        observation_json = _canonical_json(payload)

        provider = _required_text(identity.get("provider"), "source.provider")
        dataset = _required_text(
            identity.get("dataset"),
            "source.dataset",
        )
        provider_symbol = _required_text(
            identity.get("provider_symbol"),
            "source.provider_symbol",
        )
        price_field = _required_text(
            identity.get("field"),
            "source.field",
        )
        session_date = _date_iso(
            payload.get("session"),
            "observation.session",
        )
        instrument = _required_text(
            identity.get("canonical_instrument"),
            "source.canonical_instrument",
        )
        raw_payload_sha256 = _required_text(
            payload.get("raw_payload_sha256"),
            "observation.raw_payload_sha256",
        )
        if len(raw_payload_sha256) != 64:
            raise ValueError(
                "observation.raw_payload_sha256 must be a SHA-256 hex digest"
            )
        try:
            int(raw_payload_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "observation.raw_payload_sha256 must be a SHA-256 hex digest"
            ) from exc

        is_final = payload.get("is_final")
        if not isinstance(is_final, bool):
            raise ValueError("observation.is_final must be a boolean")
        return {
            "observation_sha256": hashlib.sha256(
                observation_json.encode("utf-8")
            ).hexdigest(),
            "provider": provider,
            "instrument": instrument,
            "session_date": session_date,
            "close_value": str(observation.close),
            "dataset": dataset,
            "provider_symbol": provider_symbol,
            "price_field": price_field,
            "adjustment": _required_text(
                identity.get("adjustment"),
                "source.adjustment",
            ),
            "unit": _required_text(identity.get("unit"), "source.unit"),
            "source_identity_json": _canonical_json(identity),
            "event_at": _utc_iso(
                payload.get("event_at"),
                "observation.event_at",
            ),
            "available_at": _utc_iso(
                payload.get("available_at"),
                "observation.available_at",
            ),
            "ingested_at": _utc_iso(
                payload.get("ingested_at"),
                "observation.ingested_at",
            ),
            "request_id": _required_text(
                payload.get("request_id"),
                "observation.request_id",
            ),
            "raw_payload_sha256": raw_payload_sha256,
            "payload_kind": _required_text(
                payload.get("payload_kind"),
                "observation.payload_kind",
            ),
            "availability_basis": _required_text(
                payload.get("availability_basis"),
                "observation.availability_basis",
            ),
            "is_final": int(is_final),
            "source_identity_recognized": (
                observation.source_identity_recognized
            ),
            "observation_schema_version": REGIME_DATA_SCHEMA_VERSION,
            "observation_json": observation_json,
        }

    def publish_snapshot(
        self,
        snapshot: RegimeMarketDataSnapshot,
        channel: str = "research",
    ) -> str:
        if not isinstance(snapshot, RegimeMarketDataSnapshot):
            raise TypeError("snapshot must be a RegimeMarketDataSnapshot")
        channel_text = _snapshot_channel(channel)
        snapshot_json = snapshot.to_json()
        verified = self._verify_snapshot_json(snapshot_json)
        snapshot_sha256 = verified.snapshot_sha256
        snapshot_as_of = _utc_iso(verified.as_of, "snapshot.as_of")
        published_at = _utc_iso(
            datetime.now(timezone.utc),
            "published_at",
        )

        with self._immediate_transaction():
            existing = self._connection.execute(
                """
                SELECT snapshot_as_of, snapshot_json
                FROM regime_input_snapshots
                WHERE snapshot_sha256 = ?
                """,
                (snapshot_sha256,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["snapshot_as_of"] != snapshot_as_of
                    or existing["snapshot_json"] != snapshot_json
                ):
                    raise RegimeEvidenceIntegrityError(
                        "Existing snapshot content does not match its SHA-256"
                    )
            else:
                self._connection.execute(
                    """
                    INSERT INTO regime_input_snapshots (
                        snapshot_sha256, snapshot_as_of, snapshot_json
                    ) VALUES (?, ?, ?)
                    """,
                    (snapshot_sha256, snapshot_as_of, snapshot_json),
                )
            self._connection.execute(
                """
                INSERT INTO regime_snapshot_publications (
                    channel, snapshot_sha256, published_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(channel, snapshot_sha256) DO NOTHING
                """,
                (channel_text, snapshot_sha256, published_at),
            )
        return snapshot_sha256

    def _verify_snapshot_json(self, snapshot_json: str) -> RegimeMarketDataSnapshot:
        if not isinstance(snapshot_json, str):
            raise RegimeEvidenceIntegrityError("Snapshot payload is not text")
        try:
            snapshot = load_snapshot_json(snapshot_json)
            canonical = snapshot.to_json()
        except Exception as exc:
            raise RegimeEvidenceIntegrityError(
                "Snapshot JSON failed schema or checksum validation"
            ) from exc
        if canonical != snapshot_json:
            raise RegimeEvidenceIntegrityError(
                "Snapshot JSON is not in canonical form"
            )
        return snapshot

    def _snapshot_from_row(
        self,
        row: sqlite3.Row,
        context: str,
    ) -> RegimeMarketDataSnapshot:
        snapshot = self._verify_snapshot_json(row["snapshot_json"])
        if snapshot.snapshot_sha256 != row["snapshot_sha256"]:
            raise RegimeEvidenceIntegrityError(
                f"{context} content does not match the stored SHA-256"
            )
        if _utc_iso(snapshot.as_of, "snapshot.as_of") != row["snapshot_as_of"]:
            raise RegimeEvidenceIntegrityError(
                f"{context} as_of does not match the stored timestamp"
            )
        return snapshot

    def load_snapshot(self, sha256: str) -> RegimeMarketDataSnapshot:
        snapshot_sha256 = _required_text(sha256, "sha256")
        row = self._connection.execute(
            """
            SELECT snapshot_sha256, snapshot_as_of, snapshot_json
            FROM regime_input_snapshots
            WHERE snapshot_sha256 = ?
            """,
            (snapshot_sha256,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown regime input snapshot: {snapshot_sha256}")
        return self._snapshot_from_row(row, "Snapshot")

    def latest_snapshot(
        self,
        channel: str = "research",
    ) -> RegimeMarketDataSnapshot | None:
        channel_text = _snapshot_channel(channel)
        row = self._connection.execute(
            """
            SELECT
                snapshot.snapshot_sha256,
                snapshot.snapshot_as_of,
                snapshot.snapshot_json
            FROM regime_snapshot_publications AS publication
            JOIN regime_input_snapshots AS snapshot
              ON snapshot.snapshot_sha256 = publication.snapshot_sha256
            WHERE publication.channel = ?
            ORDER BY
                snapshot.snapshot_as_of DESC,
                publication.publication_sequence DESC
            LIMIT 1
            """,
            (channel_text,),
        ).fetchone()
        if row is None:
            return None
        return self._snapshot_from_row(row, "Latest snapshot")

    def source_health(self, provider: str, instrument: str) -> dict[str, Any]:
        provider_text = _required_text(provider, "provider")
        instrument_text = _required_text(instrument, "instrument")
        row = self._connection.execute(
            """
            SELECT
                state.provider,
                state.instrument,
                state.last_attempt_id,
                state.last_attempt_at,
                state.last_attempt_status,
                state.last_success_id,
                state.last_success_at,
                attempt.completed_at AS last_attempt_completed_at,
                attempt.error_code AS last_error_code,
                attempt.error_message AS last_error_message
            FROM source_state AS state
            JOIN source_attempts AS attempt
              ON attempt.attempt_id = state.last_attempt_id
            WHERE state.provider = ? AND state.instrument = ?
            """,
            (provider_text, instrument_text),
        ).fetchone()
        if row is None:
            return {
                "provider": provider_text,
                "instrument": instrument_text,
                "last_attempt_id": None,
                "last_attempt_at": None,
                "last_attempt_completed_at": None,
                "last_attempt_status": None,
                "last_success_id": None,
                "last_success_at": None,
                "last_error_code": None,
                "last_error_message": None,
            }
        return dict(row)

    def quick_check(self) -> bool:
        try:
            rows = self._connection.execute("PRAGMA quick_check").fetchall()
        except sqlite3.DatabaseError:
            return False
        return bool(rows) and all(str(row[0]).lower() == "ok" for row in rows)
