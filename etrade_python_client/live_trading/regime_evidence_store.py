"""Durable storage for canonical regime-market-data evidence.

The store deliberately has a small surface:

* source attempts and their last-attempt/last-success health;
* exact raw provider-response BLOBs and fetch/parser receipts;
* append-only market-observation revisions linked to parser outputs; and
* immutable, content-addressed regime input snapshots with channel-scoped,
  explicitly verified or unverified publications.

It does not fetch data or run the detector.  Those concerns stay in the market
data gateway and the pure regime detector respectively.  The legacy publication
API remains explicitly unverified; verified publication reparses the retained
bytes under the store's write lock before committing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
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


SCHEMA_VERSION = 3
_BUSY_TIMEOUT_MS = 5_000
_SNAPSHOT_CHANNELS = frozenset({"research", "shadow"})
_VERIFIED_SNAPSHOT_SOURCES = (
    ("massive", "SPY"),
    ("cboe", "VIX"),
)
_SHA256_HEX_LENGTH = 64
_FETCH_HASH_DOMAIN = b"regime-fetch-receipt.v1\0"
_PARSER_HASH_DOMAIN = b"regime-parser-receipt.v1\0"
_MANIFEST_HASH_DOMAIN = b"regime-output-manifest.v1\0"
_EVIDENCE_HASH_DOMAIN = b"regime-snapshot-evidence.v1\0"
_SENSITIVE_DIAGNOSTIC = re.compile(
    r"(?:authorization|bearer\s+|basic\s+|api[-_ ]?key|"
    r"access[-_ ]?token|secret|cookie|password)",
    re.IGNORECASE,
)


class RegimeEvidenceStoreError(RuntimeError):
    """Base error for evidence-store failures."""


class RegimeEvidenceIntegrityError(RegimeEvidenceStoreError):
    """Raised when persisted evidence does not match its recorded identity."""


@dataclass(frozen=True)
class SnapshotEvidenceReport:
    """Store-derived provenance verdict for one immutable input snapshot."""

    snapshot_sha256: str
    verified: bool
    evidence_sha256: str | None
    verification_kind: str | None
    verified_at: str | None
    failures: tuple[str, ...]


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
    CREATE UNIQUE INDEX IF NOT EXISTS idx_market_observations_sha256
    ON market_observations(observation_sha256)
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_payload_blobs (
        payload_sha256  TEXT PRIMARY KEY,
        byte_length     INTEGER NOT NULL CHECK (byte_length >= 0),
        payload_bytes   BLOB NOT NULL,
        stored_at       TEXT NOT NULL,
        CHECK (byte_length = length(payload_bytes))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fetch_receipts (
        fetch_sha256          TEXT PRIMARY KEY,
        attempt_id            TEXT NOT NULL,
        payload_sha256        TEXT NOT NULL,
        provider              TEXT NOT NULL,
        endpoint              TEXT NOT NULL,
        requested_sessions_json TEXT NOT NULL,
        request_parameters_json TEXT NOT NULL,
        request_started_at    TEXT NOT NULL,
        response_completed_at TEXT NOT NULL,
        status_code           INTEGER NOT NULL
                              CHECK (status_code BETWEEN 100 AND 599),
        headers_json          TEXT NOT NULL,
        media_type            TEXT NOT NULL,
        content_encoding      TEXT NOT NULL,
        response_metadata_json TEXT NOT NULL,
        receipt_json          TEXT NOT NULL,
        FOREIGN KEY (attempt_id)
            REFERENCES source_attempts(attempt_id),
        FOREIGN KEY (payload_sha256)
            REFERENCES raw_payload_blobs(payload_sha256),
        UNIQUE (attempt_id, fetch_sha256)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fetch_receipts_attempt
    ON fetch_receipts(attempt_id, response_completed_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS parser_receipts (
        receipt_sha256         TEXT PRIMARY KEY,
        fetch_sha256           TEXT NOT NULL,
        parser_id              TEXT NOT NULL,
        parser_version         TEXT NOT NULL,
        parser_code_sha256     TEXT NOT NULL,
        parser_config_sha256   TEXT NOT NULL,
        parser_config_json     TEXT NOT NULL,
        output_manifest_sha256 TEXT NOT NULL,
        output_manifest_json   TEXT NOT NULL,
        receipt_json           TEXT NOT NULL,
        first_parsed_at        TEXT NOT NULL,
        FOREIGN KEY (fetch_sha256)
            REFERENCES fetch_receipts(fetch_sha256),
        UNIQUE (
            fetch_sha256,
            parser_id,
            parser_version,
            parser_code_sha256,
            parser_config_sha256
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS parser_receipt_outputs (
        receipt_sha256      TEXT NOT NULL,
        output_ordinal      INTEGER NOT NULL CHECK (output_ordinal >= 0),
        observation_sha256  TEXT NOT NULL,
        source_locator_json TEXT NOT NULL,
        PRIMARY KEY (receipt_sha256, output_ordinal),
        UNIQUE (receipt_sha256, observation_sha256),
        FOREIGN KEY (receipt_sha256)
            REFERENCES parser_receipts(receipt_sha256),
        FOREIGN KEY (observation_sha256)
            REFERENCES market_observations(observation_sha256)
    )
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
    CREATE TABLE IF NOT EXISTS snapshot_evidence_manifests (
        evidence_sha256    TEXT PRIMARY KEY,
        snapshot_sha256    TEXT NOT NULL,
        verification_kind TEXT NOT NULL
                          CHECK (
                              verification_kind IN (
                                  'decision_time',
                                  'verified_replay'
                              )
                          ),
        verified_at        TEXT NOT NULL,
        manifest_json      TEXT NOT NULL,
        UNIQUE (evidence_sha256, snapshot_sha256),
        FOREIGN KEY (snapshot_sha256)
            REFERENCES regime_input_snapshots(snapshot_sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snapshot_evidence_observations (
        evidence_sha256     TEXT NOT NULL,
        observation_sha256  TEXT NOT NULL,
        receipt_sha256      TEXT NOT NULL,
        PRIMARY KEY (evidence_sha256, observation_sha256),
        FOREIGN KEY (evidence_sha256)
            REFERENCES snapshot_evidence_manifests(evidence_sha256),
        FOREIGN KEY (receipt_sha256, observation_sha256)
            REFERENCES parser_receipt_outputs(
                receipt_sha256,
                observation_sha256
            )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS regime_snapshot_publications (
        publication_sequence  INTEGER PRIMARY KEY AUTOINCREMENT,
        channel               TEXT NOT NULL
                              CHECK (channel IN ('research', 'shadow')),
        snapshot_sha256       TEXT NOT NULL,
        evidence_sha256       TEXT,
        published_at          TEXT NOT NULL,
        FOREIGN KEY (snapshot_sha256)
            REFERENCES regime_input_snapshots(snapshot_sha256),
        FOREIGN KEY (evidence_sha256, snapshot_sha256)
            REFERENCES snapshot_evidence_manifests(
                evidence_sha256,
                snapshot_sha256
            )
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS
        idx_regime_publications_unverified
    ON regime_snapshot_publications(channel, snapshot_sha256)
    WHERE evidence_sha256 IS NULL
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS
        idx_regime_publications_verified
    ON regime_snapshot_publications(
        channel,
        snapshot_sha256,
        evidence_sha256
    )
    WHERE evidence_sha256 IS NOT NULL
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


def _content_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_sha256(domain: bytes, payload: str | bytes) -> str:
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(domain + encoded).hexdigest()


def _validate_sha256(value: Any, field_name: str) -> str:
    digest = _required_text(value, field_name).lower()
    if len(digest) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest") from exc
    return digest


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
        requested_path = Path(os.path.abspath(os.fspath(path)))
        if requested_path.is_symlink():
            raise RegimeEvidenceStoreError(
                "Evidence database cannot be a symbolic link"
            )
        for candidate in (
            requested_path.parent,
            *requested_path.parent.parents,
        ):
            if candidate == Path(candidate.anchor) or not candidate.is_symlink():
                continue
            link_stat = os.lstat(candidate)
            container_stat = os.lstat(candidate.parent)
            trusted_system_alias = (
                link_stat.st_uid == 0
                and container_stat.st_uid == 0
                and not stat.S_IMODE(container_stat.st_mode) & 0o022
            )
            if not trusted_system_alias:
                raise RegimeEvidenceStoreError(
                    "Evidence database parent path cannot contain symbolic "
                    "links"
                )
        self.path = requested_path.parent.resolve(strict=False) / requested_path.name
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn: sqlite3.Connection | None = None
        self._prepare_parent_directory()
        self._prepare_database_file()
        self._enforce_permissions()

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
            self._validate_database_integrity()
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
            Path(f"{self.path}-journal"),
        ):
            self._secure_evidence_file(candidate, missing_ok=True)

    def _prepare_parent_directory(self) -> None:
        parent = self.path.parent
        for candidate in (parent, *parent.parents):
            if candidate == Path(candidate.anchor):
                continue
            if candidate.is_symlink():
                raise RegimeEvidenceStoreError(
                    "Evidence database parent path cannot contain symbolic "
                    "links"
                )
        parent_stat = os.lstat(parent)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RegimeEvidenceStoreError(
                "Evidence database parent must be a directory"
            )
        if (
            hasattr(os, "geteuid")
            and parent_stat.st_uid != os.geteuid()
        ):
            raise RegimeEvidenceStoreError(
                "Evidence database parent must be owned by the current user"
            )
        if stat.S_IMODE(parent_stat.st_mode) & 0o077:
            raise RegimeEvidenceStoreError(
                "Evidence database parent permissions must be owner-only"
            )

    @staticmethod
    def _secure_evidence_file(
        candidate: Path,
        *,
        missing_ok: bool,
    ) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_NOCTTY"):
            flags |= os.O_NOCTTY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(candidate, flags)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        except OSError as exc:
            raise RegimeEvidenceStoreError(
                f"Could not securely open evidence file: {candidate.name}"
            ) from exc
        try:
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_nlink != 1
            ):
                raise RegimeEvidenceStoreError(
                    "Evidence files must be regular, single-link files"
                )
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def _prepare_database_file(self) -> None:
        if self.path.exists():
            self._secure_evidence_file(self.path, missing_ok=False)
            return
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        os.close(descriptor)

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
        if version not in (0, 2):
            raise RegimeEvidenceStoreError(
                f"No migration is defined from evidence schema {version}"
            )

        with self._immediate_transaction():
            if version == 2:
                self._migrate_schema_v2_to_v3()
            else:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _validate_database_integrity(self) -> None:
        foreign_key_failures = self._connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_failures:
            raise RegimeEvidenceIntegrityError(
                "Evidence database contains foreign-key violations"
            )
        quick_check = self._connection.execute(
            "PRAGMA quick_check"
        ).fetchall()
        if (
            not quick_check
            or any(str(row[0]).lower() != "ok" for row in quick_check)
        ):
            raise RegimeEvidenceIntegrityError(
                "Evidence database failed SQLite quick_check"
            )

    def _migrate_schema_v2_to_v3(self) -> None:
        connection = self._connection
        connection.execute(
            """
            ALTER TABLE regime_snapshot_publications
            RENAME TO regime_snapshot_publications_v2
            """
        )
        connection.execute(
            "DROP INDEX IF EXISTS idx_regime_snapshot_publications_channel"
        )
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO regime_snapshot_publications (
                publication_sequence,
                channel,
                snapshot_sha256,
                evidence_sha256,
                published_at
            )
            SELECT
                publication_sequence,
                channel,
                snapshot_sha256,
                NULL,
                published_at
            FROM regime_snapshot_publications_v2
            ORDER BY publication_sequence
            """
        )
        connection.execute("DROP TABLE regime_snapshot_publications_v2")

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        connection = self._connection
        if connection.in_transaction:
            yield
            return
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

    @staticmethod
    def _fetch_receipt_payload(
        attempt_id: str,
        response,
        payload_sha256: str,
    ) -> dict[str, Any]:
        metadata = response.to_metadata_dict()
        if not isinstance(metadata, dict):
            raise TypeError(
                "RawProviderResponse.to_metadata_dict() must return a dictionary"
            )
        return {
            "schema_version": "regime_fetch_receipt.v1",
            "attempt_id": attempt_id,
            "payload_sha256": payload_sha256,
            "byte_length": len(response.body),
            "response": metadata,
        }

    def capture_response(self, attempt_id: str, response) -> str:
        """Durably capture exact parser-input bytes and their fetch envelope."""

        from live_trading.regime_provider_evidence import RawProviderResponse

        if not isinstance(response, RawProviderResponse):
            raise TypeError("response must be a RawProviderResponse")
        payload_bytes = bytes(response.body)
        payload_sha256 = _content_sha256(payload_bytes)
        if response.body_sha256 != payload_sha256:
            raise RegimeEvidenceIntegrityError(
                "RawProviderResponse body checksum does not match its exact bytes"
            )
        receipt_payload = self._fetch_receipt_payload(
            attempt_id,
            response,
            payload_sha256,
        )
        receipt_json = _canonical_json(receipt_payload)
        fetch_sha256 = _domain_sha256(_FETCH_HASH_DOMAIN, receipt_json)
        stored_at = _utc_iso(datetime.now(timezone.utc), "stored_at")
        requested_sessions_json = _canonical_json(
            [session.isoformat() for session in response.requested_sessions]
        )
        request_parameters_json = _canonical_json(
            [list(item) for item in response.request_parameters]
        )
        headers_json = _canonical_json(
            [list(item) for item in response.headers]
        )
        response_metadata_json = _canonical_json(
            response.to_metadata_dict()
        )

        with self._immediate_transaction():
            attempt = self._attempt_row(attempt_id)
            if attempt["status"] != "in_progress":
                raise RegimeEvidenceStoreError(
                    f"Attempt {attempt_id} is already {attempt['status']}"
                )
            if response.provider != attempt["provider"]:
                raise ValueError(
                    "Raw response provider does not match the source attempt"
                )
            schedule = regime_market_schedule(
                attempt["requested_start"],
                attempt["requested_end"],
            )
            expected_sessions = {
                session.date()
                for session in schedule.index
            }
            response_sessions = set(response.requested_sessions)
            if not response_sessions or not response_sessions <= expected_sessions:
                raise ValueError(
                    "Raw response requested sessions must be a non-empty "
                    "subset of the source attempt"
                )
            attempted_at = pd.Timestamp(attempt["attempted_at"])
            if pd.Timestamp(response.request_started_at) < attempted_at:
                raise ValueError(
                    "Raw response request_started_at cannot precede attempted_at"
                )

            existing_blob = self._connection.execute(
                """
                SELECT byte_length, payload_bytes
                FROM raw_payload_blobs
                WHERE payload_sha256 = ?
                """,
                (payload_sha256,),
            ).fetchone()
            if existing_blob is None:
                self._connection.execute(
                    """
                    INSERT INTO raw_payload_blobs (
                        payload_sha256,
                        byte_length,
                        payload_bytes,
                        stored_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        payload_sha256,
                        len(payload_bytes),
                        sqlite3.Binary(payload_bytes),
                        stored_at,
                    ),
                )
            elif (
                int(existing_blob["byte_length"]) != len(payload_bytes)
                or bytes(existing_blob["payload_bytes"]) != payload_bytes
                or _content_sha256(bytes(existing_blob["payload_bytes"]))
                != payload_sha256
            ):
                raise RegimeEvidenceIntegrityError(
                    "Stored raw payload does not match its content address"
                )

            existing_fetch = self._connection.execute(
                """
                SELECT attempt_id, payload_sha256, receipt_json
                FROM fetch_receipts
                WHERE fetch_sha256 = ?
                """,
                (fetch_sha256,),
            ).fetchone()
            if existing_fetch is None:
                self._connection.execute(
                    """
                    INSERT INTO fetch_receipts (
                        fetch_sha256,
                        attempt_id,
                        payload_sha256,
                        provider,
                        endpoint,
                        requested_sessions_json,
                        request_parameters_json,
                        request_started_at,
                        response_completed_at,
                        status_code,
                        headers_json,
                        media_type,
                        content_encoding,
                        response_metadata_json,
                        receipt_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        fetch_sha256,
                        attempt_id,
                        payload_sha256,
                        response.provider,
                        response.endpoint,
                        requested_sessions_json,
                        request_parameters_json,
                        _utc_iso(
                            response.request_started_at,
                            "request_started_at",
                        ),
                        _utc_iso(
                            response.completed_at,
                            "response.completed_at",
                        ),
                        int(response.status_code),
                        headers_json,
                        response.media_type,
                        response.content_encoding,
                        response_metadata_json,
                        receipt_json,
                    ),
                )
            elif (
                existing_fetch["attempt_id"] != attempt_id
                or existing_fetch["payload_sha256"] != payload_sha256
                or existing_fetch["receipt_json"] != receipt_json
            ):
                raise RegimeEvidenceIntegrityError(
                    "Stored fetch receipt does not match its content address"
                )
        return fetch_sha256

    def _load_captured_response(
        self,
        fetch_sha256: str,
        expected_attempt_id: str | None = None,
    ):
        from live_trading.regime_provider_evidence import RawProviderResponse

        fetch_digest = _validate_sha256(fetch_sha256, "fetch_sha256")
        row = self._connection.execute(
            """
            SELECT
                fetch.*,
                blob.byte_length,
                blob.payload_bytes
            FROM fetch_receipts AS fetch
            JOIN raw_payload_blobs AS blob
              ON blob.payload_sha256 = fetch.payload_sha256
            WHERE fetch.fetch_sha256 = ?
            """,
            (fetch_digest,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown fetch receipt: {fetch_digest}")
        if (
            expected_attempt_id is not None
            and row["attempt_id"] != expected_attempt_id
        ):
            raise ValueError(
                "Fetch receipt does not belong to the source attempt"
            )

        payload_bytes = bytes(row["payload_bytes"])
        if (
            len(payload_bytes) != int(row["byte_length"])
            or _content_sha256(payload_bytes) != row["payload_sha256"]
        ):
            raise RegimeEvidenceIntegrityError(
                "Raw payload BLOB failed length or SHA-256 verification"
            )
        try:
            requested_sessions = tuple(
                date.fromisoformat(value)
                for value in json.loads(row["requested_sessions_json"])
            )
            request_parameters = tuple(
                tuple(value)
                for value in json.loads(row["request_parameters_json"])
            )
            headers = tuple(
                tuple(value)
                for value in json.loads(row["headers_json"])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RegimeEvidenceIntegrityError(
                "Fetch receipt contains invalid canonical metadata"
            ) from exc
        response = RawProviderResponse(
            provider=row["provider"],
            endpoint=row["endpoint"],
            requested_sessions=requested_sessions,
            request_parameters=request_parameters,
            request_started_at=row["request_started_at"],
            completed_at=row["response_completed_at"],
            status_code=int(row["status_code"]),
            headers=headers,
            body=payload_bytes,
        )
        metadata_json = _canonical_json(response.to_metadata_dict())
        if metadata_json != row["response_metadata_json"]:
            raise RegimeEvidenceIntegrityError(
                "Fetch response metadata is not canonical or was modified"
            )
        receipt_payload = self._fetch_receipt_payload(
            row["attempt_id"],
            response,
            row["payload_sha256"],
        )
        receipt_json = _canonical_json(receipt_payload)
        if (
            receipt_json != row["receipt_json"]
            or _domain_sha256(_FETCH_HASH_DOMAIN, receipt_json)
            != row["fetch_sha256"]
        ):
            raise RegimeEvidenceIntegrityError(
                "Fetch receipt failed canonical checksum verification"
            )
        return response, row

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
        if (
            _SENSITIVE_DIAGNOSTIC.search(error_code_text)
            or _SENSITIVE_DIAGNOSTIC.search(error_message_text)
        ):
            raise ValueError(
                "failure diagnostics cannot contain credential-bearing text"
            )

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
            self._store_serialized_observations(attempt_id, serialized)
            self._mark_attempt_success(
                attempt,
                completed_text,
                len(serialized),
            )

    def _store_serialized_observations(
        self,
        attempt_id: str,
        serialized: list[dict[str, Any]],
    ) -> None:
        for item in serialized:
            existing = self._connection.execute(
                """
                SELECT attempt_id, observation_json
                FROM market_observations
                WHERE observation_sha256 = ?
                """,
                (item["observation_sha256"],),
            ).fetchone()
            if existing is not None:
                if existing["observation_json"] != item["observation_json"]:
                    raise RegimeEvidenceIntegrityError(
                        "Observation content does not match its SHA-256"
                    )
                if existing["attempt_id"] != attempt_id:
                    raise RegimeEvidenceIntegrityError(
                        "Observation content address is already linked to "
                        "a different source attempt"
                    )
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

    def _mark_attempt_success(
        self,
        attempt: sqlite3.Row,
        completed_at: str,
        observation_count: int,
    ) -> None:
        attempt_id = attempt["attempt_id"]
        self._connection.execute(
            """
            UPDATE source_attempts
            SET completed_at = ?, status = 'success',
                observation_count = ?, error_code = NULL,
                error_message = NULL
            WHERE attempt_id = ?
            """,
            (completed_at, observation_count, attempt_id),
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
                completed_at,
                attempt_id,
                completed_at,
                completed_at,
                attempt["provider"],
                attempt["instrument"],
            ),
        )

    @staticmethod
    def _source_locator_payload(locator) -> dict[str, str]:
        if hasattr(locator, "to_dict"):
            payload = locator.to_dict()
            if isinstance(payload, dict):
                return payload
        return {
            "kind": _required_text(locator.kind, "source_locator.kind"),
            "value": _required_text(locator.value, "source_locator.value"),
        }

    def _prepare_parse_job(
        self,
        attempt_id: str,
        completed_at: str,
        job,
    ) -> dict[str, Any]:
        from live_trading.regime_provider_evidence import (
            ProviderParseJob,
            get_registered_provider_parser,
        )

        if not isinstance(job, ProviderParseJob):
            raise TypeError("parse_jobs must contain ProviderParseJob values")
        registered = get_registered_provider_parser(
            job.parser.parser_id,
            job.parser.parser_version,
        )
        if job.parser is not registered:
            raise ValueError(
                "ProviderParseJob parser must be the registered parser instance"
            )
        response, fetch_row = self._load_captured_response(
            job.fetch_sha256,
            expected_attempt_id=attempt_id,
        )
        if not 200 <= int(response.status_code) < 300:
            raise ValueError("Only successful provider responses can be parsed")
        if pd.Timestamp(response.completed_at) > pd.Timestamp(completed_at):
            raise ValueError(
                "Provider response completed after the source attempt"
            )
        if tuple(job.config.requested_sessions) != tuple(
            response.requested_sessions
        ):
            raise ValueError(
                "Parser configuration sessions must match the raw response"
            )

        batch = registered.parse(
            response,
            job.config,
            ingested_at=pd.Timestamp(completed_at),
        )
        expected_batch_identity = (
            registered.parser_id,
            registered.parser_version,
            registered.code_sha256,
            job.config.config_sha256,
            response.response_sha256,
        )
        actual_batch_identity = (
            batch.parser_id,
            batch.parser_version,
            batch.parser_code_sha256,
            batch.config_sha256,
            batch.response_sha256,
        )
        if actual_batch_identity != expected_batch_identity:
            raise RegimeEvidenceIntegrityError(
                "Registered parser returned a mismatched batch receipt"
            )

        config_json = job.config.to_json()
        try:
            if _canonical_json(json.loads(config_json)) != config_json:
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RegimeEvidenceIntegrityError(
                "Parser configuration is not canonical JSON"
            ) from exc
        parser_code_sha256 = _validate_sha256(
            registered.code_sha256,
            "parser.code_sha256",
        )
        parser_config_sha256 = _validate_sha256(
            job.config.config_sha256,
            "parser.config_sha256",
        )

        observations: list[MarketObservation] = []
        output_rows: list[dict[str, Any]] = []
        for ordinal, output in enumerate(batch.outputs):
            observation = output.observation
            serialized = self._serialize_observation(observation)
            if serialized["raw_payload_sha256"] != fetch_row["payload_sha256"]:
                raise RegimeEvidenceIntegrityError(
                    "Parser observation does not reference the exact raw BLOB"
                )
            locator_payload = self._source_locator_payload(
                output.source_locator
            )
            observations.append(observation)
            output_rows.append(
                {
                    "output_ordinal": ordinal,
                    "observation_sha256": serialized["observation_sha256"],
                    "source_locator": locator_payload,
                }
            )
        if not observations:
            raise ValueError("Registered parser returned no observations")
        parsed_sessions = tuple(
            observation.session
            for observation in observations
        )
        if (
            len(set(parsed_sessions)) != len(parsed_sessions)
            or set(parsed_sessions) != set(job.config.requested_sessions)
        ):
            raise RegimeEvidenceIntegrityError(
                "Parser outputs must exactly cover their configured sessions"
            )

        output_manifest = {
            "schema_version": "regime_parser_output_manifest.v1",
            "outputs": output_rows,
        }
        output_manifest_json = _canonical_json(output_manifest)
        output_manifest_sha256 = _domain_sha256(
            _MANIFEST_HASH_DOMAIN,
            output_manifest_json,
        )
        receipt_payload = {
            "schema_version": "regime_parser_receipt.v1",
            "fetch_sha256": fetch_row["fetch_sha256"],
            "response_sha256": response.response_sha256,
            "parser": {
                "id": registered.parser_id,
                "version": registered.parser_version,
                "code_sha256": parser_code_sha256,
                "config_sha256": parser_config_sha256,
            },
            "output_manifest_sha256": output_manifest_sha256,
        }
        receipt_json = _canonical_json(receipt_payload)
        receipt_sha256 = _domain_sha256(
            _PARSER_HASH_DOMAIN,
            receipt_json,
        )
        return {
            "receipt_sha256": receipt_sha256,
            "fetch_sha256": fetch_row["fetch_sha256"],
            "parser_id": registered.parser_id,
            "parser_version": registered.parser_version,
            "parser_code_sha256": parser_code_sha256,
            "parser_config_sha256": parser_config_sha256,
            "parser_config_json": config_json,
            "output_manifest_sha256": output_manifest_sha256,
            "output_manifest_json": output_manifest_json,
            "receipt_json": receipt_json,
            "first_parsed_at": completed_at,
            "observations": tuple(observations),
            "output_rows": tuple(output_rows),
        }

    def _store_parse_receipt(self, prepared: dict[str, Any]) -> None:
        existing = self._connection.execute(
            """
            SELECT *
            FROM parser_receipts
            WHERE
                fetch_sha256 = ?
                AND parser_id = ?
                AND parser_version = ?
                AND parser_code_sha256 = ?
                AND parser_config_sha256 = ?
            """,
            (
                prepared["fetch_sha256"],
                prepared["parser_id"],
                prepared["parser_version"],
                prepared["parser_code_sha256"],
                prepared["parser_config_sha256"],
            ),
        ).fetchone()
        if existing is not None:
            if (
                existing["receipt_sha256"] != prepared["receipt_sha256"]
                or existing["receipt_json"] != prepared["receipt_json"]
                or existing["output_manifest_json"]
                != prepared["output_manifest_json"]
            ):
                raise RegimeEvidenceIntegrityError(
                    "Parser produced conflicting output for the same "
                    "raw response, code, and configuration"
                )
            return
        self._connection.execute(
            """
            INSERT INTO parser_receipts (
                receipt_sha256,
                fetch_sha256,
                parser_id,
                parser_version,
                parser_code_sha256,
                parser_config_sha256,
                parser_config_json,
                output_manifest_sha256,
                output_manifest_json,
                receipt_json,
                first_parsed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prepared["receipt_sha256"],
                prepared["fetch_sha256"],
                prepared["parser_id"],
                prepared["parser_version"],
                prepared["parser_code_sha256"],
                prepared["parser_config_sha256"],
                prepared["parser_config_json"],
                prepared["output_manifest_sha256"],
                prepared["output_manifest_json"],
                prepared["receipt_json"],
                prepared["first_parsed_at"],
            ),
        )

    def _store_parse_outputs(self, prepared: dict[str, Any]) -> None:
        expected_rows = [
            (
                row["output_ordinal"],
                row["observation_sha256"],
                _canonical_json(row["source_locator"]),
            )
            for row in prepared["output_rows"]
        ]
        fetch_attempt = self._connection.execute(
            """
            SELECT attempt_id
            FROM fetch_receipts
            WHERE fetch_sha256 = ?
            """,
            (prepared["fetch_sha256"],),
        ).fetchone()
        if fetch_attempt is None:
            raise RegimeEvidenceIntegrityError(
                "Parser receipt references a missing fetch receipt"
            )
        for _, observation_sha256, _ in expected_rows:
            observation_attempt = self._connection.execute(
                """
                SELECT attempt_id
                FROM market_observations
                WHERE observation_sha256 = ?
                """,
                (observation_sha256,),
            ).fetchone()
            if (
                observation_attempt is None
                or observation_attempt["attempt_id"]
                != fetch_attempt["attempt_id"]
            ):
                raise RegimeEvidenceIntegrityError(
                    "Parser output observation and fetch receipt must belong "
                    "to the same source attempt"
                )
        existing_rows = self._connection.execute(
            """
            SELECT
                output_ordinal,
                observation_sha256,
                source_locator_json
            FROM parser_receipt_outputs
            WHERE receipt_sha256 = ?
            ORDER BY output_ordinal
            """,
            (prepared["receipt_sha256"],),
        ).fetchall()
        if existing_rows:
            actual_rows = [
                (
                    row["output_ordinal"],
                    row["observation_sha256"],
                    row["source_locator_json"],
                )
                for row in existing_rows
            ]
            if actual_rows != expected_rows:
                raise RegimeEvidenceIntegrityError(
                    "Parser receipt output links do not match its manifest"
                )
            return
        for ordinal, observation_sha256, locator_json in expected_rows:
            self._connection.execute(
                """
                INSERT INTO parser_receipt_outputs (
                    receipt_sha256,
                    output_ordinal,
                    observation_sha256,
                    source_locator_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    prepared["receipt_sha256"],
                    ordinal,
                    observation_sha256,
                    locator_json,
                ),
            )

    def record_success_from_captures(
        self,
        attempt_id: str,
        completed_at,
        parse_jobs,
    ) -> tuple[MarketObservation, ...]:
        """Parse retained bytes and atomically commit verified observations."""

        completed_text = _utc_iso(completed_at, "completed_at")
        try:
            jobs = tuple(parse_jobs)
        except TypeError as exc:
            raise TypeError("parse_jobs must be an iterable") from exc
        if not jobs:
            raise ValueError("parse_jobs cannot be empty")

        attempt = self._attempt_row(attempt_id)
        if attempt["status"] != "in_progress":
            raise RegimeEvidenceStoreError(
                f"Attempt {attempt_id} is already {attempt['status']}"
            )
        self._validate_completion_time(attempt, completed_text)
        prepared_jobs = [
            self._prepare_parse_job(attempt_id, completed_text, job)
            for job in jobs
        ]
        observations = tuple(
            observation
            for prepared in prepared_jobs
            for observation in prepared["observations"]
        )
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
            for prepared in prepared_jobs:
                response, row = self._load_captured_response(
                    prepared["fetch_sha256"],
                    expected_attempt_id=attempt_id,
                )
                if response.body_sha256 != row["payload_sha256"]:
                    raise RegimeEvidenceIntegrityError(
                        "Raw BLOB changed after parsing"
                    )
                self._store_parse_receipt(prepared)
            self._store_serialized_observations(attempt_id, serialized)
            for prepared in prepared_jobs:
                self._store_parse_outputs(prepared)
            self._mark_attempt_success(
                attempt,
                completed_text,
                len(serialized),
            )
        return observations

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

    def assemble_verified_snapshot(
        self,
        start,
        end,
        *,
        as_of,
    ) -> RegimeMarketDataSnapshot:
        """Assemble a causal rolling snapshot from retained verified revisions.

        For each session and required source, the latest revision whose
        ingestion time is no later than ``as_of`` is selected. Legacy
        observations without a same-attempt parser/fetch/raw-byte lineage are
        ineligible. The complete snapshot is reparsed under one write lock
        before it is returned.
        """

        start_text = _date_iso(start, "start")
        end_text = _date_iso(end, "end")
        if end_text < start_text:
            raise ValueError("end cannot be before start")
        as_of_text = _utc_iso(as_of, "as_of")
        as_of_timestamp = pd.Timestamp(as_of_text)
        schedule = regime_market_schedule(start_text, end_text)
        if schedule.empty:
            raise ValueError("requested snapshot range has no NYSE sessions")

        observations: list[MarketObservation] = []
        with self._immediate_transaction():
            for provider, instrument in _VERIFIED_SNAPSHOT_SOURCES:
                rows = self._connection.execute(
                    """
                    WITH eligible AS (
                        SELECT
                            observation.observation_sequence,
                            observation.observation_sha256,
                            observation.observation_json,
                            observation.session_date,
                            observation.revision,
                            ROW_NUMBER() OVER (
                                PARTITION BY observation.session_date
                                ORDER BY
                                    observation.ingested_at DESC,
                                    observation.revision DESC,
                                    observation.observation_sequence DESC
                            ) AS causal_rank
                        FROM market_observations AS observation
                        JOIN source_attempts AS attempt
                          ON attempt.attempt_id = observation.attempt_id
                        WHERE observation.provider = ?
                          AND observation.instrument = ?
                          AND observation.session_date BETWEEN ? AND ?
                          AND observation.ingested_at <= ?
                          AND attempt.status = 'success'
                          AND EXISTS (
                              SELECT 1
                              FROM parser_receipt_outputs AS output
                              JOIN parser_receipts AS receipt
                                ON receipt.receipt_sha256 =
                                   output.receipt_sha256
                              JOIN fetch_receipts AS fetch
                                ON fetch.fetch_sha256 =
                                   receipt.fetch_sha256
                              WHERE output.observation_sha256 =
                                    observation.observation_sha256
                                AND fetch.attempt_id =
                                    observation.attempt_id
                          )
                    )
                    SELECT
                        observation_sha256,
                        observation_json,
                        session_date,
                        revision
                    FROM eligible
                    WHERE causal_rank = 1
                    ORDER BY session_date
                    """,
                    (
                        provider,
                        instrument,
                        start_text,
                        end_text,
                        as_of_text,
                    ),
                ).fetchall()
                for row in rows:
                    try:
                        payload = json.loads(row["observation_json"])
                        observation = MarketObservation.from_dict(payload)
                    except Exception as exc:
                        raise RegimeEvidenceIntegrityError(
                            "Stored observation failed schema validation"
                        ) from exc
                    serialized = self._serialize_observation(observation)
                    if (
                        serialized["observation_sha256"]
                        != row["observation_sha256"]
                        or serialized["observation_json"]
                        != row["observation_json"]
                        or serialized["provider"] != provider
                        or serialized["instrument"] != instrument
                        or serialized["session_date"]
                        != row["session_date"]
                    ):
                        raise RegimeEvidenceIntegrityError(
                            "Stored observation does not match its indexed "
                            "identity"
                        )
                    observations.append(observation)

            expected_count = len(schedule.index) * len(
                _VERIFIED_SNAPSHOT_SOURCES
            )
            if len(observations) != expected_count:
                raise RegimeEvidenceIntegrityError(
                    "Verified stored history does not exactly cover the "
                    f"requested sessions: expected={expected_count}, "
                    f"actual={len(observations)}"
                )
            snapshot = RegimeMarketDataSnapshot(
                as_of=as_of_timestamp,
                observations=tuple(observations),
            )
            report = self._verify_snapshot_locked(snapshot)
            if not report.verified:
                raise RegimeEvidenceIntegrityError(
                    "Assembled snapshot failed evidence verification: "
                    f"{list(report.failures)}"
                )
        return snapshot

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
            self._ensure_snapshot_content(
                snapshot_sha256,
                snapshot_as_of,
                snapshot_json,
            )
            self._connection.execute(
                """
                INSERT INTO regime_snapshot_publications (
                    channel,
                    snapshot_sha256,
                    evidence_sha256,
                    published_at
                ) VALUES (?, ?, NULL, ?)
                ON CONFLICT DO NOTHING
                """,
                (channel_text, snapshot_sha256, published_at),
            )
        return snapshot_sha256

    def _ensure_snapshot_content(
        self,
        snapshot_sha256: str,
        snapshot_as_of: str,
        snapshot_json: str,
    ) -> None:
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
            return
        self._connection.execute(
            """
            INSERT INTO regime_input_snapshots (
                snapshot_sha256, snapshot_as_of, snapshot_json
            ) VALUES (?, ?, ?)
            """,
            (snapshot_sha256, snapshot_as_of, snapshot_json),
        )

    @staticmethod
    def _config_from_json(config_json: str):
        from live_trading.regime_provider_evidence import ProviderParseConfig

        try:
            payload = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise RegimeEvidenceIntegrityError(
                "Parser configuration is not valid JSON"
            ) from exc
        if _canonical_json(payload) != config_json:
            raise RegimeEvidenceIntegrityError(
                "Parser configuration is not canonical JSON"
            )
        if hasattr(ProviderParseConfig, "from_json"):
            return ProviderParseConfig.from_json(config_json)
        if hasattr(ProviderParseConfig, "from_dict"):
            return ProviderParseConfig.from_dict(payload)
        raise RegimeEvidenceIntegrityError(
            "ProviderParseConfig cannot be reconstructed for verification"
        )

    def _verify_parser_receipt(
        self,
        receipt_sha256: str,
    ) -> dict[str, Any]:
        from live_trading.regime_provider_evidence import (
            ProviderParseJob,
            get_registered_provider_parser,
        )

        receipt_digest = _validate_sha256(
            receipt_sha256,
            "receipt_sha256",
        )
        row = self._connection.execute(
            """
            SELECT
                receipt.*,
                fetch.attempt_id
            FROM parser_receipts AS receipt
            JOIN fetch_receipts AS fetch
              ON fetch.fetch_sha256 = receipt.fetch_sha256
            WHERE receipt.receipt_sha256 = ?
            """,
            (receipt_digest,),
        ).fetchone()
        if row is None:
            raise RegimeEvidenceIntegrityError(
                f"Missing parser receipt: {receipt_digest}"
            )
        config = self._config_from_json(row["parser_config_json"])
        if config.config_sha256 != row["parser_config_sha256"]:
            raise RegimeEvidenceIntegrityError(
                "Parser configuration checksum does not match its receipt"
            )
        parser = get_registered_provider_parser(
            row["parser_id"],
            row["parser_version"],
        )
        if parser.code_sha256 != row["parser_code_sha256"]:
            raise RegimeEvidenceIntegrityError(
                "Registered parser code does not match its stored receipt"
            )
        job = ProviderParseJob(
            fetch_sha256=row["fetch_sha256"],
            parser=parser,
            config=config,
        )
        prepared = self._prepare_parse_job(
            row["attempt_id"],
            row["first_parsed_at"],
            job,
        )
        comparable_fields = (
            "receipt_sha256",
            "fetch_sha256",
            "parser_id",
            "parser_version",
            "parser_code_sha256",
            "parser_config_sha256",
            "parser_config_json",
            "output_manifest_sha256",
            "output_manifest_json",
            "receipt_json",
            "first_parsed_at",
        )
        for field_name in comparable_fields:
            if prepared[field_name] != row[field_name]:
                raise RegimeEvidenceIntegrityError(
                    f"Parser receipt field was modified: {field_name}"
                )

        expected_outputs = [
            (
                item["output_ordinal"],
                item["observation_sha256"],
                _canonical_json(item["source_locator"]),
            )
            for item in prepared["output_rows"]
        ]
        stored_outputs = self._connection.execute(
            """
            SELECT
                output.output_ordinal,
                output.observation_sha256,
                output.source_locator_json,
                observation.attempt_id AS observation_attempt_id
            FROM parser_receipt_outputs AS output
            JOIN market_observations AS observation
              ON observation.observation_sha256 =
                 output.observation_sha256
            WHERE output.receipt_sha256 = ?
            ORDER BY output.output_ordinal
            """,
            (receipt_digest,),
        ).fetchall()
        if any(
            item["observation_attempt_id"] != row["attempt_id"]
            for item in stored_outputs
        ):
            raise RegimeEvidenceIntegrityError(
                "Parser output observation and fetch receipt do not share "
                "the same source attempt"
            )
        actual_outputs = [
            (
                item["output_ordinal"],
                item["observation_sha256"],
                item["source_locator_json"],
            )
            for item in stored_outputs
        ]
        if actual_outputs != expected_outputs:
            raise RegimeEvidenceIntegrityError(
                "Parser receipt output links failed deterministic reparse"
            )
        return prepared

    def verify_snapshot(
        self,
        snapshot: RegimeMarketDataSnapshot,
    ) -> SnapshotEvidenceReport:
        """Reparse every retained BLOB and derive a store-owned verdict."""

        with self._immediate_transaction():
            return self._verify_snapshot_locked(snapshot)

    def _verify_snapshot_locked(
        self,
        snapshot: RegimeMarketDataSnapshot,
    ) -> SnapshotEvidenceReport:
        if not isinstance(snapshot, RegimeMarketDataSnapshot):
            raise TypeError("snapshot must be a RegimeMarketDataSnapshot")
        snapshot_json = snapshot.to_json()
        verified_snapshot = self._verify_snapshot_json(snapshot_json)
        snapshot_sha256 = verified_snapshot.snapshot_sha256
        failures: list[str] = []
        evidence_rows: list[dict[str, Any]] = []
        verified_receipt_outputs: dict[str, frozenset[str]] = {}
        latest_parse_time: pd.Timestamp | None = None

        for observation in verified_snapshot.observations:
            for failure in observation.provenance_evidence_failures:
                failures.append(
                    f"{observation.instrument}:"
                    f"{observation.session.isoformat()}:{failure}"
                )
            serialized = self._serialize_observation(observation)
            observation_row = self._connection.execute(
                """
                SELECT observation_sha256, observation_json, revision
                FROM market_observations
                WHERE observation_sha256 = ?
                """,
                (serialized["observation_sha256"],),
            ).fetchone()
            if observation_row is None:
                failures.append(
                    "observation_not_stored:"
                    f"{observation.instrument}:"
                    f"{observation.session.isoformat()}"
                )
                continue
            if observation_row["observation_json"] != serialized["observation_json"]:
                raise RegimeEvidenceIntegrityError(
                    "Stored observation JSON does not match snapshot content"
                )
            receipt_row = self._connection.execute(
                """
                SELECT
                    output.receipt_sha256,
                    receipt.first_parsed_at,
                    receipt.fetch_sha256,
                    fetch.payload_sha256,
                    fetch.attempt_id AS fetch_attempt_id,
                    observation.attempt_id AS observation_attempt_id
                FROM parser_receipt_outputs AS output
                JOIN parser_receipts AS receipt
                  ON receipt.receipt_sha256 = output.receipt_sha256
                JOIN fetch_receipts AS fetch
                  ON fetch.fetch_sha256 = receipt.fetch_sha256
                JOIN market_observations AS observation
                  ON observation.observation_sha256 =
                     output.observation_sha256
                WHERE output.observation_sha256 = ?
                ORDER BY
                    receipt.first_parsed_at,
                    output.receipt_sha256
                LIMIT 1
                """,
                (serialized["observation_sha256"],),
            ).fetchone()
            if receipt_row is None:
                failures.append(
                    "parser_receipt_missing:"
                    f"{observation.instrument}:"
                    f"{observation.session.isoformat()}"
                )
                continue
            if (
                receipt_row["observation_attempt_id"]
                != receipt_row["fetch_attempt_id"]
            ):
                raise RegimeEvidenceIntegrityError(
                    "Selected observation and parser receipt do not share "
                    "the same source attempt"
                )
            receipt_sha256 = receipt_row["receipt_sha256"]
            output_hashes = verified_receipt_outputs.get(receipt_sha256)
            if output_hashes is None:
                prepared = self._verify_parser_receipt(receipt_sha256)
                output_hashes = frozenset(
                    item["observation_sha256"]
                    for item in prepared["output_rows"]
                )
                verified_receipt_outputs[receipt_sha256] = output_hashes
            if serialized["observation_sha256"] not in output_hashes:
                raise RegimeEvidenceIntegrityError(
                    "Reparsed receipt does not contain the selected observation"
                )
            parsed_at = pd.Timestamp(receipt_row["first_parsed_at"])
            latest_parse_time = (
                parsed_at
                if latest_parse_time is None
                else max(latest_parse_time, parsed_at)
            )
            evidence_rows.append(
                {
                    "instrument": observation.instrument,
                    "session": observation.session.isoformat(),
                    "observation_sha256": serialized["observation_sha256"],
                    "revision": int(observation_row["revision"]),
                    "receipt_sha256": receipt_row["receipt_sha256"],
                    "fetch_sha256": receipt_row["fetch_sha256"],
                    "payload_sha256": receipt_row["payload_sha256"],
                }
            )

        if failures:
            return SnapshotEvidenceReport(
                snapshot_sha256=snapshot_sha256,
                verified=False,
                evidence_sha256=None,
                verification_kind=None,
                verified_at=None,
                failures=tuple(sorted(set(failures))),
            )
        if len(evidence_rows) != len(verified_snapshot.observations):
            raise RegimeEvidenceIntegrityError(
                "Snapshot evidence cardinality is inconsistent"
            )
        verification_kind = (
            "decision_time"
            if latest_parse_time is not None
            and latest_parse_time <= verified_snapshot.as_of
            else "verified_replay"
        )
        evidence_rows.sort(
            key=lambda item: (
                item["session"],
                item["instrument"],
                item["observation_sha256"],
            )
        )
        manifest = {
            "schema_version": "regime_snapshot_evidence.v1",
            "snapshot_sha256": snapshot_sha256,
            "verification_kind": verification_kind,
            "observations": evidence_rows,
        }
        manifest_json = _canonical_json(manifest)
        evidence_sha256 = _domain_sha256(
            _EVIDENCE_HASH_DOMAIN,
            manifest_json,
        )
        verified_at = _utc_iso(
            datetime.now(timezone.utc),
            "verified_at",
        )
        snapshot_as_of = _utc_iso(
            verified_snapshot.as_of,
            "snapshot.as_of",
        )

        with self._immediate_transaction():
            self._ensure_snapshot_content(
                snapshot_sha256,
                snapshot_as_of,
                snapshot_json,
            )
            existing = self._connection.execute(
                """
                SELECT
                    snapshot_sha256,
                    verification_kind,
                    verified_at,
                    manifest_json
                FROM snapshot_evidence_manifests
                WHERE evidence_sha256 = ?
                """,
                (evidence_sha256,),
            ).fetchone()
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO snapshot_evidence_manifests (
                        evidence_sha256,
                        snapshot_sha256,
                        verification_kind,
                        verified_at,
                        manifest_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        evidence_sha256,
                        snapshot_sha256,
                        verification_kind,
                        verified_at,
                        manifest_json,
                    ),
                )
            elif (
                existing["snapshot_sha256"] != snapshot_sha256
                or existing["verification_kind"] != verification_kind
                or existing["manifest_json"] != manifest_json
            ):
                raise RegimeEvidenceIntegrityError(
                    "Existing snapshot evidence does not match its SHA-256"
                )
            else:
                verified_at = existing["verified_at"]

            stored_links = self._connection.execute(
                """
                SELECT observation_sha256, receipt_sha256
                FROM snapshot_evidence_observations
                WHERE evidence_sha256 = ?
                ORDER BY observation_sha256
                """,
                (evidence_sha256,),
            ).fetchall()
            expected_links = sorted(
                (
                    item["observation_sha256"],
                    item["receipt_sha256"],
                )
                for item in evidence_rows
            )
            if stored_links:
                actual_links = [
                    (
                        item["observation_sha256"],
                        item["receipt_sha256"],
                    )
                    for item in stored_links
                ]
                if actual_links != expected_links:
                    raise RegimeEvidenceIntegrityError(
                        "Snapshot evidence links do not match its manifest"
                    )
            else:
                for observation_sha256, receipt_sha256 in expected_links:
                    self._connection.execute(
                        """
                        INSERT INTO snapshot_evidence_observations (
                            evidence_sha256,
                            observation_sha256,
                            receipt_sha256
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            evidence_sha256,
                            observation_sha256,
                            receipt_sha256,
                        ),
                    )
        return SnapshotEvidenceReport(
            snapshot_sha256=snapshot_sha256,
            verified=True,
            evidence_sha256=evidence_sha256,
            verification_kind=verification_kind,
            verified_at=verified_at,
            failures=(),
        )

    def publish_verified_snapshot(
        self,
        snapshot: RegimeMarketDataSnapshot,
        channel: str = "research",
    ) -> str:
        """Publish only after the store rehashes and reparses every input."""

        channel_text = _snapshot_channel(channel)
        published_at = _utc_iso(
            datetime.now(timezone.utc),
            "published_at",
        )
        with self._immediate_transaction():
            report = self._verify_snapshot_locked(snapshot)
            if not report.verified or report.evidence_sha256 is None:
                raise RegimeEvidenceIntegrityError(
                    "Snapshot provenance is unverified: "
                    f"{list(report.failures)}"
                )
            evidence = self._connection.execute(
                """
                SELECT snapshot_sha256
                FROM snapshot_evidence_manifests
                WHERE evidence_sha256 = ?
                """,
                (report.evidence_sha256,),
            ).fetchone()
            if (
                evidence is None
                or evidence["snapshot_sha256"] != report.snapshot_sha256
            ):
                raise RegimeEvidenceIntegrityError(
                    "Snapshot evidence is missing or belongs to another snapshot"
                )
            self._connection.execute(
                """
                INSERT INTO regime_snapshot_publications (
                    channel,
                    snapshot_sha256,
                    evidence_sha256,
                    published_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    channel_text,
                    report.snapshot_sha256,
                    report.evidence_sha256,
                    published_at,
                ),
            )
        return report.snapshot_sha256

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
        require_verified: bool = False,
    ) -> RegimeMarketDataSnapshot | None:
        channel_text = _snapshot_channel(channel)
        if not isinstance(require_verified, bool):
            raise TypeError("require_verified must be a boolean")
        row = self._connection.execute(
            """
            SELECT
                snapshot.snapshot_sha256,
                snapshot.snapshot_as_of,
                snapshot.snapshot_json,
                publication.evidence_sha256
            FROM regime_snapshot_publications AS publication
            JOIN regime_input_snapshots AS snapshot
              ON snapshot.snapshot_sha256 = publication.snapshot_sha256
            WHERE publication.channel = ?
              AND (? = 0 OR publication.evidence_sha256 IS NOT NULL)
            ORDER BY
                snapshot.snapshot_as_of DESC,
                publication.publication_sequence DESC
            LIMIT 1
            """,
            (channel_text, int(require_verified)),
        ).fetchone()
        if row is None:
            return None
        snapshot = self._snapshot_from_row(row, "Latest snapshot")
        if require_verified:
            report = self.verify_snapshot(snapshot)
            if (
                not report.verified
                or report.evidence_sha256 != row["evidence_sha256"]
            ):
                raise RegimeEvidenceIntegrityError(
                    "Latest verified publication failed evidence validation"
                )
        return snapshot

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
