"""
Sharded SQLite cache for historical option pricing data.

Price rows live in underlying/year shards so no single file grows without
bound. Metadata tables stay in a small sidecar database.
"""

from __future__ import annotations

import hmac
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from backtesting.contract_universe import (
    CONTRACT_REFERENCE_ENVIRONMENT,
    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
    CONTRACT_REFERENCE_MAX_PAGES_PER_ROOT,
    CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE,
    CONTRACT_REFERENCE_SOURCE,
    ConfirmedContractReferenceSnapshot,
    ContractReferencePageEvidence,
    ContractUniverseError,
    OptionContractIdentity,
    contract_reference_snapshot_sha256,
)


# ``contract_reference_attempts`` is a one-transition state projection.
# Lifecycle events, pages, snapshots, snapshot rows, and heads are the
# trigger-enforced append-only/immutable evidence.
_CONTRACT_REFERENCE_TABLES = """
CREATE TABLE IF NOT EXISTS contract_reference_attempts (
    attempt_id          TEXT PRIMARY KEY,
    schema_version      INTEGER NOT NULL,
    environment         TEXT NOT NULL,
    source              TEXT NOT NULL,
    underlying          TEXT NOT NULL,
    expiration          TEXT NOT NULL,
    contract_type       TEXT NOT NULL,
    as_of_date          TEXT NOT NULL,
    expected_roots_json TEXT NOT NULL,
    status              TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    failure_code        TEXT,
    snapshot_sha256     TEXT
);
CREATE INDEX IF NOT EXISTS idx_contract_reference_attempt_lookup
    ON contract_reference_attempts(
        environment, source, schema_version, underlying, expiration,
        contract_type, as_of_date, started_at
    );
CREATE TABLE IF NOT EXISTS contract_reference_attempt_events (
    attempt_id      TEXT NOT NULL,
    event_ordinal   INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,
    failure_code    TEXT,
    snapshot_sha256 TEXT,
    PRIMARY KEY (attempt_id, event_ordinal)
);
CREATE TABLE IF NOT EXISTS contract_reference_pages (
    attempt_id          TEXT NOT NULL,
    root_ordinal        INTEGER NOT NULL,
    root_ticker         TEXT NOT NULL,
    page_ordinal        INTEGER NOT NULL,
    request_sha256      TEXT NOT NULL,
    response_sha256     TEXT NOT NULL,
    result_count        INTEGER NOT NULL,
    next_request_sha256 TEXT,
    is_terminal         INTEGER NOT NULL,
    recorded_at         TEXT NOT NULL,
    PRIMARY KEY (attempt_id, root_ordinal, page_ordinal),
    UNIQUE (attempt_id, request_sha256)
);
CREATE TABLE IF NOT EXISTS contract_reference_snapshots (
    snapshot_sha256     TEXT PRIMARY KEY,
    schema_version      INTEGER NOT NULL,
    environment         TEXT NOT NULL,
    source              TEXT NOT NULL,
    underlying          TEXT NOT NULL,
    expiration          TEXT NOT NULL,
    contract_type       TEXT NOT NULL,
    as_of_date          TEXT NOT NULL,
    attempt_id          TEXT NOT NULL UNIQUE,
    expected_roots_json TEXT NOT NULL,
    page_count          INTEGER NOT NULL,
    contract_count      INTEGER NOT NULL,
    confirmed_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contract_reference_snapshot_contracts (
    snapshot_sha256 TEXT NOT NULL,
    option_ticker   TEXT NOT NULL,
    strike_text     TEXT NOT NULL,
    contract_type   TEXT NOT NULL,
    PRIMARY KEY (snapshot_sha256, option_ticker)
);
CREATE TABLE IF NOT EXISTS contract_reference_heads (
    schema_version  INTEGER NOT NULL,
    environment     TEXT NOT NULL,
    source          TEXT NOT NULL,
    underlying      TEXT NOT NULL,
    expiration      TEXT NOT NULL,
    contract_type   TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL,
    confirmed_at    TEXT NOT NULL,
    PRIMARY KEY (
        schema_version, environment, source, underlying, expiration,
        contract_type, as_of_date
    )
);
"""

_CONTRACT_REFERENCE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempts_insert_guard
BEFORE INSERT ON contract_reference_attempts
WHEN NOT (
    NEW.status = 'STARTED'
    AND NEW.finished_at IS NULL
    AND NEW.failure_code IS NULL
    AND NEW.snapshot_sha256 IS NULL
    AND NEW.schema_version = 1
    AND NEW.environment = 'historical_research'
    AND NEW.source = 'massive_v3_reference_options_contracts'
    AND typeof(NEW.attempt_id) = 'text'
    AND length(NEW.attempt_id) = 32
    AND NEW.attempt_id NOT GLOB '*[^0-9a-f]*'
    AND typeof(NEW.underlying) = 'text'
    AND length(NEW.underlying) > 0
    AND NEW.underlying NOT GLOB '*[^A-Z0-9._-]*'
    AND NEW.contract_type IN ('put', 'call')
    AND typeof(NEW.expiration) = 'text'
    AND length(NEW.expiration) = 10
    AND typeof(NEW.as_of_date) = 'text'
    AND length(NEW.as_of_date) = 10
    AND NEW.expiration > NEW.as_of_date
    AND typeof(NEW.started_at) = 'text'
    AND length(NEW.started_at) >= 20
    AND substr(NEW.started_at, -1, 1) = 'Z'
    AND NEW.expected_roots_json = CASE
        WHEN NEW.underlying = 'SPX' THEN '["SPX","SPXW"]'
        ELSE '["' || NEW.underlying || '"]'
    END
    AND NOT EXISTS (
        SELECT 1 FROM contract_reference_attempts
        WHERE attempt_id = NEW.attempt_id
    )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'invalid contract-reference attempt insertion'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempts_transition_guard
BEFORE UPDATE ON contract_reference_attempts
WHEN NOT (
    OLD.status = 'STARTED'
    AND OLD.finished_at IS NULL
    AND OLD.failure_code IS NULL
    AND OLD.snapshot_sha256 IS NULL
    AND NEW.attempt_id IS OLD.attempt_id
    AND NEW.schema_version IS OLD.schema_version
    AND NEW.environment IS OLD.environment
    AND NEW.source IS OLD.source
    AND NEW.underlying IS OLD.underlying
    AND NEW.expiration IS OLD.expiration
    AND NEW.contract_type IS OLD.contract_type
    AND NEW.as_of_date IS OLD.as_of_date
    AND NEW.expected_roots_json IS OLD.expected_roots_json
    AND NEW.started_at IS OLD.started_at
    AND typeof(NEW.finished_at) = 'text'
    AND length(NEW.finished_at) >= 20
    AND substr(NEW.finished_at, -1, 1) = 'Z'
    AND NEW.finished_at >= OLD.started_at
    AND (
        (
            NEW.status = 'FAILED'
            AND typeof(NEW.failure_code) = 'text'
            AND length(NEW.failure_code) BETWEEN 1 AND 64
            AND NEW.failure_code NOT GLOB '*[^A-Z0-9_]*'
            AND NEW.snapshot_sha256 IS NULL
        )
        OR (
            NEW.status = 'CONFIRMED_EMPTY'
            AND NEW.failure_code IS NULL
            AND NEW.snapshot_sha256 IS NULL
        )
        OR (
            NEW.status IN ('CONFIRMED', 'DUPLICATE_CONFIRMED')
            AND NEW.failure_code IS NULL
            AND typeof(NEW.snapshot_sha256) = 'text'
            AND length(NEW.snapshot_sha256) = 64
            AND NEW.snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
        )
        OR (
            NEW.status = 'CONFLICT'
            AND NEW.failure_code = 'CONCURRENT_SNAPSHOT_CONFLICT'
            AND typeof(NEW.snapshot_sha256) = 'text'
            AND length(NEW.snapshot_sha256) = 64
            AND NEW.snapshot_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'invalid contract-reference attempt transition'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempts_no_delete
BEFORE DELETE ON contract_reference_attempts
BEGIN
    SELECT RAISE(ABORT, 'contract-reference attempts cannot be deleted');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempt_events_no_update
BEFORE UPDATE ON contract_reference_attempt_events
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference attempt events are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempt_events_no_replace
BEFORE INSERT ON contract_reference_attempt_events
WHEN EXISTS (
    SELECT 1 FROM contract_reference_attempt_events
    WHERE attempt_id = NEW.attempt_id
      AND event_ordinal = NEW.event_ordinal
)
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference attempt events are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_attempt_events_no_delete
BEFORE DELETE ON contract_reference_attempt_events
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference attempt events are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_pages_no_update
BEFORE UPDATE ON contract_reference_pages
BEGIN
    SELECT RAISE(ABORT, 'contract-reference pages are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_pages_no_replace
BEFORE INSERT ON contract_reference_pages
WHEN EXISTS (
    SELECT 1 FROM contract_reference_pages
    WHERE (
        attempt_id = NEW.attempt_id
        AND root_ordinal = NEW.root_ordinal
        AND page_ordinal = NEW.page_ordinal
    ) OR (
        attempt_id = NEW.attempt_id
        AND request_sha256 = NEW.request_sha256
    )
)
BEGIN
    SELECT RAISE(ABORT, 'contract-reference pages are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_pages_no_delete
BEFORE DELETE ON contract_reference_pages
BEGIN
    SELECT RAISE(ABORT, 'contract-reference pages are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshots_no_update
BEFORE UPDATE ON contract_reference_snapshots
BEGIN
    SELECT RAISE(ABORT, 'contract-reference snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshots_no_replace
BEFORE INSERT ON contract_reference_snapshots
WHEN EXISTS (
    SELECT 1 FROM contract_reference_snapshots
    WHERE snapshot_sha256 = NEW.snapshot_sha256
       OR attempt_id = NEW.attempt_id
)
BEGIN
    SELECT RAISE(ABORT, 'contract-reference snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshots_no_delete
BEFORE DELETE ON contract_reference_snapshots
BEGIN
    SELECT RAISE(ABORT, 'contract-reference snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshot_contracts_no_update
BEFORE UPDATE ON contract_reference_snapshot_contracts
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference snapshot contracts are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshot_contracts_no_replace
BEFORE INSERT ON contract_reference_snapshot_contracts
WHEN EXISTS (
    SELECT 1 FROM contract_reference_snapshot_contracts
    WHERE snapshot_sha256 = NEW.snapshot_sha256
      AND option_ticker = NEW.option_ticker
)
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference snapshot contracts are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_snapshot_contracts_no_delete
BEFORE DELETE ON contract_reference_snapshot_contracts
BEGIN
    SELECT RAISE(
        ABORT,
        'contract-reference snapshot contracts are immutable'
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_heads_no_update
BEFORE UPDATE ON contract_reference_heads
BEGIN
    SELECT RAISE(ABORT, 'contract-reference heads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_heads_no_replace
BEFORE INSERT ON contract_reference_heads
WHEN EXISTS (
    SELECT 1 FROM contract_reference_heads
    WHERE schema_version = NEW.schema_version
      AND environment = NEW.environment
      AND source = NEW.source
      AND underlying = NEW.underlying
      AND expiration = NEW.expiration
      AND contract_type = NEW.contract_type
      AND as_of_date = NEW.as_of_date
)
BEGIN
    SELECT RAISE(ABORT, 'contract-reference heads are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_contract_reference_heads_no_delete
BEFORE DELETE ON contract_reference_heads
BEGIN
    SELECT RAISE(ABORT, 'contract-reference heads are immutable');
END;
"""

MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT = (
    CONTRACT_REFERENCE_MAX_PAGES_PER_ROOT
)
_CONTRACT_REFERENCE_TABLE_NAMES = (
    "contract_reference_attempts",
    "contract_reference_attempt_events",
    "contract_reference_pages",
    "contract_reference_snapshots",
    "contract_reference_snapshot_contracts",
    "contract_reference_heads",
)


class ContractReferenceCacheError(RuntimeError):
    """Raised when durable reference evidence violates its state machine."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: object) -> Optional[datetime]:
    if (
        type(value) is not str
        or len(value) < 20
        or not value.endswith("Z")
    ):
        return None
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00"
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _normalize_schema_sql(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(value.strip().rstrip(";").split()).casefold()


def _table_info(
    conn: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple, ...]:
    return tuple(
        tuple(row)
        for row in conn.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    )


@lru_cache(maxsize=1)
def _contract_reference_schema_blueprint() -> dict:
    """Build the canonical SQLite object definitions from the owned DDL."""

    reference = sqlite3.connect(":memory:")
    try:
        reference.executescript(_CONTRACT_REFERENCE_TABLES)
        reference.executescript(_CONTRACT_REFERENCE_TRIGGERS)
        table_sql = {}
        columns = {}
        for table_name in _CONTRACT_REFERENCE_TABLE_NAMES:
            row = reference.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table_name,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "canonical contract-reference DDL lacks a table"
                )
            table_sql[table_name] = _normalize_schema_sql(row[0])
            columns[table_name] = _table_info(reference, table_name)

        explicit_indexes = {
            row[0]: (
                row[1],
                _normalize_schema_sql(row[2]),
            )
            for row in reference.execute(
                """
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='index' AND sql IS NOT NULL
                  AND tbl_name IN (
                    'contract_reference_attempts',
                    'contract_reference_attempt_events',
                    'contract_reference_pages',
                    'contract_reference_snapshots',
                    'contract_reference_snapshot_contracts',
                    'contract_reference_heads'
                  )
                """
            ).fetchall()
        }
        triggers = {
            row[0]: (
                row[1],
                _normalize_schema_sql(row[2]),
            )
            for row in reference.execute(
                """
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='trigger'
                  AND tbl_name IN (
                    'contract_reference_attempts',
                    'contract_reference_attempt_events',
                    'contract_reference_pages',
                    'contract_reference_snapshots',
                    'contract_reference_snapshot_contracts',
                    'contract_reference_heads'
                  )
                """
            ).fetchall()
        }
        return {
            "tables": table_sql,
            "columns": columns,
            "indexes": explicit_indexes,
            "triggers": triggers,
        }
    finally:
        reference.close()


def _ticker_contract_type(option_ticker: str) -> Optional[str]:
    try:
        clean = option_ticker[2:] if option_ticker.startswith("O:") else option_ticker
        for idx, char in enumerate(clean):
            if char.isdigit():
                flag = clean[idx + 6]
                if flag == "C":
                    return "call"
                if flag == "P":
                    return "put"
                return None
    except (IndexError, TypeError):
        return None
    return None


def _ticker_root(option_ticker: str) -> Optional[str]:
    try:
        clean = option_ticker[2:] if option_ticker.startswith("O:") else option_ticker
        for idx, char in enumerate(clean):
            if char.isdigit():
                return clean[:idx]
    except Exception:
        return None
    return None


def _root_aliases(root: Optional[str]) -> list[str]:
    if root == "SPX":
        return ["SPX", "SPXW"]
    if root == "SPXW":
        return ["SPXW", "SPX"]
    return [root] if root else []


def _ohlcv_quality(row: Optional[dict]) -> int:
    if not row:
        return 0
    bid = row.get("bid")
    ask = row.get("ask")
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return 3
    close = row.get("close")
    if close is not None and close > 0:
        return 2
    mid = row.get("mid")
    if mid is not None and mid > 0:
        return 1
    return 0


def _db_default_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backtest_cache",
        "option_data.db",
    )


class OptionDataCache:
    def __init__(self, db_path: str = _db_default_path()):
        self.db_path = db_path
        self.root_dir = os.path.dirname(db_path)
        self.shard_dir = os.path.join(self.root_dir, "option_data_shards")
        self.use_shards = os.path.isdir(self.shard_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self._mem_cache = {}
        self._mem_cache_date = None
        self._price_conns: Dict[Tuple[str, str], sqlite3.Connection] = {}

        if self.use_shards:
            os.makedirs(self.shard_dir, exist_ok=True)
            self.meta_db_path = db_path
            self.meta_conn = sqlite3.connect(self.meta_db_path, timeout=60)
            self.meta_conn.row_factory = sqlite3.Row
            try:
                self.meta_conn.execute("PRAGMA journal_mode=WAL")
                self.meta_conn.execute("PRAGMA synchronous=NORMAL")
                self._create_meta_tables()
            except Exception:
                self.meta_conn.close()
                raise
            self.conn = self.meta_conn
        else:
            self.conn = sqlite3.connect(db_path, timeout=60)
            self.conn.row_factory = sqlite3.Row
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self._create_legacy_tables()
            except Exception:
                self.conn.close()
                raise

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_contract_reference_schema(
        conn: sqlite3.Connection,
        *,
        allow_missing: bool,
    ) -> None:
        expected = _contract_reference_schema_blueprint()

        present_tables = {}
        for table_name, expected_sql in expected["tables"].items():
            row = conn.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='table' AND name=?
                """,
                (table_name,),
            ).fetchone()
            if row is None:
                if allow_missing:
                    continue
                raise ContractReferenceCacheError(
                    f"missing contract-reference table: {table_name}"
                )
            present_tables[table_name] = True
            if (
                _normalize_schema_sql(row["sql"]) != expected_sql
                or _table_info(conn, table_name)
                != expected["columns"][table_name]
            ):
                raise ContractReferenceCacheError(
                    f"contract-reference table schema mismatch: {table_name}"
                )
        if (
            not allow_missing
            and len(present_tables) != len(expected["tables"])
        ):
            raise ContractReferenceCacheError(
                "contract-reference table set is incomplete"
            )

        index_names = tuple(expected["indexes"])
        index_name_placeholders = ",".join("?" for _ in index_names)
        table_placeholders = ",".join(
            "?" for _ in _CONTRACT_REFERENCE_TABLE_NAMES
        )
        index_rows = conn.execute(
            f"""
            SELECT name, tbl_name, sql FROM sqlite_master
            WHERE type='index' AND sql IS NOT NULL
              AND (
                name IN ({index_name_placeholders})
                OR tbl_name IN ({table_placeholders})
              )
            """,
            index_names + _CONTRACT_REFERENCE_TABLE_NAMES,
        ).fetchall()
        actual_indexes = {
            row["name"]: (
                row["tbl_name"],
                _normalize_schema_sql(row["sql"]),
            )
            for row in index_rows
        }
        for name, definition in actual_indexes.items():
            if expected["indexes"].get(name) != definition:
                raise ContractReferenceCacheError(
                    f"unexpected contract-reference index: {name}"
                )
        if (
            not allow_missing
            and actual_indexes != expected["indexes"]
        ):
            raise ContractReferenceCacheError(
                "contract-reference index set is incomplete"
            )

        trigger_names = tuple(expected["triggers"])
        trigger_name_placeholders = ",".join(
            "?" for _ in trigger_names
        )
        trigger_rows = conn.execute(
            f"""
            SELECT name, tbl_name, sql FROM sqlite_master
            WHERE type='trigger'
              AND (
                name IN ({trigger_name_placeholders})
                OR tbl_name IN ({table_placeholders})
              )
            """,
            trigger_names + _CONTRACT_REFERENCE_TABLE_NAMES,
        ).fetchall()
        actual_triggers = {
            row["name"]: (
                row["tbl_name"],
                _normalize_schema_sql(row["sql"]),
            )
            for row in trigger_rows
        }
        for name, definition in actual_triggers.items():
            if expected["triggers"].get(name) != definition:
                raise ContractReferenceCacheError(
                    f"unexpected contract-reference trigger: {name}"
                )
        if (
            not allow_missing
            and actual_triggers != expected["triggers"]
        ):
            raise ContractReferenceCacheError(
                "contract-reference trigger set is incomplete"
            )

    @classmethod
    def _initialize_contract_reference_schema(
        cls,
        conn: sqlite3.Connection,
    ) -> None:
        """Create only absent owned objects, then attest the exact schema."""

        cls._verify_contract_reference_schema(
            conn,
            allow_missing=True,
        )
        try:
            conn.executescript(_CONTRACT_REFERENCE_TABLES)
            conn.executescript(_CONTRACT_REFERENCE_TRIGGERS)
        except sqlite3.Error as exc:
            conn.rollback()
            raise ContractReferenceCacheError(
                "contract-reference schema initialization failed"
            ) from exc
        cls._verify_contract_reference_schema(
            conn,
            allow_missing=False,
        )
        conn.commit()

    def _create_legacy_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS option_prices (
                underlying    TEXT NOT NULL,
                option_ticker TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                strike        REAL NOT NULL,
                expiration    TEXT NOT NULL,
                pricing_date  TEXT NOT NULL,
                open          REAL,
                high          REAL,
                low           REAL,
                close         REAL,
                volume        INTEGER,
                vwap          REAL,
                num_trades    INTEGER,
                bid           REAL,
                ask           REAL,
                mid           REAL,
                bid_size      INTEGER,
                ask_size      INTEGER,
                implied_vol   REAL,
                delta         REAL,
                gamma         REAL,
                theta         REAL,
                vega          REAL,
                open_interest INTEGER,
                is_synchronized INTEGER DEFAULT 0,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, pricing_date)
            );
            CREATE INDEX IF NOT EXISTS idx_underlying_date
                ON option_prices(underlying, pricing_date);
            CREATE INDEX IF NOT EXISTS idx_underlying_exp_date
                ON option_prices(underlying, expiration, pricing_date);
            CREATE INDEX IF NOT EXISTS idx_underlying_type_exp
                ON option_prices(underlying, contract_type, expiration, pricing_date);
            CREATE TABLE IF NOT EXISTS contracts_cache (
                underlying    TEXT NOT NULL,
                expiration    TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                as_of_date    TEXT NOT NULL,
                option_ticker TEXT NOT NULL,
                strike        REAL NOT NULL,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, as_of_date)
            );
            CREATE INDEX IF NOT EXISTS idx_contracts_lookup
                ON contracts_cache(underlying, expiration, contract_type, as_of_date);
            CREATE TABLE IF NOT EXISTS fetch_log (
                underlying    TEXT NOT NULL,
                pricing_date  TEXT NOT NULL,
                expiration    TEXT,
                contract_type TEXT,
                data_type     TEXT NOT NULL DEFAULT 'ohlcv',
                status        TEXT NOT NULL,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (underlying, pricing_date, expiration, contract_type, data_type)
            );
            CREATE TABLE IF NOT EXISTS ohlcv_fetch_log (
                option_ticker TEXT NOT NULL,
                from_date     TEXT NOT NULL,
                to_date       TEXT NOT NULL,
                bar_count     INTEGER NOT NULL DEFAULT 0,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, from_date, to_date)
            );
            """
        )
        self._initialize_contract_reference_schema(self.conn)
        try:
            self.conn.execute(
                "ALTER TABLE option_prices ADD COLUMN is_synchronized INTEGER DEFAULT 0"
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def _create_meta_tables(self):
        self.meta_conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS contracts_cache (
                underlying    TEXT NOT NULL,
                expiration    TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                as_of_date    TEXT NOT NULL,
                option_ticker TEXT NOT NULL,
                strike        REAL NOT NULL,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, as_of_date)
            );
            CREATE INDEX IF NOT EXISTS idx_contracts_lookup
                ON contracts_cache(underlying, expiration, contract_type, as_of_date);
            CREATE TABLE IF NOT EXISTS fetch_log (
                underlying    TEXT NOT NULL,
                pricing_date  TEXT NOT NULL,
                expiration    TEXT,
                contract_type TEXT,
                data_type     TEXT NOT NULL DEFAULT 'ohlcv',
                status        TEXT NOT NULL,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (underlying, pricing_date, expiration, contract_type, data_type)
            );
            CREATE TABLE IF NOT EXISTS ohlcv_fetch_log (
                option_ticker TEXT NOT NULL,
                from_date     TEXT NOT NULL,
                to_date       TEXT NOT NULL,
                bar_count     INTEGER NOT NULL DEFAULT 0,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, from_date, to_date)
            );
            """
        )
        self._initialize_contract_reference_schema(self.meta_conn)
        self.meta_conn.commit()

    def _ensure_price_conn(self, underlying: str, year: str) -> sqlite3.Connection:
        key = (underlying, year)
        conn = self._price_conns.get(key)
        if conn:
            return conn
        shard_root = os.path.join(self.shard_dir, underlying)
        os.makedirs(shard_root, exist_ok=True)
        shard_path = os.path.join(shard_root, f"{year}.db")
        conn = sqlite3.connect(shard_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS option_prices (
                underlying    TEXT NOT NULL,
                option_ticker TEXT NOT NULL,
                contract_type TEXT NOT NULL,
                strike        REAL NOT NULL,
                expiration    TEXT NOT NULL,
                pricing_date  TEXT NOT NULL,
                open          REAL,
                high          REAL,
                low           REAL,
                close         REAL,
                volume        INTEGER,
                vwap          REAL,
                num_trades    INTEGER,
                bid           REAL,
                ask           REAL,
                mid           REAL,
                bid_size      INTEGER,
                ask_size      INTEGER,
                implied_vol   REAL,
                delta         REAL,
                gamma         REAL,
                theta         REAL,
                vega          REAL,
                open_interest INTEGER,
                is_synchronized INTEGER DEFAULT 0,
                fetched_at    TEXT NOT NULL,
                PRIMARY KEY (option_ticker, pricing_date)
            );
            CREATE INDEX IF NOT EXISTS idx_underlying_date
                ON option_prices(underlying, pricing_date);
            CREATE INDEX IF NOT EXISTS idx_underlying_exp_date
                ON option_prices(underlying, expiration, pricing_date);
            CREATE INDEX IF NOT EXISTS idx_underlying_type_exp
                ON option_prices(underlying, contract_type, expiration, pricing_date);
            """
        )
        conn.commit()
        self._price_conns[key] = conn
        return conn

    def _price_candidates(self, option_ticker: str, pricing_date: str) -> list[Tuple[str, str]]:
        root = _ticker_root(option_ticker)
        year = pricing_date[:4]
        roots = _root_aliases(root)
        return [(r, year) for r in roots if r]

    def _meta(self) -> sqlite3.Connection:
        return self.meta_conn if self.use_shards else self.conn

    # ------------------------------------------------------------------
    # Contract list cache
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_contract_reference_key(
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of_date: str,
    ) -> None:
        if type(underlying) is not str or not underlying:
            raise ContractReferenceCacheError("underlying must be non-empty")
        if any(
            char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in underlying
        ):
            raise ContractReferenceCacheError(
                "underlying must be an uppercase provider ticker root"
            )
        if type(expiration) is not str or not expiration:
            raise ContractReferenceCacheError("expiration must be non-empty")
        if type(as_of_date) is not str or not as_of_date:
            raise ContractReferenceCacheError("as_of_date must be non-empty")
        try:
            expiration_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            decision_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ContractReferenceCacheError(
                "reference dates must be exact ISO dates"
            ) from exc
        if expiration_date <= decision_date:
            raise ContractReferenceCacheError(
                "reference expiration must be after as_of_date"
            )
        if contract_type not in {"put", "call"}:
            raise ContractReferenceCacheError(
                "contract_type must be put or call"
            )

    @staticmethod
    def _expected_roots_json(expected_roots: tuple[str, ...]) -> str:
        if (
            type(expected_roots) is not tuple
            or not expected_roots
            or any(type(root) is not str or not root for root in expected_roots)
            or len(set(expected_roots)) != len(expected_roots)
        ):
            raise ContractReferenceCacheError(
                "expected_roots must be a unique non-empty tuple"
            )
        return json.dumps(
            list(expected_roots),
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @staticmethod
    def _canonical_expected_roots(underlying: str) -> tuple[str, ...]:
        return ("SPX", "SPXW") if underlying == "SPX" else (underlying,)

    @staticmethod
    def _append_attempt_event(
        conn: sqlite3.Connection,
        *,
        attempt_id: str,
        event_type: str,
        recorded_at: str,
        failure_code: Optional[str] = None,
        snapshot_sha256: Optional[str] = None,
    ) -> None:
        prior = conn.execute(
            """
            SELECT event_ordinal FROM contract_reference_attempt_events
            WHERE attempt_id=?
            ORDER BY event_ordinal DESC
            LIMIT 1
            """,
            (attempt_id,),
        ).fetchone()
        event_ordinal = 1 if prior is None else prior["event_ordinal"] + 1
        conn.execute(
            """
            INSERT INTO contract_reference_attempt_events (
                attempt_id, event_ordinal, event_type, recorded_at,
                failure_code, snapshot_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                event_ordinal,
                event_type,
                recorded_at,
                failure_code,
                snapshot_sha256,
            ),
        )

    @staticmethod
    def _attempt_events_are_valid(
        conn: sqlite3.Connection,
        attempt: sqlite3.Row,
    ) -> bool:
        started_at = _utc_datetime(attempt["started_at"])
        if started_at is None:
            return False
        rows = conn.execute(
            """
            SELECT * FROM contract_reference_attempt_events
            WHERE attempt_id=?
            ORDER BY event_ordinal
            """,
            (attempt["attempt_id"],),
        ).fetchall()
        if not rows or len(rows) > 2:
            return False
        first = rows[0]
        if (
            first["event_ordinal"] != 1
            or first["event_type"] != "STARTED"
            or first["recorded_at"] != attempt["started_at"]
            or first["failure_code"] is not None
            or first["snapshot_sha256"] is not None
            or _utc_datetime(first["recorded_at"]) != started_at
        ):
            return False
        if attempt["status"] == "STARTED":
            return (
                len(rows) == 1
                and attempt["finished_at"] is None
                and attempt["failure_code"] is None
                and attempt["snapshot_sha256"] is None
            )
        if len(rows) != 2:
            return False
        terminal = rows[1]
        finished_at = _utc_datetime(attempt["finished_at"])
        if (
            finished_at is None
            or finished_at < started_at
            or _utc_datetime(terminal["recorded_at"]) != finished_at
        ):
            return False
        terminal_fields_are_valid = (
            (
                attempt["status"] == "FAILED"
                and type(attempt["failure_code"]) is str
                and 1 <= len(attempt["failure_code"]) <= 64
                and all(
                    char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                    for char in attempt["failure_code"]
                )
                and attempt["snapshot_sha256"] is None
            )
            or (
                attempt["status"] == "CONFIRMED_EMPTY"
                and attempt["failure_code"] is None
                and attempt["snapshot_sha256"] is None
            )
            or (
                attempt["status"]
                in {"CONFIRMED", "DUPLICATE_CONFIRMED"}
                and attempt["failure_code"] is None
                and _is_sha256(attempt["snapshot_sha256"])
            )
            or (
                attempt["status"] == "CONFLICT"
                and attempt["failure_code"]
                == "CONCURRENT_SNAPSHOT_CONFLICT"
                and _is_sha256(attempt["snapshot_sha256"])
            )
        )
        return (
            terminal_fields_are_valid
            and terminal["event_ordinal"] == 2
            and terminal["event_type"] == attempt["status"]
            and terminal["recorded_at"] == attempt["finished_at"]
            and terminal["failure_code"] == attempt["failure_code"]
            and terminal["snapshot_sha256"] == attempt["snapshot_sha256"]
        )

    def begin_contract_reference_attempt(
        self,
        *,
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of_date: str,
        expected_roots: tuple[str, ...],
    ) -> str:
        """Durably record an attempt before the first provider request."""

        self._validate_contract_reference_key(
            underlying,
            expiration,
            contract_type,
            as_of_date,
        )
        roots_json = self._expected_roots_json(expected_roots)
        if expected_roots != self._canonical_expected_roots(underlying):
            raise ContractReferenceCacheError(
                "expected_roots disagree with the underlying query contract"
            )
        attempt_id = uuid.uuid4().hex
        started_at = _utc_now()
        conn = self._meta()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO contract_reference_attempts (
                    attempt_id, schema_version, environment, source, underlying,
                    expiration, contract_type, as_of_date, expected_roots_json,
                    status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'STARTED', ?)
                """,
                (
                    attempt_id,
                    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                    CONTRACT_REFERENCE_ENVIRONMENT,
                    CONTRACT_REFERENCE_SOURCE,
                    underlying,
                    expiration,
                    contract_type,
                    as_of_date,
                    roots_json,
                    started_at,
                ),
            )
            self._append_attempt_event(
                conn,
                attempt_id=attempt_id,
                event_type="STARTED",
                recorded_at=started_at,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return attempt_id

    @staticmethod
    def _attempt_expected_roots(attempt: sqlite3.Row) -> tuple[str, ...]:
        try:
            decoded = json.loads(attempt["expected_roots_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractReferenceCacheError(
                "attempt expected_roots evidence is malformed"
            ) from exc
        if (
            type(decoded) is not list
            or not decoded
            or any(type(root) is not str or not root for root in decoded)
            or len(set(decoded)) != len(decoded)
        ):
            raise ContractReferenceCacheError(
                "attempt expected_roots evidence is invalid"
            )
        return tuple(decoded)

    @staticmethod
    def _load_reference_attempt(
        conn: sqlite3.Connection,
        attempt_id: str,
    ) -> sqlite3.Row:
        if type(attempt_id) is not str or not attempt_id:
            raise ContractReferenceCacheError("attempt_id must be non-empty")
        attempt = conn.execute(
            """
            SELECT * FROM contract_reference_attempts
            WHERE attempt_id=?
            """,
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise ContractReferenceCacheError("unknown contract-reference attempt")
        if (
            attempt["schema_version"]
            != CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION
            or attempt["environment"] != CONTRACT_REFERENCE_ENVIRONMENT
            or attempt["source"] != CONTRACT_REFERENCE_SOURCE
        ):
            raise ContractReferenceCacheError(
                "contract-reference attempt binding mismatch"
            )
        OptionDataCache._validate_contract_reference_key(
            attempt["underlying"],
            attempt["expiration"],
            attempt["contract_type"],
            attempt["as_of_date"],
        )
        if (
            type(attempt["attempt_id"]) is not str
            or len(attempt["attempt_id"]) != 32
            or any(
                char not in "0123456789abcdef"
                for char in attempt["attempt_id"]
            )
            or _utc_datetime(attempt["started_at"]) is None
        ):
            raise ContractReferenceCacheError(
                "contract-reference attempt identity is malformed"
            )
        if (
            OptionDataCache._attempt_expected_roots(attempt)
            != OptionDataCache._canonical_expected_roots(attempt["underlying"])
        ):
            raise ContractReferenceCacheError(
                "contract-reference root binding mismatch"
            )
        return attempt

    def record_contract_reference_page(
        self,
        *,
        attempt_id: str,
        root_ordinal: int,
        root_ticker: str,
        page_ordinal: int,
        request_sha256: str,
        response_sha256: str,
        result_count: int,
        next_request_sha256: Optional[str],
        is_terminal: bool,
    ) -> None:
        """Append one ordered, credential-free pagination evidence row."""

        if type(root_ordinal) is not int or root_ordinal < 0:
            raise ContractReferenceCacheError("root_ordinal must be non-negative")
        if (
            type(page_ordinal) is not int
            or page_ordinal < 1
            or page_ordinal > MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT
        ):
            raise ContractReferenceCacheError("page_ordinal is out of bounds")
        if type(root_ticker) is not str or not root_ticker:
            raise ContractReferenceCacheError("root_ticker must be non-empty")
        if not _is_sha256(request_sha256) or not _is_sha256(response_sha256):
            raise ContractReferenceCacheError(
                "request and response hashes must be SHA-256"
            )
        if (
            type(result_count) is not int
            or result_count < 0
            or result_count > CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE
        ):
            raise ContractReferenceCacheError(
                "result_count is outside the provider page bound"
            )
        if type(is_terminal) is not bool:
            raise ContractReferenceCacheError("is_terminal must be exact bool")
        if is_terminal:
            if next_request_sha256 is not None:
                raise ContractReferenceCacheError(
                    "terminal page cannot name a next request"
                )
        elif not _is_sha256(next_request_sha256):
            raise ContractReferenceCacheError(
                "non-terminal page requires next request SHA-256"
            )

        conn = self._meta()
        try:
            conn.execute("BEGIN IMMEDIATE")
            attempt = self._load_reference_attempt(conn, attempt_id)
            if attempt["status"] != "STARTED":
                raise ContractReferenceCacheError(
                    "pages can be appended only to STARTED attempts"
                )
            if not self._attempt_events_are_valid(conn, attempt):
                raise ContractReferenceCacheError(
                    "attempt event history is invalid"
                )
            expected_roots = self._attempt_expected_roots(attempt)
            if (
                root_ordinal >= len(expected_roots)
                or expected_roots[root_ordinal] != root_ticker
            ):
                raise ContractReferenceCacheError(
                    "page root disagrees with attempt binding"
                )
            later_root = conn.execute(
                """
                SELECT 1 FROM contract_reference_pages
                WHERE attempt_id=? AND root_ordinal>?
                LIMIT 1
                """,
                (attempt_id, root_ordinal),
            ).fetchone()
            if later_root is not None:
                raise ContractReferenceCacheError(
                    "page roots cannot be recorded out of order"
                )
            prior = conn.execute(
                """
                SELECT * FROM contract_reference_pages
                WHERE attempt_id=? AND root_ordinal=?
                ORDER BY page_ordinal DESC
                LIMIT 1
                """,
                (attempt_id, root_ordinal),
            ).fetchone()
            expected_page = 1 if prior is None else prior["page_ordinal"] + 1
            if prior is None and root_ordinal > 0:
                prior_root_terminal = conn.execute(
                    """
                    SELECT is_terminal FROM contract_reference_pages
                    WHERE attempt_id=? AND root_ordinal=?
                    ORDER BY page_ordinal DESC
                    LIMIT 1
                    """,
                    (attempt_id, root_ordinal - 1),
                ).fetchone()
                if (
                    prior_root_terminal is None
                    or prior_root_terminal["is_terminal"] != 1
                ):
                    raise ContractReferenceCacheError(
                        "prior root must terminate before the next root"
                    )
            if page_ordinal != expected_page:
                raise ContractReferenceCacheError(
                    "page ordinal is not contiguous"
                )
            if prior is not None:
                if prior["is_terminal"]:
                    raise ContractReferenceCacheError(
                        "cannot append after terminal page"
                    )
                if prior["next_request_sha256"] != request_sha256:
                    raise ContractReferenceCacheError(
                        "page request does not match prior next-page evidence"
                    )
            conn.execute(
                """
                INSERT INTO contract_reference_pages (
                    attempt_id, root_ordinal, root_ticker, page_ordinal,
                    request_sha256, response_sha256, result_count,
                    next_request_sha256, is_terminal, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    root_ordinal,
                    root_ticker,
                    page_ordinal,
                    request_sha256,
                    response_sha256,
                    result_count,
                    next_request_sha256,
                    int(is_terminal),
                    _utc_now(),
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _validated_reference_pages(
        conn: sqlite3.Connection,
        attempt: sqlite3.Row,
    ) -> tuple[ContractReferencePageEvidence, ...]:
        expected_roots = OptionDataCache._attempt_expected_roots(attempt)
        started_at = _utc_datetime(attempt["started_at"])
        finished_at = (
            _utc_datetime(attempt["finished_at"])
            if attempt["finished_at"] is not None
            else None
        )
        if (
            started_at is None
            or (
                attempt["finished_at"] is not None
                and (
                    finished_at is None
                    or finished_at < started_at
                )
            )
        ):
            raise ContractReferenceCacheError(
                "attempt timestamps are invalid"
            )
        rows = conn.execute(
            """
            SELECT * FROM contract_reference_pages
            WHERE attempt_id=?
            ORDER BY root_ordinal, page_ordinal
            """,
            (attempt["attempt_id"],),
        ).fetchall()
        pages: list[ContractReferencePageEvidence] = []
        prior_recorded_at = started_at
        for root_ordinal, root_ticker in enumerate(expected_roots):
            root_rows = [
                row for row in rows if row["root_ordinal"] == root_ordinal
            ]
            if not root_rows:
                raise ContractReferenceCacheError(
                    "expected root has no page evidence"
                )
            if len(root_rows) > MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT:
                raise ContractReferenceCacheError(
                    "reference page count exceeds bound"
                )
            for index, row in enumerate(root_rows, start=1):
                recorded_at = _utc_datetime(row["recorded_at"])
                if (
                    row["root_ticker"] != root_ticker
                    or row["page_ordinal"] != index
                    or not _is_sha256(row["request_sha256"])
                    or not _is_sha256(row["response_sha256"])
                    or type(row["result_count"]) is not int
                    or row["result_count"] < 0
                    or row["result_count"]
                    > CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE
                    or recorded_at is None
                    or recorded_at < prior_recorded_at
                    or (
                        finished_at is not None
                        and recorded_at > finished_at
                    )
                ):
                    raise ContractReferenceCacheError(
                        "page evidence is malformed"
                    )
                prior_recorded_at = recorded_at
                is_last = index == len(root_rows)
                if is_last:
                    if (
                        row["is_terminal"] != 1
                        or row["next_request_sha256"] is not None
                    ):
                        raise ContractReferenceCacheError(
                            "root page chain lacks terminal proof"
                        )
                else:
                    next_row = root_rows[index]
                    if (
                        row["is_terminal"] != 0
                        or row["next_request_sha256"]
                        != next_row["request_sha256"]
                    ):
                        raise ContractReferenceCacheError(
                            "root page chain is discontinuous"
                        )
                pages.append(
                    ContractReferencePageEvidence(
                        root_ordinal=root_ordinal,
                        root_ticker=root_ticker,
                        page_ordinal=index,
                        request_sha256=row["request_sha256"],
                        response_sha256=row["response_sha256"],
                        result_count=row["result_count"],
                        next_request_sha256=row["next_request_sha256"],
                        is_terminal=bool(row["is_terminal"]),
                    )
                )
        if len(pages) != len(rows):
            raise ContractReferenceCacheError(
                "page evidence contains an unexpected root"
            )
        return tuple(pages)

    def finish_contract_reference_failure(
        self,
        attempt_id: str,
        failure_code: str,
    ) -> None:
        """Finish a non-authorizing attempt without changing snapshot heads."""

        if (
            type(failure_code) is not str
            or not failure_code
            or len(failure_code) > 64
            or any(
                char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                for char in failure_code
            )
        ):
            raise ContractReferenceCacheError("invalid failure_code")
        conn = self._meta()
        try:
            conn.execute("BEGIN IMMEDIATE")
            attempt = self._load_reference_attempt(conn, attempt_id)
            if attempt["status"] != "STARTED":
                raise ContractReferenceCacheError(
                    "attempt is not STARTED and cannot fail"
                )
            if not self._attempt_events_are_valid(conn, attempt):
                raise ContractReferenceCacheError(
                    "attempt event history is invalid"
                )
            finished_at = _utc_now()
            conn.execute(
                """
                UPDATE contract_reference_attempts
                SET status='FAILED', finished_at=?, failure_code=?
                WHERE attempt_id=?
                """,
                (finished_at, failure_code, attempt_id),
            )
            self._append_attempt_event(
                conn,
                attempt_id=attempt_id,
                event_type="FAILED",
                recorded_at=finished_at,
                failure_code=failure_code,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def finish_contract_reference_empty(self, attempt_id: str) -> None:
        """Persist complete-empty evidence without authorizing a universe."""

        conn = self._meta()
        try:
            conn.execute("BEGIN IMMEDIATE")
            attempt = self._load_reference_attempt(conn, attempt_id)
            if attempt["status"] != "STARTED":
                raise ContractReferenceCacheError(
                    "attempt is not STARTED and cannot finish empty"
                )
            if not self._attempt_events_are_valid(conn, attempt):
                raise ContractReferenceCacheError(
                    "attempt event history is invalid"
                )
            pages = self._validated_reference_pages(conn, attempt)
            if sum(page.result_count for page in pages) != 0:
                raise ContractReferenceCacheError(
                    "CONFIRMED_EMPTY requires zero provider results"
                )
            finished_at = _utc_now()
            conn.execute(
                """
                UPDATE contract_reference_attempts
                SET status='CONFIRMED_EMPTY', finished_at=?, failure_code=NULL
                WHERE attempt_id=?
                """,
                (finished_at, attempt_id),
            )
            self._append_attempt_event(
                conn,
                attempt_id=attempt_id,
                event_type="CONFIRMED_EMPTY",
                recorded_at=finished_at,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _normalize_reference_contracts(
        contracts: Iterable[dict],
        contract_type: str,
    ) -> tuple[OptionContractIdentity, ...]:
        if isinstance(contracts, (str, bytes)):
            raise ContractReferenceCacheError("contracts must be an iterable")
        normalized: list[OptionContractIdentity] = []
        try:
            for row in contracts:
                if type(row) is not dict:
                    raise ContractReferenceCacheError(
                        "contract rows must be exact dictionaries"
                    )
                normalized.append(
                    OptionContractIdentity(
                        option_ticker=row.get("option_ticker"),
                        strike=row.get("strike"),
                        contract_type=row.get("contract_type", contract_type),
                    )
                )
        except ContractUniverseError as exc:
            raise ContractReferenceCacheError(str(exc)) from exc
        normalized.sort()
        if not normalized:
            raise ContractReferenceCacheError(
                "confirmed reference snapshot cannot be empty"
            )
        if len({item.option_ticker for item in normalized}) != len(normalized):
            raise ContractReferenceCacheError(
                "duplicate ticker in contract snapshot"
            )
        if any(item.contract_type != contract_type for item in normalized):
            raise ContractReferenceCacheError(
                "contract type disagrees with attempt"
            )
        return tuple(normalized)

    def confirm_contract_reference_attempt(
        self,
        attempt_id: str,
        contracts: Iterable[dict],
    ) -> Optional[ConfirmedContractReferenceSnapshot]:
        """Atomically publish one complete non-empty reference snapshot."""

        conn = self._meta()
        try:
            conn.execute("BEGIN IMMEDIATE")
            attempt = self._load_reference_attempt(conn, attempt_id)
            if attempt["status"] != "STARTED":
                raise ContractReferenceCacheError(
                    "attempt is not STARTED and cannot confirm"
                )
            if not self._attempt_events_are_valid(conn, attempt):
                raise ContractReferenceCacheError(
                    "attempt event history is invalid"
                )
            pages = self._validated_reference_pages(conn, attempt)
            normalized = self._normalize_reference_contracts(
                contracts,
                attempt["contract_type"],
            )
            if sum(page.result_count for page in pages) != len(normalized):
                raise ContractReferenceCacheError(
                    "parsed contract count disagrees with page evidence"
                )
            snapshot_sha256 = contract_reference_snapshot_sha256(
                schema_version=attempt["schema_version"],
                environment=attempt["environment"],
                source=attempt["source"],
                underlying=attempt["underlying"],
                expiration=attempt["expiration"],
                contract_type=attempt["contract_type"],
                as_of_date=attempt["as_of_date"],
                expected_roots=self._attempt_expected_roots(attempt),
                pages=pages,
                contracts=normalized,
            )
            finished_at = _utc_now()
            try:
                ConfirmedContractReferenceSnapshot(
                    underlying=attempt["underlying"],
                    expiration=attempt["expiration"],
                    contract_type=attempt["contract_type"],
                    as_of_date=attempt["as_of_date"],
                    attempt_id=attempt_id,
                    snapshot_sha256=snapshot_sha256,
                    confirmed_at=finished_at,
                    expected_roots=self._attempt_expected_roots(attempt),
                    pages=pages,
                    contracts=normalized,
                    environment=attempt["environment"],
                    source=attempt["source"],
                    schema_version=attempt["schema_version"],
                )
            except ContractUniverseError as exc:
                raise ContractReferenceCacheError(str(exc)) from exc
            head = conn.execute(
                """
                SELECT * FROM contract_reference_heads
                WHERE schema_version=? AND environment=? AND source=?
                  AND underlying=? AND expiration=? AND contract_type=?
                  AND as_of_date=?
                """,
                (
                    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                    CONTRACT_REFERENCE_ENVIRONMENT,
                    CONTRACT_REFERENCE_SOURCE,
                    attempt["underlying"],
                    attempt["expiration"],
                    attempt["contract_type"],
                    attempt["as_of_date"],
                ),
            ).fetchone()
            if head is not None:
                if head["snapshot_sha256"] == snapshot_sha256:
                    conn.execute(
                        """
                        UPDATE contract_reference_attempts
                        SET status='DUPLICATE_CONFIRMED', finished_at=?,
                            snapshot_sha256=?, failure_code=NULL
                        WHERE attempt_id=?
                        """,
                        (finished_at, snapshot_sha256, attempt_id),
                    )
                    self._append_attempt_event(
                        conn,
                        attempt_id=attempt_id,
                        event_type="DUPLICATE_CONFIRMED",
                        recorded_at=finished_at,
                        snapshot_sha256=snapshot_sha256,
                    )
                    conn.commit()
                    return self.get_confirmed_contract_reference(
                        underlying=attempt["underlying"],
                        expiration=attempt["expiration"],
                        contract_type=attempt["contract_type"],
                        as_of_date=attempt["as_of_date"],
                    )
                conn.execute(
                    """
                    UPDATE contract_reference_attempts
                    SET status='CONFLICT', finished_at=?,
                        failure_code='CONCURRENT_SNAPSHOT_CONFLICT',
                        snapshot_sha256=?
                    WHERE attempt_id=?
                    """,
                    (finished_at, snapshot_sha256, attempt_id),
                )
                self._append_attempt_event(
                    conn,
                    attempt_id=attempt_id,
                    event_type="CONFLICT",
                    recorded_at=finished_at,
                    failure_code="CONCURRENT_SNAPSHOT_CONFLICT",
                    snapshot_sha256=snapshot_sha256,
                )
                conn.commit()
                return None

            expected_roots_json = attempt["expected_roots_json"]
            conn.execute(
                """
                INSERT INTO contract_reference_snapshots (
                    snapshot_sha256, schema_version, environment, source,
                    underlying, expiration, contract_type, as_of_date,
                    attempt_id, expected_roots_json, page_count,
                    contract_count, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_sha256,
                    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                    CONTRACT_REFERENCE_ENVIRONMENT,
                    CONTRACT_REFERENCE_SOURCE,
                    attempt["underlying"],
                    attempt["expiration"],
                    attempt["contract_type"],
                    attempt["as_of_date"],
                    attempt_id,
                    expected_roots_json,
                    len(pages),
                    len(normalized),
                    finished_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO contract_reference_snapshot_contracts (
                    snapshot_sha256, option_ticker, strike_text, contract_type
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        snapshot_sha256,
                        contract.option_ticker,
                        contract.canonical_record()["strike"],
                        contract.contract_type,
                    )
                    for contract in normalized
                ],
            )
            conn.execute(
                """
                INSERT INTO contract_reference_heads (
                    schema_version, environment, source, underlying,
                    expiration, contract_type, as_of_date, snapshot_sha256,
                    confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                    CONTRACT_REFERENCE_ENVIRONMENT,
                    CONTRACT_REFERENCE_SOURCE,
                    attempt["underlying"],
                    attempt["expiration"],
                    attempt["contract_type"],
                    attempt["as_of_date"],
                    snapshot_sha256,
                    finished_at,
                ),
            )
            conn.execute(
                """
                UPDATE contract_reference_attempts
                SET status='CONFIRMED', finished_at=?, failure_code=NULL,
                    snapshot_sha256=?
                WHERE attempt_id=?
                """,
                (finished_at, snapshot_sha256, attempt_id),
            )
            self._append_attempt_event(
                conn,
                attempt_id=attempt_id,
                event_type="CONFIRMED",
                recorded_at=finished_at,
                snapshot_sha256=snapshot_sha256,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.get_confirmed_contract_reference(
            underlying=attempt["underlying"],
            expiration=attempt["expiration"],
            contract_type=attempt["contract_type"],
            as_of_date=attempt["as_of_date"],
        )

    def get_contract_reference_attempt(self, attempt_id: str) -> Optional[dict]:
        """Return the guarded one-transition attempt projection."""

        conn = self._meta()
        row = conn.execute(
            """
            SELECT * FROM contract_reference_attempts WHERE attempt_id=?
            """,
            (attempt_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_contract_reference_attempt_events(
        self,
        attempt_id: str,
    ) -> list[dict]:
        """Return the append-only credential-free lifecycle evidence."""

        rows = self._meta().execute(
            """
            SELECT * FROM contract_reference_attempt_events
            WHERE attempt_id=?
            ORDER BY event_ordinal
            """,
            (attempt_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_contract_reference_pages(self, attempt_id: str) -> list[dict]:
        """Return the append-only credential-free request/page evidence."""

        rows = self._meta().execute(
            """
            SELECT * FROM contract_reference_pages
            WHERE attempt_id=?
            ORDER BY root_ordinal, page_ordinal
            """,
            (attempt_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_confirmed_contract_reference(
        self,
        *,
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of_date: str,
    ) -> Optional[ConfirmedContractReferenceSnapshot]:
        """Read only a fully replay-verified, exact-tuple snapshot head."""

        try:
            self._validate_contract_reference_key(
                underlying,
                expiration,
                contract_type,
                as_of_date,
            )
            conn = self._meta()
            head = conn.execute(
                """
                SELECT * FROM contract_reference_heads
                WHERE schema_version=? AND environment=? AND source=?
                  AND underlying=? AND expiration=? AND contract_type=?
                  AND as_of_date=?
                """,
                (
                    CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                    CONTRACT_REFERENCE_ENVIRONMENT,
                    CONTRACT_REFERENCE_SOURCE,
                    underlying,
                    expiration,
                    contract_type,
                    as_of_date,
                ),
            ).fetchone()
            if head is None or not _is_sha256(head["snapshot_sha256"]):
                return None
            snapshot = conn.execute(
                """
                SELECT * FROM contract_reference_snapshots
                WHERE snapshot_sha256=?
                """,
                (head["snapshot_sha256"],),
            ).fetchone()
            if snapshot is None:
                return None
            confirmed_at = _utc_datetime(snapshot["confirmed_at"])
            exact_values = (
                snapshot["schema_version"],
                snapshot["environment"],
                snapshot["source"],
                snapshot["underlying"],
                snapshot["expiration"],
                snapshot["contract_type"],
                snapshot["as_of_date"],
                snapshot["confirmed_at"],
            )
            if exact_values != (
                CONTRACT_REFERENCE_EVIDENCE_SCHEMA_VERSION,
                CONTRACT_REFERENCE_ENVIRONMENT,
                CONTRACT_REFERENCE_SOURCE,
                underlying,
                expiration,
                contract_type,
                as_of_date,
                head["confirmed_at"],
            ) or confirmed_at is None:
                return None
            attempt = self._load_reference_attempt(
                conn,
                snapshot["attempt_id"],
            )
            if (
                attempt["status"] != "CONFIRMED"
                or attempt["snapshot_sha256"] != snapshot["snapshot_sha256"]
                or attempt["expected_roots_json"]
                != snapshot["expected_roots_json"]
                or attempt["underlying"] != underlying
                or attempt["expiration"] != expiration
                or attempt["contract_type"] != contract_type
                or attempt["as_of_date"] != as_of_date
                or attempt["finished_at"] != snapshot["confirmed_at"]
                or _utc_datetime(attempt["finished_at"])
                != confirmed_at
                or not self._attempt_events_are_valid(conn, attempt)
            ):
                return None
            pages = self._validated_reference_pages(conn, attempt)
            contract_rows = conn.execute(
                """
                SELECT option_ticker, strike_text, contract_type
                FROM contract_reference_snapshot_contracts
                WHERE snapshot_sha256=?
                ORDER BY option_ticker
                """,
                (snapshot["snapshot_sha256"],),
            ).fetchall()
            persisted_contracts = []
            for row in contract_rows:
                if type(row["strike_text"]) is not str:
                    return None
                identity = OptionContractIdentity(
                    option_ticker=row["option_ticker"],
                    strike=float(row["strike_text"]),
                    contract_type=row["contract_type"],
                )
                if not hmac.compare_digest(
                    identity.canonical_record()["strike"],
                    row["strike_text"],
                ):
                    return None
                persisted_contracts.append(identity.as_legacy_record())
            normalized = self._normalize_reference_contracts(
                persisted_contracts,
                contract_type,
            )
            if (
                type(snapshot["page_count"]) is not int
                or type(snapshot["contract_count"]) is not int
                or len(pages) != snapshot["page_count"]
                or len(normalized) != snapshot["contract_count"]
                or sum(page.result_count for page in pages)
                != len(normalized)
            ):
                return None
            recomputed = contract_reference_snapshot_sha256(
                schema_version=attempt["schema_version"],
                environment=attempt["environment"],
                source=attempt["source"],
                underlying=attempt["underlying"],
                expiration=attempt["expiration"],
                contract_type=attempt["contract_type"],
                as_of_date=attempt["as_of_date"],
                expected_roots=self._attempt_expected_roots(attempt),
                pages=pages,
                contracts=normalized,
            )
            if not hmac.compare_digest(
                recomputed,
                snapshot["snapshot_sha256"],
            ):
                return None
            return ConfirmedContractReferenceSnapshot(
                underlying=underlying,
                expiration=expiration,
                contract_type=contract_type,
                as_of_date=as_of_date,
                attempt_id=attempt["attempt_id"],
                snapshot_sha256=snapshot["snapshot_sha256"],
                confirmed_at=snapshot["confirmed_at"],
                expected_roots=self._attempt_expected_roots(attempt),
                pages=pages,
                contracts=normalized,
                environment=snapshot["environment"],
                source=snapshot["source"],
                schema_version=snapshot["schema_version"],
            )
        except (
            ContractReferenceCacheError,
            ContractUniverseError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ):
            return None

    def get_cached_contracts(
        self, underlying: str, expiration: Optional[str], contract_type: str, as_of_date: str
    ) -> Optional[list]:
        if not expiration:
            return None
        snapshot = self.get_confirmed_contract_reference(
            underlying=underlying,
            expiration=expiration,
            contract_type=contract_type,
            as_of_date=as_of_date,
        )
        return snapshot.as_legacy_records() if snapshot is not None else None

    def save_contracts(
        self,
        underlying: str,
        expiration: Optional[str],
        contract_type: str,
        as_of_date: str,
        contracts: list,
    ):
        conn = self._meta()
        now = datetime.now().isoformat()
        rows = []
        for c in contracts:
            c_exp = expiration
            if not c_exp:
                try:
                    clean = c["option_ticker"][2:] if c["option_ticker"].startswith("O:") else c["option_ticker"]
                    for i, char in enumerate(clean):
                        if char.isdigit():
                            exp_str = clean[i:i+6]
                            c_exp = f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}"
                            break
                except Exception:
                    c_exp = "UNKNOWN"
            ticker_type = _ticker_contract_type(c["option_ticker"])
            if ticker_type != contract_type:
                continue
            rows.append((underlying, c_exp, ticker_type, as_of_date, c["option_ticker"], c["strike"], now))
        if not rows:
            return
        conn.executemany(
            """
            INSERT OR REPLACE INTO contracts_cache
            (underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def set_daily_memory_cache(self, date: str, tickers: list):
        if not tickers:
            self._mem_cache = {}
            self._mem_cache_date = None
            return
        rows = {}
        for ticker in tickers:
            row = self.get_ohlcv(ticker, date)
            if row:
                rows[row["option_ticker"]] = row
                rows[ticker] = row
        self._mem_cache = rows
        self._mem_cache_date = date

    def clear_daily_memory_cache(self):
        self._mem_cache = {}
        self._mem_cache_date = None

    def _query_price_rows(self, option_ticker: str, pricing_date: str) -> list[dict]:
        results = []
        if self.use_shards:
            for underlying, year in self._price_candidates(option_ticker, pricing_date):
                conn = self._ensure_price_conn(underlying, year)
                rows = conn.execute(
                    "SELECT * FROM option_prices WHERE option_ticker=? AND pricing_date=?",
                    (option_ticker, pricing_date),
                ).fetchall()
                results.extend(dict(r) for r in rows)
                # Alias lookup for SPX/SPXW.
                alias = "SPXW" if _ticker_root(option_ticker) == "SPX" else "SPX"
                if alias and alias != underlying:
                    alias_ticker = ("O:" if option_ticker.startswith("O:") else "") + option_ticker[2:].replace(_ticker_root(option_ticker), alias, 1)
                    rows = conn.execute(
                        "SELECT * FROM option_prices WHERE option_ticker=? AND pricing_date=?",
                        (alias_ticker, pricing_date),
                    ).fetchall()
                    results.extend(dict(r) for r in rows)
            return results

        row = self.conn.execute(
            "SELECT * FROM option_prices WHERE option_ticker=? AND pricing_date=?",
            (option_ticker, pricing_date),
        ).fetchall()
        return [dict(r) for r in row]

    def get_ohlcv(self, option_ticker: str, pricing_date: str) -> Optional[dict]:
        candidates = [option_ticker]
        root = _ticker_root(option_ticker)
        prefix = "O:" if option_ticker.startswith("O:") else ""
        if root in {"SPX", "SPXW"}:
            alt = "SPXW" if root == "SPX" else "SPX"
            clean = option_ticker[2:] if prefix else option_ticker
            candidates.append(prefix + clean.replace(root, alt, 1))
        if getattr(self, "_mem_cache_date", None) == pricing_date:
            best = None
            for candidate in candidates:
                cached = self._mem_cache.get(candidate)
                if _ohlcv_quality(cached) > _ohlcv_quality(best):
                    best = cached
            if best:
                self._mem_cache[option_ticker] = best
                return best

        best = None
        for row in self._query_price_rows(option_ticker, pricing_date):
            if _ohlcv_quality(row) > _ohlcv_quality(best):
                best = row
        return best

    def get_ohlcv_range(self, option_ticker: str, from_date: str, to_date: str) -> list:
        if self.use_shards:
            rows = []
            year_start = int(from_date[:4])
            year_end = int(to_date[:4])
            for year in range(year_start, year_end + 1):
                for underlying in _root_aliases(_ticker_root(option_ticker)):
                    conn = self._ensure_price_conn(underlying, str(year))
                    rows.extend(
                        dict(r)
                        for r in conn.execute(
                            """
                            SELECT * FROM option_prices
                            WHERE option_ticker=? AND pricing_date BETWEEN ? AND ?
                            ORDER BY pricing_date ASC
                            """,
                            (option_ticker, from_date, to_date),
                        ).fetchall()
                    )
            seen = set()
            deduped = []
            for row in rows:
                key = (row["option_ticker"], row["pricing_date"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            return deduped
        rows = self.conn.execute(
            """
            SELECT * FROM option_prices
            WHERE option_ticker=? AND pricing_date BETWEEN ? AND ?
            ORDER BY pricing_date ASC
            """,
            (option_ticker, from_date, to_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chain_for_date(self, underlying: str, expiration: str, contract_type: str, pricing_date: str) -> list:
        if self.use_shards:
            rows = []
            for candidate_underlying in _root_aliases(underlying):
                conn = self._ensure_price_conn(candidate_underlying, pricing_date[:4])
                rows.extend(
                    dict(r)
                    for r in conn.execute(
                        """
                        SELECT * FROM option_prices
                        WHERE underlying=? AND expiration=? AND contract_type=? AND pricing_date=?
                        ORDER BY strike ASC
                        """,
                        (candidate_underlying, expiration, contract_type, pricing_date),
                    ).fetchall()
                )
            best_by_strike = {}
            for row in rows:
                key = float(row["strike"])
                if key not in best_by_strike or _ohlcv_quality(row) > _ohlcv_quality(best_by_strike[key]):
                    best_by_strike[key] = row
            return [best_by_strike[k] for k in sorted(best_by_strike)]

        rows = self.conn.execute(
            """
            SELECT * FROM option_prices
            WHERE underlying=? AND expiration=? AND contract_type=? AND pricing_date=?
            ORDER BY strike ASC
            """,
            (underlying, expiration, contract_type, pricing_date),
        ).fetchall()
        return [dict(r) for r in rows if _ticker_contract_type(r["option_ticker"]) == contract_type]

    def has_chain_ohlcv(self, underlying: str, expiration: str, contract_type: str, pricing_date: str) -> bool:
        return any(r.get("close") is not None for r in self.get_chain_for_date(underlying, expiration, contract_type, pricing_date))

    def has_chain_quotes(self, underlying: str, expiration: str, contract_type: str, pricing_date: str) -> bool:
        return any(r.get("bid") is not None and r.get("ask") is not None for r in self.get_chain_for_date(underlying, expiration, contract_type, pricing_date))

    def _records_to_price_groups(self, records: list) -> Dict[Tuple[str, str], list[tuple]]:
        groups: Dict[Tuple[str, str], list[tuple]] = {}
        for record in records:
            if isinstance(record, dict):
                underlying = record.get("underlying") or _ticker_root(record.get("option_ticker", "")) or "SPX"
                year = str(record.get("pricing_date"))[:4]
            else:
                underlying = record[0] or _ticker_root(record[1]) or "SPX"
                year = str(record[5])[:4]
            groups.setdefault((underlying, year), []).append(record)
        return groups

    @staticmethod
    def _full_record_row_from_tuple(record: tuple, fetched_at: str) -> tuple:
        if len(record) == 26:
            return record
        if len(record) != 19:
            raise ValueError(f"Unsupported option row shape: {len(record)}")
        return (
            record[0],  # underlying
            record[1],  # option_ticker
            record[2],  # contract_type
            record[3],  # strike
            record[4],  # expiration
            record[5],  # pricing_date
            None, None, None,  # open/high/low
            record[6],  # close
            record[7],  # volume
            None,  # vwap
            None,  # num_trades
            record[8],  # bid
            record[9],  # ask
            record[10],  # mid
            record[11],  # bid_size
            record[12],  # ask_size
            record[13],  # implied_vol
            record[14],  # delta
            record[15],  # gamma
            record[16],  # theta
            record[17],  # vega
            None,  # open_interest
            1,  # is_synchronized
            fetched_at,
        )

    @staticmethod
    def _full_record_row_from_dict(record: dict, fetched_at: str) -> tuple:
        return (
            record.get("underlying"),
            record.get("option_ticker"),
            record.get("contract_type"),
            record.get("strike"),
            record.get("expiration"),
            record.get("pricing_date"),
            record.get("open"),
            record.get("high"),
            record.get("low"),
            record.get("close"),
            record.get("volume"),
            record.get("vwap"),
            record.get("num_trades"),
            record.get("bid"),
            record.get("ask"),
            record.get("mid"),
            record.get("bid_size"),
            record.get("ask_size"),
            record.get("implied_vol"),
            record.get("delta"),
            record.get("gamma"),
            record.get("theta"),
            record.get("vega"),
            record.get("open_interest"),
            record.get("is_synchronized", 0),
            fetched_at,
        )

    def bulk_upsert_full_records(self, records: list):
        if not records:
            return
        fetched_at = datetime.now().isoformat()
        grouped = self._records_to_price_groups(records) if self.use_shards else {("LEGACY", "LEGACY"): records}
        if self.use_shards:
            for (underlying, year), recs in grouped.items():
                conn = self._ensure_price_conn(underlying, year)
                rows = []
                for record in recs:
                    if isinstance(record, dict):
                        rows.append(self._full_record_row_from_dict(record, fetched_at))
                    else:
                        rows.append(self._full_record_row_from_tuple(record, fetched_at))
                conn.executemany(
                    """
                    INSERT INTO option_prices
                       (underlying, option_ticker, contract_type, strike, expiration,
                        pricing_date, open, high, low, close, volume, vwap, num_trades,
                        bid, ask, mid, bid_size, ask_size, implied_vol, delta, gamma,
                        theta, vega, open_interest, is_synchronized, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
                        underlying=excluded.underlying,
                        contract_type=excluded.contract_type,
                        strike=excluded.strike,
                        expiration=excluded.expiration,
                        open=COALESCE(excluded.open, option_prices.open),
                        high=COALESCE(excluded.high, option_prices.high),
                        low=COALESCE(excluded.low, option_prices.low),
                        close=COALESCE(excluded.close, option_prices.close),
                        volume=COALESCE(excluded.volume, option_prices.volume),
                        vwap=COALESCE(excluded.vwap, option_prices.vwap),
                        num_trades=COALESCE(excluded.num_trades, option_prices.num_trades),
                        bid=COALESCE(excluded.bid, option_prices.bid),
                        ask=COALESCE(excluded.ask, option_prices.ask),
                        mid=COALESCE(excluded.mid, option_prices.mid),
                        bid_size=COALESCE(excluded.bid_size, option_prices.bid_size),
                        ask_size=COALESCE(excluded.ask_size, option_prices.ask_size),
                        implied_vol=COALESCE(excluded.implied_vol, option_prices.implied_vol),
                        delta=COALESCE(excluded.delta, option_prices.delta),
                        gamma=COALESCE(excluded.gamma, option_prices.gamma),
                        theta=COALESCE(excluded.theta, option_prices.theta),
                        vega=COALESCE(excluded.vega, option_prices.vega),
                        open_interest=COALESCE(excluded.open_interest, option_prices.open_interest),
                        is_synchronized=MAX(excluded.is_synchronized, option_prices.is_synchronized),
                        fetched_at=excluded.fetched_at
                    """,
                    rows,
                )
                conn.commit()
            return

        rows = []
        for record in records:
            if isinstance(record, dict):
                rows.append(self._full_record_row_from_dict(record, fetched_at))
            else:
                rows.append(self._full_record_row_from_tuple(record, fetched_at))
        self.conn.executemany(
            """
            INSERT INTO option_prices
               (underlying, option_ticker, contract_type, strike, expiration,
                pricing_date, open, high, low, close, volume, vwap, num_trades,
                bid, ask, mid, bid_size, ask_size, implied_vol, delta, gamma,
                theta, vega, open_interest, is_synchronized, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
                underlying=excluded.underlying,
                contract_type=excluded.contract_type,
                strike=excluded.strike,
                expiration=excluded.expiration,
                open=COALESCE(excluded.open, option_prices.open),
                high=COALESCE(excluded.high, option_prices.high),
                low=COALESCE(excluded.low, option_prices.low),
                close=COALESCE(excluded.close, option_prices.close),
                volume=COALESCE(excluded.volume, option_prices.volume),
                vwap=COALESCE(excluded.vwap, option_prices.vwap),
                num_trades=COALESCE(excluded.num_trades, option_prices.num_trades),
                bid=COALESCE(excluded.bid, option_prices.bid),
                ask=COALESCE(excluded.ask, option_prices.ask),
                mid=COALESCE(excluded.mid, option_prices.mid),
                bid_size=COALESCE(excluded.bid_size, option_prices.bid_size),
                ask_size=COALESCE(excluded.ask_size, option_prices.ask_size),
                implied_vol=COALESCE(excluded.implied_vol, option_prices.implied_vol),
                delta=COALESCE(excluded.delta, option_prices.delta),
                gamma=COALESCE(excluded.gamma, option_prices.gamma),
                theta=COALESCE(excluded.theta, option_prices.theta),
                vega=COALESCE(excluded.vega, option_prices.vega),
                open_interest=COALESCE(excluded.open_interest, option_prices.open_interest),
                is_synchronized=MAX(excluded.is_synchronized, option_prices.is_synchronized),
                fetched_at=excluded.fetched_at
            """,
            rows,
        )
        self.conn.commit()

    def bulk_upsert_ohlcv(self, records: list):
        if not records:
            return
        if self.use_shards:
            grouped = self._records_to_price_groups(records)
            for (underlying, year), recs in grouped.items():
                conn = self._ensure_price_conn(underlying, year)
                conn.executemany(
                    """
                    INSERT INTO option_prices
                    (underlying, option_ticker, contract_type, strike, expiration, pricing_date,
                     open, high, low, close, volume, vwap, num_trades, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
                      underlying=excluded.underlying,
                      contract_type=excluded.contract_type,
                      strike=excluded.strike,
                      expiration=excluded.expiration,
                      open=excluded.open, high=excluded.high, low=excluded.low,
                      close=excluded.close, volume=excluded.volume, vwap=excluded.vwap,
                      num_trades=excluded.num_trades, fetched_at=excluded.fetched_at
                    """,
                    recs,
                )
                conn.commit()
            return
        self.conn.executemany(
            """
            INSERT INTO option_prices
            (underlying, option_ticker, contract_type, strike, expiration,
             pricing_date, open, high, low, close, volume, vwap, num_trades, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
              underlying=excluded.underlying,
              contract_type=excluded.contract_type,
              strike=excluded.strike,
              expiration=excluded.expiration,
              open=excluded.open, high=excluded.high, low=excluded.low,
              close=excluded.close, volume=excluded.volume, vwap=excluded.vwap,
              num_trades=excluded.num_trades, fetched_at=excluded.fetched_at
            """,
            records,
        )
        self.conn.commit()

    def get_max_pricing_date(self, underlying: Optional[str] = None) -> Optional[str]:
        if not self.use_shards:
            if underlying:
                row = self.conn.execute(
                    "SELECT MAX(pricing_date) AS max_date FROM option_prices WHERE underlying=?",
                    (underlying,),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT MAX(pricing_date) AS max_date FROM option_prices"
                ).fetchone()
            return row["max_date"] if row and row["max_date"] else None

        max_date = None
        shard_root = Path(self.shard_dir)
        for shard_path in shard_root.glob("*/*.db"):
            conn = sqlite3.connect(shard_path, timeout=60)
            conn.row_factory = sqlite3.Row
            try:
                if underlying:
                    row = conn.execute(
                        "SELECT MAX(pricing_date) AS max_date FROM option_prices WHERE underlying=?",
                        (underlying,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT MAX(pricing_date) AS max_date FROM option_prices"
                    ).fetchone()
                if row and row["max_date"] and (max_date is None or row["max_date"] > max_date):
                    max_date = row["max_date"]
            finally:
                conn.close()
        return max_date

    def bulk_update_quotes(self, records: list):
        if not records:
            return
        if self.use_shards:
            grouped: Dict[Tuple[str, str], list[tuple]] = {}
            for bid, ask, mid, bid_size, ask_size, ticker, pricing_date in records:
                underlying = _ticker_root(ticker) or "SPX"
                year = pricing_date[:4]
                grouped.setdefault((underlying, year), []).append((bid, ask, mid, bid_size, ask_size, ticker, pricing_date))
            for (underlying, year), recs in grouped.items():
                conn = self._ensure_price_conn(underlying, year)
                conn.executemany(
                    """
                    UPDATE option_prices
                    SET bid=?, ask=?, mid=?, bid_size=?, ask_size=?
                    WHERE option_ticker=? AND pricing_date=?
                    """,
                    recs,
                )
                conn.commit()
            return
        self.conn.executemany(
            "UPDATE option_prices SET bid=?, ask=?, mid=?, bid_size=?, ask_size=? WHERE option_ticker=? AND pricing_date=?",
            records,
        )
        self.conn.commit()

    def upsert_full_record(self, record: dict):
        now = datetime.now().isoformat()
        is_sync = record.get("is_synchronized", 0)
        row = (
            record.get("underlying"),
            record.get("option_ticker"),
            record.get("contract_type"),
            record.get("strike"),
            record.get("expiration"),
            record.get("pricing_date"),
            record.get("open"),
            record.get("high"),
            record.get("low"),
            record.get("close"),
            record.get("volume"),
            record.get("vwap"),
            record.get("num_trades"),
            record.get("bid"),
            record.get("ask"),
            record.get("mid"),
            record.get("bid_size"),
            record.get("ask_size"),
            is_sync,
            now,
        )
        if self.use_shards:
            underlying = record.get("underlying") or _ticker_root(record.get("option_ticker", "")) or "SPX"
            year = str(record.get("pricing_date"))[:4]
            conn = self._ensure_price_conn(underlying, year)
            conn.execute(
                """
                INSERT INTO option_prices
                (underlying, option_ticker, contract_type, strike, expiration,
                 pricing_date, open, high, low, close, volume, vwap, num_trades,
                 bid, ask, mid, bid_size, ask_size, is_synchronized, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
                  underlying=COALESCE(excluded.underlying, option_prices.underlying),
                  contract_type=COALESCE(excluded.contract_type, option_prices.contract_type),
                  strike=COALESCE(excluded.strike, option_prices.strike),
                  expiration=COALESCE(excluded.expiration, option_prices.expiration),
                  open=COALESCE(excluded.open, option_prices.open),
                  high=COALESCE(excluded.high, option_prices.high),
                  low=COALESCE(excluded.low, option_prices.low),
                  close=COALESCE(excluded.close, option_prices.close),
                  volume=COALESCE(excluded.volume, option_prices.volume),
                  vwap=COALESCE(excluded.vwap, option_prices.vwap),
                  num_trades=COALESCE(excluded.num_trades, option_prices.num_trades),
                  bid=COALESCE(excluded.bid, option_prices.bid),
                  ask=COALESCE(excluded.ask, option_prices.ask),
                  mid=COALESCE(excluded.mid, option_prices.mid),
                  bid_size=COALESCE(excluded.bid_size, option_prices.bid_size),
                  ask_size=COALESCE(excluded.ask_size, option_prices.ask_size),
                  is_synchronized=MAX(excluded.is_synchronized, option_prices.is_synchronized),
                  fetched_at=excluded.fetched_at
                """,
                row,
            )
            conn.commit()
            return
        self.conn.execute(
            """
            INSERT INTO option_prices
            (underlying, option_ticker, contract_type, strike, expiration,
             pricing_date, open, high, low, close, volume, vwap, num_trades,
             bid, ask, mid, bid_size, ask_size, is_synchronized, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
              underlying=COALESCE(excluded.underlying, option_prices.underlying),
              contract_type=COALESCE(excluded.contract_type, option_prices.contract_type),
              strike=COALESCE(excluded.strike, option_prices.strike),
              expiration=COALESCE(excluded.expiration, option_prices.expiration),
              open=COALESCE(excluded.open, option_prices.open),
              high=COALESCE(excluded.high, option_prices.high),
              low=COALESCE(excluded.low, option_prices.low),
              close=COALESCE(excluded.close, option_prices.close),
              volume=COALESCE(excluded.volume, option_prices.volume),
              vwap=COALESCE(excluded.vwap, option_prices.vwap),
              num_trades=COALESCE(excluded.num_trades, option_prices.num_trades),
              bid=COALESCE(excluded.bid, option_prices.bid),
              ask=COALESCE(excluded.ask, option_prices.ask),
              mid=COALESCE(excluded.mid, option_prices.mid),
              bid_size=COALESCE(excluded.bid_size, option_prices.bid_size),
              ask_size=COALESCE(excluded.ask_size, option_prices.ask_size),
              is_synchronized=MAX(excluded.is_synchronized, option_prices.is_synchronized),
              fetched_at=excluded.fetched_at
            """,
            row,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Fetch logs
    # ------------------------------------------------------------------

    def is_fetch_complete(self, underlying: str, pricing_date: str, expiration: str = None, contract_type: str = None, data_type: str = "ohlcv") -> bool:
        conn = self._meta()
        row = conn.execute(
            """
            SELECT status FROM fetch_log
            WHERE underlying=? AND pricing_date=? AND expiration IS ? AND contract_type IS ? AND data_type=?
            """,
            (underlying, pricing_date, expiration, contract_type, data_type),
        ).fetchone()
        return row is not None and row["status"] == "complete"

    def mark_fetch_complete(self, underlying: str, pricing_date: str, expiration: str = None, contract_type: str = None, data_type: str = "ohlcv"):
        conn = self._meta()
        now = datetime.now().isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_log
            (underlying, pricing_date, expiration, contract_type, data_type, status, fetched_at)
            VALUES (?, ?, ?, ?, ?, 'complete', ?)
            """,
            (underlying, pricing_date, expiration, contract_type, data_type, now),
        )
        conn.commit()

    def is_ticker_range_fetched(self, option_ticker: str, from_date: str, to_date: str) -> bool:
        conn = self._meta()
        rows = conn.execute(
            "SELECT from_date, to_date FROM ohlcv_fetch_log WHERE option_ticker=?",
            (option_ticker,),
        ).fetchall()
        for row in rows:
            if row["from_date"] <= from_date and row["to_date"] >= to_date:
                return True
        if self.get_ohlcv(option_ticker, to_date):
            return True
        return False

    def mark_ticker_range_fetched(self, option_ticker: str, from_date: str, to_date: str, bar_count: int = 0):
        conn = self._meta()
        now = datetime.now().isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO ohlcv_fetch_log
            (option_ticker, from_date, to_date, bar_count, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (option_ticker, from_date, to_date, bar_count, now),
        )
        conn.commit()

    def bulk_mark_ticker_ranges_fetched(self, records: list):
        conn = self._meta()
        now = datetime.now().isoformat()
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_fetch_log
            (option_ticker, from_date, to_date, bar_count, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(t, f, to, bc, now) for t, f, to, bc in records],
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):
        if self.use_shards:
            for conn in self._price_conns.values():
                conn.close()
            self.meta_conn.close()
        else:
            self.conn.close()
