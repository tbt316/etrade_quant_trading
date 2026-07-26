#!/usr/bin/env python3
"""Migrate the legacy monolithic option cache into shard files."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.option_data_cache import OptionDataCache


DEFAULT_LEGACY_DB = PROJECT_ROOT / "backtest_cache" / "option_data.db"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "_option_data_legacy_backup"

OPTION_PRICE_COLUMNS = [
    "underlying",
    "option_ticker",
    "contract_type",
    "strike",
    "expiration",
    "pricing_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "num_trades",
    "bid",
    "ask",
    "mid",
    "bid_size",
    "ask_size",
    "implied_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "open_interest",
    "is_synchronized",
    "fetched_at",
]

META_TABLES = {
    "contracts_cache": [
        "underlying",
        "expiration",
        "contract_type",
        "as_of_date",
        "option_ticker",
        "strike",
        "fetched_at",
    ],
    "fetch_log": [
        "underlying",
        "pricing_date",
        "expiration",
        "contract_type",
        "data_type",
        "status",
        "fetched_at",
    ],
    "ohlcv_fetch_log": [
        "option_ticker",
        "from_date",
        "to_date",
        "bar_count",
        "fetched_at",
    ],
}


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def quote_identifier(value: str) -> str:
    return "\"" + value.replace("\"", "\"\"") + "\""


def copy_table_rows(
    src: sqlite3.Connection,
    dest: sqlite3.Connection,
    table: str,
    columns: list[str],
    batch_size: int,
) -> int:
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(quote_identifier(col) for col in columns)
    select_sql = f"SELECT {column_sql} FROM {quote_identifier(table)}"
    insert_sql = f"""
        INSERT OR REPLACE INTO {quote_identifier(table)} ({column_sql})
        VALUES ({placeholders})
    """
    cursor = src.execute(select_sql)
    total = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        dest.executemany(insert_sql, [tuple(row) for row in rows])
        dest.commit()
        total += len(rows)
        print(f"[meta] {table}: copied {total:,} rows", flush=True)
    return total


def copy_price_rows(
    src: sqlite3.Connection,
    cache: OptionDataCache,
    batch_size: int,
) -> int:
    select_sql = f"""
        SELECT {", ".join(quote_identifier(col) for col in OPTION_PRICE_COLUMNS)}
        FROM option_prices
        ORDER BY underlying, pricing_date, option_ticker
    """
    cursor = src.execute(select_sql)
    total = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        cache.bulk_upsert_full_records([tuple(row) for row in rows])
        total += len(rows)
        print(f"[prices] copied {total:,} rows", flush=True)
    return total


def backup_legacy_db(legacy_db: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{legacy_db.stem}_{stamp}{legacy_db.suffix}"
    shutil.move(str(legacy_db), str(backup_path))
    return backup_path


def migrate(legacy_db: Path, backup_dir: Path, batch_size: int) -> None:
    if not legacy_db.exists():
        raise FileNotFoundError(f"Legacy cache not found: {legacy_db}")

    legacy_conn = sqlite3.connect(legacy_db, timeout=120)
    legacy_conn.row_factory = sqlite3.Row
    try:
        if not table_exists(legacy_conn, "option_prices"):
            print(f"[skip] {legacy_db} already looks migrated", flush=True)
            return

        shard_dir = legacy_db.parent / "option_data_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        meta_tmp = legacy_db.with_name(f"{legacy_db.name}.migrating")
        if meta_tmp.exists():
            meta_tmp.unlink()

        cache = OptionDataCache(str(meta_tmp))
        try:
            meta_conn = cache.conn
            meta_total = 0
            for table, columns in META_TABLES.items():
                if table_exists(legacy_conn, table):
                    meta_total += copy_table_rows(legacy_conn, meta_conn, table, columns, batch_size)
            print(f"[meta] copied {meta_total:,} metadata rows", flush=True)

            price_total = copy_price_rows(legacy_conn, cache, batch_size)
            print(f"[prices] copied {price_total:,} price rows", flush=True)
            meta_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            meta_conn.commit()
        finally:
            cache.close()

        legacy_conn.close()
        backup_path = backup_legacy_db(legacy_db, backup_dir)
        os.replace(meta_tmp, legacy_db)

        verifier = OptionDataCache(str(legacy_db))
        try:
            max_date = verifier.get_max_pricing_date("SPX")
            print(f"[verify] SPX max pricing date: {max_date}", flush=True)
        finally:
            verifier.close()

        print(f"[done] legacy cache moved to {backup_path}", flush=True)
    finally:
        try:
            legacy_conn.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-db", type=Path, default=DEFAULT_LEGACY_DB)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate(args.legacy_db, args.backup_dir, args.batch_size)


if __name__ == "__main__":
    main()
