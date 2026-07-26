#!/usr/bin/env python3
"""Import OptionsDX SPX EOD text files into the SQLite option backtest cache."""

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.option_data_cache import OptionDataCache


DEFAULT_SOURCE_DIR = Path("/Users/btian/Documents/Downloads_Sync")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "backtest_cache" / "optionsdx_spx_eod"
DEFAULT_DB_PATH = PROJECT_ROOT / "backtest_cache" / "option_data.db"


UPSERT_OPTION_SQL = """
INSERT INTO option_prices
   (underlying, option_ticker, contract_type, strike, expiration,
    pricing_date, open, high, low, close, volume, vwap, num_trades,
    bid, ask, mid, bid_size, ask_size, implied_vol, delta, gamma, theta,
    vega, open_interest, is_synchronized, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, NULL,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)
ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
    underlying=excluded.underlying,
    contract_type=excluded.contract_type,
    strike=excluded.strike,
    expiration=excluded.expiration,
    close=COALESCE(excluded.close, option_prices.close),
    volume=COALESCE(excluded.volume, option_prices.volume),
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
    is_synchronized=MAX(excluded.is_synchronized, option_prices.is_synchronized),
    fetched_at=excluded.fetched_at
"""


def clean_key(value: str) -> str:
    return value.strip().strip("[]").strip()


def clean_value(value):
    return value.strip() if isinstance(value, str) else value


def to_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def positive_or_none(value):
    if value is None or value <= 0:
        return None
    return value


def to_int(value: str):
    parsed = to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def parse_size(value: str):
    value = (value or "").strip()
    if not value or "x" not in value.lower():
        return None, None
    left, right = value.lower().split("x", 1)
    return to_int(left), to_int(right)


def option_ticker(expiration: str, contract_type: str, strike_text: str) -> str:
    exp = datetime.strptime(expiration, "%Y-%m-%d")
    flag = "C" if contract_type == "call" else "P"
    try:
        strike_int = int((Decimal(strike_text.strip()) * Decimal("1000")).to_integral_value())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"Bad strike value: {strike_text!r}") from exc
    return f"O:SPX{exp:%y%m%d}{flag}{strike_int:08d}"


def row_to_records(row: dict, fetched_at: str):
    quote_date = row.get("QUOTE_DATE")
    expiration = row.get("EXPIRE_DATE")
    strike_text = row.get("STRIKE")
    strike = to_float(strike_text)
    if not quote_date or not expiration or strike is None:
        return []

    records = []
    for side, prefix in (("call", "C"), ("put", "P")):
        bid = positive_or_none(to_float(row.get(f"{prefix}_BID")))
        ask = positive_or_none(to_float(row.get(f"{prefix}_ASK")))
        last = positive_or_none(to_float(row.get(f"{prefix}_LAST")))
        if bid is None and ask is None and last is None:
            continue
        if bid is not None and ask is not None:
            mid = round((bid + ask) / 2.0, 6)
        else:
            mid = last
        close = last or mid
        bid_size, ask_size = parse_size(row.get(f"{prefix}_SIZE"))
        records.append(
            (
                "SPX",
                option_ticker(expiration, side, strike_text),
                side,
                strike,
                expiration,
                quote_date,
                close,
                to_int(row.get(f"{prefix}_VOLUME")),
                bid,
                ask,
                mid,
                bid_size,
                ask_size,
                to_float(row.get(f"{prefix}_IV")),
                to_float(row.get(f"{prefix}_DELTA")),
                to_float(row.get(f"{prefix}_GAMMA")),
                to_float(row.get(f"{prefix}_THETA")),
                to_float(row.get(f"{prefix}_VEGA")),
                fetched_at,
            )
        )
    return records


def find_archives(source_dir: Path):
    return sorted(source_dir.glob("spx_eod_*.7z"))


def extract_archives(source_dir: Path, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archives = find_archives(source_dir)
    if not archives:
        raise FileNotFoundError(f"No spx_eod_*.7z archives found in {source_dir}")

    extracted = 0
    skipped = 0
    for archive in archives:
        members = subprocess.check_output(["bsdtar", "-tf", str(archive)], text=True).splitlines()
        missing = [member for member in members if member.endswith(".txt") and not (raw_dir / member).exists()]
        if not missing:
            skipped += 1
            print(f"[extract] skip {archive.name}: all text files already present", flush=True)
            continue
        print(f"[extract] {archive.name}: {len(missing)} text files", flush=True)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(raw_dir)], check=True)
        extracted += 1
    return extracted, skipped


def iter_text_files(raw_dir: Path):
    return sorted(raw_dir.glob("spx_eod_*.txt"))


def init_temp_tables(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TEMP TABLE IF NOT EXISTS optionsdx_imported_contracts (
            underlying TEXT NOT NULL,
            expiration TEXT NOT NULL,
            contract_type TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            option_ticker TEXT NOT NULL,
            strike REAL NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (option_ticker, as_of_date)
        );
        CREATE TEMP TABLE IF NOT EXISTS optionsdx_imported_tickers (
            option_ticker TEXT PRIMARY KEY
        );
        CREATE TEMP TABLE IF NOT EXISTS optionsdx_imported_ranges (
            option_ticker TEXT PRIMARY KEY,
            from_date TEXT NOT NULL,
            to_date TEXT NOT NULL,
            bar_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def flush_records(cache: OptionDataCache, conn: sqlite3.Connection, records: list):
    if not records:
        return
    cache.bulk_upsert_full_records(records)
    contract_rows = [
        (r[0], r[4], r[2], r[4], r[1], r[3], r[-1])
        for r in records
    ]
    ticker_rows = [(r[1],) for r in records]
    range_rows = {}
    for r in records:
        ticker = r[1]
        pricing_date = r[5]
        if ticker not in range_rows:
            range_rows[ticker] = [pricing_date, pricing_date, 0]
        else:
            range_rows[ticker][0] = min(range_rows[ticker][0], pricing_date)
            range_rows[ticker][1] = max(range_rows[ticker][1], pricing_date)
        range_rows[ticker][2] += 1
    conn.executemany(
        """
        INSERT OR IGNORE INTO optionsdx_imported_contracts
            (underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        contract_rows,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO optionsdx_imported_tickers (option_ticker) VALUES (?)",
        ticker_rows,
    )
    conn.executemany(
        """
        INSERT INTO optionsdx_imported_ranges
            (option_ticker, from_date, to_date, bar_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(option_ticker) DO UPDATE SET
            from_date = CASE
                WHEN excluded.from_date < optionsdx_imported_ranges.from_date THEN excluded.from_date
                ELSE optionsdx_imported_ranges.from_date
            END,
            to_date = CASE
                WHEN excluded.to_date > optionsdx_imported_ranges.to_date THEN excluded.to_date
                ELSE optionsdx_imported_ranges.to_date
            END,
            bar_count = optionsdx_imported_ranges.bar_count + excluded.bar_count
        """,
        [(ticker, row[0], row[1], row[2]) for ticker, row in range_rows.items()],
    )


def finalize_cache_tables(conn: sqlite3.Connection, fetched_at: str):
    conn.execute(
        """
        INSERT OR REPLACE INTO contracts_cache
            (underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at)
        SELECT underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at
        FROM optionsdx_imported_contracts
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO fetch_log
            (underlying, pricing_date, expiration, contract_type, data_type, status, fetched_at)
        SELECT underlying, as_of_date, expiration, contract_type, 'contracts', 'complete', ?
        FROM (
            SELECT DISTINCT underlying, as_of_date, expiration, contract_type
            FROM optionsdx_imported_contracts
        )
        """,
        (fetched_at,),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO ohlcv_fetch_log
            (option_ticker, from_date, to_date, bar_count, fetched_at)
        SELECT option_ticker, from_date, to_date, bar_count, ?
        FROM optionsdx_imported_ranges
        """,
        (fetched_at,),
    )


def import_text_files(db_path: Path, raw_dir: Path, batch_size: int):
    files = iter_text_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No spx_eod_*.txt files found in {raw_dir}")

    fetched_at = datetime.now().isoformat()
    cache = OptionDataCache(str(db_path))
    conn = cache.conn
    init_temp_tables(conn)

    total_records = 0
    total_rows = 0
    try:
        for path in files:
            records = []
            file_rows = 0
            file_records = 0
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                reader.fieldnames = [clean_key(name) for name in reader.fieldnames]
                for row in reader:
                    normalized = {clean_key(key): clean_value(value) for key, value in row.items()}
                    new_records = row_to_records(normalized, fetched_at)
                    if not new_records:
                        continue
                    records.extend(new_records)
                    file_rows += 1
                    file_records += len(new_records)
                    if len(records) >= batch_size:
                        flush_records(cache, conn, records)
                        total_records += len(records)
                        records.clear()
                flush_records(cache, conn, records)
                total_records += len(records)
            total_rows += file_rows
            conn.commit()
            print(
                f"[import] {path.name}: {file_rows:,} source rows -> {file_records:,} option records",
                flush=True,
            )

        finalize_cache_tables(conn, fetched_at)
        conn.commit()
    finally:
        cache.close()
    return len(files), total_rows, total_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    raw_dir = args.cache_dir / "raw_txt"
    if not args.skip_extract:
        extracted, skipped = extract_archives(args.source_dir, raw_dir)
        print(f"[extract] complete: {extracted} extracted, {skipped} already present", flush=True)

    file_count, source_rows, option_records = import_text_files(args.db_path, raw_dir, args.batch_size)
    print(
        f"[done] imported {option_records:,} option records from {source_rows:,} source rows "
        f"across {file_count:,} text files into {args.db_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
