"""
Sharded SQLite cache for historical option pricing data.

Price rows live in underlying/year shards so no single file grows without
bound. Metadata tables stay in a small sidecar database.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


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
            self.meta_conn.execute("PRAGMA journal_mode=WAL")
            self.meta_conn.execute("PRAGMA synchronous=NORMAL")
            self._create_meta_tables()
            self.conn = self.meta_conn
        else:
            self.conn = sqlite3.connect(db_path, timeout=60)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self._create_legacy_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

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

    def get_cached_contracts(
        self, underlying: str, expiration: Optional[str], contract_type: str, as_of_date: str
    ) -> Optional[list]:
        conn = self._meta()
        if expiration:
            rows = conn.execute(
                """
                SELECT option_ticker, strike FROM contracts_cache
                WHERE underlying=? AND expiration=? AND contract_type=? AND as_of_date=?
                ORDER BY strike ASC
                """,
                (underlying, expiration, contract_type, as_of_date),
            ).fetchall()
        else:
            if not self.is_fetch_complete(underlying, as_of_date, None, contract_type, "contracts"):
                return None
            rows = conn.execute(
                """
                SELECT option_ticker, strike FROM contracts_cache
                WHERE underlying=? AND contract_type=? AND as_of_date=?
                ORDER BY strike ASC
                """,
                (underlying, contract_type, as_of_date),
            ).fetchall()
        if not rows:
            return None
        filtered = [
            {
                "option_ticker": r["option_ticker"],
                "strike": r["strike"],
                "contract_type": contract_type,
            }
            for r in rows
            if _ticker_contract_type(r["option_ticker"]) == contract_type
        ]
        return filtered or None

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
