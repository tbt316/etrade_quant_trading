"""
SQLite-based local cache for historical option pricing data.
Replaces the pickle-based caching approach with indexed, queryable storage.
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional


# Default DB path relative to project root
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backtest_cache",
    "option_data.db",
)


class OptionDataCache:
    """SQLite-backed option data cache with indexed lookups."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
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
        """)
        self.conn.commit()

    # ── Contract List Cache ─────────────────────────────────────────

    def get_cached_contracts(
        self, underlying: str, expiration: str, contract_type: str, as_of_date: str
    ) -> Optional[list]:
        """Return cached contracts list, or None if not cached."""
        rows = self.conn.execute(
            """SELECT option_ticker, strike FROM contracts_cache
               WHERE underlying=? AND expiration=? AND contract_type=? AND as_of_date=?
               ORDER BY strike ASC""",
            (underlying, expiration, contract_type, as_of_date),
        ).fetchall()
        if not rows:
            return None
        return [{"option_ticker": r["option_ticker"], "strike": r["strike"]} for r in rows]

    def save_contracts(
        self,
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of_date: str,
        contracts: list,
    ):
        """Save contracts list to cache."""
        now = datetime.now().isoformat()
        rows = [
            (underlying, expiration, contract_type, as_of_date,
             c["option_ticker"], c["strike"], now)
            for c in contracts
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO contracts_cache
               (underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    # ── OHLCV Price Cache ───────────────────────────────────────────

    def get_ohlcv(
        self, option_ticker: str, pricing_date: str
    ) -> Optional[dict]:
        """Get single OHLCV record."""
        row = self.conn.execute(
            "SELECT * FROM option_prices WHERE option_ticker=? AND pricing_date=?",
            (option_ticker, pricing_date),
        ).fetchone()
        return dict(row) if row else None

    def get_ohlcv_range(
        self, option_ticker: str, from_date: str, to_date: str
    ) -> list:
        """Get OHLCV records for a date range."""
        rows = self.conn.execute(
            """SELECT * FROM option_prices
               WHERE option_ticker=? AND pricing_date BETWEEN ? AND ?
               ORDER BY pricing_date ASC""",
            (option_ticker, from_date, to_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chain_for_date(
        self,
        underlying: str,
        expiration: str,
        contract_type: str,
        pricing_date: str,
    ) -> list:
        """Get all option prices for a chain on a specific date."""
        rows = self.conn.execute(
            """SELECT * FROM option_prices
               WHERE underlying=? AND expiration=? AND contract_type=?
                 AND pricing_date=?
               ORDER BY strike ASC""",
            (underlying, expiration, contract_type, pricing_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def has_chain_ohlcv(
        self, underlying: str, expiration: str, contract_type: str, pricing_date: str
    ) -> bool:
        """Check if we have OHLCV data for a chain on a date."""
        row = self.conn.execute(
            """SELECT COUNT(*) as cnt FROM option_prices
               WHERE underlying=? AND expiration=? AND contract_type=?
                 AND pricing_date=? AND close IS NOT NULL""",
            (underlying, expiration, contract_type, pricing_date),
        ).fetchone()
        return row["cnt"] > 0

    def has_chain_quotes(
        self, underlying: str, expiration: str, contract_type: str, pricing_date: str
    ) -> bool:
        """Check if we have bid/ask quote data for a chain on a date."""
        row = self.conn.execute(
            """SELECT COUNT(*) as cnt FROM option_prices
               WHERE underlying=? AND expiration=? AND contract_type=?
                 AND pricing_date=? AND bid IS NOT NULL AND ask IS NOT NULL""",
            (underlying, expiration, contract_type, pricing_date),
        ).fetchone()
        return row["cnt"] > 0

    def bulk_upsert_ohlcv(self, records: list):
        """Batch insert/update OHLCV records."""
        if not records:
            return
        self.conn.executemany(
            """INSERT INTO option_prices
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
                 num_trades=excluded.num_trades, fetched_at=excluded.fetched_at""",
            records,
        )
        self.conn.commit()

    def bulk_update_quotes(self, records: list):
        """Batch update bid/ask/mid on existing records."""
        if not records:
            return
        self.conn.executemany(
            """UPDATE option_prices
               SET bid=?, ask=?, mid=?, bid_size=?, ask_size=?
               WHERE option_ticker=? AND pricing_date=?""",
            records,
        )
        self.conn.commit()

    def upsert_full_record(self, record: dict):
        """Insert or update a single complete record."""
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO option_prices
               (underlying, option_ticker, contract_type, strike, expiration,
                pricing_date, open, high, low, close, volume, vwap, num_trades,
                bid, ask, mid, bid_size, ask_size, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(option_ticker, pricing_date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume, vwap=excluded.vwap,
                 num_trades=excluded.num_trades, bid=excluded.bid, ask=excluded.ask,
                 mid=excluded.mid, bid_size=excluded.bid_size, ask_size=excluded.ask_size,
                 fetched_at=excluded.fetched_at""",
            (
                record.get("underlying"), record.get("option_ticker"),
                record.get("contract_type"), record.get("strike"),
                record.get("expiration"), record.get("pricing_date"),
                record.get("open"), record.get("high"), record.get("low"),
                record.get("close"), record.get("volume"), record.get("vwap"),
                record.get("num_trades"), record.get("bid"), record.get("ask"),
                record.get("mid"), record.get("bid_size"), record.get("ask_size"),
                now,
            ),
        )
        self.conn.commit()

    # ── Fetch Log ───────────────────────────────────────────────────

    def is_fetch_complete(
        self, underlying: str, pricing_date: str,
        expiration: str = None, contract_type: str = None,
        data_type: str = "ohlcv",
    ) -> bool:
        row = self.conn.execute(
            """SELECT status FROM fetch_log
               WHERE underlying=? AND pricing_date=?
                 AND expiration IS ? AND contract_type IS ?
                 AND data_type=?""",
            (underlying, pricing_date, expiration, contract_type, data_type),
        ).fetchone()
        return row is not None and row["status"] == "complete"

    def mark_fetch_complete(
        self, underlying: str, pricing_date: str,
        expiration: str = None, contract_type: str = None,
        data_type: str = "ohlcv",
    ):
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO fetch_log
               (underlying, pricing_date, expiration, contract_type, data_type, status, fetched_at)
               VALUES (?, ?, ?, ?, ?, 'complete', ?)""",
            (underlying, pricing_date, expiration, contract_type, data_type, now),
        )
        self.conn.commit()

    # ── OHLCV Negative Cache (per-ticker date range) ────────────────

    def is_ticker_range_fetched(
        self, option_ticker: str, from_date: str, to_date: str
    ) -> bool:
        """Check if we've already attempted to fetch OHLCV for this ticker/range.
        Returns True if the requested range is fully contained within a previously fetched range."""
        rows = self.conn.execute(
            """SELECT from_date, to_date FROM ohlcv_fetch_log
               WHERE option_ticker=?""",
            (option_ticker,),
        ).fetchall()
        
        for row in rows:
            if row["from_date"] <= from_date and row["to_date"] >= to_date:
                return True
        return False

    def mark_ticker_range_fetched(
        self, option_ticker: str, from_date: str, to_date: str, bar_count: int = 0
    ):
        """Record that we've fetched (or attempted) OHLCV for this ticker/range."""
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO ohlcv_fetch_log
               (option_ticker, from_date, to_date, bar_count, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (option_ticker, from_date, to_date, bar_count, now),
        )
        self.conn.commit()

    def bulk_mark_ticker_ranges_fetched(
        self, records: list
    ):
        """Batch record fetch attempts. records: list of (ticker, from, to, bar_count)."""
        now = datetime.now().isoformat()
        self.conn.executemany(
            """INSERT OR REPLACE INTO ohlcv_fetch_log
               (option_ticker, from_date, to_date, bar_count, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(t, f, to, bc, now) for t, f, to, bc in records],
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
