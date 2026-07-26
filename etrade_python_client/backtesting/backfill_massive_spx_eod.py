#!/usr/bin/env python3
"""Backfill post-OptionsDX SPX option daily bars from Massive/Polygon."""

import argparse
import asyncio
import calendar
import sqlite3
import ssl
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import aiohttp
import certifi
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.massive_config import API_KEY, BASE_URL
from backtesting.option_data_cache import OptionDataCache


DEFAULT_DB_PATH = PROJECT_ROOT / "backtest_cache" / "option_data.db"
DEFAULT_START_DATE = "2024-01-01"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_starts(start: date, end: date) -> Iterable[date]:
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def month_end(month_start: date) -> date:
    return date(
        month_start.year,
        month_start.month,
        calendar.monthrange(month_start.year, month_start.month)[1],
    )


def iso(value: date) -> str:
    return value.isoformat()


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_option_ticker(ticker: str) -> dict:
    try:
        clean = ticker[2:] if ticker.startswith("O:") else ticker
        split_idx = next(idx for idx, char in enumerate(clean) if char.isdigit())
        underlying = clean[:split_idx]
        rem = clean[split_idx:]
        expiration = f"20{rem[:2]}-{rem[2:4]}-{rem[4:6]}"
        contract_type = "call" if rem[6] == "C" else "put"
        strike = float(rem[7:]) / 1000.0
        return {
            "underlying": underlying,
            "expiration": expiration,
            "contract_type": contract_type,
            "strike": strike,
        }
    except Exception:
        return {}


class MassiveRateLimiter:
    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.last_request = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.min_interval <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            sleep_for = self.min_interval - (now - self.last_request)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self.last_request = time.monotonic()


async def get_json(
    session: aiohttp.ClientSession,
    limiter: MassiveRateLimiter,
    url: str,
    params: dict,
    max_retries: int,
):
    params = dict(params)
    params["apiKey"] = API_KEY
    for attempt in range(max_retries + 1):
        await limiter.wait()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429 and attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt >= max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
    return None


async def fetch_contract_page_set(
    session: aiohttp.ClientSession,
    limiter: MassiveRateLimiter,
    expiration_start: date,
    expiration_end: date,
    contract_type: str,
    max_retries: int,
) -> list[dict]:
    """Fetch one month/type of historical and live SPX contracts."""
    base_params = {
        "underlying_ticker": "SPX",
        "contract_type": contract_type,
        "expiration_date.gte": iso(expiration_start),
        "expiration_date.lte": iso(expiration_end),
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
    }
    variants = [
        {**base_params, "expired": "true"},
        base_params,
    ]

    seen = set()
    contracts = []
    for first_params in variants:
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = dict(first_params)
        while url:
            data = await get_json(session, limiter, url, params, max_retries)
            if not data or "results" not in data:
                break
            for item in data["results"]:
                ticker = item.get("ticker")
                if not ticker or ticker in seen:
                    continue
                parsed = parse_option_ticker(ticker)
                if parsed.get("contract_type") != contract_type:
                    continue
                expiration = parsed.get("expiration")
                if not expiration:
                    continue
                contracts.append(
                    {
                        "option_ticker": ticker,
                        "contract_type": contract_type,
                        "strike": float(item["strike_price"]),
                        "expiration": expiration,
                    }
                )
                seen.add(ticker)
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {}
            else:
                break
    return contracts


def bulk_upsert_contracts(conn: sqlite3.Connection, contracts: list[dict], fetched_at: str):
    rows = [
        (
            "SPX",
            c["expiration"],
            c["contract_type"],
            c["expiration"],
            c["option_ticker"],
            c["strike"],
            fetched_at,
        )
        for c in contracts
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO contracts_cache
            (underlying, expiration, contract_type, as_of_date, option_ticker, strike, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    fetch_rows = sorted({("SPX", c["expiration"], c["expiration"], c["contract_type"], fetched_at) for c in contracts})
    conn.executemany(
        """
        INSERT OR REPLACE INTO fetch_log
            (underlying, pricing_date, expiration, contract_type, data_type, status, fetched_at)
        VALUES (?, ?, ?, ?, 'contracts', 'complete', ?)
        """,
        fetch_rows,
    )
    conn.commit()


def range_is_fetched(conn: sqlite3.Connection, ticker: str, from_date: str, to_date: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM ohlcv_fetch_log
        WHERE option_ticker = ? AND from_date <= ? AND to_date >= ?
        LIMIT 1
        """,
        (ticker, from_date, to_date),
    ).fetchone()
    return row is not None


async def fetch_daily_bars(
    session: aiohttp.ClientSession,
    limiter: MassiveRateLimiter,
    contract: dict,
    from_date: str,
    to_date: str,
    max_retries: int,
) -> tuple[list[tuple], tuple[str, str, str, int]]:
    url = f"{BASE_URL}/v2/aggs/ticker/{contract['option_ticker']}/range/1/day/{from_date}/{to_date}"
    params = {"adjusted": "false", "sort": "asc", "limit": 50000}
    data = await get_json(session, limiter, url, params, max_retries)
    fetched_at = datetime.now().isoformat()
    rows = []
    if data and data.get("results"):
        eastern = pytz.timezone("America/New_York")
        for bar in data["results"]:
            ts_ms = bar.get("t")
            if ts_ms is None:
                continue
            pricing_date = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.utc).astimezone(eastern).strftime("%Y-%m-%d")
            rows.append(
                (
                    "SPX",
                    contract["option_ticker"],
                    contract["contract_type"],
                    contract["strike"],
                    contract["expiration"],
                    pricing_date,
                    safe_float(bar.get("o")),
                    safe_float(bar.get("h")),
                    safe_float(bar.get("l")),
                    safe_float(bar.get("c")),
                    int(bar.get("v") or 0),
                    safe_float(bar.get("vw")),
                    int(bar.get("n") or 0),
                    fetched_at,
                )
            )
    return rows, (contract["option_ticker"], from_date, to_date, len(rows))


def flush_bars(cache: OptionDataCache, results: list[tuple[list[tuple], tuple[str, str, str, int]]]):
    price_rows = []
    range_rows = []
    for rows, range_row in results:
        price_rows.extend(rows)
        range_rows.append(range_row)
    if price_rows:
        cache.bulk_upsert_ohlcv(price_rows)
    if range_rows:
        cache.bulk_mark_ticker_ranges_fetched(range_rows)
    return len(price_rows), len(range_rows)


async def discover_contracts(args) -> list[dict]:
    limiter = MassiveRateLimiter(args.requests_per_second)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=args.max_concurrent)
    all_contracts = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for month in month_starts(args.start_date, args.expiration_end):
            exp_start = max(month, args.start_date)
            exp_end = min(month_end(month), args.expiration_end)
            monthly = []
            for contract_type in ("put", "call"):
                monthly.extend(
                    await fetch_contract_page_set(
                        session,
                        limiter,
                        exp_start,
                        exp_end,
                        contract_type,
                        args.max_retries,
                    )
                )
            all_contracts.extend(monthly)
            print(
                f"[contracts] {exp_start:%Y-%m}: {len(monthly):,} contracts "
                f"({len(all_contracts):,} cumulative)",
                flush=True,
            )
    deduped = {c["option_ticker"]: c for c in all_contracts}
    return sorted(deduped.values(), key=lambda c: (c["expiration"], c["contract_type"], c["strike"]))


def build_fetch_plan(conn: sqlite3.Connection, contracts: list[dict], start_date: date, end_date: date):
    plan = []
    for contract in contracts:
        exp_date = parse_date(contract["expiration"])
        to_date = min(exp_date, end_date)
        if to_date < start_date:
            continue
        from_date = iso(start_date)
        to_text = iso(to_date)
        if range_is_fetched(conn, contract["option_ticker"], from_date, to_text):
            continue
        plan.append((contract, from_date, to_text))
    return plan


async def fetch_plan(args, plan: list[tuple[dict, str, str]], cache: OptionDataCache):
    limiter = MassiveRateLimiter(args.requests_per_second)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=args.max_concurrent)
    total_records = 0
    total_ranges = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for start in range(0, len(plan), args.chunk_size):
            chunk = plan[start : start + args.chunk_size]
            tasks = [
                fetch_daily_bars(session, limiter, contract, from_date, to_date, args.max_retries)
                for contract, from_date, to_date in chunk
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            ok_results = []
            failures = 0
            for result in results:
                if isinstance(result, Exception):
                    failures += 1
                else:
                    ok_results.append(result)
            records, ranges = flush_bars(cache, ok_results)
            total_records += records
            total_ranges += ranges
            print(
                f"[bars] {min(start + len(chunk), len(plan)):,}/{len(plan):,} ranges "
                f"| +{records:,} rows | failures {failures:,} | total rows {total_records:,}",
                flush=True,
            )
            if failures:
                raise RuntimeError(f"{failures} requests failed in chunk starting at {start}")
    return total_records, total_ranges


def default_end_date(db_path: Path) -> date:
    cache = OptionDataCache(str(db_path))
    try:
        max_date = cache.get_max_pricing_date("SPX")
        if max_date:
            return parse_date(max_date)
    finally:
        cache.close()
    return datetime.now().date()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--start-date", type=parse_date, default=parse_date(DEFAULT_START_DATE))
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument(
        "--expiration-end",
        type=parse_date,
        default=None,
        help="Latest expiration to discover. Defaults to end-date + expiration-lookahead-days.",
    )
    parser.add_argument("--expiration-lookahead-days", type=int, default=120)
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--requests-per-second", type=float, default=45.0)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--limit-ranges", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_date is None:
        args.end_date = default_end_date(args.db_path)
    if args.expiration_end is None:
        args.expiration_end = args.end_date + timedelta(days=args.expiration_lookahead_days)
    if args.end_date < args.start_date:
        raise ValueError("end-date must be on or after start-date")
    if args.expiration_end < args.start_date:
        raise ValueError("expiration-end must be on or after start-date")

    print(
        f"[config] pricing {args.start_date} -> {args.end_date}; "
        f"expirations through {args.expiration_end}; db={args.db_path}",
        flush=True,
    )
    contracts = asyncio.run(discover_contracts(args))
    print(f"[contracts] discovered {len(contracts):,} unique contracts", flush=True)

    conn = sqlite3.connect(args.db_path, timeout=120)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        bulk_upsert_contracts(conn, contracts, datetime.now().isoformat())
        plan = build_fetch_plan(conn, contracts, args.start_date, args.end_date)
    finally:
        conn.close()

    if args.limit_ranges:
        plan = plan[: args.limit_ranges]
    print(f"[plan] {len(plan):,} ticker ranges need daily-bar fetches", flush=True)
    if args.discover_only:
        return

    cache = OptionDataCache(str(args.db_path))
    try:
        records, ranges = asyncio.run(fetch_plan(args, plan, cache))
    finally:
        cache.close()
    print(f"[done] fetched {records:,} OHLCV rows across {ranges:,} ticker ranges", flush=True)


if __name__ == "__main__":
    main()
