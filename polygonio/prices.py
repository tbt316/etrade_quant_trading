from __future__ import annotations
import time
from datetime import date, datetime, timedelta
from typing import Union

import pandas as pd
import requests

from .config import get_settings
from .paths import (
    get_yf_price_cache_file,
    get_polygon_price_cache_file,
    ensure_dir,
)
from .symbols import to_vendor_ticker


# ---------------------------------------------------------
# Yahoo Finance (chart API) — daily or 1m
# ---------------------------------------------------------

def _to_naive_dt(dt_like: Union[date, datetime]) -> datetime:
    if isinstance(dt_like, date) and not isinstance(dt_like, datetime):
        return datetime.combine(dt_like, datetime.min.time())
    return dt_like  # assume naive


def fetch_yfinance_data(
    ticker: str,
    start_date: Union[date, datetime],
    end_date: Union[date, datetime],
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download close data from Yahoo Finance and return DataFrame with 'date' and 'close'.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    start_date, end_date : date | datetime
        Range of data to fetch (inclusive for start; end treated exclusive by API so we +1d).
    interval : {"1d", "1m"}
        Data interval.
    auto_adjust : bool
        Whether to auto-adjust prices.
    """
    if interval not in {"1d", "1m"}:
        raise ValueError("interval must be '1d' or '1m'")

    yf_ticker = to_vendor_ticker(ticker, vendor="yfinance")
    start_dt = _to_naive_dt(start_date)
    end_dt = _to_naive_dt(end_date)

    # Yahoo's chart API treats period2 as exclusive
    if interval == "1d":
        end_dt = end_dt + timedelta(days=1)
    else:
        end_dt = end_dt + timedelta(seconds=60)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; YahooFinanceClient/1.0)",
            "Accept": "application/json, text/plain, */*",
        }
    )

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        f"?period1={int(start_dt.timestamp())}"
        f"&period2={int(end_dt.timestamp())}"
        f"&interval={interval}"
        f"&events=history"
        f"&adjusted={str(auto_adjust).lower()}"
    )

    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    try:
        result = payload["chart"]["result"][0]
        ts = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Unexpected response structure or no data returned")

    if not ts or closes is None:
        raise ValueError("No data returned from Yahoo Finance")

    idx = pd.to_datetime(ts, unit="s")
    df = pd.DataFrame({"datetime": idx, "close": closes}).dropna()
    df["date"] = df["datetime"].dt.date
    return df[["date", "close"]]


# ---------------------------------------------------------
# Unified price fetcher with CSV caching (Yahoo or Polygon)
# ---------------------------------------------------------

def get_historical_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    vol_lookback: int = 5,
    data_source: str = "yfinance",
) -> pd.DataFrame:
    """Fetch historical prices and compute returns/vol/MA, with CSV caching.

    Behavior matches legacy script:
    - Per-source CSV caches
    - Weekend rollover for "today" and effective_end semantics
    - Polygon pulled in ≤500-bar chunks
    """
    s = get_settings()  # currently unused, kept for future knobs

    # Caching setup
    if data_source == "yfinance":
        yf_ticker = to_vendor_ticker(ticker, vendor="yfinance")
        cache_file = get_yf_price_cache_file(yf_ticker)
    elif data_source == "polygon":
        cache_file = get_polygon_price_cache_file(ticker)
    else:
        raise ValueError("Invalid data_source. Choose 'yfinance' or 'polygon'.")

    ensure_dir(cache_file.parent)

    # Parse requested range
    today = date.today()
    requested_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    requested_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Define last weekday (if weekend, roll back to last Friday)
    if today.weekday() >= 5:  # Sat=5, Sun=6
        last_weekday = today - timedelta(days=(today.weekday() - 4))
    else:
        last_weekday = today

    # If requested end is weekend, cap at last_weekday
    effective_end = min(requested_end, last_weekday)

    # Try loading cache if it sufficiently covers the requested range
    use_cache = False
    if cache_file.exists():
        try:
            file_mod_date = date.fromtimestamp(cache_file.stat().st_mtime)
            cached_df = pd.read_csv(cache_file, parse_dates=["date"])  # 'date' as Timestamp
            cached_start = cached_df["date"].min().date()
            cached_end = cached_df["date"].max().date()
            days_gap = (cached_start - requested_start).days

            use_cache = (
                days_gap <= 7
                and cached_end >= effective_end
                and (file_mod_date >= last_weekday or requested_end < last_weekday)
            )
        except Exception as e:
            print(f"Warning: cache read error for {ticker}: {e}")
            use_cache = False

    df: pd.DataFrame | None = None

    if use_cache:
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=["date"])  # 'date' as Timestamp
            mask = (
                (cached_df["date"].dt.date >= requested_start)
                & (cached_df["date"].dt.date <= requested_end)
            )
            df = cached_df.loc[mask].copy()
        except Exception as e:
            print(f"Warning: cache processing error for {ticker}: {e}")
            df = None

    # Fetch fresh data if cache is unusable
    if df is None or df.empty:
        if data_source == "yfinance":
            print(f"Fetching price data from yfinance for {ticker}")
            fetched = fetch_yfinance_data(
                ticker,
                datetime.combine(requested_start, datetime.min.time()),
                datetime.combine(requested_end, datetime.min.time()),
            )
            fetched.to_csv(cache_file, index=False)
            df = fetched

        elif data_source == "polygon":
            # Polygon aggregates endpoint with ≤500-bar chunks
            from .config import get_settings  # local import to avoid cycles
            api_key = get_settings().polygon_api_key
            max_bars = 500
            chunk_start = requested_start
            all_chunks: list[pd.DataFrame] = []

            while chunk_start <= requested_end:
                chunk_end = min(requested_end, chunk_start + timedelta(days=max_bars - 1))
                s_str = chunk_start.strftime("%Y-%m-%d")
                e_str = chunk_end.strftime("%Y-%m-%d")
                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
                    f"/range/1/day/{s_str}/{e_str}"
                    f"?adjusted=false&sort=asc&limit={max_bars}&apiKey={api_key}"
                )

                print(f"Fetching Polygon data for {ticker}: {s_str} → {e_str}")
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    if "results" in data and data["results"]:
                        tmp = pd.DataFrame(data["results"])  # expects 't' (ms ts) and 'c' (close)
                        tmp["date"] = pd.to_datetime(tmp["t"], unit="ms")
                        tmp = tmp[["date", "c"]].rename(columns={"c": "close"})
                        all_chunks.append(tmp)
                    else:
                        print(f"  ⚠️  No data for {ticker} {s_str}–{e_str}")
                except Exception as e:
                    print(f"Polygon error {ticker} {s_str}–{e_str}: {e}")

                # advance
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.2)  # be gentle on API

            if all_chunks:
                df = (
                    pd.concat(all_chunks, ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="first")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                df.to_csv(cache_file, index=False)
            else:
                df = pd.DataFrame(columns=["date", "close"])  # empty

    # Post-processing (returns/vol/MA)
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["date", "close", "returns", "vol_20", "MA", "ticker"]
        )

    # Ensure 'date' is of date type (not Timestamp)
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.date

    df["returns"] = df["close"].pct_change()
    df["vol_20"] = df["returns"].rolling(window=vol_lookback).std()
    df["MA"] = df["close"].rolling(window=vol_lookback).mean()
    df["ticker"] = ticker

    return df

