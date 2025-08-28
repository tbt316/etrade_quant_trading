# polygonio/prices.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, date
from typing import Union, Optional, List

import pandas as pd
import requests

from .symbols import to_vendor_ticker
from .paths import yf_price_cache_file, polygon_price_cache_file
from .config import get_settings


def fetch_yfinance_data(
    ticker: str,
    start_date: Union[date, datetime],
    end_date: Union[date, datetime],
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download close data from Yahoo Finance and return a DataFrame with 'date' and 'close' columns.

    Parameters
    ----------
    ticker : str
        Vendor-normalized Yahoo symbol (e.g., BRK-B, ^VIX).
    start_date, end_date : date | datetime
        The date range for the data (inclusive).
    interval : {"1d", "1m"}
        The data interval ('1d' for daily, '1m' for 1-minute).
    auto_adjust : bool
        Whether to auto-adjust prices.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'close']
    """
    if interval not in {"1d", "1m"}:
        raise ValueError("interval must be '1d' or '1m'")

    def _to_naive_dt(dt: Union[date, datetime]) -> datetime:
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return datetime.combine(dt, datetime.min.time())
        return dt

    start_dt = _to_naive_dt(start_date)
    end_dt = _to_naive_dt(end_date)

    # For intraday, extend end_dt to now if start == end
    if interval == "1m" and start_dt.date() == end_dt.date():
        end_dt = datetime.now()

    # Yahoo's chart API treats period2 as exclusive
    if interval == "1d":
        end_dt += timedelta(days=1)
    else:  # "1m"
        end_dt += timedelta(seconds=60)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; YahooFinanceClient/1.0)",
            "Accept": "application/json, text/plain, */*",
        }
    )

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={int(start_dt.timestamp())}"
        f"&period2={int(end_dt.timestamp())}"
        f"&interval={interval}"
        f"&events=history"
        f"&adjusted={str(auto_adjust).lower()}"
    )

    resp = session.get(url, timeout=15)
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


def get_historical_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    vol_lookback: int = 5,
    data_source: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch historical prices using either Yahoo Finance (yfinance) or Polygon.io and
    return a DataFrame with calculated metrics.

    Returns columns:
      ['date', 'close', 'returns', 'vol_20', 'MA', 'ticker']
    """

    # --- Cache file selection (via helpers) ---
    if data_source == "yfinance":
        yf_ticker = to_vendor_ticker(ticker, "yfinance")
        cache_file = yf_price_cache_file(yf_ticker)
    elif data_source == "polygon":
        cache_file = polygon_price_cache_file(ticker)
    else:
        raise ValueError("Invalid data_source. Choose 'yfinance' or 'polygon'.")

    # --- Date parsing & effective end (weekend guard) ---
    today = date.today()
    requested_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    requested_end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # If today is weekend, roll back to last Friday
    if today.weekday() >= 5:  # 5=Sat, 6=Sun
        last_weekday = today - timedelta(days=(today.weekday() - 4))
    else:
        last_weekday = today

    # If end date is weekend, treat as last_weekday
    effective_end = min(requested_end, last_weekday)

    # --- Try to use cache if it covers the window ---
    use_cache = False
    if os.path.exists(cache_file):
        file_mod_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=["date"])
            cached_start = cached_df["date"].min().date() if not cached_df.empty else None
            cached_end = cached_df["date"].max().date() if not cached_df.empty else None

            if cached_start and cached_end:
                days_gap = (cached_start - requested_start).days
                use_cache = (
                    days_gap <= 7
                    and cached_end >= effective_end
                    and (file_mod_date >= last_weekday or requested_end < last_weekday)
                )
        except Exception:
            use_cache = False

    df: Optional[pd.DataFrame] = None
    if use_cache:
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=["date"])
            df = cached_df.loc[
                (cached_df["date"].dt.date >= requested_start)
                & (cached_df["date"].dt.date <= requested_end)
            ].copy()
        except Exception:
            df = None
            use_cache = False

    # --- Fetch if cache is unusable ---
    if not use_cache or df is None or df.empty:
        if data_source == "yfinance":
            # Fetch from Yahoo; save to cache
            df = fetch_yfinance_data(
                to_vendor_ticker(ticker, "yfinance"),
                requested_start,
                requested_end,
                interval="1d",
                auto_adjust=False,
            )
            # Ensure parent directory exists (helpers did it, but guard again)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_csv(cache_file, index=False)

        elif data_source == "polygon":
            # Chunked Polygon fetch (<= 500 bars per call)
            s = get_settings()  # only used here for API key
            api_key = s.polygon_api_key
            max_bars = 500
            chunk_start = requested_start
            chunks: List[pd.DataFrame] = []

            while chunk_start <= requested_end:
                chunk_end = min(requested_end, chunk_start + timedelta(days=max_bars - 1))
                s_str = chunk_start.strftime("%Y-%m-%d")
                e_str = chunk_end.strftime("%Y-%m-%d")

                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
                    f"/range/1/day/{s_str}/{e_str}"
                    f"?adjusted=false&sort=asc&limit={max_bars}&apiKey={api_key}"
                )

                try:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                    if "results" in data and data["results"]:
                        tmp = pd.DataFrame(data["results"])
                        tmp["date"] = pd.to_datetime(tmp["t"], unit="ms").dt.date
                        tmp = tmp[["date", "c"]].rename(columns={"c": "close"})
                        chunks.append(tmp)
                except Exception:
                    # gentle on the API / transient failures
                    pass

                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.2)  # be nice to API

            if chunks:
                df = (
                    pd.concat(chunks, ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="first")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_csv(cache_file, index=False)
            else:
                df = pd.DataFrame(columns=["date", "close"])

    # --- Post-processing (same as original) ---
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "returns", "vol_20", "MA", "ticker"])

    # Ensure dtypes
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"]).dt.date

    df["returns"] = pd.Series(df["close"]).pct_change()
    df["vol_20"] = df["returns"].rolling(window=vol_lookback).std()
    df["MA"] = pd.Series(df["close"]).rolling(window=vol_lookback).mean()
    df["ticker"] = ticker

    return df