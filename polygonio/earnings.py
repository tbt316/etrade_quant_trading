from __future__ import annotations
from datetime import date, datetime
from typing import Set

import pandas as pd
import yfinance as yf

from .paths import yf_earnings_cache_file, ensure_dir
from .symbols import to_vendor_ticker


# ---------------------------------------------------------
# Earnings date cache (Yahoo Finance)
# ---------------------------------------------------------

def get_earnings_dates(ticker: str, start_date: str, end_date: str) -> Set[date]:
    """Return all known earnings dates for *ticker* within [start_date, end_date].

    Behavior mirrors the legacy implementation:
    - Cache lives under data/yfinance/earnings/{TICKER}_earnings.csv
    - Cache is considered fresh only if the file was modified *today*
    - If Yahoo returns no data, write a sentinel row with `no_earnings_flag`=True
    - On exceptions, write an `error_flag` row with the message for diagnostics
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    yf_ticker = to_vendor_ticker(ticker, vendor="yfinance")

    cache_file = yf_earnings_cache_file(yf_ticker)
    ensure_dir(cache_file.parent)

    today = date.today()

    # ----------------------------
    # Try using same-day cache
    # ----------------------------
    use_cache = False
    if cache_file.exists():
        try:
            file_mod_date = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
            use_cache = (file_mod_date == today)
            if not use_cache:
                print(
                    f"Warning: Earnings cache {cache_file} outdated; mod={file_mod_date}, today={today}"
                )
        except Exception as e:
            print(f"Warning: Failed to stat cache {cache_file}: {e}")

    if use_cache:
        try:
            cached = pd.read_csv(cache_file, parse_dates=["earnings_date"], infer_datetime_format=True)
            # no_earnings sentinel
            if (
                "no_earnings_flag" in cached.columns
                and len(cached) == 1
                and bool(cached.iloc[0].get("no_earnings_flag", False))
            ):
                return set()

            out: Set[date] = set()
            for _, row in cached.iterrows():
                dt_only = row["earnings_date"].date() if hasattr(row["earnings_date"], "date") else pd.to_datetime(row["earnings_date"]).date()
                if start_dt <= dt_only <= end_dt:
                    out.add(dt_only)
            return out
        except Exception as e:
            print(f"Warning: Error reading cached earnings for {ticker}: {e}")
            # fall through to live fetch

    # ----------------------------
    # Live fetch from Yahoo Finance
    # ----------------------------
    out: Set[date] = set()
    try:
        print(f"Fetching earnings dates from yfinance for {ticker}")
        t = yf.Ticker(yf_ticker)
        df = t.get_earnings_dates(limit=50)

        if df is not None and not df.empty:
            # Persist full set for cache; filter for return
            all_dates = [pd.to_datetime(idx).date() for idx in pd.to_datetime(df.index)]
            # Write cache with a single 'earnings_date' column
            cache_df = pd.DataFrame({"earnings_date": all_dates})
            cache_df.to_csv(cache_file, index=False)

            for d in all_dates:
                if start_dt <= d <= end_dt:
                    out.add(d)
            return out
        else:
            # No data → write sentinel so we don't hammer the API repeatedly
            sent = pd.DataFrame(
                {
                    "earnings_date": [today],
                    "no_earnings_flag": [True],
                    "checked_until": [today.strftime("%Y-%m-%d")],
                }
            )
            sent.to_csv(cache_file, index=False)
            return set()

    except Exception as e:
        # On error, write an error marker for diagnostics and return empty set
        err_df = pd.DataFrame(
            {
                "earnings_date": [today],
                "error_flag": [True],
                "error_message": [str(e)],
                "checked_until": [today.strftime("%Y-%m-%d")],
            }
        )
        err_df.to_csv(cache_file, index=False)
        print(f"Warning: Unable to fetch earnings dates for {ticker}: {e}")
        return set()

