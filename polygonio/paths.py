# polygonio/paths.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Root project directory (two parents up from this file: polygonio/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Cache and output directories
CACHE_DIR = (ROOT_DIR.parent / "polygon_api_option_data").resolve()
MONTHLY_BACKTEST_DIR = ROOT_DIR / "monthly_backtest_data"
YF_PRICE_DIR = ROOT_DIR / "yfinance" / "prices"
POLY_PRICE_DIR = ROOT_DIR / "polygon" / "prices"
YF_EARNINGS_DIR = ROOT_DIR / "yfinance" / "earnings"


def ensure_dir(p: str | Path) -> None:
    """Ensure directory exists."""
    Path(p).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Option chain/price cache files
# -----------------------------

def get_price_cache_file(ticker: str) -> str:
    """
    Pickle path for option price cache of a ticker.
    Example: polygon_api_option_data/AAPL_stored_option_price.pkl
    """
    ensure_dir(CACHE_DIR)
    return str(CACHE_DIR / f"{ticker.upper()}_stored_option_price.pkl")


def get_chain_cache_file(ticker: str) -> str:
    """
    Pickle path for option chain cache of a ticker.
    Example: polygon_api_option_data/AAPL_stored_option_chain.pkl
    """
    ensure_dir(CACHE_DIR)
    return str(CACHE_DIR / f"{ticker.upper()}_stored_option_chain.pkl")


# -----------------------------
# Backtest results
# -----------------------------

def get_monthly_backtest_file(ticker: str, global_start_date: str, global_end_date: str) -> str:
    """
    Unique pickle file path for storing monthly backtest results.
    """
    ensure_dir(MONTHLY_BACKTEST_DIR)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monthly_{ticker}_{global_start_date}_to_{global_end_date}_{timestamp_str}.pkl"
    return str(MONTHLY_BACKTEST_DIR / filename)


# -----------------------------
# Historical prices
# -----------------------------

def yf_price_cache_file(ticker: str) -> str:
    """
    CSV cache file for Yahoo Finance daily prices.
    """
    ensure_dir(YF_PRICE_DIR)
    return str(YF_PRICE_DIR / f"{ticker.upper()}_prices.csv")


def polygon_price_cache_file(ticker: str) -> str:
    """
    CSV cache file for Polygon daily aggregated prices.
    """
    ensure_dir(POLY_PRICE_DIR)
    return str(POLY_PRICE_DIR / f"{ticker.upper()}_prices.csv")


# -----------------------------
# Earnings
# -----------------------------

def yf_earnings_cache_file(ticker: str) -> str:
    """
    CSV cache file for Yahoo Finance earnings dates.
    """
    ensure_dir(YF_EARNINGS_DIR)
    return str(YF_EARNINGS_DIR / f"{ticker.upper()}_earnings.csv")