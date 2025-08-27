from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------
# Paths / Filesystem helpers
# ---------------------------------------------------------

# Project/data roots (customize via env if desired)
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data"

# Legacy-compatible cache folder names (match existing script semantics)
POLYGON_OPTION_CACHE_DIRNAME = "polygon_api_option_data"  # for stored_option_price / stored_option_chain pickles
MONTHLY_BACKTEST_DIRNAME = "monthly_backtest_data"

# Vendor-specific subtrees for CSV caches
YF_DIR = DATA_ROOT / "yfinance"
YF_PRICES_DIR = YF_DIR / "prices"
YF_EARNINGS_DIR = YF_DIR / "earnings"

POLY_DIR = DATA_ROOT / "polygon"
POLY_PRICES_DIR = POLY_DIR / "prices"

# Ensure base dirs exist eagerly (safe if they already exist)
for p in [DATA_ROOT, YF_PRICES_DIR, YF_EARNINGS_DIR, POLY_PRICES_DIR, DATA_ROOT / POLYGON_OPTION_CACHE_DIRNAME, DATA_ROOT / MONTHLY_BACKTEST_DIRNAME]:
    p.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> Path:
    """Create *path* (dir) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------
# Option price / chain pickle caches (per-ticker)
# ---------------------------------------------------------

def get_price_cache_file(ticker: str) -> Path:
    """Pickle file for stored_option_price of *ticker*.
    Example: data/polygon_api_option_data/AAPL_stored_option_price.pkl
    """
    t = ticker.upper()
    return ensure_dir(DATA_ROOT / POLYGON_OPTION_CACHE_DIRNAME) / f"{t}_stored_option_price.pkl"


def get_chain_cache_file(ticker: str) -> Path:
    """Pickle file for stored_option_chain of *ticker*.
    Example: data/polygon_api_option_data/AAPL_stored_option_chain.pkl
    """
    t = ticker.upper()
    return ensure_dir(DATA_ROOT / POLYGON_OPTION_CACHE_DIRNAME) / f"{t}_stored_option_chain.pkl"


# ---------------------------------------------------------
# Underlying price CSV caches (Yahoo / Polygon)
# ---------------------------------------------------------

def get_yf_price_cache_file(yf_ticker: str) -> Path:
    """CSV file for Yahoo Finance price history cache.
    Example: data/yfinance/prices/AAPL_prices.csv
    """
    t = yf_ticker.upper()
    return ensure_dir(YF_PRICES_DIR) / f"{t}_prices.csv"


def get_polygon_price_cache_file(ticker: str) -> Path:
    """CSV file for Polygon price history cache.
    Example: data/polygon/prices/AAPL_prices.csv
    """
    t = ticker.upper()
    return ensure_dir(POLY_PRICES_DIR) / f"{t}_prices.csv"


# ---------------------------------------------------------
# Earnings CSV cache (Yahoo)
# ---------------------------------------------------------

def get_earnings_cache_file(yf_ticker: str) -> Path:
    """CSV file for cached earnings dates for *yf_ticker*.
    Example: data/yfinance/earnings/AAPL_earnings.csv
    """
    t = yf_ticker.upper()
    return ensure_dir(YF_EARNINGS_DIR) / f"{t}_earnings.csv"


# ---------------------------------------------------------
# Monthly backtest results
# ---------------------------------------------------------

def get_monthly_backtest_file(ticker: str, global_start_date: str, global_end_date: str, *, timestamp: Optional[datetime] = None) -> Path:
    """Return pkl filepath for storing monthly backtest results.
    Embeds timestamp + params in the filename for uniqueness.
    Example: data/monthly_backtest_data/monthly_AAPL_2022-01-01_to_2022-12-31_20250826_235959.pkl
    """
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    fname = f"monthly_{ticker}_{global_start_date}_to_{global_end_date}_{ts}.pkl"
    return ensure_dir(DATA_ROOT / MONTHLY_BACKTEST_DIRNAME) / fname

