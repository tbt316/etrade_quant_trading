from __future__ import annotations
import pickle
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Dict

from .paths import get_price_cache_file, get_chain_cache_file, ensure_dir, DATA_ROOT, POLYGON_OPTION_CACHE_DIRNAME

# ---------------------------------------------------------
# In-memory singletons (shared process-wide)
# ---------------------------------------------------------

stored_option_price: Dict[str, Dict[str, Any]] = {}
stored_option_chain: Dict[str, Dict[str, Any]] = {}

# Global file lock for cross-process safety (for ProcessPoolExecutor)
file_lock: Lock = Lock()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _ensure_ticker_slots(ticker: str) -> None:
    t = ticker.upper()
    if t not in stored_option_price:
        stored_option_price[t] = {}
    if t not in stored_option_chain:
        stored_option_chain[t] = {}


def merge_nested_dicts(dst: Dict, src: Dict) -> None:
    """Recursively merge *src* into *dst* without losing existing sub-dicts.
    If both sides are dicts, merge recursively; otherwise *src* overwrites *dst*.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge_nested_dicts(dst[k], v)
        else:
            dst[k] = v


# ---------------------------------------------------------
# Load / Save with merge semantics
# ---------------------------------------------------------

def load_stored_option_data(ticker: str) -> Dict[str, Any]:
    """Load cached price/chain pickles for *ticker* (if present) and merge into memory.
    Returns the most recent pricing-date dict for convenience (or empty dict).
    """
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    price_cache_file: Path = get_price_cache_file(t)
    chain_cache_file: Path = get_chain_cache_file(t)

    # Price cache
    if price_cache_file.exists():
        try:
            if price_cache_file.stat().st_size > 0:
                with price_cache_file.open("rb") as f:
                    on_disk: Dict[str, Any] = pickle.load(f)
                merge_nested_dicts(stored_option_price[t], on_disk)
            else:
                print(f"Warning: {price_cache_file} is empty; skipping.")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Failed to load {price_cache_file} ({e}); treating as empty.")

    # Chain cache
    if chain_cache_file.exists():
        try:
            if chain_cache_file.stat().st_size > 0:
                with chain_cache_file.open("rb") as f:
                    on_disk_chain: Dict[str, Any] = pickle.load(f)
                merge_nested_dicts(stored_option_chain[t], on_disk_chain)
            else:
                print(f"Warning: {chain_cache_file} is empty; skipping.")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Failed to load {chain_cache_file} ({e}); treating as empty.")

    # Return last pricing-date dict for convenience
    price_by_date = stored_option_price.get(t, {})
    if price_by_date:
        last_date = sorted(price_by_date.keys())[-1]
        return price_by_date.get(last_date, {})
    return {}


def save_stored_option_data(ticker: str) -> None:
    """Merge in-memory dicts with any existing on-disk pickles and write back atomically."""
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    price_cache_file: Path = get_price_cache_file(t)
    chain_cache_file: Path = get_chain_cache_file(t)

    # Ensure parent dir exists
    ensure_dir(DATA_ROOT / POLYGON_OPTION_CACHE_DIRNAME)

    # Load existing pickles (under lock) to avoid overwriting concurrent updates
    existing_price: Dict[str, Any] = {}
    existing_chain: Dict[str, Any] = {}

    with file_lock:
        if price_cache_file.exists() and price_cache_file.stat().st_size > 0:
            try:
                with price_cache_file.open("rb") as f:
                    existing_price = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load {price_cache_file} ({e}); using empty dict.")
        if chain_cache_file.exists() and chain_cache_file.stat().st_size > 0:
            try:
                with chain_cache_file.open("rb") as f:
                    existing_chain = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Warning: Failed to load {chain_cache_file} ({e}); using empty dict.")

        # Merge memory → existing
        merge_nested_dicts(existing_price, stored_option_price[t])
        merge_nested_dicts(existing_chain, stored_option_chain[t])

        # Write back (highest protocol)
        with price_cache_file.open("wb") as f:
            pickle.dump(existing_price, f, protocol=pickle.HIGHEST_PROTOCOL)
        with chain_cache_file.open("wb") as f:
            pickle.dump(existing_chain, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Sanity checks
        if price_cache_file.stat().st_size == 0:
            print(f"Error: {price_cache_file} is empty after write!")
        if chain_cache_file.stat().st_size == 0:
            print(f"Error: {chain_cache_file} is empty after write!")

