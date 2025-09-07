from __future__ import annotations
import pickle
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Dict

from .paths import get_price_cache_file, get_chain_cache_file, ensure_dir
import os

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

def merge_nested_dicts_with_change(dst: dict, src: dict) -> bool:
    """
    Like merge_nested_dicts, but returns True if *dst* changed as a result of the merge.
    Used to avoid unnecessary pickle writes when nothing is new.
    """
    changed = False
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            if merge_nested_dicts_with_change(dst[k], v):
                changed = True
        else:
            if k not in dst or dst.get(k) != v:
                dst[k] = v
                changed = True
    return changed

# ---------------------------------------------------------
# Load / Save with merge semantics
# ---------------------------------------------------------

def load_stored_option_data(ticker: str, cache_dir: Path | None = None) -> Dict[str, Any]:
    """Load cached price/chain pickles for *ticker* (if present) and merge into memory.
    If *cache_dir* is provided, use that directory instead of the default cache path.
    Returns the most recent pricing-date dict for convenience (or empty dict).
    """
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    if cache_dir is not None:
        ensure_dir(cache_dir)
        price_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_price.pkl"
        chain_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_chain.pkl"
    else:
        price_cache_file: Path = Path(get_price_cache_file(t))
        chain_cache_file: Path = Path(get_chain_cache_file(t))

    # Price cache
    if price_cache_file.exists():
        try:
            if price_cache_file.stat().st_size > 0:
                with price_cache_file.open("rb") as f:
                    on_disk: Dict[str, Any] = pickle.load(f)
                merge_nested_dicts(stored_option_price[t], on_disk)
                print(f"[DEBUG] Loaded price data from: {price_cache_file}")
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
                print(f"[DEBUG] Loaded option data from: {chain_cache_file}")

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

def option_data_needs_update(ticker: str, cache_dir: Path | None = None) -> bool:
    """
    Determine whether in-memory option data differs from what is stored on disk.
    This allows callers to decide if saving is necessary without performing the
    costly save operation.
    """
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    if cache_dir is not None:
        price_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_price.pkl"
        chain_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_chain.pkl"
    else:
        price_cache_file = Path(get_price_cache_file(t))
        chain_cache_file = Path(get_chain_cache_file(t))

    existing_price: Dict[str, Any] = {}
    existing_chain: Dict[str, Any] = {}

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

    price_changed = merge_nested_dicts_with_change(existing_price, stored_option_price[t])
    chain_changed = merge_nested_dicts_with_change(existing_chain, stored_option_chain[t])

    return price_changed or chain_changed


def save_stored_option_data(ticker: str, cache_dir: Path | None = None) -> None:
    """Persist in-memory option data to disk."""
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    if cache_dir is not None:
        ensure_dir(cache_dir)
        price_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_price.pkl"
        chain_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_chain.pkl"
    else:
        price_cache_file = Path(get_price_cache_file(t))
        chain_cache_file = Path(get_chain_cache_file(t))

    ensure_dir(price_cache_file.parent)
    ensure_dir(chain_cache_file.parent)

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

        # Merge memory into existing and write back
        merge_nested_dicts(existing_price, stored_option_price[t])
        merge_nested_dicts(existing_chain, stored_option_chain[t])

        with price_cache_file.open("wb") as f:
            pickle.dump(existing_price, f, protocol=pickle.HIGHEST_PROTOCOL)
        with chain_cache_file.open("wb") as f:
            pickle.dump(existing_chain, f, protocol=pickle.HIGHEST_PROTOCOL)

