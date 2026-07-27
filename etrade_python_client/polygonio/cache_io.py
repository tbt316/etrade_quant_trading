from __future__ import annotations
import json
import pickle
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import (
    get_price_cache_file,
    get_chain_cache_file,
    ensure_dir,
    risk_free_cache_file,
)

# ---------------------------------------------------------
# In-memory singletons (shared process-wide)
# ---------------------------------------------------------

stored_option_price: Dict[str, Dict[str, Any]] = {}
stored_option_chain: Dict[str, Dict[str, Any]] = {}
# Track how many option records have been added or changed since the last save
unsaved_option_entries: Dict[str, int] = {}
risk_free_rate_cache: Dict[str, Dict[str, float]] = {}
_risk_free_cache_loaded = False

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
    if t not in unsaved_option_entries:
        unsaved_option_entries[t] = 0


def _risk_free_cache_path() -> Path:
    return Path(risk_free_cache_file())


def _load_risk_free_cache() -> None:
    global _risk_free_cache_loaded
    if _risk_free_cache_loaded:
        return
    cache_path = _risk_free_cache_path()
    risk_free_rate_cache.clear()
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for sym, mapping in raw.items():
                    if not isinstance(mapping, dict):
                        continue
                    normalized_sym = str(sym).upper()
                    normalized_mapping: Dict[str, float] = {}
                    for dstr, val in mapping.items():
                        try:
                            normalized_mapping[str(dstr)] = float(val)
                        except (TypeError, ValueError):
                            continue
                    if normalized_mapping:
                        risk_free_rate_cache[normalized_sym] = normalized_mapping
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"Warning: Failed to load risk-free cache {cache_path} ({exc}); using empty dict."
            )
    _risk_free_cache_loaded = True


def _save_risk_free_cache() -> None:
    cache_path = _risk_free_cache_path()
    ensure_dir(cache_path.parent)
    serializable = {
        sym: {date_key: float(rate) for date_key, rate in mapping.items()}
        for sym, mapping in risk_free_rate_cache.items()
    }
    with file_lock:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, sort_keys=True)


def get_cached_risk_free_rate(symbol: str, date_str: str) -> Optional[float]:
    if not symbol or not date_str:
        return None
    _load_risk_free_cache()
    sym = str(symbol).upper()
    date_key = str(date_str)
    try:
        value = risk_free_rate_cache.get(sym, {}).get(date_key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def cache_risk_free_rate(symbol: str, date_str: str, rate: float) -> None:
    if not symbol or not date_str:
        return
    try:
        normalized_rate = float(rate)
    except (TypeError, ValueError):
        return
    _load_risk_free_cache()
    sym = str(symbol).upper()
    date_key = str(date_str)
    bucket = risk_free_rate_cache.setdefault(sym, {})
    if bucket.get(date_key) == normalized_rate:
        return
    bucket[date_key] = normalized_rate
    _save_risk_free_cache()


def cache_risk_free_rates(symbol: str, rates: Dict[str, float]) -> None:
    """Store multiple risk-free rates for *symbol* in a single pass."""
    if not symbol or not rates:
        return
    _load_risk_free_cache()
    sym = str(symbol).upper()
    bucket = risk_free_rate_cache.setdefault(sym, {})
    updated = False
    for date_key, rate in rates.items():
        if not date_key:
            continue
        try:
            normalized_rate = float(rate)
        except (TypeError, ValueError):
            continue
        normalized_rate = max(0.0, normalized_rate)
        normalized_key = str(date_key)
        if bucket.get(normalized_key) == normalized_rate:
            continue
        bucket[normalized_key] = normalized_rate
        updated = True
    if updated:
        _save_risk_free_cache()


def merge_nested_dicts(dst: Dict, src: Dict) -> None:
    """Recursively merge *src* into *dst* without losing existing sub-dicts.
    If both sides are dicts, merge recursively; otherwise *src* overwrites *dst*.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge_nested_dicts(dst[k], v)
        else:
            dst[k] = v


def _count_leaves(d: Dict) -> int:
    """Count the number of non-dict leaf nodes in *d*."""
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += _count_leaves(v)
        else:
            total += 1
    return total


def merge_nested_dicts_with_count(dst: Dict, src: Dict) -> int:
    """Merge *src* into *dst* and return number of leaf entries added or changed."""
    count = 0
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            count += merge_nested_dicts_with_count(dst[k], v)
        else:
            if k not in dst or dst.get(k) != v:
                dst[k] = v
                if isinstance(v, dict):
                    count += _count_leaves(v)
                else:
                    count += 1
    return count


def record_unsaved_option_entries(ticker: str, count: int) -> None:
    """Increment unsaved entry counter for *ticker* by *count*."""
    if count <= 0:
        return
    t = ticker.upper()
    unsaved_option_entries[t] = unsaved_option_entries.get(t, 0) + count

# ---------------------------------------------------------
# Load / Save with merge semantics
# ---------------------------------------------------------

def _suffix_for_strike_type(strike_type: Optional[str]) -> str:
    s = (strike_type or "").strip().lower()
    if s in {"atm", "otm"}:
        return f"_{s}"
    return ""


def load_stored_option_data(
    ticker: str,
    cache_dir: Path | None = None,
    *,
    strike_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Load cached price/chain pickles for *ticker* (if present) and merge into memory.
    If *cache_dir* is provided, use that directory instead of the default cache path.
    Returns the most recent pricing-date dict for convenience (or empty dict).
    """
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    suffix = _suffix_for_strike_type(strike_type)

    if cache_dir is not None:
        ensure_dir(cache_dir)
        price_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_price{suffix}.pkl"
        chain_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_chain{suffix}.pkl"
    else:
        # Default cache locations; append suffix if provided
        price_cache = Path(get_price_cache_file(t))
        chain_cache = Path(get_chain_cache_file(t))
        if suffix:
            price_cache = price_cache.with_name(price_cache.stem + suffix + price_cache.suffix)
            chain_cache = chain_cache.with_name(chain_cache.stem + suffix + chain_cache.suffix)
        price_cache_file = price_cache
        chain_cache_file = chain_cache

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


def option_data_unsaved_count(ticker: str, cache_dir: Path | None = None) -> int:  # cache_dir unused
    """Return number of option entries added/updated since last save."""
    _ensure_ticker_slots(ticker)
    return unsaved_option_entries.get(ticker.upper(), 0)


def save_stored_option_data(
    ticker: str,
    cache_dir: Path | None = None,
    *,
    strike_type: Optional[str] = None,
) -> None:
    """Persist in-memory option data to disk."""
    _ensure_ticker_slots(ticker)
    t = ticker.upper()

    suffix = _suffix_for_strike_type(strike_type)

    if cache_dir is not None:
        ensure_dir(cache_dir)
        price_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_price{suffix}.pkl"
        chain_cache_file: Path = Path(cache_dir) / f"{t}_stored_option_chain{suffix}.pkl"
    else:
        price_cache_file = Path(get_price_cache_file(t))
        chain_cache_file = Path(get_chain_cache_file(t))
        if suffix:
            price_cache_file = price_cache_file.with_name(
                price_cache_file.stem + suffix + price_cache_file.suffix
            )
            chain_cache_file = chain_cache_file.with_name(
                chain_cache_file.stem + suffix + chain_cache_file.suffix
            )

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
        unsaved_option_entries[t] = 0
