"""Compatibility wrapper for the shard-aware option cache."""

from backtesting.sharded_option_data_cache import (
    MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT,
    ContractReferenceCacheError,
    OptionDataCache,
    _db_default_path,
)


DEFAULT_DB_PATH = _db_default_path()
