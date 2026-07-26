"""Compatibility wrapper for the shard-aware option cache."""

from backtesting.sharded_option_data_cache import OptionDataCache, _db_default_path


DEFAULT_DB_PATH = _db_default_path()

