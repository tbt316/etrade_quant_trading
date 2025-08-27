from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Optional

from .paths import DATA_ROOT, ensure_dir

# ---------------------------------------------------------
# Logging setup (file + optional console) with a unified format
# ---------------------------------------------------------

_LOGGER_INITIALIZED = False
_LOG_FILE_NAME = "backtest_debug.log"  # preserve legacy filename


def init_logging(level: int = logging.INFO, *, log_to_console: bool = True, max_bytes: int = 10_000_000, backup_count: int = 5) -> None:
    """Initialize root logger once with a rotating file handler and optional console.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_to_console : bool
        If True, also attach a StreamHandler to stdout.
    max_bytes : int
        Maximum bytes for each rotated file.
    backup_count : int
        Number of rotated file backups to keep.
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        set_level(level)
        return

    ensure_dir(DATA_ROOT)
    log_path = DATA_ROOT / _LOG_FILE_NAME

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # File handler (rotating)
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    # Optional console handler
    if log_to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    _LOGGER_INITIALIZED = True


def set_level(level: int) -> None:
    """Update logging level for all handlers on the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)


@contextmanager
def timed(msg: str, *, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG) -> Iterator[None]:
    """Context manager to time a code block and log the elapsed duration.

    Usage
    -----
    >>> from polygonio.logging_setup import timed
    >>> with timed("load prices"):
    ...     df = get_historical_prices(...)
    """
    log = logger or logging.getLogger(__name__)
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = (perf_counter() - start) * 1000.0  # ms
        log.log(level, f"{msg} finished in {elapsed:.2f} ms")

