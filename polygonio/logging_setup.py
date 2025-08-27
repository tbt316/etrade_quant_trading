# polygonio/logging_setup.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def init_logging(level: str = "INFO", logfile: str = "backtest_debug.log") -> None:
    # ensure the directory for the logfile exists
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    # root logger level
    root = logging.getLogger()
    root.setLevel(level.upper())

    # clear existing handlers to avoid duplicates on repeated runs
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # rotating file
    fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    root.addHandler(fh)