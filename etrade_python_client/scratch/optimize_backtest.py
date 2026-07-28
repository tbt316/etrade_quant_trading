"""Quarantined legacy regime-aware backtest optimizer."""

DISABLED_REASON = (
    "UNSAFE_REGIME_BACKTEST_DISABLED: this optimizer selected parameters "
    "while using a globally fitted HMM and unlagged evaluation-period states. "
    "Migrate experiments to backtesting/backtest_runner.py with explicit "
    "calibration/test ranges, walk-forward inference, and lagged regime maps."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
