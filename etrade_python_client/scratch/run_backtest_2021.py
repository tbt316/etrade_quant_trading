"""Quarantined legacy regime-aware backtest entry point."""

DISABLED_REASON = (
    "UNSAFE_REGIME_BACKTEST_DISABLED: this script fitted one global HMM over "
    "the 2021 evaluation sample and used unlagged states. Use "
    "backtesting/backtest_runner.py with an explicit pre-test calibration "
    "cutoff and a one-session-lagged causal regime map."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
