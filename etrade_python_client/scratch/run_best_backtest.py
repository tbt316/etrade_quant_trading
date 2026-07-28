"""Quarantined legacy regime-aware backtest entry point."""

DISABLED_REASON = (
    "UNSAFE_REGIME_BACKTEST_DISABLED: this script fitted one global HMM over "
    "the evaluation sample and fed same-day states into historical trades. "
    "Use backtesting/backtest_runner.py with a recorded pre-test calibration "
    "cutoff, causal walk-forward trace, and one-session-lagged regime map."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
