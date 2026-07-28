"""Quarantined legacy regime-aware backtest comparison."""

DISABLED_REASON = (
    "UNSAFE_REGIME_BACKTEST_DISABLED: this comparison used one full-sample "
    "HMM and same-day regime states in performance claims. Use "
    "backtesting/backtest_runner.py with recorded causal calibration, "
    "walk-forward inference, and a one-session-lagged regime map."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
