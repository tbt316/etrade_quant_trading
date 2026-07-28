"""Quarantined panic-transition roll comparison entry point."""

DISABLED_REASON = (
    "UNSAFE_REGIME_BACKTEST_DISABLED: this experiment used a final stress "
    "overlay to open replacement risk during panic transitions. Final-overlay "
    "signals may only veto entries or reduce existing risk. Use "
    "backtesting/backtest_runner.py with BacktestRegimeProtocol and typed "
    "RegimeDecisionEvidence for maintained regime-aware research."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
