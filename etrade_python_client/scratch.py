"""Quarantined top-level global-HMM scratch entry point."""

DISABLED_REASON = (
    "UNSAFE_REGIME_RESEARCH_DISABLED: declare calibration and test ranges, "
    "then call train_regime_hmm with an explicit pre-test fit_end. Use "
    "market_regime_detect.py for the maintained causal review workflow."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
