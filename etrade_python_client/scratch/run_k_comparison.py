"""Quarantined legacy HMM K-performance comparison."""

DISABLED_REASON = (
    "UNSAFE_REGIME_RESEARCH_DISABLED: this script selected K after global "
    "feature preparation and evaluated forward labels without a fully "
    "reproducible causal manifest. Rebuild it with a pre-test feature fit "
    "cutoff, strictly resolved targets, and untouched OOS evaluation."
)


def main():
    raise RuntimeError(DISABLED_REASON)


if __name__ == "__main__":
    main()
