# Raw HMM Taxonomy and Return-Bucket Contract

The legacy regime engine exposed two unrelated numeric namespaces as plain
integers:

- raw HMM states, whose meaning depends on one exact fitted model; and
- final stress-overlay states used for display and risk diagnostics.

Those values are no longer interchangeable. Raw return distributions use
`RawHMMStateRef` and `RegimeReturnBuckets`. Final overlays use
`FinalRiskRegimeRef` and the separate `Overlay_Regime_State` /
`Overlay_Regime_Label` plotting columns. `HMM_State` and `Regime_Label` always
remain the raw fitted-HMM outputs. The overlay mapping is closed to Expansion
(0), Cautious Decline (1), and Panic / Crisis (2); arbitrary state/label pairs
are rejected.

## Exact taxonomy identity

Every usable fitted HMM carries a recomputable SHA-256 taxonomy ID. The ID
binds:

- ordered semantic labels for every raw state;
- the causal feature-manifest hash, including its training-data prefix;
- the exact model training cutoff;
- the taxonomy/snapshot pipeline version and model class; and
- fitted start probabilities, transition matrix, emission means,
  covariances, and mixture weights when present.

A cached model without this binding is legacy content and is rejected. A
parameter, label, cutoff, feature, or pipeline change produces a different ID.
The probability engine requires the model and buckets to carry the same ID.
Fixed-model traces record that ID on every row. Expanding-window traces record
`Raw_HMM_Taxonomy_ID` per row because each refit is a distinct fitted
taxonomy; callers must not project the final returned HMM identity backward
over earlier rows.

## Strict outcome availability

`horizon_calendar_days` is converted once to `horizon_trading_days`. For each
historical entry row, the engine records the exact terminal NYSE session used
to calculate its forward return. An outcome is eligible only when:

```text
terminal_resolution_session < inference_as_of
```

Same-day terminal outcomes are excluded. The typed bucket contract records
`resolved_outcomes_through`, and every raw state must have at least one
strictly resolved observation. Statistical construction still requires the
higher minimum sample count enforced by `get_probability_engine`.

Live callers do not use an in-progress daily bar. They resolve an explicit
`as_of_date` from the latest NYSE session whose regular close is already in the
past. `build_regime_return_arrays` rejects an omitted `as_of_date`; there is no
implicit wall-clock fallback. A missing calendar, ambiguous timezone, or
intraday timestamp fails closed.

The standalone `market_regime_detect.py` review also separates calibration
from evaluation. Its default calibration fetch begins eight years before the
requested test start and ends on the final NYSE session strictly before that
start. Callers may declare an earlier calibration start/end. The HMM receives
the resolved pre-test `fit_end`, only requested out-of-sample rows are plotted,
and the returned review record includes the calibration range, test range,
walk-forward inference method, and signal timestamp.

## Cache and execution boundary

Snapshot pipeline version 3 stores the typed buckets, exact model, daily
models, taxonomy ID, and a value-binding bucket manifest. Legacy dictionaries,
missing states, changed values, mismatched horizons, tampered fitted
parameters, and unbound models invalidate the cache.

The current contract remains research-only:

```text
validity_status = UNVERIFIED
execution_eligible = false
```

Taxonomy integrity prevents namespace mixing and look-ahead reuse; it does not
certify the data provider, model calibration, probability quality, or live
order eligibility.

## Historical caller boundary

Every tracked expanding-window HMM caller must declare an explicit
`fit_end` strictly before its test range. A repository AST test enforces this
at every call site. Entry alignment uses the exact prior NYSE session; it
never forward-fills across a missing source row. The shared session mapper
accepts only a nonempty, ordered list or tuple of canonical `YYYY-MM-DD`
strings so canonical entry keys cannot be lost during normalization.

Legacy scripts that globally fit the HMM over their evaluation sample, use
same-day states, or turn a panic overlay into replacement risk are stable
tombstones. Maintained regime-aware backtests must use
`BacktestRegimeProtocol` and persist typed decision evidence. The final
overlay may veto new entries or reduce existing exposure, but it cannot add
or replace risk.
