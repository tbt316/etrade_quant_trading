# Backtest Regime Protocol

Regime-aware backtests use two deliberately separate namespaces:

- `FinalRiskRegimeRef` is the closed `0/1/2` stress overlay. Its value for an
  entry session is taken from the exact prior NYSE session. It may veto a new
  opening or reduce existing risk, but it cannot increase delta, width, DTE, or
  quantity.
- `RawHMMStateRef` is bound to one fitted-model taxonomy. It may select a
  return bucket only through `RegimeReturnBuckets.bucket_for(...)`; it never
  indexes by a final overlay state.

Every regime-aware run persists an immutable `BacktestRegimeProtocol` with:

- calibration end strictly before the out-of-sample test start;
- test date range;
- inference method;
- exact one-NYSE-session lag and signal timestamp;
- fixed raw HMM component count;
- `UNVERIFIED` validity and `execution_eligible=false`.

Every attempted regime-aware opening either receives immutable
`RegimeDecisionEvidence` or is blocked. The evidence records the exact prior
signal session, final overlay, raw taxonomy/state when required, model training
end, bucket as-of and resolved-through dates, horizons, assignment probability,
and a value-bound return-bucket manifest digest. Missing rows, sparse buckets,
taxonomy mismatches, failed model fits, and failed roots remain stable
`UNAVAILABLE` outcomes. There is no fixed-delta or zero-array fallback.

The raw bucket and assignment caches are run-local and exact-keyed. Both
successes and failures are cached. Post-hoc plots read the entry-time evidence;
they do not refit models or reconstruct return buckets.

Untyped external `regimes: dict` input is audit/plot-compatible for a baseline
run, but it cannot authorize a regime-aware backtest. A regime-aware external
context requires both an explicit `BacktestRegimeProtocol` and a validated
`LaggedFinalRiskMap` covering the exact test sessions.

These artifacts remain research-only. They do not authorize live E*TRADE
orders.
