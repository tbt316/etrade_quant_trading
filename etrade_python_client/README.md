# E*TRADE Quantitative Options Research and Safety Infrastructure

This source tree contains an options-research simulator, causal market-regime
research, and fail-closed infrastructure for a future E*TRADE control plane.

> **Safety status:** unattended live trading is not approved. Legacy broker
> mutation paths, service installation, runtime bootstrap, and remote restart
> are disabled. The durable order stack is isolated and has no live caller.
>
> **Research status:** historical execution and regime-aware results remain
> `UNVERIFIED`. They are not investment-performance evidence.

Install only from the Git-root `pyproject.toml` and the hash-locked files under
`requirements/`. Legacy per-directory dependency files are unsupported.

## 1. Architecture

```text
etrade_python_client/
├── backtesting/
│   ├── backtest_runner.py          # Historical research simulator
│   ├── massive_api_client.py       # Massive historical-data adapter
│   ├── regime_bridge.py            # Typed causal regime/backtest boundary
│   └── strategies/*.yaml           # Strategy research configurations
├── live_trading/
│   ├── data_ingestion.py           # Prefix-causal feature preparation
│   ├── pca_fusion.py               # Causal feature fusion
│   ├── ev_engine.py                # HMM states and resolved return models
│   ├── regime_taxonomy.py          # Raw-HMM and final-overlay identities
│   ├── market_sessions.py          # Exact NYSE-session alignment
│   ├── pretrade_risk.py            # Pure fail-closed risk decision
│   ├── opening_risk_lineage.py     # Replayable, non-authorizing evidence
│   ├── order_intent_ledger.py      # Durable schema-17 state machine
│   ├── etrade_broker_reader.py     # Isolated durable broker reads
│   ├── etrade_broker_transport.py  # No-retry mutation transport
│   ├── etrade_order_gateway.py     # Isolated order coordinator
│   └── read_only_dashboard.py      # Broker-isolated operator display
├── docs/                           # Architecture and safety contracts
├── scripts/                        # Offline repository/release gates
└── tests/                          # Deterministic contract tests
```

The legacy live monitor remains useful for contained broker reads and artifact
publication, but its order, close, reprice, and cancellation compatibility
surfaces are unconditional tombstones. The new ledger, reader, transport,
gateway, and risk components are deliberately not composed into that process.

## 2. Market-regime boundary

The regime stack is a research system with explicit decision-time and taxonomy
contracts:

- Feature selection, fractional differencing, scaling, and PCA preparation are
  bound to an explicit historical prefix. Volatility smoothing is trailing; it
  is a filter, not an earnings-calendar model.
- The regime model is a Gaussian HMM. Gaussian-mixture models are used
  separately for regime-conditioned return distributions.
- Historical inference stores only the last posterior from each eligible
  prefix. Viterbi-smoothed history is forbidden for decisions.
- Every fitted raw HMM has a SHA-256 taxonomy identity binding its feature
  manifest, training cutoff, ordered labels, pipeline version, and fitted
  parameters.
- `RawHMMStateRef` may select only a `RegimeReturnBuckets` object with the exact
  same taxonomy. Outcomes must resolve strictly before the inference session.
- `FinalRiskRegimeRef` belongs to a separate closed namespace: Expansion,
  Cautious Decline, or Panic / Crisis. It may veto or reduce risk, but it must
  never index a raw-HMM return bucket or increase/replace exposure.
- Daily close signals use the exact prior NYSE session for a next-session
  decision. Missing sessions stay missing; there is no forward-fill fallback.

`BacktestRegimeProtocol` records the calibration cutoff, test range, inference
method, `close_T_for_next_session` timestamp, and exact one-session lag.
`RegimeDecisionEvidence` records the final overlay, optional taxonomy-bound raw
state and probability lineage, bucket cutoff, resolved-outcome cutoff, and
horizon. Missing required evidence blocks regime-aware openings. These records
remain `UNVERIFIED` and `execution_eligible=false`.

Regime V2 is a separate shadow-only background-plus-shock detector. It is useful
for distinguishing persistent stress from event-driven volatility spikes, but
it cannot authorize E*TRADE activity and has not completed prospective
promotion.

See:

- [`docs/market_regime_detect_specs.md`](docs/market_regime_detect_specs.md)
- [`docs/regime_taxonomy_contract.md`](docs/regime_taxonomy_contract.md)
- [`docs/regime_detection_v2_design.md`](docs/regime_detection_v2_design.md)

## 3. Historical simulation and mark evidence

The backtester remains a research simulator. Its maintained causal boundaries
include:

- point-in-time contract-reference snapshots keyed to the trade date;
- no strike-acquisition filter based on future underlying prices;
- explicit, independent entry and exit historical-mark policies;
- a regime-unaware baseline control;
- typed regime protocols and decision evidence for regime-aware runs; and
- persisted validity reasons instead of an executable-fill claim.

`strict_nbbo` validates an observed Massive quote only when every exact contract
has a finite, non-crossed bid/ask, a UTC event timestamp in the requested
regular NYSE session, acceptable age at the exchange-calendar close, and
acceptable multi-leg timestamp skew. A bundle contains exactly two or three
unique contracts from one pricing date.

That is strict historical **mark** evidence, not proof that an order could have
filled. Displayed size may be zero, modeled quantity may exceed available size,
and the default mark is a midpoint. Consequently
`historical_execution_proven`, `entry_execution_proven`, and
`exit_execution_proven` remain false.

`research_fallback` may use explicitly labeled observed quotes, synchronized
minute aggregates, trade prints, theoretical values, or daily closes. Those
sources never pass strict validation. Expiration closes are research-only;
settlement style and the official settlement reference remain `UNVERIFIED`,
including AM-settlement risk for index options.

See [`docs/historical_fill_evidence.md`](docs/historical_fill_evidence.md).

## 4. Live safety boundary

The isolated schema-17 execution core provides durable opening and closing
intent identity, exact capacity reservations, one-send fencing, known-order
reconciliation, one-shot cancellation, and zero/full terminal absorption.
The pure pretrade policy evaluates typed authority, quote, portfolio, and
overlay evidence.

This is not live authorization. Current opening-risk lineage is deliberately
`INDEPENDENT_EVIDENCE_PENDING`: retained inputs cannot yet reconstruct all
existing-position/open-order risk, Greeks, marked daily P&L, or durable
exchange-session counters. Partial fills, replacement chains, transformed
lots, and assignment/exercise remain blocked. A reviewed live composition
root, sandbox lifecycle evidence, operational deployment, credential
remediation, and provider entitlements are still required.

See:

- [`docs/order_intent_ledger.md`](docs/order_intent_ledger.md)
- [`docs/runtime_configuration.md`](docs/runtime_configuration.md)
- [`docs/production_readiness_upgrade_plan.md`](docs/production_readiness_upgrade_plan.md)
- [`RELIABILITY.md`](RELIABILITY.md)

## 5. Causal review checklist

A regime-aware historical result is not valid unless it records and verifies:

1. the training/calibration end date;
2. the out-of-sample test range;
3. the causal inference method;
4. an exact regime lag of at least one trading session; and
5. return buckets containing only outcomes resolved before the decision.

The raw HMM taxonomy must match the return bucket exactly. The final risk
overlay is never a substitute for that taxonomy.

## 6. Canonical offline source verification

Run these commands from `etrade_python_client/` in the hash-locked test
environment:

```bash
python scripts/check_secret_content.py --start .
python scripts/check_repo_hygiene.py --start .
python scripts/check_etrade_mutation_boundary.py
ETRADE_TEST_NETWORK=deny MASSIVE_OFFLINE_ONLY=1 \
  python -m pytest -q -m "not integration" tests
```

The hygiene gate also invokes the exact-Git-blob secret scanner. The explicit
secret command above makes that evidence visible during review. The complete
artifact-first release sequence is documented in the Git-root
[`README.md`](../README.md#offline-verification).

Scripts that acquire Massive, Yahoo, Cboe, or E*TRADE data, populate caches,
generate provider-backed plots, or run long historical experiments are
explicit integration/research operations. They are not part of the default
offline gate and must not be used as evidence that live trading, provider
provenance, execution, or deployment is ready.
