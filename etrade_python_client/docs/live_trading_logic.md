# Live Runtime Logic

## Current execution boundary

R7f converts the legacy live agent into a read-only monitoring and analytics
runtime. There is no supported E*TRADE place, change, cancel, close,
neutralize, repricing, or margin-release path. Auto-open is forced off, and the
historical `--trade` flag is not an instruction or override for live trading.

The dashboard is authenticated and loopback-only. Service installation and
remote restart remain suspended.

## First-in-cycle positions publication

R8e-B publishes the broker-isolated dashboard's static positions read model
before model loading, candidate scans, and other slower analytics. Publication
requires two consecutive page-complete E*TRADE portfolio reads. The first is a
minimal identity/quantity scan; the second is the complete display scan. Their
canonical symbol, security type, option contract, and quantity fingerprints
must match.

An HTTP, pagination, account-identity, schema, stability, projection,
signature, filesystem, or freshness failure prevents replacement and preserves
the last confirmed artifact. That older artifact eventually becomes visibly
stale; it is never silently refreshed by file modification time alone.
Publication cadence is no slower than half
`data.max_snapshot_age_seconds`.

The publisher remains composed inside the transitional legacy monitor rather
than a standalone read collector. The historical `--trade` CLI spelling is
therefore still required to start that monitor even though all compatibility
mutation paths are unconditional tombstones. The static artifact is strictly
for operator display and cannot be used as a risk or execution snapshot.

## Retained analytical logic

The runtime may still calculate and display:

- market-session status;
- portfolio, balance, margin, position, and quote-freshness views;
- option-chain and preview-only spread selection;
- GEX and neutralization previews;
- cash-flow and position history;
- extrinsic-value and assignment-risk observations;
- potential high-gain or expiry-risk conditions;
- the shadow-only two-axis regime advisory.

Target premium, VIX adjustment, DTE, delta, spread width, earnings filters,
profit thresholds, and candidate rankings are analytical inputs only. They may
describe or preview a possible strategy, but they do not authorize execution.
Alerts and previews are not durable risk decisions.

## Disabled historical actions

The following dashboard routes are fixed authenticated tombstones and return
HTTP `503 LEGACY_EXECUTION_DISABLED` before reading a body or causing side
effects:

- `/api/execute_manual_order`
- `/api/execute_neutralize_order`
- `/api/review_close_position`
- `/api/close_position`
- `/api/execute_close_order`

Legacy Python methods for preview/place/change/cancel, stale-order nudging,
automatic same-day ITM closing, high-gain closing, margin release, scheduler
workers, and facade placement reject unconditionally. No settings value,
action PIN, environment, arm file, or operator choice can enable them.

## Enforcement and verification

Run:

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

The static checker protects the reviewed mutation boundary and exact
tombstones across tracked sources. R8e-B dashboard behavior was visually
checked through the exact broker-isolated local handler and signed generated
position artifact at desktop and mobile sizes, including digest-pinned iframe
delivery. The deployed service was not restarted or inspected, and no E*TRADE
order path was exercised.

## Promotion path

Future live execution must use one reviewed composition root. It must compose
the durable E*TRADE reader, intent ledger, order gateway, and no-retry mutation
transport with a pure full pre-trade risk policy. The existing isolated stack
is not connected to the live agent. Partial/replacement/assignment recovery,
durable cancellation, closing capacity, operational migration, and deployed
verification must be complete before that boundary can be promoted.
