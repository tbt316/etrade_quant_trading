# `etrade_cover_call_new.py` — Current Functionality Specification

## Status: R7f read-only quarantine

`live_trading/etrade_cover_call_new.py` is currently a local E*TRADE monitoring
and analytics runtime. It is **not a supported automated trading entry point**.
All known legacy broker-mutation paths are quarantined.

Historical descriptions of automatic opening, email-approved execution,
same-day ITM closing, high-gain closing, stale-order price chasing, margin
release, manual execution, and position neutralization no longer describe
executable behavior. The legacy `--trade` flag remains in the entry point while
the runtime is decomposed, but it does not authorize or restore those paths.
There is no supported live-start or deployment recipe in this document.

## Supported behavior

### Startup and identity safety

- Requires an explicit `sandbox` or `production` environment before OAuth
  construction.
- Production requires the exact account ID, account key, and institution type
  plus an owner-only, HMAC-signed, short-lived arm document.
- Revalidates the exact account identity after refresh.
- Sources OAuth credentials outside tracked source.
- Uses owner-only settings, OAuth state, and touched log files.

These controls are prerequisites for safe reads. They do not make the legacy
runtime order-capable.

### Read-only broker and market data

The runtime may obtain and process confirmed:

- account, balance, portfolio, and position data;
- quotes and option chains;
- order history used for cash-flow and position tracking;
- market-session state and market-calendar data.

Failed or incomplete upstream reads must preserve the last confirmed artifact;
they must not publish a false empty portfolio or erase confirmed history.

### Local analytics and artifacts

The runtime may:

- render `screened_option_pairs.html`;
- serve the authenticated dashboard on loopback;
- refresh confirmed portfolio and analytical data;
- display position P&L, margin, cash flow, quote freshness, and risk
  observations;
- calculate preview-only spreads, GEX, and neutralization analytics;
- display a redacted V2 `background + shock` regime advisory.

Auto-open is forced off. Preview results and the V2 signal are explicitly
non-authoritative and cannot authorize execution. The dashboard and generated
position artifact contain no execute, close, or neutralize controls.

## HTTP surface

The retained read-only surface includes authenticated dashboard, refresh,
settings, status, positions, preview, GEX, and regime-advisory responses.
These endpoints provide observations or local configuration only.

Five historical execution endpoints remain so older clients fail explicitly
instead of receiving an ambiguous response:

| Method | Route | Result |
|---|---|---|
| `POST` | `/api/execute_manual_order` | fixed authenticated `503` |
| `POST` | `/api/execute_neutralize_order` | fixed authenticated `503` |
| `POST` | `/api/review_close_position` | fixed authenticated `503` |
| `POST` | `/api/close_position` | fixed authenticated `503` |
| `POST` | `/api/execute_close_order` | fixed authenticated `503` |

The response schema is:

```json
{
  "code": "LEGACY_EXECUTION_DISABLED",
  "error": "Trading actions are disabled.",
  "read_only": true,
  "execution_enabled": false
}
```

Authentication precedes route disclosure. For an authenticated request, the
handler rejects before reading the request body, writing files or logs,
changing a queue, or calling a collaborator.

## Python mutation quarantine

Known legacy mutation methods in the order compatibility layer, live facade,
schedulers, workers, repricer, automatic close path, and margin-release path
are unconditional `LegacyExecutionDisabled` tombstones. They have no
environment, configuration, arm-file, action-PIN, or operator override.

The hardened R7 mutation transport is private to the future gateway boundary.
The live monolith has no public transport handle and does not instantiate the
gateway.

## Deployment status

`deploy/install_pi_service.sh` rejects service installation before privileged
or service-manager work. `deploy/sync_to_pi.sh --restart` rejects before SSH,
rsync, or restart work. A sync-only/state-inspection workflow may remain for
source comparison, but it is not authorization to start the runtime.

No deployed service was restarted or verified as part of R7f, and no E*TRADE
session or broker mutation was exercised.

## Static enforcement

Run the mutation-boundary gate before the test suite:

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

The checker parses tracked application Python files, including tracked scratch.
It rejects:

- raw HTTP mutation I/O and mutation-capable imports/reflection;
- E*TRADE mutation URLs, XML, and methods outside the reviewed boundary;
- access to the hardened transport outside its allowlist;
- public gateway transport exposure;
- legacy mutation calls;
- changes to the exact required tombstones.

CI runs this checker before tests.

## Verification scope

R7f tests cover fixed reject-only methods and routes, response ordering and
schema, read-only HTML, gateway privacy, mutation-bypass patterns, and
install/restart containment. The real local handler and generated positions
artifact were rendered in an isolated process at desktop and mobile widths.
That inspection verified the source-served read-only UI, not the user's
deployed Pi.

## Future production mutation architecture

Any future broker mutation must flow through a single reviewed composition
root:

```text
operator/strategy request
  -> pure pre-trade risk policy
  -> ETradeOrderGateway
     -> durable ETradeBrokerReader evidence
     -> OrderIntentLedger reservation and authorization
     -> uniquely fenced ETradeBrokerTransport send
     -> parsed durable response
     -> restart-safe reconciliation
```

The ledger, origin-pinned reader, opening/reprice gateway, hardened no-retry
transport, and exact zero/full terminal-risk absorption have been implemented
and tested in isolation. They are deliberately unreachable from the live
runtime.

Promotion remains blocked on:

- partial-fill, replacement, and assignment recovery;
- durable per-intent cancellation;
- closing-order capacity and lifecycle;
- a pure full pre-trade risk policy;
- one production composition root;
- operational migration, secret rotation/history cleanup, and deployed
  verification.

Until those gates are complete, the correct runtime contract is read-only.
