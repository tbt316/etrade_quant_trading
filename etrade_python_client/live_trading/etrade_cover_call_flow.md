# E*TRADE Read-Only Runtime Flow

This document describes the R7f behavior of
`live_trading/etrade_cover_call_new.py`. It replaces the historical automated
trading flow. The current runtime is a local monitoring and analytics service;
it has no supported broker-mutation path.

## 1. Startup boundary

1. Resolve an explicit `sandbox` or `production` environment.
2. For production, validate the exact account ID, account key, institution
   type, and an owner-only signed arm document.
3. Construct OAuth and read-only E*TRADE collaborators only after the runtime
   boundary succeeds.
4. Select and revalidate the exact account identity.
5. Start the authenticated HTTP server on loopback.

The legacy entry point still contains a historical `--trade` gate. It is not a
supported live-trading command and cannot bypass R7f tombstones. The deployment
scripts reject service installation and remote restart before privileged or
remote actions.

## 2. Read-only monitoring loop

The retained loop may:

- renew the OAuth session and revalidate the account boundary;
- inspect market status;
- fetch confirmed portfolio, balance, quote, option-chain, position, and order
  history data;
- update cash-flow, margin, position, quote-freshness, and assignment-risk
  analytics;
- render `screened_option_pairs.html`;
- serve dashboard refresh, settings, status, GEX, preview-only analytics, and
  the redacted V2 regime advisory.

Market windows and strategy parameters may still influence analytics or
candidate previews. They do not authorize an order. Auto-open is forced off,
and no scheduler or worker owns an operative place/change/cancel/close path.

## 3. Dashboard flow

```text
Authenticated browser on loopback
  -> dashboard_template.html
  -> read-only refresh/status/settings/preview requests
  -> confirmed broker and local analytical data
  -> generated positions artifact and advisory cards
  -> no order mutation
```

The dashboard and generated position rows contain no execute, close, or
neutralize controls. Preview labels explicitly state that submission is
disabled. The V2 background/shock regime signal remains a shadow advisory and
cannot affect order eligibility.

## 4. Historical execution-route rejection

These exact paths are retained only as compatibility tombstones:

- `/api/execute_manual_order`
- `/api/execute_neutralize_order`
- `/api/review_close_position`
- `/api/close_position`
- `/api/execute_close_order`

After authentication, each route returns HTTP `503` with:

```json
{
  "code": "LEGACY_EXECUTION_DISABLED",
  "error": "Trading actions are disabled.",
  "read_only": true,
  "execution_enabled": false
}
```

The rejection occurs before request-body parsing, log/file mutation, queue
changes, or collaborator calls. Legacy Python preview/place/change/cancel,
repricing, automatic-close, margin-release, and facade mutation methods are
also unconditional reject-only tombstones.

## 5. Static containment

`scripts/check_etrade_mutation_boundary.py` scans every tracked application
Python source, including tracked scratch. CI runs it before the test suite. The
gate restricts raw broker mutation to the reviewed transport, restricts exact
transport calls to the gateway, rejects public transport exposure and legacy
call sites, and verifies the fixed tombstone policy.

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

## 6. Intended future mutation flow

```text
single reviewed production composition root
  -> pure pre-trade risk policy
  -> durable ETradeOrderGateway
     -> durable account/order reader
     -> intent ledger and capacity reservation
     -> uniquely fenced no-retry transport send
     -> parsed response receipt
     -> restart-safe reconciliation
```

The ledger, reader, opening/reprice gateway, transport, and zero/full
terminal-risk absorption exist only as an isolated stack. Live composition is
blocked until partial/replacement/assignment handling, per-intent
cancellation, closing capacity, the full risk policy, operational migration,
and deployed verification are complete.
