# E*TRADE Read-Only Runtime

## Current status

The live runtime is under an R7f mutation quarantine. It may read broker and
market data and render local analytics, but this repository currently provides
**no supported path for placing, changing, cancelling, closing, or
neutralizing an E*TRADE order**.

Do not use the historical `--trade` flag as a live-trading instruction. The
flag remains in the legacy entry point while the runtime is being decomposed,
but it cannot re-enable quarantined mutation methods or routes. Service
installation and remote restart are also suspended by the deployment scripts.

This is source-level containment, not a production-readiness claim. No live
service was restarted and no broker mutation was exercised while verifying
R7f.

## Retained read-only capabilities

- Authenticated, loopback-only dashboard delivery.
- Explicit sandbox/production environment resolution and exact account
  identity checks before E*TRADE construction.
- Portfolio, balance, quote, option-chain, order-history, and position reads.
- On-demand portfolio refresh and generated position HTML.
- Position, cash-flow, margin, quote-freshness, GEX, and risk-monitoring
  analytics.
- Preview-only spread and neutralization calculations. A preview is not an
  authorization or an executable order.
- Redacted V2 background/shock regime advisory. It is shadow-only and cannot
  authorize execution.
- Owner-only handling for settings, OAuth state, and touched local logs.

Auto-open is forced off. The dashboard and generated position view contain no
execute, close, or neutralize controls and identify themselves as read-only.

## Fixed disabled-route contract

After authentication, each retained historical execution route returns HTTP
`503` with the same fail-closed payload before reading the request body,
changing files, queueing work, or invoking a collaborator:

| Route |
|---|
| `/api/execute_manual_order` |
| `/api/execute_neutralize_order` |
| `/api/review_close_position` |
| `/api/close_position` |
| `/api/execute_close_order` |

```json
{
  "code": "LEGACY_EXECUTION_DISABLED",
  "error": "Trading actions are disabled.",
  "read_only": true,
  "execution_enabled": false
}
```

Legacy preview/place/change/cancel/reprice/close methods outside the reviewed
R7 stack are unconditional tombstones. There is no environment variable,
configuration setting, production arm, action PIN, or operator override that
can re-enable them.

## Verification

Run the tracked-source mutation gate before tests:

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

The checker parses tracked application Python sources, including tracked
scratch files. It rejects raw broker mutation I/O, mutation-capable imports and
reflection, E*TRADE mutation literals, access to the hardened transport outside
its allowlist, legacy mutation calls, and drift in required tombstones.

R7f was also rendered through an isolated instance of the real local handler
and generated positions artifact at desktop and mobile widths. That inspection
did not start the trading loop, create an E*TRADE session, restart a deployed
service, or prove the state of the deployed Pi.

## Future live architecture

The intended live mutation path is a single composition root around:

1. `order_intent_ledger.py` for durable intent, reservation, authorization,
   send, response, and reconciliation evidence.
2. `etrade_broker_reader.py` for bounded, origin-pinned, durable broker reads.
3. `etrade_order_gateway.py` for account-bound coordination and the immutable
   risk ceiling.
4. `etrade_broker_transport.py` as the only reviewed mutation adapter.

Those components are implemented in isolation and are not instantiated by the
live runtime. Partial/replacement/assignment recovery, durable cancellation,
closing capacity, a pure full pre-trade risk policy, the single production
composition root, operational migration, and deployed verification remain
required before any live wiring is supported.
