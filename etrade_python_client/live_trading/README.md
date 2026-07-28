# E*TRADE supervised trading runtime

## Current status

The dashboard supports one narrow broker mutation: an authenticated operator
may manually open a server-selected SPY or SPX two-leg `PUT` or `CALL` credit
spread. This is an improved restoration of the previous Execute workflow, not
a restoration of the legacy mutation code.

The normal flow remains familiar:

1. Select ticker, side, width, expiration, delta, and quantity.
2. Refresh the live spread preview.
3. Click **Execute SPY Spread** or **Execute SPX Spread**.
4. Review the exact account, broker symbol, expiration, legs, limit credit,
   quantity, total credit, maximum loss, proposal expiry, proposal ID, and
   request ID.
5. Re-enter the dashboard PIN and click **Submit Once**.

The PIN is never stored in browser storage and must be re-entered for the exact
final confirmation. A broker acknowledgement means only that E*TRADE
acknowledged the submission; it is **not** a fill confirmation.

Auto-open, closing, neutralization, cancellation, repricing, and automatic
retry remain unavailable from the dashboard. The V2 background/shock regime
display is advisory-only and cannot authorize an order.

## Safety boundary

The browser and legacy scanner never supply authoritative order economics. The
scanner selects only candidate contract identity. During an open NYSE regular
session, the manual service makes one origin-pinned, no-retry E*TRADE quote
read for the exact two OSI contracts. Both quote rows must be
`quoteStatus=REALTIME`, unadjusted standard 100-share options, usable
non-crossed NBBOs, fresh under the configured age limit, and timestamped no
more than five seconds apart. The server derives the exact two-leg midpoint
credit and creates a short-lived HMAC-signed proposal bound to:

- the exact account and broker environment;
- the immutable runtime-configuration digest;
- ticker, broker symbol, side, expiration, strikes, and raw OSI identities;
- the retained quote receipt/snapshot hashes, both bid/ask pairs, and both
  exchange timestamps;
- the server-derived midpoint limit credit and oldest quote timestamp; and
- an expiry no later than the configured quote-age ceiling.

The executable preview and confirmation render the proposal projection, not
the scanner payload: exact contracts, strikes, expiry, and limit credit are
overwritten from the signed broker-quote result. Scanner delta and OTM remain
advisory selection context only. The dashboard banner and final confirmation
show `SANDBOX` or `PRODUCTION`, and a preview bound to another environment or
runtime configuration is disabled.

Confirmation sends only that signed proposal, a bounded quantity, a canonical
request ID, and the freshly entered action PIN. The signed `proposal_id` is the
durable idempotency key; the UUID `request_id` correlates only this HTTP
request/response. Changing the request ID cannot turn one proposal into a
second durable order. The server verifies the PIN and proposal before invoking
the sole reviewed mutation composition:

```text
dashboard
  -> ManualOpenService
  -> EtradeOrderGateway
  -> OrderIntentLedger
  -> ETradeBrokerTransport
  -> E*TRADE
```

Before a proposal can be issued, the runtime must have:

- schema-2 `sandbox` or `live` opt-in;
- one exact allowlisted account;
- a current environment/account runtime arm;
- readable, structurally valid durable history;
- no unresolved placement, cancellation, closing, amendment, or terminal
  absorption blocker;
- an open NYSE regular session; and
- the exact two-leg broker quote evidence described above.

The signed proposal deadline is also capped fifteen seconds before that
session's exact NYSE close. The same deadline is enforced again before broker
preview and placement, preventing a near-close request from being staged for
the next session.

Proposal issuance does not read or reserve account capacity. Final submission
does, so a valid proposal can still fail safely before broker I/O. That fresh
capacity-v3 decision enforces two independent limits:

- `max_account_open_risk_cents` bounds current opening risk. Schema-19 policy
  V2 caps it at raw broker buying power and subtracts external position risk,
  external order risk, and represented managed filled risk from the immutable
  account budget; active local reservations are subtracted separately.
- `max_daily_loss_cents` is a New York calendar-day budget for newly authorized
  maximum loss. It is not trading P&L, is not tied to NYSE session boundaries,
  and is not refunded after a reservation exists, even if later processing
  fails.

Unsupported, adjusted, naked, ambiguous, or incompletely identified broker
positions or active orders fail closed. Capacity is never inferred from an
unverified browser value. Historical capacity-policy V1 decisions remain
replay-only; only V2 can authorize a new reservation or submission claim.

## No automatic retry

The browser writes a non-secret recovery marker before sending the request,
clears the confirmation PIN, and aborts its wait after 30 seconds. The marker
contains only proposal/request identity plus account, environment, and
configuration bindings. It is cleared only when the server returns an exact
matching `request_id` with `submission_disposition=NOT_ATTEMPTED`. A malformed,
unrecognized, mismatched, timed-out, or otherwise uncorrelated response is
treated as `SUBMISSION_UNKNOWN`, even if its HTTP status is below 500. On
reload:

- a matching durable record is displayed;
- `SUBMITTED` is labelled broker-acknowledged, not filled;
- an unresolved or missing record displays **DO NOT RETRY** and disables every
  manual-open button; and
- no request is resubmitted automatically.

The durable ledger owns idempotency and ambiguous-outcome handling. Network
timeouts, malformed responses, crashes, and post-capable lease expiry cannot
turn an uncertain submission back into a retryable browser action.
The UI refreshes the local ledger status periodically (nominally every 30
seconds); this is not broker-order polling. It never queries E*TRADE order
state, retries, resubmits, or reprices.

## Dashboard authentication and generation

Login and PIN failures use separate bounded process-local throttles. Login,
PIN verification, settings, and manual-open JSON bodies are content-type and
size bounded. The owner-only dashboard authentication master is a generated
256-bit lowercase-hex value; invalid legacy values rotate on load, and
credential changes rotate it again to revoke current sessions. Session and
proposal signatures use separate HMAC derivation domains.

At process start the backend pins one bounded, regular UTF-8 template whose
protocol marker and nonce placeholders match its own protocol version. It
serves those pinned bytes with their SHA-256, preventing a newly edited HTML
file from being mixed with an already-running backend generation.

## Legacy mutation quarantine

These historical routes remain fixed `503` tombstones after authentication and
cannot read the request body or reach legacy collaborators:

| Route |
|---|
| `/api/execute_manual_order` |
| `/api/execute_neutralize_order` |
| `/api/review_close_position` |
| `/api/close_position` |
| `/api/execute_close_order` |

The old queue/fast-worker, execute, close, neutralize, automatic strategy,
automatic-close, cancel, change, margin-release, and repricing paths remain
reject-only or uncomposed. No environment variable, action PIN, or dashboard
setting can revive them.

## Operator and deployment requirements

Do not enable live mutations by editing `live_trading_settings.json`. Use the
schema-versioned runtime configuration and independently signed runtime arm
described in [runtime_configuration.md](../docs/runtime_configuration.md).
Keep the dashboard bound to loopback behind the reviewed authenticated tunnel,
use owner-only runtime directories/files, and rotate any credentials that
previously appeared in repository history.

After an upgrade, run migration and recovery drills against a copied ledger
before activating the service. If the dashboard reports an unresolved durable
state, stop and reconcile it; do not create a replacement proposal merely
because no broker order is visible in the browser.

## Verification

Run the mutation boundary before the maintained tests:

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

The checker confines raw mutation I/O to the reviewed transport, transport
construction to the reviewed composition root, and transport calls to the
durable gateway. Dashboard-facing changes additionally require rendering the
served HTML at desktop and mobile widths and exercising lock, preview,
confirmation, durable result, and reload-recovery states.

This source boundary is not evidence of a completed live deployment. A full
E*TRADE sandbox and live preview/place/fill/reconciliation/restart lifecycle,
service rollout/rollback drill, credential rotation, and exact deployed
endpoint verification remain required before calling the deployed system
production-ready or using it for unattended trading. The complete pure
pretrade engine—account Greeks, concentration, marked P&L, and regime
authorization—is not composed into this manual path. Partial fills,
replacement chains, assignment/exercise, and related recovery also remain
blocked. Verification to date is source/local visual verification, not an
E*TRADE or deployed-service lifecycle.
