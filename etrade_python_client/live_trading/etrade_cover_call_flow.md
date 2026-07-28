# E*TRADE Supervised Manual-Open Runtime Flow

This document describes the current source-level behavior of
`live_trading/etrade_cover_call_new.py`. The runtime remains primarily a local
monitoring and analytics service, with one deliberately narrow mutation
capability: an authenticated operator may open a familiar SPY or SPX `PUT` or
`CALL` two-leg net-credit vertical after reviewing a short-lived server
proposal and confirming it with the action PIN.

This restoration is not unattended automation and is not evidence of a
completed live or sandbox order lifecycle.

## 1. Startup boundary

1. Load and validate an explicit schema-versioned runtime configuration.
2. Resolve the exact `sandbox` or `production` environment and account
   identity through the independent runtime safety boundary.
3. In production, validate the owner-only, signed, short-lived runtime arm.
4. Construct OAuth and read-only E*TRADE collaborators only after that boundary
   succeeds, then select and revalidate the exact account.
5. If and only if schema 2 has
   `execution.broker_mutations_enabled=true`, construct the private ledger,
   broker reader, no-retry transport, and durable gateway through
   `execution_runtime.build_manual_open_service`.
6. Return only the narrow `ManualOpenService` capability to the dashboard
   process and start the authenticated HTTP server on loopback.

Schema-2 opt-in is a configuration prerequisite, not authority. Manual opening
also requires an exact runtime/account match, the independent production arm
when applicable, dashboard authentication, the action PIN, a current signed
proposal, and gateway validation. If composition or OAuth-session replacement
cannot rebuild that exact capability, manual opening remains unavailable.

The `--trade` process flag enters this supervised runtime; it does not restore
the historical automated trading path. Deployment scripts continue to reject
service installation and remote restart before privileged or remote actions.

## 2. Monitoring and proposal flow

The retained monitoring loop may:

- renew OAuth and revalidate the account boundary;
- inspect market status;
- fetch confirmed portfolio, balance, quote, option-chain, position, and order
  history data;
- update cash-flow, margin, position, quote-freshness, and assignment-risk
  analytics;
- render `screened_option_pairs.html`;
- serve dashboard refresh, settings, status, GEX, read-only analytics, and the
  redacted V2 regime advisory; and
- expose current SPY/SPX `PUT` and `CALL` spread economics for explicit
  operator review.

Auto-open remains forced off. Market windows, strategy settings, previews, and
the regime advisory do not independently authorize an order.

## 3. Supervised manual-open flow

```text
Authenticated operator selects a current SPY/SPX PUT or CALL vertical
  -> scanner supplies candidate contract identity only
  -> while NYSE regular trading is open, the durable reader retains one
     origin-pinned E*TRADE REALTIME response for both exact OSI contracts
  -> strict replay validates both NBBOs, exchange timestamps, freshness,
     standard contract identity, and at most five seconds of leg skew
  -> ManualOpenService issues a signed, short-lived proposal
     bound to runtime config, environment, exact account, symbol,
     expiration, strikes, quote receipt/snapshot, both NBBOs/timestamps,
     derived midpoint credit, oldest observation time, and expiry
  -> operator reviews the exact proposal and quantity
  -> operator re-enters the independent dashboard action PIN
  -> server revalidates signature, freshness, account/runtime binding,
     UUID request shape, quantity, quote age, and per-order maximum loss
  -> gateway obtains fresh capacity-v3 evidence and may fail before any send
  -> one SubmitOpeningCommand enters EtradeOrderGateway
  -> durable intent, fenced no-retry send, receipt, and reconciliation
```

`GET /api/status` advertises the fail-closed manual-open capability and reason.
When execution is eligible, `GET /api/preview_spread` replaces every
execution-facing candidate identity and price field with the signed proposal's
exact contracts and strictly replayed E*TRADE two-leg midpoint. It also
projects the retained quote hashes/timestamps, proposal token/ID/expiry, signed
maximum quantity, and `execution_enabled=true`; delta and OTM remain advisory
scanner context only. The dashboard banner and confirmation display the exact
`SANDBOX` or `PRODUCTION` environment, and the browser disables a proposal if
its environment or configuration digest differs from current durable status.
Otherwise the preview remains non-executable and carries an unavailable
reason. Proposal issuance requires the open NYSE regular session but does not
read or reserve account capacity. Its signed deadline is capped fifteen
seconds before the exact session close and is enforced again before broker
preview and placement.

`POST /api/manual_open` is confirmation only. It accepts exactly the action
PIN, proposal token, quantity, and a browser-generated UUIDv4 request ID. The
browser cannot rewrite the signed economics at confirmation. Durable
idempotency is bound to the signed `proposal_id`; the `request_id` is HTTP
correlation only, so changing it cannot turn one reviewed proposal into a
second order. The PIN remains only in browser session memory, is cleared before
the request is sent, and must be re-entered for another confirmation. Existing
unlock/settings requests and this confirmation request transmit it to the
authenticated same-origin server, but it is never written into the dashboard
log.

The composed flow supports only:

- underlying `SPY` or `SPX` (`SPX`/`SPXW` broker symbols);
- option side `PUT` or `CALL`;
- exactly one `SELL_OPEN` leg and one `BUY_OPEN` leg;
- a standard two-leg `VERTICAL` with `NET_CREDIT` and `GOOD_FOR_DAY`;
- a positive integral quantity within `risk.max_order_contracts`; and
- maximum loss within `risk.max_order_loss_cents`.

The proposal lifetime is bounded by `risk.max_quote_age_seconds` and can never
extend beyond the age of the oldest leg quote. The service submits the reviewed
credit exactly as proposed. It does not nudge or reprice an order.

Final submission reads fresh capacity and enforces two independent budgets:
`max_account_open_risk_cents` bounds current account opening risk, while
`max_daily_loss_cents` is a New York calendar-day ceiling on newly authorized
maximum loss. The daily field is not realized/marked P&L or a trading-session
counter, and a reservation is not refunded from that daily total after later
failure. Schema-19 V2 caps account capacity at raw broker buying power and the
account budget net of external position/order risk and represented managed
filled risk; active local reservations are applied separately. Historical V1
capacity decisions are replay-only and cannot authorize a fresh reserve or
claim.

The UI sends one confirmation request with a 30-second browser abort and never
automatically retries, resubmits, or reprices it. Its recovery marker is cleared
only by an exact correlated `NOT_ATTEMPTED` response. Any malformed,
unrecognized, request-ID-mismatched, timed-out, or otherwise uncertain
response becomes `SUBMISSION_UNKNOWN`, regardless of a superficially
non-server-error HTTP status. The dashboard refreshes local durable status
periodically (nominally every 30 seconds), but that endpoint reads only the
ledger; it does not poll E*TRADE orders. Acknowledged `SUBMITTED` is not a fill
confirmation.

Login and PIN failures have separate bounded throttles, and JSON request bodies
are content-type and size bounded. The dashboard master secret uses a strict
generated 256-bit hex format; credential changes rotate it, while sessions and
manual proposals use separate derivation domains. The backend pins one
protocol-matched template generation at process start and serves its exact
bytes/hash so disk edits cannot create a mixed frontend/backend generation.

## 4. Mutation quarantine that remains in force

These historical dashboard paths remain compatibility tombstones:

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

The legacy queue/fast worker, execute, close, neutralize, automatic strategy,
automatic-close, cancel, change, margin-release, and repricing/nudging paths
remain reject-only or uncomposed. The supervised manual-open service exposes
only `submit_opening`; it does not expose the gateway's other mutation methods.

## 5. Static containment

`scripts/check_etrade_mutation_boundary.py` scans tracked application Python
source. CI runs it before the test suite. The gate confines raw broker mutation
to the reviewed transport, confines exact transport calls to the durable
gateway, restricts construction to the reviewed execution composition root,
and verifies that the legacy tombstones do not drift.

```bash
python scripts/check_etrade_mutation_boundary.py
pytest -q tests
```

## 6. Readiness boundary

The source now composes a narrow durable manual-opening path; it does not make
the repository fully production-ready for unattended trading. The following
remain unverified or incomplete:

- an end-to-end E*TRADE sandbox preview/place/fill/reconciliation/restart
  lifecycle using the restored dashboard path;
- a live production order lifecycle;
- independently replayable account-wide open-risk, daily P&L, concentration,
  delta, active-order, and session-boundary evidence, including full pure
  pretrade Greeks, concentration, marked P&L, and regime authorization;
- partial fills, replacements, transformed lots, assignment/exercise, and
  complex recovery;
- deployed service installation, restart, migration, rollback, and operator
  drills; and
- supported closing, neutralization, cancellation, or repricing from the
  dashboard.

Do not describe the supervised source composition as completed live
verification or as authorization for unattended production use. Current
evidence is source and local visual verification only; no E*TRADE sandbox/live
lifecycle or deployed-service activation has been verified.
