# Live Dashboard Reliability Ledger

This is the evolving source of truth for failures that affect live E*TRADE data,
derived calculations, generated dashboard HTML, or the page delivered to the
browser. Read it before changing those paths and update it after a verified fix.

## How to use this file

For every reliability incident:

1. Record the user-visible symptom and the exact affected artifact.
2. Separate the triggering event from the architectural root cause.
3. State the invariant that would have prevented the incident.
4. Link the implementation and regression test.
5. Verify the exact served dashboard, not only source code or an offline fixture.
6. Keep unresolved risks explicit. Do not mark an incident resolved from a code
   patch alone.

## Non-negotiable invariants

### External data

- A failed, malformed, rate-limited, or partially paginated upstream response
  must never delete previously confirmed data.
- A range may replace cached history only after every page in that range has
  completed successfully.
- Every rendered market-price series must participate independently in
  missing-date detection. Complete SPY or SPX history must not suppress a VIX
  backfill.
- Retried requests must be bounded and must preserve the last confirmed state
  when retries are exhausted.
- Avoid unused or duplicative upstream calls in a live refresh path. E*TRADE
  request volume is part of the reliability budget.
- Persist both the last attempt and the last successful synchronization. A
  stale-but-confirmed result must be distinguishable from a fresh result.
- Regime inputs must persist provider/field identity, exchange event time,
  `available_at`, `ingested_at`, finality, and a payload checksum. A caller-
  supplied provenance flag or filesystem modification time is not evidence;
  verified status additionally requires a durable raw response and parser
  receipt linked to the selected observation.
- A regime-data range is complete only when every expected NYSE session has
  one independent final SPY close and one independent final VIX close. A
  partial leg, holiday-only row, or conflicting correction must not overwrite
  the last confirmed snapshot.
- A continuously scheduled or order-adjacent market-data collector must have a
  current entitlement covering non-display strategy use, raw-response
  retention, the exact dataset, and the deployed subscriber. Missing, expired,
  or scope-mismatched rights fail before the first network request; an API key
  or publicly accessible CSV is not entitlement evidence.
- Retained provider responses are evidence, not dashboard content. Raw bytes
  must never be committed, logged, embedded in generated HTML, or returned by a
  dashboard endpoint.

### Calculations

- The green cash-flow series is cumulative **net cash flow**, not trade count or
  gross trading activity. Offsetting buys and sells can legitimately make it
  flat even when trades occurred.
- A missing daily source value may be carried forward for plotting only when the
  sync health is known. It must not silently disguise an incomplete destructive
  refresh.
- For a trading day under investigation, compare the rendered cumulative value
  with the raw executions, the cached daily net flow, and the prior cumulative
  value.

### Live startup and local security

- Every order-capable process must require an explicit `sandbox` or `production`
  environment. Missing or contradictory mode input must never fall through to
  production.
- Production startup requires an exact account ID, account key, and institution
  type plus an owner-only, signed, short-lived arm document bound to the same
  identity. The arm and identity must be revalidated after account refresh and
  immediately before every order API call. R7 must preserve that invariant
  while replacing the compatibility guard with one durable mutation gateway.
- Every broker mutation must have one durable, uniquely fenced send claim
  before I/O and one parsed response receipt before acknowledgement. An
  ambiguous send is reconciliation-only and is never retried. A broker query
  may clear it only for a durably known broker order ID whose exact account,
  environment, and canonical economic terms match the authorization.
- Account opening capacity must use an immutable gateway/operator risk ceiling,
  not a strategy-command value. Incomplete, stale, unstable, or non-durable
  balance/position/open-order evidence cannot authorize exposure.
- An order-capable dashboard binds to loopback by default. Public tunneling is a
  separately supervised operator action, and wildcard CORS is forbidden.
- Dashboard credentials must reject defaults and weak values. Settings, arm
  documents, OAuth state, and touched local logs must be owner-only regular
  files and must not expose credentials, OAuth verifiers, full account
  responses, order payloads, or account-bearing URLs.
- Removing a credential from the current source does not remediate a public Git
  history. Any published key must be treated as compromised, revoked and
  rotated at its issuer, then removed from history through a coordinated rewrite
  and downstream clone/cache cleanup.

### Delivery and verification

- The live service must be restarted or reloaded after backend changes.
- Regenerate the production HTML through the same endpoint used by the user.
- Confirm cache headers/generations and reload the exact served artifact.
- For charts, inspect the actual canvas at the user's time span and viewport.
- Record the final rendered label and plotted values, and compare them with the
  current cache/source of truth.

## Data path

```text
E*TRADE executed Orders API
  -> complete paginated range fetch
  -> atomic recent-window reconciliation
  -> spy_gains_cache.json (daily flows, trades, sync health)
  -> get_spy_tracking_history_with_gains()
  -> accounts/accounts_bo.py chart series
  -> generated portfolio HTML
  -> /api/positions
  -> browser canvas
```

The generated HTML and the cache are artifacts, not independent sources of
truth. A source patch is not live until this whole path has been exercised.

## Incident ledger

| ID | Status | Symptom | Root cause | Guardrail |
|---|---|---|---|---|
| INC-2026-07-17-01 | Resolved | Green cash-flow curve appeared flat for several recent trading days despite executions. | A partial Orders fetch was treated as complete, then the entire 14-day cache window was deleted and rebuilt from the partial response. Missing chart dates were carried forward, hiding the loss. Repeated per-chunk Orders and unused Transactions calls also increased rate-limit exposure. | One paginated range request, bounded transient retries, complete-fetch gate before reconciliation, atomic file write, persisted sync health, and a visible stale-data badge. |
| INC-2026-07-17-02 | Resolved | Underlying quote shown from a position card could remain stale after Refresh Data. | Dashboard refresh and quote generation were not tied to a clearly observable fresh generation, and quote freshness was not exposed alongside the displayed value. | Refresh generation tracking, current E*TRADE quote metadata, no-store delivery, and visible quote timestamp/status. |
| INC-2026-07-17-03 | Resolved | Mobile view fragmented each SPX position into separate cards, making portfolio-level action thresholds hard to scan. | Layout was organized around individual position cards instead of decision-making by underlying. | One responsive table per ticker, pair-level gain/loss, DTE, strikes, and action emphasis. |
| INC-2026-07-18-01 | Resolved | A weekend Refresh Data request changed the live portfolio value and margin budget to zero. | The E*TRADE portfolio request returned 401 after OAuth expiry. `portfolio()` silently converted the failed response to an empty list, and the dashboard writer treated it as a confirmed empty portfolio and overwrote the production HTML. | Retry an expired-token portfolio request through the shared auth callback, and require every dashboard-producing portfolio fetch to succeed before replacing the served artifact. |
| INC-2026-07-24-01 | Resolved | The orange VIX line in the one-month benchmark chart stopped after July 6 while SPY and option-value data continued through July 24. | Market-history refresh checked only SPY and SPX for missing cached dates, so a VIX-only gap never triggered a download. Yahoo also returned malformed responses when the corrected refresh attempted the backfill. | Include VIX in independent missing-date detection and fall back to Cboe's official daily VIX history when Yahoo does not return the requested closes. |
| INC-2026-07-26-02 | Open | The live process could enter production without an explicit environment, bind a mutable account-list position, and expose an order-capable dashboard beyond loopback; OAuth credentials were also embedded in tracked source. | Environment, arming, account identity, dashboard exposure, local file permissions, and credential sourcing were independent conventions rather than one fail-closed startup boundary. The repository is public, so removing values from the current source cannot revoke copies retained in Git history. | R6 adds explicit environment resolution, exact production identity, a signed short-lived arm, placement-time revalidation, loopback-only dashboard service, strong local credentials, owner-only files, and guarded client logs. Keep this incident open until external keys are revoked/rotated, history is purged, R7 replaces compatibility paths with a durable single mutation owner, and the deployed service is verified. |

## INC-2026-07-17-01 — Recent cash-flow history regressed

### Evidence

During repeated refreshes for the same July 3–17 range, logs showed the fetched
order count changing from 22 to 25, then dropping to 16, and later recovering to
27. The cache simultaneously shrank from 344 to 341 days before recovering.
An E*TRADE rate-limit response occurred in the same refresh cycle.

The code previously broke out of a request loop on any non-200 response, retained
the orders fetched so far, deleted all cached flows/trades/close-days in the
lookback window, and merged only that partial set. The chart then carried the
last cumulative value forward across dates that still existed in other series.

For July 13–17, a subsequent complete response produced these confirmed daily net
cash flows:

| Date | Daily net cash flow | Cumulative cash flow |
|---|---:|---:|
| 2026-07-13 | $2,050.00 | $268,509.45 |
| 2026-07-14 | $2,050.00 | $270,559.45 |
| 2026-07-15 | $1,295.00 | $271,854.45 |
| 2026-07-16 | $2,080.00 | $273,934.45 |
| 2026-07-17 | $1,435.00 | $275,369.45 |

The screenshot's flat level matched the July 14 cumulative value, consistent
with the recent window having been partially erased and then carried forward.

### Implementation

- `live_trading/spy_position_tracker.py`
  - Fetches the whole lookback window with pagination instead of many date
    chunks.
  - Retries transient/rate-limit failures with bounded backoff.
  - Discards all pages from an incomplete fetch.
  - Reconciles the recent cache window only after complete pagination.
  - Preserves confirmed flows, trades, close days, and `last_update_date` after
    a failed attempt.
  - Writes the cache with `os.replace`.
  - Persists `sync_health` with attempt time, last success, range, page/order
    counts, and error.
  - Removes unused Transactions API traffic from this calculation path.
- `accounts/accounts_bo.py`
  - Displays whether cash-flow history is freshly synchronized or is showing the
    last confirmed history after a delayed sync.
- `tests/test_spy_position_tracker_sync.py`
  - Covers complete pagination, transient retry, failure after a successful
    first page, and preservation of confirmed cache data.

### Remaining risk

E*TRADE or the network can still be temporarily unavailable. In that case, new
activity cannot appear until a later successful refresh. The corrected behavior
is deliberately stale-but-confirmed: history no longer regresses, the failure is
recorded, and the dashboard labels the delayed sync. A truly flat green line can
also be legitimate when same-day inflows and outflows net to zero; inspect trade
details or add a separate gross-activity series if trade frequency itself needs
to be visualized.

### Verification record

- Unit regression suite: 7 tests passed in both the system Python and project
  virtualenv runtimes on 2026-07-17.
- Live service: launchd generation restarted successfully and loaded the new
  code. Its startup refresh fetched 27 orders in one complete page, retained 344
  cached days and 181 close-event days, and persisted `sync_health.status=ok`.
- Exact `/api/positions` artifact: opened through the authenticated in-app
  browser. Response delivery was `no-store`; served HTML contained the
  `data-sync-status="ok"` badge reading “Cash-flow orders synced Jul 17, 2026
  12:12 PM ET · 27 orders.”
- Three-month canvas: visually inspected with the 3month control active. It
  rendered 63 points from 2026-04-17 through `2026-07-17 (Live)`. The final
  plotted values were cumulative cash flow $275,369.45, realized gain
  $242,843.45, and margin $662,000.00.
- Final five cash-flow values in the exact served chart matched the cache:
  $268,509.45, $270,559.45, $271,854.45, $273,934.45, and $275,369.45 for
  July 13–17 respectively.

## INC-2026-07-18-01 — Failed portfolio fetch published as zero

### Evidence

The exact live dashboard showed a `2026-07-18 (Live)` portfolio point of $0.00,
a current calculated margin of $0.00, and no positions. The comparison panel in
the same served artifact still showed the last confirmed July 17 snapshot:
30 tracked positions, option value -$36,250.00, and margin $662,000.00.

The service log for the refresh recorded `oauth_problem=token_expired`, followed
by a successful HTML write. `Accounts.portfolio()` stopped pagination on the
non-200 response and returned the empty accumulator. The refresh path then
screened that false-empty result and replaced `screened_option_pairs.html`.

### Implementation

- `accounts/accounts_bo.py`
  - Retries a token-expired portfolio page after invoking the shared E*TRADE
    authentication refresh callback.
  - Supports `require_success=True`, which raises on a non-200 portfolio page
    instead of returning a false empty result.
- `live_trading/etrade_cover_call_new.py`
  - Uses the required-success mode for manual, startup, after-hours, outside
    trading-window, tracker, and position-management portfolio refreshes that
    can lead to a production dashboard write.
- `tests/test_dashboard_quotes.py`
  - Covers successful portfolio retry after OAuth expiry and fail-closed
    behavior when an unsuccessful portfolio response remains.

### Remaining risk

If OAuth renewal or E*TRADE remains unavailable, the refresh reports an error
and cannot show new portfolio activity until a later successful attempt. The
intended behavior is to keep serving the last confirmed HTML rather than publish
zero. A successful 200 response containing a genuinely empty portfolio remains
valid and can still render zero.

### Verification record

- Regression suite: 9 focused dashboard/reliability tests passed in the project
  virtualenv on 2026-07-18; the changed Python files also compiled successfully.
- Live service: launchd restarted from PID 2184 to PID 66689 after the source
  changes. Both its startup refresh and a second Refresh Data request completed
  with `CLOSED_HOLIDAY` and regenerated the production artifact.
- Exact served dashboard: after the post-restart refresh, the portfolio table
  showed total option price -$51,442.00 and calculated margin $662,000.00. The
  tracked benchmark's final label was `2026-07-18 (Live)` with option value
  -$31,347.50; SPY and VIX values were null rather than fabricated weekend
  quotes.
- Benchmark canvas: visually inspected with 1month active. Hovering the final
  plotted point displayed `2026-07-18 (Live)` and -$31,347.50.
- Cash-flow/margin canvas: visually inspected with 1month active. Hovering the
  final point displayed `2026-07-18 (Live)`, cumulative cash flow $275,369.45,
  realized gain $246,298.45, and total margin $662,000.00. These matched the
  generated chart arrays, `spy_gains_cache.json` (complete 28-order sync and
  cumulative cash flow), and `spy_tracking_data.json` (realized gain and the
  confirmed $662,000 margin).

## INC-2026-07-24-01 — VIX benchmark history stopped early

### Evidence

The authenticated one-month `SPY Benchmark Performance` chart showed SPY and
option-value points through July 24, but its orange VIX line stopped after July
6. The production `spy_vix_price_cache.json` confirmed the split: SPY and SPX
had closes through July 23, while the last non-null VIX close was July 6.

The refresh decision in `accounts/accounts_bo.py` considered only missing SPY
and SPX dates. Because those two series were complete, the VIX-only gap was
reported as fully cached and no history request ran. After VIX was added to the
missing-date check, Yahoo returned malformed responses and zero VIX rows,
showing that a second independent source was also required.

### Implementation

- `accounts/accounts_bo.py`
  - Adds `_missing_market_close_dates()` so SPY, SPX, and VIX each participate
    in the refresh decision.
  - Preserves the existing cache when a source fails.
  - Uses Cboe's official daily VIX history CSV for requested VIX closes that
    Yahoo did not return.
  - Parses and merges only the requested Cboe dates.
- `tests/test_dashboard_quotes.py`
  - Covers a VIX-only cache gap when SPY and SPX are complete.
  - Covers parsing a requested close from the Cboe history format.

### Remaining risk

If both Yahoo and Cboe are temporarily unavailable, the dashboard continues to
show the last confirmed VIX history until a later successful refresh. The
intraday live label may legitimately have null SPY/VIX closes because the chart
uses confirmed daily closes; it must not fabricate a value for that point.

### Verification record

- Focused regression suite: 11 tests passed; the changed Python files also
  compiled successfully.
- Live service: launchd generation restarted on PID 67172 and completed refresh
  generation 4 without an error. The refresh detected 13 missing VIX trading
  dates and populated all 13 through July 23.
- Exact `/api/positions` artifact: authenticated response returned 200 with
  `Cache-Control: no-store, max-age=0`; all four benchmark arrays contained 309
  aligned entries.
- Source comparison: rendered and cached VIX closes for every restored date
  from July 7 through July 23 matched the Cboe daily history with zero
  mismatches. The restored values were 16.13, 16.90, 15.84, 15.03, 17.16,
  16.50, 15.67, 16.73, 18.77, 18.65, 17.05, 16.64, and 18.70.
- One-month canvas: visually inspected in the authenticated live dashboard with
  `1month` active. The dashed orange line visibly continued through the latest
  historical point. Hovering that point displayed `2026-07-23`, option value
  -$42,787.50, SPY close $738.18, and VIX close 18.70.
- Final chart label: `2026-07-24 (Live)` displayed option value -$39,100.00 with
  SPY and VIX null, as expected for an intraday live point without confirmed
  closes.

## INC-2026-07-26-01 — Regime output could not distinguish climate from shocks

### Evidence

The legacy HMM trace emitted the same raw state on every saved row, while its
final overlay ignored the HMM and used fixed absolute thresholds. The active
dashboard could also fall back to a stale trace when regime computation was
disabled. This made a persistent late-March/early-April stress climate and a
quieter June/July period with isolated VIX jumps appear operationally similar.

The local normalized replay contains two VIX-only non-NYSE rows and one
conflicting overlapping cached value. It has no retained provider bytes or
decision-time receipts, so it cannot establish production provenance.

### Implementation

- `live_trading/regime_detector_v2.py`
  - Separates `calm/elevated/persistent_stress` background state from
    `none/active/aftershock` event state.
  - Uses causal prior-only percentiles, explicit next-session availability,
    immutable configuration hashes, a portable detector/clock source hash,
    and a separate exact runtime fingerprint.
- `live_trading/regime_calibration.py`
  - Freezes a small candidate grid and rejects shock-lane tuning.
  - Uses purged 2016–2024 folds and only fully resolved post-lag future-risk
    labels.
  - Makes coverage, switching, aggregate occupancy, state diversity, and
    isolated-shock false-persistence checks binding selection guardrails.
  - Requires exact plan, price-prefix, calendar, source-policy, build, and
    deployment-pinned artifact hashes.
  - Rejects provider-provenance claims from loose DataFrames and keeps every R3
    artifact research-only and execution-ineligible.
- `docs/regime_v2_calibration_plan.json`
  - Commits the canonical protocol before the prospective observation window
    beginning 2026-07-27.
- `research_reports/regime_v2_calibration_artifact.json`
  - Records the deterministic legacy-data evaluation and selected baseline.
- `live_trading/regime_signal.py`
  - Defines an immutable two-axis signal with canonical hashing, exact
    close-T-to-next-session timing, artifact/evidence lineage, the mandatory
    causal record, and a hard-coded non-authoritative action boundary.
- `backtesting/backtest_runner.py`
  - Accepts V2 only as an exact-date audit annotation and persists it without
    consulting it for trade behavior.
  - No longer backfills later HMM probabilities into earlier dates.
  - Requires empirical forward-return outcomes to resolve strictly before the
    entry session.
- `live_trading/regime_shadow_publish.py`
  - Requires an explicit entitlement capability before provider I/O.
  - Publishes only the exact newly verified decision-time channel head and a
    non-unavailable tail signal; all failures preserve the prior sealed file.
- `live_trading/regime_shadow_store.py`
  - Uses owner-only, descriptor-relative, no-follow reads and atomic
    replacement for the sealed V2 dashboard read model.
  - Expires the signal at its exact effective-session joint finalization and
    emits only a fixed redacted, non-authoritative dashboard schema.
- `live_trading/etrade_cover_call_new.py` and `dashboard_template.html`
  - Add a separate authenticated, same-origin, no-store
    `/api/regime_v2_shadow` endpoint.
  - Render the V2 background and shock axes with text-only DOM updates and a
    permanent `CANNOT AUTHORIZE EXECUTION` label; the endpoint never enters
    GEX, HMM, EV, account, refresh-queue, or order paths.

### Remaining risk

The selected profile is not approved for E*TRADE orders. Its calibration data
are explicitly `legacy_normalized_unverified`; the provider entitlement gate is
unresolved; the prospective window has not accumulated; no current verified
evidence database or sealed production signal exists in this workspace; the
user's deployed dashboard process has not been restarted and inspected with
this source; and no centralized fail-closed risk engine consumes the signal.
The correct operational result remains `Unavailable`,
`Calibration_Abstain=true`, and `Execution_Eligible=false`.

### Verification record

- Plan SHA-256:
  `e40f762f8adadd871ec0a6919fc6e6e2e99ef800a39bad365e88e00c862c2299`.
- Artifact SHA-256:
  `95bd8b003a7b4e3ed13ab077031b392e679dc6aaf007131a07b0d0980fa55de4`.
- Selection retained `b0_baseline`: score 0.3810, 100% evaluation coverage,
  6.01 switches per 252 sessions, maximum aggregate occupancy 0.516, and two
  false persistent transitions across 60 qualifying isolated shocks.
- The baseline led the 2025 retrospective candidates with tail F1 0.4898.
- March 20–April 7 produced 12/12 `persistent_stress` sessions. June 15–July
  24 produced 22 `elevated`, six `calm`, and five independently active shocks.
- Unit coverage includes canonical/tamper checks, strict purging, exact
  post-lag outcomes, prefix invariance, stale build/data/pin rejection, and
  binding coverage/chatter/false-persistence guards.
- R4 coverage verifies closed two-axis types, canonical signal hashes,
  artifact pins, stripped-attribute rejection, exact Friday-to-Monday mapping,
  no-fill lookup, numeric-HMM separation, audit serialization, and strictly
  prior return-bucket resolution.
- R5 focused coverage verifies entitlement-before-network, exact verified-head
  publication, old-file preservation, descriptor/race/path protections,
  tamper and fixed-schema redaction, exact effective-session staleness,
  authentication, restrictive CORS, endpoint isolation, and text-only DOM
  rendering.
- An isolated instance of the actual `RefreshHandler` and source-served
  `dashboard_template.html` was inspected in the in-app browser on
  2026-07-26. A sealed synthetic Friday-to-Monday fixture visibly rendered
  `Elevated`, `Active news/volatility shock`, effective session `2026-07-27`,
  `Shadow advisory`, and `CANNOT AUTHORIZE EXECUTION`. Restarting the isolated
  handler without a signal visibly rendered both axes and status as
  `Unavailable`, no effective date, and
  `No verified shadow snapshot has been published.` The isolated process did
  not start the trading loop or create an E*TRADE session. This verifies the
  code path and HTML, not deployment into the user's currently running service.

### CI portability incident

The first R3 draft bound the detector/build hash to exact Python and dependency
versions. The frozen plan was generated under Python 3.10.5 and therefore
failed deterministically on GitHub's Python 3.10.20 runner even though the
committed algorithm and market-clock source were identical. Build identity is
now the portable hash of those two source modules; the exact Python, NumPy,
pandas, and calendar versions remain recorded in a separate runtime
fingerprint. A future promotion gate must validate both a reviewed source hash
and an approved runtime/lockfile identity, but a patch release cannot rewrite
the research protocol identity.

## INC-2026-07-26-02 — Live startup could fail open and source exposed OAuth credentials

### Evidence

Before R6, omitted mode input could resolve to the production E*TRADE endpoint,
the selected brokerage account depended on mutable list position `1`, and there
was no independently signed, expiring production arm. The same process could
bind its order-capable dashboard to all interfaces, start ngrok automatically,
and emit wildcard CORS. Local settings and request logs were not consistently
created with owner-only protections, and the existing local dashboard
credentials do not meet the new minimum-strength policy.

Two tracked legacy E*TRADE utilities also contained hardcoded OAuth credentials.
Those literals have been removed from the current source, but the repository is
currently public. The affected keys must therefore be treated as compromised;
their presence in Git history is security evidence, not a resolved source-only
finding.

### Implementation

- `live_trading/runtime_safety.py`
  - Requires an explicit `sandbox` or `production` environment before OAuth
    construction.
  - Requires production to name the exact account ID, account key, and
    institution type.
  - Validates an owner-only, HMAC-signed, versioned arm document bound to that
    identity, with issued/expiry timestamps and a maximum 15-minute lifetime.
  - Provides an operator CLI that creates the arm atomically with mode `0600`
    and reads the signing secret only from
    `ETRADE_PRODUCTION_ARMING_SECRET`.
  - Rejects default dashboard usernames, passwords shorter than 16 characters,
    and action PINs shorter than eight digits.
- `accounts/accounts_bo.py` and
  `live_trading/etrade_cover_call_new.py`
  - Select production accounts by exact identity rather than list position.
  - Revalidate the account identity and current arm at startup and account
    refresh.
  - Bind the dashboard to loopback, do not launch ngrok automatically, and do
    not return wildcard CORS.
  - Use owner-only handling for local settings and touched logs and redact
    dashboard request records.
- `order/order_bo.py`, `order/order.py`, and
  `core_api/stock_trade_class.py`
  - Pass the immutable runtime boundary into the reusable order client.
  - Revalidate the current arm, environment, and exact account immediately
    before every order API POST/PUT.
  - Reject order API access without that boundary and quarantine the older
    interactive order client.
- Shared E*TRADE client logging
  - Uses owner-only files, does not propagate to unfiltered root handlers,
    redacts authorization-bearing headers, and records only SHA-256/size
    metadata for order payloads, account/order response bodies, and
    account-bearing URLs.
- `live_trading/order_intent_ledger.py`,
  `live_trading/etrade_broker_transport.py`, and
  `live_trading/etrade_order_gateway.py`
  - Persist immutable intent, reservation, authorization, exact send, preview,
    and parsed-response evidence.
  - Permit only opening vertical submission and price-only amendment through a
    no-retry, no-redirect, isolated mutation exchange.
  - Keep every unknown no-ID result permanently blocked and require a known
    broker ID plus exact canonical order-term hash before reconciliation.
  - Revalidate the exact runtime/account/environment around each mutation and
    enforce a gateway-owned account risk ceiling.
  - Remain intentionally disconnected from the live agent until the read side
    and migration gates below are complete.
- `live_trading/etrade_check_option.py` and
  `live_trading/etrade_option_chains.py`
  - Remove hardcoded OAuth credentials from the current source and require
    local configuration or environment variables.

### Remaining risk

This is containment in the current source, not production readiness. The
compatibility order client still owns current live calls. The isolated R7a–R7c
stack now provides a schema 10 intent ledger, hardened mutation transport, and
opening/reprice coordinator, but no live code instantiates it.

The next gate is a concrete private E*TRADE reader with complete pagination,
stable two-pass capacity snapshots, origin-bound durable raw/parser receipts,
and atomic reconciliation evidence. Position absorption, closing capacity,
durable per-intent cancellation, a single production composition root, and a
static ban on every direct legacy mutation path also remain required.

The exposed OAuth keys still require external revocation and rotation. A later
coordinated history purge must remove them from all refs and arrange cleanup of
downstream clones, forks, caches, and build artifacts; making the repository
private now would not undo prior exposure. The repository remains public.

The current local dashboard settings intentionally fail the new credential
policy and must be reprovisioned before startup. No live service has been
restarted, no E*TRADE session or order path has been exercised, and the exact
deployed dashboard has not been reloaded or visually inspected with R6.

### Verification record

- R5 / PR #28 passed its clean CI job with 134 tests. This establishes the
  clean dependency/import baseline for the preceding shadow-dashboard change;
  it is not deployment evidence for R6.
- R6 includes focused runtime-safety coverage for explicit environments,
  production-arm schema/signature/lifetime, exact account matching, owner-only
  files, guarded response/request logging, order calls without a boundary,
  placement-time arm expiry, dashboard credential rejection, and loopback/CORS
  behavior.
- The current R7 ledger/transport/coordinator focused suite includes 88
  deterministic tests for immutable identity, exact authorization/XML,
  transport deadlines, send fencing, parsed receipts, crash/timeout recovery,
  capacity arithmetic, gateway-owned risk ceilings, environment/account
  binding, exact order-term reconciliation, amendment replay, terminal-risk
  retention, and additive schema 8→9→10 migration. Independent adversarial
  reviews found no remaining reproducible P0/P1 mutation-state defect in the
  isolated stack; they explicitly retain a no-go on live wiring until the
  durable reader and remaining protocols are delivered.
- Deployment verification is deliberately recorded as incomplete. This
  incident remains open until the remaining-risk conditions above are
  satisfied.

## Checklist for future dashboard incidents

- [ ] Capture the exact URL, selected time span, viewport, and screenshot.
- [ ] Identify the authoritative upstream data for the incorrect element.
- [ ] Check the latest successful sync separately from the latest attempted sync.
- [ ] Check counts and pagination completeness before comparing calculations.
- [ ] Confirm that cache rows never decrease after an incomplete fetch.
- [ ] Compare raw executions -> daily net -> cumulative values.
- [ ] Regenerate through the live endpoint.
- [ ] Verify response caching/generation behavior.
- [ ] Inspect the rendered DOM/canvas and record the final values.
- [ ] Add or update a regression test.
- [ ] Update this ledger with evidence, invariant, fix, residual risk, and result.
