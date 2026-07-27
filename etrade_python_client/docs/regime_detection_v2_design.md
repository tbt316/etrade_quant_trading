# Regime Detection V2: Persistent Climate + Event Shock

**Review date:** 2026-07-26

**Status:** R5 entitlement-gated shadow publisher and visually verified
dashboard read model; not connected to order execution
**Related roadmap:** `docs/production_readiness_upgrade_plan.md`

## Decision

The current regime mechanism should not be repaired by tuning another
three-state Gaussian HMM. The central modeling error is that one latent label is
being asked to represent two processes with different time scales:

1. A persistent volatility climate that lasts days or weeks.
2. A fast news/event shock that can appear and mean-revert within one session.

The replacement should return two independent states:

```text
background_state = calm | elevated | persistent_stress
shock_state      = unavailable | none | active | aftershock
```

This produces combinations that are both statistically and operationally
meaningful:

- `calm + none`
- `calm + active`
- `elevated + intermittent shocks`
- `persistent_stress + active`

For the market behavior in the user example, this is the difference between a
multi-week stress climate and a calmer climate that still experiences isolated
VIX jumps.

The first production candidate should be a small, causal, interpretable score
and state machine. A robust duration model can be evaluated later as a
challenger. This sequence makes failures observable and keeps model output
auditable while the repository's market-data clock is repaired.

## What the 2026 tape says

The local audit combines the longer SPY/VIX Parquet history with the fresher
dashboard cache and gives the dashboard cache priority on overlapping dates.
The latest observation is the July 24, 2026 close.

The cache does not persist trustworthy per-symbol `available_at` provenance.
The replay therefore wraps normalized cache values and historical finalization
assumptions in an immutable evidence snapshot marked `UNVERIFIED`; its results
are diagnostic and explicitly ineligible to drive a live order. A filesystem
modification time is never treated as market-data provenance.

| Window | Observed behavior | Legacy overlay | V2 shadow result |
|---|---|---|---|
| Mar 20–Apr 7, 12 sessions | VIX median 25.97, maximum 31.05; mean V2 background score 0.916 | 9 `cautious_decline`, 3 `expansion` | 12/12 `persistent_stress`; 3 fresh shocks |
| Full April, 21 sessions | VIX median 18.92; early stress followed by rapid normalization | 20 `expansion`, 1 `cautious_decline` | 18 `persistent_stress`, then 3 `elevated` because exit requires confirmation |
| Jun 15–Jul 24, 28 sessions | VIX median 16.81, maximum 19.49; five VIX increases above 10% | 28/28 `expansion` | 22 `elevated`, 6 `calm`; all five jumps detected |
| Jul 9–Jul 24, 12 sessions | Mean background score 0.494; three VIX increases above 10% | 12/12 `expansion` | Jul 17 and Jul 23 were `calm + active`; Jul 24 was `calm + aftershock` |

The recent raw shock dates were:

| Date | VIX close | Daily VIX change | V2 background | V2 shock |
|---|---:|---:|---|---|
| 2026-06-17 | 18.44 | +12.37% | elevated | active |
| 2026-06-23 | 19.49 | +12.79% | elevated | active |
| 2026-07-13 | 17.16 | +14.17% | elevated | active |
| 2026-07-17 | 18.77 | +12.19% | calm | active |
| 2026-07-23 | 18.70 | +12.38% | calm | active |

The user's intuition is therefore supported, with one useful refinement: the
stressed interval is concentrated in late March and early April. April as a
whole is transitional rather than one homogeneous regime. This matches the
broader contemporaneous record: the IMF reported financial-market volatility
and economic disruption on March 3, and Cboe described an outsized volatility
and hedging-demand increase on March 9. By July, Cboe described a calmer market
that was often discounting the conflict even while intraday volatility
continued.

These dates are used only to test whether the mechanism can express the desired
distinction. They are not a locked out-of-sample validation set, and the
prototype results must not be presented as investment-performance evidence.

## Why the current system returns the same answer

### 1. The active dashboard is not running the regime model

The latest restart log launches the dashboard with `--no-regime`
(`live_dashboard_restart.log:22`). In that mode the regime card can fall back to
`research_reports/regime_diagnostics/causal_regime_trace.csv`
(`live_trading/etrade_cover_call_new.py:1622-1636,1707-1753`).

That artifact ends on 2026-05-22. A refresh can therefore repeat a stale answer
even when the live market has changed. A production dashboard must display
`Unavailable — model disabled` or `Stale — last signal 2026-05-22`; it must not
present a stale row as current.

### 2. The final actionable label ignores the HMM

`_apply_causal_stress_overlay()` initializes every row as `Expansion` and
changes it only when fixed VIX, SPY return, or SPY drawdown thresholds fire
(`live_trading/ev_engine.py:524-586`).

The final classification does not use HMM state or probability.
`base_state_count` is unused. A VIX move from 16 to 18 is still below the
absolute VIX level of 25, so it remains `Expansion` even when the one-day
percentage move is extreme.

### 3. The feature pipeline erases the event the user wants to detect

Every feature whose name contains `VIX`, `VVIX`, or `Vol` is replaced by a
five-row trailing median (`live_trading/data_ingestion.py:972-979`). This filter
is causal, but it deliberately suppresses a one-day pulse.

On June 5, the raw VIX close was 21.51 after a 39.7% daily increase. The
five-row median was only 16.05, suppressing roughly one quarter of the raw level
on the exact day that should have fired the event channel.

Smoothing is appropriate for the slow background lane. It must never overwrite
the raw series used by the shock lane.

### 4. The saved HMM is statistically collapsed

`research_reports/regime_diagnostics/causal_regime_trace.csv` contains 1,260
rows from 2020-01-03 through 2026-05-22. On every row:

```text
Raw_HMM_State = 2
prob_state_0  = 0.0
prob_state_1  = 0.0
prob_state_2  = 1.0
```

All 117 recorded refits contain a pair of duplicate emission means. The model
has not learned three distinguishable regimes.

The code already contains useful occupancy, posterior-entropy, and
state-separation checks in `validate_hmm_quality()`
(`live_trading/ev_engine.py:58-145`). However, a fresh fixed-K fit returns after
only the numerical `is_hmm_healthy()` check
(`live_trading/ev_engine.py:460-484`). The statistical quality gate is applied
to warm starts but bypassed by the fresh fallback path. A numerically valid,
one-state solution can therefore be cached as a healthy three-state model.

### 5. Research and cached/live inference use different inputs

The direct expanding training path applies the final overlay to the raw caller
frame (`live_trading/ev_engine.py:1081`). Cached scoring applies it to the
stationary, median-filtered feature frame
(`live_trading/ev_engine.py:1557`).

The same date can consequently receive different labels depending on which
entry point generated it. A production feature contract must explicitly carry
both `raw_*` and `slow_*` values and use the same versioned detector in research,
backtest, shadow, and live paths.

### 6. The VIX3M cache is invalid

The local `INDEX_VIX3M.parquet` close is exactly equal to
`INDEX_VVIX.parquet` on all 229 overlapping rows from 2025-06-02 through
2026-04-29. The values are roughly 90–100 when the trusted VIX3M report is near
20. VIX3M also stops on 2026-04-29.

The cache coverage rule treats symbols such as `^VIX3M` as macro-like because
of their length and allows a 90-day stale buffer
(`live_trading/data_ingestion.py:177-186`). The merged dataset is then broadly
forward-filled.

Until the series is refreshed and passes identity, range, timestamp, and
freshness checks:

- VIX3M must be quarantined.
- Term-structure features must be marked unavailable.
- The detector must not substitute VVIX or a stale forward-filled value.
- Missing term structure must degrade confidence, never default the regime to
  `Expansion`.

### 7. The market clock is not trustworthy

The saved trace has weekend refit dates and almost no Mondays. Multi-calendar
data are outer-joined and forward-filled, then rows are removed by a later
all-column `dropna()` (`live_trading/data_ingestion.py:418-424,930,793-854`).
As a result, “three rows,” “five rows,” and “21 rows” do not always mean the
same number of exchange sessions.

The current dashboard cache also contains VIX-only observations on two NYSE
holidays: 16.59 on 2026-05-25 and 16.78 on 2026-06-19, with SPY missing. The V2
audit explicitly quarantines and reports those non-session rows. It never drops
a missing observation on a valid NYSE session; that condition fails closed.

All detector features must be computed on a canonical NYSE session index.
Every source observation needs `event_time`, `available_at`, `source`, and
freshness metadata.

### 8. Downstream consumers mix taxonomies

Regime return buckets are grouped by raw HMM state
(`live_trading/ev_engine.py:1657-1666`), while the backtester prefers final
overlay state (`backtesting/backtest_runner.py:136-172`). A numeric state ID in
one taxonomy is not semantically compatible with the same integer in another.

There is also a safety inversion: `regime_dynamic_delta` changes the short-put
delta from -0.10 to -0.20 in crisis
(`backtesting/backtest_runner.py:2164-2171`). Stress must reduce or block short
convexity exposure, not increase it.

## Recommended architecture

```mermaid
flowchart LR
    RAW["Raw SPY / VIX / optional intraday data"] --> CLOCK["NYSE clock + freshness + provenance gates"]
    CLOCK --> SLOW["Slow lane: 10d VIX median, realized vol, drawdown"]
    CLOCK --> FAST["Fast lane: raw VIX jump, SPY tail move"]
    SLOW --> BG["Background state machine"]
    FAST --> SHOCK["Immediate shock + aftershock decay"]
    BG --> PACKET["Versioned evidence packet"]
    SHOCK --> PACKET
    PACKET --> SHADOW["Shadow log and dashboard"]
    SHADOW --> VALIDATE["Walk-forward statistical validation"]
    VALIDATE --> POLICY["Separately approved risk policy"]
```

The detector produces evidence. A separate risk policy decides whether that
evidence blocks or reduces an order. This prevents statistical-model code from
directly mutating execution behavior.

## V2 transparent baseline

The new shadow module is
`live_trading/regime_detector_v2.py`. It intentionally has no HMM, PCA, network
call, cache mutation, or broker dependency.

### Slow background features

For close T:

```text
VIX_slow(T) = median of the last 10 VIX closes
RV10(T)     = annualized standard deviation of the last 10 SPY log returns
DD21(T)     = current SPY drawdown from the trailing 21-session high
```

Each value is ranked against the previous 756 sessions, excluding T:

```text
score_raw(T) =
    0.45 * percentile(VIX_slow(T))
  + 0.35 * percentile(RV10(T))
  + 0.20 * percentile(DD21(T))
```

The score receives a causal EWMA with half-life two sessions. The score is an
ordinal stress score, not a calibrated probability.

At least 252 prior valid observations of each slow feature are required. The raw
price history must be longer because rolling-feature warm-up precedes that
calibration history. Before then, the background result is `unavailable`, not
`calm`.

### Absolute stress evidence

A relative percentile alone must not label the quietest state in a calm sample
as crisis. Entry into `persistent_stress` therefore also requires at least one
absolute condition:

```text
VIX_slow >= 22
or RV10 >= 20% annualized
or DD21 >= 6%
```

These are prototype risk anchors. Their final values must be selected in a
frozen calibration period and tested in later walk-forward folds.

Once a valid term-structure source exists, sustained `VIX / VIX3M >= 1` can be
added as another absolute stress condition. It must not be enabled against the
current cache.

### Hysteresis

The baseline applies asymmetric confirmation:

```text
enter persistent_stress: 3 closes with score >= 0.75 and absolute evidence
exit persistent_stress:  5 closes with score < 0.65
enter calm:               3 closes with score <= 0.45 and no absolute evidence
exit calm to elevated:    3 closes with score > 0.55
otherwise:                elevated
```

The longer exit condition prevents a single relief day from declaring a crisis
over. Absolute stress evidence moves `calm` to at least `elevated` immediately,
even before persistent-stress confirmation. State age is recorded for every
row.

### Fast shock features

The shock lane uses unsmoothed observations:

```text
active shock if:
    VIX daily percentage change >= 10%
    or SPY daily log return <= -1.5%
```

The flag fires for session T as soon as both required daily inputs are final. It
is never delayed by background debounce. Two subsequent quiet sessions are
labeled `aftershock`.

This is a daily shock proxy, not a statistically identified price jump. A later
intraday version should use five-minute bars and realized-versus-bipower
variation to separate continuous and jump variation.

The close-based baseline cannot protect an order placed earlier on T from an
intraday shock on T. Before regime evidence can influence live E*TRADE orders,
the centralized risk engine also needs a broker-independent, timestamped
intraday circuit breaker. That gate should consume fresh raw VIX/SPY quotes,
expire automatically, and fail closed on stale or unavailable market data. It
must remain distinct from the next-session research/backtest signal.

### Output contract

Each row contains:

```json
{
  "Background_State": "calm",
  "Background_Score": 0.50,
  "Background_Regime_Age": 5,
  "Shock_State": "active",
  "Shock_Age": 1,
  "Composite_Regime": "calm+active",
  "Absolute_Stress_Evidence": false,
  "Data_Quality": "exchange_sessions_and_source_times_valid",
  "Market_Close_At": "2026-07-23T20:00:00+00:00",
  "VIX_Finalization_At": "2026-07-23T20:15:00+00:00",
  "Signal_Available_At": "2026-07-23T20:16:00+00:00",
  "Tradable_Session": "2026-07-24",
  "Detector_Version": "regime_v2_shadow_0.3.0",
  "Config_Hash": "<sha256>",
  "Input_Snapshot_SHA256": "<sha256>",
  "Input_Schema_Version": "regime_market_data.v2",
  "Calendar_Policy_Version": "nyse+cboe_index_options.v1",
  "Source_Policy_Version": "regime_source_identity.v1",
  "Source_Policy_SHA256": "<sha256>",
  "Input_Provenance_Status": "verified | unverified",
  "Input_Provenance_Evidence": "durable_raw_payload_and_parser_receipts_verified | complete_but_not_durably_verified | incomplete",
  "Evidence_Manifest_SHA256": "<sha256> | null",
  "Evidence_Verification_Kind": "decision_time | verified_replay | null",
  "Evidence_Verified_At": "<UTC timestamp> | null",
  "Evidence_Decision_Time_Eligible": true,
  "Detector_Stage": "shadow",
  "Execution_Eligible": false,
  "Reason_Codes": "vix_daily_change_extreme",
  "Regime_Signal_Timestamp": "after_spy_vix_finalization_T_for_next_session"
}
```

The typed shadow entry point accepts only a
`RegimeMarketDataSnapshot`. It derives provenance from immutable source
metadata rather than accepting a caller-supplied Boolean. Each observation
carries provider/dataset/symbol/field identity, payload checksum and kind,
event time, availability time, ingestion time, request ID, finality, and a
frozen source-policy verdict. The snapshot binds the version and hash of that
policy, verifies exact contiguous NYSE sessions, one independent SPY and VIX
observation per session, recognized identities, deterministic input and
calendar hashes, and timezone-aware clock ordering.

R2 does not let the snapshot or caller certify provenance. The
`detect_regimes_from_verified_snapshot(snapshot, evidence_store)` boundary
calls `evidence_store.verify_snapshot(snapshot)` itself, binds the resulting
manifest to that exact snapshot hash, and rejects incomplete evidence. It does
not accept a caller-provided Boolean or report. Only this retained-byte reparse
path can emit the `verified` variant above. The ordinary snapshot, loose-array,
and legacy-cache paths emit `unverified` and remain execution-ineligible.

`decision_time` means the evidence was parsed no later than the snapshot's
recorded decision time. `verified_replay` proves byte/parser lineage but does
not prove that lineage existed at the historical decision time; it sets
`Evidence_Decision_Time_Eligible=false`, adds a data-quality warning, and must
be rejected by future backtest or live promotion gates. Both variants remain
execution-ineligible in R2.

SPY event time comes from the versioned NYSE schedule. VIX event time comes
from the versioned `CBOE_Index_Options` schedule, including shortened sessions;
the paired publication boundary is the later of the two exchange closes. The
final row must be the latest jointly finalized session at `as_of`.

`Signal_Available_At` is the cumulative maximum of both source timestamps
through T. This dependency watermark matters because returns, rolling features,
the EWMA, and the state machines all consume prior rows. `Tradable_Session` is
the first subsequent NYSE session whose market open is strictly after that
watermark. If either the current row or a required historical row arrives after
the next session has opened, the signal rolls to the following session instead
of being backdated into that morning.

The compatibility entry point for loose arrays is research-only and rejects
any attempt to claim verified provenance. Unverified provenance is carried in
`Data_Quality`, makes `freshness_assessed=false`, and must hard-block any
regime-dependent execution. Both entry points currently emit
`Execution_Eligible=false`; promotion requires a later, separately reviewed
risk-policy release. Invalid, missing, non-finite, nonpositive, non-session,
stale, premature, duplicated, or tampered required data fail closed.
Insufficient calibration history returns `unavailable`, and the first row's
shock state is `unavailable` because a daily change cannot yet be observed.
None of these cases silently produces `calm` or `none`.

### R2 provider evidence gateway

R2 closes the durable-lineage gap for the provider-backed shadow path.
`live_trading/regime_market_data_gateway.py` is a library, not a scheduler: it
does not fetch at import time, read a secret configuration, start a worker, or
connect to E*TRADE execution. An explicit caller supplies a transport, clock,
sleep function, and bounded requested range. The production transport requests
one Massive Daily Ticker Summary per NYSE session for SPY (`adjusted=false`) and
one official Cboe `VIX_History.csv` response for the same session range.

For every complete HTTP response, including retryable and non-2xx responses,
the gateway captures the exact **decoded parser-input bytes** before deciding
whether to retry. SQLite stores those content-addressed bytes separately from a
credential-free fetch receipt. A registered strict parser then produces the
chosen observation(s), a parser receipt bound to parser source/configuration,
and stable source locators. The store re-reads the retained bytes and reparses
them when creating the snapshot-evidence manifest; a snapshot is verified only
when every selected SPY and VIX observation links to that manifest. A checksum
alone remains unverified.

SPY and VIX have independent source attempts and outcomes. The gateway never
holds a store transaction over HTTP. It records each captured response, closes
each attempt as success or failure, and can publish only after both legs cover
the same exact contiguous NYSE sessions, satisfy their exchange clocks, and
pass store-owned verification. A one-leg failure retains the prior verified
`shadow` channel head; it must not partially publish or erase confirmed history.

Incremental collection is a **snapshot-start history assembly** process. The
first collection requests a bounded calibration/bootstrap range. Later runs
extend the explicitly recorded `snapshot_start` through the latest jointly
eligible session and assemble a new complete contiguous candidate from the
previous verified history plus new independent observations. A one-session
refresh is therefore not a substitute for the detector's required trailing
history. The resulting publication is still a full immutable snapshot, never a
mutable in-place append.

`available_at` means this collector observed a completed response after the
versioned NYSE/Cboe exchange-close clock. It is not a claim that Massive or
Cboe has supplied a contractual per-row finality guarantee. Provider
corrections are retained as append-only observation revisions; a correction is
eligible only through a newly verified snapshot and cannot rewrite an existing
publication.

The [provider-entitlement gate](regime_data_provider_entitlements.md) remains
unresolved. Until it is satisfied, raw provider bytes stay local, are not
redistributed or rendered, and every provider-backed result remains
shadow/research-only with `Execution_Eligible=false`.

`live_trading/regime_evidence_store.py` is the durable manifest boundary. It
stores source attempts and health, raw BLOBs, fetch and parser receipts,
append-only observation revisions, evidence manifests, and channel-scoped
content-addressed snapshots in SQLite with WAL, `synchronous=FULL`, foreign
keys, restrictive file permissions, and immediate write transactions. Snapshot
reload and verification recheck canonical hashes and stored parser lineage. The
legacy JSON and Parquet caches remain research/display artifacts and cannot be
promoted into verified evidence.

The database must live in a dedicated current-user-owned `0700` directory;
database and SQLite companion files are preflighted without following links and
forced to `0600`. This protects the service boundary from shared-directory and
pre-existing link/special-file attacks. It does not claim isolation from a
malicious process already running as the same OS user, so production deployment
must use a dedicated service identity.

### R3 causal calibration and validation

R3 keeps the transparent two-lane detector and rejects a more complex model
unless it produces a material improvement. The immutable protocol is
`docs/regime_v2_calibration_plan.json`; the deterministic result is
`research_reports/regime_v2_calibration_artifact.json`. Both documents use
canonical strict-schema JSON and content hashes. Runtime research inference
also requires an externally supplied expected artifact hash and verifies the
exact historical price prefix, detector/clock source hash, calibration-engine hash,
exchange schedule, calendar policy, and source policy.

The detector source identity is intentionally portable across machines: it
hashes the detector and market-clock implementation, not the interpreter patch
release. Every output separately records an exact runtime fingerprint covering
Python, NumPy, pandas, and `pandas_market_calendars`. Research replay may run
under a different recorded runtime; any future executable promotion must bind
an approved runtime or lockfile in addition to the portable source identity.

The predeclared candidate set is deliberately small:

| ID | Difference from control |
|---|---|
| `b0_baseline` | 10-session slow VIX and realized-volatility windows; existing thresholds |
| `c1_slow_15d` | 15-session slow VIX and realized-volatility windows |
| `c3_stress_entry_080` | persistent-stress entry score raised from 0.75 to 0.80 |
| `c4_stress_confirm_4d` | persistent-stress confirmation raised from three to four sessions |

All candidates use the same raw-shock thresholds and aftershock duration. The
calibration plan rejects a candidate grid that tunes the shock lane while
selecting the background model.

The statistical target is future market risk, not an invented “regime truth”
label or probability. A close-T signal is first tradable during T+1. Because
the legacy dataset contains only closes, R3 conservatively excludes the
close-T to close-T+1 move and measures:

```text
annualized RMS log-return volatility over close T+1 through close T+h+1
maximum peak-to-trough drawdown over close T+1 through close T+h+1
h in {5, 20}
```

Each measure is converted to an empirical percentile using only the fold's
reference outcomes. A tail event occurs when any measure reaches its
training-only 95th percentile. Reference outcomes must resolve strictly before
the evaluation fold begins; unresolved labels are rejected, never silently
dropped.

Candidate selection uses annual evaluation folds from 2016 through 2024. Each
reference end is purged through the 21-session maximum outcome resolution.
The score is:

```text
tail F1 + 0.25 * max(mean background-risk Spearman, 0)
```

A challenger must improve that score by at least 0.01. Binding guardrails
require at least 120 rows per fold, at least 99% detector availability, no more
than 24 background switches per 252 sessions, aggregate occupancy below 95%,
at least two observed background states, and at least 50 qualifying isolated
shock episodes. Across those episodes, no more than 10% may cause a
`persistent_stress` state during the next five sessions. This last check
measures the behavior the user identified: a discrete news spike must not
silently turn into a persistent background classification.

The 2025 calendar year is a retrospective validation window and is never used
for candidate selection. The March–April and June–July 2026 windows are
retrospective case studies only. The committed prospective observation window
starts on 2026-07-27; it cannot authorize trading and must accumulate before a
later paper-promotion review.

Selection-fold results on the unverified legacy cache are:

| Candidate | Tail F1 | Recall | Precision | Mean Spearman | Selection score | False persistence | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `b0_baseline` | 0.3488 | 0.6500 | 0.2384 | 0.1286 | **0.3810** | 2/60 | selected |
| `c1_slow_15d` | 0.3514 | 0.6615 | 0.2392 | 0.1146 | 0.3800 | 3/61 | rejected |
| `c3_stress_entry_080` | 0.3438 | 0.6308 | 0.2363 | 0.1215 | 0.3742 | 2/60 | rejected |
| `c4_stress_confirm_4d` | 0.3472 | 0.6423 | 0.2379 | 0.1250 | 0.3784 | 2/60 | rejected |

The control also had the strongest 2025 tail F1: 0.4898 versus 0.4242,
0.4583, and 0.4742 for C1, C3, and C4. C1 is especially undesirable for the
stated use case: it labels all 28 sessions from June 15 through July 24, 2026
as `elevated`, eliminating the six `calm` sessions while leaving the same five
active shocks.

The chosen profile reproduces the requested distinction:

- March 20–April 7: 12/12 `persistent_stress`, with three active shocks.
- June 15–July 24: 22 `elevated`, six `calm`, and five independently active
  shocks.

These results establish detector behavior, not investment edge. The source
cache has two quarantined non-NYSE rows and one conflicting overlapping value,
has no retained raw/provider receipts, and is explicitly
`legacy_normalized_unverified`. The artifact therefore has
`promotion_status=research_only`, `Calibration_Abstain=true`, and
`Execution_Eligible=false`.

### R4 typed shadow parity

`live_trading/regime_signal.py` is the only compatibility boundary for V2
consumers. It converts detector rows into frozen, content-addressed
`RegimeSignal` objects without collapsing the two axes into an HMM integer.
Each object is keyed to its explicit effective session; consumers receive no
forward fill and no state fallback.

The adapter validates:

- close-T is a real NYSE session and the effective date is exactly T+1;
- signal availability is no earlier than joint SPY/VIX finalization and
  strictly before the T+1 market open;
- detector version, configuration, source-code identity, runtime fingerprint,
  artifact/plan identity, and any snapshot/evidence hashes remain distinct;
- calibrated traces retain immutable DataFrame attributes, match an externally
  pinned artifact hash, explicitly abstain, and remain execution-ineligible;
- canonical serialization reproduces the signal hash; and
- all attempted action projections raise.

The embedded causal record makes the mandatory audit fields explicit:

| Field | R4 value |
|---|---|
| Training/calibration end | Last selection fold; 2024-12-31 for the committed artifact |
| Test range | Retrospective 2025 range from the pinned plan |
| Inference | `causal_prefix_filter` for the artifact; null and abstaining for raw research traces |
| Regime lag | Exactly one NYSE session |
| Return buckets | `not_used_shadow_annotation` |

`backtesting/backtest_runner.py` accepts these objects only through
`regime_v2_annotations`. It records their primitive, hash-verified envelope on
the daily path and each new trade, but the signal is not consulted by expiration,
strike, delta, quantity, probability, entry, exit, or roll logic. The legacy
numeric HMM lane remains a separate compatibility path. R4 also removes
backward filling of HMM probabilities, requires forward-return outcomes to
resolve strictly before an EOD entry session.

This is semantic parity, not a trading-policy promotion. A later phase must
build verified provider history, finish the prospective holdout, define a
separate monotonic risk policy, and pass paper/canary gates before V2 can veto
or resize an opening order.

### R5 shadow publication and dashboard isolation

R5 adds an end-to-end advisory publication path without connecting V2 to the
legacy HMM, EV, GEX, account, or order paths:

1. `regime_shadow_publish.py` requires an explicit entitlement capability
   before its first provider refresh.
2. The provider gateway must publish a new, verified `shadow` snapshot and the
   returned hash must exactly equal the current verified channel head.
3. The detector must emit a decision-time-eligible, non-unavailable row for the
   requested close session. Only that final row is adapted to a
   `RegimeSignal`.
4. `regime_shadow_store.py` seals the canonical signal to an owner-only file
   using descriptor-relative, no-follow, atomic replacement.
5. The dashboard reads only that file through
   `/api/regime_v2_shadow`. The endpoint is authenticated, same-origin,
   no-store, and separate from `/api/gex`.

The dashboard response is a fixed redacted view model. It exposes only the two
closed state axes, close/effective sessions, whitelisted reason codes, advisory
status, and the literal `may_authorize_execution=false`. It never exposes raw
prices, provider bytes, filesystem paths, evidence hashes, artifact hashes, or
internal exceptions. The renderer uses `textContent`, keeps the legacy lane
unchanged, and displays a permanent `CANNOT AUTHORIZE EXECUTION` badge.

Publication and display both fail closed. A missing, tampered, unsafe,
not-yet-available, or stale signal returns an explicit unavailable model; no
last-good signal is reused. A Friday-close signal remains displayable during
the weekend for Monday, then expires exactly at the Monday SPY/VIX joint
finalization boundary. Failed entitlement, provider, evidence, detector, or
seal checks leave any prior file untouched.

The workspace has no current verified evidence database, and the provider
entitlement gate remains unresolved. Once deployed, this source correctly
displays `Unavailable`; it must not manufacture a current signal from the
unverified legacy caches. An isolated actual-handler browser verification used
a sealed synthetic Friday-to-Monday fixture and confirmed both rendered
states:

- available fixture: `Elevated` background, `Active news/volatility shock`,
  effective session `2026-07-27`, `Shadow advisory`, and the permanent
  execution lock;
- absent fixture: both axes and status `Unavailable`, no effective session,
  and `No verified shadow snapshot has been published.`

The isolated server did not initialize an E*TRADE session or trading loop.
Deploying the source to the user's running dashboard and accumulating real
verified history remain separate operational gates.

## Why not another single HMM

A single latent state must choose among three bad behaviors:

1. Switch on every outlier and destroy regime persistence.
2. Become very sticky and miss genuine structural breaks.
3. Create cross-product states such as calm/shock and stress/shock, fragmenting
   a limited sample and making labels unstable after refits.

The repository currently exhibits the second failure, followed by a hard
threshold overlay that hides it.

A factorial HMM could represent two latent causes formally, but it adds
identification and inference complexity without solving the source-clock and
quality problems. The explicit two-lane design captures the important
separation while remaining testable and explainable.

## Model challengers after the baseline is trusted

### 1. Duration-aware robust background model

Compare the transparent state machine against a fixed three-state hidden
semi-Markov model:

- Student-t emissions rather than Gaussian emissions.
- Explicit duration distributions instead of an implicit geometric dwell time.
- Two or three slow, interpretable features; no PCA initially.
- Fixed semantic anchors and emission-distance alignment across refits.
- Mandatory out-of-sample occupancy, entropy, KL-separation, and duplicate-
  centroid gates.

The challenger must beat the baseline on locked statistical metrics before it
can replace it.

### 2. Online change-point evidence

Bayesian online change-point detection can run on the compact background score
and produce a causal posterior over current run length. It should accelerate a
candidate transition when slow features shift together. It should not define
the semantic state by itself.

### 3. Intraday jump decomposition

With trustworthy five-minute SPY data:

```text
RV(T) = sum of intraday squared returns
JumpVariation(T) ~= max(RV(T) - BipowerVariation(T), 0)
```

This supports a statistically grounded event channel and distinguishes
continuous volatility from discrete jumps. Without intraday observations, the
daily VIX/SPY rules must continue to be labeled proxies.

### 4. Valid implied-volatility curve

After VIX3M is repaired, add:

- `VIX9D / VIX3M` for acute short-end stress.
- `VIX / VIX3M` for 30-day versus three-month stress.
- Curve breadth: whether only the front is elevated or the whole curve shifted.
- VVIX as fragility/vol-of-vol evidence, never as a substitute for VIX3M.

The curve can distinguish a short event premium from a durable repricing of
volatility.

## Risk-policy mapping

Detection and execution must remain separate. A conservative short-put policy
would move in only one direction as stress increases:

| Composite condition | Recommended policy behavior |
|---|---|
| data quality or provenance not verified | Hard block regime-dependent execution |
| either axis unavailable | Hard block regime-dependent execution; surface the failed evidence gate |
| calm + none | Strategy's normal limits |
| calm + active/aftershock | Pause new short-vol entries until the shock gate returns to `none`, or require a separately approved tighter limit |
| elevated + none | Reduce size and tighten liquidity/quote-age requirements |
| elevated + active/aftershock | Block new short-vol exposure while the shock gate is not `none` |
| persistent_stress + none | Block or materially reduce new short-put exposure |
| persistent_stress + active/aftershock | Block new short-vol exposure |

This table is a design requirement, not an authorization to change live order
behavior. The unified fail-closed risk engine in the production-readiness
roadmap must own the final decision. Every stress transition must be monotone in
risk: the policy must never increase absolute short-put delta or size when
either axis worsens.

## Validation design

### Mandatory causal record

Every regime experiment must persist:

| Field | V2 prototype status | Production requirement |
|---|---|---|
| Training/calibration end date | Selection folds end 2024-12-31; retrospective outcomes resolve through 2026-02-02; exact folds and hashes are stored | Rebuild on verified provider evidence before paper promotion |
| Test date range | 2025 is retrospective validation; 2026 examples are retrospective case studies; prospective observation starts 2026-07-27 | Accumulate and review the untouched prospective window without changing the frozen plan |
| Inference method | `causal_prefix_filter`, bound in the plan and artifact | Preserve the same method in backtest, shadow, and live consumers |
| Regime lag | Exactly one session; close-T labels begin post-lag outcomes at close T+1 | Assert the same exact trade-entry mapping in R4 |
| Return-bucket causality | Fold reference outcomes resolve strictly before evaluation; post-lag labels resolve at T+h+1 | R4 must use the same resolved-only convention and taxonomy |

The current prototype is therefore **UNVERIFIED for investment claims** even
though its inference mechanics are causal.

### Models to compare

1. Current HMM plus hard overlay.
2. Transparent V2 background only.
3. Transparent V2 background plus shock lane.
4. Duration-aware robust background plus the same shock lane.
5. Optional BOCPD evidence added to models 3 and 4.

### Statistical metrics

- One-, five-, and 20-session realized-volatility forecast loss.
- Brier and log score for future tail-move events.
- Conditional future drawdown and option-loss calibration.
- Detection delay for pre-registered stress windows.
- False persistent transitions after isolated shocks.
- Switches per year, state occupancy, and dwell-time stability.
- Stability of state meanings across refits.
- Data-availability rate and stale/invalid fail-closed rate.

Economic results are evaluated only after the statistical model passes. The
regime-unaware strategy remains the control, and costs, turnover, spread width,
and skipped-trade effects must be included.

### Non-negotiable tests

- Appending future rows cannot alter any earlier feature, score, or state.
- A single +12% VIX day in a calm sample produces `calm/elevated + active`, not
  `persistent_stress`.
- A sustained high-VIX/high-realized-volatility sequence can enter
  `persistent_stress` only after confirmation.
- Shock state fires on the observation date and decays after the configured
  quiet period.
- Empty, missing, non-finite, nonpositive, duplicate, unexpected, missing-
  session, stale, or premature required observations fail closed.
- A background state with absolute stress evidence is never labeled `calm`.
- Unobservable first-return shock evidence is `unavailable`, not `none`.
- Output maps jointly finalized session T to the exact next NYSE trading
  session, including holiday boundaries.
- A 4:01 p.m. ET run cannot publish session T while VIX is still inside its
  calculation window.
- A signal delivered after the next session's open rolls to a later tradable
  session rather than leaking backward into that open.
- A delayed historical dependency propagates its availability watermark to
  every later signal that consumes it.
- Filesystem modification time never upgrades source provenance to verified.
- A checksum without retained raw bytes and a linked parser receipt remains
  unverified.
- Source-policy version, hash, and frozen verdict are bound into the snapshot
  identity.
- A stale observation from an older attempt cannot advance last-success
  freshness for a new attempt.
- Publishing an older backfill cannot regress the latest-as-of snapshot in the
  research or shadow channel.
- Invalid VIX3M cannot enter a term-structure calculation.
- A probabilistic challenger is rejected if any state exceeds 95% OOS
  occupancy, centroids duplicate, posterior entropy collapses, or minimum
  state separation fails.
- Backtests consume close-T output no earlier than the next trading session.
- No `bfill()` may create a historical regime value.
- A more stressed output can never increase short-put exposure.

## Migration plan

### Phase 0 — stop false confidence

- Keep regime-dependent order behavior disabled.
- Change the dashboard fallback from a stale label to explicit
  `disabled/stale/unavailable` status.
- Quarantine the corrupted VIX3M cache.
- Repair the NYSE session clock and source freshness metadata.
- Make the existing HMM quality test mandatory on every fresh and fallback fit.

### Phase 1 — shadow the transparent detector

- Run `regime_detector_v2.py` after each confirmed market close.
- Persist raw inputs, derived features, both state axes, reason codes,
  configuration hash, source timestamps, and publication timestamp.
- Display the shadow result beside the legacy result without affecting orders.
- Alert on stale inputs, unavailable calibration, or publication failure.

### Phase 2 — causal replay and locked validation

- Freeze calibration/validation/test partitions.
- Compare the baseline and challengers using the statistical metrics above.
- Audit 2008, 2011, 2018, 2020, 2022, and 2026 without choosing thresholds from
  the final test windows.
- Publish an experiment manifest and prefix-invariance proof for every run.
- Keep the committed R3 plan unchanged while the prospective window
  accumulates.

### Phase 3 — unify research, backtest, and dashboard semantics

- Replace numeric regime IDs with versioned typed values.
- Use the same detector package and configuration schema in all paths.
- Apply the one-trading-session lag in the backtester.
- Remove raw-HMM/final-overlay bucket mixing.
- Keep the regime-unaware strategy as the default and control.

### Phase 4 — risk integration

- Integrate only through the centralized fail-closed `RiskDecision`.
- Start in paper mode, then shadow live decisions, then a bounded canary.
- Persist what the order would have been without the regime rule.
- Require rollback to disable the regime policy without disabling monitoring.

## Prototype artifacts and verification

- Detector: `live_trading/regime_detector_v2.py`
- Evidence contract: `live_trading/regime_market_data.py`
- Evidence store: `live_trading/regime_evidence_store.py`
- Provider parser contract: `live_trading/regime_provider_evidence.py`
- Provider gateway: `live_trading/regime_market_data_gateway.py`
- Calibration engine: `live_trading/regime_calibration.py`
- Typed shadow contract: `live_trading/regime_signal.py`
- Entitlement-gated publisher: `live_trading/regime_shadow_publish.py`
- Atomic dashboard read model: `live_trading/regime_shadow_store.py`
- Authenticated dashboard consumer:
  `live_trading/etrade_cover_call_new.py` and
  `live_trading/dashboard_template.html`
- Frozen calibration plan: `docs/regime_v2_calibration_plan.json`
- Research artifact:
  `research_reports/regime_v2_calibration_artifact.json`
- Release gate: `docs/regime_data_provider_entitlements.md`
- Focused tests: `tests/test_regime_detector_v2.py`,
  `tests/test_regime_market_data.py`, and
  `tests/test_regime_evidence_store.py`,
  `tests/test_regime_market_data_gateway.py`,
  `tests/test_regime_calibration.py`,
  `tests/test_regime_signal.py`, and
  `tests/test_regime_backtest_parity.py`,
  `tests/test_regime_shadow_publish.py`,
  `tests/test_regime_shadow_store.py`, and
  `tests/test_regime_shadow_dashboard.py`
- Read-only replay: `scratch/regime_detector_v2_audit.py`
- Deterministic calibration runner:
  `scratch/regime_detector_v2_calibrate.py`

The focused tests verify:

- Isolated shock versus persistent background.
- Sustained stress entry.
- Aftershock decay.
- Prefix invariance.
- Fail-closed invalid, non-session, stale, and premature inputs.
- No `calm` label while absolute stress evidence is present.
- Explicit market-close timestamp and exact next tradable NYSE session.
- Per-session source availability and delayed-delivery rollover.
- Actual NYSE/Cboe regular and shortened-session clocks.
- Deterministic snapshot, calendar, and source-policy hashes.
- Exact requested-range coverage and transactional failure rollback.
- Exact retained provider bytes, credential-free fetch receipts, deterministic
  parser receipts, and store-owned snapshot verification.
- Stale-attempt rejection and channel-scoped, monotonic snapshot publication.
- Unavailable first-return shock evidence.
- Closed-enum, canonical V2 signals with no numeric HMM projection.
- Exact T-to-T+1 backtest annotations with no fill or action effect.
- Deployment artifact pins, immutable trace attributes, and strict
  pre-entry return-outcome resolution.

The read-only replay prints the causal record and reproduces the 2026 comparison
without downloading or mutating data.

## Primary references

- Cboe, VIX and volatility-index term structure:
  <https://www.cboe.com/tradable-products/vix/term-structure>
- Cboe, VIX calculation and trading-hour specification:
  <https://www.cboe.com/tradable-products/vix/vix-options/specifications/>
- Cboe, current VIX methodology:
  <https://cdn.cboe.com/api/global/us_indices/governance/VIX_Methodology.pdf>
- Cboe, exchange hours and shortened sessions:
  <https://www.cboe.com/about/hours/us-options>
- Cboe DataShop, VIX end-of-day calculation inputs and early-close timing:
  <https://datashop.cboe.com/vix-index-eod-calculation-inputs>
- Cboe, March 2026 volatility and hedging-demand decomposition:
  <https://www.cboe.com/insights/posts/stagflation-fears-drive-widening-volatility-risk-premium>
- Cboe, June 2026 volatility spike:
  <https://www.cboe.com/insights/posts/week-of-6-8-2026-downside-risks-rise-as-tech-volatility-spikes/>
- Cboe, July 2026 calmer but intraday-volatile conditions:
  <https://www.cboe.com/insights/posts/markets-appear-to-be-shaking-off-mideast-conflict>
- IMF, March 3, 2026 statement on market volatility and Middle East disruption:
  <https://www.imf.org/en/news/articles/2026/03/03/pr-26068-statement-on-middle-east>
- Corsi, heterogeneous autoregressive realized volatility:
  <https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf>
- Barndorff-Nielsen and Shephard, power and bipower variation with jumps:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=409160>
- Adams and MacKay, Bayesian online change-point detection:
  <https://arxiv.org/abs/0710.3742>
- Hamilton, Markov-switching time-series inference:
  <https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html>
- Fox et al., sticky state-persistence extension:
  <https://arxiv.org/abs/0905.2592>
