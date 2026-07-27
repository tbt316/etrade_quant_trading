# Durable Order Intent Ledger

**Delivery status:** R7a ledger + R7b transport + R7c coordinator + R7d
durable reader, isolated

**Production status:** not connected to live E*TRADE mutation paths

`live_trading/order_intent_ledger.py` is the durable, broker-agnostic state
machine for order identity, capacity reservation, fencing, and ambiguous broker
outcomes. It performs no network I/O. `etrade_broker_transport.py` owns the
reviewed no-retry mutation exchange, `etrade_broker_reader.py` owns the exact
origin-bound GET surface, and `etrade_order_gateway.py` coordinates opening
submissions, price-only amendments, and restart reconciliation. The live agent
still instantiates none of them, so this stack does not yet protect the current
legacy order paths.

## Supported scope

- Explicit `sandbox` or `production` environment and exact account identity.
- Two-leg vertical option spreads with one buy leg and one sell leg.
- Opening credit and debit spreads whose price direction and maximum exposure
  can be derived from immutable order fields.
- No new closing orders until a later slice supplies typed position and open-order
  capacity evidence. Previously persisted closing orders remain readable and
  reconcilable after migration, but cannot be newly claimed or amended.
- No equity orders, naked opening options, arbitrary multi-leg spreads, bulk
  cancellation, or terminal reservation release. Unsupported operations fail
  closed.

## State and crash boundary

```mermaid
stateDiagram-v2
    [*] --> INTENT
    INTENT --> CLAIMED: fenced lease
    CLAIMED --> FAILED: definitive pre-POST failure
    CLAIMED --> SUBMISSION_UNKNOWN: exact semantic bytes persisted
    SUBMISSION_UNKNOWN --> SUBMITTED: broker acknowledgement or query
    SUBMISSION_UNKNOWN --> FILLED: broker query
    SUBMISSION_UNKNOWN --> CANCELLED: broker query
    SUBMISSION_UNKNOWN --> REJECTED: broker query
    SUBMISSION_UNKNOWN --> EXPIRED: broker query
    SUBMITTED --> SUBMITTED: fenced amendment
    SUBMITTED --> FILLED: broker query
    SUBMITTED --> CANCELLED: broker query
    SUBMITTED --> REJECTED: broker query
    SUBMITTED --> EXPIRED: broker query
```

The transport persists the exact send attempt and transitions to
`SUBMISSION_UNKNOWN` or an in-doubt amendment immediately before the
corresponding broker mutation. A timeout, malformed response, process crash, or
expired post-capable lease never returns the intent to a retryable state. A
broker query for a durably known broker order ID must prove the exact authorized
economic payload before reconciliation. Because E*TRADE does not echo
`clientOrderId`, an ambiguous placement without a durable broker order ID
remains blocked rather than guessed or retried.

## Enforced invariants

- An idempotency scope/key is permanently bound to one immutable economic
  envelope. Stable client order IDs include account and environment identity.
- Client IDs and broker order IDs cannot be rebound across original orders,
  active amendments, or immutable amendment history.
- Submission and amendment leases use monotonic fencing tokens. Stale workers
  cannot begin, release, acknowledge, or overwrite newer work, even when the
  owner name is reused.
- `BEGIN IMMEDIATE` serializes account-capacity decisions and state
  transitions. New account mutations stop while any submission or amendment is
  unresolved.
- Opening reservations use fresh, typed quote, portfolio, and buying-power
  evidence. The caller's asserted maximum loss cannot be below the exposure
  derived from strike width, price, contract multiplier, and quantity.
- An opening terminal state retains its reservation as
  `FILLED_PENDING_ABSORPTION`. The schema-11 R7a–R7d stack never releases it:
  neither a newer timestamp nor a different account digest proves that a
  specific partial or terminal fill is reflected in positions. A later
  position-absorption slice must add order-bound position evidence.
- Evidence dataclasses and their security-critical string, timestamp, integer,
  byte, and `Decimal` fields must use exact built-in types. Subclass overrides
  cannot replace validation or comparison behavior.
- Preparation returns a frozen `OutboundAuthorization` containing immutable
  serialized broker-schema bytes. The transport deterministically derives the
  final E*TRADE XML from those bytes and a durably recorded preview ID. Every
  preview/place send is uniquely claimed before I/O; every parsed response is
  recorded before it can update or leave the durable state machine.
- The coordinator requires one immutable account-level opening-risk ceiling.
  Strategy commands cannot raise it. Reconciliation requires the reader's
  normalized order payload to match the durable original, pending amendment, or
  latest completed amendment hash.
- Capacity can be set only from a content-addressed schema-11 manifest. A
  complete manifest brackets two independent scans with exact account-list
  reads; fully traverses balance, every portfolio page, and all reviewed active
  order status lanes; and requires both economic scans to match. Moving balance
  effective timestamps are freshness-checked separately from economic
  stability.
- Every usable broker GET is pinned to the configured E*TRADE origin and exact
  account, has no redirects/retries/ambient session state, runs in a bounded
  disposable process, and is persisted before its result can leave the reader.
  The ledger deterministically reruns the installed parser over the raw bytes
  before accepting the normalized receipt.
- Known-order reconciliation performs only the direct lookup for the already
  durable broker order ID. A 404, partial fill, replacement ambiguity, payload
  mismatch, or undocumented shape remains unresolved and cannot clear a
  blocker.
- Order events, broker-order history, amendment history, outbound
  authorizations, transport attempts, preview receipts, mutation responses,
  broker-read responses, read manifests, and capacity decisions are
  append-only.
- The ledger database must live in an owner-only `0700` directory and remain an
  owner-only regular `0600` file. SQLite sidecars receive the same validation.

## Schema policy

Schema 11 adds raw broker-read receipts, ordered semantic manifests, durable
capacity decisions, and provenance foreign keys on caps, reservations, and
events. Additive schema 8→9→10→11 migration is one explicit SQLite transaction
and verifies required columns, foreign keys, append-only triggers, journal
mode, foreign-key integrity, and `quick_check` before version promotion.
Unknown or malformed schemas fail closed. A production operator must still
take an atomic private backup and complete a rollback drill before migration.

## Remaining production release gates

The isolated R7 stack must not be used as evidence that live execution is
production-ready. The following remain required before any order-capable
process is enabled:

1. Position-level fill evidence must safely absorb terminal opening
   reservations, including partial fills, replacements, assignment/exercise,
   and zero-fill proof for cancelled/rejected/expired orders.
2. Closing-position capacity and one-shot per-intent cancellation need durable,
   crash-tested protocols. Bulk cancellation remains disabled.
3. The live composition root must construct the exact ledger, reader,
   transport, and coordinator, and static enforcement must reject direct legacy
   order mutations outside that root.
4. Sandbox restart/crash fixtures must cover pagination drift, stale evidence,
   every nonterminal/terminal broker status, replacement chains, cancellation,
   closing, and process death at each durable/I/O boundary.
5. Operational migration needs a private database backup, integrity check,
   rollback drill, credential rotation/history purge, deployment restart, and
   observation of the exact served/live artifacts.

The focused ledger/reader/transport/coordinator suites exercise raw-parser
binding, pagination and marker drift, two-scan stability, schema rollback,
exact payload reconciliation, mutation fencing, and crash/timeout behavior.
They contain 112 deterministic tests; the maintained repository `tests/` suite
contains 277 passing tests in the clean Python 3.10 environment. This is source
verification only; position, cancellation/closing, live composition, and
operational migration gates above remain open.
