# Runtime Configuration

The schema-versioned runtime configuration is the static, non-secret startup
contract for the E*TRADE application. It separates strategy, data, model,
execution, and risk settings and resolves runtime state outside the installed
package. Loading this configuration does not authenticate to E*TRADE, construct
the durable order gateway, or authorize a broker mutation.

The canonical example is distributed with the installed package as
`live_trading/runtime_config.example.json`. It is a valid, disabled `paper`
configuration and contains no credentials or account identity.

## Safety status

Schema versions 1 and 2 recognize four explicit modes:

| Mode | Broker environment | Exact account required | Schema-1 mutation capability | Schema-2 mutation capability |
|---|---|---:|---:|---:|
| `paper` | None | No; account settings are forbidden | Never | Never |
| `sandbox` | E*TRADE sandbox | Yes | Never | Supervised manual open, opt-in only |
| `shadow` | E*TRADE production, read-only | Yes | Never | Never |
| `live` | E*TRADE production | Yes | Never | Supervised manual open, opt-in only |

Schema 1 remains permanently read-only. Schema 2 may set
`execution.broker_mutations_enabled=true` only in `sandbox` or `live` mode, and
only for the supervised SPY/SPX two-leg credit-spread opening flow. That field
records operator intent; it is not broker authority.

Every order-capable startup must separately pass the exact runtime environment
and account boundary. Production additionally requires the independently
signed, short-lived runtime arm. Each submission also requires an authenticated
dashboard session, a freshly re-entered action PIN, a current server-signed
proposal backed by an origin-pinned two-leg `REALTIME` E*TRADE quote, an open
NYSE regular session, and the durable gateway's fresh capacity validation. No
configuration field can replace any of those checks.

This narrow composition does not enable legacy execute, close, neutralize, or
automatic workers, and it exposes no repricing path. Full unattended
production readiness and the live/sandbox order lifecycle remain unverified.

The configuration is also distinct from the historical strategy YAML under
`backtesting/strategies/`. Those research inputs remain subject to the separate
typed `BacktestSpec` and causal-validity work in Phase 4.

## Schema versions 1 and 2

Every field is required. Unknown keys, duplicate JSON keys, implicit defaults,
wrong types, and invalid ranges are errors.

```json
{
  "schema_version": 1,
  "mode": "paper",
  "runtime_root": "runtime",
  "strategy": {
    "enabled": false,
    "strategy_id": "disabled",
    "symbols": []
  },
  "data": {
    "require_complete_snapshots": true,
    "max_snapshot_age_seconds": 300
  },
  "model": {
    "enabled": false,
    "required_for_entry": false,
    "max_signal_age_seconds": 86400
  },
  "execution": {
    "selected_account_id_key": null,
    "account_allowlist": [],
    "broker_mutations_enabled": false
  },
  "risk": {
    "max_order_contracts": 0,
    "max_order_loss_cents": 0,
    "max_account_open_risk_cents": 0,
    "max_daily_loss_cents": 0,
    "max_quote_age_seconds": 30
  }
}
```

Broker-backed modes use an exact account allowlist. The selected account key
must occur exactly once, and all three broker-returned identity fields must
match:

```json
{
  "selected_account_id_key": "opaque-etrade-account-key",
  "account_allowlist": [
    {
      "account_id": "12345678",
      "account_id_key": "opaque-etrade-account-key",
      "institution_type": "BROKERAGE"
    }
  ],
  "broker_mutations_enabled": false
}
```

Account identifiers are not OAuth credentials, but they should still be
handled as private operational metadata and must never be logged in full.

Schema 2 uses the same exact document shape. To opt into the supervised manual
opening capability, the document must use:

```json
{
  "schema_version": 2,
  "mode": "sandbox",
  "strategy": {
    "enabled": true,
    "strategy_id": "supervised-credit-spreads",
    "symbols": ["SPY", "SPX"]
  },
  "model": {
    "enabled": false,
    "required_for_entry": false,
    "max_signal_age_seconds": 86400
  },
  "execution": {
    "selected_account_id_key": "opaque-etrade-account-key",
    "account_allowlist": [
      {
        "account_id": "12345678",
        "account_id_key": "opaque-etrade-account-key",
        "institution_type": "BROKERAGE"
      }
    ],
    "broker_mutations_enabled": true
  }
}
```

This excerpt is not a complete configuration; all fields from the schema-1
example remain required. Positive risk limits are mandatory when the strategy
is enabled. `risk.max_quote_age_seconds` also bounds the lifetime of a signed
manual-open proposal to at most 300 seconds. The proposal expires from its
oldest per-leg broker quote timestamp, not from a browser or scanner timestamp.

## Validation rules

- `schema_version` is the exact integer `1` or `2`; booleans and numeric
  strings are rejected.
- `mode` is exactly `sandbox`, `shadow`, `paper`, or `live`.
- `paper` forbids a selected account and requires an empty account allowlist.
  The other modes require one selected identity from a non-empty allowlist.
- Account keys are unique, and the selected key must match exactly one
  allowlisted account.
- `model.required_for_entry=true` requires `model.enabled=true`.
- `data.require_complete_snapshots` must remain `true` in both versions.
- An enabled strategy requires a non-empty, unique symbol list and positive
  risk limits.
- Schema 1 rejects `execution.broker_mutations_enabled=true`.
- In schema 2, broker mutations require `sandbox` or `live`, an enabled
  strategy whose symbols are a non-empty subset of `SPY` and `SPX`, and
  `model.required_for_entry=false`. The selected `account_id` must also be the
  exact positive numeric E*TRADE account ID accepted by the broker transport.
- Monetary limits are integer cents rather than binary floating-point values.
  Boolean values are not accepted as integers.
- `max_order_loss_cents` cannot exceed
  `max_account_open_risk_cents`.
- Quote age is bounded to 1–300 seconds. Snapshot and model-signal ages are
  bounded to 1–86,400 seconds.
- All monetary values must fit a signed SQLite 64-bit integer.

Schema migrations are explicit. The normal loader does not upgrade, rewrite,
or add defaults to an older document.

## Runtime paths

`runtime_root` is resolved once relative to the configuration file, never the
process working directory. The resolved immutable path set uses fixed names
under that root for local secrets, OAuth state, the production arm, process
lock, durable order ledger, dashboard settings and logs, data, and model state.
Loading the configuration creates none of those paths.

The required directory set is:

```text
runtime_root/
├── state/
├── cache/
├── logs/
├── artifacts/
├── execution/
├── data/
└── model/
```

Every directory, including `runtime_root`, must already exist, be owned by the
service user, be a real directory rather than a symbolic link, and have exact
mode `0700` when `--check-directories` is used.

Raw traversal components, NUL bytes, user-controlled symbolic links,
non-directory roots, and unsafe group- or world-writable parent directories
are rejected. A root-owned sticky system temporary directory may be traversed
only when an existing owner- or root-controlled non-writable directory forms a
private boundary below it; the shared directory itself and a directly derived
missing runtime root are rejected. The configuration document itself must also
be a regular file that is not group- or world-writable.
Resolution is not a filesystem authorization: sensitive files must still be
opened descriptor-relatively with no-follow semantics and revalidated at every
use.

The packaged example is documentation, not an operational state file. Copy it
outside the source checkout and installed package before use:

```bash
install -d -m 700 /path/to/private/etrade
install -m 600 \
  /path/to/repo/live_trading/runtime_config.example.json \
  /path/to/private/etrade/runtime_config.json
```

Do not edit the copy to add credentials.

## Secrets

Secrets are deliberately absent from the runtime configuration. The service or
OS environment is authoritative. The supported E*TRADE names are:

- `ETRADE_SANDBOX_CONSUMER_KEY`
- `ETRADE_SANDBOX_CONSUMER_SECRET`
- `ETRADE_LIVE_CONSUMER_KEY`
- `ETRADE_LIVE_CONSUMER_SECRET`
- `ETRADE_USER`
- `ETRADE_PASS`
- `ETRADE_DASHBOARD_USER`
- `ETRADE_DASHBOARD_PASSWORD`
- `ETRADE_DASHBOARD_PIN`
- `ETRADE_DASHBOARD_SESSION_SECRET`
- `ETRADE_POSITIONS_ARTIFACT_HMAC_KEY`

A local development fallback, when permitted by mode, must be a
current-user-owned regular file with no group or world permissions under the
configured runtime root. It is read only for values missing from the service
environment and is never packaged, logged, or printed. `live` mode does not
permit the local fallback.

The fallback is `runtime_root/state/secrets.json`. It has exact schema version
1 and exactly these keys; broker fields may be `null` only when the selected
mode has no broker:

```json
{
  "schema_version": 1,
  "etrade_consumer_key": null,
  "etrade_consumer_secret": null,
  "etrade_username": null,
  "etrade_password": null,
  "dashboard_username": "local-operator",
  "dashboard_password": "replace-with-a-long-random-value",
  "dashboard_pin": "replace-with-8-to-64-random-characters",
  "dashboard_session_secret": "replace-with-at-least-43-random-characters"
}
```

These strings are illustrative placeholders, not usable credentials.
Generate the dashboard session key from a cryptographic random source, for
example `secrets.token_urlsafe(32)`, and rotate it with the dashboard password
so every existing stateless session is invalidated.

`ETRADE_PRODUCTION_ARMING_SECRET` remains environment-only. It is never valid
in the runtime configuration or local fallback, and it must not be supplied on
the command line.

`ETRADE_POSITIONS_ARTIFACT_HMAC_KEY` is also environment-only and is not a
valid key in `secrets.json`. Generate a unique value for each runtime and
broker environment with `secrets.token_urlsafe(32)`. It must not reuse the
dashboard-session or production-arming secret. The validator requires at
exactly the canonical 43-character unpadded URL-safe encoding of 32 bytes,
rejects low-diversity and repeating patterns, and cannot prove randomness;
generation from a cryptographic random source remains an operator invariant.

The positions publisher and reader derive a non-reversible runtime binding
from the exact mode, broker environment, allowlisted account identity,
canonical runtime root, schema version, and raw configuration SHA-256. Any
configuration-byte change, including whitespace, changes that SHA-256 and
invalidates the prior artifact until it is republished. Moving the runtime
root or changing mode, environment, or account has the same fail-closed
effect.

`data.max_snapshot_age_seconds` governs both publisher admission and reader
freshness for the positions display. The integrated producer requires a
window of at least 60 seconds and polls at no more than half the configured
window. The reader independently checks both the signed full-scan start
time and the artifact file modification time, with at most five seconds of
future clock skew. The HMAC key is currently required at dashboard startup in
all modes, including `paper`; `paper` still returns
`broker_positions_disabled`.

Do not auto-load `.env`, accept passwords through command-line arguments, or
copy local secret/state files through the code-deployment path.

The supervised legacy dashboard has an additional owner-only
`dashboard_auth_secret` setting used as a local master key. Its exact accepted
format is 64 lowercase hexadecimal characters representing 32 bytes, with
low-diversity/default patterns rejected. Invalid legacy values are replaced
with a new `secrets.token_hex(32)` value when settings load. Dashboard session
signatures and manual-open proposal signatures use distinct HMAC derivation
domains, so they do not reuse the master key directly. Changing the dashboard
username or password rotates this master key and invalidates existing browser
sessions. This legacy setting is not a substitute for
`ETRADE_PRODUCTION_ARMING_SECRET` or
`ETRADE_POSITIONS_ARTIFACT_HMAC_KEY`.

## Operator checks

Configuration validation is offline and must complete before OAuth or any
broker collaborator is constructed. A valid result establishes only that the
static document is well-formed. It does not establish broker readiness,
production arming, current account identity, fresh data, reconciliation, or a
passing risk decision. In particular, a valid schema-2 mutation opt-in is not
authority to submit an order.

Validate the static document:

```bash
python -m live_trading.runtime_config validate \
  --config /path/to/private/etrade/runtime_config.json
```

After an operator has pre-created every derived directory with mode `0700`,
validate the directory boundary as well:

```bash
python -m live_trading.runtime_config validate \
  --config /path/to/private/etrade/runtime_config.json \
  --check-directories
```

Use `--check-secrets` only in the service environment that supplies the
credential variables. The command prints non-secret mode, path, schema, and
source-hash evidence; it never prints account identifiers or secret values.

The broker-isolated dashboard has a narrower check:

```bash
python -m live_trading.runtime_config validate \
  --config /path/to/private/etrade/runtime_config.json \
  --check-directories \
  --check-dashboard-secrets
```

That path reads only `ETRADE_DASHBOARD_USER`,
`ETRADE_DASHBOARD_PASSWORD`, `ETRADE_DASHBOARD_SESSION_SECRET`, and
`ETRADE_POSITIONS_ARTIFACT_HMAC_KEY`. It never reads the combined fallback,
broker credential variables, or `ETRADE_DASHBOARD_PIN`.

The release gate verifies that the credential-free example is present in both
the wheel and source distribution, matches the committed bytes, parses through
the installed runtime module, and produces no working-directory state.

## Supervised manual-open composition

`live_trading/etrade_cover_call_new.py` may construct the order-capable runtime
only after schema 2 opts in and the independent runtime safety boundary is
already armed for the exact environment and account. The reviewed
`live_trading/execution_runtime.py` composition root creates the private
ledger, broker reader, no-retry transport, and durable order gateway, then
returns only the narrow `ManualOpenService` capability.

The dashboard supports the familiar operator-reviewed SPY/SPX `PUT` or `CALL`
vertical workflow:

1. During an open NYSE regular session, the server treats the scanner result as
   candidate identity only and obtains one retained, origin-pinned E*TRADE
   response for the exact two OSI contracts. Both rows must be
   `quoteStatus=REALTIME`, have usable non-crossed bid/ask values, carry
   exchange timestamps no more than five seconds apart, and be fresh under
   `max_quote_age_seconds`.
2. The server derives the exact two-leg midpoint credit and creates a
   short-lived proposal bound to the quote receipt/snapshot hashes, per-leg
   values/timestamps, runtime, environment, exact account, and configuration.
   Its deadline is capped fifteen seconds before the exact session close and
   is enforced again before broker preview and placement. Proposal issuance
   does not read or reserve account capacity.
3. The authenticated operator reviews the exact two-leg net-credit economics
   and re-enters the action PIN for that confirmation.
4. The server validates the proposal, quantity, quote age, per-order loss
   ceiling, and request correlation before issuing exactly one durable opening
   command. Final submission obtains fresh capacity-v3 account evidence, so a
   valid proposal may still fail safely without a broker send.

The browser cannot rewrite the signed economics at confirmation. The service
passes `max_account_open_risk_cents` and `max_daily_loss_cents` to the gateway
as two independent limits:

- the account limit bounds current opening risk. Under schema-19
  `OPENING_MAX_LOSS_V2`, fresh capacity is the lesser of raw broker buying
  power and the non-negative account budget after external position risk,
  external order risk, and represented managed filled risk; active local
  reservations are subtracted separately when the new reservation is created;
- the daily limit is a New York calendar-day ceiling on newly authorized
  maximum loss. Despite its compatibility name, it is not realized or marked
  P&L, does not reset by NYSE trading-session boundaries, and is not refunded
  when work fails after a reservation was created.

Historical V1 capacity decisions are replay-only. They cannot authorize a
fresh reservation or claim; schema-19 migration fails/releases only pristine
untraced V1 reservations and moves traced work to `SUBMISSION_UNKNOWN` while
retaining risk.

The signed `proposal_id` is the durable idempotency key. The UUID
`request_id` is only HTTP correlation. The browser writes a non-secret recovery
marker before sending, clears the PIN, aborts the request after 30 seconds, and
clears that marker only for an exact correlated `NOT_ATTEMPTED` response. Any
unrecognized, malformed, mismatched, or network-ambiguous result is shown as
`SUBMISSION_UNKNOWN` with no automatic retry. Periodic dashboard status reads
the local ledger; it does not poll E*TRADE orders, resubmit, or reprice.

Login and PIN failures are separately throttled with bounded failure windows
and lockouts, and dashboard JSON requests use an exact bounded-body reader.
The handler loads one bounded, regular, UTF-8 dashboard template at process
start, verifies its protocol marker/nonce contract, and serves that pinned
generation with its SHA-256. These controls prevent a new HTML file from being
mixed with an already-running backend generation.

The full pure pretrade engine is not composed into this manual workflow:
complete account Greeks, concentration, marked P&L, and regime authorization
remain readiness gaps.

The service does not expose repricing, closing, neutralization, cancellation,
or automatic execution. The historical execute/close/neutralize routes and
automatic workers remain reject-only tombstones. Source composition is not
proof of a successful E*TRADE preview, placement, fill, reconciliation, or
restart cycle; those sandbox/live lifecycle checks are still outstanding.

## Read-only composition

The supported operator-plane composition is
[`read_only_dashboard.py`](../live_trading/read_only_dashboard.py). It validates
configuration, directories, and dashboard-only secrets before opening a
listening socket, and
it constructs no OAuth, broker, market-data, model, or order collaborator.
Dashboard credentials remain immutable and are not stored in or editable
through legacy live settings. The operator process never reads or retains
E*TRADE credentials or the unused action PIN.

The positions HMAC key is the only secret shared with the transitional
publisher. The dashboard receives only an authenticated, runtime-bound static
artifact reader capability, not the full runtime path set or a publication
capability.

See [`read_only_dashboard.md`](read_only_dashboard.md) for provisioning,
startup, endpoint, and artifact-integrity details. This read-only service does
not construct the manual-open service and does not make the durable order stack
live-ready.
