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

Schema version 1 recognizes four explicit modes:

| Mode | Broker environment | Exact account required | Broker mutation authority |
|---|---|---:|---:|
| `paper` | None | No; account settings are forbidden | Never |
| `sandbox` | E*TRADE sandbox | Yes | Never in schema version 1 |
| `shadow` | E*TRADE production, read-only | Yes | Never |
| `live` | E*TRADE production | Yes | Starts unarmed; never from configuration |

`live` is an explicit operational intent, not an authorization. The
configuration field `execution.broker_mutations_enabled` must be `false` in
schema version 1. An `armed` field is not part of the schema. A future live
composition root may accept the existing independently signed, short-lived
production arm only after the cancellation, closing, and complete risk-policy
gates are delivered. R8d does not compose the broker reader, transport, ledger,
or order gateway.

The configuration is also distinct from the historical strategy YAML under
`backtesting/strategies/`. Those research inputs remain subject to the separate
typed `BacktestSpec` and causal-validity work in Phase 4.

## Schema version 1

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

## Validation rules

- `schema_version` is the exact integer `1`; booleans and numeric strings are
  rejected.
- `mode` is exactly `sandbox`, `shadow`, `paper`, or `live`.
- `paper` forbids a selected account and requires an empty account allowlist.
  The other modes require one selected identity from a non-empty allowlist.
- Account keys are unique, and the selected key must match exactly one
  allowlisted account.
- `model.required_for_entry=true` requires `model.enabled=true`.
- `data.require_complete_snapshots` must remain `true` in version 1.
- An enabled strategy requires a non-empty, unique symbol list and positive
  risk limits.
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
  /path/to/runtime_config.example.json \
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

## Operator checks

Configuration validation is offline and must complete before OAuth or any
broker collaborator is constructed. A valid result establishes only that the
static document is well-formed. It does not establish broker readiness,
production arming, current account identity, fresh data, reconciliation, or a
passing risk decision.

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
not make the isolated durable order stack live-ready.
