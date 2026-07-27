# Read-only Operator Dashboard

`live_trading.read_only_dashboard` is the broker-isolated operator plane. It
loads the schema-versioned runtime configuration, validates every runtime
directory, resolves immutable dashboard credentials, and then serves only
pre-generated local artifacts.

It does not import or construct the E*TRADE client, the legacy live agent, the
order gateway, a market-data provider, or a backtest engine. It has no preview,
place, change, close, cancel, settings-mutation, refresh-queue, or PIN action
route.

> **Integration status:** this service is an isolated acceptance surface.
> Production still writes the legacy `screened_option_pairs.html`; no supported
> publisher writes `runtime_root/artifacts/positions.html` yet. Do not replace
> the deployed dashboard until the pure static publisher migration and
> served-artifact verification are complete.

## Capability boundary

The process may:

- authenticate one dashboard operator;
- display the packaged read-only dashboard shell;
- read a descriptor-validated positions artifact from
  `runtime_root/artifacts/positions.html`;
- read the sealed Regime V2 shadow signal from
  `runtime_root/model/regime-v2-shadow.json`;
- report minimal liveness, readiness, and artifact provenance.

The process may not:

- call E*TRADE or any market-data provider;
- create an OAuth session;
- accept account IDs, passwords, or a PIN on the command line;
- mutate strategy settings or credentials;
- trigger a refresh, backtest, model fit, or broker action;
- bind outside loopback.

`sandbox`, `shadow`, and unarmed `live` configurations may describe a broker
identity, but that identity is not exposed to the dashboard and grants no
broker capability. `paper` works without any broker credentials.

## Startup

First provision the directory tree documented in
[`runtime_configuration.md`](runtime_configuration.md), with exact mode
`0700`. Supply the three dashboard-only secrets through the service
environment. This process never opens the combined broker secret fallback.

Validate the complete startup boundary:

```bash
etrade-runtime-config validate \
  --config /path/to/private/etrade/runtime-config.json \
  --check-directories \
  --check-dashboard-secrets
```

Start the dashboard:

```bash
etrade-read-only-dashboard serve \
  --config /path/to/private/etrade/runtime-config.json \
  --host 127.0.0.1 \
  --port 8765
```

The module form is equivalent:

```bash
python -m live_trading.read_only_dashboard serve \
  --config /path/to/private/etrade/runtime-config.json
```

The server never opens a browser or tunnel. A reverse proxy or private overlay
network, when approved, remains an external deployment responsibility.
The built-in server handles one request at a time, caps the accept queue, and
sets a ten-second socket deadline so it cannot create an unbounded thread pool.
When—and only when—the browser-facing reverse proxy provides HTTPS, pass
`--secure-cookie`; forwarded headers do not implicitly change cookie security.

## Artifact contract

The positions artifact must be:

- a current-user-owned regular file with no group or world permissions;
- stored at the exact configured path;
- at most 8 MiB;
- no older than `data.max_snapshot_age_seconds` and no more than five seconds
  in the future;
- UTF-8;
- marked `Read only — all E*TRADE order actions are disabled`;
- free of every historical execution route and action marker.

The reader opens both parent and artifact with no-follow semantics and checks
identity, metadata, and directory entry again after the read. Missing,
replaced, changing, unsafe, executable, or oversized content returns a fixed
`503` fallback whose CSP disables scripts, network access, and forms. Stale or
future-dated content also returns `503`; the status payload preserves its
checksum and modification time for diagnosis but never displays it as current.

The Regime V2 endpoint returns only the sealed public projection. Missing,
invalid, future, or stale state remains unavailable and can never authorize
execution.

## Authentication

Dashboard username, password, and session signing key are immutable
dashboard-only environment secrets, not editable dashboard settings. The
process does not read or retain E*TRADE credentials or the action PIN. Login
compares both fields using constant-time primitives and permits five failed
attempts per rolling minute. The stateless session is HMAC-bound to the
configured username, expires after at most seven days, and is delivered as
`HttpOnly; SameSite=Strict`. `Secure` is added only by the explicit
`--secure-cookie` startup option.

Provision `ETRADE_DASHBOARD_SESSION_SECRET` with at least 32 random bytes
(for example, the 43-character output of `secrets.token_urlsafe(32)`). Repeated
or low-diversity values fail startup. Sessions are bound to the current
password, so a password change invalidates them; rotating the session key also
invalidates every session immediately.

No credential, PIN, account identity, or secret source appears in status
payloads, object representations, logs, or artifacts. Request logs contain
only a timestamp, method, normalized path, and response status.
The cookie name and signed session audience are bound to the canonical runtime
root plus configuration digest, preventing sessions from crossing dashboard
instances.

## Health endpoints

- `GET /healthz` is an unauthenticated, minimal process-liveness response.
- `GET /readyz` is authenticated and proves that configuration, directories,
  secrets, and the packaged template loaded successfully.
- `GET /api/status` is authenticated and reports redacted artifact/regime
  availability and checksums.

Artifact absence does not crash the operator plane. It is shown explicitly as
unavailable so monitoring remains possible while upstream publishers recover.
