# Read-only Operator Dashboard

`live_trading.read_only_dashboard` is the broker-isolated operator plane. It
loads the schema-versioned runtime configuration, validates every directory
exposed to this process (the root, artifacts, and model directories), resolves
immutable dashboard credentials, and then serves only pre-generated local
artifacts.

It does not import or construct the E*TRADE client, the legacy live agent, the
order gateway, a market-data provider, or a backtest engine. It has no preview,
place, change, close, cancel, settings-mutation, refresh-queue, or PIN action
route.

> **Integration status:** R8e-B implements the supported static publisher in
> source. The legacy monitor now requires two consecutive, complete portfolio
> scans with identical position identity and quantity, projects the second
> scan into a typed display-only snapshot, signs the exact rendered bytes, and
> atomically replaces `runtime_root/artifacts/positions.html`. This has been
> verified through the local production handler, not deployed to or restarted
> on the Pi. Installation and remote restart remain suspended.

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
`0700`. Supply the four dashboard-process secrets through the service
environment: the username, password, session secret, and positions-artifact
HMAC key. This process never opens the combined broker secret fallback.

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
The Refresh button only polls the local status/read-model endpoints. It never
requests a broker refresh.

The dashboard and publisher currently run as the same Unix identity because
the artifact contract requires current-user ownership. The dashboard service
must therefore receive an OS-enforced read-only mount view of the runtime
tree; ordinary file permissions alone are not sufficient separation. See
[`read_only_dashboard_runbook.md`](read_only_dashboard_runbook.md) and the
inert hardened unit example under `deploy/`.

## Publication flow

The transitional publisher is composed by
`live_trading/etrade_cover_call_new.py`; it is not yet a standalone broker-read
collector. On every cycle it:

1. fetches one minimal, page-complete E*TRADE portfolio scan;
2. fetches a second complete scan and compares only canonical contract
   identity and quantity, including the exact OSI key, adjustment flag,
   multiplier, and deliverables for options;
3. preserves the prior artifact if either scan fails or the scans differ;
4. converts the confirmed result into primitive, bounded display DTOs;
5. renders deterministic HTML and signs the exact bytes with
   HMAC-SHA256;
6. publishes through an owner-only lock, owner-only temporary file,
   `fsync`, atomic replacement, and parent-directory `fsync`.

The publisher rejects stale or future source times, source-time regressions,
same-time generation conflicts, mismatched broker environments, unsafe
directories, unsafe existing files or locks, concurrent publishers, missing
option identity evidence, adjusted options, and any option multiplier other
than the standard 100 shares. The operator artifact labels each accepted
option with its exact OSI key and `Standard ×100`; unsupported contracts
preserve the prior artifact rather than being rendered ambiguously.
`positions_artifact_commit_unknown` is deliberately distinct from a failed
pre-commit operation: the new bytes may already be visible and must be
reconciled by digest before retrying.

The artifact is a display read model. It is not a broker transaction, risk,
margin, or execution snapshot. `source_as_of` is the conservative local UTC
time captured immediately before the complete second portfolio scan; it is
not an exchange timestamp.

## Artifact contract

The positions artifact must be:

- a current-user-owned regular file with no group or world permissions;
- stored at the exact configured path;
- at most 8 MiB;
- no older than `data.max_snapshot_age_seconds` and no more than five seconds
  in the future;
- UTF-8;
- signed over its exact bytes with the configured HMAC key;
- bound to the exact mode, broker environment, allowlisted account, canonical
  runtime root, schema version, and configuration SHA-256;
- bound to the exact installed `positions_artifact.py` source bytes that
  define the renderer/verifier contract;
- marked `Read only — all E*TRADE order actions are disabled`;
- free of every historical execution route and action marker.

The reader opens both parent and artifact with no-follow semantics and checks
identity, metadata, and directory entry again after the read. Missing,
replaced, changing, unsafe, executable, or oversized content returns a fixed
`503` fallback whose CSP disables scripts, network access, and forms. Stale or
future-dated content also returns `503`; the status payload preserves its
checksum and modification time for diagnosis but never displays it as current.
Both signed `source_as_of` and file modification time must be fresh.
Changing text, CSS, rows, metadata, environment, runtime binding, or source
time without the key makes the whole artifact untrusted.

The shell and iframe use one exact generation. `/api/status` reports the
verified artifact SHA-256, and the shell fetches
`/api/positions?sha256=<digest>` before assigning the response to the
sandboxed iframe. A missing or malformed digest returns `400`; an artifact
replacement between those two requests returns a fail-closed `409`; missing,
stale, unavailable, or untrusted content returns `503`. The browser records a
generation only after the expected response succeeds, so the next status poll
retries the same digest after a transient failure. The server never silently
serves generation B under generation A's status.

Status includes the exact freshness expiry. The browser arms a local expiry
timer, checks again when a background tab becomes visible, and hides the
iframe immediately when freshness expires or a status poll fails. A previously
verified page therefore cannot remain presented indefinitely after the
dashboard loses its server connection.

Common status reasons are:

- `missing`: no artifact has been published;
- `unsafe_parent` or `unsafe_file`: ownership, type, link, or mode failed;
- `changed`: the file or directory entry moved during the verified read;
- `artifact_io_error`: a bounded filesystem read failed;
- `oversized`: the bounded artifact size was exceeded;
- `untrusted_artifact`: signature, grammar, environment, or runtime binding
  failed;
- `future_timestamp` or `stale`: signed source time or file time is outside
  the configured window;
- `broker_positions_disabled`: `paper` mode intentionally has no broker
  positions capability.

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

Provision `ETRADE_POSITIONS_ARTIFACT_HMAC_KEY` as a different
`secrets.token_urlsafe(32)` value. The accepted representation is exactly the
canonical 43-character unpadded URL-safe encoding of 32 bytes; startup also
rejects low-diversity, repeating, and reused dashboard secrets. No validator
can prove randomness, so cryptographic generation remains mandatory. The key
is shared only by the publisher and reader, never accepted in configuration or
the local combined secret file, and must not reuse the production-arming
secret. This is symmetric integrity rather than asymmetric producer
attestation; protecting the dashboard process with a read-only service sandbox
remains required.

No credential, PIN, account identity, or secret source appears in status
payloads, object representations, logs, or artifacts. Request logs contain
only a timestamp, method, normalized path, and response status.
The cookie name and signed session audience are bound to the canonical runtime
root plus configuration digest, preventing sessions from crossing dashboard
instances.

## Health endpoints

- `GET /healthz` is an unauthenticated, minimal process-liveness response.
- `GET /readyz` is authenticated and proves that configuration, directories,
  secrets, and the packaged template loaded successfully. It does not prove
  that current position data exists.
- `GET /api/status` is authenticated and reports redacted artifact/regime
  availability and checksums.

Artifact absence does not crash the operator plane. It is shown explicitly as
unavailable so monitoring remains possible while upstream publishers recover.
Data-readiness monitoring must inspect the `positions.available` and
`positions.reason` fields in `/api/status`; that endpoint itself remains HTTP
`200` when the display artifact is unavailable.
