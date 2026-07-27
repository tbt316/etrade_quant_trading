# Read-only Dashboard Operations Runbook

This runbook covers the R8e-B broker-isolated dashboard and signed positions
read model. It does not authorize live deployment. Service installation,
remote restart, and Pi activation remain suspended until the wider production
gates in `production_readiness_upgrade_plan.md` close.

## Operating contract

- The dashboard binds only `127.0.0.1` and has no broker, provider, writer,
  refresh-queue, order, settings, or PIN capability.
- The transitional publisher remains inside the legacy monitor and uses the
  same Unix identity as the dashboard.
- The dashboard's systemd mount namespace exposes only the runtime root,
  `artifacts/`, and `model/` read-only. It keeps `state/`, `cache/`, `logs/`,
  `execution/`, and `data/` inaccessible. The publisher runs outside that
  namespace and owns the only positions writer capability.
- `positions.html` is an operator display read model. Never use it for risk,
  reconciliation, order eligibility, or execution.
- A previous artifact is preserved on producer failure and becomes unavailable
  after the configured freshness window.

The inactive example unit is
`deploy/etrade-read-only-dashboard.service.example`. It contains an approval
condition and no `[Install]` section, so copying it alone cannot enable the
service.

## Provisioning checklist

1. Install one reviewed immutable wheel under `/opt/etrade`; do not run from a
   mutable checkout.
2. Create the configuration and runtime tree outside the package. Configuration
   and files use mode `0600`; every runtime directory uses exact mode `0700`
   and the service/publisher user owns them.
3. Configure reliable UTC clock synchronization. More than five seconds of
   future skew makes the artifact fail closed.
4. Generate separate random values for the dashboard session secret and
   positions artifact key:

   ```bash
   python -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

5. Put dashboard username, password, session secret, and
   `ETRADE_POSITIONS_ARTIFACT_HMAC_KEY` in a root-owned `0600` systemd
   environment file. Never place secret values in a unit, configuration,
   command line, log, or repository.
6. Keep the reverse proxy on the same host or an approved private overlay.
   Terminate TLS at that boundary and pass `--secure-cookie`.
7. Substitute reviewed absolute paths in the example unit, run
   `systemd-analyze security`, and repeat the complete offline and browser
   acceptance suite on the target OS before creating the explicit approval
   marker.

Validate before either process performs OAuth or opens a socket:

```bash
etrade-runtime-config validate \
  --config /var/lib/etrade/runtime-config.json \
  --check-directories \
  --check-dashboard-secrets
```

## Startup and shutdown order

When activation is eventually approved:

1. Validate configuration, directories, clock, and secrets.
2. Start the read-only broker monitor/publisher.
3. Wait for two stable complete scans and a durable publication receipt.
4. Independently validate artifact owner, mode, source time, and SHA-256.
5. Start the sandboxed dashboard.
6. Authenticate, inspect `/api/status`, and verify
   `positions.available=true`.
7. Reload the exact browser endpoint at desktop and mobile widths and inspect
   the actual positions iframe.

Stop the dashboard first and publisher second. A dashboard restart must never
restart, refresh, or otherwise signal the publisher.

## Monitoring

- `/healthz` proves only that the HTTP process responds.
- Authenticated `/readyz` proves only that startup composition succeeded.
- Authenticated `/api/status` is the data-readiness source. Alert when
  `positions.available` is not `true`, when `positions.stale` is true, or when
  the source age approaches the configured limit.
- Record the status SHA-256 and source generation, not artifact contents or
  account identity.
- Treat `positions.expires_at` as a hard client and monitoring deadline. The
  browser hides previously verified content on expiry or status-poll failure.
- The iframe request must contain the exact status SHA-256. `409` indicates a
  safe publication race; poll status again. `400` indicates a client contract
  error. `503` indicates unavailable, stale, or untrusted data.
- Alert on repeated publisher conflicts, unsafe filesystem errors,
  scan-instability rejections, and `positions_artifact_commit_unknown`.

## Failure recovery

`missing`
: Confirm that the publisher uses the same canonical configuration and runtime
  root, then wait for a new two-scan publication.

`stale` or `future_timestamp`
: Check time synchronization and publisher health. Do not touch the file time
  to make it appear fresh.

`untrusted_artifact`
: Treat this as key, configuration, environment/account binding, or byte
  integrity failure. Compare non-secret configuration digests and restart from
  a reviewed artifact; never bypass signature validation.

`unsafe_parent`, `unsafe_file`, or `changed`
: Stop both processes. Inspect every path component, ownership, mode, file
  type, link count, and resolved inode. Repair only the exact reviewed path;
  do not recursively change ownership or permissions.

`artifact_io_error`
: Treat the snapshot as unavailable and investigate the filesystem or storage
  device. The request handler should remain up and serve the fixed fallback;
  repeated I/O errors are an operational incident.

Unsafe publish lock
: Stop every publisher and prove none retains the lock. A valid lock is a
  current-user-owned regular single-link file at mode `0600` and may remain in
  place. Quarantine an unsafe object only after resolving its exact
  descriptor/path identity; never delete it while a publisher may be active.

`positions_artifact_commit_unknown`
: The atomic replacement may already be visible. Read through the authenticated
  dashboard reader and compare its SHA-256 with the exception receipt. If they
  match, treat the generation as committed. If they differ or validation
  fails, keep the old view unavailable and perform a fresh stable scan; do not
  blindly replay the same write.

## Key and configuration rotation

There is one active symmetric artifact key and no key ID or grace set. Rotation
therefore has a short, intentional fail-closed window:

1. Generate a new unique key.
2. Stop the dashboard.
3. update the root-owned publisher and dashboard environment sources;
4. restart the publisher and obtain a durable artifact signed with the new key;
5. start the dashboard and verify the exact new digest.

Never retain the old key in configuration or reuse it elsewhere.

Any configuration-byte change changes the runtime binding, even whitespace.
For a configuration, account, mode, or runtime-root change, stop both
processes, validate the new tree, start the publisher, require a new durable
artifact, and only then start the dashboard.

## Rollback

Rollback means switching to a previously reviewed immutable release whose
artifact schema and configuration contract are compatible. Keep the dashboard
stopped during the switch. Validate directories and secrets, publish a fresh
artifact with that release, then re-run handler and browser acceptance before
activation. Never roll back by restoring OAuth state, copying a stale
positions file, disabling HMAC checks, or re-enabling the legacy installer.
