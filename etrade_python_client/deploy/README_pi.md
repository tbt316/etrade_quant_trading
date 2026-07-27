# Raspberry Pi Code Sync (Live Deployment Suspended)

Live E*TRADE service installation and remote restart are suspended while the
durable order gateway migration is incomplete. The repository may be copied to
a Raspberry Pi for offline inspection, but these scripts must not install,
start, enable, or restart an order-capable process.

`deploy/install_pi_service.sh` always exits nonzero without writing a service
file or invoking privileged commands. `deploy/sync_to_pi.sh --restart` also
exits nonzero before SSH or rsync. This is an intentional production safety
boundary, not a deployment outage workaround.

## Allowed Workflow

- Edit, test, and review on the MacBook.
- Use `./deploy/sync_to_pi.sh` for code-only synchronization.
- Use `./deploy/sync_to_pi.sh --state` only for a deliberate private-state
  migration.
- Keep every copied runtime stopped.

Installation and restart instructions will return only after the durable
gateway is the sole mutation owner and the production-readiness gates have
passed.

## Code-Only Sync

From the Mac, copy an immutable archive of the current `HEAD` commit to the
configured Pi destination for offline inspection:

```bash
./deploy/sync_to_pi.sh
```

The command rejects staged or unstaged tracked changes before creating a
snapshot or contacting the Pi. Untracked and ignored working-tree files are
never part of the code source presented to rsync. The extracted snapshot must
contain every regular file in a NUL-delimited committed-tree manifest, so Git
attributes cannot silently omit code and symbolic links or submodules fail
before remote work. Git repository/index selector environment variables are
also rejected, replacement objects are disabled, and the resolved Git
directory must remain identical before the script switches to the repository
root. A custom `PI_DIR` must be a normalized absolute path whose final directory
is `etrade_python_client`; broad targets such as `/` or `/home` are rejected
before remote work.

## Private Runtime State

The live bot needs files that are intentionally not committed:

- `config.ini`
- `.env`, if used
- `.etrade_oauth`, if present
- `live_trading_settings.json`
- `trade_status.json`
- `manual_order_status.json`
- `spy_tracking_data.json`
- `spy_vix_price_cache.json`
- `nudge_history.json`, if present
- `nudge_state.json`, if present

Sync those from the Mac only when you intentionally want the Pi to inherit the current runtime state:

```bash
./deploy/sync_to_pi.sh --state
```

Only regular, non-symlink files from that exact allowlist are eligible; they
are copied into a separate owner-only temporary snapshot before any remote
action. Copying private state does not arm, install, start, or authorize a
service. After a copy, keep the Pi stopped and protect those files as secrets.

## Sync-Only Behavior

Code-only sync excludes secrets, OAuth tokens, logs, virtual environments, and
generated runtime files as defense in depth around the committed archive.
`--state` adds only the explicitly listed private runtime files. `--restart`
is unavailable and fails before Git, temporary-file, remote-command, or file
transfer work.

By default rsync does not delete unrelated files already present on the Pi.
`SYNC_DELETE=1 ./deploy/sync_to_pi.sh` requests deletion for an exact code
mirror while retaining excluded private state. Until versioned release
directories and atomic activation are delivered, even code-only sync remains
an offline-inspection tool, not an approved deployment.
