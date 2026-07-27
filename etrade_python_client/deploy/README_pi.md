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

From the Mac, copy the current working tree to the configured Pi destination
for offline inspection:

```bash
./deploy/sync_to_pi.sh
```

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

Copying private state does not arm, install, start, or authorize a service.
After a copy, keep the Pi stopped and protect those files as secrets.

## Sync-Only Behavior

Code-only sync excludes secrets, OAuth tokens, logs, virtual environments, and
generated runtime files. `--state` adds only the explicitly listed private
runtime files. `--restart` is unavailable and fails before any remote command
or file transfer.
