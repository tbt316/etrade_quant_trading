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
- Use `./deploy/sync_to_pi.sh` only to publish and select a stopped,
  owner-read-only, reverified code release.
- Keep private-state transfer suspended until the runtime consumes a complete
  state generation through one atomic pointer.
- Keep every copied runtime stopped.

Installation and restart instructions will return only after the durable
gateway is the sole mutation owner and the production-readiness gates have
passed.

The repository includes an inert hardened read-only-dashboard unit example for
future review. It has an unmet approval condition, no `[Install]` section, and
must not be activated while deployment is suspended. Provisioning, monitoring,
rotation, failure recovery, and rollback requirements are documented in
[`../docs/read_only_dashboard_runbook.md`](../docs/read_only_dashboard_runbook.md).

## Versioned Code Release

From the Mac, publish one exact archive of the current `HEAD` commit to the
configured Pi destination and atomically select it for offline inspection:

```bash
./deploy/sync_to_pi.sh
```

The release ID is `<exact-commit>-<archive-sha256>`. The command rejects staged
or unstaged tracked changes before creating a snapshot or contacting the Pi.
Untracked working-tree files, including ignored local files, are never
archived. A forced-tracked ignored path is still part of Git, so the exact
resolved tree is also checked against the repository's semantic deployment
hygiene policy before SSH. Known credential, authentication, session, runtime
state, cache, database, log, backup, and generated-artifact path classes fail
with redacted diagnostics. Static JSON contracts and examples are allowed by
their semantics rather than rejected merely because a broad local ignore rule
matches `*.json`. This path policy does not inspect file contents; code review
and credential scanning remain required for secrets hidden under an otherwise
safe source name.

The extracted snapshot is checked against a NUL-delimited committed-tree
manifest: every entry must be a regular Git blob, its executable bit must
match, and its bytes must equal the exact object returned by `git cat-file`.
Effective `export-subst` is rejected and `export-ignore` omissions cannot pass.
Symbolic links, submodules, unsafe names, and non-regular entries also fail
before remote work. Git repository/index selector environment
variables are rejected, replacement objects are disabled, and the resolved Git
directory must remain identical before the script switches to the repository
root.

A custom `PI_DIR` must be a normalized absolute path whose final directory is
`etrade_python_client`; broad targets such as `/` or `/home` are rejected. The
remote helper also rejects symbolic-link path components and creates this
owner-only layout:

```text
etrade_python_client/
├── artifacts/<release-id>.tar
├── releases/<release-id>/
├── verified/<release-id>
├── incoming/
├── selections/<unique-generation> -> ../releases/<release-id>
└── current -> selections/<unique-generation>
```

Every remote prepare, install, verification, activation, and rollback action
takes the same owner-only `.deploy-lock`. Prepare returns the exact current
generation observed before upload. Ordinary activation is a compare-and-swap:
if another publication changes `current` before activation, the stale
publication remains verified but cannot select itself. Explicit rollback is
the only operation allowed to intentionally select an older verified release.
Every successful activation or rollback creates a new 64-hex selection
generation under the operation lock, including when the selected release ID
returns from A to B to A; release-ID ABA therefore cannot satisfy an older
prepare token.
A retained archive must still hash to the digest in the release ID; its
extracted tree must pass the structural precheck and match the stored release
byte-for-byte before `current` can move. Every upload, retained artifact,
extracted file, release file, and verification marker must have exactly one
filesystem link. Upload, checksum, extraction, precheck, alias, or verification
failures occur before the pointer switch and leave `current` unchanged. Ambient
files at the deployment root are never part of `current`.

The lock deliberately has no automatic stale-lock eviction. If a host crash
leaves `.deploy-lock`, all later operations fail closed. An operator must first
confirm that no release helper is running, preserve the lock marker for the
incident record, and remove that one exact lock directory manually.

`current` is changed by creating a relative temporary symlink and replacing the
old symlink with one filesystem rename. The helper re-reads the expected
current generation under the lock immediately before that rename. A process
failure immediately before the rename leaves the old target; a failure
reported immediately after it may leave the new, already verified target.
Inspect `readlink "$PI_DIR/current"` and then the referenced link under
`"$PI_DIR/selections"` to resolve that case.

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

Private-state transfer is deliberately suspended:

```bash
./deploy/sync_to_pi.sh --state
```

That command exits with status 78 before Git, snapshot, SSH, or rsync activity.
Publishing files one-by-one could expose a mixed configuration generation, and
moving them to a new directory would not help until every consumer is migrated
to that path. State sync can return only with an owner-only, whole-generation
format and an atomic consumer pointer. Continue to protect the local files
above as secrets.

## Deterministic Rollback

Rollback requires the exact release ID printed by a successful publish:

```bash
./deploy/sync_to_pi.sh --rollback <commit>-<archive-sha256>
```

The target is not selected by timestamp, directory order, or a mutable
`previous` label. Rollback revalidates the target's marker and retained archive,
fresh-extracts the archive, compares that tree with the stored release, checks
the current pointer again, and then performs the same atomic pointer switch.
Missing, unverified, or modified rollback targets fail closed. A damaged
current release does not prevent switching away from it to a separately
verified target.

## Remaining Deployment Boundary

`--restart` remains unavailable and fails before Git, temporary-file,
remote-command, or transfer work. Installation and bootstrap remain inert.
`SYNC_DELETE=1` is accepted only for compatibility and has no effect because a
verified release directory is never updated in place.

The atomic rename prevents readers from observing a half-written pointer, but
this shell protocol does not claim power-loss durability: it does not prove
that the release directories and parent-directory rename were fsynced to
stable storage. Owner-read-only modes are not operating-system immutability:
the release owner can restore write permission, so the eventual service must
run under a different, unprivileged identity or use an enforced read-only
release filesystem. The service definition also has not yet been migrated to
execute through `current`, and the structural shell precheck is not an
application health check. Those migrations, a stopped-process Pi rehearsal,
durability validation, retention policy, and exact served-artifact
verification are required before installation or restart can be re-enabled.
Failed pointer replacement may also leave an unselected generation link; it
cannot become current implicitly and is covered by the future retention/GC
policy.
