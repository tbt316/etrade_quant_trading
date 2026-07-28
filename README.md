# E*TRADE Quant Trading

Causal options research, deterministic backtesting, and fail-closed E*TRADE
trading infrastructure.

> **Safety status:** live order execution, service installation, runtime
> bootstrap, and remote restart are intentionally disabled while the durable
> execution migration is incomplete. This repository is not approved for
> unattended trading.

## Reproducible setup

Use the exact CPython release in `.python-version`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install \
  --require-hashes \
  --only-binary=:all: \
  -r requirements/build.lock

python -m pip install \
  --require-hashes \
  --no-build-isolation \
  --only-binary=:all: \
  --no-binary=pyetrade,rauth \
  -r requirements/test.lock

python -m pip install --no-deps --no-build-isolation .
```

`pyproject.toml` is the only direct dependency and package definition.
`requirements/*.lock` contains the complete hash-pinned environments. See
[`requirements/README.md`](requirements/README.md) for the lock policy and
regeneration commands.

## Offline verification

The default suite is deterministic and must not call brokers or market-data
providers:

```bash
set -euo pipefail
umask 022
python etrade_python_client/scripts/check_repo_hygiene.py
python etrade_python_client/scripts/check_etrade_mutation_boundary.py
PYTHONPATH=etrade_python_client python -m pytest -q -m "not integration" \
  etrade_python_client/tests/test_etrade_mutation_boundary.py \
  etrade_python_client/tests/test_repo_hygiene.py

release_source="$(mktemp -d)"
release_dist="$(mktemp -d)"
sdist_source="$(mktemp -d)"
sdist_dist="$(mktemp -d)"
git archive --format=tar HEAD | tar -xf - -C "$release_source"
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
python -m build --sdist --wheel --no-isolation \
  --outdir "$release_dist" "$release_source"
python etrade_python_client/scripts/normalize_sdist.py \
  --sdist "$release_dist"/*.tar.gz \
  --epoch "$SOURCE_DATE_EPOCH"
python etrade_python_client/scripts/check_release_artifacts.py \
  --dist-dir "$release_dist" \
  --repository-root . \
  --git-revision HEAD
tar -xzf "$release_dist"/*.tar.gz \
  --strip-components=1 \
  -C "$sdist_source"
python -m build --wheel --no-isolation \
  --outdir "$sdist_dist" "$sdist_source"
cmp "$release_dist"/*.whl "$sdist_dist"/*.whl
python -m pip install --no-deps --no-build-isolation "$release_dist"/*.whl
env -u PYTHONPATH ETRADE_TEST_ARTIFACT=1 ETRADE_TEST_NETWORK=deny \
  MASSIVE_OFFLINE_ONLY=1 \
  python -m pytest -q -m "not integration" etrade_python_client/tests \
  --ignore=etrade_python_client/tests/test_etrade_mutation_boundary.py \
  --ignore=etrade_python_client/tests/test_repo_hygiene.py
```

CI runs the same artifact-first checks on CPython 3.10.20. It audits the wheel
and sdist against the exact committed Git blobs, verifies wheel integrity and
metadata, rebuilds the wheel from the inspected sdist, and proves runtime
imports come from the installed artifact. CI runs the smoke and functional
tests as the unprivileged runner inside a loopback-only Linux network
namespace. The runtime smoke uses a whitelisted environment after removing
build-only installers from its virtual environment. The
`ETRADE_TEST_NETWORK=deny` command above is Python-level defense-in-depth when
run outside that namespace. Network, broker, and provider tests must be
explicitly marked as integration tests and are never part of the default gate.

## Source layout

The installable packages live under `etrade_python_client/`:

- `backtesting/`: historical simulator, cache, strategy loader, and experiment
  tooling.
- `live_trading/`: broker safety boundary, durable intent components,
  read-only dashboard, and causal regime pipeline.
- `accounts/`, `market/`, `order/`, `core_api/`: legacy E*TRADE adapters kept
  behind mutation tombstones during migration.
- `polygonio/` and `strategies/`: reusable historical-data and strategy
  modules.
- `tests/`: deterministic reliability and contract tests; not distributed in
  the runtime wheel.
- `scratch/`: research-only scripts and outputs; not distributed.

The wheel includes only the ten explicitly allowlisted packages, the dashboard
template, the credential-free runtime-configuration example, and three strategy
YAML files. It excludes credentials, operational configuration, runtime state,
caches, reports, tests, and scratch material.

## Runtime configuration

[`docs/runtime_configuration.md`](etrade_python_client/docs/runtime_configuration.md)
defines the strict schema-versioned startup contract and mode semantics for
`sandbox`, `shadow`, `paper`, and `live`. The packaged example is a valid,
disabled `paper` configuration containing no credentials or account identity.
Schema version 1 grants no broker-mutation authority in any mode; `live` always
loads unarmed, and configuration is never a substitute for the independently
signed production arm.

Copy the example outside the checkout or installed package and protect the
operational copy before editing non-secret settings:

```bash
install -d -m 700 /path/to/private/etrade
install -m 600 \
  etrade_python_client/live_trading/runtime_config.example.json \
  /path/to/private/etrade/runtime_config.json
```

## Credentials and local state

Never commit OAuth values, market-data keys, sessions, arming documents, broker
responses, operational runtime configuration, local secret files, or runtime
databases. Prefer the service or OS environment for secrets. A permitted local
development fallback must be an ignored, owner-only regular file and is never
valid as a source for the production arming secret. The repository hygiene
check enforces the tracked-index boundary.

Removing a credential from the current tree does not remove it from Git
history. Previously exposed E*TRADE keys must still be revoked and rotated, and
history cleanup must be coordinated separately.

## Architecture and operating rules

- [`docs/project_overview.md`](etrade_python_client/docs/project_overview.md)
  maps the current backtest, live, dashboard, and regime flows.
- [`docs/production_readiness_upgrade_plan.md`](etrade_python_client/docs/production_readiness_upgrade_plan.md)
  records the phased production-readiness gates.
- [`docs/runtime_configuration.md`](etrade_python_client/docs/runtime_configuration.md)
  defines the typed non-secret runtime configuration and local-secret boundary.
- [`RELIABILITY.md`](etrade_python_client/RELIABILITY.md) is the incident and
  invariant ledger.
- [`docs/market_regime_detect_specs.md`](etrade_python_client/docs/market_regime_detect_specs.md)
  is mandatory for every regime-aware backtest or live-model change.

Backtests that cannot prove their calibration cutoff, out-of-sample range,
causal inference method, one-session regime lag, and resolved-only probability
buckets are `UNVERIFIED` or `INVALID`; they must not be presented as valid
performance evidence.
