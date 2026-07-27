# Reproducible Python environments

`pyproject.toml` is the only direct dependency definition. The committed lock
files pin the complete transitive environment and every accepted distribution
hash for CPython 3.10.

Bootstrap the exact build tools:

```bash
python -m pip install \
  --require-hashes \
  --only-binary=:all: \
  -r requirements/build.lock
```

Install the production runtime:

```bash
python -m pip install \
  --require-hashes \
  --no-build-isolation \
  --only-binary=:all: \
  --no-binary=pyetrade,rauth \
  -r requirements/runtime.lock
```

Install the offline test and build environment:

```bash
python -m pip install \
  --require-hashes \
  --no-build-isolation \
  --only-binary=:all: \
  --no-binary=pyetrade,rauth \
  -r requirements/test.lock
```

`pyetrade==2.1.1` and its OAuth dependency `rauth==0.7.3` are the only approved
source-distribution exceptions because their publishers do not provide wheels.
Their source archives are hash-pinned, build isolation is disabled, and each
build runs only under `build.lock`. That bootstrap lock also pins
`poetry-core`, the backend declared by the `pyetrade` source distribution.

Regenerate both locks only with the Python version in `.python-version` and the
pinned `pip-tools` version declared by the `dev` extra:

```bash
python -m piptools compile pyproject.toml \
  --generate-hashes \
  --allow-unsafe \
  --resolver=backtracking \
  --strip-extras \
  --output-file=requirements/runtime.lock

python -m piptools compile pyproject.toml \
  --extra=dev \
  --generate-hashes \
  --allow-unsafe \
  --resolver=backtracking \
  --strip-extras \
  --output-file=requirements/test.lock
```

Normal regeneration does not use `--upgrade`. Dependency upgrades are explicit,
one reviewed direct pin at a time, followed by both lock compilations and the
artifact-first CI suite. Refresh `build.lock` from the matching entries in the
new test lock whenever a bootstrap-tool pin changes.
