# Reproducible build inputs

Use Python 3.12.13. The reviewed dependency graphs are
`requirements-runtime.lock`, `requirements-dev.lock` (runtime plus test/audit
tools), and `requirements-build.lock`. `requirements.lock` is a compatibility
entry point. CI uses the same pins and hashes as Docker. Do not resolve new
dependencies during project installation.

```sh
python -m venv .venv
# Activate the environment for your shell.
python -m pip install --require-hashes --only-binary=:all: -r requirements-dev.lock -r requirements-build.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m pip_audit --no-deps --disable-pip -r requirements-dev.lock -r requirements-build.lock
ruff check .
ruff format --check .
mypy src tests
pytest
python scripts/prepare_model.py --cache-dir .fastembed_cache
# Use EMBEDDING_CACHE_DIR=.fastembed_cache and HF_HUB_OFFLINE=1 for evaluations.
doc-assistant eval --suite demo --output eval/results/demo-retrieval.json
```

Run the portfolio suite against the revision in `eval/portfolio-source.json`;
CI checks out that exact revision. When portfolio corpus/questions change,
update this reference through a reviewed companion change. A Cited pass against
an older corpus does not qualify the new portfolio corpus for release.

## Updating locks

Locks were generated with uv 0.12.2, used as a compiler for pip requirements.
Installation remains pip-based. Preserve existing pins unless the change
intentionally updates them. Review package identity, versions, hashes, platform
markers and licenses; run the complete gate after changes.

```sh
uv pip compile pyproject.toml --extra api --universal --python-version 3.12 --generate-hashes --output-file requirements-runtime.lock --no-annotate --no-header
uv pip compile pyproject.toml --extra api --extra dev --constraint requirements-runtime.lock --universal --python-version 3.12 --generate-hashes --output-file requirements-dev.lock --no-annotate --no-header
uv pip compile requirements-build.in --constraint requirements-dev.lock --universal --python-version 3.12 --generate-hashes --output-file requirements-build.lock --no-annotate --no-header
```

The Dockerfile pins the Python image digest in both stages, installs only
hashed wheels, and installs the project with the locked build backend.
`model.lock.json` identifies the exact upstream revision and SHA-256 of all five
model/tokenizer files. `prepare_model.py` fetches that revision, verifies its
bytes, and binds the cache reference; container indexing and serving are offline.
Changing the lock requires model/vector compatibility and retrieval checks.

```sh
docker build --build-arg BACKEND_COMMIT="$(git rev-parse HEAD)" --tag cited:verify .
```

CI runs this build with no provider credentials. A Windows source/test pass is
not evidence that the Linux container built. If no container runtime is
available locally, record that limitation and require the CI container check
before release. Pinned inputs do not promise byte-identical OCI images across
builders, architectures or build timestamps; record the actual built image
digest and the actual vectors in the release manifest.

Sources: [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/),
[uv compile](https://docs.astral.sh/uv/pip/compile/).
