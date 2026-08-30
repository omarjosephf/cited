# Two stages. The first installs dependencies and downloads the embedding model;
# the second copies only what is needed to run, so build tooling and pip caches
# never reach the published image.

FROM python:3.12-slim AS build

# Compilers are needed to install some wheels and are not needed to run
# anything, which is precisely why this happens in a stage that gets discarded.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# `src/` is copied before the install because the build backend needs it to
# build the wheel, which means editing a docstring invalidates this layer and
# reinstalls every dependency, ONNX Runtime included.
#
# That is accepted rather than overlooked. Avoiding it needs either a stub
# package installed first and then overwritten, or a separate stage pinning
# fastembed on its own — and that second pin would silently control the layout
# of the model cache the runtime stage reads. Trading a build-cache miss for a
# version coupling that fails at runtime, on a project that rebuilds rarely, is
# a bad trade. Revisit if builds become frequent enough to be felt.
# LICENSE is here because pyproject.toml declares `license = { file = "LICENSE" }`,
# and hatchling reads it while generating metadata. Omitting it does not degrade
# the image, it stops the build outright with "License file does not exist" —
# which happens on the deploy platform, after a push, rather than locally.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --prefix=/install ".[api]"

# Bake the embedding model into the image (~64 MB, measured).
#
# Without this the first request downloads the model before it can answer, which
# on a cold container reads as "the demo is broken" rather than "the demo is
# loading". Doing it here also means a network failure breaks the build, where
# it is visible, rather than a visitor's first question.
#
# The path is passed explicitly rather than through an environment variable that
# fastembed would read: it reads none. `cache_dir` is a constructor argument,
# and `EMBEDDING_CACHE_DIR` below is this project's own variable, honoured by
# `FastEmbedEmbedder`. Setting a plausible-looking `FASTEMBED_CACHE_PATH` does
# nothing at all — verified, after writing exactly that mistake.
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

# Embed every corpus in the image, here, on a build machine that is not
# throttled. Measured on the portfolio corpus: 7.4s of an 8.4s cold start goes
# into this, and three to four minutes of it on the deployment's shared CPU
# slice, where the calling route gives up after twenty.
#
# Built here rather than committed beside each corpus, because a derived file
# checked in next to its source is a file that can be forgotten. Building it
# from the corpus that is in this image means the two cannot disagree by
# omission. They can still disagree if the chunker or `Chunk.indexed_text`
# changes without a rebuild, which is why `vectors.py` binds each matrix to a
# digest of the strings it embedded, and the service refuses to start on a
# mismatch rather than quietly re-embedding.
#
# One file per corpus, named for its deployment. Which one is read is decided by
# CORPUS_VECTORS_FILE at run time, next to the CORPUS_DIR it has to match. It is
# deliberately NOT defaulted below: an image pointed at somebody else's
# documents should embed them at startup, not fail a digest check for vectors it
# was never told about.
COPY content/ ./content/
COPY deploy/ ./deploy/
RUN PYTHONPATH=/install/lib/python3.12/site-packages \
    EMBEDDING_CACHE_DIR=/opt/models \
    sh -eux -c 'mkdir -p /opt/vectors; \
      python -m assistant.cli --corpus content embed --out /opt/vectors/content.npz; \
      for corpus in deploy/*/content; do \
        [ -d "$corpus" ] || continue; \
        app=$(basename "$(dirname "$corpus")"); \
        python -m assistant.cli --corpus "$corpus" embed --out "/opt/vectors/$app.npz"; \
      done'


FROM python:3.12-slim AS runtime

# Runs as a non-root user. The process only ever reads its own files and makes
# outbound HTTPS calls, so root buys nothing and costs the usual: any code
# execution bug becomes a root code execution bug.
RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build --chown=app:app /opt/models /opt/models
# Roughly 100 KB per corpus, read once at startup instead of being recomputed.
COPY --from=build --chown=app:app /opt/vectors /opt/vectors
COPY --chown=app:app src/ ./src/
COPY --chown=app:app content/ ./content/
# Corpus artifacts staged by another repository, for deployments that serve
# someone else's documents. Empty for the default image — `deploy/.gitkeep` is
# tracked precisely so this COPY has a source and the default build does not
# break when the staging area is cleaned.
#
# One image, many corpora: which one is served is decided by CORPUS_DIR at run
# time, so a corpus change is a redeploy of the same code rather than a fork of
# it. The service refuses to start if what it finds does not match the checksum
# the artifact carries.
COPY --chown=app:app deploy/ ./deploy/
# LICENSE travels with the image: this is AGPL-3.0, and §13 obliges anyone
# running it as a network service to offer the source to its users. Shipping the
# terms alongside the code is the least that requires.
COPY --chown=app:app pyproject.toml README.md LICENSE ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    EMBEDDING_CACHE_DIR=/opt/models \
    PORT=8080

USER app
EXPOSE 8080

# Bound to 0.0.0.0 because the platform's proxy reaches the container over its
# private network; 127.0.0.1 would make the service unreachable while appearing
# to start correctly.
#
# A single worker is deliberate. Each one loads its own copy of the model, and
# the daily budget lives in process memory — a second worker would quietly
# double the ceiling it is supposed to enforce.
CMD ["sh", "-c", "uvicorn assistant.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
