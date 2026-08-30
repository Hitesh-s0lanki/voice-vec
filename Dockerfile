# syntax=docker/dockerfile:1.9

# The backend *is* this repository's root — frontend/ is the other half, and
# .dockerignore keeps it out of the context. So the build is run from here:
#
#     docker build -t voice-vec .
#
# Two stages, so the runtime image carries the virtualenv and the source, but
# none of the build tooling or lockfile machinery that produced them.

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.7

# --------------------------------------------------------------------------
# Stage 1 — builder: resolve the locked dependency set into /app/.venv.
# --------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# pyproject.toml declares no [build-system], so uv.lock records this project as
# `source = { virtual = "." }`: uv installs the dependencies and never builds a
# wheel of src/. That is why the sync needs only these two files bind-mounted —
# editing anything under src/ reuses this layer instead of re-resolving 100
# packages — and why the runtime imports work off PYTHONPATH rather than an
# installed distribution.
#
# --frozen installs exactly what uv.lock pins and never re-resolves. CI proves
# the lock is current (`uv lock --check` in backend-ci.yml), so a build must not
# silently drift from it.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev

COPY src ./src
COPY scripts ./scripts

# Nothing to prefetch. This used to bake ~670 MB of ONNX weights in, because
# the lifespan loaded them before accepting traffic and a cold container would
# otherwise fetch them from HuggingFace inside the platform's health-check
# window. Embedding is a call to `text-embedding-3` now
# (docs/25-no-local-embedder.md), so the image carries the source and the
# virtualenv and nothing else — and a deployment that has 512 MiB to live in
# can actually run it.

# A checkout that lost the executable bit (Windows, a zip export) would
# otherwise produce an image whose entrypoint cannot run.
RUN chmod +x scripts/docker-entrypoint.sh

# --------------------------------------------------------------------------
# Stage 2 — runtime.
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Unprivileged. A fixed uid/gid keeps bind-mounted file ownership predictable.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --no-create-home app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/app/.venv/bin:$PATH \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    RUN_MIGRATIONS=true \
    DATASET_DIR=/app/data/datasets

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

# One DuckDB file per attached dataset is written here at runtime
# (docs/18-datasets.md). Ephemeral unless the platform mounts a disk over it —
# which is the intended shape: a dataset is rebuilt from its source URL.
RUN mkdir -p /app/data/datasets && chown app:app /app/data /app/data/datasets

USER app

EXPOSE 8000

# Probes the same endpoint the platform does. urllib rather than curl, so the
# image needs no extra apt package; a non-200 raises and exits 1.
#
# 200 is the bar, not `"status":"ok"`. /health answers `degraded` — still 200 —
# when the reply model or the embedder is missing, and a process that is up and
# misconfigured must be reported as up, or a rolling deploy never completes and
# the real fault never surfaces.
#
# start-period is short because boot is short: there is no model to load, so
# the process is answering about as soon as uvicorn binds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=4).status==200 or exit(1)"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["serve"]
