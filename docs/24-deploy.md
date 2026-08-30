# Deploying

The backend is this repository's root — `frontend/` is the other half — so the
image is built from the root with `.dockerignore` keeping the Next.js app, the
docs and `data/` out of the context.

```bash
docker build -t voice-vec .
docker run -p 8000:8000 --env-file .env voice-vec
```

## The image

Two stages. The builder resolves `uv.lock` into `/app/.venv`; the runtime
carries that virtualenv, `src/`, `scripts/` and the embedding model, under an
unprivileged uid 1001.

`pyproject.toml` declares no `[build-system]`, so uv records this project as
`source = { virtual = "." }` — it installs the dependencies and never builds a
wheel of `src/`. That is why the dependency layer needs only `pyproject.toml`
and `uv.lock` bind-mounted, and why imports work off `PYTHONPATH=/app` rather
than an installed distribution.

### Nothing is baked in

This image used to carry ~670 MB of ONNX weights, because the lifespan loaded an
embedding model before accepting traffic and a cold container would otherwise
have fetched them from HuggingFace inside the platform's health-check window.

That model is gone ([25-no-local-embedder.md](25-no-local-embedder.md)) — it held
~700 MiB resident, which a 512 MiB instance cannot pay — and embedding is a call
to `text-embedding-3` now. What is left in the image is the virtualenv and the
source:

| | Before | After |
| --- | --- | --- |
| Image | ~1.5 GB | **120 MB** |
| Resident at startup | 922 MiB | **184 MiB** |
| Boot to first response | ~7 s | **1.6 s** |

Verified under `--memory=512m --memory-swap=512m`: 182 MiB used, healthcheck
healthy.

### The entrypoint

`scripts/docker-entrypoint.sh` takes three shapes:

- `serve` (the default) — runs the migrations when `DATABASE_URL` is set, then
  execs uvicorn. Not `python -m src.main`: that entrypoint sets `reload=True`.
- `migrate` — `python -m scripts.migrate` alone, so the image is also the
  migration runner.
- anything else — executed verbatim.

A `DATABASE_URL` that is set and unreachable is fatal. An *unset* one is not:
the app degrades to "this build does not remember anything", which is a
supported deployment, and the log says so.

`WEB_CONCURRENCY` stays at 1 by default: this server waits on upstream
providers rather than burning local CPU, so a second worker mostly buys another
copy of the process. It is now safe to raise on a box with the memory for it —
the ~700 MiB ONNX session each worker used to load is gone.

`/app/data/datasets` holds one DuckDB file per attached dataset
([18-datasets.md](18-datasets.md)). It is ephemeral unless the platform mounts a
disk over it — which is the intended shape, since a dataset is rebuilt from its
source URL.

### Health

The `HEALTHCHECK` and the platform both probe `GET /health`, and the bar is
**200, not `"status":"ok"`**. `/health` answers `degraded` — still 200 — when
the reply model or the embedder is missing. A process that is up and
misconfigured must be reported as up, or a rolling deploy never completes and
the real fault never surfaces.

## The pipelines

Both live in `.github/workflows/` and are filtered by `paths-ignore` rather than
`backend/**`, because the backend is the root.

**`backend-ci.yml`** — correctness of the code.

| job | what it proves |
| --- | --- |
| `lock` | `uv.lock` matches `pyproject.toml`, and `requirements*.txt` still regenerate from it byte for byte |
| `test` | the suite passes on Settings' defaults, with no provider key |
| `test-postgres` | every `ensure_schema` builds a working schema on real pgvector, twice — the entrypoint runs them on every boot, so idempotence is load-bearing |
| `smoke` | the app boots and answers with `--no-dev` installed only |
| `backend-ci` | one stable check to require in branch protection |

**`backend-docker.yml`** — the artefact, on `main` and `release` only, so a fork
cannot push to the registry.

`build` publishes to GHCR and pins the digest. `verify` boots *that digest*
against a throwaway pgvector and asserts: `/health` is 200 with
it answered as this service (and `embedder_ready:true` when an
`OPENAI_API_KEY` was supplied, which is a promise only then),
`/ask` and `/voice/config` are in the schema, the entrypoint printed
`schema ready`, the process is uid 1001, and the image's own healthcheck agrees.
`configure` mirrors GitHub secrets and variables into the Render service one key
at a time. `deploy` posts the verified digest to the Render deploy hook.

Every one of those degrades to a no-op when its secret is absent, so the
pipeline is green on a repository where nothing has been configured yet and gets
stricter as each value is filled in.

### What to set

Secrets: `SARVAM_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `REDIS_URL`,
`CLERK_JWT_KEY`, `COMPOSIO_ENCRYPTION_KEY`, `AGENT_MEMORY_API_KEY`,
`RENDER_API_KEY`, `RENDER_DEPLOY_HOOK_URL`.

Variables: `RENDER_SERVICE_ID`, `CLERK_PUBLISHABLE_KEY`, `CORS_ORIGINS`,
`FRONTEND_URL`, `AGENT_MEMORY_ENDPOINT`, `AGENT_MEMORY_STORE_ID`, `TTS_SPEAKER`.

`CORS_ORIGINS` must be a **JSON array** — `["https://vec.example.com"]`.
`Settings.cors_origins` is a `list[str]`, and pydantic-settings parses complex
fields as JSON; a bare comma-separated string fails to boot. The `configure` job
writes a warning into the run summary when it sees one that is not.

`ENVIRONMENT`, `HOST`, `PORT`, `EMBED_CACHE_DIR` and `DATASET_DIR` are
deliberately *not* mirrored to Render — the image pins them, and a copy in the
dashboard would be a second place for them to drift.
