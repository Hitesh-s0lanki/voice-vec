#!/bin/sh
# Container entrypoint. `serve` (the default CMD) starts the API; `migrate`
# runs the schema check on its own so the image can be used as a one-shot
# migration runner; anything else is executed verbatim, which is what makes
# `docker run <image> python -m scripts.migrate` and a debugging shell work.
set -eu

run_migrations() {
    # There is no chunk table and no corpus here — this creates the tables the
    # app itself owns (conversations, tool calls, connected accounts, profiles,
    # datasets) and proves the DSN resolves. Every ensure_schema is idempotent
    # and the stores call their own on first use, so this is not a
    # prerequisite; it runs at deploy time because a wrong DSN is cheap to find
    # here and expensive to find from a spoken turn.
    echo "==> python -m scripts.migrate"
    python -m scripts.migrate
}

case "${1:-serve}" in
    serve)
        if [ "${RUN_MIGRATIONS:-true}" = "true" ] && [ -n "${DATABASE_URL:-}" ]; then
            # Deliberately fatal. The app degrades to "this build does not
            # remember anything" when DATABASE_URL is unset, which is a
            # supported deployment — but a DATABASE_URL that is set and
            # unreachable is a misconfiguration, and a deploy that serves
            # through it would lose every turn silently.
            run_migrations
        elif [ -z "${DATABASE_URL:-}" ]; then
            echo "==> DATABASE_URL unset — skipping migrations; conversations will not be saved"
        else
            echo "==> RUN_MIGRATIONS is not 'true' — skipping migrations"
        fi

        # Not `python -m src.main`: that entrypoint runs uvicorn with
        # reload=True, which is a development server. Same app, production
        # server settings.
        #
        # WEB_CONCURRENCY stays at 1 by default: this server's work is waiting
        # on upstream providers rather than burning local CPU, so a second
        # worker mostly buys another copy of the process. It is now safe to
        # raise on a box with the memory for it — the ~700 MiB ONNX session
        # each worker used to load is gone (docs/25-no-local-embedder.md).
        #
        # proxy-headers, because Render and every other platform terminate TLS
        # in front of this and the voice socket needs the real scheme.
        echo "==> uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000} (${WEB_CONCURRENCY:-1} worker(s))"
        exec uvicorn src.main:app \
            --host "${HOST:-0.0.0.0}" \
            --port "${PORT:-8000}" \
            --workers "${WEB_CONCURRENCY:-1}" \
            --proxy-headers \
            --forwarded-allow-ips '*' \
            --timeout-graceful-shutdown 20
        ;;
    migrate)
        run_migrations
        ;;
    *)
        exec "$@"
        ;;
esac
