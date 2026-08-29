#!/usr/bin/env bash
# Boot the server the way the container does and prove it answers.
#
#     ./scripts/smoke.sh [port]
#
# Used by backend-ci.yml against a runtime-only install (no dev group), so it
# catches the failure a unit test cannot: an import that only the test
# environment satisfied, or a lifespan that raises before the first request.
set -euo pipefail

PORT="${1:-8123}"
HOST=127.0.0.1
LOG="$(mktemp -t vec-smoke.XXXXXX)"

# No DATABASE_URL, no REDIS_URL and no provider keys on purpose: this asserts
# the boot path a deployment takes before any of its secrets are filled in.
# /health answers `degraded` in that state — still 200 — and that is the point.
export HOST PORT
export ENVIRONMENT=ci
export CORS_ORIGINS='["http://localhost:3002"]'
export EMBED_CACHE_DIR="${EMBED_CACHE_DIR:-data/models}"
# Off, so the warm loop cannot keep the process alive past the teardown below.
export KEEPALIVE_SECONDS=0

cleanup() {
    if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    echo '==> server log (tail)'
    tail -40 "$LOG" || true
    rm -f "$LOG"
}
trap cleanup EXIT

echo "==> booting uvicorn on ${HOST}:${PORT}"
uvicorn src.main:app --host "$HOST" --port "$PORT" > "$LOG" 2>&1 &
PID=$!

# Generous, because the first boot on a cold runner loads (and possibly
# downloads) the ONNX embedding model before the socket starts answering.
for _ in $(seq 1 150); do
    if curl -sf "http://${HOST}:${PORT}/health" -o /tmp/vec-health.json; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo '::error::the server exited during startup'
        exit 1
    fi
    sleep 2
done

echo '==> GET /health'
curl -fsS "http://${HOST}:${PORT}/health" -o /tmp/vec-health.json
cat /tmp/vec-health.json
echo
grep -q '"embedder_ready":true' /tmp/vec-health.json \
    || { echo '::error::the embedder did not load'; exit 1; }

echo '==> GET /openapi.json'
curl -fsS -o /tmp/vec-openapi.json -w 'HTTP %{http_code}\n' "http://${HOST}:${PORT}/openapi.json"
# A router that silently stopped being included would still leave /health
# green. `WS /voice/ws` cannot be asserted here — FastAPI keeps websocket
# routes out of the OpenAPI schema — so its sibling on the same router stands
# in for it, alongside the HTTP half of the same pipeline.
for path in '"/ask"' '"/voice/config"'; do
    grep -q "$path" /tmp/vec-openapi.json \
        || { echo "::error::$path is missing from the schema"; exit 1; }
done

echo '==> smoke passed'
