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

# There is no model to load any more (docs/25-no-local-embedder.md), so this
# should bind in about a second. The retries are for a slow runner, not for a
# slow boot.
for _ in $(seq 1 30); do
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
# Not `embedder_ready`: with no OPENAI_API_KEY there is nothing to embed with,
# and reporting that honestly is exactly what this run is checking. What must
# be true is that the app booted and answered as itself.
grep -q '"service":"voice-vec"' /tmp/vec-health.json \
    || { echo '::error::/health did not answer as this service'; exit 1; }
grep -qE '"status":"(ok|degraded)"' /tmp/vec-health.json \
    || { echo '::error::/health reported neither ok nor degraded'; exit 1; }

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
