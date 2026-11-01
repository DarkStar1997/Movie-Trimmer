#!/bin/sh
# All-in-one launcher: llama-server (VLM arbiter, loopback only) + uvicorn
# (scanner API) in one container. The scanner reaches the arbiter at
# ARBITER_BASE_URL=http://127.0.0.1:${VLM_PORT} — one container runs the
# whole pipeline: scan -> arbiter -> crowd -> GPU blur render.
#
# Ordering matters on a 24GB GPU: llama-server claims ~20.5GB at model
# load, so it must be healthy BEFORE the API starts taking requests that
# build TRT engines (~2GB). This mirrors the old two-container stack's
# depends_on: service_healthy.
#
# Env-tunable: VLM_PORT API_PORT MODEL MMPROJ DRAFT CTX PARALLEL EXTRA_ARGS
set -e

VLM_PORT="${VLM_PORT:-8082}"
API_PORT="${API_PORT:-8081}"

DRAFT_ARGS=""
if [ -n "$DRAFT" ] && [ -f "$DRAFT" ]; then
    DRAFT_ARGS="--model-draft $DRAFT --spec-type draft-mtp --spec-draft-n-max 4"
fi

/opt/llama/llama-server \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    $DRAFT_ARGS \
    -ngl 999 \
    -fa on \
    -c "$CTX" \
    -np "$PARALLEL" \
    --host 127.0.0.1 \
    --port "$VLM_PORT" \
    $EXTRA_ARGS &
VLM_PID=$!

cleanup() {
    kill "$VLM_PID" 2>/dev/null || true
}
trap cleanup TERM INT

# health-gate (python, not curl: the scanner runtime has no curl).
echo "[allinone] waiting for llama-server on 127.0.0.1:${VLM_PORT} (model load)..."
i=0
until python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${VLM_PORT}/health', timeout=2).status == 200 else 1)" >/dev/null 2>&1; do
    if ! kill -0 "$VLM_PID" 2>/dev/null; then
        echo "[allinone] llama-server exited during load" >&2
        exit 1
    fi
    i=$((i + 1))
    if [ "$i" -ge 300 ]; then
        echo "[allinone] llama-server health timeout (600s)" >&2
        exit 1
    fi
    sleep 2
done
echo "[allinone] llama-server healthy — starting scanner API on :${API_PORT}"

uvicorn scanner.main:app --host 0.0.0.0 --port "$API_PORT" \
    --timeout-keep-alive 3600 &
API_PID=$!

# the container lives with the API; TERM/INT (docker stop) reaps both via
# the cleanup trap
wait "$API_PID"
