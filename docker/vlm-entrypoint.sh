#!/bin/sh
# llama-server launcher: explicit local model + mmproj; optional MTP drafter
# (only when DRAFT points at a file — gemma-4 ships one, qwen3-vl does not).
# Env-tunable: HOST PORT MODEL MMPROJ DRAFT CTX PARALLEL EXTRA_ARGS
set -e

DRAFT_ARGS=""
if [ -n "$DRAFT" ] && [ -f "$DRAFT" ]; then
    DRAFT_ARGS="--model-draft $DRAFT --spec-type draft-mtp --spec-draft-n-max 4"
fi

exec /opt/llama/llama-server \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    $DRAFT_ARGS \
    -ngl 999 \
    -fa on \
    -c "$CTX" \
    -np "$PARALLEL" \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA_ARGS
