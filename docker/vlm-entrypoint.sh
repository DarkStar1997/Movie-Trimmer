#!/bin/sh
# llama-server launcher: explicit local model + mmproj + MTP drafter.
# Env-tunable: HOST PORT MODEL MMPROJ DRAFT CTX PARALLEL EXTRA_ARGS
set -e

exec /opt/llama/llama-server \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --model-draft "$DRAFT" \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    -ngl 999 \
    -fa on \
    -c "$CTX" \
    -np "$PARALLEL" \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA_ARGS
