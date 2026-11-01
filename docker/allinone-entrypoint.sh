#!/bin/sh
# All-in-one launcher: scanner API (uvicorn) only. The VLM arbiter
# (llama-server) is ON-DEMAND — scanner/vlm.py spawns it when arbiter
# work starts and stops it when the pass ends, so the ~20GB of VRAM
# stays free for detector/blur jobs (see NSFW_SCANNER_DESIGN.md §13).
#
# Env consumed by the supervisor (same knobs the old boot path used):
# VLM_PORT via ARBITER_BASE_URL, MODEL, MMPROJ, DRAFT, CTX, PARALLEL,
# EXTRA_ARGS, VLM_START_TIMEOUT_S.
set -e

API_PORT="${API_PORT:-8081}"

echo "[allinone] starting scanner API on :${API_PORT} (VLM on-demand)"

exec uvicorn scanner.main:app --host 0.0.0.0 --port "$API_PORT" \
    --timeout-keep-alive 3600
