# All-in-one appliance: scanner API + VLM arbiter (llama-server) in ONE
# container — the full pipeline (scan -> arbiter -> crowd -> GPU blur
# render) runs from a single `docker compose up -d` / `docker run`.
#
# Composed from the two published component images (no rebuild of
# llama.cpp, no re-copy of weights from models/ — the VLM image already
# carries them): /opt/llama binaries + Qwen3-VL-30B-A3B GGUF from the
# sidecar image, everything else from the scanner image. Same GPU budget
# as the two-container pair: llama-server ~20.5GB + TRT engines ~2GB of 24.
#
# Build (components are ARGs; build them first when they change):
#   docker build -f scanner.Dockerfile \
#     -t ghcr.io/darkstar1997/movie-trimmer-scanner:v1 .
#   docker build -f vlm.Dockerfile \
#     --build-arg MODEL_DIR=models/qwen3-vl-30b-a3b \
#     --build-arg MODEL_FILE=Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
#     --build-arg MMPROJ_FILE=mmproj-BF16.gguf \
#     -t ghcr.io/darkstar1997/movie-trimmer-vlm-qwen3vl-30b-a3b:v1 .
#   docker build -f allinone.Dockerfile \
#     -t ghcr.io/darkstar1997/movie-trimmer:v1 .
#
# Run (volumes = engine cache, movie dir, blur outputs):
#   docker run -d --gpus all -p 8081:8081 \
#     -v $PWD/trt_cache:/cache/trt -v $PWD:/movies:ro \
#     -v $PWD/blur_outputs:/blur_outputs ghcr.io/darkstar1997/movie-trimmer:v1
#   # or just: docker compose up -d
#
# The scanner-only and sidecar-only images remain published for hosts that
# want independent lifecycles; see NSFW_SCANNER_DESIGN.md §10.

ARG SCANNER_IMAGE=ghcr.io/darkstar1997/movie-trimmer-scanner:v1
ARG VLM_IMAGE=ghcr.io/darkstar1997/movie-trimmer-vlm-qwen3vl-30b-a3b:v1

FROM ${VLM_IMAGE} AS vlm
FROM ${SCANNER_IMAGE}

ARG MODEL_FILE=Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf
ARG MMPROJ_FILE=mmproj-BF16.gguf

COPY --from=vlm /opt/llama /opt/llama
COPY --from=vlm /models /models
COPY docker/allinone-entrypoint.sh /usr/local/bin/allinone-entrypoint.sh
RUN chmod +x /usr/local/bin/allinone-entrypoint.sh

# llama-server knobs (same as the sidecar image; loopback only — nothing
# outside the container can reach it) + the scanner pointing at it.
# LD_LIBRARY_PATH must add the pip-wheel CUDA libs (torch's own nvidia/*
# wheels carry cudart/cublas/nccl) — the scanner base keeps them outside
# the default ld path, unlike the vlm image's nvidia/cuda base.
ENV LD_LIBRARY_PATH=/opt/llama:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
    MODEL=/models/${MODEL_FILE} \
    MMPROJ=/models/${MMPROJ_FILE} \
    CTX=8192 \
    PARALLEL=4 \
    EXTRA_ARGS="-ub 2048" \
    VLM_PORT=8082 \
    API_PORT=8081 \
    ARBITER_BASE_URL=http://127.0.0.1:8082

# API health == container health (the entrypoint already gated startup on
# the arbiter). start-period covers the ~17GB model load on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=10 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('API_PORT','8081') + '/health', timeout=3).status == 200 else 1)"

EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/allinone-entrypoint.sh"]
