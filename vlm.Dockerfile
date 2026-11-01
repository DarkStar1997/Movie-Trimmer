# Gemma-4-12B-it VLM arbiter candidate — llama.cpp (CUDA + flash attention)
# with mmproj vision and the MTP speculative drafter, all baked in.
#
# Multi-stage:
#   builder    — llama.cpp v0.8.0 built from source, CUDA + FA all-quants,
#                arch pinned via CUDA_ARCH (default 89 = RTX 4090)
#   downloader — model + mmproj + MTP drafter pulled from HF with hf_transfer
#                (HF_TOKEN passed as a BuildKit secret, never baked in)
#   runtime    — CUDA runtime base, binaries + weights only; nothing
#                downloads at container start
#
# Build (token from the environment):
#   docker build -f vlm.Dockerfile \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t ghcr.io/darkstar1997/movie-trimmer-vlm-gemma:v1 .
# Run:
#   docker run --gpus all -p 8082:8082 \
#     ghcr.io/darkstar1997/movie-trimmer-vlm-gemma:v1
# Judge a frame (OpenAI-compatible):
#   POST /v1/chat/completions  {"messages":[{"role":"user","content":[
#     {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}},
#     {"type":"text","text":"..."}]}]}

ARG CUDA_VERSION=13.0.1
ARG LLAMA_TAG=v0.8.0
ARG CUDA_ARCH=89

# --- stage 1: build llama.cpp from source --------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder
ARG LLAMA_TAG
ARG CUDA_ARCH
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ccache \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --branch "${LLAMA_TAG}" --depth 1 https://github.com/ggml-org/llama.cpp
# GGML_CUDA_FA_ALL_QUANTS: flash-attention kernels for every quant mix, so the
# Q4_K_XL target and the Q4_0 MTP drafter both run under -fa on
RUN cmake -S llama.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DLLAMA_CURL=OFF \
    && cmake --build build --target llama-server -j"$(nproc)"

# --- stage 2: download weights at full speed -----------------------------------
# unsloth/gemma-4-12B-it-qat-GGUF (apache-2.0, public):
#   gemma-4-12B-it-qat-UD-Q4_K_XL.gguf  6.7 GB  target (QAT, unsloth dynamic)
#   mmproj-BF16.gguf                    175 MB  vision projector
#   mtp-gemma-4-12B-it.gguf             253 MB  MTP drafter (smart Q4_0,
#                                              root file -hf auto-detects)
FROM python:3.12-slim AS downloader
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]"
WORKDIR /models
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download unsloth/gemma-4-12B-it-qat-GGUF \
        gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
        mmproj-BF16.gguf \
        mtp-gemma-4-12B-it.gguf \
        --local-dir /models

# --- stage 3: runtime ------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1001 llama
# v0.8.0 refactors llama-server into a thin binary + libllama-server-impl.so
# (plus the ggml/llama shared libs) — copy the whole bin dir together so the
# $ORIGIN rpaths keep resolving
COPY --from=builder /src/build/bin/ /opt/llama/
COPY --from=downloader /models /models
COPY docker/vlm-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R llama:llama /models /opt/llama
USER llama
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/opt/llama \
    PORT=8082 \
    HOST=0.0.0.0 \
    MODEL=/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
    MMPROJ=/models/mmproj-BF16.gguf \
    DRAFT=/models/mtp-gemma-4-12B-it.gguf \
    CTX=8192 \
    PARALLEL=4 \
    EXTRA_ARGS=""
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -sf "http://localhost:${PORT}/health" || exit 1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
