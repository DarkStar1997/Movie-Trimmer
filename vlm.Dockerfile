# VLM arbiter image — llama.cpp server (CUDA + flash attention, optional
# MTP drafter). Ships Qwen3-VL-30B-A3B-Instruct Q4_K_XL; any llama.cpp-
# compatible VLM (e.g. Gemma-4-12B-it + drafter, Qwen3-VL-8B) stays
# ARG-swappable. Weights live in the repo's gitignored models/ dir (same
# convention as the detector ONNXs); the image only COPYs them — nothing
# downloads at build or run time.
#
# Multi-stage:
#   builder  — llama.cpp v0.8.0 built from source, CUDA + FA all-quants,
#              arch pinned via CUDA_ARCH (default 89 = RTX 4090)
#   runtime  — CUDA runtime base, binaries + local weights
#
# One-time model download into the current directory (host):
#   HF_TOKEN=... hf download unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF \
#     Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
#     --local-dir models/qwen3-vl-30b-a3b
#
# Build (defaults = the shipped Qwen3-VL-30B-A3B Q4_K_XL):
#   docker build -f vlm.Dockerfile \
#     -t ghcr.io/darkstar1997/movie-trimmer-vlm-qwen3vl-30b-a3b:v1 .
# Any other candidate is ARGs away, e.g. Gemma-4-12B + its MTP drafter
# (download gemma-4-12B-it-qat-UD-Q4_K_XL.gguf, mmproj-BF16.gguf,
# mtp-gemma-4-12B-it.gguf into models/gemma-4-12b/ first):
#   docker build -f vlm.Dockerfile \
#     --build-arg MODEL_DIR=models/gemma-4-12b \
#     --build-arg MODEL_FILE=gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
#     --build-arg DRAFT_FILE=mtp-gemma-4-12B-it.gguf \
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

# candidate model, fully parameterized so one Dockerfile serves any
# llama.cpp-compatible VLM (defaults = the shipped 30B MoE; e.g.
# gemma-4-12b + MTP drafter via ARGs — empty DRAFT_FILE disables the
# speculative flags)
ARG MODEL_DIR=models/qwen3-vl-30b-a3b
ARG MODEL_FILE=Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf
ARG MMPROJ_FILE=mmproj-BF16.gguf
ARG DRAFT_FILE=

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

# --- stage 2: runtime ------------------------------------------------------------
# Weights come from the local models/ dir in the build context (one-time
# host download, see header) — no network in this stage.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1001 llama
# v0.8.0 refactors llama-server into a thin binary + libllama-server-impl.so
# (plus the ggml/llama shared libs) — copy the whole bin dir together so the
# $ORIGIN rpaths keep resolving
COPY --from=builder /src/build/bin/ /opt/llama/
COPY docker/vlm-entrypoint.sh /usr/local/bin/entrypoint.sh
# global ARGs must be re-declared per stage or they expand empty — an
# empty MODEL_DIR once COPY'd the whole repo into the image (32GB lesson)
ARG MODEL_DIR
ARG MODEL_FILE
ARG MMPROJ_FILE
ARG DRAFT_FILE
COPY --chown=1001:1001 ${MODEL_DIR}/ /models/
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R llama:llama /opt/llama
USER llama
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/opt/llama \
    PORT=8082 \
    HOST=0.0.0.0 \
    MODEL=/models/${MODEL_FILE} \
    MMPROJ=/models/${MMPROJ_FILE} \
    DRAFT=${DRAFT_FILE:+/models/${DRAFT_FILE}} \
    CTX=8192 \
    PARALLEL=4 \
    EXTRA_ARGS=""
EXPOSE 8082
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -sf "http://localhost:${PORT}/health" || exit 1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
