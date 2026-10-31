# Weight-only OCI image: Qwen3-VL-8B-Instruct for the Movie-Trimmer
# scanner's stage-B arbiter. Stock Apache-2.0 weights (Alibaba Qwen), staged
# at models/qwen3-vl-8b-instruct/ via huggingface_hub snapshot_download and
# re-shipped so the scanner image is fully self-contained — no HF downloads
# at runtime (enforced with HF_HUB_OFFLINE=1).
#
# Build & push (context = repo root; models/ is gitignored but buildable):
#   docker build -f qwen3vl-weights.Dockerfile \
#     -t ghcr.io/darkstar1997/movietrimmer-qwen3vl-weights:v1 .
#   docker push ghcr.io/darkstar1997/movietrimmer-qwen3vl-weights:v1
FROM scratch
LABEL org.opencontainers.image.title="Movie-Trimmer Qwen3-VL-8B-Instruct arbiter" \
      org.opencontainers.image.description="Stock Qwen3-VL-8B-Instruct (Apache-2.0, Alibaba Qwen) for the Movie-Trimmer scanner's sparse VLM arbiter: segment verification + blind-gap sampling." \
      org.opencontainers.image.source="https://github.com/DarkStar1997/Movie-Trimmer" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY models/qwen3-vl-8b-instruct/ /models/qwen3-vl-8b-instruct/
