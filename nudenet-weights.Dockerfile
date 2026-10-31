# Weight-only OCI image: NudeNet v3.4 ONNX detectors for Movie-Trimmer.
# Baked at build time so the scanner image (and every replica) ships the
# exact weights version; the OCI source label links the package to the repo.
#
# Build & push:
#   docker build -f nudenet-weights.Dockerfile \
#     -t ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1 .
#   docker push ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1
FROM scratch
LABEL org.opencontainers.image.title="Movie-Trimmer NudeNet weights" \
      org.opencontainers.image.description="NudeNet v3.4 YOLOv8 nudity detectors (320n @320px, 640m @640px) consumed by the Movie-Trimmer NSFW scanner" \
      org.opencontainers.image.source="https://github.com/DarkStar1997/Movie-Trimmer" \
      org.opencontainers.image.version="v1" \
      org.opencontainers.image.licenses="AGPL-3.0"
COPY models/nudenet_320n_v1.onnx /models/nudenet_320n_v1.onnx
COPY models/nudenet_640m_v1.onnx /models/nudenet_640m_v1.onnx
