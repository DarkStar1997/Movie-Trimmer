# Scanner image: NudeNet ONNX -> TensorRT FP16, weights baked in from
# ghcr.io/darkstar1997/movietrimmer-nudenet-weights; RF-DETR Seg Medium from
# ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights. The VLM arbiter is
# NOT bundled: it runs as a llama-server sidecar (Qwen3-VL-30B-A3B MoE)
# reached over ARBITER_BASE_URL.
# NOTHING downloads at runtime: HF_HUB_OFFLINE=1 turns any accidental hub
# access into a hard error. Engine cache is served by a bind mount.
#
# Build:
#   docker build -f scanner.Dockerfile -t ghcr.io/darkstar1997/movie-trimmer-scanner:v1 .
# Run the whole pipeline in ONE container (scanner + arbiter inside):
#   docker compose up -d          # builds allinone.Dockerfile from this
#                                 # image + the vlm sidecar image
# (independent-lifecycle hosts can still run scanner + sidecar separately)
#
# Base matches the visiontouch/fitness-coach stack (torch 2.14.0 / CUDA 13.0).
FROM ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1 AS weights
FROM ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1 AS seg-weights

FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    MODEL_DIR=/opt/models \
    VLM_DIR=/opt/vlm \
    TRT_CACHE_DIR=/cache/trt \
    SCAN_WORKSPACE_DIR=/scan_workspaces \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# PEP 668: the base python is externally managed. venv with
# --system-site-packages keeps the base's torch 2.14.0/cu130 visible while
# everything else installs into the venv (visiontouch pattern).
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# tensorrt-cu13 11.2.1.2: same pin as visiontouch (FP8 graph miscompiles
# under 11.1; engines are TRT-version keyed). Engine cache is keyed by
# gpu_slug/trt_slug/weights_v so a base bump never reuses a stale plan.
RUN pip install --no-cache-dir tensorrt-cu13==11.2.1.2
RUN pip install --no-cache-dir nvidia-modelopt[onnx]==0.45.0
# modelopt pins cupy-cuda12x by name; on the cu130 base we want the CUDA-13 build
RUN pip uninstall -y cupy-cuda12x && pip install --no-cache-dir cupy-cuda13x==14.2.0

RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    python-multipart==0.0.22 \
    opencv-python-headless==5.0.0.93 \
    onnx==1.23.2 \
    onnxruntime==1.31.1 \
    onnxslim

# stage B arbiter (Qwen3-VL via transformers; torch comes from the base).
# accelerate for device_map="cuda" sharding. The model itself is baked in
# below — no runtime downloads (HF_HUB_OFFLINE=1 enforces it).
RUN pip install --no-cache-dir transformers==5.16.1 accelerate

# stage C render: full-GPU-native NVDEC->torch->NVENC (audio passthrough,
# zero-copy DLPack frames; pycuda NOT needed — context via torch)
RUN pip install --no-cache-dir PyNvVideoCodec==2.2.3

# weights baked in (single source of truth: the weight images). The VLM
# arbiter ships in the sidecar image, not here — ARBITER_BASE_URL points
# at it (docker-compose.yml); an unset base URL falls back to in-process
# transformers IF weights are mounted into VLM_DIR (none bundled).
COPY --from=weights /models/nudenet_320n_v1.onnx /opt/models/
COPY --from=weights /models/nudenet_640m_v1.onnx /opt/models/
COPY --from=seg-weights /models/person_seg_rfdetr_seg_medium_v1.onnx /opt/models/

COPY scanner/ /app/scanner/
WORKDIR /app

# engines land in the bind-mounted cache on first run (~1-3 min), then reuse
VOLUME ["/cache/trt"]
EXPOSE 8081
CMD ["uvicorn", "scanner.main:app", "--host", "0.0.0.0", "--port", "8081", \
     "--timeout-keep-alive", "3600"]
