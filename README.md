# A simple FastAPI based server that handles movie trimming and subtitles merging

See `ANALYSIS.md` for a code review of the trimmer, and `NSFW_SCANNER_DESIGN.md`
for the auto-detection layer (`scanner/`) that finds NSFW/nudity segments and
feeds them to this trimmer.

Installation:

1. Make sure ffmpeg is installed and is in the path
2. Project is tested with **python 3.14** but should work with earlier versions as well
3. Run `pip install -r requirements.txt`

Running:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 3600 --workers 2
```

## NSFW Scanner

GPU scanner (NudeNet YOLOv8 → TensorRT FP16) that auto-detects explicit
segments and emits `remove` segments for the trimmer.

```bash
# build & push weight image (models baked in, linked to this repo)
docker build -f nudenet-weights.Dockerfile -t ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1 .
docker push  ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1

# Scanner image (weights copied from the weight image; torch 2.14 / CUDA 13.0 / TRT-cu13)
docker build -f scanner.Dockerfile -t ghcr.io/darkstar1997/movie-trimmer-scanner:v1 .

# Weight images (self-contained copies, linked to this repo):
#   ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1     NudeNet 320n + 640m
#   ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1  RF-DETR Seg Medium (stage C blurring)
#   ghcr.io/darkstar1997/movietrimmer-qwen3vl-weights:v1     Qwen3-VL-8B (stage B arbiter)
# All models are baked into the scanner image: nothing downloads at runtime
# (HF_HUB_OFFLINE=1 makes any accidental hub access a hard error).

# run — engine cache and all scan artifacts stay repo-local (gitignored):
# ./trt_cache (TRT engines, first run builds them ~1-3 min),
# ./scan_workspaces (upload scratch), ./scans (results)
docker run --gpus all -p 8081:8081 \
  -v "$PWD/trt_cache:/cache/trt" \
  -v "$PWD/blur_outputs:/outputs" -e BLUR_OUTPUT_DIR=/outputs \
  -v "$PWD:/movies:ro" \
  ghcr.io/darkstar1997/movie-trimmer-scanner:v1
```

```bash
# scan a mounted file
curl -X POST http://localhost:8081/scan -F video_path=/movies/film.mkv
# or upload: -F video=@film.mkv
curl http://localhost:8081/scan/{scan_id}     # segments (seconds)

# custom policy / sampling
curl -X POST http://localhost:8081/scan \
  -F video_path=/movies/film.mkv \
  -F sample_fps=6 \
  -F 'policy={"triggers": {"FEMALE_BREAST_EXPOSED": 0.4}, "review": {}}'

# stage A tracking (default on): person-cluster tracker bridges sub-threshold
# flicker inside a scene — add -F track=false to disable.
# stage B arbiter (opt-in): sparse Qwen3-VL-8B pass — 3 verify frames per
# segment, a 15s global sweep (guaranteed 50% of the 700-frame budget, so
# far-from-any-segment scenes still get sampled), and 1s-grid windows around
# segments. arbiter_mode=apply merges VLM-found segments in (add-only);
# verify reports verdicts without changing segments. Knobs (env): 
# ARBITER_SWEEP_EVERY_S, ARBITER_MAX_FRAMES, ARBITER_SWEEP_BUDGET_FRAC,
# ARBITER_SWEEP_PAD_S (sweep hits pad wide — point samples of unknown scenes).
# Measured: 108-min film scans in ~8 min total (13x realtime), 0 refusals.
curl -X POST http://localhost:8081/scan \
  -F video_path=/movies/film.mkv \
  -F arbiter=true -F arbiter_mode=apply

# stage C blur: mask-precision pixelation instead of cutting. Segments from
# a scan_id (its policy is reused), explicit [[start, end]] pairs, or full
# segment dicts (a saved scan's segments array — labels/mean_score survive,
# so arbiter-found "VLM" segments get masked); optional preview window.
curl -X POST http://localhost:8081/blur \
  -F video_path=/movies/film.mkv -F scan_id={scan_id}
curl -X POST http://localhost:8081/blur \
  -F video_path=/movies/film.mkv \
  -F 'segments=[[844.8, 859.5]]' \
  -F window_start=840 -F window_end=865    # fast preview render
curl http://localhost:8081/blur/{blur_id}   # status -> output path in /outputs
# full renders are one continuous GPU-native pass: NVDEC -> torch (zero-copy
# NV12) -> RF-DETR person-mask pixelation on the GPU -> NVENC (hevc, P5 hq,
# VBR + explicit CQ); frames never touch the CPU, audio/subs passthrough with
# bit-exact PTS — no cut boundaries, so the timeline cannot drift.
# Previews re-encode the window only (pipe path).
# masking: the FULL person is blurred (matched RF-DETR mask; padded box
# fallback for close-ups), tracked across detector flicker until they leave
# the frame; arbiter/VLM-found segments mask every person (centered-frame
# fallback). Every flagged frame also gets red top-left overlay text
# "VULGARITY DETECTED / <conf>% <label>" — visible even where the mosaic
# isn't (smooth/dim scenes). BLUR_OVERLAY=0 env disables it.

# feed a scan result to the trimmer (segments JSON = seconds)
curl -X POST http://localhost:8000/submit -F file=@film.mkv \
  -F subtitle_file=@film.srt \
  -F 'segments=[[12.5, 45.0], [620.25, 700.5]]'
```

Tests (host, CPU-only, no torch needed): `.venv/bin/python -m pytest tests/ -q`

