"""Scanner configuration: paths, models, defaults.

Paths are env-overridable so the Docker image (models baked into /opt/models,
bind-mounted engine cache) and a host checkout share one code path.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Baked-in weights (container: /opt/models, host checkout: ./models)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
# Baked-in VLM for the arbiter stage (container: /opt/vlm, host: ./models).
# Weights ship in the scanner image — nothing downloads at runtime
# (HF_HUB_OFFLINE=1 in the container enforces it).
VLM_DIR = Path(os.environ.get("VLM_DIR", "models"))
# TRT engine cache — bind-mount this in the container (see scanner.Dockerfile).
TRT_CACHE_DIR = Path(os.environ.get("TRT_CACHE_DIR", "trt_cache"))
# Uploaded scan inputs (host or container-local scratch)
SCAN_WORKSPACE_DIR = Path(os.environ.get("SCAN_WORKSPACE_DIR", "scan_workspaces"))

# --- Models ----------------------------------------------------------------
# name -> (onnx filename stem, inference resolution). Resolution must match the
# model the weights were trained at (NudeNet v3.4: 320n@320, 640m@640).
MODELS: dict[str, tuple[str, int]] = {
    "320n": ("nudenet_320n_v1.onnx", 320),
    "640m": ("nudenet_640m_v1.onnx", 640),
}
DEFAULT_MODEL = "640m"

# Decode resolution fed to the detector. Wider than tall: the detector
# letterbox-pads to square itself (reference NudeNet behaviour), so a 16:9
# frame arrives as full width + black bottom pad. Full-width detail is what
# matters; height padding costs the model nothing we can recover anyway.
DECODE_WIDTH = 640
DECODE_HEIGHT = 360

# --- Scan defaults ----------------------------------------------------------
DEFAULT_SAMPLE_FPS = 4.0
DEFAULT_BATCH_SIZE = 16
MAX_CONCURRENT_SCANS = 1

# --- NudeNet label taxonomy (EXACT order of ONNX output channels 4..21) -----
NUDENET_LABELS: list[str] = [
    "FEMALE_GENITALIA_COVERED",   # 0
    "FACE_FEMALE",                # 1
    "BUTTOCKS_EXPOSED",           # 2
    "FEMALE_BREAST_EXPOSED",      # 3
    "FEMALE_GENITALIA_EXPOSED",   # 4
    "MALE_BREAST_EXPOSED",        # 5
    "ANUS_EXPOSED",               # 6
    "FEET_EXPOSED",               # 7
    "BELLY_COVERED",              # 8
    "FEET_COVERED",               # 9
    "ARMPITS_COVERED",            # 10
    "ARMPITS_EXPOSED",            # 11
    "FACE_MALE",                  # 12
    "BELLY_EXPOSED",              # 13
    "MALE_GENITALIA_EXPOSED",     # 14
    "ANUS_COVERED",               # 15
    "FEMALE_BREAST_COVERED",      # 16
    "BUTTOCKS_COVERED",           # 17
]

# --- Default label policy ----------------------------------------------------
# triggers  -> any qualifying detection marks the frame NSFW (auto-cut segment)
# review    -> flagged separately for human review, never auto-cut
DEFAULT_TRIGGER_LABELS: dict[str, float] = {
    "FEMALE_GENITALIA_EXPOSED": 0.45,
    "MALE_GENITALIA_EXPOSED": 0.45,
    "ANUS_EXPOSED": 0.45,
    "FEMALE_BREAST_EXPOSED": 0.45,
    "BUTTOCKS_EXPOSED": 0.45,
}
DEFAULT_REVIEW_LABELS: dict[str, float] = {
    "BELLY_EXPOSED": 0.60,
    "ARMPITS_EXPOSED": 0.60,
    "FEET_EXPOSED": 0.60,
}
# Ignore boxes smaller than this fraction of frame area (posters, background
# TVs, tiny far-away figures).
DEFAULT_MIN_BOX_AREA_FRAC = 0.005

# --- Temporal smoothing defaults ---------------------------------------------
# (see temporal.SmoothingConfig docstring for semantics)
DEFAULT_SMOOTHING = {
    "open_min_consecutive": 2,     # sampled frames to open a segment
    "close_min_consecutive": 2,    # clean frames to close it
    "bridge_gap_s": 2.0,           # merge on/off flicker shorter than this
    "drop_below_s": 1.0,           # discard blips shorter than this ...
    "drop_rescue_score": 0.50,     # ... unless mean trigger score above this
                                   # (0.70 kept killing real sub-second
                                   # flashes — e.g. breast flash @3150.5 in
                                   # Ex Machina GT, mean ~0.52, dropped)
    "pad_s": 1.5,                  # safety padding on both ends
}

# --- Detector post-processing (parity with nudenet 3.4.2 reference) ----------
DETECT_SCORE_THRESHOLD = 0.2   # candidate collection threshold
NMS_SCORE_THRESHOLD = 0.25    # NMS entry threshold
NMS_IOU_THRESHOLD = 0.45     # class-agnostic, as in the reference

# --- Stage A: tracking (explicit-hold bridging) --------------------------------
# ByteTrack-style person-cluster tracking (scanner/tracking.py). Deterministic,
# no extra model: keeps a scene flagged while its person track (which last
# carried an explicit label within TRACK_EXPLICIT_HOLD_S) stays alive through
# sub-threshold flicker. A *_COVERED counterpart clears the hold (re-clothed).
DEFAULT_TRACK = True
TRACK_EXPLICIT_HOLD_S = 8.0
TRACK_REFRESH_FLOOR = 0.15    # sub-trigger explicit score that still refreshes
TRACK_MAX_LOST_SAMPLES = 3    # coasting samples (~0.75s @4fps) before retirement

# --- Stage C: person segmentation + blur render --------------------------------
# RF-DETR Seg Medium (COCO person class only at runtime; FP32 TRT engine).
PERSON_SEG = {
    "onnx": "person_seg_rfdetr_seg_medium_v1.onnx",
    "resolution": 432,
}
# Blur render defaults. Blur region = person mask AND padded explicit box.
BLUR_KEYFRAME_FPS = 8.0     # NudeNet+personseg rate inside flagged ranges
BLUR_BOX_PAD_FRAC = 0.35    # explicit box padding before mask intersection
BLUR_MASK_DILATE_FRAC = 0.012   # mask dilation, fraction of frame width
BLUR_PIXELS = 24            # pixelate block size at 1080p (scaled by width)
# Fallback chain, first usable wins. NB: av1_nvenc is deliberately absent —
# it fails with -22 EINVAL on this host/driver even bare (rawvideo-pipe path),
# while hevc/h264 accept the full quality args. libx264 = CPU last resort.
BLUR_CODECS = ("hevc_nvenc", "h264_nvenc", "libx264")
BLUR_CQ = 26                # NVENC constant-quality (libx264 crf equivalent)
# Warning overlay: every flagged frame in the final render carries red
# top-left text "VULGARITY DETECTED" + confidence% + label. This is the
# remedy for mosaic-invisible misses on ultra-smooth/dim scenes: the
# viewer always sees WHAT was flagged even when the blur itself is subtle.
# Env kill-switch: BLUR_OVERLAY=0 disables.
BLUR_OVERLAY = os.environ.get("BLUR_OVERLAY", "1").lower() not in (
    "0", "false", "no", "off")
# VLM-sourced (arbiter-found) segments have no NudeNet box; when the person
# segmenter also finds nobody (extreme close-ups), mask this fraction of the
# frame centered — conservative by design (VLM confidence is >=0.9 there).
BLUR_VLM_CENTER_FRAC = 0.7
# Kill a decode that produced no frame for this long (NVDEC wedge quirk)
# instead of blocking the job forever; pipe-path callers retry once on CPU.
BLUR_STALL_TIMEOUT_S = float(os.environ.get("BLUR_STALL_TIMEOUT_S", "120"))
# Writable output dir (bind-mount it; movies are mounted read-only)
BLUR_OUTPUT_DIR = str(Path(os.environ.get("BLUR_OUTPUT_DIR", "blur_outputs")))

# --- Stage D: crowd timestamps (CringeMDB, opt-in) -------------------------------
# POST /scan with crowd=true: fetch the movie's Scene-by-Scene guide from
# CringeMDB (slug resolved from the filename's Title.Year pattern) and
# merge in-scope category ranges as first-class segments — same Stage C
# treatment as VLM blind spots (all-person mask + tracking + overlay).
# Fail-soft: fetch/parse errors are reported in result['crowd'], the scan
# proceeds. Community entries are unverified and "off by a minute or so"
# (their words) — add-only, never removes pipeline-found segments.
DEFAULT_CROWD = False
CROWD_CATEGORIES = tuple(
    os.environ.get("CROWD_CATEGORIES", "nudity,sex").split(","))
CROWD_PAD_S = float(os.environ.get("CROWD_PAD_S", "1.5"))
CROWD_SCORE = 0.90          # overlay confidence shown for crowd segments
CROWD_FETCH_TIMEOUT_S = float(os.environ.get("CROWD_FETCH_TIMEOUT_S", "15"))
CROWD_CACHE_DIR = os.environ.get(
    "CROWD_CACHE_DIR",
    str(Path(os.environ.get("SCAN_WORKSPACE_DIR", "scan_workspaces"))
        / "crowd_cache"))

# --- Stage B: VLM arbiter (sparse Qwen3-VL pass) -------------------------------
# Never every frame: verify N frames per final segment, sample windows around
# segments (flicker gaps / short continuations), plus a coarse global sweep for
# true blind spots — capped at ARBITER_MAX_FRAMES per movie.
DEFAULT_ARBITER = False
# Bare name resolved against VLM_DIR first (baked weights); an HF repo id
# also works on a networked host (the container runs offline by design).
DEFAULT_ARBITER_MODEL = "Qwen3-VL-8B-Instruct"
ARBITER_VERIFY_FRAMES = 3       # per segment, at 20/50/80% of its span
ARBITER_WINDOW_BEFORE_S = 10.0  # sampled before each segment start
ARBITER_WINDOW_AFTER_S = 45.0   # sampled after each segment end
ARBITER_WINDOW_STEP_S = 1.0     # sample spacing inside windows
ARBITER_SWEEP_EVERY_S = float(os.environ.get("ARBITER_SWEEP_EVERY_S", "10.0"))
ARBITER_MAX_FRAMES = int(os.environ.get("ARBITER_MAX_FRAMES", "1100"))
# The sweep is the ONLY global blind-spot coverage and used to lose the
# budget race entirely (47 segments: verify 141 + windows left ~9 sweep
# frames for a 108-min movie). It now gets a guaranteed share of the cap;
# windows refine around known segments with the remainder.
ARBITER_SWEEP_BUDGET_FRAC = float(
    os.environ.get("ARBITER_SWEEP_BUDGET_FRAC", "0.6"))
# A sweep hit is a POINT sample of a scene of unknown extent (window hits
# sit on a 1s grid, so 1.5s pad finds their edges; sweep hits do not).
# Pad each sweep hit by this much when clustering into new segments.
ARBITER_SWEEP_PAD_S = float(os.environ.get("ARBITER_SWEEP_PAD_S", "8.0"))
# Sub-threshold NudeNet trigger hits (score in [ARBITER_WEAK_MIN_SCORE,
# the label's trigger threshold)) never reach smoothing, but in the dim/
# distant blind-spot region they are the detector's only voice (Ex Machina
# GT: FEMALE_GENITALIA 0.25-0.26 @3327/3342.5, BUTTOCKS 0.36-0.44 @3388 —
# raw scores below the 0.45 trigger). Cluster peaks go to the VLM as
# targeted "weak" frames; a confirmed hit becomes a padded add-only
# segment exactly like a sweep hit.
ARBITER_WEAK_MIN_SCORE = float(
    os.environ.get("ARBITER_WEAK_MIN_SCORE", "0.25"))
ARBITER_WEAK_MAX_FRAMES = int(
    os.environ.get("ARBITER_WEAK_MAX_FRAMES", "120"))
# Second pass: measure the extent of explicit point hits (sweep/weak) by
# re-sampling their ±ARBITER_SWEEP_PAD_S neighborhood at window density.
# Capped; highest-confidence hits expand first.
ARBITER_EXPAND_MAX_FRAMES = int(
    os.environ.get("ARBITER_EXPAND_MAX_FRAMES", "96"))
ARBITER_MODES = ("verify", "apply")  # verify: annotate; apply: also merge/remove
ARBITER_FRAME_W = 1024          # extraction width for VLM frames
# Off-box arbiter: when ARBITER_BASE_URL is set (e.g. a llama-server
# sidecar speaking the OpenAI /v1/chat/completions API), judgments go
# over HTTP and the in-process transformers VLM is NEVER loaded — same
# prompts, same parse/aggregate path, ~17GB VRAM freed for the sidecar.
# Empty (default) = in-process transformers with the baked weights.
ARBITER_BASE_URL = os.environ.get("ARBITER_BASE_URL", "")
ARBITER_HTTP_CONCURRENCY = int(
    os.environ.get("ARBITER_HTTP_CONCURRENCY", "4"))
ARBITER_HTTP_TIMEOUT_S = float(
    os.environ.get("ARBITER_HTTP_TIMEOUT_S", "120"))
