# NSFW Auto-Detection Layer — Design Proposal

Status: **implemented** (`scanner/`, see README for build/run). Steps 1–3 of
the build order are done: parity-tested detector (ORT + TRT backends), policy,
temporal smoothing, `/scan` API, weight + scanner images on ghcr, bind-mounted
engine cache. Remaining: real-movie threshold tuning, review UI, trimmer
auto-submit wiring.
Goal: replace hand-written cut timestamps with an automatic scan pass that finds
NSFW / nudity / explicit-body-part segments in a movie and feeds them to the existing
trimmer as `remove` segments. Gore/violence explicitly **out of scope**.

---

## 1. Research summary (what's out there, 2026-10)

| Model | Type | Labels | Weights | TRT path | License | Verdict |
|---|---|---|---|---|---|---|
| **NudeNet 3.4** (`notAI-tech/NudeNet`, 2.5k★, active) | YOLOv8n (320n) / YOLOv8m (640m) object detector | 18 anatomy states: `FEMALE_BREAST_EXPOSED`, `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`, `BUTTOCKS_EXPOSED`, covered variants, belly/armpits/feet, face | official ONNX + PT from GitHub releases; pip `nudenet==3.4.2` | ONNX → TRT straightforward; raw output `[1,20,8400]`, NMS done in torch post-proc (or deepghs `nms-yolov8.onnx`) | code AGPL-3.0; weights repo has no explicit license | **Recommended primary** — exactly the right label granularity |
| `Pakaho-q1/yolo26-nsfw-labels` v3 | YOLO26s detector, 19 classes | same taxonomy family + covered/exposed | ships `.onnx` on HF | **NMS-free** (YOLO26 end-to-end) → cleanest TRT graph | AGPL-3.0 | Strong modern alternative; NMS-free helps TRT |
| `Falconsai/nsfw_image_detection` (1.7M dl/mo) | ViT-base binary (normal/nsfw) | 1 bit | HF; `onnx-community/...-ONNX` exists | easy | Apache-2.0 | Optional cheap arbiter / gate, not sufficient alone (no labels) |
| `viddexa/nsfw-detection-2-mini/-nano` | EfficientNet-B4/B0 5-class | nudity/porn tiers | HF | export via optimum | Apache-2.0, F1 96.1 (LSPD) | Optional second opinion |
| `deepghs/nudenet_onnx` | NudeNet 320n + NMS as ONNX | 18 | HF | reference impl | repo Apache-2.0 | Dev reference for pre/post-processing |
| Qwen3-VL (already used in visiontouch) | VLM | free-form | HF | TRT-LLM / raw torch | Apache-2.0 | Optional late-stage arbitration on ambiguous windows (dim light, paintings, distance shots) |

No credible **temporal/video-native** NSFW model surfaced (HF search is dominated by
generative LoRAs) → frame-sampled detection + temporal smoothing is the correct design.

### What NudeNet actually is (and isn't)
NudeNet is **not a dataset and not a novel architecture**. It is:
1. **Trained weights** — stock ultralytics **YOLOv8n (320n) / YOLOv8m (640m)** fine-tuned for nudity;
2. a small **Python library** wrapping inference (pre/post-proc, censor helper);
3. a **label taxonomy** (the 18 anatomy classes).

The training dataset is **not public** — issues #135, #161, #178, #200 on the repo are all
people asking for it. The value is baked into the weights, not the architecture.

**Implementation parity notes (discovered while porting, verified by A/B test):**
- The reference feeds **BGR** blobs to the network: cv2.imread gives BGR, its
  `cvtColor(RGBA2BGR)` on 3-channel input *swaps* to RGB, then
  `blobFromImage(swapRB=True)` swaps back to BGR. Our RGB decode pipeline
  reproduces the identical blob with a single `swapRB=True`.
- Preprocessing is **pad-to-square** (right/bottom black pad), resize to
  (res, res), /255 — not a plain stretch.
- NMS is **class-agnostic** (cv2.dnn.NMSBoxes, collect ≥ 0.2, NMS ≥ 0.25,
  IoU 0.45) — an overlapping FACE_FEMALE box suppresses a lower-scoring
  FEMALE_BREAST_EXPOSED box. `scanner/detector.py` replicates this exactly;
  parity test: `tests/test_detector_parity.py` (0 mismatches, scores equal
  to 3 decimals on real frames).

**RF-DETR reimplementation feasibility:** any single-stage detector can emit these 18 classes,
but there is no public per-body-part labeled dataset to train on. The only real path is
**distillation**: run NudeNet 640m over a large frame corpus, use detections as pseudo-labels,
human-verify a sample, train RF-DETR on that. Viable as a phase-2 (stack unification with
fitness-coach, licensing independence) but it inherits the teacher's blind spots, and note
our own trt.py lesson: RF-DETR needs FP32 TRT engines (FP16 drifts detection logits), while
YOLOv8 runs FP16 fine — so an RF-DETR student would also be slower in TRT than the YOLO teacher.

### Why a detector (not a classifier) as primary
The requirement is "explicit body parts shown", which is a *localized* property: a beach
scene with bikinis is fine, the same movie's bedroom scene is not. NudeNet-style labels
let us write a policy over `EXPOSED` labels while ignoring `COVERED` ones, plus gate by
box size (background poster / TV-in-scene false positives).

---

## 2. Architecture

```
                       ┌────────────────────── scanner container (new) ──────────────────────┐
 movie file ──────────►│ ffmpeg NVDEC decode ──► sample @4fps ──► NudeNet 640m (TRT FP16)     │
 (volume mount)        │   (H.264/HEVC, GPU)      scale 640      [320n cascade option]        │
                       │                                        │ per-frame detections         │
                       │                       label policy ──► temporal smoothing ──► merge  │
                       └──────────────────────────────────────────┬───────────────────────────┘
                                                                    │ candidate segments (JSON,
                                                                    │ + per-frame timeline, conf)
                                                                    ▼ (human review, then)
                       existing FastAPI trimmer  POST /submit(segments="HH:MM:SS-HH:MM:SS,...")
                                                                    ▼
                       GPU re-encode trim + subtitle resync (already works, frame-accurate)
```

### 2.1 Decode + sampling
- Primary: `ffmpeg -hwaccel cuda -i in.mkv -vf scale=640:360 -r 4 -f rawvideo -pix_fmt rgb24 pipe:`
  → fixed-size frames off a pipe → pinned torch buffer → H2D. NVDEC does the heavy lifting;
  pixels hit CPU only at 640×360@4fps (~2.8 MB/s). No OpenCV dependency needed for the hot path.
- Fallback: threaded `cv2.VideoCapture` (visiontouch pattern), pure CPU, ~2× realtime at 1080p.
- `sample_fps=4` default (configurable 1–10). Rationale: cut boundaries get ±padding anyway;
  4 fps catches ≥0.5s flashes with the smoothing rules below. Optionally add scene-cut
  snapping (ffmpeg `select='gt(scene,0.4)'` prescan) for cleaner boundaries.

### 2.2 Inference
- NudeNet **640m** ONNX → TRT FP16 engine (batch 16). Per-frame: top-k per class, torch NMS
  on GPU (8400 proposals is trivial) — or the YOLO26-NSFW model which is NMS-free.
- Cascade option for weak GPUs: 320n everywhere, 640m re-run only on ±3s windows around
  borderline frames. On a 4090 both models are decode-bound anyway; cascade is config, not default.
- Engine cache copied from the fitness-coach/visiontouch pattern:
  `trt_cache/{gpu_slug}/{trt_slug}/{model}.{weights_v}.{engine}` — port `trt.py` (TrtEngine,
  EngineCache, versioned_name) nearly verbatim; YOLO detectors get FP16 (their FP32-for-logits
  lesson applies to RF-DETR-style detection heads; if we ever see label flips, flip `fp32=True`).

### 2.3 Label policy (default; all configurable per request)

```yaml
triggers:            # any detection of these => frame is NSFW
  FEMALE_GENITALIA_EXPOSED: {score: 0.45}
  MALE_GENITALIA_EXPOSED:   {score: 0.45}
  ANUS_EXPOSED:             {score: 0.45}
  FEMALE_BREAST_EXPOSED:    {score: 0.45}
  BUTTOCKS_EXPOSED:         {score: 0.45}
review_only:         # flagged in output, not auto-cut
  BELLY_EXPOSED:            {score: 0.60}
  ARMPITS_EXPOSED:          {score: 0.60}
  FEET_EXPOSED:             {score: 0.60}
min_box_area_frac: 0.005    # ignore boxes < 0.5% of frame (posters, background TVs)
```

### 2.4 Temporal smoothing (sampled flags → cut segments)
1. Frame flagged iff any trigger class ≥ score AND box area ≥ min fraction.
2. **Hysteresis**: need ≥2 consecutive flagged samples to open a segment (kills single-frame noise);
   close after ≥2 consecutive clean samples.
3. **Bridge** gaps ≤ 2 s (same scene cutting back and forth).
4. **Drop** segments < 1.0 s unless mean score > 0.7 (real flash vs artifact).
5. **Pad** ±1.5 s, cap at neighboring segment start; snap to nearest scene cut within 1 s if prescanned.
6. Merge overlaps, sort, emit `[[start_s, end_s], ...]` (seconds — `parse_segments_input` accepts JSON).

### 2.5 API contract (scanner service)

```
POST /scan        multipart video | {video_path}, policy overrides
  → 202 {scan_id}
GET  /scan/{id}   → {status, segments, review_segments, timeline (binned per-second flags),
                     label_stats, duration, wall_time}
```

Review step first (web or CLI listing segments w/ thumbnails); auto-submit to the trimmer is a
one-liner once the user trusts the policy. Keep human-in-the-loop initially.

---

## 3. Container (mirrors visiontouch/fitness-coach stack)

```dockerfile
FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime   # torch 2.14 / CUDA 13.0, as visiontouch
ENV PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HUB_OFFLINE=1   # all weights baked in; nothing downloads at runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tensorrt-cu13==11.2.1.2 opencv-python-headless \
    fastapi uvicorn numpy
# weights baked at build (versioned like their weight images)
ADD https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx \
    /opt/models/nudenet_640m_v1.onnx
ADD https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx \
    /opt/models/nudenet_320n_v1.onnx
# app code: scanner.py, trt.py (ported), policy.py, temporal.py
COPY app/ /app/
VOLUME /cache/trt          # engines keyed gpu_slug/trt_slug/weights_v
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Run: `docker run --gpus all -v /mnt/movies:/movies -v trt_cache:/cache/trt -p 8080:8080 nsfw-scanner:v1`
First run pays a one-time ~1–3 min engine build per model, then cached.

### Performance estimate (2h 1080p H.264 movie, 4090, estimates not benchmarks)
| Stage | Cost |
|---|---|
| NVDEC decode 1080p H.264 (~900 fps) | ~3–4 min |
| NudeNet 640m TRT FP16 b16 @4fps (28.8k frames, ~400–600 fps) | ~1 min (overlapped with decode) |
| **Total wall time** | **≈ 4–6 min**, single pass, no CPU bottleneck |

---

## 4. Licensing notes
- NudeNet code: AGPL-3.0; weights: no explicit license on the models repo (personal use fine;
  needs review before any product-ization). `deepghs/nudenet_onnx` conversion repo says Apache-2.0.
- YOLO26-NSFW: AGPL-3.0 (ultralytics lineage).
- Falconsai / viddexa classifiers: Apache-2.0 — clean fallbacks if licensing forces a swap:
  binary gate + a person detector won't match NudeNet's label granularity, though.

## 5. Suggested build order
1. Scanner skeleton + NudeNet 320n via the pip package (CPU/ORT) — end-to-end correctness,
   golden-test on a short clip with known segments.
2. Port `trt.py`, switch to 640m ONNX→TRT FP16, ffmpeg pipe decode; benchmark real movies.
3. Temporal smoothing + policy config + `/scan` API + review output (thumbnails).
4. Wire into trimmer (`/scan` → review → `/submit`), then quality-tune thresholds on a test set.
5. Optional: YOLO26-NSFW A/B, VLM arbitration for ambiguous windows.

## 6. Calibration findings (Ex Machina 1080p, 2026-10-31)

Measured: 108 min scanned in 131.7s (**49x realtime**), 435 fps inference,
NVDEC-bound. Default policy (triggers 0.45, area gate 0.5%) found the two
main explicit clusters but missed ~10 real regions. Root causes, verified
experimentally:

1. **Sub-threshold continuations** — scenes right after confirmed segments
   scoring 0.30-0.44 (dim/occluded framing). Fix: triggers lowered to 0.30.
2. **Area-gated distant shots** — scores >0.45 but boxes <0.5% of frame
   (wide shots). Fix: `min_box_area_frac` lowered to 0.001.
3. **True blind spots** — near-zero signal (<0.08) in some unusual-framing
   shots. Ruled out experimentally: gamma/CLAHE brightening (confirmed hits
   are themselves luma 26-47 — the model handles this film's darkness) and
   SAHI-style 2x2 tiling (quarters centered figures, destroys whole-body
   context NudeNet needs; scores dropped 0.33 -> 0.00). These need the VLM
   arbiter (phase 5).

**Stage plan:** A) *implemented* — hand-rolled numpy ByteTrack-style
tracking on NudeNet person clusters (`scanner/tracking.py`; `boxmot`
rejected: imports torch into the torch-free host venv). Two-band IoU
association + EMA velocity coasting; explicit-hold bridging (8s hold,
sub-trigger refresh floor 0.15, `*_COVERED` counterpart clears). Validated
on Ex Machina: 21 -> 41 segments — +1-2s tail extensions everywhere, rescue
of isolated sub-threshold hits (previously killed by the 2-consecutive
rule), and long fills inside continuous scenes (47:57-48:32, 72:20-72:28).
B) *implemented* — sparse Qwen3-VL-8B arbitration (`scanner/arbiter.py`):
3 verify frames/segment + ~1fps windows hugging segments (10s before /
45s after, clipped between neighbours) + 60s global sweep, capped at 250
frames by striding, never every frame. Strict-JSON moderation prompt,
refusal-tolerant parsing; `arbiter_mode=verify` (annotate) or `apply`
(add blind-spot segments, drop rejected). Model baked in — nothing
downloads at runtime (HF_HUB_OFFLINE=1; weights re-shipped at
ghcr.io/darkstar1997/movietrimmer-qwen3vl-weights:v1).
C) RF-DETR Seg Medium person masks (re-shipped at
ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1, FP32 TRT engine
per the fitness-coach logit-drift lesson) for mask-precision blurring:
blur region = person mask AND padded explicit box. SAM 3.1 dropped in
favour of the in-house RF-DETR seg model.

**Tuned policy (v2):** triggers {5 explicit labels: 0.30},
`min_box_area_frac` 0.001, `bridge_gap_s` 2.5. Hysteresis (2-sample open) +
min-duration + drop-rescue keep the lower threshold's FP cost low; the first
0.39-mean candidate region should be human-verified before trusting auto-cut.

## 7. Arbiter calibration (Qwen3-VL-8B on Ex Machina, empirical)

Two integration bugs found and fixed before any prompt conclusions:
- **Images never reached the model**: transformers 5.16.1's single-shot
  `apply_chat_template(tokenize=True)` does not expand image placeholders
  for Qwen3-VL. Fix: the visiontouch two-step pattern (render to string
  with `tokenize=False, add_generation_prompt=True`, then
  `processor(text=..., images=...)`). Symptom: confident `explicit:false`
  on every frame plus 159/250 unparseable outputs.
- **Sparse-sample chaining**: reusing `smooth_segments` for arbiter hits
  counted *array-adjacent* samples as consecutive, chaining hits minutes
  apart into phantom 600s segments. Fix: time-proximity clustering
  (`_cluster_hits`, chain gap = 1.5x median frame spacing, clamped
  [2.5, 6.0]s).
- The 8B model holds ~17GB VRAM: it must be **unloaded after each run**
  (`_unload_vlm`), not cached, or the next GPU workload OOMs.

Prompt findings (7 variants probed on explicit / borderline / benign
controls, deterministic decoding):

| variant | result |
|---|---|
| wordy "moderation" user prompt | sandbags: confident false on real nudity, 64% refusals |
| machine-to-machine emphasis | no refusals, still misses borderline |
| clinical framing | catches high-confidence only |
| free-form system classify | most sensitive (saw 1 borderline) but unparseable |
| **terse system classifier + strict JSON** | **adopted**: 0 refusals, 0 FPs on controls, confirms all high-confidence segments |

Measured behavior profile: reliable on clearly-visible nudity and clean
non-nudity; **false negatives on dim/borderline scenes** (NudeNet 0.2-0.36
regions usually judged false). Consequence: `arbiter_mode=apply` is
**add-only** — VLM-found segments merge in, detector segments are never
removed on non-confirmation; `rejected` verdicts stay informational.

Frame budget: 41 segments x 3 verify + thinned 10s-before/45s-after windows
consumed all 250 slots, leaving none for the global sweep — known isolated
spots (e.g. 01:40:23) can go unsampled; raise `ARBITER_MAX_FRAMES` or trim
`ARBITER_WINDOW_AFTER_S` if sweep coverage matters.
