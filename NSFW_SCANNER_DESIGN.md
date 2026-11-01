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

### Frame budget & sweep starvation (recall fix, 2026-11-01)

The original budget order (verify → windows → sweep) starved the global
sweep to near-zero whenever a movie had many segments: on Ex Machina the
47 segments' verify+window candidates consumed the whole 250-frame cap,
leaving the sweep — the *only* sampling of regions far from any detector
segment — with single-digit frames (60s+ sampling gaps, 240s worst-case
blind gap). That was the structural cause of "nudity not detected" in
scene-change/edit-heavy films. Three changes (`scanner/config.py`, all
env-overridable):

1. **Allocation order is verify → sweep → windows.** The sweep gets a
   guaranteed `ARBITER_SWEEP_BUDGET_FRAC` (0.5) share of the cap before
   windows refine already-flagged neighborhoods; windows take the rest.
2. **Denser defaults**: `ARBITER_SWEEP_EVERY_S` 60 → **15**, 
   `ARBITER_MAX_FRAMES` 250 → **700**. On the 6481s reference film:
   141 verify + 350 sweep + 209 window frames, 15s median blind-spot
   spacing, 61.8s worst gap (was 240s).
3. **`ARBITER_SWEEP_PAD_S` (8.0)**: a sweep hit is a point sample of a
   scene of unknown extent — it pads ±8s (window hits keep the 1.5s
   smoothing pad since their neighborhood is on a 1s grid), and adjacent
   sweep hits chain into one segment.

Measured A/B (same tuned policy, only the arbiter changed): flagged
coverage 349.0s → 393.8s (+12.8%), VLM segments 6 → 9 — three scenes the
detector missed entirely (t≈3406, 4276, 5752), the hotel-scene fragments
chained into continuous coverage (4222–4239, 4333–4348), and the finale
extended to 5784–5846 unbroken. The 26s of "lost" coverage was 0.3–0.5s
pad slivers plus 4 borderline detector segments (score 0.26–0.46) that
flicker run-to-run regardless of the arbiter. Cost: arbiter wall
343.5s for 700 frames (0 refusals) — scan total 8m03s (13.4× realtime),
detector pass unchanged (~2m20s).

Residual: sub-15s scenes between grid points can still slip through; the
next lever is a denser sweep (cap 1200 / every 8s ≈ +2min arbiter).

### CringeMDB ground truth: the 52–58 min blind spot (2026-11-01)

CringeMDB publishes timestamped content scenes for the 1h48m release that
matches our reference file; 3 of its 4 nudity scenes were spot-verified by
skin-fraction at the stated frames. Saved as
`scans/gt_cringemdb_exmachina.json` + `scripts/eval_gt.py` (regression
metric). The tuned pipeline (640m, sweep15, arbiter apply) covered
**5.6%** of GT nudity time (13/237s) — all four scenes sit in the dim /
distant / brief-exposure weak spot.

Isolation probes (raw detector calls on ffmpeg-grabbed frames — no
pipeline interference; control frame t=5820 fires 0.38–0.71 in every run):

- **NudeNet is blind on the GT frames at any size**: 320n and 640m, TRT
  fp16 *and* pure-ONNX ORT, full-res 1920×808 *and* 640×270 pre-downsize,
  raw/CLAHE/gamma — zero trigger boxes on all 12 GT frames. A dense 0.5s
  sweep of the whole 3131–3499 region produced 4 firing moments in 6 min:
  3150.5 FEMALE_BREAST 0.51–0.54, 3327/3342.5 FEMALE_GENITALIA 0.25–0.26,
  3388 BUTTOCKS 0.36–0.44. Luma explains only the bed scene (mean 9);
  the motionless (66) and sex-scene (81) regions are *brighter* than
  regions we do detect (finale 45). 640m was already the shipped default
  — swapping models is not the lever here.
- **Qwen3-VL is the only model that sees any of it** — but under-calls
  with the terse prompt: `explicit:false conf 0.0` on 13/14 GT frames,
  *including the two frames NudeNet fires on* (3150.5, 3388). No
  refusals; clean JSON every time — a calibration problem, not format.
- **Prompt fix (probe: 5 variants × 16 frames × 9 clean controls)**:
  appending a dim/distant-aware paragraph to the terse system prompt
  ("look carefully at dim regions and distant/partially occluded
  figures; however small, dim, or brief — set explicit=true") quadrupled
  GT-frame recall 1/16 → 4/16 at **zero control FPs** (dress-open scene,
  sub-second flash, dim buttocks). "Search the whole frame" phrasing and
  a label-free binary variant gained nothing; a 2× center-crop zoom
  gained nothing (the distant couple stays sub-resolution). Adopted as
  the new `ARBITER_SYSTEM_PROMPT`.
- **Scene-level ceiling (dense S1-prompt VLM pass)**: motionless scene
  3197–3247 — zero flips at 5s spacing, no skin signal, no detector
  signal: the CringeMDB range over-spans whatever is briefly visible;
  unreachable and honestly missable. Distant sex scene 3301–3472 —
  exactly one flippable moment (t=3431, MALE_GENITALIA, conf 0.95) that
  NudeNet never fired on at any resolution; it sits inside our
  review-only segment 3427–3435 (ARMPITS/BELLY). Bed flash bounded to
  3150.50–3150.75 (~0.3–0.5s).

Fix package (this repo, all defaults env-overridable):

1. **S1 dim-aware arbiter prompt** (see `scanner/arbiter.py`).
2. **Weak-hit routing**: sub-threshold trigger scores
   (`ARBITER_WEAK_MIN_SCORE` 0.25 → label threshold 0.45) are collected
   per frame (`Policy.weak_triggers`), clustered to ≤2s-burst peaks
   (`_cluster_weak`, cap `ARBITER_WEAK_MAX_FRAMES` 120), and planned as
   top-priority "weak" arbiter frames; a confirmed hit pads like a sweep
   hit (±8s, add-only). These are the blind region's only detector
   voice.
3. **Flash rescue**: `drop_rescue_score` 0.70 → **0.50** — the old line
   discarded every real sub-second flash (2-sample run at mean ~0.52
   opened fine, then died at the drop stage).
4. **Denser sweep**: `ARBITER_SWEEP_EVERY_S` 15 → **10**,
   `ARBITER_MAX_FRAMES` 700 → **1100**, `ARBITER_SWEEP_BUDGET_FRAC`
   0.5 → **0.6** (full 10s grid coverage of blind spots; brief scenes
   like the dress-open still need luck — the sweep is a point sample).
5. **Orphan-flag routing**: flagged samples (≥ trigger threshold) whose
   run smoothing discarded — single-frame hits killed by
   `open_min_consecutive=2`, sub-1s blips below the rescue line — join
   the weak candidates at score 1.0 (`_orphan_flagged`). A detector-
   strong single frame is the best blind-spot lead there is; it must
   never vanish silently. Orphan samples also get **±0.5s companion
   frames** judged: a sub-second flash is wider than one sampling
   phase, and the orphan sample can land at the edge of the
   visually-explicit window (measured: the flash's orphan landed 0.17s
   before the VLM's window; companions at ±0.5s cover that class).
   Weak-band peaks (0.25–0.45) get no companions — measured VLM-
   rejected 15/15, not worth the budget.
6. **Point-hit expansion second pass**: explicit sweep/weak/orphan hits
   are point samples; `plan_expansion` re-samples their
   ±`ARBITER_SWEEP_PAD_S` neighborhood at 1s with the same loaded model
   (cap `ARBITER_EXPAND_MAX_FRAMES` 96, best hits first). Neighbors
   that also flip chain with the hit — a measured segment instead of a
   symmetric ±8s guess. Add-only by construction: non-explicit
   neighbors never shrink the hit's own pad.

v1 A/B (prompt + weak routing + rescue 0.50 + sweep 10s/900, before
expansion/orphans), same movie and request params as the sweep15
baseline: GT recall **5.6% → 9.7%** exact (23/237s) and **6.5% →
19.8%** at ±60s tolerance; dress-open scene 0% → 57%, bed scene covered
under tolerance, sex scene 7.8% → 8.8%. Regression audit: +95s gained
vs 45s lost (all losses 0.2–4s boundary slivers of still-covered
scenes), zero clean-control regions newly flagged, 0 VLM refusals.
Mechanism attribution from frame verdicts: dress scene = sweep hit at
3490.0, sex-scene moment = window hit at 3433.3, weak tier judged 15
frames with 0 flips and 0 FPs (VLM correctly rejects blurry sub-
threshold distant hits — precision holds). Wall 544s (11.9× realtime)
with 900 VLM frames judged in 157s.

Final A/B (v3: + orphan routing, full 10s grid at 1100 frames; v4/v5:
+ orphan companions): **GT recall 5.6% → 12.0%** exact (28/237s; 15.0%
observed under one batch binning — borderline VLM verdicts shift a
sweep hit or two when the batch composition changes, detector segments
stay byte-identical), **6.5% → 28.9%** at ±60s tolerance; dress-open
0% → 68%, the sub-second flash captured as a real segment (the 52:16
CringeMDB entry is plausibly this flash within their ±1min accuracy),
sex scene 7.8% → ~12-15%. v4 and v5 (same code) reproduce exactly.
Regression audit across all five runs: no clean-control region ever
newly flagged, losses are 0.2-8s boundary slivers of still-covered
scenes, 0 VLM refusals, 47 segments / ~505s flagged (baseline 50 /
394s).

### The remaining gap is model-bound (evidence)

A 1s-density VLM pass (99 frames) over the two big uncovered GT ranges
found **zero** explicit moments: motionless 3197–3247 (51 frames) and
sex-scene tail 3435–3472 (38 frames) are invisible to Qwen3-VL at
maximum practical density — matching the NudeNet dense 0.5s sweep
(zero boxes) and the skin-fraction calibration (no separation; the
teal/orange grade defeats YCrCb rules). CringeMDB ranges span whole
*scenes*; the actually-visible nudity inside them is concentrated in
the slivers every model does see (now covered). Raising coverage
further needs a differently-biased model (e.g. ShieldGemma-2 second
vote), not more mechanisms on this stack.


## 8. Stage C: mask-precision blur render (empirical, 2026-10-31)

Architecture: two passes. **Analysis** decodes only flagged ranges at
8 fps (BLUR_KEYFRAME_FPS) at scan resolution, runs NudeNet per batch, and
composes the per-keyframe mask as

    detector segments:  union over explicit boxes of
                          full person mask matched to that box
                        (padded box-rect fallback when no person matches)
    VLM (arbiter) segments:
                          union mask of every person RF-DETR finds
                        (centered ~0.7-frame rect fallback when it finds
                         nobody)

with two added behaviors:

- **Full-person blur.** The matched person's whole mask is used, not the
  part-box intersection — the requirement is to blur the person, not the
  part. Association accepts IoU > 0.3 OR >=60% containment of the part box
  inside the person box (part boxes have inherently low IoU vs person
  boxes); when nothing matches, the padded box rect is used — an unmatched
  person mask risks blurring the wrong person.
- **Person tracking across detector flicker.** The last matched person box
  is remembered per segment; on keyframes where NudeNet finds nothing, the
  box is re-matched and the full-person mask persists until the person
  leaves the frame (mask + overlay survive sub-threshold flicker).

Masks are dilated (~1.2% of frame width) and stored packbits-compressed
per keyframe, each carrying (label, score) for the overlay. **Render**
(full movies: GPU-native path below) decodes the whole movie, and for
frames inside a segment holds the most recent keyframe mask (upscaled
nearest), pixelates through it (24 px blocks at 1080p), composites the
warning overlay, and encodes with NVENC.

### Warning overlay (2026-11-01)

Every flagged frame carries red top-left text:

    VULGARITY DETECTED
    87%  FEMALE_BREAST_EXPOSED

Rationale: the mosaic destroys *texture*, not *silhouettes* — on
ultra-smooth/dim scenes the blur is nearly invisible, so the viewer needs
an explicit, unmissable flag on exactly the frames the pipeline acted on.
Confidence is the triggering detection's score (or the arbiter's VLM
confidence for arbiter-found segments, label `VLM`). Composited after the
pixelate so the text stays crisp: pipe path draws via cv2 on RGB; the
GPU-native path builds the text canvas once per keyframe (cv2), converts
to NV12 masks, and does two `masked_fill_` passes with BT.709
limited-range constants (red Y63 Cb102 Cr240, black outline Y16 Cb128
Cr128). Kill-switch: `BLUR_OVERLAY=0` env.

### Validation findings (Ex Machina 1080p)

- Mask coverage of the explicit box: **96-100% on every keyframe** of the
  5628.8-5636.2 test segment (person-mask intersection is doing real work
  vs a bare box: blur region tracks the torso, not a full-frame box).
- Only keyframes with passing detections carry masks (22-27 of 68 per
  segment) — the hold logic bridges detector flicker *within* the segment,
  and pre-first-keyframe frames reuse the first mask (blur early, not
  late).
- Policy consistency: analysis must run the SAME policy the scan judged
  by. Default policy (0.45 triggers) vs the scan's tuned (0.30/0.001) lost
  the first 1.25s of the test segment. `/blur` now reuses the scan's
  policy automatically when `scan_id` is given and no override is passed.

### NVENC findings (RTX 4090, driver 615.78.08, ffmpeg n9.0.2)

- **`av1_nvenc` fails with `-22 EINVAL` on the rawvideo-pipe path** —
  bare encoder, every preset/rc combination. The trimmer's ANALYSIS.md
  av1 success went through `-hwaccel_output_format cuda` (frames already
  on the GPU); with piped CPU frames the encoder rejects the session.
  av1 is therefore excluded from `BLUR_CODECS`.
- `hevc_nvenc` / `h264_nvenc` with `-preset p5 -tune hq -rc vbr -cq N
  -b:v 0 -spatial-aq 1 -rc-lookahead 32` work; the chain falls back
  `hevc_nvenc -> h264_nvenc -> libx264` (a codec that fails fast at init
  is skipped after a 1.5s probe).
- Explicit rate control is mandatory: NVENC defaults to ~2 Mbps VBR,
  visibly inadequate for 1080p film (trimmer ANALYSIS.md finding; CQ 26
  default here).

### Numbers

- Window render (25s window, 1 segment): analysis + render = 15-32s wall.
- Full-movie GPU-native render (Ex Machina 1080p, 6481s, 155,385 frames,
  41 segments / 920 mask keyframes): **285s wall = 22.7x realtime**, 7,452
  frames blurred (4.8%), NVDEC→torch→NVENC never touching CPU memory.

### Full-movie verification (Ex Machina, GPU-native render, 2026-11-01)

Every check passed end-to-end:

- container: hevc+aac, duration 6480.96s == source exactly.
- decode clean (`-err_detect explode` over the full file).
- **155,385 video packets == source packet count exactly** — one
  continuous pass, no cut boundaries, so nothing can drift.
- **audio bit-exact**: packet pts/dts/size md5 identical to the source.
- frame identity: sampled output frames each match a source frame to zero
  hot pixels at display time. (The output's raw PTS grid sits 2-3 frames
  earlier than the source's edit-list offset — the muxer compensates the
  source's 0.083s video start; display timelines align.)
- blur present inside segments: local texture (3x3 std) crushed to
  ~0.2-0.4x of source inside mask regions; clean ranges pixel-exact up to
  codec noise.
- One content caveat, not a defect: on ultra-smooth soft-focus scenes
  (seg 0, source local-std p95 ≈ 4) the 24px mosaic changes values by <30
  for 98.7% of mask pixels — mathematically applied (verified against an
  in-process emulation: 671 vs 672 hot px) but barely visible, because
  pixelate destroys *texture*, not *silhouettes*. A bigger BLUR_PIXELS
  does not change this; a different blur style (gaussian/solid) would.

### Torch↔NVENC stream ordering (found via the overlay, 2026-11-01)

The in-place torch ops on the from_dlpack NV12 surface are async on
torch's stream; NVENC reads the surface on its own stream without
waiting for that queue. Without an explicit `torch.cuda.synchronize()`
after the pixelate+overlay, the race drops the queue TAIL — measured on
a 240-flagged-frame repro: the overlay (submitted last) was missing on
135/240 frames while the pixelate (earlier kernels) always landed.
Cost of the sync: none measurable (3.13s repro before and after).

### Splice rendering — attempted, then abandoned (documented failure)

To avoid re-encoding 108 minutes for 350s of flagged footage, a
stream-copy-splice renderer (copy unflagged ranges, re-encode only
[segment, next keyframe) windows, MPEG-TS concat) was built and validated
in prototype — then failed twice on real-movie boundaries despite verified
packet counts: ffmpeg's copy-cut semantics (dts vs pts, per-edge rounding,
mp4 timescale quantization) made every part boundary a fresh off-by-one,
and each fix relocated the failure. The lesson: **never trust ffmpeg's
cutter at frame granularity — prefer designs with no cuts at all.**

### GPU-native rendering (production path)

The replacement — one continuous pass, so there are no cut boundaries to
get wrong:

    CreateDemuxer -> CreateDecoder (NVDEC, device memory)
                  -> torch.from_dlpack(frame)   # NV12 view, zero-copy
                  -> in-place block pixelate of Y+UV through the held
                     person mask (avg_pool + nearest-up + where, on GPU)
                  -> CreateEncoder (NVENC hevc, P5, high_quality tuning,
                     VBR + explicit CQ)
                  -> FFmpegMuxer (audio/subs passthrough, bit-exact PTS)

- PyNvVideoCodec 2.2.3 (no pycuda needed: `cudacontext=0` shares torch's
  primary context; pycuda won't even build in the runtime image). DLPack
  hands NVDEC's NV12 surface to torch as a (h*3/2, w) uint8 view and
  in-place writes flow into NVENC — frames never touch CPU memory.
- PTS set verified identical to the source on a PoC; the muxer maps output
  PTS from the source packets (RecordSourceVideoPacket), so the timeline
  and A/V sync are structurally exact.
- NV12 masks: luma mask at full res; chroma mask = 2x2-union of luma
  repeated to the interleaved UV plane (a masked macropixel's U and V
  share the decision).
- NVDEC-wedge insurance: a per-packet progress watchdog aborts a hung
  decode instead of blocking the job forever.

### TRT-only backends (policy)

A silent ORT fallback let the first full render crawl at 28 CPU cores for 8
minutes (personseg auto-picked CPU ORT in-container). Backends are now TRT
by default and fail loudly when TRT cannot init ("ort" remains an explicit
host-test-only opt-in). The RF-DETR Seg Medium FP32 engine builds in ~12s
and analysis of a 7s segment drops ~27s → ~1.5s.

## 9. Stage D: crowd timestamps — CringeMDB scene guide (2026-11-01)

`POST /scan` with `crowd=true` adds a third segment source: CringeMDB's
crowd-sourced Scene-by-Scene guide, fetched at scan time. Community
ranges become **first-class segments** — identical Stage C treatment to
VLM-found blind spots: mask every person RF-DETR finds (tracking across
keyframes), center-box fallback, and the red warning overlay reads
`VULGARITY DETECTED / 90% CRINGEMDB:NUDITY` (category from the entry).

Design decisions, each grounded in how the site actually behaves:

- **Slug resolution, not search.** `cringemdb.com/search?q=` always
  returns `[]` (useless), but slugs are deterministic:
  `/movie/{title-dash-year}`. The movie title/year are parsed from the
  filename (`Ex.Machina.2015.1080p.BluRay.x264.YIFY.mp4` → "Ex Machina",
  2015), with `crowd_title`/`crowd_year` Form overrides for odd names.
  A wrong slug 404s cleanly into the fail-soft block.
- **Parsing.** Scenes are `<article class="sg-scene">` blocks; the
  category comes from the `sg-dot {nudity|sex|violence}` CSS class (a
  stable token), not display text. Times are `<strong>` + optional
  "to" span; point scenes get `end = start`.
- **Scope.** Only `CROWD_CATEGORIES` (default `nudity,sex`) become
  segments — violence is out of this pipeline's scope and lands in the
  result's `skipped` list for auditability.
- **Add-only.** Crowd merges AFTER the arbiter and can never remove
  detector/VLM-found segments — community data extends coverage, it
  does not veto models.
- **Fail-soft.** Fetch/parse/HTTP errors land in `result['crowd']
  ['error']`; the scan proceeds. CringeMDB is an enhancement, not a
  model — it must never fail a scan. HTML is cached under
  `scan_workspaces/crowd_cache/{slug}.html` (config: `CROWD_*`).
- **Honesty about accuracy.** Entries are unverified and "can be off by
  a minute or so" (their words) — hence score 0.90 (below model
  confidence), ±`CROWD_PAD_S` padding, and per-entry `desc` in the
  result block for human audit.

End-to-end validation on Ex Machina (live container): filename alone
resolved `ex-machina-2015`, 4 in-scope entries (3 nudity + 1 sex) merged
padded at exactly the scene ranges (3134.5–3139.5, 3195.5–3248.5,
3299.5–3473.5, 3474.5–3491.5), the 1:29:52 violence entry skipped. A
blur render of the dress-scene crowd segment showed the overlay
(`VULGARITY DETECTED / 90% CRINGEMDB:NUDITY`, OCR-verified) and a
tracked person mask whose pixels sit exactly on the 24px block grid
(quantization error 0.4–1.6 vs 35–55 for source), absent before segment
start. Host tests: 9 crowd unit tests on a saved real-page fixture plus
an end-to-end `run_scan(crowd=true)` smoke test (98/98, network-free).

## 10. Gemma-4-12B arbiter candidate — rejected (llama.cpp, 2026-11-01)

Infrastructure (kept: `vlm.Dockerfile`, 3 stages): llama.cpp **v0.8.0**
built from source (CUDA 13, flash attention all-quants, arch 89), server
running `unsloth/gemma-4-12B-it-qat-GGUF` (UD-Q4_K_XL) + `mmproj-BF16`
+ the **MTP drafter** (`--spec-type draft-mtp --spec-draft-n-max 4`,
`-fa on`, `-ub 2048` to uncap the 1120-token image budget; thinking off
via `chat_template_kwargs.enable_thinking=false` → 0.27 s/frame
single-stream, 18 output tokens/frame). Build notes: CUDA 13 needs
`-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs` for the driver-API
link; v0.8.0's server refactor ships `libllama-server-impl.so` so the
whole `build/bin` dir must be copied together.

Detection verdict on the reference movie — **rejected**:

- Contested GT ranges (motionless / sex-tail / dress-head, 89 frames at
  1s density — the exact set Qwen3-VL-8B zeroes out): **1 flip**, and
  it mislabels it (MALE_GENITALIA on the dress-open scene).
- Known positives the production stack already catches: **1/4** (missed
  the flash, the sex-scene moment, dress-3490; all conf=1.0 clean).
- Controls: 0/9 — precision holds, but so does Qwen's.
- Diagnosis: not safety-blind in general (flags the close-up Ava nudity
  at conf 0.98) and not resolution-starved (identical verdicts at
  `--image-max-tokens 1120`; still clean on a 3× zoomed crop of the
  masked person region). It is **dominant-figure biased**: asked to
  describe the exact frames, it narrates the foreground man in the
  doorway and never registers the small distant nude figure — the
  dim/distant regime §7 built the whole recall fix around.
- Throughput: 0.27 s/frame single-stream vs Qwen's ~0.14 s/frame batched
  — no speed win either, at 12 GB VRAM held.

Conclusion: a bigger VLM is not a better arbiter for this task;
Qwen3-VL-8B stays. The llama.cpp serving stack is kept for the next
candidate (e.g. a Qwen3-VL GGUF under the same harness would isolate
serving-runtime from model bias).

### Qwen3-VL-8B GGUF under the same harness (same day)

`unsloth/Qwen3-VL-8B-Instruct-GGUF` Q8_0 + mmproj-BF16, same llama.cpp
image, no drafter, weights local under `models/qwen3-vl-8b/` (the image
COPYs them; global ARGs must be re-declared per Docker stage — an empty
`${MODEL_DIR}` once COPY'd the whole repo into a 67 GB image).

- Positives 4/4, controls 0/9 — parity with the production
  transformers path on the decisive sets.
- Contested ranges: 4 flips (dress-head 3482-3483 same as transformers;
  sex-tail 3445-3446 new) — subject to this model's known borderline
  flicker; the two new flips sit inside GT ranges, no FP risk.
- Labels degrade: nearly every hit reads MALE_GENITALIA where
  transformers says FEMALE_BREAST on the same frames/prompt — Q8
  quantization or template interplay; overlay text would suffer if
  adopted as-is.
- Throughput: 0.366 s/frame single-stream, **0.101 s/frame at 4
  concurrent** (saturates its 4 slots) vs 0.14 s/frame for the
  production transformers batching — ~28% faster at 12.5 GB vs 17 GB.

Verdict: llama.cpp serving wins on speed and memory at recall parity;
transformers keeps label fidelity. Adopting it means an OpenAI-client
arbiter path (currently in-process transformers) — worth doing only if
the label degradation is understood first.

### Qwen3.8-27B UD-Q4_K_M + MTP drafter (same harness, same day)

Bigger-model hypothesis: recover more contested GT and fix the 8B GGUF's
label degradation. Weights under `models/qwen3.8-27b/` (UD-Q4_K_M 16.5GB
+ mmproj-BF16 + Q4_0 MTP drafter, the largest quant that fits 24GB with
KV headroom). It is a thinking model: `chat_template_kwargs:
{"enable_thinking": false}` (or `reasoning_effort: "none"`) is required —
the default burns the whole budget on reasoning and returns empty
verdicts.

- Positives 2/4 — misses both dress-head positives (3483.0/3490.0) as
  high-confidence clean, stably across repeats. Worse, one of its two
  hits (3433.3) flips back to clean on re-run: borderline verdicts are
  non-deterministic under MTP/batching numerics.
- Controls 0/9 (only metric at parity).
- Contested ranges: 0 flips — recovers none of the ceiling.
- Labels correct (FEMALE_BREAST) — evidence the 8B GGUF's
  MALE_GENITALIA degradation is quant/model-specific, not a
  llama.cpp-serving artifact.
- Throughput 0.387 s/frame at 4 workers (MTP drafter on) vs 0.101 for
  the 8B GGUF; VRAM 22.6/24 GB.

Verdict: REJECTED. A larger model of a different family is less
sensitive on exactly the borderline-nudity frames this arbiter exists
to catch, and slower by ~4x. Qwen3-VL-8B (transformers, in-process)
remains the arbiter; the model-bound ceiling stands. Bigger is not
better here — twice tested, twice true (Gemma-4-12B, Qwen3.8-27B).

### Qwen3-VL-30B-A3B UD-Q4_K_XL (same harness, same day) — best yet

Same family as the production 8B but MoE: 30B total, ~3B active per
token. Weights under `models/qwen3-vl-30b-a3b/` (17.7GB + mmproj).
`qwen3vlmoe` arch is supported by the pinned llama.cpp v0.8.0. No MTP
drafter exists for it — MoE decode is already cheap.

- Positives 3/4: 3490.0 flickers (1 explicit of 4 judgments across
  probe re-runs + recheck) — borderline, not blind: contrast the 27B's
  stable high-confidence misses.
- Controls 0/9.
- Contested ranges: **10 flips** (sex-tail 3445-3460, dress-head
  3474-3483), identical across two full probe runs — 5x the
  transformers record (2) and 2.5x the 8B GGUF (4). First candidate to
  lift the model-bound ceiling, and by a lot.
- Labels: correct-dominant with richer multi-label output
  (FEMALE_BREAST where transformers says so; sex-tail frames add
  MALE_GENITALIA/BUTTOCKS, plausible for that scene).
- Throughput: **0.054 s/frame at 4 concurrent** — ~2x the 8B GGUF
  (0.101), 2.6x production transformers (0.14). VRAM 20.5/24 GB.

Verdict: strongest candidate measured. Adopting it means moving the
arbiter from in-process transformers to an OpenAI-client path against
a llama-server sidecar (scanner image stops bundling the VLM). The
parked Qwen3-VL-8B BF16 GGUF A/B remains the label-degradation
root-cause test for the 8B line.

### Full-movie validation of the 30B-A3B arbiter (same day)

ARBITER_BASE_URL HTTP backend added (scanner/arbiter.py): OpenAI
/v1/chat/completions client, env-switched, ARBITER_HTTP_CONCURRENCY
parallel requests; the in-process transformers VLM is never loaded
when set (pure-function http_request_body/http_response_text tested
network-free on the host). Combined scan rerun end-to-end (detector +
30B arbiter apply + crowd) → scans/exmachina_combined_crowd_arbiter_
30b.json; blur render blur_outputs/blurred_21453785.mp4.

- 63 segments vs 46; 28 carry the arbiter label vs 17.
- In-pipeline ceiling lift confirmed: frames the 8B called clean at
  conf 0.98 (t=3460/3500/3510) flip explicit under the 30B; its
  dress-head/sex-tail hits land inside the crowd range and extend its
  tail 3491.5 -> 3518 (labels cringemdb + arbiter:4 merged).
- Clean controls 0/9 in any non-crowd segment; all four probe
  positives covered; verify_blur ALL PASS (bit-exact audio, packet
  parity, blur present on new VLM segments 442-468/3299-3518/3552-
  3568, 9/9 clean controls exact-match source).
- Cost neutral: scan wall 750.9s vs 754.3s — the HTTP inference gain
  (~65s vs ~157s judging) is masked by the per-frame-seek extraction
  fallback; fixing extraction is the next optimization, not the model.

### Adoption (same day)

The 30B-A3B sidecar is the production arbiter from here on: scanner
image no longer bundles the 8B (17GB lighter), docker-compose.yml
defines the pair (scanner + llama-server sidecar on one GPU,
ARBITER_BASE_URL wiring, sidecar health-gated startup). The in-process
transformers path remains as a generic env-switch fallback for hosts
that mount weights into VLM_DIR.







