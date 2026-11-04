# SAM 3.1 vs RF-DETR PersonSegmenter — mask-stage timing (2026-11-04)

Question: how much slower would the render/mask stage get if the RF-DETR
person segmenter were replaced by SAM 3.1, using the visiontouch TRT FP32
recipe? Live benchmark on this host's RTX 4090 (idle), same 14 real
1080p frames from GG/SB/Wolf masked ranges for every engine.

## How visiontouch runs SAM 3.1 (video_pipeline.py Sam3TRTWrapper)

- Model: **FP8-quantized dynk ONNX** (`sam3_grounding_fp8_dynk.onnx`,
  1.67 GB — image `ghcr.io/dishaeye/sam3-models:fp8-dynk-onnx`; the
  Dockerfile deploys exactly this). "FP32" in their code is the *build
  mode*, not unquantized weights: `_trt_build_engine(..., fp32=True)` =
  strongly-typed network, no FP16 builder flag → engine runs the graph
  as quantized (FP8 kernels on Ada).
- Build: workspace 20 GB, dynamic-K optimization profile (KMIN/KOPT/KMAX
  = 1/3/8) on every dynamic-dim input (`text_tokens (K,32)`, `img_ids (K,)`),
  input 1008×1008, TRT 11.2.1.2 (same as our container).
- Tokens: pre-computed 32-token text rows in npz tables (CLIP-style,
  minted offline by training/make_sam3_tokens.py); one row per prompt.
- Per pass: CPU preprocess (resize 1008², normalize) → persistent GPU
  buffers → set shapes → execute_async_v3 → sync → probs
  (sigmoid(logits)·sigmoid(presence)), boxes, and best-query mask
  (288² logits → bilinear → frame size → threshold → dilate).

## Measured (RTX 4090, median ms/frame, one person-prompt pass)

| engine | ms/frame | split | vs RF-DETR |
|---|---|---|---|
| **RF-DETR PersonSegmenter** 432² FP32 (current) | **6.7** | — | 1.0× |
| SAM 3.1 FP8 ONNX / FP32-mode engine, K=1 | **36.2** | pre 2.7 / exec 33.4 / decode 0.1 | 5.4× |
| SAM 3.1 FP8, K=3 (visiontouch opt) | 43.0 | pre 2.7 / exec 40.2 | 6.4× |
| SAM 3.1 unquantized FP32, K=1 | **175.4** | pre 2.8 / exec 172.4 | 26.2× |
| SAM 3.1 unquantized FP32, K=3 | 234.7 | pre 2.8 / exec 231.8 | 35.0× |

Decode of all queries ≥0.5 (the all-persons masking we need) costs the
same as best-query (0.14 ms) — mask decode is not a factor.

Build/size: FP8 engine 1,067 MB, built in 72 s; FP32 engine 3,472 MB,
built in 49 s. VRAM with engine + buffers resident: FP8 ≈ 5.1 GB total
(fine alongside the VLM), FP32 ≈ **21.7 GB — would not coexist with the
arbiter VLM on a 24 GB card**.

## Projected mask-stage walls (keyframes × per-frame ms)

Keyframes (full-frame-rate mask analysis inside flagged ranges):
GG 17,007 · SB 40,473 · Wolf 42,676. Current segmentation share
(keyframes × 6.7 ms) reconciles with the observed render walls
(e.g. SB: 271 s of the 459 s wall + encode ≈ wall ✓).

| movie | render wall now | +SAM 3.1 FP8 (K=1) | wall → | +SAM 3.1 FP32 (K=1) | wall → |
|---|---|---|---|---|---|
| GG | 574 s | **+8.4 min** | 1,076 s (1.9×) | +47.8 min | 3,442 s (6.0×) |
| SB | 459 s | **+19.9 min** | 1,654 s (3.6×) | +113.8 min | 7,285 s (15.9×) |
| Wolf | 589 s | **+21.0 min** | 1,849 s (3.1×) | +120.0 min | 7,787 s (13.2×) |

## Notes / caveats

- Bench frames were 1080p; production analysis frames are 640×360 —
  only the ~2.7 ms CPU preprocess shrinks (to ~1.5 ms), exec is
  resolution-independent (fixed 1008² input). Deltas hold ±1 ms.
- Prompt tokens: bench reused the football table's "player" row; a real
  deployment mints a person/nudity prompt — same (1,32) shape, same cost.
- Visiontouch's own numbers were never recorded in-repo (their tests use
  mocked ms); this benchmark *is* the measurement, on identical hardware
  (same GPU, same TRT 11.2.1.2 as their Dockerfile pins).
- If SAM 3.1's mask quality made a lower keyframe rate acceptable (its
  masks are semantically stabler across motion), part of the 5.4× could
  be clawed back by halving the keyframe fps — changes hold behavior,
  would need its own validation.
- Bench artifacts (gitignored, `trt_cache/bench/`): both ONNX pairs,
  both built engines, `bench_sam3.py`, `results*.json`, 14 frames.

Bottom line: **SAM 3.1 (the way visiontouch ships it) costs ~5.4× the
current segmenter per keyframe — +8 to +21 min per movie on these three.
The unquantized FP32 variant is 26× and does not fit VRAM with the VLM.**

## Measured: full SAM 3.1 renders (2026-11-04, after the timing bench)

The three NudeNet-free scan segment sets re-rendered with
`segmenter=sam3` (per-frame SAM 3.1 masks at native fps + temporal
smoothing; identical segments/policy/smoothing to the `*.masked.vlm.mp4`
renders — mask engine is the only variable). All verify gates ALL PASS.

| | GG | SB | Wolf |
|---|---|---|---|
| mask keyframes | 17,008 | 40,481 | 42,688 |
| frames masked (identical to vlm) | 16,997 | 40,497 | 42,675 |
| analysis (SAM 3.1) wall | 12.0 m | ~24 m | ~27 m |
| encode + mux wall | ~9.7 m | ~9.5 m | ~9.9 m |
| mask-phase total (submit→done) | **22.1 m** | **33.6 m** | **36.9 m** |
| — same movie on RF-DETR | 10.0 m | 12.7 m | 14.0 m |
| — delta | +12.1 m | +20.9 m | +22.9 m |
| verify gate | 11/11 | 11/11 | 10/10 |

Projections held (bench: +8.4/+19.9/+21.0 m; measured +12/+21/+23 —
encode ran ~1–2 m slower than the vlm renders, larger masks cost a
little more per-frame apply). Outputs: `mask_outputs/*.masked.sam3.mp4`.
Idle-GPU headroom (SM ≤12%) confirmed during both phases — throughput
fixes (CUDA graphs / batching / pipelining) are available later if the
quality verdict favors SAM 3.1.
