# Throughput (2026-11-05)

Where the walls go and what was changed. Quality constraint: mask
content and encode settings stay byte-identical — speed comes from
scheduling, not from cheaper inference (user decision: NVENC preset
stays P5; P4/P3 measured and declined for now).

## Render side — settled

Probe (`trt_cache/bench/probe_gpu_render.py`, 4090, GG 10-bit + Wolf
8-bit, 12k frames):

- NVDEC decode-only: **2,316 fps** — never the bottleneck.
- NVENC P5-HQ encode: **~2.2 ms/frame (~450 fps ceiling)**, synchronous
  per frame. Not the per-frame `torch.cuda.synchronize()` — event/none
  sync modes just move the same wait into the Encode timer. The sync
  stays (it orders our async torch kernels before NVENC reads the
  surface; without it overlays flicker off).
- Preset knobs (Wolf 8-bit, CQ 26, 12k frames): P4 493 fps / +1.6%
  size; P3 780 fps / +15.5% size. **Kept P5 per user decision.**

**Change: analysis/render overlap** (`MASK_OVERLAP=1`, default on;
commits 43a87d5 + 8939a0b). Full renders start the GPU render as soon
as segment 0's masks exist; analysis of the remaining segments runs in
a companion thread (`on_segment(si, entries)` hands entries + a
per-segment event to the render head, which builds each segment's
keyframe index lazily via `seg_ready`). Measured GG:

| | serial (baseline) | overlap |
|---|---|---|
| total job wall | ~700 s (114 analysis + 574 render) | **585.7 s** |
| render phase | 574 s | 580.6 s (unaffected by sharing) |
| analysis | 114 s serial | 140.2 s, fully hidden |

Verify gate after the fix: ALL PASS, seg 0 masked 703,154 px ==
baseline render's 703,154 px; segs 5/9/13/18/19 match within re-encode
noise; frames_masked 16,997 ~= 17,007 keyframes; audio bit-exact,
packet counts equal.

Post-mortem worth keeping: the first overlap build shipped segments the
render head entered during analysis **unmasked** — `on_segment` fired
with the data still locked inside `analysis_pass`'s local dict (only
returned after EVERY segment). Caught by the verify gate (seg 0:
7,676 px vs 703,154). Fix: the callback carries the entries;
regression test `test_analysis_pass_on_segment_delivers_entries_inline`.

Record keys changed: `wall_time_s` is now the true total (it was
clobbered by `**render` with the render-phase wall — old records
understate jobs by the analysis time); new `analysis_wall_s` +
`render_wall_s`.

## Scan side — instrumented, optimizing

`run_scan` now emits `stages` in every record: `decode_gate_s` (scan
start through decode+detector+gate+smoothing+gate-join), `crowd_s`,
`guide_s`, `arbiter_s` (lead clustering + VLM judging + apply),
`finalize_s` (residual).

Baseline walls (pre-instrumentation records): GG 1245 s / SB 1249 s /
Wolf 1426 s; arbiter frames_checked 2059/2010/2606, one image per HTTP
request through llama-server `-np 4` slots at
`ARBITER_HTTP_CONCURRENCY=4`.

**Measured (GG, 2026-11-05, crowd+guide cached):**

| stage | seconds | share |
|---|---|---|
| decode+gate | 186 | 15% |
| crowd | 1.0 | — |
| guide | 0.4 | — |
| **arbiter** | **1086** | **85%** |
| wall total | 1273.6 | |

The arbiter is the scan. Two costs inside it:

1. **VLM judging** — 2059 frames x ~2.1 s each through 4 server slots
   (`-np 4`, `-c 8192`, pool 4). Lever: slots/pool 4 -> 8 with
   `-c 16384` (per-slot context stays 2048) via run.py env.
2. **Frame extraction** — the select pass requested the 0.25 s sample
   grid with +-5 ms windows, but mkv's 1 ms timebase quantizes frame
   PTS up to ~21 ms off the grid: only ~26% of windows hit (478/1867),
   and EVERY frame then re-extracted by individual ffmpeg seeks
   (~2,000 seeks/run). Fixed (see below): requests snap to actual
   frame PTS from a demux-only ffprobe pass; only genuinely missing
   frames take the seek path.

(np8 + snap-fix measurements land below as they run)
