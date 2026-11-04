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

**B: slots 4 -> 8 (PARALLEL=8, CTX=16384, pool 8) — NULL RESULT.**
arbiter_s 1080 vs 1086: judging is compute-bound on this 4090 already
at 4 slots (30B-A3B prefills saturate SMs; more slots just queue).
Env kept: parity in wall, per-slot context unchanged (2048), VRAM
headroom verified with the server up. Not a lever; recorded so nobody
retries it blind.

**C: extraction fix** (commit 6bec98f): ffprobe packet-PTS snap
(0.7s) + select pass on exact frame timestamps (1856/1856 delivered,
one 126s decode, in-container) + batches <400 frames per-seek directly
(windows/expansion: ~0.2s/frame beats another full-movie decode).
Sweep extraction: ~500s -> ~130s. Host ffmpeg note: this host's build
fails to init >1.5k-term select chains (Cannot allocate memory); the
container build handles 1.9k — unit tests keep chains small.

**C result (GG, shipped build 6bec98f):**

| stage | A baseline | C | |
|---|---|---|---|
| decode+gate | 186 | 185 | control, unchanged |
| crowd+guide | 1.5 | 1.5 | cached |
| arbiter | **1086** | **609** | −44% |
| **wall** | **1273.6** | **795.7** | **−37.5% (1.60x)** |

Arbiter decomposition at C: VLM judging ~410s (compute-bound, the
honest floor for 2,059 frames through the 30B) + one 126s sweep
decode + ~40s small-batch seeks. Segment output parity across A/B/C
(21-24 segs, 709-726s flagged — run-to-run VLM variance, same as
historical repeat variance).

**Env reverted to -np 4 / CTX 8192 / pool 4** after C: slots 4->8
measured parity but took VRAM from ~6GB to ~2.1GB headroom with the
server up (22.4/24.6GB) — headroom a concurrent mask job's TRT
engines may need. The speedup is entirely the extraction fix.

## Where this leaves end-to-end (GG, 4090)

| | before | after | |
|---|---|---|---|
| scan | 1274s | 796s | 1.60x |
| mask render (total job) | ~700s | 586s | 1.19x |
| scan + render | ~1974s | ~1382s | **1.43x (−30%)** |

Remaining known costs, for whenever throughput matters again: VLM
judging (~410s/2k frames, compute-bound — a smaller/faster judge model
is the only real lever, a quality decision), the sweep decode (126s —
shared with decode_gate if ever worth pipelining), NVENC P5 encode
(~450fps ceiling — P4/P3 declined, table above).
