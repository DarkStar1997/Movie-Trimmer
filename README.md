# Movie-Trimmer

GPU-native NSFW auto-detection and masking for movies: a FastAPI service that
scans a film for explicit content, then renders a copy in which every flagged
moment is masked with **person-shaped masks** — no cuts, no timeline drift,
audio and subtitles bit-exact. It began as a movie trimmer / subtitle merger
(`main.py`, still here) and grew the scanner (`scanner/`) into the main event.

The whole pipeline — detector, person gate, VLM arbiter, community
timestamps, GPU mask render — runs in **one container** on one GPU.

![pipeline](docs/pipeline.svg)

Design principle throughout: **add-only**. Detector, crowd and VLM can each
only add territory; nothing can shrink or remove a flagged range, so
re-scanning can never reduce coverage.

## Quick start

Prereqs: an NVIDIA GPU (24 GB for the VLM arbiter; scans without the arbiter
fit in ~4 GB), the NVIDIA container toolkit, and ffmpeg/ffprobe on the host
(verification only).

```bash
# put the movie anywhere in the repo (mounted read-only at /movies)
mv My.Movie.2023.1080p.mkv .

# run the published appliance (pulls from GHCR; --build rebuilds locally)
python3 run.py --wait 120          # healthy → http://localhost:8081/health
```

`run.py` runs a `--rm` container under the definite name `movie-trimmer`,
mirroring `docker-compose.yml` (engine cache, repo read-only at `/movies`,
mask outputs). It also removes any foreign container squatting port 8081
(e.g. an old compose stack). `docker stop movie-trimmer` tears it down.

## Scanning

```bash
# detector-only (fast, no VLM): NudeNet + tracking + smoothing
curl -X POST http://localhost:8081/scan -F video_path=/movies/film.mkv

# full stack: + person-gated VLM arbiter + CringeMDB crowd timestamps
curl -X POST http://localhost:8081/scan \
  -F video_path=/movies/film.mkv -F arbiter=true -F crowd=true

curl http://localhost:8081/scan/{scan_id}    # 202-async → status + segments
```

Each request returns immediately; poll `GET /scan/{scan_id}` until
`completed`. The result JSON carries the segment list (seconds, `labels`
naming the source: NudeNet class / `arbiter` / `cringemdb:<category>`),
per-frame arbiter verdicts, crowd bookkeeping, and timings.

**How the sources combine:**

- **Detector (NudeNet 640m, TRT fp16)** — the always-on backbone:
  **39–54× realtime** (a 2-hour film ≈ 3 min; ~935 fps sustained decode,
  every frame decoded, 1-in-6 kept at 0.25 s sampling); weak sub-threshold
  flicker is bridged by a person-cluster tracker.
- **Person gate (RF-DETR Seg Medium, TRT fp32)** — presence pass riding
  the same decode at zero extra decode cost (0.5 s resolution); computes
  ranges where a person is on screen. The arbiter only spends frames
  there: an empty hallway cannot host explicit content. Gate recall is
  audited, not assumed (78/78 explicit hits covered on the calibration
  movie).
- **VLM arbiter (Qwen3-VL-30B-A3B via llama-server)** — judges sparse,
  curated frames: a person-gated global sweep (every 10 s), dense 1 s-spaced
  windows hugging every trusted segment's edges (10 s before / 45 s after —
  scene spill lives there), and ±10 s ungated expansion around new hits.
  Explicit findings become new segments; existing ones are never re-judged.
  The server is **on-demand**: spawned when arbiter work starts (~20 GB,
  6–8 s warm / minutes cold), stopped when it ends — idle VRAM use is zero.
  Bring-up failure is loud (error block in the result JSON), never silent.
- **Crowd (CringeMDB scene guides)** — community timestamp ranges for
  nudity/sex categories, cached on disk, fail-soft (site down → scan
  proceeds). Entries are padded and treated as trusted territory.

## Masking

```bash
# from a completed scan (same container run); output_name names the
# final file, and mask-region chapters are baked automatically
curl -X POST http://localhost:8081/mask \
  -F video_path=/movies/film.mkv -F scan_id={scan_id} \
  -F output_name=Film.2023.masked.mp4

# from explicit segments — send FULL segment dicts, not bare pairs:
# labels are load-bearing ('arbiter'/'cringemdb' segments get the
# mask-every-person path; bare pairs lose them and render unmasked)
curl -X POST http://localhost:8081/mask \
  -F video_path=/movies/film.mkv \
  -F 'segments=[{"start": 42.0, "end": 58.0, "labels": {"arbiter": 1}}]'

# fast preview of one window (chapters skipped: offset timeline)
curl -X POST http://localhost:8081/mask -F video_path=/movies/film.mkv \
  -F 'segments=[[844.8, 859.5]]' -F window_start=840 -F window_end=865 \
  -F bake_chapters=false

curl http://localhost:8081/mask/{mask_id}    # → output path in mask_outputs/
```

Full renders are one continuous GPU-native pass: NVDEC → zero-copy torch →
RF-DETR person-mask pixelation on the GPU → NVENC (hevc). Frames never touch
the CPU; audio and subtitles pass through bit-exact — frame and packet counts
match the source exactly. Masks are segmented per frame at the source's
native fps (`MASK_KEYFRAME_FPS=0`; a fixed economy rate can be set). Every
flagged frame also carries a red overlay
(`VULGARITY DETECTED / <conf>% <label>`); `MASK_OVERLAY=0` disables it.

## Chapters

Every full render ends with mask-region chapters baked in (stream copy,
no re-encode): each segment becomes a named chapter — `▶ MASK · VLM 94%` —
and gap fillers bracket every mask with seek-bar ticks in mpv. Names flash
on seek and drive chapter navigation (`PgUp`/`PgDn`). A completed render
can be (re-)chaptered standalone:

```bash
curl -X POST http://localhost:8081/chapters \
  -F output_path=/mask_outputs/OUT.mp4 \
  -F 'segments=[{"start": 42.0, "end": 58.0, "labels": {"arbiter": 1}}]'
```

## Verification

`POST /verify` (in-container, CPU) re-checks a render against its source
without trusting the pipeline: container/duration sanity, full decode with
`-err_detect explode`, exact video packet count, audio payload md5
(bit-exact), mask **presence** inside flagged segments (hot-pixel diff +
blackout + texture crushing against the nearest source frames, matched by
raw PTS), and mask **absence** in clean gaps. Spots are auto-derived from
segment provenance (detector / crowd / arbiter + evenly-spread clean
times) unless given explicitly:

```bash
curl -X POST http://localhost:8081/verify \
  -F video_path=/movies/film.mkv -F output_path=/mask_outputs/OUT.mp4 \
  -F 'segments=[...]'            # or -F mask_id={id} alone
curl http://localhost:8081/verify/{verify_id}   # → checks[], all_pass
```

## Validated end-to-end

Two full movies, no manual intervention between phases, judged by the gate:

| | Gone Girl (2014) | Saltburn (2023) |
|---|---|---|
| runtime / frames | 8942 s · 214,355 | 7906 s · 189,741 |
| segments found | 38 | 76 |
| flagged time | 701 s (7.8 %) | 1747 s (22.1 %) |
| frames masked | 15,612 | 39,827 |
| VLM new / edge-ext / merged | 16 / 5 / 2 | 11 / 31 / 7 |
| scan → mask → verify | 882 / 507 / ~300 s | 1351 / 540 / 157 s |
| throughput | decode ~935 fps · render 423 fps · arbiter 81–85 % of scan wall | |
| verify gate | 24/24 PASS | 37/37 PASS |

Saltburn end-to-end: **34.3 min** for a 131-minute film. Source codecs
covered: hevc 10-bit 16:9 and h264 8-bit 4:3.

## API summary

| endpoint | purpose |
|---|---|
| `POST /scan` | async scan (form: `video_path`, `arbiter`, `crowd`, `guide`, `arbiter_mode=apply\|verify`, `sample_fps`, `policy`, `smoothing`, `track`, …) |
| `GET /scan/{id}` | status + result JSON |
| `POST /mask` | async render (form: `video_path`, `scan_id` or `segments`, `cq`, `window_start/end`, `output_name`, `bake_chapters`) — solid-black person masks + chapters |
| `GET /mask/{id}` | status → `output` path, render stats, chapters |
| `POST /verify` | async render gate (form: `video_path`+`output_path`, or `mask_id`; optional `masked_i[]`, `clean_t[]`) |
| `GET /verify/{id}` | status → `checks[]`, `all_pass` |
| `POST /chapters` | bake mask chapters into a render in place (`output_path` or `mask_id`, + `segments`/`scan_id`) |
| `GET /scans`, `GET /masks`, `GET /verifies` | lists |
| `GET /health` | GPU/TRT state, models, on-demand VLM server state |

## Configuration

Key env knobs (all have sane defaults; see `scanner/config.py`):

| env | default | meaning |
|---|---|---|
| `PERSON_GATE_ENABLED` | 1 | RF-DETR gate for the arbiter sweep |
| `ARBITER_SWEEP_EVERY_S` | 10 | global sweep spacing (person-gated) |
| `ARBITER_MAX_FRAMES` | 0 | arbiter frame cap (0 = uncapped) |
| `ARBITER_SWEEP_PAD_S` | 10 | pad around sweep/weak hits |
| `ARBITER_EXPAND_MAX_FRAMES` | 96 | expansion budget around new hits |
| `CROWD_CATEGORIES` | nudity,sex | in-scope crowd categories |
| `CROWD_PAD_S` | 1.5 | padding around crowd ranges |
| `GUIDE_CACHE_DIR` | scans/guide_cache | parents-guide context cache (guide=true; IMDb blocks live fetches, so guides are seeded here browser-assisted) |
| `VLM_START_TIMEOUT_S` | 900 | llama-server bring-up budget |
| `MASK_KEYFRAME_FPS` | 0 | mask analysis rate; 0 = every frame at native fps (capped 30) |
| `MASK_OVERLAY` | 1 | red warning overlay on flagged frames |
| `DEFAULT_ARBITER_MODE` | apply | `verify` reports verdicts, changes nothing |

## Repo layout

```
scanner/          the NSFW pipeline (decode, detector, person gate, arbiter,
                  crowd, mask, API) — packaged into the images
scripts/          segments_table.py, person_probe.py, eval_gt.py, eval/
                  (prompt-replay harness — dev-time, host-side)
docker/           entrypoints for the scanner / all-in-one images
tests/            137 CPU-only tests: pytest tests/ -q
run.py            run the appliance (docker run --rm, port hygiene)
build.py          build + push all images
docker-compose.yml  alternative launch (same container, restart: unless-stopped)
NSFW_SCANNER_DESIGN.md  the full design doc — research, calibration,
                  architecture and every incident (§1–13)
ANALYSIS.md       code review of the original trimmer
```

Models are baked into versioned weight images (nothing downloads at run
time; `HF_HUB_OFFLINE=1` makes accidental hub access a hard error):

```
ghcr.io/darkstar1997/movietrimmer-nudenet-weights:v1      NudeNet 320n + 640m
ghcr.io/darkstar1997/movietrimmer-rfdetr-seg-weights:v1   RF-DETR Seg Medium
ghcr.io/darkstar1997/movie-trimmer-vlm-qwen3vl-30b-a3b:v1 Qwen3-VL-30B GGUF
ghcr.io/darkstar1997/movie-trimmer-scanner:v1             scanner (stages A–C)
ghcr.io/darkstar1997/movie-trimmer:v1                     all-in-one appliance
```

## Development

```bash
python3 build.py            # build scanner + all-inone
python3 build.py --push     # build + push to GHCR
.venv/bin/python -m pytest tests/ -q    # host, CPU-only, no torch needed
```

TRT engines are cached in `trt_cache/`, keyed by a fingerprint of the ONNX
content and build recipe (fp16/fp32) — first use builds (~1–3 min per model),
after which scans start cold in seconds. Scan/mask artifacts land in
`scans/` and `mask_outputs/` (gitignored).

## The original trimmer

`main.py` (port 8000) is the pre-scanner core: cut-list trimming with
subtitle re-merging, frame-accurate, audio passthrough. A scan's segments
feed it directly. See `ANALYSIS.md` for its code review.
