# Pipeline Analysis — 2026-10-31

Analysis of `main.py` (FastAPI + ffmpeg/ffmpeg-NVENC trimming service) before further work.
Environment tested: ffmpeg n9.0.2, RTX 4090 (driver 615.78.08), Python 3.14.

## Architecture (what the code does)

1. `POST /submit` — accepts video + optional `.srt` + cut spec (`segments` = ranges to **remove**,
   as `"HH:MM:SS-HH:MM:SS, ..."` or JSON `[[start, end], ...]`). Saved to `temp_workspaces/<job_id>/`.
2. Keep-segment math — inverts the remove-ranges into keep-ranges. Handles unsorted and
   overlapping input correctly (`main.py:210-219`).
3. Per-segment encode — each keep-range re-encoded in one pass:
   `-hwaccel cuda -hwaccel_output_format cuda` (NVDEC, frames stay on GPU) → `av1_nvenc -preset p4`
   + `libopus` (`main.py:240-252`). Frame-accurate input seeking (`-ss` before `-i`).
4. Subtitle processing (parallel branch) — pysrt clips subtitles to keep-ranges, splits subs that
   span a cut boundary, and shifts timestamps to the new concatenated timeline (`main.py:110-162`).
5. Concat + mux — concat demuxer with `-c:v copy -c:a copy`, SRT muxed as soft `mov_text`
   (`main.py:254-282`). Output: `finished_jobs/processed_<job_id>.mp4`, served via `GET /download`.

**Core design is sound**: exactly one re-encode, GPU zero-copy decode→encode, lossless concat,
correct subtitle timeline math. Jobs run as FastAPI `BackgroundTasks`, throttled by
`asyncio.Semaphore(MAX_CONCURRENT_JOBS=2)`.

## Verified empirically on this machine

| Check | Result |
|---|---|
| `av1_nvenc` available | ✅ (h264/hevc/av1 nvenc all present) |
| `-ss` before `-i` + CUDA decode frame-accurate | ✅ 30/30 frames on a 1s @ 30fps cut |
| `av1_nvenc` default rate control | ⚠️ ~2 Mbps VBR (CPB avg 2000000) — no `-b:v`/`-cq` set in code |
| Opus→MP4 concat **without** `-strict -2` | ✅ works on ffmpeg 9.0.2 (older builds reject it) |
| Keep-segment inversion + subtitle clip/shift math | ✅ correct by inspection |

## Findings

### High

- **H1 — `--workers 2` breaks in-memory state** (`README.md:12`, `main.py:33-34, 39-41`)
  `job_store` and the semaphore are per-process: status/download polls 404 on the other worker,
  effective concurrency becomes 2×2, and startup `shutil.rmtree(TEMP_DIR)` races across workers.
  *Fix (quick):* run one worker, raise `MAX_CONCURRENT_JOBS` (4090 has 2 NVENC chips; 3–4
  concurrent AV1 sessions is reasonable). *Fix (real):* SQLite/Redis job store + real queue.

- **H2 — Blocking upload in async handler** (`main.py:304-305`)
  `shutil.copyfileobj` runs on the event loop; a multi-GB upload stalls every endpoint.
  *Fix:* `await run_in_executor(None, shutil.copyfileobj, ...)` or chunked `await file.read()`.

- **H3 — Filename path traversal** (`main.py:296`)
  `work_dir / f"input_{file.filename}"` with a crafted `filename` (`../..`, absolute path)
  writes outside the job workspace. *Fix:* `Path(file.filename).name`.

- **H4 — No video bitrate/quality set** (`main.py:247`)
  `av1_nvenc` defaults to ~2 Mbps VBR (verified) — visible artifacts on 1080p/4K film content.
  *Fix:* `-rc vbr -b:v 0 -cq 24 -spatial-aq 1`, plus explicit `-b:a 128k` for Opus.

### Medium

- **M1 — No `-map` on segment extraction** (`main.py:240-251`)
  ffmpeg auto-selection keeps only one audio track: multi-audio movies (dub tracks) silently lose
  the rest. The `-map 0:a` on the concat step is too late — the segments already have one track.
  *Fix:* `-map 0:v:0 -map 0:a?` in the extract command.

- **M2 — SRT indexes not renumbered** (`main.py:152-162`)
  Filtered subtitles are deepcopied with original indexes → gaps/duplicates. Most players tolerate;
  strict parsers don't. *Fix:* `final_subs.clean_indexes()` before save.

- **M3 — No job/file lifecycle** (`main.py:33`, `main.py:349-374`)
  `job_store` and `finished_jobs/` grow forever; no upload size cap; no delete/cleanup endpoint;
  server restart orphans finished files with no record. Disk-fill risk.

- **M4 — Segment validation gaps** (`main.py:57-94`)
  Inverted (`start > end`) or negative ranges are accepted → negative `-t` → cryptic ffmpeg
  failure mid-job. *Fix:* validate `0 <= start < end` at parse time.

### Low

- **L1 — Opus-in-MP4 portability** (`main.py:261-272`): concat omits `-strict -2`; fine on the
  current build, older ffmpeg would fail. Adding it is harmless.
- **L2 — Logging/timeouts** (`main.py:101-106`): ffmpeg output from concurrent jobs interleaves
  untagged (hard to attribute); no timeout if an encode hangs while holding a semaphore slot.
- **L3 — Output compatibility**: AV1 + Opus-in-MP4 has weak support on TVs/QuickTime/older
  players. Consider an optional H.264/AAC output profile.

## Recommended order of attack

1. H3 + H4 + M2 + M4 (small, local fixes — sanitize filename, quality flags, `clean_indexes`,
   segment validation).
2. H2 (async upload) + M1 (`-map`).
3. H1 decision: single worker (README change) now, or jump straight to persistent job store.
4. M3 retention/cleanup, then the L items as convenient.
