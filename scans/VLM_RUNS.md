# NudeNet-free pipeline runs (2026-11-04) — review packet

`detector=false`: person gate (RF-DETR) + arbiter sweep (5s, person-gated,
windows) + crowd + guide context. No NudeNet anywhere — scan or render
(all-person masks via the segmenter, with the new temporal smoothing).
Outputs `*.masked.vlm.mp4`, records `scans/*_vlm.json`. All three verify
gates ALL PASS.

## Runs vs detector-path baselines

| | GG | SB | Wolf |
|---|---|---|---|
| segments (arb + crowd, **0 detector**) | 21 (17+4) | 33 (22+11) | 31 (22+9) |
| flagged time | 709 s | 1,686 s | 1,780 s |
| — detector-path final | 720 s | 1,664 s | 2,018 s |
| scan wall (speedup) | 1,245 s (7.2×, was 1,483) | 1,249 s (6.3×, was 1,521) | 1,426 s (**7.6×**, was 1,925) |
| render wall | 574 s | 459 s | 589 s |
| e2e (scan→gate) | 35m11s | ~34m | ~44m |
| frames masked | 16,997 | 40,497 | 42,675 |
| verify gate | 11/11 | 11/11 | 10/10 |
| fail-safes | 0 (mechanism gone) | 0 | 0 |

Scans are 16–26% faster with NudeNet gone; Wolf (3h movie) dropped
53m→~44m e2e. Masks are temporally smoothed in all three renders.

## Segment diffs vs the adjudicated finals

**Compliant with your adjudications (15 of the ranges you ruled on):**
- GG detector FPs 4158/4741/5504/6670: gone ✓ (and the arbiter re-found
  none of them — matches the sweep sim, 0/4)
- GG 1275–1295, 1930–1950, 4115–4135, 8350–8370: stay gone ✓
- SB 950–970, 3620.5–3623.5, 6700–6725: stay gone ✓
- Wolf 4045.2–4050.2: gone ✓

**Regressions vs your adjudications — 5 ranges (~100 s) re-added:**
| range | your prior verdict |
|---|---|
| GG 1435–1455 | clean (guide FP) |
| GG 7165–7185 | clean (inside 7155–7195) |
| SB 4545–4565 | clean (fp4 rejection) |
| SB 5760–5780 | clean (guide FP remnant) |
| SB 7215–7235 | clean (fp4 rejection) |

Pattern: all are 20s window-formed arbiter segments in borderline zones —
verdicts there are sampling-layout-sensitive (the sweep now covers old
detector territory, so different frames get judged than in any prior run).

**Detector-only ranges dropped without prior adjudication** (matches the
sweep sim; your `scans/review2/` verdicts confirm or veto):
- SB: 1.5–5.5, 1009.8–1016.8, 3845.5–3859.5, 4740.5–4744.5, 5289.8–5298.5,
  6583.8–6599.8 (~54 s)
- Wolf: 22 ranges (~316 s, mostly brief MALE_GENITALIA singles)
- Also dropped, unadjudicated arbiter ranges: SB 6435–6460, 6635–6650;
  GG 7210–7230; Wolf 5232.2–5235.2

**New catches no prior run flagged** — need your verdict
(frames in `scans/review3/`, 3 per range):
- SB 80.5–120 (39.5s), 3599.5–3620 (20.5s), 4348–4375 (27s),
  **4395–4515 (120s!)**
- Wolf 2296.5–2395 (98.5s), **6495–6655 (160s!)**, 8830–8875 (45s),
  9815–9875 (60s)

## What's owed for the final verdict

1. `scans/review2/` — SB/Wolf detector-only ranges (are the drops right?)
2. `scans/review3/` — the 12 new/re-added ranges above (mask or clean?)
3. The 5 re-added regressions say the arbiter's borderline 20s windows
   need a stability rule (e.g. require ≥2 explicit sweep hits before a
   window forms) — worth designing after your verdicts.

If verdicts land mostly "drops right, new catches real", the NudeNet-free
pipeline becomes the default and the NudeNet weights leave the image.

## Verdict — NudeNet removed completely (2026-11-05)

User decision on the adjudication material: **"Remove NudeNet completely.
Not needed."** The NudeNet-free pipeline is the only pipeline; the
review2/review3 adjudication queue is superseded — detector-only ranges
stay dropped, the new catches stay in.

Shipped as commit 497c6a5 (+ images scanner 3aac2e63 /
allinone a0f2df11, pushed to GHCR):

- `scanner/detector.py`, `policy.py`, `tracking.py` deleted; weak/
  companion leads, person_points proxies, PRIO_WEAK tier gone from the
  arbiter; mask.py is vlm-only (all-persons union + center fallback);
  personseg keeps `persons_union`/`person_presence` only (match_box
  machinery removed with its only consumer).
- API: `/scan` loses `policy`/`model`/`backend`/`fp32`/`track`/
  `detector`; `/mask` loses `policy`. Detector-era record fields stay as
  empty structures for consumers. 110 tests green (smoke test now needs
  no weights).
- Docker: nudenet-weights stage + COPYs dropped from scanner.Dockerfile.

Post-strip GG confirmation scan (NudeNet-free build, same env as the C
stage measurement): see THROUGHPUT.md note below.

## Infinity Pool (2026-11-07) — guide default-on validation

First run with `DEFAULT_GUIDE = True` (guide context on unless
`guide=false`), on a fourth movie: Infinity Pool (2023), tt10365998,
117 min h264 1080p. Guide seeded browser-assisted into
`scans/guide_cache/tt10365998.json` (Chrome via browser-harness; IMDb's
embedded page JSON mined directly — 4 nudity items, Severe 368/505).

Submitted with **no explicit guide field** — exercising the default:

- scan 720 s (decode+gate 189.2 · arbiter 502.6 · guide 1.0) →
  **29 segments, 646 s flagged (9.2%)**, all `arbiter`-labeled
- guide: `from_cache: true`, tt10365998, 1,115-char context block —
  the default-on path works end-to-end from filename resolution
- crowd: CringeMDB has no page for this film (deterministic slug
  404s) — the fail-soft miss path exercised on a real movie;
  `crowd.error` recorded, scan unaffected
- render 511 s → 15,486 frames masked, 59 chapters (29 mask + 30
  gap), output 1.77 GB (hevc p5)
- verify 100 s → **ALL PASS 10/10** (spots auto-derived: 1 masked +
  3 clean)
- e2e 22:11 (scan 05:15 → verify 05:37)

Images: scanner 182abd482056 / allinone 9c5b504e2067 (guide-default
build) pushed to GHCR.

