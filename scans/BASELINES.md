# v3 Baselines — the three validated movies (2026-11-04)

Config: sweep 5s · expand 192 · v3 context-gated arbiter prompt (apply,
add-only) · crowd=CringeMDB · bridge-noise demotion · solid-black masks
· per-frame mask keyframes (native fps) · auto-chapters. These are the
control group for the guide-context A/B (`guide=true`, IMDb Parents
Guide film-context). Masked outputs are final artifacts, kept as-is.

## Scan (stage A+B+D)

| | Gone Girl (2014) | Saltburn (2023) | Wolf of Wall St (2013) |
|---|---|---|---|
| runtime | 8,942 s (2h29) | 7,906 s (2h11) | 10,792 s (3h00) |
| scan wall | 1,483 s (6.0×) | 1,521 s (5.2×) | 1,925 s (5.6×) |
| final segments | 29 | 51 | 66 |
| — detector / crowd / arbiter | 4 / 4 / 21 | 10 / 11 / 30 | 28 / 9 / 29 |
| flagged time | 847 s (9.5%) | 1,844 s (23.3%) | 2,033 s (18.8%) |
| bridge-noise demoted | 12 | 20 | 89 |
| arbiter frames checked | 2,408 | 2,705 | 4,092 |
| — sweep / window / companion / weak | 1630/437/131/18 | 1271/976/240/26 | 1669/1922/240/69 |
| arbiter new segments (pre-merge) | 25 | 37 | 50 |
| scan record | `gone_girl_mask_v3b.json` | `saltburn_mask_v3b.json` | `wolf_mask_v3.json` |

## Render + gate (stage C) + chapters

| | Gone Girl | Saltburn | Wolf |
|---|---|---|---|
| frames | 214,355 | 189,741 | 258,758 |
| frames masked | 20,300 (9.5%) | 39,827 (21.0%) | 48,737 (18.9%) |
| mask keyframes | 20,094 (per-frame) | ~5.6k (8fps era)* | 46,066 (per-frame) |
| render wall | 585 s | 540 s* | 597 s (18.1×) |
| fail-safe renders | 1 (4158–4163) | 2 (1009–1016, 3845–3859) | 0 |
| chapters baked | 29 (+30 fillers) | 51 (+52) | 66 (+67) |
| verify gate | ALL PASS (12/12) | ALL PASS | ALL PASS (10/10, 149 s) |
| masked output | `Gone.Girl.2014.masked.mp4` | `Saltburn.2023.masked.mp4` | `Wolf.of.Wall.Street.2013.masked.mp4` |

*Saltburn was rendered before the per-frame switch (MASK_KEYFRAME_FPS
was still 8) and was not re-rendered; its scan record and chapter set
are current. Any A/B render delta on Saltburn should account for this.

## End-to-end timing (Wolf, the first full 3-call chain run)

Boot → `POST /scan` → `POST /mask` (auto-chapters) → `POST /verify`:

| phase | wall | notes |
|---|---|---|
| container boot | 6 s | models baked, engines cached |
| scan phase | 1,951 s | 1,925 s scan + poll/save overhead |
| render phase | 1,081 s | 597 s render (incl. per-frame analysis) + ~470 s chapter mux (stream copy + faststart of 2.2 GB) + polls |
| verify phase | 150 s | 149 s in-container gate, auto spots |
| **total** | **3,188 s (53m08s)** | **3.4× realtime** for a 3h movie |

Chapter mux is now the second-largest render-phase cost; it can be
folded into the encode (chapters written at mux time) if it matters.

## Observations for the review

- Wolf demoted 89 bridge-only ranges (vs 12/20 on GG/SB): heavy
  scene-cut flicker; the demotion path did its job at scale.
- Zero fail-safes on Wolf (GG 1, SB 2): NudeNet re-detection
  reproduced on every labeled segment.
- The gate's auto-spot selection collapsed to `[0]` on Wolf: union
  merges accumulate labels, so segment 0 satisfies every provenance
  class. Gate still verified 79%-black 544k-px mask presence + 3 clean
  spots. Improvement candidate: pick per-class segments by top label.
