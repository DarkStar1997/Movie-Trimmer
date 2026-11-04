# Guide-context A/B — full results for review (2026-11-04)

Everything from the IMDb-Parents-Guide experiment, in one place. Companion
docs: `REPLAY_AB.md` (frame-level replay), `BASELINES.md` (v3 baselines).

## 1. What was tested

`guide=true` appends the movie's IMDb Parents Guide (Sex & Nudity items +
community severity) to the arbiter system prompt, worded as STEP-2
tie-break-only background. Everything else identical: same sweep/expand
params, same crowd, same v3 prompt, same models, per-frame masks,
auto-chapters. Three full runs (scan → mask → verify), outputs kept
distinct (`*_masked_guide.mp4`, `scans/*_mask_guide.json`).

## 2. Frame-level replay verdict (the mechanism test) — GATE FAILED

766 adjudicated frames, same session, same JPEGs, only delta = the
context block. 74 flips (9.7%), strongly clean-ward (59 vs 15):

| class | n | v3 explicit | guide explicit | want |
|---|---|---|---|---|
| fp4 (you rejected) | 5 | 3 | 2 | 0 |
| anat (in anatomy segs) | 107 | 98 | 92 | 107 |
| arb (baseline-explicit) | 446 | 343 | 304 | mostly keep |
| ctrl (baseline-clean) | 208 | 2 | **4** | **0 — FAIL** |

The mechanism's core promise does not hold: the model uses the context
to suppress **STEP-1 visibility** — anatomy labels vanish (incl.
FEMALE_GENITALIA) with confidence unchanged, despite the block
forbidding exactly that. Same frames, same decode settings; the guide
text is the only variable.

## 3. Full-pipeline results (what actually ships)

| | GG v3 | GG guide | SB v3 | SB guide | Wolf v3 | Wolf guide |
|---|---|---|---|---|---|---|
| final segments | 29 | 27 | 51 | 50 | 66 | 65 |
| — detector/crowd/arbiter | 4/4/21 | 4/4/19 | 10/11/30 | 9*/11/30 | 28/9/29 | 28/9/28 |
| flagged time | 847 s | 740 s | 1,844 s | 1,727 s | 2,033 s | 2,018 s |
| arbiter flips (→clean/→expl) | — | 27/7 | — | 64/13 | — | 30/9 |
| frames masked (render) | 20,300 | 17,739 | 39,827† | 41,492† | 48,737 | 48,390 |
| fail-safe renders | 1 | 0 | 2 | 0 | 0 | 0 |
| verify gate | PASS | PASS | PASS | PASS | PASS | PASS |
| scan wall | 1,483 s | 1,417 s | 1,521 s | 1,564 s | 1,925 s | 1,921 s |

*SB detector 10→9 is a merge artifact: detector seg 4581–4588 union-merged
with an arbiter seg that now starts at 4588; coverage and anatomy labels
fully preserved in the merged segment.
†SB baseline render was 8fps-era; guide render is per-frame keyframes —
frame counts are not comparable (see BASELINES.md note).

Guide context costs ~nothing at scan time (cache read + one suggestion
call; wall times within run-to-run noise).

## 4. Mask-coverage changes — YOUR ADJUDICATION LIST

Ranges whose masking actually changed (>50%-cover diff of final segment
sets). Review frames: `scans/review/{movie}_{start}_{LOST|NEW}_f*.jpg`
(3 frames per range from the source).

**GG — v3 masked, guide does not (FN risk on validated content):**

| range | len | notes |
|---|---|---|
| 1275–1295 | 20 s | guide keeps only 1285–1288 |
| 1930–1950 | 20 s | |
| 4115–4135 | 20 s | |
| 7155–7195 | 40 s | biggest single loss in the A/B |
| 8350–8370 | 20 s | |

**GG — guide masks, v3 does not (new catch or FP):** 1435–1455 (20 s).

**SB:**

| range | len | change | notes |
|---|---|---|---|
| 6700–6725 | 25 s | v3 → gone | **WIN: this is your rejected fp4 range 6705–6725** |
| 950–970 | 20 s | v3 → gone | you accepted this content in review — FN risk |
| 5752.8–5795 | 42 s | v3 → partial (5760–5780 kept) | net ~19 s unmasked |
| 3620.5–3623.5 | 3 s | v3 → gone | 3643.5–3646.5 appears nearby |
| 3643.5–3646.5 | 3 s | new | |

fp4 scorecard: 2652–2665 was already clean in v3; **6705–6725 fixed by
guide**; 4545–4565 and 7215–7235 still flagged by both.

**Wolf:** 4045.2–4050.2 (5 s) unmasked. Otherwise neutral — despite the
longest guide context, Wolf's arbiter barely moved.

## 5. Gates + artifacts

All three `*_masked_guide.mp4` outputs pass the full verify gate
(codec/duration/packets/audio bit-exact/mask-present/clean-spot checks),
chapters auto-baked (27/50/65). Notably the guide runs rendered **zero
fail-safes** (baselines: GG 1, SB 2) — GG's old fail-safe zone 4158–4163
now carries a real arbiter-found mask (4160.9, 415k px).

Outputs: `mask_outputs/{Gone.Girl.2014,Saltburn.2023,Wolf.of.Wall.Street.2013}.masked_guide.mp4`
Records: `scans/{gone_girl,saltburn,wolf}_mask_guide.json`
Comparison tool: `scripts/eval/guide_ab_report.py`

## 6. Recommendation

**Keep guide opt-in (`guide=true`), default off — v3 remains the shipping
configuration.** The replay shows the mechanism violates its STEP-1-
safety constraint and adds clean-space FPs; the pipeline shows real loss
of validated GG/SB content (up to 40 s per range) against one genuine
win (your rejected 6700–6725) and near-neutral Wolf behavior. The
arbiter's STEP-2 context call remains the weakest joint, but parents-
guide text is not a safe lever for it in this model's hands.

If the adjudicated verdicts in §4 come back mostly "guide was right",
revisit with a stricter block (e.g. severity only, no scene items).
