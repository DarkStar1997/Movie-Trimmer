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

## 4. Mask-coverage changes — ADJUDICATED 2026-11-04: ALL LEAVE-CLEAN

The user reviewed every changed range (frames in `scans/review/`,
3 per range): verdict **leave clean on all 11**. That flips the
reading — every range the guide unmasks was a **v3 false positive**,
and the guide's own additions are its only errors.

**v3 FPs confirmed** (masked in the shipped baselines, should be clean)
— all pure-arbiter sweep-window ranges, no detector/crowd backing:

| movie | ranges | total |
|---|---|---|
| GG | 1275–1295, 1930–1950, 4115–4135, 7155–7195 (40 s), 8350–8370 | 120 s |
| SB | 950–970, 5752.8–5795, 3620.5–3623.5, 6700–6725 (= fp4 rejection) | 90 s |
| Wolf | 4045.2–4050.2 | 5 s |

**~215 s of v3 over-masking total.** The "validated" baselines were
gate-validated, not frame-adjudicated — the fp4 class was larger than
the four SB ranges originally identified.

**Guide FPs confirmed** (masked only in guide runs, should be clean):
GG 1435–1455 (20 s); SB 3643.5–3646.5 (3 s) + 5760–5780 (20 s remnant
of the rejected range). **~43 s total.**

Still open from the original fp4 rejection: **SB 4545–4565 and
7215–7237 remain flagged by BOTH runs** — user-rejected content that
neither prompt version catches; the corrected finals drop them
explicitly.

## 5. Gates + artifacts

All three `*_masked_guide.mp4` outputs pass the full verify gate
(codec/duration/packets/audio bit-exact/mask-present/clean-spot checks),
chapters auto-baked (27/50/65). Notably the guide runs rendered **zero
fail-safes** (baselines: GG 1, SB 2) — GG's old fail-safe zone 4158–4163
now carries a real arbiter-found mask (4160.9, 415k px).

Outputs: `mask_outputs/{Gone.Girl.2014,Saltburn.2023,Wolf.of.Wall.Street.2013}.masked_guide.mp4`
Records: `scans/{gone_girl,saltburn,wolf}_mask_guide.json`
Comparison tool: `scripts/eval/guide_ab_report.py`

## 6. Verdict (post-adjudication) — flipped

The adjudication reverses §2's practical pessimism while confirming its
mechanism finding:

- **v3's arbiter over-masks borderline sweep ranges**: ~215 s across
  the three baselines was adjudicated clean (10 pure-arbiter ranges,
  no detector/crowd backing — that class is the FP pattern).
- **Guide context fixed all 10** and added ~43 s of its own FPs
  (3 ranges). Net on changed ranges: 215 s of wrong masking removed
  vs 43 s added.
- The replay findings stand (STEP-1 leakage, +2 clean-space FPs) — the
  mechanism is real, but on these three severe-rated guides its net
  direction matched the user's judgment.

Disposition:

- **Corrected finals**: GG/SB re-rendered from the guide segment sets
  minus the adjudicated FP ranges (+ the two still-flagged fp4 ranges
  on SB); Wolf's guide output already matches the adjudication as-is.
- **guide=true stays opt-in** until validated on a movie with a
  Mild/None-rated guide — the leakage risk is untested in the
  under-tagging direction, which is where it would bite.
- Follow-up candidate: offline pass over recorded verdicts for an
  arbiter new-segment evidence threshold — pure-arbiter sweep ranges
  without detector backing were 100% of the confirmed FPs; no
  detector-backed range was ever wrongly masked.
