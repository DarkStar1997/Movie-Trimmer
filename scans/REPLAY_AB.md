# Guide-context replay A/B (2026-11-04) — gate result: FAIL

Same-session replay: 766 adjudicated frames (SB5+SBbase+GGbase runs),
each judged twice through the production sidecar path — "v3" (baked
production prompt) vs "v3guide" (identical prompt + per-movie IMDb
parents-guide context block, the guide=true runtime wiring). Same
extracted JPEGs, temp 0, zero refusals both variants (119s/118s). This
is a clean causal comparison: the only delta is the system-prompt
suffix. (Agreement vs the ORIGINAL recorded verdicts is 84.9% for v3 —
that gap is JPEG re-extraction + decode jitter, the harness noise
floor; it does not contaminate the same-session diff below.)

## Direct diff (v3 -> v3guide, 766 frames)

| | n | flip | reading |
|---|---|---|---|
| explicit -> clean | 59 | 49 arb / 9 anat / 1 fp4 | guide suppresses |
| clean -> explicit | 15 | 10 arb / 3 anat / 2 ctrl | |
| **net** | **74 (9.7%)** | | strong clean-ward shift |

Class outcomes (frames explicit):

| class | n | v3 | v3guide | want |
|---|---|---|---|---|
| fp4 (user-rejected) | 5 | 3 | 2 | 0 — improved, not fixed |
| anat (in anatomy segs) | 107 | 98 | 92 | all — WRONG direction |
| arb (baseline-explicit) | 446 | 343 | 304 | mostly stay |
| ctrl (baseline-clean) | 208 | 2 | 4 | 0 — **+2 new FPs** |

## The two findings that fail the gate

1. **New FPs on baseline-clean frames** (zero required): GG 2667.25
   (FEMALE_BREAST,MALE_GENITALIA @0.9), SB 1288.50 (BUTTOCKS,
   MALE_GENITALIA @0.9).

2. **STEP-1 visibility leakage** — the block says "It never changes
   STEP 1: tag anatomy you can see regardless of this text", but the
   model demonstrably uses the context to suppress visibility: many
   flips DROP anatomy labels entirely with confidence unchanged or
   higher, e.g. GG 4637/4638/4641/4643
   (FEMALE_GENITALIA,FEMALE_BREAST,BUTTOCKS -> nothing @0.95), SB
   2485/2490 (FEMALE_GENITALIA -> nothing), SB 3600.75/3603.75, SB
   4397.50. Same JPEG, same session, same decode settings — the only
   cause is the guide text. Genitalia-labeled frames going clean is
   false-negative risk on content the baselines validated.

Genuine win observed too: SB 6715.00 (inside the user-rejected
6705-6725 range) correctly flips to clean.

## Verdict

- **Ship-as-default: NO.** Guide stays opt-in (guide=true), default
  off. The zero-new-FP gate failed and the mechanism's core promise
  (tie-break-only, STEP-1-safe) does not hold for this model.
- The full 3-movie pipeline A/B still runs (user request — review
  material); its segment-level outcome is expected to show net
  arbiter-segment loss on GG/SB, partially absorbed by smoothing.
- Full flip lists: mask_outputs/replay_v{3,3guide}.json; scorer:
  scripts/eval/score_prompt_replay.py; this diff joined on frame key.
