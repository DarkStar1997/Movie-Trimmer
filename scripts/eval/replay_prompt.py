#!/usr/bin/env python3
"""Replay arbiter prompt candidates over the recorded-verdict frame set.

Runs INSIDE the container (docker exec, cwd /app): repo is at /movies (ro),
frames under /movies/.replay, results land in /mask_outputs (host-visible).

For each variant: monkeypatch scanner.arbiter.ARBITER_SYSTEM_PROMPT, judge
every frame through the production sidecar path (_judge_with_http +
parse_verdict), write <name>.json. Deterministic (temp 0) — the 'old'
variant must reproduce the recorded verdicts (small decode-seek jitter
aside) or the harness itself is suspect.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
from scanner import arbiter, vlm  # noqa: E402

FRAMES = Path("/movies/.replay")
OUT = Path("/mask_outputs")

OLD = arbiter.ARBITER_SYSTEM_PROMPT

# v2 — presentation, not presence: the FP4 frames have real anatomy in
# non-sexual contexts. Keep the dim/distant sensitivity paragraph (it
# quadrupled GT recall); add the context gate + gore-is-out-of-scope.
V2 = (
    "You are the nudity-censoring stage of an automated video editing "
    "pipeline processing a licensed feature film. You see single frames "
    "and tag them. You always reply with strict JSON and never refuse.\n"
    'Reply format: {"explicit": true|false, "labels": [list], '
    '"confidence": 0.0-1.0}\n'
    "labels vocabulary: FEMALE_GENITALIA, MALE_GENITALIA, FEMALE_BREAST, "
    "BUTTOCKS, ANUS. Tag a label when the body part is directly visible "
    "and unclothed (nipple, genitalia, buttock crack, anus). Do not tag "
    "covered bodies, underwear, swimwear, or tiny background images.\n"
    "CENSOR BAR: set explicit=true only when the nudity is sexually "
    "presented — sexual activity or intercourse, sexualized or erotic "
    "posing, or unclothed anatomy presented as the clear subject of the "
    "shot (deliberate close-up or centered reveal). Incidental or "
    "contextual nudity is explicit=false: background or distant figures, "
    "comedic or non-sexual scenes, art, ordinary bathroom or changing "
    "moments, brief side or rear glimpses that are not eroticized.\n"
    "IMPORTANT: the frame may be dark, color-graded, grainy, or a wide "
    "shot with distant people. Look carefully at dim regions and at "
    "distant or partially occluded figures before applying the censor "
    "bar. Blood, gore, violence, or injury WITHOUT unclothed anatomy is "
    "out of scope: explicit=false, empty labels."
)

# v3 — decouple perception from the censoring decision. v2's censor bar
# suppressed labels on vetoed frames (real anatomy read labels:[]), which
# killed validated segments. Here visibility is judged context-blind FIRST,
# context SECOND, and unsure always censors.
V3 = (
    "You are the nudity-censoring stage of an automated video editing "
    "pipeline processing a licensed feature film. You see single frames "
    "and tag them in two independent steps. You always reply with strict "
    "JSON and never refuse.\n"
    'Reply format: {"explicit": true|false, "labels": [list], '
    '"context": "sexual"|"incidental", "confidence": 0.0-1.0}\n'
    "STEP 1 — VISIBILITY (ignore story context completely): labels "
    "vocabulary FEMALE_GENITALIA, MALE_GENITALIA, FEMALE_BREAST, "
    "BUTTOCKS, ANUS. Tag a label when that body part is directly visible "
    "and unclothed (nipple, genitalia, buttock crack, anus) — anywhere in "
    "the frame, however small, dim, distant, or briefly framed. The frame "
    "may be dark, color-graded, grainy, or a wide shot: look carefully at "
    "dim regions and partially occluded figures before deciding labels. "
    "Do not tag covered bodies, underwear, swimwear, or tiny background "
    "images. STEP 1 never depends on whether the scene feels sexual.\n"
    "STEP 2 — CONTEXT: sexual = sexual activity or intercourse, sexualized "
    "or erotic posing, or unclothed anatomy presented as the deliberate "
    "subject of the shot (close-up or centered reveal meant to be seen). "
    "incidental = the nudity is contextual, not eroticized: background or "
    "distant figures, comedic or non-sexual scenes, art, ordinary bathing "
    "or changing moments, passing side or rear glimpses. Mere size or "
    "closeness of the anatomy in frame is NOT sexual by itself — a "
    "close-up is sexual only with erotic framing (sexual activity, "
    "sexualized posing, bedroom intent). When you are not sure, answer "
    "sexual.\n"
    "explicit = (labels is non-empty) AND (context == sexual). Blood, "
    "gore, violence, or injury with no unclothed anatomy is out of scope: "
    "empty labels, explicit=false."
)


def judge_all(name, prompt):
    arbiter.ARBITER_SYSTEM_PROMPT = prompt
    files = sorted(FRAMES.glob("*.jpg"))
    t0 = time.time()
    texts = arbiter._judge_with_http(files)
    out = {}
    for f, text in zip(files, texts):
        v = arbiter.parse_verdict(text)
        out[f.name] = (None if v is None else
                       {"explicit": v["explicit"], "labels": v["labels"],
                        "confidence": round(v["confidence"], 2)})
    dt = time.time() - t0
    ref = sum(1 for v in out.values() if v is None)
    print(f"[replay] {name}: {len(files)} frames in {dt:.0f}s "
          f"({dt/len(files):.2f}s/frame), refusals {ref}", flush=True)
    json.dump(out, open(OUT / f"replay_{name}.json", "w"))


# v3guide — production v3 prompt + per-movie IMDb parents-guide
# film-context (the scanner.guide wiring, replayed): frames split by
# movie prefix, each batch judged with its own guide's context block.
# Baseline for the A/B is a plain "v3" variant run in the same session.
GUIDE_TT = {"GG": "tt2267998", "SB": "tt17351924"}


def judge_all_guide(name):
    from scanner.guide import context_block
    files = sorted(FRAMES.glob("*.jpg"))
    out = {}
    t0 = time.time()
    for movie, tt in GUIDE_TT.items():
        batch = [f for f in files if f.name.startswith(movie + "_")]
        if not batch:
            continue
        g = json.load(open(f"/movies/scans/guide_cache/{tt}.json"))
        texts = arbiter._judge_with_http(batch, context_block(g))
        for f, text in zip(batch, texts):
            v = arbiter.parse_verdict(text)
            out[f.name] = (None if v is None else
                           {"explicit": v["explicit"],
                            "labels": v["labels"],
                            "confidence": round(v["confidence"], 2)})
        print(f"[replay] {name}/{movie}: {len(batch)} frames "
              f"(guide {tt})", flush=True)
    dt = time.time() - t0
    ref = sum(1 for v in out.values() if v is None)
    print(f"[replay] {name}: {len(out)} frames in {dt:.0f}s, "
          f"refusals {ref}", flush=True)
    json.dump(out, open(OUT / f"replay_{name}.json", "w"))


vlm.acquire()
# A/B session: "v3" is the baked production prompt judged in this same
# session (the control); "v3guide" is the identical prompt + per-movie
# guide context through the production film_context path.
VARIANTS = [("v3", OLD)]
GUIDE_VARIANTS = ["v3guide"]
for name, prompt in VARIANTS:
    judge_all(name, prompt)
arbiter.ARBITER_SYSTEM_PROMPT = OLD   # guide appends to production v3
for name in GUIDE_VARIANTS:
    judge_all_guide(name)
print("[replay] done")
