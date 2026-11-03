#!/usr/bin/env python3
"""Build the arbiter-prompt replay set from recorded scans.

Classes:
  fp4       SB5 hit frames inside the 4 user-rejected ranges  -> must flip clean
  anat      explicit frames inside baseline anatomy segments  -> must stay
  arb       all other explicit verdict frames (baseline+SB5)  -> should stay
  ctrl      stratified clean verdict sample                   -> must stay clean

Frames extracted at the arbiter's analysis width (640) as JPEGs under
.replay/, manifest at .replay/manifest.json. Repo root is mounted at
/movies in the container, so the replay script sees them at /movies/.replay.
"""
import json
import os
import random
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOVIES = {
    "SB": "Saltburn.2023.1080p.AMZN.WEBRip.1600MB.DD5.1.x264-GalaxyRG.mkv",
    "GG": "Gone.Girl.2014.1080p.WEBRip.DDP5.1.x265.10bit-GalaxyRG265.mkv",
}
RUNS = [
    ("SB5", "SB", "saltburn_mask_sweep5.json"),
    ("SBbase", "SB", "saltburn_2023_allinone.json"),
    ("GGbase", "GG", "gone_girl_30b_allinone.json"),
]
ANATOMY = {
    "FEMALE_GENITALIA_COVERED", "FEMALE_GENITALIA_EXPOSED", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "MALE_GENITALIA_COVERED", "MALE_BREAST_COVERED", "BUTTOCKS_COVERED",
}
FP4 = [(2652.5, 2665.0), (4545.0, 4565.0), (6705.0, 6725.0), (7215.0, 7235.0)]

random.seed(11)
os.makedirs(f"{ROOT}/.replay", exist_ok=True)
frames = {}          # (movie, t) -> record  (dedupe across runs)

def add(movie, t, cls, run, recorded):
    key = (movie, round(t, 2))
    if key not in frames:                    # first run wins; classes agree
        frames[key] = {"movie": movie, "t": round(t, 2), "cls": cls,
                       "runs": [run], "recorded": recorded}
    else:
        frames[key]["runs"].append(run)

for run, movie, fn in RUNS:
    d = json.load(open(f"{ROOT}/scans/{fn}"))
    anat = [(s["start"], s["end"]) for s in d["segments"]
            if any(l in ANATOMY for l in s.get("labels", {}))]
    for v in d["arbiter"]["frame_verdicts"]:
        if v.get("explicit") is None:         # refusal / unparsed -> skip
            continue
        rec = {"explicit": v["explicit"], "labels": v.get("labels") or [],
               "confidence": v.get("confidence")}
        if v["explicit"]:
            if movie == "SB" and any(a <= v["t"] <= b for a, b in FP4):
                cls = "fp4"
            elif any(a <= v["t"] <= b for a, b in anat):
                cls = "anat"
            else:
                cls = "arb"
        else:
            continue                          # controls sampled below
        add(movie, v["t"], cls, run, rec)
    # clean controls: 60/run stratified by reason
    by_reason = {}
    for v in d["arbiter"]["frame_verdicts"]:
        if v.get("explicit") is False:
            by_reason.setdefault(v["reason"], []).append(v)
    for reason, vs in by_reason.items():
        for v in random.sample(vs, min(20, len(vs))):
            add(movie, v["t"], "ctrl", run,
                {"explicit": False, "labels": [], "confidence": v["confidence"]})

manifest = {"frames": sorted(frames.values(), key=lambda f: (f["movie"], f["t"]))}
json.dump(manifest, open(f"{ROOT}/.replay/manifest.json", "w"), indent=1)

n = 0
for f in manifest["frames"]:
    run = f["runs"][0]
    out = f"{ROOT}/.replay/{f['movie']}_{f['t']:07.2f}.jpg"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-ss", str(f["t"]), "-i", f"{ROOT}/{MOVIES[f['movie']]}",
                        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3",
                        "-y", out], check=True)
    n += 1

from collections import Counter
c = Counter(f["cls"] for f in manifest["frames"])
print(f"frames: {n}  classes: {dict(c)}")
