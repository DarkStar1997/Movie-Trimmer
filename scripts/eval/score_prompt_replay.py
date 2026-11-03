#!/usr/bin/env python3
"""Score replay_prompt.py outputs against the manifest + recorded scans.

Per variant:
  harness sanity : 'old' vs recorded verdict agreement (expect high; ffmpeg
                   seek may grab a neighbour frame occasionally)
  class rates    : fp4 flip-to-clean (want 100%), anat/arb stay-explicit,
                   ctrl stay-clean (want 100%)
  segment impact : per movie, which recorded arbiter segments keep enough
                   explicit hits to survive (a segment lives while any of
                   its forming frames still reads explicit)
"""
import json
import sys

ROOT = "/home/rohan/repos/Movie-Trimmer"
RUNS = [
    ("SB5", "SB", "saltburn_mask_sweep5.json"),
    ("SBbase", "SB", "saltburn_2023_allinone.json"),
    ("GGbase", "GG", "gone_girl_30b_allinone.json"),
]

manifest = json.load(open(f"{ROOT}/.replay/manifest.json"))
by_file = {f"{f['movie']}_{f['t']:07.2f}.jpg": f for f in manifest["frames"]}

# recorded explicit frames per run, and arbiter segments per run
rec_explicit, segs = {}, {}
for run, movie, fn in RUNS:
    d = json.load(open(f"{ROOT}/scans/{fn}"))
    rec_explicit[run] = {(movie, round(v["t"], 2))
                         for v in d["arbiter"]["frame_verdicts"]
                         if v.get("explicit")}
    segs[run] = [(s["start"], s["end"]) for s in d["arbiter"]["new_segments"]]

for variant in sys.argv[1:]:
    res = json.load(open(f"{ROOT}/blur_outputs/replay_{variant}.json"))
    agree = flips = 0
    cls_stat = {}
    for fn, v in res.items():
        f = by_file[fn]
        rec = f["recorded"]["explicit"]
        if v is not None:
            agree += (v["explicit"] == rec)
            if v["explicit"] != rec:
                flips += 1
        cls_stat.setdefault(f["cls"], [0, 0, 0])  # n, explicit, clean
        cls_stat[f["cls"]][0] += 1
        cls_stat[f["cls"]][1 if v and v["explicit"] else 2] += 1
    n = len(res)
    print(f"== {variant} ==  agreement(old-contract) {agree}/{n} "
          f"({agree/n:.1%}), flips {flips}")
    for cls in ("fp4", "anat", "arb", "ctrl"):
        tot, ex, cl = cls_stat[cls]
        print(f"  {cls:5s} {tot:4d}  explicit {ex:4d}  clean {cl:4d}")
    # segment survival: a segment lives if ANY explicit replay frame falls in it
    for run, movie, _ in RUNS:
        live = res_exp = 0
        dead = []
        for a, b in segs[run]:
            hit = any(v and v["explicit"]
                      and fn.startswith(movie)
                      and a <= by_file[fn]["t"] <= b
                      for fn, v in res.items())
            if hit:
                live += 1
            else:
                dead.append((a, b))
        print(f"  {run:7s} arbiter segs: {live}/{len(segs[run])} live")
        if dead:
            print(f"           dead: {[(round(a,1), round(b,1)) for a, b in dead]}")
    print()
