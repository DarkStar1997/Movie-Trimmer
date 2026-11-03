"""Replay the arbiter frame planner under different coverage settings.

Inputs come from the recorded baseline scans (segments_pre_arbiter, person
gate ranges, weak hits) so scenarios are directly comparable to what ran.
Run from repo root: python3 scripts/eval/arbiter_coverage_sim.py
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scanner.arbiter import plan_frames  # noqa: E402

MOVIES = {
    "gg": ("scans/gone_girl_30b_allinone.json", 716.36, 881.71, 8942.046),
    "sb": ("scans/saltburn_2023_allinone.json", 1139.67, 1344.48, 7905.88),
}

SCENARIOS = {
    "current  sweep10  win 10/45/1": {},
    "user     sweep5   win  5/5/1":  {"sweep_every_s": 5.0,
                                      "window_before_s": 5.0,
                                      "window_after_s": 5.0},
    "sweep5   win 10/45/1":          {"sweep_every_s": 5.0},
}

for tag, (path, arb_s, total_s, dur) in MOVIES.items():
    d = json.loads(open(path).read())
    a = d["arbiter"]
    segs = a["segments_pre_arbiter"]
    gate = d.get("person_gate", {}).get("ranges")
    person_ranges = ([(r["start"], r["end"]) for r in gate]
                     if gate else None)
    weak = [h["t"] for h in d.get("weak_trigger_hits", [])]

    print(f"== {tag}: duration {dur:.0f}s | pre-arbiter segs {len(segs)} | "
          f"gate {'on' if person_ranges else 'OFF (scan predates it)'}")
    print(f"   recorded: planned {a['planned_frames']} + expand "
          f"{a['expansion_frames']} = checked {a['frames_checked']} "
          f"({arb_s:.0f}s arbiter, {total_s:.0f}s scan)")

    # value signal: which reasons found explicit frames / new segments
    from collections import Counter
    expl = Counter(v["reason"] for v in a["frame_verdicts"] if v["explicit"])
    print(f"   explicit frames by reason: {dict(expl)}")
    for ns in a["new_segments"][:60]:
        rs = {v["reason"] for v in a["frame_verdicts"]
              if ns["start"] - 1 <= v["t"] <= ns["end"] + 1}
    finder = Counter()
    for ns in a["new_segments"]:
        inside = [v["reason"] for v in a["frame_verdicts"]
                  if ns["start"] - 1 <= v["t"] <= ns["end"] + 1]
        finder[inside[0] if inside else "?"] += 1
    print(f"   new segments first-seen-by: {dict(finder)}")

    rate = arb_s / a["frames_checked"]
    for name, kw in SCENARIOS.items():
        frames = plan_frames(dur, segs, person_ranges=person_ranges,
                             weak_hits=weak, **kw)
        from collections import Counter
        rc = Counter(f.reason for f in frames)
        n = len(frames)
        new_arb = (n + a["expansion_frames"]) * rate
        new_total = total_s - arb_s + new_arb
        print(f"   {name}: {n:5d} frames {dict(rc)} "
              f"-> arbiter {new_arb:6.0f}s, scan {new_total:6.0f}s "
              f"({dur / new_total:.1f}x realtime)")
    print()
