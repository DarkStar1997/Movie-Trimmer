#!/usr/bin/env python3
"""Blurred-sections table from a scan JSON.

Rows: index, timecode in/out, duration, sources with labels, mean conf.
Sources: arbiter (VLM explicit frames + labels from
arbiter.frame_verdicts), crowd (cringemdb category). Pre-2026-11-05
records may still carry NudeNet detector labels (legacy decode branch).
Usage: table.py SCAN.json [--json rows.json]
"""
import json
import sys

scan = json.load(open(sys.argv[1]))
segs = sorted(scan["segments"], key=lambda s: s["start"])
ar = scan.get("arbiter") or {}
fv = ar.get("frame_verdicts") or []
sv = ar.get("segment_verdicts") or []


def tc(s):
    m, sec = divmod(round(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def vlm_labels(a, b):
    """Explicit arbiter frames inside [a, b] -> (n, label->max conf)."""
    n, labs = 0, {}
    for f in fv:
        if f.get("explicit") and a <= f["t"] <= b:
            n += 1
            for lab in f.get("labels") or []:
                labs[lab] = max(labs.get(lab, 0.0), f.get("confidence", 0.0))
    return n, labs


def seg_verdict(a, b):
    """Candidate-verification record overlapping [a, b] (else None)."""
    for v in sv:
        if v["start"] <= b and v["end"] >= a:
            return v
    return None


rows = []
for i, s in enumerate(segs):
    a, b = s["start"], s["end"]
    src = []
    for k, v in s.get("labels", {}).items():
        if k == "arbiter":
            n, labs = vlm_labels(a, b)
            det = ", ".join(f"{l} ({c:.2f})" for l, c in sorted(labs.items()))
            src.append(f"VLM arbiter: {v} explicit frames"
                       + (f" — {det}" if det else ""))
        elif k.startswith("cringemdb:"):
            src.append(f"CringeMDB crowd: {k.split(':', 1)[1]}")
        else:
            src.append(f"detector (legacy record): {k} ×{v}")
    if not src:
        v = seg_verdict(a, b)
        if v and v.get("valid_frames"):
            src.append(f"arbiter {v['explicit_frames']}/"
                       f"{v['valid_frames']} frames explicit (kept: add-only)")
        else:
            src.append("trusted (crowd) — not re-judged by the VLM")
    rows.append({
        "i": i, "in": tc(a), "out": tc(b), "dur": round(b - a, 1),
        "sources": src, "conf": round(s.get("mean_score", 0.0), 2),
    })

if "--json" in sys.argv:
    json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
for r in rows:
    print(f"{r['i']:2d}  {r['in']}-{r['out']}  {r['dur']:5.1f}s  "
          f"conf {r['conf']:.2f}  " + " | ".join(r["sources"]))
print(f"\n{len(rows)} segments, total blurred "
      f"{sum(r['dur'] for r in rows):.0f}s of {scan['duration']:.0f}s "
      f"({100*sum(r['dur'] for r in rows)/scan['duration']:.1f}%)")
