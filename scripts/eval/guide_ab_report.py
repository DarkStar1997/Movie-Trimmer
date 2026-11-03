#!/usr/bin/env python3
"""Baseline-vs-guide A/B report over paired scan records.

For each movie pair (v3 baseline, guide run) this prints:
  segment table    provenance split, flagged time, demoted, arbiter stats
  verdict flips    frames judged by BOTH runs where explicit changed
  segment diffs    arbiter-only segments gained/lost with guide (IoU match)

Pairs are hardcoded (the A/B set); --self KEY sanity-checks the join
logic by comparing a record to itself (must report zero diffs).

Usage: python scripts/eval/guide_ab_report.py [--self gg|sb|wolf]
"""
import json
import sys

ROOT = "/home/rohan/repos/Movie-Trimmer"
PAIRS = {
    "gg":   ("gone_girl_mask_v3b.json", "gone_girl_mask_guide.json"),
    "sb":   ("saltburn_mask_v3b.json", "saltburn_mask_guide.json"),
    "wolf": ("wolf_mask_v3.json", "wolf_mask_guide.json"),
}


def load(fn):
    return json.load(open(f"{ROOT}/scans/{fn}"))


def provenance(s):
    labs = s.get("labels") or {}
    if any("cringemdb" in k for k in labs):
        return "crowd"
    if "arbiter" in labs:
        return "arbiter"
    return "detector"


def seg_table(d):
    prov = {}
    for s in d["segments"]:
        p = provenance(s)
        prov[p] = prov.get(p, 0) + 1
    flagged = sum(s["end"] - s["start"] for s in d["segments"])
    demoted = sum(1 for s in d.get("review_segments", [])
                  if "TRACK_BRIDGE" in (s.get("labels") or {}))
    a = d.get("arbiter") or {}
    return {"segments": len(d["segments"]), "prov": prov,
            "flagged_s": round(flagged),
            "flagged_pct": round(100 * flagged / d["duration"], 1),
            "demoted": demoted,
            "arb_checked": a.get("frames_checked"),
            "arb_new": len(a.get("new_segments") or [])}


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def fmt_seg(s):
    labs = ",".join(sorted((s.get("labels") or {}).keys()))[:60]
    return (f"{s['start']:8.1f}-{s['end']:8.1f} "
            f"({s['end']-s['start']:6.1f}s) {provenance(s):8s} {labs}")


def report(key, base, guide):
    print(f"===== {key} =====")
    g = guide.get("guide") or {}
    print(f"guide: tt={g.get('tt')} cache={g.get('from_cache')} "
          f"context={len(g.get('context_block') or '')}ch error={g.get('error')}")
    tb, tg = seg_table(base), seg_table(guide)
    print(f"{'':16s}{'v3':>12s}{'guide':>12s}")
    rows = [("segments", "segments"), ("flagged_s", "flagged_s"),
            ("flagged_pct", "flagged_pct"), ("demoted", "demoted"),
            ("arb_checked", "arb_checked"), ("arb_new", "arb_new")]
    for label, k in rows:
        print(f"{label:16s}{str(tb[k]):>12}{str(tg[k]):>12}")
    for p in ("detector", "crowd", "arbiter"):
        print(f"prov {p:10s}{str(tb['prov'].get(p, 0)):>12}"
              f"{str(tg['prov'].get(p, 0)):>12}")

    # verdict flips, joined on (round(t,2), reason)
    def verdicts(d):
        return {(round(v["t"], 2), v.get("reason")): v
                for v in (d.get("arbiter") or {}).get("frame_verdicts", [])
                if v.get("explicit") is not None}
    vb, vg = verdicts(base), verdicts(guide)
    common = set(vb) & set(vg)
    flips = [(t, r, vb[(t, r)], vg[(t, r)]) for (t, r) in sorted(common)
             if vb[(t, r)]["explicit"] != vg[(t, r)]["explicit"]]
    to_clean = sum(1 for *_, b, g in flips if b["explicit"])
    to_expl = len(flips) - to_clean
    print(f"verdict frames: base {len(vb)} / guide {len(vg)} / joined {len(common)}"
          f" | flips {len(flips)} (explicit->clean {to_clean}, clean->explicit {to_expl})")
    for t, r, b, g in flips[:40]:
        print(f"  flip {t:9.2f}s [{r}] {'EXPL' if b['explicit'] else 'clean'}"
              f"(c={b.get('confidence')}) -> "
              f"{'EXPL' if g['explicit'] else 'clean'}(c={g.get('confidence')})"
              f" {','.join(g.get('labels') or [])[:40]}")
    if len(flips) > 40:
        print(f"  ... {len(flips)-40} more")

    # arbiter-only segment diffs (trusted detector/crowd never changes)
    ab = [(s["start"], s["end"]) for s in base["segments"]
          if provenance(s) == "arbiter"]
    ag = [(s["start"], s["end"]) for s in guide["segments"]
          if provenance(s) == "arbiter"]
    gained = [s for s in guide["segments"]
              if provenance(s) == "arbiter"
              and not any(iou((s["start"], s["end"]), b) > 0.5 for b in ab)]
    lost = [s for s in base["segments"]
            if provenance(s) == "arbiter"
            and not any(iou((s["start"], s["end"]), g) > 0.5 for g in ag)]
    print(f"arbiter segments: {len(ab)} -> {len(ag)} "
          f"(gained {len(gained)}, lost {len(lost)})")
    for s in gained:
        print(f"  + {fmt_seg(s)}")
    for s in lost:
        print(f"  - {fmt_seg(s)}")
    print()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--self":
        key = sys.argv[2]
        d = load(PAIRS[key][0])
        print(f"[self-check {key}: comparing {PAIRS[key][0]} to itself; "
              "flips and diffs must be 0]\n")
        report(key, d, d)
    else:
        for key, (bf, gf) in PAIRS.items():
            try:
                report(key, load(bf), load(gf))
            except FileNotFoundError as e:
                print(f"===== {key} =====  skipping ({e.filename} not yet present)\n")
