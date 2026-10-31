#!/usr/bin/env python3
"""Evaluate a scan's segment coverage against a ground-truth scene list.

Usage:
    python scripts/eval_gt.py scans/scan.json scans/gt_cringemdb_exmachina.json

Prints per-scene coverage and total time-recall. GT scenes are also
credited within --tolerance seconds (CringeMDB warns timestamps can be
off by a minute; default tolerance 0).
"""
import argparse
import json


def union(ivs):
    out = []
    for x, y in sorted(ivs):
        if out and x <= out[-1][1]:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    return out


def overlap(a0, a1, ivs):
    return sum(max(0.0, min(y, a1) - max(x, a0)) for x, y in ivs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_json")
    ap.add_argument("gt_json")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="credit coverage this many seconds away from GT")
    args = ap.parse_args()

    segs = json.load(open(args.scan_json))["segments"]
    gt = json.load(open(args.gt_json))
    scenes = [s for s in gt["scenes"]]
    cov = union([(s["start"], s["end"]) for s in segs])

    print(f"scan: {len(segs)} segments, "
          f"{sum(e - s for s, e in [(x['start'], x['end']) for x in segs]):.0f}s flagged")
    print(f"gt:   {len(scenes)} scenes, "
          f"{sum(s['end'] - s['start'] for s in scenes):.0f}s "
          f"(source: {gt.get('source', '?')})\n")

    tot = hit = 0
    for sc in scenes:
        a, b = sc["start"], sc["end"]
        if args.tolerance:
            covd = overlap(a - args.tolerance, b + args.tolerance, cov)
        else:
            covd = overlap(a, b, cov)
        tot += b - a
        hit += min(covd, b - a)
        pct = 100 * min(covd, b - a) / (b - a)
        mark = "MISS" if pct < 10 else ("partial" if pct < 80 else "OK")
        print(f"  [{a:7.1f}-{b:7.1f}] {sc['label']:20s} "
              f"{min(covd, b - a):5.0f}/{b - a:<5.0f}s {pct:5.1f}%  {mark}")

    print(f"\nGT recall: {hit:.0f}/{tot:.0f}s = {100 * hit / tot:.1f}%"
          + (f"  (tolerance +/-{args.tolerance:.0f}s)" if args.tolerance else ""))


if __name__ == "__main__":
    main()
