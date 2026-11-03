"""Score all-frames detector runs against the validated segment ground truth.

Host-side, numpy only. Run from the repo root.

GT structure (learned the hard way): the scan JSON's final segments are the
union of (a) detector-origin segments — sparse NudeNet hits at 4 fps, padded
and merged — and (b) arbiter-only segments the Qwen3-VL 30B added on top
(GG: 30/38, SB: 52/76 — the detector stack genuinely misses most of the
dark/stylized explicit content; that is why the arbiter exists). So:

  * frame-level recall vs the padded union is meaningless as a headline.
  * the replacement question is SEGMENT-level: can a candidate replace
    NudeNet as first stage = fire inside detector-origin segments.
  * firing inside arbiter-only segments is a bonus: content the cheap
    detector could have caught without the 30B arbiter.
  * false alarms are counted outside the full union (definitely bad).

Usage: python3 scripts/eval/score_candidates.py
Writes evals/scores.json + evals/scores.md
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

FPS = 24.0
MOVIES = {
    "gg": ("evals/gg/allframes.npz", "scans/gone_girl_30b_allinone.json"),
    "sb": ("evals/sb/allframes.npz", "scans/saltburn_2023_allinone.json"),
}

# F1-optimal search ranges (model-aware: fp32 EraX cls scores run low)
SWEEP = {
    "nudenet": np.arange(0.10, 0.80, 0.05),
    "nsfw2mini": np.arange(0.10, 0.80, 0.05),
    "eraxn": np.arange(0.02, 0.45, 0.03),
    "eraxs": np.arange(0.02, 0.45, 0.03),
    "eraxm": np.arange(0.02, 0.45, 0.03),
}
MODELS = ("nudenet", "nsfw2mini", "eraxn", "eraxs", "eraxm")

# Qwen3-VL-30B arbiter reference (measured on the baseline scans; its
# accept/reject decisions DEFINE part of the GT, so it is a reference
# row, not a scored competitor — and it only ever saw detector-flagged
# frames, not every frame).
VLM_REFERENCE = {
    "per_movie": {"gg": {"judged_frames": 1196, "new_segments": 28,
                         "refusals": 1},
                  "sb": {"judged_frames": 2239, "new_segments": 54,
                         "refusals": 31}},
    "s_per_judged_frame": 0.054, "effective_fps": 18.5,
}


def load_gt(scan_path, n_frames):
    """Three segment provenances live in the scan JSON:
    anatomy-label (NudeNet found them), external metadata priors
    (e.g. cringemdb:sex), and arbiter-only (Qwen3-VL additions)."""
    from scanner.config import DEFAULT_TRIGGER_LABELS
    anatomy = set(DEFAULT_TRIGGER_LABELS)
    segs = json.loads(pathlib.Path(scan_path).read_text())["segments"]
    det_idx, ext_idx, arb_idx = [], [], []
    union = np.zeros(n_frames, dtype=bool)
    for s in segs:
        a, b = int(s["start"] * FPS), min(n_frames, int(np.ceil(s["end"] * FPS)))
        union[a:b] = True
        labels = set(s["labels"])
        if labels & anatomy:
            det_idx.append((a, b))
        elif labels - {"arbiter"}:
            ext_idx.append((a, b))          # external metadata prior
        else:
            arb_idx.append((a, b))
    return det_idx, ext_idx, arb_idx, union


def frame_scores(npz, model):
    if model == "nudenet":
        return npz["nudenet_max"]
    if model == "nsfw2mini":
        p = npz["nsfw2_probs"]  # [N,5]: safe, hentai, porn, sexy, drawing
        return np.maximum(p[:, 1], p[:, 2])          # explicit = porn|hentai
    return npz[f"{model}_max"].max(axis=1)           # best of 5 anatomy classes


def seg_recall(pred, segs_idx):
    if not segs_idx:
        return 0.0
    return sum(1 for a, b in segs_idx if pred[a:b].any()) / len(segs_idx)


def metrics(sc, t, det_idx, ext_idx, arb_idx, union):
    pred = sc >= t
    fp_frames = int((pred & ~union).sum())
    return {
        "threshold": round(float(t), 3),
        "seg_recall_det": round(seg_recall(pred, det_idx), 3),
        "seg_recall_ext": round(seg_recall(pred, ext_idx), 3),
        "seg_recall_arb": round(seg_recall(pred, arb_idx), 3),
        "false_alarm_min": round(fp_frames / FPS / 60, 2),
        "hot_frames": int(pred.sum()),
    }


def main():
    out = {"movies": {}, "models": {}, "vlm_reference": VLM_REFERENCE}
    for tag, (npz_path, scan_path) in MOVIES.items():
        npz = np.load(npz_path)
        meta = json.loads(pathlib.Path(npz_path).parent.joinpath("meta.json")
                          .read_text())
        n = len(npz["nudenet_max"])
        det_idx, ext_idx, arb_idx, union = load_gt(scan_path, n)
        out["movies"][tag] = {
            "frames": n,
            "segments": len(det_idx) + len(ext_idx) + len(arb_idx),
            "det_origin_segments": len(det_idx),
            "metadata_segments": len(ext_idx),
            "arbiter_only_segments": len(arb_idx),
            "gt_seconds": round(float(union.sum()) / FPS, 1),
            "inference_ms_per_frame": meta["inference_ms_per_frame"],
            "inference_fps": meta["effective_inference_fps"],
            "decode_wall_s": round(meta["decode_wall_s"], 1),
        }
        for model in MODELS:
            sc = frame_scores(npz, model)
            rows = [metrics(sc, t, det_idx, ext_idx, arb_idx, union)
                    for t in SWEEP[model]]
            # operating point: highest det-seg-recall whose fp <= 2 min;
            # tie-break on arbiter coverage then lower threshold
            ok = [r for r in rows if r["false_alarm_min"] <= 2.0] or rows
            best = max(ok, key=lambda r: (r["seg_recall_det"],
                                          r["seg_recall_arb"],
                                          -r["threshold"]))
            out["models"].setdefault(model, {})[tag] = {
                "operating": best, "sweep": rows}

    pathlib.Path("evals/scores.json").write_text(json.dumps(out, indent=1))

    lines = ["# candidate detector scores (vs validated final segments)", ""]
    v = VLM_REFERENCE
    lines.append("## VLM arbiter reference (Qwen3-VL-30B; defines part of GT)")
    lines.append(f"- {v['s_per_judged_frame']} s/judged frame "
                 f"({v['effective_fps']} fps effective); on-demand only")
    for tag, r in v["per_movie"].items():
        lines.append(f"- {tag}: {r['judged_frames']} judged frames, "
                     f"{r['new_segments']} new segments, {r['refusals']} refusals")
    lines.append("")
    for tag, m in out["movies"].items():
        lines.append(f"## {tag}: {m['frames']} frames | {m['det_origin_segments']} det"
                     f"+{m['metadata_segments']} meta"
                     f"+{m['arbiter_only_segments']} arb segs "
                     f"({m['gt_seconds']}s GT)")
        lines.append("ms/frame: " + ", ".join(
            f"{k}={v:.2f}" for k, v in m["inference_ms_per_frame"].items()))
        lines.append("")
        lines.append("| model | thr | det-seg-R | meta-seg-R | arb-seg-R | fp min |")
        lines.append("|---|---|---|---|---|---|")
        for model in MODELS:
            b = out["models"][model][tag]["operating"]
            lines.append(f"| {model} | {b['threshold']} | {b['seg_recall_det']:.2f} "
                         f"| {b['seg_recall_ext']:.2f} | {b['seg_recall_arb']:.2f} "
                         f"| {b['false_alarm_min']} |")
        lines.append("")
    pathlib.Path("evals/scores.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
