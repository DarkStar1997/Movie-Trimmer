"""Temporal smoothing: per-sample flags -> cut segments.

Pipeline (defaults in config.DEFAULT_SMOOTHING):
  1. hysteresis — a flagged run must reach `open_min_consecutive` samples to
     open a segment (shorter blips never open); once open, it closes only
     after `close_min_consecutive` consecutive clean samples (shorter gaps
     keep it open, i.e. runs separated by less merge together).
  2. bridge     — merge two segments separated by a gap <= `bridge_gap_s`.
  3. drop       — discard segments shorter than `drop_below_s` unless their
     mean trigger score exceeds `drop_rescue_score`.
  4. pad        — extend both ends by `pad_s`, clipped to [0, duration].
  5. merge      — collapse overlaps (padding can push segments into each
     other) and return sorted, non-overlapping segments.

Sample timestamps are the decode clock (frame k at k / sample_fps), so a
segment boundary has +/- half-sample accuracy; padding absorbs that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class SmoothingConfig:
    open_min_consecutive: int = config.DEFAULT_SMOOTHING["open_min_consecutive"]
    close_min_consecutive: int = config.DEFAULT_SMOOTHING["close_min_consecutive"]
    bridge_gap_s: float = config.DEFAULT_SMOOTHING["bridge_gap_s"]
    drop_below_s: float = config.DEFAULT_SMOOTHING["drop_below_s"]
    drop_rescue_score: float = config.DEFAULT_SMOOTHING["drop_rescue_score"]
    pad_s: float = config.DEFAULT_SMOOTHING["pad_s"]

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "SmoothingConfig":
        if not overrides:
            return cls()
        known = set(config.DEFAULT_SMOOTHING)
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown smoothing keys: {sorted(unknown)}")
        return cls(**{**config.DEFAULT_SMOOTHING, **overrides})


def _flagged_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Maximal consecutive flagged index runs, inclusive."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _hysteresis(flags: list[bool], ts: np.ndarray, scores: np.ndarray,
                cfg: SmoothingConfig) -> list[tuple[float, float, float]]:
    """Runs -> segments [(start_s, end_s, mean_score)] after open/close rules."""
    runs = [r for r in _flagged_runs(flags)
            if (r[1] - r[0] + 1) >= cfg.open_min_consecutive]
    if not runs:
        return []
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.25

    # close rule: merge runs whose clean gap is shorter than the threshold
    merged = [runs[0]]
    for r in runs[1:]:
        ps, pe = merged[-1]
        if (r[0] - pe - 1) < cfg.close_min_consecutive:
            merged[-1] = (ps, r[1])
        else:
            merged.append(r)

    segments = []
    for s_idx, e_idx in merged:
        start = float(ts[s_idx])
        end = float(ts[e_idx]) + dt  # flagged material spans to next sample
        mean_score = float(scores[s_idx:e_idx + 1].mean()) if e_idx >= s_idx else 0.0
        segments.append((start, end, mean_score))
    return segments


def smooth_segments(flags: list[bool], ts: list[float], scores: list[float],
                    duration: float,
                    cfg: SmoothingConfig | None = None) -> list[dict]:
    """Full smoothing chain. Returns [{'start', 'end', 'mean_score'}] in
    seconds, sorted and non-overlapping."""
    cfg = cfg or SmoothingConfig()
    ts_arr = np.asarray(ts, dtype=float)
    sc_arr = np.asarray(scores, dtype=float)
    if len(flags) != len(ts_arr) or len(flags) != len(sc_arr):
        raise ValueError("flags/ts/scores length mismatch")

    segs = _hysteresis(flags, ts_arr, sc_arr, cfg)
    if not segs:
        return []

    # bridge gaps
    bridged: list[tuple[float, float, float]] = [segs[0]]
    for s, e, m in segs[1:]:
        ps, pe, pm = bridged[-1]
        if s - pe <= cfg.bridge_gap_s:
            bridged[-1] = (ps, e, (pm + m) / 2.0)
        else:
            bridged.append((s, e, m))

    # drop blips (unless high-confidence)
    kept = [(s, e, m) for s, e, m in bridged
            if (e - s) >= cfg.drop_below_s or m >= cfg.drop_rescue_score]

    # pad + clip
    out: list[dict] = []
    for s, e, m in kept:
        s = max(0.0, s - cfg.pad_s)
        e = min(duration, e + cfg.pad_s)
        if e > s:
            out.append({"start": round(s, 3), "end": round(e, 3),
                        "mean_score": round(m, 4)})

    # merge overlaps/touches introduced by padding
    merged_out: list[dict] = []
    for seg in sorted(out, key=lambda d: d["start"]):
        if merged_out and seg["start"] <= merged_out[-1]["end"]:
            prev = merged_out[-1]
            prev["end"] = max(prev["end"], seg["end"])
            prev["mean_score"] = max(prev["mean_score"], seg["mean_score"])
        else:
            merged_out.append(seg)
    return merged_out


def bin_timeline(flags: list[bool], ts: list[float],
                 duration: float) -> list[dict]:
    """Per-second bins for UI timelines: count of flagged samples."""
    n_bins = max(1, int(np.ceil(duration)))
    counts = [0] * n_bins
    for flagged, t in zip(flags, ts):
        if flagged:
            b = min(n_bins - 1, int(t))
            counts[b] += 1
    return [{"t": i, "hits": c} for i, c in enumerate(counts)]
