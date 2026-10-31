"""Scan pipeline: decode -> detect -> policy -> temporal smoothing.

Stage A (tracking, default on): person-cluster tracking with explicit-hold
bridging — sub-threshold flicker inside a scene keeps it flagged.
Stage B (arbiter, opt-in): sparse VLM verification + blind-gap sampling.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Optional

import numpy as np

from . import config
from .arbiter import run_arbiter
from .decode import (analysis_size, open_decoder, probe_duration,
                     probe_dimensions)
from .detector import NudeNetDetector
from .policy import Policy
from .temporal import SmoothingConfig, bin_timeline, smooth_segments
from .tracking import TrackBridge, cluster_persons

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """One live detector per (model, backend) — TRT engines deserialize in
    ~a second; scans reuse the loaded engine."""

    def __init__(self) -> None:
        self._detectors: dict[tuple, NudeNetDetector] = {}
        self._lock = threading.Lock()

    def get(self, model: str, backend: str, batch_size: int,
            fp32: bool) -> NudeNetDetector:
        key = (model, backend, batch_size, fp32)
        with self._lock:
            det = self._detectors.get(key)
            if det is None:
                det = NudeNetDetector(model, backend=backend,
                                      batch_size=batch_size, fp32=fp32)
                self._detectors[key] = det
            return det


registry = DetectorRegistry()


def run_scan(
    video_path: str,
    *,
    policy: Policy,
    smoothing: SmoothingConfig,
    sample_fps: float = config.DEFAULT_SAMPLE_FPS,
    model: str = None,
    backend: str = "auto",
    fp32: bool = False,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    track: bool = config.DEFAULT_TRACK,
    arbiter: bool = config.DEFAULT_ARBITER,
    arbiter_model: str = config.DEFAULT_ARBITER_MODEL,
    arbiter_mode: str = "verify",
) -> dict:
    """Scan one video; returns the result dict (see schemas.ScanResult)."""
    detector = registry.get(model or config.DEFAULT_MODEL, backend,
                            batch_size, fp32)
    duration = probe_duration(video_path)
    src_w, src_h = probe_dimensions(video_path)
    aw, ah = analysis_size(src_w, src_h, config.DECODE_WIDTH)
    t0 = time.monotonic()
    infer_s = 0.0

    ts: list[float] = []
    trig_flags: list[bool] = []
    trig_scores: list[float] = []
    trig_labels: list[dict] = []
    rev_flags: list[bool] = []
    rev_scores: list[float] = []
    rev_labels: list[dict] = []
    bridge_only: list[bool] = []
    label_stats: Counter = Counter()

    # stage A: tracker over person clusters at the sample cadence
    bridge = TrackBridge(
        dt=1.0 / sample_fps,
        max_lost_samples=config.TRACK_MAX_LOST_SAMPLES,
        explicit_hold_s=config.TRACK_EXPLICIT_HOLD_S,
        refresh_floor=config.TRACK_REFRESH_FLOOR) if track else None
    explicit_labels = set(policy.triggers)

    decoder, eff_fps = open_decoder(
        video_path, fps=sample_fps, width=aw, height=ah)
    if bridge is not None:
        bridge.dt = 1.0 / eff_fps
    logger.info("[scan] %s: %.1fs @ %.2ffps effective, %dx%d (%.2f:1), "
                "backend=%s, track=%s", video_path, duration, eff_fps, aw, ah,
                aw / ah, detector.backend_name, track)

    batch: list[np.ndarray] = []

    def flush(frames: list[np.ndarray], first_idx: int) -> None:
        nonlocal infer_s
        t_infer = time.monotonic()
        results = detector.detect_frames(frames)
        infer_s += time.monotonic() - t_infer
        for j, dets in enumerate(results):
            idx = first_idx + j
            t = idx / eff_fps
            verdict = policy.evaluate(dets, decoder.width, decoder.height)
            flagged = verdict.triggered
            score = verdict.max_trigger_score
            if bridge is not None:
                tb = bridge.step(t, cluster_persons(dets), explicit_labels)
                if tb["bridged"] and not flagged:
                    flagged = True
                    score = max(score, tb["score"])
                    bridge_only.append(True)
                else:
                    bridge_only.append(False)
            else:
                bridge_only.append(False)
            trig_flags.append(flagged)
            trig_scores.append(score)
            trig_labels.append(dict(verdict.trigger_labels))
            rev_flags.append(verdict.review)
            rev_scores.append(max(verdict.review_labels.values(), default=0.0))
            rev_labels.append(dict(verdict.review_labels))
            ts.append(t)
            for det in dets:
                label_stats[det.label] += 1

    try:
        with decoder:
            idx = 0
            for frame in decoder.frames():
                batch.append(frame)
                if len(batch) >= batch_size:
                    flush(batch, idx)
                    idx += len(batch)
                    batch = []
            if batch:
                flush(batch, idx)
    finally:
        decoder.close()

    segments = smooth_segments(trig_flags, ts, trig_scores, duration, smoothing)
    review_segments = smooth_segments(rev_flags, ts, rev_scores, duration,
                                      smoothing)

    def _annotate(segs: list[dict], label_dicts: list[dict]) -> None:
        """Attach per-segment label counts (sample hits per label inside the
        segment window) for review UIs."""
        for seg in segs:
            counts: dict[str, int] = {}
            for t, ld in zip(ts, label_dicts):
                if ld and seg["start"] <= t <= seg["end"]:
                    for lab in ld:
                        counts[lab] = counts.get(lab, 0) + 1
            seg["labels"] = dict(sorted(counts.items(),
                                        key=lambda kv: -kv[1]))

    _annotate(segments, trig_labels)
    _annotate(review_segments, rev_labels)
    for seg in segments:
        seg["bridged_samples"] = sum(
            1 for t, bo in zip(ts, bridge_only)
            if bo and seg["start"] <= t <= seg["end"])
    timeline = bin_timeline(trig_flags, ts, duration)

    # stage B: sparse VLM arbitration (opt-in; failure never fails the scan)
    arbiter_block: Optional[dict] = None
    if arbiter:
        arbiter_block = run_arbiter(
            video_path, duration, segments, smoothing=smoothing,
            model_id=arbiter_model, mode=arbiter_mode)
        # apply-mode is ADD-ONLY: VLM-found blind-spot segments merge in,
        # but detector-flagged segments are never removed on VLM
        # non-confirmation — measured Qwen3-VL behavior on dim scenes is
        # false negatives, and auto-removal would systematically eat them.
        if (arbiter_mode == "apply" and not arbiter_block.get("error")
                and arbiter_block["new_segments"]):
            pre = [dict(s) for s in segments]
            segments = _apply_arbiter(segments, arbiter_block)
            arbiter_block["segments_pre_arbiter"] = pre

    bridged_total = sum(bridge_only)
    wall = time.monotonic() - t0

    return {
        "duration": round(duration, 3),
        "model": detector.model_name,
        "backend": detector.backend_name,
        "sample_fps": eff_fps,
        "frames_scanned": len(ts),
        "segments": segments,
        "review_segments": review_segments,
        "timeline": timeline,
        "label_stats": dict(label_stats),
        "policy": {"triggers": policy.triggers, "review": policy.review,
                   "min_box_area_frac": policy.min_box_area_frac},
        "smoothing": {"open_min_consecutive": smoothing.open_min_consecutive,
                      "close_min_consecutive": smoothing.close_min_consecutive,
                      "bridge_gap_s": smoothing.bridge_gap_s,
                      "drop_below_s": smoothing.drop_below_s,
                      "drop_rescue_score": smoothing.drop_rescue_score,
                      "pad_s": smoothing.pad_s},
        "tracking": {
            "enabled": track,
            "explicit_hold_s": config.TRACK_EXPLICIT_HOLD_S,
            "refresh_floor": config.TRACK_REFRESH_FLOOR,
            "bridged_samples": bridged_total,
            "segments_with_bridging": sum(
                1 for s in segments if s.get("bridged_samples")),
        } if track else {"enabled": False, "bridged_samples": 0},
        "arbiter": arbiter_block,
        "wall_time_s": round(wall, 2),
        "infer_time_s": round(infer_s, 2),
    }


def _apply_arbiter(segments: list[dict], arb: dict) -> list[dict]:
    """apply-mode (add-only): merge in VLM-found segments, drop nothing."""
    kept = [dict(s) for s in segments]
    kept.extend(dict(s) for s in arb["new_segments"])
    kept.sort(key=lambda s: s["start"])
    merged: list[dict] = []
    for s in kept:
        if merged and s["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], s["end"])
            for lab, n in s.get("labels", {}).items():
                prev["labels"] = prev.get("labels") or {}
                prev["labels"][lab] = prev["labels"].get(lab, 0) + n
        else:
            merged.append(s)
    return merged
