"""Scan pipeline: decode -> detect -> policy -> temporal smoothing.

Stage A (tracking, default on): person-cluster tracking with explicit-hold
bridging — sub-threshold flicker inside a scene keeps it flagged.
Stage A2 (person gate, arbiter runs only): RF-DETR presence pass on the
same decode stream -> person_ranges the arbiter sweep intersects with.
Stage B (arbiter, opt-in): trusted-source planning + person-gated sparse
VLM sweep; apply mode merges found segments (add-only).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .crowd import fetch_crowd_segments, merge_crowd_segments
from .guide import fetch_guide_context
from .arbiter import run_arbiter
from .decode import (analysis_size, open_decoder, probe_duration,
                     probe_dimensions)
from .detector import NudeNetDetector
from .persongate import PersonGateRunner
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


def demote_bridge_only(segments: list[dict],
                       review_segments: list[dict]) -> tuple[list[dict],
                                                             list[dict]]:
    """Pure track-bridge segments (no sample carried a passing trigger
    label — only tracker-held sub-threshold flicker) are scan noise:
    NudeNet re-detection at mask time finds nothing in them (Gone Girl
    2026-11-03: 7 such segments, every one a 0-keyframe no-op render).
    Demote them to review so remove-class renders never see them; the
    TRACK_BRIDGE label records provenance for review UIs. Labeled bridge
    tails (real detections + held flicker) are unaffected."""
    noise = [s for s in segments if not s.get("labels")]
    if not noise:
        return segments, review_segments
    for s in noise:
        s["labels"] = {"TRACK_BRIDGE": s.get("bridged_samples", 0)}
    ids = {id(s) for s in noise}
    kept = [s for s in segments if id(s) not in ids]
    merged = sorted(review_segments + noise, key=lambda s: s["start"])
    logger.info("[scan] %d pure-bridge segment(s) demoted to review "
                "(track-held flicker, no passing label)", len(noise))
    return kept, merged


def run_scan(
    video_path: str,
    *,
    policy: Policy,
    smoothing: SmoothingConfig,
    sample_fps: float = config.DEFAULT_SAMPLE_FPS,
    model: str = None,
    backend: str = "trt",
    fp32: bool = False,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    track: bool = config.DEFAULT_TRACK,
    arbiter: bool = config.DEFAULT_ARBITER,
    arbiter_model: str = config.DEFAULT_ARBITER_MODEL,
    arbiter_mode: str = config.DEFAULT_ARBITER_MODE,
    crowd: bool = config.DEFAULT_CROWD,
    crowd_title: str | None = None,
    crowd_year: int | None = None,
    guide: bool = config.DEFAULT_GUIDE,
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
    weak_raw: list[tuple[float, float, str]] = []   # (t, score, label)
    proxy_pts: list[float] = []     # NudeNet proxy-class hits (gate evidence)

    # stage A2: RF-DETR person gate (only needed when the arbiter runs;
    # fail-open — an engine problem costs VLM time, never coverage)
    gate_runner: Optional[PersonGateRunner] = None
    if arbiter and config.PERSON_GATE_ENABLED:
        try:
            gate_runner = PersonGateRunner(backend="trt")
        except Exception as e:  # noqa: BLE001
            logger.warning("[person-gate] init failed open: %s", e)

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
    gate_stride = max(1, round(eff_fps / max(config.PERSON_GATE_FPS, 0.1))) \
        if gate_runner is not None else 1

    batch: list[np.ndarray] = []

    def flush(frames: list[np.ndarray], first_idx: int) -> None:
        nonlocal infer_s
        t_infer = time.monotonic()
        results = detector.detect_frames(frames)
        infer_s += time.monotonic() - t_infer
        if gate_runner is not None:
            gate_runner.feed_batch(frames, first_idx, eff_fps, gate_stride)
            gate_runner.sync()
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
            weak = policy.weak_triggers(dets, decoder.width, decoder.height,
                                         config.ARBITER_WEAK_MIN_SCORE)
            if weak:
                top_label = max(weak, key=weak.get)
                weak_raw.append((t, weak[top_label], top_label))
            # person-gate proxy evidence: a NudeNet body-part detection
            # (any non-trigger class) means somebody is there
            if any(det.label in config.PERSON_GATE_PROXY_LABELS
                   and det.score >= config.PERSON_GATE_PROXY_SCORE
                   for det in dets):
                proxy_pts.append(round(t, 3))
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
    segments, review_segments = demote_bridge_only(segments,
                                                   review_segments)
    timeline = bin_timeline(trig_flags, ts, duration)

    # person gate: join the side stream and smooth presence into ranges
    # (fail-open: runner errors return None ranges = whole timeline)
    if gate_runner is not None:
        person_ranges, gate_stats = gate_runner.finish(duration)
    else:
        person_ranges, gate_stats = None, {
            "enabled": bool(arbiter), "reason": "disabled" if not arbiter
            else "person gate off"}

    # stage C: crowd timestamps (CringeMDB, opt-in, fail-soft) — merged
    # BEFORE the arbiter: crowd ranges are TRUSTED (mask-final), so the
    # VLM must not spend frames inside them either. Union-merge only:
    # community ranges can never remove model-found segments.
    crowd_block: Optional[dict] = None
    if crowd:
        crowd_block = fetch_crowd_segments(
            Path(video_path).name, title=crowd_title, year=crowd_year,
            categories=config.CROWD_CATEGORIES, pad_s=config.CROWD_PAD_S,
            cache_dir=Path(config.CROWD_CACHE_DIR))
        if crowd_block["segments"]:
            segments = merge_crowd_segments(segments, crowd_block)

    # stage E: parents-guide film-context (IMDb, opt-in, fail-soft) —
    # loaded BEFORE the arbiter so its Sex & Nudity items can serve as
    # STEP-2 tie-break background in the system prompt. A miss (blocked
    # fetch, no cache, no IMDb match) is recorded and the scan proceeds
    # with the byte-identical context-free prompt.
    guide_block: Optional[dict] = None
    if guide:
        guide_block = fetch_guide_context(
            Path(video_path).name, title=crowd_title, year=crowd_year)

    # stage B: sparse VLM arbitration (opt-in; failure never fails the scan)
    # weak hits: cluster sub-threshold trigger bursts (<=2s gap) to their
    # peak sample, capped by striding — each peak is one targeted VLM frame.
    # Orphaned flagged samples (>= threshold but smoothing dropped the run)
    # join them at score 1.0: a detector-strong single frame is the best
    # blind-spot lead there is (Ex Machina GT: the 0.3s flash at 6fps).
    weak_raw += [(t, 1.0, "ORPHAN_TRIGGER")
                 for t in _orphan_flagged(ts, trig_flags, segments)]
    orphans = [t for t, s, lab in weak_raw if lab == "ORPHAN_TRIGGER"]
    weak_peaks = _cluster_weak(weak_raw, config.ARBITER_WEAK_MAX_FRAMES)
    # +-0.5s companions for orphan samples only: weak-band peaks are
    # VLM-rejected as a rule (measured 15/15), orphans are detector-
    # confirmed sub-second flashes whose sample can land at the edge of
    # the visually-explicit window
    companions = sorted({round(min(max(t + d, 0.0), duration), 3)
                         for t in orphans for d in (-0.5, 0.5)
                         if 0.0 <= t + d <= duration})
    arbiter_block: Optional[dict] = None
    if arbiter:
        arbiter_block = run_arbiter(
            video_path, duration, segments, smoothing=smoothing,
            model_id=arbiter_model, mode=arbiter_mode,
            weak_hits=[t for t, _, _ in weak_peaks],
            companion_hits=companions,
            person_ranges=person_ranges, person_points=proxy_pts,
            film_context=(guide_block or {}).get("context_block") or "")
        # apply-mode is ADD-ONLY: VLM-found blind-spot segments merge in,
        # but trusted (detector/crowd) segments are never removed on VLM
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
        "weak_trigger_hits": [{"t": round(t, 3), "score": round(s, 3),
                               "label": lab}
                              for t, s, lab in weak_peaks],
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
        "person_gate": gate_stats,
        "crowd": crowd_block,
        "guide": guide_block,
        "wall_time_s": round(wall, 2),
        "infer_time_s": round(infer_s, 2),
    }


def _orphan_flagged(ts: list[float], flags: list[bool],
                    segments: list[dict]) -> list[float]:
    """Flagged sample times that no final segment covers — the detector
    fired at/above trigger threshold there, but smoothing discarded the
    run (single-frame hits, sub-1s blips below the rescue line). Pure."""
    ivs = [(s["start"], s["end"]) for s in segments]
    return [t for t, f in zip(ts, flags)
            if f and not any(a <= t <= b for a, b in ivs)]


def _cluster_weak(raw: list[tuple[float, float, str]],
                  max_keep: int) -> list[tuple[float, float, str]]:
    """(t, score, label) samples -> one peak per <=2s burst, thinned to
    max_keep. Pure; sorted output."""
    clusters: list[list[tuple[float, float, str]]] = []
    for t, s, lab in sorted(raw):
        if clusters and t - clusters[-1][-1][0] <= 2.0:
            clusters[-1].append((t, s, lab))
        else:
            clusters.append([(t, s, lab)])
    peaks = [max(c, key=lambda x: x[1]) for c in clusters]
    if len(peaks) > max_keep > 0:
        step = len(peaks) / max_keep
        peaks = [peaks[int(i * step)] for i in range(max_keep)]
    return peaks


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
