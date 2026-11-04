"""Scan pipeline: decode -> person gate -> VLM arbitration.

Stage A (person gate): RF-DETR presence pass on the decode stream ->
person_ranges the arbiter sweep intersects with. A frame with no person
cannot host explicit content, so the VLM never samples there. Fail-open:
a gate problem costs VLM time, never coverage.
Stage B (arbiter): trusted-source planning + person-gated sparse VLM
sweep; apply mode merges found segments (add-only).
Stage C (crowd, opt-in): community timestamps merged as trusted
segments. Stage E (guide, opt-in): parents-guide film context for the
arbiter's tie-break step.

NudeNet was removed 2026-11-05 (user decision): discovery is VLM-first,
validated across the three-movie corpus (scans/VLM_RUNS.md) — the
detector timeline, tracking, weak/orphan leads and policy evaluation
all went with it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .crowd import fetch_crowd_segments, merge_crowd_segments
from .guide import fetch_guide_context
from .arbiter import run_arbiter
from .decode import (analysis_size, open_decoder, probe_duration,
                     probe_dimensions)
from .persongate import PersonGateRunner
from .temporal import SmoothingConfig

logger = logging.getLogger(__name__)

# decode batch handed to the person gate per flush (the gate strides
# internally to PERSON_GATE_FPS; the batch is just feed granularity)
_GATE_BATCH = 16


def run_scan(
    video_path: str,
    *,
    smoothing: SmoothingConfig,
    sample_fps: float = config.DEFAULT_SAMPLE_FPS,
    arbiter: bool = config.DEFAULT_ARBITER,
    arbiter_model: str = config.DEFAULT_ARBITER_MODEL,
    arbiter_mode: str = config.DEFAULT_ARBITER_MODE,
    crowd: bool = config.DEFAULT_CROWD,
    crowd_title: str | None = None,
    crowd_year: int | None = None,
    guide: bool = config.DEFAULT_GUIDE,
) -> dict:
    """Scan one video; returns the result dict (see schemas.ScanResult).

    Discovery is the arbiter sweep (person-gated) + windows, plus crowd;
    the detector timeline fields (review_segments, timeline, label_stats)
    stay in the record as empty structures for consumer compatibility."""
    duration = probe_duration(video_path)
    src_w, src_h = probe_dimensions(video_path)
    aw, ah = analysis_size(src_w, src_h, config.DECODE_WIDTH)
    t0 = time.monotonic()

    # stage A: RF-DETR person gate — the arbiter's pre-filter
    # (fail-open: an engine problem costs VLM time, never coverage)
    gate_runner: Optional[PersonGateRunner] = None
    if arbiter and config.PERSON_GATE_ENABLED:
        try:
            gate_runner = PersonGateRunner(backend="trt")
        except Exception as e:  # noqa: BLE001
            logger.warning("[person-gate] init failed open: %s", e)

    decoder, eff_fps = open_decoder(
        video_path, fps=sample_fps, width=aw, height=ah)
    logger.info("[scan] %s: %.1fs @ %.2ffps effective, %dx%d (%.2f:1), "
                "gate-only", video_path, duration, eff_fps, aw, ah,
                aw / ah)
    gate_stride = max(1, round(eff_fps / max(config.PERSON_GATE_FPS, 0.1))) \
        if gate_runner is not None else 1

    batch: list[np.ndarray] = []

    def flush(frames: list[np.ndarray], first_idx: int) -> None:
        if gate_runner is not None:
            gate_runner.feed_batch(frames, first_idx, eff_fps, gate_stride)
            gate_runner.sync()

    try:
        with decoder:
            idx = 0
            for frame in decoder.frames():
                batch.append(frame)
                if len(batch) >= _GATE_BATCH:
                    flush(batch, idx)
                    idx += len(batch)
                    batch = []
            if batch:
                flush(batch, idx)
    finally:
        decoder.close()

    segments: list[dict] = []

    # person gate: join the side stream and smooth presence into ranges
    # (fail-open: runner errors return None ranges = whole timeline)
    if gate_runner is not None:
        person_ranges, gate_stats = gate_runner.finish(duration)
    else:
        person_ranges, gate_stats = None, {
            "enabled": bool(arbiter), "reason": "disabled" if not arbiter
            else "person gate off"}

    # per-stage walls (throughput instrumentation): decode_gate covers
    # scan start through decode + gate feeding and the gate-stream
    # join; crowd/guide/arbiter each measure their own span (arbiter
    # includes lead clustering + apply); finalize is the residual.
    t_stage = time.monotonic()
    decode_gate_s = t_stage - t0

    # stage C: crowd timestamps (CringeMDB, opt-in, fail-soft) — merged
    # BEFORE the arbiter: crowd ranges are TRUSTED (mask-final), so the
    # VLM must not spend frames inside them either. Union-merge only:
    # community ranges can never remove pipeline-found segments.
    crowd_block: Optional[dict] = None
    t_stage = time.monotonic()
    if crowd:
        crowd_block = fetch_crowd_segments(
            Path(video_path).name, title=crowd_title, year=crowd_year,
            categories=config.CROWD_CATEGORIES, pad_s=config.CROWD_PAD_S,
            cache_dir=Path(config.CROWD_CACHE_DIR))
        if crowd_block["segments"]:
            segments = merge_crowd_segments(segments, crowd_block)
    crowd_s = time.monotonic() - t_stage

    # stage E: parents-guide film-context (IMDb, opt-in, fail-soft) —
    # loaded BEFORE the arbiter so its Sex & Nudity items can serve as
    # STEP-2 tie-break background in the system prompt. A miss (blocked
    # fetch, no cache, no IMDb match) is recorded and the scan proceeds
    # with the byte-identical context-free prompt.
    guide_block: Optional[dict] = None
    t_stage = time.monotonic()
    if guide:
        guide_block = fetch_guide_context(
            Path(video_path).name, title=crowd_title, year=crowd_year)
    guide_s = time.monotonic() - t_stage

    # stage B: sparse VLM arbitration (opt-in; failure never fails the
    # scan)
    t_stage = time.monotonic()          # arbiter span incl. lead clustering
    arbiter_block: Optional[dict] = None
    if arbiter:
        arbiter_block = run_arbiter(
            video_path, duration, segments, smoothing=smoothing,
            model_id=arbiter_model, mode=arbiter_mode,
            person_ranges=person_ranges,
            film_context=(guide_block or {}).get("context_block") or "")
        # apply-mode is ADD-ONLY: VLM-found blind-spot segments merge in,
        # but trusted (crowd) segments are never removed on VLM
        # non-confirmation — measured Qwen3-VL behavior on dim scenes is
        # false negatives, and auto-removal would systematically eat them.
        if (arbiter_mode == "apply" and not arbiter_block.get("error")
                and arbiter_block["new_segments"]):
            pre = [dict(s) for s in segments]
            segments = _apply_arbiter(segments, arbiter_block)
            arbiter_block["segments_pre_arbiter"] = pre

    arbiter_s = time.monotonic() - t_stage
    wall = time.monotonic() - t0
    finalize_s = wall - decode_gate_s - crowd_s - guide_s - arbiter_s

    return {
        "duration": round(duration, 3),
        "sample_fps": eff_fps,
        "frames_scanned": 0,
        "segments": segments,
        # detector-era record fields, kept as empty structures for
        # consumer compatibility (review UIs, scripts)
        "review_segments": [],
        "timeline": [],
        "label_stats": {},
        "weak_trigger_hits": [],
        "tracking": {"enabled": False, "bridged_samples": 0},
        "smoothing": {"open_min_consecutive": smoothing.open_min_consecutive,
                      "close_min_consecutive": smoothing.close_min_consecutive,
                      "bridge_gap_s": smoothing.bridge_gap_s,
                      "drop_below_s": smoothing.drop_below_s,
                      "drop_rescue_score": smoothing.drop_rescue_score,
                      "pad_s": smoothing.pad_s},
        "arbiter": arbiter_block,
        "person_gate": gate_stats,
        "crowd": crowd_block,
        "guide": guide_block,
        "stages": {
            "decode_gate_s": round(decode_gate_s, 2),
            "crowd_s": round(crowd_s, 2) if crowd else None,
            "guide_s": round(guide_s, 2) if guide else None,
            "arbiter_s": round(arbiter_s, 2) if arbiter else None,
            "finalize_s": round(finalize_s, 2),
        },
        "wall_time_s": round(wall, 2),
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
