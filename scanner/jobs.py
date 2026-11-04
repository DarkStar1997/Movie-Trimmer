"""Scan job store and background worker (single-process, mirroring the
trimmer's main.py job pattern; run exactly one uvicorn worker)."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .pipeline import run_scan
from .policy import Policy
from .temporal import SmoothingConfig

logger = logging.getLogger(__name__)

scan_store: Dict[str, Dict[str, Any]] = {}
scan_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SCANS)


def new_scan_entry(video_name: str) -> str:
    scan_id = str(uuid.uuid4())
    scan_store[scan_id] = {
        "scan_id": scan_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "video": video_name,
        "segments": [],
        "review_segments": [],
        "timeline": [],
        "label_stats": {},
        "error": None,
    }
    return scan_id


async def run_scan_job(scan_id: str, video_path: str,                       policy_overrides: Optional[dict],
                       smoothing_overrides: Optional[dict],
                       sample_fps: float, model: Optional[str],
                       backend: str, fp32: bool,
                       workspace: Optional[Path] = None, *,
                       track: bool = True, arbiter: bool = False,
                       arbiter_model: Optional[str] = None,
                       arbiter_mode: str = config.DEFAULT_ARBITER_MODE,
                       crowd: bool = False,
                       crowd_title: Optional[str] = None,
                       crowd_year: Optional[int] = None,
                       guide: bool = False,
                       detector_enabled: bool = True) -> None:
    """Background task: run the scan off the event loop, store the result."""
    entry = scan_store[scan_id]
    try:
        async with scan_semaphore:
            entry["status"] = "processing"
            policy = Policy.from_overrides(policy_overrides)
            smoothing = SmoothingConfig.from_overrides(smoothing_overrides)
            fn = functools.partial(
                run_scan, video_path,
                policy=policy, smoothing=smoothing,
                sample_fps=sample_fps, model=model,
                backend=backend, fp32=fp32,
                track=track, arbiter=arbiter,
                arbiter_model=arbiter_model, arbiter_mode=arbiter_mode,
                crowd=crowd, crowd_title=crowd_title, crowd_year=crowd_year,
                guide=guide, detector_enabled=detector_enabled)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fn)

            entry.update(result)
            entry["status"] = "completed"
            logger.info("[scan %s] completed: %d cut segments, %d review, "
                        "%.1fx realtime", scan_id, len(result["segments"]),
                        len(result["review_segments"]),
                        result["duration"] / max(result["wall_time_s"], 0.01))
    except Exception as e:  # noqa: BLE001
        logger.exception("[scan %s] failed", scan_id)
        entry["status"] = "failed"
        entry["error"] = str(e)
    finally:
        if workspace and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


# --- mask jobs (stage C) ---------------------------------------------------------

mask_store: Dict[str, Dict[str, Any]] = {}
mask_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SCANS)


def new_mask_entry(video_name: str) -> str:
    mask_id = str(uuid.uuid4())
    mask_store[mask_id] = {
        "mask_id": mask_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "video": video_name,
        "output": None,
        "error": None,
    }
    return mask_id


def safe_output_name(name: str) -> str:
    """Sanitize a user-supplied output filename to a bare .mp4 basename;
    raises ValueError on anything that escapes the output dir."""
    n = Path(name).name.strip()
    if not n or n.startswith(".") or n in {".", ".."}:
        raise ValueError(f"invalid output_name: {name!r}")
    if not n.endswith(".mp4"):
        n += ".mp4"
    return n


def resolve_segments(segments_json: Any, scan_id: str | None) -> list[dict]:
    """Segments for a mask job: explicit JSON list, or a completed scan's.
    JSON items may be [start, end] pairs or full segment dicts (labels and
    mean_score are preserved for the VLM-segment masking path)."""
    if scan_id is not None:
        if scan_id not in scan_store:
            raise ValueError(f"scan id not found: {scan_id}")
        entry = scan_store[scan_id]
        if entry["status"] != "completed":
            raise ValueError(f"scan {scan_id} is {entry['status']}, not completed")
        return [dict(s) for s in entry["segments"]]
    if isinstance(segments_json, str):
        segments_json = json.loads(segments_json)
    if not isinstance(segments_json, (list, tuple)) or not segments_json:
        raise ValueError("segments must be a non-empty [[start, end], ...]")
    out = []
    for seg in segments_json:
        if isinstance(seg, dict):
            # full segment dicts (a saved scan's `segments` array): keep
            # labels/mean_score — 'arbiter' in labels marks VLM-found
            # segments that stage C must mask without NudeNet boxes
            a, b = float(seg["start"]), float(seg["end"])
        else:
            a, b = float(seg[0]), float(seg[1])
        if not (0.0 <= a < b):
            raise ValueError(f"invalid segment [{a}, {b}]")
        entry = {"start": a, "end": b}
        if isinstance(seg, dict):
            for k in ("labels", "mean_score"):
                if k in seg:
                    entry[k] = seg[k]
        out.append(entry)
    return out


async def run_mask_job(mask_id: str, video_path: str, segments: list[dict],
                        policy_overrides: Optional[dict],
                        window: Optional[tuple], cq: Optional[int],
                        output_name: Optional[str] = None,
                        bake_chapters: bool = True,
                        segmenter: Optional[str] = None) -> None:
    entry = mask_store[mask_id]
    entry["video_path"] = video_path
    entry["segments"] = segments
    if segmenter:
        entry["segmenter"] = segmenter
    try:
        async with mask_semaphore:
            entry["status"] = "processing"
            from .mask import run_mask
            policy = Policy.from_overrides(policy_overrides)
            out_dir = Path(config.MASK_OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / (output_name or f"masked_{mask_id[:8]}.mp4")
            entry["output"] = str(out)
            fn = functools.partial(
                run_mask, video_path, segments, str(out), policy=policy,
                window=window, cq=cq,
                segmenter=segmenter or config.MASK_SEGMENTER)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fn)
            entry.update(result)
            if bake_chapters:
                if window is None:
                    # chapters are full-movie-timeline metadata; window
                    # renders have an offset timeline and would get
                    # garbage chapter times
                    from .chapters import bake
                    entry["chapters"] = await loop.run_in_executor(
                        None, bake, str(out), segments)
                else:
                    entry["chapters"] = {"note": "window render — "
                                     "chapters skipped (offset timeline)"}
            entry["status"] = "completed"
            logger.info("[mask %s] completed: %d frames, %d masked, "
                        "codec=%s%s", mask_id, result["frames"],
                        result["frames_masked"], result["codec"],
                        (f", {entry['chapters']['mask_chapters']} "
                         "chapters" if "mask_chapters" in entry.get(
                             "chapters", {}) else ""))
    except Exception as e:  # noqa: BLE001
        logger.exception("[mask %s] failed", mask_id)
        entry["status"] = "failed"
        entry["error"] = str(e)


# --- verify jobs (render gate) ---------------------------------------------------

verify_store: Dict[str, Dict[str, Any]] = {}
verify_semaphore = asyncio.Semaphore(1)   # CPU/ffmpeg heavy: serialize


def new_verify_entry(source: str, output: str) -> str:
    verify_id = str(uuid.uuid4())
    verify_store[verify_id] = {
        "verify_id": verify_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "source": source,
        "output": output,
        "all_pass": None,
        "checks": [],
        "error": None,
    }
    return verify_id


async def run_verify_job(verify_id: str, source: str, output: str,
                         segments: list[dict],
                         masked_i: Optional[list[int]],
                         clean_t: Optional[list[float]],
                         k: int = 6) -> None:
    entry = verify_store[verify_id]
    try:
        async with verify_semaphore:
            entry["status"] = "processing"
            from .verify import run_verification
            fn = functools.partial(run_verification, source, output,
                                   segments, masked_i, clean_t, k)
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(None, fn)
            entry.update(report)
            entry["status"] = "completed"
            n_pass = sum(1 for c in report["checks"] if c["pass"])
            logger.info("[verify %s] %s: %d/%d checks passed", verify_id,
                        "ALL PASS" if report["all_pass"] else "FAILURES",
                        n_pass, len(report["checks"]))
    except Exception as e:  # noqa: BLE001
        logger.exception("[verify %s] failed", verify_id)
        entry["status"] = "failed"
        entry["error"] = str(e)


# --- chapter jobs ----------------------------------------------------------------

chapters_store: Dict[str, Dict[str, Any]] = {}
chapters_semaphore = asyncio.Semaphore(1)


def new_chapters_entry(output: str) -> str:
    chapters_id = str(uuid.uuid4())
    chapters_store[chapters_id] = {
        "chapters_id": chapters_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "output": output,
        "error": None,
    }
    return chapters_id


async def run_chapters_job(chapters_id: str, output: str,
                           segments: list[dict]) -> None:
    entry = chapters_store[chapters_id]
    try:
        async with chapters_semaphore:
            entry["status"] = "processing"
            from .chapters import bake
            loop = asyncio.get_event_loop()
            entry.update(await loop.run_in_executor(
                None, bake, output, segments))
            entry["status"] = "completed"
    except Exception as e:  # noqa: BLE001
        logger.exception("[chapters %s] failed", chapters_id)
        entry["status"] = "failed"
        entry["error"] = str(e)
