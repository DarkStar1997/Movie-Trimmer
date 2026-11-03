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
                       crowd_year: Optional[int] = None) -> None:
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
                crowd=crowd, crowd_title=crowd_title, crowd_year=crowd_year)
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


# --- blur jobs (stage C) ---------------------------------------------------------

blur_store: Dict[str, Dict[str, Any]] = {}
blur_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SCANS)


def new_blur_entry(video_name: str) -> str:
    blur_id = str(uuid.uuid4())
    blur_store[blur_id] = {
        "blur_id": blur_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "video": video_name,
        "output": None,
        "error": None,
    }
    return blur_id


def resolve_segments(segments_json: Any, scan_id: str | None) -> list[dict]:
    """Segments for a blur job: explicit JSON list, or a completed scan's.
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


async def run_blur_job(blur_id: str, video_path: str, segments: list[dict],
                        policy_overrides: Optional[dict],
                        window: Optional[tuple], cq: Optional[int]) -> None:
    entry = blur_store[blur_id]
    try:
        async with blur_semaphore:
            entry["status"] = "processing"
            from .blur import run_blur
            policy = Policy.from_overrides(policy_overrides)
            out_dir = Path(config.BLUR_OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"masked_{blur_id[:8]}.mp4"
            fn = functools.partial(
                run_blur, video_path, segments, str(out), policy=policy,
                window=window, cq=cq)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fn)
            entry.update(result)
            entry["status"] = "completed"
            logger.info("[blur %s] completed: %d frames, %d blurred, codec=%s",
                        blur_id, result["frames"], result["frames_blurred"],
                        result["codec"])
    except Exception as e:  # noqa: BLE001
        logger.exception("[blur %s] failed", blur_id)
        entry["status"] = "failed"
        entry["error"] = str(e)
