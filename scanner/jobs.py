"""Scan job store and background worker (single-process, mirroring the
trimmer's main.py job pattern; run exactly one uvicorn worker)."""

from __future__ import annotations

import asyncio
import functools
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


async def run_scan_job(scan_id: str, video_path: str,
                       policy_overrides: Optional[dict],
                       smoothing_overrides: Optional[dict],
                       sample_fps: float, model: Optional[str],
                       backend: str, fp32: bool,
                       workspace: Optional[Path] = None, *,
                       track: bool = True, arbiter: bool = False,
                       arbiter_model: Optional[str] = None,
                       arbiter_mode: str = "verify") -> None:
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
                arbiter_model=arbiter_model, arbiter_mode=arbiter_mode)
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
