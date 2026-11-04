"""Scanner FastAPI service.

Endpoints:
  POST /scan       multipart `video` file OR `video_path` (volume-mounted),
                   plus optional JSON form fields `policy`, `smoothing`,
                   `sample_fps`, `model`, `backend`, arbiter fields, and
                   crowd fields (`crowd`, `crowd_title`, `crowd_year` —
                   CringeMDB scene-guide timestamps as first-class
                   segments, fail-soft)
  GET  /scan/{id}  status + result (segments in seconds)
  GET  /scans      list
  POST /mask       mask-precision mask render (stage C): segments from a
                   completed scan_id or explicit JSON; optional preview
                   window, output_name, and automatic chapter bake;
                   NVDEC decode, NVENC encode with explicit CQ
  GET  /mask/{id}  status + output path
  GET  /masks      list
  POST /verify     render gate: source vs render (container, decode,
                   timeline, audio md5, mask presence/absence spot
                   checks); spots auto-derived from segment provenance
  GET  /verify/{id}  status + report
  GET  /verifies   list
  POST /chapters   bake mask-region chapters into a render in place
                   (stream copy; also automatic on full mask renders)
  GET  /chapters/{id}  status + summary
  GET  /health     gpu/engine/model diagnostics

Run single-worker only (in-memory job store):
  uvicorn scanner.main:app --host 0.0.0.0 --port 8081
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import config
from .jobs import (chapters_store, mask_store, new_chapters_entry,
                   new_mask_entry, new_scan_entry, new_verify_entry,
                   resolve_segments, run_chapters_job, run_mask_job,
                   run_scan_job, run_verify_job, safe_output_name,
                   scan_store, verify_store)
from .pipeline import registry
from .schemas import ScanResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("nsfw_scanner")

app = FastAPI(
    title="NSFW Scanner for Movie-Trimmer",
    description="Scans a movie for nudity/explicit content and emits "
                "remove-segments for the trimmer.",
    version="0.1.0")


def _parse_json_field(raw: str | None, field_name: str) -> dict | None:
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"{field_name}: invalid JSON: {e}")
    if not isinstance(data, dict):
        raise HTTPException(400, f"{field_name} must be a JSON object")
    return data


@app.post("/scan", status_code=202)
async def submit_scan(
    background_tasks: BackgroundTasks,
    video: UploadFile | None = File(None, description="Video file"),
    video_path: str | None = Form(None, description="Path to a mounted video"),
    policy: str | None = Form(None, description="JSON policy overrides"),
    smoothing: str | None = Form(None, description="JSON smoothing overrides"),
    sample_fps: float = Form(config.DEFAULT_SAMPLE_FPS),
    model: str = Form(config.DEFAULT_MODEL),
    backend: str = Form("trt"),
    fp32: bool = Form(False),
    track: bool = Form(config.DEFAULT_TRACK),
    arbiter: bool = Form(config.DEFAULT_ARBITER),
    arbiter_model: str = Form(config.DEFAULT_ARBITER_MODEL),
    arbiter_mode: str = Form(config.DEFAULT_ARBITER_MODE),
    crowd: bool = Form(config.DEFAULT_CROWD),
    crowd_title: str | None = Form(None),
    crowd_year: int | None = Form(None),
    guide: bool = Form(config.DEFAULT_GUIDE),
    detector: bool = Form(True,
                          description="false = NudeNet-free scan: person "
                                      "gate + arbiter sweep + crowd only"),
):
    if model not in config.MODELS:
        raise HTTPException(400, f"unknown model {model!r}")
    if not (0.2 <= sample_fps <= 12.0):
        raise HTTPException(400, "sample_fps must be in [0.2, 12]")
    if arbiter_mode not in config.ARBITER_MODES:
        raise HTTPException(400, f"arbiter_mode must be one of "
                                 f"{config.ARBITER_MODES}")
    policy_overrides = _parse_json_field(policy, "policy")
    smoothing_overrides = _parse_json_field(smoothing, "smoothing")
    # validate early so bad config 400s instead of failing the job
    from .policy import Policy
    from .temporal import SmoothingConfig
    try:
        Policy.from_overrides(policy_overrides)
        SmoothingConfig.from_overrides(smoothing_overrides)
    except ValueError as e:
        raise HTTPException(400, str(e))

    workspace = None
    if video is not None and video.filename:
        scan_id = new_scan_entry(video.filename)
        workspace = config.SCAN_WORKSPACE_DIR / scan_id
        workspace.mkdir(parents=True, exist_ok=True)
        safe_name = Path(video.filename).name
        dest = workspace / f"input_{safe_name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(video.file, out)
        video.file.close()
        source = str(dest)
    elif video_path:
        source = video_path
        if not Path(source).exists():
            raise HTTPException(400, f"video_path not found: {source}")
        scan_id = new_scan_entry(Path(source).name)
    else:
        raise HTTPException(400, "provide multipart `video` or `video_path`")

    background_tasks.add_task(
        run_scan_job, scan_id, source, policy_overrides, smoothing_overrides,
        sample_fps, model, backend, fp32, workspace,
        track=track, arbiter=arbiter, arbiter_model=arbiter_model,
        arbiter_mode=arbiter_mode, crowd=crowd, crowd_title=crowd_title,
        crowd_year=crowd_year, guide=guide,
        detector_enabled=detector)

    return JSONResponse({
        "scan_id": scan_id,
        "status": scan_store[scan_id]["status"],
        "status_url": f"/scan/{scan_id}",
        "note": "segments are seconds; feed to trimmer /submit as JSON "
                "[[start, end], ...]",
    }, status_code=202)


@app.get("/scan/{scan_id}")
def get_scan(scan_id: str):
    if scan_id not in scan_store:
        raise HTTPException(404, "scan id not found")
    return scan_store[scan_id]


def _safe_output_name(name: str) -> str:
    try:
        return safe_output_name(name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/mask", status_code=202)
async def submit_mask(
    background_tasks: BackgroundTasks,
    video_path: str = Form(..., description="Path to a mounted video"),
    segments: str | None = Form(None,
                                description="JSON [[start, end], ...] seconds"),
    scan_id: str | None = Form(None,
                               description="Take segments from this scan "
                                           "(its policy is reused unless "
                                           "overridden)"),
    policy: str | None = Form(None, description="JSON policy overrides"),
    window_start: float | None = Form(None,
                                      description="Render only from here (s)"),
    window_end: float | None = Form(None,
                                    description="Render only until here (s)"),
    cq: int | None = Form(None, ge=1, le=51),
    output_name: str | None = Form(None,
                                    description="Final filename in the "
                                                "output dir (basename)"),
    bake_chapters: bool = Form(True,
                                description="Bake mask-region chapters "
                                            "into the render (full movies)"),
):
    if not Path(video_path).exists():
        raise HTTPException(400, f"video_path not found: {video_path}")
    if (window_start is None) != (window_end is None):
        raise HTTPException(400, "window_start and window_end come together")
    if output_name is not None:
        output_name = _safe_output_name(output_name)
    policy_overrides = _parse_json_field(policy, "policy")
    try:
        segs = resolve_segments(segments, scan_id)
        if policy_overrides is None and scan_id is not None:
            # consistency: mask with the same policy the scan was judged by
            policy_overrides = scan_store[scan_id].get("policy")
        from .policy import Policy
        Policy.from_overrides(policy_overrides)
    except ValueError as e:
        raise HTTPException(400, str(e))

    mask_id = new_mask_entry(Path(video_path).name)
    window = ((window_start, window_end) if window_start is not None
              else None)
    background_tasks.add_task(
        run_mask_job, mask_id, video_path, segs, policy_overrides,
        window, cq, output_name=output_name, bake_chapters=bake_chapters)
    return JSONResponse({
        "mask_id": mask_id,
        "status": mask_store[mask_id]["status"],
        "status_url": f"/mask/{mask_id}",
        "note": "output lands in the container's MASK_OUTPUT_DIR"
                + (f" as {output_name}" if output_name else "")
                + ("; chapters baked automatically"
                   if bake_chapters and window is None else ""),
    }, status_code=202)


@app.get("/mask/{mask_id}")
def get_mask(mask_id: str):
    if mask_id not in mask_store:
        raise HTTPException(404, "mask id not found")
    return mask_store[mask_id]


@app.get("/masks")
def list_masks():
    return list(mask_store.values())


@app.post("/verify", status_code=202)
async def submit_verify(
    background_tasks: BackgroundTasks,
    video_path: str | None = Form(None,
                                  description="Source video (mounted path)"),
    output_path: str | None = Form(None,
                                   description="Masked render (mounted path)"),
    mask_id: str | None = Form(None,
                               description="Resolve source/output/segments "
                                           "from this completed mask job"),
    segments: str | None = Form(None, description="JSON segments"),
    scan_id: str | None = Form(None, description="Take segments from a scan"),
    masked_i: list[int] | None = Form(
        None, description="Segment indices to spot-check (repeatable; "
                          "default: auto — seg 0 + first of each "
                          "provenance class)"),
    clean_t: list[float] | None = Form(
        None, description="Clean-spot seconds (repeatable; default: auto "
                          "— evenly spread, clear of segments)"),
    k: int = Form(6, description="Neighbor frames searched per spot"),
):
    segs_resolved = None
    if mask_id is not None:
        if mask_id not in mask_store:
            raise HTTPException(404, "mask id not found")
        entry = mask_store[mask_id]
        if entry["status"] != "completed":
            raise HTTPException(400, f"mask job is {entry['status']}")
        output_path = output_path or entry.get("output")
        video_path = video_path or entry.get("video_path")
        segs_resolved = entry.get("segments")
    if not video_path or not output_path:
        raise HTTPException(400, "provide video_path + output_path, "
                                 "or a completed mask_id")
    for p, what in ((video_path, "video_path"), (output_path, "output_path")):
        if not Path(p).exists():
            raise HTTPException(400, f"{what} not found: {p}")
    try:
        if segments is not None or scan_id is not None:
            segs = resolve_segments(segments, scan_id)
        elif segs_resolved:
            segs = [dict(s) for s in segs_resolved]
        else:
            raise ValueError("provide segments, scan_id, or mask_id")
    except ValueError as e:
        raise HTTPException(400, str(e))

    verify_id = new_verify_entry(video_path, output_path)
    background_tasks.add_task(
        run_verify_job, verify_id, video_path, output_path, segs,
        masked_i, clean_t, k)
    return JSONResponse({
        "verify_id": verify_id,
        "status": verify_store[verify_id]["status"],
        "status_url": f"/verify/{verify_id}",
    }, status_code=202)


@app.get("/verify/{verify_id}")
def get_verify(verify_id: str):
    if verify_id not in verify_store:
        raise HTTPException(404, "verify id not found")
    return verify_store[verify_id]


@app.get("/verifies")
def list_verifies():
    return list(verify_store.values())


@app.post("/chapters", status_code=202)
async def submit_chapters(
    background_tasks: BackgroundTasks,
    output_path: str | None = Form(None,
                                   description="Render to chapter-ize "
                                               "(mounted path)"),
    mask_id: str | None = Form(None,
                               description="Resolve output/segments from "
                                           "this completed mask job"),
    segments: str | None = Form(None, description="JSON segments"),
    scan_id: str | None = Form(None, description="Take segments from a scan"),
):
    segs_resolved = None
    if mask_id is not None:
        if mask_id not in mask_store:
            raise HTTPException(404, "mask id not found")
        entry = mask_store[mask_id]
        if entry["status"] != "completed":
            raise HTTPException(400, f"mask job is {entry['status']}")
        output_path = output_path or entry.get("output")
        segs_resolved = entry.get("segments")
    if not output_path:
        raise HTTPException(400, "provide output_path or a completed "
                                 "mask_id")
    if not Path(output_path).exists():
        raise HTTPException(400, f"output_path not found: {output_path}")
    try:
        if segments is not None or scan_id is not None:
            segs = resolve_segments(segments, scan_id)
        elif segs_resolved:
            segs = [dict(s) for s in segs_resolved]
        else:
            raise ValueError("provide segments, scan_id, or mask_id")
    except ValueError as e:
        raise HTTPException(400, str(e))

    chapters_id = new_chapters_entry(output_path)
    background_tasks.add_task(
        run_chapters_job, chapters_id, output_path, segs)
    return JSONResponse({
        "chapters_id": chapters_id,
        "status": chapters_store[chapters_id]["status"],
        "status_url": f"/chapters/{chapters_id}",
    }, status_code=202)


@app.get("/chapters/{chapters_id}")
def get_chapters(chapters_id: str):
    if chapters_id not in chapters_store:
        raise HTTPException(404, "chapters id not found")
    return chapters_store[chapters_id]


@app.get("/scans")
def list_scans():
    return list(scan_store.values())


@app.get("/health")
def health():
    from .arbiter import resolve_vlm_path
    from .detector import _trt_available
    from .trt import gpu_slug, trt_slug
    from . import vlm as vlm_supervisor
    vlm_model = config.DEFAULT_ARBITER_MODEL
    vlm = resolve_vlm_path(vlm_model) if vlm_model else ""
    return {
        "gpu": gpu_slug(),
        "tensorrt": trt_slug() if _trt_available() else "unavailable",
        "trt_cache_dir": str(config.TRT_CACHE_DIR),
        "models": {name: Path(config.MODEL_DIR / fname).exists()
                   for name, (fname, _) in config.MODELS.items()},
        "vlm": {
            "model": vlm_model or "(llama-server via ARBITER_BASE_URL)",
            "path": vlm,
            "available": bool(vlm) and Path(vlm).is_dir(),
            # on-demand server state — reported, never probed: /health
            # must not spawn the ~20GB model (vlm.acquire does that)
            "server": vlm_supervisor.status(),
        },
        "loaded_detectors": [f"{m}/{b}"
                             for (m, b, _, _) in registry._detectors.keys()],
        "scan_workspaces": str(config.SCAN_WORKSPACE_DIR),
    }


# Backwards-compat alias so ScanResult import is used (kept for API docs)
__all__ = ["app", "ScanResult"]
