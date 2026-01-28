import os
import shutil
import uuid
import subprocess
import json
import sys
import asyncio
import logging
import copy
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

import pysrt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# --- Configuration ---
TEMP_DIR = Path("temp_workspaces")
FINISHED_DIR = Path("finished_jobs")
MAX_CONCURRENT_JOBS = 2 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GPU_Cutter")

# --- Global State ---
job_store: Dict[str, Dict[str, Any]] = {}
gpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if TEMP_DIR.exists():
        logger.warning(f"Startup: Cleaning up {TEMP_DIR}...")
        shutil.rmtree(TEMP_DIR)
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINISHED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Startup: Directories ready.")
    yield
    logger.info("Shutdown: Server stopping.")

app = FastAPI(
    title="Async GPU Video Cutter + Subtitles",
    description="Cut video and subtitles (SRT) concurrently.",
    lifespan=lifespan
)

# --- Helpers ---

def parse_time_str(time_str: str) -> float:
    try:
        parts = time_str.strip().split(':')
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        raise ValueError(f"Invalid timestamp: {time_str}")

def parse_segments_input(raw_input: str) -> List[List[float]]:
    raw_input = raw_input.strip()
    segments_in_seconds = []
    
    # Try JSON
    if raw_input.startswith("["):
        try:
            data = json.loads(raw_input)
            for start, end in data:
                segments_in_seconds.append([parse_time_str(str(start)), parse_time_str(str(end))])
            return segments_in_seconds
        except Exception:
            pass 

    # Try Simple String "00:00-00:10, 00:20-00:30"
    try:
        ranges = raw_input.split(',')
        for rng in ranges:
            if not rng.strip(): continue
            parts = rng.replace(" ", "").split('-')
            if len(parts) == 2:
                segments_in_seconds.append([parse_time_str(parts[0]), parse_time_str(parts[1])])
            else:
                raise ValueError
        if not segments_in_seconds: raise ValueError
        return segments_in_seconds
    except Exception:
        raise HTTPException(400, "Invalid segments format.")

def get_video_duration(input_file: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", input_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)['format']['duration'])

def run_ffmpeg_command(cmd: List[str], job_id: str):
    logger.info(f"[Job {job_id}] Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

# --- Subtitle Processing Logic ---

def process_subtitles(input_srt_path: str, output_srt_path: str, keep_segments: List[List[float]]):
    """
    Reads source SRT, extracts only parts within keep_segments, 
    shifts their timestamps to align with the new concatenated video, 
    and saves the result.
    """
    # Load original subtitles (try-catch for encoding)
    try:
        subs = pysrt.open(input_srt_path, encoding='utf-8')
    except Exception:
        # Fallback for common windows encodings
        subs = pysrt.open(input_srt_path, encoding='iso-8859-1')

    final_subs = pysrt.SubRipFile()
    current_timeline_offset = 0.0

    for seg_start, seg_end in keep_segments:
        # Iterate over all subs to find overlaps
        # (Optimization: could filter sorted subs, but simple iteration is fine for typical sizes)
        for sub in subs:
            # Convert pysrt times to seconds (float)
            s_start = sub.start.ordinal / 1000.0
            s_end = sub.end.ordinal / 1000.0

            # Calculate overlap with the current Keep Segment
            # We clip the subtitle to the segment boundaries
            effective_start = max(s_start, seg_start)
            effective_end = min(s_end, seg_end)

            if effective_end > effective_start:
                # Subtitle is visible in this segment
                
                # Calculate new start/end relative to the NEW timeline
                # 1. Start relative to the beginning of this segment
                rel_start = effective_start - seg_start
                rel_end = effective_end - seg_start
                
                # 2. Add the offset of previous segments
                new_start_sec = rel_start + current_timeline_offset
                new_end_sec = rel_end + current_timeline_offset

                # Create new sub item
                new_sub = copy.deepcopy(sub)
                new_sub.start = pysrt.SubRipTime.from_ordinal(int(new_start_sec * 1000))
                new_sub.end = pysrt.SubRipTime.from_ordinal(int(new_end_sec * 1000))
                
                final_subs.append(new_sub)

        # Increment offset for the next segment
        current_timeline_offset += (seg_end - seg_start)

    # Save processed SRT
    final_subs.save(output_srt_path, encoding='utf-8')

# --- Background Worker ---

async def background_video_processor(
    job_id: str, 
    input_path: str, 
    subtitle_path: Optional[str],
    segments: List[List[float]]
):
    work_dir = Path(input_path).parent
    final_output_filename = f"processed_{job_id}.mp4"
    final_output_path = FINISHED_DIR / final_output_filename

    try:
        async with gpu_semaphore:
            job_store[job_id]["status"] = "processing"
            logger.info(f"[Job {job_id}] Started processing...")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                _sync_ffmpeg_logic, 
                job_id, input_path, subtitle_path, str(final_output_path), segments
            )

            job_store[job_id]["status"] = "completed"
            job_store[job_id]["output_file"] = final_output_filename
            logger.info(f"[Job {job_id}] Completed. Saved to {final_output_filename}")

    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)

def _sync_ffmpeg_logic(
    job_id: str, 
    input_path: str, 
    subtitle_path: Optional[str],
    output_path: str, 
    remove_segments: List[List[float]]
):
    total_duration = get_video_duration(input_path)
    
    # Calculate Keep Segments
    remove_segments.sort(key=lambda x: x[0])
    keep_segments = []
    current_pos = 0.0

    for r_start, r_end in remove_segments:
        if r_start > current_pos:
            keep_segments.append((current_pos, r_start))
        current_pos = max(current_pos, r_end)
    if current_pos < total_duration:
        keep_segments.append((current_pos, total_duration))

    if not keep_segments:
        raise ValueError("Removed segments cover the entire video.")

    temp_dir = os.path.dirname(input_path)
    
    # --- 1. Process Subtitles (if present) ---
    processed_srt_path = None
    if subtitle_path and os.path.exists(subtitle_path):
        logger.info(f"[Job {job_id}] Processing subtitles...")
        processed_srt_path = os.path.join(temp_dir, "processed_subs.srt")
        process_subtitles(subtitle_path, processed_srt_path, keep_segments)

    # --- 2. Extract Video Segments ---
    concat_files = []
    for i, (start, end) in enumerate(keep_segments):
        seg_filename = os.path.join(temp_dir, f"seg_{i}.mp4")
        concat_files.append(seg_filename)
        duration = end - start
        
        cmd_extract = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-ss", str(start),
            "-t", str(duration),
            "-i", input_path,
            "-c:v", "av1_nvenc", "-preset", "p4",
            "-c:a", "libopus",
            "-strict", "-2",
            seg_filename
        ]
        run_ffmpeg_command(cmd_extract, job_id)

    # --- 3. Concatenate and Mux Subtitles ---
    list_file = os.path.join(temp_dir, "concat.txt")
    with open(list_file, "w") as f:
        for tf in concat_files:
            safe_path = Path(tf).absolute().as_posix()
            f.write(f"file '{safe_path}'\n")

    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file
    ]

    # If we have subtitles, add them as input 1
    if processed_srt_path:
        cmd_concat.extend(["-i", processed_srt_path])

    # Map video/audio from concat (stream 0)
    cmd_concat.extend(["-c:v", "copy", "-c:a", "copy"])
    
    # Map subtitles (stream 1) if available
    if processed_srt_path:
        # -map 1:s selects subtitle stream from input 1 (srt file)
        # -c:s mov_text converts SRT to MP4 compatible soft subs
        cmd_concat.extend(["-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:s", "mov_text"])

    cmd_concat.append(output_path)
    
    run_ffmpeg_command(cmd_concat, job_id)

# --- Endpoints ---

@app.post("/submit", summary="Submit Video (+ optional Subtitle)")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video File"),
    subtitle_file: Optional[UploadFile] = File(None, description="Optional .srt file"),
    segments: str = Form(..., description="'00:00:00-00:00:10, ...'")
):
    job_id = str(uuid.uuid4())
    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)
    input_video_path = work_dir / f"input_{file.filename}"
    input_sub_path = None

    try:
        # Validate segments
        segments_parsed = parse_segments_input(segments)
        
        # Save Video
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()

        # Save Subtitle (if provided)
        if subtitle_file:
            # Check extension (basic check)
            if not subtitle_file.filename.lower().endswith('.srt'):
                # Warning: We just log/ignore or could raise error. 
                # Let's save it anyway and let pysrt try to handle it.
                pass
                
            input_sub_path = str(work_dir / f"input.srt")
            with open(input_sub_path, "wb") as buffer:
                shutil.copyfileobj(subtitle_file.file, buffer)
            subtitle_file.file.close()

    except Exception as e:
        shutil.rmtree(work_dir)
        raise HTTPException(400, f"Submission failed: {str(e)}")

    # Queue Job
    job_store[job_id] = {
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "original_filename": file.filename,
        "has_subtitles": bool(input_sub_path)
    }
    
    background_tasks.add_task(
        background_video_processor, 
        job_id, 
        str(input_video_path), 
        input_sub_path,
        segments_parsed
    )

    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "has_subtitles": bool(input_sub_path),
        "status_url": f"/jobs/{job_id}",
        "download_url": f"/download/{job_id}"
    })

@app.get("/jobs", summary="List All Jobs")
def list_jobs():
    return job_store

@app.get("/jobs/{job_id}", summary="Get Job Status")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(404, "Job ID not found")
    return job_store[job_id]

@app.get("/download/{job_id}", summary="Download Finished File")
def download_file(job_id: str):
    if job_id not in job_store:
        raise HTTPException(404, "Job ID not found")
    
    job_info = job_store[job_id]
    if job_info["status"] != "completed":
        raise HTTPException(400, f"Job not ready. Status: {job_info['status']}")
    
    filename = job_info.get("output_file")
    file_path = FINISHED_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(500, "File missing from disk")
        
    return FileResponse(path=file_path, filename=filename, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
