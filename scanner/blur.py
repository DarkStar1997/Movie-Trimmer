"""Stage C: mask-precision blur render.

Two passes over the movie:

  analysis (cheap, flagged ranges only)
    For each remove-segment, decode [start, end] at BLUR_KEYFRAME_FPS and
    analysis size (the scan's decode size — detection parity). Run NudeNet
    per batch; for every keyframe whose detections pass the policy triggers,
    compute the blur mask:

        mask = union over explicit boxes of
                 ( person mask matched to that box  AND  padded box )
               falling back to the padded box rectangle when the person
               segmenter finds no match (close-up crops where the person
               detector fails are exactly when the box matters).

    Masks are dilated at analysis resolution (covers keyframe-hold motion)
    and stored packed per keyframe timestamp.

  render (two modes)
    gpu-native (default, full renders): one continuous pass — NVDEC decode
    to CUDA frames, torch.from_dlpack zero-copy NV12 view, in-place pixelate
    through the held person mask, NVENC encode (hevc P5 hq, VBR + explicit
    CQ), FFmpegMuxer with audio/subs passthrough and bit-exact PTS. No cut
    boundaries exist, so none can be gotten wrong (the abandoned splice
    approach failed twice on ffmpeg's copy-cut dts/pts semantics).
    pipe (window previews): ffmpeg NVDEC full-res rawvideo pipe -> per frame
    inside a segment, take the most recent keyframe mask, upscale nearest to
    full res, pixelate the full frame (block downscale/upscale) and composite
    through the mask -> pipe into a second ffmpeg running NVENC with explicit
    quality control (hevc_nvenc p5 hq, VBR -cq, spatial AQ — the ANALYSIS.md
    finding was that NVENC defaults to ~2 Mbps; the CQ is set explicitly
    here. av1_nvenc is empirically broken on this host's rawvideo path:
    -22 EINVAL). Audio is stream-copied; subtitles pass through as mov_text.
"""

from __future__ import annotations

import bisect
import json
import logging
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .decode import (DecoderStalled, FFmpegPipeDecoder, probe_dimensions,
                     probe_duration)
from .detector import Detection, NudeNetDetector
from .personseg import PersonSegmenter
from .policy import Policy

logger = logging.getLogger(__name__)


def probe_fps_fraction(path: str) -> str:
    """Source fps as an exact fraction string, e.g. '24000/1001'."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]["r_frame_rate"]


def probe_fps(path: str) -> float:
    num, den = probe_fps_fraction(path).split("/")
    return float(num) / float(den)


# --- pure helpers (host-testable) ------------------------------------------------

def pad_box(box_xyxy: np.ndarray, pad_frac: float,
            frame_w: int, frame_h: int) -> np.ndarray:
    """Expand a box by pad_frac of its own w/h per side, clipped to frame."""
    x1, y1, x2, y2 = box_xyxy.astype(np.float64)
    w, h = x2 - x1, y2 - y1
    return np.clip([x1 - w * pad_frac, y1 - h * pad_frac,
                    x2 + w * pad_frac, y2 + h * pad_frac],
                   0, [frame_w, frame_h, frame_w, frame_h])


def box_mask(box_xyxy: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    m = np.zeros((frame_h, frame_w), dtype=bool)
    x1, y1, x2, y2 = np.clip(box_xyxy, 0, None).astype(int)
    x2, y2 = min(x2, frame_w), min(y2, frame_h)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def intersect_with_box(mask: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    x1, y1, x2, y2 = np.clip(box_xyxy, 0, None).astype(int)
    m[:max(0, y1), :] = False
    m[min(h, y2):, :] = False
    m[:, :max(0, x1)] = False
    m[:, min(w, x2):] = False
    return m


def explicit_boxes(dets: list[Detection], policy: Policy,
                   frame_w: int, frame_h: int) -> list[np.ndarray]:
    """xyxy boxes of detections that pass the policy triggers."""
    out = []
    for d in dets:
        thr = policy.triggers.get(d.label)
        if thr is None or d.score < thr:
            continue
        if d.area_fraction(frame_w, frame_h) < policy.min_box_area_frac:
            continue
        x, y, w, h = d.box
        out.append(np.asarray([x, y, x + w, y + h], dtype=np.float32))
    return out


def dilate_mask(mask: np.ndarray, kernel_px: int) -> np.ndarray:
    if kernel_px <= 1 or not mask.any():
        return mask
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def pixelate(frame: np.ndarray, mask: np.ndarray, block_px: int) -> np.ndarray:
    """Block-pixelate `frame` inside `mask` (block_px at frame width)."""
    if not mask.any() or block_px <= 1:
        return frame
    h, w = frame.shape[:2]
    down = cv2.resize(frame, (max(1, w // block_px), max(1, h // block_px)),
                      interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[mask] = pix[mask]
    return out


def keyframe_masks_for(dets_per_frame, frames, policy, segmenter,
                       dilate_px: int) -> list[np.ndarray]:
    """Per keyframe blur mask (analysis res) for one batch of frames."""
    masks = []
    for frame, dets in zip(frames, dets_per_frame):
        h, w = frame.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        boxes = explicit_boxes(dets, policy, w, h)
        for bx in boxes:
            pmask, _, _ = segmenter.infer(frame, match_box=bx)
            padded = pad_box(bx, config.BLUR_BOX_PAD_FRAC, w, h)
            if pmask is not None:
                union |= intersect_with_box(pmask, padded)
            else:                      # no person match: box-rect fallback
                union |= box_mask(padded, w, h)
        union = dilate_mask(union, dilate_px)
        masks.append(union)
    return masks


# --- analysis pass ----------------------------------------------------------------

def analysis_pass(video_path: str, segments: list[dict], policy: Policy,
                  detector: NudeNetDetector, segmenter: PersonSegmenter,
                  aw: int, ah: int, batch_size: int = 16) -> dict[int, list]:
    """-> {segment_index: [(t, packed_mask_bytes), ...]} sorted by t."""
    kf_fps = config.BLUR_KEYFRAME_FPS
    # dilation covers keyframe-hold motion; scaled to the analysis resolution
    dilate_px = max(3, int(round(config.BLUR_MASK_DILATE_FRAC * aw)))
    per_seg: dict[int, list] = {}
    for si, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        entries: list[tuple] = []
        t0 = max(0.0, start - 0.5)

        def flush(frames, first_idx):
            results = detector.detect_frames(frames)
            masks = keyframe_masks_for(results, frames, policy, segmenter,
                                       dilate_px)
            for j, m in enumerate(masks):
                if m.any():
                    t = t0 + (first_idx + j) / kf_fps
                    if start <= t <= end:
                        entries.append((t, np.packbits(m).tobytes()))

        def analyze(hw: bool) -> None:
            dec = FFmpegPipeDecoder(video_path, fps=kf_fps, width=aw,
                                    height=ah, start=t0,
                                    duration=end - start + 1.0,
                                    use_hwaccel=hw)
            batch: list[np.ndarray] = []
            try:
                with dec:
                    idx = 0
                    for frame in dec.frames():
                        batch.append(frame)
                        if len(batch) >= batch_size:
                            flush(batch, idx)
                            idx += len(batch)
                            batch = []
                    if batch:
                        flush(batch, idx)
            finally:
                dec.close()

        try:
            try:
                analyze(True)
            except DecoderStalled:
                logger.warning("[blur] NVDEC stall in segment %d — "
                               "retrying on CPU decode", si)
                analyze(False)
        except Exception as e:  # noqa: BLE001
            logger.warning("[blur] analysis of segment %d failed: %s", si, e)
        per_seg[si] = entries
        logger.info("[blur] segment %d [%d-%d]: %d mask keyframes",
                    si, int(start), int(end), len(entries))
    return per_seg


# --- render pass -------------------------------------------------------------------

def _writer_args(codec: str, cq: int) -> list[str]:
    if codec == "libx264":
        return ["-preset", "medium", "-crf", str(cq)]
    args = ["-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
            "-spatial-aq", "1", "-rc-lookahead", "32"]
    if codec != "av1_nvenc":
        args += ["-tune", "hq"]
    return args


def _start_writer(video_path: str, out_path: str, w: int, h: int,
                  fps_frac: str, codecs: tuple[str, ...], cq: int,
                  window: tuple[float, float] | None = None):
    """Spawn the NVENC writer (window previews); falls back down the codec
    chain on fast failure. Full renders use the GPU-native path
    (scanner/gpu_render.py) instead."""
    last_err = ""
    for codec in codecs:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", fps_frac, "-i", "pipe:0"]
        if window is not None:
            cmd += ["-ss", f"{window[0]:.3f}", "-t",
                    f"{window[1] - window[0]:.3f}"]
        cmd += ["-i", video_path,
                "-map", "0:v", "-map", "1:a?", "-map", "1:s?",
                "-c:v", codec, *_writer_args(codec, cq),
                "-c:a", "copy", "-c:s", "mov_text",
                "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        time.sleep(1.5)
        if proc.poll() not in (None, 0):
            last_err = proc.stderr.read().decode(errors="replace")[-400:]
            logger.warning("[blur] writer codec %s failed: %s", codec,
                           last_err)
            continue
        logger.info("[blur] writer: %s", codec)
        return proc, codec
    raise RuntimeError(f"no usable encoder (tried {codecs}): {last_err}")


def _render_range(video_path: str, t0: float, t1: float, w: int, h: int,
                  aw: int, ah: int, per_seg: dict, seg_bounds: list,
                  starts: list, kf_index: dict, block_px: int,
                  cq: int, codecs: tuple, out_path: str
                  ) -> tuple[int, int, str]:
    """Decode [t0, t1), blur masked frames, write through NVENC (pipe path,
    used for window previews). NVDEC wedge (known quirk) surfaces as
    DecoderStalled; one CPU-decode retry, then fail loud."""
    fps_frac = probe_fps_fraction(video_path)
    src_fps = probe_fps(video_path)
    for hw in (True, False):
        reader = FFmpegPipeDecoder(video_path, fps=24.0, width=w, height=h,
                                   start=t0, duration=t1 - t0 + 2.0 / src_fps,
                                   exact=True, use_hwaccel=hw)
        writer = None
        frames_out = frames_blurred = 0
        try:
            with reader:
                writer, codec = _start_writer(
                    video_path, out_path, w, h, fps_frac,
                    codecs, cq, (t0, t1))
                held = {"si": -1, "ki": -1, "mask": None}
                for idx, frame in enumerate(reader.frames()):
                    t = t0 + idx / src_fps
                    si = bisect.bisect_right(starts, t) - 1
                    inside = (0 <= si < len(seg_bounds)
                              and seg_bounds[si][0] <= t <= seg_bounds[si][1])
                    if inside and per_seg.get(si):
                        entries = per_seg[si]
                        ki = max(0, bisect.bisect_right(kf_index[si], t) - 1)
                        if held["si"] != si or held["ki"] != ki:
                            packed = np.frombuffer(entries[ki][1], dtype=np.uint8)
                            m = np.unpackbits(
                                packed, count=ah * aw).reshape(ah, aw)
                            held = {"si": si, "ki": ki, "mask": cv2.resize(
                                m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)}
                        frame = pixelate(frame, held["mask"], block_px)
                        frames_blurred += 1
                    writer.stdin.write(frame.tobytes())
                    frames_out += 1
                writer.stdin.close()
                rc = writer.wait(timeout=300)
                if rc != 0:
                    err = writer.stderr.read().decode(errors="replace")[-400:]
                    raise RuntimeError(f"writer exited {rc}: {err}")
            return frames_out, frames_blurred, codec
        except DecoderStalled:
            if writer is not None and writer.poll() is None:
                writer.terminate()
            if not hw:
                raise
            logger.warning("[blur] NVDEC stall in [%.3f,%.3f) — "
                           "retrying range on CPU decode", t0, t1)
    raise RuntimeError("unreachable")


def render_pass(video_path: str, out_path: str, segments: list[dict],
                per_seg: dict[int, list], w: int, h: int, aw: int, ah: int,
                window: tuple[float, float] | None,
                cq: int = None, pixels: int = None) -> dict:
    """Pipe-path render for window previews (small ranges; full renders use
    the GPU-native scanner/gpu_render.py)."""
    cq = cq or config.BLUR_CQ
    pixels = pixels or config.BLUR_PIXELS
    block_px = max(2, int(round(pixels * w / 1920)))
    seg_bounds = [(float(s["start"]), float(s["end"])) for s in segments]
    starts = [a for a, _ in seg_bounds]
    kf_index = {si: [t for t, _ in entries] for si, entries in per_seg.items()}

    frames_out, frames_blurred, codec = _render_range(
        video_path, window[0], window[1], w, h, aw, ah, per_seg, seg_bounds,
        starts, kf_index, block_px, cq, config.BLUR_CODECS,
        out_path=out_path)
    return {"frames": frames_out, "frames_blurred": frames_blurred,
            "codec": codec}


def run_blur(video_path: str, segments: list[dict], out_path: str, *,
             policy: Policy, backend: str = "trt",
             window: tuple[float, float] | None = None,
             cq: int = None, pixels: int = None) -> dict:
    """Full stage-C pipeline: analysis (masks) -> render.

    Full renders take the GPU-native path (NVDEC -> torch NV12 zero-copy ->
    in-place pixelate -> NVENC, audio passthrough, one continuous pass — no
    cut boundaries). A `window` takes the pipe-path preview render."""
    t0 = time.monotonic()
    if window is not None:
        a, b = window
        segments = [dict(s) for s in segments
                    if s["end"] > a and s["start"] < b]
        for s in segments:
            s["start"], s["end"] = max(s["start"], a), min(s["end"], b)
    if not segments:
        raise ValueError("no segments to blur")

    w, h = probe_dimensions(video_path)
    from .decode import analysis_size
    aw, ah = analysis_size(w, h, config.DECODE_WIDTH)

    detector = NudeNetDetector(backend=backend)
    segmenter = PersonSegmenter(backend="trt" if backend != "ort" else "ort")
    per_seg = analysis_pass(video_path, segments, policy, detector,
                            segmenter, aw, ah)
    if not any(per_seg.values()):
        raise RuntimeError("analysis produced no masks — nothing to blur")

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if window is not None:
        render = render_pass(video_path, out_path, segments, per_seg, w, h,
                             aw, ah, window, cq=cq, pixels=pixels)
    else:
        from .gpu_render import render_gpu
        render = render_gpu(video_path, out_path, segments, per_seg,
                            w, h, aw, ah, cq=cq, pixels=pixels)

    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", out_path], capture_output=True, text=True,
        check=True).stdout.strip()
    return {
        "output": out_path,
        "segments_blurred": len(segments),
        "keyframes": sum(len(v) for v in per_seg.values()),
        "duration_out": float(dur),
        "wall_time_s": round(time.monotonic() - t0, 2),
        **render,
    }
