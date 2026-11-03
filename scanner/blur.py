"""Stage C: mask-precision blackout render.

Two passes over the movie:

  analysis (cheap, flagged ranges only)
    For each remove-segment, decode [start, end] at BLUR_KEYFRAME_FPS and
    analysis size (the scan's decode size — detection parity). Run NudeNet
    per batch; per keyframe the censor mask is:

      detector segments — union over trigger-passing boxes of the MATCHED
      person's full-body mask (the whole person is blurred, not just the
      part box); padded box-rect fallback when no person matches (close-up
      crops where the person detector fails are exactly when the box
      matters). When NudeNet flickers off, the last matched person box is
      re-matched so the person stays masked while in frame (tracking).
      Arbiter/VLM-found segments have no boxes by construction (NudeNet's
      blind spots) — every person the segmenter finds is masked, with a
      centered-frame-rectangle fallback.

    Every masked keyframe carries (label, score) for the warning overlay;
    at render time each flagged frame gets red top-left text
    "VULGARITY DETECTED / <conf>% <label>" (BLUR_OVERLAY kill-switch) — a
    black silhouette needs no help being invisible, but the flag tells the
    viewer what happened and why a person-shaped void walks through the
    scene.

    Masks are dilated at analysis resolution (covers keyframe-hold motion)
    and stored packed per keyframe timestamp.

  render (two modes)
    gpu-native (default, full renders): one continuous pass — NVDEC decode
    to CUDA frames, torch.from_dlpack zero-copy NV12 view, in-place
    solid-black fill through the held person mask, NVENC encode (hevc P5
    hq, VBR + explicit CQ), FFmpegMuxer with audio/subs passthrough and
    bit-exact PTS. No cut boundaries exist, so none can be gotten wrong
    (the abandoned splice approach failed twice on ffmpeg's copy-cut
    dts/pts semantics).
    pipe (window previews): ffmpeg NVDEC full-res rawvideo pipe -> per frame
    inside a segment, take the most recent keyframe mask, upscale nearest to
    full res, fill solid black through the mask -> pipe into a second ffmpeg
    running NVENC with explicit quality control (hevc_nvenc p5 hq, VBR -cq,
    spatial AQ — the ANALYSIS.md finding was that NVENC defaults to ~2 Mbps;
    the CQ is set explicitly here. av1_nvenc is empirically broken on this
    host's rawvideo path: -22 EINVAL). Audio is stream-copied; subtitles
    pass through as mov_text.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import subprocess
import tempfile
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


def probe_pix_fmt(path: str) -> str:
    """Source pixel format, e.g. 'yuv420p' (8-bit) or 'yuv420p10le' (10-bit)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=pix_fmt", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]["pix_fmt"]


def count_subtitle_streams(path: str) -> int:
    """Subtitle stream count in the container (0 for none)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "s",
           "-show_entries", "stream=index", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return len(json.loads(out.stdout).get("streams", []))


def strip_subtitles(path: str, workdir: str) -> str:
    """Sub-free stream-copy of the source (video+audio bit-exact) for the
    GPU render: PyNvVideoCodec's muxer cannot carry subtitle streams over —
    the MP4 muxer EINVALs at header write and the MKV muxer segfaults (a
    45-track SRT WEBRip found both). Subtitles are dropped from blur
    outputs."""
    fd, tmp = tempfile.mkstemp(suffix=".mkv", dir=workdir)
    os.close(fd)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                        "-map", "0:v:0", "-map", "0:a", "-c", "copy",
                        "-sn", "-dn", tmp], check=True)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return tmp


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


def blackout_rgb(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Solid black inside `mask` (person silhouette shape).

    Replaced the block-mosaic pixelate: mosaic kept silhouettes legible
    and bright scenes readable; a constant fill leaves nothing.
    """
    if not mask.any():
        return frame
    out = frame.copy()
    out[mask] = 0
    return out


# warning overlay: every flagged frame tells the viewer what happened —
# a black silhouette is unmissable, but the text says WHY it is there
OVERLAY_HEAD = "VULGARITY DETECTED"
OVERLAY_FILL_RGB = (255, 0, 0)     # pure red (RGB frame order)
OVERLAY_OUTLINE_RGB = (0, 0, 0)    # black outline for contrast on red scenes


def draw_warning(canvas_rgb: np.ndarray, label: str, score: float,
                 fill: tuple = OVERLAY_FILL_RGB,
                 outline: tuple = OVERLAY_OUTLINE_RGB) -> None:
    """In-place red warning text, top-left: VULGARITY DETECTED / conf% label.

    Resolution-scaled (font ~ 1.15x at 1080p width); the outline is a
    thicker stroke drawn under the fill. Callers compositing in NV12 pass
    sentinel colors and derive masks from the channels."""
    h, w = canvas_rgb.shape[:2]
    fs = max(0.55, 1.15 * w / 1920)
    th = max(2, int(round(2.5 * w / 1920)))
    lines = (OVERLAY_HEAD,
             f"{int(round(float(score) * 100))}%  {label}" if label
             else f"{int(round(float(score) * 100))}%")
    x = max(8, int(0.018 * w))
    y = max(int(0.055 * h), int(fs * 30))
    step = int(fs * 42)
    for line in lines:
        cv2.putText(canvas_rgb, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, outline, th + 3, cv2.LINE_AA)
        cv2.putText(canvas_rgb, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, fill, th, cv2.LINE_AA)
        y += step


def keyframe_masks_for(dets_per_frame, frames, policy, segmenter,
                       dilate_px: int, vlm_source: bool = False,
                       vlm_score: float = 0.9,
                       src_label: str = "VLM",
                       track_state: dict | None = None
                       ) -> list[tuple[np.ndarray, str, float]]:
    """Per keyframe (mask, label, score) at analysis res for one batch.

    Detector-sourced segments: for each explicit-part box the matched
    person's FULL mask is used (not the part-box intersection — the brief
    is to blur the whole person). When no person matches the box, the
    padded box rectangle is the fallback (close-up crops where the person
    detector fails are exactly when the box matters; an unmatched person
    mask would risk blurring the wrong person). NudeNet flicker is bridged
    by re-matching the last matched person box (track_state) so the person
    stays masked for as long as they remain in frame.

    VLM-sourced segments (arbiter-found blind spots): NudeNet is blind
    exactly there, so masks come from masking EVERY person the segmenter
    finds, with a center-box fallback when it finds nobody.
    """
    state = track_state if track_state is not None else {}
    out: list[tuple[np.ndarray, str, float]] = []
    for frame, dets in zip(frames, dets_per_frame):
        h, w = frame.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        label, score = "", 0.0

        if vlm_source:
            # arbiter/crowd segment: no NudeNet boxes by construction —
            # mask all persons; the source saw nudity, so err wide
            pmask = segmenter.all_persons(frame)
            if pmask is None or not pmask.any():
                frac = config.BLUR_VLM_CENTER_FRAC
                pmask = box_mask(
                    np.asarray([(1 - frac) * w / 2, (1 - frac) * h / 2,
                                (1 + frac) * w / 2, (1 + frac) * h / 2]),
                    w, h)
            out.append((dilate_mask(pmask, dilate_px), src_label, vlm_score))
            continue

        # strongest trigger-passing detection this frame (for the overlay)
        for d in dets:
            thr = policy.triggers.get(d.label)
            if (thr is not None and d.score >= thr and d.score > score
                    and d.area_fraction(w, h) >= policy.min_box_area_frac):
                label, score = d.label, d.score

        boxes = explicit_boxes(dets, policy, w, h)
        if boxes:
            for bx in boxes:
                pmask, _, _, pbbox = segmenter.infer(frame, match_box=bx)
                if pmask is not None and pbbox is not None:
                    union |= pmask               # FULL person mask
                    state["bbox"] = pbbox        # track for flicker frames
                    state["label"], state["score"] = label, score
                else:
                    padded = pad_box(bx, config.BLUR_BOX_PAD_FRAC, w, h)
                    union |= box_mask(padded, w, h)
        elif state.get("bbox") is not None:
            # NudeNet flickered off but the person may still be in frame:
            # re-match the tracked person and keep their full mask on
            pmask, _, _, pbbox = segmenter.infer(frame,
                                                 match_box=state["bbox"])
            if pmask is not None and pbbox is not None:
                union |= pmask
                state["bbox"] = pbbox
                label = state.get("label", "")
                score = state.get("score", 0.0)
            else:
                state["bbox"] = None             # person left the frame
        out.append((dilate_mask(union, dilate_px), label, score))
    return out


# --- analysis pass ----------------------------------------------------------------

def analysis_pass(video_path: str, segments: list[dict], policy: Policy,
                  detector: NudeNetDetector, segmenter: PersonSegmenter,
                  aw: int, ah: int, batch_size: int = 16) -> dict[int, list]:
    """-> {segment_index: [(t, packed_mask_bytes, label, score), ...]} by t.

    label/score ride along for the warning overlay: the trigger label +
    score for detector segments, "VLM" + arbiter confidence for
    arbiter-found segments (identified by 'arbiter' in seg['labels']),
    "CRINGEMDB:<category>" + crowd score for community-timestamp
    segments (labels starting with 'cringemdb'). Both non-detector
    sources mask every person, with a center fallback."""
    kf_fps = config.BLUR_KEYFRAME_FPS
    # dilation covers keyframe-hold motion; scaled to the analysis resolution
    dilate_px = max(3, int(round(config.BLUR_MASK_DILATE_FRAC * aw)))
    per_seg: dict[int, list] = {}
    for si, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        entries: list[tuple] = []
        t0 = max(0.0, start - 0.5)
        labs = seg.get("labels") or {}
        crowd_lab = next((l for l in labs if str(l).startswith("cringemdb")),
                         None)
        vlm = "arbiter" in labs or crowd_lab is not None
        vlm_score = float(seg.get("mean_score")
                          or (config.CROWD_SCORE if crowd_lab else 0.9))
        src_label = crowd_lab.upper() if crowd_lab else "VLM"
        track_state: dict = {}

        def flush(frames, first_idx):
            results = detector.detect_frames(frames)
            masks = keyframe_masks_for(results, frames, policy, segmenter,
                                       dilate_px, vlm_source=vlm,
                                       vlm_score=vlm_score,
                                       src_label=src_label,
                                       track_state=track_state)
            for j, (m, label, score) in enumerate(masks):
                if m.any():
                    t = t0 + (first_idx + j) / kf_fps
                    if start <= t <= end:
                        entries.append((t, np.packbits(m).tobytes(),
                                        label, score))

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
        logger.info("[blur] segment %d [%d-%d]%s: %d mask keyframes",
                    si, int(start), int(end),
                    " (vlm)" if vlm else "", len(entries))
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
                  starts: list, kf_index: dict,
                  cq: int, codecs: tuple, out_path: str
                  ) -> tuple[int, int, str]:
    """Decode [t0, t1), black out masked frames, write through NVENC (pipe
    path, used for window previews). NVDEC wedge (known quirk) surfaces as
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
                                interpolation=cv2.INTER_NEAREST).astype(bool),
                                "label": entries[ki][2],
                                "score": entries[ki][3]}
                        frame = blackout_rgb(frame, held["mask"])
                        if config.BLUR_OVERLAY:
                            draw_warning(frame, held["label"], held["score"])
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
                cq: int = None) -> dict:
    """Pipe-path render for window previews (small ranges; full renders use
    the GPU-native scanner/gpu_render.py)."""
    cq = cq or config.BLUR_CQ
    seg_bounds = [(float(s["start"]), float(s["end"])) for s in segments]
    starts = [a for a, _ in seg_bounds]
    kf_index = {si: [e[0] for e in entries]
                for si, entries in per_seg.items()}

    frames_out, frames_blurred, codec = _render_range(
        video_path, window[0], window[1], w, h, aw, ah, per_seg, seg_bounds,
        starts, kf_index, cq, config.BLUR_CODECS,
        out_path=out_path)
    return {"frames": frames_out, "frames_blurred": frames_blurred,
            "codec": codec}


def run_blur(video_path: str, segments: list[dict], out_path: str, *,
             policy: Policy, backend: str = "trt",
             window: tuple[float, float] | None = None,
             cq: int = None) -> dict:
    """Full stage-C pipeline: analysis (masks) -> render.

    Full renders take the GPU-native path (NVDEC -> torch NV12 zero-copy ->
    in-place solid-black fill -> NVENC, audio passthrough, one continuous
    pass — no cut boundaries). A `window` takes the pipe-path preview
    render."""
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

    # analysis is done: free its TRT engines so the render has the whole
    # GPU (render re-imports nothing from analysis)
    del detector, segmenter
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if window is not None:
        render = render_pass(video_path, out_path, segments, per_seg, w, h,
                             aw, ah, window, cq=cq)
    else:
        from .gpu_render import render_gpu
        # subtitle-bearing sources render from a sub-free copy (video+audio
        # stay bit-exact; see strip_subtitles)
        render_path = (strip_subtitles(video_path, Path(out_path).parent)
                       if count_subtitle_streams(video_path) else video_path)
        try:
            render = render_gpu(render_path, out_path, segments, per_seg,
                                w, h, aw, ah, cq=cq)
        finally:
            if render_path != video_path:
                Path(render_path).unlink(missing_ok=True)

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
