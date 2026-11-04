"""Video decode: ffmpeg NVDEC pipe (primary) and OpenCV (fallback).

The ffmpeg path decodes on GPU when available, drops to the analysis frame
rate *before* scaling (so swscale only touches sampled frames), and streams
fixed-size RGB24 frames over a pipe. The OpenCV path is the portable CPU
fallback and mirrors the visiontouch video_pipeline pattern.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class DecoderStalled(RuntimeError):
    """ffmpeg stopped producing frames (NVDEC wedge). The caller may retry
    once with use_hwaccel=False."""


def probe_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def probe_dimensions(path: str) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def analysis_size(src_w: int, src_h: int, target_w: int = 640) -> tuple[int, int]:
    """Aspect-preserving analysis frame size: width pinned to target_w for
    landscape sources (height rounded to even for yuv), height pinned for
    portrait. Scope (2.39:1) films keep their aspect."""
    if src_w >= src_h:
        w = target_w
        h = max(2, int(round(src_h * target_w / src_w / 2)) * 2)
    else:
        h = target_w
        w = max(2, int(round(src_w * target_w / src_h / 2)) * 2)
    return w, h


class FFmpegPipeDecoder:
    """Yields RGB uint8 frames of exactly (width, height) at `fps`.

    Frame k arrives at timestamp k / fps (the fps filter's clock), which the
    pipeline uses as the sample timeline.
    """

    def __init__(self, path: str, *, fps: float, width: int, height: int,
                 use_hwaccel: bool = True, start: float | None = None,
                 duration: float | None = None, exact: bool = False,
                 stall_timeout_s: float | None = None):
        self.path = path
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * 3
        self.start = float(start) if start is not None else None
        self.duration = float(duration) if duration is not None else None
        # exact=True: no fps filter — frames pass through 1:1 (render path,
        # where frame counts must match source packet counts exactly)
        self.exact = exact
        self.stall_timeout_s = (config.MASK_STALL_TIMEOUT_S
                                if stall_timeout_s is None else stall_timeout_s)
        self._stalled = False
        self._watchdog: threading.Thread | None = None
        self._last_read = time.monotonic()
        self._cmd_base = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
        ]
        if self.start is not None:
            # input seeking: keyframe seek + decode-forward, frame-accurate
            self._cmd_base += ["-ss", f"{self.start:.3f}"]
        if self.duration is not None:
            self._cmd_base += ["-t", f"{self.duration:.3f}"]
        if use_hwaccel:
            self._cmd_base += ["-hwaccel", "cuda"]
        self._proc: subprocess.Popen | None = None
        self.frames_yielded = 0

    def _full_cmd(self) -> list[str]:
        args = self._cmd_base + ["-i", self.path]
        if not self.exact:
            args += ["-vf",
                     f"fps={self.fps},scale={self.width}:{self.height}"]
        return args + [
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]

    def __enter__(self) -> "FFmpegPipeDecoder":
        # idempotent: open_decoder() already entered; a second enter (the
        # caller's `with`) must not spawn a second ffmpeg — the first would
        # be orphaned (self._proc replaced) and leak ~0.5 GiB VRAM per scan
        if self._proc is not None:
            return self
        try:
            self._start()
        except RuntimeError:
            if "-hwaccel" in self._cmd_base:
                logger.warning("[decode] NVDEC failed, falling back to CPU decode")
                self._cmd_base = [c for c in self._cmd_base
                                  if c not in ("-hwaccel", "cuda")]
                self._start()
            else:
                raise
        return self

    def _start(self) -> None:
        proc = subprocess.Popen(self._full_cmd(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # Give an immediately-failing command (bad file, no CUDA) time to die
        # before we hand out the pipe.
        time.sleep(0.5)
        if proc.poll() not in (None, 0):
            err = proc.stderr.read().decode(errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg exited early: {err}")
        self._proc = proc

    def __exit__(self, *exc) -> None:
        self.close()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames; raises DecoderStalled if ffmpeg wedges (NVDEC quirk:
        the process survives but stops producing output — kill + fail loud
        instead of blocking forever)."""
        assert self._proc is not None and self._proc.stdout is not None
        self._start_watchdog()
        try:
            while True:
                buf = self._proc.stdout.read(self.frame_size)
                self._last_read = time.monotonic()
                if not buf:
                    break
                if len(buf) < self.frame_size:
                    # truncated tail frame — drop it
                    break
                self.frames_yielded += 1
                yield np.frombuffer(buf, dtype=np.uint8).reshape(
                    self.height, self.width, 3)
        finally:
            self._stop_watchdog()
        if self._stalled:
            raise DecoderStalled(
                f"decoder produced no frame for {self.stall_timeout_s}s "
                f"(NVDEC wedge?) — killed. {self.path} @ {self.start}")

    def _start_watchdog(self) -> None:
        if self._watchdog is not None:
            return

        def _watch() -> None:
            proc = self._proc
            while proc is not None and proc.poll() is None:
                time.sleep(2.0)
                if time.monotonic() - self._last_read > self.stall_timeout_s:
                    self._stalled = True
                    logger.warning("[decode] stalled %.0fs — killing ffmpeg",
                                   self.stall_timeout_s)
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    return

        self._watchdog = threading.Thread(target=_watch, daemon=True)
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        # daemon thread: exits on its own once the process is dead
        self._watchdog = None

    def close(self) -> None:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)   # reap: never leave orphans
                    except subprocess.TimeoutExpired:
                        logger.warning("[decode] ffmpeg refused to die")


class OpenCVDecoder:
    """CPU fallback: cv2 grab/retrieve, keeping every Nth frame."""

    def __init__(self, path: str, *, fps: float, width: int, height: int):
        import cv2  # local import: heavy
        self._cv2 = cv2
        self.path = path
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self._step = max(1, round(src_fps / self.fps))
        # effective sampling fps after integer stepping
        self.effective_fps = src_fps / self._step
        self.frames_yielded = 0

    def frames(self) -> Iterator[np.ndarray]:
        cap, cv2 = self.cap, self._cv2
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % self._step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame = cv2.resize(frame, (self.width, self.height))
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames_yielded += 1
            idx += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "OpenCVDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_decoder(path: str, *, fps: float, width: int, height: int,
                 prefer_ffmpeg: bool = True) -> tuple:
    """Return (context_manager_decoder, effective_fps). Prefers the ffmpeg
    pipe (NVDEC when present), falls back to OpenCV."""
    if prefer_ffmpeg and shutil.which("ffmpeg"):
        dec = FFmpegPipeDecoder(path, fps=fps, width=width, height=height)
        try:
            dec.__enter__()
            return dec, dec.fps
        except Exception as e:  # noqa: BLE001 — fall back
            logger.warning("[decode] ffmpeg pipe unusable (%s); OpenCV fallback", e)
    dec = OpenCVDecoder(path, fps=fps, width=width, height=height)
    dec.__enter__()
    return dec, dec.effective_fps
