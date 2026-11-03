"""On-demand llama-server supervisor for the all-in-one container.

The VLM (Qwen3-VL arbiter) owns ~20GB of VRAM — held for the whole
container lifetime it leaves a 24GB GPU with ~3.5GB of headroom for
EVERY other stage, most of which never touch it. This module gives it
a deterministic lifecycle instead:

    acquire()  — spawn llama-server (unless already up), wait for its
                 /health, and pin it. Called by run_arbiter BEFORE frame
                 extraction so the model load overlaps ffmpeg.
    release()  — un-pin; the LAST releaser terminates the server and
                 frees the VRAM. Called in run_arbiter's finally.

Refcounted: MAX_CONCURRENT_SCANS mask jobs never acquire, concurrent
arbiter runs share one server and only the final release stops it.
Bring-up failure raises VlmUnavailable — the caller (run_arbiter)
records it as an error block; a scan NEVER silently degrades, the
failure is loud in the result JSON.

Process management lives in the API process (uvicorn runs single).
Host/test environments fake Popen; nothing imports tensorrt here.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_refs: int = 0


class VlmUnavailable(RuntimeError):
    """llama-server could not be brought up (load failed / timeout)."""


def _base_url() -> str:
    return config.ARBITER_BASE_URL.rstrip("/")


def _healthy(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{_base_url()}/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _argv() -> list[str]:
    env = os.environ
    model = env.get("MODEL", "")
    mmproj = env.get("MMPROJ", "")
    if not model or not mmproj:
        raise VlmUnavailable(
            "MODEL/MMPROJ env not set (all-in-one image provides them)")
    argv = ["/opt/llama/llama-server",
            "-m", model,
            "--mmproj", mmproj,
            "-ngl", "999",
            "-fa", "on",
            "-c", env.get("CTX", "8192"),
            "-np", env.get("PARALLEL", "4"),
            "--host", "127.0.0.1",
            "--port", _base_url().rsplit(":", 1)[-1]]
    if env.get("DRAFT") and os.path.isfile(env["DRAFT"]):
        argv += ["--model-draft", env["DRAFT"],
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]
    extra = env.get("EXTRA_ARGS", "").split()
    return argv + extra


def acquire(timeout_s: float | None = None) -> None:
    """Pin the VLM up (spawning + health-gating if needed). Raises
    VlmUnavailable on failure; on failure nothing stays pinned."""
    global _proc, _refs
    timeout_s = config.VLM_START_TIMEOUT_S if timeout_s is None else timeout_s
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _refs += 1
            return
        argv = _argv()
        logger.info("[vlm] starting llama-server (load may take minutes"
                    " cold): %s", " ".join(argv[:2]))
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if proc.poll() is not None:
                proc.wait()
                raise VlmUnavailable(
                    f"llama-server exited during load (rc={proc.returncode})")
            if _healthy():
                _proc, _refs = proc, _refs + 1
                logger.info("[vlm] healthy in %.0fs (pinned by %d)",
                            time.monotonic() - t0, _refs)
                return
            time.sleep(2)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        raise VlmUnavailable(
            f"llama-server health timeout after {timeout_s:.0f}s")


def release() -> None:
    """Un-pin; the last releaser stops the server and frees the VRAM."""
    global _proc, _refs
    with _lock:
        _refs = max(0, _refs - 1)
        if _refs > 0 or _proc is None:
            return
        proc, _proc = _proc, None
        logger.info("[vlm] last release — stopping llama-server")
        proc.terminate()
        try:
            proc.wait(timeout=config.VLM_STOP_GRACE_S)
        except subprocess.TimeoutExpired:  # pragma: no cover
            logger.warning("[vlm] graceful stop timed out; killing")
            proc.kill()
        logger.info("[vlm] stopped — VRAM freed")


def status() -> dict:
    with _lock:
        return {
            "running": _proc is not None and _proc.poll() is None,
            "pinned_by": _refs,
            "base_url": _base_url(),
        }


@atexit.register
def _shutdown() -> None:  # pragma: no cover — process-exit safety net
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc, _refs = None, 0
