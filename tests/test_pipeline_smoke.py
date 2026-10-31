"""End-to-end pipeline smoke test (host, CPU, ORT backend, no torch).
Generates a tiny synthetic clip and runs the full decode -> detect ->
policy -> smoothing chain. Catches integration breakage (imports, decode
wiring, result shape) that unit tests miss."""

import glob
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not glob.glob("models/nudenet_320n_v1.onnx"),
    reason="weights not present")


def test_run_scan_end_to_end():
    from scanner.pipeline import run_scan
    from scanner.policy import Policy
    from scanner.temporal import SmoothingConfig

    with tempfile.TemporaryDirectory() as td:
        clip = f"{td}/smoke.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=640x270:rate=24",
             "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
            check=True)
        res = run_scan(clip, policy=Policy(), smoothing=SmoothingConfig(),
                       sample_fps=2.0, model="320n", backend="ort")

    assert res["duration"] == pytest.approx(4.0, abs=0.2)
    assert res["frames_scanned"] == pytest.approx(8, abs=2)
    assert res["segments"] == []            # synthetic content: no triggers
    assert res["review_segments"] == []
    assert res["timeline"] and all(b["hits"] == 0 for b in res["timeline"])
    assert res["model"] == "320n" and res["backend"].startswith("ort")
    assert res["wall_time_s"] > 0
    # stage A on by default; stage B opt-in (absent unless requested)
    assert res["tracking"]["enabled"] is True
    assert res["tracking"]["bridged_samples"] == 0
    assert res["arbiter"] is None


def test_open_decoder_double_enter_single_ffmpeg():
    """open_decoder() enters; the caller's `with` enters again. The second
    enter must not spawn a second ffmpeg — the first would be orphaned
    (self._proc replaced, never killed) and leak ~0.5 GiB VRAM per scan."""
    import os
    import shutil
    import time

    import cv2
    import numpy as np

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not present")

    with tempfile.TemporaryDirectory() as td:
        clip = f"{td}/smoke.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
             "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
            check=True)
        from scanner.decode import FFmpegPipeDecoder

        dec = FFmpegPipeDecoder(clip, fps=4.0, width=320, height=180,
                                use_hwaccel=False)
        dec.__enter__()
        first_pid = dec._proc.pid
        try:
            with dec:                      # the double-enter under test
                assert dec._proc.pid == first_pid
                for i, _ in enumerate(dec.frames()):
                    if i >= 4:
                        break
        finally:
            dec.close()
        assert dec._proc is None
        # the exact pid is gone (close reaped it)
        def alive(pid):
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
        deadline = time.monotonic() + 5
        while alive(first_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(first_pid)
