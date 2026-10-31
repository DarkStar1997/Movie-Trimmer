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
