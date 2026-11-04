"""End-to-end pipeline smoke test (host, CPU, no weights needed).
Generates a tiny synthetic clip and runs the full decode -> person-gate
feed chain (gate/arbiter off — no engines). Catches integration breakage
(imports, decode wiring, result shape) that unit tests miss."""

import subprocess
import tempfile

import pytest


def _clip(td, name="smoke.mp4", t="4"):
    path = f"{td}/{name}"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x270:rate=24",
         "-t", t, "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True)
    return path


def test_run_scan_end_to_end():
    from scanner.pipeline import run_scan
    from scanner.temporal import SmoothingConfig

    with tempfile.TemporaryDirectory() as td:
        clip = _clip(td)
        res = run_scan(clip, smoothing=SmoothingConfig(), sample_fps=2.0)

    assert res["duration"] == pytest.approx(4.0, abs=0.2)
    assert res["segments"] == []            # synthetic content: no sources
    # detector-era record fields, kept empty for consumer compatibility
    assert res["review_segments"] == []
    assert res["timeline"] == [] and res["label_stats"] == {}
    assert res["wall_time_s"] > 0
    # stage B opt-in (absent unless requested); gate follows the arbiter
    assert res["arbiter"] is None
    assert res["person_gate"] == {"enabled": False, "reason": "disabled"}
    assert res["stages"]["decode_gate_s"] > 0


def test_run_scan_crowd_end_to_end(monkeypatch, tmp_path):
    """crowd=True must flow through run_scan: block present, scene-guide
    ranges merged as first-class segments (fetch monkeypatched — host
    tests stay network-free). Guards the crowd wiring that a unit test
    on crowd.py alone cannot see."""
    from pathlib import Path

    import scanner.crowd as crowd_mod
    from scanner.pipeline import run_scan
    from scanner.temporal import SmoothingConfig

    fixture = (Path(__file__).parent / "fixtures" /
               "cringemdb_exmachina.html").read_text()
    monkeypatch.setattr(crowd_mod, "fetch_page",
                        lambda url, timeout: fixture)
    monkeypatch.setattr("scanner.config.CROWD_CACHE_DIR", str(tmp_path))

    with tempfile.TemporaryDirectory() as td:
        clip = _clip(td, "Ex.Machina.2015.smoke.mp4")
        res = run_scan(clip, smoothing=SmoothingConfig(), sample_fps=2.0,
                       crowd=True)

    cr = res["crowd"]
    assert cr is not None and cr["error"] is None
    assert cr["slug"] == "ex-machina-2015"
    assert len(cr["entries"]) == 4 and len(cr["skipped"]) == 1
    crowd_segs = [s for s in res["segments"]
                  if any(str(l).startswith("cringemdb") for l in s["labels"])]
    assert len(crowd_segs) == 4
    assert res["crowd"].get("cached") is not True


def test_open_decoder_double_enter_single_ffmpeg():
    """open_decoder() enters; the caller's `with` enters again. The second
    enter must not spawn a second ffmpeg — the first would be orphaned
    (self._proc replaced, never killed) and leak ~0.5 GiB VRAM per scan."""
    import os
    import shutil
    import time

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not present")

    with tempfile.TemporaryDirectory() as td:
        clip = _clip(td, t="2")
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
