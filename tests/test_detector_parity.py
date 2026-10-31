"""Parity: our ORT backend vs the reference nudenet 3.4.2 package on real
frames. Requires models/nudenet_320n_v1.onnx and a sample video (any
visiontouch sports clip); skips otherwise. Runs CPU-only, torch-free."""

import glob
import os
import subprocess
import tempfile
from collections import Counter

import pytest

SAMPLE_GLOBS = [
    "/home/rohan/repos/Yukin-Global-NG/visiontouch/netball_test_clips/*.mp4",
    "/home/rohan/repos/Yukin-Global-NG/visiontouch/SampleFootballVideos/**/*.mp4",
    "/home/rohan/repos/Yukin-Global-NG/visiontouch/SampleFootballVideos/**/*.webm",
    "/home/rohan/repos/Yukin-Global-NG/visiontouch/NFL-Videos/**/*.mp4",
    "/home/rohan/repos/Yukin-Global-NG/visiontouch/NFL-Videos/**/*.webm",
]

MODEL = "models/nudenet_320n_v1.onnx"


def _sample_video():
    for g in SAMPLE_GLOBS:
        hits = sorted(glob.glob(g, recursive=True))
        if hits:
            return hits[0]
    return None


pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL) or _sample_video() is None,
    reason="weights or sample video not present")


def _extract_frames(video, n=3, size="640x360"):
    out_dir = tempfile.mkdtemp(prefix="parity_")
    for i in range(n):
        t = 10 + i * 30
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", str(t), "-i", video, "-frames:v", "1",
             "-vf", f"scale={size}", f"{out_dir}/f{i}.png"],
            check=True)
    return sorted(glob.glob(f"{out_dir}/*.png"))


def test_parity_with_reference_package():
    import cv2
    from nudenet import NudeDetector
    from scanner.detector import NudeNetDetector

    ref = NudeDetector(model_path=os.path.abspath(MODEL))
    ours = NudeNetDetector("320n", backend="ort")

    frames = _extract_frames(_sample_video())
    assert frames, "no frames extracted"

    rgbs = []
    for path in frames:
        bgr = cv2.imread(path)
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    our_results = ours.detect_frames(rgbs)
    for path, ours_dets in zip(frames, our_results):
        ref_dets = ref.detect(path)
        ref_bag = Counter((d["class"], round(d["score"], 2)) for d in ref_dets)
        our_bag = Counter((d.label, round(d.score, 2)) for d in ours_dets)
        # labels+scores must match (boxes checked spot-wise below)
        assert our_bag == ref_bag, f"{path}: ref={ref_dets} ours={ours_dets}"
        ref_by_label = {}
        for d in ref_dets:
            ref_by_label.setdefault(d["class"], []).append(d)
        for od in ours_dets:
            rb = ref_by_label.get(od.label, [])
            assert any(
                all(abs(a - b) <= 2 for a, b in zip(od.box, (d["box"][0], d["box"][1], d["box"][2], d["box"][3])))
                for d in rb), f"box mismatch for {od.label}: {od.box} vs {rb}"
