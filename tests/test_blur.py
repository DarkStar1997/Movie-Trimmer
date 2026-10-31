"""Stage C tests: personseg decode math, blur helpers, end-to-end render.

The pure-function tests need no weights. The end-to-end test mocks the
mask source (synthetic clips have no nudity) but runs the real decode ->
packbits -> render -> encode chain, including the libx264 fallback when
NVENC is absent (host). Weight-dependent inference is exercised in the
container validation instead.
"""

import subprocess
import tempfile

import numpy as np
import pytest

from scanner.blur import (box_mask, dilate_mask, explicit_boxes,
                          intersect_with_box, pad_box, pixelate,
                          _writer_args)
from scanner.detector import Detection
from scanner.personseg import PERSON_CLASS, decode, preprocess
from scanner.policy import Policy


# --- personseg -------------------------------------------------------------------

def test_preprocess_shape_and_norm():
    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
    t = preprocess(frame, 432)
    assert t.shape == (1, 3, 432, 432) and t.dtype == np.float32
    # white pixel normalized: (1 - mean) / std
    import scanner.personseg as ps
    expect = (1.0 - ps._MEANS.ravel()) / ps._STDS.ravel()
    assert np.allclose(t[0, :, 0, 0], expect, atol=1e-5)


def _fake_outs(q=6, c=91, h=40, w=60):
    rng = np.random.default_rng(0)
    labels = rng.normal(-6, 1, (1, q, c)).astype(np.float32)  # sigmoid ~0
    dets = rng.uniform(0.05, 0.2, (1, q, 4)).astype(np.float32)
    masks = rng.normal(0, 1, (1, q, 108, 108)).astype(np.float32)
    return labels, dets, masks


def test_decode_selects_matching_person():
    labels, dets, masks = _fake_outs()
    # query 3, class PERSON: dominant score; a person box in the left half
    labels[0, 3, PERSON_CLASS] = 3.0                 # sigmoid ~0.95
    dets[0, 3] = [0.25, 0.5, 0.3, 0.5]               # cx cy w h normalized
    masks[0, 3] = 5.0                                # whole mask positive
    match_box = np.asarray([5.0, 5.0, 25.0, 35.0])   # overlaps left half
    mask, area, score = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, match_box)
    assert mask is not None and score > 0.9
    assert mask.shape == (40, 60)
    assert area > 0.9                                 # mask fully positive


def test_decode_no_person_returns_none():
    labels, dets, masks = _fake_outs()
    labels[0, 2, 5] = 4.0        # high score, but class 5 (not person)
    mask, area, score = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, None)
    assert mask is None and score == 0.0


# --- blur helpers -------------------------------------------------------------------

def test_pad_box_clips():
    out = pad_box(np.asarray([0., 0., 100., 50.]), 0.5, 200, 100)
    # pad 50% of own w/h per side: [-50,-25,150,75] clipped to the frame
    assert out.tolist() == [0.0, 0.0, 150.0, 75.0]


def test_box_mask_and_intersect():
    bm = box_mask(np.asarray([10, 10, 30, 20]), 60, 40)
    assert bm.sum() == 20 * 10
    person = np.zeros((40, 60), dtype=bool)
    person[5:35, 5:40] = True
    inter = intersect_with_box(person, np.asarray([10, 10, 30, 20]))
    assert inter.sum() == 10 * 20
    assert not inter[:10, :].any()


def test_explicit_boxes_filters_by_policy():
    policy = Policy()   # 0.45 triggers, 0.005 area floor at defaults
    dets = [
        Detection("FEMALE_BREAST_EXPOSED", 0.60, (10, 10, 60, 40)),   # ok
        Detection("FEMALE_BREAST_EXPOSED", 0.30, (10, 10, 60, 40)),   # score
        Detection("BUTTOCKS_EXPOSED", 0.90, (0, 0, 2, 2)),            # area
        Detection("BELLY_EXPOSED", 0.90, (10, 10, 60, 40)),           # review
    ]
    boxes = explicit_boxes(dets, policy, 640, 270)
    assert len(boxes) == 1
    assert boxes[0].tolist() == [10, 10, 70, 50]


def test_pixelate_only_inside_mask():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:12] = 255                    # boundary at row 12 — NOT block-aligned
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, :8] = True               # the straddled 8px block rows 8-15
    out = pixelate(frame, mask, 8)
    # the straddled block averages white+black -> strictly between 0 and 255
    assert ((out[8:16, :8] > 0) & (out[8:16, :8] < 255)).all()
    # every pixel outside the mask is untouched
    assert (out[~mask] == frame[~mask]).all()


def test_dilate_mask_grows():
    m = np.zeros((40, 40), dtype=bool)
    m[20, 20] = True
    d = dilate_mask(m, 7)
    assert d.sum() > 1 and d[20, 20]


def test_writer_args():
    assert "-crf" in _writer_args("libx264", 23)
    nv = _writer_args("hevc_nvenc", 26)
    assert "-cq" in nv and "0" in nv           # -b:v 0: no default bitrate cap
    assert "-spatial-aq" in nv and "-tune" in nv


# --- end-to-end (real ffmpeg chain, mocked masks) -----------------------------------

def test_run_blur_end_to_end(monkeypatch):
    import scanner.blur as blur
    from scanner.blur import run_blur
    from scanner.policy import Policy as P

    with tempfile.TemporaryDirectory() as td:
        clip = f"{td}/src.mp4"
        out = f"{td}/out.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=640x270:rate=24",
             "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
            check=True)

        # fake the two model stages: no detections, center-block mask
        monkeypatch.setattr(
            blur, "keyframe_masks_for",
            lambda results, frames, policy, seg, dilate_px: [
                _center_mask(f.shape[0], f.shape[1]) for f in frames])

        res = run_blur(clip, [{"start": 2.0, "end": 4.0}], out,
                       policy=P(), backend="ort",
                       window=(0.5, 5.5))

        # window previews take the pipe path; this host has NVENC (hevc)
        assert res["codec"] in ("hevc_nvenc", "h264_nvenc", "libx264")
        assert res["frames_blurred"] > 0
        assert res["keyframes"] > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name", "-of", "json",
             out], capture_output=True, text=True, check=True)
        import json
        meta = json.loads(probe.stdout)
        assert 5.0 < float(meta["format"]["duration"]) < 7.0


def _center_mask(h, w):
    m = np.zeros((h, w), dtype=bool)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return m
