"""Stage C tests: personseg decode math, mask helpers, end-to-end render.

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

from scanner.mask import (blackout_rgb, box_mask, dilate_mask,
                          explicit_boxes, intersect_with_box, pad_box,
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
    mask, area, score, pbbox = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, match_box)
    assert mask is not None and score > 0.9
    assert mask.shape == (40, 60)
    assert area > 0.9                                 # mask fully positive
    # the person box (6,10,24,30) overlaps match_box enough to count as
    # matched -> callers may use this mask as the full-person mask
    assert pbbox is not None
    assert pbbox.tolist() == pytest.approx([6.0, 10.0, 24.0, 30.0],
                                           abs=1e-3)


def test_decode_containment_match():
    """A part box (breast/genitalia) sits INSIDE the person box: IoU is low
    by construction, but containment >=60% must still count as a match."""
    labels, dets, masks = _fake_outs()
    labels[0, 1, PERSON_CLASS] = 3.0
    dets[0, 1] = [0.5, 0.5, 0.6, 0.8]               # person: 12..48 x 4..36
    masks[0, 1] = 5.0
    part = np.asarray([22.0, 14.0, 34.0, 24.0])      # fully inside the person
    mask, area, score, pbbox = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, part)
    assert mask is not None and pbbox is not None    # containment matched


def test_decode_unmatched_returns_no_bbox():
    """When nothing matches the box the caller must be told: the returned
    mask is just the top-scoring person and must NOT be used as a
    full-person mask for the box's subject (wrong-person risk)."""
    labels, dets, masks = _fake_outs()
    labels[0, 3, PERSON_CLASS] = 3.0
    dets[0, 3] = [0.5, 0.5, 0.2, 0.2]               # person: 24..36 x 16..24
    masks[0, 3] = 5.0
    far = np.asarray([0.0, 0.0, 10.0, 10.0])         # no overlap, no contain
    mask, area, score, pbbox = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, far)
    assert mask is not None and pbbox is None


def test_decode_no_person_returns_none():
    labels, dets, masks = _fake_outs()
    labels[0, 2, 5] = 4.0        # high score, but class 5 (not person)
    mask, area, score, pbbox = decode(
        {"dets": dets, "labels": labels, "masks": masks},
        {"dets": "dets", "labels": "labels", "masks": "masks"},
        40, 60, None)
    assert mask is None and score == 0.0 and pbbox is None


# --- mask helpers -------------------------------------------------------------------

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


def test_blackout_only_inside_mask():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, :8] = True               # silhouette-shaped region
    out = blackout_rgb(frame, mask)
    # inside: solid black, nothing legible
    assert (out[8:16, :8] == 0).all()
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


# --- keyframe mask semantics (full person / fallback / VLM / tracking) ---------------

class _FakeSeg:
    """Scripted person segmenter: pops scripted infer() results."""
    def __init__(self, results):
        self.results = list(results)      # [(mask, bbox) | None per call]
    def infer(self, frame, match_box=None):
        mask, bbox = self.results.pop(0)
        return mask, (float(mask.mean()) if mask is not None else 0.0), \
            0.9, bbox
    def all_persons(self, frame):
        mask, _ = self.results.pop(0)
        return mask


def _person_mask(h=40, w=60):
    m = np.zeros((h, w), dtype=bool)
    m[4:36, 6:48] = True                  # a body-shaped region
    return m


def test_keyframe_masks_full_person_not_part_intersection():
    seg = _FakeSeg([(_person_mask(), np.asarray([6.0, 4.0, 48.0, 36.0]))])
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    dets = [Detection("FEMALE_BREAST_EXPOSED", 0.60, (20, 15, 10, 10))]
    from scanner.mask import keyframe_masks_for
    (mask, label, score), = keyframe_masks_for(
        [dets], [frame], Policy(), seg, dilate_px=1)
    # FULL person mask: far larger than the part box (20,15,30,25)
    assert mask.sum() == _person_mask().sum()
    assert label == "FEMALE_BREAST_EXPOSED" and score == 0.60


def test_keyframe_box_fallback_when_unmatched():
    # person found but does NOT match the box -> padded box rect, never the
    # wrong person's mask. box (2,2,8,8) padded 0.35/side -> (0,0,10,10)
    seg = _FakeSeg([(_person_mask(), None)])
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    dets = [Detection("FEMALE_BREAST_EXPOSED", 0.60, (2, 2, 6, 6))]
    from scanner.mask import keyframe_masks_for
    (mask, _, _), = keyframe_masks_for(
        [dets], [frame], Policy(), seg, dilate_px=1)
    assert mask.sum() == 10 * 10
    assert mask[:10, :10].all()
    assert mask.sum() != _person_mask().sum()   # NOT the wrong person


def test_keyframe_tracking_bridges_detector_flicker():
    # frame 1: detector fires, person matches (bbox tracked)
    # frame 2: detector silent -> tracked bbox re-matched, full mask stays
    # frame 3: tracker loses the person -> empty mask, tracking resets
    seg = _FakeSeg([
        (_person_mask(), np.asarray([6.0, 4.0, 48.0, 36.0])),
        (_person_mask(), np.asarray([6.0, 4.0, 48.0, 36.0])),
        (_person_mask(), None),
    ])
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    dets = [Detection("FEMALE_BREAST_EXPOSED", 0.60, (20, 15, 10, 10))]
    from scanner.mask import keyframe_masks_for
    m1, m2, m3 = keyframe_masks_for(
        [dets, [], []], [frame] * 3, Policy(), seg, dilate_px=1)
    assert m1[0].sum() == _person_mask().sum()
    assert m2[0].sum() == _person_mask().sum()   # bridged: still masked
    assert m2[1] == "FEMALE_BREAST_EXPOSED"      # overlay label carried
    assert not m3[0].any()                       # person gone: no mask


def test_keyframe_vlm_source_masks_all_persons():
    seg = _FakeSeg([(_person_mask(), None)])
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    from scanner.mask import keyframe_masks_for
    (mask, label, score), = keyframe_masks_for(
        [[]], [frame], Policy(), seg, dilate_px=1,
        vlm_source=True, vlm_score=0.98)
    assert mask.sum() == _person_mask().sum()
    assert label == "VLM" and score == 0.98


def test_keyframe_vlm_center_fallback():
    seg = _FakeSeg([(None, None)])               # segmenter finds nobody
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    from scanner.mask import keyframe_masks_for
    from scanner import config
    (mask, label, score), = keyframe_masks_for(
        [[]], [frame], Policy(), seg, dilate_px=1,
        vlm_source=True, vlm_score=0.95)
    frac = config.MASK_VLM_CENTER_FRAC
    expect = int(round(60 * frac)) * int(round(40 * frac))
    assert abs(mask.sum() - expect) < 8          # centered rect ~ frac^2 area
    assert mask[20, 30] and not mask[0, 0]       # center yes, corner no


# --- warning overlay -----------------------------------------------------------------

def test_draw_warning_red_text_top_left():
    from scanner.mask import draw_warning
    canvas = np.zeros((270, 640, 3), dtype=np.uint8)
    draw_warning(canvas, "FEMALE_BREAST_EXPOSED", 0.87)
    red = (canvas[..., 0] > 180) & (canvas[..., 1] < 90) \
        & (canvas[..., 2] < 90)
    assert red.sum() > 150                        # substantial red text
    assert red[270 // 3:, :].sum() == 0           # confined to the top third
    assert red[:, 640 // 2:].sum() == 0           # confined to the left half
    outline = (canvas[..., 0] < 60) & (canvas[..., 2] < 60) \
        & ~red
    assert outline.sum() > 50                     # black outline present


# --- end-to-end (real ffmpeg chain, mocked masks) -----------------------------------

def test_run_mask_end_to_end(monkeypatch):
    import scanner.mask as mask
    from scanner.mask import run_mask
    from scanner.policy import Policy as P

    with tempfile.TemporaryDirectory() as td:
        clip = f"{td}/src.mp4"
        out = f"{td}/out.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=640x270:rate=24",
             "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
            check=True)

        # fake the two model stages: no detections, center-block mask +
        # label/score for the overlay
        monkeypatch.setattr(
            mask, "keyframe_masks_for",
            lambda results, frames, policy, seg, dilate_px, **kw: [
                (_center_mask(f.shape[0], f.shape[1]),
                 "FEMALE_BREAST_EXPOSED", 0.83) for f in frames])

        res = run_mask(clip, [{"start": 2.0, "end": 4.0}], out,
                       policy=P(), backend="ort",
                       window=(0.5, 5.5))

        # window previews take the pipe path; this host has NVENC (hevc)
        assert res["codec"] in ("hevc_nvenc", "h264_nvenc", "libx264")
        assert res["frames_masked"] > 0
        assert res["keyframes"] > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name", "-of", "json",
             out], capture_output=True, text=True, check=True)
        import json
        meta = json.loads(probe.stdout)
        assert 5.0 < float(meta["format"]["duration"]) < 7.0

        # every flagged frame carries the red warning: grab t=3.0 (inside
        # the segment) and look for red text pixels in the top-left corner
        png = f"{td}/probe.png"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", "2.5", "-i", out, "-copyts",
             "-vf", r"select=between(t\,2.99\,3.01)",
             "-fps_mode", "passthrough", "-frames:v", "1", "-y", png],
            check=True)
        import cv2
        img = cv2.imread(png)                     # BGR: red = channel 2
        red = ((img[..., 2] > 180) & (img[..., 1] < 90)
               & (img[..., 0] < 90))
        # testsrc2 itself contains red bars, so assert on the top-left
        # quadrant where the overlay text lives
        assert red[:270 // 3, :640 // 2].sum() > 100, \
            "no red warning overlay on flagged frame"


def _center_mask(h, w):
    m = np.zeros((h, w), dtype=bool)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return m


def test_analysis_pass_on_segment_delivers_entries_inline(tmp_path):
    """The overlap contract (2026-11-05): on_segment(si, entries) must
    deliver each segment's FINAL entries during the run — analysis_pass's
    returned dict only exists once every segment is done, so overlap
    consumers must take the data from the callback. The first overlap
    build consumed a still-empty shared dict at callback time and the
    render shipped those segments unmasked (caught by the verify gate:
    seg 0 masked 7,676 px vs the baseline's 703,154)."""
    import subprocess

    from scanner.mask import analysis_pass

    clip = str(tmp_path / "overlap.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x270:rate=24",
         "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip],
        check=True)
    aw, ah = 160, 90
    body = np.zeros((ah, aw), dtype=bool)
    body[20:60, 40:100] = True

    class _StaticSeg:
        def all_persons(self, frame):
            return body

    segs = [{"start": 0.5, "end": 1.5, "labels": {"arbiter": 1}},
            {"start": 2.5, "end": 3.5, "labels": {"arbiter": 1}}]
    events: list = []
    got: dict = {}

    def on_segment(si, entries):
        events.append(("cb", si))
        got[si] = entries

    ret = analysis_pass(clip, segs, Policy(), None, _StaticSeg(),
                        aw, ah, on_segment=on_segment)
    events.append(("ret", None))

    # callbacks fired in segment order, all before the return existed
    assert [si for kind, si in events[:-1] if kind == "cb"] == [0, 1]
    assert events[-1] == ("ret", None)
    # the delivered entries ARE the returned entries (same objects), and
    # they are non-empty at callback time — the render may consume them
    # the moment the callback fires
    assert set(got) == set(ret) == {0, 1}
    for si, entries in ret.items():
        assert got[si] is entries
        assert len(entries) > 0
