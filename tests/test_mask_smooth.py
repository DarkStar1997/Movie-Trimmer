"""Temporal mask smoothing (mask.temporal_step + keyframe_masks_for
smoothing path): dropouts bridge, edges stabilise, masks still track."""
from __future__ import annotations

import numpy as np

from scanner import config
from scanner.mask import keyframe_masks_for, temporal_step


def box(w, h, x0, y0, s=40):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y0 + s, x0:x0 + s] = True
    return m


W, H = 320, 180


def test_moving_box_stays_covered():
    """Union with the eroding carry covers the box's current position
    every frame — a moving person is never left partially unmasked."""
    prev = None
    for i in range(30):
        raw = box(W, H, 20 + 3 * i, 60)          # 3 px/frame motion
        out = temporal_step(raw, prev, dilate_px=4, smooth_px=3)
        assert out[60:100, 20 + 3 * i:60 + 3 * i].all()
        prev = out


def test_static_converges_no_runaway_growth():
    prev = None
    areas = []
    for _ in range(12):
        out = temporal_step(box(W, H, 100, 60), prev, 4, 3)
        areas.append(int(out.sum()))
        prev = out
    assert areas[-1] == areas[-2]               # converged
    assert areas[-1] < areas[1] * 1.1           # no runaway halo


def test_dropout_bridged_and_decays():
    """A detection dropout keeps the mask on (eroding), then fully
    recovers when detection returns."""
    prev = temporal_step(box(W, H, 100, 60), None, 4, 3)
    base = int(prev.sum())
    m = temporal_step(np.zeros((H, W), bool), prev, 4, 3)   # dropout
    assert 0 < int(m.sum()) < base                           # carried
    for _ in range(40):                                      # long dropout
        prev = m
        m = temporal_step(np.zeros((H, W), bool), prev, 4, 3)
    assert int(m.sum()) == 0                                 # decayed
    back = temporal_step(box(W, H, 100, 60), m, 4, 3)        # recovered
    assert back.any()


def test_smooth_off_is_plain_dilation():
    raw = box(W, H, 100, 60)
    prev = temporal_step(box(W, H, 200, 60), None, 4, 3)
    from scanner.mask import dilate_mask
    assert np.array_equal(temporal_step(raw, prev, 4, 0),
                          dilate_mask(raw, 4))
    assert np.array_equal(temporal_step(raw, None, 4, 3),
                          dilate_mask(raw, 4))


class _StubSegmenter:
    """all_persons returns a moving person mask, None on dropout frames."""

    def __init__(self, drop_frames=()):
        self.i = -1
        self.drop = set(drop_frames)

    def all_persons(self, frame):
        self.i += 1
        if self.i in self.drop:
            return None
        return box(W, H, 30 + 2 * self.i, 60)


def _frames(n):
    return [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n)]


def test_vlm_dropout_no_center_square_jump():
    """Mid-segment segmenter dropout must ride the eroding carry, not
    jump to the MASK_VLM_CENTER_FRAC fallback square (0.7 center box)."""
    seg = _StubSegmenter(drop_frames={5, 6})
    state = {}
    masks = keyframe_masks_for(
        _frames(10), segmenter=seg, dilate_px=4,
        vlm_score=0.9, src_label="VLM", state=state, smooth_px=3)
    areas = [int(m.sum()) for m, _, _ in masks]
    # steady-state person box ~40x40 + dilation; a center-square jump
    # would be ~(0.7*320)*(0.7*180) = ~35k px — 2x+ the person area
    assert max(areas) < 2.2 * min(a for a in areas if a > 0)
    assert all(a > 0 for a in areas)             # never flickered off


def test_vlm_first_frame_nobody_uses_center_fallback():
    """Nobody detected from frame 0 (no carry alive): the center
    fallback still applies — err wide when the source saw nudity."""
    class _Empty:
        def all_persons(self, frame):
            return None

    masks = keyframe_masks_for(
        _frames(2), segmenter=_Empty(), dilate_px=4,
        vlm_score=0.9, src_label="VLM", state={}, smooth_px=3)
    frac = config.MASK_VLM_CENTER_FRAC
    expect = (frac * W) * (frac * H)
    assert all(expect <= int(m.sum()) <= expect * 1.3 for m, _, _ in masks)


def test_carry_persists_across_batches():
    """state carries sm_prev between keyframe_masks_for calls
    (analysis_pass batches frames): the second batch continues the
    smoothing chain instead of restarting it."""
    state = {}
    for _ in (0, 1):
        frames = _frames(4)
        keyframe_masks_for(
            frames, segmenter=_StubSegmenter(), dilate_px=4,
            vlm_score=0.9, src_label="VLM", state=state, smooth_px=3)
    assert state.get("sm_prev") is not None and state["sm_prev"].any()
