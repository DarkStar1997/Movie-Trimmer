"""Bridge-noise demotion (pipeline) + render fail-safe (mask).

Both behaviors validated on Gone Girl 2026-11-03: 8 flagged segments
rendered as no-ops — 7 pure track-bridge (no passing label, sub-threshold
held flicker) and 1 labeled segment whose 4fps scan detection did not
reproduce at the 8fps mask re-detection.
"""
import numpy as np

from scanner.mask import failsafe_entries
from scanner.pipeline import demote_bridge_only


def _seg(start, end, labels=None, score=0.3, bridged=0):
    return {"start": start, "end": end, "mean_score": score,
            "labels": dict(labels or {}), "bridged_samples": bridged}


# --- pipeline: pure-bridge noise demoted to review -----------------------------

def test_bridge_only_segments_demoted_to_review():
    segs = [
        _seg(0.0, 5.0, {"FEMALE_BREAST_EXPOSED": 2}),   # real
        _seg(10.0, 14.0),                               # pure bridge
        _seg(20.0, 26.0, {"ARBITER": 1}),               # arbiter-born
        _seg(30.0, 33.0, bridged=6),                    # pure bridge
    ]
    review = [_seg(50.0, 55.0, {"BELLY_EXPOSED": 1})]
    kept, rev = demote_bridge_only(segs, review)
    assert [s["start"] for s in kept] == [0.0, 20.0]
    # demoted land in review, sorted, labeled with their provenance
    assert [s["start"] for s in rev] == [10.0, 30.0, 50.0]
    assert rev[0]["labels"] == {"TRACK_BRIDGE": 0}
    assert rev[1]["labels"] == {"TRACK_BRIDGE": 6}
    # originals untouched otherwise
    assert kept[0]["labels"] == {"FEMALE_BREAST_EXPOSED": 2}


def test_demotion_noop_when_all_labeled():
    segs = [_seg(0.0, 5.0, {"FEMALE_BREAST_EXPOSED": 2})]
    kept, rev = demote_bridge_only(segs, [])
    assert kept is segs and rev == []


# --- mask: full-frame fail-safe for labeled-but-undetectable -------------------

def test_failsafe_full_frame_for_labeled_segment():
    seg = _seg(100.0, 105.0, {"MALE_GENITALIA_EXPOSED": 1}, score=0.52)
    ents = failsafe_entries(seg, aw=640, ah=360)
    assert len(ents) == 2
    t0, packed, label, score = ents[0]
    assert t0 == 100.0 and ents[1][0] == 105.0
    assert label == "MALE_GENITALIA_EXPOSED" and score == 0.52
    m = np.unpackbits(np.frombuffer(packed, dtype=np.uint8),
                      count=360 * 640).reshape(360, 640)
    assert m.all()                       # full frame — nothing legible


def test_failsafe_empty_for_unlabeled_segment():
    assert failsafe_entries(_seg(10.0, 14.0), 640, 360) == []
