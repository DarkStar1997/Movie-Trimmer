"""Stage B arbiter tests: planning, verdict parsing, aggregation.

Pure-function coverage only — no torch/transformers on the host.
"""

import pytest

from scanner.arbiter import (PlannedFrame, _cluster_hits, _subtract,
                             aggregate, parse_verdict, plan_expansion,
                             plan_frames, resolve_vlm_path)
from scanner import config
from scanner.temporal import SmoothingConfig


# --- planning --------------------------------------------------------------------

def test_plan_verify_frames_inside_segments():
    segs = [{"start": 100.0, "end": 110.0}, {"start": 300.0, "end": 310.0}]
    planned = plan_frames(600.0, segs)
    verify = [f for f in planned if f.reason == "verify"]
    assert len(verify) == 6                      # 3 per segment
    for f in verify:
        seg = segs[f.seg_idx]
        assert seg["start"] <= f.t <= seg["end"]


def test_plan_windows_outside_segments():
    segs = [{"start": 100.0, "end": 110.0}]
    planned = plan_frames(600.0, segs)
    win = [f for f in planned if f.reason == "window"]
    assert win
    for f in win:
        assert not (100.0 <= f.t <= 110.0)
        assert (90.0 < f.t < 100.0) or (110.0 < f.t < 155.0)


def test_plan_window_clipped_between_adjacent_segments():
    """The after-window of seg0 must stop at seg1's start (not sample inside)."""
    segs = [{"start": 100.0, "end": 110.0}, {"start": 120.0, "end": 130.0}]
    planned = plan_frames(600.0, segs)
    for f in planned:
        if f.reason != "verify":
            for a, b in [(100, 110), (120, 130)]:
                assert not a < f.t < b, f.t


def test_plan_sweep_covers_blind_spots():
    planned = plan_frames(600.0, [])             # no segments at all
    sweep = [f for f in planned if f.reason == "sweep"]
    ts = [f.t for f in sweep]
    assert ts[0] == 0.0
    assert any(abs(t - 300.0) < 1e-6 for t in ts)
    assert max(t for t in ts) <= 600.0


def test_plan_respects_cap_by_thinning_not_truncating():
    # 30 segments x 3 verify = 90 verify + windows + sweep, cap 150
    segs = [{"start": 10.0 + i * 20.0, "end": 14.0 + i * 20.0}
            for i in range(30)]
    planned = plan_frames(600.0, segs, max_frames=150)
    assert len(planned) <= 150
    verify = [f for f in planned if f.reason == "verify"]
    assert len(verify) == 90                     # verify tier always kept
    # thinned windows must still span the whole movie, not just its start
    win = [f.t for f in planned if f.reason == "window"]
    assert win and max(win) > 400.0


def test_plan_empty_duration():
    assert plan_frames(0.0, []) == []


def test_plan_sweep_keeps_guaranteed_budget_share():
    """Regression: with many segments the old verify->windows->sweep order
    starved the sweep to single digits for the WHOLE movie (47 segments on
    a 108-min film left ~9 sweep frames). The sweep is the only global
    blind-spot coverage and now gets its share before windows refine
    already-flagged neighborhoods."""
    # sparse segments: windows cover only their 45s afterglow, leaving the
    # second half of the movie as true blind spots for the sweep
    segs = [{"start": 100.0 + i * 55.0, "end": 104.0 + i * 55.0}
            for i in range(20)]                    # 20 segs x 3 = 60 verify
    planned = plan_frames(3600.0, segs, sweep_every_s=15.0, max_frames=300,
                          sweep_budget_frac=0.5)
    assert len(planned) <= 300
    sweep = [f for f in planned if f.reason == "sweep"]
    # budget after verify = 240; sweep share = min(240, int(300*0.5)) = 150
    assert len(sweep) == 150


def test_plan_sweep_spans_whole_movie_when_capped():
    """A thinned sweep must still stride the full duration, not truncate."""
    segs = [{"start": 100.0 + i * 55.0, "end": 104.0 + i * 55.0}
            for i in range(20)]
    planned = plan_frames(3600.0, segs, sweep_every_s=15.0, max_frames=300)
    ts = [f.t for f in planned if f.reason == "sweep"]
    assert ts[0] <= 15.0 and max(ts) > 3400.0


# --- verdict parsing ---------------------------------------------------------------

def test_parse_plain_json():
    v = parse_verdict('{"explicit": true, "labels": ["FEMALE_BREAST"], "confidence": 0.9}')
    assert v == {"explicit": True, "labels": ["FEMALE_BREAST"], "confidence": 0.9}


def test_parse_fenced_json_with_prose():
    text = 'Sure! Here is the assessment:\n```json\n{"explicit": false, "labels": [], "confidence": 0.2}\n```'
    v = parse_verdict(text)
    assert v["explicit"] is False and v["confidence"] == 0.2


def test_parse_refusal():
    assert parse_verdict("I'm sorry, I can't help with analyzing this image.") is None
    assert parse_verdict("I cannot assist with that request.") is None


def test_parse_garbage_and_invalid():
    assert parse_verdict("") is None
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"explicit": "yes"}') is None
    assert parse_verdict('{"labels": []}') is None


def test_parse_clamps_and_filters():
    v = parse_verdict('{"explicit": true, "labels": ["FEMALE_BREAST", "NUKE", 7], "confidence": 3.0}')
    assert v["labels"] == ["FEMALE_BREAST"]
    assert v["confidence"] == 1.0


# --- aggregation --------------------------------------------------------------------

def _pf(t, reason="window", seg_idx=None):
    return PlannedFrame(t=t, prio=2 if reason == "window" else 3,
                        reason=reason, seg_idx=seg_idx)


SM = SmoothingConfig()


def test_aggregate_segment_verdicts():
    segs = [{"start": 100.0, "end": 110.0}, {"start": 200.0, "end": 210.0}]
    planned = ([_pf(102, "verify", 0), _pf(105, "verify", 0), _pf(108, "verify", 0),
                _pf(202, "verify", 1), _pf(205, "verify", 1), _pf(208, "verify", 1)])
    verdicts = [
        {"explicit": True, "labels": [], "confidence": 0.9},
        {"explicit": True, "labels": [], "confidence": 0.8},
        {"explicit": False, "labels": [], "confidence": 0.3},
        {"explicit": False, "labels": [], "confidence": 0.9},
        {"explicit": False, "labels": [], "confidence": 0.9},
        None,  # refusal on last verify frame of segment 1
    ]
    out = aggregate(planned, verdicts, 600.0, SM, segs)
    assert out["segment_verdicts"][0]["verdict"] == "confirmed"   # 2/3
    assert out["segment_verdicts"][1]["verdict"] == "rejected"    # 0/2 valid
    assert out["refusals"] == 1


def test_aggregate_single_valid_frame_is_uncertain():
    segs = [{"start": 100.0, "end": 110.0}]
    planned = [_pf(102, "verify", 0), _pf(105, "verify", 0), _pf(108, "verify", 0)]
    verdicts = [{"explicit": True, "labels": [], "confidence": 0.9}, None, None]
    out = aggregate(planned, verdicts, 600.0, SM, segs)
    assert out["segment_verdicts"][0]["verdict"] == "uncertain"


def test_aggregate_new_segments_from_window_hits():
    """Two explicit window hits a second apart -> one new segment, padded."""
    segs = [{"start": 100.0, "end": 110.0}]
    planned = [_pf(102, "verify", 0), _pf(105, "verify", 0), _pf(108, "verify", 0),
               _pf(200.0), _pf(201.0), _pf(202.0), _pf(300.0)]
    verdicts = [
        {"explicit": True, "labels": [], "confidence": 0.9},
        {"explicit": True, "labels": [], "confidence": 0.9},
        {"explicit": True, "labels": [], "confidence": 0.9},
        {"explicit": True, "labels": [], "confidence": 0.8},
        {"explicit": True, "labels": [], "confidence": 0.9},
        {"explicit": True, "labels": [], "confidence": 0.8},
        {"explicit": False, "labels": [], "confidence": 0.1},
    ]
    out = aggregate(planned, verdicts, 600.0, SM, segs)
    assert len(out["new_segments"]) == 1
    seg = out["new_segments"][0]
    # smoothing pads 1.5s both ways around 200-202
    assert seg["start"] <= 198.6 and seg["end"] >= 203.4
    # not overlapping the existing segment
    assert seg["start"] >= 110.0


def test_aggregate_new_segments_clipped_by_existing():
    """A window hit adjacent to an existing segment must not re-add covered
    time: the overlap is clipped away (pad crossing into the segment)."""
    segs = [{"start": 100.0, "end": 110.0}]
    planned = [_pf(110.6), _pf(111.4)]           # just after the segment
    verdicts = [{"explicit": True, "labels": [], "confidence": 0.9},
                {"explicit": True, "labels": [], "confidence": 0.9}]
    out = aggregate(planned, verdicts, 600.0, SmoothingConfig(pad_s=1.5), segs)
    for ns in out["new_segments"]:
        assert ns["start"] >= 110.0


def test_aggregate_distant_hits_never_chain():
    """Two explicit hits 30s apart are NOT consecutive samples on a sparse
    timeline — they must form two separate small segments, not one 30s
    span (regression: the phantom giant new-segments bug)."""
    segs = [{"start": 100.0, "end": 110.0}]
    planned = [_pf(200.0), _pf(230.0), _pf(260.0)]   # sparse window frames
    verdicts = [{"explicit": True, "labels": [], "confidence": 0.9}] * 3
    out = aggregate(planned, verdicts, 600.0, SM, segs)
    assert len(out["new_segments"]) == 3


def test_aggregate_sweep_hit_pads_wide():
    """A sweep hit is a POINT sample of a scene of unknown extent — its new
    segment must be padded by ARBITER_SWEEP_PAD_S, much wider than a window
    hit whose neighborhood was sampled on a 1s grid."""
    planned = [_pf(100.0, "sweep"), _pf(400.0)]      # window frame at 400
    verdicts = [{"explicit": True, "labels": [], "confidence": 0.9}] * 2
    out = aggregate(planned, verdicts, 600.0, SM, [])
    ns = sorted(out["new_segments"], key=lambda s: s["start"])
    pad = config.ARBITER_SWEEP_PAD_S
    assert abs(ns[0]["start"] - (100.0 - pad)) < 0.01
    assert abs(ns[0]["end"] - (100.0 + pad)) < 0.01
    assert (ns[0]["end"] - ns[0]["start"]) > \
        (ns[1]["end"] - ns[1]["start"])               # sweep >> window pad


def test_cluster_hits_chains_overlapping_sweep_pads():
    """Sweep samples 15s apart describe ONE ongoing scene when their wide
    pads overlap (gap after padding <= chain_gap); window hits 15s apart
    with narrow pads stay two segments."""
    segs = _cluster_hits([(100.0, 0.9, 8.0), (115.0, 0.8, 8.0)], 6.0)
    assert len(segs) == 1
    assert (segs[0][0], segs[0][1]) == (92.0, 123.0)
    assert segs[0][2] == pytest.approx(0.85)
    assert len(_cluster_hits([(100.0, 0.9, 1.5), (115.0, 0.8, 1.5)], 6.0)) == 2


def test_subtract():
    assert _subtract([(0, 10)], [(4, 6)]) == [(0, 4), (6, 10)]
    assert _subtract([(0, 10), (20, 30)], [(5, 25)]) == [(0, 5), (25, 30)]
    assert _subtract([(0, 10)], [(0, 10)]) == []
    assert _subtract([(0, 10)], [(15, 20)]) == [(0, 10)]


# --- VLM path resolution --------------------------------------------------------

def test_resolve_vlm_path_prefers_local(tmp_path, monkeypatch):
    import scanner.config as config

    baked = tmp_path / "qwen3-vl-8b-instruct"     # lowercase on disk
    baked.mkdir()
    monkeypatch.setattr(config, "VLM_DIR", tmp_path)

    assert resolve_vlm_path("Qwen3-VL-8B-Instruct") == str(baked)
    assert resolve_vlm_path("Qwen/Qwen3-VL-8B-Instruct") == str(baked)
    # a direct existing path wins
    direct = tmp_path / "other-model"
    direct.mkdir()
    assert resolve_vlm_path(str(direct)) == str(direct)


def test_resolve_vlm_path_falls_back_to_id(tmp_path, monkeypatch):
    import scanner.config as config

    monkeypatch.setattr(config, "VLM_DIR", tmp_path)
    assert resolve_vlm_path("Qwen3-VL-8B-Instruct") == "Qwen3-VL-8B-Instruct"


# --- weak detector hits ----------------------------------------------------------

def test_plan_weak_hits_added_and_skipped_inside_segments():
    segs = [{"start": 100.0, "end": 110.0}]
    planned = plan_frames(600.0, segs, weak_hits=[50.0, 105.0, 200.0])
    weak = [f for f in planned if f.reason == "weak"]
    assert sorted(f.t for f in weak) == [50.0, 200.0]   # 105 sits in a segment


def test_plan_weak_capped_without_starving_sweep():
    planned = plan_frames(600.0, [], weak_hits=[float(i * 5) for i in range(10)],
                          weak_max=3, max_frames=50)
    weak = [f for f in planned if f.reason == "weak"]
    assert len(weak) == 3
    assert any(f.reason == "sweep" for f in planned)


def test_aggregate_pads_weak_hits_like_sweep(monkeypatch):
    monkeypatch.setattr(config, "ARBITER_SWEEP_PAD_S", 8.0)
    planned = [PlannedFrame(100.0, 4, "weak"),
               PlannedFrame(400.0, 1, "sweep"),
               PlannedFrame(500.0, 2, "window")]
    ex = {"explicit": True, "labels": [], "confidence": 0.9}
    out = aggregate(planned, [ex] * 3, 600.0, SmoothingConfig(pad_s=1.5), [])
    starts = {round(s["start"], 3) for s in out["new_segments"]}
    assert starts == {92.0, 392.0, 498.5}


def test_plan_expansion_grid_and_exclusions():
    segs = [{"start": 100.0, "end": 110.0}]
    exp = plan_expansion([200.0], {200.0, 195.0}, segs, 600.0,
                         pad_s=8.0, step_s=1.0, max_frames=50)
    ts = [f.t for f in exp]
    assert ts and all(f.reason == "expand" for f in exp)
    assert 192.0 in ts and 208.0 in ts      # full +-8 window on a 1s grid
    assert 200.0 not in ts and 195.0 not in ts           # already judged
    assert not any(100.0 <= t <= 110.0 for t in ts)      # inside a segment


def test_plan_expansion_budget_spent_on_best_hits_first():
    exp = plan_expansion([400.0, 100.0], set(), [], 600.0,
                         pad_s=8.0, step_s=1.0, max_frames=10)
    ts = [f.t for f in exp]
    assert len(ts) == 10                    # first (highest-conf) hit only
    assert all(392.0 <= t <= 408.0 for t in ts)
