import pytest

from scanner.temporal import SmoothingConfig, bin_timeline, smooth_segments

FPS = 4.0
DT = 1.0 / FPS


def ts_of(n):
    return [i / FPS for i in range(n)]


def flags_from_runs(n, runs):
    """runs: list of (start_idx, end_idx) inclusive"""
    flags = [False] * n
    for s, e in runs:
        for i in range(s, e + 1):
            flags[i] = True
    return flags


DUR = 100.0


def test_no_flags_no_segments():
    assert smooth_segments([False] * 40, ts_of(40), [0.0] * 40, DUR) == []


def test_single_sample_blip_never_opens():
    # 1 flagged sample < open_min_consecutive(2) -> nothing
    flags = flags_from_runs(40, [(10, 10)])
    assert smooth_segments(flags, ts_of(40), [0.0] * 40, DUR) == []


def test_short_low_confidence_run_dropped():
    # 2 flagged samples = 0.5s < drop_below_s(1.0), low score -> dropped
    # (rescue line is 0.50 since the flash-recall fix: mean >= 0.50 survives)
    flags = flags_from_runs(40, [(10, 11)])
    scores = [0.45 if f else 0.0 for f in flags]
    assert smooth_segments(flags, ts_of(40), scores, DUR) == []


def test_short_mid_confidence_flash_rescued():
    # Ex Machina GT regression: breast flash @3150.5 fires ~0.52 for <0.5s —
    # the old 0.70 rescue line dropped it; 0.50 keeps real detector-strong
    # flashes while still discarding sub-0.5 noise blips
    flags = flags_from_runs(40, [(10, 11)])
    scores = [0.52 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(40), scores, DUR)
    assert len(segs) == 1


def test_short_high_confidence_run_rescued():
    flags = flags_from_runs(40, [(10, 11)])
    scores = [0.8 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(40), scores, DUR)
    assert len(segs) == 1
    # raw 2.5..3.0s (idx10 @2.5, idx11 end 3.0), pad 1.5 -> [1.0, 4.5]
    assert segs[0]["start"] == pytest.approx(1.0)
    assert segs[0]["end"] == pytest.approx(4.5)


def test_solid_run_produces_padded_segment():
    flags = flags_from_runs(40, [(8, 19)])  # 2.0s .. 5.0s
    scores = [0.6 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(40), scores, DUR)
    assert len(segs) == 1
    assert segs[0]["start"] == pytest.approx(0.5)   # 2.0 - 1.5
    assert segs[0]["end"] == pytest.approx(6.5)     # 5.0 + 1.5


def test_close_rule_merges_short_gaps():
    # runs separated by 1 clean sample (< close_min_consecutive 2) -> merged
    flags = flags_from_runs(40, [(8, 11), (13, 19)])
    scores = [0.6 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(40), scores, DUR)
    assert len(segs) == 1


def test_bridge_gap_merges():
    # gap of 1.5s <= bridge_gap_s(2.0) -> merged into one segment
    flags = flags_from_runs(60, [(4, 15), (23, 35)])   # gap idx16..22 = 1.75s
    scores = [0.6 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(60), scores, DUR)
    assert len(segs) == 1


def test_large_gap_stays_split():
    flags = flags_from_runs(80, [(4, 15), (40, 51)])   # gap ~6s
    scores = [0.6 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(80), scores, DUR)
    assert len(segs) == 2


def test_padding_merges_nearby_segments():
    # gap of 2.6s survives bridging (>2.0) but padding (1.5+1.5) closes it
    cfg = SmoothingConfig(bridge_gap_s=2.0, pad_s=1.5)
    flags = flags_from_runs(80, [(4, 15), (26, 39)])
    scores = [0.6 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(80), scores, DUR, cfg)
    assert len(segs) == 1
    assert segs[0]["start"] == pytest.approx(0.0)   # 1.0 - 1.5 clipped
    assert segs[0]["end"] == pytest.approx(11.5)    # 10.0 + 1.5


def test_padding_clipped_to_duration():
    flags = flags_from_runs(12, [(4, 11)])
    scores = [0.9 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(12), scores, duration=3.0)
    assert segs[-1]["end"] == 3.0


def test_segments_sorted_and_disjoint():
    flags = flags_from_runs(200, [(4, 15), (60, 71), (150, 161)])
    scores = [0.8 if f else 0.0 for f in flags]
    segs = smooth_segments(flags, ts_of(200), scores, DUR)
    for a, b in zip(segs, segs[1:]):
        assert a["end"] < b["start"]


def test_bin_timeline():
    flags = flags_from_runs(10, [(0, 3), (8, 9)])
    tl = bin_timeline(flags, ts_of(10), duration=3.0)
    assert tl[0]["hits"] == 4   # t=0.0,0.25,0.5,0.75
    assert tl[1]["hits"] == 0
    assert tl[2]["hits"] == 2   # t=2.0, 2.25
