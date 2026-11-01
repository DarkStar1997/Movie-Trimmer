"""Stage A2 person-gate tests: presence decode + range smoothing.

Pure-function coverage only — no torch/CUDA on the host.
"""

import numpy as np

from scanner.personseg import person_presence
from scanner.persongate import smooth_person_presence


# --- presence decode -----------------------------------------------------------

def test_person_presence_pure():
    # person query at 2.0 (sigmoid ~0.88); another class far higher —
    # presence must NOT depend on the global top-k cut
    logits = np.full((1, 5, 91), -10.0, dtype=np.float32)
    logits[0, 2, 1] = 2.0
    logits[0, 3, 50] = 5.0
    ok, score = person_presence({"labels": logits}, {"labels": "labels"}, 0.5)
    assert ok
    assert 0.85 < score < 0.92


def test_person_presence_absent():
    logits = np.full((1, 5, 91), -10.0, dtype=np.float32)
    logits[0, 1, 40] = 6.0            # only a non-person class fires
    ok, score = person_presence({"labels": logits}, {"labels": "labels"}, 0.5)
    assert not ok and score < 0.5


# --- range smoothing -------------------------------------------------------------

def test_smooth_hysteresis_open_and_bridge():
    times = [round(i * 0.5, 3) for i in range(20)]     # 2fps, 0..9.5
    present = ([False] * 4 + [True, True] + [False, False]
               + [True, True] + [False] * 6 + [True, True, True] + [False])
    ranges = smooth_person_presence(times, present, open_min=2, close_gap=4)
    # run @2.0 opens; 2-sample gap bridges into the run @4.0 (one range);
    # 6-sample gap closes it; run @8.0 opens its own range
    assert ranges == [(2.0, 4.5), (8.0, 9.0)]


def test_smooth_single_sample_never_opens():
    times = [0.0, 0.5, 1.0]
    present = [True, False, True]
    assert smooth_person_presence(times, present,
                                  open_min=2, close_gap=1) == []


def test_smooth_long_gap_splits_ranges():
    times = [round(i * 0.5, 3) for i in range(12)]
    present = [True] * 4 + [False] * 5 + [True] * 3
    ranges = smooth_person_presence(times, present, open_min=2, close_gap=4)
    assert ranges == [(0.0, 1.5), (4.5, 5.5)]
