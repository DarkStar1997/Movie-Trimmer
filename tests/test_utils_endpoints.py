"""Tests for the in-container utility endpoints' logic: chapter
metadata build (scanner/chapters.py), verify spot auto-selection
(scanner/verify.py), and output-name sanitization (scanner/jobs.py)."""
from __future__ import annotations

import pytest

from scanner.chapters import build_metadata, chapter_title
from scanner.jobs import safe_output_name
from scanner.verify import auto_spots


# --- chapter titles ------------------------------------------------------------


def test_chapter_title_ranks_labels_and_caps_confidence():
    seg = {"labels": {"FEMALE_BREAST_EXPOSED": 0.87, "MALE_GENITALIA_EXPOSED": 0.4},
           "mean_score": 0.87}
    assert chapter_title(seg) == "\u25b6 MASK \u00b7 FEMALE_BREAST_EXPOSED + MALE_GENITALIA_EXPOSED 87%"


def test_chapter_title_maps_arbiter_label_to_vlm():
    seg = {"labels": {"arbiter": 0.94}, "mean_score": 0.94}
    assert chapter_title(seg) == "\u25b6 MASK \u00b7 VLM 94%"


def test_chapter_title_unlabeled_segment_says_flagged_with_default_conf():
    assert chapter_title({}) == "\u25b6 MASK \u00b7 FLAGGED 90%"


# --- chapter metadata ----------------------------------------------------------


def _meta_chapters(meta: str) -> list[tuple[int, int, str]]:
    out = []
    for block in meta.split("[CHAPTER]")[1:]:
        start = int(next(l for l in block.splitlines()
                         if l.startswith("START=")).split("=")[1])
        end = int(next(l for l in block.splitlines()
                       if l.startswith("END=")).split("=")[1])
        title = next(l for l in block.splitlines()
                     if l.startswith("title=")).split("=", 1)[1]
        out.append((start, end, title))
    return out


def test_build_metadata_brackets_masks_with_fillers():
    segs = [{"start": 10.0, "end": 20.0, "labels": {"arbiter": 0.9}}]
    ch = _meta_chapters(build_metadata(segs, 60.0))
    assert len(ch) == 3
    assert ch[0][2] == "\u00b7" and ch[0][:2] == (0, 10000)      # leading filler
    assert "MASK" in ch[1][2] and ch[1][:2] == (10000, 20000)    # the mask
    assert ch[2][2] == "\u00b7" and ch[2][:2] == (20000, 60000)  # trailing filler


def test_build_metadata_sorts_segments_and_drops_empty_gaps():
    segs = [
        {"start": 30.0, "end": 40.0, "labels": {"arbiter": 0.9}},
        {"start": 10.0, "end": 30.0, "labels": {"arbiter": 0.9}},  # abuts prev
        {"start": 50.0, "end": 60.0, "labels": {"arbiter": 0.9}},
    ]
    ch = _meta_chapters(build_metadata(segs, 60.0))
    # fillers: 0-10, 40-50; masks: 10-30, 30-40, 50-60; no trailing filler
    assert [c[:2] for c in ch] == [(0, 10000), (10000, 30000),
                                   (30000, 40000), (40000, 50000),
                                   (50000, 60000)]
    assert sum("MASK" in c[2] for c in ch) == 3


def test_build_metadata_clamps_mask_end_to_duration():
    segs = [{"start": 50.0, "end": 90.0, "labels": {"arbiter": 0.9}}]
    ch = _meta_chapters(build_metadata(segs, 60.0))
    assert ch[-1][:2] == (50000, 60000)  # clamped, no empty chapter after


# --- verify spot auto-selection ------------------------------------------------


def test_auto_spots_picks_provenance_classes_and_dedupes():
    segs = [
        {"start": 0, "end": 5, "labels": {"FEMALE_BREAST_EXPOSED": 0.9}},   # 0: both seg-0 and EXPOSED
        {"start": 100, "end": 110, "labels": {"arbiter": 0.94}},            # arbiter
        {"start": 200, "end": 210, "labels": {"cringemdb:sex": 0.9}},       # crowd
        {"start": 300, "end": 310, "labels": {"MALE_GENITALIA_EXPOSED": 0.9}},
    ]
    idx, cleans = auto_spots(segs, 1000.0)
    assert idx == [0, 1, 2]
    assert len(cleans) == 3
    for t in cleans:
        assert 0 <= t < 995
        assert not any(s["start"] - 5 <= t <= s["end"] + 5 for s in segs)


def test_auto_spots_cleans_nudge_clear_of_long_segment():
    # a segment covering the whole first third swallows the 0.15-fraction
    # clean spot; it must be nudged forward past it
    segs = [{"start": 0, "end": 300, "labels": {"arbiter": 0.9}}]
    idx, cleans = auto_spots(segs, 1000.0)
    assert idx == [0]
    for t in cleans:
        assert t > 300 + 5


def test_auto_spots_short_movie_gets_no_cleans():
    idx, cleans = auto_spots([{"start": 0, "end": 5}], 30.0)
    assert idx == [0] and cleans == []


# --- output-name sanitization --------------------------------------------------


def test_safe_output_name_appends_mp4_and_strips_dirs():
    assert safe_output_name("../evil/Gone.Girl") == "Gone.Girl.mp4"
    assert safe_output_name("Wolf.of.Wall.Street.2013.masked.mp4") == \
        "Wolf.of.Wall.Street.2013.masked.mp4"


@pytest.mark.parametrize("bad", ["", ".", "..", ".hidden", ".hidden.mp4"])
def test_safe_output_name_rejects_escapes(bad):
    with pytest.raises(ValueError):
        safe_output_name(bad)
