"""Crowd timestamp source tests (CringeMDB fetch/parse/merge).

Network-free: the parser runs on a saved fixture of the real Ex Machina
scene guide; the fetch path is monkeypatched.
"""

from pathlib import Path

import pytest

from scanner import config
from scanner.crowd import (fetch_crowd_segments, merge_crowd_segments,
                           parse_filename, parse_scenes, parse_time,
                           slug_for)

FIXTURE = Path(__file__).parent / "fixtures" / "cringemdb_exmachina.html"


# --- pure helpers ----------------------------------------------------------------

def test_parse_time_formats():
    assert parse_time("52:16") == pytest.approx(3136.0)
    assert parse_time("1:29:52") == pytest.approx(5392.0)
    assert parse_time("0:00") == pytest.approx(0.0)


def test_parse_filename_title_year():
    cases = {
        "Ex.Machina.2015.1080p.BluRay.x264.YIFY.mp4": ("Ex Machina", 2015),
        "The Godfather 1972 720p.mkv": ("The Godfather", 1972),
        "dont-breathe-2016-[1080p].mp4": ("dont breathe", 2016),
        "Movie.Without.Year.mp4": ("Movie Without Year", None),
    }
    for name, (t, y) in cases.items():
        assert parse_filename(name) == (t, y)


def test_slug_for():
    assert slug_for("Ex Machina", 2015) == "ex-machina-2015"
    assert slug_for("Don't Breathe", 2016) == "don-t-breathe-2016"
    assert slug_for("Nope", None) == "nope"


# --- parser on the saved real page --------------------------------------------------

def test_parse_scenes_fixture_matches_gt():
    scenes = parse_scenes(FIXTURE.read_text())
    assert len(scenes) == 5
    by_cat = {}
    for sc in scenes:
        by_cat.setdefault(sc["category"], []).append(sc)
    assert len(by_cat["nudity"]) == 3
    assert len(by_cat["sex"]) == 1
    assert len(by_cat["violence"]) == 1
    nud = sorted(by_cat["nudity"], key=lambda s: s["start_s"])
    assert (nud[0]["start_s"], nud[0]["end_s"]) == (3136.0, 3138.0)
    assert (nud[1]["start_s"], nud[1]["end_s"]) == (3197.0, 3247.0)
    assert (nud[2]["start_s"], nud[2]["end_s"]) == (3476.0, 3490.0)
    sex = by_cat["sex"][0]
    assert (sex["start_s"], sex["end_s"]) == (3301.0, 3472.0)
    assert "distance" in sex["desc"]
    assert by_cat["violence"][0]["start_s"] == pytest.approx(5392.0)


# --- fetch (monkeypatched network) ---------------------------------------------------

def _fake_fetch(url, timeout):
    return FIXTURE.read_text()


def test_fetch_crowd_segments_filters_and_pads(monkeypatch):
    import scanner.crowd as c
    monkeypatch.setattr(c, "fetch_page", _fake_fetch)
    block = fetch_crowd_segments("Ex.Machina.2015.1080p.mp4",
                                 cache_dir=None)
    assert block["error"] is None
    assert block["slug"] == "ex-machina-2015"
    assert len(block["entries"]) == 4          # nudity x3 + sex
    assert len(block["skipped"]) == 1          # violence out of scope
    assert block["skipped"][0]["category"] == "violence"
    starts = sorted(s["start"] for s in block["segments"])
    assert starts[0] == pytest.approx(3136.0 - config.CROWD_PAD_S)
    labs = [l for s in block["segments"] for l in s["labels"]]
    assert any(l.startswith("cringemdb:") for l in labs)


def test_fetch_crowd_404_is_failsoft(monkeypatch):
    import scanner.crowd as c
    import urllib.error
    def boom(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(c, "fetch_page", boom)
    block = fetch_crowd_segments("No.Such.Movie.2099.mp4", cache_dir=None)
    assert block["segments"] == []
    assert "404" in block["error"]


def test_fetch_crowd_uses_cache(monkeypatch, tmp_path):
    import scanner.crowd as c
    calls = []
    def counting_fetch(url, timeout):
        calls.append(url)
        return FIXTURE.read_text()
    monkeypatch.setattr(c, "fetch_page", counting_fetch)
    fetch_crowd_segments("Ex.Machina.2015.mp4", cache_dir=tmp_path)
    fetch_crowd_segments("Ex.Machina.2015.mp4", cache_dir=tmp_path)
    assert len(calls) == 1                     # second run hit the cache


# --- merge ---------------------------------------------------------------------------

def test_merge_crowd_union_and_labels():
    segs = [{"start": 3130.0, "end": 3140.0, "mean_score": 0.5,
             "labels": {"FEMALE_BREAST_EXPOSED": 3}}]
    crowd = {"segments": [
        {"start": 3135.0, "end": 3150.0, "mean_score": 0.9,
         "labels": {"cringemdb:nudity": 1}},
        {"start": 4000.0, "end": 4010.0, "mean_score": 0.9,
         "labels": {"cringemdb:sex": 1}}]}
    out = merge_crowd_segments(segs, crowd)
    assert len(out) == 2
    assert out[0]["start"] == 3130.0 and out[0]["end"] == 3150.0
    assert out[0]["labels"]["FEMALE_BREAST_EXPOSED"] == 3
    assert out[0]["labels"]["cringemdb:nudity"] == 1
    assert out[1]["start"] == 4000.0


def test_merge_crowd_none_is_identity():
    segs = [{"start": 1.0, "end": 2.0, "labels": {}}]
    assert merge_crowd_segments(segs, None) == segs
    assert merge_crowd_segments(segs, {"segments": []}) == segs
