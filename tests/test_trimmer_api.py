"""Original trimmer (main.py) unit coverage — the app now mounted at /trim
inside the scanner appliance. CPU-only: segment parsing, subtitle
re-timing/clipping, keep-segment workspace hygiene inputs, and the mount
itself. No ffmpeg, no weights (the cut/concat path is exercised live in
the container)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import main as trimmer
from scanner.main import app as scanner_app


# --- segment parsing -----------------------------------------------------------


def test_parse_segments_string_form():
    assert trimmer.parse_segments_input("00:00-00:10, 01:00-01:05") == [
        [0.0, 10.0], [60.0, 65.0]
    ]


def test_parse_segments_json_form_mixed_formats():
    assert trimmer.parse_segments_input('[[0, 5], ["00:20", "00:25"]]') == [
        [0.0, 5.0], [20.0, 25.0]
    ]


def test_parse_segments_rejects_garbage():
    with pytest.raises(HTTPException) as e:
        trimmer.parse_segments_input("not-a-range")
    assert e.value.status_code == 400


# --- subtitle re-timing --------------------------------------------------------


def test_process_subtitles_clips_and_shifts(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:05,000\nalpha\n\n"
        "2\n"
        "00:00:03,000 --> 00:00:10,000\nbeta\n\n"
        "3\n"
        "00:00:25,000 --> 00:00:28,000\ngamma\n\n",
        encoding="utf-8")
    out = tmp_path / "out.srt"
    # keep (2,8) then (20,30): alpha clipped to 0-3s, beta to 1-6s on the
    # new timeline; gamma lands at 5-8s + 6s offset = 11-14s.
    trimmer.process_subtitles(str(src), str(out), [(2.0, 8.0), (20.0, 30.0)])
    subs = trimmer.pysrt.open(str(out))
    assert [s.text for s in subs] == ["alpha", "beta", "gamma"]
    times = [(s.start.ordinal, s.end.ordinal) for s in subs]
    assert times == [(0, 3000), (1000, 6000), (11000, 14000)]


def test_process_subtitles_drops_cues_outside_keeps(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text(
        "1\n00:00:00,100 --> 00:00:00,200\nflash\n\n"
        "2\n00:00:40,000 --> 00:00:41,000\nkept\n\n",
        encoding="utf-8")
    out = tmp_path / "out.srt"
    trimmer.process_subtitles(str(src), str(out), [(0.0, 1.0), (30.0, 60.0)])
    subs = trimmer.pysrt.open(str(out))
    assert [s.text for s in subs] == ["flash", "kept"]


# --- mount wiring --------------------------------------------------------------


def test_trimmer_is_mounted_under_trim():
    mounts = [r for r in scanner_app.routes if getattr(r, "path", "") == "/trim"]
    assert mounts, "scanner API must mount the trimmer at /trim"
    assert mounts[0].app is trimmer.app


def test_submit_accepts_upload_or_path():
    # the endpoint exposes both an upload field and an appliance-style
    # video_path form field (schema name is FastAPI-version dependent, so
    # find the body schema by shape)
    schemas = trimmer.app.openapi()["components"]["schemas"]
    bodies = [s for s in schemas.values()
              if {"file", "video_path", "segments"} <= set(s.get("properties", {}))]
    assert bodies, "submit body must accept file, video_path and segments"
