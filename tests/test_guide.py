"""Tests for the experimental parents-guide film-context stage
(scanner/guide.py) and its arbiter prompt injection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner import guide as guide_mod
from scanner.arbiter import (ARBITER_SYSTEM_PROMPT, film_context_block,
                             http_request_body)
from scanner.guide import context_block, fetch_guide_context, parse_suggestion


# --- IMDb id resolution --------------------------------------------------------


def _suggest(*items):
    return {"d": [{"id": i, "l": l, "y": y, "q": q}
                  for i, l, y, q in items]}


def test_parse_suggestion_exact_title_year_feature_preferred():
    payload = _suggest(
        ("tt123", "Gone Girl", 2014, "feature"),
        ("tt124", "gone girl", 2014, "short"),      # case-insensitive
        ("tt125", "Gone Girl", 2021, "feature"),    # wrong year
        ("tt126", "Gone Girls", 2014, "feature"),   # wrong title
    )
    assert parse_suggestion(payload, "Gone Girl", 2014) == "tt123"


def test_parse_suggestion_no_year_takes_first_exact():
    payload = _suggest(("tt1", "Saltburn", 2023, "feature"),
                       ("tt2", "Saltburn", 1999, "feature"))
    assert parse_suggestion(payload, "Saltburn", None) == "tt1"


def test_parse_suggestion_no_match_returns_none():
    assert parse_suggestion(_suggest(("tt1", "Alien", 1979, "feature")),
                            "Saltburn", 2023) is None


# --- context block -------------------------------------------------------------


def _guide(nudity_items, severity="Severe", votes="729 of 1232"):
    return {"tt": "tt2267998", "title": "Gone Girl (2014)",
            "sections": {"nudity": {"name": "Sex & Nudity",
                                    "severity": severity, "votes": votes,
                                    "items": nudity_items}}}


def test_context_block_severity_votes_and_items():
    block = context_block(_guide(["A woman's breasts are visible.",
                                  "A sex scene with thrusting."]))
    assert block.startswith("FILM CONTEXT")
    assert "Community nudity severity: Severe (729 of 1232)." in block
    assert "- A woman's breasts are visible." in block
    assert "ONLY to break ties in STEP 2" in block


def test_context_block_caps_items_and_truncates_long_ones():
    items = [f"Scene {i} " + "x" * 400 for i in range(20)]
    block = context_block(_guide(items), max_items=5, max_chars=60)
    assert block.count("- Scene") == 5
    line = [l for l in block.splitlines() if l.startswith("- ")][0]
    assert len(line) <= 62 and line.endswith("\u2026")


def test_context_block_empty_section_is_empty_string():
    assert context_block({"sections": {"nudity": {"items": []}}}) == ""
    assert context_block({}) == ""


# --- arbiter prompt injection --------------------------------------------------


def test_prompt_byte_identical_without_film_context():
    body = http_request_body("aGVsbG8=")
    assert (body["messages"][0]["content"]
            == ARBITER_SYSTEM_PROMPT)                    # regression pin
    assert film_context_block("") == ""


def test_prompt_appends_film_context_verbatim():
    ctx = "FILM CONTEXT\n- item"
    body = http_request_body("aGVsbG8=", film_context=ctx)
    content = body["messages"][0]["content"]
    assert content == ARBITER_SYSTEM_PROMPT + "\n\n" + ctx
    assert content.startswith(ARBITER_SYSTEM_PROMPT)


# --- fetch (fail-soft, cache-first) ---------------------------------------------


def test_fetch_guide_context_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(guide_mod, "resolve_imdb_id",
                        lambda t, y, to: "tt2267998")
    cache = tmp_path / "tt2267998.json"
    cache.write_text(json.dumps(_guide(["Breasts seen briefly."])))
    out = fetch_guide_context("Gone.Girl.2014.1080p.mkv",
                              cache_dir=tmp_path)
    assert out["error"] is None and out["from_cache"] is True
    assert out["tt"] == "tt2267998"
    assert "Breasts seen briefly." in out["context_block"]


def test_fetch_guide_context_no_match_fail_soft(tmp_path, monkeypatch):
    monkeypatch.setattr(guide_mod, "resolve_imdb_id",
                        lambda t, y, to: None)
    out = fetch_guide_context("Nowhere.Movie.2019.mkv", cache_dir=tmp_path)
    assert out["context_block"] == "" and out["error"] is not None


def test_fetch_guide_context_blocked_live_fetch_fail_soft(
        tmp_path, monkeypatch):
    monkeypatch.setattr(guide_mod, "resolve_imdb_id",
                        lambda t, y, to: "tt9999999")
    monkeypatch.setattr(guide_mod, "fetch_page",
                        lambda url, to: "<html>202 challenge</html>")
    out = fetch_guide_context("Some.Movie.2020.mkv", cache_dir=tmp_path)
    assert out["context_block"] == ""
    assert "blocked" in out["error"] or "seed the cache" in out["error"]


def test_fetch_guide_context_unparsed_title_fail_soft(tmp_path):
    out = fetch_guide_context("1998 x264 aac.mp4", cache_dir=tmp_path)
    assert out["context_block"] == "" and out["error"]
