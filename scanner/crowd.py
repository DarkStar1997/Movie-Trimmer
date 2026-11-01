"""Crowd timestamp source: CringeMDB scene guide (opt-in).

Given a video filename, resolve the movie's CringeMDB page
(/movie/{title-dash-year} — deterministic slugs, the site's /search API
returns nothing useful), fetch the Scene-by-Scene guide, keep the
in-scope categories (nudity/sex by default — this pipeline's scope has
always been nudity, not violence), and hand the ranges to the scan as
first-class segments. They get exactly the VLM-segment treatment in
Stage C: mask every person (RF-DETR) with tracking, center fallback when
nobody is found, red overlay naming the source and category.

Community entries are unverified and "can be off by a minute or so" —
they are ranges padded like every other segment, never trusted to
REMOVE anything, and the result block records exactly which entries
were used and skipped so a human can audit.

Fail-soft by design: CringeMDB is an enhancement, not a model — any
fetch/parse problem lands in result['crowd']['error'] and the scan
proceeds without crowd segments.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

CROWD_BASE = "https://cringemdb.com/movie/"

_SCENE_RE = re.compile(
    r'<article class="sg-scene"[^>]*>(.*?)</article>', re.DOTALL)
_TIME_RE = re.compile(
    r"<strong>([\d:]+)</strong>(?:\s*<span>to\s+([\d:]+)</span>)?",
    re.DOTALL)
_CAT_RE = re.compile(
    r'<i class="sg-dot ([a-z-]+)"></i>\s*([^<]*?)\s*(?:<small>([^<]*)'
    r"</small>)?", re.DOTALL)
_DESC_RE = re.compile(r"<p>([^<]+)</p>", re.DOTALL)
_TITLE_JUNK = re.MULTILINE


def parse_time(s: str) -> float:
    """'52:16' -> 3136.0, '1:29:52' -> 5392.0 (H:M:S, M:S accepted)."""
    parts = [int(p) for p in s.strip().split(":")]
    if not parts or len(parts) > 3:
        raise ValueError(f"bad time {s!r}")
    return float(sum(p * 60 ** i for i, p in enumerate(reversed(parts))))


_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")
_JUNK_TOKENS = {"1080p", "720p", "2160p", "480p", "web", "webdl", "web-dl",
                "bluray", "blu-ray", "brrip", "bdrip", "dvdrip", "hdrip",
                "hdtv", "cam", "hdcam", "x264", "x265", "h264", "h265",
                "hevc", "avc", "xvid", "divx", "aac", "ac3", "dts",
                "mp3", "flac", "truehd", "atmos", "10bit", "8bit",
                "remux", "proper", "repack", "extended", "unrated",
                "imax", "hdr", "hdr10", "dv", "dolby", "vision", "multi",
                "subbed", "subs", "nf", "amzn", "dsnp", "hmax", "ma",
                "yify", "rarbg", "ettv", "eztv", "fgt", "galaxyrg",
                "playhd", "psa", "qris"}


def parse_filename(name: str) -> tuple[str, int | None]:
    """'Ex.Machina.2015.1080p.BluRay.x264.YIFY.mp4' -> ('Ex Machina', 2015).

    Splits on . _ - ( brackets ), finds the first plausible year token;
    everything before it (minus release junk) is the title. Pure."""
    stem = Path(name).stem
    tokens = re.split(r"[._\- ()\[\]]+", stem)
    year = None
    title_parts: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        if year is None and _YEAR_RE.match(tok):
            year = int(tok)
            break               # title = everything before the year
        title_parts.append(tok)
    while title_parts and title_parts[-1].lower() in _JUNK_TOKENS:
        title_parts.pop()
    title = " ".join(t for t in title_parts if t).strip()
    return (title, year)


def slug_for(title: str, year: int | None) -> str:
    """'Ex Machina', 2015 -> 'ex-machina-2015' (CringeMDB URL convention)."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if year is not None:
        slug = f"{slug}-{year}"
    return slug


def parse_scenes(page_html: str) -> list[dict]:
    """Scene guide HTML -> [{start_s, end_s, category, intensity, desc}].

    Category comes from the sg-dot CSS class (stable token), not the
    display text. Point scenes (no 'to' span) get end = start. Pure.
    """
    scenes: list[dict] = []
    for block in _SCENE_RE.finditer(page_html):
        chunk = block.group(1)
        tm = _TIME_RE.search(chunk)
        cm = _CAT_RE.search(chunk)
        if not tm or not cm:
            continue
        try:
            start_s = parse_time(tm.group(1))
            end_s = parse_time(tm.group(2)) if tm.group(2) else start_s
        except ValueError:
            continue
        dm = _DESC_RE.search(chunk)
        scenes.append({
            "start_s": start_s,
            "end_s": end_s,
            "category": cm.group(1),
            "category_text": html_mod.unescape(cm.group(2)).strip(),
            "intensity": html_mod.unescape(cm.group(3) or "").strip(),
            "desc": html_mod.unescape(dm.group(1)).strip() if dm else "",
        })
    return scenes


def fetch_page(url: str, timeout: float) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "movie-trimmer-scanner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_crowd_segments(
        video_name: str, *, title: str | None = None,
        year: int | None = None,
        categories: tuple[str, ...] = config.CROWD_CATEGORIES,
        pad_s: float = config.CROWD_PAD_S,
        cache_dir: Path | None = None) -> dict:
    """Resolve + fetch + filter crowd scenes for one movie. Fail-soft:
    errors are returned in the block, never raised.

    -> {"source", "url", "slug", "title", "year", "entries",
        "skipped", "segments", "error"}
          segments: [{"start","end","mean_score","labels"}] in scope,
          padded, clipped to >=0, sorted.
    """
    out: dict = {"source": "cringemdb", "url": None, "slug": None,
                 "title": None, "year": None, "entries": [],
                 "skipped": [], "segments": [], "error": None}
    try:
        if title is None or year is None:
            t, y = parse_filename(video_name)
            title = title or t
            year = year if year is not None else y
        out["title"], out["year"] = title, year
        if not title:
            out["error"] = ("could not parse a movie title from "
                            f"{video_name!r}; pass crowd_title/crowd_year")
            return out
        slug = slug_for(title, year)
        out["slug"] = slug
        url = CROWD_BASE + slug
        out["url"] = url

        cache: Path | None = None
        page_html: str | None = None
        if cache_dir is not None:
            cache = Path(cache_dir) / f"{slug}.html"
            if cache.exists():
                page_html = cache.read_text(errors="replace")
                out["cached"] = True
        if page_html is None:
            page_html = fetch_page(url, config.CROWD_FETCH_TIMEOUT_S)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(page_html)

        scenes = parse_scenes(page_html)
        if not scenes:
            out["error"] = (f"no scene-guide entries parsed from {url} "
                            "(wrong slug or page layout changed?)")
            return out
        for sc in scenes:
            entry = {"start": sc["start_s"], "end": sc["end_s"],
                     "category": sc["category"],
                     "category_text": sc["category_text"],
                     "intensity": sc["intensity"], "desc": sc["desc"]}
            if sc["category"] in categories:
                out["entries"].append(entry)
            else:
                out["skipped"].append(entry)
        out["segments"] = [
            {"start": round(max(0.0, e["start"] - pad_s), 3),
             "end": round(e["end"] + pad_s, 3),
             "mean_score": config.CROWD_SCORE,
             "labels": {f"cringemdb:{e['category']}": 1}}
            for e in out["entries"]]
        logger.info("[crowd] %s: %d in-scope / %d skipped scenes",
                    slug, len(out["entries"]), len(out["skipped"]))
    except urllib.error.HTTPError as e:
        out["error"] = f"cringemdb fetch failed: HTTP {e.code} ({out['url']})"
    except Exception as e:  # noqa: BLE001 — crowd must never fail the scan
        out["error"] = f"crowd source error: {type(e).__name__}: {e}"
    return out


def merge_crowd_segments(segments: list[dict], crowd: dict | None
                         ) -> list[dict]:
    """Union-merge crowd segments into the scan's (same shape as
    _apply_arbiter: sort, overlap-merge, labels accumulate). Pure."""
    if not crowd or not crowd.get("segments"):
        return [dict(s) for s in segments]
    kept = [dict(s) for s in segments]
    kept.extend(dict(s) for s in crowd["segments"])
    kept.sort(key=lambda s: s["start"])
    merged: list[dict] = []
    for s in kept:
        if merged and s["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], s["end"])
            for lab, n in s.get("labels", {}).items():
                prev["labels"] = prev.get("labels") or {}
                prev["labels"][lab] = prev["labels"].get(lab, 0) + n
        else:
            merged.append(s)
    return merged
