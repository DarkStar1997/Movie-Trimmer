"""Stage E (experimental): IMDb Parents Guide as arbiter film-context.

The v3 arbiter prompt judges each frame's visibility context-blind, then
its context (sexual vs incidental). Borderline context calls are its
weakest joint — exactly where community scene descriptions help: they
say what KIND of film this is and which sequences are sexual vs
incidental, the same evidence a human adjudicator uses.

This module resolves a movie's Parents Guide and renders its Sex &
Nudity items (plus community severity) as a short film-context block
appended to the arbiter's system prompt. The block is BACKGROUND ONLY:
its wording pins it to STEP 2 tie-breaking and forbids it from changing
visibility either way; an empty block leaves the prompt byte-identical.

Access reality (2026-11-04): IMDb walls programmatic page fetches
(Akamai 202/403 on both the page and api.graphql.imdb.com), so
acquisition is browser-assisted — extract the guide in a real browser
into scans/guide_cache/{tt}.json, which the runtime reads directly. ID
resolution uses IMDb's open suggestion endpoint (unauthenticated). A
live page fetch is still attempted when no cache exists; today that
lands in result['guide']['error'] and the scan proceeds context-free —
guide is an enhancement, never a dependency (fail-soft like crowd).

Opt-in: POST /scan with guide=true. Default off (experimental A/B).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import config
from .crowd import fetch_page, parse_filename

logger = logging.getLogger(__name__)

SUGGEST_URL = "https://v2.sg.media-imdb.com/suggestion/{key}/{query}.json"
GUIDE_PAGE = "https://www.imdb.com/title/{tt}/parentalguide/"


def parse_suggestion(payload: dict, title: str,
                     year: int | None) -> str | None:
    """IMDb suggestion JSON -> best matching tt id, or None.

    Exact title match (case-insensitive); year narrows when given;
    features (movies) preferred over episodes/shorts. Pure."""
    want = title.strip().lower()
    cands = [it for it in payload.get("d", ())
             if str(it.get("id", "")).startswith("tt")
             and str(it.get("l", "")).strip().lower() == want]
    if year is not None:
        same_year = [it for it in cands if it.get("y") == year]
        if same_year:
            cands = same_year
    features = [it for it in cands if it.get("q") == "feature"]
    pool = features or cands
    return str(pool[0]["id"]) if pool else None


def resolve_imdb_id(title: str, year: int | None,
                    timeout: float) -> str | None:
    """Live suggestion-endpoint lookup. Raises on network errors."""
    query = urllib.parse.quote(title.strip())
    key = urllib.parse.quote(title.strip()[0].lower())
    url = SUGGEST_URL.format(key=key, query=query)
    req = urllib.request.Request(
        url, headers={"User-Agent": "movie-trimmer-scanner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    return parse_suggestion(payload, title, year)


def context_block(guide: dict, *,
                  max_items: int = config.GUIDE_MAX_ITEMS,
                  max_chars: int = config.GUIDE_ITEM_MAX_CHARS) -> str:
    """Guide JSON -> film-context prompt block ('' when nothing useful).

    Uses ONLY the Sex & Nudity section: this pipeline's scope. Pure."""
    sec = (guide.get("sections") or {}).get("nudity") or {}
    items = [re.sub(r"\s+", " ", str(it)).strip()
             for it in sec.get("items") or []]
    items = [it if len(it) <= max_chars else it[:max_chars - 1] + "\u2026"
             for it in items if it][:max_items]
    if not items and not sec.get("severity"):
        return ""
    lines = [
        "FILM CONTEXT (community parents-guide summaries; may be "
        "inaccurate, incomplete, or describe scenes absent from this "
        "cut). Use it ONLY to break ties in STEP 2 (context). It never "
        "changes STEP 1: tag anatomy you can see regardless of this "
        "text, and never tag anatomy you cannot see because of it."]
    if sec.get("severity"):
        votes = (f" ({sec['votes']})" if sec.get("votes") else "")
        lines.append(f"Community nudity severity: "
                     f"{sec['severity']}{votes}.")
    lines += [f"- {it}" for it in items]
    return "\n".join(lines)


def fetch_guide_context(video_name: str, *, title: str | None = None,
                        year: int | None = None,
                        cache_dir: Path | None = None) -> dict:
    """Resolve + load the parents-guide context for one movie. Fail-soft:
    errors land in the block, never raised.

    -> {"source", "tt", "title", "url", "context_block", "from_cache",
        "error"} — context_block '' means "no context, proceed".
    """
    out: dict = {"source": "imdb-parents-guide", "tt": None, "title": None,
                 "url": None, "context_block": "", "from_cache": False,
                 "error": None}
    try:
        if title is None or year is None:
            t, y = parse_filename(video_name)
            title = title or t
            year = year if year is not None else y
        out["title"] = title
        if not title:
            out["error"] = ("could not parse a movie title from "
                            f"{video_name!r}; pass crowd_title/crowd_year")
            return out
        tt = resolve_imdb_id(title, year, config.GUIDE_FETCH_TIMEOUT_S)
        if not tt:
            out["error"] = (f"no IMDb match for {title!r} ({year}) in the "
                            "suggestion endpoint")
            return out
        out["tt"], out["url"] = tt, GUIDE_PAGE.format(tt=tt)

        cache = Path(cache_dir if cache_dir is not None
                     else config.GUIDE_CACHE_DIR) / f"{tt}.json"
        if cache.exists():
            guide = json.loads(cache.read_text(errors="replace"))
            out["from_cache"] = True
            out["context_block"] = context_block(guide)
            if not out["context_block"]:
                out["error"] = f"guide cache {cache} has no nudity section"
            else:
                logger.info("[guide] %s: %s context from cache (%d chars)",
                            tt, title, len(out["context_block"]))
            return out

        # No cache: attempt a live fetch (Akamai blocks this today; the
        # attempt keeps the path honest if IMDb ever opens up). The
        # React page shell is ~small; a real guide page is >1MB and
        # references the nudity anchor.
        html = fetch_page(out["url"], config.GUIDE_FETCH_TIMEOUT_S)
        if len(html) < 20_000 or "nudity" not in html:
            out["error"] = (f"IMDb blocked the live parents-guide fetch "
                            f"for {tt} (bot wall); seed the cache: "
                            f"scans/guide_cache/{tt}.json via a real "
                            "browser extraction")
            return out
        # Unblocked servers of the classic page would land here; without
        # a live specimen to build against, treat it as unavailable
        # rather than parse blind.
        out["error"] = (f"live parents-guide fetch for {tt} succeeded but "
                        "no parser for the served format; seed "
                        f"scans/guide_cache/{tt}.json")
    except Exception as e:  # noqa: BLE001 — guide must never fail the scan
        out["error"] = f"guide source error: {type(e).__name__}: {e}"
    return out
