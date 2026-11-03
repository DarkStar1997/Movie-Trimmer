"""Bake mask-region chapters into a masked render (stream copy, no
re-encode, NVDEC path untouched) — the in-container twin of the old
scripts/bake_mask_chapters.py host tool.

Each scan segment becomes a named chapter — "▶ MASK · FEMALE_BREAST 87%"
— and filler chapters fill the gaps so every mask is bracketed by ticks
on mpv's seek bar (a chapter's end tick IS the next chapter's start).
Chapter names are the payload: mpv flashes them on every seek, and
chapter navigation (pgup/pgdn, --chapter) lands on mask boundaries.
Because the renders are self-announcing (blackout + red overlay), no OSD
script or sidecar is needed.

Exposed as POST /chapters and run automatically at the end of every
full-movie mask render (bake_chapters=true).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def chapter_title(seg: dict) -> str:
    labels = seg.get("labels") or {}
    ranked = sorted(labels, key=lambda k: -labels[k])
    parts = []
    for lab in ranked[:2]:
        parts.append("VLM" if lab == "arbiter" else str(lab))
    conf = round(100 * float(seg.get("mean_score") or 0.9))
    name = " + ".join(parts) if parts else "FLAGGED"
    return f"\u25b6 MASK \u00b7 {name} {conf}%"


def build_metadata(segments: list[dict], duration_s: float) -> str:
    """FFmetadata with one chapter per mask + gap fillers, in order."""
    lines = [";FFMETADATA1"]
    spans = [(float(s["start"]), float(s["end"]), chapter_title(s))
             for s in sorted(segments, key=lambda s: float(s["start"]))]
    chapters: list[tuple[float, float, str]] = []
    t = 0.0
    for start, end, title in spans:
        if t < start - 0.001:
            chapters.append((t, start, "\u00b7"))          # filler "·"
        chapters.append((start, min(end, duration_s), title))
        t = max(t, end)
    if t < duration_s - 0.001:
        chapters.append((t, duration_s, "\u00b7"))
    for a, b, title in chapters:
        if b - a < 0.001:                                  # drop empties
            continue
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(round(a * 1000))}",
                  f"END={int(round(b * 1000))}",
                  f"title={title}"]
    return "\n".join(lines) + "\n"


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True,
        check=True).stdout.strip()
    return float(out)


def bake(render_path: str, segments: list[dict]) -> dict:
    """Mux chapters into render_path in place (stream copy). -> summary."""
    out = Path(render_path)
    duration = ffprobe_duration(str(out))
    meta = build_metadata(segments, duration)
    # meta + tmp output live in the same dir as the render so the final
    # swap is an atomic rename on the same filesystem
    fd, meta_path = tempfile.mkstemp(suffix=".ffmeta", dir=out.parent)
    os.write(fd, meta.encode())
    os.close(fd)
    tmp_out = out.with_name(f".{out.name}.chaptering.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(out), "-i", meta_path,
             "-map", "0", "-map_metadata", "1", "-c", "copy",
             "-movflags", "+faststart", str(tmp_out)],
            check=True, capture_output=True, text=True)
        os.replace(tmp_out, out)
    finally:
        os.unlink(meta_path)
        if tmp_out.exists():
            tmp_out.unlink()
    n = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters",
         "-show_entries", "chapter_tags=title", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout
    titles = [l for l in n.splitlines() if l.strip()]
    masks = [l for l in titles if "MASK" in l]
    logger.info("[chapters] baked %d mask chapters (+%d fillers) into %s",
                len(masks), len(titles) - len(masks), out.name)
    return {"output": str(out), "chapters": len(titles),
            "mask_chapters": len(masks)}
