#!/usr/bin/env python3
"""Bake mask-region chapters into a masked render (stream copy, no
re-encode, NVDEC path untouched).

Each scan segment becomes a named chapter — "▶ MASK · FEMALE_BREAST 87%"
— and filler chapters fill the gaps so every mask is bracketed by ticks
on mpv's seek bar (a chapter's end tick IS the next chapter's start).
Chapter names are the payload: mpv flashes them on every seek, and
chapter navigation (pgup/pgdn, --chapter) lands on mask boundaries.

    python scripts/bake_mask_chapters.py scans/saltburn_mask_v3b.json \
        blur_outputs/Saltburn.2023.masked.mp4

Blur outputs are container-written (root-owned); when the output path
isn't host-writable the mux runs inside the movie-trimmer container
automatically (ffmpeg is in the image).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

CONTAINER = "movie-trimmer"


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
    for i, (a, b, title) in enumerate(chapters):
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


def mux(video: str, meta_path: str) -> None:
    """Stream-copy mux with chapters; root-owned dirs go via container."""
    tmp = os.path.join(tempfile.gettempdir(),
                       f"chapters_{os.getpid()}_{os.path.basename(video)}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", video, "-i", meta_path,
           "-map", "0", "-map_metadata", "1", "-c", "copy",
           "-movflags", "+faststart", tmp]
    writable = os.access(os.path.dirname(os.path.abspath(video)), os.W_OK)
    if writable:
        subprocess.run(cmd, check=True)
        shutil.move(tmp, video)
    else:
        # blur_outputs is root-owned: same mux inside the container.
        # The repo root is mounted there at /movies, so any host path
        # under it maps 1:1.
        root = "/home/rohan/repos/Movie-Trimmer"
        def to_container(p: str) -> str:
            p = os.path.abspath(p)
            if not p.startswith(root):
                return p
            rel = os.path.relpath(p, root)
            # blur_outputs has its own rw mount (the repo mount is ro)
            if rel.startswith("blur_outputs/"):
                return "/blur_outputs/" + rel[len("blur_outputs/"):]
            return "/movies/" + rel
        dvideo, dmeta = to_container(video), to_container(meta_path)
        dtmp = "/tmp/" + os.path.basename(tmp)
        dcmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", dvideo, "-i", dmeta,
                "-map", "0", "-map_metadata", "1", "-c", "copy",
                "-movflags", "+faststart", dtmp]
        subprocess.run(["docker", "exec", CONTAINER] + dcmd, check=True)
        subprocess.run(["docker", "exec", CONTAINER, "mv", dtmp,
                        dvideo], check=True)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    scan_json, video = sys.argv[1], sys.argv[2]
    segs = json.load(open(scan_json))["segments"]
    duration = ffprobe_duration(video)
    meta = build_metadata(segs, duration)
    meta_path = os.path.join(os.path.dirname(os.path.abspath(scan_json)),
                             f".chapters_{os.path.basename(video)}.txt")
    with open(meta_path, "w") as f:
        f.write(meta)
    mux(video, meta_path)
    os.unlink(meta_path)
    n = subprocess.run(["ffprobe", "-v", "error", "-show_chapters",
                        "-show_entries", "chapter_tags=title",
                        "-of", "csv=p=0", video],
                       capture_output=True, text=True, check=True).stdout
    masks = [l for l in n.splitlines() if "MASK" in l]
    print(f"[chapters] baked {len(masks)} mask chapters (+fillers) "
          f"into {video}")
    for l in masks[:3]:
        print(f"           e.g. {l}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
