"""End-to-end verification of a masked render against its source — the
in-container twin of the old scripts/verify_blur.py host gate.

Checks:
  1. container: streams (hevc + audio expected), duration == source
  2. decode integrity: full decode with -err_detect explode
  3. timeline: video packet count == source packet count
  4. audio: packet payload md5 identical (bit-exact passthrough)
  5. mask presence: inside flagged segments, some plausible source
     counterpart frame shows the diff, the region went solid black,
     and local texture is crushed
  6. mask absence: clean-range frames match a source frame exactly

Frames are grabbed by raw PTS (select between(t,...) with -copyts after a
short input seek), so matching is immune to cross-codec -ss keyframe
artifacts. Counterpart search spans several neighboring source frames:
the output's raw PTS grid sits a constant 2-3 frames from the source's
(the muxer compensates the source's edit-list offset), and on static
smooth scenes a clean neighbor can score a lower mean diff than the true
masked pair — presence therefore requires SOME plausible counterpart
with a significant diff, absence requires the best match to be exact.

Exposed as POST /verify. Pure CPU (ffprobe/ffmpeg/cv2); heavy, so it
runs as a background job.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# counterpart mean-diff bounds; see check 5 notes in the module docstring
PLAUSIBLE_MEAN = 3.0
FULLMASK_MEAN = 96.0
HOT_THRESH = 20
HOT_MIN_PX = 200
FRAC_BLACK_MIN = 0.2
CLEAN_MEAN_MAX = 4.0
CLEAN_HOT_MAX = 2000


def auto_spots(segments: list[dict], duration_s: float,
               n_clean: int = 3) -> tuple[list[int], list[float]]:
    """Deterministic default spot-check selection: segment 0 plus the
    first segment of each provenance class (legacy detector-era
    EXPOSED labels / crowd / arbiter),
    and clean times evenly spread across the movie, nudged clear of all
    segments. Generic — derived from labels, never hardcoded indices."""
    def first_where(pred) -> int | None:
        for i, seg in enumerate(segments):
            if pred(seg.get("labels") or {}):
                return i
        return None

    idx: list[int] = []
    for i in (0,
              first_where(lambda l: any("EXPOSED" in str(k) for k in l)),
              first_where(lambda l: any(str(k).startswith("cringemdb")
                                        for k in l)),
              first_where(lambda l: "arbiter" in l)):
        if i is not None and i not in idx:
            idx.append(i)
    idx.sort()   # reports read in timeline order

    spans = [(float(s["start"]), float(s["end"])) for s in segments]
    cleans: list[float] = []
    if duration_s > 60:
        fracs = [0.15, 0.5, 0.85][:n_clean] if n_clean < 3 else \
            [i / (n_clean + 1) for i in range(1, n_clean + 1)]
        for t in (duration_s * f for f in fracs):
            # nudge forward until clear of every segment (+5s margin)
            while t < duration_s - 5 and any(a - 5 <= t <= b + 5
                                             for a, b in spans):
                t += 7.3
            if t < duration_s - 5:
                cleans.append(round(float(t), 1))
    return idx, cleans


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check(results: list, name: str, cond: bool, detail: str = "") -> bool:
    results.append({"name": name, "pass": bool(cond), "detail": detail})
    return cond


def _grab(path: str, pts: float, tmpdir: str) -> np.ndarray:
    png = os.path.join(tmpdir, f"v_{uuid.uuid4().hex[:8]}.png")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.0, pts - 5):.3f}", "-i", path, "-copyts",
         "-vf", f"select=between(t\\,{pts - 0.01:.3f}\\,{pts + 0.01:.3f})",
         "-fps_mode", "passthrough", "-frames:v", "1", "-y", png],
        capture_output=True, text=True)
    if not os.path.exists(png):
        raise RuntimeError(f"no frame at {pts}: {r.stderr[:120]}")
    img = cv2.imread(png).astype(np.int16)
    os.unlink(png)
    return img


def _local_std(gray: np.ndarray) -> float:
    r = gray.astype(np.float32)
    m = cv2.blur(r, (3, 3))
    m2 = cv2.blur(r * r, (3, 3))
    return float(np.sqrt(np.maximum(m2 - m * m, 0)).mean())


def run_verification(source: str, output: str, segments: list[dict],
                     masked_i: list[int] | None = None,
                     clean_t: list[float] | None = None,
                     k: int = 6) -> dict:
    """-> report dict {checks: [{name, pass, detail}], all_pass, ...}."""
    segs = [(float(s["start"]), float(s["end"])) for s in segments]
    if not masked_i or not clean_t:
        dur = float(_run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", source]).stdout.strip())
        auto_i, auto_clean = auto_spots(segments, dur)
        masked_i = masked_i or auto_i
        clean_t = clean_t or auto_clean
    results: list[dict] = []

    meta = _run(["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type,codec_name:format=duration",
                 "-of", "json", output])
    streams_meta = json.loads(meta.stdout)
    streams = {s["codec_type"]: s["codec_name"]
               for s in streams_meta["streams"]}
    dur_out = float(streams_meta["format"]["duration"])
    dur_src = float(_run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0",
                          source]).stdout.strip())
    _check(results, "video codec hevc", streams.get("video") == "hevc",
           str(streams))
    _check(results, "audio present", streams.get("audio") is not None, "")
    _check(results, "duration == source (+/-0.5s)",
           abs(dur_out - dur_src) < 0.5, f"{dur_out:.2f} vs {dur_src:.2f}")

    r = _run(["ffmpeg", "-v", "error", "-err_detect", "explode", "-i",
              output, "-f", "null", "-"])
    _check(results, "full decode clean (err_detect explode)",
           not r.stderr.strip(), r.stderr[:120])

    def video_packets(path: str) -> int:
        return int(_run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-count_packets", "-show_entries",
                         "stream=nb_read_packets", "-of", "csv=p=0",
                         path]).stdout.split()[0])

    n_out, n_src = video_packets(output), video_packets(source)
    _check(results, "video packet count == source", n_out == n_src,
           f"{n_out} vs {n_src}")

    def audio_sig(path: str) -> str:
        # md5 over packet payload bytes via ffmpeg's md5 muxer (binary in
        # a text capture breaks; the md5 muxer prints one text line).
        # FULL stream only — no seeks: seek windows land on different
        # packet boundaries per container. Byte-exact payload is the
        # invariant that survives container differences.
        o = _run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
                  "-c", "copy", "-f", "md5", "-"])
        return o.stdout.strip().split("=")[-1]

    _check(results, "audio packets bit-exact (payload md5)",
           audio_sig(output) == audio_sig(source), "")

    with tempfile.TemporaryDirectory(prefix="verify_") as tmp:
        def grab(path: str, pts: float) -> np.ndarray:
            return _grab(path, pts, tmp)

        def plist(path: str) -> list[float]:
            o = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "packet=pts_time", "-of", "csv=p=0",
                      path]).stdout.strip().split("\n")
            return [float(x) for x in o if x.strip()]

        sp, op = plist(source), plist(output)

        def cands_near(t: float, k: int) -> list[float]:
            return sorted(sp, key=lambda p: abs(p - t))[:k]

        for i in masked_i:
            a, b = segs[i]
            t = (a + b) / 2
            po = min(op, key=lambda p: abs(p - t))
            out_frame = grab(output, po)
            cands = [(float(np.abs(grab(source, ps) - out_frame)
                            .mean(axis=2).mean()), ps)
                     for ps in cands_near(t + 0.083, k)]
            plausible = [ps for m, ps in cands if m < PLAUSIBLE_MEAN]
            # full-person / VLM masks cover most of the frame and push
            # the mean above PLAUSIBLE_MEAN against EVERY source
            # candidate (measured 5.4-9.4 on legit full-person masks).
            # Fall back to the closest candidate; the hot-pixel/std/
            # black checks below still fail a wrong-frame or garbage
            # match — the bound only pre-filters candidates.
            if not plausible and cands and min(cands)[0] < FULLMASK_MEAN:
                plausible = [min(cands)[1]]
            best_hot, pair = 0, None
            for ps in plausible:
                src_frame = grab(source, ps)
                d = np.abs(src_frame - out_frame).mean(axis=2)
                hot = int((d > HOT_THRESH).sum())
                if hot > best_hot:
                    best_hot, pair = hot, (src_frame, d)
            if pair is None:
                if plausible:
                    # counterpart matched but ZERO hot pixels anywhere:
                    # the frame is (near-)identical to source — the mask
                    # is MISSING here, not unmatchable
                    _check(results, f"mask present @ seg {i} ({t:.1f}s)",
                           False, "counterpart matched, no mask diff "
                                  "(unmasked frame)")
                else:
                    _check(results, f"mask present @ seg {i} ({t:.1f}s)",
                           False, "no plausible counterpart")
                continue
            src_frame, d = pair
            ys, xs = np.nonzero(d > HOT_THRESH)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            ga = cv2.cvtColor(src_frame[y0:y1, x0:x1].astype(np.uint8),
                              cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(out_frame[y0:y1, x0:x1].astype(np.uint8),
                              cv2.COLOR_BGR2GRAY)
            sa, sb = _local_std(ga), _local_std(gb)
            smooth = sa < 6.0
            # blackout assert: a meaningful share of the diff region is
            # near-black in the output (the old mosaic preserved local
            # mean, so it would fail this — deliberately).
            frac_black = float((gb < 16).mean())
            _check(results, f"mask present @ seg {i} ({t:.1f}s)",
                   len(ys) > HOT_MIN_PX and frac_black >= FRAC_BLACK_MIN
                   and (sb < 0.8 * sa or smooth),
                   f"{len(ys)} px, black {frac_black:.0%}, "
                   f"std {sa:.2f}->{sb:.2f}"
                   + (" (source ultra-smooth)" if smooth else ""))

        for t in clean_t:
            if any(a <= t <= b for a, b in segs):
                _check(results, f"clean @ {t:.0f}s outside segments",
                       False, "clean spot is inside a segment")
                continue
            po = min(op, key=lambda p: abs(p - t))
            out_frame = grab(output, po)
            best = min(float(np.abs(grab(source, ps) - out_frame)
                             .mean(axis=2).mean())
                       for ps in cands_near(t + 0.083, k))
            best_hot = min(int((np.abs(grab(source, ps) - out_frame)
                                .mean(axis=2) > 30).sum())
                           for ps in cands_near(t + 0.083, k))
            _check(results, f"clean @ {t:.0f}s exact source match",
                   best < CLEAN_MEAN_MAX and best_hot < CLEAN_HOT_MAX,
                   f"best mean {best:.2f}, hot {best_hot}")

    all_pass = all(c["pass"] for c in results)
    return {
        "source": source, "output": output,
        "masked_indices": masked_i, "clean_times": clean_t,
        "checks": results, "all_pass": all_pass,
    }
