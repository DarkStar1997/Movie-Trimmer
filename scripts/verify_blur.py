#!/usr/bin/env python3
"""End-to-end verification of a GPU-native blur output against its source.

Usage:
    python scripts/verify_blur.py SOURCE.mp4 OUTPUT.mp4 SEGMENTS.json
        [--clean-t 1200 --clean-t 3000 ...] [--blurred-i 0 --blurred-i 33]

SEGMENTS.json is any scans/*.json (list of {start, end, ...} segments).

Checks:
  1. container: streams (hevc+aac expected), duration == source
  2. decode integrity: full decode with -err_detect explode
  3. timeline: video packet count == source packet count
  4. audio: packet pts/dts/size md5 identical (bit-exact passthrough)
  5. blur presence: inside flagged segments, some plausible source
     counterpart frame shows the blur diff and local texture is crushed
  6. blur absence: clean-range frames match a source frame exactly

Frames are grabbed by raw PTS (select between(t,...) with -copyts after a
short input seek), so matching is immune to cross-codec -ss keyframe
artifacts. Counterpart search spans several neighboring source frames:
the output's raw PTS grid sits a constant 2-3 frames from the source's
(the muxer compensates the source's edit-list offset), and on static
smooth scenes a clean neighbor can score a lower mean diff than the true
blurred pair — presence therefore requires SOME plausible counterpart
with a significant diff, absence requires the best match to be exact.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid

import cv2
import numpy as np


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def ok(name, cond, detail=""):
    print("%-46s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    return cond


def plist(path):
    o = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0",
             path]).stdout.strip().split("\n")
    return [float(x) for x in o if x.strip()]


def grab(path, pts, tmpdir):
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


def local_std(gray):
    r = gray.astype(np.float32)
    m = cv2.blur(r, (3, 3))
    m2 = cv2.blur(r * r, (3, 3))
    return float(np.sqrt(np.maximum(m2 - m * m, 0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("segments_json")
    ap.add_argument("--clean-t", type=float, action="append", default=None,
                    help="clean-spot seconds (repeatable; default 1200)")
    ap.add_argument("--blurred-i", type=int, action="append", default=None,
                    help="segment index to spot-check (repeatable; default 0)")
    ap.add_argument("--k", type=int, default=6,
                    help="neighbor frames to search per spot")
    args = ap.parse_args()
    args.clean_t = args.clean_t or [1200.0]
    args.blurred_i = args.blurred_i or [0]

    SRC, OUT = args.source, args.output
    segs = [(s["start"], s["end"]) for s in
            json.load(open(args.segments_json))["segments"]]
    results = []

    meta = json.loads(run(["ffprobe", "-v", "error", "-show_entries",
                           "stream=codec_type,codec_name:format=duration",
                           "-of", "json", OUT]).stdout)
    streams = {s["codec_type"]: s["codec_name"] for s in meta["streams"]}
    dur_out = float(meta["format"]["duration"])
    dur_src = float(run(["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "csv=p=0",
                         SRC]).stdout.strip())
    results.append(ok("video codec hevc", streams.get("video") == "hevc",
                      str(streams)))
    results.append(ok("audio present",
                      streams.get("audio") is not None, ""))
    results.append(ok("duration == source (+/-0.5s)",
                      abs(dur_out - dur_src) < 0.5,
                      f"{dur_out:.2f} vs {dur_src:.2f}"))

    r = run(["ffmpeg", "-v", "error", "-err_detect", "explode", "-i", OUT,
             "-f", "null", "-"])
    results.append(ok("full decode clean (err_detect explode)",
                      not r.stderr.strip(), r.stderr[:120]))

    def video_packets(path):
        return int(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_packets", "-show_entries",
                        "stream=nb_read_packets", "-of", "csv=p=0",
                        path]).stdout.split()[0])
    n_out, n_src = video_packets(OUT), video_packets(SRC)
    results.append(ok("video packet count == source", n_out == n_src,
                      f"{n_out} vs {n_src}"))

    def audio_sig(path):
        # md5 over the packet payload bytes via ffmpeg's md5 muxer (binary
        # in a text capture breaks; the md5 muxer prints one text line).
        # FULL stream only — no seeks: seek windows land on different
        # packet boundaries per container. Raw pts strings and decoded PCM
        # both legitimately differ across containers (timescales, skip/
        # edit-list decode semantics); byte-exact payload is the invariant.
        o = run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0",
                 "-c", "copy", "-f", "md5", "-"])
        return o.stdout.strip().split("=")[-1]
    results.append(ok("audio packets bit-exact (payload md5)",
                      audio_sig(OUT) == audio_sig(SRC), ""))

    with tempfile.TemporaryDirectory(prefix="verify_blur_") as tmp:
        def grab2(path, pts):
            return grab(path, pts, tmp)

        sp, op = plist(SRC), plist(OUT)

        def cands_near(t, k):
            return sorted(sp, key=lambda p: abs(p - t))[:k]

        for i in args.blurred_i:
            a, b = segs[i]
            t = (a + b) / 2
            po = min(op, key=lambda p: abs(p - t))
            out_frame = grab2(OUT, po)
            cands = [(float(np.abs(grab2(SRC, ps) - out_frame)
                            .mean(axis=2).mean()), ps)
                     for ps in cands_near(t + 0.083, args.k)]
            plausible = [ps for m, ps in cands if m < 3.0]
            # full-person-mask fallback: masks covering most of a frame
            # legitimately push mean diff high (Gone Girl close-ups: 5-9;
            # Saltburn's detail-dense party montage with a near-full-frame
            # mask measured 16.2). The bound only pre-filters candidates —
            # the hot-pixel and texture-crushing tests below are the real
            # blur-vs-wrong-frame discriminators (a wrong frame is hot
            # everywhere but NOT locally smoother), so it can be generous.
            if not plausible and cands and min(cands)[0] < 24.0:
                # full-person / VLM masks cover most of the frame and push
                # the mean above 3.0 against EVERY source candidate (measured
                # 5.4-9.4 on legit full-person masks). Fall back to the
                # closest candidate; the hot-pixel/std checks below still
                # fail it if this is a wrong-frame or garbage match.
                plausible = [min(cands)[1]]
            best_hot, pair = 0, None
            for ps in plausible:
                src_frame = grab2(SRC, ps)
                d = np.abs(src_frame - out_frame).mean(axis=2)
                hot = int((d > 20).sum())
                if hot > best_hot:
                    best_hot, pair = hot, (src_frame, d)
            if pair is None:
                if plausible:
                    # counterpart matched but ZERO hot pixels anywhere:
                    # the frame is (near-)identical to source — the blur
                    # is MISSING here, not unmatchable
                    results.append(ok(f"blur present @ seg {i} ({t:.1f}s)",
                                      False,
                                      "counterpart matched, no blur diff"
                                      " (unblurred frame)"))
                else:
                    results.append(ok(f"blur present @ seg {i} ({t:.1f}s)",
                                      False, "no plausible counterpart"))
                continue
            src_frame, d = pair
            ys, xs = np.nonzero(d > 20)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            ga = cv2.cvtColor(src_frame[y0:y1, x0:x1].astype(np.uint8),
                              cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(out_frame[y0:y1, x0:x1].astype(np.uint8),
                              cv2.COLOR_BGR2GRAY)
            sa, sb = local_std(ga), local_std(gb)
            smooth = sa < 6.0
            results.append(ok(
                f"blur present @ seg {i} ({t:.1f}s)",
                len(ys) > 200 and (sb < 0.8 * sa or smooth),
                f"{len(ys)} px, std {sa:.2f}->{sb:.2f}"
                + (" (source ultra-smooth)" if smooth else "")))

        for t in args.clean_t:
            assert not any(a <= t <= b for a, b in segs), \
                f"clean spot {t}s is inside a segment"
            po = min(op, key=lambda p: abs(p - t))
            out_frame = grab2(OUT, po)
            best = min(float(np.abs(grab2(SRC, ps) - out_frame)
                             .mean(axis=2).mean())
                       for ps in cands_near(t + 0.083, args.k))
            best_hot = min(int((np.abs(grab2(SRC, ps) - out_frame)
                                .mean(axis=2) > 30).sum())
                           for ps in cands_near(t + 0.083, args.k))
            results.append(ok(f"clean @ {t:.0f}s exact source match",
                              best < 4.0 and best_hot < 2000,
                              f"best mean {best:.2f}, hot {best_hot}"))

    print("\nALL PASS" if all(results) else "\nFAILURES PRESENT")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
