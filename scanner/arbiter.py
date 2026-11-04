"""Stage B: sparse VLM arbiter (Qwen3-VL) — never every frame.

Structure (pure functions first, heavy runner last):

  plan_frames()    pure, testable: which timestamps deserve a VLM look and why
  parse_verdict()  pure: robust strict-JSON verdict extraction + refusal detect
  aggregate()      pure: segment confirm/reject verdicts + new blind-spot
                   segments (SmoothingConfig params drive clustering)
  extract_frames() ffmpeg single-pass frame extraction for the plan
  run_arbiter()    the runner: extract -> lazy VLM load -> batched inference
                   -> aggregate. torch/transformers import happens HERE only,
                   so the module is importable in the torch-free host venv.

Planning policy (trusted sources + person gate; see config):
  - trusted: crowd ranges are mask-final regardless of VLM opinion
    (apply mode is add-only) — NEVER sampled
  - person gate: the RF-DETR pass (scanner/persongate.py) marks where
    anybody is on screen. Gate off/failed = whole timeline eligible
    (fail-open)
  - window: ~1fps sampling in the gaps hugging trusted edges (before
    starts, after ends) — flicker continuations and near-miss
    boundaries live there; person-gated
  - sweep: one frame per ARBITER_SWEEP_EVERY_S across person-present,
    nothing-trusted timeline — the blind-spot backbone
  - ARBITER_MAX_FRAMES <= 0 (default): uncapped; a cap thins by striding
    (never truncated to only the early part of the movie)
"""

from __future__ import annotations

import base64
import bisect
import json
import logging
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config
from .temporal import SmoothingConfig
from . import vlm

logger = logging.getLogger(__name__)

# priority tiers for the frame budget
PRIO_WINDOW = 2
PRIO_SWEEP = 1

_LABEL_WHITELIST = {
    "FEMALE_GENITALIA", "MALE_GENITALIA", "FEMALE_BREAST",
    "MALE_BREAST", "BUTTOCKS", "ANUS",
}

# Arbiter prompting, validated empirically in three rounds (see
# NSFW_SCANNER_DESIGN.md §7): a terse SYSTEM-role classifier prompt with a
# strict-JSON contract gets zero refusals and reliably confirms
# high-confidence nudity. Wordy "moderation"-style user prompts made
# Qwen3-VL sandbag (confident false on real nudity); free-form output was
# slightly more sensitive on borderline dim scenes but unparseable in
# aggregate. The dim/distant-aware paragraph quadrupled GT-frame recall on
# Ex Machina and stays.
#
# Context gate (2026-11-03, prompt replay on 766 recorded-verdict frames
# from Saltburn + Gone Girl — scripts/eval/{build_prompt_replay,
# replay_prompt,score_prompt_replay}.py): the presence-only prompt flagged
# contextual nudity a human reviewer called uncensorable (Saltburn
# 5s-sweep catches at 44:15/75:55/1:51:55/2:00:25). A coupled censor bar
# (v2: "explicit only if sexually presented") suppressed label perception
# on vetoed frames — real anatomy read labels:[] and validated segments
# died. v3 decouples instead: visibility is judged context-blind FIRST,
# context SECOND as its own field, explicit = anatomy AND sexual, and
# unsure always censors. Replay: 2 of 4 human-rejected catches flip clean
# (the other two are a confident model-vs-human disagreement no wording
# fixed), 9 unadjudicated arbiter-only segments prune, clean controls
# 199/208 -> 207/208. Coverage is add-only by construction
# (pipeline._apply_arbiter drops nothing), so the gate can only narrow
# NEW blind-spot segments, never lose detector-flagged ones.
ARBITER_SYSTEM_PROMPT = (
    "You are the nudity-censoring stage of an automated video editing "
    "pipeline processing a licensed feature film. You see single frames "
    "and tag them in two independent steps. You always reply with strict "
    "JSON and never refuse.\n"
    'Reply format: {"explicit": true|false, "labels": [list], '
    '"context": "sexual"|"incidental", "confidence": 0.0-1.0}\n'
    "STEP 1 — VISIBILITY (ignore story context completely): labels "
    "vocabulary FEMALE_GENITALIA, MALE_GENITALIA, FEMALE_BREAST, "
    "BUTTOCKS, ANUS. Tag a label when that body part is directly visible "
    "and unclothed (nipple, genitalia, buttock crack, anus) — anywhere in "
    "the frame, however small, dim, distant, or briefly framed. The frame "
    "may be dark, color-graded, grainy, or a wide shot: look carefully at "
    "dim regions and partially occluded figures before deciding labels. "
    "Do not tag covered bodies, underwear, swimwear, or tiny background "
    "images. STEP 1 never depends on whether the scene feels sexual.\n"
    "STEP 2 — CONTEXT: sexual = sexual activity or intercourse, sexualized "
    "or erotic posing, or unclothed anatomy presented as the deliberate "
    "subject of the shot (close-up or centered reveal meant to be seen). "
    "incidental = the nudity is contextual, not eroticized: background or "
    "distant figures, comedic or non-sexual scenes, art, ordinary bathing "
    "or changing moments, passing side or rear glimpses. When you are not "
    "sure, answer sexual.\n"
    "explicit = (labels is non-empty) AND (context == sexual). Blood, "
    "gore, violence, or injury with no unclothed anatomy is out of scope: "
    "empty labels, explicit=false."
)
ARBITER_USER_TEXT = "Tag this frame. JSON only."


def film_context_block(film_context: str) -> str:
    """Film-context additions appended to the system prompt ('' = none).

    The context text itself (built by scanner.guide.context_block)
    carries its own STEP-2-tie-break-only wording; this stays a pure
    append so an empty context is byte-identical to v3."""
    return f"\n\n{film_context}" if film_context else ""

_REFUSAL_RE = re.compile(
    r"\b(i can(?:no|')t|cannot|can not|unable to|sorry|i won't|"
    r"not able to)\s+(help|assist|analyze|describe|provide|comply|identify)",
    re.IGNORECASE)


# --- planning (pure) ----------------------------------------------------------

@dataclass(frozen=True)
class PlannedFrame:
    t: float
    prio: int
    reason: str          # "verify" | "window" | "sweep"
    seg_idx: int | None = None


def plan_frames(duration: float, segments: list[dict], *,
                person_ranges: list[tuple[float, float]] | None = None,
                window_before_s: float = config.ARBITER_WINDOW_BEFORE_S,
                window_after_s: float = config.ARBITER_WINDOW_AFTER_S,
                window_step_s: float = config.ARBITER_WINDOW_STEP_S,
                sweep_every_s: float = config.ARBITER_SWEEP_EVERY_S,
                max_frames: int = config.ARBITER_MAX_FRAMES,
                sweep_budget_frac: float = config.ARBITER_SWEEP_BUDGET_FRAC,
                ) -> list[PlannedFrame]:
    """Curate the sparse VLM frame list.

    Trusted-source policy: `segments` here are crowd ranges, mask-final
    regardless of VLM opinion (apply mode is add-only) — the VLM NEVER
    samples inside them. It watches only where they don't fire AND
    (person gate) where somebody is present: `person_ranges` from the
    RF-DETR pass (None = gate off or failed open — the whole timeline
    is eligible). max_frames <= 0 = uncapped; a cap is applied by
    striding, never truncation."""
    if duration <= 0:
        return []
    planned: dict[float, PlannedFrame] = {}   # rounded t -> frame (max prio)

    def add(t: float, prio: int, reason: str, seg_idx: int | None = None,
            margin: float = 0.0) -> None:
        t = round(min(max(t + margin, 0.0), duration), 3)
        if t < 0.0 or t > duration:
            return
        cur = planned.get(t)
        if cur is None or prio > cur.prio:
            planned[t] = PlannedFrame(t, prio, reason, seg_idx)

    seg_bounds = [(s["start"], s["end"]) for s in segments]

    def in_segment(t: float) -> bool:
        return any(a <= t <= b for a, b in seg_bounds)

    gate_bounds = list(person_ranges) if person_ranges is not None else None

    def person_present(t: float) -> bool:
        if gate_bounds is None:
            return True                    # gate off / failed open
        return any(a <= t <= b for a, b in gate_bounds)

    # 1) windows hugging trusted segments (never inside one): their edges
    #    are where scene spill lives — starts that begin early, ends that
    #    run late. Person-gated: an empty hallway after a segment end
    #    cannot host explicit content.
    for i, (a, b) in enumerate(seg_bounds):
        prev_end = seg_bounds[i - 1][1] if i > 0 else 0.0
        w0 = max(a - window_before_s, prev_end)
        t = a - window_step_s
        while t > w0:
            if not in_segment(t) and person_present(t):
                add(t, PRIO_WINDOW, "window")
            t -= window_step_s
        next_start = seg_bounds[i + 1][0] if i + 1 < len(seg_bounds) \
            else duration
        w1 = min(b + window_after_s, next_start)
        t = b + window_step_s
        while t < w1:
            if not in_segment(t) and person_present(t):
                add(t, PRIO_WINDOW, "window")
            t += window_step_s

    # 2) global sweep for blind spots — the person-gated backbone: every
    #    sweep_every_s of timeline where a person is present and nothing
    #    is trusted
    k = 0.0
    while k <= duration:
        if not in_segment(k) and person_present(k):
            add(k, PRIO_SWEEP, "sweep")
        k += sweep_every_s

    # --- budget: sweep guaranteed a share; windows the rest.
    # max_frames <= 0 = UNCAPPED: every planned frame is judged. ---
    capped = max_frames > 0
    frames = list(planned.values())
    by_reason: dict[str, list[PlannedFrame]] = {}
    for f in sorted(frames, key=lambda f: f.t):
        by_reason.setdefault(f.reason, []).append(f)

    def thin(items: list[PlannedFrame],
             budget: int | None) -> list[PlannedFrame]:
        if budget is None or len(items) <= budget:
            return items
        step = len(items) / budget
        return [items[int(i * step)] for i in range(budget)]

    budget = max_frames if capped else None
    # sweep BEFORE windows: it is the only global coverage and loses the
    # budget race otherwise (windows hug already-flagged segments)
    sweeps = thin(by_reason.get("sweep", []),
                  min(budget, int(max_frames * sweep_budget_frac))
                  if capped else None)
    budget = (budget - len(sweeps)) if capped else None
    windows = thin(by_reason.get("window", []), budget)

    out = sorted(windows + sweeps, key=lambda f: f.t)
    return out


def plan_expansion(hit_ts: list[float], judged: set[float],
                   segments: list[dict], duration: float, *,
                   pad_s: float | None = None,
                   step_s: float = config.ARBITER_WINDOW_STEP_S,
                   max_frames: int = config.ARBITER_EXPAND_MAX_FRAMES,
                   ) -> list[PlannedFrame]:
    """Second-pass frames measuring the extent of explicit point hits.

    Sweep hits are point samples padded blind (±ARBITER_SWEEP_PAD_S).
    When one flips explicit, its neighborhood goes back on the VLM at
    window density (1s grid): neighbors that also flip chain with the
    original hit and measure the scene's real extent — a measured segment
    instead of a symmetric guess. `hit_ts` is the explicit point-hit
    times, highest-confidence first (budget spends there first). Pure.
    """
    if pad_s is None:
        pad_s = config.ARBITER_SWEEP_PAD_S
    seg_bounds = [(s["start"], s["end"]) for s in segments]

    def in_segment(t: float) -> bool:
        return any(a <= t <= b for a, b in seg_bounds)

    cand: dict[float, PlannedFrame] = {}
    for ht in hit_ts:
        k = ht - pad_s
        while k <= ht + pad_s:
            t = round(min(max(k, 0.0), duration), 3)
            if t not in judged and t not in cand and not in_segment(t):
                cand[t] = PlannedFrame(t, PRIO_WINDOW, "expand")
                if len(cand) >= max_frames:
                    break
            k += step_s
        if len(cand) >= max_frames:
            break
    return sorted(cand.values(), key=lambda f: f.t)


# --- verdict parsing (pure) ---------------------------------------------------

def parse_verdict(text: str) -> dict | None:
    """-> {"explicit": bool, "labels": [...], "confidence": float} | None.

    None = refusal / unparseable (counted separately, treated as invalid).
    """
    if not text:
        return None
    if _REFUSAL_RE.search(text):
        return None
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("explicit"), bool):
        return None
    labels = [l for l in (data.get("labels") or [])
              if isinstance(l, str) and l.upper() in _LABEL_WHITELIST]
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"explicit": data["explicit"],
            "labels": labels,
            "confidence": min(max(conf, 0.0), 1.0)}


# --- aggregation (pure) --------------------------------------------------------

def aggregate(planned: list[PlannedFrame], verdicts: list[dict | None],
              duration: float, smoothing: SmoothingConfig,
              existing_segments: list[dict]) -> dict:
    """Fold per-frame verdicts into segment verdicts + new blind-spot segments.

    Segment rules (its verify frames, refusals skipped):
      confirmed — >= half explicit      rejected — zero explicit (>=2 valid)
      uncertain — anything between / 1 valid frame
    New segments cluster explicit window/sweep hits by TIME proximity
    (chain gap derived from the actual frame spacing) — never by array
    adjacency, which would chain hits minutes apart into phantom spans
    (the reason smooth_segments is not reused here: its consecutive-sample
    semantics assume a dense, uniform timeline).
    """
    refusals = sum(1 for v in verdicts if v is None)

    seg_verdicts: list[dict] = []
    for i, seg in enumerate(existing_segments):
        vs = [v for f, v in zip(planned, verdicts)
              if f.seg_idx == i and v is not None]
        n_ex = sum(1 for v in vs if v["explicit"])
        if not vs:
            verdict, ratio, conf = "unclear", 0.0, 0.0
        else:
            ratio = n_ex / len(vs)
            conf = sum(v["confidence"] for v in vs) / len(vs)
            verdict = ("confirmed" if ratio >= 0.5 and len(vs) >= 2
                       else "rejected" if ratio == 0.0 and len(vs) >= 2
                       else "uncertain")
        seg_verdicts.append({
            "start": seg["start"], "end": seg["end"],
            "explicit_frames": n_ex, "valid_frames": len(vs),
            "verdict": verdict,
            "mean_confidence": round(conf, 3),
        })

    # hits outside segments -> new-segment candidates. Sweep hits are
    # POINT samples (no 1s grid around them): pad them wide; window hits
    # keep the normal pad (their neighborhood is densely sampled).
    hits = sorted((f.t, v["confidence"],
                   config.ARBITER_SWEEP_PAD_S
                   if f.reason == "sweep" else smoothing.pad_s)
                  for f, v in zip(planned, verdicts)
                  if v is not None and v["explicit"] and f.seg_idx is None)
    chain_gap = _chain_gap(planned)
    raw_segs = _cluster_hits(hits, chain_gap)
    clipped: list[dict] = []
    for a, b, score in raw_segs:
        pieces = _subtract([(a, b)],
                           [(s["start"], s["end"])
                            for s in existing_segments])
        for x, y in pieces:
            if y - x >= 0.5:
                clipped.append({"start": round(x, 3), "end": round(y, 3),
                                "mean_score": round(score, 3),
                                "labels": {"arbiter": 1}})
    return {
        "segment_verdicts": seg_verdicts,
        "new_segments": clipped,
        "rejected_segments": [
            {"start": sv["start"], "end": sv["end"]}
            for sv in seg_verdicts if sv["verdict"] == "rejected"],
        "refusals": refusals,
        "chain_gap_s": round(chain_gap, 2),
    }


def _chain_gap(planned: list[PlannedFrame]) -> float:
    """Merge distance for explicit hits: 1.5x the typical spacing of the
    sparse arbiter frames (1fps windows, wider when thinned), clamped to
    [2.5, 6.0]."""
    ts = sorted(f.t for f in planned if f.seg_idx is None)
    if len(ts) < 2:
        return 2.5
    diffs = sorted(b - a for a, b in zip(ts, ts[1:]) if b - a > 0.01)
    if not diffs:
        return 2.5
    typical = diffs[len(diffs) // 2]      # median spacing
    return min(6.0, max(2.5, typical * 1.5))


def _cluster_hits(hits: list[tuple], chain_gap: float
                  ) -> list[tuple]:
    """[(t, conf, pad_s)] -> [(start, end, mean_conf)], each hit padded by
    its own pad (sweep hits pad wide), chained by chain_gap."""
    segs: list[tuple] = []
    for t, conf, pad in hits:
        a, b = t - pad, t + pad
        if segs and a - segs[-1][1] <= chain_gap:
            pa, pb, (pc, pn) = segs[-1]
            segs[-1] = (pa, max(pb, b), (pc + conf, pn + 1))
        else:
            segs.append((a, b, (conf, 1)))
    return [(a, b, c / n) for a, b, (c, n) in segs]


def _subtract(intervals: list[tuple], holes: list[tuple]) -> list[tuple]:
    """Clip `intervals` by removing all `holes` (both sorted, closed)."""
    out = []
    for a, b in intervals:
        cur = [(a, b)]
        for h0, h1 in holes:
            nxt = []
            for x, y in cur:
                if h1 <= x or h0 >= y:
                    nxt.append((x, y))
                else:
                    if x < h0:
                        nxt.append((x, h0))
                    if h1 < y:
                        nxt.append((h1, y))
            cur = nxt
        out.extend(cur)
    return out


# --- frame extraction ----------------------------------------------------------

def extract_frames(video_path: str, times: list[float], workdir: Path,
                   frame_w: int = config.ARBITER_FRAME_W) -> list[Path]:
    """Extract JPEGs for `times` (sorted, seconds) in `times` order.

    Requests are first snapped to actual frame timestamps (demux-only
    ffprobe pass), so one exact-window ffmpeg select pass delivers
    everything. The old count-matched pass requested the sample grid
    directly: ms-quantized container timebases (mkv) sit up to half a
    frame interval off the grid, only ~26% of +-5ms windows hit (GG:
    478/1867), and EVERY frame was then re-extracted by individual
    seeks. Now only genuinely missing frames (probe failure, VFR
    oddity) take the per-frame seek path.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for old in workdir.glob("frame_*.jpg"):
        old.unlink()                # stale files broke count checks
    snapped = _snap_times(times, _video_frame_pts(video_path))
    uniq = sorted({s for s in snapped if s is not None})
    by_time: dict[float, Path] = {}
    # small batches go straight to per-seek: a select pass costs one
    # full-movie decode (~126s on GG) while seeks cost ~0.2s/frame —
    # under a few hundred frames seeking is cheaper than decoding the
    # movie again (the sweep is ~1.9k frames -> select; window and
    # expansion passes are ~0.2k -> seek)
    if uniq and len(uniq) >= config.ARBITER_SELECT_MIN:
        files = _extract_select(video_path, uniq, workdir, frame_w)
        if files is not None:
            by_time = dict(zip(uniq, files))
    missing = sum(1 for s in snapped if s is None or s not in by_time)
    if missing:
        logger.warning("[arbiter] %d/%d frames missing after select "
                       "pass — per-frame seek for those only",
                       missing, len(times))
    out: list[Path] = []
    for t, s in zip(times, snapped):
        f = by_time.get(s) if s is not None else None
        if f is None:
            f = _extract_seek_one(video_path, t, workdir, len(out),
                                  frame_w)
        if f is not None:
            out.append(f)
    return out


def _video_frame_pts(video_path: str) -> list[float]:
    """Video packet PTS seconds, demux-only (no decode — seconds for a
    full movie). These are the frame timestamps the select filter's
    `t` actually takes; requesting anything else is what silently
    starved the old pass on ms-timebase containers."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time", "-of", "csv=p=0",
           video_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             check=True, timeout=600).stdout
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("[arbiter] packet-pts probe failed: %s", e)
        return []
    pts: list[float] = []
    for line in out.splitlines():
        try:
            pts.append(float(line.strip()))
        except ValueError:
            pass
    return pts


def _snap_times(times: list[float],
                pts: list[float]) -> list[float | None]:
    """Snap each requested time to the nearest actual frame timestamp,
    -> None where no frame sits within half a frame interval (those
    keep their original time and take the seek fallback)."""
    pts = sorted(pts)   # ffprobe emits packets in DECODE order
    # (B-frames); bisect requires presentation order
    if not pts:
        return [None] * len(times)
    gaps = sorted(b - a for a, b in zip(pts, pts[1:]) if b > a)
    # 0.5x the median gap: a request can sit at most half a frame
    # interval from its nearest frame (worst case: dead center between
    # two). +2ms absorbs sub-ms float/timestamp noise at that boundary.
    # Request spacing must exceed one frame interval to guarantee no
    # double-mapping — the arbiter samples at >=0.25s vs ~42ms
    # frames, so that holds by construction.
    half_frame = 0.5 * (gaps[len(gaps) // 2] if gaps else 1 / 24.0) + 0.002
    out: list[float | None] = []
    for t in times:
        i = bisect.bisect_left(pts, t)
        cand = [j for j in (i - 1, i) if 0 <= j < len(pts)]
        if not cand:
            out.append(None)
            continue
        j = min(cand, key=lambda j: abs(pts[j] - t))
        out.append(pts[j] if abs(pts[j] - t) <= half_frame else None)
    return out


def _extract_select(video_path: str, times: list[float], workdir: Path,
                    frame_w: int) -> list[Path] | None:
    """One exact-window ffmpeg select pass over the movie. -> sorted
    paths when exactly len(times) frames came out, else None (caller
    per-seeks the gap; partial output cannot be aligned reliably)."""
    if not times:
        return []
    eps = 0.005
    terms = "+".join(f"between(t,{t - eps:.3f},{t + eps:.3f})" for t in times)
    out_base = workdir / "frame"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if shutil.which("nvidia-smi"):
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", video_path,
            "-vf", f"select='{terms}',scale='min({frame_w},iw)':-2",
            "-fps_mode", "vfr", "-q:v", "3",
            str(out_base) + "_%05d.jpg"]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"")[-300:].decode(errors="replace")
        logger.warning("[arbiter] select pass failed (rc=%s): %s",
                       e.returncode, tail)   # never log the multi-MB cmd
        return None
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("[arbiter] select pass failed: %s", e)
        return None
    files = sorted(workdir.glob("frame_*.jpg"))
    return files if len(files) == len(times) else None


def _extract_seek_one(video_path: str, t: float, workdir: Path, i: int,
                      frame_w: int) -> Path | None:
    """Single-frame seek extraction — the fallback for frames the
    select pass could not deliver."""
    dest = workdir / f"seek_{i:05d}_{t:.2f}.jpg"
    dest.unlink(missing_ok=True)     # same name can recur across a run's
    # successive extract calls; never return a stale frame
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, t - 0.05):.3f}", "-i", video_path,
           "-frames:v", "1",
           "-vf", f"scale='min({frame_w},iw)':-2", "-q:v", "3",
           str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("[arbiter] seek extract failed at %.2fs: %s", t, e)
        return None
    return dest if dest.exists() else None


# --- HTTP backend (OpenAI-compatible sidecar, e.g. llama-server) ---------------
#
# Same prompts and verdict contract as the in-process path; the sidecar
# owns the model (any llama.cpp-compatible VLM — a Qwen3-VL-30B-A3B MoE
# measured 0.054s/frame at 4 concurrent slots, ~2.6x the in-process 8B's
# batched rate, see NSFW_SCANNER_DESIGN.md §10). Enabled by
# ARBITER_BASE_URL; per-frame failures degrade to "" which parse_verdict
# counts as a refusal — the scan itself never fails on the sidecar.

_HTTP_MAX_TOKENS = 120        # verdict JSON + template overhead (~20 measured)
_THINK_KWARG_OK: bool | None = None   # None = probe; set after first response


def http_request_body(image_b64: str, *, thinking_off: bool = True,
                      film_context: str = "") -> dict:
    """Pure: /v1/chat/completions body for one frame (JPEG data URL)."""
    body: dict = {
        "max_tokens": _HTTP_MAX_TOKENS,
        "temperature": 0.0,
        "messages": [
            {"role": "system",
             "content": ARBITER_SYSTEM_PROMPT
                        + film_context_block(film_context)},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": ARBITER_USER_TEXT}]}]}
    if thinking_off:
        # thinking-capable templates (Qwen3.8-27B) burn the whole budget
        # on reasoning and return an empty verdict; templates that reject
        # the kwarg 400 and are probed once without it
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def http_response_text(response: dict) -> str:
    """Pure: content text from a chat completion ("" on odd shapes)."""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _http_judge_one(url: str, path: Path, film_context: str = "") -> str:
    global _THINK_KWARG_OK
    b64 = base64.b64encode(path.read_bytes()).decode()
    attempts = ([_THINK_KWARG_OK] if _THINK_KWARG_OK is not None
                else [True, False])
    for thinking_off in attempts:
        body = json.dumps(
            http_request_body(b64, thinking_off=thinking_off,
                              film_context=film_context)).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(
                    req, timeout=config.ARBITER_HTTP_TIMEOUT_S) as r:
                out = http_response_text(json.load(r))
                if _THINK_KWARG_OK is None:
                    _THINK_KWARG_OK = thinking_off
                return out
        except urllib.error.HTTPError as e:
            if e.code == 400 and _THINK_KWARG_OK is None and thinking_off:
                _THINK_KWARG_OK = False    # template rejects the kwarg
                continue
            logger.warning("[arbiter] http judge failed on %s: %s", path, e)
            return ""
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("[arbiter] http judge error on %s: %s", path, e)
            return ""
    return ""


def _judge_with_http(files: list[Path],
                     film_context: str = "") -> list[str]:
    """Judge frames against the sidecar concurrently — its slots are the
    throughput unit (ARBITER_HTTP_CONCURRENCY should match them)."""
    url = config.ARBITER_BASE_URL.rstrip("/") + "/v1/chat/completions"
    with ThreadPoolExecutor(
            max_workers=config.ARBITER_HTTP_CONCURRENCY) as ex:
        return list(ex.map(lambda f: _http_judge_one(url, f, film_context),
                           files))


# --- runner (lazy torch/transformers — runs in the container only) -------------


def resolve_vlm_path(model_id: str) -> str:
    """Baked/local directory for the model if present, else the id as-is.

    Resolution order: an existing path verbatim, then a case-insensitive
    match of the repo name among VLM_DIR's children (lets "Qwen3-VL-8B-Instruct"
    resolve a baked "qwen3-vl-8b-instruct" directory). The container sets
    HF_HUB_OFFLINE=1 with weights baked into /opt/vlm, so an unresolved id
    fails fast instead of downloading at runtime.
    """
    p = Path(model_id)
    if p.is_dir():
        return str(p)
    name = p.name.lower()
    try:
        children = list(config.VLM_DIR.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return model_id
    for child in children:
        if child.is_dir() and child.name.lower() == name:
            return str(child)
    return model_id


def is_hf_weights_dir(path: str | Path) -> bool:
    """True if `path` is a transformers-loadable weights directory.

    config.json is the canonical HF marker; a GGUF dir (llama-server
    weights, e.g. the shipped qwen3-vl-30b-a3b) is deliberately NOT —
    the in-process path must reject it instead of dying inside
    AutoProcessor. Pure."""
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file()


def _load_vlm(model_id: str):
    """Load (model, processor); bf16 on CUDA. Not cached across runs: the
    8B model holds ~17GB of VRAM, and parking it idle between scans would
    starve everything else on the GPU. Disk load is ~1-2 min per arbiter
    run — cheap next to the inference itself."""
    resolved = resolve_vlm_path(model_id)
    import torch  # noqa: F401 — container-only
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    t0 = time.monotonic()
    logger.info("[arbiter] loading VLM %s (from %s)", model_id, resolved)
    processor = AutoProcessor.from_pretrained(resolved)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    logger.info("[arbiter] loaded %s in %.1fs", model_id,
                time.monotonic() - t0)
    return model, processor


def _unload_vlm() -> None:
    """Release VLM VRAM. Callers must have dropped their own references
    BEFORE calling (del/None their locals): empty_cache() only returns
    *unreferenced* blocks to the driver, so a caller frame still holding
    the model keeps all ~17GB parked in the allocator pool."""
    import gc

    import torch
    for _ in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[arbiter] VLM unloaded")


def _judge_with_vlm(files: list[Path], model_id: str,
                    film_context: str = "") -> list[str]:
    """Load, judge, and fully release the VLM. run_arbiter must never hold
    model references itself — see _unload_vlm."""
    model = processor = None
    try:
        model, processor = _load_vlm(model_id)
        return _judge_batch(files, model, processor,
                            film_context=film_context)
    finally:
        model = processor = None      # drop frame refs BEFORE unload
        _unload_vlm()


def _judge_batch(files: list[Path], model, processor,
                 batch_size: int = 8, *,
                 film_context: str = "") -> list[str]:
    """Raw model outputs for the frame files, batched.

    Two-step input prep (visiontouch pattern, proven on transformers
    5.16.1 + Qwen3-VL): render the chat template to a string first, then
    tokenize via processor(text=..., images=...) — the single-shot
    apply_chat_template(tokenize=True) path does not expand image
    placeholders for this model, leaving it to answer blind.
    """
    import torch
    from PIL import Image

    texts: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(files), batch_size):
            chunk = files[i:i + batch_size]
            images = [Image.open(f).convert("RGB") for f in chunk]
            prompts = [processor.apply_chat_template(
                [{"role": "system",
                  "content": ARBITER_SYSTEM_PROMPT
                             + film_context_block(film_context)},
                 {"role": "user", "content": [
                     {"type": "image", "image": img},
                     {"type": "text", "text": ARBITER_USER_TEXT}]}],
                add_generation_prompt=True, tokenize=False)
                for img in images]
            inputs = processor(text=prompts, images=images,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen = model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id)
            new = gen[:, inputs["input_ids"].shape[1]:]
            texts.extend(processor.batch_decode(new, skip_special_tokens=True))
    return texts


def run_arbiter(video_path: str, duration: float, segments: list[dict],
                *, smoothing: SmoothingConfig,
                model_id: str = config.DEFAULT_ARBITER_MODEL,
                mode: str = "verify",
                person_ranges: list[tuple[float, float]] | None = None,
                workdir: Path | None = None,
                film_context: str = "") -> dict:
    """Full sparse arbitration pass. Never raises: returns an error block.

    `segments` are the TRUSTED ranges (crowd): never sampled inside.
    `person_ranges` gates the sweep/windows (None = gate off or failed
    open = whole timeline). `film_context` (guide=true) appends community
    parents-guide items to the system prompt as STEP-2 tie-break
    background; '' keeps the prompt byte-identical."""
    t0 = time.monotonic()
    result: dict = {"model": model_id, "mode": mode, "frames_checked": 0,
                    "refusals": 0, "segment_verdicts": [], "new_segments": [],
                    "rejected_segments": [], "error": None}
    workdir = workdir or Path(config.SCAN_WORKSPACE_DIR) / "arbiter"
    vlm_acquired = False
    try:
        planned = plan_frames(duration, segments,
                              person_ranges=person_ranges)
        result["planned_frames"] = len(planned)
        result["planned_by_reason"] = dict(
            Counter(f.reason for f in planned))
        if not planned:
            result["wall_time_s"] = round(time.monotonic() - t0, 2)
            return result

        # on-demand VLM: bring llama-server up BEFORE frame extraction so
        # the model load overlaps ffmpeg. A bring-up failure raises into
        # the handler below -> loud error block, scan continues without
        # VLM discovery (the same contract as any arbiter failure).
        if config.ARBITER_BASE_URL:
            vlm.acquire()
            vlm_acquired = True

        shutil.rmtree(workdir, ignore_errors=True)  # stale frames from prior run
        files = extract_frames(video_path, [f.t for f in planned], workdir)
        model = processor = None
        loaded_transformers = False
        try:
            if config.ARBITER_BASE_URL:
                result["backend"] = "http"
                result["backend_url"] = config.ARBITER_BASE_URL

                def judge(fs: list[Path]) -> list[str]:
                    return _judge_with_http(fs, film_context)
            else:
                resolved = resolve_vlm_path(model_id)
                if not is_hf_weights_dir(resolved):
                    raise RuntimeError(
                        f"arbiter requested without ARBITER_BASE_URL and "
                        f"'{resolved}' is not an HF weights dir "
                        f"(arbiter_model={model_id!r}). The default "
                        "Qwen3-VL-30B-A3B ships as GGUF for llama-server: "
                        "set ARBITER_BASE_URL (or run the all-in-one image, "
                        "whose supervisor owns the server). In-process "
                        "transformers judging is opt-in — point arbiter_model "
                        "at a local HF-format dir (config.json + "
                        "safetensors). Nothing downloads at runtime.")
                model, processor = _load_vlm(model_id)
                loaded_transformers = True

                def judge(fs: list[Path]) -> list[str]:
                    return _judge_batch(fs, model, processor,
                                        film_context=film_context)

            raw = judge(files)
            all_planned = list(planned)
            verdicts = [parse_verdict(r) for r in raw]

            # second pass: explicit point hits (sweep) get their
            # neighborhood measured at window density with the SAME loaded
            # model — converts a blind ±pad guess into a chained, measured
            # segment. Highest-confidence hits first; capped. Add-only:
            # non-explicit neighbors never shrink the hit's own pad.
            point_hits = sorted(
                ((f, v) for f, v in zip(planned, verdicts)
                 if v is not None and v["explicit"] and f.seg_idx is None
                 and f.reason == "sweep"),
                key=lambda fv: -fv[1]["confidence"])
            if point_hits:
                exp = plan_expansion(
                    [f.t for f, _ in point_hits],
                    {f.t for f in planned}, segments, duration)
                if exp:
                    exp_files = extract_frames(
                        video_path, [f.t for f in exp], workdir)
                    raw2 = judge(exp_files)
                    all_planned += exp
                    verdicts += [parse_verdict(r) for r in raw2]
                    result["expansion_frames"] = len(exp)

            result["refusals"] = sum(1 for v in verdicts if v is None)
            result["frames_checked"] = len(verdicts)
            # per-frame verdicts: trust/debug data for review UIs
            result["frame_verdicts"] = [
                {"t": f.t, "reason": f.reason,
                 **(v if v is not None else {"explicit": None})}
                for f, v in zip(all_planned, verdicts)]
            result.update(aggregate(all_planned, verdicts, duration,
                                    smoothing, segments))
        finally:
            if loaded_transformers:
                model = processor = None  # drop frame refs BEFORE unload
                _unload_vlm()
    except Exception as e:  # noqa: BLE001 — arbiter must not fail the scan
        logger.exception("[arbiter] failed")
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        if vlm_acquired:
            vlm.release()   # last releaser stops llama-server, frees ~20GB
        result["wall_time_s"] = round(time.monotonic() - t0, 2)
    return result
