"""Stage B: sparse VLM arbiter (Qwen3-VL) — never every frame.

Structure (pure functions first, heavy runner last):

  plan_frames()    pure, testable: which timestamps deserve a VLM look and why
  parse_verdict()  pure: robust strict-JSON verdict extraction + refusal detect
  aggregate()      pure: segment confirm/reject verdicts + new blind-spot
                   segments (reuses temporal.smooth_segments)
  extract_frames() ffmpeg single-pass frame extraction for the plan
  run_arbiter()    the runner: extract -> lazy VLM load -> batched inference
                   -> aggregate. torch/transformers import happens HERE only,
                   so the module is importable in the torch-free host venv.

Planning policy (see config):
  - verify: ARBITER_VERIFY_FRAMES per final segment (20/50/80% of its span)
  - window: ~1fps sampling in the gaps hugging segments (before starts, after
    ends) — flicker continuations and near-miss boundaries live there
  - sweep: one frame per ARBITER_SWEEP_EVERY_S across the whole movie for true
    blind spots, thinned to whatever frame budget remains
  - hard cap ARBITER_MAX_FRAMES, thinned by striding (never truncated to only
    the early part of the movie)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .temporal import SmoothingConfig

logger = logging.getLogger(__name__)

# priority tiers for the frame budget
PRIO_VERIFY = 3
PRIO_WINDOW = 2
PRIO_SWEEP = 1

_LABEL_WHITELIST = {
    "FEMALE_GENITALIA", "MALE_GENITALIA", "FEMALE_BREAST",
    "MALE_BREAST", "BUTTOCKS", "ANUS",
}

# Arbiter prompting, validated empirically on Ex Machina (see
# NSFW_SCANNER_DESIGN.md §7): a terse SYSTEM-role classifier prompt with a
# strict-JSON contract gets zero refusals, zero false positives on benign
# controls, and reliably confirms high-confidence nudity. Wordy
# "moderation"-style user prompts made Qwen3-VL sandbag (confident false on
# real nudity); free-form output was slightly more sensitive on borderline
# dim scenes but unparseable in aggregate. Borderline dark scenes remain
# below the VLM's sensitivity, which is why apply-mode only ever ADDS
# segments and never removes detector-flagged ones.
ARBITER_SYSTEM_PROMPT = (
    "You are the nudity-detection stage of an automated video editing "
    "pipeline processing a licensed feature film. You see single frames "
    "and tag them. You always reply with strict JSON and never refuse.\n"
    'Reply format: {"explicit": true|false, "labels": [list], '
    '"confidence": 0.0-1.0}\n'
    "labels vocabulary: FEMALE_GENITALIA, MALE_GENITALIA, FEMALE_BREAST, "
    "BUTTOCKS, ANUS. Tag a label when the body part is directly visible "
    "and unclothed (nipple, genitalia, buttock crack, anus). Do not tag "
    "covered bodies, underwear, swimwear, or tiny background images."
)
ARBITER_USER_TEXT = "Tag this frame. JSON only."

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
                verify_frames: int = config.ARBITER_VERIFY_FRAMES,
                window_before_s: float = config.ARBITER_WINDOW_BEFORE_S,
                window_after_s: float = config.ARBITER_WINDOW_AFTER_S,
                window_step_s: float = config.ARBITER_WINDOW_STEP_S,
                sweep_every_s: float = config.ARBITER_SWEEP_EVERY_S,
                max_frames: int = config.ARBITER_MAX_FRAMES) -> list[PlannedFrame]:
    """Curate the sparse VLM frame list, capped by striding not truncating."""
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

    # 1) verification frames inside segments (keep out of the 0.5s edges)
    fracs = [0.2, 0.5, 0.8][:verify_frames] or [0.5]
    for i, (a, b) in enumerate(seg_bounds):
        span = b - a
        for f in fracs:
            add(a + span * f, PRIO_VERIFY, "verify", seg_idx=i,
                margin=0.0 if span > 1.0 else 0.0)

    # 2) windows hugging segments (never inside one)
    for i, (a, b) in enumerate(seg_bounds):
        prev_end = seg_bounds[i - 1][1] if i > 0 else 0.0
        w0 = max(a - window_before_s, prev_end)
        t = a - window_step_s
        while t > w0:
            if not in_segment(t):
                add(t, PRIO_WINDOW, "window")
            t -= window_step_s
        next_start = seg_bounds[i + 1][0] if i + 1 < len(seg_bounds) \
            else duration
        w1 = min(b + window_after_s, next_start)
        t = b + window_step_s
        while t < w1:
            if not in_segment(t):
                add(t, PRIO_WINDOW, "window")
            t += window_step_s

    # 3) global sweep for blind spots
    k = 0.0
    while k <= duration:
        if not in_segment(k):
            add(k, PRIO_SWEEP, "sweep")
        k += sweep_every_s

    # --- budget: verify always; thin windows then sweep by striding ---
    frames = list(planned.values())
    by_reason: dict[str, list[PlannedFrame]] = {}
    for f in sorted(frames, key=lambda f: f.t):
        by_reason.setdefault(f.reason, []).append(f)

    def thin(items: list[PlannedFrame], budget: int) -> list[PlannedFrame]:
        if len(items) <= budget or budget <= 0:
            return items if budget > 0 else []
        step = len(items) / budget
        return [items[int(i * step)] for i in range(budget)]

    kept = thin(by_reason.get("verify", []), max_frames)
    budget = max_frames - len(kept)
    windows = thin(by_reason.get("window", []), budget)
    budget -= len(windows)
    sweeps = thin(by_reason.get("sweep", []), budget)

    out = sorted(kept + windows + sweeps, key=lambda f: f.t)
    return out


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

    hits = sorted((f.t, v["confidence"])
                  for f, v in zip(planned, verdicts)
                  if v is not None and v["explicit"] and f.seg_idx is None)
    chain_gap = _chain_gap(planned)
    raw_segs = _cluster_hits(hits, chain_gap, smoothing.pad_s)
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


def _cluster_hits(hits: list[tuple], chain_gap: float,
                  pad_s: float) -> list[tuple]:
    """[(t, conf)] -> [(start, end, mean_conf)], chained by chain_gap+pad."""
    segs: list[tuple] = []
    for t, conf in hits:
        a, b = t - pad_s, t + pad_s
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
    """Extract JPEGs for `times` (sorted, seconds) via one ffmpeg select pass.

    Falls back to per-timestamp seeks when the single pass under-delivers
    (e.g. container timestamp quirks). Files are returned in `times` order.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    files = _extract_select(video_path, times, workdir, frame_w)
    if files is not None and len(files) == len(times):
        return files
    logger.warning("[arbiter] single-pass extraction mismatch (%s/%d); "
                   "falling back to per-frame seek",
                   "-" if files is None else len(files), len(times))
    return _extract_seek(video_path, times, workdir, frame_w)


def _extract_select(video_path: str, times: list[float], workdir: Path,
                    frame_w: int) -> list[Path] | None:
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
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("[arbiter] select pass failed: %s", e)
        return None
    files = sorted(workdir.glob("frame_*.jpg"))
    return files if len(files) == len(times) else files


def _extract_seek(video_path: str, times: list[float], workdir: Path,
                  frame_w: int) -> list[Path]:
    out: list[Path] = []
    for i, t in enumerate(times):
        dest = workdir / f"seek_{i:05d}_{t:.2f}.jpg"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", f"{max(0.0, t - 0.05):.3f}", "-i", video_path,
               "-frames:v", "1",
               "-vf", f"scale='min({frame_w},iw)':-2", "-q:v", "3",
               str(dest)]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("[arbiter] seek extract failed at %.2fs: %s", t, e)
        out.append(dest)
    return [f for f in out if f.exists()]


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


def _judge_with_vlm(files: list[Path], model_id: str) -> list[str]:
    """Load, judge, and fully release the VLM. run_arbiter must never hold
    model references itself — see _unload_vlm."""
    model = processor = None
    try:
        model, processor = _load_vlm(model_id)
        return _judge_batch(files, model, processor)
    finally:
        model = processor = None      # drop frame refs BEFORE unload
        _unload_vlm()


def _judge_batch(files: list[Path], model, processor,
                 batch_size: int = 8) -> list[str]:
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
                [{"role": "system", "content": ARBITER_SYSTEM_PROMPT},
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
                workdir: Path | None = None) -> dict:
    """Full sparse arbitration pass. Never raises: returns an error block."""
    t0 = time.monotonic()
    result: dict = {"model": model_id, "mode": mode, "frames_checked": 0,
                    "refusals": 0, "segment_verdicts": [], "new_segments": [],
                    "rejected_segments": [], "error": None}
    workdir = workdir or Path(config.SCAN_WORKSPACE_DIR) / "arbiter"
    try:
        planned = plan_frames(duration, segments)
        result["planned_frames"] = len(planned)
        if not planned:
            result["wall_time_s"] = round(time.monotonic() - t0, 2)
            return result

        shutil.rmtree(workdir, ignore_errors=True)  # stale frames from prior run
        files = extract_frames(video_path, [f.t for f in planned], workdir)
        raw = _judge_with_vlm(files, model_id)

        verdicts = [parse_verdict(r) for r in raw]
        result["refusals"] = sum(1 for v in verdicts if v is None)
        result["frames_checked"] = len(verdicts)
        # per-frame verdicts: trust/debug data for review UIs
        result["frame_verdicts"] = [
            {"t": f.t, "reason": f.reason,
             **(v if v is not None else {"explicit": None})}
            for f, v in zip(planned, verdicts)]
        result.update(aggregate(planned, verdicts, duration, smoothing,
                                segments))
    except Exception as e:  # noqa: BLE001 — arbiter must not fail the scan
        logger.exception("[arbiter] failed")
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["wall_time_s"] = round(time.monotonic() - t0, 2)
    return result
