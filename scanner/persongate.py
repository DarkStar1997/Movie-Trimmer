"""Stage A2: RF-DETR person gate — where the arbiter is allowed to look.

The VLM sweep exists for content only a vision LLM catches (dim, distant,
color-graded) — but a frame with no person in it cannot host that content.
This stage answers "is anybody here" at PERSON_GATE_FPS, piggybacked on
the scan's existing decode stream (no second decode pass), on its own
CUDA stream so the gate engine overlaps decode instead of serializing
behind it, then smooths per-sample answers into person_ranges
the arbiter planner intersects its sweep with.

Failure policy: FAIL-OPEN. If the engine cannot init or the pass dies
midway, ranges=None — the planner then watches the whole timeline, i.e.
exactly the pre-gate behavior. A gate problem must cost VLM time, never
coverage.
"""

from __future__ import annotations

import logging
import threading

from . import config
from .personseg import PersonSegmenter, person_presence

logger = logging.getLogger(__name__)

_singleton: PersonSegmenter | None = None
_singleton_lock = threading.Lock()


def _segmenter(backend: str) -> PersonSegmenter:
    """One shared segmenter (engine deserialize ~1s; scans reuse it)."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = PersonSegmenter(backend)
        return _singleton


def smooth_person_presence(times: list[float], present: list[bool], *,
                            open_min: int, close_gap: int
                            ) -> list[tuple[float, float]]:
    """Hysteresis-smooth uniform presence samples into (start, end) ranges.

    open_min   — consecutive present samples to OPEN a range (single-sample
                 noise — a poster, a misfire — never opens one)
    close_gap  — consecutive absent samples tolerated before CLOSING (brief
                 occlusion / person leaving frame for a beat stays one range)

    Range bounds are the first/last present sample times. Pure."""
    n = len(times)
    ranges: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not present[i]:
            i += 1
            continue
        j = i
        while j < n and present[j]:
            j += 1
        if j - i < open_min:
            i = j
            continue
        start, end = times[i], times[j - 1]
        # bridge short absence gaps into the open range
        while True:
            a = j
            while a < n and not present[a]:
                a += 1
            if a < n and (a - j) <= close_gap:
                b = a
                while b < n and present[b]:
                    b += 1
                end = times[b - 1]
                j = b
            else:
                break
        ranges.append((round(start, 3), round(end, 3)))
        i = j
    return ranges


class PersonGateRunner:
    """Fed batches by the scan loop (the decode samples, strided to
    PERSON_GATE_FPS); finish() returns smoothed ranges + stats.

    TRT path enqueues per frame on a side stream and joins per batch, so
    the gate engine overlaps decode. ORT path (host tests)
    is synchronous. Any exception fails the gate open (ranges=None)."""

    def __init__(self, backend: str = "trt"):
        self.segmenter = _segmenter(backend)
        self.backend = backend
        self._stream = None
        if backend == "trt":
            self._stream = self.segmenter._torch.cuda.Stream()
        self._pending: list[tuple[float, object, str]] = []
        self.times: list[float] = []
        self.scores: list[float] = []
        self.error: str | None = None

    # -- scan loop interface ------------------------------------------------
    def feed_batch(self, frames: list, first_idx: int, eff_fps: float,
                   stride: int) -> None:
        """Enqueue gate passes for every stride-th sampled frame."""
        if self.error:
            return
        try:
            for j, frame in enumerate(frames):
                idx = first_idx + j
                if idx % stride:
                    continue
                t = idx / eff_fps
                if self._stream is not None:
                    tensor, name = self.segmenter.presence_enqueue(
                        frame, self._stream)
                    self._pending.append((t, tensor, name))
                else:
                    _, score = self.segmenter.presence(frame)
                    self.times.append(t)
                    self.scores.append(score)
        except Exception as e:  # noqa: BLE001 — fail-open by contract
            self._fail(e)

    def sync(self) -> None:
        """Join the side stream and drain pending results to scores."""
        if self.error:
            return
        try:
            if self._stream is not None and self._pending:
                self._stream.synchronize()
                for t, tensor, name in self._pending:
                    outs = {"labels": tensor.float().cpu().numpy()}
                    _, score = person_presence(outs, {"labels": name}, 0.0)
                    self.times.append(t)
                    self.scores.append(score)
                self._pending.clear()
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def finish(self, duration: float
               ) -> tuple[list[tuple[float, float]] | None, dict]:
        """-> (ranges | None (fail-open), stats). Threshold applied here, so
        the effective PERSON_GATE_SCORE is visible in the stats block."""
        if self.error:
            return None, {"enabled": True, "error": self.error}
        self.sync()
        if self.error:
            return None, {"enabled": True, "error": self.error}
        thr = config.PERSON_GATE_SCORE
        present = [s >= thr for s in self.scores]
        ranges = smooth_person_presence(
            self.times, present,
            open_min=config.PERSON_GATE_OPEN_SAMPLES,
            close_gap=config.PERSON_GATE_CLOSE_GAP_SAMPLES)
        coverage = sum(b - a for a, b in ranges)
        stats = {
            "enabled": True,
            "backend": self.backend,
            "samples": len(self.times),
            "present_samples": sum(present),
            "threshold": thr,
            "open_samples": config.PERSON_GATE_OPEN_SAMPLES,
            "close_gap_samples": config.PERSON_GATE_CLOSE_GAP_SAMPLES,
            "ranges": [{"start": a, "end": b} for a, b in ranges],
            "coverage_s": round(coverage, 1),
            "coverage_frac": round(coverage / duration, 3) if duration else 0.0,
            "mean_present_score": round(
                sum(self.scores) / len(self.scores), 3) if self.scores else 0.0,
        }
        return ranges, stats

    def _fail(self, e: Exception) -> None:
        logger.warning("[person-gate] failed open (VLM watches everything):"
                       " %s: %s", type(e).__name__, e)
        self.error = f"{type(e).__name__}: {e}"
        self._pending.clear()
