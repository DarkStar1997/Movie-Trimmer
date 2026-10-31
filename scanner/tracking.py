"""ByteTrack-style person tracking over NudeNet detections (numpy-only).

Why hand-rolled instead of `boxmot`: boxmot imports torch, and this
project's host venv is deliberately torch-free (CPU-only unit tests; the
container is the only torch runtime). We need single-class tracking of
person clusters at ~4fps, which is ByteTrack's functional core:

  - constant-velocity motion model (EMA velocity, no covariance bookkeeping)
  - two-band IoU association: high-confidence detections first, then
    low-confidence detections against the remaining tracks
  - lifecycle: tentative (min_hits to matter) -> matched -> coasting
    (misses <= max_lost_samples, still matchable) -> retired

Person clustering: NudeNet emits per-body-part boxes (face above breast
above belly above feet...), so one person's parts are vertically stacked
but rarely overlapping. Two boxes join a cluster when their
vertically-expanded rectangles intersect — vertical expansion keeps
horizontally-separated people apart while stacking one person's parts.

Explicit-hold bridging: a track that carried an explicit (trigger) label
within `explicit_hold_s` and is still alive (matched this frame, or
coasting within max_lost_samples) keeps the scene flagged through
sub-threshold flicker — the deterministic fix for continuation dips
(0.2-0.44 scoring frames between real hits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Detection


# --- per-frame person clustering ---------------------------------------------

@dataclass
class PersonCluster:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    members: list[Detection]

    @property
    def score(self) -> float:
        return max((d.score for d in self.members), default=0.0)

    @property
    def labels(self) -> set[str]:
        return {d.label for d in self.members}


def cluster_persons(detections: list[Detection],
                    v_expand_frac: float = 0.45) -> list[PersonCluster]:
    """Group vertically-stacked part boxes into person clusters."""
    n = len(detections)
    if n == 0:
        return []
    boxes = np.asarray(
        [(d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3])
         for d in detections], dtype=float)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    exp_h = (boxes[:, 3] - boxes[:, 1]) * v_expand_frac
    for i in range(n):
        for j in range(i + 1, n):
            if (boxes[i][0] < boxes[j][2] and boxes[j][0] < boxes[i][2]
                    and boxes[i][1] - exp_h[i] < boxes[j][3] + exp_h[j]
                    and boxes[j][1] - exp_h[j] < boxes[i][3] + exp_h[i]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = []
    for idxs in groups.values():
        x1 = min(boxes[i][0] for i in idxs)
        y1 = min(boxes[i][1] for i in idxs)
        x2 = max(boxes[i][2] for i in idxs)
        y2 = max(boxes[i][3] for i in idxs)
        clusters.append(PersonCluster(
            box=(x1, y1, x2, y2), members=[detections[i] for i in idxs]))
    return clusters


# --- tracker -------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of [N,4] / [M,4] xyxy arrays -> [N,M]."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


class _Track:
    __slots__ = ("tid", "box", "vel", "hits", "misses",
                 "last_explicit_t", "explicit_score", "labels_seen")

    def __init__(self, tid: int, box: np.ndarray):
        self.tid = tid
        self.box = box
        self.vel = np.zeros(4)
        self.hits = 1
        self.misses = 0
        self.last_explicit_t: float | None = None
        self.explicit_score = 0.0
        self.labels_seen: set[str] = set()

    def holds_explicit(self, t: float, hold_s: float, min_hits: int) -> bool:
        return (self.last_explicit_t is not None
                and self.hits >= min_hits
                and t - self.last_explicit_t <= hold_s)


EXPLICIT_COUNTERPARTS = {
    "FEMALE_GENITALIA_EXPOSED": "FEMALE_GENITALIA_COVERED",
    "FEMALE_BREAST_EXPOSED": "FEMALE_BREAST_COVERED",
    "MALE_GENITALIA_EXPOSED": None,      # no covered label in the taxonomy
    "ANUS_EXPOSED": "ANUS_COVERED",
    "BUTTOCKS_EXPOSED": "BUTTOCKS_COVERED",
}


class TrackBridge:
    """Track person clusters; flag explicit-hold bridging per frame.

    Hold semantics (refresh / clear):
      - an explicit-label detection >= refresh_floor (below the policy
        trigger, but at/above the detector's collect threshold) refreshes
        `last_explicit_t` — sub-threshold flicker keeps the hold alive
      - a *_COVERED counterpart with NO explicit hit >= refresh_floor
        clears the hold — the person demonstrably re-clothed, so stop
        extending the scene past them
    """

    def __init__(self, *, dt: float = 0.25, iou_thr: float = 0.2,
                 high_score: float = 0.5, max_lost_samples: int = 3,
                 min_hits: int = 2, explicit_hold_s: float = 8.0,
                 refresh_floor: float = 0.15):
        self.dt = dt
        self.iou_thr = iou_thr
        self.high_score = high_score
        self.max_lost_samples = max_lost_samples
        self.min_hits = min_hits
        self.explicit_hold_s = explicit_hold_s
        self.refresh_floor = refresh_floor
        self._tracks: list[_Track] = []
        self._next_id = 1

    # -- explicit-hold state on a track ------------------------------------

    def _explicit_state(self, members: list[Detection],
                        explicit_labels: set[str]) -> tuple[bool, float]:
        """(any explicit signal >= refresh_floor, best such score)."""
        best = 0.0
        hit = False
        for d in members:
            if d.label in explicit_labels and d.score >= self.refresh_floor:
                hit = True
                best = max(best, d.score)
        return hit, best

    def _reclothed(self, members: list[Detection],
                   explicit_labels: set[str]) -> bool:
        """True when covered counterparts fire and no weak explicit does."""
        hit, _ = self._explicit_state(members, explicit_labels)
        if hit:
            return False
        labels = {d.label for d in members}
        for expl in explicit_labels:
            cov = EXPLICIT_COUNTERPARTS.get(expl)
            if cov is not None and cov in labels:
                return True
        return False

    def _absorb(self, trk: _Track, cluster: PersonCluster, t: float,
                explicit_labels: set[str]) -> None:
        """Fold a matched/spawned cluster's explicit evidence into a track."""
        hit, best = self._explicit_state(cluster.members, explicit_labels)
        if hit:
            if trk.last_explicit_t is None or t >= trk.last_explicit_t:
                trk.last_explicit_t = t
            trk.explicit_score = max(trk.explicit_score, best)
        elif self._reclothed(cluster.members, explicit_labels):
            trk.last_explicit_t = None
            trk.explicit_score = 0.0

    def _spawn(self, box: np.ndarray, cluster: PersonCluster, t: float,
               explicit_labels: set[str]) -> _Track:
        trk = _Track(self._next_id, box.copy())
        self._next_id += 1
        trk.labels_seen = set(cluster.labels)
        self._absorb(trk, cluster, t, explicit_labels)
        self._tracks.append(trk)
        return trk

    def step(self, t: float, clusters: list[PersonCluster],
             explicit_labels: set[str]) -> dict:
        """Advance one frame. Returns:
        bridged          — explicit-hold keeps this frame flagged
        score            — score backing the bridge (max held explicit score)
        tracks           — tracks matched this frame
        explicit_tracks  — matched tracks currently holding explicit state
        """
        if clusters:
            boxes = np.asarray([c.box for c in clusters], dtype=float)
            scores = np.asarray([c.score for c in clusters], dtype=float)
        else:
            boxes = np.zeros((0, 4), dtype=float)
            scores = np.zeros(0, dtype=float)

        # --- two-band greedy IoU association ---
        matched_track_ids: set[int] = set()
        matched_det: dict[int, int] = {}          # det_idx -> track id
        unmatched_dets = set(range(len(boxes)))
        for band_high in (True, False):
            if not unmatched_dets:
                break
            pool = [trk for trk in self._tracks
                    if id(trk) not in matched_track_ids]
            if not pool:
                break
            cand = [i for i in unmatched_dets
                    if (scores[i] >= self.high_score) == band_high]
            if not cand:
                continue
            pred = np.stack([trk.box + trk.vel * self.dt for trk in pool])
            iou = _iou(boxes[cand], pred)
            order = sorted(((iou[ci, ti], ci, ti)
                            for ci in range(len(cand))
                            for ti in range(len(pool))),
                           key=lambda x: (-x[0], x[2], x[1]))
            for val, ci, ti in order:
                if val < self.iou_thr:
                    break
                if cand[ci] in matched_det or id(pool[ti]) in matched_track_ids:
                    continue
                matched_det[cand[ci]] = id(pool[ti])
                matched_track_ids.add(id(pool[ti]))
            unmatched_dets -= set(matched_det)

        # --- update matched tracks ---
        for di, tid_ in matched_det.items():
            trk = next(t_ for t_ in self._tracks if id(t_) == tid_)
            new_box = boxes[di]
            if trk.hits > 1:
                inst_vel = (new_box - trk.box) / self.dt
                trk.vel = 0.6 * trk.vel + 0.4 * inst_vel
            trk.box = new_box.copy()
            trk.hits += 1
            trk.misses = 0
            trk.labels_seen |= clusters[di].labels
            self._absorb(trk, clusters[di], t, explicit_labels)

        # --- age unmatched tracks / retire stale ---
        alive: list[_Track] = []
        for trk in self._tracks:
            if id(trk) not in matched_track_ids:
                trk.misses += 1
                if trk.misses > self.max_lost_samples:
                    continue
            alive.append(trk)
        self._tracks = alive

        # --- spawn new tracks ---
        for di in unmatched_dets:
            self._spawn(boxes[di], clusters[di], t, explicit_labels)

        # --- explicit-hold bridging over all alive tracks ---
        bridged, score = False, 0.0
        live = explicit_live = 0
        for trk in self._tracks:
            if trk.misses == 0:
                live += 1
                if trk.holds_explicit(t, self.explicit_hold_s, self.min_hits):
                    explicit_live += 1
            if trk.misses <= self.max_lost_samples and \
                    trk.holds_explicit(t, self.explicit_hold_s, self.min_hits):
                bridged = True
                score = max(score, trk.explicit_score)

        return {"bridged": bridged, "score": float(score),
                "tracks": live, "explicit_tracks": explicit_live}
