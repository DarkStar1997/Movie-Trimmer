"""Label policy: per-frame verdict from raw detections.

Two tiers:
  - triggers: any qualifying detection flags the frame NSFW (auto-cut)
  - review:   flagged separately for human review, never auto-cut

A detection qualifies when score >= its per-label threshold AND its box area
is at least `min_box_area_frac` of the frame (kills background posters/TVs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import config
from .detector import Detection


@dataclass
class Policy:
    triggers: dict[str, float] = field(
        default_factory=lambda: dict(config.DEFAULT_TRIGGER_LABELS))
    review: dict[str, float] = field(
        default_factory=lambda: dict(config.DEFAULT_REVIEW_LABELS))
    min_box_area_frac: float = config.DEFAULT_MIN_BOX_AREA_FRAC

    def __post_init__(self) -> None:
        for label in list(self.triggers) + list(self.review):
            if label not in config.NUDENET_LABELS:
                raise ValueError(
                    f"unknown label {label!r}; valid: {config.NUDENET_LABELS}")
        for name, thr in (("triggers", self.triggers), ("review", self.review)):
            for label, score in thr.items():
                if not 0.0 < score <= 1.0:
                    raise ValueError(f"{name}[{label}] threshold must be in (0, 1]")
        if not 0.0 <= self.min_box_area_frac < 1.0:
            raise ValueError("min_box_area_frac must be in [0, 1)")

    @classmethod
    def from_overrides(cls, overrides: Optional[dict]) -> "Policy":
        """Merge user overrides onto defaults. Accepted keys:
        triggers, review (label->score dicts, fully replacing the default
        dict when given), min_box_area_frac."""
        if not overrides:
            return cls()
        known = {"triggers", "review", "min_box_area_frac"}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return cls(
            triggers=overrides.get("triggers",
                                   dict(config.DEFAULT_TRIGGER_LABELS)),
            review=overrides.get("review", dict(config.DEFAULT_REVIEW_LABELS)),
            min_box_area_frac=overrides.get(
                "min_box_area_frac", config.DEFAULT_MIN_BOX_AREA_FRAC),
        )

    def evaluate(self, detections: list[Detection],
                 frame_w: int, frame_h: int) -> "FrameVerdict":
        """Evaluate one frame's detections against the policy."""
        triggered: dict[str, float] = {}
        review: dict[str, float] = {}
        for det in detections:
            if det.area_fraction(frame_w, frame_h) < self.min_box_area_frac:
                continue
            t = self.triggers.get(det.label)
            if t is not None and det.score >= t:
                triggered[det.label] = max(triggered.get(det.label, 0.0), det.score)
                continue
            r = self.review.get(det.label)
            if r is not None and det.score >= r:
                review[det.label] = max(review.get(det.label, 0.0), det.score)
        return FrameVerdict(
            triggered=bool(triggered),
            review=bool(review) and not triggered,
            trigger_labels=triggered,
            review_labels=review,
            max_trigger_score=max(triggered.values(), default=0.0),
        )

    def weak_triggers(self, detections: list[Detection],
                      frame_w: int, frame_h: int,
                      min_score: float) -> dict[str, float]:
        """Sub-threshold trigger hits: area-qualified detections whose score
        sits in [min_score, the label's trigger threshold). These never flag
        the frame (smoothing never sees them) but are the detector's "almost"
        signal — valuable exactly where dim/distant bodies live. The arbiter
        routes their cluster peaks to the VLM for a targeted second look."""
        weak: dict[str, float] = {}
        for det in detections:
            t = self.triggers.get(det.label)
            if t is None or not min_score <= det.score < t:
                continue
            if det.area_fraction(frame_w, frame_h) < self.min_box_area_frac:
                continue
            weak[det.label] = max(weak.get(det.label, 0.0), det.score)
        return weak


@dataclass
class FrameVerdict:
    triggered: bool
    review: bool
    trigger_labels: dict[str, float]
    review_labels: dict[str, float]
    max_trigger_score: float
