"""Temporal smoothing configuration.

The arbiter's aggregation (scanner/arbiter.aggregate) consumes these
parameters when clustering explicit point hits into segments: hits are
chained, bridged by `bridge_gap_s`, dropped below `drop_below_s` unless
confidence rescues them, and padded by `pad_s` (see _chain_gap and
_cluster_hits there). The old detector-era smoothing chain
(hysteresis over per-sample flags) was removed with NudeNet.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class SmoothingConfig:
    open_min_consecutive: int = config.DEFAULT_SMOOTHING["open_min_consecutive"]
    close_min_consecutive: int = config.DEFAULT_SMOOTHING["close_min_consecutive"]
    bridge_gap_s: float = config.DEFAULT_SMOOTHING["bridge_gap_s"]
    drop_below_s: float = config.DEFAULT_SMOOTHING["drop_below_s"]
    drop_rescue_score: float = config.DEFAULT_SMOOTHING["drop_rescue_score"]
    pad_s: float = config.DEFAULT_SMOOTHING["pad_s"]

    @classmethod
    def from_overrides(cls, overrides: dict | None) -> "SmoothingConfig":
        if not overrides:
            return cls()
        known = set(config.DEFAULT_SMOOTHING)
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown smoothing keys: {sorted(unknown)}")
        return cls(**{**config.DEFAULT_SMOOTHING, **overrides})
