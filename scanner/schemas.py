"""Pydantic models for the scan API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScanResultSegment(BaseModel):
    start: float = Field(..., description="segment start, seconds")
    end: float = Field(..., description="segment end, seconds")
    mean_score: float = Field(..., description="mean trigger score in segment")
    labels: Optional[dict[str, int]] = Field(
        None, description="trigger-label -> flagged-sample count in segment")


class ScanResult(BaseModel):
    scan_id: str
    status: str
    video: str
    duration: Optional[float] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    sample_fps: Optional[float] = None
    frames_scanned: Optional[int] = None
    segments: list[ScanResultSegment] = Field(
        default_factory=list,
        description="auto-cut segments (NSFW), seconds — feed to trimmer /submit")
    review_segments: list[ScanResultSegment] = Field(
        default_factory=list,
        description="borderline segments for human review, not auto-cut")
    timeline: list[dict] = Field(default_factory=list,
                                 description="per-second flagged-sample bins")
    label_stats: dict[str, int] = Field(default_factory=dict)
    policy: Optional[dict] = None
    smoothing: Optional[dict] = None
    wall_time_s: Optional[float] = None
    error: Optional[str] = None
