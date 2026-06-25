"""Structural interfaces for the swappable pipeline components.

Using ``typing.Protocol`` keeps the real (GPU) and mock (CPU) implementations
interchangeable without a shared base class or heavy imports.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .models import BBox, Detection, Embedding, Track


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]:
        """Return person detections for a single BGR frame."""
        ...


@runtime_checkable
class Tracker(Protocol):
    def update(self, detections: list[Detection], frame: Any | None = None) -> list[Track]:
        """Associate detections across frames, returning tracks with track_ids."""
        ...


@runtime_checkable
class ReIDExtractor(Protocol):
    def extract(self, frame: Any, bboxes: Sequence[BBox]) -> list[Embedding]:
        """Return one appearance embedding per bbox (same order)."""
        ...
