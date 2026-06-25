"""Domain models shared across the pipeline.

Deliberately stdlib-only. Embeddings are plain ``list[float]`` so the Identity
Manager and its tests have no numpy dependency; the real ReID extractor converts
its numpy output with ``.tolist()`` at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

# (x1, y1, x2, y2) in pixel coordinates.
BBox = tuple[float, float, float, float]
Embedding = list[float]


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iso(ts: float) -> str:
    """Epoch seconds -> ISO 8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass(slots=True)
class Detection:
    """Raw detector output for one person."""
    bbox: BBox
    confidence: float


@dataclass(slots=True)
class Track:
    """A ByteTrack association. ``track_id`` is INTERNAL ONLY and must never be
    exposed through the public API (Tracking spec / Identity Rule 5)."""
    track_id: int
    bbox: BBox
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)


@dataclass(slots=True)
class Identity:
    """An audience member as tracked by the Identity Manager (source of truth)."""
    gid: int
    first_seen: float
    last_seen: float
    embedding_avg: Optional[Embedding] = None
    visible: bool = True
    active_track_id: Optional[int] = None
    last_bbox: Optional[BBox] = None
    last_confidence: float = 0.0
    # Accumulated wall-clock time the identity has actually been visible.
    duration_seen_seconds: float = 0.0
    # Number of times this GID has been recovered after going lost (ReID hits).
    recoveries: int = 0
    update_count: int = 0

    @property
    def center(self) -> Optional[tuple[float, float]]:
        return bbox_center(self.last_bbox) if self.last_bbox else None

    def to_state(self) -> "AudienceState":
        return AudienceState(
            gid=self.gid,
            visible=self.visible,
            bbox=self.last_bbox,
            center=self.center,
            confidence=self.last_confidence,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            duration_seen_seconds=round(self.duration_seen_seconds, 3),
        )


@dataclass(slots=True)
class AudienceState:
    """Serializable, GID-only view of an audience member (public model)."""
    gid: int
    visible: bool
    bbox: Optional[BBox]
    center: Optional[tuple[float, float]]
    confidence: float
    first_seen: float
    last_seen: float
    duration_seen_seconds: float

    # ---- serialization variants used by the API ---- #
    def summary(self) -> dict:
        """Compact form for GET /audience and WebSocket broadcasts."""
        return {
            "gid": self.gid,
            "visible": self.visible,
            "center": _round_point(self.center),
            "bbox": _round_bbox(self.bbox),
        }

    def detail(self) -> dict:
        """Full form for GET /audience/{gid} and GET /snapshot."""
        return {
            "gid": self.gid,
            "visible": self.visible,
            "bbox": _round_bbox(self.bbox),
            "center": _round_point(self.center),
            "confidence": round(self.confidence, 3),
            "first_seen": iso(self.first_seen),
            "last_seen": iso(self.last_seen),
            "duration_seen_seconds": round(self.duration_seen_seconds, 3),
        }


def _round_point(p: Optional[Sequence[float]]) -> Optional[list[int]]:
    return [int(round(p[0])), int(round(p[1]))] if p else None


def _round_bbox(b: Optional[Sequence[float]]) -> Optional[list[int]]:
    return [int(round(v)) for v in b] if b else None
