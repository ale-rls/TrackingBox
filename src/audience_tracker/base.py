"""Structural interfaces for the swappable pipeline components.

Using ``typing.Protocol`` keeps the real (GPU) and mock (CPU) implementations
interchangeable without a shared base class or heavy imports.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from .models import BBox, Detection, Embedding, Frame, Track


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


@runtime_checkable
class FrameSource(Protocol):
    """Abstract source of frames for the Tracking Pipeline (Video Ingestion spec).

    Implementations include a local OpenCV camera/file/RTSP reader, the synthetic
    simulator, and the Modal ingestion queue fed over WebSocket. The pipeline only
    ever calls ``next_frame`` / ``release`` and inspects ``exhausted`` — it never
    learns the underlying transport.
    """

    def next_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Return the next frame, or ``None`` if none arrived within ``timeout``.

        ``None`` is transient (a live source may simply have nothing yet); callers
        decide whether to keep waiting by checking :attr:`exhausted`.
        """
        ...

    def release(self) -> None:
        """Release any underlying resources (camera handle, socket, etc.)."""
        ...

    @property
    def exhausted(self) -> bool:
        """True once the source is permanently finished (file/simulator ended)."""
        ...
