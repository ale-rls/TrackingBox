"""Audience Tracking System — real-time anonymous audience tracking.

Anonymous-by-design: assigns persistent Global IDs (GIDs) to audience members
via YOLO11m detection, ByteTrack tracking and OSNet ReID. No facial
recognition, no name lookup.
"""

from __future__ import annotations

from .config import Config
from .identity import IdentityManager
from .models import AudienceState, Detection, Identity, Track

__version__ = "1.0.0"

__all__ = [
    "Config",
    "IdentityManager",
    "AudienceState",
    "Detection",
    "Identity",
    "Track",
    "__version__",
]
