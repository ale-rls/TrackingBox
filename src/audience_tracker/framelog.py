"""Replayable per-frame logging.

Writes one JSON object per processed frame (JSONL) capturing the data required
by the spec — timestamp, gid, bbox, center, visible — for debugging,
benchmarking and evaluation. ``replay()`` reads it back. Pure stdlib.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Iterator

from .models import AudienceState, iso


class FrameLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._fh = None
        if enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def log(self, frame_index: int, timestamp: float, states: list[AudienceState]) -> None:
        if not self.enabled or self._fh is None:
            return
        record = {
            "frame": frame_index,
            "timestamp": iso(timestamp),
            "people": [
                {
                    "gid": s.gid,
                    "bbox": [int(round(v)) for v in s.bbox] if s.bbox else None,
                    "center": [int(round(c)) for c in s.center] if s.center else None,
                    "visible": s.visible,
                }
                for s in states
            ],
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "FrameLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def replay(path: str) -> Iterator[dict]:
    """Yield logged frame records in order."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
