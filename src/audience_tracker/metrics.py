"""Runtime performance metrics. Pure stdlib (+ optional GPU probe).

Tracks the values required by ``GET /metrics``: fps, latency_ms,
gpu_utilization, active_people, active_tracks, reid_inference_time_ms, plus a
best-effort id_switches counter.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


def _gpu_utilization() -> Optional[float]:
    """Return GPU utilization percent, or None if unavailable."""
    try:  # NVML is the most accurate source when present.
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        return float(util)
    except Exception:
        return None


class _EMA:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * sample
        return self.value


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fps = _EMA(0.1)
        self._latency = _EMA(0.2)
        self._reid_ms = _EMA(0.2)
        self._last_frame_ts: Optional[float] = None
        self._active_people = 0
        self._active_tracks = 0
        self._id_switches = 0
        self._frames = 0
        self._gpu_every = 30  # probe NVML every N frames (it's not free)
        self._gpu_util: Optional[float] = None

    def record_frame(
        self,
        latency_ms: float,
        active_people: int,
        active_tracks: int,
        reid_ms: float = 0.0,
        id_switches: int = 0,
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._last_frame_ts is not None:
                dt = now - self._last_frame_ts
                if dt > 0:
                    self._fps.update(1.0 / dt)
            self._last_frame_ts = now
            self._latency.update(latency_ms)
            if reid_ms > 0:
                self._reid_ms.update(reid_ms)
            self._active_people = active_people
            self._active_tracks = active_tracks
            self._id_switches = id_switches
            self._frames += 1
            if self._frames % self._gpu_every == 1:
                self._gpu_util = _gpu_utilization()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "fps": round(self._fps.value, 2) if self._fps.value else 0.0,
                "latency_ms": round(self._latency.value, 1) if self._latency.value else 0.0,
                "gpu_utilization": self._gpu_util,
                "active_people": self._active_people,
                "active_tracks": self._active_tracks,
                "reid_inference_time_ms": round(self._reid_ms.value, 2)
                if self._reid_ms.value
                else 0.0,
                "id_switches": self._id_switches,
                "frames_processed": self._frames,
            }
