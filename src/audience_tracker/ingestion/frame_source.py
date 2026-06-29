"""Concrete :class:`~audience_tracker.base.FrameSource` implementations.

* :class:`OpenCVFrameSource` — local USB camera / capture card / file / RTSP via
  OpenCV (used by the Capture Agent and for local runs).
* :class:`SimulatorFrameSource` — the synthetic crowd (no camera/GPU).
* :class:`QueueFrameSource` — the Modal ingestion queue: a bounded, drop-oldest
  buffer fed by the WebSocket ingestion handler and drained by the pipeline.

All keep frames strictly ordered by ``frame_id`` and expose ``exhausted`` so the
pipeline can tell "nothing yet" (live, keep waiting) from "finished" (file end).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from ..config import CameraConfig
from ..models import Frame


class _FiniteFrameSource:
    """Shared ``read()`` compatibility shim for pull-based finite sources.

    Lets existing ``cv2.VideoCapture``-style callers (CLI, benchmark, tests) keep
    using ``ok, frame = src.read()`` while the pipeline uses ``next_frame()``.
    """

    _exhausted: bool

    def next_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:  # pragma: no cover
        raise NotImplementedError

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        frame = self.next_frame()
        if frame is None:
            return False, None
        return True, frame.image

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def release(self) -> None:  # pragma: no cover - overridden where needed
        self._exhausted = True


class OpenCVFrameSource(_FiniteFrameSource):
    """Reads frames from any OpenCV-openable source: device index, file, RTSP/HTTP."""

    def __init__(
        self,
        source: str | int,
        max_frames: Optional[int] = None,
        camera: Optional[CameraConfig] = None,
    ) -> None:
        import cv2  # local import: only the venue/GPU box has OpenCV

        self._source = source
        self._max_frames = max_frames
        self._exhausted = False
        self._next_id = 0
        self._cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if camera is not None and str(source).isdigit():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
            self._cap.set(cv2.CAP_PROP_FPS, camera.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

    def next_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._exhausted:
            return None
        if self._max_frames is not None and self._next_id >= self._max_frames:
            self._exhausted = True
            return None
        ok, image = self._cap.read()
        if not ok or image is None:
            self._exhausted = True
            return None
        h, w = image.shape[:2]
        frame = Frame(image=image, timestamp=time.time(), frame_id=self._next_id, width=w, height=h)
        self._next_id += 1
        return frame

    def release(self) -> None:
        self._exhausted = True
        try:
            self._cap.release()
        except Exception:  # pragma: no cover
            pass


class SimulatorFrameSource(_FiniteFrameSource):
    """Wraps the synthetic :class:`~audience_tracker.simulator.Simulator`."""

    def __init__(self, simulator, max_frames: Optional[int] = None) -> None:
        self._sim = simulator
        self._max_frames = max_frames
        self._exhausted = False
        self._next_id = 0

    def next_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        if self._exhausted:
            return None
        if self._max_frames is not None and self._next_id >= self._max_frames:
            self._exhausted = True
            return None
        self._sim.step()
        image = self._sim.render()
        h, w = image.shape[:2]
        frame = Frame(image=image, timestamp=time.time(), frame_id=self._next_id, width=w, height=h)
        self._next_id += 1
        return frame

    def release(self) -> None:
        self._exhausted = True


class QueueFrameSource:
    """Bounded, drop-oldest frame queue — the Modal Ingestion Service buffer.

    The WebSocket ingestion handler calls :meth:`push` for each decoded frame; the
    pipeline thread calls :meth:`next_frame` to drain it. Per spec:

    * depth is bounded (default 2-3); when full the **oldest** frame is dropped so
      latency never grows unbounded;
    * frames are kept strictly ordered — a frame whose ``frame_id`` is not greater
      than the last accepted one is rejected (late/duplicate);
    * it is a *live* source: an empty queue means "nothing yet", not "finished",
      so the pipeline keeps running while the agent is briefly disconnected.
    """

    def __init__(self, maxsize: int = 3) -> None:
        self._maxsize = max(1, maxsize)
        self._buf: deque[Frame] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._last_id: int = -1
        # Metrics (spec /status style on the receiving side).
        self.received = 0
        self.dropped_overflow = 0
        self.dropped_out_of_order = 0

    def begin_session(self) -> None:
        """Reset ordering for a new Capture Agent connection.

        The agent's ``frame_id`` restarts from scratch whenever the agent process
        restarts, but this (long-lived) queue would otherwise keep the previous
        session's high ``last_frame_id`` and reject every new frame as
        out-of-order. Single camera => single agent, so a fresh connection means a
        fresh frame-id sequence; drop any stale buffered frames too.
        """
        with self._cond:
            self._buf.clear()
            self._last_id = -1

    # -- write side (WebSocket ingestion thread) -- #
    def push(self, frame: Frame) -> bool:
        """Enqueue a frame. Returns False if rejected as late/out-of-order."""
        with self._cond:
            if self._closed:
                return False
            if frame.frame_id <= self._last_id:
                self.dropped_out_of_order += 1
                return False
            self._last_id = frame.frame_id
            self.received += 1
            if len(self._buf) >= self._maxsize:
                self._buf.popleft()  # drop oldest — prioritise freshness
                self.dropped_overflow += 1
            self._buf.append(frame)
            self._cond.notify()
            return True

    # -- read side (pipeline thread) -- #
    def next_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        with self._cond:
            if not self._buf and not self._closed:
                self._cond.wait(timeout)
            if self._buf:
                return self._buf.popleft()
            return None

    def release(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def exhausted(self) -> bool:
        with self._cond:
            return self._closed and not self._buf

    def status(self) -> dict:
        with self._cond:
            return {
                "depth": len(self._buf),
                "capacity": self._maxsize,
                "received": self.received,
                "dropped_overflow": self.dropped_overflow,
                "dropped_out_of_order": self.dropped_out_of_order,
                "last_frame_id": self._last_id,
                "closed": self._closed,
            }
