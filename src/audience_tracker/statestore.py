"""Shared audience state + change broadcasting.

The tracking pipeline (a worker thread) publishes the latest audience snapshot;
the REST/WebSocket API (asyncio) reads it and streams changes. This is the
"Shared State" component from the spec — in-memory for V1, with a Redis-backed
implementation left as a documented extension point.

Thread-safety: ``publish`` is called from the pipeline thread; reads and
WebSocket subscription happen on the asyncio loop thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from .models import AudienceState


class InMemoryStateStore:
    def __init__(self, ws_queue_size: int = 256) -> None:
        self._lock = threading.Lock()
        self._by_gid: dict[int, AudienceState] = {}
        self._stats: dict = {"active_people": 0, "total_people_seen": 0}
        self._metrics: dict = {}
        self._updated_at: float = time.time()
        self._frame_jpeg: Optional[bytes] = None

        # WebSocket fan-out (asyncio side).
        self._ws_queue_size = ws_queue_size
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._prev_summary: dict[int, tuple] = {}

    # ------------------------------------------------------------------ #
    # Write side (pipeline thread)
    # ------------------------------------------------------------------ #
    def publish(
        self,
        snapshot: list[AudienceState],
        stats: dict,
        metrics: Optional[dict] = None,
    ) -> None:
        events: list[dict] = []
        with self._lock:
            self._by_gid = {s.gid: s for s in snapshot}
            self._stats = stats
            if metrics is not None:
                self._metrics = metrics
            self._updated_at = time.time()

            # Diff against the previous publish to emit only changes.
            current: dict[int, tuple] = {}
            for s in snapshot:
                summ = s.summary()
                key = (summ["visible"], tuple(summ["center"] or ()), tuple(summ["bbox"] or ()))
                current[s.gid] = key
                if self._prev_summary.get(s.gid) != key:
                    events.append(summ)
            # Identities that dropped out of the snapshot entirely -> gone/hidden.
            for gid in self._prev_summary.keys() - current.keys():
                events.append({"gid": gid, "visible": False, "center": None, "bbox": None})
            self._prev_summary = current

        for ev in events:
            self._broadcast(ev)

    # ------------------------------------------------------------------ #
    # Read side (API thread) — shapes match the REST spec
    # ------------------------------------------------------------------ #
    def get_active(self) -> list[dict]:
        with self._lock:
            return [s.summary() for s in self._by_gid.values() if s.visible]

    def get_member(self, gid: int) -> Optional[dict]:
        with self._lock:
            s = self._by_gid.get(gid)
            return s.detail() if s else None

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_metrics(self) -> dict:
        with self._lock:
            return dict(self._metrics)

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame_jpeg = jpeg

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpeg

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "timestamp": _iso(self._updated_at),
                "active_people": sum(1 for s in self._by_gid.values() if s.visible),
                "people": [s.detail() for s in self._by_gid.values()],
            }

    # ------------------------------------------------------------------ #
    # WebSocket fan-out
    # ------------------------------------------------------------------ #
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once by the API on startup so cross-thread publishes can enqueue."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._ws_queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, message: dict) -> None:
        loop = self._loop
        if loop is None or not self._subscribers:
            return
        for q in list(self._subscribers):
            loop.call_soon_threadsafe(self._safe_put, q, message)

    @staticmethod
    def _safe_put(q: asyncio.Queue, message: dict) -> None:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            # Slow consumer: drop the oldest message to stay live.
            try:
                q.get_nowait()
                q.put_nowait(message)
            except Exception:
                pass


class RedisStateStore:  # pragma: no cover - future work
    """Placeholder for the documented Redis-backed shared state.

    Would mirror :class:`InMemoryStateStore` using Redis hashes for audience
    state and Pub/Sub for change broadcasting, enabling the API service and GPU
    worker to run as separate Modal containers.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "RedisStateStore is a future extension; use InMemoryStateStore for V1."
        )


def _iso(ts: float) -> str:
    from .models import iso

    return iso(ts)
