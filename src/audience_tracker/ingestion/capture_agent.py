"""The local Capture Agent (Video Ingestion spec).

Runs on the venue computer. Connects to the physical camera, reads frames
continuously, timestamps and JPEG-compresses them, and streams them over a
single long-lived WebSocket to the Modal Ingestion Service — reconnecting
automatically with exponential backoff. Exposes ``GET /status`` with live
metrics. Performs **no** computer-vision inference.

Design notes that satisfy the spec:

* The capture loop never blocks on the network. It hands the *latest* encoded
  packet to the sender through a single-slot buffer; if the sender is behind,
  the older unsent packet is dropped (``frames_dropped``) so we always send the
  freshest frame. Real-time beats completeness.
* ``frame_id`` is an agent-owned monotonic counter, so it stays strictly
  increasing across both camera replug and WebSocket reconnects.
* Camera unplug -> ``read()`` fails -> the agent reopens the device with backoff,
  no restart required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..config import Config
from .frame_source import OpenCVFrameSource
from .packet import encode_packet

log = logging.getLogger("audience_tracker.capture_agent")

# Connection states (spec "Connection Management").
CONNECTED = "CONNECTED"
CONNECTING = "CONNECTING"
DISCONNECTED = "DISCONNECTED"
ERROR = "ERROR"


class _RollingFPS:
    """Frames-per-second over a trailing 1-second window."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window = window_s
        self._ts: deque[float] = deque()

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._ts.append(now)
        self._prune(now)

    def value(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        self._prune(now)
        return round(len(self._ts) / self._window, 1)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()


class CaptureAgent:
    def __init__(self, cfg: Config, source: Optional[str | int] = None) -> None:
        self.cfg = cfg
        self.acfg = cfg.capture_agent
        # Acquire from an explicit source, else the configured camera device index.
        self._source = source if source is not None else cfg.camera.device_index

        self._running = False
        self._started_at = time.monotonic()
        self._frame_id = 0

        # Sender single-slot buffer (loop-thread only) + readiness signalling.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._latest: Optional[bytes] = None
        self._has_frame: Optional[asyncio.Event] = None

        # Metrics (guarded; read by the status HTTP thread).
        self._lock = threading.Lock()
        self._state = DISCONNECTED
        self._fps_capture = _RollingFPS()
        self._fps_sent = _RollingFPS()
        self._latency_ms = 0.0
        self._frames_dropped = 0

        self._capture_thread: Optional[threading.Thread] = None
        self._status_server: Optional[ThreadingHTTPServer] = None

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        with self._lock:
            return {
                "connected": self._state == CONNECTED,
                "state": self._state,
                "fps_capture": self._fps_capture.value(),
                "fps_sent": self._fps_sent.value(),
                "latency_ms": round(self._latency_ms, 1),
                "frames_dropped": self._frames_dropped,
                "uptime_seconds": round(time.monotonic() - self._started_at, 1),
            }

    def _set_state(self, state: str) -> None:
        with self._lock:
            if state != self._state:
                log.info("Capture Agent: %s -> %s", self._state, state)
            self._state = state

    # ------------------------------------------------------------------ #
    # Capture (its own thread; blocking OpenCV reads)
    # ------------------------------------------------------------------ #
    def _open_camera(self) -> OpenCVFrameSource:
        return OpenCVFrameSource(self._source, camera=self.cfg.camera)

    def _encode(self, image) -> Optional[bytes]:
        import cv2

        a = self.acfg
        if a.resize_width and a.resize_height:
            image = cv2.resize(image, (a.resize_width, a.resize_height))
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality])
        return buf.tobytes() if ok else None

    def _capture_loop(self) -> None:
        backoff = self.acfg.reconnect_initial_s
        cam: Optional[OpenCVFrameSource] = None
        while self._running:
            if cam is None:
                try:
                    cam = self._open_camera()
                    backoff = self.acfg.reconnect_initial_s
                except Exception as exc:
                    log.warning("Camera open failed (%s); retrying in %.1fs", exc, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.acfg.reconnect_max_s)
                    continue
            frame = cam.next_frame()
            if frame is None:  # device read failed (e.g. unplugged) -> reopen
                log.warning("Camera read failed; reopening device in %.1fs", backoff)
                cam.release()
                cam = None
                time.sleep(backoff)  # avoid a tight reopen spin while unplugged
                backoff = min(backoff * 2, self.acfg.reconnect_max_s)
                continue
            jpeg = self._encode(frame.image)
            if jpeg is None:
                continue
            h, w = frame.image.shape[:2]
            self._frame_id += 1
            packet = encode_packet(self._frame_id, frame.timestamp, jpeg, w, h)
            with self._lock:
                self._fps_capture.tick()
            self._offer(packet)
        if cam is not None:
            cam.release()

    def _offer(self, packet: bytes) -> None:
        """Hand the newest packet to the sender, dropping any older unsent one."""
        loop, ev = self._loop, self._has_frame
        if loop is None or ev is None:
            return

        def deliver() -> None:
            if self._latest is not None:  # sender hasn't taken the previous one
                with self._lock:
                    self._frames_dropped += 1
            self._latest = packet
            ev.set()

        loop.call_soon_threadsafe(deliver)

    # ------------------------------------------------------------------ #
    # Sender (asyncio: one long-lived WSS connection + reconnect/backoff)
    # ------------------------------------------------------------------ #
    async def _sender_main(self) -> None:
        import websockets

        self._loop = asyncio.get_running_loop()
        self._has_frame = asyncio.Event()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True
        )
        self._capture_thread.start()

        headers = [("Authorization", f"Bearer {self.acfg.token}")] if self.acfg.token else []
        backoff = self.acfg.reconnect_initial_s
        while self._running:
            self._set_state(CONNECTING)
            try:
                async with websockets.connect(
                    self.acfg.server_url, additional_headers=headers, max_size=None
                ) as ws:
                    self._set_state(CONNECTED)
                    backoff = self.acfg.reconnect_initial_s  # reset on success
                    await self._send_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set_state(ERROR if "403" in str(exc) or "401" in str(exc) else DISCONNECTED)
                log.warning("Ingest connection lost (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.acfg.reconnect_max_s)
        self._set_state(DISCONNECTED)

    async def _send_loop(self, ws) -> None:
        assert self._has_frame is not None
        while self._running:
            await self._has_frame.wait()
            self._has_frame.clear()
            packet = self._latest
            self._latest = None
            if packet is None:
                continue
            t0 = time.perf_counter()
            await ws.send(packet)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._latency_ms = dt_ms
                self._fps_sent.tick()

    # ------------------------------------------------------------------ #
    # Status HTTP server (its own thread)
    # ------------------------------------------------------------------ #
    def _start_status_server(self) -> None:
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") not in ("/status", ""):
                    self.send_error(404)
                    return
                body = json.dumps(agent.status()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # silence default logging
                pass

        self._status_server = ThreadingHTTPServer(
            (self.acfg.status_host, self.acfg.status_port), Handler
        )
        threading.Thread(
            target=self._status_server.serve_forever, name="status", daemon=True
        ).start()
        log.info("Status endpoint: http://%s:%d/status", self.acfg.status_host, self.acfg.status_port)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._running = True
        self._started_at = time.monotonic()
        self._start_status_server()
        log.info("Streaming %s -> %s", self._source, self.acfg.server_url)
        try:
            asyncio.run(self._sender_main())
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._status_server is not None:
            self._status_server.shutdown()


def run_agent(cfg: Config, source: Optional[str | int] = None) -> None:
    CaptureAgent(cfg, source=source).run()
