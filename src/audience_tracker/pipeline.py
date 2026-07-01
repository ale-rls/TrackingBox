"""The real-time tracking pipeline.

Per frame:  detect -> track -> (scheduled) ReID -> identity reconcile ->
publish state -> log -> overlay. Can run in the foreground or as a background
worker thread feeding an :class:`InMemoryStateStore`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from .base import Detector, ReIDExtractor, Tracker
from .config import Config
from .framelog import FrameLogger
from .identity import IdentityManager
from .metrics import MetricsCollector
from .models import AudienceState, Embedding, Track
from .overlay import OverlayRenderer
from .projection import FloorProjector, ProjectionError
from .reid import NullReID
from .statestore import InMemoryStateStore

log = logging.getLogger("audience_tracker.pipeline")


class TrackingPipeline:
    def __init__(
        self,
        cfg: Config,
        detector: Detector,
        tracker: Tracker,
        reid: ReIDExtractor,
        store: InMemoryStateStore,
        identity_manager: Optional[IdentityManager] = None,
        metrics: Optional[MetricsCollector] = None,
        frame_logger: Optional[FrameLogger] = None,
        overlay: Optional[OverlayRenderer] = None,
        floor_projector: Optional[FloorProjector] = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.tracker = tracker
        self.reid = reid
        self.store = store
        self.identity = identity_manager or IdentityManager(cfg.identity, cfg.reid)
        self.metrics = metrics or MetricsCollector()
        self.frame_logger = frame_logger or FrameLogger(
            cfg.logging.frame_log_path, cfg.logging.enabled
        )
        self.overlay = overlay or OverlayRenderer(cfg.overlay)
        self.floor_projector = floor_projector or self._build_floor_projector()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reid_enabled = cfg.reid.enabled and not isinstance(reid, NullReID)

    # ------------------------------------------------------------------ #
    # Single frame
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: Any, frame_index: int, now: Optional[float] = None) -> Any:
        now = time.time() if now is None else now
        t0 = time.perf_counter()

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        embeddings = self._embed(frame, tracks, frame_index)
        self.identity.update(tracks, embeddings, now)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        snapshot = self.identity.snapshot(now)
        visible = [s for s in snapshot if s.visible]
        counters = self.identity.counters()
        reid_ms = float(getattr(self.reid, "last_inference_ms", 0.0))

        self._project_floor(snapshot)
        self.metrics.record_frame(
            latency_ms=latency_ms,
            active_people=counters["active_people"],
            active_tracks=counters["active_tracks"],
            reid_ms=reid_ms,
            id_switches=counters["id_switches"],
        )
        self.store.publish(snapshot, self.identity.stats(), self.metrics.snapshot())
        self.frame_logger.log(frame_index, now, snapshot)

        if self.cfg.pipeline.render_overlay:
            frame = self.overlay.render(frame, visible, self.cfg.overlay.debug)
            if self.cfg.pipeline.stream_overlay:
                self._publish_frame(frame)
        return frame

    def _build_floor_projector(self) -> FloorProjector | None:
        try:
            return FloorProjector(self.cfg.lens_calibration, self.cfg.calibration)
        except ProjectionError as exc:
            log.warning("Floor projection disabled: %s", exc)
            return None

    def _project_floor(self, snapshot: list[AudienceState]) -> None:
        if self.floor_projector is None:
            return
        for state in snapshot:
            result = self.floor_projector.project_bbox(state.bbox, gid=state.gid)
            state.floor = result.floor
            state.floor_valid = result.valid

    def _publish_frame(self, frame: Any) -> None:
        try:
            import cv2

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                self.store.set_frame(buf.tobytes())
        except Exception:  # OpenCV missing or encode failed — skip the stream.
            pass

    def _embed(self, frame: Any, tracks: list[Track], frame_index: int) -> dict[int, Embedding]:
        """ReID scheduling: new/unbound tracks every frame, known tracks every N."""
        if not self._reid_enabled or not tracks:
            return {}
        known = self.identity.known_track_ids()
        due = frame_index % max(1, self.cfg.reid.update_every) == 0
        targets = [t for t in tracks if t.track_id not in known or due]
        if not targets:
            return {}
        embeddings = self.reid.extract(frame, [t.bbox for t in targets])
        return {t.track_id: e for t, e in zip(targets, embeddings) if e}

    # ------------------------------------------------------------------ #
    # Run loop
    # ------------------------------------------------------------------ #
    def run(self, source: Any, writer: Any = None, max_frames: Optional[int] = None) -> int:
        """Process frames from a :class:`FrameSource` until it ends or stop().

        The source is consumed via ``next_frame`` so the pipeline is agnostic to
        the transport (camera / file / RTSP / simulator / ingestion queue). A
        ``None`` return is transient — for a live source (the ingestion queue
        while the Capture Agent is disconnected) the loop keeps waiting; it only
        ends when the source reports ``exhausted`` (file/simulator finished)."""
        processed = 0
        timeout = self.cfg.ingest.frame_timeout_s
        min_dt = 1.0 / self.cfg.pipeline.max_fps if self.cfg.pipeline.max_fps else 0.0
        try:
            while not self._stop.is_set():
                if max_frames is not None and processed >= max_frames:
                    break
                loop_t0 = time.perf_counter()
                frame = source.next_frame(timeout=timeout)
                if frame is None:
                    if getattr(source, "exhausted", False):
                        break
                    continue  # live source idle — keep tracking state, wait
                annotated = self.process_frame(frame.image, frame.frame_id, frame.timestamp)
                if writer is not None:
                    writer.write(annotated)
                processed += 1
                if min_dt:
                    elapsed = time.perf_counter() - loop_t0
                    if elapsed < min_dt:
                        time.sleep(min_dt - elapsed)
        finally:
            source.release()
            if writer is not None:
                writer.release()
            self.frame_logger.close()
        return processed

    def start_background(self, source: Any) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run, args=(source,), name="tracking-pipeline", daemon=True
        )
        self._thread.start()
        log.info("Tracking pipeline started in background")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
