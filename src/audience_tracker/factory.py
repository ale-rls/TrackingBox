"""Builds pipeline components from a :class:`Config`, choosing the real ML stack
or the synthetic mock stack.

Backend resolution (``cfg.pipeline.backend``):
  * ``real`` — force YOLO11m + ByteTrack + OSNet (requires the ``ml`` extra).
  * ``mock`` — synthetic simulator + colour detector + colour ReID (CPU only).
  * ``auto`` — use ``real`` if the ML stack imports, else ``mock``.
"""

from __future__ import annotations

import importlib.util
import logging

from .config import Config

log = logging.getLogger("audience_tracker.factory")


def _can_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def resolve_backend(cfg: Config) -> str:
    backend = cfg.pipeline.backend
    if backend in ("real", "mock"):
        return backend
    have_real = _can_import("ultralytics") and _can_import("supervision")
    return "real" if have_real else "mock"


class SimulatedCamera:
    """Drop-in for ``cv2.VideoCapture`` backed by the synthetic simulator."""

    def __init__(self, simulator, max_frames: int | None = None) -> None:
        self.sim = simulator
        self.max_frames = max_frames
        self._i = 0

    def read(self):
        if self.max_frames is not None and self._i >= self.max_frames:
            return False, None
        self.sim.step()
        self._i += 1
        return True, self.sim.render()

    def release(self) -> None:
        pass


def open_source(cfg: Config, simulator=None, max_frames: int | None = None):
    """Return a camera-like object exposing ``read()`` and ``release()``."""
    if simulator is not None:
        return SimulatedCamera(simulator, max_frames=max_frames)
    import cv2

    src = cfg.pipeline.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src!r}")
    return cap


def build_components(cfg: Config, num_people: int = 24) -> dict:
    """Construct detector, tracker, reid and (for mock) a simulator."""
    backend = resolve_backend(cfg)
    log.info("Using %s backend", backend)

    if backend == "real":
        from .detection import YOLODetector
        from .reid import NullReID, OSNetExtractor
        from .tracking import ByteTrackTracker

        detector = YOLODetector(cfg.detector, cfg.pipeline.device)
        tracker = ByteTrackTracker(cfg.tracker)
        reid = OSNetExtractor(cfg.reid, cfg.pipeline.device) if cfg.reid.enabled else NullReID()
        simulator = None
    else:
        from .detection import MockDetector
        from .reid import MockReID, NullReID
        from .simulator import Simulator
        from .tracking import IoUTracker

        simulator = Simulator(num_people=num_people)
        detector = MockDetector(simulator.palette)
        tracker = IoUTracker(cfg.tracker)
        reid = MockReID() if cfg.reid.enabled else NullReID()

    return {
        "backend": backend,
        "detector": detector,
        "tracker": tracker,
        "reid": reid,
        "simulator": simulator,
    }


def build_pipeline(cfg: Config, store=None, num_people: int = 24):
    """Construct a ready-to-run pipeline plus its components/store.

    Returns a dict with keys: ``pipeline``, ``store``, ``simulator``, ``backend``.
    """
    from .pipeline import TrackingPipeline
    from .statestore import InMemoryStateStore

    store = store or InMemoryStateStore()
    parts = build_components(cfg, num_people=num_people)
    pipeline = TrackingPipeline(
        cfg=cfg,
        detector=parts["detector"],
        tracker=parts["tracker"],
        reid=parts["reid"],
        store=store,
    )
    return {
        "pipeline": pipeline,
        "store": store,
        "simulator": parts["simulator"],
        "backend": parts["backend"],
    }
