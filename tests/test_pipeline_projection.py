"""Pipeline integration for derived floor coordinates."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker.config import Config  # noqa: E402
from audience_tracker.models import Detection, Track  # noqa: E402
from audience_tracker.pipeline import TrackingPipeline  # noqa: E402
from audience_tracker.statestore import InMemoryStateStore  # noqa: E402


class OnePersonDetector:
    def detect(self, frame):
        return [Detection(bbox=(10.0, 20.0, 30.0, 80.0), confidence=0.9)]


class PassthroughTracker:
    def update(self, detections, frame):
        return [
            Track(track_id=1, bbox=detections[0].bbox, confidence=detections[0].confidence)
        ]


class NoReID:
    last_inference_ms = 0.0

    def extract(self, frame, bboxes):
        return [[] for _ in bboxes]


def test_pipeline_publishes_floor_position_from_bbox_anchor():
    cfg = Config()
    cfg.reid.enabled = False
    cfg.pipeline.render_overlay = False
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False
    cfg.calibration.enabled = True
    cfg.calibration.image_points = [[0, 0], [100, 0], [100, 100], [0, 100]]
    cfg.calibration.floor_points = [[0, 0], [1, 0], [1, 1], [0, 1]]

    store = InMemoryStateStore()
    pipeline = TrackingPipeline(
        cfg=cfg,
        detector=OnePersonDetector(),
        tracker=PassthroughTracker(),
        reid=NoReID(),
        store=store,
    )

    pipeline.process_frame(frame=object(), frame_index=1, now=1.0)

    active = store.get_active()
    assert len(active) == 1
    assert active[0]["floor"] == [0.2, 0.8]
    assert active[0]["floor_valid"] is True


def test_pipeline_assigns_floor_zone_and_counts():
    cfg = Config()
    cfg.reid.enabled = False
    cfg.pipeline.render_overlay = False
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False
    cfg.calibration.enabled = True
    cfg.calibration.image_points = [[0, 0], [100, 0], [100, 100], [0, 100]]
    cfg.calibration.floor_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
    cfg.zones.enabled = True
    cfg.zones.default_zone = "outside"
    cfg.zones.zones = [
        {"id": "front", "rect": [0, 0, 1, 0.5]},
        {"id": "back", "rect": [0, 0.5, 1, 1]},
    ]

    store = InMemoryStateStore()
    pipeline = TrackingPipeline(
        cfg=cfg,
        detector=OnePersonDetector(),
        tracker=PassthroughTracker(),
        reid=NoReID(),
        store=store,
    )

    pipeline.process_frame(frame=object(), frame_index=1, now=1.0)

    active = store.get_active()
    assert active[0]["zone"] == "back"
    assert store.get_zone_counts() == {"front": 0, "back": 1, "outside": 0}
