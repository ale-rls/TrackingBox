"""End-to-end: frames flow through the QueueFrameSource into the pipeline.

Proves the FrameSource abstraction (Video Ingestion spec) drives the existing
detect->track->reid->identity pipeline, decoupled from any camera/transport.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")

from audience_tracker.config import Config  # noqa: E402
from audience_tracker.factory import build_components  # noqa: E402
from audience_tracker.ingestion.frame_source import QueueFrameSource  # noqa: E402
from audience_tracker.models import Frame  # noqa: E402
from audience_tracker.pipeline import TrackingPipeline  # noqa: E402
from audience_tracker.statestore import InMemoryStateStore  # noqa: E402

FRAMES = 60
PEOPLE = 6


def test_pipeline_consumes_queue_frame_source():
    cfg = Config()
    cfg.pipeline.backend = "mock"
    cfg.pipeline.render_overlay = False
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False

    parts = build_components(cfg, num_people=PEOPLE)
    store = InMemoryStateStore()
    pipeline = TrackingPipeline(
        cfg=cfg,
        detector=parts["detector"],
        tracker=parts["tracker"],
        reid=parts["reid"],
        store=store,
    )

    # Render synthetic frames and stream them in through the ingestion queue.
    sim = parts["simulator"]
    q = QueueFrameSource(maxsize=128)  # large enough to avoid drops in the test
    for fid in range(1, FRAMES + 1):
        sim.step()
        img = sim.render()
        q.push(Frame(image=img, timestamp=time.time(), frame_id=fid,
                     width=img.shape[1], height=img.shape[0]))
    q.release()  # no more frames -> source becomes exhausted once drained

    processed = pipeline.run(q, max_frames=FRAMES)

    assert processed == FRAMES
    stats = pipeline.identity.stats()
    assert stats["total_people_seen"] >= 1
    # IDs that the public API would expose are GIDs, never tracker ids.
    assert all("track_id" not in s.summary() for s in pipeline.identity.snapshot(time.time()))


if __name__ == "__main__":
    test_pipeline_consumes_queue_frame_source()
    print("ok")
