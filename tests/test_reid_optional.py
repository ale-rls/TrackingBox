"""ReID-optional path: detection + tracking still produce persistent GIDs.

With ReID disabled (the Windows/TouchDesigner default), the Identity Manager
binds GIDs by track continuity. GIDs must still be assigned and stable while a
track lives; only cross-occlusion recovery is sacrificed.
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
from audience_tracker.pipeline import TrackingPipeline  # noqa: E402
from audience_tracker.reid import NullReID  # noqa: E402
from audience_tracker.statestore import InMemoryStateStore  # noqa: E402

FRAMES = 80
PEOPLE = 6


def test_gids_assigned_with_reid_disabled():
    cfg = Config()
    cfg.pipeline.backend = "mock"
    cfg.reid.enabled = False           # the --no-reid path
    cfg.pipeline.render_overlay = False
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False

    parts = build_components(cfg, num_people=PEOPLE)
    # Factory must not wire a real ReID extractor when disabled.
    assert isinstance(parts["reid"], NullReID)

    pipeline = TrackingPipeline(
        cfg=cfg,
        detector=parts["detector"],
        tracker=parts["tracker"],
        reid=parts["reid"],
        store=InMemoryStateStore(),
    )
    sim = parts["simulator"]
    for i in range(FRAMES):
        sim.step()
        pipeline.process_frame(sim.render(), i)

    stats = pipeline.identity.stats()
    assert stats["total_people_seen"] >= 1
    assert stats["active_people"] >= 1
    # No appearance-based recovery happened (ReID off).
    assert pipeline.identity.counters()["recoveries"] == 0


if __name__ == "__main__":
    test_gids_assigned_with_reid_disabled()
    print("ok")
