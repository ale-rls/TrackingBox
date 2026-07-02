"""End-to-end smoke test of the full pipeline on the synthetic backend.

Exercises simulator -> mock detector -> IoU tracker -> mock ReID -> Identity
Manager -> state store, with no GPU/camera. Verifies the pipeline runs and that
ReID keeps GID churn bounded (broken ReID would balloon the GID count).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")  # numpy required for the frame pipeline

from audience_tracker.config import Config  # noqa: E402
from audience_tracker.factory import build_pipeline, open_source  # noqa: E402

NUM_PEOPLE = 8
FRAMES = 150


def _make(max_frames=FRAMES):
    """Mock-backend pipeline + finite simulator source (headless flags set)."""
    cfg = Config()
    cfg.pipeline.backend = "mock"
    cfg.pipeline.render_overlay = False
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False
    built = build_pipeline(cfg, num_people=NUM_PEOPLE)
    source = open_source(cfg, simulator=built["simulator"], max_frames=max_frames)
    return built["pipeline"], source


def _run():
    pipeline, camera = _make()
    processed = 0
    for i in range(FRAMES):
        ok, frame = camera.read()
        if not ok:
            break
        pipeline.process_frame(frame, i)
        processed += 1
    return pipeline, processed


def test_pipeline_runs_and_assigns_gids():
    pipeline, processed = _run()
    assert processed == FRAMES
    stats = pipeline.identity.stats()
    assert stats["total_people_seen"] >= 1
    assert stats["active_people"] >= 1


def test_reid_keeps_gid_count_bounded():
    pipeline, _ = _run()
    total = pipeline.identity.stats()["total_people_seen"]
    # With working ReID, re-entries reuse GIDs; without it the count explodes.
    assert total <= NUM_PEOPLE * 2, f"GID count {total} suggests ReID is not recovering identities"


def test_metrics_populated():
    pipeline, _ = _run()
    snap = pipeline.metrics.snapshot()
    assert snap["fps"] > 0
    assert snap["frames_processed"] == FRAMES
    assert "reid_inference_time_ms" in snap


def test_run_survives_frame_processing_errors():
    """A failing frame must not kill the run loop mid-show: the loop logs,
    skips the frame, and keeps consuming until the source actually ends."""
    pipeline, source = _make(max_frames=20)

    real_detect = pipeline.detector.detect
    calls = {"n": 0}

    def flaky_detect(frame):
        calls["n"] += 1
        if calls["n"] in (3, 4):
            raise RuntimeError("simulated detector fault")
        return real_detect(frame)

    pipeline.detector.detect = flaky_detect
    processed = pipeline.run(source)
    assert calls["n"] == 20    # every frame was attempted
    assert processed == 18     # the two faulty frames were skipped, not fatal


if __name__ == "__main__":
    test_pipeline_runs_and_assigns_gids()
    test_reid_keeps_gid_count_bounded()
    test_metrics_populated()
    test_run_survives_frame_processing_errors()
    print("ok")
