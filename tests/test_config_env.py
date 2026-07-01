"""Env-var override coercion — including fields whose default is None
(regression: ``AT_PIPELINE_MAX_FPS=30`` used to arrive as the string "30" and
crash the pipeline run loop at ``1.0 / max_fps``).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker.config import Config  # noqa: E402


def test_none_default_float_env_becomes_float():
    cfg = Config().apply_env({"AT_PIPELINE_MAX_FPS": "30"})
    assert cfg.pipeline.max_fps == 30.0
    assert isinstance(cfg.pipeline.max_fps, float)


def test_none_default_str_env_stays_str():
    cfg = Config().apply_env({"AT_PIPELINE_OUTPUT_PATH": "out.mp4"})
    assert cfg.pipeline.output_path == "out.mp4"


def test_typed_defaults_still_coerce():
    cfg = Config().apply_env(
        {
            "AT_REID_SIMILARITY_THRESHOLD": "0.7",
            "AT_INGEST_QUEUE_SIZE": "5",
            "AT_PIPELINE_RENDER_OVERLAY": "false",
        }
    )
    assert cfg.reid.similarity_threshold == 0.7
    assert cfg.ingest.queue_size == 5
    assert cfg.pipeline.render_overlay is False
