"""Tests for calibration script helpers that do not require a camera."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from calibration_common import (  # noqa: E402
    camera_settings,
    capture_source,
    parse_points,
    update_section,
)


def test_parse_points_reads_semicolon_separated_pairs():
    assert parse_points("0,0;1.5,2; 3, 4 ") == [[0.0, 0.0], [1.5, 2.0], [3.0, 4.0]]


def test_update_section_preserves_existing_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"api":{"port":8000},"calibration":{"enabled":false}}', encoding="utf-8")

    updated = update_section(path, "calibration", {"enabled": True, "floor_points": [[0, 0]]})

    assert updated["api"] == {"port": 8000}
    assert updated["calibration"]["enabled"] is True
    assert updated["calibration"]["floor_points"] == [[0, 0]]


def test_capture_source_converts_numeric_camera_indices():
    assert capture_source("12") == 12
    assert capture_source("rtsp://camera/live") == "rtsp://camera/live"


def test_camera_settings_reads_runtime_camera_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"camera":{"device_index":12,"width":1920,"height":1080,"fps":30}}',
        encoding="utf-8",
    )

    assert camera_settings(path) == {"width": 1920, "height": 1080, "fps": 30}
