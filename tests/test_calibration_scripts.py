"""Tests for calibration script helpers that do not require a camera."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from calibration_common import parse_points, update_section  # noqa: E402


def test_parse_points_reads_semicolon_separated_pairs():
    assert parse_points("0,0;1.5,2; 3, 4 ") == [[0.0, 0.0], [1.5, 2.0], [3.0, 4.0]]


def test_update_section_preserves_existing_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"api":{"port":8000},"calibration":{"enabled":false}}', encoding="utf-8")

    updated = update_section(path, "calibration", {"enabled": True, "floor_points": [[0, 0]]})

    assert updated["api"] == {"port": 8000}
    assert updated["calibration"]["enabled"] is True
    assert updated["calibration"]["floor_points"] == [[0, 0]]
