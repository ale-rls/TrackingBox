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
from generate_checkerboard import black_squares, make_layout, write_pdf, write_svg  # noqa: E402


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


def test_checkerboard_layout_uses_inner_corner_counts():
    layout = make_layout(
        page="a4",
        orientation="landscape",
        board_cols=9,
        board_rows=6,
        square_size_mm=25,
        margin_mm=10,
    )

    assert layout["page_w_mm"] == 297.0
    assert layout["page_h_mm"] == 210.0
    assert layout["squares_x"] == 10
    assert layout["squares_y"] == 7
    assert len(black_squares(layout)) == 35


def test_checkerboard_writers_create_printable_files(tmp_path):
    layout = make_layout(
        page="a4",
        orientation="landscape",
        board_cols=9,
        board_rows=6,
        square_size_mm=25,
        margin_mm=10,
    )
    pdf_path = tmp_path / "board.pdf"
    svg_path = tmp_path / "board.svg"

    write_pdf(pdf_path, layout)
    write_svg(svg_path, layout)

    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")
    assert 'width="297.0mm"' in svg_path.read_text(encoding="utf-8")
