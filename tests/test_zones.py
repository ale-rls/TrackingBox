"""Tests for floor-space zone geometry."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from audience_tracker.config import Config, ZonesConfig  # noqa: E402
from audience_tracker.zones import ZoneError, ZoneMap, parse_zone, point_in_polygon  # noqa: E402


def test_rect_zone_matches_floor_points_and_boundary():
    zones = ZoneMap(
        ZonesConfig(
            enabled=True,
            default_zone="outside",
            zones=[
                {"id": "left", "label": "Left", "rect": [0, 0, 0.5, 1]},
                {"id": "right", "label": "Right", "x": 0.5, "y": 0, "width": 0.5, "height": 1},
            ],
        )
    )

    assert zones.zone_for((0.25, 0.5)) == "left"
    assert zones.zone_for((0.75, 0.5)) == "right"
    assert zones.zone_for((0.5, 0.5)) == "left"
    assert zones.zone_for((1.2, 0.5)) == "outside"


def test_polygon_zone_and_disabled_config():
    enabled = ZoneMap(
        ZonesConfig(
            enabled=True,
            zones=[
                {
                    "id": "triangle",
                    "points": [[0, 0], [1, 0], [0, 1]],
                }
            ],
        )
    )
    disabled = ZoneMap(ZonesConfig(enabled=False, zones=[{"id": "all", "rect": [0, 0, 1, 1]}]))

    assert enabled.zone_for((0.2, 0.2)) == "triangle"
    assert enabled.zone_for((0.9, 0.9)) is None
    assert disabled.zone_for((0.2, 0.2)) is None


def test_counts_include_configured_and_default_zones():
    zones = ZoneMap(
        ZonesConfig(
            enabled=True,
            default_zone="outside",
            zones=[
                {"id": "a", "rect": [0, 0, 0.5, 1]},
                {"id": "b", "rect": [0.5, 0, 1, 1]},
            ],
        )
    )

    assert zones.counts_for([(0.1, 0.1), (0.8, 0.1), (1.2, 0.1), None]) == {
        "a": 1,
        "b": 1,
        "outside": 1,
    }


def test_invalid_zone_config_raises_helpful_error():
    with pytest.raises(ZoneError, match="zone id is required"):
        parse_zone({"rect": [0, 0, 1, 1]})

    with pytest.raises(ZoneError, match="must have at least 3 points"):
        point_in_polygon((0, 0), [[0, 0], [1, 1]])


def test_zones_load_from_config_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "zones": {
            "enabled": true,
            "default_zone": "outside",
            "zones": [{"id": "stage", "rect": [0, 0, 1, 1]}]
          }
        }
        """,
        encoding="utf-8",
    )

    cfg = Config.load(str(path))

    assert cfg.zones.enabled is True
    assert cfg.zones.default_zone == "outside"
    assert cfg.zones.zones == [{"id": "stage", "rect": [0, 0, 1, 1]}]
