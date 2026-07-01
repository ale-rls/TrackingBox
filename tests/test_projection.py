"""Tests for floor-plane projection and calibration math."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from audience_tracker.config import CalibrationConfig, LensCalibrationConfig  # noqa: E402
from audience_tracker.projection import (  # noqa: E402
    FloorProjector,
    ProjectionError,
    apply_homography,
    bbox_anchor,
    compute_homography,
)


def assert_point_close(actual, expected, tol=1e-6):
    assert actual[0] == pytest.approx(expected[0], abs=tol)
    assert actual[1] == pytest.approx(expected[1], abs=tol)


def test_bbox_anchor_uses_bottom_center_by_default():
    bbox = (10.0, 20.0, 30.0, 80.0)

    assert bbox_anchor(bbox) == (20.0, 80.0)
    assert bbox_anchor(bbox, "center") == (20.0, 50.0)


def test_unknown_bbox_anchor_rejected():
    with pytest.raises(ProjectionError):
        bbox_anchor((0.0, 0.0, 10.0, 20.0), "nose")


def test_homography_maps_image_points_to_floor_points():
    matrix = compute_homography(
        image_points=[[0, 0], [100, 0], [100, 200], [0, 200]],
        floor_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )

    assert_point_close(apply_homography((50, 100), matrix), (0.5, 0.5))
    assert_point_close(apply_homography((100, 200), matrix), (1.0, 1.0))


def test_homography_accepts_more_than_four_pairs():
    matrix = compute_homography(
        image_points=[[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        floor_points=[[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]],
    )

    assert_point_close(apply_homography((25, 75), matrix), (2.5, 7.5))


def test_projector_disabled_returns_invalid_result():
    projector = FloorProjector(LensCalibrationConfig(), CalibrationConfig(enabled=False))

    result = projector.project_bbox((0.0, 0.0, 10.0, 20.0), gid=1)

    assert result.floor is None
    assert result.valid is False


def test_projector_projects_bbox_bottom_center_and_smooths_by_gid():
    cfg = CalibrationConfig(
        enabled=True,
        image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        floor_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        smoothing_alpha=0.5,
    )
    projector = FloorProjector(LensCalibrationConfig(), cfg)

    first = projector.project_bbox((0.0, 0.0, 20.0, 50.0), gid=7)
    second = projector.project_bbox((0.0, 0.0, 40.0, 70.0), gid=7)

    assert first.valid is True
    assert_point_close(first.floor, (0.1, 0.5))
    assert second.valid is True
    assert_point_close(second.floor, (0.15, 0.6))
