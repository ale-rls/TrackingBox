"""Floor-plane projection helpers.

This module maps camera image points to a top-down floor coordinate system. The
runtime path is:

    bbox -> anchor pixel -> optional lens undistortion -> homography -> floor

The homography solver is dependency-free so the mock/dev path remains light.
OpenCV is imported only when lens undistortion is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import CalibrationConfig, LensCalibrationConfig
from .models import BBox

Point = tuple[float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class ProjectionError(ValueError):
    """Raised when calibration data cannot produce a usable projection."""


@dataclass(slots=True)
class ProjectionResult:
    floor: Point | None
    valid: bool


def bbox_anchor(bbox: BBox, anchor: str = "bottom_center") -> Point:
    """Return the image point that should represent a person's floor contact."""

    x1, y1, x2, y2 = bbox
    if anchor == "bottom_center":
        return ((x1 + x2) / 2.0, y2)
    if anchor == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    raise ProjectionError(f"Unsupported calibration anchor: {anchor!r}")


def compute_homography(
    image_points: Sequence[Sequence[float]],
    floor_points: Sequence[Sequence[float]],
) -> Matrix3:
    """Compute a homography from image points to floor points.

    Four point pairs are the minimum. More pairs are solved with least squares
    via normal equations.
    """

    src = [_point(p, "image point") for p in image_points]
    dst = [_point(p, "floor point") for p in floor_points]
    if len(src) != len(dst):
        raise ProjectionError("image_points and floor_points must have the same length")
    if len(src) < 4:
        raise ProjectionError("at least 4 image/floor point pairs are required")

    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)

    normal = [[0.0 for _ in range(8)] for _ in range(8)]
    target = [0.0 for _ in range(8)]
    for row, b in zip(rows, rhs):
        for i in range(8):
            target[i] += row[i] * b
            for j in range(8):
                normal[i][j] += row[i] * row[j]

    h = _solve_linear_system(normal, target)
    return (
        (h[0], h[1], h[2]),
        (h[3], h[4], h[5]),
        (h[6], h[7], 1.0),
    )


def apply_homography(point: Sequence[float], matrix: Matrix3) -> Point:
    x, y = _point(point, "point")
    den = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(den) < 1e-12:
        raise ProjectionError("homography maps point to infinity")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / den,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / den,
    )


class FloorProjector:
    """Project person bboxes into the configured floor coordinate space."""

    def __init__(self, lens: LensCalibrationConfig, calibration: CalibrationConfig) -> None:
        self.lens = lens
        self.calibration = calibration
        self._history: dict[int, Point] = {}
        self._undistorter = _PointUndistorter(lens)
        if self.enabled:
            image_points = [self._undistorter.undistort(p) for p in calibration.image_points]
            self._homography = compute_homography(image_points, calibration.floor_points)
        else:
            self._homography = None

    @property
    def enabled(self) -> bool:
        return bool(self.calibration.enabled)

    def project_bbox(self, bbox: BBox | None, gid: int | None = None) -> ProjectionResult:
        if not self.enabled or bbox is None or self._homography is None:
            return ProjectionResult(floor=None, valid=False)

        point = bbox_anchor(bbox, self.calibration.anchor)
        point = self._undistorter.undistort(point)
        floor = apply_homography(point, self._homography)
        floor = self._smooth(gid, floor)
        return ProjectionResult(floor=floor, valid=True)

    def image_to_floor(self, point: Sequence[float]) -> ProjectionResult:
        if not self.enabled or self._homography is None:
            return ProjectionResult(floor=None, valid=False)
        undistorted = self._undistorter.undistort(point)
        return ProjectionResult(floor=apply_homography(undistorted, self._homography), valid=True)

    def _smooth(self, gid: int | None, floor: Point) -> Point:
        if gid is None:
            return floor
        alpha = self.calibration.smoothing_alpha
        if alpha <= 0.0 or alpha >= 1.0:
            self._history[gid] = floor
            return floor
        prev = self._history.get(gid)
        if prev is None:
            self._history[gid] = floor
            return floor
        smoothed = (
            prev[0] * (1.0 - alpha) + floor[0] * alpha,
            prev[1] * (1.0 - alpha) + floor[1] * alpha,
        )
        self._history[gid] = smoothed
        return smoothed


class _PointUndistorter:
    def __init__(self, lens: LensCalibrationConfig) -> None:
        self.lens = lens
        self.enabled = bool(lens.enabled)
        self._cv2 = None
        self._np = None
        if self.enabled:
            if not lens.camera_matrix or not lens.distortion_coeffs:
                raise ProjectionError(
                    "lens calibration is enabled but matrix/coefficients are missing"
                )
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
            except Exception as exc:  # pragma: no cover - depends on optional runtime deps
                raise ProjectionError("lens calibration requires OpenCV and numpy") from exc
            self._cv2 = cv2
            self._np = np

    def undistort(self, point: Sequence[float]) -> Point:
        x, y = _point(point, "point")
        if not self.enabled:
            return (x, y)

        assert self._cv2 is not None and self._np is not None
        pts = self._np.array([[[x, y]]], dtype=self._np.float64)
        camera = self._np.array(self.lens.camera_matrix, dtype=self._np.float64)
        dist = self._np.array(self.lens.distortion_coeffs, dtype=self._np.float64).reshape(-1, 1)
        if self.lens.model == "fisheye":
            out = self._cv2.fisheye.undistortPoints(pts, camera, dist, P=camera)
        elif self.lens.model == "pinhole":
            out = self._cv2.undistortPoints(pts, camera, dist, P=camera)
        else:
            raise ProjectionError(f"Unsupported lens calibration model: {self.lens.model!r}")
        return (float(out[0][0][0]), float(out[0][0][1]))


def _point(raw: Sequence[float], label: str) -> Point:
    if len(raw) != 2:
        raise ProjectionError(f"{label} must contain exactly two values")
    return (float(raw[0]), float(raw[1]))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ProjectionError("calibration points are singular or nearly singular")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        scale = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= scale

        for r in range(n):
            if r == col:
                continue
            factor = augmented[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[r][j] -= factor * augmented[col][j]

    return [augmented[i][n] for i in range(n)]
