"""Floor-space zones used for counts, game answers, and operator maps.

Zones are defined in calibrated floor coordinates, not camera pixels. The first
enabled zone that contains a point wins, which lets operators intentionally layer
small regions before larger fallback regions in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import ZonesConfig

Point = tuple[float, float]


class ZoneError(ValueError):
    """Invalid zone configuration."""


@dataclass(frozen=True, slots=True)
class ZoneDefinition:
    id: str
    label: str
    points: tuple[Point, ...]

    def contains(self, point: Sequence[float]) -> bool:
        return point_in_polygon(_point(point, "floor point"), self.points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "points": [[x, y] for x, y in self.points],
        }


class ZoneMap:
    def __init__(self, cfg: ZonesConfig) -> None:
        self.enabled = bool(cfg.enabled)
        self.default_zone = cfg.default_zone
        self.zones = tuple(parse_zone(raw) for raw in cfg.zones if raw.get("enabled", True))

    def zone_for(self, floor_point: Sequence[float] | None) -> str | None:
        if not self.enabled or floor_point is None:
            return None
        point = _point(floor_point, "floor point")
        for zone in self.zones:
            if zone.contains(point):
                return zone.id
        return self.default_zone

    def counts_for(self, points: Sequence[Sequence[float] | None]) -> dict[str, int]:
        counts = {zone.id: 0 for zone in self.zones}
        if self.default_zone:
            counts.setdefault(self.default_zone, 0)
        for point in points:
            zone_id = self.zone_for(point)
            if zone_id is not None:
                counts[zone_id] = counts.get(zone_id, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_zone": self.default_zone,
            "zones": [zone.as_dict() for zone in self.zones],
        }


def parse_zone(raw: dict[str, Any]) -> ZoneDefinition:
    zone_id = str(raw.get("id", "")).strip()
    if not zone_id:
        raise ZoneError("zone id is required")
    label = str(raw.get("label") or raw.get("name") or zone_id)

    points = raw.get("points") or raw.get("polygon")
    if points is not None:
        polygon = tuple(_point(point, f"zone {zone_id} point") for point in points)
    else:
        polygon = _rect_points(raw, zone_id)

    if len(polygon) < 3:
        raise ZoneError(f"zone {zone_id} must have at least 3 points")
    return ZoneDefinition(id=zone_id, label=label, points=polygon)


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return True when point is inside or on the boundary of a polygon."""
    x, y = _point(point, "floor point")
    vertices = [_point(p, "polygon point") for p in polygon]
    if len(vertices) < 3:
        raise ZoneError("polygon must have at least 3 points")

    inside = False
    j = len(vertices) - 1
    for i, current in enumerate(vertices):
        previous = vertices[j]
        if _on_segment((x, y), previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_at_y:
                inside = not inside
        j = i
    return inside


def _rect_points(raw: dict[str, Any], zone_id: str) -> tuple[Point, ...]:
    rect = raw.get("rect")
    if rect is not None:
        if len(rect) != 4:
            raise ZoneError(f"zone {zone_id} rect must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(v) for v in rect)
    elif all(key in raw for key in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = (float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
    elif all(key in raw for key in ("x", "y", "width", "height")):
        x1 = float(raw["x"])
        y1 = float(raw["y"])
        x2 = x1 + float(raw["width"])
        y2 = y1 + float(raw["height"])
    else:
        raise ZoneError(f"zone {zone_id} needs points, polygon, rect, or x/y/width/height")

    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _point(point: Sequence[float], label: str) -> Point:
    if len(point) != 2:
        raise ZoneError(f"{label} must be [x, y]")
    return (float(point[0]), float(point[1]))


def _on_segment(point: Point, a: Point, b: Point, epsilon: float = 1e-9) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    return (
        min(ax, bx) - epsilon <= px <= max(ax, bx) + epsilon
        and min(ay, by) - epsilon <= py <= max(ay, by) + epsilon
    )
