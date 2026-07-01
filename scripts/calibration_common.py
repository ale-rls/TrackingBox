"""Shared helpers for on-site calibration scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def update_section(path: str | Path, section: str, values: dict[str, Any]) -> dict[str, Any]:
    config = load_config(path)
    existing = config.get(section)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(values)
    config[section] = existing
    save_config(path, config)
    return config


def parse_points(raw: str) -> list[list[float]]:
    """Parse points from 'x,y;x,y;...'."""

    points: list[list[float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point {item!r}; expected x,y")
        points.append([float(parts[0]), float(parts[1])])
    if not points:
        raise ValueError("At least one point is required")
    return points
