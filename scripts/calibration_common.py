"""Shared helpers for on-site calibration scripts."""

from __future__ import annotations

import json
import sys
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


def capture_source(source: str) -> int | str:
    """Convert numeric camera indices while leaving files/URLs untouched."""

    return int(source) if str(source).isdigit() else source


def camera_settings(path: str | Path) -> dict[str, int]:
    config = load_config(path)
    camera = config.get("camera")
    if not isinstance(camera, dict):
        return {}
    settings: dict[str, int] = {}
    for key in ("width", "height", "fps"):
        value = camera.get(key)
        if isinstance(value, (int, float)) and value > 0:
            settings[key] = int(value)
    return settings


def open_capture(source: str, config_path: str | Path):
    """Open a live/video source with the same camera settings as runtime."""

    import cv2

    resolved = capture_source(source)
    backends = [None]
    if isinstance(resolved, int) and sys.platform.startswith("win"):
        backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])

    last_cap = None
    for backend in backends:
        cap = cv2.VideoCapture(resolved) if backend is None else cv2.VideoCapture(resolved, backend)
        _apply_camera_settings(cap, config_path)
        if cap.isOpened():
            return cap
        cap.release()
        last_cap = cap
    return last_cap


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


def _apply_camera_settings(cap, config_path: str | Path) -> None:
    import cv2

    settings = camera_settings(config_path)
    if "width" in settings:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
    if "height" in settings:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
    if "fps" in settings:
        cap.set(cv2.CAP_PROP_FPS, settings["fps"])
