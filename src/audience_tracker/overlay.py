"""Overlay renderer — draws the anonymous GID above each audience member.

Renders ``#<gid>`` (plus ``conf:`` in debug mode). Uses OpenCV when available
for crisp text; otherwise falls back to numpy rectangle markers so the pipeline
still runs headless.
"""

from __future__ import annotations

from typing import Any

from .config import OverlayConfig
from .models import AudienceState

try:  # OpenCV is optional for headless/metric-only runs.
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - exercised only without opencv
    _HAS_CV2 = False


class OverlayRenderer:
    def __init__(self, cfg: OverlayConfig) -> None:
        self.cfg = cfg

    def render(self, frame: Any, states: list[AudienceState], debug: bool | None = None) -> Any:
        debug = self.cfg.debug if debug is None else debug
        if _HAS_CV2:
            return self._render_cv2(frame, states, debug)
        return self._render_numpy(frame, states)

    # --------------------------------------------------------------- #
    def _render_cv2(self, frame, states, debug):
        cfg = self.cfg
        for st in states:
            if not st.bbox:
                continue
            x1, y1, x2, y2 = (int(v) for v in st.bbox)
            color = cfg.box_color if st.visible else cfg.lost_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)
            label = f"#{st.gid}"
            if debug:
                label += f"  conf: {st.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1
            )
            ty = max(th + 4, y1)
            cv2.rectangle(frame, (x1, ty - th - base - 4), (x1 + tw + 6, ty), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, ty - base - 2),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.text_color, 1, cv2.LINE_AA,
            )
        return frame

    def _render_numpy(self, frame, states):
        cfg = self.cfg
        t = max(1, cfg.box_thickness)
        for st in states:
            if not st.bbox:
                continue
            x1, y1, x2, y2 = (int(v) for v in st.bbox)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            color = cfg.box_color if st.visible else cfg.lost_color
            frame[y1 : y1 + t, x1:x2] = color
            frame[max(0, y2 - t) : y2, x1:x2] = color
            frame[y1:y2, x1 : x1 + t] = color
            frame[y1:y2, max(0, x2 - t) : x2] = color
            # Tag bar whose width encodes the GID (no font available).
            bar = min(x2, x1 + 6 + 4 * (st.gid % 16))
            frame[max(0, y1 - 6) : y1, x1:bar] = color
        return frame
