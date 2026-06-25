"""Synthetic scene generator for running the full pipeline without a camera/GPU.

Each simulated person is drawn as a solid, uniquely-coloured rectangle that
moves, crosses others (occlusion) and periodically leaves the frame. The mock
detector recovers boxes by colour and the mock ReID derives a stable appearance
embedding from crop colour — so identity persistence and ReID recovery can be
demonstrated and benchmarked end-to-end on a laptop.
"""

from __future__ import annotations

import math

import numpy as np

# Background colour the detector treats as "empty". Kept off pure black so that
# anti-aliasing is irrelevant (we render hard-edged rectangles).
BACKGROUND = (16, 16, 16)


def _hsv_to_bgr(h_deg: float, s: float, v: float) -> tuple[int, int, int]:
    """Pure HSV->BGR (no OpenCV dependency). h in degrees, s/v in [0, 1]."""
    c = v * s
    x = c * (1 - abs((h_deg / 60.0) % 2 - 1))
    m = v - c
    sextant = int(h_deg // 60) % 6
    r, g, b = [
        (c, x, 0.0), (x, c, 0.0), (0.0, c, x),
        (0.0, x, c), (x, 0.0, c), (c, 0.0, x),
    ][sextant]
    return (
        int(round((b + m) * 255)),
        int(round((g + m) * 255)),
        int(round((r + m) * 255)),
    )


def build_palette(n: int) -> list[tuple[int, int, int]]:
    """``n`` visually distinct BGR colours, none equal to the background."""
    # Golden-angle hue hop keeps adjacent ids far apart on the colour wheel.
    return [_hsv_to_bgr((i * 137.508) % 360, 0.85, 0.90) for i in range(n)]


class Simulator:
    def __init__(
        self,
        num_people: int = 12,
        width: int = 1280,
        height: int = 720,
        seed: int = 7,
    ) -> None:
        self.width = width
        self.height = height
        self.num_people = num_people
        self.palette = build_palette(max(num_people, 1))
        rng = np.random.default_rng(seed)
        self._t = 0.0
        self.people = []
        for i in range(num_people):
            self.people.append(
                {
                    "id": i,
                    "color": self.palette[i],
                    "cx": rng.uniform(0.15, 0.85) * width,
                    "cy": rng.uniform(0.25, 0.80) * height,
                    "ax": rng.uniform(0.10, 0.35) * width,
                    "ay": rng.uniform(0.03, 0.10) * height,
                    "fx": rng.uniform(0.05, 0.18),
                    "fy": rng.uniform(0.05, 0.18),
                    "phase": rng.uniform(0, 2 * math.pi),
                    "w": int(rng.uniform(0.04, 0.07) * width),
                    "h": int(rng.uniform(0.16, 0.24) * height),
                    # Duty cycle controlling when the person is in-frame.
                    "duty": rng.uniform(0.55, 0.95),
                    "duty_phase": rng.uniform(0, 2 * math.pi),
                    "duty_f": rng.uniform(0.02, 0.06),
                }
            )

    def step(self, dt: float = 1.0) -> None:
        self._t += dt

    def _box(self, p: dict) -> tuple[int, int, int, int] | None:
        t = self._t
        present = math.sin(t * p["duty_f"] + p["duty_phase"]) < (2 * p["duty"] - 1)
        if not present:
            return None
        x = p["cx"] + p["ax"] * math.sin(t * p["fx"] + p["phase"])
        y = p["cy"] + p["ay"] * math.cos(t * p["fy"] + p["phase"])
        x1 = int(x - p["w"] / 2)
        y1 = int(y - p["h"] / 2)
        x2, y2 = x1 + p["w"], y1 + p["h"]
        if x2 < 0 or y2 < 0 or x1 > self.width or y1 > self.height:
            return None
        return (
            max(0, x1),
            max(0, y1),
            min(self.width, x2),
            min(self.height, y2),
        )

    def ground_truth(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """(person_id, bbox) for every person currently in frame."""
        out = []
        for p in self.people:
            box = self._box(p)
            if box is not None:
                out.append((p["id"], box))
        return out

    def render(self) -> np.ndarray:
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = BACKGROUND
        # Later ids drawn on top -> deterministic occlusion ordering.
        for p in self.people:
            box = self._box(p)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            frame[y1:y2, x1:x2] = p["color"]
        return frame
