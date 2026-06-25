"""Person detection adapters.

``YOLODetector`` wraps Ultralytics YOLO11m (real path). ``MockDetector``
recovers boxes from the synthetic simulator's colour-coded frames (CPU path).
Both satisfy :class:`audience_tracker.base.Detector`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DetectorConfig
from .models import Detection


class YOLODetector:
    """Ultralytics YOLO11m person detector."""

    def __init__(self, cfg: DetectorConfig, device: str = "auto") -> None:
        from ultralytics import YOLO  # heavy import, real path only

        self.cfg = cfg
        self.device = None if device == "auto" else device
        self.model = YOLO(cfg.model_path)

    def detect(self, frame: Any) -> list[Detection]:
        results = self.model.predict(
            frame,
            classes=[self.cfg.person_class_id],
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.image_size,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            out.append(Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                                  confidence=float(c)))
        return out


class MockDetector:
    """Detects the simulator's solid-colour rectangles by colour mask.

    Knowing only the colour palette (a constant), this reads boxes straight from
    the pixels — so occlusion (one rectangle drawn over another) naturally
    shrinks or hides the covered detection, exercising the ReID recovery path.
    """

    def __init__(self, palette: list[tuple[int, int, int]], min_area: int = 200) -> None:
        self.palette = palette
        self.min_area = min_area
        # Pack each BGR colour into a single int key for fast pixel matching.
        self._keys = {(b << 16) | (g << 8) | r: (b, g, r) for (b, g, r) in palette}

    def detect(self, frame: Any) -> list[Detection]:
        b = frame[:, :, 0].astype(np.int32)
        g = frame[:, :, 1].astype(np.int32)
        r = frame[:, :, 2].astype(np.int32)
        keys = (b << 16) | (g << 8) | r
        present = set(np.unique(keys).tolist())  # skip colours not in this frame

        out: list[Detection] = []
        for key in self._keys:
            if key not in present:
                continue
            mask = keys == key
            count = int(mask.sum())
            if count < self.min_area:
                continue
            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
            # Confidence ~ how filled the bbox is (occlusion lowers it).
            conf = max(0.30, min(0.99, count / area))
            out.append(Detection(bbox=(float(x1), float(y1), float(x2 + 1), float(y2 + 1)),
                                  confidence=float(round(conf, 3))))
        return out
