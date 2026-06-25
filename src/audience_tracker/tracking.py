"""Tracking adapters.

``ByteTrackTracker`` wraps supervision's ByteTrack (real path). ``IoUTracker``
is a dependency-free greedy-IoU fallback used by the mock backend and in tests.
Both satisfy :class:`audience_tracker.base.Tracker`.

Track IDs are INTERNAL ONLY and must never leave the system (spec).
"""

from __future__ import annotations

from typing import Any

from .config import TrackerConfig
from .models import BBox, Detection, Track


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ByteTrackTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        import supervision as sv  # heavy import, real path only

        self._sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
        )

    def update(self, detections: list[Detection], frame: Any | None = None) -> list[Track]:
        import numpy as np

        sv = self._sv
        if detections:
            xyxy = np.array([d.bbox for d in detections], dtype=float)
            conf = np.array([d.confidence for d in detections], dtype=float)
            class_id = np.zeros(len(detections), dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=float)
            conf = np.empty((0,), dtype=float)
            class_id = np.empty((0,), dtype=int)
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
        tracked = self.tracker.update_with_detections(det)
        out: list[Track] = []
        for box, cid, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
            if tid is None:
                continue
            x1, y1, x2, y2 = box
            out.append(Track(track_id=int(tid),
                             bbox=(float(x1), float(y1), float(x2), float(y2)),
                             confidence=float(cid)))
        return out


class IoUTracker:
    """Greedy-IoU tracker. No third-party deps; good enough for the mock path."""

    def __init__(self, cfg: TrackerConfig, min_iou: float = 0.2) -> None:
        self.min_iou = min_iou
        self.lost_buffer = cfg.lost_track_buffer
        self._next_id = 1
        # track_id -> {"bbox", "conf", "lost"}
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[Detection], frame: Any | None = None) -> list[Track]:
        track_ids = list(self._tracks.keys())
        # Build all candidate (iou, track_id, det_index) pairs above threshold.
        pairs = []
        for ti in track_ids:
            tb = self._tracks[ti]["bbox"]
            for di, det in enumerate(detections):
                score = iou(tb, det.bbox)
                if score >= self.min_iou:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        result: list[Track] = []
        for score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            det = detections[di]
            self._tracks[ti].update(bbox=det.bbox, conf=det.confidence, lost=0)
            result.append(Track(track_id=ti, bbox=det.bbox, confidence=det.confidence))

        # Unmatched detections -> new tracks.
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            ti = self._next_id
            self._next_id += 1
            self._tracks[ti] = {"bbox": det.bbox, "conf": det.confidence, "lost": 0}
            result.append(Track(track_id=ti, bbox=det.bbox, confidence=det.confidence))

        # Age + retire unmatched tracks.
        for ti in track_ids:
            if ti in used_tracks:
                continue
            self._tracks[ti]["lost"] += 1
            if self._tracks[ti]["lost"] > self.lost_buffer:
                del self._tracks[ti]

        return result
