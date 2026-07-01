"""ReID (appearance embedding) adapters.

``OSNetExtractor`` wraps torchreid's OSNet x1.0 (real path). ``MockReID``
produces a stable colour descriptor from the crop (CPU path) — enough for the
synthetic simulator to demonstrate identity recovery. ``NullReID`` disables
ReID. All satisfy :class:`audience_tracker.base.ReIDExtractor`.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from .config import ReIDConfig
from .models import BBox, Embedding
from .vecmath import normalize


def _crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return frame[y1:y2, x1:x2]


class OSNetExtractor:
    def __init__(self, cfg: ReIDConfig, device: str = "auto") -> None:
        # ImportError (not just ModuleNotFoundError) also covers a torchreid
        # release where the module exists but FeatureExtractor moved.
        try:
            from torchreid.utils import FeatureExtractor  # type: ignore[import-not-found]
        except ImportError:
            from torchreid.reid.utils import FeatureExtractor  # type: ignore[import-not-found]

        if device == "auto":
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device
        self.extractor = FeatureExtractor(model_name=cfg.model_name, device=dev)
        self.last_inference_ms = 0.0

    def extract(self, frame: Any, bboxes: Sequence[BBox]) -> list[Embedding]:
        if not bboxes:
            return []
        # torchreid treats ndarray input as RGB; our frames are OpenCV BGR.
        crops = [np.ascontiguousarray(_crop(frame, b)[:, :, ::-1]) for b in bboxes]
        t0 = time.perf_counter()
        feats = self.extractor(crops)  # torch.Tensor (N, 512)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        feats = feats.cpu().numpy()
        return [normalize(f.tolist()) for f in feats]


class MockReID:
    """Stable, colour-based appearance descriptor.

    Mean BGR over the whole crop plus the top and bottom halves, normalised.
    For the simulator's solid-colour people this is unique and stable, so the
    Identity Manager can recover GIDs after occlusion.
    """

    def __init__(self) -> None:
        self.last_inference_ms = 0.0

    def extract(self, frame: Any, bboxes: Sequence[BBox]) -> list[Embedding]:
        out: list[Embedding] = []
        t0 = time.perf_counter()
        for b in bboxes:
            crop = _crop(frame, b)
            if crop.size == 0:
                out.append(normalize([0.0] * 9))
                continue
            top = crop[: max(1, crop.shape[0] // 2)]
            bot = crop[crop.shape[0] // 2 :]
            feat = np.concatenate(
                [crop.reshape(-1, 3).mean(0),
                 top.reshape(-1, 3).mean(0),
                 bot.reshape(-1, 3).mean(0)]
            ) / 255.0
            out.append(normalize(feat.tolist()))
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return out


class NullReID:
    """ReID disabled — returns no embeddings."""

    last_inference_ms = 0.0

    def extract(self, frame: Any, bboxes: Sequence[BBox]) -> list[Embedding]:
        return [[] for _ in bboxes]
