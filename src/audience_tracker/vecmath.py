"""Tiny pure-Python vector helpers for ReID embeddings.

Kept numpy-free so the Identity Manager (and its tests) run with only the
standard library. Embeddings are short (OSNet x1.0 = 512 dims) and matched
against a small set of lost identities, so pure Python is fast enough.
"""

from __future__ import annotations

import math
from typing import Sequence

Vector = Sequence[float]


def normalize(vec: Vector) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def ema(avg: Vector | None, new: Vector, alpha: float) -> list[float]:
    """Exponential moving average of two equal-length vectors, L2-normalized.

    ``alpha`` weights the new sample. With ``avg is None`` the new sample is
    adopted directly.
    """
    if avg is None or len(avg) != len(new):
        return normalize(new)
    blended = [(1.0 - alpha) * a + alpha * b for a, b in zip(avg, new)]
    return normalize(blended)
