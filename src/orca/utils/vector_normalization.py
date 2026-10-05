"""Vector normalization helpers shared by ORCA algorithms."""

from __future__ import annotations

import numpy as np


def safe_normalize(v: np.ndarray, atol: float = 1.0e-14) -> np.ndarray:
    """Return ``v / ||v||_2`` or a zero vector when ``v`` is near zero."""

    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= atol:
        return np.zeros_like(arr, dtype=float)
    return arr / norm


def normalize_rows(matrix: np.ndarray, atol: float = 1.0e-14) -> np.ndarray:
    """Normalize each row of a two-dimensional matrix."""

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return np.vstack([safe_normalize(row, atol=atol) for row in arr]) if arr.size else arr.copy()


def normalize_last_axis(tensor: np.ndarray, atol: float = 1.0e-14) -> np.ndarray:
    """Normalize vectors along the last axis of an arbitrary-dimensional array."""

    arr = np.asarray(tensor, dtype=float)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr, dtype=float), where=norms > atol)
