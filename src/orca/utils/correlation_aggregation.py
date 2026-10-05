"""Aggregation of local ORCA interactions into objective correlation matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def aggregate_interactions_to_adjacency(
    strengths: np.ndarray,
    weights: np.ndarray,
    *,
    zero_weight_value: float = 0.5,
    self_correlation: float = 1.0,
    min_correlation: float = 0.0,
    max_correlation: float = 1.0,
    rescaling_factor: float = 0.5,
    atol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate local strengths into an ORCA objective adjacency matrix.

    For each objective pair ``(i, j)``, this computes

        A_ij = 0.5 * (1 + sum(W_ijkn S_ijkn) / sum(W_ijkn)).

    The summation is over all axes after the first two objective axes.

    Returns
    -------
    adj_matrix, total_weights, weighted_strengths
    """

    s = np.asarray(strengths, dtype=float)
    w = np.asarray(weights, dtype=float)
    if s.shape != w.shape:
        raise ValueError("strengths and weights must have the same shape")
    if s.ndim < 3:
        raise ValueError("strengths must have at least three dimensions")
    if s.shape[0] != s.shape[1]:
        raise ValueError("first two strengths axes must both index objectives")

    sum_axes = tuple(range(2, s.ndim))
    total_weights = np.sum(w, axis=sum_axes)
    weighted_strengths = np.sum(w * s, axis=sum_axes)

    num_objectives = s.shape[0]
    adj = np.eye(num_objectives, dtype=float) * self_correlation
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            if total_weights[i, j] <= atol:
                value = zero_weight_value
            else:
                value = rescaling_factor * (self_correlation + weighted_strengths[i, j] / total_weights[i, j])
            value = float(np.clip(value, min_correlation, max_correlation))
            adj[i, j] = adj[j, i] = value

    return adj, total_weights, weighted_strengths
