"""Competition-sensitive weights for local ORCA interactions."""

from __future__ import annotations

from typing import Optional

import numpy as np

EXPONENT_CLIP_LIMIT = 700.0


def logistic_competition_weight(
    strength: np.ndarray | float,
    *,
    alpha_weight: float = 0.9,
    beta_weight: float = 100.0,
) -> np.ndarray | float:
    """Compute ORCA logistic weights for local interaction strengths.

    The local interaction strength ``S`` is usually a projected cosine
    similarity in ``[-1, 1]``. The weight is

        W(S) = 1 - alpha_weight / (1 + exp(-beta_weight * S)).

    Negative strengths therefore receive larger weights, making competitive
    local interactions more influential in the aggregated ORCA adjacency score.
    """

    s = np.asarray(strength, dtype=float)
    clipped_exponent = np.clip(-beta_weight * s, -EXPONENT_CLIP_LIMIT, EXPONENT_CLIP_LIMIT)
    logistic_value = 1.0 / (1.0 + np.exp(clipped_exponent))
    weights = 1.0 - alpha_weight * logistic_value
    if np.isscalar(strength):
        return float(weights)
    return weights


def compute_interaction_weights(
    strengths: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    equality_mask: Optional[np.ndarray] = None,
    alpha_weight: float = 0.9,
    beta_weight: float = 100.0,
) -> np.ndarray:
    """Compute weights with optional equality and validity masks.

    Parameters
    ----------
    strengths:
        Local interaction strengths with shape ``(m, m, n_points, n_constraints)``.
    valid_mask:
        Boolean mask with the same shape as ``strengths``. Invalid entries get
        zero weight.
    equality_mask:
        Optional boolean vector of length ``n_constraints``. Equality-constraint
        interactions are assigned weight 1 rather than logistic weights.
    """

    s = np.asarray(strengths, dtype=float)
    weights = np.asarray(
        logistic_competition_weight(s, alpha_weight=alpha_weight, beta_weight=beta_weight),
        dtype=float,
    )

    if equality_mask is not None:
        eq = np.asarray(equality_mask, dtype=bool)
        if eq.ndim != 1 or eq.shape[0] != s.shape[-1]:
            raise ValueError("equality_mask must be a 1D array matching the final strengths axis")
        weights[..., eq] = 1.0

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != s.shape:
            raise ValueError("valid_mask must have the same shape as strengths")
        weights = np.where(mask, weights, 0.0)

    return weights
