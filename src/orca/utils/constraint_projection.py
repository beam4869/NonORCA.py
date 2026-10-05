"""Projection utilities for objective directions and constraint surfaces."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from orca.utils.vector_normalization import safe_normalize


def normal_component(direction: np.ndarray, normal: np.ndarray, atol: float = 1.0e-14) -> np.ndarray:
    """Return the component of ``direction`` parallel to ``normal``.

    Mathematically, for direction ``d`` and constraint normal ``a``, this is

        d_N = (d^T a / a^T a) a.
    """

    d = np.asarray(direction, dtype=float)
    a = np.asarray(normal, dtype=float)
    denom = float(np.dot(a, a))
    if denom <= atol:
        return np.zeros_like(d, dtype=float)
    return (float(np.dot(d, a)) / denom) * a


def tangent_component(direction: np.ndarray, normal: np.ndarray, atol: float = 1.0e-14) -> np.ndarray:
    """Return the component of ``direction`` tangent to a constraint surface."""

    d = np.asarray(direction, dtype=float)
    return d - normal_component(d, normal, atol=atol)


def project_onto_constraint_surface(
    direction: np.ndarray,
    normal: np.ndarray,
    atol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a direction into tangent and normal components.

    Returns
    -------
    projected, normal_part:
        ``projected`` is the tangent component ``d_P`` and ``normal_part`` is
        the normal component ``d_N``.
    """

    n_part = normal_component(direction, normal, atol=atol)
    return np.asarray(direction, dtype=float) - n_part, n_part


def points_outward(direction: np.ndarray, normal: np.ndarray, tol: float = 0.0) -> bool:
    """Return true if ``direction`` points toward the infeasible side.

    For an inequality ``g(x) <= 0`` with outward normal ``a = grad g(x)``, a
    local movement ``d`` points outward when ``a^T d > 0``.
    """

    return float(np.dot(np.asarray(normal, dtype=float), np.asarray(direction, dtype=float))) > tol


def modified_nonlinear_direction(
    direction: np.ndarray,
    normal: np.ndarray,
    atol: float = 1.0e-14,
) -> np.ndarray:
    """Return the nonlinear ORCA local direction for one objective-constraint pair.

    If the objective descent direction points outward, the tangent projection is
    used. If it points inward or tangentially, the original descent direction is
    retained. The returned vector is normalized.
    """

    d = np.asarray(direction, dtype=float)
    a = np.asarray(normal, dtype=float)
    if np.linalg.norm(a) <= atol:
        return np.zeros_like(d, dtype=float)
    if points_outward(d, a, tol=0.0):
        return safe_normalize(tangent_component(d, a, atol=atol), atol=atol)
    return safe_normalize(d, atol=atol)
