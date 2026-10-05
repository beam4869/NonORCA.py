"""Local objective-interaction calculations for linear and nonlinear ORCA."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from orca.utils.constraint_projection import (
    modified_nonlinear_direction,
    points_outward,
    project_onto_constraint_surface,
)
from orca.utils.vector_normalization import safe_normalize


def _empty_matrix(num_columns: int) -> np.ndarray:
    return np.zeros((0, num_columns), dtype=float)


def _pairwise_dot_strengths(local_dirs: np.ndarray, valid_constraints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise dot products from normalized local directions.

    Parameters
    ----------
    local_dirs:
        Array with shape ``(num_points, num_constraints, num_objectives, num_vars)``.
    valid_constraints:
        Boolean mask with shape ``(num_points, num_constraints)``.
    """

    n_points, n_constraints, n_objectives, _ = local_dirs.shape
    strengths = np.zeros((n_objectives, n_objectives, n_points, n_constraints), dtype=float)
    valid_mask = np.zeros_like(strengths, dtype=bool)

    for i in range(n_objectives):
        strengths[i, i, :, :] = 1.0
        valid_mask[i, i, :, :] = valid_constraints
        for j in range(i + 1, n_objectives):
            dots = np.sum(local_dirs[:, :, i, :] * local_dirs[:, :, j, :], axis=-1)
            strengths[i, j, :, :] = strengths[j, i, :, :] = dots
            valid_mask[i, j, :, :] = valid_mask[j, i, :, :] = valid_constraints

    return strengths, valid_mask


def compute_linear_local_objective_interactions(
    objective_matrix: np.ndarray,
    inequality_matrix: np.ndarray,
    equality_matrix: Optional[np.ndarray] = None,
    *,
    filter_inactive_inequalities: bool = True,
    atol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute linear ORCA local projected-cosine interactions.

    Parameters
    ----------
    objective_matrix:
        Matrix whose rows are objective coefficient vectors ``c_i``.
    inequality_matrix:
        Matrix whose rows are inequality normals ``a_k``.
    equality_matrix:
        Optional matrix whose rows are equality normals.

    Returns
    -------
    strengths, valid_mask, metadata:
        ``strengths`` and ``valid_mask`` use the shared shape
        ``(num_objectives, num_objectives, 1, num_constraints_total)``.
        The final constraint axis concatenates inequalities followed by
        equalities. ``metadata["equality_mask"]`` identifies equality entries.
    """

    Jobj = np.asarray(objective_matrix, dtype=float)
    Jineq = np.asarray(inequality_matrix, dtype=float)
    if Jobj.ndim != 2:
        raise ValueError("objective_matrix must be two-dimensional")
    num_objectives, num_vars = Jobj.shape

    if Jineq.size == 0:
        Jineq = _empty_matrix(num_vars)
    if Jineq.ndim != 2 or Jineq.shape[1] != num_vars:
        raise ValueError("inequality_matrix must have the same number of columns as objective_matrix")

    if equality_matrix is None or np.asarray(equality_matrix).size == 0:
        Jeq = _empty_matrix(num_vars)
    else:
        Jeq = np.asarray(equality_matrix, dtype=float)
        if Jeq.ndim != 2 or Jeq.shape[1] != num_vars:
            raise ValueError("equality_matrix must have the same number of columns as objective_matrix")

    normals = np.vstack([Jineq, Jeq]) if Jeq.size else Jineq.copy()
    num_ineq = Jineq.shape[0]
    num_constraints = normals.shape[0]
    equality_mask = np.zeros(num_constraints, dtype=bool)
    if Jeq.shape[0] > 0:
        equality_mask[num_ineq:] = True

    # Linear objective descent directions are constant: d_i = -c_i.
    descent_dirs = np.vstack([safe_normalize(-row, atol=atol) for row in Jobj])

    local_dirs = np.zeros((1, num_constraints, num_objectives, num_vars), dtype=float)
    valid_constraints_by_pair = np.zeros((num_objectives, num_objectives, 1, num_constraints), dtype=bool)
    strengths = np.zeros_like(valid_constraints_by_pair, dtype=float)

    for k, normal in enumerate(normals):
        if np.linalg.norm(normal) <= atol:
            continue

        projected_unit_dirs = np.zeros((num_objectives, num_vars), dtype=float)
        normal_parts = np.zeros((num_objectives, num_vars), dtype=float)
        for i in range(num_objectives):
            projected, n_part = project_onto_constraint_surface(descent_dirs[i, :], normal, atol=atol)
            projected_unit_dirs[i, :] = safe_normalize(projected, atol=atol)
            normal_parts[i, :] = n_part
            local_dirs[0, k, i, :] = projected_unit_dirs[i, :]

        for i in range(num_objectives):
            strengths[i, i, 0, k] = 1.0
            valid_constraints_by_pair[i, i, 0, k] = True
            for j in range(i + 1, num_objectives):
                if equality_mask[k]:
                    include = True
                elif filter_inactive_inequalities:
                    include = points_outward(normal_parts[i, :], normal, tol=0.0) and points_outward(
                        normal_parts[j, :], normal, tol=0.0
                    )
                else:
                    include = True

                if include:
                    val = float(np.dot(projected_unit_dirs[i, :], projected_unit_dirs[j, :]))
                    strengths[i, j, 0, k] = strengths[j, i, 0, k] = val
                    valid_constraints_by_pair[i, j, 0, k] = valid_constraints_by_pair[j, i, 0, k] = True

    metadata = {
        "equality_mask": equality_mask,
        "num_inequality_constraints": int(num_ineq),
        "num_equality_constraints": int(Jeq.shape[0]),
        "filter_inactive_inequalities": bool(filter_inactive_inequalities),
        "projected_directions": local_dirs,
    }
    return strengths, valid_constraints_by_pair, metadata


def compute_nonlinear_local_objective_interactions(
    objective_gradients: np.ndarray,
    constraint_jacobians: np.ndarray,
    *,
    constraint_values: Optional[np.ndarray] = None,
    active_constraint_tolerance: Optional[float] = None,
    atol: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute nonlinear ORCA local objective interactions.

    Parameters
    ----------
    objective_gradients:
        Array with shape ``(num_points, num_objectives, num_vars)`` containing
        ``grad f_i(x_n)``.
    constraint_jacobians:
        Array with shape ``(num_points, num_constraints, num_vars)`` containing
        ``grad g_k(x_n)`` for inequalities ``g_k(x) <= 0``.
    constraint_values:
        Optional array with shape ``(num_points, num_constraints)`` containing
        ``g_k(x_n)``.
    active_constraint_tolerance:
        If provided, only constraints with ``g_k(x_n) >= -tol`` are included.
        If ``None``, every nonzero constraint Jacobian row is included.

    Returns
    -------
    strengths, valid_mask, metadata:
        ``strengths[i, j, n, k]`` is the nonlinear ORCA local interaction
        strength between objectives ``i`` and ``j`` at point ``n`` on constraint
        ``k``.
    """

    obj_grads = np.asarray(objective_gradients, dtype=float)
    con_jacs = np.asarray(constraint_jacobians, dtype=float)

    if obj_grads.ndim != 3:
        raise ValueError("objective_gradients must have shape (num_points, num_objectives, num_vars)")
    if con_jacs.ndim != 3:
        raise ValueError("constraint_jacobians must have shape (num_points, num_constraints, num_vars)")
    if obj_grads.shape[0] != con_jacs.shape[0] or obj_grads.shape[2] != con_jacs.shape[2]:
        raise ValueError("objective_gradients and constraint_jacobians have incompatible shapes")

    n_points, n_objectives, n_vars = obj_grads.shape
    _, n_constraints, _ = con_jacs.shape

    if constraint_values is not None:
        con_vals = np.asarray(constraint_values, dtype=float)
        if con_vals.shape != (n_points, n_constraints):
            raise ValueError("constraint_values must have shape (num_points, num_constraints)")
    else:
        con_vals = None

    # Nonlinear objective descent directions are local: d_i(x_n) = -grad f_i(x_n).
    descent_dirs = np.zeros_like(obj_grads, dtype=float)
    for n in range(n_points):
        for i in range(n_objectives):
            descent_dirs[n, i, :] = safe_normalize(-obj_grads[n, i, :], atol=atol)

    if active_constraint_tolerance is not None:
        if con_vals is None:
            raise ValueError("constraint_values are required when active_constraint_tolerance is set")
        active_pairs = [
            (n, k)
            for n in range(n_points)
            for k in range(n_constraints)
            if con_vals[n, k] >= -active_constraint_tolerance and np.linalg.norm(con_jacs[n, k, :]) > atol
        ]
        if not active_pairs:
            strengths = np.zeros((n_objectives, n_objectives, 1, 1), dtype=float)
            valid_mask = np.zeros_like(strengths, dtype=bool)
            metadata = {
                "active_constraint_tolerance": active_constraint_tolerance,
                "projected_directions": np.zeros((0, 1, n_objectives, n_vars), dtype=float),
                "num_points": int(n_points),
                "num_constraints": int(n_constraints),
                "compact_active_constraints": True,
                "num_active_constraint_pairs": 0,
            }
            return strengths, valid_mask, metadata

        local_dirs = np.zeros((len(active_pairs), 1, n_objectives, n_vars), dtype=float)
        valid_constraints = np.ones((len(active_pairs), 1), dtype=bool)
        for pair_idx, (n, k) in enumerate(active_pairs):
            normal = con_jacs[n, k, :]
            for i in range(n_objectives):
                local_dirs[pair_idx, 0, i, :] = modified_nonlinear_direction(
                    descent_dirs[n, i, :], normal, atol=atol
                )

        strengths, valid_mask = _pairwise_dot_strengths(local_dirs, valid_constraints)
        metadata = {
            "active_constraint_tolerance": active_constraint_tolerance,
            "projected_directions": local_dirs,
            "num_points": int(n_points),
            "num_constraints": int(n_constraints),
            "compact_active_constraints": True,
            "num_active_constraint_pairs": int(len(active_pairs)),
            "active_constraint_pairs": active_pairs,
        }
        return strengths, valid_mask, metadata

    local_dirs = np.zeros((n_points, n_constraints, n_objectives, n_vars), dtype=float)
    valid_constraints = np.ones((n_points, n_constraints), dtype=bool)

    for n in range(n_points):
        for k in range(n_constraints):
            normal = con_jacs[n, k, :]
            if np.linalg.norm(normal) <= atol:
                valid_constraints[n, k] = False
                continue
            if active_constraint_tolerance is not None:
                if con_vals is None:
                    raise ValueError("constraint_values are required when active_constraint_tolerance is set")
                if con_vals[n, k] < -active_constraint_tolerance:
                    valid_constraints[n, k] = False
                    continue
            for i in range(n_objectives):
                local_dirs[n, k, i, :] = modified_nonlinear_direction(
                    descent_dirs[n, i, :], normal, atol=atol
                )

    strengths, valid_mask = _pairwise_dot_strengths(local_dirs, valid_constraints)
    metadata = {
        "active_constraint_tolerance": active_constraint_tolerance,
        "projected_directions": local_dirs,
        "num_points": int(n_points),
        "num_constraints": int(n_constraints),
    }
    return strengths, valid_mask, metadata
