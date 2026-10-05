"""Gradient and Jacobian evaluation for nonlinear ORCA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orca.data_schema import SampledGradientBlocks
from orca.nonlinear.problem_interface import NonlinearORCAProblem


def evaluate_objective_gradients(problem: NonlinearORCAProblem, points: np.ndarray) -> np.ndarray:
    """Evaluate objective gradients at selected points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (num_points, num_variables)")
    gradients = [np.asarray(problem.objective_gradients(x), dtype=float) for x in pts]
    return np.stack(gradients, axis=0)


def evaluate_constraint_jacobians(problem: NonlinearORCAProblem, points: np.ndarray) -> np.ndarray:
    """Evaluate constraint Jacobians at selected points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (num_points, num_variables)")
    jacobians = [np.asarray(problem.constraint_jacobian(x), dtype=float) for x in pts]
    return np.stack(jacobians, axis=0)


def evaluate_constraint_values(problem: NonlinearORCAProblem, points: np.ndarray) -> np.ndarray:
    """Evaluate inequality constraint values at selected points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (num_points, num_variables)")
    values = [np.asarray(problem.constraint_values(x), dtype=float) for x in pts]
    return np.stack(values, axis=0)


def evaluate_sampled_gradient_blocks(
    problem: NonlinearORCAProblem,
    points: Sequence[np.ndarray] | np.ndarray,
) -> SampledGradientBlocks:
    """Evaluate all local-gradient data required by nonlinear ORCA."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (num_points, num_variables)")
    obj_grads = evaluate_objective_gradients(problem, pts)
    con_jacs = evaluate_constraint_jacobians(problem, pts)
    con_vals = evaluate_constraint_values(problem, pts)
    return SampledGradientBlocks(
        points=pts,
        objective_gradients=obj_grads,
        constraint_jacobians=con_jacs,
        constraint_values=con_vals,
    )
