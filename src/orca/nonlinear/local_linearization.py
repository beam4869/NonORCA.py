"""Explicit first-order local linearization utilities for nonlinear ORCA."""

from __future__ import annotations

import numpy as np

from orca.data_schema import LocalLinearizationBlocks, SampledGradientBlocks
from orca.nonlinear.problem_interface import NonlinearORCAProblem


def linearize_from_sampled_gradients(
    problem: NonlinearORCAProblem,
    blocks: SampledGradientBlocks,
) -> LocalLinearizationBlocks:
    """Construct first-order affine approximations from sampled gradients.

    For each selected point ``x_n`` and function ``h``, this computes the
    intercept in

        h(x) ~= grad h(x_n)^T x + intercept,

    where

        intercept = h(x_n) - grad h(x_n)^T x_n.
    """

    points = np.asarray(blocks.points, dtype=float)
    obj_grads = np.asarray(blocks.objective_gradients, dtype=float)
    con_jacs = np.asarray(blocks.constraint_jacobians, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must have shape (num_points, num_variables)")
    if obj_grads.ndim != 3 or con_jacs.ndim != 3:
        raise ValueError("gradient arrays must be three-dimensional")
    if obj_grads.shape[0] != points.shape[0] or con_jacs.shape[0] != points.shape[0]:
        raise ValueError("gradient arrays must have one block per sampled point")

    objective_values = np.stack([np.asarray(problem.objective_values(x), dtype=float) for x in points], axis=0)
    constraint_values = (
        blocks.constraint_values
        if blocks.constraint_values is not None
        else np.stack([np.asarray(problem.constraint_values(x), dtype=float) for x in points], axis=0)
    )

    objective_intercepts = objective_values - np.einsum("nmp,np->nm", obj_grads, points)
    constraint_intercepts = constraint_values - np.einsum("nkp,np->nk", con_jacs, points)

    return LocalLinearizationBlocks(
        points=points,
        objective_gradients=obj_grads,
        objective_intercepts=objective_intercepts,
        constraint_jacobians=con_jacs,
        constraint_intercepts=constraint_intercepts,
        constraint_values=constraint_values,
    )


def linearize_problem_at_points(problem: NonlinearORCAProblem, points: np.ndarray) -> LocalLinearizationBlocks:
    """Evaluate sampled gradients and return explicit local linearizations."""

    from orca.nonlinear.gradient_evaluation import evaluate_sampled_gradient_blocks

    blocks = evaluate_sampled_gradient_blocks(problem, points)
    return linearize_from_sampled_gradients(problem, blocks)
