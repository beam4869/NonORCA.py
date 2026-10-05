"""Shared dataclasses that define the data passed through ORCA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class LinearMatrixBlocks:
    """Matrix representation of a linear ORCA problem.

    Attributes
    ----------
    Jineq:
        Inequality coefficient matrix. Each row is a constraint normal ``a_k``
        for an inequality standardized as ``a_k^T x <= b_k``.
    Jobj:
        Objective coefficient matrix. Each row is an objective vector ``c_i``
        for a linear objective ``f_i(x) = c_i^T x``.
    Jeq:
        Optional equality coefficient matrix. Each row is an equality normal.
    var_names:
        Optional variable names corresponding to the matrix columns.
    """

    Jineq: np.ndarray
    Jobj: np.ndarray
    Jeq: Optional[np.ndarray] = None
    var_names: Optional[list[str]] = None


@dataclass
class SampledGradientBlocks:
    """Sampled local-gradient representation of a nonlinear ORCA problem.

    Attributes
    ----------
    points:
        Selected feasible fixed points, with shape ``(num_points, num_vars)``.
    objective_gradients:
        Objective gradients evaluated at selected points, with shape
        ``(num_points, num_objectives, num_vars)``.
    constraint_jacobians:
        Constraint Jacobians evaluated at selected points, with shape
        ``(num_points, num_constraints, num_vars)``. Each row corresponds to
        ``grad g_k(x_n)`` for inequalities ``g_k(x) <= 0``.
    constraint_values:
        Optional constraint values at selected points, with shape
        ``(num_points, num_constraints)``. Useful for feasibility checks and
        active-constraint filtering.
    """

    points: np.ndarray
    objective_gradients: np.ndarray
    constraint_jacobians: np.ndarray
    constraint_values: Optional[np.ndarray] = None


@dataclass
class LocalLinearizationBlocks:
    """Explicit first-order local linearizations at selected points.

    For each sampled point ``x_n``, a nonlinear function ``h(x)`` is approximated
    as ``h(x) ~= grad_h(x_n)^T x + intercept`` where
    ``intercept = h(x_n) - grad_h(x_n)^T x_n``.
    """

    points: np.ndarray
    objective_gradients: np.ndarray
    objective_intercepts: np.ndarray
    constraint_jacobians: np.ndarray
    constraint_intercepts: np.ndarray
    constraint_values: Optional[np.ndarray] = None


@dataclass
class ORCAInteractionData:
    """Local interaction quantities computed before final aggregation.

    Attributes
    ----------
    strengths:
        Local pairwise objective interaction strengths. The shared convention
        is ``strengths[i, j, n, k]`` where ``i`` and ``j`` are objective indices,
        ``n`` is a selected point index, and ``k`` is a constraint index. Linear
        ORCA uses ``n = 0``.
    weights:
        Weights applied to local interaction strengths. Same shape as
        ``strengths``.
    total_weights:
        Pairwise total weights after summing over all sampled points and
        constraints. Shape is ``(num_objectives, num_objectives)``.
    valid_mask:
        Boolean mask indicating which local interactions were included.
    projected_directions:
        Optional local directions used to compute strengths. This can be useful
        for debugging but may be memory-heavy.
    metadata:
        Optional implementation details, such as filtering rules and tolerance.
    """

    strengths: np.ndarray
    weights: np.ndarray
    total_weights: np.ndarray
    valid_mask: Optional[np.ndarray] = None
    projected_directions: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ORCAResult:
    """Final output of an ORCA run.

    Attributes
    ----------
    adj_matrix:
        Objective correlation adjacency matrix. Values near 1 indicate strong
        correlation; values near 0 indicate strong competition.
    groups:
        Objective group labels returned by the grouping method.
    interaction_data:
        Optional local interaction strengths, weights, and diagnostics.
    input_data:
        Optional input data object, such as ``LinearMatrixBlocks`` or
        ``SampledGradientBlocks``.
    metadata:
        Hyperparameters and run information.
    """

    adj_matrix: np.ndarray
    groups: np.ndarray
    interaction_data: Optional[ORCAInteractionData] = None
    input_data: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
