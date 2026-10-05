"""Reusable utilities for ORCA-style objective correlation analysis in Python.

This module translates the core logic of the uploaded Julia code:
- linear_jac.jl: extract linear constraint/objective coefficient matrices
- correlation_strength.jl: compute projection-based objective correlations

The matrix-based functions are independent of any modeling package. The Pyomo
extractor is optional and only required when extracting matrices directly from a
Pyomo ConcreteModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import igraph as ig  # type: ignore
import leidenalg  # type: ignore

from orca.legacy.orca_constants import (
    CORRELATION_RESCALING_FACTOR,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_FILTER_INACTIVE_INEQUALITIES,
    DEFAULT_GROUPING_METHOD,
    DEFAULT_NUM_GROUPS,
    DEFAULT_NUM_SELECTED_POINTS,
    DEFAULT_ZERO_WEIGHT_CORRELATION,
    DEFAULT_RESOLUTION_START,
    DEFAULT_RESOLUTION_STEPS,
    DEFAULT_RESOLUTION_STOP,
    EXPONENT_CLIP_LIMIT,
    GROUP_LABEL_START,
    LEIDEN_GRAPH_MODE,
    LEIDEN_WEIGHT_ATTRIBUTE,
    MAX_CORRELATION,
    MIN_CORRELATION,
    MIN_GROUP_COUNT,
    NEGATIVE_INFINITY,
    NUMERIC_ATOL,
    SELF_CORRELATION,
)

@dataclass
class LinearBlocks:
    """Linear matrix blocks extracted from an optimization model.

    Attributes
    ----------
    Jeq:
        Equality constraint Jacobian rows.
    Jineq:
        Inequality constraint Jacobian rows, standardized as <= 0.
    Jobj:
        Objective coefficient or gradient rows. One row per objective.
    var_names:
        Column order used by all matrices.
    """

    Jeq: np.ndarray
    Jineq: np.ndarray
    Jobj: np.ndarray
    var_names: List[str]


@dataclass
class ORCAResults:
    """Results from objective correlation analysis."""

    adj_matrix: np.ndarray
    groups: np.ndarray
    total_weights: np.ndarray
    total_ineq_weights: np.ndarray
    ineq_strength: np.ndarray
    eq_strength: np.ndarray


def _safe_normalize(v: np.ndarray, atol: float = NUMERIC_ATOL) -> np.ndarray:
    """Return v / ||v||, or a zero vector if v is numerically zero."""

    v = np.asarray(v, dtype=float)
    nrm = np.linalg.norm(v)
    if nrm <= atol:
        return np.zeros_like(v, dtype=float)
    return v / nrm


def _empty_matrix(num_columns: int) -> np.ndarray:
    """Return an empty two-dimensional matrix with the requested column count."""

    return np.zeros((0, num_columns), dtype=float)


def project_onto_constraint_surface(
    direction: np.ndarray,
    normal: np.ndarray,
    *,
    atol: float = NUMERIC_ATOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an objective descent direction onto a constraint surface.

    Parameters
    ----------
    direction:
        Normalized descent direction of an objective, corresponding to -c_i.
    normal:
        Constraint normal vector a_k.
    atol:
        Absolute tolerance used to identify zero normal vectors.

    Returns
    -------
    projected, normal_component:
        projected is the component along the constraint surface;
        normal_component is the component parallel to the constraint normal.
    """

    direction = np.asarray(direction, dtype=float)
    normal = np.asarray(normal, dtype=float)
    denom = float(np.dot(normal, normal))
    if denom <= atol:
        return np.zeros_like(direction), np.zeros_like(direction)
    normal_component = (np.dot(direction, normal) / denom) * normal
    projected = direction - normal_component
    return projected, normal_component


def logistic_weight(
    strength: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """Weight a constraint-level interaction strength.

    This follows the uploaded Julia implementation:
        W = 1 - alpha / (1 + exp(-beta * S))

    Negative strengths receive larger weights, so competition is emphasized.
    """

    clipped_exponent = float(np.clip(-beta * strength, -EXPONENT_CLIP_LIMIT, EXPONENT_CLIP_LIMIT))
    logistic_value = SELF_CORRELATION / (SELF_CORRELATION + np.exp(clipped_exponent))
    return float(SELF_CORRELATION - alpha * logistic_value)


def correlation_strength_matrix(
    ineq_matrix: np.ndarray,
    obj_matrix: np.ndarray,
    eq_matrix: Optional[np.ndarray] = None,
    *,
    num_selected_points: int = DEFAULT_NUM_SELECTED_POINTS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    filter_inactive_inequalities: bool = True,
    zero_weight_value: float = DEFAULT_ZERO_WEIGHT_CORRELATION,
    atol: float = NUMERIC_ATOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ORCA objective correlation adjacency matrix.

    Parameters
    ----------
    ineq_matrix:
        Inequality constraint Jacobian rows, standardized as <= 0.
    obj_matrix:
        Objective coefficient or gradient rows. If num_selected_points > 1,
        rows should be ordered as obj1_sp1, obj1_sp2, ..., obj2_sp1, ...,
        matching the uploaded Julia code.
    eq_matrix:
        Equality constraint Jacobian rows. If None, treated as empty.
    num_selected_points:
        Number of sampled points per objective. For linear models, use 1.
    alpha, beta:
        Logistic weighting parameters.
    filter_inactive_inequalities:
        If True, mimic the Julia code and ignore an inequality when the normal
        components do not pass the sign check. If False, include all inequality
        constraint surfaces.
    zero_weight_value:
        Correlation assigned if a pair has zero total weight. 0.5 means neutral.
    atol:
        Absolute tolerance used for zero checks.

    Returns
    -------
    adj_matrix, total_weights, total_ineq_weights, ineq_strength, eq_strength
    """

    ineq_matrix = np.asarray(ineq_matrix, dtype=float)
    obj_matrix = np.asarray(obj_matrix, dtype=float)
    if eq_matrix is None:
        eq_matrix = _empty_matrix(obj_matrix.shape[1])
    else:
        eq_matrix = np.asarray(eq_matrix, dtype=float)

    if obj_matrix.ndim != 2:
        raise ValueError("obj_matrix must be a 2D array")
    if num_selected_points <= 0:
        raise ValueError("num_selected_points must be positive")
    if obj_matrix.shape[0] % num_selected_points != 0:
        raise ValueError("obj_matrix rows must be divisible by num_selected_points")

    num_objectives = obj_matrix.shape[0] // num_selected_points
    num_variables = obj_matrix.shape[1]

    if ineq_matrix.size == 0:
        ineq_matrix = _empty_matrix(num_variables)
    if eq_matrix.size == 0:
        eq_matrix = _empty_matrix(num_variables)
    if ineq_matrix.shape[1] != num_variables or eq_matrix.shape[1] != num_variables:
        raise ValueError("constraint matrices must have the same number of columns as obj_matrix")

    # Descent directions -normalize(c_i), matching the Julia code.
    normalized_objectives = np.vstack([_safe_normalize(-row, atol=atol) for row in obj_matrix])

    num_ineq_constraints = ineq_matrix.shape[0]
    num_eq_constraints = eq_matrix.shape[0]

    ineq_strength = np.zeros(
        (num_objectives, num_objectives, num_ineq_constraints, num_selected_points),
        dtype=float,
    )
    total_ineq_weights = np.zeros_like(ineq_strength)
    eq_strength = np.zeros(
        (num_objectives, num_objectives, num_eq_constraints, num_selected_points),
        dtype=float,
    )
    total_eq_weights = np.zeros_like(eq_strength)

    for obj_i in range(num_objectives):
        for obj_j in range(obj_i + 1, num_objectives):
            for constraint_idx in range(num_ineq_constraints):
                normal = ineq_matrix[constraint_idx, :]
                if np.linalg.norm(normal) <= atol:
                    continue
                for selected_point_idx in range(num_selected_points):
                    row_i = obj_i * num_selected_points + selected_point_idx
                    row_j = obj_j * num_selected_points + selected_point_idx
                    proj_i, normal_i = project_onto_constraint_surface(
                        normalized_objectives[row_i, :], normal, atol=atol
                    )
                    proj_j, normal_j = project_onto_constraint_surface(
                        normalized_objectives[row_j, :], normal, atol=atol
                    )
                    proj_i_unit = _safe_normalize(proj_i, atol=atol)
                    proj_j_unit = _safe_normalize(proj_j, atol=atol)
                    strength = float(np.dot(proj_i_unit, proj_j_unit))

                    include_constraint = True
                    if filter_inactive_inequalities:
                        # This reproduces the uploaded Julia sign check:
                        # if dot(IneqMatrix[k,:], cn1) > 0 && dot(IneqMatrix[k,:], cn2) > 0
                        include_constraint = (
                            np.dot(normal, normal_i) > MIN_CORRELATION
                            and np.dot(normal, normal_j) > MIN_CORRELATION
                        )

                    if include_constraint:
                        ineq_strength[obj_i, obj_j, constraint_idx, selected_point_idx] = strength
                        total_ineq_weights[obj_i, obj_j, constraint_idx, selected_point_idx] = logistic_weight(
                            strength, alpha=alpha, beta=beta
                        )

            for constraint_idx in range(num_eq_constraints):
                normal = eq_matrix[constraint_idx, :]
                if np.linalg.norm(normal) <= atol:
                    continue
                for selected_point_idx in range(num_selected_points):
                    row_i = obj_i * num_selected_points + selected_point_idx
                    row_j = obj_j * num_selected_points + selected_point_idx
                    proj_i, _ = project_onto_constraint_surface(
                        normalized_objectives[row_i, :], normal, atol=atol
                    )
                    proj_j, _ = project_onto_constraint_surface(
                        normalized_objectives[row_j, :], normal, atol=atol
                    )
                    strength = float(
                        np.dot(_safe_normalize(proj_i, atol=atol), _safe_normalize(proj_j, atol=atol))
                    )
                    eq_strength[obj_i, obj_j, constraint_idx, selected_point_idx] = strength
                    total_eq_weights[obj_i, obj_j, constraint_idx, selected_point_idx] = SELF_CORRELATION

    total_weights = np.zeros((num_objectives, num_objectives), dtype=float)
    weighted_strengths = np.zeros((num_objectives, num_objectives), dtype=float)

    for obj_i in range(num_objectives):
        for obj_j in range(obj_i + 1, num_objectives):
            weight_sum = (
                total_ineq_weights[obj_i, obj_j, :, :].sum()
                + total_eq_weights[obj_i, obj_j, :, :].sum()
            )
            weighted_strength_sum = (
                (total_ineq_weights[obj_i, obj_j, :, :] * ineq_strength[obj_i, obj_j, :, :]).sum()
                + (total_eq_weights[obj_i, obj_j, :, :] * eq_strength[obj_i, obj_j, :, :]).sum()
            )
            total_weights[obj_i, obj_j] = total_weights[obj_j, obj_i] = weight_sum
            weighted_strengths[obj_i, obj_j] = weighted_strengths[obj_j, obj_i] = weighted_strength_sum

    adj_matrix = np.eye(num_objectives, dtype=float)
    for obj_i in range(num_objectives):
        for obj_j in range(obj_i + 1, num_objectives):
            if total_weights[obj_i, obj_j] <= atol:
                correlation_value = zero_weight_value
            else:
                correlation_value = CORRELATION_RESCALING_FACTOR * (
                    SELF_CORRELATION + weighted_strengths[obj_i, obj_j] / total_weights[obj_i, obj_j]
                )
            adj_matrix[obj_i, obj_j] = adj_matrix[obj_j, obj_i] = float(
                np.clip(correlation_value, MIN_CORRELATION, MAX_CORRELATION)
            )

    return adj_matrix, total_weights, total_ineq_weights, ineq_strength, eq_strength


def group_objectives(
    adj_matrix: np.ndarray,
    num_groups: int = DEFAULT_NUM_GROUPS,
    *,
    method: str = DEFAULT_GROUPING_METHOD,
    resolution_grid: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Group objectives from an adjacency matrix.

    method="auto" tries Python Leiden packages first. If they are unavailable,
    it falls back to a deterministic agglomerative grouping based on average
    inter-group similarity. The fallback does not exactly reproduce Leiden, but
    it provides a dependency-light grouping with the requested number of groups.
    """

    adj = np.asarray(adj_matrix, dtype=float)
    num_nodes = adj.shape[0]
    if adj.shape != (num_nodes, num_nodes):
        raise ValueError("adj_matrix must be square")
    if num_groups < MIN_GROUP_COUNT or num_groups > num_nodes:
        raise ValueError("num_groups must be between 1 and the number of objectives")
    if num_groups == num_nodes:
        return np.arange(GROUP_LABEL_START, num_nodes + GROUP_LABEL_START, dtype=int)
    if num_groups == GROUP_LABEL_START:
        return np.ones(num_nodes, dtype=int)

    if method in {DEFAULT_GROUPING_METHOD, "leiden"}:
        try:

            graph = ig.Graph.Weighted_Adjacency(
                adj.tolist(), mode=LEIDEN_GRAPH_MODE, attr=LEIDEN_WEIGHT_ATTRIBUTE, loops=False
            )
            if resolution_grid is None:
                resolution_grid = np.linspace(
                    DEFAULT_RESOLUTION_START,
                    DEFAULT_RESOLUTION_STOP,
                    DEFAULT_RESOLUTION_STEPS,
                )

            best_membership = None
            best_gap = num_nodes + GROUP_LABEL_START
            for resolution in resolution_grid:
                partition = leidenalg.find_partition(
                    graph,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=LEIDEN_WEIGHT_ATTRIBUTE,
                    resolution_parameter=float(resolution),
                )
                membership = np.asarray(partition.membership, dtype=int)
                found_groups = len(np.unique(membership))
                gap = abs(found_groups - num_groups)
                if gap < best_gap:
                    best_gap = gap
                    best_membership = membership
                if found_groups == num_groups:
                    break
            if best_membership is not None:
                return _relabel_groups(best_membership)
        except ImportError:
            if method == "leiden":
                raise ImportError("Install python-igraph and leidenalg for Leiden grouping")

    # Fallback: average-linkage agglomerative clustering by similarity.
    groups = [{node_idx} for node_idx in range(num_nodes)]
    while len(groups) > num_groups:
        best_pair = None
        best_similarity = NEGATIVE_INFINITY
        for group_a in range(len(groups)):
            for group_b in range(group_a + 1, len(groups)):
                similarities = [adj[i, j] for i in groups[group_a] for j in groups[group_b]]
                similarity = float(np.mean(similarities)) if similarities else NEGATIVE_INFINITY
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (group_a, group_b)
        assert best_pair is not None
        group_a, group_b = best_pair
        groups[group_a] = groups[group_a].union(groups[group_b])
        del groups[group_b]

    labels = np.zeros(num_nodes, dtype=int)
    for label, group in enumerate(groups, start=GROUP_LABEL_START):
        for node_idx in sorted(group):
            labels[node_idx] = label
    return _relabel_groups(labels)


def _relabel_groups(labels: np.ndarray) -> np.ndarray:
    """Relabel arbitrary community labels to 1, 2, ... in order of first appearance."""

    mapping: dict[int, int] = {}
    out = np.zeros_like(labels, dtype=int)
    next_label = GROUP_LABEL_START
    for idx, label in enumerate(labels.tolist()):
        if label not in mapping:
            mapping[label] = next_label
            next_label += GROUP_LABEL_START
        out[idx] = mapping[label]
    return out


def orca_from_matrices(
    Jineq: np.ndarray,
    Jobj: np.ndarray,
    Jeq: Optional[np.ndarray] = None,
    *,
    num_groups: int = DEFAULT_NUM_GROUPS,
    num_selected_points: int = DEFAULT_NUM_SELECTED_POINTS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    grouping_method: str = DEFAULT_GROUPING_METHOD,
    filter_inactive_inequalities: bool = True,
    zero_weight_value: float = DEFAULT_ZERO_WEIGHT_CORRELATION,
    atol: float = NUMERIC_ATOL,
) -> ORCAResults:
    """Run the full ORCA workflow from precomputed matrices."""

    adj, total_weights, total_ineq_weights, ineq_strength, eq_strength = correlation_strength_matrix(
        Jineq,
        Jobj,
        Jeq,
        num_selected_points=num_selected_points,
        alpha=alpha,
        beta=beta,
        filter_inactive_inequalities=filter_inactive_inequalities,
        zero_weight_value=zero_weight_value,
        atol=atol,
    )
    groups = group_objectives(adj, num_groups=num_groups, method=grouping_method)
    return ORCAResults(
        adj_matrix=adj,
        groups=groups,
        total_weights=total_weights,
        total_ineq_weights=total_ineq_weights,
        ineq_strength=ineq_strength,
        eq_strength=eq_strength,
    )


def extract_linear_blocks_pyomo(model, objective_exprs: Sequence) -> LinearBlocks:
    """Extract linear blocks from a Pyomo model.

    This is the Python analog of linear_jac.extract_linear_blocks in the Julia code.
    It requires Pyomo and assumes the constraints/objectives passed here are linear.

    Inequalities are standardized as <= 0:
        body <= upper        ->  grad(body)
        lower <= body        -> -grad(body)
        lower <= body <= upper -> both rows

    Equalities are stored in Jeq.
    """

    try:
        import pyomo.environ as pyo  # type: ignore
        from pyomo.repn.standard_repn import generate_standard_repn  # type: ignore
    except ImportError as exc:
        raise ImportError("extract_linear_blocks_pyomo requires Pyomo") from exc

    var_list = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    var_to_idx = {id(var): idx for idx, var in enumerate(var_list)}
    var_names = [var.name for var in var_list]
    num_variables = len(var_list)

    def row_from_expr(expr) -> np.ndarray:
        repn = generate_standard_repn(expr, compute_values=False)
        if repn.nonlinear_expr is not None or repn.quadratic_vars:
            raise ValueError(f"Expression is not linear: {expr}")
        row = np.zeros(num_variables, dtype=float)
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            idx = var_to_idx.get(id(var))
            if idx is not None:
                row[idx] += float(coef)
        return row

    eq_rows: List[np.ndarray] = []
    ineq_rows: List[np.ndarray] = []

    for constraint in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        if constraint.body is None:
            continue
        body_row = row_from_expr(constraint.body)
        has_lower_bound = constraint.lower is not None
        has_upper_bound = constraint.upper is not None

        if constraint.equality:
            eq_rows.append(body_row)
        else:
            if has_upper_bound:
                ineq_rows.append(body_row)
            if has_lower_bound:
                ineq_rows.append(-body_row)

    obj_rows = [row_from_expr(expr) for expr in objective_exprs]

    Jeq = np.vstack(eq_rows) if eq_rows else _empty_matrix(num_variables)
    Jineq = np.vstack(ineq_rows) if ineq_rows else _empty_matrix(num_variables)
    Jobj = np.vstack(obj_rows) if obj_rows else _empty_matrix(num_variables)
    return LinearBlocks(Jeq=Jeq, Jineq=Jineq, Jobj=Jobj, var_names=var_names)
