"""Baseline grouping comparisons for objective-reduction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np

from orca.config import NonlinearORCAConfig
from orca.nonlinear.main import nonlinear_orca
from orca.nonlinear.problem_interface import NonlinearORCAProblem
from orca.utils.objective_grouping import group_objectives


@dataclass(frozen=True)
class BaselineGroupingResult:
    """One grouping method's result against known labels."""

    method: str
    groups: np.ndarray
    ari: float
    adj_matrix: np.ndarray


def adjusted_rand_index(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    """Compute the adjusted Rand index without adding a scikit-learn dependency."""

    true = np.asarray(labels_true, dtype=int)
    pred = np.asarray(labels_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape")
    n = true.size
    if n < 2:
        return 1.0

    true_labels = np.unique(true)
    pred_labels = np.unique(pred)
    contingency = np.zeros((true_labels.size, pred_labels.size), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            contingency[i, j] = int(np.sum((true == true_label) & (pred == pred_label)))

    sum_comb = float(sum(comb(int(value), 2) for value in contingency.ravel()))
    sum_true = float(sum(comb(int(value), 2) for value in np.sum(contingency, axis=1)))
    sum_pred = float(sum(comb(int(value), 2) for value in np.sum(contingency, axis=0)))
    total = float(comb(n, 2))
    if total == 0.0:
        return 1.0
    expected = sum_true * sum_pred / total
    maximum = 0.5 * (sum_true + sum_pred)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0 if sum_comb == maximum else 0.0
    return float((sum_comb - expected) / denominator)


def sample_box_points(
    problem: NonlinearORCAProblem,
    *,
    num_points: int = 256,
    random_seed: int = 0,
) -> np.ndarray:
    """Sample feasible points by drawing in a known box and projecting."""

    rng = np.random.default_rng(random_seed)
    num_variables = problem.num_variables()
    lower = np.asarray(getattr(problem, "lower_bounds", np.zeros(num_variables)), dtype=float)
    upper = np.asarray(getattr(problem, "upper_bounds", np.ones(num_variables)), dtype=float)
    points = []
    for seed in problem.initial_points():
        points.append(problem.project_feasible(np.asarray(seed, dtype=float)))
    while len(points) < num_points:
        trial = rng.uniform(lower, upper)
        projected = problem.project_feasible(trial)
        if np.all(problem.constraint_values(projected) <= 1.0e-8):
            points.append(projected)
    return np.vstack(points[:num_points])


def objective_value_correlation_adjacency(
    problem: NonlinearORCAProblem,
    points: np.ndarray,
) -> np.ndarray:
    """Build a baseline adjacency from Pearson correlations of objective values."""

    values = np.vstack([problem.objective_values(point) for point in np.asarray(points, dtype=float)])
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (corr + 1.0)
    np.fill_diagonal(adj, 1.0)
    return np.clip(adj, 0.0, 1.0)


def gradient_cosine_adjacency(
    problem: NonlinearORCAProblem,
    points: np.ndarray,
    *,
    atol: float = 1.0e-14,
) -> np.ndarray:
    """Build a baseline adjacency from average cosine similarity of gradients."""

    gradients = np.asarray([problem.objective_gradients(point) for point in np.asarray(points, dtype=float)])
    num_objectives = problem.num_objectives()
    adj = np.eye(num_objectives, dtype=float)
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            sims = []
            for point_grads in gradients:
                gi = point_grads[i]
                gj = point_grads[j]
                denom = float(np.linalg.norm(gi) * np.linalg.norm(gj))
                if denom <= atol:
                    continue
                sims.append(float(np.dot(gi, gj) / denom))
            mean_sim = float(np.mean(sims)) if sims else 0.0
            adj[i, j] = adj[j, i] = 0.5 * (mean_sim + 1.0)
    return np.clip(adj, 0.0, 1.0)


def cyclic_random_groups(num_objectives: int, num_groups: int, *, random_seed: int = 0) -> np.ndarray:
    """Deterministic random-control labels with all requested groups represented."""

    rng = np.random.default_rng(random_seed)
    labels = (np.arange(num_objectives) % num_groups) + 1
    rng.shuffle(labels)
    return labels.astype(int)


def compare_grouping_methods(
    problem: NonlinearORCAProblem,
    expected_groups: Sequence[int],
    *,
    num_groups: int,
    points: np.ndarray | None = None,
    sample_points: int = 256,
    random_seed: int = 0,
    grouping_method: str = "average_linkage",
    orca_config: NonlinearORCAConfig | None = None,
) -> list[BaselineGroupingResult]:
    """Compare ORCA against simple objective-reduction baselines."""

    expected = np.asarray(expected_groups, dtype=int)
    if expected.shape != (problem.num_objectives(),):
        raise ValueError("expected_groups must have one label per objective")

    sampled_points = points if points is not None else sample_box_points(
        problem,
        num_points=sample_points,
        random_seed=random_seed,
    )
    cfg = orca_config or NonlinearORCAConfig(
        num_groups=num_groups,
        grouping_method=grouping_method,
        random_seed=random_seed,
        num_points_per_seed=3,
        include_seed_points=True,
    )
    orca_result = nonlinear_orca(problem, cfg)

    outputs: list[BaselineGroupingResult] = [
        BaselineGroupingResult(
            method="ORCA",
            groups=orca_result.groups,
            ari=adjusted_rand_index(expected, orca_result.groups),
            adj_matrix=orca_result.adj_matrix,
        )
    ]

    for method, adj in [
        ("objective_value_correlation", objective_value_correlation_adjacency(problem, sampled_points)),
        ("gradient_cosine", gradient_cosine_adjacency(problem, sampled_points)),
    ]:
        groups = group_objectives(adj, num_groups, method=grouping_method)
        outputs.append(
            BaselineGroupingResult(
                method=method,
                groups=groups,
                ari=adjusted_rand_index(expected, groups),
                adj_matrix=adj,
            )
        )

    random_groups = cyclic_random_groups(problem.num_objectives(), num_groups, random_seed=random_seed)
    outputs.append(
        BaselineGroupingResult(
            method="random_control",
            groups=random_groups,
            ari=adjusted_rand_index(expected, random_groups),
            adj_matrix=np.eye(problem.num_objectives(), dtype=float),
        )
    )
    return outputs
