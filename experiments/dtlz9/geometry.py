"""Geometry, aggregation, and grouping metrics for DTLZ9 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from . import orca_bridge as _orca_bridge  # noqa: F401
from orca.config import NonlinearORCAConfig
from orca.data_schema import SampledGradientBlocks
from orca.nonlinear.gradient_evaluation import evaluate_sampled_gradient_blocks
from orca.nonlinear.main import nonlinear_orca_from_sampled_gradients
from orca.utils.interaction_weights import logistic_competition_weight
from orca.utils.local_objective_interactions import compute_nonlinear_local_objective_interactions
from orca.utils.objective_grouping import group_objectives_average_linkage


@dataclass(frozen=True)
class ProjectorDiagnostics:
    projector: np.ndarray
    rank: int
    singular_values: np.ndarray
    row_norms: np.ndarray
    condition_number: float
    symmetry_residual: float
    idempotence_residual: float
    tangent_residual: float


def _relative_frobenius(matrix: np.ndarray, reference: np.ndarray | None = None) -> float:
    denom_source = matrix if reference is None else reference
    denom = max(float(np.linalg.norm(denom_source, ord="fro")), np.finfo(float).tiny)
    return float(np.linalg.norm(matrix, ord="fro") / denom)


def joint_tangent_projector(
    active_jacobian: np.ndarray,
    *,
    rtol: float | None = None,
    row_atol: float = 1.0e-300,
) -> ProjectorDiagnostics:
    """Project onto the nullspace of all active constraint rows using SVD."""

    jac = np.asarray(active_jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("active_jacobian must be two-dimensional")
    n_variables = jac.shape[1]
    if jac.shape[0] == 0:
        projector = np.eye(n_variables, dtype=float)
        return ProjectorDiagnostics(
            projector=projector,
            rank=0,
            singular_values=np.empty(0, dtype=float),
            row_norms=np.empty(0, dtype=float),
            condition_number=1.0,
            symmetry_residual=0.0,
            idempotence_residual=0.0,
            tangent_residual=0.0,
        )

    row_norms = np.linalg.norm(jac, axis=1)
    keep = row_norms > row_atol
    if not np.any(keep):
        raise ValueError("active_jacobian has no nonzero rows")
    normalized = jac[keep] / row_norms[keep, None]
    _, singular_values, vh = np.linalg.svd(normalized, full_matrices=False)
    if singular_values.size == 0:
        rank = 0
    else:
        threshold = (
            float(rtol) * singular_values[0]
            if rtol is not None
            else max(normalized.shape) * np.finfo(float).eps * singular_values[0]
        )
        rank = int(np.sum(singular_values > threshold))

    if rank:
        row_basis = vh[:rank]
        projector = np.eye(n_variables, dtype=float) - row_basis.T @ row_basis
        smallest = singular_values[rank - 1]
        condition = float(singular_values[0] / smallest) if smallest > 0.0 else float("inf")
    else:
        projector = np.eye(n_variables, dtype=float)
        condition = 1.0

    symmetry = _relative_frobenius(projector - projector.T, projector)
    idempotence = _relative_frobenius(projector @ projector - projector, projector)
    tangent = _relative_frobenius(normalized @ projector, normalized)
    return ProjectorDiagnostics(
        projector=projector,
        rank=rank,
        singular_values=singular_values,
        row_norms=row_norms,
        condition_number=condition,
        symmetry_residual=symmetry,
        idempotence_residual=idempotence,
        tangent_residual=tangent,
    )


def cosine_matrix(vectors: np.ndarray, *, atol: float = 1.0e-14) -> np.ndarray:
    """Return a signed cosine matrix, preserving undefined entries as NaN."""

    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    norms = np.linalg.norm(arr, axis=1)
    result = np.full((arr.shape[0], arr.shape[0]), np.nan, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            denom = norms[i] * norms[j]
            if denom > atol:
                result[i, j] = float(np.clip(np.dot(arr[i], arr[j]) / denom, -1.0, 1.0))
    return result


def expected_signed_matrix(num_objectives: int) -> np.ndarray:
    target = np.ones((num_objectives, num_objectives), dtype=float)
    target[:-1, -1] = -1.0
    target[-1, :-1] = -1.0
    return target


def aggregate_signed_matrices(
    matrices: Sequence[np.ndarray],
    *,
    competition_weighted: bool,
    alpha_weight: float = 0.9,
    beta_weight: float = 100.0,
) -> np.ndarray:
    """Aggregate pointwise signed matrices while retaining invalid NaNs."""

    stack = np.asarray(matrices, dtype=float)
    if stack.ndim != 3:
        raise ValueError("matrices must have shape (n_points, m, m)")
    valid = np.isfinite(stack)
    if competition_weighted:
        weights = np.asarray(
            logistic_competition_weight(stack, alpha_weight=alpha_weight, beta_weight=beta_weight),
            dtype=float,
        )
    else:
        weights = np.ones_like(stack)
    weights = np.where(valid, weights, 0.0)
    denom = np.sum(weights, axis=0)
    numerator = np.nansum(weights * np.where(valid, stack, 0.0), axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    np.fill_diagonal(out, 1.0)
    return out


def raw_signed_matrix(objective_gradients: np.ndarray) -> np.ndarray:
    return cosine_matrix(np.asarray(objective_gradients, dtype=float))


def joint_signed_matrix(
    objective_gradients: np.ndarray,
    constraint_jacobian: np.ndarray,
) -> tuple[np.ndarray, ProjectorDiagnostics]:
    diagnostics = joint_tangent_projector(constraint_jacobian)
    projected = np.asarray(objective_gradients, dtype=float) @ diagnostics.projector
    return cosine_matrix(projected), diagnostics


def matrices_from_sampled_blocks(
    blocks: SampledGradientBlocks,
    *,
    active_tolerance: float = 1.0e-10,
    num_groups: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute raw, current, single-constraint, and joint matrices."""

    raw_pointwise = [raw_signed_matrix(grads) for grads in blocks.objective_gradients]
    raw = aggregate_signed_matrices(raw_pointwise, competition_weighted=False)

    strengths, valid_mask, local_meta = compute_nonlinear_local_objective_interactions(
        blocks.objective_gradients,
        blocks.constraint_jacobians,
        constraint_values=blocks.constraint_values,
        active_constraint_tolerance=active_tolerance,
    )
    single_stack = np.where(valid_mask, strengths, np.nan)
    single = aggregate_signed_matrices(
        [single_stack[:, :, idx, 0] for idx in range(single_stack.shape[2])],
        competition_weighted=False,
    )

    config = NonlinearORCAConfig(
        num_groups=num_groups,
        grouping_method="average_linkage",
        active_constraint_tolerance=active_tolerance,
    )
    current_result = nonlinear_orca_from_sampled_gradients(blocks, config=config)
    current = 2.0 * current_result.adj_matrix - 1.0

    joint_pointwise: list[np.ndarray] = []
    projector_records: list[ProjectorDiagnostics] = []
    active_counts: list[int] = []
    for n, (grads, jac, values) in enumerate(
        zip(blocks.objective_gradients, blocks.constraint_jacobians, blocks.constraint_values)
    ):
        active = np.asarray(values >= -active_tolerance, dtype=bool)
        active_counts.append(int(np.sum(active)))
        selected_jac = jac[active]
        matrix, diagnostics = joint_signed_matrix(grads, selected_jac)
        joint_pointwise.append(matrix)
        projector_records.append(diagnostics)
    joint = aggregate_signed_matrices(joint_pointwise, competition_weighted=True)

    metadata: dict[str, Any] = {
        "current_result": current_result,
        "local_metadata": local_meta,
        "joint_diagnostics": projector_records,
        "active_counts": active_counts,
    }
    return {
        "Raw-Cosine": raw,
        "Single-Constraint-Mean": single,
        "ORCA-current": current,
        "ORCA-joint": joint,
    }, metadata


def sampled_blocks(
    problem: Any,
    points: np.ndarray,
    *,
    gradient_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> SampledGradientBlocks:
    blocks = evaluate_sampled_gradient_blocks(problem, points)
    if gradient_noise <= 0.0:
        return blocks
    generator = rng or np.random.default_rng(0)
    gradients = blocks.objective_gradients.copy()
    row_norms = np.linalg.norm(gradients, axis=2, keepdims=True)
    scale = gradient_noise * np.maximum(row_norms, np.finfo(float).tiny) / np.sqrt(gradients.shape[2])
    gradients += generator.normal(size=gradients.shape) * scale
    return SampledGradientBlocks(
        points=blocks.points,
        objective_gradients=gradients,
        constraint_jacobians=blocks.constraint_jacobians,
        constraint_values=blocks.constraint_values,
    )


def value_correlation_matrices(values: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    pearson = np.corrcoef(arr, rowvar=False)
    ranks = np.column_stack([rankdata(arr[:, j], method="average") for j in range(arr.shape[1])])
    spearman = np.corrcoef(ranks, rowvar=False)
    return {"Pearson": pearson, "Spearman": spearman}


def fixed_k_groups(signed_matrix: np.ndarray, num_groups: int = 2) -> np.ndarray:
    adjacency = 0.5 * (1.0 + np.nan_to_num(signed_matrix, nan=0.0))
    np.fill_diagonal(adjacency, 1.0)
    return group_objectives_average_linkage(adjacency, num_groups)


def positive_component_groups(signed_matrix: np.ndarray, threshold: float = 1.0e-8) -> np.ndarray:
    """Discover groups as connected components of positive-correlation edges."""

    matrix = np.asarray(signed_matrix, dtype=float)
    n = matrix.shape[0]
    labels = np.zeros(n, dtype=int)
    next_label = 1
    for start in range(n):
        if labels[start] != 0:
            continue
        stack = [start]
        labels[start] = next_label
        while stack:
            node = stack.pop()
            neighbors = np.where(np.isfinite(matrix[node]) & (matrix[node] > threshold))[0]
            for neighbor in neighbors:
                if neighbor != node and labels[neighbor] == 0:
                    labels[neighbor] = next_label
                    stack.append(int(neighbor))
        next_label += 1
    return labels


def adjusted_rand_index(true_labels: Sequence[int], pred_labels: Sequence[int]) -> float:
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    n = true.size
    if n < 2:
        return 1.0
    true_ids = {value: idx for idx, value in enumerate(np.unique(true))}
    pred_ids = {value: idx for idx, value in enumerate(np.unique(pred))}
    contingency = np.zeros((len(true_ids), len(pred_ids)), dtype=int)
    for t, p in zip(true, pred):
        contingency[true_ids[t], pred_ids[p]] += 1
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_cells = float(np.sum(comb2(contingency)))
    sum_rows = float(np.sum(comb2(np.sum(contingency, axis=1))))
    sum_cols = float(np.sum(comb2(np.sum(contingency, axis=0))))
    total = comb2(n)
    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = 0.5 * (sum_rows + sum_cols)
    if abs(maximum - expected) <= 1.0e-15:
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def normalized_mutual_information(true_labels: Sequence[int], pred_labels: Sequence[int]) -> float:
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    n = float(true.size)
    true_values, true_counts = np.unique(true, return_counts=True)
    pred_values, pred_counts = np.unique(pred, return_counts=True)
    mutual = 0.0
    for t, t_count in zip(true_values, true_counts):
        for p, p_count in zip(pred_values, pred_counts):
            joint = float(np.sum((true == t) & (pred == p)))
            if joint > 0.0:
                mutual += (joint / n) * np.log((joint * n) / (t_count * p_count))
    entropy_true = -sum((count / n) * np.log(count / n) for count in true_counts)
    entropy_pred = -sum((count / n) * np.log(count / n) for count in pred_counts)
    denom = np.sqrt(entropy_true * entropy_pred)
    if denom <= 1.0e-15:
        return 1.0 if np.array_equal(true, pred) else 0.0
    return float(mutual / denom)


def pairwise_group_scores(true_labels: Sequence[int], pred_labels: Sequence[int]) -> tuple[float, float, float]:
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    tp = fp = fn = 0
    for i in range(true.size):
        for j in range(i + 1, true.size):
            same_true = true[i] == true[j]
            same_pred = pred[i] == pred[j]
            tp += int(same_true and same_pred)
            fp += int((not same_true) and same_pred)
            fn += int(same_true and (not same_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def grouping_metrics(true_labels: Sequence[int], pred_labels: Sequence[int]) -> dict[str, float | int]:
    precision, recall, f1 = pairwise_group_scores(true_labels, pred_labels)
    return {
        "ari": adjusted_rand_index(true_labels, pred_labels),
        "nmi": normalized_mutual_information(true_labels, pred_labels),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "estimated_groups": int(np.unique(pred_labels).size),
        "exact_recovery": int(adjusted_rand_index(true_labels, pred_labels) >= 1.0 - 1.0e-12),
    }


def edge_summary(matrix: np.ndarray) -> dict[str, float]:
    arr = np.asarray(matrix, dtype=float)
    within = arr[:-1, :-1][np.triu_indices(arr.shape[0] - 1, k=1)]
    between = arr[:-1, -1]
    within_finite = within[np.isfinite(within)]
    between_finite = between[np.isfinite(between)]
    within_mean = float(np.mean(within_finite)) if within_finite.size else float("nan")
    within_min = float(np.min(within_finite)) if within_finite.size else float("nan")
    between_mean = float(np.mean(between_finite)) if between_finite.size else float("nan")
    between_max = float(np.max(between_finite)) if between_finite.size else float("nan")
    return {
        "within_mean": within_mean if within.size else 1.0,
        "within_min": within_min if within.size else 1.0,
        "between_mean": between_mean,
        "between_max": between_max,
        "separation_margin": (
            within_min - between_max
            if np.isfinite(within_min) and np.isfinite(between_max)
            else float("nan")
        ),
    }
