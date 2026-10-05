"""Generate source data for the DTLZ manuscript figures.

The script performs two bounded, reproducible tasks:

1. Recompute the standard Python DTLZ5(5, 16) objective-affinity matrix and
   compare the fixed-K average-linkage and Leiden partitions.
2. Run three-objective DTLZ5/DTLZ6 validation experiments and save every
   returned solution in the original objective space.  In three objectives the
   analytic g=0 front is an unambiguous quarter-circle after collapsing the
   correlated {f1, f2} block with its Euclidean norm.

Run with the Python environment that contains the ORCA package dependencies.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
ORCA_REPO = Path(os.environ.get("ORCA_REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ORCA_REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from orca.benchmarks import DTLZ5Problem, DTLZ6Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.experiments.downstream_optimizer import (
    _evaluate_full_objectives,
    _reference_directions,
    _run_nsga3,
)
from orca.nonlinear.main import nonlinear_orca


STRUCTURE_SEED = 0
STRUCTURE_I = 5
STRUCTURE_M = 16
STRUCTURE_K_TAIL = 10

PARETO_I = 2
PARETO_M = 3
PARETO_K_TAIL = 10
PARETO_SEEDS = tuple(range(5))
MIN_POPULATION_SIZE = 24
FULL_GENERATIONS = 200
REFERENCE_POINTS = 2001


def labels_to_text(labels: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in labels)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    labels = [f"f{i}" for i in range(1, matrix.shape[0] + 1)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def generate_structure_source_data() -> None:
    problem = DTLZ5Problem(
        intrinsic_dimension=STRUCTURE_I,
        num_objectives=STRUCTURE_M,
        k_tail=STRUCTURE_K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy="deterministic",
        optimizer_backend="scipy",
    )
    common = dict(
        num_groups=STRUCTURE_I,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=STRUCTURE_SEED,
        step_size=0.03,
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )
    average_result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(grouping_method="average_linkage", **common),
    )
    leiden_result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(grouping_method="leiden", **common),
    )
    expected = expected_dtlz5_groups(STRUCTURE_M, STRUCTURE_I)

    write_matrix(HERE / "dtlz5_m16_affinity_seed0.csv", np.asarray(average_result.adj_matrix, dtype=float))
    rows = [
        ("known structure", expected),
        ("average linkage (fixed K=5)", np.asarray(average_result.groups, dtype=int)),
        ("Leiden (target K=5)", np.asarray(leiden_result.groups, dtype=int)),
    ]
    with (HERE / "dtlz5_m16_partitions_seed0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "labels", "num_groups", "ari_vs_known", "exact_match"])
        for method, labels in rows:
            writer.writerow(
                [
                    method,
                    labels_to_text(labels),
                    len(set(labels.tolist())),
                    f"{adjusted_rand_index(expected, labels):.12g}",
                    int(np.array_equal(expected, labels)),
                ]
            )


def aggregate_l2_by_group(objective_values: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    values = np.atleast_2d(np.asarray(objective_values, dtype=float))
    labels = np.asarray(groups, dtype=int)
    ordered_labels = list(dict.fromkeys(labels.tolist()))
    reduced = np.zeros((values.shape[0], len(ordered_labels)), dtype=float)
    for group_index, label in enumerate(ordered_labels):
        reduced[:, group_index] = np.linalg.norm(values[:, labels == label], axis=1)
    return reduced


class ReducedL2Problem:
    """Lazy pymoo wrapper for the L2-aggregated objective groups."""

    def __new__(cls, problem, groups: Sequence[int] | None = None):  # noqa: ANN001
        from pymoo.core.problem import Problem

        lower = np.zeros(problem.num_variables(), dtype=float)
        upper = np.ones(problem.num_variables(), dtype=float)
        sample = 0.5 * (lower + upper)
        num_constraints = len(np.asarray(problem.constraint_values(sample), dtype=float))
        num_objectives = problem.num_objectives() if groups is None else len(set(groups))

        class WrappedProblem(Problem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=problem.num_variables(),
                    n_obj=num_objectives,
                    n_ieq_constr=num_constraints,
                    xl=lower,
                    xu=upper,
                )

            def _evaluate(self, X, out, *args, **kwargs) -> None:  # noqa: ANN001
                rows = np.atleast_2d(np.asarray(X, dtype=float))
                full_values = np.vstack([problem.objective_values(row) for row in rows])
                out["F"] = full_values if groups is None else aggregate_l2_by_group(full_values, groups)
                out["G"] = np.vstack([problem.constraint_values(row) for row in rows])

        return WrappedProblem()


def actual_evaluations(result, fallback: int) -> int:  # noqa: ANN001
    try:
        return int(result.algorithm.evaluator.n_eval)
    except Exception:
        return int(fallback)


def analytic_front(num_points: int = REFERENCE_POINTS) -> np.ndarray:
    theta = np.linspace(0.0, 0.5 * math.pi, num_points)
    radial = np.cos(theta)
    return np.column_stack([radial / math.sqrt(2.0), radial / math.sqrt(2.0), np.sin(theta)])


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    distances = np.empty(query.shape[0], dtype=float)
    indices = np.empty(query.shape[0], dtype=int)
    block_size = 256
    for start in range(0, query.shape[0], block_size):
        block = query[start : start + block_size]
        delta = block[:, None, :] - reference[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        idx = np.argmin(squared, axis=1)
        distances[start : start + len(block)] = np.sqrt(squared[np.arange(len(block)), idx])
        indices[start : start + len(block)] = idx
    return distances, indices


def run_pareto_seed(family: str, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    problem_class = DTLZ5Problem if family == "DTLZ5" else DTLZ6Problem
    problem = problem_class(
        intrinsic_dimension=PARETO_I,
        num_objectives=PARETO_M,
        k_tail=PARETO_K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy="optimize",
        optimizer_backend="scipy",
    )
    expected = expected_dtlz5_groups(PARETO_M, PARETO_I)
    config = NonlinearORCAConfig(
        num_groups=PARETO_I,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=seed,
        step_size=0.03,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )
    grouping_start = time.perf_counter()
    orca_result = nonlinear_orca(problem, config)
    grouping_seconds = time.perf_counter() - grouping_start
    groups = np.asarray(orca_result.groups, dtype=int)
    ari = adjusted_rand_index(expected, groups)

    full_ref_dirs = _reference_directions(PARETO_M, MIN_POPULATION_SIZE)
    reduced_ref_dirs = _reference_directions(PARETO_I, MIN_POPULATION_SIZE)
    full_eval_budget = len(full_ref_dirs) * FULL_GENERATIONS
    reduced_generations = int(math.ceil(full_eval_budget / len(reduced_ref_dirs)))

    full_result, full_seconds, _ = _run_nsga3(
        ReducedL2Problem(problem, groups=None),
        num_objectives=PARETO_M,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=FULL_GENERATIONS,
        random_seed=seed,
    )
    reduced_result, reduced_seconds, _ = _run_nsga3(
        ReducedL2Problem(problem, groups=groups),
        num_objectives=PARETO_I,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=reduced_generations,
        random_seed=seed,
    )

    method_data = [
        (
            "Full NSGA-III",
            np.asarray(full_result.X, dtype=float),
            full_result,
            full_seconds,
            0.0,
            FULL_GENERATIONS,
            len(full_ref_dirs),
        ),
        (
            "ORCA-reduced NSGA-III (L2)",
            np.asarray(reduced_result.X, dtype=float),
            reduced_result,
            reduced_seconds,
            grouping_seconds,
            reduced_generations,
            len(reduced_ref_dirs),
        ),
    ]

    reference = analytic_front()
    solution_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for method, X, result, optimizer_seconds, method_grouping_seconds, n_gen, ref_count in method_data:
        F = _evaluate_full_objectives(problem, X)
        distance_to_front, nearest_indices = nearest_distances(F, reference)
        distance_from_front, _ = nearest_distances(reference, F)
        collapsed = np.column_stack([np.linalg.norm(F[:, :2], axis=1), F[:, 2]])
        radial_gap = np.linalg.norm(F, axis=1) - 1.0
        angles = np.arctan2(collapsed[:, 1], collapsed[:, 0])
        evals = actual_evaluations(result, ref_count * n_gen)
        total_seconds = optimizer_seconds + method_grouping_seconds

        for solution_index, (x_row, f_row) in enumerate(zip(X, F)):
            nearest = reference[nearest_indices[solution_index]]
            solution_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "method": method,
                    "solution_index": solution_index,
                    "f1": f_row[0],
                    "f2": f_row[1],
                    "f3": f_row[2],
                    "collapsed_block_l2": collapsed[solution_index, 0],
                    "radial_gap": radial_gap[solution_index],
                    "distance_to_true_front": distance_to_front[solution_index],
                    "nearest_true_f1": nearest[0],
                    "nearest_true_f2": nearest[1],
                    "nearest_true_f3": nearest[2],
                    "g_value": problem._g_value(x_row),  # noqa: SLF001
                    "evals": evals,
                    "total_seconds": total_seconds,
                    "groups": labels_to_text(groups),
                    "ari_vs_known": ari,
                }
            )

        metric_rows.append(
            {
                "family": family,
                "seed": seed,
                "method": method,
                "num_solutions": F.shape[0],
                "evals": evals,
                "n_gen": n_gen,
                "reference_directions": ref_count,
                "optimizer_seconds": optimizer_seconds,
                "grouping_seconds": method_grouping_seconds,
                "total_seconds": total_seconds,
                "gd_true": float(np.mean(distance_to_front)),
                "gd95_true": float(np.quantile(distance_to_front, 0.95)),
                "igd_true": float(np.mean(distance_from_front)),
                "radial_gap_median": float(np.median(radial_gap)),
                "radial_gap_mean": float(np.mean(radial_gap)),
                "angular_coverage": float((np.max(angles) - np.min(angles)) / (0.5 * math.pi)),
                "groups": labels_to_text(groups),
                "ari_vs_known": ari,
            }
        )
    return solution_rows, metric_rows


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_pareto_source_data() -> None:
    all_solutions: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    for family in ("DTLZ5", "DTLZ6"):
        for seed in PARETO_SEEDS:
            print(f"Running {family}(2,3), seed {seed + 1}/{len(PARETO_SEEDS)}", flush=True)
            solution_rows, metric_rows = run_pareto_seed(family, seed)
            all_solutions.extend(solution_rows)
            all_metrics.extend(metric_rows)
    write_dict_rows(HERE / "dtlz3_pareto_solutions.csv", all_solutions)
    write_dict_rows(HERE / "dtlz3_pareto_metrics.csv", all_metrics)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    generate_structure_source_data()
    generate_pareto_source_data()
    print(f"Source data written to {HERE}")


if __name__ == "__main__":
    main()
