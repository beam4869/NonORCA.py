"""Reproduce the paired DTLZ5(2,3) analytic-front NSGA-III audit.

This is a minimal extraction of the protocol in
``figures/dtlz_paper/generate_dtlz_source_data.py``.  It intentionally uses
the current project implementation of DTLZ5(I,M), ORCA, and the shared pymoo
NSGA-III helpers rather than copying those implementations here.

The analytic front is fixed before either algorithm is run.  Distances are
computed after applying the fixed external bounds [0,1] to every original
objective.  This scaling is the identity and is stated explicitly to rule out
run-wise or pooled-output normalization.  The primary 2,001-point grid matches
the persisted source experiment.  A 10,001-point sensitivity calculation is
also saved, but it is not the source of the reported manuscript values.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pymoo
import scipy


HERE = Path(__file__).resolve().parent
ORCA_REPO = Path(
    os.environ.get("ORCA_REPO", str(Path(__file__).resolve().parents[2]))
).resolve()
sys.path.insert(0, str(ORCA_REPO))

from ORCA_python.benchmarks import DTLZ5Problem, expected_dtlz5_groups
from ORCA_python.config import NonlinearORCAConfig
from ORCA_python.experiments.baseline_comparison import adjusted_rand_index
from ORCA_python.experiments.downstream_optimizer import (
    _evaluate_full_objectives,
    _reference_directions,
    _run_nsga3,
)
from ORCA_python.nonlinear.main import nonlinear_orca


INTRINSIC_DIMENSION = 2
NUM_OBJECTIVES = 3
K_TAIL = 10
SEEDS = tuple(range(5))
MIN_POPULATION_SIZE = 24
FULL_GENERATIONS = 200
REFERENCE_POINTS = 2_001
SENSITIVITY_REFERENCE_POINTS = 10_001
FIXED_SCALE_LOWER = np.zeros(NUM_OBJECTIVES, dtype=float)
FIXED_SCALE_UPPER = np.ones(NUM_OBJECTIVES, dtype=float)


def labels_to_text(labels: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in labels)


def aggregate_l2_by_group(
    objective_values: np.ndarray, groups: Sequence[int]
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(objective_values, dtype=float))
    labels = np.asarray(groups, dtype=int)
    ordered_labels = list(dict.fromkeys(labels.tolist()))
    reduced = np.zeros((values.shape[0], len(ordered_labels)), dtype=float)
    for group_index, label in enumerate(ordered_labels):
        reduced[:, group_index] = np.linalg.norm(
            values[:, labels == label], axis=1
        )
    return reduced


class ReducedL2Problem:
    """Create the same lazy pymoo wrapper used by the source experiment."""

    def __new__(cls, problem, groups: Sequence[int] | None = None):  # noqa: ANN001
        from pymoo.core.problem import Problem

        lower = np.zeros(problem.num_variables(), dtype=float)
        upper = np.ones(problem.num_variables(), dtype=float)
        num_constraints = len(
            np.asarray(problem.constraint_values(0.5 * (lower + upper)), dtype=float)
        )
        num_objectives = (
            problem.num_objectives() if groups is None else len(set(groups))
        )

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
                full_values = np.vstack(
                    [problem.objective_values(row) for row in rows]
                )
                out["F"] = (
                    full_values
                    if groups is None
                    else aggregate_l2_by_group(full_values, groups)
                )
                out["G"] = np.vstack(
                    [problem.constraint_values(row) for row in rows]
                )

        return WrappedProblem()


def actual_evaluations(result, fallback: int) -> int:  # noqa: ANN001
    try:
        return int(result.algorithm.evaluator.n_eval)
    except Exception:
        return int(fallback)


def analytic_front(num_points: int) -> np.ndarray:
    theta = np.linspace(0.0, 0.5 * math.pi, num_points)
    radial = np.cos(theta)
    return np.column_stack(
        [radial / math.sqrt(2.0), radial / math.sqrt(2.0), np.sin(theta)]
    )


def apply_fixed_scaling(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) - FIXED_SCALE_LOWER) / (
        FIXED_SCALE_UPPER - FIXED_SCALE_LOWER
    )


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    distances = np.empty(query.shape[0], dtype=float)
    block_size = 256
    for start in range(0, query.shape[0], block_size):
        block = query[start : start + block_size]
        delta = block[:, None, :] - reference[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        distances[start : start + len(block)] = np.sqrt(
            np.min(squared, axis=1)
        )
    return distances


def gd_igd(values: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    scaled_values = apply_fixed_scaling(values)
    scaled_reference = apply_fixed_scaling(reference)
    gd = float(np.mean(nearest_distances(scaled_values, scaled_reference)))
    igd = float(np.mean(nearest_distances(scaled_reference, scaled_values)))
    return gd, igd


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seed(seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    problem = DTLZ5Problem(
        intrinsic_dimension=INTRINSIC_DIMENSION,
        num_objectives=NUM_OBJECTIVES,
        k_tail=K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy="optimize",
        optimizer_backend="scipy",
    )
    expected = expected_dtlz5_groups(NUM_OBJECTIVES, INTRINSIC_DIMENSION)
    config = NonlinearORCAConfig(
        num_groups=INTRINSIC_DIMENSION,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=seed,
        step_size=0.03,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )
    orca_result = nonlinear_orca(problem, config)
    groups = np.asarray(orca_result.groups, dtype=int)
    ari = adjusted_rand_index(expected, groups)

    full_ref_dirs = _reference_directions(
        NUM_OBJECTIVES, MIN_POPULATION_SIZE
    )
    reduced_ref_dirs = _reference_directions(
        INTRINSIC_DIMENSION, MIN_POPULATION_SIZE
    )
    full_budget = len(full_ref_dirs) * FULL_GENERATIONS
    reduced_generations = int(math.ceil(full_budget / len(reduced_ref_dirs)))

    full_result, _, _ = _run_nsga3(
        ReducedL2Problem(problem),
        num_objectives=NUM_OBJECTIVES,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=FULL_GENERATIONS,
        random_seed=seed,
    )
    reduced_result, _, _ = _run_nsga3(
        ReducedL2Problem(problem, groups=groups),
        num_objectives=INTRINSIC_DIMENSION,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=reduced_generations,
        random_seed=seed,
    )

    runs = (
        (
            "Full NSGA-III",
            full_result,
            FULL_GENERATIONS,
            len(full_ref_dirs),
        ),
        (
            "Grouped NSGA-III, L2",
            reduced_result,
            reduced_generations,
            len(reduced_ref_dirs),
        ),
    )
    reference = analytic_front(REFERENCE_POINTS)
    sensitivity_reference = analytic_front(SENSITIVITY_REFERENCE_POINTS)
    solution_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for method, result, generations, ref_count in runs:
        X = np.atleast_2d(np.asarray(result.X, dtype=float))
        F = _evaluate_full_objectives(problem, X)
        evals = actual_evaluations(result, ref_count * generations)
        gd, igd = gd_igd(F, reference)
        gd_sensitivity, igd_sensitivity = gd_igd(F, sensitivity_reference)
        metric_rows.append(
            {
                "seed": seed,
                "method": method,
                "groups": labels_to_text(groups),
                "ari_vs_known": ari,
                "num_solutions": F.shape[0],
                "reference_directions": ref_count,
                "generations": generations,
                "evaluations": evals,
                "reference_points": REFERENCE_POINTS,
                "gd": gd,
                "igd": igd,
                "gd_with_10001_points": gd_sensitivity,
                "igd_with_10001_points": igd_sensitivity,
            }
        )
        for solution_index, (x_row, f_row) in enumerate(zip(X, F)):
            row: dict[str, object] = {
                "seed": seed,
                "method": method,
                "solution_index": solution_index,
                "f1": f_row[0],
                "f2": f_row[1],
                "f3": f_row[2],
            }
            row.update({f"x{i + 1}": value for i, value in enumerate(x_row)})
            solution_rows.append(row)
    return solution_rows, metric_rows


def main() -> None:
    all_solutions: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    for seed in SEEDS:
        print(f"Running paired seed {seed}", flush=True)
        solution_rows, metric_rows = run_seed(seed)
        all_solutions.extend(solution_rows)
        all_metrics.extend(metric_rows)

    write_rows(HERE / "dtlz5_i2_m3_analytic_reproduction_metrics.csv", all_metrics)
    write_rows(HERE / "dtlz5_i2_m3_analytic_reproduction_solutions.csv", all_solutions)

    methods = list(dict.fromkeys(row["method"] for row in all_metrics))
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        selected = [row for row in all_metrics if row["method"] == method]
        gd = np.asarray([row["gd"] for row in selected], dtype=float)
        igd = np.asarray([row["igd"] for row in selected], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "seeds": labels_to_text(SEEDS),
                "evaluations": selected[0]["evaluations"],
                "reference_points": REFERENCE_POINTS,
                "gd_mean": float(np.mean(gd)),
                "gd_sample_sd": float(np.std(gd, ddof=1)),
                "igd_mean": float(np.mean(igd)),
                "igd_sample_sd": float(np.std(igd, ddof=1)),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pymoo": pymoo.__version__,
            }
        )
    write_rows(HERE / "dtlz5_i2_m3_analytic_reproduction_summary.csv", summary_rows)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
