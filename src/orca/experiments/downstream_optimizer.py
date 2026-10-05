"""Downstream optimizer experiments for ORCA objective reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

import numpy as np

from orca.nonlinear.problem_interface import NonlinearORCAProblem


@dataclass(frozen=True)
class DownstreamComparisonResult:
    """Full-objective vs ORCA-reduced optimizer comparison in full objective space."""

    groups: np.ndarray
    full_hv: float
    reduced_hv: float
    full_igd: float
    reduced_igd: float
    full_runtime_seconds: float
    reduced_runtime_seconds: float
    full_objectives: np.ndarray
    reduced_objectives: np.ndarray
    reference_objectives: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def aggregate_objectives_by_group(
    objective_values: np.ndarray,
    groups: Sequence[int],
    *,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate full objective values into group-level reduced objectives."""

    values = np.asarray(objective_values, dtype=float)
    was_1d = values.ndim == 1
    if was_1d:
        values = values[None, :]
    labels = np.asarray(groups, dtype=int)
    if values.shape[1] != labels.size:
        raise ValueError("groups must have one label per full objective")

    ordered_labels = list(dict.fromkeys(labels.tolist()))
    reduced = np.zeros((values.shape[0], len(ordered_labels)), dtype=float)
    for group_idx, label in enumerate(ordered_labels):
        cols = values[:, labels == label]
        if method == "mean":
            reduced[:, group_idx] = np.mean(cols, axis=1)
        elif method == "sum":
            reduced[:, group_idx] = np.sum(cols, axis=1)
        elif method == "max":
            reduced[:, group_idx] = np.max(cols, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    return reduced[0] if was_1d else reduced


def _infer_bounds(problem: NonlinearORCAProblem) -> tuple[np.ndarray, np.ndarray]:
    num_variables = problem.num_variables()
    lower = np.asarray(getattr(problem, "lower_bounds", np.zeros(num_variables)), dtype=float)
    upper = np.asarray(getattr(problem, "upper_bounds", np.ones(num_variables)), dtype=float)
    if lower.shape != (num_variables,) or upper.shape != (num_variables,):
        raise ValueError("Problem bounds must have one entry per variable")
    return lower, upper


class _ORCAPymooProblem:
    """Lazy pymoo wrapper to keep pymoo optional for non-experiment users."""

    def __new__(
        cls,
        problem: NonlinearORCAProblem,
        *,
        groups: Sequence[int] | None = None,
        aggregation: str = "mean",
    ):
        try:
            from pymoo.core.problem import Problem  # type: ignore
        except ImportError as exc:
            raise ImportError("Downstream optimizer experiments require pymoo") from exc

        lower, upper = _infer_bounds(problem)
        sample = np.clip(0.5 * (lower + upper), lower, upper)
        num_constraints = len(np.asarray(problem.constraint_values(sample), dtype=float))
        num_objectives = problem.num_objectives() if groups is None else len(set(np.asarray(groups, dtype=int).tolist()))

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
                if groups is None:
                    out["F"] = full_values
                else:
                    out["F"] = aggregate_objectives_by_group(full_values, groups, method=aggregation)
                out["G"] = np.vstack([problem.constraint_values(row) for row in rows])

        return WrappedProblem()


def _reference_directions(num_objectives: int, min_population_size: int) -> np.ndarray:
    try:
        from pymoo.util.ref_dirs import get_reference_directions  # type: ignore
    except ImportError as exc:
        raise ImportError("NSGA-III reference directions require pymoo") from exc

    for partitions in range(1, 30):
        ref_dirs = get_reference_directions("das-dennis", num_objectives, n_partitions=partitions)
        if len(ref_dirs) >= min_population_size:
            return np.asarray(ref_dirs, dtype=float)
    raise RuntimeError("Failed to generate enough NSGA-III reference directions")


def _run_nsga3(
    pymoo_problem,
    *,
    num_objectives: int,
    min_population_size: int,
    n_gen: int,
    random_seed: int,
):
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3  # type: ignore
        from pymoo.optimize import minimize  # type: ignore
    except ImportError as exc:
        raise ImportError("NSGA-III experiments require pymoo") from exc

    ref_dirs = _reference_directions(num_objectives, min_population_size)
    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs)
    start = perf_counter()
    result = minimize(
        pymoo_problem,
        algorithm,
        ("n_gen", n_gen),
        seed=random_seed,
        verbose=False,
    )
    runtime = perf_counter() - start
    if result.X is None:
        raise RuntimeError("NSGA-III did not return a solution set")
    return result, runtime, ref_dirs


def _evaluate_full_objectives(problem: NonlinearORCAProblem, X: np.ndarray) -> np.ndarray:
    rows = np.atleast_2d(np.asarray(X, dtype=float))
    return np.vstack([problem.objective_values(row) for row in rows])


def _non_dominated_reference(F: np.ndarray) -> np.ndarray:
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # type: ignore
    except ImportError as exc:
        raise ImportError("Non-dominated sorting requires pymoo") from exc

    idx = NonDominatedSorting().do(np.asarray(F, dtype=float), only_non_dominated_front=True)
    return np.asarray(F, dtype=float)[idx]


def _normalize_against_union(*arrays: np.ndarray) -> list[np.ndarray]:
    union = np.vstack([np.asarray(array, dtype=float) for array in arrays])
    lower = np.min(union, axis=0)
    upper = np.max(union, axis=0)
    denom = upper - lower
    denom[denom <= 1.0e-12] = 1.0
    return [(np.asarray(array, dtype=float) - lower) / denom for array in arrays]


def _quality_metrics(F_full: np.ndarray, F_reduced: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    try:
        from pymoo.indicators.hv import HV  # type: ignore
        from pymoo.indicators.igd import IGD  # type: ignore
    except ImportError as exc:
        raise ImportError("HV/IGD metrics require pymoo") from exc

    pooled_reference = _non_dominated_reference(np.vstack([F_full, F_reduced]))
    F_full_norm, F_reduced_norm, reference_norm = _normalize_against_union(F_full, F_reduced, pooled_reference)
    ref_point = np.full(F_full.shape[1], 1.1, dtype=float)
    hv = HV(ref_point=ref_point)
    igd = IGD(reference_norm)
    return (
        float(hv(F_full_norm)),
        float(hv(F_reduced_norm)),
        float(igd(F_full_norm)),
        float(igd(F_reduced_norm)),
        pooled_reference,
    )


def compare_full_vs_reduced_nsga3(
    problem: NonlinearORCAProblem,
    groups: Sequence[int],
    *,
    aggregation: str = "mean",
    min_population_size: int = 40,
    n_gen: int = 40,
    random_seed: int = 0,
) -> DownstreamComparisonResult:
    """Run NSGA-III on full objectives and on group-reduced objectives.

    Both returned solution sets are evaluated in the original full-objective
    space. HV and IGD are computed on normalized full-objective values using the
    pooled non-dominated union as an empirical reference set.
    """

    labels = np.asarray(groups, dtype=int)
    if labels.shape != (problem.num_objectives(),):
        raise ValueError("groups must have one label per objective")

    full_problem = _ORCAPymooProblem(problem, groups=None, aggregation=aggregation)
    reduced_problem = _ORCAPymooProblem(problem, groups=labels, aggregation=aggregation)

    full_result, full_runtime, full_ref_dirs = _run_nsga3(
        full_problem,
        num_objectives=problem.num_objectives(),
        min_population_size=min_population_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )
    reduced_result, reduced_runtime, reduced_ref_dirs = _run_nsga3(
        reduced_problem,
        num_objectives=len(set(labels.tolist())),
        min_population_size=min_population_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )

    full_objectives = _evaluate_full_objectives(problem, np.asarray(full_result.X, dtype=float))
    reduced_objectives = _evaluate_full_objectives(problem, np.asarray(reduced_result.X, dtype=float))
    full_hv, reduced_hv, full_igd, reduced_igd, reference = _quality_metrics(full_objectives, reduced_objectives)
    return DownstreamComparisonResult(
        groups=labels,
        full_hv=full_hv,
        reduced_hv=reduced_hv,
        full_igd=full_igd,
        reduced_igd=reduced_igd,
        full_runtime_seconds=float(full_runtime),
        reduced_runtime_seconds=float(reduced_runtime),
        full_objectives=full_objectives,
        reduced_objectives=reduced_objectives,
        reference_objectives=reference,
        metadata={
            "aggregation": aggregation,
            "min_population_size": min_population_size,
            "n_gen": n_gen,
            "random_seed": random_seed,
            "full_ref_dirs": len(full_ref_dirs),
            "reduced_ref_dirs": len(reduced_ref_dirs),
        },
    )
