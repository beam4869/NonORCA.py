"""Runtime-scaling experiments for ORCA vs solution-set generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from orca.config import NonlinearORCAConfig
from orca.nonlinear.main import nonlinear_orca
from orca.nonlinear.problem_interface import NonlinearORCAProblem
from orca.experiments.downstream_optimizer import _run_nsga3, aggregate_objectives_by_group


@dataclass(frozen=True)
class RuntimeScalingRow:
    """Timing row for one benchmark instance."""

    dataset: str
    num_objectives: int
    num_variables: int
    num_groups: int
    n_gen: int
    min_population_size: int
    full_ref_dirs: int
    reduced_ref_dirs: int
    orca_points: int
    orca_grouping_seconds: float
    full_optimizer_seconds: float
    reduced_optimizer_seconds: float
    orca_plus_reduced_seconds: float
    full_solutions: int
    reduced_solutions: int
    groups: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["groups"] = " ".join(str(value) for value in self.groups)
        return data


def _bounds(problem: NonlinearORCAProblem) -> tuple[np.ndarray, np.ndarray]:
    n = problem.num_variables()
    lower = np.asarray(getattr(problem, "lower_bounds", np.zeros(n)), dtype=float)
    upper = np.asarray(getattr(problem, "upper_bounds", np.ones(n)), dtype=float)
    return lower, upper


class _BoxOnlyPymooProblem:
    """Pymoo wrapper using variable bounds only, for runtime experiments."""

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
            raise ImportError("Runtime scaling experiments require pymoo") from exc

        lower, upper = _bounds(problem)
        num_objectives = problem.num_objectives() if groups is None else len(set(np.asarray(groups, dtype=int).tolist()))

        class WrappedProblem(Problem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=problem.num_variables(),
                    n_obj=num_objectives,
                    n_ieq_constr=0,
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

        return WrappedProblem()


def run_runtime_scaling_case(
    dataset: str,
    problem: NonlinearORCAProblem,
    *,
    num_groups: int,
    n_gen: int = 100,
    min_population_size: int = 24,
    random_seed: int = 0,
    orca_point_limit: int = 5,
    grouping_method: str = "average_linkage",
    aggregation: str = "mean",
) -> RuntimeScalingRow:
    """Measure ORCA grouping time vs full and ORCA-reduced NSGA-III time."""

    seed_points = list(problem.initial_points())[:orca_point_limit]
    orca_cfg = NonlinearORCAConfig(
        num_groups=num_groups,
        num_points_per_seed=0,
        include_seed_points=True,
        random_seed=random_seed,
        grouping_method=grouping_method,
    )
    start = perf_counter()
    orca_result = nonlinear_orca(problem, orca_cfg, points=np.asarray(seed_points, dtype=float))
    orca_seconds = perf_counter() - start

    full_problem = _BoxOnlyPymooProblem(problem, groups=None, aggregation=aggregation)
    full_result, full_seconds, full_ref_dirs = _run_nsga3(
        full_problem,
        num_objectives=problem.num_objectives(),
        min_population_size=min_population_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )

    reduced_problem = _BoxOnlyPymooProblem(problem, groups=orca_result.groups, aggregation=aggregation)
    reduced_result, reduced_seconds, reduced_ref_dirs = _run_nsga3(
        reduced_problem,
        num_objectives=len(set(orca_result.groups.tolist())),
        min_population_size=min_population_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )

    return RuntimeScalingRow(
        dataset=dataset,
        num_objectives=problem.num_objectives(),
        num_variables=problem.num_variables(),
        num_groups=len(set(orca_result.groups.tolist())),
        n_gen=n_gen,
        min_population_size=min_population_size,
        full_ref_dirs=len(full_ref_dirs),
        reduced_ref_dirs=len(reduced_ref_dirs),
        orca_points=len(seed_points),
        orca_grouping_seconds=float(orca_seconds),
        full_optimizer_seconds=float(full_seconds),
        reduced_optimizer_seconds=float(reduced_seconds),
        orca_plus_reduced_seconds=float(orca_seconds + reduced_seconds),
        full_solutions=0 if full_result.X is None else int(np.asarray(full_result.X).shape[0]),
        reduced_solutions=0 if reduced_result.X is None else int(np.asarray(reduced_result.X).shape[0]),
        groups=tuple(int(value) for value in orca_result.groups.tolist()),
    )
