"""Multi-seed full-vs-reduced downstream reduction studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import BaselineGroupingResult, compare_grouping_methods
from orca.experiments.downstream_optimizer import (
    _ORCAPymooProblem,
    _evaluate_full_objectives,
    _non_dominated_reference,
    _normalize_against_union,
    _run_nsga3,
)
from orca.nonlinear.problem_interface import NonlinearORCAProblem


@dataclass(frozen=True)
class ReductionStudyRow:
    """One optimizer run summarized in the original objective space."""

    dataset: str
    seed: int
    method: str
    groups: tuple[int, ...] | None
    ari: float | None
    optimized_objectives: int
    hv: float
    igd: float
    runtime_seconds: float
    max_constraint_violation: float
    mean_constraint_violation: float
    num_solutions: int

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["groups"] = "" if self.groups is None else " ".join(str(value) for value in self.groups)
        return data


@dataclass(frozen=True)
class ReductionStudyResult:
    """Rows and grouping metadata for one benchmark's reduction study."""

    dataset: str
    groupings: tuple[BaselineGroupingResult, ...]
    rows: tuple[ReductionStudyRow, ...]


def _constraint_violation(problem: NonlinearORCAProblem, X: np.ndarray) -> tuple[float, float]:
    rows = np.atleast_2d(np.asarray(X, dtype=float))
    violations = []
    for row in rows:
        positive = np.maximum(np.asarray(problem.constraint_values(row), dtype=float), 0.0)
        violations.append(float(np.sum(positive)))
    if not violations:
        return 0.0, 0.0
    return float(np.max(violations)), float(np.mean(violations))


def _approximate_hv(
    values: np.ndarray,
    *,
    ref_point: np.ndarray,
    samples: np.ndarray,
) -> float:
    """Approximate minimization HV using shared samples in the reference box."""

    dominated = np.zeros(samples.shape[0], dtype=bool)
    for point in np.asarray(values, dtype=float):
        dominated |= np.all(point <= samples, axis=1)
    return float(np.prod(ref_point) * np.mean(dominated))


def _sobol_reference_box_samples(num_dimensions: int, ref_point: np.ndarray, *, num_samples: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc  # type: ignore
    except ImportError:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, ref_point, size=(num_samples, num_dimensions))

    power = int(np.ceil(np.log2(max(2, num_samples))))
    sampler = qmc.Sobol(d=num_dimensions, scramble=True, seed=seed)
    unit = sampler.random_base2(power)
    return unit[:num_samples] * ref_point


def _common_quality_metrics(
    objective_sets: Sequence[np.ndarray],
    *,
    exact_hv_max_dimensions: int = 7,
    hv_samples: int = 8192,
    hv_seed: int = 0,
) -> tuple[list[float], list[float]]:
    try:
        from pymoo.indicators.hv import HV  # type: ignore
        from pymoo.indicators.igd import IGD  # type: ignore
    except ImportError as exc:
        raise ImportError("HV/IGD metrics require pymoo") from exc

    pooled_reference = _non_dominated_reference(np.vstack([np.asarray(values, dtype=float) for values in objective_sets]))
    normalized = _normalize_against_union(*objective_sets, pooled_reference)
    normalized_sets = normalized[:-1]
    reference_norm = normalized[-1]
    ref_point = np.full(objective_sets[0].shape[1], 1.1, dtype=float)
    hv_indicator = HV(ref_point=ref_point) if objective_sets[0].shape[1] <= exact_hv_max_dimensions else None
    igd_indicator = IGD(reference_norm)
    hv_reference_samples = None
    if hv_indicator is None:
        hv_reference_samples = _sobol_reference_box_samples(
            objective_sets[0].shape[1],
            ref_point,
            num_samples=hv_samples,
            seed=hv_seed,
        )
    return (
        [
            float(hv_indicator(values))
            if hv_indicator is not None
            else _approximate_hv(values, ref_point=ref_point, samples=hv_reference_samples)
            for values in normalized_sets
        ],
        [float(igd_indicator(values)) for values in normalized_sets],
    )


def run_reduction_study(
    dataset: str,
    problem: NonlinearORCAProblem,
    expected_groups: Sequence[int],
    *,
    num_groups: int,
    seeds: Iterable[int],
    n_gen: int = 100,
    min_population_size: int = 24,
    sample_points: int = 192,
    grouping_seed: int = 2,
    aggregation: str = "mean",
    grouping_method: str = "average_linkage",
    orca_config: NonlinearORCAConfig | None = None,
    exact_hv_max_dimensions: int = 7,
    hv_samples: int = 8192,
) -> ReductionStudyResult:
    """Compare full objectives with ORCA/correlation/gradient reduced objectives.

    Groupings are computed once per benchmark. For each optimizer seed, NSGA-III
    is run on the full objective set and on each reduced objective set. All
    solution sets are then evaluated in the original full-objective space, and
    HV/IGD are computed against a common pooled non-dominated reference set for
    that seed.
    """

    cfg = orca_config or NonlinearORCAConfig(
        num_groups=num_groups,
        num_points_per_seed=3,
        include_seed_points=True,
        random_seed=11,
        step_size=0.03,
        grouping_method=grouping_method,
    )
    all_groupings = tuple(
        result
        for result in compare_grouping_methods(
            problem,
            expected_groups,
            num_groups=num_groups,
            sample_points=sample_points,
            random_seed=grouping_seed,
            grouping_method=grouping_method,
            orca_config=cfg,
        )
        if result.method in {"ORCA", "objective_value_correlation", "gradient_cosine"}
    )

    rows: list[ReductionStudyRow] = []
    for seed in seeds:
        full_problem = _ORCAPymooProblem(problem, groups=None, aggregation=aggregation)
        full_result, full_runtime, _ = _run_nsga3(
            full_problem,
            num_objectives=problem.num_objectives(),
            min_population_size=min_population_size,
            n_gen=n_gen,
            random_seed=int(seed),
        )
        method_names = ["full"]
        method_groups: list[np.ndarray | None] = [None]
        method_ari: list[float | None] = [None]
        method_runtimes = [float(full_runtime)]
        method_X = [np.asarray(full_result.X, dtype=float)]
        objective_sets = [_evaluate_full_objectives(problem, method_X[0])]

        for grouping in all_groupings:
            reduced_problem = _ORCAPymooProblem(problem, groups=grouping.groups, aggregation=aggregation)
            reduced_result, reduced_runtime, _ = _run_nsga3(
                reduced_problem,
                num_objectives=len(set(grouping.groups.tolist())),
                min_population_size=min_population_size,
                n_gen=n_gen,
                random_seed=int(seed),
            )
            x_values = np.asarray(reduced_result.X, dtype=float)
            method_names.append(grouping.method)
            method_groups.append(grouping.groups)
            method_ari.append(float(grouping.ari))
            method_runtimes.append(float(reduced_runtime))
            method_X.append(x_values)
            objective_sets.append(_evaluate_full_objectives(problem, x_values))

        hv_values, igd_values = _common_quality_metrics(
            objective_sets,
            exact_hv_max_dimensions=exact_hv_max_dimensions,
            hv_samples=hv_samples,
            hv_seed=int(seed),
        )
        for idx, name in enumerate(method_names):
            max_violation, mean_violation = _constraint_violation(problem, method_X[idx])
            labels = None if method_groups[idx] is None else tuple(int(value) for value in method_groups[idx].tolist())
            rows.append(
                ReductionStudyRow(
                    dataset=dataset,
                    seed=int(seed),
                    method=name,
                    groups=labels,
                    ari=method_ari[idx],
                    optimized_objectives=problem.num_objectives() if labels is None else len(set(labels)),
                    hv=hv_values[idx],
                    igd=igd_values[idx],
                    runtime_seconds=method_runtimes[idx],
                    max_constraint_violation=max_violation,
                    mean_constraint_violation=mean_violation,
                    num_solutions=int(objective_sets[idx].shape[0]),
                )
            )

    return ReductionStudyResult(dataset=dataset, groupings=all_groupings, rows=tuple(rows))


def summarize_reduction_rows(rows: Sequence[ReductionStudyRow]) -> list[dict[str, object]]:
    """Aggregate rows by dataset and method as mean/std dictionaries."""

    output: list[dict[str, object]] = []
    keys = sorted({(row.dataset, row.method) for row in rows})
    for dataset, method in keys:
        group = [row for row in rows if row.dataset == dataset and row.method == method]
        first = group[0]
        output.append(
            {
                "dataset": dataset,
                "method": method,
                "optimized_objectives": first.optimized_objectives,
                "ari": first.ari,
                "hv_mean": float(np.mean([row.hv for row in group])),
                "hv_std": float(np.std([row.hv for row in group], ddof=1)) if len(group) > 1 else 0.0,
                "igd_mean": float(np.mean([row.igd for row in group])),
                "igd_std": float(np.std([row.igd for row in group], ddof=1)) if len(group) > 1 else 0.0,
                "runtime_mean": float(np.mean([row.runtime_seconds for row in group])),
                "runtime_std": float(np.std([row.runtime_seconds for row in group], ddof=1)) if len(group) > 1 else 0.0,
                "max_constraint_violation": float(np.max([row.max_constraint_violation for row in group])),
                "mean_constraint_violation": float(np.mean([row.mean_constraint_violation for row in group])),
                "num_seeds": len(group),
            }
        )
    return output
