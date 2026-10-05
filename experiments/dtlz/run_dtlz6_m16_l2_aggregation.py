"""Test L2 aggregation for the reducible DTLZ6 objective block.

For DTLZ6(5,16), the known ORCA group is:
    {f1, ..., f12}, {f13}, {f14}, {f15}, {f16}

This script compares full-objective NSGA-III against ORCA-reduced NSGA-III
using mean, max, and L2 aggregation for the first group. All returned solutions
are evaluated in the original 16-objective space before HV/IGD are computed.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from orca.benchmarks import DTLZ6Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.experiments.downstream_optimizer import (
    _evaluate_full_objectives,
    _non_dominated_reference,
    _normalize_against_union,
    _reference_directions,
    _run_nsga3,
    aggregate_objectives_by_group,
)
from orca.nonlinear.main import nonlinear_orca


OUTDIR = Path(os.environ.get("ORCA_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "dtlz")))
M = 16
I = 5
K_TAIL = 10
NUM_GROUPS = I
MIN_POPULATION_SIZE = 24
N_GEN_FULL = 100
N_SEEDS = 3
HV_SAMPLES = 50_000
REF_POINT_VALUE = 1.1


def labels_to_str(labels: np.ndarray | None) -> str:
    if labels is None:
        return ""
    return " ".join(str(int(value)) for value in np.asarray(labels, dtype=int).tolist())


def actual_evaluations(result, fallback: int) -> int:  # noqa: ANN001
    try:
        return int(result.algorithm.evaluator.n_eval)
    except Exception:
        return int(fallback)


def aggregate_l2_by_group(objective_values: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    values = np.asarray(objective_values, dtype=float)
    was_1d = values.ndim == 1
    if was_1d:
        values = values[None, :]

    labels = np.asarray(groups, dtype=int)
    if values.shape[1] != labels.size:
        raise ValueError("groups must have one label per objective")

    ordered_labels = list(dict.fromkeys(labels.tolist()))
    reduced = np.zeros((values.shape[0], len(ordered_labels)), dtype=float)
    for group_idx, label in enumerate(ordered_labels):
        cols = values[:, labels == label]
        reduced[:, group_idx] = np.sqrt(np.sum(cols**2, axis=1))
    return reduced[0] if was_1d else reduced


class ReducedPymooProblem:
    """Lazy pymoo wrapper with mean/max/L2 grouped objectives."""

    def __new__(
        cls,
        problem: DTLZ6Problem,
        *,
        groups: Sequence[int] | None = None,
        aggregation: str = "mean",
    ):
        try:
            from pymoo.core.problem import Problem  # type: ignore
        except ImportError as exc:
            raise ImportError("Downstream optimizer experiments require pymoo") from exc

        lower = np.asarray(getattr(problem, "lower_bounds", np.zeros(problem.num_variables())), dtype=float)
        upper = np.asarray(getattr(problem, "upper_bounds", np.ones(problem.num_variables())), dtype=float)
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
                elif aggregation == "l2":
                    out["F"] = aggregate_l2_by_group(full_values, groups)
                else:
                    out["F"] = aggregate_objectives_by_group(full_values, groups, method=aggregation)
                out["G"] = np.vstack([problem.constraint_values(row) for row in rows])

        return WrappedProblem()


def max_constraint_violation(problem: DTLZ6Problem, X: np.ndarray) -> tuple[float, float]:
    values = np.vstack([problem.constraint_values(row) for row in np.atleast_2d(X)])
    violations = np.maximum(values, 0.0)
    return float(np.max(violations)), float(np.mean(violations))


def g_statistics(problem: DTLZ6Problem, X: np.ndarray) -> dict[str, float]:
    values = np.array([problem._g_value(row) for row in np.atleast_2d(X)], dtype=float)  # noqa: SLF001
    return {
        "g_min": float(np.min(values)),
        "g_median": float(np.median(values)),
        "g_mean": float(np.mean(values)),
        "g_max": float(np.max(values)),
    }


def approximate_hv(
    F_norm: np.ndarray,
    samples: np.ndarray,
    *,
    ref_point_value: float = REF_POINT_VALUE,
    block_size: int = 4096,
) -> float:
    F = np.asarray(F_norm, dtype=float)
    dominated_total = 0
    for start in range(0, samples.shape[0], block_size):
        sample_block = samples[start : start + block_size]
        dominated = np.any(np.all(F[None, :, :] <= sample_block[:, None, :], axis=2), axis=1)
        dominated_total += int(np.sum(dominated))
    return float((dominated_total / samples.shape[0]) * (ref_point_value ** F.shape[1]))


def empirical_igd(F_norm: np.ndarray, reference_norm: np.ndarray, *, block_size: int = 256) -> float:
    F = np.asarray(F_norm, dtype=float)
    reference = np.asarray(reference_norm, dtype=float)
    distances = []
    for start in range(0, reference.shape[0], block_size):
        ref_block = reference[start : start + block_size]
        diff = ref_block[:, None, :] - F[None, :, :]
        distances.append(np.min(np.linalg.norm(diff, axis=2), axis=1))
    return float(np.mean(np.concatenate(distances)))


def run_seed(seed: int) -> list[dict[str, object]]:
    problem = DTLZ6Problem(
        intrinsic_dimension=I,
        num_objectives=M,
        k_tail=K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy="optimize",
        optimizer_backend="scipy",
    )
    expected_groups = expected_dtlz5_groups(M, I)

    orca_cfg = NonlinearORCAConfig(
        num_groups=NUM_GROUPS,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=seed,
        step_size=0.03,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )
    grouping_start = time.perf_counter()
    orca_result = nonlinear_orca(problem, orca_cfg)
    grouping_seconds = time.perf_counter() - grouping_start
    groups = np.asarray(orca_result.groups, dtype=int)
    ari = adjusted_rand_index(expected_groups, groups)

    full_ref_dirs = _reference_directions(M, MIN_POPULATION_SIZE)
    reduced_ref_dirs = _reference_directions(NUM_GROUPS, MIN_POPULATION_SIZE)
    full_eval_budget = len(full_ref_dirs) * N_GEN_FULL
    reduced_n_gen_equal = int(math.ceil(full_eval_budget / len(reduced_ref_dirs)))

    solution_sets: dict[str, dict[str, object]] = {}

    full_problem = ReducedPymooProblem(problem, groups=None)
    full_result, full_optimizer_seconds, _ = _run_nsga3(
        full_problem,
        num_objectives=M,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=N_GEN_FULL,
        random_seed=seed,
    )
    solution_sets["full_nsga3"] = {
        "result": full_result,
        "X": np.asarray(full_result.X, dtype=float),
        "optimizer_seconds": full_optimizer_seconds,
        "grouping_seconds": 0.0,
        "n_gen": N_GEN_FULL,
        "ref_dirs": len(full_ref_dirs),
        "optimized_objectives": M,
        "aggregation": "none",
        "budget": "full",
    }

    for aggregation in ["mean", "max", "l2"]:
        reduced_problem = ReducedPymooProblem(problem, groups=groups, aggregation=aggregation)
        for budget_label, n_gen in [("natural", N_GEN_FULL), ("equal_eval", reduced_n_gen_equal)]:
            reduced_result, reduced_optimizer_seconds, _ = _run_nsga3(
                reduced_problem,
                num_objectives=NUM_GROUPS,
                min_population_size=MIN_POPULATION_SIZE,
                n_gen=n_gen,
                random_seed=seed,
            )
            solution_sets[f"orca_nsga3_{aggregation}_{budget_label}"] = {
                "result": reduced_result,
                "X": np.asarray(reduced_result.X, dtype=float),
                "optimizer_seconds": reduced_optimizer_seconds,
                "grouping_seconds": grouping_seconds,
                "n_gen": n_gen,
                "ref_dirs": len(reduced_ref_dirs),
                "optimized_objectives": NUM_GROUPS,
                "aggregation": aggregation,
                "budget": budget_label,
            }

    full_objective_sets = {
        method: _evaluate_full_objectives(problem, data["X"]) for method, data in solution_sets.items()
    }
    empirical_reference = _non_dominated_reference(np.vstack(list(full_objective_sets.values())))
    normalized = _normalize_against_union(*list(full_objective_sets.values()), empirical_reference)
    normalized_sets = dict(zip(full_objective_sets.keys(), normalized[:-1]))
    reference_norm = normalized[-1]

    rng = np.random.default_rng(300_000 + seed)
    hv_samples = rng.uniform(0.0, REF_POINT_VALUE, size=(HV_SAMPLES, M))

    rows: list[dict[str, object]] = []
    for method, data in solution_sets.items():
        X = np.asarray(data["X"], dtype=float)
        F_norm = normalized_sets[method]
        max_viol, mean_viol = max_constraint_violation(problem, X)
        fallback_evals = int(data["ref_dirs"]) * int(data["n_gen"])
        stats = g_statistics(problem, X)
        rows.append(
            {
                "dataset": f"DTLZ6({I},{M}), k_tail={K_TAIL}",
                "seed": seed,
                "method": method,
                "aggregation": data["aggregation"],
                "budget": data["budget"],
                "optimized_objectives": int(data["optimized_objectives"]),
                "ref_dirs": int(data["ref_dirs"]),
                "n_gen": int(data["n_gen"]),
                "evals": actual_evaluations(data["result"], fallback_evals),
                "num_solutions": int(X.shape[0]),
                "orca_grouping_seconds": float(data["grouping_seconds"]),
                "optimizer_seconds": float(data["optimizer_seconds"]),
                "total_seconds": float(data["grouping_seconds"] + data["optimizer_seconds"]),
                "approx_hv": approximate_hv(F_norm, hv_samples),
                "empirical_igd": empirical_igd(F_norm, reference_norm),
                "max_constraint_violation": max_viol,
                "mean_constraint_violation": mean_viol,
                "expected_groups": labels_to_str(expected_groups),
                "groups": "" if method == "full_nsga3" else labels_to_str(groups),
                "ari_vs_expected": np.nan if method == "full_nsga3" else ari,
                **stats,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["dataset", "method", "aggregation", "budget"], sort=False)
        .agg(
            optimized_objectives=("optimized_objectives", "first"),
            ref_dirs=("ref_dirs", "first"),
            n_gen_mean=("n_gen", "mean"),
            evals_mean=("evals", "mean"),
            approx_hv_mean=("approx_hv", "mean"),
            approx_hv_std=("approx_hv", lambda s: float(np.std(s, ddof=0))),
            empirical_igd_mean=("empirical_igd", "mean"),
            empirical_igd_std=("empirical_igd", lambda s: float(np.std(s, ddof=0))),
            optimizer_seconds_mean=("optimizer_seconds", "mean"),
            orca_grouping_seconds_mean=("orca_grouping_seconds", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_std=("total_seconds", lambda s: float(np.std(s, ddof=0))),
            num_solutions_mean=("num_solutions", "mean"),
            max_constraint_violation=("max_constraint_violation", "max"),
            mean_constraint_violation=("mean_constraint_violation", "mean"),
            g_median_mean=("g_median", "mean"),
            g_mean_mean=("g_mean", "mean"),
            g_max_mean=("g_max", "mean"),
            ari_vs_expected_mean=("ari_vs_expected", "mean"),
            groups_mode=("groups", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            n=("seed", "size"),
        )
        .reset_index()
    )


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    method_order = [
        "full_nsga3",
        "orca_nsga3_mean_natural",
        "orca_nsga3_mean_equal_eval",
        "orca_nsga3_max_natural",
        "orca_nsga3_max_equal_eval",
        "orca_nsga3_l2_natural",
        "orca_nsga3_l2_equal_eval",
    ]
    labels = ["Full", "Mean\nnat.", "Mean\nequal", "Max\nnat.", "Max\nequal", "L2\nnat.", "L2\nequal"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#FF9DA6"]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), constrained_layout=True)
    for ax, metric, title, ylabel, err_col in [
        (axes[0], "total_seconds_mean", "Runtime", "seconds; lower is better", "total_seconds_std"),
        (axes[1], "approx_hv_mean", f"Approx. HV ({HV_SAMPLES:,} MC)", "higher is better", "approx_hv_std"),
        (axes[2], "empirical_igd_mean", "Empirical IGD", "lower is better", "empirical_igd_std"),
        (axes[3], "g_median_mean", "Median g(x)", "lower is closer to PF", None),
    ]:
        vals = []
        errs = []
        for method in method_order:
            row = summary[summary["method"] == method]
            vals.append(float(row[metric].iloc[0]))
            errs.append(0.0 if err_col is None else float(row[err_col].iloc[0]))
        ax.bar(np.arange(len(vals)), vals, yerr=errs if err_col else None, capsize=3, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=18, ha="right")

    fig.suptitle("DTLZ6(5,16): L2 group aggregation for ORCA-reduced NSGA-III", fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in range(N_SEEDS):
        print(f"Running DTLZ6(5,16) L2 aggregation seed {seed + 1}/{N_SEEDS}", flush=True)
        rows.extend(run_seed(seed))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)

    raw_path = OUTDIR / "orca_dtlz6_m16_l2_aggregation_raw.csv"
    summary_path = OUTDIR / "orca_dtlz6_m16_l2_aggregation_summary.csv"
    figure_path = OUTDIR / "orca_dtlz6_m16_l2_aggregation_comparison.png"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, figure_path)

    print(summary.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
