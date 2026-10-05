"""Follow-up DTLZ5/DTLZ6 tests for ORCA and gradient-cosine.

Outputs:
1. gradient-cosine and ORCA grouping accuracy on DTLZ5/6 cases.
2. Full NSGA-III vs ORCA+NSGA-III on DTLZ5/6(5,16), comparing mean and max
   reduced-objective aggregation.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from orca.benchmarks import DTLZ5Problem, DTLZ6Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index, gradient_cosine_adjacency
from orca.experiments.downstream_optimizer import (
    _ORCAPymooProblem,
    _evaluate_full_objectives,
    _non_dominated_reference,
    _normalize_against_union,
    _reference_directions,
    _run_nsga3,
)
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.main import nonlinear_orca
from orca.utils.objective_grouping import group_objectives


OUTDIR = Path(os.environ.get("ORCA_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "dtlz")))
I = 5
NUM_GROUPS = 5
MIN_POPULATION_SIZE = 24
N_GEN = 100
GROUPING_SEEDS = 5
DOWNSTREAM_SEEDS = 3
HV_SAMPLES = 50_000
REF_POINT_VALUE = 1.1


@dataclass(frozen=True)
class CaseSpec:
    family: str
    num_objectives: int
    k_tail: int

    @property
    def name(self) -> str:
        return f"{self.family}({I},{self.num_objectives}), k_tail={self.k_tail}"

    def build_problem(self, *, initial_point_strategy: str = "deterministic"):
        cls: Callable[..., object] = DTLZ5Problem if self.family == "DTLZ5" else DTLZ6Problem
        return cls(
            intrinsic_dimension=I,
            num_objectives=self.num_objectives,
            k_tail=self.k_tail,
            gradient_backend="analytic",
            initial_point_strategy=initial_point_strategy,
            optimizer_backend="scipy",
        )


GROUPING_CASES = [
    CaseSpec("DTLZ5", 16, 10),
    CaseSpec("DTLZ6", 16, 10),
    CaseSpec("DTLZ5", 20, 100),
    CaseSpec("DTLZ6", 20, 100),
    CaseSpec("DTLZ5", 20, 500),
    CaseSpec("DTLZ6", 20, 500),
]

DOWNSTREAM_CASES = [
    CaseSpec("DTLZ5", 16, 10),
    CaseSpec("DTLZ6", 16, 10),
]


def labels_to_str(labels: np.ndarray | None) -> str:
    if labels is None:
        return ""
    return " ".join(str(int(value)) for value in np.asarray(labels, dtype=int).tolist())


def selected_points(problem, *, seed: int) -> np.ndarray:  # noqa: ANN001
    cfg = NonlinearORCAConfig(
        num_groups=NUM_GROUPS,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=seed,
        step_size=0.03,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )
    return generate_fixed_points(problem, cfg)


def run_grouping_cases() -> pd.DataFrame:
    rows = []
    for case in GROUPING_CASES:
        expected = expected_dtlz5_groups(case.num_objectives, I)
        for seed in range(GROUPING_SEEDS):
            problem = case.build_problem(initial_point_strategy="deterministic")
            point_start = time.perf_counter()
            points = selected_points(problem, seed=seed)
            point_seconds = time.perf_counter() - point_start

            cfg = NonlinearORCAConfig(
                num_groups=NUM_GROUPS,
                grouping_method="average_linkage",
                active_constraint_tolerance=1.0e-8,
                random_seed=seed,
            )
            orca_start = time.perf_counter()
            orca_result = nonlinear_orca(problem, cfg, points=points)
            orca_seconds = time.perf_counter() - orca_start

            grad_start = time.perf_counter()
            grad_adj = gradient_cosine_adjacency(problem, points)
            grad_groups = group_objectives(grad_adj, NUM_GROUPS, method="average_linkage")
            grad_seconds = time.perf_counter() - grad_start

            for method, groups, seconds in [
                ("ORCA", orca_result.groups, orca_seconds),
                ("gradient_cosine", grad_groups, grad_seconds),
            ]:
                rows.append(
                    {
                        "dataset": case.name,
                        "family": case.family,
                        "num_objectives": case.num_objectives,
                        "k_tail": case.k_tail,
                        "seed": seed,
                        "method": method,
                        "selected_points": int(points.shape[0]),
                        "point_generation_seconds": point_seconds,
                        "grouping_seconds": seconds,
                        "expected_groups": labels_to_str(expected),
                        "groups": labels_to_str(groups),
                        "ari": adjusted_rand_index(expected, groups),
                        "exact_match": int(np.array_equal(np.asarray(expected, dtype=int), np.asarray(groups, dtype=int))),
                    }
                )
    return pd.DataFrame(rows)


def actual_evaluations(result, fallback: int) -> int:  # noqa: ANN001
    try:
        return int(result.algorithm.evaluator.n_eval)
    except Exception:
        return int(fallback)


def max_constraint_violation(problem, X: np.ndarray) -> tuple[float, float]:  # noqa: ANN001
    values = np.vstack([problem.constraint_values(row) for row in np.atleast_2d(X)])
    violations = np.maximum(values, 0.0)
    return float(np.max(violations)), float(np.mean(violations))


def approximate_hv(F_norm: np.ndarray, samples: np.ndarray, *, block_size: int = 4096) -> float:
    F = np.asarray(F_norm, dtype=float)
    dominated_total = 0
    for start in range(0, samples.shape[0], block_size):
        sample_block = samples[start : start + block_size]
        dominated = np.any(np.all(F[None, :, :] <= sample_block[:, None, :], axis=2), axis=1)
        dominated_total += int(np.sum(dominated))
    return float((dominated_total / samples.shape[0]) * (REF_POINT_VALUE ** F.shape[1]))


def empirical_igd(F_norm: np.ndarray, reference_norm: np.ndarray, *, block_size: int = 256) -> float:
    F = np.asarray(F_norm, dtype=float)
    reference = np.asarray(reference_norm, dtype=float)
    distances = []
    for start in range(0, reference.shape[0], block_size):
        ref_block = reference[start : start + block_size]
        diff = ref_block[:, None, :] - F[None, :, :]
        distances.append(np.min(np.linalg.norm(diff, axis=2), axis=1))
    return float(np.mean(np.concatenate(distances)))


def run_downstream_seed(case: CaseSpec, seed: int) -> list[dict[str, object]]:
    problem = case.build_problem(initial_point_strategy="optimize")
    expected = expected_dtlz5_groups(case.num_objectives, I)

    cfg = NonlinearORCAConfig(
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
    orca_result = nonlinear_orca(problem, cfg)
    grouping_seconds = time.perf_counter() - grouping_start
    groups = np.asarray(orca_result.groups, dtype=int)
    ari = adjusted_rand_index(expected, groups)

    full_ref_dirs = _reference_directions(case.num_objectives, MIN_POPULATION_SIZE)
    reduced_ref_dirs = _reference_directions(NUM_GROUPS, MIN_POPULATION_SIZE)
    full_eval_budget = len(full_ref_dirs) * N_GEN
    reduced_n_gen_equal = int(math.ceil(full_eval_budget / len(reduced_ref_dirs)))

    solution_sets: dict[str, dict[str, object]] = {}
    full_problem = _ORCAPymooProblem(problem, groups=None)
    full_result, full_seconds, _ = _run_nsga3(
        full_problem,
        num_objectives=case.num_objectives,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=N_GEN,
        random_seed=seed,
    )
    solution_sets["full_nsga3"] = {
        "result": full_result,
        "X": np.asarray(full_result.X, dtype=float),
        "optimizer_seconds": full_seconds,
        "grouping_seconds": 0.0,
        "n_gen": N_GEN,
        "ref_dirs": len(full_ref_dirs),
        "optimized_objectives": case.num_objectives,
        "aggregation": "none",
    }

    for aggregation in ["mean", "max"]:
        reduced_problem = _ORCAPymooProblem(problem, groups=groups, aggregation=aggregation)
        for budget_label, n_gen in [("natural", N_GEN), ("equal_eval", reduced_n_gen_equal)]:
            reduced_result, reduced_seconds, _ = _run_nsga3(
                reduced_problem,
                num_objectives=NUM_GROUPS,
                min_population_size=MIN_POPULATION_SIZE,
                n_gen=n_gen,
                random_seed=seed,
            )
            solution_sets[f"orca_nsga3_{aggregation}_{budget_label}"] = {
                "result": reduced_result,
                "X": np.asarray(reduced_result.X, dtype=float),
                "optimizer_seconds": reduced_seconds,
                "grouping_seconds": grouping_seconds,
                "n_gen": n_gen,
                "ref_dirs": len(reduced_ref_dirs),
                "optimized_objectives": NUM_GROUPS,
                "aggregation": aggregation,
            }

    full_objective_sets = {
        method: _evaluate_full_objectives(problem, np.asarray(data["X"], dtype=float))
        for method, data in solution_sets.items()
    }
    empirical_reference = _non_dominated_reference(np.vstack(list(full_objective_sets.values())))
    normalized = _normalize_against_union(*list(full_objective_sets.values()), empirical_reference)
    normalized_sets = dict(zip(full_objective_sets.keys(), normalized[:-1]))
    reference_norm = normalized[-1]

    rng = np.random.default_rng(200_000 + 1000 * case.num_objectives + seed)
    hv_samples = rng.uniform(0.0, REF_POINT_VALUE, size=(HV_SAMPLES, case.num_objectives))

    rows = []
    for method, data in solution_sets.items():
        F_norm = normalized_sets[method]
        max_viol, mean_viol = max_constraint_violation(problem, np.asarray(data["X"], dtype=float))
        fallback_evals = int(data["ref_dirs"]) * int(data["n_gen"])
        rows.append(
            {
                "dataset": case.name,
                "family": case.family,
                "seed": seed,
                "method": method,
                "aggregation": data["aggregation"],
                "optimized_objectives": int(data["optimized_objectives"]),
                "ref_dirs": int(data["ref_dirs"]),
                "n_gen": int(data["n_gen"]),
                "evals": actual_evaluations(data["result"], fallback_evals),
                "num_solutions": int(np.asarray(data["X"]).shape[0]),
                "orca_grouping_seconds": float(data["grouping_seconds"]),
                "optimizer_seconds": float(data["optimizer_seconds"]),
                "total_seconds": float(data["grouping_seconds"] + data["optimizer_seconds"]),
                "approx_hv": approximate_hv(F_norm, hv_samples),
                "empirical_igd": empirical_igd(F_norm, reference_norm),
                "max_constraint_violation": max_viol,
                "mean_constraint_violation": mean_viol,
                "expected_groups": labels_to_str(expected),
                "groups": labels_to_str(groups) if method != "full_nsga3" else "",
                "ari_vs_expected": ari if method != "full_nsga3" else np.nan,
            }
        )
    return rows


def run_downstream_cases() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in DOWNSTREAM_CASES:
        for seed in range(DOWNSTREAM_SEEDS):
            print(f"Downstream {case.name} seed {seed + 1}/{DOWNSTREAM_SEEDS}", flush=True)
            rows.extend(run_downstream_seed(case, seed))
    return pd.DataFrame(rows)


def summarize_grouping(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "method"], sort=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            exact_match_rate=("exact_match", "mean"),
            selected_points=("selected_points", "first"),
            grouping_seconds_mean=("grouping_seconds", "mean"),
            groups_mode=("groups", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            n=("ari", "size"),
        )
        .reset_index()
    )


def summarize_downstream(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "method", "aggregation"], sort=False)
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
            ari_vs_expected_mean=("ari_vs_expected", "mean"),
            groups_mode=("groups", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            n=("seed", "size"),
        )
        .reset_index()
    )


def plot_grouping(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary["dataset"].drop_duplicates().tolist()
    methods = ["ORCA", "gradient_cosine"]
    colors = {"ORCA": "#4C78A8", "gradient_cosine": "#F58518"}
    x = np.arange(len(datasets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    for idx, method in enumerate(methods):
        values = [
            float(summary[(summary["dataset"] == dataset) & (summary["method"] == method)]["ari_mean"].iloc[0])
            for dataset in datasets
        ]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=method, color=colors[method])
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("ARI vs known DTLZ grouping")
    ax.set_title("DTLZ5/DTLZ6 grouping accuracy: ORCA vs gradient-cosine")
    ax.set_xticks(x, datasets, rotation=22, ha="right")
    ax.legend(frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_downstream(summary: pd.DataFrame, output: Path) -> None:
    method_order = [
        "full_nsga3",
        "orca_nsga3_mean_natural",
        "orca_nsga3_mean_equal_eval",
        "orca_nsga3_max_natural",
        "orca_nsga3_max_equal_eval",
    ]
    labels = ["Full", "Mean\nnatural", "Mean\nequal", "Max\nnatural", "Max\nequal"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    families = summary["dataset"].drop_duplicates().tolist()

    fig, axes = plt.subplots(len(families), 3, figsize=(13, 7.2), constrained_layout=True)
    for row_idx, dataset in enumerate(families):
        subset = summary[summary["dataset"] == dataset]
        for col_idx, (metric, title, ylabel) in enumerate(
            [
                ("total_seconds_mean", "Runtime", "seconds; lower is better"),
                ("approx_hv_mean", f"Approx. HV ({HV_SAMPLES:,} MC)", "higher is better"),
                ("empirical_igd_mean", "Empirical IGD", "lower is better"),
            ]
        ):
            ax = axes[row_idx, col_idx] if len(families) > 1 else axes[col_idx]
            vals = []
            errs = []
            for method in method_order:
                row = subset[subset["method"] == method]
                vals.append(float(row[metric].iloc[0]))
                if metric == "approx_hv_mean":
                    errs.append(float(row["approx_hv_std"].iloc[0]))
                elif metric == "empirical_igd_mean":
                    errs.append(float(row["empirical_igd_std"].iloc[0]))
                else:
                    errs.append(float(row["total_seconds_std"].iloc[0]))
            ax.bar(np.arange(len(vals)), vals, yerr=errs, capsize=3, color=colors)
            ax.set_title(f"{dataset}\n{title}")
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(len(vals)), labels, rotation=18, ha="right")
    fig.suptitle("DTLZ5 vs DTLZ6 (5,16): downstream NSGA-III quality and time", fontweight="bold")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Running grouping checks...", flush=True)
    grouping_raw = run_grouping_cases()
    grouping_summary = summarize_grouping(grouping_raw)
    grouping_raw.to_csv(OUTDIR / "orca_dtlz5_dtlz6_gradient_grouping_raw.csv", index=False)
    grouping_summary.to_csv(OUTDIR / "orca_dtlz5_dtlz6_gradient_grouping_summary.csv", index=False)
    plot_grouping(grouping_summary, OUTDIR / "orca_dtlz5_dtlz6_gradient_grouping.png")

    print("Running downstream optimizer checks...", flush=True)
    downstream_raw = run_downstream_cases()
    downstream_summary = summarize_downstream(downstream_raw)
    downstream_raw.to_csv(OUTDIR / "orca_dtlz5_dtlz6_m16_aggregation_raw.csv", index=False)
    downstream_summary.to_csv(OUTDIR / "orca_dtlz5_dtlz6_m16_aggregation_summary.csv", index=False)
    plot_downstream(downstream_summary, OUTDIR / "orca_dtlz5_dtlz6_m16_aggregation_comparison.png")

    print("\nGrouping summary")
    print(grouping_summary.to_string(index=False))
    print("\nDownstream summary")
    print(downstream_summary.to_string(index=False))


if __name__ == "__main__":
    main()
