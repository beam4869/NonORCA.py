"""Compare active-constraint ORCA against gradient-cosine baselines.

The goal is not to make full Pareto sets. It isolates the objective-structure
step under identical selected feasible points:

* ORCA active: projected/tangent local interactions using active constraints.
* ORCA all constraints: projected/tangent interactions using all constraints.
* gradient-cosine: raw objective-gradient cosine similarity, no constraints.

Constrained DTLZ6 has a known objective grouping. LIR-CMOP has no objective
reduction ground truth, so the script reports adjacency/group differences.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from orca.benchmarks import ConstrainedDTLZ6Problem, LIRCMOPProblem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index, gradient_cosine_adjacency
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.main import nonlinear_orca
from orca.nonlinear.problem_interface import NonlinearORCAProblem
from orca.utils.objective_grouping import group_objectives


OUTPUT_DIR = Path(os.environ.get("ORCA_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "dtlz")))


@dataclass(frozen=True)
class ProblemCase:
    name: str
    problem: NonlinearORCAProblem
    num_groups: int
    expected_groups: np.ndarray | None
    num_points: int
    generation_points_per_seed: int
    generation_step_size: float


def labels_to_str(labels: Sequence[int] | np.ndarray | None) -> str:
    if labels is None:
        return ""
    return " ".join(str(int(value)) for value in np.asarray(labels, dtype=int).tolist())


def matrix_to_str(matrix: np.ndarray) -> str:
    return " ".join(f"{float(value):.6g}" for value in np.asarray(matrix, dtype=float).reshape(-1))


def offdiag_mean(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[0] < 2:
        return 1.0
    return float(np.mean(arr[np.triu_indices_from(arr, k=1)]))


def offdiag_min(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape[0] < 2:
        return 1.0
    return float(np.min(arr[np.triu_indices_from(arr, k=1)]))


def active_constraint_stats(problem: NonlinearORCAProblem, points: np.ndarray, tolerance: float) -> tuple[int, float, float]:
    values = np.asarray([problem.constraint_values(point) for point in points], dtype=float)
    active = values >= -tolerance
    return int(np.sum(active)), float(np.mean(np.sum(active, axis=1))), float(np.max(values))


def selected_points(case: ProblemCase, *, seed: int) -> np.ndarray:
    if case.generation_points_per_seed <= 0:
        points = np.asarray(case.problem.initial_points(), dtype=float)
    else:
        cfg = NonlinearORCAConfig(
            num_groups=case.num_groups,
            num_points_per_seed=case.generation_points_per_seed,
            include_seed_points=True,
            random_seed=seed,
            step_size=case.generation_step_size,
            grouping_method="average_linkage",
            active_constraint_tolerance=1.0e-8,
            max_projection_failures=20,
        )
        points = generate_fixed_points(case.problem, cfg)
    return np.asarray(points[: case.num_points], dtype=float)


def run_case(case: ProblemCase, *, seed: int, active_tolerance: float) -> list[dict[str, object]]:
    points_start = time.perf_counter()
    points = selected_points(case, seed=seed)
    points_seconds = time.perf_counter() - points_start

    active_pairs, active_per_point, max_violation = active_constraint_stats(case.problem, points, active_tolerance)

    cfg_active = NonlinearORCAConfig(
        num_groups=case.num_groups,
        grouping_method="average_linkage",
        active_constraint_tolerance=active_tolerance,
        random_seed=seed,
    )
    cfg_all = NonlinearORCAConfig(
        num_groups=case.num_groups,
        grouping_method="average_linkage",
        active_constraint_tolerance=None,
        random_seed=seed,
    )

    active_start = time.perf_counter()
    active_result = nonlinear_orca(case.problem, cfg_active, points=points)
    active_seconds = time.perf_counter() - active_start

    all_start = time.perf_counter()
    all_result = nonlinear_orca(case.problem, cfg_all, points=points)
    all_seconds = time.perf_counter() - all_start

    grad_start = time.perf_counter()
    grad_adj = gradient_cosine_adjacency(case.problem, points)
    grad_groups = group_objectives(grad_adj, case.num_groups, method="average_linkage")
    grad_seconds = time.perf_counter() - grad_start

    method_rows = [
        ("ORCA_active", active_result.groups, active_result.adj_matrix, active_seconds),
        ("ORCA_all_constraints", all_result.groups, all_result.adj_matrix, all_seconds),
        ("gradient_cosine", grad_groups, grad_adj, grad_seconds),
    ]

    rows = []
    for method, groups, adj, seconds in method_rows:
        rows.append(
            {
                "dataset": case.name,
                "method": method,
                "seed": int(seed),
                "num_objectives": int(case.problem.num_objectives()),
                "num_variables": int(case.problem.num_variables()),
                "num_groups": int(case.num_groups),
                "selected_points": int(points.shape[0]),
                "point_generation_seconds": float(points_seconds),
                "method_seconds": float(seconds),
                "active_constraint_pairs": int(active_pairs),
                "active_constraints_per_point_mean": float(active_per_point),
                "max_constraint_violation": float(max_violation),
                "expected_groups": labels_to_str(case.expected_groups),
                "groups": labels_to_str(groups),
                "ari_vs_expected": "" if case.expected_groups is None else float(adjusted_rand_index(case.expected_groups, groups)),
                "ari_vs_orca_active": float(adjusted_rand_index(active_result.groups, groups)),
                "mean_adjacency": offdiag_mean(adj),
                "min_adjacency": offdiag_min(adj),
                "mean_abs_adj_diff_vs_gradient": float(np.mean(np.abs(active_result.adj_matrix - grad_adj))),
                "max_abs_adj_diff_vs_gradient": float(np.max(np.abs(active_result.adj_matrix - grad_adj))),
                "adjacency_matrix_flat": matrix_to_str(adj),
            }
        )
    return rows


def make_cases() -> list[ProblemCase]:
    dtlz12 = ConstrainedDTLZ6Problem(
        intrinsic_dimension=5,
        num_objectives=12,
        k_tail=100,
        gradient_backend="analytic",
        initial_point_strategy="deterministic",
        optimizer_backend="scipy",
    )
    dtlz20 = ConstrainedDTLZ6Problem(
        intrinsic_dimension=5,
        num_objectives=20,
        k_tail=100,
        gradient_backend="analytic",
        initial_point_strategy="deterministic",
        optimizer_backend="scipy",
    )

    return [
        ProblemCase(
            name="Constrained-DTLZ6(5,12), k_tail=100",
            problem=dtlz12,
            num_groups=5,
            expected_groups=expected_dtlz5_groups(12, 5),
            num_points=36,
            generation_points_per_seed=2,
            generation_step_size=0.03,
        ),
        ProblemCase(
            name="Constrained-DTLZ6(5,20), k_tail=100",
            problem=dtlz20,
            num_groups=5,
            expected_groups=expected_dtlz5_groups(20, 5),
            num_points=40,
            generation_points_per_seed=1,
            generation_step_size=0.03,
        ),
        ProblemCase(
            name="LIR-CMOP5",
            problem=LIRCMOPProblem("LIRCMOP5", num_variables=30),
            num_groups=2,
            expected_groups=None,
            num_points=4,
            generation_points_per_seed=0,
            generation_step_size=0.0,
        ),
        ProblemCase(
            name="LIR-CMOP6",
            problem=LIRCMOPProblem("LIRCMOP6", num_variables=30),
            num_groups=2,
            expected_groups=None,
            num_points=4,
            generation_points_per_seed=0,
            generation_step_size=0.0,
        ),
        ProblemCase(
            name="LIR-CMOP13",
            problem=LIRCMOPProblem("LIRCMOP13", num_variables=30),
            num_groups=2,
            expected_groups=None,
            num_points=4,
            generation_points_per_seed=0,
            generation_step_size=0.0,
        ),
        ProblemCase(
            name="LIR-CMOP14",
            problem=LIRCMOPProblem("LIRCMOP14", num_variables=30),
            num_groups=2,
            expected_groups=None,
            num_points=4,
            generation_points_per_seed=0,
            generation_step_size=0.0,
        ),
    ]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in df.groupby(["dataset", "method"], sort=False):
        ari_vals = pd.to_numeric(group["ari_vs_expected"], errors="coerce")
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": int(group.shape[0]),
                "selected_points": int(group["selected_points"].iloc[0]),
                "method_seconds_mean": float(group["method_seconds"].mean()),
                "method_seconds_std": float(group["method_seconds"].std(ddof=1)) if group.shape[0] > 1 else 0.0,
                "active_constraints_per_point_mean": float(group["active_constraints_per_point_mean"].mean()),
                "groups_mode": group["groups"].mode().iloc[0],
                "ari_vs_expected_mean": "" if ari_vals.isna().all() else float(ari_vals.mean()),
                "ari_vs_orca_active_mean": float(group["ari_vs_orca_active"].mean()),
                "mean_adjacency_mean": float(group["mean_adjacency"].mean()),
                "min_adjacency_mean": float(group["min_adjacency"].mean()),
                "mean_abs_adj_diff_vs_gradient": float(group["mean_abs_adj_diff_vs_gradient"].mean()),
                "max_abs_adj_diff_vs_gradient": float(group["max_abs_adj_diff_vs_gradient"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    method_order = ["ORCA_active", "ORCA_all_constraints", "gradient_cosine"]
    colors = {"ORCA_active": "#16a34a", "ORCA_all_constraints": "#3b82f6", "gradient_cosine": "#f97316"}

    dtlz = summary[summary["dataset"].str.startswith("Constrained-DTLZ6")].copy()
    x = np.arange(dtlz["dataset"].nunique())
    width = 0.24
    for idx, method in enumerate(method_order):
        subset = dtlz[dtlz["method"] == method]
        axes[0].bar(x + (idx - 1) * width, subset["ari_vs_expected_mean"].astype(float), width=width, label=method, color=colors[method])
    axes[0].set_xticks(x, dtlz["dataset"].drop_duplicates().str.replace(", k_tail=100", "", regex=False), rotation=15, ha="right")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("ARI vs known DTLZ6 grouping")
    axes[0].set_title("Known-Group Recovery")
    axes[0].grid(axis="y", alpha=0.25)

    lirs = summary[summary["dataset"].str.startswith("LIR")].copy()
    for method in method_order:
        subset = lirs[lirs["method"] == method]
        axes[1].plot(subset["dataset"], subset["mean_adjacency_mean"], marker="o", label=method, color=colors[method])
    axes[1].set_ylabel("mean off-diagonal adjacency")
    axes[1].set_title("LIR-CMOP Objective Similarity")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25)

    active_only = summary[summary["method"] == "ORCA_active"].copy()
    axes[2].barh(active_only["dataset"], active_only["max_abs_adj_diff_vs_gradient"], color="#64748b")
    axes[2].set_xlabel("max |ORCA active adjacency - gradient cosine|")
    axes[2].set_title("Constraint-Induced Difference")
    axes[2].grid(axis="x", alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seeds = [0, 1, 2]
    for case in make_cases():
        case_seeds = seeds if case.name.startswith("Constrained-DTLZ6") else [0]
        for seed in case_seeds:
            rows.extend(run_case(case, seed=seed, active_tolerance=1.0e-8))
            print(f"finished {case.name}, seed={seed}", flush=True)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw_path = OUTPUT_DIR / "orca_active_constraints_vs_gradient_raw.csv"
    summary_path = OUTPUT_DIR / "orca_active_constraints_vs_gradient_summary.csv"
    fig_path = OUTPUT_DIR / "orca_active_constraints_vs_gradient.png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, fig_path)

    print("Summary:")
    print(summary.to_string(index=False))
    print("Wrote:")
    print(raw_path)
    print(summary_path)
    print(fig_path)


if __name__ == "__main__":
    main()
