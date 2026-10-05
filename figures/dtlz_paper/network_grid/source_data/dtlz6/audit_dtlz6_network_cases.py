"""Audit nonlinear ORCA grouping on the five requested DTLZ6 cases.

The audit keeps the benchmark definition and expected labels fixed.  Every
configuration is applied uniformly to all cases and seeds; no case-specific
tuning is performed.  CSV outputs are intended as raw provenance for the
network-grid manuscript figure.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ORCA_REPO = Path(os.environ.get("ORCA_REPO", Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.benchmarks import DTLZ6Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.nonlinear.main import nonlinear_orca


K_TAIL = 10
CASES = ((2, 3), (3, 5), (4, 7), (6, 10), (7, 16))
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class AuditConfig:
    name: str
    initial_point_strategy: str
    num_points_per_seed: int
    active_constraint_tolerance: float | None


CONFIGS = (
    # Configuration used in the current DTLZ grouping follow-up experiments.
    AuditConfig("deterministic_n1_active", "deterministic", 1, 1.0e-8),
    # Uniform point-count checks; these are not selected per case.
    AuditConfig("deterministic_n5_active", "deterministic", 5, 1.0e-8),
    AuditConfig("deterministic_n10_active", "deterministic", 10, 1.0e-8),
    # Matches the optimized seed strategy used by the three-objective audit.
    AuditConfig("optimize_n1_active", "optimize", 1, 1.0e-8),
    # Current config default for constraint handling: aggregate all box faces.
    AuditConfig("deterministic_n1_all_constraints", "deterministic", 1, None),
    AuditConfig("optimize_n1_all_constraints", "optimize", 1, None),
)


def labels_text(labels: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in np.asarray(labels, dtype=int))


def off_diagonal_stats(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return {
        "affinity_min": float(np.min(values)),
        "affinity_max": float(np.max(values)),
        "affinity_mean": float(np.mean(values)),
    }


def within_between_stats(matrix: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    expected = np.asarray(expected, dtype=int)
    within: list[float] = []
    between: list[float] = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            target = within if expected[i] == expected[j] else between
            target.append(float(matrix[i, j]))
    return {
        "within_min": float(np.min(within)) if within else float("nan"),
        "within_max": float(np.max(within)) if within else float("nan"),
        "within_mean": float(np.mean(within)) if within else float("nan"),
        "between_min": float(np.min(between)) if between else float("nan"),
        "between_max": float(np.max(between)) if between else float("nan"),
        "between_mean": float(np.mean(between)) if between else float("nan"),
    }


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    labels = [f"f{i}" for i in range(1, matrix.shape[0] + 1)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def run() -> None:
    rows: list[dict[str, object]] = []
    for config in CONFIGS:
        for intrinsic_dimension, num_objectives in CASES:
            expected = expected_dtlz5_groups(num_objectives, intrinsic_dimension)
            for seed in SEEDS:
                problem = DTLZ6Problem(
                    intrinsic_dimension=intrinsic_dimension,
                    num_objectives=num_objectives,
                    k_tail=K_TAIL,
                    gradient_backend="analytic",
                    initial_point_strategy=config.initial_point_strategy,
                    optimizer_backend="scipy",
                )
                orca_config = NonlinearORCAConfig(
                    num_groups=intrinsic_dimension,
                    grouping_method="average_linkage",
                    random_seed=seed,
                    num_points_per_seed=config.num_points_per_seed,
                    include_seed_points=True,
                    step_size=0.03,
                    active_constraint_tolerance=config.active_constraint_tolerance,
                    max_projection_failures=20,
                )
                start = time.perf_counter()
                result = nonlinear_orca(problem, orca_config)
                elapsed = time.perf_counter() - start
                groups = np.asarray(result.groups, dtype=int)
                matrix = np.asarray(result.adj_matrix, dtype=float)
                row: dict[str, object] = {
                    "configuration": config.name,
                    "initial_point_strategy": config.initial_point_strategy,
                    "num_points_per_seed": config.num_points_per_seed,
                    "active_constraint_tolerance": config.active_constraint_tolerance,
                    "family": "DTLZ6",
                    "intrinsic_dimension": intrinsic_dimension,
                    "num_objectives": num_objectives,
                    "k_tail": K_TAIL,
                    "seed": seed,
                    "selected_points": int(result.input_data.points.shape[0]),
                    "groups": labels_text(groups),
                    "expected_groups": labels_text(expected),
                    "ari": adjusted_rand_index(expected, groups),
                    "exact_match": int(np.array_equal(groups, expected)),
                    "runtime_seconds": elapsed,
                }
                row.update(off_diagonal_stats(matrix))
                row.update(within_between_stats(matrix, expected))
                rows.append(row)

                if config.name == "deterministic_n1_active" and seed == 0:
                    write_matrix(
                        HERE / f"dtlz6_I{intrinsic_dimension}_M{num_objectives}_seed0_affinity.csv",
                        matrix,
                    )

    fieldnames = list(rows[0])
    with (HERE / "dtlz6_network_case_audit_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for config in CONFIGS:
        for intrinsic_dimension, num_objectives in CASES:
            subset = [
                row
                for row in rows
                if row["configuration"] == config.name
                and row["intrinsic_dimension"] == intrinsic_dimension
                and row["num_objectives"] == num_objectives
            ]
            summary_rows.append(
                {
                    "configuration": config.name,
                    "case": f"DTLZ6({intrinsic_dimension},{num_objectives})",
                    "runs": len(subset),
                    "exact_runs": sum(int(row["exact_match"]) for row in subset),
                    "mean_ari": float(np.mean([float(row["ari"]) for row in subset])),
                    "min_ari": float(np.min([float(row["ari"]) for row in subset])),
                    "groups_by_seed": " | ".join(str(row["groups"]) for row in subset),
                    "expected_groups": subset[0]["expected_groups"],
                    "mean_selected_points": float(np.mean([float(row["selected_points"]) for row in subset])),
                    "mean_runtime_seconds": float(np.mean([float(row["runtime_seconds"]) for row in subset])),
                    "min_affinity": float(np.min([float(row["affinity_min"]) for row in subset])),
                    "max_affinity": float(np.max([float(row["affinity_max"]) for row in subset])),
                    "mean_within_affinity": float(np.mean([float(row["within_mean"]) for row in subset])),
                    "mean_between_affinity": float(np.mean([float(row["between_mean"]) for row in subset])),
                }
            )

    with (HERE / "dtlz6_network_case_audit_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    primary_rows: list[dict[str, object]] = []
    primary_name = "deterministic_n1_active"
    for intrinsic_dimension, num_objectives in CASES:
        subset = [
            row
            for row in rows
            if row["configuration"] == primary_name
            and row["intrinsic_dimension"] == intrinsic_dimension
            and row["num_objectives"] == num_objectives
        ]
        seed_zero = next(row for row in subset if row["seed"] == 0)
        primary_rows.append(
            {
                "case": f"DTLZ6({intrinsic_dimension},{num_objectives})",
                "groups_seed0": seed_zero["groups"],
                "expected_groups": seed_zero["expected_groups"],
                "ari_seed0": seed_zero["ari"],
                "exact_seed0": seed_zero["exact_match"],
                "exact_runs_out_of_5": sum(int(row["exact_match"]) for row in subset),
                "selected_points_seed0": seed_zero["selected_points"],
                "affinity_min_seed0": seed_zero["affinity_min"],
                "affinity_max_seed0": seed_zero["affinity_max"],
                "within_mean_seed0": seed_zero["within_mean"],
                "between_mean_seed0": seed_zero["between_mean"],
                "runtime_median_seconds": float(np.median([float(row["runtime_seconds"]) for row in subset])),
                "runtime_mean_seconds": float(np.mean([float(row["runtime_seconds"]) for row in subset])),
            }
        )

    with (HERE / "dtlz6_network_case_primary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(primary_rows[0]))
        writer.writeheader()
        writer.writerows(primary_rows)


if __name__ == "__main__":
    run()
