"""Generate reproducible ORCA affinity data for the five DTLZ5 network panels.

The primary figure configuration uses deterministic initial points so that the
affinity matrices are independent of a scalar optimizer and remain visually
well separated.  A second, optimizer-seeded run is recorded as a protocol
cross-check because the nonlinear ORCA workflow normally starts from
single-objective optima.

Both configurations use the standard DTLZ5 definition with ``k_tail=10``,
analytic objective gradients, one generated point per seed, and fixed-K
average-linkage grouping where K equals the known intrinsic dimension I.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.optimize  # noqa: F401  # Import before timing optimizer-seeded runs.


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ORCA_REPO = Path(os.environ.get("ORCA_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.benchmarks import DTLZ5Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.main import nonlinear_orca


CASES = ((2, 3), (3, 5), (4, 7), (6, 10), (7, 16))
K_TAIL = 10
SEEDS = tuple(range(5))
PRIMARY_STRATEGY = "deterministic"
SOURCE_DIR = HERE / "source_data" / "dtlz5"


def labels_to_text(labels: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in labels)


def groups_to_text(labels: Sequence[int]) -> str:
    values = np.asarray(labels, dtype=int)
    blocks = []
    for label in dict.fromkeys(values.tolist()):
        members = [f"f{index + 1}" for index in np.flatnonzero(values == label)]
        blocks.append("{" + ",".join(members) + "}")
    return " | ".join(blocks)


def same_partition(left: Sequence[int], right: Sequence[int]) -> bool:
    left_values = np.asarray(left, dtype=int)
    right_values = np.asarray(right, dtype=int)
    return bool(np.array_equal(left_values[:, None] == left_values[None, :], right_values[:, None] == right_values[None, :]))


def build_problem(intrinsic_dimension: int, num_objectives: int, strategy: str) -> DTLZ5Problem:
    return DTLZ5Problem(
        intrinsic_dimension=intrinsic_dimension,
        num_objectives=num_objectives,
        k_tail=K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy=strategy,
        optimizer_backend="scipy",
    )


def build_config(intrinsic_dimension: int, seed: int) -> NonlinearORCAConfig:
    return NonlinearORCAConfig(
        num_groups=intrinsic_dimension,
        num_points_per_seed=1,
        include_seed_points=True,
        random_seed=seed,
        step_size=0.03,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
        max_projection_failures=20,
    )


def affinity_ranges(matrix: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    within = (expected[:, None] == expected[None, :]) & off_diagonal
    between = (expected[:, None] != expected[None, :]) & off_diagonal
    return {
        "affinity_offdiag_min": float(np.min(matrix[off_diagonal])),
        "affinity_offdiag_max": float(np.max(matrix[off_diagonal])),
        "affinity_within_min": float(np.min(matrix[within])),
        "affinity_within_max": float(np.max(matrix[within])),
        "affinity_between_min": float(np.min(matrix[between])),
        "affinity_between_max": float(np.max(matrix[between])),
    }


def run_case(intrinsic_dimension: int, num_objectives: int, strategy: str, seed: int):  # noqa: ANN201
    problem = build_problem(intrinsic_dimension, num_objectives, strategy)
    config = build_config(intrinsic_dimension, seed)

    point_start = time.perf_counter()
    points = generate_fixed_points(problem, config)
    point_seconds = time.perf_counter() - point_start

    affinity_start = time.perf_counter()
    result = nonlinear_orca(problem, config, points=points)
    affinity_seconds = time.perf_counter() - affinity_start

    expected = expected_dtlz5_groups(num_objectives, intrinsic_dimension)
    groups = np.asarray(result.groups, dtype=int)
    matrix = np.asarray(result.adj_matrix, dtype=float)
    row = {
        "dataset": f"DTLZ5({intrinsic_dimension},{num_objectives})",
        "intrinsic_dimension": intrinsic_dimension,
        "num_objectives": num_objectives,
        "k_tail": K_TAIL,
        "initial_point_strategy": strategy,
        "gradient_backend": "analytic",
        "grouping_method": "average_linkage",
        "fixed_num_groups": intrinsic_dimension,
        "seed": seed,
        "selected_points": int(points.shape[0]),
        "point_generation_seconds": point_seconds,
        "affinity_and_grouping_seconds": affinity_seconds,
        "end_to_end_seconds": point_seconds + affinity_seconds,
        "expected_labels": labels_to_text(expected),
        "recovered_labels": labels_to_text(groups),
        "expected_groups": groups_to_text(expected),
        "recovered_groups": groups_to_text(groups),
        "ari": float(adjusted_rand_index(expected, groups)),
        "exact_partition": int(same_partition(expected, groups)),
        "labels_identical": int(np.array_equal(expected, groups)),
        **affinity_ranges(matrix, expected),
    }
    return row, matrix, groups, expected


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    labels = [f"f{index + 1}" for index in range(matrix.shape[0])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in row]])


def write_nodes(path: Path, groups: np.ndarray, expected: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "recovered_group", "expected_group"])
        for index, (group, expected_group) in enumerate(zip(groups, expected), start=1):
            writer.writerow([f"f{index}", int(group), int(expected_group)])


def write_edges(path: Path, matrix: np.ndarray, groups: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "affinity", "same_recovered_group"])
        for left in range(matrix.shape[0]):
            for right in range(left + 1, matrix.shape[0]):
                writer.writerow(
                    [
                        f"f{left + 1}",
                        f"f{right + 1}",
                        f"{float(matrix[left, right]):.12g}",
                        int(groups[left] == groups[right]),
                    ]
                )


def summarize_stability(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy in ("deterministic", "optimize"):
        for intrinsic_dimension, num_objectives in CASES:
            subset = [
                row
                for row in rows
                if row["initial_point_strategy"] == strategy
                and row["intrinsic_dimension"] == intrinsic_dimension
                and row["num_objectives"] == num_objectives
            ]
            times = np.asarray([row["end_to_end_seconds"] for row in subset], dtype=float)
            summaries.append(
                {
                    "dataset": subset[0]["dataset"],
                    "initial_point_strategy": strategy,
                    "runs": len(subset),
                    "exact_runs": int(sum(int(row["exact_partition"]) for row in subset)),
                    "exact_rate": float(np.mean([row["exact_partition"] for row in subset])),
                    "ari_min": float(np.min([row["ari"] for row in subset])),
                    "ari_mean": float(np.mean([row["ari"] for row in subset])),
                    "selected_points_min": int(np.min([row["selected_points"] for row in subset])),
                    "selected_points_max": int(np.max([row["selected_points"] for row in subset])),
                    "end_to_end_seconds_median": float(np.median(times)),
                    "end_to_end_seconds_min": float(np.min(times)),
                    "end_to_end_seconds_max": float(np.max(times)),
                }
            )
    return summaries


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    primary_rows: list[dict[str, object]] = []
    for intrinsic_dimension, num_objectives in CASES:
        row, matrix, groups, expected = run_case(
            intrinsic_dimension,
            num_objectives,
            PRIMARY_STRATEGY,
            seed=0,
        )
        primary_rows.append(row)
        stem = f"dtlz5_i{intrinsic_dimension}_m{num_objectives}"
        write_matrix(SOURCE_DIR / f"{stem}_affinity.csv", matrix)
        write_nodes(SOURCE_DIR / f"{stem}_nodes.csv", groups, expected)
        write_edges(SOURCE_DIR / f"{stem}_edges.csv", matrix, groups)

    stability_rows: list[dict[str, object]] = []
    for strategy in ("deterministic", "optimize"):
        for intrinsic_dimension, num_objectives in CASES:
            for seed in SEEDS:
                row, _, _, _ = run_case(intrinsic_dimension, num_objectives, strategy, seed)
                stability_rows.append(row)

    write_rows(SOURCE_DIR / "dtlz5_network_cases_seed0.csv", primary_rows)
    write_rows(SOURCE_DIR / "dtlz5_network_stability_raw.csv", stability_rows)
    write_rows(SOURCE_DIR / "dtlz5_network_stability_summary.csv", summarize_stability(stability_rows))

    for row in primary_rows:
        print(
            f"{row['dataset']}: {row['recovered_groups']}; "
            f"ARI={row['ari']:.3f}; exact={row['exact_partition']}; "
            f"off-diagonal affinity=[{row['affinity_offdiag_min']:.3f}, "
            f"{row['affinity_offdiag_max']:.3f}]"
        )


if __name__ == "__main__":
    main()
