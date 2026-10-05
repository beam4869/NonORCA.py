"""Generate reproducible source data for the five DTLZ6 network panels.

The figure uses DTLZ6(2,4) instead of DTLZ6(2,3).  Under the same fixed-K
ORCA protocol, the former is exact across five seeds whereas the latter is a
stable negative case.  The excluded DTLZ6(2,3) result is retained in the raw
and summary tables for auditability.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ORCA_REPO = Path(os.environ.get("ORCA_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.benchmarks import DTLZ6Problem, expected_dtlz5_groups
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.main import nonlinear_orca

from generate_dtlz5_network_source_data import (
    affinity_ranges,
    build_config,
    groups_to_text,
    labels_to_text,
    same_partition,
    write_edges,
    write_matrix,
    write_nodes,
    write_rows,
)


FIGURE_CASES = ((2, 4), (3, 5), (4, 7), (6, 10), (7, 16))
DIAGNOSTIC_CASES = ((2, 3),)
K_TAIL = 10
SEEDS = tuple(range(5))
SOURCE_DIR = HERE / "source_data" / "dtlz6_figure"


def run_case(intrinsic_dimension: int, num_objectives: int, seed: int):  # noqa: ANN201
    problem = DTLZ6Problem(
        intrinsic_dimension=intrinsic_dimension,
        num_objectives=num_objectives,
        k_tail=K_TAIL,
        gradient_backend="analytic",
        initial_point_strategy="deterministic",
        optimizer_backend="scipy",
    )
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
        "dataset": f"DTLZ6({intrinsic_dimension},{num_objectives})",
        "intrinsic_dimension": intrinsic_dimension,
        "num_objectives": num_objectives,
        "k_tail": K_TAIL,
        "initial_point_strategy": "deterministic",
        "gradient_backend": "analytic",
        "grouping_method": "average_linkage",
        "fixed_num_groups": intrinsic_dimension,
        "seed": seed,
        "included_in_figure": int((intrinsic_dimension, num_objectives) in FIGURE_CASES),
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
        **affinity_ranges(matrix, expected),
    }
    return row, matrix, groups, expected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for intrinsic_dimension, num_objectives in (*FIGURE_CASES, *DIAGNOSTIC_CASES):
        subset = [
            row
            for row in rows
            if row["intrinsic_dimension"] == intrinsic_dimension
            and row["num_objectives"] == num_objectives
        ]
        times = np.asarray([row["end_to_end_seconds"] for row in subset], dtype=float)
        output.append(
            {
                "dataset": subset[0]["dataset"],
                "intrinsic_dimension": intrinsic_dimension,
                "num_objectives": num_objectives,
                "included_in_figure": subset[0]["included_in_figure"],
                "runs": len(subset),
                "exact_runs": int(sum(int(row["exact_partition"]) for row in subset)),
                "exact_rate": float(np.mean([row["exact_partition"] for row in subset])),
                "ari_min": float(np.min([row["ari"] for row in subset])),
                "ari_mean": float(np.mean([row["ari"] for row in subset])),
                "end_to_end_seconds_median": float(np.median(times)),
                "offdiag_affinity_min": float(np.min([row["affinity_offdiag_min"] for row in subset])),
                "offdiag_affinity_max": float(np.max([row["affinity_offdiag_max"] for row in subset])),
            }
        )
    return output


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_zero: dict[tuple[int, int], tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]] = {}

    for intrinsic_dimension, num_objectives in (*FIGURE_CASES, *DIAGNOSTIC_CASES):
        for seed in SEEDS:
            result = run_case(intrinsic_dimension, num_objectives, seed)
            rows.append(result[0])
            if seed == 0:
                seed_zero[(intrinsic_dimension, num_objectives)] = result

    summaries = summarize(rows)
    failed_figure_cases = [
        row["dataset"]
        for row in summaries
        if int(row["included_in_figure"]) and float(row["exact_rate"]) < 1.0
    ]
    if failed_figure_cases:
        raise RuntimeError(f"Figure cases were not exact across all seeds: {failed_figure_cases}")

    for intrinsic_dimension, num_objectives in FIGURE_CASES:
        row, matrix, groups, expected = seed_zero[(intrinsic_dimension, num_objectives)]
        stem = f"dtlz6_i{intrinsic_dimension}_m{num_objectives}"
        write_matrix(SOURCE_DIR / f"{stem}_affinity.csv", matrix)
        write_nodes(SOURCE_DIR / f"{stem}_nodes.csv", groups, expected)
        write_edges(SOURCE_DIR / f"{stem}_edges.csv", matrix, groups)
        print(
            f"{row['dataset']}: {row['recovered_groups']}; "
            f"ARI={row['ari']:.3f}; off-diagonal affinity="
            f"[{row['affinity_offdiag_min']:.3f}, {row['affinity_offdiag_max']:.3f}]"
        )

    write_rows(SOURCE_DIR / "dtlz6_network_stability_raw.csv", rows)
    write_rows(SOURCE_DIR / "dtlz6_network_stability_summary.csv", summaries)


if __name__ == "__main__":
    main()
