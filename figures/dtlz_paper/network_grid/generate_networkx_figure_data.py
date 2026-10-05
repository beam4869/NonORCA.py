"""Generate the exact DTLZ source data used by the NetworkX-style figures."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ORCA_REPO = Path(os.environ.get("ORCA_REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.benchmarks import DTLZ5Problem, DTLZ6Problem, expected_dtlz5_groups
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


FIGURE_CASES = {
    "DTLZ5": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 12)),
    "DTLZ6": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 12)),
}
DIAGNOSTIC_CASES = (("DTLZ6", 2, 3),)
SEEDS = tuple(range(5))
K_TAIL = 10
SOURCE_DIR = HERE / "source_data" / "networkx_all_labels"


def run_case(family: str, intrinsic_dimension: int, num_objectives: int, seed: int):  # noqa: ANN201
    problem_class = DTLZ5Problem if family == "DTLZ5" else DTLZ6Problem
    problem = problem_class(
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
    recovered = np.asarray(result.groups, dtype=int)
    matrix = np.asarray(result.adj_matrix, dtype=float)
    in_figure = (intrinsic_dimension, num_objectives) in FIGURE_CASES.get(family, ())
    row = {
        "dataset": f"{family}({intrinsic_dimension},{num_objectives})",
        "family": family,
        "intrinsic_dimension": intrinsic_dimension,
        "num_objectives": num_objectives,
        "k_tail": K_TAIL,
        "seed": seed,
        "included_in_figure": int(in_figure),
        "selected_points": int(points.shape[0]),
        "point_generation_seconds": point_seconds,
        "affinity_and_grouping_seconds": affinity_seconds,
        "end_to_end_seconds": point_seconds + affinity_seconds,
        "expected_labels": labels_to_text(expected),
        "recovered_labels": labels_to_text(recovered),
        "expected_groups": groups_to_text(expected),
        "recovered_groups": groups_to_text(recovered),
        "ari": float(adjusted_rand_index(expected, recovered)),
        "exact_partition": int(same_partition(expected, recovered)),
        **affinity_ranges(matrix, expected),
    }
    return row, matrix, recovered, expected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cases = [
        (family, intrinsic_dimension, num_objectives)
        for family, family_cases in FIGURE_CASES.items()
        for intrinsic_dimension, num_objectives in family_cases
    ] + list(DIAGNOSTIC_CASES)
    output: list[dict[str, object]] = []
    for family, intrinsic_dimension, num_objectives in cases:
        subset = [
            row
            for row in rows
            if row["family"] == family
            and row["intrinsic_dimension"] == intrinsic_dimension
            and row["num_objectives"] == num_objectives
        ]
        times = np.asarray([row["end_to_end_seconds"] for row in subset], dtype=float)
        output.append(
            {
                "dataset": subset[0]["dataset"],
                "family": family,
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
    requested_cases = [
        (family, intrinsic_dimension, num_objectives)
        for family, family_cases in FIGURE_CASES.items()
        for intrinsic_dimension, num_objectives in family_cases
    ]
    all_cases = requested_cases + list(DIAGNOSTIC_CASES)
    rows: list[dict[str, object]] = []
    seed_zero: dict[tuple[str, int, int], tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]] = {}

    for family, intrinsic_dimension, num_objectives in all_cases:
        for seed in SEEDS:
            result = run_case(family, intrinsic_dimension, num_objectives, seed)
            rows.append(result[0])
            if seed == 0:
                seed_zero[(family, intrinsic_dimension, num_objectives)] = result

    summaries = summarize(rows)
    failed = [
        row["dataset"]
        for row in summaries
        if int(row["included_in_figure"]) and float(row["exact_rate"]) < 1.0
    ]
    if failed:
        raise RuntimeError(f"Displayed cases were not exact in every seed: {failed}")

    for family, intrinsic_dimension, num_objectives in requested_cases:
        row, matrix, recovered, expected = seed_zero[(family, intrinsic_dimension, num_objectives)]
        stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}"
        write_matrix(SOURCE_DIR / f"{stem}_affinity.csv", matrix)
        write_nodes(SOURCE_DIR / f"{stem}_nodes.csv", recovered, expected)
        write_edges(SOURCE_DIR / f"{stem}_edges.csv", matrix, recovered)
        print(
            f"{row['dataset']}: ARI={row['ari']:.3f}; "
            f"off-diagonal affinity=[{row['affinity_offdiag_min']:.3f}, "
            f"{row['affinity_offdiag_max']:.3f}]"
        )

    write_rows(SOURCE_DIR / "networkx_figure_runs.csv", rows)
    write_rows(SOURCE_DIR / "networkx_figure_case_summary.csv", summaries)


if __name__ == "__main__":
    main()
