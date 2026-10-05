"""Compare ORCA+NSGA-III against full NSGA-III on DTLZ5(5,16).

The script evaluates all returned solutions in the original 16-objective space.
HV is Monte-Carlo approximated because exact high-dimensional HV can be slow and
unstable for quick iteration.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from orca.benchmarks import DTLZ5Problem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.experiments.baseline_comparison import adjusted_rand_index
from orca.experiments.downstream_optimizer import (
    _ORCAPymooProblem,
    _evaluate_full_objectives,
    _non_dominated_reference,
    _normalize_against_union,
    _reference_directions,
    _run_nsga3,
)
from orca.nonlinear.main import nonlinear_orca


OUTDIR = Path(os.environ.get("ORCA_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "dtlz")))
M = 16
I = 5
K_TAIL = 10
NUM_GROUPS = I
MIN_POPULATION_SIZE = 24
N_GEN_FULL = 100
N_SEEDS = 5
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


def max_constraint_violation(problem: DTLZ5Problem, X: np.ndarray) -> tuple[float, float]:
    values = np.vstack([problem.constraint_values(row) for row in np.atleast_2d(X)])
    violations = np.maximum(values, 0.0)
    return float(np.max(violations)), float(np.mean(violations))


def approximate_hv(
    F_norm: np.ndarray,
    samples: np.ndarray,
    *,
    ref_point_value: float = REF_POINT_VALUE,
    block_size: int = 4096,
) -> float:
    """Monte-Carlo dominated volume under a shared sample set."""

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
    problem = DTLZ5Problem(
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

    full_problem = _ORCAPymooProblem(problem, groups=None)
    reduced_problem = _ORCAPymooProblem(problem, groups=groups, aggregation="mean")

    full_result, full_optimizer_seconds, _ = _run_nsga3(
        full_problem,
        num_objectives=M,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=N_GEN_FULL,
        random_seed=seed,
    )
    natural_result, natural_optimizer_seconds, _ = _run_nsga3(
        reduced_problem,
        num_objectives=NUM_GROUPS,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=N_GEN_FULL,
        random_seed=seed,
    )
    equal_result, equal_optimizer_seconds, _ = _run_nsga3(
        reduced_problem,
        num_objectives=NUM_GROUPS,
        min_population_size=MIN_POPULATION_SIZE,
        n_gen=reduced_n_gen_equal,
        random_seed=seed,
    )

    solution_sets = {
        "NSGAIII_full": {
            "result": full_result,
            "X": np.asarray(full_result.X, dtype=float),
            "optimizer_seconds": full_optimizer_seconds,
            "grouping_seconds": 0.0,
            "n_gen": N_GEN_FULL,
            "ref_dirs": len(full_ref_dirs),
            "optimized_objectives": M,
            "groups": None,
            "ari": np.nan,
        },
        "ORCA_NSGAIII_reduced_natural_budget": {
            "result": natural_result,
            "X": np.asarray(natural_result.X, dtype=float),
            "optimizer_seconds": natural_optimizer_seconds,
            "grouping_seconds": grouping_seconds,
            "n_gen": N_GEN_FULL,
            "ref_dirs": len(reduced_ref_dirs),
            "optimized_objectives": NUM_GROUPS,
            "groups": groups,
            "ari": ari,
        },
        "ORCA_NSGAIII_reduced_equal_eval": {
            "result": equal_result,
            "X": np.asarray(equal_result.X, dtype=float),
            "optimizer_seconds": equal_optimizer_seconds,
            "grouping_seconds": grouping_seconds,
            "n_gen": reduced_n_gen_equal,
            "ref_dirs": len(reduced_ref_dirs),
            "optimized_objectives": NUM_GROUPS,
            "groups": groups,
            "ari": ari,
        },
    }

    full_objective_sets = {
        method: _evaluate_full_objectives(problem, data["X"]) for method, data in solution_sets.items()
    }
    empirical_reference = _non_dominated_reference(np.vstack(list(full_objective_sets.values())))
    normalized = _normalize_against_union(*list(full_objective_sets.values()), empirical_reference)
    normalized_sets = dict(zip(full_objective_sets.keys(), normalized[:-1]))
    reference_norm = normalized[-1]

    rng = np.random.default_rng(100_000 + seed)
    hv_samples = rng.uniform(0.0, REF_POINT_VALUE, size=(HV_SAMPLES, M))

    rows: list[dict[str, object]] = []
    for method, data in solution_sets.items():
        F_norm = normalized_sets[method]
        max_viol, mean_viol = max_constraint_violation(problem, data["X"])
        fallback_evals = int(data["ref_dirs"]) * int(data["n_gen"])
        rows.append(
            {
                "dataset": f"DTLZ5({I},{M}), k_tail={K_TAIL}",
                "seed": seed,
                "method": method,
                "optimized_objectives": int(data["optimized_objectives"]),
                "ref_dirs": int(data["ref_dirs"]),
                "n_gen": int(data["n_gen"]),
                "evals": actual_evaluations(data["result"], fallback_evals),
                "num_solutions": int(data["X"].shape[0]),
                "orca_grouping_seconds": float(data["grouping_seconds"]),
                "optimizer_seconds": float(data["optimizer_seconds"]),
                "total_seconds": float(data["grouping_seconds"] + data["optimizer_seconds"]),
                "approx_hv": approximate_hv(F_norm, hv_samples),
                "empirical_igd": empirical_igd(F_norm, reference_norm),
                "max_constraint_violation": max_viol,
                "mean_constraint_violation": mean_viol,
                "expected_groups": labels_to_str(expected_groups),
                "groups": labels_to_str(data["groups"]),
                "ari_vs_expected": data["ari"],
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in raw.groupby("method", sort=False):
        rows.append(
            {
                "dataset": group["dataset"].iloc[0],
                "method": method,
                "optimized_objectives": int(group["optimized_objectives"].iloc[0]),
                "ref_dirs": int(group["ref_dirs"].iloc[0]),
                "n_gen_mean": float(group["n_gen"].mean()),
                "evals_mean": float(group["evals"].mean()),
                "approx_hv_mean": float(group["approx_hv"].mean()),
                "approx_hv_std": float(group["approx_hv"].std(ddof=0)),
                "empirical_igd_mean": float(group["empirical_igd"].mean()),
                "empirical_igd_std": float(group["empirical_igd"].std(ddof=0)),
                "optimizer_seconds_mean": float(group["optimizer_seconds"].mean()),
                "orca_grouping_seconds_mean": float(group["orca_grouping_seconds"].mean()),
                "total_seconds_mean": float(group["total_seconds"].mean()),
                "total_seconds_std": float(group["total_seconds"].std(ddof=0)),
                "num_solutions_mean": float(group["num_solutions"].mean()),
                "max_constraint_violation": float(group["max_constraint_violation"].max()),
                "ari_vs_expected_mean": float(pd.to_numeric(group["ari_vs_expected"], errors="coerce").mean()),
                "groups_mode": group["groups"].mode().iloc[0] if not group["groups"].mode().empty else "",
                "num_seeds": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    labels = ["Full NSGA-III", "ORCA reduced\nnatural", "ORCA reduced\nequal eval"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    axes[0].bar(x, summary["total_seconds_mean"], color=colors)
    axes[0].set_title("Runtime (s)")
    axes[0].set_ylabel("lower is better")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")

    axes[1].bar(x, summary["approx_hv_mean"], yerr=summary["approx_hv_std"], color=colors, capsize=3)
    axes[1].set_title(f"Approx. HV ({HV_SAMPLES:,} MC samples)")
    axes[1].set_ylabel("higher is better")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")

    axes[2].bar(x, summary["empirical_igd_mean"], yerr=summary["empirical_igd_std"], color=colors, capsize=3)
    axes[2].set_title("Empirical IGD")
    axes[2].set_ylabel("lower is better")
    axes[2].set_xticks(x, labels, rotation=15, ha="right")

    fig.suptitle("DTLZ5(5,16): ORCA+NSGA-III vs full NSGA-III", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for seed in range(N_SEEDS):
        print(f"Running seed {seed + 1}/{N_SEEDS}...")
        all_rows.extend(run_seed(seed))

    raw = pd.DataFrame(all_rows)
    summary = summarize(raw)

    raw_path = OUTDIR / "orca_dtlz5_5_16_nsga3_raw.csv"
    summary_path = OUTDIR / "orca_dtlz5_5_16_nsga3_summary.csv"
    figure_path = OUTDIR / "orca_dtlz5_5_16_nsga3_comparison.png"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, figure_path)

    print(summary.to_string(index=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
