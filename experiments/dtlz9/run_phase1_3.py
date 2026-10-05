"""Run DTLZ9 Phase 1-3 validation and grouping-recovery experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.dtlz9.geometry import (
    cosine_matrix,
    edge_summary,
    expected_signed_matrix,
    fixed_k_groups,
    grouping_metrics,
    joint_signed_matrix,
    matrices_from_sampled_blocks,
    positive_component_groups,
    raw_signed_matrix,
    sampled_blocks,
    value_correlation_matrices,
)
from experiments.dtlz9.problem import DTLZ9Problem
from experiments.dtlz9 import orca_bridge as _orca_bridge  # noqa: F401
from orca.config import NonlinearORCAConfig
from orca.nonlinear.fixed_point_generation import generate_fixed_points


METHOD_ORDER = [
    "Raw-Cosine",
    "Single-Constraint-Mean",
    "ORCA-current",
    "ORCA-joint",
    "Pearson",
    "Spearman",
]


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denom)


def _finite_difference_jacobian(function, x: np.ndarray, h: float = 1.0e-7) -> np.ndarray:
    base = np.asarray(function(x), dtype=float)
    jac = np.zeros((base.size, x.size), dtype=float)
    for idx in range(x.size):
        step = np.zeros_like(x)
        step[idx] = h
        jac[:, idx] = (np.asarray(function(x + step)) - np.asarray(function(x - step))) / (2.0 * h)
    return jac


def _matrix_to_long(matrix: np.ndarray, method: str, m: int, block_size: int) -> list[dict[str, object]]:
    rows = []
    for i in range(m):
        for j in range(m):
            rows.append(
                {
                    "M": m,
                    "block_size": block_size,
                    "method": method,
                    "objective_i": i + 1,
                    "objective_j": j + 1,
                    "signed_strength": float(matrix[i, j]),
                    "adjacency": float(0.5 * (1.0 + matrix[i, j])),
                }
            )
    return rows


def run_phase2(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    geometry_rows: list[dict[str, object]] = []
    derivative_rows: list[dict[str, object]] = []
    invariance_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    for m in (3, 5, 10):
        for block_size in (1, 10):
            problem = DTLZ9Problem(m, m * block_size)
            theta_values = np.linspace(0.15, 0.5 * np.pi - 0.15, 25)
            points = problem.exact_pf_decisions(theta_values)
            target = expected_signed_matrix(m)

            for theta, x in zip(theta_values, points):
                gradients = problem.objective_gradients(x)
                jacobian = problem.constraint_jacobian(x)
                raw = raw_signed_matrix(gradients)
                joint, diagnostics = joint_signed_matrix(gradients, jacobian)
                offdiag = raw[~np.eye(m, dtype=bool)]
                geometry_rows.append(
                    {
                        "M": m,
                        "block_size": block_size,
                        "n": problem.num_variables(),
                        "theta": float(theta),
                        "pf_constraint_max_abs": float(np.max(np.abs(problem.constraint_values(x)))),
                        "raw_offdiag_max_abs": float(np.nanmax(np.abs(offdiag))),
                        "joint_matrix_max_abs_error": float(np.nanmax(np.abs(joint - target))),
                        "joint_matrix_median_abs_error": float(np.nanmedian(np.abs(joint - target))),
                        "projector_rank": diagnostics.rank,
                        "expected_rank": m - 1,
                        "projector_symmetry_residual": diagnostics.symmetry_residual,
                        "projector_idempotence_residual": diagnostics.idempotence_residual,
                        "projector_tangent_residual": diagnostics.tangent_residual,
                        "active_jacobian_condition": diagnostics.condition_number,
                        "singular_values": " ".join(f"{value:.16g}" for value in diagnostics.singular_values),
                        "gradient_norm_min": float(np.min(np.linalg.norm(gradients, axis=1))),
                        "gradient_norm_max": float(np.max(np.linalg.norm(gradients, axis=1))),
                        "variables_below_1e-8": int(np.sum(x <= 1.0e-8)),
                    }
                )

            if block_size == 1:
                x_mid = problem.exact_pf_decision(np.pi / 4.0)
                objective_fd = _finite_difference_jacobian(problem.objective_values, x_mid)
                constraint_fd = _finite_difference_jacobian(problem.constraint_values, x_mid)
                derivative_rows.append(
                    {
                        "M": m,
                        "block_size": block_size,
                        "theta": float(np.pi / 4.0),
                        "objective_jacobian_relative_error": _relative_error(
                            objective_fd, problem.objective_gradients(x_mid)
                        ),
                        "constraint_jacobian_relative_error": _relative_error(
                            constraint_fd, problem.constraint_jacobian(x_mid)
                        ),
                    }
                )

            blocks = sampled_blocks(problem, points)
            matrices, _ = matrices_from_sampled_blocks(blocks)
            values = np.vstack([problem.objective_values(point) for point in points])
            matrices.update(value_correlation_matrices(values))
            matrices["Oracle-Joint-Tangent"] = target
            if m == 5 and block_size == 10:
                for method, matrix in matrices.items():
                    representative_rows.extend(_matrix_to_long(matrix, method, m, block_size))

            rng = np.random.default_rng(1000 + 10 * m + block_size)
            x = problem.exact_pf_decision(0.7)
            gradients = problem.objective_gradients(x)
            jacobian = problem.constraint_jacobian(x)
            base_raw = cosine_matrix(gradients)
            base_joint, base_diag = joint_signed_matrix(gradients, jacobian)
            constraints = problem.constraint_values(x)
            for scale in (0.1, 1.0, 10.0):
                alpha = scale * rng.normal(size=(m, m - 1))
                extended_gradients = gradients + alpha @ jacobian
                extended_raw = cosine_matrix(extended_gradients)
                extended_joint, _ = joint_signed_matrix(extended_gradients, jacobian)
                extended_values = problem.objective_values(x) + alpha @ constraints
                invariance_rows.append(
                    {
                        "M": m,
                        "block_size": block_size,
                        "test": "normal_extension",
                        "case": f"alpha_scale_{scale:g}",
                        "condition_number": base_diag.condition_number,
                        "objective_value_max_abs_error": float(
                            np.max(np.abs(extended_values - problem.objective_values(x)))
                        ),
                        "raw_matrix_max_abs_change": float(np.nanmax(np.abs(extended_raw - base_raw))),
                        "joint_matrix_max_abs_change": float(np.nanmax(np.abs(extended_joint - base_joint))),
                        "projector_max_abs_change": 0.0,
                    }
                )

            left, _ = np.linalg.qr(rng.normal(size=(m - 1, m - 1)))
            right, _ = np.linalg.qr(rng.normal(size=(m - 1, m - 1)))
            mixing_cases = {
                "orthogonal": left,
                "condition_10": left @ np.diag(np.geomspace(1.0, 10.0, m - 1)) @ right.T,
                "condition_1e8_report_only": left
                @ np.diag(np.geomspace(1.0, 1.0e8, m - 1))
                @ right.T,
            }
            for name, mixing in mixing_cases.items():
                mixed_joint, mixed_diag = joint_signed_matrix(gradients, mixing @ jacobian)
                invariance_rows.append(
                    {
                        "M": m,
                        "block_size": block_size,
                        "test": "row_mixing",
                        "case": name,
                        "condition_number": float(np.linalg.cond(mixing)),
                        "objective_value_max_abs_error": 0.0,
                        "raw_matrix_max_abs_change": 0.0,
                        "joint_matrix_max_abs_change": float(np.nanmax(np.abs(mixed_joint - base_joint))),
                        "projector_max_abs_change": float(
                            np.max(np.abs(mixed_diag.projector - base_diag.projector))
                        ),
                    }
                )
            print(f"phase2 complete M={m}, block_size={block_size}", flush=True)

    geometry = pd.DataFrame(geometry_rows)
    derivatives = pd.DataFrame(derivative_rows)
    invariance = pd.DataFrame(invariance_rows)
    representatives = pd.DataFrame(representative_rows)
    geometry.to_csv(output_dir / "phase2_geometry_validation.csv", index=False)
    derivatives.to_csv(output_dir / "phase2_derivative_validation.csv", index=False)
    invariance.to_csv(output_dir / "phase2_invariance_validation.csv", index=False)
    representatives.to_csv(output_dir / "phase2_representative_matrices.csv", index=False)
    return geometry, derivatives, invariance, representatives


def _partial_active_points(
    problem: DTLZ9Problem,
    rng: np.random.Generator,
    *,
    q: float,
    count: int,
    inactive_delta: float = 0.05,
    heterogeneous: bool = False,
) -> np.ndarray:
    points = []
    active_count = int(round(q * (problem.num_objectives() - 1)))
    active_count = min(max(active_count, 0), problem.num_objectives() - 1)
    for _ in range(count):
        theta_low = max(problem.theta_min, 0.45) if heterogeneous else problem.theta_min
        theta_high = min(0.5 * np.pi - problem.theta_min, 1.20) if heterogeneous else 0.5 * np.pi - problem.theta_min
        theta = float(rng.uniform(theta_low, theta_high))
        active = (
            rng.choice(problem.num_objectives() - 1, size=active_count, replace=False)
            if active_count
            else np.empty(0, dtype=int)
        )
        if heterogeneous:
            deltas = rng.uniform(0.02, 0.60, size=problem.num_objectives() - 1)
            deltas[active] = 0.0
            points.append(problem.controlled_off_front_decision(theta, deltas))
        else:
            jitter = float(rng.uniform(0.5, 1.5))
            points.append(
                problem.adjoining_surface_decision(
                    theta,
                    active,
                    inactive_delta=inactive_delta * jitter,
                )
            )
    return np.vstack(points)


def _record_grouping_rows(
    matrices: dict[str, np.ndarray],
    *,
    m: int,
    block_size: int,
    regime: str,
    seed: int,
    active_fraction: float,
    runtime_seconds: float,
) -> list[dict[str, object]]:
    true_labels = np.asarray([1] * (m - 1) + [2], dtype=int)
    target = expected_signed_matrix(m)
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        matrix = matrices[method]
        fixed = fixed_k_groups(matrix, 2)
        automatic = positive_component_groups(matrix, 1.0e-8)
        edge = edge_summary(matrix)
        for grouping_mode, labels in (("fixed_K2", fixed), ("unknown_K", automatic)):
            metrics = grouping_metrics(true_labels, labels)
            row: dict[str, object] = {
                "M": m,
                "block_size": block_size,
                "n": m * block_size,
                "regime": regime,
                "seed": seed,
                "method": method,
                "grouping_mode": grouping_mode,
                "active_fraction": active_fraction,
                "runtime_seconds": runtime_seconds,
                "signed_matrix_mae_vs_pf_oracle": float(np.nanmean(np.abs(matrix - target))),
                "labels": " ".join(str(int(value)) for value in labels),
                "groups_threshold_neg005": int(np.unique(positive_component_groups(matrix, -0.05)).size),
                "groups_threshold_zero": int(np.unique(positive_component_groups(matrix, 0.0)).size),
                "groups_threshold_pos005": int(np.unique(positive_component_groups(matrix, 0.05)).size),
                **edge,
                **metrics,
            }
            rows.append(row)
    return rows


def _evaluate_case(
    problem: DTLZ9Problem,
    points: np.ndarray,
    *,
    regime: str,
    seed: int,
    gradient_noise: float,
    active_tolerance: float,
) -> list[dict[str, object]]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed + 99991)
    blocks = sampled_blocks(problem, points, gradient_noise=gradient_noise, rng=rng)
    matrices, metadata = matrices_from_sampled_blocks(blocks, active_tolerance=active_tolerance)
    values = np.vstack([problem.objective_values(point) for point in points])
    matrices.update(value_correlation_matrices(values))
    runtime = time.perf_counter() - start
    active_counts = np.asarray(metadata["active_counts"], dtype=int)
    active_fraction = float(np.mean(active_counts == problem.num_objectives() - 1))
    return _record_grouping_rows(
        matrices,
        m=problem.num_objectives(),
        block_size=problem.block_size,
        regime=regime,
        seed=seed,
        active_fraction=active_fraction,
        runtime_seconds=runtime,
    )


def run_phase3(
    output_dir: Path,
    *,
    seeds: int,
    points_per_case: int,
    active_tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for m in (3, 5, 10, 20):
        for block_size in (1, 10):
            problem = DTLZ9Problem(m, m * block_size)
            exact_theta = np.linspace(problem.theta_min, 0.5 * np.pi - problem.theta_min, 25)
            exact_points = problem.exact_pf_decisions(exact_theta)
            rows.extend(
                _evaluate_case(
                    problem,
                    exact_points,
                    regime="exact_pf",
                    seed=-1,
                    gradient_noise=0.0,
                    active_tolerance=active_tolerance,
                )
            )

            for seed in range(seeds):
                rng = np.random.default_rng(100000 * m + 1000 * block_size + seed)
                for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                    points = _partial_active_points(
                        problem,
                        rng,
                        q=q,
                        count=points_per_case,
                        inactive_delta=0.05,
                    )
                    rows.extend(
                        _evaluate_case(
                            problem,
                            points,
                            regime=f"active_fraction_q{q:g}",
                            seed=seed,
                            gradient_noise=0.0,
                            active_tolerance=active_tolerance,
                        )
                    )

                for q in (0.25, 0.5, 0.75):
                    points = _partial_active_points(
                        problem,
                        rng,
                        q=q,
                        count=points_per_case,
                        heterogeneous=True,
                    )
                    rows.extend(
                        _evaluate_case(
                            problem,
                            points,
                            regime=f"heterogeneous_contamination_q{q:g}",
                            seed=seed,
                            gradient_noise=0.0,
                            active_tolerance=active_tolerance,
                        )
                    )

                noisy_theta = rng.uniform(
                    problem.theta_min,
                    0.5 * np.pi - problem.theta_min,
                    size=points_per_case,
                )
                noisy_points = problem.exact_pf_decisions(noisy_theta)
                for noise in (0.01, 0.05):
                    rows.extend(
                        _evaluate_case(
                            problem,
                            noisy_points,
                            regime=f"exact_pf_gradient_noise_{noise:g}",
                            seed=seed,
                            gradient_noise=noise,
                            active_tolerance=active_tolerance,
                        )
                    )
            print(f"phase3 synthetic complete M={m}, block_size={block_size}", flush=True)

    for m in (3, 5, 10):
        for block_size in (1, 10):
            problem = DTLZ9Problem(m, m * block_size)
            for seed in range(seeds):
                rng = np.random.default_rng(700000 + 10000 * m + 100 * block_size + seed)
                initial = _partial_active_points(problem, rng, q=0.5, count=2, inactive_delta=0.08)
                config = NonlinearORCAConfig(
                    num_groups=2,
                    grouping_method="average_linkage",
                    active_constraint_tolerance=active_tolerance,
                    num_points_per_seed=3,
                    include_seed_points=True,
                    random_seed=seed,
                    step_size=0.01,
                    max_projection_failures=20,
                )
                try:
                    selected = generate_fixed_points(problem, config, seeds=initial)
                    rows.extend(
                        _evaluate_case(
                            problem,
                            selected,
                            regime="orca_selected_points",
                            seed=seed,
                            gradient_noise=0.0,
                            active_tolerance=active_tolerance,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "M": m,
                            "block_size": block_size,
                            "n": m * block_size,
                            "regime": "orca_selected_points",
                            "seed": seed,
                            "method": "POINT_GENERATION",
                            "grouping_mode": "failure",
                            "active_fraction": np.nan,
                            "runtime_seconds": np.nan,
                            "error": str(exc),
                        }
                    )
            print(f"phase3 selected-point complete M={m}, block_size={block_size}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "phase3_grouping_recovery_raw.csv", index=False)
    valid = raw[raw["method"].isin(METHOD_ORDER)].copy()
    summary = (
        valid.groupby(
            ["M", "block_size", "regime", "method", "grouping_mode"],
            dropna=False,
            as_index=False,
        )
        .agg(
            runs=("seed", "count"),
            exact_recovery_rate=("exact_recovery", "mean"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            pairwise_f1_mean=("pairwise_f1", "mean"),
            estimated_groups_mean=("estimated_groups", "mean"),
            matrix_mae_mean=("signed_matrix_mae_vs_pf_oracle", "mean"),
            within_mean=("within_mean", "mean"),
            between_mean=("between_mean", "mean"),
            separation_margin_mean=("separation_margin", "mean"),
            active_fraction_mean=("active_fraction", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
    )
    summary["ari_std"] = summary["ari_std"].fillna(0.0)
    summary.to_csv(output_dir / "phase3_grouping_recovery_summary.csv", index=False)
    return raw


def write_run_manifest(
    output_dir: Path,
    *,
    seeds: int,
    points_per_case: int,
    active_tolerance: float,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "experiment": "DTLZ9 ORCA Phase 1-3",
        "verification_status": "VERIFIED_BY_EXECUTION",
        "p": 0.1,
        "phase2_M": [3, 5, 10],
        "phase3_M": [3, 5, 10, 20],
        "block_sizes": [1, 10],
        "phase2_exact_pf_points": 25,
        "stochastic_seeds": seeds,
        "stochastic_points_per_case": points_per_case,
        "partial_active_q": [0.0, 0.25, 0.5, 0.75, 1.0],
        "heterogeneous_contamination_q": [0.25, 0.5, 0.75],
        "gradient_noise_levels": [0.01, 0.05],
        "active_tolerance": active_tolerance,
        "box_bounds_exposed_to_orca": False,
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dtlz9_phase3"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--points-per-case", type=int, default=12)
    parser.add_argument("--active-tolerance", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1 or args.points_per_case < 2:
        raise ValueError("seeds must be positive and points-per-case must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    run_phase2(args.output_dir)
    run_phase3(
        args.output_dir,
        seeds=args.seeds,
        points_per_case=args.points_per_case,
        active_tolerance=args.active_tolerance,
    )
    elapsed = time.perf_counter() - start
    write_run_manifest(
        args.output_dir,
        seeds=args.seeds,
        points_per_case=args.points_per_case,
        active_tolerance=args.active_tolerance,
        elapsed_seconds=elapsed,
    )
    print(f"completed Phase 1-3 in {elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
