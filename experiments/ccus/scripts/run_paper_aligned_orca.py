"""Aggregate paper-aligned nonlinear ORCA interactions for sampled CCUS points."""

from __future__ import annotations

import os
import sys
from pathlib import Path


import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
ORCA_REPO = Path(
    os.environ.get("ORCA_REPO", str(Path(__file__).resolve().parents[3]))
).expanduser()
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.utils.correlation_aggregation import (  # noqa: E402
    aggregate_interactions_to_adjacency,
)
from orca.utils.interaction_weights import compute_interaction_weights  # noqa: E402
from orca.utils.local_objective_interactions import (  # noqa: E402
    compute_linear_local_objective_interactions,
    compute_nonlinear_local_objective_interactions,
)
from orca.utils.objective_grouping import group_objectives  # noqa: E402


OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
ACTIVE_TOLERANCE = 1.0e-6
ALPHA_WEIGHT = 0.9
BETA_WEIGHT = 100.0


def numeric_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda column: int(column[len(prefix) :]),
    )


def equality_interactions(objective_gradients: np.ndarray, equality_jacobian: np.ndarray):
    strengths = []
    valid = []
    empty_inequalities = np.zeros((0, objective_gradients.shape[-1]), dtype=float)
    for local_objectives in objective_gradients:
        local_strengths, local_valid, _ = compute_linear_local_objective_interactions(
            local_objectives,
            empty_inequalities,
            equality_jacobian,
            filter_inactive_inequalities=False,
        )
        strengths.append(local_strengths.reshape(3, 3, -1))
        valid.append(local_valid.reshape(3, 3, -1))
    return np.concatenate(strengths, axis=-1), np.concatenate(valid, axis=-1)


def aggregate_one_replicate(
    objective_gradients: np.ndarray,
    inequality_values: np.ndarray,
    inequality_jacobian: np.ndarray,
    equality_jacobian: np.ndarray,
    *,
    include_equalities: bool,
):
    repeated_inequality_jacobian = np.broadcast_to(
        inequality_jacobian,
        (len(objective_gradients), *inequality_jacobian.shape),
    )
    ineq_strengths, ineq_valid, ineq_meta = compute_nonlinear_local_objective_interactions(
        objective_gradients,
        repeated_inequality_jacobian,
        constraint_values=inequality_values,
        active_constraint_tolerance=ACTIVE_TOLERANCE,
    )
    strengths = ineq_strengths.reshape(3, 3, -1)
    valid = ineq_valid.reshape(3, 3, -1)
    equality_mask = np.zeros(strengths.shape[-1], dtype=bool)

    if include_equalities:
        eq_strengths, eq_valid = equality_interactions(
            objective_gradients, equality_jacobian
        )
        strengths = np.concatenate([strengths, eq_strengths], axis=-1)
        valid = np.concatenate([valid, eq_valid], axis=-1)
        equality_mask = np.concatenate(
            [equality_mask, np.ones(eq_strengths.shape[-1], dtype=bool)]
        )

    weights = compute_interaction_weights(
        strengths,
        valid_mask=valid,
        equality_mask=equality_mask,
        alpha_weight=ALPHA_WEIGHT,
        beta_weight=BETA_WEIGHT,
    )
    adjacency, total_weights, _ = aggregate_interactions_to_adjacency(
        strengths, weights
    )
    signed = 2.0 * adjacency - 1.0
    leiden = group_objectives(adjacency, 2, method="leiden")
    average = group_objectives(adjacency, 2, method="average_linkage")
    return {
        "adjacency": adjacency,
        "signed": signed,
        "total_weights": total_weights,
        "leiden": leiden,
        "average": average,
        "active_inequality_interactions": int(
            ineq_meta.get("num_active_constraint_pairs", strengths.shape[-1])
        ),
        "equality_interactions": int(
            len(objective_gradients) * len(equality_jacobian)
            if include_equalities
            else 0
        ),
    }


def main() -> None:
    label = os.environ.get("CCUS_ORCA_RUN_LABEL", "paper_main")
    profile = os.environ.get("CCUS_ORCA_PROFILE", "direct_use_expansion")
    prefix = f"{profile}_paper_orca_{label}_"
    points = pd.read_csv(RESULT_DIR / f"{prefix}points.csv")
    gradients = pd.read_csv(RESULT_DIR / f"{prefix}objective_gradients.csv")
    equality = pd.read_csv(RESULT_DIR / f"{prefix}equality_jacobian.csv")
    inequality = pd.read_csv(RESULT_DIR / f"{prefix}inequality_jacobian.csv")
    z_columns = numeric_columns(points, "z")
    g_columns = numeric_columns(points, "g")
    dz_columns = numeric_columns(gradients, "dz")
    equality_jacobian = equality[dz_columns].to_numpy(dtype=float)
    inequality_jacobian = inequality[dz_columns].to_numpy(dtype=float)

    pair_rows = []
    grouping_rows = []
    for replicate, replicate_points in points.groupby("replicate", sort=True):
        point_ids = replicate_points["point_id"].tolist()
        objective_gradients = np.stack(
            [
                gradients.loc[gradients["point_id"] == point_id]
                .set_index("objective")
                .loc[OBJECTIVES, dz_columns]
                .to_numpy(dtype=float)
                for point_id in point_ids
            ],
            axis=0,
        )
        inequality_values = replicate_points[g_columns].to_numpy(dtype=float)
        metadata = replicate_points.iloc[0]
        for handling, include_equalities in (
            ("equality_tangent", True),
            ("inequalities_only", False),
        ):
            result = aggregate_one_replicate(
                objective_gradients,
                inequality_values,
                inequality_jacobian,
                equality_jacobian,
                include_equalities=include_equalities,
            )
            for i, j in PAIRS:
                pair_rows.append(
                    {
                        "run_label": label,
                        "replicate": int(replicate),
                        "rng_seed": int(metadata["rng_seed"]),
                        "step_size": float(metadata["step_size"]),
                        "points_per_seed": int(metadata["points_per_seed"]),
                        "selected_points": len(replicate_points),
                        "constraint_handling": handling,
                        "objective_i": OBJECTIVES[i],
                        "objective_j": OBJECTIVES[j],
                        "pair": f"{OBJECTIVES[i]}__{OBJECTIVES[j]}",
                        "orca_adjacency_01": result["adjacency"][i, j],
                        "orca_signed_correlation": result["signed"][i, j],
                        "total_interaction_weight": result["total_weights"][i, j],
                        "active_inequality_interactions": result[
                            "active_inequality_interactions"
                        ],
                        "equality_interactions": result["equality_interactions"],
                    }
                )
            grouping_rows.append(
                {
                    "run_label": label,
                    "replicate": int(replicate),
                    "rng_seed": int(metadata["rng_seed"]),
                    "step_size": float(metadata["step_size"]),
                    "points_per_seed": int(metadata["points_per_seed"]),
                    "selected_points": len(replicate_points),
                    "constraint_handling": handling,
                    "leiden_TAC": int(result["leiden"][0]),
                    "leiden_TotEmiss": int(result["leiden"][1]),
                    "leiden_ISI": int(result["leiden"][2]),
                    "average_TAC": int(result["average"][0]),
                    "average_TotEmiss": int(result["average"][1]),
                    "average_ISI": int(result["average"][2]),
                }
            )

    pairwise = pd.DataFrame(pair_rows)
    pairwise["rank_descending"] = (
        pairwise.groupby(["replicate", "constraint_handling"])[
            "orca_signed_correlation"
        ]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    summary = (
        pairwise.groupby(["run_label", "constraint_handling", "pair"], as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            selected_points_mean=("selected_points", "mean"),
            signed_mean=("orca_signed_correlation", "mean"),
            signed_sd=("orca_signed_correlation", "std"),
            signed_min=("orca_signed_correlation", "min"),
            signed_max=("orca_signed_correlation", "max"),
            adjacency_mean=("orca_adjacency_01", "mean"),
            rank_mean=("rank_descending", "mean"),
            strongest_count=("rank_descending", lambda values: int((values == 1).sum())),
        )
        .sort_values(["constraint_handling", "signed_mean"], ascending=[True, False])
    )

    pairwise.to_csv(RESULT_DIR / f"{prefix}pairwise.csv", index=False)
    summary.to_csv(RESULT_DIR / f"{prefix}summary.csv", index=False)
    pd.DataFrame(grouping_rows).to_csv(
        RESULT_DIR / f"{prefix}groupings.csv", index=False
    )
    (RESULT_DIR / f"{prefix}aggregation_run.txt").write_text(
        "\n".join(
            [
                f"orca_repo = {ORCA_REPO}",
                "method = Wang-Allman 2025 nonlinear modified-gradient interactions",
                f"run_label = {label}",
                f"profile = {profile}",
                f"active_inequality_tolerance = {ACTIVE_TOLERANCE}",
                f"alpha_weight = {ALPHA_WEIGHT}",
                f"beta_weight = {BETA_WEIGHT}",
                "equality_extension = tangent projection with unit weights",
                "grouping_primary = Leiden, K=2",
                "grouping_audit = average linkage, K=2",
                f"replicates = {pairwise['replicate'].nunique()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
