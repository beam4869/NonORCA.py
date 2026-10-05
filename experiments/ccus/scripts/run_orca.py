"""Run pointwise ORCA on the converged CCUS scenario solutions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
ORCA_REPO = Path(os.environ.get("ORCA_REPO", str(Path(__file__).resolve().parents[3]))).expanduser()
if not (ORCA_REPO / "src" / "orca").is_dir():
    raise RuntimeError(f"ORCA package was not found at {ORCA_REPO}")
sys.path.insert(0, str(ORCA_REPO / "src"))

from orca.utils.correlation_aggregation import aggregate_interactions_to_adjacency
from orca.utils.interaction_weights import compute_interaction_weights
from orca.utils.local_objective_interactions import (
    compute_linear_local_objective_interactions,
)
from orca.utils.objective_grouping import group_objectives


OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
ACTIVE_TOLERANCE = 1.0e-6
ALPHA_WEIGHT = 0.9
BETA_WEIGHT = 100.0


def normalized_cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-14):
        raise ValueError("An objective gradient has effectively zero norm")
    unit = matrix / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def pointwise_orca(
    objective_gradients: np.ndarray,
    equality_gradients: np.ndarray,
    active_inequality_gradients: np.ndarray,
):
    strengths, valid, metadata = compute_linear_local_objective_interactions(
        objective_gradients,
        active_inequality_gradients,
        equality_gradients,
        filter_inactive_inequalities=False,
    )
    return strengths, valid, metadata["equality_mask"]


def main() -> None:
    solves = pd.read_csv(RESULT_DIR / "scenario_solves.csv")
    points = pd.read_csv(RESULT_DIR / "selected_points.csv")
    objective_df = pd.read_csv(RESULT_DIR / "objective_gradients.csv")
    equality_df = pd.read_csv(RESULT_DIR / "equality_gradients.csv")
    inequality_df = pd.read_csv(RESULT_DIR / "inequality_gradients.csv")
    gradient_columns = sorted(
        (column for column in objective_df.columns if column.startswith("dz")),
        key=lambda column: int(column[2:]),
    )

    scenario_order = solves["scenario"].drop_duplicates().tolist()
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    point_rows: list[dict] = []

    for scenario in scenario_order:
        scenario_points = points.loc[points["scenario"] == scenario]
        if scenario_points.empty:
            failed = solves.loc[solves["scenario"] == scenario].iloc[0]
            summary_rows.append(
                {
                    "scenario": scenario,
                    "status": "skipped_no_accepted_points",
                    "n_points": 0,
                    "termination": failed["termination"],
                    "max_inequality_violation": failed["max_inequality_violation"],
                }
            )
            continue

        all_strengths = []
        all_valid = []
        all_equality_masks = []
        raw_cosines = []
        active_counts = []
        for point_id in scenario_points["point_id"]:
            obj_point = objective_df.loc[objective_df["point_id"] == point_id].set_index("objective")
            eq_point = equality_df.loc[equality_df["point_id"] == point_id]
            ineq_point = inequality_df.loc[inequality_df["point_id"] == point_id]
            if set(obj_point.index) != set(OBJECTIVES):
                raise ValueError(f"Objective gradients are incomplete at {point_id}")

            Jobj = obj_point.loc[OBJECTIVES, gradient_columns].to_numpy(dtype=float)
            Jeq = eq_point[gradient_columns].to_numpy(dtype=float)
            active = ineq_point["value"].to_numpy(dtype=float) >= -ACTIVE_TOLERANCE
            Jineq = ineq_point.loc[active, gradient_columns].to_numpy(dtype=float)
            strengths, valid, equality_mask = pointwise_orca(Jobj, Jeq, Jineq)
            all_strengths.append(strengths)
            all_valid.append(valid)
            all_equality_masks.append(equality_mask)
            raw_cosines.append(normalized_cosine_matrix(Jobj))
            active_counts.append(int(active.sum()))
            point_rows.append(
                {
                    "scenario": scenario,
                    "point_id": point_id,
                    "solve_type": scenario_points.loc[
                        scenario_points["point_id"] == point_id, "solve_type"
                    ].iloc[0],
                    "active_inequalities": int(active.sum()),
                    "equality_constraints": int(Jeq.shape[0]),
                    "local_interactions": int(strengths.shape[-1]),
                }
            )

        strengths = np.concatenate(all_strengths, axis=-1)
        valid = np.concatenate(all_valid, axis=-1)
        equality_mask = np.concatenate(all_equality_masks)
        weights = compute_interaction_weights(
            strengths,
            valid_mask=valid,
            equality_mask=equality_mask,
            alpha_weight=ALPHA_WEIGHT,
            beta_weight=BETA_WEIGHT,
        )
        adjacency, total_weights, _ = aggregate_interactions_to_adjacency(strengths, weights)
        signed = 2.0 * adjacency - 1.0
        raw_cosine = np.mean(raw_cosines, axis=0)
        groups = group_objectives(adjacency, 2, method="average_linkage")

        if not np.allclose(adjacency, adjacency.T, atol=1.0e-12):
            raise AssertionError(f"ORCA adjacency is not symmetric for {scenario}")
        if not np.allclose(np.diag(adjacency), 1.0, atol=1.0e-12):
            raise AssertionError(f"ORCA adjacency diagonal is invalid for {scenario}")
        if not np.isfinite(adjacency).all():
            raise AssertionError(f"ORCA adjacency contains a non-finite value for {scenario}")

        for i, objective_i in enumerate(OBJECTIVES):
            for j in range(i + 1, len(OBJECTIVES)):
                objective_j = OBJECTIVES[j]
                pair_rows.append(
                    {
                        "scenario": scenario,
                        "objective_i": objective_i,
                        "objective_j": objective_j,
                        "pair": f"{objective_i}__{objective_j}",
                        "orca_adjacency_01": adjacency[i, j],
                        "orca_signed_correlation": signed[i, j],
                        "raw_gradient_cosine": raw_cosine[i, j],
                        "total_interaction_weight": total_weights[i, j],
                        "group_i_k2": int(groups[i]),
                        "group_j_k2": int(groups[j]),
                        "same_group_k2": bool(groups[i] == groups[j]),
                    }
                )

        summary_rows.append(
            {
                "scenario": scenario,
                "status": "ok",
                "n_points": len(scenario_points),
                "mean_active_inequalities": float(np.mean(active_counts)),
                "total_local_interactions": int(strengths.shape[-1]),
                "group_TAC_k2": int(groups[0]),
                "group_TotEmiss_k2": int(groups[1]),
                "group_ISI_k2": int(groups[2]),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    baseline = pairwise.loc[pairwise["scenario"] == "baseline"].set_index("pair")
    pairwise["delta_orca_signed_vs_baseline"] = pairwise.apply(
        lambda row: row["orca_signed_correlation"]
        - baseline.loc[row["pair"], "orca_signed_correlation"],
        axis=1,
    )
    pairwise["delta_raw_cosine_vs_baseline"] = pairwise.apply(
        lambda row: row["raw_gradient_cosine"]
        - baseline.loc[row["pair"], "raw_gradient_cosine"],
        axis=1,
    )

    pairwise.to_csv(RESULT_DIR / "orca_pairwise.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(RESULT_DIR / "orca_scenario_summary.csv", index=False)
    pd.DataFrame(point_rows).to_csv(RESULT_DIR / "orca_point_diagnostics.csv", index=False)
    (RESULT_DIR / "orca_run.txt").write_text(
        "\n".join(
            [
                f"orca_repo = {ORCA_REPO}",
                f"active_inequality_tolerance = {ACTIVE_TOLERANCE}",
                f"alpha_weight = {ALPHA_WEIGHT}",
                f"beta_weight = {BETA_WEIGHT}",
                "equalities = always included",
                "grouping = average_linkage, K=2 (descriptive only)",
                f"successful_scenarios = {(pd.DataFrame(summary_rows)['status'] == 'ok').sum()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved ORCA results for {pairwise['scenario'].nunique()} feasible scenarios")


if __name__ == "__main__":
    main()
