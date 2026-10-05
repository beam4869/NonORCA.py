"""Run pointwise ORCA on the parameter-diversity objective optima."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_orca import (
    ACTIVE_TOLERANCE,
    ALPHA_WEIGHT,
    BETA_WEIGHT,
    OBJECTIVES,
    aggregate_interactions_to_adjacency,
    compute_interaction_weights,
    group_objectives,
    normalized_cosine_matrix,
    pointwise_orca,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
BASELINE = "baseline_forced"


def main() -> None:
    profiles = pd.read_csv(RESULT_DIR / "diversity_candidate_summary.csv")
    points = pd.read_csv(RESULT_DIR / "diversity_objective_solutions.csv")
    objective_df = pd.read_csv(RESULT_DIR / "diversity_objective_gradients.csv")
    equality_df = pd.read_csv(RESULT_DIR / "diversity_equality_gradients.csv")
    inequality_df = pd.read_csv(RESULT_DIR / "diversity_inequality_gradients.csv")
    gradient_columns = sorted(
        (column for column in objective_df.columns if column.startswith("dz")),
        key=lambda column: int(column[2:]),
    )

    pair_rows: list[dict] = []
    point_pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    point_rows: list[dict] = []
    for scenario in profiles["profile"]:
        scenario_points = points.loc[points["scenario"] == scenario]
        if len(scenario_points) != len(OBJECTIVES):
            summary_rows.append(
                {"scenario": scenario, "status": "skipped_incomplete", "n_points": len(scenario_points)}
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

            objective_gradients = obj_point.loc[OBJECTIVES, gradient_columns].to_numpy(dtype=float)
            equality_gradients = eq_point[gradient_columns].to_numpy(dtype=float)
            active = ineq_point["value"].to_numpy(dtype=float) >= -ACTIVE_TOLERANCE
            inequality_gradients = ineq_point.loc[active, gradient_columns].to_numpy(dtype=float)
            strengths, valid, equality_mask = pointwise_orca(
                objective_gradients, equality_gradients, inequality_gradients
            )
            point_weights = compute_interaction_weights(
                strengths,
                valid_mask=valid,
                equality_mask=equality_mask,
                alpha_weight=ALPHA_WEIGHT,
                beta_weight=BETA_WEIGHT,
            )
            point_adjacency, point_total_weights, _ = aggregate_interactions_to_adjacency(
                strengths, point_weights
            )
            point_signed = 2.0 * point_adjacency - 1.0
            point_cosine = normalized_cosine_matrix(objective_gradients)
            solve_type = scenario_points.loc[
                scenario_points["point_id"] == point_id, "objective"
            ].iloc[0]
            for i, objective_i in enumerate(OBJECTIVES):
                for j in range(i + 1, len(OBJECTIVES)):
                    point_pair_rows.append(
                        {
                            "scenario": scenario,
                            "point_id": point_id,
                            "solve_type": solve_type,
                            "objective_i": objective_i,
                            "objective_j": OBJECTIVES[j],
                            "pair": f"{objective_i}__{OBJECTIVES[j]}",
                            "orca_adjacency_01": point_adjacency[i, j],
                            "orca_signed_correlation": point_signed[i, j],
                            "raw_gradient_cosine": point_cosine[i, j],
                            "total_interaction_weight": point_total_weights[i, j],
                            "active_inequalities": int(active.sum()),
                        }
                    )
            all_strengths.append(strengths)
            all_valid.append(valid)
            all_equality_masks.append(equality_mask)
            raw_cosines.append(point_cosine)
            active_counts.append(int(active.sum()))
            point_rows.append(
                {
                    "scenario": scenario,
                    "point_id": point_id,
                    "solve_type": solve_type,
                    "active_inequalities": int(active.sum()),
                    "equality_constraints": int(equality_gradients.shape[0]),
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
                pair_rows.append(
                    {
                        "scenario": scenario,
                        "objective_i": objective_i,
                        "objective_j": OBJECTIVES[j],
                        "pair": f"{objective_i}__{OBJECTIVES[j]}",
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
    baseline = pairwise.loc[pairwise["scenario"] == BASELINE].set_index("pair")
    pairwise["delta_orca_signed_vs_baseline"] = pairwise.apply(
        lambda row: row["orca_signed_correlation"]
        - baseline.loc[row["pair"], "orca_signed_correlation"],
        axis=1,
    )
    pairwise.to_csv(RESULT_DIR / "diversity_orca_pairwise.csv", index=False)
    pd.DataFrame(point_pair_rows).to_csv(
        RESULT_DIR / "diversity_orca_pointwise_pairwise.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(RESULT_DIR / "diversity_orca_summary.csv", index=False)
    pd.DataFrame(point_rows).to_csv(RESULT_DIR / "diversity_orca_point_diagnostics.csv", index=False)
    (RESULT_DIR / "diversity_orca_run.txt").write_text(
        "\n".join(
            [
                f"baseline = {BASELINE}",
                f"active_inequality_tolerance = {ACTIVE_TOLERANCE}",
                f"alpha_weight = {ALPHA_WEIGHT}",
                f"beta_weight = {BETA_WEIGHT}",
                "points_per_profile = TAC, TotEmiss, ISI optima",
                "grouping = average_linkage, K=2 (descriptive only)",
                f"successful_profiles = {(pd.DataFrame(summary_rows)['status'] == 'ok').sum()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved diversity ORCA results for {pairwise['scenario'].nunique()} profiles")


if __name__ == "__main__":
    main()
