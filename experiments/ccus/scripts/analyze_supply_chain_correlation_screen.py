"""Apply the local ORCA package to supply-chain parameter-screen endpoints."""

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
    pointwise_orca,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
BASELINE = "supply_chain_condition1"


def main() -> None:
    status = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_status.csv"
    )
    points = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_solutions.csv"
    )
    objective_df = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_objective_gradients.csv"
    )
    equality_df = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_equality_gradients.csv"
    )
    inequality_df = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_inequality_gradients.csv"
    )
    gradient_columns = sorted(
        (column for column in objective_df.columns if column.startswith("dz")),
        key=lambda column: int(column[2:]),
    )

    pair_rows: list[dict] = []
    point_rows: list[dict] = []
    grouping_rows: list[dict] = []
    for scenario in status["scenario"]:
        scenario_points = points.loc[points["scenario"].eq(scenario)]
        if len(scenario_points) != len(OBJECTIVES):
            continue

        strengths_all = []
        valid_all = []
        equality_masks = []
        for point_id in scenario_points["point_id"]:
            obj = objective_df.loc[objective_df["point_id"].eq(point_id)].set_index(
                "objective"
            )
            eq = equality_df.loc[equality_df["point_id"].eq(point_id)]
            ineq = inequality_df.loc[inequality_df["point_id"].eq(point_id)]
            Jobj = obj.loc[OBJECTIVES, gradient_columns].to_numpy(float)
            Jeq = eq[gradient_columns].to_numpy(float)
            active = ineq["value"].to_numpy(float) >= -ACTIVE_TOLERANCE
            Jineq = ineq.loc[active, gradient_columns].to_numpy(float)
            strengths, valid, equality_mask = pointwise_orca(Jobj, Jeq, Jineq)

            weights = compute_interaction_weights(
                strengths,
                valid_mask=valid,
                equality_mask=equality_mask,
                alpha_weight=ALPHA_WEIGHT,
                beta_weight=BETA_WEIGHT,
            )
            adjacency, _, _ = aggregate_interactions_to_adjacency(strengths, weights)
            signed = 2.0 * adjacency - 1.0
            solve_type = scenario_points.loc[
                scenario_points["point_id"].eq(point_id), "solve_type"
            ].iloc[0]
            for i, objective_i in enumerate(OBJECTIVES):
                for j in range(i + 1, len(OBJECTIVES)):
                    point_rows.append(
                        {
                            "scenario": scenario,
                            "point_id": point_id,
                            "solve_type": solve_type,
                            "pair": f"{objective_i}__{OBJECTIVES[j]}",
                            "orca_signed_correlation": signed[i, j],
                            "active_inequalities": int(active.sum()),
                        }
                    )
            strengths_all.append(strengths)
            valid_all.append(valid)
            equality_masks.append(equality_mask)

        strengths = np.concatenate(strengths_all, axis=-1)
        valid = np.concatenate(valid_all, axis=-1)
        equality_mask = np.concatenate(equality_masks)
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
        groups = group_objectives(adjacency, 2, method="average_linkage")
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
                        "total_interaction_weight": total_weights[i, j],
                        "group_i_k2": int(groups[i]),
                        "group_j_k2": int(groups[j]),
                        "same_group_k2": bool(groups[i] == groups[j]),
                    }
                )
        grouping_rows.append(
            {
                "scenario": scenario,
                "group_TAC_k2": int(groups[0]),
                "group_TotEmiss_k2": int(groups[1]),
                "group_ISI_k2": int(groups[2]),
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    wide = pairwise.pivot(
        index="scenario", columns="pair", values="orca_signed_correlation"
    ).reset_index()
    wide["target_margin"] = wide["TAC__TotEmiss"] - wide[
        ["TAC__ISI", "TotEmiss__ISI"]
    ].max(axis=1)
    wide["target_is_strongest"] = wide["target_margin"] > 0.0
    wide = wide.merge(pd.DataFrame(grouping_rows), on="scenario", how="left")
    wide["target_same_group_k2"] = (
        wide["group_TAC_k2"] == wide["group_TotEmiss_k2"]
    )
    mechanisms = status[["scenario", "mechanism"]]
    wide = wide.merge(mechanisms, on="scenario", how="left")
    wide = wide.sort_values(
        ["target_is_strongest", "target_margin", "TAC__TotEmiss"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    wide.insert(0, "screen_rank", np.arange(1, len(wide) + 1))

    baseline = pairwise.loc[pairwise["scenario"].eq(BASELINE)].set_index("pair")
    pairwise["delta_signed_vs_condition1"] = pairwise.apply(
        lambda row: row["orca_signed_correlation"]
        - baseline.loc[row["pair"], "orca_signed_correlation"],
        axis=1,
    )
    pairwise.to_csv(
        RESULT_DIR / "supply_chain_correlation_screen_orca_pairwise.csv",
        index=False,
    )
    pd.DataFrame(point_rows).to_csv(
        RESULT_DIR / "supply_chain_correlation_screen_orca_pointwise.csv",
        index=False,
    )
    wide.to_csv(
        RESULT_DIR / "supply_chain_correlation_screen_ranking.csv", index=False
    )
    print(
        wide[
            [
                "screen_rank",
                "scenario",
                "TAC__TotEmiss",
                "TAC__ISI",
                "TotEmiss__ISI",
                "target_margin",
                "target_is_strongest",
                "target_same_group_k2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
