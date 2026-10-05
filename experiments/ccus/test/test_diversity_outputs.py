from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


def main() -> None:
    summary = pd.read_csv(RESULT_DIR / "diversity_candidate_summary.csv")
    solutions = pd.read_csv(RESULT_DIR / "diversity_objective_solutions.csv")
    pairwise = pd.read_csv(RESULT_DIR / "diversity_orca_pairwise.csv")
    pointwise = pd.read_csv(RESULT_DIR / "diversity_orca_pointwise_pairwise.csv")

    assert len(summary) == 8
    assert len(solutions) == 24
    assert (summary["accepted_objectives"] == 3).all()
    assert solutions["point_id"].is_unique
    assert (solutions["max_abs_equality"] <= 1.0e-6).all()
    assert (solutions["max_abs_source_carbon_diagnostic"] <= 1.0e-6).all()
    assert (solutions["max_inequality_violation"] <= 1.0e-6).all()

    exploratory = summary.loc[summary["evidence_class"] == "exploratory"]
    assert (exploratory["distinct_main_sinks"] == 3).all()
    assert (
        summary.loc[summary["profile"] == "direct_use_expansion", "pretreatment_range"].iloc[0]
        > 0.60
    )

    direct_use = solutions.loc[solutions["profile"] == "direct_use_expansion"].set_index("objective")
    assert direct_use.loc["TAC", "main_sink"] == "Urea"
    assert direct_use.loc["TotEmiss", "main_sink"] == "Saline Storage"
    assert direct_use.loc["ISI", "main_sink"] == "Greenhouse"

    assert len(pairwise) == 24
    assert np.isfinite(pairwise["orca_signed_correlation"]).all()
    assert pairwise["orca_signed_correlation"].between(-1.0, 1.0).all()
    assert pairwise.groupby("scenario")["pair"].nunique().eq(3).all()
    assert len(pointwise) == 72
    assert pointwise.groupby(["scenario", "solve_type"])["pair"].nunique().eq(3).all()
    three_way = pointwise.loc[
        (pointwise["scenario"] == "three_way_competition")
        & (pointwise["pair"] == "TAC__TotEmiss")
    ].set_index("solve_type")
    assert three_way.loc["TAC", "orca_signed_correlation"] > 0.0
    assert three_way.loc["ISI", "orca_signed_correlation"] < 0.0
    print("Diversity outputs passed structural, feasibility, and ORCA checks")


if __name__ == "__main__":
    main()
