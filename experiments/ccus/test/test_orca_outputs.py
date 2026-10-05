"""Sanity checks for generated CCUS ORCA artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


def main() -> None:
    pairwise = pd.read_csv(RESULT_DIR / "orca_pairwise.csv")
    summary = pd.read_csv(RESULT_DIR / "orca_scenario_summary.csv")
    solves = pd.read_csv(RESULT_DIR / "scenario_solves.csv")
    equalities = pd.read_csv(RESULT_DIR / "equality_gradients.csv")

    successful = summary.loc[summary["status"] == "ok"]
    assert len(successful) == 11
    assert len(pairwise) == 33
    assert pairwise.groupby("scenario").size().eq(3).all()
    assert pairwise["orca_adjacency_01"].between(0.0, 1.0).all()
    assert pairwise["orca_signed_correlation"].between(-1.0, 1.0).all()
    assert np.isfinite(pairwise["total_interaction_weight"]).all()

    gradient_columns = sorted(
        (column for column in equalities.columns if column.startswith("dz")),
        key=lambda column: int(column[2:]),
    )
    equality_ranks = [
        np.linalg.matrix_rank(group[gradient_columns].to_numpy(dtype=float))
        for _, group in equalities.groupby("point_id")
    ]
    assert equality_ranks and set(equality_ranks) == {34}

    infeasible = solves.loc[
        (solves["scenario"] == "capture_infeasible")
        & (solves["solve_type"] == "feasibility")
    ].iloc[0]
    assert infeasible["termination"] == "LOCALLY_INFEASIBLE"
    assert not bool(infeasible["accepted"])
    assert float(infeasible["max_inequality_violation"]) > 0.02
    print("ORCA output checks passed")


if __name__ == "__main__":
    main()
