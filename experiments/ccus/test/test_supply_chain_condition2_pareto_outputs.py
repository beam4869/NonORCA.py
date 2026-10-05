from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
PREFIX = "supply_chain_condition2_"
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]


def test_condition2_frontier_is_feasible_and_nondominated() -> None:
    frontier = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_frontier.csv")
    assert len(frontier) == 279
    assert frontier[[f"{name}_normalized" for name in OBJECTIVES]].notna().all().all()

    values = frontier[OBJECTIVES].to_numpy(float)
    ranges = np.ptp(values, axis=0)
    scaled = (values - values.min(axis=0)) / ranges
    for index, point in enumerate(scaled):
        other = np.delete(scaled, index, axis=0)
        dominated = np.any(
            np.all(other <= point + 1.0e-7, axis=1)
            & np.any(other < point - 1.0e-7, axis=1)
        )
        assert not dominated

    raw = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_full_raw.csv")
    assert raw["max_abs_equality"].max() < 1.0e-6
    assert raw["max_abs_source_carbon_diagnostic"].max() < 1.0e-6
    assert raw["max_inequality_violation"].max() < 1.0e-6
    assert raw["epsilon_violation"].max() < 1.0e-6


def test_condition2_frontier_has_rich_sink_structure() -> None:
    structures = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_system_structures.csv"
    )
    assert structures["frontier_id"].nunique() == 279
    assert structures["main_sink"].nunique() == 5
    assert {"Methanol", "Urea", "Acetic Acid", "Greenhouse", "Algae"}.issubset(
        set(structures["main_sink"])
    )
    assert structures["pretreated_fraction"].max() - structures[
        "pretreated_fraction"
    ].min() > 0.70


def test_condition2_exact_loss_matches_orca_at_fine_resolution() -> None:
    comparison = pd.read_csv(
        RESULT_DIR / f"{PREFIX}exact_info_loss_quantile21_comparison.csv"
    )
    assert comparison["represented_slices"].eq(11).all()
    assert comparison["multi_point_slices"].eq(11).all()
    assert comparison["exact_loss_rank"].tolist() == [1, 2, 3]
    assert comparison["orca_rank"].tolist() == [1, 2, 3]
    assert comparison.iloc[0]["grouping"] == "TAC + total emissions"
    assert comparison.iloc[-1]["grouping"] == "Total emissions + ISI"

    projections = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_pairwise_projection_source.csv"
    )
    counts = projections.groupby("pair")["is_2d_nondominated"].sum().to_dict()
    assert counts == {"Emissions-ISI": 60, "TAC-ISI": 40, "TAC-emissions": 2}
