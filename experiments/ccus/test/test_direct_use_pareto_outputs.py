"""Integrity checks for the direct-use Pareto and ORCA comparison."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
PREFIX = "direct_use_expansion_"
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]


def nondominated(values: np.ndarray, tolerance: float = 1.0e-7) -> bool:
    for index, point in enumerate(values):
        other = np.delete(values, index, axis=0)
        if np.any(
            np.all(other <= point + tolerance, axis=1)
            & np.any(other < point - tolerance, axis=1)
        ):
            return False
    return True


def assert_nested_failures_have_witnesses(log: pd.DataFrame, raw: pd.DataFrame):
    for method, subset in log.groupby("method"):
        accepted = raw.loc[raw["method"] == method]
        for row in subset.loc[~subset["accepted"]].itertuples():
            if pd.notna(row.index_b):
                witness = accepted.loc[
                    (accepted["index_a"] >= row.index_a)
                    & (accepted["index_b"] >= row.index_b)
                ]
            else:
                witness = accepted.loc[accepted["index_a"] >= row.index_a]
            if witness.empty:
                continue
            point = witness.iloc[0]
            if method == "full_TAC":
                assert point.TotEmiss <= row.epsilon_TotEmiss + 1.0e-5
                assert point.ISI <= row.epsilon_ISI + 1.0e-5
            elif method == "full_TotEmiss":
                assert point.TAC <= row.epsilon_TAC + 1.0e-2
                assert point.ISI <= row.epsilon_ISI + 1.0e-5
            elif method == "full_ISI":
                assert point.TAC <= row.epsilon_TAC + 1.0e-2
                assert point.TotEmiss <= row.epsilon_TotEmiss + 1.0e-5
            elif method == "group_TAC_TotEmiss":
                assert point.ISI <= row.epsilon_ISI + 1.0e-5
            elif method == "group_TAC_ISI":
                assert point.TotEmiss <= row.epsilon_TotEmiss + 1.0e-5
            else:
                assert point.TAC <= row.epsilon_TAC + 1.0e-2


def main() -> None:
    full = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_full_raw.csv")
    reduced = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_reduced_raw.csv")
    log = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_solve_log.csv")
    frontier = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_frontier.csv")
    validation = pd.read_csv(RESULT_DIR / f"{PREFIX}orca_pareto_validation.csv")
    sensitivity = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_information_loss_sensitivity.csv"
    )
    source_sensitivity = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_candidate_source_sensitivity.csv"
    )
    convergence = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_grid_convergence.csv"
    ).iloc[0]
    structures = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_system_structures.csv"
    )

    assert len(full) == 834
    assert len(reduced) == 181
    assert set(full["termination"]) <= {"LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED"}
    for column in (
        "max_abs_equality",
        "max_abs_source_carbon_diagnostic",
        "max_inequality_violation",
        "epsilon_violation",
    ):
        assert max(full[column].max(), reduced[column].max()) <= 1.0e-6

    normalized = [f"{name}_normalized" for name in OBJECTIVES]
    assert len(frontier) == 225
    assert nondominated(frontier[normalized].to_numpy(dtype=float))
    assert frontier[normalized].min().min() >= -1.0e-8
    assert frontier[normalized].max().max() <= 1.0 + 1.0e-8

    top_orca = validation.loc[validation["orca_strength_01"].idxmax(), "grouping"]
    best_loss = validation.loc[
        validation["mean_information_loss_bins"].idxmin(), "grouping"
    ]
    best_coverage = validation.loc[validation["igd_mean"].idxmin(), "grouping"]
    assert top_orca == "Total emissions + ISI"
    assert best_loss == best_coverage == "TAC + total emissions"

    winners = sensitivity.loc[
        sensitivity.groupby(["metric", "resolution"])["information_loss"].idxmin(),
        "grouping",
    ]
    assert len(winners) == 14
    assert set(winners) == {"TAC + total emissions"}
    assert set(source_sensitivity.loc[source_sensitivity["bins_rank"] == 1, "grouping"]) == {
        "TAC + total emissions"
    }
    assert set(
        source_sensitivity.loc[source_sensitivity["neighbors_rank"] == 1, "grouping"]
    ) == {"TAC + total emissions"}
    assert convergence["p95_normalized_distance_fine_to_coarse"] < 0.10

    assert len(structures) == len(frontier)
    assert set(structures["main_sink"]) == {"Urea", "Saline Storage", "Greenhouse"}
    assert structures["pretreated_fraction"].between(0, 1).all()
    assert structures["direct_fraction"].between(0, 1).all()

    assert_nested_failures_have_witnesses(
        log.loc[log["method"].str.startswith("full_")], full
    )
    assert_nested_failures_have_witnesses(
        log.loc[log["method"].str.startswith("group_")], reduced
    )

    for stem in ("ccus_pareto_frontier_and_groupings", "ccus_orca_pareto_validation"):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = FIGURE_DIR / f"{PREFIX}{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10_000
        svg = FIGURE_DIR / f"{PREFIX}{stem}.svg"
        assert "<text" in svg.read_text(encoding="utf-8")

    report = RESULT_DIR / f"{PREFIX}pareto_orca_analysis.md"
    assert report.exists() and report.stat().st_size > 5_000
    print("Direct-use Pareto and ORCA validation checks passed")


if __name__ == "__main__":
    main()

