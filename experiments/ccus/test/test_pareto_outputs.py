"""Validation checks for deterministic Pareto and ORCA comparison outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
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


def main() -> None:
    raw = pd.read_csv(RESULT_DIR / "pareto_full_raw.csv")
    reduced = pd.read_csv(RESULT_DIR / "pareto_reduced_raw.csv")
    frontier = pd.read_csv(RESULT_DIR / "pareto_frontier.csv")
    validation = pd.read_csv(RESULT_DIR / "orca_pareto_validation.csv")
    sensitivity = pd.read_csv(RESULT_DIR / "pareto_information_loss_sensitivity.csv")
    convergence = pd.read_csv(RESULT_DIR / "pareto_grid_convergence.csv").iloc[0]

    assert len(raw) == 457
    assert len(reduced) == 123
    assert set(raw["termination"]) <= {"LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED"}
    assert raw["max_abs_equality"].max() <= 1.0e-6
    assert raw["max_abs_source_carbon_diagnostic"].max() <= 1.0e-6
    assert raw["max_inequality_violation"].max() <= 1.0e-6
    assert raw["epsilon_violation"].max() <= 1.0e-6

    normalized_columns = [f"{name}_normalized" for name in OBJECTIVES]
    assert len(frontier) == 136
    assert nondominated(frontier[normalized_columns].to_numpy(dtype=float))
    assert frontier[normalized_columns].min().min() >= -1.0e-8
    assert frontier[normalized_columns].max().max() <= 1.0 + 1.0e-8

    best_orca = validation.loc[validation["orca_strength_01"].idxmax(), "grouping"]
    best_loss = validation.loc[
        validation["mean_information_loss_bins"].idxmin(), "grouping"
    ]
    best_coverage = validation.loc[validation["igd_mean"].idxmin(), "grouping"]
    assert best_orca == best_loss == best_coverage == "Total emissions + ISI"
    winners = sensitivity.loc[
        sensitivity.groupby(["metric", "resolution"])["information_loss"].idxmin(),
        "grouping",
    ]
    assert len(winners) == 14
    assert set(winners) == {"Total emissions + ISI"}
    assert convergence["p95_normalized_distance_fine_to_coarse"] < 0.10

    for stem in ("ccus_pareto_frontier_and_groupings", "ccus_orca_pareto_validation"):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = FIGURE_DIR / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10_000
        assert "<text" in (FIGURE_DIR / f"{stem}.svg").read_text(encoding="utf-8")
    print("Pareto and ORCA validation checks passed")


if __name__ == "__main__":
    main()
