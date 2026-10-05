"""Integrity checks for the paper-faithful conditional information-loss run."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
STEM = "direct_use_expansion_exact_info_loss_quantile21_"
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
EXPECTED_RANK = {
    "Total emissions + ISI": 1,
    "TAC + total emissions": 2,
    "TAC + ISI": 3,
}


def main() -> None:
    points = pd.read_csv(RESULT_DIR / f"{STEM}points.csv")
    audited = pd.read_csv(RESULT_DIR / f"{STEM}points_audited.csv")
    solve_log = pd.read_csv(RESULT_DIR / f"{STEM}solve_log.csv")
    slices = pd.read_csv(RESULT_DIR / f"{STEM}slices.csv")
    summary = pd.read_csv(RESULT_DIR / f"{STEM}summary.csv")
    comparison = pd.read_csv(RESULT_DIR / f"{STEM}comparison.csv")
    bounds = pd.read_csv(RESULT_DIR / f"{STEM}normalization_bounds.csv")
    frontier = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_frontier.csv"
    )

    assert len(solve_log) == 3 * 11 * 21
    assert int(solve_log["accepted"].sum()) == len(points) == len(audited)
    for column in (
        "max_abs_equality",
        "max_inequality_violation",
        "epsilon_violation",
    ):
        assert points[column].max() <= 1.0e-6
    active = points.loc[points["retained_active"]]
    assert len(active) > 0
    assert active["retained_normalized_error"].max() <= 1.0e-6

    assert set(bounds["objective"]) == set(OBJECTIVES)
    assert np.all(bounds["upper"].to_numpy() > bounds["lower"].to_numpy())
    objective_ranges = dict(
        zip(bounds["objective"], bounds["upper"] - bounds["lower"])
    )
    targets = points[
        ["grouping", "retained", "retained_index", "retained_target"]
    ].drop_duplicates()
    assert len(targets) == 33
    for row in targets.itertuples(index=False):
        distance = np.abs(
            frontier[row.retained].to_numpy(dtype=float) - row.retained_target
        ).min()
        assert distance <= 1.0e-6 * objective_ranges[row.retained]

    assert set(summary["retained_resolution"]) == {6, 11}
    assert set(summary["epsilon_resolution"]) == {11, 21}
    assert len(summary) == 12
    for _, subset in summary.groupby(
        ["retained_resolution", "epsilon_resolution"]
    ):
        ranks = dict(zip(subset["grouping"], subset["rank"]))
        assert ranks == EXPECTED_RANK
        assert (subset["represented_slices"] == subset["requested_slices"]).all()

    assert slices["information_loss"].between(0.0, 2.0 + 1.0e-8).all()
    recalculated = slices.groupby(
        ["retained_resolution", "epsilon_resolution", "grouping"]
    )["information_loss"].mean()
    reported = summary.set_index(
        ["retained_resolution", "epsilon_resolution", "grouping"]
    )["mean_information_loss"]
    assert np.allclose(recalculated.sort_index(), reported.sort_index())

    assert dict(zip(comparison["grouping"], comparison["exact_loss_rank"])) == (
        EXPECTED_RANK
    )
    assert dict(zip(comparison["grouping"], comparison["orca_rank"])) == (
        EXPECTED_RANK
    )
    assert np.allclose(
        comparison.sort_values("exact_loss_rank")["mean_information_loss"],
        [0.2899573385, 0.3452983280, 0.7428999238],
        atol=1.0e-8,
    )

    figure_stem = (
        FIGURE_DIR
        / "direct_use_expansion_exact_information_loss_quantile21_validation"
    )
    for extension in ("svg", "pdf", "png", "tiff"):
        path = figure_stem.with_suffix(f".{extension}")
        assert path.exists() and path.stat().st_size > 10_000
    assert "<text" in figure_stem.with_suffix(".svg").read_text(encoding="utf-8")

    report = RESULT_DIR / "direct_use_expansion_exact_information_loss_analysis.md"
    assert report.exists() and report.stat().st_size > 5_000
    print("Exact conditional information-loss checks passed")


if __name__ == "__main__":
    main()
