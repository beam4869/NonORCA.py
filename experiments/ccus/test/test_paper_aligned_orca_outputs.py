"""Integrity checks for the paper-aligned nonlinear ORCA experiment."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
PREFIX = "direct_use_expansion_paper_orca_paper_main_"


def main() -> None:
    points = pd.read_csv(RESULT_DIR / f"{PREFIX}points.csv")
    attempts = pd.read_csv(RESULT_DIR / f"{PREFIX}projection_attempts.csv")
    summary = pd.read_csv(RESULT_DIR / f"{PREFIX}summary.csv")
    comparison = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_comparison.csv"
    )
    sensitivity = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_sensitivity.csv"
    )
    groupings = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_groupings.csv"
    )
    pareto_validation = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_orca_pareto_validation.csv"
    )

    assert len(points) == 2460
    assert points["replicate"].nunique() == 20
    assert points.groupby(["replicate", "seed_objective"]).size().eq(41).all()
    assert points["max_abs_equality"].max() <= 1.0e-6
    assert points["max_abs_source_carbon_diagnostic"].max() <= 1.0e-6
    assert points["max_inequality_violation"].max() <= 1.0e-6

    assert len(attempts) == 2400
    assert attempts["projection_accepted"].all()
    assert attempts["kept"].all()
    assert set(attempts["termination"]) <= {
        "LOCALLY_SOLVED",
        "ALMOST_LOCALLY_SOLVED",
    }

    primary = summary.loc[summary["constraint_handling"] == "equality_tangent"].set_index(
        "pair"
    )
    audit = summary.loc[summary["constraint_handling"] == "inequalities_only"].set_index(
        "pair"
    )
    assert set(primary.index) == {"TAC__TotEmiss", "TAC__ISI", "TotEmiss__ISI"}
    assert primary.loc["TotEmiss__ISI", "strongest_count"] == 20
    assert primary.loc["TAC__TotEmiss", "rank_mean"] == 2
    assert primary.loc["TAC__ISI", "rank_mean"] == 3
    assert np.isclose(primary.loc["TotEmiss__ISI", "signed_mean"], 0.7662818074)
    assert audit["signed_mean"].idxmax() == "TotEmiss__ISI"

    assert comparison.set_index("pair").loc["TAC__TotEmiss", "endpoint_rank"] == 3
    assert comparison.set_index("pair").loc["TAC__TotEmiss", "paper_rank"] == 2
    assert comparison.set_index("pair").loc["TAC__ISI", "endpoint_rank"] == 2
    assert comparison.set_index("pair").loc["TAC__ISI", "paper_rank"] == 3

    for _, subset in sensitivity.groupby("run_label"):
        strongest = subset.loc[subset["signed_mean"].idxmax()]
        assert strongest["pair"] == "TotEmiss__ISI"
        assert strongest["strongest_count"] == strongest["replicates"]

    assert len(groupings) == 80
    for prefix in ("leiden", "average"):
        assert (groupings[f"{prefix}_TotEmiss"] == groupings[f"{prefix}_ISI"]).all()
        assert (groupings[f"{prefix}_TAC"] != groupings[f"{prefix}_TotEmiss"]).all()

    pareto = pareto_validation.set_index("grouping")
    assert pareto.loc["Total emissions + ISI", "orca_rank_descending"] == 1
    assert pareto.loc["TAC + total emissions", "information_loss_rank_ascending"] == 1
    assert np.isclose(
        pareto.loc["Total emissions + ISI", "orca_signed_correlation"],
        primary.loc["TotEmiss__ISI", "signed_mean"],
    )

    stem = FIGURE_DIR / "direct_use_expansion_paper_aligned_orca_validation"
    for extension in ("svg", "pdf", "png", "tiff"):
        path = stem.with_suffix(f".{extension}")
        assert path.exists() and path.stat().st_size > 10_000
    assert "<text" in stem.with_suffix(".svg").read_text(encoding="utf-8")

    report = RESULT_DIR / "direct_use_expansion_paper_aligned_orca_analysis.md"
    assert report.read_text(encoding="utf-8").startswith("# Material Passport")
    assert report.stat().st_size > 5_000
    print("Paper-aligned nonlinear ORCA checks passed")


if __name__ == "__main__":
    main()
