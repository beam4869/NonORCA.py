from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"


def test_all_screening_scenarios_are_feasible() -> None:
    status = pd.read_csv(RESULT_DIR / "supply_chain_correlation_screen_status.csv")
    assert len(status) >= 60
    assert status["feasible"].all()
    assert status["accepted_objectives"].eq(3).all()


def test_condition2_main_orca_selects_tac_emissions() -> None:
    prefix = "supply_chain_condition2_paper_orca_condition2_main_"
    summary = pd.read_csv(RESULT_DIR / f"{prefix}summary.csv")
    main = summary.loc[summary["constraint_handling"].eq("equality_tangent")].set_index(
        "pair"
    )
    assert main.loc["TAC__TotEmiss", "replicates"] == 10
    assert main.loc["TAC__TotEmiss", "strongest_count"] == 10
    assert main.loc["TAC__TotEmiss", "signed_mean"] > main.loc["TAC__ISI", "signed_mean"]
    assert main.loc["TAC__TotEmiss", "signed_mean"] > main.loc["TotEmiss__ISI", "signed_mean"]

    groups = pd.read_csv(RESULT_DIR / f"{prefix}groupings.csv")
    for method in ("leiden", "average"):
        assert (groups[f"{method}_TAC"] == groups[f"{method}_TotEmiss"]).all()
        assert (groups[f"{method}_TAC"] != groups[f"{method}_ISI"]).all()


def test_condition2_points_are_feasible() -> None:
    points = pd.read_csv(
        RESULT_DIR / "supply_chain_condition2_paper_orca_condition2_main_points.csv"
    )
    assert len(points) == 1230
    assert points["max_abs_equality"].max() <= 1.0e-6
    assert points["max_abs_source_carbon_diagnostic"].max() <= 1.0e-6
    assert points["max_inequality_violation"].max() <= 1.0e-6
    assert np.isfinite(points[["TAC", "TotEmiss", "ISI"]].to_numpy()).all()


def test_condition2_step_sensitivity_keeps_pair_rank() -> None:
    for label in ("condition2_step_low", "condition2_step_high"):
        summary = pd.read_csv(
            RESULT_DIR / f"supply_chain_condition2_paper_orca_{label}_summary.csv"
        )
        main = summary.loc[
            summary["constraint_handling"].eq("equality_tangent")
        ].set_index("pair")
        assert main.loc["TAC__TotEmiss", "strongest_count"] == 5
        assert main.loc["TAC__TotEmiss", "signed_mean"] > main.loc["TAC__ISI", "signed_mean"]
        assert main.loc["TAC__TotEmiss", "signed_mean"] > main.loc["TotEmiss__ISI", "signed_mean"]


def test_condition_comparison_artifacts_exist() -> None:
    assert (RESULT_DIR / "supply_chain_condition_orca_analysis.md").stat().st_size > 1000
    assert (RESULT_DIR / "supply_chain_condition_comparison_source_data.csv").stat().st_size > 1000
    parameters = pd.read_csv(
        RESULT_DIR / "supply_chain_condition_parameter_comparison.csv"
    )
    assert len(parameters) == 12
    assert set(parameters["condition"]) == {
        "Supply-chain condition 1",
        "Supply-chain condition 2",
    }
    stem = FIGURE_DIR / "supply_chain_condition_orca_comparison"
    for extension in ("png", "pdf", "svg", "tiff"):
        path = stem.with_suffix(f".{extension}")
        assert path.exists()
        assert path.stat().st_size > 1000
