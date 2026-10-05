from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"


def test_epsilon_outputs_are_feasible_and_complete() -> None:
    raw = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_raw.csv"
    )
    assert len(raw) == 121
    assert raw["epsilon_index"].tolist() == list(range(1, 122))
    assert raw["max_abs_equality"].max() <= 1.0e-6
    assert raw["max_abs_source_carbon_diagnostic"].max() <= 1.0e-6
    assert raw["max_inequality_violation"].max() <= 1.0e-6
    assert raw["epsilon_violation"].max() <= 1.0e-6


def test_screened_frontier_is_dense_and_nondominated() -> None:
    frontier = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_frontier.csv"
    )
    assert len(frontier) >= 100
    assert set(frontier["main_sink"]) == {"Urea", "Saline Storage"}
    assert np.all(np.diff(frontier["TAC"].to_numpy()) > 0.0)
    assert np.all(np.diff(frontier["J_normalized"].to_numpy()) < 0.0)

    values = frontier[["TAC_normalized", "J_outer_normalized"]].to_numpy()
    for index, point in enumerate(values):
        dominates = np.all(values <= point + 1.0e-7, axis=1) & np.any(
            values < point - 1.0e-7, axis=1
        )
        assert not np.any(dominates), f"frontier row {index} is dominated"


def test_epsilon_endpoints_match_payoff_table() -> None:
    frontier = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_frontier.csv"
    )
    payoff = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_payoff.csv"
    )
    minimum_tac = payoff.loc[payoff["endpoint"].eq("Minimum TAC")].iloc[0]
    minimum_j = payoff.loc[payoff["endpoint"].eq("Minimum J")].iloc[0]
    assert np.isclose(frontier.iloc[0]["TAC"], minimum_tac["TAC"], rtol=1.0e-8)
    assert np.isclose(
        frontier.iloc[-1]["J_normalized"], minimum_j["J_normalized"], rtol=1.0e-8
    )


def test_epsilon_figure_artifacts_exist() -> None:
    stem = FIGURE_DIR / "direct_use_expansion_epsilon_normalized_frontier"
    for extension in ("png", "pdf", "svg", "tiff"):
        path = stem.with_suffix(f".{extension}")
        assert path.exists()
        assert path.stat().st_size > 1000
