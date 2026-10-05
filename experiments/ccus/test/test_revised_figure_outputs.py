"""Integrity checks for the revised Pareto and ORCA demonstration figures."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
STEMS = (
    "direct_use_expansion_pareto_3d_standalone",
    "direct_use_expansion_grouping_projections_horizontal",
    "direct_use_expansion_pareto_orca_network",
)


def main() -> None:
    for stem in STEMS:
        for extension in ("svg", "pdf", "png", "tiff"):
            path = FIGURE_DIR / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10_000
        svg = (FIGURE_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg

    retained = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_grouping_projection_retained_points.csv"
    )
    assert retained.groupby("grouping").size().to_dict() == {
        "TAC + ISI": 61,
        "TAC + total emissions": 61,
        "Total emissions + ISI": 54,
    }
    projected = {
        name: len(
            subset[["TotEmiss_normalized", "ISI_normalized"]]
            .round(6)
            .drop_duplicates()
        )
        for name, subset in retained.groupby("grouping")
    }
    assert projected == {
        "TAC + ISI": 1,
        "TAC + total emissions": 24,
        "Total emissions + ISI": 18,
    }

    network = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_orca_network_source.csv"
    )
    assert set(network["pair"]) == {
        "TAC__TotEmiss",
        "TAC__ISI",
        "TotEmiss__ISI",
    }
    assert np.allclose(
        network["orca_adjacency"], (network["orca_signed"] + 1.0) / 2.0
    )
    assert network.loc[
        network["orca_adjacency"].idxmax(), "pair"
    ] == "TotEmiss__ISI"
    network_svg = (
        FIGURE_DIR / "direct_use_expansion_pareto_orca_network.svg"
    ).read_text(encoding="utf-8")
    for required in ("Selected group", "0.778", "0.756", "0.883"):
        assert required in network_svg
    for removed in (
        "ORCA-selected group",
        "ORCA objective network",
        "Edge width and color encode adjacency",
        "n=45",
        "n=73",
        "n=107",
    ):
        assert removed not in network_svg
    print("Revised figure output checks passed")


if __name__ == "__main__":
    main()
