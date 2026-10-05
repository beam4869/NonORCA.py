from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"


def main() -> None:
    frontier = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_TAC_EI_pareto_frontier.csv"
    )
    keys = pd.read_csv(RESULT_DIR / "direct_use_expansion_TAC_EI_key_points.csv")
    links = pd.read_csv(RESULT_DIR / "direct_use_expansion_TAC_EI_sankey_links.csv")

    assert len(frontier) == 57
    assert frontier["TAC"].is_monotonic_increasing
    assert frontier["EI_grouped_raw"].is_monotonic_decreasing
    assert np.allclose(
        frontier["EI_grouped_raw"],
        0.20 * frontier["TotEmiss"] + 0.80 * frontier["ISI"],
    )
    assert (frontier["main_sink"] == "Greenhouse").sum() == 0
    assert set(keys["key_id"]) == {"A", "B", "C", "D", "E"}
    assert set(keys["key_role"]) == {
        "Minimum TAC",
        "10% saline allocation",
        "Sink-switch point",
        "90% saline allocation",
        "Minimum raw emissions + ISI",
    }
    assert keys["max_sink_mass_error"].max() < 1.0e-4
    assert set(links["flow_type"]) == {"treated", "direct"}
    assert (links["flow_kt_per_year"] > 1.0).all()

    stem = FIGURE_DIR / "direct_use_expansion_TAC_EI_pareto_sankey"
    for extension in ("png", "svg", "pdf", "tiff"):
        path = stem.with_suffix(f".{extension}")
        assert path.exists() and path.stat().st_size > 1_000
    with Image.open(stem.with_suffix(".png")) as image:
        assert image.width >= 2000
        assert image.height >= 1200

    print("Grouped Pareto and Sankey output checks passed")


if __name__ == "__main__":
    main()
