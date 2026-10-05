from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"


def main() -> None:
    raw = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_normalized_raw.csv"
    )
    supported = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_supported_frontier.csv"
    )
    reference = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_reference_frontier.csv"
    )

    assert len(raw) == 101
    assert np.allclose(raw["weight_TAC"] + raw["weight_grouped"], 1.0)
    assert raw["max_abs_equality"].max() <= 1.0e-6
    assert raw["max_abs_source_carbon_diagnostic"].max() <= 1.0e-6
    assert raw["max_inequality_violation"].max() <= 1.0e-6
    assert set(raw["main_sink"]) == {"Urea", "Saline Storage"}
    assert raw.loc[raw["main_sink"].eq("Urea"), "weight_TAC"].min() == 0.51

    assert len(supported) == 2
    assert set(supported["main_sink"]) == {"Urea", "Saline Storage"}
    values = supported[["TAC", "J_normalized"]].to_numpy(dtype=float)
    for point in values:
        dominates = np.all(values <= point + 1.0e-7, axis=1) & np.any(
            values < point - 1.0e-7, axis=1
        )
        assert not np.any(dominates)
    assert len(reference) == 54

    stem = FIGURE_DIR / "direct_use_expansion_weighted_sum_normalized_frontier"
    for extension in ("png", "svg", "pdf", "tiff"):
        path = stem.with_suffix(f".{extension}")
        assert path.exists() and path.stat().st_size > 1_000
    with Image.open(stem.with_suffix(".png")) as image:
        assert image.width >= 1800
        assert image.height >= 700

    print("Weighted-sum normalized output checks passed")


if __name__ == "__main__":
    main()
