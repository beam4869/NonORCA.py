"""Summarize raw objective-gradient cosine for the primary CCUS run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "direct_use_expansion_paper_orca_paper_main_objective_gradients.csv"
RAW_OUTPUT = RESULTS / "direct_use_expansion_paper_orca_paper_main_raw_gradient_cosine.csv"
SUMMARY_OUTPUT = RESULTS / "direct_use_expansion_paper_orca_paper_main_raw_gradient_cosine_summary.csv"
OBJECTIVES = ("TAC", "TotEmiss", "ISI")
PAIRS = (("TAC", "TotEmiss"), ("TAC", "ISI"), ("TotEmiss", "ISI"))


def numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix)],
        key=lambda column: int(column[len(prefix) :]),
    )


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0.0:
        raise ValueError("A stored objective gradient has zero norm")
    return float(np.dot(first, second) / denominator)


def main() -> None:
    gradients = pd.read_csv(INPUT)
    derivative_columns = numbered_columns(gradients, "dz")
    raw_rows: list[dict[str, object]] = []

    for replicate, replicate_frame in gradients.groupby("replicate", sort=True):
        pair_values: dict[str, list[float]] = {
            f"{first}__{second}": [] for first, second in PAIRS
        }
        for _, point_frame in replicate_frame.groupby("point_id", sort=False):
            by_objective = point_frame.set_index("objective")
            missing = set(OBJECTIVES).difference(by_objective.index)
            if missing:
                raise ValueError(f"Missing objective rows: {sorted(missing)}")
            for first, second in PAIRS:
                first_gradient = by_objective.loc[first, derivative_columns].to_numpy(dtype=float)
                second_gradient = by_objective.loc[second, derivative_columns].to_numpy(dtype=float)
                pair_values[f"{first}__{second}"].append(
                    cosine(first_gradient, second_gradient)
                )

        replicate_means = {
            pair: float(np.mean(values)) for pair, values in pair_values.items()
        }
        ordered = sorted(replicate_means, key=replicate_means.get, reverse=True)
        rank = {pair: index + 1 for index, pair in enumerate(ordered)}
        for pair, value in replicate_means.items():
            raw_rows.append(
                {
                    "replicate": int(replicate),
                    "pair": pair,
                    "raw_gradient_cosine": value,
                    "rank_descending": rank[pair],
                }
            )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(RAW_OUTPUT, index=False)
    summary = (
        raw.groupby("pair", as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            mean=("raw_gradient_cosine", "mean"),
            sample_sd=("raw_gradient_cosine", "std"),
            minimum=("raw_gradient_cosine", "min"),
            maximum=("raw_gradient_cosine", "max"),
            strongest_count=("rank_descending", lambda values: int((values == 1).sum())),
        )
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(SUMMARY_OUTPUT, index=False)


if __name__ == "__main__":
    main()
