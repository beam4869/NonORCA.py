"""Plot the directly reoptimized weighted-sum solutions for TAC versus J."""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_weighted_sum_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
DOMINANCE_TOLERANCE = 1.0e-7
DUPLICATE_TOLERANCE = 1.0e-5
COLORS = {
    "reference": "#BCC4D2",
    "reference_line": "#8F9AAF",
    "urea": "#C9879A",
    "saline": "#668BB3",
    "tac": "#B96678",
    "grouped": "#557EA7",
    "grid": "#E5E7EB",
    "text": "#343A46",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(
            values <= point + DOMINANCE_TOLERANCE, axis=1
        ) & np.any(values < point - DOMINANCE_TOLERANCE, axis=1)
        keep[index] = not np.any(dominates)
    return keep


def deduplicate(frontier: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    ordered = frontier.sort_values([x, y]).copy()
    scaled = np.column_stack(
        [
            (ordered[x] - ordered[x].min())
            / (ordered[x].max() - ordered[x].min()),
            (ordered[y] - ordered[y].min())
            / (ordered[y].max() - ordered[y].min()),
        ]
    )
    positions: list[int] = []
    for position in range(len(ordered)):
        if not positions or np.max(
            np.abs(scaled[position] - scaled[positions[-1]])
        ) > DUPLICATE_TOLERANCE:
            positions.append(position)
    return ordered.iloc[positions].sort_values(x).reset_index(drop=True)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weighted = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_normalized_raw.csv"
    )
    weighted_keep = nondominated_mask(
        weighted[["TAC_normalized", "J_outer_normalized"]].to_numpy(dtype=float)
    )
    supported = deduplicate(
        weighted.loc[weighted_keep], "TAC", "J_normalized"
    )
    supported.insert(0, "supported_id", [f"WS{index:02d}" for index in range(1, len(supported) + 1)])
    weight_ranges = weighted.groupby("main_sink")["weight_TAC"].agg(["min", "max"])
    supported["weight_TAC_min"] = supported["main_sink"].map(weight_ranges["min"])
    supported["weight_TAC_max"] = supported["main_sink"].map(weight_ranges["max"])

    candidates = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_frontier.csv"
    )
    payoff = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_payoff_table.csv"
    )
    payoff = payoff.loc[payoff["anchor"] != "feasibility"]
    lower = payoff[OBJECTIVES].min().to_numpy(dtype=float)
    upper = payoff[OBJECTIVES].max().to_numpy(dtype=float)
    ranges = upper - lower
    candidates["TotEmiss_normalized_recomputed"] = (
        candidates["TotEmiss"] - lower[1]
    ) / ranges[1]
    candidates["ISI_normalized_recomputed"] = (
        candidates["ISI"] - lower[2]
    ) / ranges[2]
    candidates["TAC_normalized_recomputed"] = (
        candidates["TAC"] - lower[0]
    ) / ranges[0]
    candidates["J_normalized"] = 0.5 * (
        candidates["TotEmiss_normalized_recomputed"]
        + candidates["ISI_normalized_recomputed"]
    )
    reference_keep = nondominated_mask(
        candidates[["TAC_normalized_recomputed", "J_normalized"]].to_numpy(
            dtype=float
        )
    )
    reference = deduplicate(
        candidates.loc[reference_keep], "TAC", "J_normalized"
    )

    supported.to_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_supported_frontier.csv",
        index=False,
    )
    reference[
        [
            "frontier_id",
            "TAC",
            "TotEmiss",
            "ISI",
            "TAC_normalized_recomputed",
            "TotEmiss_normalized_recomputed",
            "ISI_normalized_recomputed",
            "J_normalized",
        ]
    ].to_csv(
        RESULT_DIR / "direct_use_expansion_weighted_sum_reference_frontier.csv",
        index=False,
    )
    return weighted, supported, reference


def plot_results(
    weighted: pd.DataFrame,
    supported: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.92, wspace=0.31)

    ax = axes[0]
    reference_ordered = reference.sort_values("J_normalized")
    ax.plot(
        reference_ordered["J_normalized"],
        reference_ordered["TAC"] / 1e6,
        color=COLORS["reference_line"],
        linewidth=1.0,
        alpha=0.72,
        label="Candidate-set nondominated envelope",
        zorder=1,
    )
    ax.scatter(
        reference_ordered["J_normalized"],
        reference_ordered["TAC"] / 1e6,
        s=12,
        color=COLORS["reference"],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    endpoint_colors = {
        "Urea": COLORS["urea"],
        "Saline Storage": COLORS["saline"],
    }
    for _, row in supported.iterrows():
        ax.scatter(
            row["J_normalized"],
            row["TAC"] / 1e6,
            s=62,
            color=endpoint_colors[row["main_sink"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label=f"Weighted-sum: {row['main_sink']}",
        )
        offset = (8, -20) if row["main_sink"] == "Saline Storage" else (-70, 12)
        ax.annotate(
            f"{row['main_sink']}\n"
            f"TAC={row['TAC'] / 1e6:.2f} M,  J={row['J_normalized']:.3f}",
            (row["J_normalized"], row["TAC"] / 1e6),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.0,
            color=COLORS["text"],
            arrowprops={"arrowstyle": "-", "color": endpoint_colors[row["main_sink"]], "linewidth": 0.7},
        )

    ax.set_xlabel(r"Normalized grouped objective, $J=0.5\hat{E}+0.5\widehat{ISI}$")
    ax.set_ylabel(r"TAC ($10^6$ cost units yr$^{-1}$)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.5)
    ax.text(-0.17, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1]
    ordered = weighted.sort_values("weight_TAC")
    ax.plot(
        ordered["weight_TAC"],
        ordered["TAC_normalized"],
        color=COLORS["tac"],
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        markevery=5,
        label="Normalized TAC",
    )
    ax.plot(
        ordered["weight_TAC"],
        ordered["J_outer_normalized"],
        color=COLORS["grouped"],
        linewidth=1.5,
        marker="s",
        markersize=2.4,
        markevery=5,
        label="Normalized grouped objective",
    )
    switch_weight = ordered.loc[
        ordered["main_sink"].eq("Urea"), "weight_TAC"
    ].min()
    ax.axvline(switch_weight - 0.005, color="#717784", linestyle="--", linewidth=0.8)
    ax.text(
        switch_weight + 0.012,
        0.52,
        f"Topology switch\n$w_{{TAC}}$ = {switch_weight:.2f}",
        fontsize=6.0,
        color=COLORS["text"],
        va="center",
    )
    ax.set_xlabel(r"Weighted-sum coefficient, $w_{TAC}$")
    ax.set_ylabel("Outer payoff-normalized value")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.055, 1.055)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=5.7)
    ax.text(-0.16, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

    stem = FIGURE_DIR / "direct_use_expansion_weighted_sum_normalized_frontier"
    for extension in ("png", "svg", "pdf", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    weighted, supported, reference = load_results()
    plot_results(weighted, supported, reference)
    print(f"Accepted weighted-sum solves: {len(weighted)}")
    print(f"Unique supported Pareto points: {len(supported)}")
    print(f"Reference candidate-set frontier points: {len(reference)}")
    print(
        supported[
            ["supported_id", "weight_TAC", "TAC", "TotEmiss", "ISI", "J_normalized", "main_sink"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
