"""Compare endpoint and paper-aligned nonlinear ORCA results."""

from __future__ import annotations

import os
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["legend.frameon"] = False

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
LABELS = ["paper_main", "step_low", "step_high", "points_low", "points_high"]
PAIR_ORDER = ["TAC__TotEmiss", "TAC__ISI", "TotEmiss__ISI"]
DISPLAY = {
    "TAC__TotEmiss": "TAC + emissions",
    "TAC__ISI": "TAC + ISI",
    "TotEmiss__ISI": "Emissions + ISI",
}
COLORS = {
    "TAC__TotEmiss": "#8E9AAF",
    "TAC__ISI": "#E6A15C",
    "TotEmiss__ISI": "#4C78A8",
}
CONFIG_DISPLAY = {
    "step_low": "0.01\n40",
    "paper_main": "0.03\n40",
    "step_high": "0.05\n40",
    "points_low": "0.03\n20",
    "points_high": "0.03\n80",
}


def load_all_pairwise() -> pd.DataFrame:
    frames = [
        pd.read_csv(
            RESULT_DIR / f"direct_use_expansion_paper_orca_{label}_pairwise.csv"
        )
        for label in LABELS
    ]
    return pd.concat(frames, ignore_index=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(FIGURE_DIR / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_pairwise = load_all_pairwise()
    primary = all_pairwise.loc[
        (all_pairwise["run_label"] == "paper_main")
        & (all_pairwise["constraint_handling"] == "equality_tangent")
    ].copy()
    endpoint = pd.read_csv(RESULT_DIR / "diversity_orca_pairwise.csv")
    endpoint = endpoint.loc[endpoint["scenario"] == "direct_use_expansion"].copy()

    primary_summary = (
        primary.groupby("pair", as_index=False)
        .agg(
            paper_signed_mean=("orca_signed_correlation", "mean"),
            paper_signed_sd=("orca_signed_correlation", "std"),
            paper_signed_min=("orca_signed_correlation", "min"),
            paper_signed_max=("orca_signed_correlation", "max"),
            paper_adjacency_mean=("orca_adjacency_01", "mean"),
            replicates=("replicate", "nunique"),
        )
    )
    comparison = endpoint[
        ["pair", "orca_signed_correlation", "orca_adjacency_01"]
    ].rename(
        columns={
            "orca_signed_correlation": "endpoint_signed",
            "orca_adjacency_01": "endpoint_adjacency",
        }
    ).merge(primary_summary, on="pair", validate="one_to_one")
    comparison["delta_paper_minus_endpoint_signed"] = (
        comparison["paper_signed_mean"] - comparison["endpoint_signed"]
    )
    comparison["endpoint_rank"] = comparison["endpoint_signed"].rank(
        ascending=False, method="min"
    ).astype(int)
    comparison["paper_rank"] = comparison["paper_signed_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    comparison.to_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_comparison.csv",
        index=False,
    )

    sensitivity = (
        all_pairwise.loc[all_pairwise["constraint_handling"] == "equality_tangent"]
        .groupby(["run_label", "pair"], as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            selected_points_mean=("selected_points", "mean"),
            signed_mean=("orca_signed_correlation", "mean"),
            signed_sd=("orca_signed_correlation", "std"),
            signed_min=("orca_signed_correlation", "min"),
            signed_max=("orca_signed_correlation", "max"),
            strongest_count=("rank_descending", lambda values: int((values == 1).sum())),
        )
    )
    sensitivity.to_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_sensitivity.csv",
        index=False,
    )

    grouping_frames = []
    for label in LABELS:
        groupings = pd.read_csv(
            RESULT_DIR / f"direct_use_expansion_paper_orca_{label}_groupings.csv"
        )
        grouping_frames.append(groupings)
    groupings = pd.concat(grouping_frames, ignore_index=True)
    groupings.to_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_groupings.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.42})
    x = np.arange(3)
    ordered_comparison = comparison.set_index("pair").loc[PAIR_ORDER]
    axes[0].bar(
        x - 0.18,
        ordered_comparison["endpoint_signed"],
        width=0.36,
        color="white",
        edgecolor=[COLORS[pair] for pair in PAIR_ORDER],
        linewidth=1.25,
        label="Three endpoints",
    )
    axes[0].bar(
        x + 0.18,
        ordered_comparison["paper_signed_mean"],
        yerr=ordered_comparison["paper_signed_sd"],
        width=0.36,
        color=[COLORS[pair] for pair in PAIR_ORDER],
        edgecolor="black",
        linewidth=0.45,
        error_kw={"elinewidth": 0.7, "capsize": 2.0, "capthick": 0.7},
        label="Full nonlinear ORCA",
    )
    axes[0].set_xticks(x, [DISPLAY[pair].replace(" + ", "\n+") for pair in PAIR_ORDER])
    axes[0].set_ylabel("Signed ORCA correlation")
    axes[0].set_ylim(0.42, 0.82)
    axes[0].set_title("Endpoint vs full workflow", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=6.2)

    rng = np.random.default_rng(37)
    for pair_index, pair in enumerate(PAIR_ORDER):
        values = primary.loc[
            primary["pair"] == pair, "orca_signed_correlation"
        ].to_numpy()
        jitter = rng.uniform(-0.075, 0.075, len(values))
        axes[1].scatter(
            pair_index + jitter,
            values,
            s=13,
            color=COLORS[pair],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.85,
        )
        axes[1].plot(
            [pair_index - 0.14, pair_index + 0.14],
            [values.mean(), values.mean()],
            color="black",
            lw=1.2,
        )
    axes[1].set_xticks(x, [DISPLAY[pair].replace(" + ", "\n+") for pair in PAIR_ORDER])
    axes[1].set_ylim(0.42, 0.82)
    axes[1].set_title("Twenty random replicates", fontsize=8)
    axes[1].set_ylabel("Signed ORCA correlation")

    config_order = ["step_low", "paper_main", "step_high", "points_low", "points_high"]
    for pair in PAIR_ORDER:
        local = sensitivity.set_index(["run_label", "pair"])
        means = [local.loc[(label, pair), "signed_mean"] for label in config_order]
        errors = [local.loc[(label, pair), "signed_sd"] for label in config_order]
        axes[2].errorbar(
            np.arange(len(config_order)),
            means,
            yerr=errors,
            marker="o",
            ms=3.5,
            lw=1.15,
            capsize=2,
            color=COLORS[pair],
            label=DISPLAY[pair],
        )
    axes[2].set_xticks(
        np.arange(len(config_order)),
        [CONFIG_DISPLAY[label] for label in config_order],
    )
    axes[2].set_xlabel("Step size / points per seed")
    axes[2].set_ylim(0.42, 0.82)
    axes[2].set_ylabel("Signed ORCA correlation")
    axes[2].set_title("Sampling sensitivity", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=5.8)

    for label, ax in zip("abc", axes):
        ax.text(-0.17, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.tick_params(width=0.7, length=3, labelsize=6.3)
    save_figure(fig, "direct_use_expansion_paper_aligned_orca_validation")
    print(comparison.sort_values("paper_rank").to_string(index=False))
    print("\nSensitivity strongest counts:")
    print(
        sensitivity.loc[sensitivity["strongest_count"] > 0]
        .sort_values(["run_label", "pair"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
