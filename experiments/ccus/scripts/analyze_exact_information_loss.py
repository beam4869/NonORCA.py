"""Analyze exact retained-objective Pareto slices using Russell-Allman Eqs. 10-11."""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_exact_info_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
PROFILE = os.environ.get("CCUS_PARETO_PROFILE", "direct_use_expansion")
LABEL = os.environ.get("CCUS_EXACT_INFO_LABEL", "main")
PREFIX = f"{PROFILE}_"
POINT_PREFIX = f"{PREFIX}exact_info_loss_{LABEL}_"
RETAINED_RESOLUTIONS = tuple(
    int(value)
    for value in os.environ.get("CCUS_EXACT_RETAINED_RESOLUTIONS", "11,21").split(",")
)
EPSILON_RESOLUTIONS = tuple(
    int(value)
    for value in os.environ.get("CCUS_EXACT_EPSILON_RESOLUTIONS", "11,21").split(",")
)
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
GROUPS = {
    "TAC + total emissions": {"grouped": ("TAC", "TotEmiss"), "retained": "ISI"},
    "TAC + ISI": {"grouped": ("TAC", "ISI"), "retained": "TotEmiss"},
    "Total emissions + ISI": {
        "grouped": ("TotEmiss", "ISI"),
        "retained": "TAC",
    },
}
COLORS = {
    "TAC + total emissions": "#8E9AAF",
    "TAC + ISI": "#E6A15C",
    "Total emissions + ISI": "#4C78A8",
}
SHORT = {
    "TAC + total emissions": "TAC + emissions",
    "TAC + ISI": "TAC + ISI",
    "Total emissions + ISI": "Emissions + ISI",
}
ACTIVE_TOLERANCE = 1.0e-6


def nondominated_mask(values: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        other = np.delete(values, index, axis=0)
        keep[index] = not np.any(
            np.all(other <= point + tolerance, axis=1)
            & np.any(other < point - tolerance, axis=1)
        )
    return keep


def embedded_indices(maximum: int, resolution: int) -> list[int]:
    if resolution > maximum or (maximum - 1) % (resolution - 1):
        raise ValueError(f"Resolution {resolution} is not embedded in {maximum}")
    stride = (maximum - 1) // (resolution - 1)
    return list(range(1, maximum + 1, stride))


def calculate_slices(
    points: pd.DataFrame,
    frontier: pd.DataFrame,
    lower: np.ndarray,
    ranges: np.ndarray,
    retained_resolution: int,
    epsilon_resolution: int,
    max_retained: int,
    max_epsilon: int,
) -> pd.DataFrame:
    retained_indices = embedded_indices(max_retained, retained_resolution)
    epsilon_indices = embedded_indices(max_epsilon, epsilon_resolution)
    rows = []
    for grouping, spec in GROUPS.items():
        retained = spec["retained"]
        retained_position = OBJECTIVES.index(retained)
        grouped_positions = [OBJECTIVES.index(name) for name in spec["grouped"]]
        group_points = points.loc[
            (points["grouping"] == grouping)
            & points["retained_active"]
            & points["globally_nondominated"]
            & points["retained_index"].isin(retained_indices)
            & points["epsilon_index"].isin(epsilon_indices)
        ]
        for retained_index in retained_indices:
            target_rows = points.loc[
                (points["grouping"] == grouping)
                & (points["retained_index"] == retained_index)
            ]
            if target_rows.empty:
                continue
            fraction = float(target_rows["retained_fraction"].iloc[0])
            target = float(target_rows["retained_target"].iloc[0])
            generated = group_points.loc[
                group_points["retained_index"] == retained_index, OBJECTIVES
            ]
            existing = frontier.loc[
                np.abs(frontier[retained].to_numpy(dtype=float) - target)
                <= ACTIVE_TOLERANCE * ranges[retained_position],
                OBJECTIVES,
            ]
            local = pd.concat([generated, existing], ignore_index=True)
            if local.empty:
                continue
            scaled = (local[OBJECTIVES].to_numpy(dtype=float) - lower) / ranges
            scaled = np.unique(np.round(scaled, 9), axis=0)
            component_ranges = np.ptp(scaled[:, grouped_positions], axis=0)
            rows.append(
                {
                    "retained_resolution": retained_resolution,
                    "epsilon_resolution": epsilon_resolution,
                    "grouping": grouping,
                    "retained": retained,
                    "retained_index": retained_index,
                    "retained_fraction": fraction,
                    "points_in_slice": len(scaled),
                    "normalized_range_1": component_ranges[0],
                    "normalized_range_2": component_ranges[1],
                    "information_loss": component_ranges.sum(),
                }
            )
    return pd.DataFrame(rows)


def summarize(slices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, subset in slices.groupby(
        ["retained_resolution", "epsilon_resolution", "grouping"]
    ):
        retained_resolution, epsilon_resolution, grouping = keys
        values = subset["information_loss"].to_numpy(dtype=float)
        rows.append(
            {
                "retained_resolution": retained_resolution,
                "epsilon_resolution": epsilon_resolution,
                "grouping": grouping,
                "requested_slices": retained_resolution,
                "represented_slices": len(subset),
                "multi_point_slices": int((subset["points_in_slice"] >= 2).sum()),
                "mean_information_loss": values.mean(),
                "sd_across_slices": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min_information_loss": values.min(),
                "max_information_loss": values.max(),
            }
        )
    result = pd.DataFrame(rows)
    result["rank"] = (
        result.groupby(["retained_resolution", "epsilon_resolution"])[
            "mean_information_loss"
        ]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    return result


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    final = summary.loc[
        (summary["retained_resolution"] == max(RETAINED_RESOLUTIONS))
        & (summary["epsilon_resolution"] == max(EPSILON_RESOLUTIONS))
    ].copy()
    previous = pd.read_csv(RESULT_DIR / f"{PREFIX}orca_pareto_validation.csv")
    selected = previous[
        [
            "grouping",
            "orca_strength_01",
            "orca_signed_correlation",
            "mean_information_loss_bins",
            "sd_information_loss_bins",
            "mean_information_loss_neighbors",
            "sd_information_loss_neighbors",
        ]
    ]
    comparison = final.merge(selected, on="grouping", how="left")
    comparison["exact_loss_rank"] = comparison["mean_information_loss"].rank(
        ascending=True, method="min"
    ).astype(int)
    comparison["orca_rank"] = comparison["orca_strength_01"].rank(
        ascending=False, method="min"
    ).astype(int)
    return comparison.sort_values("exact_loss_rank")


def plot_results(slices: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    max_retained = max(RETAINED_RESOLUTIONS)
    max_epsilon = max(EPSILON_RESOLUTIONS)
    final_slices = slices.loc[
        (slices["retained_resolution"] == max_retained)
        & (slices["epsilon_resolution"] == max_epsilon)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for grouping in GROUPS:
        subset = final_slices.loc[final_slices["grouping"] == grouping].sort_values(
            "retained_fraction"
        )
        axes[0].plot(
            subset["retained_fraction"],
            subset["information_loss"],
            marker="o",
            ms=3,
            lw=1.3,
            color=COLORS[grouping],
            label=SHORT[grouping],
        )
    axes[0].set_xlabel("Normalized retained-objective level, $p$")
    axes[0].set_ylabel("Exact-slice information loss, $B_p$")
    axes[0].set_title("Conditional Pareto slices")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=5.7, loc="upper left")

    ordered = comparison.sort_values("exact_loss_rank")
    x = np.arange(len(ordered))
    width = 0.24
    axes[1].bar(
        x - width,
        ordered["mean_information_loss"],
        width,
        yerr=np.vstack(
            [
                np.minimum(
                    ordered["mean_information_loss"], ordered["sd_across_slices"]
                ),
                ordered["sd_across_slices"],
            ]
        ),
        color=[COLORS[name] for name in ordered["grouping"]],
        edgecolor="black",
        linewidth=0.45,
        capsize=2,
        label="Exact conditional",
    )
    axes[1].bar(
        x,
        ordered["mean_information_loss_bins"],
        width,
        color="white",
        edgecolor=[COLORS[name] for name in ordered["grouping"]],
        linewidth=1.1,
        label="Equal-bin approximation",
    )
    axes[1].bar(
        x + width,
        ordered["mean_information_loss_neighbors"],
        width,
        color="#D9D9D9",
        edgecolor="black",
        linewidth=0.45,
        label="Nearest-neighbor approximation",
    )
    axes[1].set_xticks(x, [SHORT[name].replace(" + ", "\n+") for name in ordered["grouping"]])
    axes[1].set_ylabel("Average information loss")
    axes[1].set_title("Exact vs earlier approximations")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=5.3)

    configurations = summary.copy()
    configurations["configuration"] = configurations.apply(
        lambda row: f'{int(row["retained_resolution"])}×{int(row["epsilon_resolution"])}',
        axis=1,
    )
    config_order = [
        f"{r}×{e}" for r in RETAINED_RESOLUTIONS for e in EPSILON_RESOLUTIONS
    ]
    for grouping in GROUPS:
        subset = configurations.loc[configurations["grouping"] == grouping].set_index(
            "configuration"
        ).reindex(config_order)
        axes[2].plot(
            np.arange(len(config_order)),
            subset["mean_information_loss"],
            marker="o",
            ms=3.3,
            lw=1.3,
            color=COLORS[grouping],
            label=SHORT[grouping],
        )
    axes[2].set_xticks(np.arange(len(config_order)), config_order)
    axes[2].set_xlabel("Retained × epsilon levels")
    axes[2].set_ylabel("Average information loss")
    axes[2].set_title("Grid convergence")
    axes[2].set_ylim(bottom=0)

    for label, axis in zip("abc", axes):
        axis.text(-0.17, 1.05, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
        axis.tick_params(width=0.7, length=3, labelsize=6.2)
    fig.tight_layout(w_pad=1.8)
    stem = FIGURE_DIR / f"{PREFIX}exact_information_loss_{LABEL}_validation"
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = pd.read_csv(RESULT_DIR / f"{POINT_PREFIX}points.csv")
    frontier = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_frontier.csv")
    max_retained = int(points["retained_index"].max())
    max_epsilon = int(points["epsilon_index"].max())

    candidate_values = np.vstack(
        [points[OBJECTIVES].to_numpy(dtype=float), frontier[OBJECTIVES].to_numpy(dtype=float)]
    )
    provisional_lower = candidate_values.min(axis=0)
    provisional_ranges = candidate_values.max(axis=0) - provisional_lower
    scaled_candidates = (candidate_values - provisional_lower) / provisional_ranges
    keep = nondominated_mask(scaled_candidates)
    nondominated_values = candidate_values[keep]
    lower = nondominated_values.min(axis=0)
    upper = nondominated_values.max(axis=0)
    ranges = upper - lower
    points = points.copy()
    points["globally_nondominated"] = keep[: len(points)]

    all_slices = []
    for retained_resolution in RETAINED_RESOLUTIONS:
        for epsilon_resolution in EPSILON_RESOLUTIONS:
            all_slices.append(
                calculate_slices(
                    points,
                    frontier,
                    lower,
                    ranges,
                    retained_resolution,
                    epsilon_resolution,
                    max_retained,
                    max_epsilon,
                )
            )
    slices = pd.concat(all_slices, ignore_index=True)
    summary = summarize(slices)
    comparison = comparison_table(summary)

    points.to_csv(RESULT_DIR / f"{POINT_PREFIX}points_audited.csv", index=False)
    slices.to_csv(RESULT_DIR / f"{POINT_PREFIX}slices.csv", index=False)
    summary.to_csv(RESULT_DIR / f"{POINT_PREFIX}summary.csv", index=False)
    comparison.to_csv(RESULT_DIR / f"{POINT_PREFIX}comparison.csv", index=False)
    bounds = pd.DataFrame({"objective": OBJECTIVES, "lower": lower, "upper": upper})
    bounds.to_csv(RESULT_DIR / f"{POINT_PREFIX}normalization_bounds.csv", index=False)
    plot_results(slices, summary, comparison)

    print(comparison.to_string(index=False))
    print("\nGrid summaries:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
