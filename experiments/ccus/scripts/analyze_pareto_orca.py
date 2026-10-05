"""Analyze the deterministic CCUS Pareto set and compare it with ORCA.

Figure contract
---------------
Core conclusion: the ORCA-strongest pair should produce the least information
loss and best full-frontier coverage when grouped, following Russell & Allman
section 4.1.
Archetype: asymmetric quantitative grid.
Backend: Python/matplotlib only.
Outputs: editable SVG plus PDF, TIFF, and PNG; source-data CSV files.
Reviewer risks: local rather than global NLP solutions, continuous-slice
approximation of the paper's discrete metric, and triangulation across holes.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_pareto_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE = os.environ.get("CCUS_PARETO_PROFILE", "baseline")
OUTPUT_PREFIX = "" if PROFILE == "baseline" else f"{PROFILE}_"

OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
GROUPS = {
    "TAC + total emissions": {
        "indices": (0, 1),
        "retained": 2,
        "method": "group_TAC_TotEmiss",
        "orca_pair": "TAC__TotEmiss",
        "color": "#8E9AAF",
    },
    "TAC + ISI": {
        "indices": (0, 2),
        "retained": 1,
        "method": "group_TAC_ISI",
        "orca_pair": "TAC__ISI",
        "color": "#E6A15C",
    },
    "Total emissions + ISI": {
        "indices": (1, 2),
        "retained": 0,
        "method": "group_TotEmiss_ISI",
        "orca_pair": "TotEmiss__ISI",
        "color": "#4C78A8",
    },
}


def result_path(filename: str) -> Path:
    return RESULT_DIR / f"{OUTPUT_PREFIX}{filename}"


def nondominated_mask(values: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    """Return points not dominated under minimization of every column."""
    keep = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        dominates = np.all(values <= point + tolerance, axis=1) & np.any(
            values < point - tolerance, axis=1
        )
        if np.any(dominates):
            keep[i] = False
    return keep


def scale_values(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (values - lower) / (upper - lower)


def prepare_frontier(full: pd.DataFrame, reduced: pd.DataFrame):
    full_copy = full.copy()
    full_copy["source"] = "full_epsilon_grid"
    reduced_copy = reduced.copy()
    reduced_copy["source"] = "reduced_group_grid"
    candidates = pd.concat([full_copy, reduced_copy], ignore_index=True, sort=False)
    values = candidates[OBJECTIVES].to_numpy(dtype=float)
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    scaled = scale_values(values, lower, upper)

    rounded = pd.DataFrame(np.round(scaled, 8)).drop_duplicates().index.to_numpy()
    candidates = candidates.iloc[rounded].reset_index(drop=True)
    scaled = scaled[rounded]
    keep = nondominated_mask(scaled)
    frontier = candidates.loc[keep].copy().reset_index(drop=True)
    scaled_frontier = scaled[keep]
    for index, name in enumerate(OBJECTIVES):
        frontier[f"{name}_normalized"] = scaled_frontier[:, index]
    frontier.insert(0, "frontier_id", [f"PF{i + 1:04d}" for i in range(len(frontier))])
    return frontier, lower, upper


def equal_width_information_loss(
    scaled_frontier: np.ndarray,
    grouped: tuple[int, int],
    retained: int,
    bins: int,
):
    retained_values = scaled_frontier[:, retained]
    bin_ids = np.minimum((retained_values * bins).astype(int), bins - 1)
    losses = []
    for bin_id in range(bins):
        local = scaled_frontier[bin_ids == bin_id]
        if len(local) >= 2:
            losses.append(np.ptp(local[:, grouped[0]]) + np.ptp(local[:, grouped[1]]))
    return float(np.mean(losses)), len(losses)


def nearest_slice_information_loss(
    scaled_frontier: np.ndarray,
    grouped: tuple[int, int],
    retained: int,
    neighbors: int,
    centers: int = 21,
):
    losses = []
    for center in np.linspace(0.0, 1.0, centers):
        local_ids = np.argsort(np.abs(scaled_frontier[:, retained] - center))[
            : min(neighbors, len(scaled_frontier))
        ]
        local = scaled_frontier[local_ids]
        losses.append(np.ptp(local[:, grouped[0]]) + np.ptp(local[:, grouped[1]]))
    return float(np.mean(losses)), len(losses)


def reduced_front_diagnostics(
    reduced: pd.DataFrame,
    method: str,
    lower: np.ndarray,
    upper: np.ndarray,
    scaled_frontier: np.ndarray,
):
    subset = reduced.loc[reduced["method"] == method].copy()
    scaled = scale_values(subset[OBJECTIVES].to_numpy(dtype=float), lower, upper)
    dominated = ~nondominated_mask(np.vstack([scaled_frontier, scaled]))[-len(scaled) :]
    retained = scaled[~dominated]
    if len(retained) == 0:
        raise RuntimeError(f"No nondominated reduced-front points remain for {method}")
    distances = cdist(scaled_frontier, retained).min(axis=1)
    return {
        "points_total": len(scaled),
        "points_nondominated": len(retained),
        "dominated_fraction": float(dominated.mean()),
        "igd_mean": float(distances.mean()),
        "igd_p95": float(np.quantile(distances, 0.95)),
        "igd_max": float(distances.max()),
    }, subset.loc[~dominated].copy()


def grid_convergence(full: pd.DataFrame, lower: np.ndarray, upper: np.ndarray):
    tac_grid = full.loc[full["method"] == "full_TAC"].copy()
    fine_size = int(max(tac_grid["index_a"].max(), tac_grid["index_b"].max()))
    fine_scaled = scale_values(tac_grid[OBJECTIVES].to_numpy(dtype=float), lower, upper)
    fine = fine_scaled[nondominated_mask(fine_scaled)]
    coarse_rows = tac_grid.loc[
        (tac_grid["index_a"].astype(int) % 2 == 1)
        & (tac_grid["index_b"].astype(int) % 2 == 1)
    ]
    coarse_scaled = scale_values(coarse_rows[OBJECTIVES].to_numpy(dtype=float), lower, upper)
    coarse = coarse_scaled[nondominated_mask(coarse_scaled)]
    distances = cdist(fine, coarse).min(axis=1)
    coarse_size = len(range(1, fine_size + 1, 2))
    return pd.DataFrame(
        [
            {
                "fine_grid": f"{fine_size}x{fine_size}",
                "coarse_embedded_grid": f"{coarse_size}x{coarse_size}",
                "fine_nondominated_points": len(fine),
                "coarse_nondominated_points": len(coarse),
                "mean_normalized_distance_fine_to_coarse": distances.mean(),
                "p95_normalized_distance_fine_to_coarse": np.quantile(distances, 0.95),
                "max_normalized_distance_fine_to_coarse": distances.max(),
            }
        ]
    )


def candidate_source_sensitivity(full: pd.DataFrame, reduced: pd.DataFrame):
    pools = {
        "full_TAC_only": full.loc[full["method"] == "full_TAC"].copy(),
        "all_full_grids": full.copy(),
        "all_full_plus_reduced": pd.concat(
            [full, reduced], ignore_index=True, sort=False
        ),
    }
    rows = []
    for source, candidates in pools.items():
        values = candidates[OBJECTIVES].to_numpy(dtype=float)
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        scaled = scale_values(values, lower, upper)
        unique_ids = (
            pd.DataFrame(np.round(scaled, 8)).drop_duplicates().index.to_numpy()
        )
        scaled = scaled[unique_ids]
        scaled = scaled[nondominated_mask(scaled)]
        for group_name, spec in GROUPS.items():
            bin_losses = [
                equal_width_information_loss(
                    scaled, spec["indices"], spec["retained"], bins
                )[0]
                for bins in (8, 10, 12, 15, 18, 20, 25, 30)
            ]
            neighbor_losses = [
                nearest_slice_information_loss(
                    scaled, spec["indices"], spec["retained"], neighbors
                )[0]
                for neighbors in (8, 12, 16, 20, 24, 30)
            ]
            rows.append(
                {
                    "candidate_source": source,
                    "grouping": group_name,
                    "frontier_points": len(scaled),
                    "mean_information_loss_bins": np.mean(bin_losses),
                    "mean_information_loss_neighbors": np.mean(neighbor_losses),
                }
            )
    result = pd.DataFrame(rows)
    result["bins_rank"] = (
        result.groupby("candidate_source")["mean_information_loss_bins"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    result["neighbors_rank"] = (
        result.groupby("candidate_source")["mean_information_loss_neighbors"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    return result


def save_figure(fig: plt.Figure, stem: str):
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            FIGURE_DIR / f"{OUTPUT_PREFIX}{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def plot_pareto_figure(
    frontier: pd.DataFrame,
    reduced_retained: dict[str, pd.DataFrame],
    lower: np.ndarray,
    upper: np.ndarray,
):
    fig = plt.figure(figsize=(7.2, 6.5))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.35, 1.0), hspace=0.42, wspace=0.48)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    scaled = scale_values(frontier[OBJECTIVES].to_numpy(dtype=float), lower, upper)
    x = scaled[:, 1]
    y = scaled[:, 2]
    z = scaled[:, 0]

    triangulation = mtri.Triangulation(x, y)
    triangles = triangulation.triangles
    triangle_points = np.stack([x[triangles], y[triangles]], axis=-1)
    edge_lengths = np.linalg.norm(
        triangle_points - np.roll(triangle_points, shift=1, axis=1), axis=2
    )
    triangulation.set_mask(edge_lengths.max(axis=1) > 0.18)
    ax3d.plot_trisurf(
        triangulation,
        z,
        color="#9BC8FA",
        alpha=0.22,
        linewidth=0.15,
        edgecolor="#6C8EAD",
        antialiased=True,
    )
    ax3d.scatter(x, y, z, c=z, cmap="cividis_r", s=8, alpha=0.82, depthshade=False)
    ax3d.set_xlabel("Normalized total emissions", labelpad=7)
    ax3d.set_ylabel("Normalized ISI", labelpad=7)
    ax3d.set_zlabel("")
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.view_init(elev=25, azim=-56)
    ax3d.xaxis.pane.set_alpha(0.03)
    ax3d.yaxis.pane.set_alpha(0.03)
    ax3d.zaxis.pane.set_alpha(0.03)
    ax3d.set_box_aspect((1.0, 1.0, 0.82))
    ax3d.text2D(0.015, 0.57, "Normalized TAC", transform=ax3d.transAxes,
                rotation=90, ha="center", va="center", fontsize=7)
    ax3d.text2D(-0.08, 1.02, "a", transform=ax3d.transAxes, fontweight="bold", fontsize=10)
    profile_label = PROFILE.replace("_", " ")
    ax3d.set_title(
        f"{profile_label}: deterministic three-objective Pareto frontier",
        fontsize=8.5,
        pad=10,
    )

    projection_axes = []
    for panel_index, (group_name, spec) in enumerate(GROUPS.items()):
        ax = fig.add_subplot(grid[panel_index, 1])
        projection_axes.append(ax)
        ax.scatter(
            scaled[:, 1],
            scaled[:, 2],
            c=scaled[:, 0],
            cmap="Greys_r",
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.65,
            linewidths=0,
            rasterized=False,
        )
        retained = reduced_retained[group_name]
        retained_scaled = scale_values(
            retained[OBJECTIVES].to_numpy(dtype=float), lower, upper
        )
        ax.scatter(
            retained_scaled[:, 1],
            retained_scaled[:, 2],
            s=14,
            facecolor="#B64342",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.95,
            zorder=4,
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Normalized ISI")
        if panel_index == 2:
            ax.set_xlabel("Normalized total emissions")
        else:
            ax.set_xticklabels([])
        ax.set_title(f"Grouped: {group_name}", fontsize=7.5, loc="left")
        ax.text(-0.17, 1.04, chr(ord("b") + panel_index), transform=ax.transAxes,
                fontweight="bold", fontsize=10)
        style_axes(ax)

    fig.text(0.98, 0.018, "Red: nondominated points retained after objective grouping",
             ha="right", fontsize=6.3, color="#6A3333")
    save_figure(fig, "ccus_pareto_frontier_and_groupings")


def plot_validation_figure(
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"wspace": 0.42})
    ordered = validation.sort_values("orca_strength_01", ascending=False)
    labels = ordered["grouping"].str.replace("Total emissions", "Emissions").tolist()
    short_labels = [label.replace(" + ", "\n+") for label in labels]
    colors = [GROUPS[name]["color"] for name in ordered["grouping"]]

    axes[0].bar(range(3), ordered["orca_strength_01"], color=colors, width=0.68,
                edgecolor="black", linewidth=0.45)
    axes[0].set_xticks(range(3), short_labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("ORCA correlation strength")
    axes[0].set_title("Structural prediction", fontsize=8)
    for index, value in enumerate(ordered["orca_strength_01"]):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=6.2)

    binned = sensitivity.loc[sensitivity["metric"] == "equal_width_bins"]
    for name, group in binned.groupby("grouping", sort=False):
        axes[1].plot(group["resolution"], group["information_loss"], marker="o",
                     ms=3.2, lw=1.25, color=GROUPS[name]["color"], label=name)
    axes[1].set_xlabel("Number of retained-axis slices")
    axes[1].set_ylabel("Average information loss")
    axes[1].set_title("Section 4.1 metric", fontsize=8)

    coverage = validation.sort_values("igd_mean")
    coverage_colors = [GROUPS[name]["color"] for name in coverage["grouping"]]
    axes[2].bar(range(3), coverage["igd_mean"], color=coverage_colors, width=0.68,
                edgecolor="black", linewidth=0.45)
    axes[2].set_xticks(
        range(3),
        [name.replace("Total emissions", "Emissions").replace(" + ", "\n+")
         for name in coverage["grouping"]],
    )
    axes[2].set_ylabel("Mean distance to reduced frontier")
    axes[2].set_title("Trade-off coverage (lower is better)", fontsize=8)
    for index, value in enumerate(coverage["igd_mean"]):
        axes[2].text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=6.2)

    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        style_axes(ax)
        ax.tick_params(labelsize=6.4)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
               ncol=3, frameon=False, fontsize=6.2)
    save_figure(fig, "ccus_orca_pareto_validation")


def main():
    full = pd.read_csv(result_path("pareto_full_raw.csv"))
    reduced = pd.read_csv(result_path("pareto_reduced_raw.csv"))
    if PROFILE == "baseline":
        orca = pd.read_csv(RESULT_DIR / "orca_pairwise.csv")
        scenario_orca = orca.loc[orca["scenario"] == "baseline"].set_index("pair")
    elif PROFILE == "direct_use_expansion":
        comparison = pd.read_csv(
            RESULT_DIR / "direct_use_expansion_paper_aligned_orca_comparison.csv"
        )
        scenario_orca = comparison.rename(
            columns={
                "paper_adjacency_mean": "orca_adjacency_01",
                "paper_signed_mean": "orca_signed_correlation",
            }
        ).set_index("pair")
    elif PROFILE == "supply_chain_condition2":
        comparison = pd.read_csv(
            RESULT_DIR
            / "supply_chain_condition2_paper_orca_condition2_main_summary.csv"
        )
        comparison = comparison.loc[
            comparison["constraint_handling"].eq("equality_tangent")
        ].copy()
        scenario_orca = comparison.rename(
            columns={
                "adjacency_mean": "orca_adjacency_01",
                "signed_mean": "orca_signed_correlation",
            }
        ).set_index("pair")
    else:
        raise ValueError(f"Unknown CCUS_PARETO_PROFILE: {PROFILE}")

    frontier, lower, upper = prepare_frontier(full, reduced)
    scaled_frontier = frontier[[f"{name}_normalized" for name in OBJECTIVES]].to_numpy()
    frontier.to_csv(result_path("pareto_frontier.csv"), index=False)

    sensitivity_rows = []
    validation_rows = []
    retained_sets = {}
    for group_name, spec in GROUPS.items():
        for bins in (8, 10, 12, 15, 18, 20, 25, 30):
            loss, valid = equal_width_information_loss(
                scaled_frontier, spec["indices"], spec["retained"], bins
            )
            sensitivity_rows.append(
                {
                    "grouping": group_name,
                    "metric": "equal_width_bins",
                    "resolution": bins,
                    "information_loss": loss,
                    "valid_slices": valid,
                }
            )
        for neighbors in (8, 12, 16, 20, 24, 30):
            loss, valid = nearest_slice_information_loss(
                scaled_frontier, spec["indices"], spec["retained"], neighbors
            )
            sensitivity_rows.append(
                {
                    "grouping": group_name,
                    "metric": "nearest_neighbor_slices",
                    "resolution": neighbors,
                    "information_loss": loss,
                    "valid_slices": valid,
                }
            )

        diagnostics, retained = reduced_front_diagnostics(
            reduced, spec["method"], lower, upper, scaled_frontier
        )
        retained_sets[group_name] = retained
        pair_sensitivity = [
            row for row in sensitivity_rows if row["grouping"] == group_name
        ]
        bin_losses = [
            row["information_loss"] for row in pair_sensitivity
            if row["metric"] == "equal_width_bins"
        ]
        neighbor_losses = [
            row["information_loss"] for row in pair_sensitivity
            if row["metric"] == "nearest_neighbor_slices"
        ]
        validation_rows.append(
            {
                "grouping": group_name,
                "orca_strength_01": scenario_orca.loc[spec["orca_pair"], "orca_adjacency_01"],
                "orca_signed_correlation": scenario_orca.loc[
                    spec["orca_pair"], "orca_signed_correlation"
                ],
                "mean_information_loss_bins": np.mean(bin_losses),
                "sd_information_loss_bins": np.std(bin_losses, ddof=1),
                "mean_information_loss_neighbors": np.mean(neighbor_losses),
                "sd_information_loss_neighbors": np.std(neighbor_losses, ddof=1),
                **diagnostics,
            }
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    validation = pd.DataFrame(validation_rows)
    validation["orca_rank_descending"] = validation["orca_strength_01"].rank(
        ascending=False, method="min"
    ).astype(int)
    validation["information_loss_rank_ascending"] = validation[
        "mean_information_loss_bins"
    ].rank(ascending=True, method="min").astype(int)
    validation["coverage_rank_ascending"] = validation["igd_mean"].rank(
        ascending=True, method="min"
    ).astype(int)

    rank_orca = validation["orca_strength_01"].to_numpy()
    rank_loss = -validation["mean_information_loss_bins"].to_numpy()
    spearman = spearmanr(rank_orca, rank_loss)
    kendall = kendalltau(rank_orca, rank_loss)
    top_orca = validation.loc[validation["orca_strength_01"].idxmax(), "grouping"]
    top_loss = validation.loc[
        validation["mean_information_loss_bins"].idxmin(), "grouping"
    ]
    top_coverage = validation.loc[validation["igd_mean"].idxmin(), "grouping"]

    convergence = grid_convergence(full, lower, upper)
    source_sensitivity = candidate_source_sensitivity(full, reduced)
    sensitivity.to_csv(result_path("pareto_information_loss_sensitivity.csv"), index=False)
    validation.to_csv(result_path("orca_pareto_validation.csv"), index=False)
    convergence.to_csv(result_path("pareto_grid_convergence.csv"), index=False)
    source_sensitivity.to_csv(
        result_path("pareto_candidate_source_sensitivity.csv"), index=False
    )
    result_path("pareto_validation_summary.txt").write_text(
        "\n".join(
            [
                "method = Russell & Allman section 4.1 adapted to a continuous frontier",
                f"profile = {PROFILE}",
                f"candidate_points = {len(full) + len(reduced)}",
                f"nondominated_unique_points = {len(frontier)}",
                f"top_ORCA_group = {top_orca}",
                f"lowest_information_loss_group = {top_loss}",
                f"best_reduced_frontier_coverage_group = {top_coverage}",
                f"spearman_rank_correlation = {spearman.statistic}",
                f"spearman_pvalue_descriptive_only_n3 = {spearman.pvalue}",
                f"kendall_tau = {kendall.statistic}",
                f"kendall_pvalue_descriptive_only_n3 = {kendall.pvalue}",
                "global_optimality = not claimed; deterministic local NLP frontier approximation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_pareto_figure(frontier, retained_sets, lower, upper)
    plot_validation_figure(validation, sensitivity)
    print(validation.to_string(index=False))
    print(convergence.to_string(index=False))


if __name__ == "__main__":
    main()
