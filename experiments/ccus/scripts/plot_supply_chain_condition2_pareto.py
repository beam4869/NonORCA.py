"""Plot condition2 Pareto projections and exact information-loss diagnostics.

Figure contract
---------------
Core conclusion: condition2 makes TAC-total emissions the ORCA-selected group
while retaining multiple sink branches on the three-objective Pareto frontier.
Evidence chain: a style-matched 3D-frontier/ORCA-network composite and three
literal pairwise projections that distinguish the full 3D set from the points
that remain nondominated in each 2D objective plane.
Archetype: asymmetric mixed-modality figure plus quantitative grid.
Backend/export: Python/Matplotlib; PDF, SVG, TIFF, and PNG plus source CSV.
Reviewer risks: 3D occlusion, confusing a projection with a 2D nondominated
subset, and treating local Ipopt solutions as globally certified optima.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_condition2_pareto_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
PREFIX = "supply_chain_condition2_"
LABEL = os.environ.get("CCUS_EXACT_INFO_LABEL", "quantile21")
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
DISPLAY = {"TAC": "TAC", "TotEmiss": "Total emissions", "ISI": "ISI"}
NORMALIZED_DISPLAY = {
    "TAC": "Normalized TAC",
    "TotEmiss": "Normalized total emissions",
    "ISI": "Normalized ISI",
}
PAIR_SPECS = [
    ("TAC", "TotEmiss", "TAC-emissions"),
    ("TAC", "ISI", "TAC-ISI"),
    ("TotEmiss", "ISI", "Emissions-ISI"),
]
PROJECTION_SPECS = [
    (
        "TotEmiss", "TAC", "TAC-emissions", "TAC + total emissions",
        "TAC + total emissions",
    ),
    ("ISI", "TAC", "TAC-ISI", "TAC + ISI", "TAC + ISI"),
    (
        "TotEmiss", "ISI", "Emissions-ISI", "Total emissions + ISI",
        "Total emissions + ISI",
    ),
]
GROUP_SPECS = [
    ("TAC + total emissions", "group_TAC_TotEmiss", "TAC + total emissions"),
    ("TAC + ISI", "group_TAC_ISI", "TAC + ISI"),
    ("Total emissions + ISI", "group_TotEmiss_ISI", "Total emissions + ISI"),
]
SINK_ORDER = ["Urea", "Methanol", "Acetic Acid", "Greenhouse", "Algae"]
SINK_STYLES = {
    "Algae": ("#B8A9C9", "P"),
    "Greenhouse": ("#7FA58A", "^"),
    "Saline Storage": ("#6E8EB5", "s"),
    "Methanol": ("#D4A45F", "v"),
    "Urea": ("#C995A5", "o"),
    "Acetic Acid": ("#8FA6A0", "D"),
}
SIGNAL = "#B94D48"
NEUTRAL = "#D5D8DD"
TEXT = "#303640"

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


def nondominated_mask(values: np.ndarray, tolerance: float = 1.0e-8) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        other = np.delete(values, index, axis=0)
        keep[index] = not np.any(
            np.all(other <= point + tolerance, axis=1)
            & np.any(other < point - tolerance, axis=1)
        )
    return keep


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            FIGURE_DIR / f"{PREFIX}{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def load_data():
    frontier = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_frontier.csv")
    structures = pd.read_csv(
        RESULT_DIR / f"{PREFIX}pareto_system_structures.csv"
    )[["frontier_id", "main_sink"]]
    frontier = frontier.merge(structures, on="frontier_id", validate="one_to_one")
    reduced = pd.read_csv(RESULT_DIR / f"{PREFIX}pareto_reduced_raw.csv")
    bounds = pd.read_csv(
        RESULT_DIR
        / f"{PREFIX}exact_info_loss_{LABEL}_normalization_bounds.csv"
    ).set_index("objective").loc[OBJECTIVES]
    lower = bounds["lower"].to_numpy(float)
    ranges = (bounds["upper"] - bounds["lower"]).to_numpy(float)
    frontier[[f"{name}_plot" for name in OBJECTIVES]] = (
        frontier[OBJECTIVES].to_numpy(float) - lower
    ) / ranges
    summary = pd.read_csv(
        RESULT_DIR / f"{PREFIX}exact_info_loss_{LABEL}_summary.csv"
    )
    summary = summary.loc[
        summary["retained_resolution"].eq(summary["retained_resolution"].max())
        & summary["epsilon_resolution"].eq(summary["epsilon_resolution"].max())
    ].set_index("grouping")
    orca = pd.read_csv(
        RESULT_DIR
        / "supply_chain_condition2_paper_orca_condition2_main_summary.csv"
    )
    orca = orca.loc[
        orca["constraint_handling"].eq("equality_tangent")
    ].set_index("pair")
    return frontier, reduced, lower, ranges, summary, orca


def draw_sink_points(ax, data: pd.DataFrame, x: str, y: str, *, size: float) -> None:
    for sink in sorted(data["main_sink"].unique()):
        subset = data.loc[data["main_sink"].eq(sink)]
        color, marker = SINK_STYLES.get(sink, ("#969696", "o"))
        ax.scatter(
            subset[x],
            subset[y],
            s=size,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.28,
            alpha=0.86,
            label=sink,
            zorder=2,
        )


def plot_frontier_and_pairwise(frontier: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 5.5))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.38, 1.0, 1.0), hspace=0.38, wspace=0.38
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    for sink in sorted(frontier["main_sink"].unique()):
        subset = frontier.loc[frontier["main_sink"].eq(sink)]
        color, marker = SINK_STYLES.get(sink, ("#969696", "o"))
        ax3d.scatter(
            subset["TotEmiss_plot"],
            subset["ISI_plot"],
            subset["TAC_plot"],
            s=12,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.88,
            depthshade=False,
            label=sink,
        )
    ax3d.set_xlabel("Total emissions\n(normalized)", labelpad=7)
    ax3d.set_ylabel("ISI\n(normalized)", labelpad=7)
    ax3d.set_zlabel("")
    ax3d.text2D(
        0.91,
        0.54,
        "TAC\n(normalized)",
        transform=ax3d.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
    )
    ax3d.view_init(elev=18, azim=45)
    ax3d.set_box_aspect((1.08, 1.0, 0.82))
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_alpha(0.02)
        axis._axinfo["grid"]["color"] = (0.84, 0.85, 0.87, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax3d.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax3d.set_xticks([0, 0.5, 1])
    ax3d.set_yticks([0, 0.5, 1])
    ax3d.set_zticks([0, 0.5, 1])
    ax3d.legend(loc="upper left", bbox_to_anchor=(0, 0.98), fontsize=5.6)
    ax3d.text2D(-0.08, 1.02, "a", transform=ax3d.transAxes,
                fontsize=9, fontweight="bold")

    axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 1:]),
    ]
    projection_rows = []
    for panel, ax, (x_name, y_name, pair_name) in zip(
        "bcd", axes, PAIR_SPECS
    ):
        x_col, y_col = f"{x_name}_plot", f"{y_name}_plot"
        values = frontier[[x_col, y_col]].to_numpy(float)
        projected = nondominated_mask(values)
        ax.scatter(
            frontier[x_col], frontier[y_col], s=9, color=NEUTRAL,
            edgecolor="none", alpha=0.62, zorder=1,
        )
        draw_sink_points(ax, frontier.loc[projected], x_col, y_col, size=18)
        front = frontier.loc[projected].sort_values(x_col)
        ax.plot(front[x_col], front[y_col], color="#6F7782", lw=0.65, alpha=0.75)
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(NORMALIZED_DISPLAY[x_name])
        ax.set_ylabel(NORMALIZED_DISPLAY[y_name])
        ax.text(-0.15, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold")
        ax.text(
            0.97, 0.96, f"2D nondominated: {projected.sum()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
            color=TEXT,
        )
        for index, row in frontier.iterrows():
            projection_rows.append(
                {
                    "pair": pair_name,
                    "frontier_id": row["frontier_id"],
                    "x_objective": x_name,
                    "y_objective": y_name,
                    "x_normalized": row[x_col],
                    "y_normalized": row[y_col],
                    "main_sink": row["main_sink"],
                    "is_2d_nondominated": bool(projected[index]),
                }
            )
    pd.DataFrame(projection_rows).to_csv(
        RESULT_DIR / f"{PREFIX}pareto_pairwise_projection_source.csv", index=False
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.09, top=0.98)
    save_figure(fig, "pareto_3d_and_pairwise_2d")


def draw_style_matched_frontier(ax, frontier: pd.DataFrame) -> None:
    for sink in SINK_ORDER:
        subset = frontier.loc[frontier["main_sink"].eq(sink)]
        if subset.empty:
            continue
        color, marker = SINK_STYLES[sink]
        ax.scatter(
            subset["TotEmiss_plot"],
            subset["ISI_plot"],
            subset["TAC_plot"],
            s=13,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.30,
            alpha=0.82,
            depthshade=False,
            label=sink,
        )
    ax.set_xlabel("Emissions", labelpad=7)
    ax.set_ylabel("ISI", labelpad=7)
    ax.set_zlabel("")
    ax.text2D(
        0.94,
        0.53,
        "TAC",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.0,
    )
    ax.view_init(elev=18, azim=45)
    ax.set_box_aspect((1.12, 1.0, 0.82))
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.018)
        axis._axinfo["grid"]["color"] = (0.79, 0.80, 0.82, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.58
    ax.tick_params(labelsize=6.2, pad=1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.03, 0.97),
        borderaxespad=0,
        fontsize=5.8,
        handletextpad=0.45,
        labelspacing=0.35,
        markerscale=1.05,
    )


def draw_style_matched_orca_network(ax, orca: pd.DataFrame) -> None:
    positions = {
        "ISI": np.array([0.50, 0.76]),
        "TAC": np.array([0.23, 0.27]),
        "TotEmiss": np.array([0.77, 0.27]),
    }
    labels = {"TAC": "TAC", "TotEmiss": "Total\nemissions", "ISI": "ISI"}
    node_colors = {
        "TAC": "#C1D2E3",
        "TotEmiss": "#C1D2E3",
        "ISI": "#E6C0C8",
    }
    text_colors = {
        "TAC": "#294968",
        "TotEmiss": "#294968",
        "ISI": "#733947",
    }
    selected = Ellipse(
        (0.50, 0.27),
        width=0.88,
        height=0.30,
        facecolor="#EFF4FA",
        edgecolor="#6E8FBA",
        linewidth=0.85,
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(selected)
    ax.text(
        0.50,
        0.075,
        "Selected group",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#456D9C",
        zorder=5,
    )

    edge_specs = [
        ("TAC", "TotEmiss", "TAC__TotEmiss", (0.50, 0.215)),
        ("TAC", "ISI", "TAC__ISI", (0.30, 0.52)),
        ("TotEmiss", "ISI", "TotEmiss__ISI", (0.70, 0.52)),
    ]
    source_rows = []
    for node_a, node_b, pair, label_position in edge_specs:
        adjacency = float(orca.loc[pair, "adjacency_mean"])
        signed = float(orca.loc[pair, "signed_mean"])
        start, end = positions[node_a], positions[node_b]
        selected_edge = pair == "TAC__TotEmiss"
        edge_color = "#4B82B5" if selected_edge else "#AEB7C4"
        width = 1.0 + 3.0 * (adjacency - 0.58) / (0.76 - 0.58)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=edge_color,
            linewidth=width,
            solid_capstyle="round",
            zorder=1,
        )
        ax.text(
            label_position[0],
            label_position[1],
            f"{adjacency:.3f}",
            ha="center",
            va="center",
            fontsize=5.8,
            bbox={
                "boxstyle": "round,pad=0.13",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=3,
        )
        source_rows.append(
            {
                "pair": pair,
                "orca_adjacency": adjacency,
                "orca_signed_strength": signed,
                "selected_group_edge": selected_edge,
            }
        )

    for node, position in positions.items():
        ax.add_patch(
            Circle(
                position,
                radius=0.105,
                facecolor=node_colors[node],
                edgecolor="#5B687D",
                linewidth=0.75,
                zorder=4,
            )
        )
        ax.text(
            position[0],
            position[1],
            labels[node],
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color=text_colors[node],
            zorder=5,
        )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.01, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    pd.DataFrame(source_rows).to_csv(
        RESULT_DIR / f"{PREFIX}orca_network_source.csv", index=False
    )


def plot_style_matched_frontier_orca(
    frontier: pd.DataFrame, orca: pd.DataFrame
) -> None:
    fig = plt.figure(figsize=(7.2, 3.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.08)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    draw_style_matched_frontier(ax3d, frontier)
    ax3d.text2D(
        -0.05,
        1.02,
        "a",
        transform=ax3d.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    ax_network = fig.add_subplot(grid[0, 1])
    draw_style_matched_orca_network(ax_network, orca)
    ax_network.text(
        -0.05,
        1.02,
        "b",
        transform=ax_network.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.98)
    save_figure(fig, "pareto_orca_network")


def visible_greys() -> mcolors.LinearSegmentedColormap:
    base = plt.colormaps["Greys_r"]
    return mcolors.LinearSegmentedColormap.from_list(
        "condition2_visible_greys", base(np.linspace(0.04, 0.76, 256))
    )


def grouped_retained(
    frontier: pd.DataFrame,
    reduced: pd.DataFrame,
    method: str,
    lower: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    full = (frontier[OBJECTIVES].to_numpy(float) - lower) / ranges
    subset = reduced.loc[reduced["method"].eq(method), OBJECTIVES]
    candidate = (subset.to_numpy(float) - lower) / ranges
    keep = nondominated_mask(np.vstack([full, candidate]))
    return candidate[keep[len(full):]]


def plot_grouped_projections(
    frontier: pd.DataFrame,
    reduced: pd.DataFrame,
    lower: np.ndarray,
    ranges: np.ndarray,
    summary: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(7.2, 2.45))
    grid = fig.add_gridspec(
        1, 4, width_ratios=(1, 1, 1, 0.045),
        left=0.075, right=0.94, bottom=0.20, top=0.90, wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    color_axis = fig.add_subplot(grid[0, 3])
    cmap = visible_greys()
    norm = mcolors.Normalize(0, 1)
    source_rows = []
    for panel, ax, (group, method, title) in zip("abc", axes, GROUP_SPECS):
        retained = grouped_retained(frontier, reduced, method, lower, ranges)
        ax.scatter(
            frontier["TotEmiss_plot"], frontier["ISI_plot"],
            c=frontier["TAC_plot"], cmap=cmap, norm=norm,
            s=9, alpha=0.70, linewidth=0, zorder=1,
        )
        ax.scatter(
            retained[:, 1], retained[:, 2], s=18, color=SIGNAL,
            edgecolor="white", linewidth=0.3, alpha=0.94, zorder=3,
        )
        loss = float(summary.loc[group, "mean_information_loss"])
        ax.set_title(f"Grouped: {title}", fontsize=7.2, loc="left", pad=5)
        ax.text(-0.17, 1.05, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold")
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        source_rows.append(
            pd.DataFrame(
                {
                    "grouping": group,
                    "TotEmiss_normalized": retained[:, 1],
                    "ISI_normalized": retained[:, 2],
                    "TAC_normalized": retained[:, 0],
                    "mean_information_loss": loss,
                }
            )
        )
    axes[0].set_ylabel("Normalized ISI")
    fig.supxlabel("Normalized total emissions", fontsize=7, y=0.055)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis
    )
    colorbar.set_label("Normalized TAC", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8, width=0.6, length=2.5)
    pd.concat(source_rows, ignore_index=True).to_csv(
        RESULT_DIR / f"{PREFIX}grouping_projection_source.csv", index=False
    )
    save_figure(fig, "grouping_projections_information_loss")


def plot_pairwise_pareto_projections(
    frontier: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Plot the full 3D Pareto set in each objective plane.

    Grey points are all 3D-nondominated solutions after projection. Red open
    circles mark only the points that remain nondominated in that 2D plane.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.20, top=0.78, wspace=0.34
    )
    source_rows = []
    for panel, ax, spec in zip("abc", axes, PROJECTION_SPECS):
        x_name, y_name, pair_name, title, summary_key = spec
        x_col, y_col = f"{x_name}_plot", f"{y_name}_plot"
        values = frontier[[x_col, y_col]].to_numpy(float)
        projected = nondominated_mask(values)

        ax.scatter(
            frontier[x_col],
            frontier[y_col],
            s=10,
            color="#C8CDD3",
            edgecolor="none",
            alpha=0.62,
            zorder=1,
            label="3D Pareto set",
        )
        ax.scatter(
            frontier.loc[projected, x_col],
            frontier.loc[projected, y_col],
            s=21,
            facecolor="white",
            edgecolor=SIGNAL,
            linewidth=0.85,
            alpha=0.98,
            zorder=3,
            label="2D nondominated subset",
        )

        loss = float(summary.loc[summary_key, "mean_information_loss"])
        ax.set_title(f"Projection: {title}", fontsize=7.2, loc="left", pad=5)
        ax.text(
            -0.17, 1.06, panel, transform=ax.transAxes,
            fontsize=9, fontweight="bold",
        )
        ax.text(
            0.97,
            0.95,
            f"2D nondominated: {projected.sum()}\nInformation loss: {loss:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.6,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
            },
            zorder=4,
        )
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(NORMALIZED_DISPLAY[x_name])
        ax.set_ylabel(NORMALIZED_DISPLAY[y_name])
        ax.tick_params(labelsize=6.2, width=0.7, length=3)

        for row_position, (_, row) in enumerate(frontier.iterrows()):
            source_rows.append(
                {
                    "pair": pair_name,
                    "frontier_id": row["frontier_id"],
                    "x_objective": x_name,
                    "y_objective": y_name,
                    "x_normalized": row[x_col],
                    "y_normalized": row[y_col],
                    "main_sink": row["main_sink"],
                    "is_2d_nondominated": bool(projected[row_position]),
                    "mean_information_loss": loss,
                }
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=2,
        fontsize=5.8,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    pd.DataFrame(source_rows).to_csv(
        RESULT_DIR / f"{PREFIX}pareto_pairwise_projection_source.csv",
        index=False,
    )
    save_figure(fig, "pairwise_pareto_projections")


def main() -> None:
    frontier, reduced, lower, ranges, summary, orca = load_data()
    plot_style_matched_frontier_orca(frontier, orca)
    plot_pairwise_pareto_projections(frontier, summary)
    print("Saved condition2 Pareto/ORCA and literal pairwise projections")


if __name__ == "__main__":
    main()
