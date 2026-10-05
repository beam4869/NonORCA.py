"""Create revised direct-use Pareto and ORCA demonstration figures.

Figure contract
---------------
Core conclusion: the Pareto set contains distinct sink branches, while ORCA
selects total emissions + ISI as the strongest objective grouping.
Archetype: quantitative grid plus a compact schematic-led composite.
Backend: Python/matplotlib only.
Outputs: editable SVG/PDF plus 600-dpi TIFF and PNG previews.
Reviewer risks: 3D occlusion, ambiguous color meaning, and confusion between
ORCA adjacency and signed correlation.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_revised_pareto_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd


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

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
NORMALIZED = [f"{name}_plot" for name in OBJECTIVES]

SINK_STYLES = {
    "Urea": {"color": "#D6A8B8", "marker": "o"},
    "Saline Storage": {"color": "#7884B4", "marker": "D"},
    "Greenhouse": {"color": "#93AD9A", "marker": "^"},
}

GROUPS = [
    {
        "name": "TAC + total emissions",
        "title": "TAC + emissions",
        "method": "group_TAC_TotEmiss",
        "panel": "a",
    },
    {
        "name": "TAC + ISI",
        "title": "TAC + ISI",
        "method": "group_TAC_ISI",
        "panel": "b",
    },
    {
        "name": "Total emissions + ISI",
        "title": "Emissions + ISI",
        "method": "group_TotEmiss_ISI",
        "panel": "c",
    },
]


def nondominated_mask(values: np.ndarray, tolerance: float = 1.0e-7) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        keep[index] = not np.any(
            np.all(values <= point + tolerance, axis=1)
            & np.any(values < point - tolerance, axis=1)
        )
    return keep


def save_figure(fig: plt.Figure, stem: str) -> None:
    target = FIGURE_DIR / stem
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            target.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def load_data():
    frontier = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_frontier.csv"
    )
    structures = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_system_structures.csv"
    )[["frontier_id", "main_sink"]]
    frontier = frontier.merge(structures, on="frontier_id", validate="one_to_one")
    reduced = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_reduced_raw.csv"
    )
    bounds = (
        pd.read_csv(
            RESULT_DIR
            / "direct_use_expansion_exact_info_loss_quantile21_normalization_bounds.csv"
        )
        .set_index("objective")
        .loc[OBJECTIVES]
    )
    lower = bounds["lower"].to_numpy(dtype=float)
    ranges = (bounds["upper"] - bounds["lower"]).to_numpy(dtype=float)
    frontier = frontier.copy()
    frontier[NORMALIZED] = (
        frontier[OBJECTIVES].to_numpy(dtype=float) - lower
    ) / ranges

    orca = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_paper_aligned_orca_comparison.csv"
    ).set_index("pair")
    return frontier, reduced, lower, ranges, orca


def retained_points(
    frontier: pd.DataFrame,
    reduced: pd.DataFrame,
    method: str,
    lower: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    full_scaled = frontier[NORMALIZED].to_numpy(dtype=float)
    subset = reduced.loc[reduced["method"] == method, OBJECTIVES]
    reduced_scaled = (subset.to_numpy(dtype=float) - lower) / ranges
    combined = np.vstack([full_scaled, reduced_scaled])
    keep = nondominated_mask(combined)
    return reduced_scaled[keep[len(full_scaled) :]]


def style_3d_axis(ax, compact: bool = False) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ticks = [0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    if compact:
        ax.set_xlabel("Emissions", labelpad=3)
        ax.set_ylabel("ISI", labelpad=3)
        ax.set_zlabel("")
        ax.text2D(
            0.94,
            0.54,
            "TAC",
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.4,
        )
    else:
        ax.set_xlabel("Total emissions\n(normalized)", labelpad=9)
        ax.set_ylabel("ISI\n(normalized)", labelpad=9)
        ax.set_zlabel("TAC\n(normalized)", labelpad=8)
    ax.view_init(elev=20, azim=132)
    ax.set_box_aspect((1.12, 1.0, 0.82))
    ax.xaxis.pane.set_alpha(0.015)
    ax.yaxis.pane.set_alpha(0.015)
    ax.zaxis.pane.set_alpha(0.015)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.84, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax.grid(True)
    ax.tick_params(labelsize=5.8 if compact else 6.4, pad=1)


def draw_frontier_3d(ax, frontier: pd.DataFrame, compact: bool = False) -> None:
    for sink, style in SINK_STYLES.items():
        subset = frontier.loc[frontier["main_sink"] == sink]
        x = subset["TotEmiss_plot"].to_numpy(dtype=float).copy()
        y = subset["ISI_plot"].to_numpy(dtype=float).copy()
        z = subset["TAC_plot"].to_numpy(dtype=float).copy()
        ax.scatter(
            x,
            y,
            z,
            s=9 if compact else 12,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.30,
            alpha=0.88,
            depthshade=False,
            label=sink,
        )
    style_3d_axis(ax, compact)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        fontsize=5.4 if compact else 6.2,
        handletextpad=0.35,
        borderaxespad=0,
        markerscale=1.15,
    )


def plot_frontier_standalone(frontier: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(5.2, 4.25))
    ax = fig.add_subplot(111, projection="3d")
    draw_frontier_3d(ax, frontier)
    ax.set_title(
        "Three-objective Pareto frontier by selected CO$_2$ sink",
        fontsize=9,
        pad=8,
    )
    fig.text(
        0.5,
        0.015,
        "Color and marker encode the dominant sink; all three axes are minimized.",
        ha="center",
        fontsize=6.2,
        color="#3F3F3F",
    )
    save_figure(fig, "direct_use_expansion_pareto_3d_standalone")


def truncated_greys() -> mcolors.LinearSegmentedColormap:
    """Preserve the original grayscale encoding without allowing pure white."""
    base = plt.colormaps["Greys_r"]
    return mcolors.LinearSegmentedColormap.from_list(
        "greys_visible", base(np.linspace(0.03, 0.78, 256))
    )


def plot_grouping_row(
    frontier: pd.DataFrame,
    reduced: pd.DataFrame,
    lower: np.ndarray,
    ranges: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(4.8, 7.8))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1, 0.045),
        left=0.14,
        right=0.88,
        bottom=0.075,
        top=0.975,
        hspace=0.34,
        wspace=0.14,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    colorbar_axis = fig.add_subplot(grid[:, 1])
    cmap = truncated_greys()
    norm = mcolors.Normalize(0, 1)
    retained_source_rows = []
    for ax, spec in zip(axes, GROUPS):
        retained = retained_points(
            frontier, reduced, spec["method"], lower, ranges
        )
        ax.scatter(
            frontier["TotEmiss_plot"],
            frontier["ISI_plot"],
            c=frontier["TAC_plot"],
            cmap=cmap,
            norm=norm,
            s=7,
            alpha=0.65,
            linewidth=0,
            zorder=1,
        )
        ax.scatter(
            retained[:, 1],
            retained[:, 2],
            s=14,
            color="#B64342",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.95,
            zorder=3,
        )
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f'Grouped: {spec["name"]}', fontsize=7.5, pad=5, loc="left")
        ax.text(
            -0.14,
            1.06,
            spec["panel"],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        ax.tick_params(width=0.7, length=3, labelsize=6.2)
        retained_source_rows.append(
            pd.DataFrame(
                {
                    "grouping": spec["name"],
                    "TotEmiss_normalized": retained[:, 1],
                    "ISI_normalized": retained[:, 2],
                    "TAC_normalized": retained[:, 0],
                }
            )
        )

    fig.supxlabel("Normalized total emissions", fontsize=7.2, y=0.018)
    fig.supylabel("Normalized ISI", fontsize=7.2, x=0.035)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label("Normalized TAC", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8, width=0.6, length=2.5)
    pd.concat(retained_source_rows, ignore_index=True).to_csv(
        RESULT_DIR / "direct_use_expansion_grouping_projection_retained_points.csv",
        index=False,
    )
    save_figure(fig, "direct_use_expansion_grouping_projections_vertical")


def draw_orca_network(ax, orca: pd.DataFrame) -> None:
    positions = {
        "TAC": np.array([0.50, 0.76]),
        "TotEmiss": np.array([0.23, 0.27]),
        "ISI": np.array([0.77, 0.27]),
    }
    labels = {"TAC": "TAC", "TotEmiss": "Total\nemissions", "ISI": "ISI"}
    node_colors = {
        "TAC": "#E8C3CA",
        "TotEmiss": "#C4D4E5",
        "ISI": "#C4D4E5",
    }
    node_text_colors = {
        "TAC": "#6E3543",
        "TotEmiss": "#263F5D",
        "ISI": "#263F5D",
    }
    edges = [
        ("TAC", "TotEmiss", "TAC__TotEmiss"),
        ("TAC", "ISI", "TAC__ISI"),
        ("TotEmiss", "ISI", "TotEmiss__ISI"),
    ]
    group_patch = Ellipse(
        (0.50, 0.27),
        0.88,
        0.30,
        facecolor="#F5F7FA",
        edgecolor="#6F89B5",
        linewidth=0.7,
        linestyle="--",
        zorder=0,
    )
    ax.add_patch(group_patch)
    ax.text(
        0.50,
        0.075,
        "Selected group",
        ha="center",
        color="#3E5F91",
        fontsize=5.8,
    )

    label_offsets = {
        "TAC__TotEmiss": np.array([-0.055, 0.010]),
        "TAC__ISI": np.array([0.055, 0.010]),
        "TotEmiss__ISI": np.array([0.0, -0.055]),
    }
    source_rows = []
    for node_a, node_b, pair in edges:
        row = orca.loc[pair]
        adjacency = float(row["paper_adjacency_mean"])
        signed = float(row["paper_signed_mean"])
        start, end = positions[node_a], positions[node_b]
        width = 0.7 + 2.8 * (adjacency - 0.70) / 0.20
        edge_color = "#4477AA" if pair == "TotEmiss__ISI" else "#AAB3C2"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=edge_color,
            linewidth=width,
            solid_capstyle="round",
            zorder=1,
        )
        midpoint = (start + end) / 2 + label_offsets[pair]
        ax.text(
            midpoint[0],
            midpoint[1],
            f"{adjacency:.3f}",
            ha="center",
            va="center",
            fontsize=5.2,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=3,
        )
        source_rows.append(
            {"pair": pair, "orca_adjacency": adjacency, "orca_signed": signed}
        )

    for node, position in positions.items():
        patch = Circle(
            position,
            radius=0.105,
            facecolor=node_colors[node],
            edgecolor="#59647A",
            linewidth=0.7,
            zorder=4,
        )
        ax.add_patch(patch)
        ax.text(
            position[0],
            position[1],
            labels[node],
            color=node_text_colors[node],
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            zorder=5,
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.02, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    pd.DataFrame(source_rows).to_csv(
        RESULT_DIR / "direct_use_expansion_orca_network_source.csv", index=False
    )


def plot_frontier_orca(frontier: pd.DataFrame, orca: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(4.8, 6.6))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.35, 1.0), hspace=0.02)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    draw_frontier_3d(ax3d, frontier, compact=True)
    ax3d.text2D(
        -0.08,
        1.02,
        "a",
        transform=ax3d.transAxes,
        fontsize=9,
        fontweight="bold",
    )

    ax_network = fig.add_subplot(grid[1, 0])
    draw_orca_network(ax_network, orca)
    ax_network.text(
        -0.08,
        1.02,
        "b",
        transform=ax_network.transAxes,
        fontsize=9,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.035, top=0.985)
    save_figure(fig, "direct_use_expansion_pareto_orca_network_vertical")


def main() -> None:
    frontier, reduced, lower, ranges, orca = load_data()
    plot_frontier_standalone(frontier)
    plot_grouping_row(frontier, reduced, lower, ranges)
    plot_frontier_orca(frontier, orca)
    print("Saved revised Pareto and ORCA demonstration figures")


if __name__ == "__main__":
    main()
