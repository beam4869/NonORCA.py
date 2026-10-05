"""Reproduce the earlier NetworkX/Matplotlib graph style for DTLZ5 and DTLZ6.

Every case is first exported as a standalone square figure.  The same drawing
function is then used to assemble the requested four-small-plus-one-large
composite.  All pairwise affinity strings are displayed in every rendering.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / ".matplotlib-cache"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data" / "networkx_all_labels"
OUTPUT_DIR = HERE / "networkx_original_style"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
COMBINED_DIR = OUTPUT_DIR / "combined"

COMBINED_WIDTH_MM = 183.0
COMBINED_HEIGHT_MM = 90.0
INDIVIDUAL_SIZE_MM = 140.0
MM_PER_INCH = 25.4

CASES = {
    "DTLZ5": ((2, 3), (3, 5), (4, 7), (6, 10), (7, 12)),
    "DTLZ6": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 12)),
}

COMBINED_LAYOUT_MM = {
    "a": (6.0, 46.5, 39.5, 39.5),
    "b": (49.5, 46.5, 39.5, 39.5),
    "c": (6.0, 3.0, 39.5, 39.5),
    "d": (49.5, 3.0, 39.5, 39.5),
    "e": (94.0, 3.0, 83.0, 83.0),
}

# The first six colors follow the earlier script; cornflower blue extends the
# palette for a seventh recovered group.
GROUP_COLORS = (
    "lightcoral",
    "brown",
    "pink",
    "grey",
    "olive",
    "lime",
    "cornflowerblue",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def canonical_groups(values: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    output = np.empty(len(values), dtype=int)
    for index, value in enumerate(np.asarray(values, dtype=int)):
        mapping.setdefault(int(value), len(mapping))
        output[index] = mapping[int(value)]
    return output


def read_case(family: str, intrinsic_dimension: int, num_objectives: int):  # noqa: ANN201
    stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}"
    matrix_frame = pd.read_csv(SOURCE_DIR / f"{stem}_affinity.csv", index_col=0)
    nodes = pd.read_csv(SOURCE_DIR / f"{stem}_nodes.csv")
    summary = pd.read_csv(SOURCE_DIR / "networkx_figure_case_summary.csv")
    row = summary[summary["dataset"] == f"{family}({intrinsic_dimension},{num_objectives})"].iloc[0]
    labels = matrix_frame.index.tolist()
    matrix = matrix_frame.to_numpy(dtype=float)
    groups = canonical_groups(nodes["recovered_group"].to_numpy(dtype=int))
    return labels, matrix, groups, row


def build_graph(labels: list[str], matrix: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    edges = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            edges.append((labels[left], labels[right], float(matrix[left, right])))
    graph.add_weighted_edges_from(edges)
    return graph


def node_size_for(panel_width_mm: float, num_objectives: int) -> float:
    width_points = panel_width_mm / MM_PER_INCH * 72.0
    if num_objectives <= 4:
        diameter_fraction = 0.16
    elif num_objectives <= 5:
        diameter_fraction = 0.14
    elif num_objectives <= 7:
        diameter_fraction = 0.125
    elif num_objectives <= 10:
        diameter_fraction = 0.105
    else:
        diameter_fraction = 0.098
    return float((width_points * diameter_fraction) ** 2)


def node_font_size(num_objectives: int, *, standalone: bool, hero: bool) -> float:
    if standalone:
        return 10.0 if num_objectives <= 7 else 8.5
    if hero:
        return 6.2
    if num_objectives <= 5:
        return 5.0
    if num_objectives <= 7:
        return 4.4
    return 3.8


def edge_font_size(num_objectives: int, *, standalone: bool, hero: bool) -> float:
    if standalone:
        return {3: 9.0, 4: 8.5, 5: 7.5, 7: 5.8, 10: 4.2, 12: 3.8}.get(num_objectives, 4.0)
    if hero:
        return 3.3
    return {3: 4.7, 4: 4.4, 5: 3.8, 7: 2.7, 10: 1.95}.get(num_objectives, 2.2)


def draw_original_style_network(
    ax: plt.Axes,
    family: str,
    intrinsic_dimension: int,
    num_objectives: int,
    *,
    panel_width_mm: float,
    standalone: bool,
    hero: bool,
    panel_letter: str | None = None,
) -> None:
    labels, matrix, groups, summary = read_case(family, intrinsic_dimension, num_objectives)
    graph = build_graph(labels, matrix)
    positions = nx.circular_layout(graph)
    node_colors = [GROUP_COLORS[int(groups[index]) % len(GROUP_COLORS)] for index in range(len(labels))]
    node_size = node_size_for(panel_width_mm, num_objectives)

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="#3F444A",
        linewidths=0.75 if standalone or hero else 0.55,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#4F555B",
        width=0.75 if standalone else (0.62 if hero else 0.38),
        alpha=0.58 if standalone else 0.50,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=node_font_size(num_objectives, standalone=standalone, hero=hero),
        font_color="black",
        font_weight="normal",
        ax=ax,
    )

    # Exactly as in the earlier workflow, every pairwise edge receives a text
    # label.  Values are rounded rather than truncated as raw strings.
    edge_labels = {
        (left, right): f"{float(data['weight']):.3f}"
        for left, right, data in graph.edges(data=True)
    }
    node_index = {label: index for index, label in enumerate(labels)}
    for (left, right), value in edge_labels.items():
        # The earlier script placed every label at the exact midpoint.  Five
        # deterministic positions retain every string while separating the
        # diametric edges that otherwise collide at the centre of the circle.
        slot = (7 * node_index[left] + 3 * node_index[right]) % 5
        label_position = 0.34 + 0.08 * slot
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={(left, right): value},
            label_pos=label_position,
            font_size=edge_font_size(num_objectives, standalone=standalone, hero=hero),
            font_color="#202428",
            rotate=True,
            bbox={"boxstyle": "round,pad=0.015", "facecolor": "white", "edgecolor": "none", "alpha": 0.62},
            node_size=node_size,
            ax=ax,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.16 if standalone else 0.14)

    if standalone:
        ax.text(
            0.0,
            1.035,
            f"{family}({intrinsic_dimension},{num_objectives})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
        )
        ax.text(
            1.0,
            1.035,
            f"ARI {float(summary['ari_mean']):.2f} · {int(summary['exact_runs'])}/{int(summary['runs'])} exact",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="#626C78",
        )
    else:
        ax.text(
            0.0,
            1.005,
            panel_letter or "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0 if hero else 7.3,
            fontweight="bold",
        )
        ax.text(
            0.10,
            1.005,
            f"{family}({intrinsic_dimension},{num_objectives})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0 if hero else 5.8,
            fontweight="bold",
        )
        ax.text(
            1.0,
            1.005,
            f"ARI {float(summary['ari_mean']):.2f} · {int(summary['exact_runs'])}/{int(summary['runs'])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.0 if hero else 3.9,
            color="#626C78",
        )


def save_figure(fig: plt.Figure, base: Path, *, png_dpi: int = 300) -> None:
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=png_dpi)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def make_individual_figures() -> None:
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    for family, family_cases in CASES.items():
        for intrinsic_dimension, num_objectives in family_cases:
            fig = plt.figure(
                figsize=(INDIVIDUAL_SIZE_MM / MM_PER_INCH, INDIVIDUAL_SIZE_MM / MM_PER_INCH)
            )
            ax = fig.add_axes([0.055, 0.045, 0.89, 0.86])
            draw_original_style_network(
                ax,
                family,
                intrinsic_dimension,
                num_objectives,
                panel_width_mm=INDIVIDUAL_SIZE_MM * 0.89,
                standalone=True,
                hero=False,
            )
            stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}_all_edge_labels"
            save_figure(fig, INDIVIDUAL_DIR / stem)


def add_mm_axes(fig: plt.Figure, box_mm: tuple[float, float, float, float]) -> plt.Axes:
    x, y, width, height = box_mm
    return fig.add_axes(
        [
            x / COMBINED_WIDTH_MM,
            y / COMBINED_HEIGHT_MM,
            width / COMBINED_WIDTH_MM,
            height / COMBINED_HEIGHT_MM,
        ]
    )


def make_combined_figures() -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for family, family_cases in CASES.items():
        fig = plt.figure(
            figsize=(COMBINED_WIDTH_MM / MM_PER_INCH, COMBINED_HEIGHT_MM / MM_PER_INCH)
        )
        for panel_letter, (intrinsic_dimension, num_objectives) in zip(
            COMBINED_LAYOUT_MM,
            family_cases,
        ):
            box = COMBINED_LAYOUT_MM[panel_letter]
            ax = add_mm_axes(fig, box)
            draw_original_style_network(
                ax,
                family,
                intrinsic_dimension,
                num_objectives,
                panel_width_mm=box[2],
                standalone=False,
                hero=panel_letter == "e",
                panel_letter=panel_letter,
            )
        save_figure(fig, COMBINED_DIR / f"figure_{family.lower()}_networkx_all_edge_labels")


def main() -> None:
    make_individual_figures()
    make_combined_figures()
    print(f"Individual and combined NetworkX-style figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
