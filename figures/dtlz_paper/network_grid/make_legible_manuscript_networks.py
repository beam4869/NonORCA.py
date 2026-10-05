"""Create the final legible DTLZ objective-network figure.

Figure contract
---------------
Core conclusion: fixed-K ORCA recovers the prescribed block-and-singleton
partition in representative DTLZ5 and D6 positive controls.
Archetype: quantitative grid.
Backend: Python and Matplotlib only.
Output: editable PDF and SVG plus high-resolution PNG and TIFF.
Evidence hierarchy: node colors show the recovered groups, edge width shows
all affinities, and numeric labels identify within-block relations and the
strongest edge incident to each singleton. Full matrices remain source data.
Reviewer risk: labeling every edge of a complete 12-node graph is illegible at
manuscript width, so the dense cases are documented in the table and archive.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(HERE.parents[2] / ".matplotlib-cache"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SOURCE_DIR = HERE / "source_data" / "networkx_all_labels"
OUTPUT_DIR = HERE / "networkx_original_code_adjusted_ab_scaled_adaptive_labels" / "manuscript"

PANELS = (
    ("a", "DTLZ5", "DTLZ5", 2, 4),
    ("b", "D6", "DTLZ6", 2, 4),
    ("c", "DTLZ5", "DTLZ5", 4, 7),
    ("d", "D6", "DTLZ6", 4, 7),
)

GROUP_COLORS = (
    "#EF7F82",
    "#A92327",
    "#F3B4C0",
    "#898989",
    "#A29A00",
    "#22D93A",
    "#6797E5",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
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


def read_result(
    source_family: str,
    intrinsic_dimension: int,
    num_objectives: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    stem = f"{source_family.lower()}_i{intrinsic_dimension}_m{num_objectives}"
    matrix_frame = pd.read_csv(SOURCE_DIR / f"{stem}_affinity.csv", index_col=0)
    nodes = pd.read_csv(SOURCE_DIR / f"{stem}_nodes.csv")
    return (
        matrix_frame.index.tolist(),
        matrix_frame.to_numpy(dtype=float),
        canonical_groups(nodes["recovered_group"].to_numpy(dtype=int)),
    )


def labeled_edges(matrix: np.ndarray, groups: np.ndarray) -> set[tuple[int, int]]:
    """Return within-group edges and one strongest edge per singleton."""

    num_objectives = len(groups)
    counts = {group: int(np.sum(groups == group)) for group in np.unique(groups)}
    selected: set[tuple[int, int]] = set()
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            if groups[i] == groups[j] and counts[int(groups[i])] > 1:
                selected.add((i, j))
    for i in range(num_objectives):
        if counts[int(groups[i])] != 1:
            continue
        candidates = [j for j in range(num_objectives) if j != i]
        j = max(candidates, key=lambda candidate: float(matrix[i, candidate]))
        selected.add(tuple(sorted((i, j))))
    return selected


def draw_panel(
    ax: plt.Axes,
    panel: str,
    display_family: str,
    source_family: str,
    intrinsic_dimension: int,
    num_objectives: int,
) -> None:
    labels, matrix, groups = read_result(
        source_family,
        intrinsic_dimension,
        num_objectives,
    )
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    all_edges: list[tuple[str, str]] = []
    widths: list[float] = []
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            all_edges.append((labels[i], labels[j]))
            widths.append(0.35 + 1.65 * float(matrix[i, j]))
    graph.add_edges_from(all_edges)
    positions = nx.circular_layout(graph)

    nx.draw_networkx_edges(
        graph,
        positions,
        width=widths,
        edge_color="#707986",
        alpha=0.46,
        ax=ax,
    )
    node_size = 2450 if num_objectives == 4 else 1250
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[GROUP_COLORS[int(group)] for group in groups],
        node_size=node_size,
        edgecolors="#3F4752",
        linewidths=0.85,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=11 if num_objectives == 4 else 9.5,
        font_color="#15191E",
        ax=ax,
    )

    for label_index, (i, j) in enumerate(sorted(labeled_edges(matrix, groups))):
        edge = (labels[i], labels[j])
        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={edge: f"{matrix[i, j]:.3f}"},
            label_pos=0.38 if label_index % 2 == 0 else 0.62,
            font_size=8.5,
            rotate=True,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            ax=ax,
        )

    ax.text(
        0.01,
        0.99,
        f"{panel}  {display_family}({intrinsic_dimension},{num_objectives})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color="#111827",
    )
    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.3))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.985, wspace=0.08, hspace=0.11)
    for ax, panel_spec in zip(axes.flat, PANELS):
        draw_panel(ax, *panel_spec)

    base = OUTPUT_DIR / "figure_dtlz5_d6_legible"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
