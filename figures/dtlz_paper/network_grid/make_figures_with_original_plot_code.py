"""Draw current ORCA results with the user's original NetworkX code style.

The network panels are rendered one at a time with the same plotting calls and
display choices as the earlier script.  The composite figures are then made by
placing those already-rendered PNG panels on a new Matplotlib canvas; the
networks are not redrawn for the composite.
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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data" / "networkx_all_labels"
OUTPUT_DIR = HERE / "networkx_original_code_adjusted_ab_scaled_adaptive_labels"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
COMBINED_DIR = OUTPUT_DIR / "combined"

CASES = {
    "DTLZ5": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 12)),
    "DTLZ6": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 12)),
}

# The first six entries are copied from the earlier plotting code.  The final
# entry is needed because the requested (7, 12) panels contain seven groups.
GROUP_COLORS = (
    "lightcoral",
    "brown",
    "pink",
    "grey",
    "olive",
    "lime",
    "cornflowerblue",
)

MM_PER_INCH = 25.4
COMBINED_WIDTH_MM = 183.0
COMBINED_HEIGHT_MM = 90.0
COMBINED_LAYOUT_MM = {
    "a": (1.0, 45.5, 43.5, 43.5),
    "b": (45.5, 45.5, 43.5, 43.5),
    "c": (1.0, 1.0, 43.5, 43.5),
    "d": (45.5, 1.0, 43.5, 43.5),
    "e": (94.0, 2.0, 86.0, 86.0),
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def canonical_groups(values: np.ndarray) -> np.ndarray:
    """Renumber recovered groups by first appearance for stable colors."""

    mapping: dict[int, int] = {}
    output = np.empty(len(values), dtype=int)
    for index, value in enumerate(np.asarray(values, dtype=int)):
        mapping.setdefault(int(value), len(mapping))
        output[index] = mapping[int(value)]
    return output


def read_current_result(
    family: str,
    intrinsic_dimension: int,
    num_objectives: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read the unchanged current affinity matrix and recovered partition."""

    stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}"
    matrix_frame = pd.read_csv(SOURCE_DIR / f"{stem}_affinity.csv", index_col=0)
    nodes = pd.read_csv(SOURCE_DIR / f"{stem}_nodes.csv")
    labels = matrix_frame.index.tolist()
    matrix = matrix_frame.to_numpy(dtype=float)
    groups = canonical_groups(nodes["recovered_group"].to_numpy(dtype=int))
    return labels, matrix, groups


def original_edge_string(value: float) -> str:
    """Use the exact string truncation rule from the earlier plotting code."""

    return str(float(value))[0:5]


def node_size_for(intrinsic_dimension: int, num_objectives: int) -> float:
    """Apply the requested case-specific node-size adjustments."""

    case = (intrinsic_dimension, num_objectives)
    if case == (2, 4):
        # NetworkX node_size is marker area: 4x area gives 2x radius.
        return 52000.0
    if case == (3, 5):
        # 2.25x area gives 1.5x radius.
        return 29250.0
    if case == (7, 12):
        return 7500.0
    return 10000.0


def font_sizes_for(
    intrinsic_dimension: int,
    num_objectives: int,
) -> tuple[float, float]:
    """Enlarge text in the four supporting panels."""

    case = (intrinsic_dimension, num_objectives)
    if case == (2, 4):
        return 34.0, 26.0
    if case == (3, 5):
        return 25.5, 19.5
    if num_objectives < 12:
        return 17.0, 13.0
    return 14.0, 11.0


def draw_adaptive_edge_labels(
    graph: nx.Graph,
    positions: dict[str, np.ndarray],
    edge_labels: dict[tuple[str, str], str],
    edge_font_size: float,
    ax: plt.Axes,
) -> None:
    """Distribute center-crowded labels along their original straight edges."""

    edge_distance_to_center = {}
    for edge in edge_labels:
        left, right = edge
        midpoint = (positions[left] + positions[right]) / 2.0
        edge_distance_to_center[edge] = float(np.linalg.norm(midpoint))

    def sorted_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(
            edges,
            key=lambda edge: (int(edge[0][1:]), int(edge[1][1:])),
        )

    def assign_alternating_positions(
        output: dict[tuple[str, str], float],
        edges: list[tuple[str, str]],
        choices: tuple[float, float],
    ) -> None:
        for index, edge in enumerate(sorted_edges(edges)):
            output[edge] = choices[index % 2]

    num_objectives = len(graph)
    adaptive_positions: dict[tuple[str, str], float] = {}
    center_edges = [
        edge for edge, distance in edge_distance_to_center.items()
        if distance < 0.06
    ]
    if num_objectives == 12:
        # Separate the three innermost chord families into distinct label bands.
        assign_alternating_positions(adaptive_positions, center_edges, (0.28, 0.72))
        assign_alternating_positions(
            adaptive_positions,
            [
                edge for edge, distance in edge_distance_to_center.items()
                if 0.18 <= distance < 0.38
            ],
            (0.40, 0.60),
        )
        assign_alternating_positions(
            adaptive_positions,
            [
                edge for edge, distance in edge_distance_to_center.items()
                if 0.42 <= distance < 0.58
            ],
            (0.35, 0.65),
        )
    else:
        center_label_positions = {
            4: (0.35, 0.65),
            10: (0.25, 0.75),
        }.get(num_objectives, (0.32, 0.68))
        assign_alternating_positions(
            adaptive_positions,
            center_edges,
            center_label_positions,
        )

    for edge, label in edge_labels.items():
        label_pos = adaptive_positions.get(edge, 0.50)

        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels={edge: label},
            label_pos=label_pos,
            font_size=edge_font_size,
            rotate=True,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            ax=ax,
        )


def draw_one_with_original_code(
    family: str,
    intrinsic_dimension: int,
    num_objectives: int,
) -> plt.Figure:
    """Apply the earlier NetworkX/Matplotlib drawing sequence verbatim."""

    labels, adj_matrix, groups = read_current_result(
        family,
        intrinsic_dimension,
        num_objectives,
    )

    # Create a graph.
    graph = nx.Graph()

    # Define the nodes (circles) with their labels.
    graph.add_nodes_from(labels)

    # Define all edges and use the earlier five-character value strings.
    edges = []
    for i in range(num_objectives):
        for j in np.arange(i + 1, num_objectives):
            edges.append(
                (
                    f"f{i + 1}",
                    f"f{j + 1}",
                    original_edge_string(adj_matrix[i][j]),
                )
            )
    graph.add_weighted_edges_from(edges)

    # Get positions of the nodes in a circular layout.
    positions = nx.circular_layout(graph)

    # Apply the earlier palette to the current algorithm's recovered groups.
    node_colors = [GROUP_COLORS[int(group)] for group in groups]
    node_size = node_size_for(intrinsic_dimension, num_objectives)
    node_font_size, edge_font_size = font_sizes_for(
        intrinsic_dimension,
        num_objectives,
    )

    # Keep the earlier drawing calls while applying the requested size changes.
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_edges(graph, positions, ax=ax)
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=node_font_size,
        font_color="black",
        ax=ax,
    )
    edge_labels = {
        (left, right): f"{data['weight']}"
        for left, right, data in graph.edges(data=True)
    }
    draw_adaptive_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        edge_font_size=edge_font_size,
        ax=ax,
    )
    ax.set_aspect("equal")
    case = (intrinsic_dimension, num_objectives)
    if case == (2, 4):
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
    elif case == (3, 5):
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-1.30, 1.30)
    ax.axis("off")
    if num_objectives < 12:
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    return fig


def save_individual(fig: plt.Figure, base: Path) -> None:
    """Export one square panel before it is used by the composite."""

    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def add_rendered_panel(
    fig: plt.Figure,
    image_path: Path,
    rectangle_mm: tuple[float, float, float, float],
    panel_letter: str,
    title: str,
) -> None:
    """Place an already-rendered panel without redrawing its network."""

    x_mm, y_mm, width_mm, height_mm = rectangle_mm
    ax = fig.add_axes(
        [
            x_mm / COMBINED_WIDTH_MM,
            y_mm / COMBINED_HEIGHT_MM,
            width_mm / COMBINED_WIDTH_MM,
            height_mm / COMBINED_HEIGHT_MM,
        ]
    )
    ax.imshow(mpimg.imread(image_path), interpolation="lanczos")
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        f"{panel_letter}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0 if panel_letter == "e" else 6.2,
        fontweight="bold",
        color="black",
    )


def make_combined(family: str) -> None:
    """Stitch the five standalone PNGs into the requested asymmetric plate."""

    fig = plt.figure(
        figsize=(COMBINED_WIDTH_MM / MM_PER_INCH, COMBINED_HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    for panel_letter, (intrinsic_dimension, num_objectives) in zip(
        COMBINED_LAYOUT_MM,
        CASES[family],
    ):
        stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}_original_code"
        add_rendered_panel(
            fig,
            INDIVIDUAL_DIR / f"{stem}.png",
            COMBINED_LAYOUT_MM[panel_letter],
            panel_letter,
            f"{family}({intrinsic_dimension},{num_objectives})",
        )

    base = COMBINED_DIR / f"figure_{family.lower()}_stitched_original_code"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    for family, cases in CASES.items():
        for intrinsic_dimension, num_objectives in cases:
            stem = f"{family.lower()}_i{intrinsic_dimension}_m{num_objectives}_original_code"
            figure = draw_one_with_original_code(
                family,
                intrinsic_dimension,
                num_objectives,
            )
            save_individual(figure, INDIVIDUAL_DIR / stem)
            print(f"saved {family}({intrinsic_dimension},{num_objectives})")

    for family in CASES:
        make_combined(family)
        print(f"stitched {family}")


if __name__ == "__main__":
    main()
