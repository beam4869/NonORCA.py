"""Draw the five-panel DTLZ5 and DTLZ6 objective-affinity network figures.

Matplotlib is used directly for every visual element.  The canvas is fixed at
183 x 90 mm: four 39.5-mm square panels form an 83-mm square on the left, and
one 83-mm square hero panel occupies the right.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[3] / ".matplotlib-cache"),
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "source_data"

WIDTH_MM = 183.0
HEIGHT_MM = 90.0
MM_PER_INCH = 25.4

PANEL_LAYOUT_MM = {
    "a": (6.0, 46.5, 39.5, 39.5),
    "b": (49.5, 46.5, 39.5, 39.5),
    "c": (6.0, 3.0, 39.5, 39.5),
    "d": (49.5, 3.0, 39.5, 39.5),
    "e": (94.0, 3.0, 83.0, 83.0),
}

FIGURE_CASES = {
    "DTLZ5": ((2, 3), (3, 5), (4, 7), (6, 10), (7, 16)),
    "DTLZ6": ((2, 4), (3, 5), (4, 7), (6, 10), (7, 16)),
}

GROUP_COLORS = (
    "#F08080",  # light-coral correlated block, matching the earlier figure vocabulary
    "#B52D2D",  # brown-red
    "#F5B2C2",  # pink
    "#929292",  # grey
    "#9A9300",  # olive
    "#42CF79",  # moderated lime
    "#6E9ED6",  # blue extension for a seventh group
)
INK = "#20252B"
EDGE = "#424952"
MUTED = "#67717E"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def mm_axes(fig: plt.Figure, box_mm: tuple[float, float, float, float]) -> plt.Axes:
    x, y, width, height = box_mm
    return fig.add_axes(
        [
            x / WIDTH_MM,
            y / HEIGHT_MM,
            width / WIDTH_MM,
            height / HEIGHT_MM,
        ]
    )


def canonical_groups(values: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    canonical = np.empty(len(values), dtype=int)
    for index, value in enumerate(np.asarray(values, dtype=int)):
        mapping.setdefault(int(value), len(mapping))
        canonical[index] = mapping[int(value)]
    return canonical


def text_color(hex_color: str) -> str:
    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 145 else INK


def read_case(family: str, intrinsic_dimension: int, num_objectives: int):  # noqa: ANN201
    lower = family.lower()
    folder = SOURCE_ROOT / ("dtlz5" if family == "DTLZ5" else "dtlz6_figure")
    stem = f"{lower}_i{intrinsic_dimension}_m{num_objectives}"
    matrix_frame = pd.read_csv(folder / f"{stem}_affinity.csv", index_col=0)
    nodes = pd.read_csv(folder / f"{stem}_nodes.csv")
    matrix = matrix_frame.to_numpy(dtype=float)
    groups = canonical_groups(nodes["recovered_group"].to_numpy(dtype=int))

    if family == "DTLZ5":
        stability = pd.read_csv(folder / "dtlz5_network_stability_summary.csv")
        row = stability[
            (stability["dataset"] == f"DTLZ5({intrinsic_dimension},{num_objectives})")
            & (stability["initial_point_strategy"] == "deterministic")
        ].iloc[0]
    else:
        stability = pd.read_csv(folder / "dtlz6_network_stability_summary.csv")
        row = stability[
            stability["dataset"] == f"DTLZ6({intrinsic_dimension},{num_objectives})"
        ].iloc[0]
    return matrix_frame.index.tolist(), matrix, groups, row


def maximum_spanning_tree_edges(matrix: np.ndarray, members: list[int]) -> set[tuple[int, int]]:
    if len(members) <= 1:
        return set()
    parent = {member: member for member in members}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    edges = sorted(
        (
            (float(matrix[left, right]), left, right)
            for pos, left in enumerate(members)
            for right in members[pos + 1 :]
        ),
        reverse=True,
    )
    selected: set[tuple[int, int]] = set()
    for _, left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        selected.add((min(left, right), max(left, right)))
        if len(selected) == len(members) - 1:
            break
    return selected


def labelled_edge_subset(matrix: np.ndarray, groups: np.ndarray) -> set[tuple[int, int]]:
    num_objectives = matrix.shape[0]
    if num_objectives <= 5:
        return {
            (left, right)
            for left in range(num_objectives)
            for right in range(left + 1, num_objectives)
        }

    group_members = {
        group: np.flatnonzero(groups == group).tolist()
        for group in np.unique(groups)
    }
    main_group = max(group_members, key=lambda group: len(group_members[group]))
    main_members = group_members[main_group]
    selected = maximum_spanning_tree_edges(matrix, main_members)

    for group, members in group_members.items():
        if group == main_group:
            continue
        for member in members:
            partner = max(main_members, key=lambda candidate: float(matrix[member, candidate]))
            selected.add((min(member, partner), max(member, partner)))
    return selected


def node_radius(num_objectives: int, is_hero: bool) -> float:
    if is_hero:
        return 0.11
    if num_objectives <= 3:
        return 0.22
    if num_objectives <= 5:
        return 0.19
    if num_objectives <= 7:
        return 0.16
    return 0.14


def draw_edge_label(
    ax: plt.Axes,
    left: int,
    right: int,
    weight: float,
    positions: np.ndarray,
    *,
    fontsize: float,
) -> None:
    start, end = positions[left], positions[right]
    fraction = 0.37 if (left + right) % 2 == 0 else 0.63
    point = (1.0 - fraction) * start + fraction * end
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length > 0:
        normal = np.array([-direction[1], direction[0]]) / length
        point = point + normal * (0.022 if (left * 3 + right) % 2 == 0 else -0.022)
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    label = ax.text(
        point[0],
        point[1],
        f"{weight:.3f}",
        ha="center",
        va="center",
        rotation=angle,
        rotation_mode="anchor",
        fontsize=fontsize,
        color=INK,
        zorder=3,
    )
    label.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])


def draw_network(
    ax: plt.Axes,
    family: str,
    intrinsic_dimension: int,
    num_objectives: int,
    panel: str,
    *,
    is_hero: bool,
) -> None:
    labels, matrix, groups, stability = read_case(family, intrinsic_dimension, num_objectives)
    radius = 0.88 if is_hero else 0.78
    angles = 2.0 * math.pi * np.arange(num_objectives) / num_objectives
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    selected_labels = labelled_edge_subset(matrix, groups)

    edges = sorted(
        (
            (float(matrix[left, right]), left, right)
            for left in range(num_objectives)
            for right in range(left + 1, num_objectives)
        ),
        key=lambda item: item[0],
    )
    for weight, left, right in edges:
        ax.plot(
            [positions[left, 0], positions[right, 0]],
            [positions[left, 1], positions[right, 1]],
            color=EDGE,
            lw=(0.18 + 1.20 * weight**2) * (1.06 if is_hero else 0.92),
            alpha=0.10 + 0.65 * weight**2,
            solid_capstyle="round",
            zorder=1,
        )

    edge_fontsize = 4.5 if is_hero else (3.8 if num_objectives <= 5 else 3.35)
    for weight, left, right in edges:
        if (left, right) in selected_labels:
            draw_edge_label(
                ax,
                left,
                right,
                weight,
                positions,
                fontsize=edge_fontsize,
            )

    circle_radius = node_radius(num_objectives, is_hero)
    node_fontsize = 5.8 if is_hero else (5.1 if num_objectives <= 7 else 4.7)
    for index, label in enumerate(labels):
        color = GROUP_COLORS[int(groups[index]) % len(GROUP_COLORS)]
        ax.add_patch(
            Circle(
                positions[index],
                circle_radius,
                facecolor=color,
                edgecolor="#4A4F55",
                linewidth=0.72 if is_hero else 0.58,
                zorder=5,
            )
        )
        ax.text(
            positions[index, 0],
            positions[index, 1],
            label,
            ha="center",
            va="center",
            fontsize=node_fontsize,
            fontweight="bold",
            color=text_color(color),
            zorder=6,
        )

    title_size = 7.0 if is_hero else 6.15
    metric_size = 5.0 if is_hero else 4.25
    ax.text(
        -1.03,
        1.17,
        f"{family}({intrinsic_dimension},{num_objectives})",
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        1.22,
        1.17,
        f"ARI {float(stability['ari_mean']):.2f} · {int(stability['exact_runs'])}/{int(stability['runs'])}",
        ha="right",
        va="center",
        fontsize=metric_size,
        color=MUTED,
    )
    ax.text(
        -1.27,
        1.18,
        panel,
        ha="left",
        va="center",
        fontsize=8.0 if is_hero else 7.5,
        fontweight="bold",
        color=INK,
    )
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def export_figure(fig: plt.Figure, stem: str) -> None:
    base = HERE / stem
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def make_family_figure(family: str) -> None:
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    axes = {panel: mm_axes(fig, box) for panel, box in PANEL_LAYOUT_MM.items()}
    for panel, (intrinsic_dimension, num_objectives) in zip(
        PANEL_LAYOUT_MM,
        FIGURE_CASES[family],
    ):
        draw_network(
            axes[panel],
            family,
            intrinsic_dimension,
            num_objectives,
            panel,
            is_hero=panel == "e",
        )
    fig.text(
        0.5,
        0.008,
        "Node fill: inferred group · edge width/opacity: ORCA affinity · all edges shown; selected weights labelled · fixed K = I",
        ha="center",
        va="bottom",
        fontsize=4.3,
        color=MUTED,
    )
    export_figure(fig, f"figure_{family.lower()}_affinity_network_grid")


def main() -> None:
    for family in FIGURE_CASES:
        make_family_figure(family)
    print(f"Network figures written to {HERE}")


if __name__ == "__main__":
    main()
