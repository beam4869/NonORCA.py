from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
ORCA_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src"
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib-cache"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ORCA_PACKAGE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge

from orca.config import NonlinearORCAConfig
from orca.nonlinear import nonlinear_orca

from oval_objective_reduction_demo import (
    AXES,
    OBJECTIVES,
    OvalORCAProblem,
    ellipse_grad,
    point_metadata_from_package_order,
    tangent_line_segment,
    unit,
)


COLORS = {
    "ink": "#18212b",
    "muted": "#667085",
    "grid": "#d9e0ea",
    "ellipse": "#4f83cc",
    "ellipse_fill": "#e9f3ff",
    "green": "#4c9a6a",
    "green2": "#71aa59",
    "purple": "#796bc8",
    "yellow": "#c59a2f",
}
SEED_COLORS = [COLORS["green"], COLORS["green2"], COLORS["purple"], COLORS["yellow"]]


def candidate_descent_directions(x: np.ndarray) -> np.ndarray:
    """Directions used by nonlinear ORCA fixed-point generation.

    The package projects each objective descent direction onto the selected
    local constraint surface. If the tangent projection vanishes, it falls back
    to the original descent direction.
    """

    normal = ellipse_grad(x)
    directions = []
    for obj in OBJECTIVES:
        descent = unit(-obj.gradient(x))
        denom = float(np.dot(normal, normal))
        projected = descent.copy()
        if denom > 1.0e-12:
            projected = descent - float(np.dot(descent, normal)) / denom * normal
        projected = unit(projected)
        if np.linalg.norm(projected) <= 1.0e-12:
            projected = descent
        directions.append(unit(projected))
    return np.vstack(directions)


def minimal_angle_arc(angles: np.ndarray) -> tuple[float, float, float]:
    """Return theta_start, theta_end, width for the shortest arc containing angles."""

    vals = np.sort(np.mod(angles, 2.0 * math.pi))
    extended = np.concatenate([vals, vals[:1] + 2.0 * math.pi])
    gaps = np.diff(extended)
    largest_gap_idx = int(np.argmax(gaps))
    start = extended[largest_gap_idx + 1] % (2.0 * math.pi)
    end = extended[largest_gap_idx]
    if end < start:
        end += 2.0 * math.pi
    return start, end, end - start


def draw_objective_cone(ax, x: np.ndarray, color: str, radius: float = 0.28) -> None:
    """Draw the positive cone spanned by projected objective descent directions."""

    dirs = candidate_descent_directions(x)
    nonzero = dirs[np.linalg.norm(dirs, axis=1) > 1.0e-12]
    if len(nonzero) == 0:
        return

    rank = np.linalg.matrix_rank(nonzero, tol=1.0e-8)
    angles = np.arctan2(nonzero[:, 1], nonzero[:, 0])

    if rank <= 1:
        theta = math.degrees(float(angles[0]))
        for center_angle in (theta, theta + 180.0):
            ax.add_patch(
                Wedge(
                    x,
                    radius,
                    center_angle - 7.0,
                    center_angle + 7.0,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.16,
                    linewidth=0.8,
                    zorder=4,
                )
            )
            direction = np.array([math.cos(math.radians(center_angle)), math.sin(math.radians(center_angle))])
            ax.plot(
                [x[0], x[0] + radius * direction[0]],
                [x[1], x[1] + radius * direction[1]],
                color=color,
                linewidth=1.0,
                alpha=0.55,
                zorder=5,
            )
        return

    start, end, width = minimal_angle_arc(angles)
    if width > math.pi:
        # If directions are not contained in a half-plane, the conic hull is
        # effectively very broad. Draw a full local disk to avoid pretending
        # there are sharp boundary rays.
        theta1, theta2 = 0.0, 360.0
    else:
        theta1, theta2 = math.degrees(start), math.degrees(end)

    ax.add_patch(
        Wedge(
            x,
            radius,
            theta1,
            theta2,
            facecolor=color,
            edgecolor=color,
            alpha=0.16,
            linewidth=0.9,
            zorder=4,
        )
    )
    for theta in (theta1, theta2):
        direction = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
        ax.plot(
            [x[0], x[0] + radius * direction[0]],
            [x[1], x[1] + radius * direction[1]],
            color=color,
            linewidth=1.0,
            alpha=0.65,
            zorder=5,
        )


def draw_tangent_constraint(ax, x: np.ndarray, color: str, alpha: float, linewidth: float) -> None:
    segment = tangent_line_segment(x, (-1.58, 1.58), (-1.08, 1.08))
    if segment is None:
        return
    a, b = segment
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        linestyle=(0, (5, 4)),
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=2,
    )


def setup_axis(ax, title: str) -> None:
    ax.add_patch(
        Ellipse(
            (0.0, 0.0),
            width=2.0 * AXES[0],
            height=2.0 * AXES[1],
            facecolor=COLORS["ellipse_fill"],
            edgecolor=COLORS["ellipse"],
            linewidth=1.8,
            zorder=1,
        )
    )
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(color="#edf1f6", linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=COLORS["ink"])


def build_points() -> tuple[dict[int, list[np.ndarray]], list[tuple[int, int]]]:
    problem = OvalORCAProblem()
    config = NonlinearORCAConfig(
        num_groups=3,
        num_points_per_seed=16,
        include_seed_points=True,
        random_seed=14,
        step_size=0.24,
        grouping_method="average_linkage",
        active_constraint_tolerance=1.0e-8,
    )
    result = nonlinear_orca(problem, config)
    points = np.asarray(result.input_data.points, dtype=float)
    meta = point_metadata_from_package_order(points, config)
    by_seed: dict[int, list[np.ndarray]] = {}
    for point, item in zip(points, meta):
        by_seed.setdefault(item[0], []).append(point)
    return by_seed, meta


def make_matplotlib_cone_figure() -> None:
    by_seed, _ = build_points()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlepad": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.09, top=0.86, wspace=0.16, hspace=0.28)
    fig.text(0.045, 0.94, "Objective-cone view of selected-point generation", fontsize=18, fontweight="bold", color=COLORS["ink"])
    fig.text(
        0.045,
        0.905,
        "Each step shows four seed-objective processes. Cones show positive combinations of projected objective descent directions.",
        fontsize=10,
        color=COLORS["muted"],
    )

    panels: list[tuple[str, int | None]] = [("Initial oval", None)]
    panels.extend((f"Step {step}", step) for step in range(5))

    for ax, (title, step_level) in zip(axes.flat, panels):
        setup_axis(ax, title)
        if step_level is None:
            ax.text(-1.52, -0.98, "Feasible region: g(x) <= 0", fontsize=7.5, color=COLORS["muted"])
            continue

        for seed_idx, color in enumerate(SEED_COLORS, start=1):
            trajectory = by_seed.get(seed_idx, [])
            shown = trajectory[: min(step_level + 1, len(trajectory))]
            if not shown:
                continue

            current = shown[-1]
            for old_point in shown[:-1]:
                draw_tangent_constraint(ax, old_point, "#c7d0dd", alpha=0.32, linewidth=0.7)
            draw_tangent_constraint(ax, current, color, alpha=0.84, linewidth=1.25)
            draw_objective_cone(ax, current, color, radius=0.26)

            arr = np.asarray(shown)
            if len(arr) > 1:
                ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.8, alpha=0.82, zorder=6)
                ax.scatter(arr[:-1, 0], arr[:-1, 1], s=10, color="#667085", edgecolor="white", linewidth=0.4, zorder=7)
            ax.scatter([current[0]], [current[1]], s=34, color=color, edgecolor="white", linewidth=0.8, zorder=9)

            grad = unit(OBJECTIVES[seed_idx - 1].gradient(current))
            ax.arrow(
                current[0],
                current[1],
                0.16 * grad[0],
                0.16 * grad[1],
                width=0.006,
                head_width=0.045,
                head_length=0.055,
                length_includes_head=True,
                color=color,
                alpha=0.95,
                zorder=10,
            )
            ax.text(
                current[0] + 0.18 * grad[0],
                current[1] + 0.18 * grad[1],
                f"grad f{seed_idx}",
                color=color,
                fontsize=6.5,
                ha="center",
                va="center",
                zorder=11,
            )

            if step_level < len(trajectory) - 1:
                nxt = trajectory[step_level + 1]
                direction = unit(nxt - current)
                ax.arrow(
                    current[0],
                    current[1],
                    0.18 * direction[0],
                    0.18 * direction[1],
                    width=0.003,
                    head_width=0.035,
                    head_length=0.045,
                    length_includes_head=True,
                    color="#111827",
                    alpha=0.58,
                    zorder=8,
                )

        if step_level == 0:
            ax.text(-1.52, -0.98, "Dashed: local feasible constraint; shaded: objective cone", fontsize=7.2, color=COLORS["muted"])

    legend_y = 0.045
    fig.text(
        0.045,
        legend_y,
        "Colored cones: positive cone of projected objective descent directions at the current selected point. "
        "Black arrows: actual generated direction toward the next selected point.",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    out_base = HERE / "oval_algorithm_steps_matplotlib_cones"
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    fig.savefig(out_base.with_suffix(".tiff"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_matplotlib_cone_figure()
