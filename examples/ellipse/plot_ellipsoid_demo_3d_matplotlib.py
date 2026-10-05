from __future__ import annotations

import csv
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / ".matplotlib-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


AXES = np.array([1.40, 0.90, 0.60], dtype=float)
Q = np.diag(1.0 / AXES**2)
C = np.array(
    [
        [-1.00, 0.00, 0.00],
        [-0.99, -0.05, 0.00],
        [1.00, 0.00, 0.00],
        [0.00, 0.00, -1.00],
    ],
    dtype=float,
)

COLORS = {
    "ink": "#18212b",
    "muted": "#667085",
    "blue": "#4f83cc",
    "blue_light": "#d9ebff",
    "green": "#4c9a6a",
    "green2": "#71aa59",
    "orange": "#dd8a22",
    "purple": "#796bc8",
    "projection": "#7b6fc8",
    "panel": "#f7f9fc",
}

SEED_COLORS = {
    1: COLORS["green"],
    2: COLORS["green2"],
    3: COLORS["purple"],
    4: COLORS["orange"],
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
    }
)


def load_points() -> tuple[dict[int, list[np.ndarray]], dict[int, np.ndarray], np.ndarray]:
    trajectories: dict[int, list[np.ndarray]] = {}
    with (HERE / "ellipsoid_selected_points.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(float(row["seed_objective"]))
            x = np.array([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            trajectories.setdefault(seed, []).append(x)

    vertices: dict[int, np.ndarray] = {}
    with (HERE / "ellipsoid_objective_vertices.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            vertices[idx] = np.array([float(row["x1"]), float(row["x2"]), float(row["x3"])])

    matrix = []
    with (HERE / "ellipsoid_correlation_matrix.csv").open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            matrix.append([float(v) for v in row[1:]])

    return trajectories, vertices, np.array(matrix)


def load_highlight_step(seed: int = 4, step: int = 9) -> tuple[dict[str, np.ndarray | float], list[dict[str, np.ndarray | float | int]]]:
    step_row: dict[str, np.ndarray | float] | None = None
    with (HERE / "ellipsoid_step_details.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["seed_objective"])) == seed and int(float(row["step"])) == step:
                step_row = {
                    "seed": seed,
                    "step": step,
                    "current": np.array([float(row["current_x1"]), float(row["current_x2"]), float(row["current_x3"])]),
                    "step_direction": np.array(
                        [
                            float(row["step_direction_x1"]),
                            float(row["step_direction_x2"]),
                            float(row["step_direction_x3"]),
                        ]
                    ),
                    "trial": np.array([float(row["trial_x1"]), float(row["trial_x2"]), float(row["trial_x3"])]),
                    "projected": np.array(
                        [
                            float(row["projected_x1"]),
                            float(row["projected_x2"]),
                            float(row["projected_x3"]),
                        ]
                    ),
                    "trial_g": float(row["trial_g"]),
                    "projection_distance": float(row["projection_distance"]),
                }
                break
    if step_row is None:
        raise ValueError(f"No highlighted step found for seed={seed}, step={step}")

    directions: list[dict[str, np.ndarray | float | int]] = []
    with (HERE / "ellipsoid_projected_directions.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["seed_objective"])) == seed and int(float(row["step"])) == step:
                directions.append(
                    {
                        "objective": int(float(row["objective"])),
                        "weight": float(row["random_weight"]),
                        "descent": np.array(
                            [float(row["descent_x1"]), float(row["descent_x2"]), float(row["descent_x3"])]
                        ),
                        "normal_dot": float(row["normal_dot_descent"]),
                        "projected": np.array(
                            [float(row["projected_x1"]), float(row["projected_x2"]), float(row["projected_x3"])]
                        ),
                        "projected_norm": float(row["projected_norm"]),
                        "unit_projected": np.array(
                            [
                                float(row["unit_projected_x1"]),
                                float(row["unit_projected_x2"]),
                                float(row["unit_projected_x3"]),
                            ]
                        ),
                    }
                )
    return step_row, directions


def ellipsoid_value(x: np.ndarray) -> float:
    return float(x @ Q @ x - 1.0)


def ellipsoid_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * Q @ x


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n <= 1e-12 else v / n


def tangent_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = unit(ellipsoid_grad(x))
    probe = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, probe))) > 0.85:
        probe = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(normal, probe))
    v = unit(np.cross(normal, u))
    return u, v


def tangent_project(vec: np.ndarray, x: np.ndarray) -> np.ndarray:
    normal = ellipsoid_grad(x)
    projected = vec - np.dot(vec, normal) / np.dot(normal, normal) * normal
    return unit(projected)


def ellipsoid_mesh(nu: int = 80, nv: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(-np.pi / 2, np.pi / 2, nv)
    uu, vv = np.meshgrid(u, v)
    x = AXES[0] * np.cos(vv) * np.cos(uu)
    y = AXES[1] * np.cos(vv) * np.sin(uu)
    z = AXES[2] * np.sin(vv)
    return x, y, z


def set_clean_3d_axis(ax) -> None:
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.12, 1.12)
    ax.set_zlim(-0.76, 0.76)
    ax.set_box_aspect((2.8, 1.8, 1.2))
    ax.view_init(elev=22, azim=-38)
    ax.set_axis_off()


def add_custom_axes(ax) -> None:
    origin = np.array([-1.55, -1.00, -0.66])
    axes = [
        (np.array([0.48, 0.00, 0.00]), "$x_1$"),
        (np.array([0.00, 0.38, 0.00]), "$x_2$"),
        (np.array([0.00, 0.00, 0.30]), "$x_3$"),
    ]
    for vec, label in axes:
        ax.quiver(*origin, *vec, color=COLORS["muted"], linewidth=0.9, arrow_length_ratio=0.18)
        endpoint = origin + vec * 1.10
        ax.text(*endpoint, label, color=COLORS["muted"], fontsize=7)


def add_tangent_plane(ax, x: np.ndarray) -> None:
    u, v = tangent_basis(x)
    scale_u, scale_v = 0.34, 0.25
    corners = np.array(
        [
            x - scale_u * u - scale_v * v,
            x + scale_u * u - scale_v * v,
            x + scale_u * u + scale_v * v,
            x - scale_u * u + scale_v * v,
        ]
    )
    plane = Poly3DCollection([corners], facecolors=COLORS["orange"], edgecolors=COLORS["orange"], alpha=0.24, linewidths=0.8)
    ax.add_collection3d(plane)


def add_projected_direction_cone(ax, x: np.ndarray, directions: list[dict[str, np.ndarray | float | int]]) -> None:
    # Draw exactly what the Julia audit file reports: each p_i is the
    # tangent-projected descent direction used to form the random conic step.
    u, v = tangent_basis(x)
    projected = []
    for record in directions:
        idx = int(record["objective"])
        norm = float(record["projected_norm"])
        d = np.array(record["unit_projected"], dtype=float)
        if norm <= 1e-10:
            continue
        angle = math.atan2(float(np.dot(d, v)), float(np.dot(d, u)))
        projected.append((angle, idx, d))

    projected.sort(key=lambda item: item[0])
    endpoints = [x + 0.36 * d for _, _, d in projected]
    triangles = []
    for p, q in zip(endpoints, endpoints[1:] + endpoints[:1]):
        triangles.append(np.array([x, p, q]))
    if triangles:
        poly = Poly3DCollection(triangles, facecolors=COLORS["green"], edgecolors=COLORS["green"], alpha=0.13, linewidths=0.6)
        ax.add_collection3d(poly)

    for _, idx, d in projected:
        ax.quiver(*x, *(0.32 * d), color=SEED_COLORS[idx], linewidth=1.35, arrow_length_ratio=0.18)


def add_highlight_step(ax, step_row: dict[str, np.ndarray | float], directions: list[dict[str, np.ndarray | float | int]]) -> None:
    x_n = np.array(step_row["current"], dtype=float)
    trial = np.array(step_row["trial"], dtype=float)
    x_next = np.array(step_row["projected"], dtype=float)
    delta = np.array(step_row["step_direction"], dtype=float)
    add_tangent_plane(ax, x_n)
    add_projected_direction_cone(ax, x_n, directions)

    # Composite direction Δ before projection.
    ax.quiver(*x_n, *(0.22 * delta), color=COLORS["ink"], linewidth=1.8, arrow_length_ratio=0.16)

    ax.plot(
        [x_n[0], trial[0]],
        [x_n[1], trial[1]],
        [x_n[2], trial[2]],
        color=COLORS["orange"],
        lw=2.0,
        ls=(0, (3, 2)),
        zorder=8,
    )
    ax.plot(
        [trial[0], x_next[0]],
        [trial[1], x_next[1]],
        [trial[2], x_next[2]],
        color=COLORS["projection"],
        lw=2.0,
        zorder=8,
    )
    ax.scatter(*x_n, s=70, color=COLORS["ink"], edgecolor="white", linewidth=0.7, depthshade=False, zorder=9)
    ax.scatter(*trial, s=65, color="white", edgecolor=COLORS["orange"], linewidth=1.2, depthshade=False, zorder=9)
    ax.scatter(*x_next, s=70, color=COLORS["orange"], edgecolor="white", linewidth=0.7, depthshade=False, zorder=9)

    ax.text(*(x_n + np.array([-0.08, 0.02, 0.13])), "$x_n$", color=COLORS["ink"], fontsize=8)
    ax.text(*(x_n + 0.25 * delta + np.array([0.02, 0.02, 0.08])), r"$\Delta$", color=COLORS["ink"], fontsize=9)
    ax.text(*(trial + np.array([0.03, 0.02, 0.10])), r"$x_n+\alpha\Delta$", color=COLORS["orange"], fontsize=8)
    ax.text(*(x_next + np.array([0.06, 0.01, -0.10])), "$x_{n+1}$", color=COLORS["orange"], fontsize=8)
    ax.text(*(x_n + np.array([-0.10, 0.10, 0.35])), "projected objective directions", color=COLORS["green"], fontsize=8)


def plot_3d_panel(
    ax,
    trajectories: dict[int, list[np.ndarray]],
    vertices: dict[int, np.ndarray],
    step_row: dict[str, np.ndarray | float],
    directions: list[dict[str, np.ndarray | float | int]],
) -> None:
    x, y, z = ellipsoid_mesh()
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=COLORS["blue_light"], alpha=0.26, linewidth=0, shade=False)
    ax.plot_wireframe(x, y, z, rstride=5, cstride=7, color=COLORS["blue"], linewidth=0.25, alpha=0.28)

    for seed, points in trajectories.items():
        arr = np.array(points)
        # Keep all selected points visible but faint; this avoids over-emphasizing
        # the local loops near f1/f2 optima.
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color=SEED_COLORS[seed], alpha=0.16, s=10, depthshade=False, zorder=4)

    labels = {
        1: ("$f_1^*$", np.array([-0.14, -0.06, -0.06])),
        2: ("$f_2^*$", np.array([0.05, 0.05, 0.09])),
        3: ("$f_3^*$", np.array([0.04, 0.02, 0.12])),
        4: ("$f_4^*$", np.array([0.03, 0.04, 0.08])),
    }
    for idx, point in vertices.items():
        ax.scatter(*point, color=SEED_COLORS[idx], edgecolor="white", linewidth=0.8, s=70, depthshade=False, zorder=9)
        label, offset = labels[idx]
        ax.text(*(point + offset), label, color=SEED_COLORS[idx], fontsize=9, weight="bold")

    add_highlight_step(ax, step_row, directions)
    set_clean_3d_axis(ax)
    add_custom_axes(ax)
    ax.set_title("One selected-point update on the ellipsoid", loc="left", pad=8, fontsize=10, weight="bold", color=COLORS["ink"])


def plot_objective_panel(ax) -> None:
    ax.axis("off")
    ax.set_facecolor(COLORS["panel"])
    ax.text(0.02, 0.94, "Toy objectives", fontsize=10, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    entries = [
        (COLORS["green"], r"$f_1(x)=-x_1$"),
        (COLORS["green2"], r"$f_2(x)=-0.99x_1-0.05x_2$"),
        (COLORS["purple"], r"$f_3(x)=x_1$"),
        (COLORS["orange"], r"$f_4(x)=-x_3$"),
    ]
    for idx, (color, label) in enumerate(entries):
        y = 0.80 - idx * 0.16
        ax.scatter([0.06], [y], s=70, color=color, transform=ax.transAxes)
        ax.text(0.15, y - 0.02, label, fontsize=9, color=COLORS["ink"], transform=ax.transAxes)
    ax.text(
        0.02,
        0.18,
        r"Feasible set: $(x_1/1.40)^2 + (x_2/0.90)^2 + (x_3/0.60)^2 \leq 1$",
        fontsize=8,
        color=COLORS["muted"],
        transform=ax.transAxes,
        wrap=True,
    )
    ax.text(
        0.02,
        0.06,
        r"Colored arrows in the 3D panel are $p_i$, the projected objective directions. Black arrow is their weighted sum $\Delta$.",
        fontsize=7.5,
        color=COLORS["muted"],
        transform=ax.transAxes,
        wrap=True,
    )


def plot_matrix_panel(ax, matrix: np.ndarray) -> None:
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(4), [f"$f_{i}$" for i in range(1, 5)])
    ax.set_yticks(range(4), [f"$f_{i}$" for i in range(1, 5)])
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(4):
        for j in range(4):
            value = matrix[i, j]
            text_color = "white" if value >= 0.65 else COLORS["ink"]
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color, weight=weight)
    ax.set_title("Projected objective correlation matrix", loc="left", fontsize=10, weight="bold", color=COLORS["ink"], pad=8)
    ax.text(0.00, -0.28, r"$G_1=\{f_1,f_2\}$", transform=ax.transAxes, fontsize=9, color=COLORS["green"], weight="bold")
    ax.text(0.00, -0.43, r"$G_2=\{f_3\}$", transform=ax.transAxes, fontsize=9, color=COLORS["purple"], weight="bold")
    ax.text(0.00, -0.58, r"$G_3=\{f_4\}$", transform=ax.transAxes, fontsize=9, color=COLORS["orange"], weight="bold")
    return im


def plot_projection_inset(fig) -> None:
    ax = fig.add_axes([0.070, 0.620, 0.215, 0.175])
    ax.set_zorder(20)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#c7d1dc")
        spine.set_linewidth(0.8)
    ax.set_xlim(-0.12, 1.08)
    ax.set_ylim(-0.30, 1.00)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Projection onto a linearized constraint", loc="left", fontsize=7.5, weight="bold", color=COLORS["ink"], pad=4)

    ax.plot([-0.05, 1.02], [0.0, 0.0], color=COLORS["muted"], lw=1.0)
    ax.text(0.53, -0.17, "linearized constraint", ha="center", fontsize=6.5, color=COLORS["muted"])

    origin = (0.18, 0.0)
    raw_end = (0.72, 0.72)
    proj_end = (0.72, 0.0)
    normal_end = (0.18, 0.72)

    ax.scatter([origin[0]], [origin[1]], s=18, color=COLORS["ink"], zorder=5)
    ax.text(origin[0] - 0.06, origin[1] - 0.12, r"$x_n$", fontsize=7, color=COLORS["ink"])

    ax.add_patch(
        FancyArrowPatch(origin, raw_end, arrowstyle="-|>", mutation_scale=9, lw=1.2, color=COLORS["green"], shrinkA=0, shrinkB=0)
    )
    ax.text(raw_end[0] + 0.02, raw_end[1] - 0.02, r"$-\nabla f_i$", fontsize=7, color=COLORS["green"])

    ax.add_patch(
        FancyArrowPatch(origin, proj_end, arrowstyle="-|>", mutation_scale=9, lw=1.4, color=COLORS["orange"], shrinkA=0, shrinkB=0)
    )
    ax.text(0.44, 0.06, r"$p_i$", fontsize=7, color=COLORS["orange"], weight="bold")

    ax.add_patch(
        FancyArrowPatch(origin, normal_end, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=COLORS["muted"], linestyle="--", shrinkA=0, shrinkB=0)
    )
    ax.text(normal_end[0] - 0.08, normal_end[1] + 0.02, r"$n$", fontsize=7, color=COLORS["muted"])

    ax.plot([raw_end[0], proj_end[0]], [raw_end[1], proj_end[1]], color=COLORS["projection"], lw=1.0, ls=(0, (2, 2)))
    ax.text(raw_end[0] + 0.03, 0.31, "removed\nnormal\npart", fontsize=5.8, color=COLORS["projection"], va="center")


def main() -> None:
    trajectories, vertices, matrix = load_points()
    step_row, directions = load_highlight_step(seed=4, step=9)

    fig = plt.figure(figsize=(10.8, 7.1), dpi=300)
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[2.35, 1.0],
        height_ratios=[0.95, 1.05],
        left=0.045,
        right=0.965,
        top=0.88,
        bottom=0.08,
        wspace=0.11,
        hspace=0.30,
    )

    fig.suptitle(
        "Simple ellipsoid demonstration of nonlinear objective reduction",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=15,
        weight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.045,
        0.925,
        "A single audited update shows how projected objective directions form the step; the final matrix shows the reducible objective structure.",
        ha="left",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    plot_3d_panel(ax3d, trajectories, vertices, step_row, directions)

    ax_obj = fig.add_subplot(gs[0, 1])
    plot_objective_panel(ax_obj)

    ax_mat = fig.add_subplot(gs[1, 1])
    im = plot_matrix_panel(ax_mat, matrix)
    plot_projection_inset(fig)

    cax = fig.add_axes([0.94, 0.16, 0.012, 0.24])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("projected correlation", fontsize=7)
    cb.ax.tick_params(labelsize=7, length=2)

    for out_stem in [
        HERE / "ellipsoid_algorithm_demo_3d_matplotlib",
        HERE / "ellipsoid_algorithm_demo_3d_matrix_labels_fixed",
        HERE / "ellipsoid_algorithm_demo_3d_with_projection_inset_v2",
    ]:
        fig.savefig(out_stem.with_suffix(".png"), dpi=600)
        fig.savefig(out_stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(out_stem.with_suffix(".svg"))
        fig.savefig(out_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
