"""Draw the selected-point workflow for final-size manuscript use.

Figure contract
---------------
Core conclusion: one residual-selected surface modifies the proposal direction,
and a separate feasibility projection repairs the trial point.
Archetype: schematic-led vertical sequence.
Backend: Python and Matplotlib only.
Output: editable PDF and SVG plus high-resolution PNG and TIFF.
Reviewer risk: a wide three-panel strip becomes illegible at journal width.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, Ellipse

A = 2.6
B = 1.3
XLIM = (-3.05, 3.30)
YLIM = (-1.60, 1.72)
OBJECTIVE_NAMES = ["f1", "f2", "f3", "f4"]
SEED_ANGLES = np.array([6.0, -6.0, 180.0, 90.0])
SELECTED_ANGLES = {
    0: np.array([6.0, -6.0, 180.0, 90.0]),
    1: np.array([12.0, -11.0, 158.0, 82.0]),
    2: np.array([18.0, -16.0, 140.0, 74.0]),
}

COLORS = {
    "ink": "#18212b",
    "muted": "#667085",
    "grid": "#e9eef5",
    "axis": "#cfd8e3",
    "ellipse": "#4f83cc",
    "ellipse_fill": "#e8f2ff",
    "f1": "#2f8f5b",
    "f2": "#21867a",
    "f3": "#7568c9",
    "f4": "#d19a24",
    "black": "#111827",
}
OBJECTIVE_COLORS = [COLORS["f1"], COLORS["f2"], COLORS["f3"], COLORS["f4"]]

CAPTION = (
    "Schematic toy problem illustrating selected fixed-point generation for nonlinear objective "
    "reduction. The feasible set is an ellipse, and each panel shows the current selected "
    "points for four seed objectives. Dashed lines are local first-order linearizations of "
    "the nonlinear constraint at selected points; pale dashed lines denote linearizations "
    "generated in previous steps. The inset shows how objective descent directions are "
    "projected onto the local tangent constraint in a zoomed local view that preserves "
    "the original orientation."
)


def ellipse_point(theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    return np.array([A * math.cos(theta), B * math.sin(theta)], dtype=float)


def g(x: np.ndarray) -> float:
    return float((x[0] / A) ** 2 + (x[1] / B) ** 2 - 1.0)


def grad_g(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * x[0] / (A**2), 2.0 * x[1] / (B**2)], dtype=float)


def unit(v: np.ndarray, atol: float = 1.0e-12) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    return np.zeros_like(v, dtype=float) if nrm <= atol else v / nrm


def objective_coefficients() -> np.ndarray:
    rows = []
    for theta_deg in SEED_ANGLES:
        theta = math.radians(theta_deg)
        rows.append(-np.array([math.cos(theta) / A, math.sin(theta) / B], dtype=float))
    return np.vstack(rows)


C = objective_coefficients()
DESCENT_DIRECTIONS = -C


def tangent_line_points(x0: np.ndarray, xlim: tuple[float, float] = XLIM) -> tuple[np.ndarray, np.ndarray]:
    normal = grad_g(x0)
    offset = float(np.dot(normal, x0))
    if abs(normal[1]) > 1.0e-12:
        xs = np.array(xlim, dtype=float)
        ys = (offset - normal[0] * xs) / normal[1]
        return np.array([xs[0], ys[0]]), np.array([xs[1], ys[1]])
    x_const = offset / normal[0]
    return np.array([x_const, YLIM[0]], dtype=float), np.array([x_const, YLIM[1]], dtype=float)


def project_to_tangent(v: np.ndarray, x0: np.ndarray) -> np.ndarray:
    n = unit(grad_g(x0))
    return np.asarray(v, dtype=float) - float(np.dot(v, n)) * n


def project_to_ellipse(y: np.ndarray) -> np.ndarray:
    if g(y) <= 1.0e-12:
        return y.copy()
    q = np.array([1.0 / (A**2), 1.0 / (B**2)], dtype=float)

    def residual(lam: float) -> float:
        return float(np.sum(q * y**2 / (1.0 + lam * q) ** 2) - 1.0)

    lo, hi = 0.0, 1.0
    while residual(hi) > 0.0:
        hi *= 2.0
    for _ in range(140):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return y / (1.0 + lam * q)


def theta_path(start: float, end: float, n: int = 80) -> np.ndarray:
    delta = end - start
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return start + np.linspace(0.0, delta, n)


def draw_ellipse(ax) -> None:
    ax.add_patch(
        Ellipse(
            (0.0, 0.0),
            width=2.0 * A,
            height=2.0 * B,
            facecolor=COLORS["ellipse_fill"],
            edgecolor=COLORS["ellipse"],
            linewidth=2.0,
            zorder=1,
        )
    )
    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=COLORS["axis"], linewidth=0.8, zorder=0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(color=COLORS["grid"], linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_tangent(ax, x0: np.ndarray, color: str, *, alpha: float, linewidth: float) -> None:
    p0, p1 = tangent_line_points(x0)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        linestyle=(0, (5, 3.5)),
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=2,
    )


def draw_boundary_arrow(ax, theta0: float, theta1: float, color: str) -> None:
    samples = theta_path(theta0, theta1, n=70)
    pts = np.vstack([ellipse_point(theta) for theta in samples])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.45, zorder=5)
    tail, head = pts[-8], pts[-1]
    ax.annotate(
        "",
        xy=head,
        xytext=tail,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, alpha=0.55, shrinkA=0, shrinkB=0),
        zorder=6,
    )


def draw_descent_arrow(ax, x0: np.ndarray, idx: int, color: str) -> None:
    direction = unit(DESCENT_DIRECTIONS[idx])
    length = 0.26 if idx == 3 else 0.34
    ax.annotate(
        "",
        xy=x0 + length * direction,
        xytext=x0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=0, shrinkB=0),
        zorder=9,
    )
    label_distance = 0.38 if idx == 3 else length + 0.16
    label_xy = x0 + label_distance * direction
    extra = {
        0: (0.08, 0.04),
        1: (0.08, -0.06),
        2: (-0.05, 0.04),
        3: (0.04, 0.04),
    }[idx]
    ax.text(
        label_xy[0] + extra[0],
        label_xy[1] + extra[1],
        rf"$-\nabla f_{idx + 1}$",
        color=color,
        fontsize=6.7,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.62, pad=0.4),
        zorder=11,
    )


def label_offset(idx: int) -> tuple[float, float]:
    return {
        0: (0.15, 0.25),
        1: (0.16, -0.32),
        2: (-0.62, 0.14),
        3: (0.11, 0.15),
    }[idx]


def draw_panel(ax, step: int) -> None:
    draw_ellipse(ax)
    ax.set_title(f"({chr(ord('a') + step)}) Step {step}", loc="left", fontsize=11.5, fontweight="bold", color=COLORS["ink"])

    for idx, color in enumerate(OBJECTIVE_COLORS):
        for previous_step in range(step):
            previous_point = ellipse_point(float(SELECTED_ANGLES[previous_step][idx]))
            draw_tangent(ax, previous_point, color, alpha=0.38, linewidth=1.15)
        current_angle = float(SELECTED_ANGLES[step][idx])
        current_point = ellipse_point(current_angle)
        draw_tangent(ax, current_point, color, alpha=0.92, linewidth=1.9)

        for segment_step in range(1, step + 1):
            draw_boundary_arrow(
                ax,
                float(SELECTED_ANGLES[segment_step - 1][idx]),
                float(SELECTED_ANGLES[segment_step][idx]),
                color,
            )

        ax.scatter(
            [current_point[0]],
            [current_point[1]],
            s=44,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=8,
        )
        dx, dy = label_offset(idx)
        ax.text(
            current_point[0] + dx,
            current_point[1] + dy,
            f"{OBJECTIVE_NAMES[idx]} SP{step}",
            color=color,
            fontsize=7.1,
            fontweight="bold",
            zorder=10,
        )
        draw_descent_arrow(ax, current_point, idx, color)

def draw_arrow(ax, start: np.ndarray, vector: np.ndarray, color: str, label: str | None = None, *, lw: float = 1.2, alpha: float = 1.0, scale: float = 1.0) -> None:
    end = start + scale * vector
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    if label:
        ax.text(end[0], end[1], label, color=color, fontsize=6.2, ha="left", va="bottom", zorder=9)


def draw_projection_inset(
    ax_parent,
    demo_point: np.ndarray,
    objective_gradients: np.ndarray,
    colors: list[str],
) -> None:
    inset = ax_parent.inset_axes([0.13, 0.035, 0.72, 0.54], zorder=20)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.97)
    for spine in inset.spines.values():
        spine.set_color("#c7d0dd")
        spine.set_linewidth(0.75)

    callout_radius = 0.26
    source_marker = Circle(
        demo_point,
        radius=callout_radius,
        facecolor=(1.0, 1.0, 1.0, 0.16),
        edgecolor="#374151",
        linewidth=1.3,
        linestyle=(0, (3.2, 2.2)),
        zorder=16,
    )
    ax_parent.add_patch(source_marker)
    ax_parent.text(
        demo_point[0] + 0.08,
        demo_point[1] - 0.36,
        "zoom",
        fontsize=7.0,
        color="#374151",
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.5),
        zorder=17,
    )
    for source_xy, inset_xy in [
        (demo_point + np.array([0.18, -0.18]), (0.03, 0.95)),
        (demo_point + np.array([0.25, 0.05]), (0.03, 0.75)),
    ]:
        connector = ConnectionPatch(
            xyA=source_xy,
            xyB=inset_xy,
            coordsA="data",
            coordsB="axes fraction",
            axesA=ax_parent,
            axesB=inset,
            color="#6b7280",
            linewidth=0.8,
            alpha=0.62,
            zorder=19,
            clip_on=False,
        )
        ax_parent.add_artist(connector)

    n = unit(grad_g(demo_point))
    t = np.array([-n[1], n[0]], dtype=float)

    def to_inset(point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=float) - demo_point

    demo_angle = float(SELECTED_ANGLES[1][2])
    theta_vals = np.linspace(demo_angle - 18.0, demo_angle + 18.0, 160)
    curve = np.vstack([to_inset(ellipse_point(theta)) for theta in theta_vals])
    inset.plot(curve[:, 0], curve[:, 1], color=COLORS["ellipse"], lw=1.35, alpha=0.76, zorder=1)
    inset.fill_between(curve[:, 0], curve[:, 1], -0.80, color=COLORS["ellipse_fill"], alpha=0.42, zorder=0)

    tangent_span = 1.55
    tangent_pts = np.vstack([-tangent_span * t, tangent_span * t])
    inset.plot(
        tangent_pts[:, 0],
        tangent_pts[:, 1],
        color="#374151",
        linestyle=(0, (4, 2.8)),
        lw=1.65,
        zorder=2,
    )
    inset.text(
        0.60,
        0.83,
        "tangent constraint",
        fontsize=7.1,
        color="#374151",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.4),
        zorder=12,
    )
    inset_colors = ["#2ca25f", "#1f9e89", "#756bb1", "#d99a21"]
    descent_vectors = []
    for gradient in objective_gradients:
        d = unit(-np.asarray(gradient, dtype=float))
        descent_vectors.append(d)
    descent_vectors = np.vstack(descent_vectors)
    direction_length = 0.88
    d_label_offsets = [(0.12, -0.035), (0.12, -0.060), (-0.13, 0.070), (-0.11, 0.080)]
    p_label_offsets = [(0.050, 0.105), (0.050, -0.105), (-0.180, 0.080), (-0.220, 0.100)]
    tips: list[tuple[np.ndarray, np.ndarray, str]] = []

    for idx, (direction, color) in enumerate(zip(descent_vectors, inset_colors)):
        tip = direction_length * direction
        projected = direction - float(np.dot(direction, n)) * n
        proj_tip = direction_length * projected
        tips.append((tip, proj_tip, color))

        inset.annotate(
            "",
            xy=tip,
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.20,
                alpha=0.70,
                mutation_scale=9,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )
        inset.plot(
            [tip[0], proj_tip[0]],
            [tip[1], proj_tip[1]],
            color="#8a94a6",
            linestyle=":",
            lw=0.72,
            alpha=0.45,
            zorder=4,
        )
        dx, dy = d_label_offsets[idx]
        inset.text(
            tip[0] + dx,
            tip[1] + dy,
            rf"$d_{idx + 1}$",
            color=color,
            fontsize=7.3,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.35),
            zorder=14,
        )

    for idx in [3, 0, 1, 2]:
        _, proj_tip, color = tips[idx]
        lw = {0: 1.08, 1: 1.18, 2: 1.22, 3: 1.40}[idx]
        mutation_scale = {0: 8.0, 1: 8.5, 2: 8.5, 3: 10.0}[idx]
        inset.annotate(
            "",
            xy=proj_tip,
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=0.96,
                mutation_scale=mutation_scale,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=7,
        )

    inset.scatter([0.0], [0.0], s=30, color=COLORS["black"], edgecolor="white", linewidth=0.55, zorder=12)
    inset.text(-0.115, 0.080, r"$x_s$", fontsize=7.7, color=COLORS["black"], ha="right", va="bottom", zorder=13)

    for idx, (_, proj_tip, color) in enumerate(tips):
        p_dx, p_dy = p_label_offsets[idx]
        p_ha = "left" if p_dx >= 0 else "right"
        inset.text(
            proj_tip[0] + p_dx,
            proj_tip[1] + p_dy,
            rf"$p_{idx + 1}$",
            color=color,
            fontsize=7.3,
            ha=p_ha,
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.35),
            zorder=14,
        )

    inset.text(
        0.30,
        -0.66,
        r"$p_i=d_i-(d_i\cdot n)n$",
        fontsize=7.2,
        color=COLORS["black"],
        ha="left",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5),
        zorder=15,
    )
    inset.set_xlim(-1.35, 1.35)
    inset.set_ylim(-0.75, 1.05)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Local tangent projection", fontsize=8.4, color=COLORS["ink"], pad=2)


def save_caption() -> None:
    (HERE / "toy_selected_point_process_revised8_caption.txt").write_text(CAPTION + "\n")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.9), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.985, bottom=0.025, hspace=0.12)

    for step, ax in enumerate(axes):
        draw_panel(ax, step)
    draw_projection_inset(axes[1], ellipse_point(float(SELECTED_ANGLES[1][2])), C, OBJECTIVE_COLORS)

    out_base = HERE / "toy_selected_point_process_vertical"
    save_options = {"bbox_inches": "tight", "pad_inches": 0.10}
    fig.savefig(out_base.with_suffix(".pdf"), **save_options)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, **save_options)
    fig.savefig(out_base.with_suffix(".svg"), **save_options)
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, **save_options)
    plt.close(fig)
    save_caption()


if __name__ == "__main__":
    main()
