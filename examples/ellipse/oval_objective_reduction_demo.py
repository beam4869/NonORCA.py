from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize_scalar


HERE = Path(__file__).resolve().parent
ORCA_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src"
if ORCA_PACKAGE_ROOT.exists() and str(ORCA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCA_PACKAGE_ROOT))

from orca.config import NonlinearORCAConfig
from orca.nonlinear import NonlinearORCAProblem, nonlinear_orca


AXES = np.array([1.35, 0.82], dtype=float)
OBJ_LABELS = ["f1", "f2", "f3", "f4"]
OBJ_DESCRIPTIONS = [
    "f1(x) = -x1",
    "f2(x) = -0.94*x1 - 0.10*x2 + 0.05*x2^2",
    "f3(x) =  x1",
    "f4(x) = -x2 + 0.18*x1^2",
]


@dataclass(frozen=True)
class Objective:
    label: str
    description: str
    value: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]


OBJECTIVES = [
    Objective(
        "f1",
        OBJ_DESCRIPTIONS[0],
        lambda x: -x[0],
        lambda x: np.array([-1.0, 0.0]),
    ),
    Objective(
        "f2",
        OBJ_DESCRIPTIONS[1],
        lambda x: -0.94 * x[0] - 0.10 * x[1] + 0.05 * x[1] ** 2,
        lambda x: np.array([-0.94, -0.10 + 0.10 * x[1]]),
    ),
    Objective(
        "f3",
        OBJ_DESCRIPTIONS[2],
        lambda x: x[0],
        lambda x: np.array([1.0, 0.0]),
    ),
    Objective(
        "f4",
        OBJ_DESCRIPTIONS[3],
        lambda x: -x[1] + 0.18 * x[0] ** 2,
        lambda x: np.array([0.36 * x[0], -1.0]),
    ),
]


def unit(v: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if nrm <= atol:
        return np.zeros_like(v, dtype=float)
    return v / nrm


def ellipse_point(theta: float) -> np.ndarray:
    return AXES * np.array([math.cos(theta), math.sin(theta)])


def ellipse_value(x: np.ndarray) -> float:
    return float(np.sum((x / AXES) ** 2) - 1.0)


def ellipse_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * x / (AXES**2)


def tangent_projection(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = ellipse_grad(x)
    denom = float(np.dot(n, n))
    if denom <= 1e-12:
        return v.copy()
    return v - float(np.dot(v, n)) / denom * n


def project_to_ellipse(y: np.ndarray) -> np.ndarray:
    if ellipse_value(y) <= 1e-12:
        return y.copy()
    q = 1.0 / (AXES**2)

    def residual(lam: float) -> float:
        return float(np.sum(q * y**2 / (1.0 + lam * q) ** 2) - 1.0)

    lo, hi = 0.0, 1.0
    while residual(hi) > 0.0:
        hi *= 2.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return y / (1.0 + lam * q)


def objective_values(x: np.ndarray) -> list[float]:
    return [obj.value(x) for obj in OBJECTIVES]


def solve_boundary_minimum(obj: Objective) -> tuple[float, np.ndarray]:
    grid = np.linspace(0.0, 2.0 * math.pi, 2401, endpoint=False)
    values = np.array([obj.value(ellipse_point(theta)) for theta in grid])
    best_theta = float(grid[int(np.argmin(values))])
    step = 2.0 * math.pi / len(grid)

    def wrapped_value(theta: float) -> float:
        return obj.value(ellipse_point(theta % (2.0 * math.pi)))

    res = minimize_scalar(
        wrapped_value,
        bounds=(best_theta - 4.0 * step, best_theta + 4.0 * step),
        method="bounded",
        options={"xatol": 1e-13},
    )
    theta = float(res.x % (2.0 * math.pi))
    return theta, ellipse_point(theta)


class OvalORCAProblem(NonlinearORCAProblem):
    """Two-variable oval test problem using the nonlinear ORCA package API."""

    def __init__(self) -> None:
        self.vertex_solutions = [solve_boundary_minimum(obj) for obj in OBJECTIVES]

    def num_variables(self) -> int:
        return 2

    def num_objectives(self) -> int:
        return len(OBJECTIVES)

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(objective_values(x), dtype=float)

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([ellipse_value(x)], dtype=float)

    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([obj.gradient(x) for obj in OBJECTIVES])

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([ellipse_grad(x)], dtype=float)

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return project_to_ellipse(np.asarray(y, dtype=float))

    def initial_points(self) -> list[np.ndarray]:
        return [point.copy() for _, point in self.vertex_solutions]

    def initial_thetas(self) -> list[float]:
        return [theta for theta, _ in self.vertex_solutions]


def selected_point_direction(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    projected = []
    for obj in OBJECTIVES:
        descent = unit(-obj.gradient(x))
        projected.append(tangent_projection(descent, x))
    weights = rng.random(len(OBJECTIVES))
    direction = sum(weights[i] * projected[i] for i in range(len(OBJECTIVES))) / float(np.sum(weights))
    if np.linalg.norm(direction) <= 1e-10:
        direction = projected[int(np.argmax([np.linalg.norm(p) for p in projected]))]
    return direction, weights, projected


def generate_selected_points(
    vertices: list[np.ndarray],
    *,
    step_scale: float = 0.24,
    steps_per_seed: int = 16,
    seed: int = 14,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[list[float]], list[list[float]]]:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    point_meta: list[tuple[int, int]] = []
    step_rows: list[list[float]] = []
    direction_rows: list[list[float]] = []

    for seed_idx, vertex in enumerate(vertices, start=1):
        x = vertex.copy()
        points.append(x.copy())
        point_meta.append((seed_idx, 0))
        for step in range(1, steps_per_seed + 1):
            direction, weights, projected = selected_point_direction(x, rng)
            trial = x + step_scale * direction
            next_x = project_to_ellipse(trial)
            step_rows.append(
                [
                    seed_idx,
                    step,
                    x[0],
                    x[1],
                    ellipse_value(x),
                    direction[0],
                    direction[1],
                    trial[0],
                    trial[1],
                    ellipse_value(trial),
                    next_x[0],
                    next_x[1],
                    ellipse_value(next_x),
                    float(np.linalg.norm(next_x - trial)),
                ]
            )

            for obj_idx, obj in enumerate(OBJECTIVES, start=1):
                descent = unit(-obj.gradient(x))
                tangent = projected[obj_idx - 1]
                direction_rows.append(
                    [
                        seed_idx,
                        step,
                        obj_idx,
                        weights[obj_idx - 1],
                        descent[0],
                        descent[1],
                        float(np.dot(descent, ellipse_grad(x))),
                        tangent[0],
                        tangent[1],
                        float(np.linalg.norm(tangent)),
                        unit(tangent)[0],
                        unit(tangent)[1],
                    ]
                )

            x = next_x
            points.append(x.copy())
            point_meta.append((seed_idx, step))

    return points, point_meta, step_rows, direction_rows


def logistic_weight(s: float, *, posmin: float = 0.9, beta: float = 100.0) -> float:
    return 1.0 - posmin * (1.0 / (1.0 + math.exp(-beta * s)))


def pair_strength(obj_i: Objective, obj_j: Objective, x: np.ndarray) -> tuple[float, float, float, float]:
    vi = unit(-obj_i.gradient(x))
    vj = unit(-obj_j.gradient(x))
    normal = ellipse_grad(x)
    pi = tangent_projection(vi, x)
    pj = tangent_projection(vj, x)
    si = float(np.dot(vi, normal))
    sj = float(np.dot(vj, normal))

    use_pair = True
    if abs(si) > 1e-12 and abs(sj) > 1e-12:
        use_pair = si > 0.0 and sj > 0.0

    if not use_pair:
        return 0.0, 0.0, si, sj
    return float(np.dot(unit(pi), unit(pj))), 1.0, si, sj


def objective_correlation(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = len(OBJECTIVES)
    corr = np.eye(m)
    raw_alignment = np.eye(m)
    total_weight = np.eye(m)
    strength_weight = np.eye(m)

    for i in range(m):
        for j in range(i + 1, m):
            weighted_sum = 0.0
            weight_total = 0.0
            raw_sum = 0.0
            strength_sum = 0.0
            for x in points:
                s, active, _, _ = pair_strength(OBJECTIVES[i], OBJECTIVES[j], x)
                w = active * logistic_weight(s)
                weighted_sum += w * s
                strength_sum += w * s
                weight_total += w
                raw_sum += s
            score = 0.0 if weight_total <= 1e-12 else 0.5 * (1.0 + weighted_sum / weight_total)
            corr[i, j] = corr[j, i] = score
            raw_alignment[i, j] = raw_alignment[j, i] = raw_sum / len(points)
            total_weight[i, j] = total_weight[j, i] = weight_total
            strength_weight[i, j] = strength_weight[j, i] = strength_sum

    return corr, raw_alignment, total_weight, strength_weight


def unprojected_objective_angle_matrix(points: list[np.ndarray]) -> np.ndarray:
    m = len(OBJECTIVES)
    matrix = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            dots = []
            for x in points:
                vi = unit(-OBJECTIVES[i].gradient(x))
                vj = unit(-OBJECTIVES[j].gradient(x))
                dots.append(float(np.dot(vi, vj)))
            score = 0.5 * (1.0 + float(np.mean(dots)))
            matrix[i, j] = matrix[j, i] = score
    return matrix


def group_sets_from_labels(labels: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for objective_idx, group_label in enumerate(np.asarray(labels, dtype=int), start=1):
        groups.setdefault(int(group_label), []).append(objective_idx)
    return [groups[label] for label in sorted(groups)]


def point_metadata_from_package_order(points: np.ndarray, config: NonlinearORCAConfig) -> list[tuple[int, int]]:
    per_seed = config.num_points_per_seed + (1 if config.include_seed_points else 0)
    meta: list[tuple[int, int]] = []
    for idx in range(len(points)):
        seed_idx = idx // per_seed + 1
        within_seed = idx % per_seed
        step = within_seed if config.include_seed_points else within_seed + 1
        meta.append((seed_idx, step))
    return meta


def package_step_rows(points: np.ndarray, point_meta: list[tuple[int, int]]) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    previous_by_seed: dict[int, np.ndarray] = {}
    for x, (seed_idx, step) in zip(points, point_meta):
        previous = previous_by_seed.get(seed_idx, x)
        delta = x - previous
        rows.append(
            [
                seed_idx,
                step,
                previous[0],
                previous[1],
                ellipse_value(previous),
                delta[0],
                delta[1],
                x[0],
                x[1],
                ellipse_value(x),
                x[0],
                x[1],
                ellipse_value(x),
                0.0,
            ]
        )
        previous_by_seed[seed_idx] = x
    return rows


def package_direction_rows(
    points: np.ndarray,
    point_meta: list[tuple[int, int]],
    projected_directions: np.ndarray | None,
) -> list[list[float | int | str]]:
    rows: list[list[float | int | str]] = []
    if projected_directions is None:
        return rows
    for point_idx, (x, (seed_idx, step)) in enumerate(zip(points, point_meta)):
        normal = ellipse_grad(x)
        for obj_idx, obj in enumerate(OBJECTIVES, start=1):
            descent = unit(-obj.gradient(x))
            package_dir = projected_directions[point_idx, 0, obj_idx - 1, :]
            rows.append(
                [
                    seed_idx,
                    step,
                    obj_idx,
                    "package",
                    descent[0],
                    descent[1],
                    float(np.dot(descent, normal)),
                    package_dir[0],
                    package_dir[1],
                    float(np.linalg.norm(package_dir)),
                    unit(package_dir)[0],
                    unit(package_dir)[1],
                ]
            )
    return rows


def raw_alignment_from_interactions(strengths: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    m = strengths.shape[0]
    raw = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            if valid_mask is None:
                values = strengths[i, j, :, :].reshape(-1)
            else:
                values = strengths[i, j, :, :][valid_mask[i, j, :, :]]
            raw[i, j] = raw[j, i] = 0.0 if len(values) == 0 else float(np.mean(values))
    return raw


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["objective", *OBJ_LABELS])
        for label, row in zip(OBJ_LABELS, matrix):
            writer.writerow([label, *[f"{value:.10f}" for value in row]])


def write_outputs(
    vertices: list[np.ndarray],
    vertex_thetas: list[float],
    points: list[np.ndarray],
    point_meta: list[tuple[int, int]],
    step_rows: list[list[float]],
    direction_rows: list[list[float]],
    corr: np.ndarray,
    raw_alignment: np.ndarray,
    total_weight: np.ndarray,
    strength_weight: np.ndarray,
    unprojected: np.ndarray,
    groups: list[list[int]],
) -> None:
    with (HERE / "oval_objective_vertices.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["objective", "description", "theta_rad", "x1", "x2", *OBJ_LABELS])
        for obj, theta, x in zip(OBJECTIVES, vertex_thetas, vertices):
            writer.writerow([obj.label, obj.description, f"{theta:.12f}", f"{x[0]:.12f}", f"{x[1]:.12f}", *objective_values(x)])

    with (HERE / "oval_selected_points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "seed_objective", "step", "x1", "x2", "g(x)"])
        for idx, (x, meta) in enumerate(zip(points, point_meta), start=1):
            writer.writerow([idx, meta[0], meta[1], f"{x[0]:.12f}", f"{x[1]:.12f}", f"{ellipse_value(x):.12e}"])

    with (HERE / "oval_step_details.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed_objective",
                "step",
                "current_x1",
                "current_x2",
                "current_g",
                "step_direction_x1",
                "step_direction_x2",
                "trial_x1",
                "trial_x2",
                "trial_g",
                "projected_x1",
                "projected_x2",
                "projected_g",
                "projection_distance",
            ]
        )
        writer.writerows(step_rows)

    with (HERE / "oval_projected_directions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed_objective",
                "step",
                "objective",
                "direction_source",
                "descent_x1",
                "descent_x2",
                "normal_dot_descent",
                "projected_x1",
                "projected_x2",
                "projected_norm",
                "unit_projected_x1",
                "unit_projected_x2",
            ]
        )
        writer.writerows(direction_rows)

    write_matrix(HERE / "oval_correlation_matrix.csv", corr)
    write_matrix(HERE / "oval_raw_alignment_matrix.csv", raw_alignment)
    write_matrix(HERE / "oval_total_weight_matrix.csv", total_weight)
    write_matrix(HERE / "oval_strength_weight_matrix.csv", strength_weight)
    write_matrix(HERE / "oval_unprojected_objective_matrix.csv", unprojected)

    with (HERE / "oval_groups.txt").open("w") as f:
        f.write("Objective grouping returned by orca.nonlinear.nonlinear_orca\n")
        f.write("grouping_method = average_linkage\n")
        f.write("requested_groups = 3\n")
        for idx, group in enumerate(groups, start=1):
            labels = ", ".join(f"f{i}" for i in group)
            f.write(f"G{idx} = {{{labels}}}\n")

    with (HERE / "oval_case_study_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "axes", "objectives", "selected_points", "grouping_method", "group_sets"])
        writer.writerow(
            [
                "2D oval with four objectives",
                f"x1/{AXES[0]} and x2/{AXES[1]}",
                " | ".join(OBJ_DESCRIPTIONS),
                len(points),
                "average_linkage",
                str(groups),
            ]
        )


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(44, True),
    "subtitle": font(26),
    "label": font(24, True),
    "body": font(22),
    "small": font(19),
    "tiny": font(16),
}

COLORS = {
    "ink": "#18212b",
    "muted": "#667085",
    "grid": "#d5dde8",
    "ellipse_fill": "#e9f3ff",
    "ellipse": "#4f83cc",
    "green": "#4c9a6a",
    "yellow": "#c59a2f",
    "purple": "#796bc8",
    "red": "#c85b5b",
    "panel": "#f7f9fc",
    "white": "#ffffff",
}

SEED_COLORS = {
    1: COLORS["green"],
    2: "#71aa59",
    3: COLORS["purple"],
    4: COLORS["yellow"],
}


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 5) -> None:
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=rgb(color), width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head = 18
    left = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    right = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, left, right], fill=rgb(color))


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    width: int = 3,
    dash: int = 16,
    gap: int = 10,
) -> None:
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy)
    if length <= 1e-12:
        return
    ux, uy = (ex - sx) / length, (ey - sy) / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        draw.line(
            [(sx + ux * pos, sy + uy * pos), (sx + ux * end_pos, sy + uy * end_pos)],
            fill=rgb(color),
            width=width,
        )
        pos += dash + gap


def normal_line_segment(
    point: np.ndarray,
    normal: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    c = float(np.dot(normal, point))
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    candidates: list[np.ndarray] = []
    if abs(normal[1]) > 1e-12:
        for x in (xmin, xmax):
            y = (c - normal[0] * x) / normal[1]
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append(np.array([x, y], dtype=float))
    if abs(normal[0]) > 1e-12:
        for y in (ymin, ymax):
            x = (c - normal[1] * y) / normal[0]
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append(np.array([x, y], dtype=float))

    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.linalg.norm(candidate - existing) < 1e-8 for existing in unique):
            unique.append(candidate)
    if len(unique) < 2:
        return None

    best_pair = (unique[0], unique[1])
    best_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = float(np.linalg.norm(unique[i] - unique[j]))
            if dist > best_dist:
                best_pair = (unique[i], unique[j])
                best_dist = dist
    return best_pair


def tangent_line_segment(
    point: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    return normal_line_segment(point, ellipse_grad(point), x_bounds, y_bounds)


def make_step_figure(points: list[np.ndarray], point_meta: list[tuple[int, int]]) -> None:
    by_seed: dict[int, list[np.ndarray]] = {}
    for x, meta in zip(points, point_meta):
        by_seed.setdefault(meta[0], []).append(x)

    panels: list[tuple[str, int | None]] = [("Initial oval", None)]
    panels.extend((f"Step {step}", step) for step in range(5))

    width, height = 2700, 1800
    image = Image.new("RGB", (width, height), rgb("#ffffff"))
    draw = ImageDraw.Draw(image)

    draw.text((90, 58), "Four seed-objective selected-point processes", fill=rgb(COLORS["ink"]), font=FONTS["title"])
    draw.text(
        (90, 114),
        "Each process starts from one single-objective optimum; dashed lines are the local oval constraints at selected points.",
        fill=rgb(COLORS["muted"]),
        font=FONTS["subtitle"],
    )

    panel_w, panel_h = 780, 650
    left, top = 85, 205
    gap_x, gap_y = 85, 80
    data_x = (-1.58, 1.58)
    data_y = (-1.08, 1.08)
    seed_colors = [COLORS["green"], "#71aa59", COLORS["purple"], COLORS["yellow"]]
    label_offsets = {
        1: (12, -28),
        2: (14, 4),
        3: (-74, -26),
        4: (12, -28),
    }

    for panel_idx, (title, step_level) in enumerate(panels):
        row, col = divmod(panel_idx, 3)
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        plot = (x0 + 34, y0 + 72, x0 + panel_w - 34, y0 + panel_h - 52)
        px0, py0, px1, py1 = plot
        plot_w, plot_h = px1 - px0, py1 - py0
        scale = min(plot_w / (data_x[1] - data_x[0]), plot_h / (data_y[1] - data_y[0]))
        cx = 0.5 * (px0 + px1)
        cy = 0.5 * (py0 + py1)

        def to_px_data(v: np.ndarray) -> tuple[float, float]:
            return cx + scale * v[0], cy - scale * v[1]

        draw.text((x0 + 24, y0 + 22), title, fill=rgb(COLORS["ink"]), font=FONTS["label"])
        if step_level is not None:
            draw.text((x0 + 130, y0 + 26), "four current selected points", fill=rgb(COLORS["muted"]), font=FONTS["small"])

        for gx in np.linspace(-1.5, 1.5, 7):
            xp = cx + scale * gx
            draw.line([(xp, py0), (xp, py1)], fill=rgb("#eef2f7"), width=1)
        for gy in np.linspace(-1.0, 1.0, 5):
            yp = cy - scale * gy
            draw.line([(px0, yp), (px1, yp)], fill=rgb("#eef2f7"), width=1)
        draw.line([(px0, cy), (px1, cy)], fill=rgb(COLORS["grid"]), width=2)
        draw.line([(cx, py0), (cx, py1)], fill=rgb(COLORS["grid"]), width=2)

        ellipse_bbox = (
            cx - scale * AXES[0],
            cy - scale * AXES[1],
            cx + scale * AXES[0],
            cy + scale * AXES[1],
        )
        draw.ellipse(ellipse_bbox, fill=rgb(COLORS["ellipse_fill"]), outline=rgb(COLORS["ellipse"]), width=4)

        if step_level is None:
            draw.text((px0 + 10, py1 - 28), "Feasible region: g(x) <= 0", fill=rgb(COLORS["muted"]), font=FONTS["tiny"])
            continue

        for seed_idx in range(1, len(OBJECTIVES) + 1):
            trajectory = by_seed.get(seed_idx, [])
            if not trajectory:
                continue
            shown = trajectory[: min(step_level + 1, len(trajectory))]
            color = seed_colors[seed_idx - 1]

            # Previous local feasible constraints are retained lightly; the
            # current one is emphasized. Every line is tangent to the oval.
            for old_point in shown[:-1]:
                segment = tangent_line_segment(old_point, data_x, data_y)
                if segment is not None:
                    draw_dashed_line(draw, to_px_data(segment[0]), to_px_data(segment[1]), "#cbd3df", width=1, dash=12, gap=10)
            current = shown[-1]
            segment = tangent_line_segment(current, data_x, data_y)
            if segment is not None:
                draw_dashed_line(draw, to_px_data(segment[0]), to_px_data(segment[1]), color, width=3, dash=14, gap=9)

            trail_pixels = [to_px_data(point) for point in shown]
            if len(trail_pixels) > 1:
                draw.line(trail_pixels, fill=rgb(color), width=4, joint="curve")
            for old_point in shown[:-1]:
                p_old = to_px_data(old_point)
                draw.ellipse((p_old[0] - 6, p_old[1] - 6, p_old[0] + 6, p_old[1] + 6), fill=rgb("#687386"), outline=rgb("#ffffff"), width=2)

            p = to_px_data(current)
            draw.ellipse((p[0] - 12, p[1] - 12, p[0] + 12, p[1] + 12), fill=rgb(color), outline=rgb("#ffffff"), width=3)
            dx, dy = label_offsets[seed_idx]
            draw.text((p[0] + dx, p[1] + dy), f"f{seed_idx} SP{step_level}", fill=rgb(color), font=FONTS["tiny"])

            grad_direction = unit(OBJECTIVES[seed_idx - 1].gradient(current))
            grad_end = to_px_data(current + 0.18 * grad_direction)
            draw_arrow(draw, p, grad_end, color, width=4)
            gx_off, gy_off = (8, -18) if seed_idx != 3 else (-42, 7)
            draw.text((grad_end[0] + gx_off, grad_end[1] + gy_off), f"grad f{seed_idx}", fill=rgb(color), font=FONTS["tiny"])

        if panel_idx == 1:
            draw.text((px0 + 10, py1 - 28), "Dashed lines: grad g(xs)^T (x - xs) = 0", fill=rgb(COLORS["muted"]), font=FONTS["tiny"])

    draw.text((90, 1690), "Colored dashed lines are local feasible constraints at the current selected points, so they are tangent to the oval.", fill=rgb(COLORS["muted"]), font=FONTS["small"])
    draw.text((90, 1722), "Colored arrows show grad f_i(xs) for the corresponding seed objective; pale dashed lines are earlier local constraints.", fill=rgb(COLORS["muted"]), font=FONTS["small"])

    image.save(HERE / "oval_algorithm_steps_2d.png", dpi=(300, 300))
    image.save(HERE / "oval_algorithm_steps_2d.tiff", dpi=(300, 300))


def make_figure(points: list[np.ndarray], point_meta: list[tuple[int, int]], vertices: list[np.ndarray], corr: np.ndarray, groups: list[list[int]]) -> None:
    width, height = 2200, 1500
    image = Image.new("RGB", (width, height), rgb("#ffffff"))
    draw = ImageDraw.Draw(image)

    draw.text((80, 58), "Two-dimensional nonlinear ORCA demonstration", fill=rgb(COLORS["ink"]), font=FONTS["title"])
    draw.text(
        (80, 114),
        "A flat oval makes the selected-point geometry and objective grouping visible.",
        fill=rgb(COLORS["muted"]),
        font=FONTS["subtitle"],
    )

    plot_box = (95, 205, 1280, 1320)
    px0, py0, px1, py1 = plot_box
    cx, cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    scale = 380.0

    def to_px(x: np.ndarray) -> tuple[float, float]:
        return cx + scale * x[0], cy - scale * x[1]

    for gx in np.linspace(-1.5, 1.5, 7):
        xpix = cx + scale * gx
        draw.line([(xpix, py0), (xpix, py1)], fill=rgb("#eef2f7"), width=2)
    for gy in np.linspace(-1.0, 1.0, 5):
        ypix = cy - scale * gy
        draw.line([(px0, ypix), (px1, ypix)], fill=rgb("#eef2f7"), width=2)
    draw.line([(px0, cy), (px1, cy)], fill=rgb(COLORS["grid"]), width=3)
    draw.line([(cx, py0), (cx, py1)], fill=rgb(COLORS["grid"]), width=3)
    draw.text((px1 - 34, cy + 12), "x1", fill=rgb(COLORS["muted"]), font=FONTS["small"])
    draw.text((cx + 12, py0 + 6), "x2", fill=rgb(COLORS["muted"]), font=FONTS["small"])

    ellipse_bbox = (
        cx - scale * AXES[0],
        cy - scale * AXES[1],
        cx + scale * AXES[0],
        cy + scale * AXES[1],
    )
    draw.ellipse(ellipse_bbox, fill=rgb(COLORS["ellipse_fill"]), outline=rgb(COLORS["ellipse"]), width=7)

    by_seed: dict[int, list[np.ndarray]] = {}
    for x, meta in zip(points, point_meta):
        by_seed.setdefault(meta[0], []).append(x)
    for seed, trajectory in by_seed.items():
        color = SEED_COLORS[seed]
        pix = [to_px(x) for x in trajectory]
        if len(pix) > 1:
            draw.line(pix, fill=rgb(color), width=5, joint="curve")
        for p in pix[::4]:
            draw.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill=rgb(color), outline=rgb("#ffffff"), width=2)

    for idx, x in enumerate(vertices, start=1):
        p = to_px(x)
        color = SEED_COLORS[idx]
        draw.ellipse((p[0] - 18, p[1] - 18, p[0] + 18, p[1] + 18), fill=rgb(color), outline=rgb("#ffffff"), width=4)
        draw.text((p[0] + 18, p[1] - 20), f"{OBJ_LABELS[idx - 1]}*", fill=rgb(COLORS["ink"]), font=FONTS["label"])

    origin = np.array([0.0, 0.0])
    draw.text((165, 1138), "Dominant descent directions", fill=rgb(COLORS["muted"]), font=FONTS["tiny"])
    vector_specs = [
        ((165, 1195), (315, 1195), "f1 descent", COLORS["green"], (12, -12)),
        ((165, 1250), (315, 1225), "f2 descent", "#71aa59", (12, -12)),
        ((630, 1195), (480, 1195), "f3 descent", COLORS["purple"], (-118, -36)),
        ((630, 1250), (630, 1125), "f4 descent", COLORS["yellow"], (12, -12)),
    ]
    for start, end, label, color, offset in vector_specs:
        draw_arrow(draw, start, end, color, width=5)
        draw.text((end[0] + offset[0], end[1] + offset[1]), label, fill=rgb(color), font=FONTS["small"])

    side_x = 1385
    draw.text((side_x, 230), "Objectives", fill=rgb(COLORS["ink"]), font=FONTS["label"])
    y = 278
    for idx, desc in enumerate(OBJ_DESCRIPTIONS, start=1):
        color = SEED_COLORS[idx]
        draw.rounded_rectangle((side_x, y - 12, side_x + 40, y + 28), radius=6, fill=rgb(color))
        draw.text((side_x + 55, y - 8), desc, fill=rgb(COLORS["ink"]), font=FONTS["small"])
        y += 62

    draw.text((side_x, 570), "Correlation strength", fill=rgb(COLORS["ink"]), font=FONTS["label"])
    cell = 92
    mat_x, mat_y = side_x + 90, 620
    for i, label in enumerate(OBJ_LABELS):
        draw.text((mat_x + i * cell + 26, mat_y - 38), label, fill=rgb(COLORS["muted"]), font=FONTS["small"])
        draw.text((mat_x - 54, mat_y + i * cell + 26), label, fill=rgb(COLORS["muted"]), font=FONTS["small"])
    for i in range(len(OBJ_LABELS)):
        for j in range(len(OBJ_LABELS)):
            val = float(corr[i, j])
            low = np.array(rgb("#f4f6fa"))
            high = np.array(rgb("#4f83cc"))
            fill = tuple(np.round(low * (1.0 - val) + high * val).astype(int))
            x0 = mat_x + j * cell
            y0 = mat_y + i * cell
            draw.rectangle((x0, y0, x0 + cell - 4, y0 + cell - 4), fill=fill, outline=rgb("#ffffff"), width=3)
            text = f"{val:.2f}"
            bbox = draw.textbbox((0, 0), text, font=FONTS["small"])
            draw.text(
                (x0 + (cell - (bbox[2] - bbox[0])) / 2 - 2, y0 + (cell - (bbox[3] - bbox[1])) / 2 - 4),
                text,
                fill=rgb(COLORS["ink"] if val < 0.72 else COLORS["white"]),
                font=FONTS["small"],
            )

    draw.text((side_x, 1030), "Grouping", fill=rgb(COLORS["ink"]), font=FONTS["label"])
    gy = 1080
    for idx, group in enumerate(groups, start=1):
        labels = ", ".join(f"f{i}" for i in group)
        draw.rounded_rectangle((side_x, gy, side_x + 500, gy + 54), radius=8, fill=rgb(COLORS["panel"]), outline=rgb("#d9e0ea"), width=2)
        draw.text((side_x + 22, gy + 14), f"G{idx} = {{{labels}}}", fill=rgb(COLORS["ink"]), font=FONTS["body"])
        gy += 72

    draw.text(
        (95, 1370),
        "Selected points start from each single-objective optimum and move along projected conic combinations of objective gradients.",
        fill=rgb(COLORS["muted"]),
        font=FONTS["small"],
    )

    image.save(HERE / "oval_algorithm_demo_2d.png", dpi=(300, 300))
    image.save(HERE / "oval_algorithm_demo_2d.tiff", dpi=(300, 300))


def main() -> None:
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

    vertex_thetas = problem.initial_thetas()
    vertices = problem.initial_points()
    points = np.asarray(result.input_data.points, dtype=float)
    point_meta = point_metadata_from_package_order(points, config)
    step_rows = package_step_rows(points, point_meta)
    direction_rows = package_direction_rows(
        points,
        point_meta,
        None if result.interaction_data is None else result.interaction_data.projected_directions,
    )
    corr = result.adj_matrix
    total_weight = result.interaction_data.total_weights
    strength_weight = result.interaction_data.metadata["weighted_strengths"]
    raw_alignment = raw_alignment_from_interactions(result.interaction_data.strengths, result.interaction_data.valid_mask)
    unprojected = unprojected_objective_angle_matrix(list(points))
    groups = group_sets_from_labels(result.groups)

    write_outputs(
        vertices,
        vertex_thetas,
        list(points),
        point_meta,
        step_rows,
        direction_rows,
        corr,
        raw_alignment,
        total_weight,
        strength_weight,
        unprojected,
        groups,
    )
    make_figure(list(points), point_meta, vertices, corr, groups)
    make_step_figure(list(points), point_meta)

    print("2D oval nonlinear ORCA package demonstration")
    print(f"axes = {AXES.tolist()}")
    print(f"selected points = {len(points)}")
    print(f"ORCA package root = {ORCA_PACKAGE_ROOT}")
    print(f"group labels = {result.groups.tolist()}")
    print("\nObjective correlation matrix:")
    for row in corr:
        print("  " + "  ".join(f"{value:0.3f}" for value in row))
    print("\nGroups:")
    for idx, group in enumerate(groups, start=1):
        labels = ", ".join(f"f{i}" for i in group)
        print(f"  G{idx} = {{{labels}}}")


if __name__ == "__main__":
    main()
