from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
OUT = HERE

W, H = 2400, 1600
AXES = np.array([1.40, 0.90, 0.60], dtype=float)
Q = np.diag(1.0 / AXES**2)

PALETTE = {
    "ink": "#18212b",
    "muted": "#667085",
    "grid": "#d3dbe7",
    "blue": "#4f83cc",
    "blue_light": "#d9ebff",
    "green": "#4c9a6a",
    "green_light": "#dbf1e3",
    "orange": "#dd8a22",
    "purple": "#796bc8",
    "red": "#c85b5b",
    "white": "#ffffff",
    "panel": "#f7f9fc",
}

SEED_COLORS = {
    1: PALETTE["green"],
    2: "#71aa59",
    3: PALETTE["purple"],
    4: PALETTE["orange"],
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    return (*rgb(hex_color), alpha)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(52, True),
    "subtitle": font(29),
    "label": font(28, True),
    "body": font(25),
    "small": font(21),
    "tiny": font(18),
}


def load_points() -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, np.ndarray], np.ndarray]:
    trajectories: dict[int, list[tuple[int, np.ndarray]]] = {}
    with (HERE / "ellipsoid_selected_points.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(float(row["seed_objective"]))
            step = int(float(row["step"]))
            x = np.array([float(row["x1"]), float(row["x2"]), float(row["x3"])])
            trajectories.setdefault(seed, []).append((step, x))

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


def rotation_matrix(elev_deg: float = 22.0, azim_deg: float = -38.0) -> np.ndarray:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    rz = np.array(
        [
            [math.cos(azim), -math.sin(azim), 0.0],
            [math.sin(azim), math.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev), -math.sin(elev)],
            [0.0, math.sin(elev), math.cos(elev)],
        ]
    )
    return rx @ rz


ROT = rotation_matrix()
SCALE = 455.0
CENTER = np.array([925.0, 840.0])


def project(p: np.ndarray) -> tuple[float, float]:
    q = ROT @ p
    return CENTER[0] + SCALE * q[0], CENTER[1] - SCALE * q[1]


def depth(p: np.ndarray) -> float:
    return float((ROT @ p)[2])


def ellipsoid_value(x: np.ndarray) -> float:
    return float(x @ Q @ x - 1.0)


def ellipsoid_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * Q @ x


def tangent_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = ellipsoid_grad(x)
    n = n / np.linalg.norm(n)
    probe = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(n, probe))) > 0.85:
        probe = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, probe)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def tangent_project(vec: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = ellipsoid_grad(x)
    out = vec - np.dot(vec, n) / np.dot(n, n) * n
    return out / np.linalg.norm(out)


def draw_arrow_2d(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    width: int = 6,
    head: int = 20,
    dash: bool = False,
) -> None:
    sx, sy = start
    ex, ey = end
    if dash:
        segments = 18
        for i in range(segments):
            if i % 2 == 0:
                a = i / segments
                b = (i + 1) / segments
                draw.line(
                    [(sx + (ex - sx) * a, sy + (ey - sy) * a), (sx + (ex - sx) * b, sy + (ey - sy) * b)],
                    fill=rgb(color),
                    width=width,
                )
    else:
        draw.line([start, end], fill=rgb(color), width=width)

    angle = math.atan2(ey - sy, ex - sx)
    left = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
    right = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
    draw.polygon([end, left, right], fill=rgb(color))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str = PALETTE["ink"]) -> None:
    x, y = xy
    pad = 9
    bbox = draw.textbbox((x, y), text, font=FONTS["small"])
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=10,
        fill=(255, 255, 255, 218),
        outline=rgb("#d5dce8"),
        width=2,
    )
    draw.text((x, y), text, fill=rgb(color), font=FONTS["small"])


def ellipsoid_patches(nu: int = 48, nv: int = 24) -> list[tuple[float, list[tuple[float, float]], float]]:
    patches = []
    us = np.linspace(0, 2 * math.pi, nu + 1)
    vs = np.linspace(-math.pi / 2, math.pi / 2, nv + 1)
    for i in range(nu):
        for j in range(nv):
            corners = []
            depths = []
            for u, v in [(us[i], vs[j]), (us[i + 1], vs[j]), (us[i + 1], vs[j + 1]), (us[i], vs[j + 1])]:
                p = AXES * np.array([math.cos(v) * math.cos(u), math.cos(v) * math.sin(u), math.sin(v)])
                corners.append(project(p))
                depths.append(depth(p))
            patches.append((float(np.mean(depths)), corners, float(np.mean(depths))))
    return sorted(patches, key=lambda item: item[0])


def draw_ellipsoid(layer: Image.Image) -> None:
    draw = ImageDraw.Draw(layer, "RGBA")
    for _, corners, d in ellipsoid_patches():
        shade = int(np.clip(210 + 38 * d, 178, 238))
        fill = (shade, min(246, shade + 8), 255, 92)
        outline = (79, 131, 204, 54)
        draw.polygon(corners, fill=fill, outline=outline)

    # Latitude and longitude guide curves.
    for v in np.linspace(-0.8, 0.8, 5):
        pts = []
        for u in np.linspace(0, 2 * math.pi, 180):
            p = AXES * np.array([math.cos(v) * math.cos(u), math.cos(v) * math.sin(u), math.sin(v)])
            pts.append(project(p))
        draw.line(pts, fill=rgba(PALETTE["blue"], 95), width=2)
    for u in np.linspace(0, 2 * math.pi, 9, endpoint=False):
        pts = []
        for v in np.linspace(-math.pi / 2, math.pi / 2, 140):
            p = AXES * np.array([math.cos(v) * math.cos(u), math.cos(v) * math.sin(u), math.sin(v)])
            pts.append(project(p))
        draw.line(pts, fill=rgba(PALETTE["blue"], 65), width=2)


def draw_axes(draw: ImageDraw.ImageDraw) -> None:
    origin = np.array([0.0, 0.0, 0.0])
    axes = [
        (np.array([1.65, 0.0, 0.0]), "x1"),
        (np.array([0.0, 1.15, 0.0]), "x2"),
        (np.array([0.0, 0.0, 0.82]), "x3"),
    ]
    o = project(origin)
    for vec, lab in axes:
        end = project(vec)
        draw_arrow_2d(draw, o, end, PALETTE["muted"], width=3, head=13)
        draw.text((end[0] + 8, end[1] - 8), lab, fill=rgb(PALETTE["muted"]), font=FONTS["tiny"])


def draw_trajectory(draw: ImageDraw.ImageDraw, points: list[np.ndarray], color: str, width: int = 5) -> None:
    pts2d = [project(p) for p in points]
    if len(pts2d) > 1:
        draw.line(pts2d, fill=rgb(color), width=width, joint="curve")
        for idx in range(3, len(pts2d), 5):
            draw_arrow_2d(draw, pts2d[idx - 1], pts2d[idx], color, width=width, head=13)
    for p in points[::5]:
        x, y = project(p)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=rgb(color), outline=rgb(PALETTE["white"]), width=2)


def draw_tangent_plane(layer: Image.Image, x: np.ndarray) -> None:
    draw = ImageDraw.Draw(layer, "RGBA")
    u, v = tangent_basis(x)
    scale_u, scale_v = 0.33, 0.25
    corners = [
        x - scale_u * u - scale_v * v,
        x + scale_u * u - scale_v * v,
        x + scale_u * u + scale_v * v,
        x - scale_u * u + scale_v * v,
    ]
    draw.polygon([project(p) for p in corners], fill=rgba("#f0b76a", 92), outline=rgba(PALETTE["orange"], 170))


def draw_algorithm_step(draw: ImageDraw.ImageDraw, traj: list[np.ndarray], step_index: int = 9) -> None:
    x_n = traj[step_index]
    x_next = traj[step_index + 1]
    step_vec = x_next - x_n
    trial = x_n + 2.1 * step_vec
    if ellipsoid_value(trial) <= 0:
        trial = x_n + 3.0 * step_vec

    # Two projected descent directions that define the visually important cone.
    c1 = np.array([-1.0, 0.0, 0.0])
    c2 = np.array([-0.99, -0.05, 0.0])
    d1 = tangent_project(-c1, x_n)
    d2 = tangent_project(-c2, x_n)
    p0 = project(x_n)
    p1 = project(x_n + 0.34 * d1)
    p2 = project(x_n + 0.34 * d2)
    draw.polygon([p0, p1, p2], fill=(76, 154, 106, 54), outline=rgb(PALETTE["green"]))
    draw_arrow_2d(draw, p0, p1, PALETTE["green"], width=5, head=16)
    draw_arrow_2d(draw, p0, p2, PALETTE["green"], width=5, head=16)

    draw_arrow_2d(draw, p0, project(trial), PALETTE["orange"], width=5, head=18, dash=True)
    draw_arrow_2d(draw, project(trial), project(x_next), PALETTE["purple"], width=5, head=18)

    for p, color, r in [(x_n, PALETTE["ink"], 12), (trial, PALETTE["white"], 12), (x_next, PALETTE["orange"], 13)]:
        x, y = project(p)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb(color), outline=rgb(PALETTE["ink"]), width=3)

    draw_label(draw, (project(x_n)[0] - 72, project(x_n)[1] + 28), "current x_n")
    draw_label(draw, (project(trial)[0] + 28, project(trial)[1] - 70), "trial x_n + alpha d", PALETTE["orange"])
    draw_label(draw, (project(x_next)[0] + 42, project(x_next)[1] + 38), "projected x_(n+1)", PALETTE["orange"])
    draw.text((project(x_n)[0] - 40, project(x_n)[1] - 86), "projected-gradient cone", fill=rgb(PALETTE["green"]), font=FONTS["small"])


def draw_matrix_inset(draw: ImageDraw.ImageDraw, matrix: np.ndarray) -> None:
    x0, y0 = 1650, 915
    cell = 76
    draw.rounded_rectangle((x0 - 45, y0 - 80, x0 + cell * 4 + 185, y0 + cell * 4 + 185), radius=26, fill=rgb(PALETTE["panel"]), outline=rgb("#d4dce8"), width=3)
    draw.text((x0 - 12, y0 - 48), "Output: objective correlation", fill=rgb(PALETTE["ink"]), font=FONTS["label"])
    for i in range(4):
        draw.text((x0 + i * cell + 25, y0 - 26), f"f{i + 1}", fill=rgb(PALETTE["muted"]), font=FONTS["tiny"])
        draw.text((x0 - 35, y0 + i * cell + 26), f"f{i + 1}", fill=rgb(PALETTE["muted"]), font=FONTS["tiny"])
    for i in range(4):
        for j in range(4):
            v = matrix[i, j]
            intensity = int(242 - 100 * v)
            fill = (intensity, 236, int(255 - 50 * v))
            draw.rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell), fill=fill, outline=rgb(PALETTE["white"]), width=3)
            if i != j:
                draw.text((x0 + j * cell + 18, y0 + i * cell + 25), f"{v:.2f}", fill=rgb(PALETTE["ink"]), font=FONTS["tiny"])
    draw.text((x0 - 12, y0 + 4 * cell + 48), "G1 = {f1, f2}", fill=rgb(PALETTE["green"]), font=FONTS["body"])
    draw.text((x0 - 12, y0 + 4 * cell + 90), "G2 = {f3}", fill=rgb(PALETTE["purple"]), font=FONTS["body"])
    draw.text((x0 - 12, y0 + 4 * cell + 132), "G3 = {f4}", fill=rgb(PALETTE["orange"]), font=FONTS["body"])


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    x0, y0 = 1555, 250
    draw.rounded_rectangle((x0, y0, x0 + 650, y0 + 250), radius=26, fill=rgb(PALETTE["panel"]), outline=rgb("#d4dce8"), width=3)
    draw.text((x0 + 34, y0 + 34), "Toy objectives", fill=rgb(PALETTE["ink"]), font=FONTS["label"])
    entries = [
        (PALETTE["green"], "f1 = -x1"),
        ("#71aa59", "f2 = -0.99x1 - 0.05x2"),
        (PALETTE["purple"], "f3 = x1"),
        (PALETTE["orange"], "f4 = -x3"),
    ]
    for idx, (color, label) in enumerate(entries):
        yy = y0 + 88 + idx * 36
        draw.ellipse((x0 + 38, yy - 12, x0 + 62, yy + 12), fill=rgb(color), outline=rgb(PALETTE["white"]), width=2)
        draw.text((x0 + 80, yy - 16), label, fill=rgb(PALETTE["ink"]), font=FONTS["small"])


def main() -> None:
    trajectories, vertices, matrix = load_points()

    base = Image.new("RGBA", (W, H), rgb(PALETTE["white"]) + (255,))
    draw = ImageDraw.Draw(base, "RGBA")

    draw.text((88, 62), "Simple ellipsoid demonstration of the nonlinear reduction algorithm", fill=rgb(PALETTE["ink"]), font=FONTS["title"])
    draw.text((90, 126), "Selected points follow projected objective-gradient directions on a nonlinear feasible set, then objective correlations reveal the reducible structure.", fill=rgb(PALETTE["muted"]), font=FONTS["subtitle"])

    scene = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw_ellipsoid(scene)
    highlight_traj = [p for _, p in trajectories[3]]
    draw_tangent_plane(scene, highlight_traj[9])
    base.alpha_composite(scene)
    draw = ImageDraw.Draw(base, "RGBA")
    draw_axes(draw)

    # Draw trajectories behind/over surface according to seed color.
    for seed, rows in trajectories.items():
        pts = [p for _, p in rows]
        draw_trajectory(draw, pts, SEED_COLORS[seed], width=5 if seed in (1, 2) else 4)

    # Single-objective optima.
    label_offsets = {
        1: (-70, -34),
        2: (-20, -60),
        3: (20, -36),
        4: (34, -104),
    }
    for idx, x in vertices.items():
        px, py = project(x)
        color = SEED_COLORS[idx]
        draw.ellipse((px - 15, py - 15, px + 15, py + 15), fill=rgb(color), outline=rgb(PALETTE["white"]), width=4)
        ox, oy = label_offsets[idx]
        draw_label(draw, (px + ox, py + oy), f"f{idx} optimum", color)

    draw_algorithm_step(draw, highlight_traj, step_index=9)
    draw_legend(draw)
    draw_matrix_inset(draw, matrix)

    # Callout for the nonlinear boundary.
    draw.rounded_rectangle((90, 1325, 1260, 1458), radius=24, fill=rgb(PALETTE["panel"]), outline=rgb("#d4dce8"), width=3)
    draw.text((126, 1364), "Feasible set: (x1/1.40)^2 + (x2/0.90)^2 + (x3/0.60)^2 <= 1", fill=rgb(PALETTE["ink"]), font=FONTS["body"])
    draw.text((126, 1410), "Dashed orange arrow: random step. Purple arrow: Euclidean projection back to the ellipsoid.", fill=rgb(PALETTE["muted"]), font=FONTS["small"])

    image = base.convert("RGB")
    image.save(OUT / "ellipsoid_algorithm_demo_3d.png", dpi=(300, 300))
    image.save(OUT / "ellipsoid_algorithm_demo_3d.tiff", dpi=(600, 600), compression="tiff_lzw")


if __name__ == "__main__":
    main()
