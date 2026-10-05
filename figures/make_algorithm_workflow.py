from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
W, H = 2400, 1350


PALETTE = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "grid": "#d9dee7",
    "panel": "#f7f9fc",
    "panel_edge": "#d5dbe6",
    "blue": "#4f83cc",
    "blue_fill": "#dcecff",
    "orange": "#d98b2b",
    "orange_fill": "#f9ddbf",
    "green": "#4c9a6a",
    "green_fill": "#dff0e6",
    "red": "#c85b5b",
    "purple": "#7b6fc8",
    "purple_fill": "#ebe8fb",
    "yellow": "#f2c94c",
    "white": "#ffffff",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def alpha_rgb(value: str, alpha: float, bg: str = "#ffffff") -> tuple[int, int, int]:
    fg = np.array(hex_to_rgb(value), dtype=float)
    back = np.array(hex_to_rgb(bg), dtype=float)
    return tuple(np.round(alpha * fg + (1 - alpha) * back).astype(int))


def font_path(bold: bool = False) -> str | None:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/DejaVu Sans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = font_path(bold=bold)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


class FigureCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), PALETTE["white"])
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def rect(
        self,
        xy: tuple[float, float, float, float],
        fill: str = "#ffffff",
        outline: str | None = None,
        width: int = 1,
        radius: int = 0,
    ) -> None:
        rgb_fill = None if fill == "none" else hex_to_rgb(fill)
        rgb_outline = hex_to_rgb(outline) if outline else None
        if radius:
            self.draw.rounded_rectangle(xy, radius=radius, fill=rgb_fill, outline=rgb_outline, width=width)
            self.svg.append(
                f'<rect x="{xy[0]:.1f}" y="{xy[1]:.1f}" width="{xy[2]-xy[0]:.1f}" height="{xy[3]-xy[1]:.1f}" '
                f'rx="{radius}" fill="{fill}" stroke="{outline or "none"}" stroke-width="{width}"/>'
            )
        else:
            self.draw.rectangle(xy, fill=rgb_fill, outline=rgb_outline, width=width)
            self.svg.append(
                f'<rect x="{xy[0]:.1f}" y="{xy[1]:.1f}" width="{xy[2]-xy[0]:.1f}" height="{xy[3]-xy[1]:.1f}" '
                f'fill="{fill}" stroke="{outline or "none"}" stroke-width="{width}"/>'
            )

    def ellipse(
        self,
        xy: tuple[float, float, float, float],
        fill: str | tuple[int, int, int] | None = None,
        outline: str | None = None,
        width: int = 1,
    ) -> None:
        fill_rgb = fill if isinstance(fill, tuple) else (hex_to_rgb(fill) if fill else None)
        outline_rgb = hex_to_rgb(outline) if outline else None
        self.draw.ellipse(xy, fill=fill_rgb, outline=outline_rgb, width=width)
        fill_svg = (
            f"rgb{fill}" if isinstance(fill, tuple) else (fill if fill else "none")
        )
        self.svg.append(
            f'<ellipse cx="{(xy[0]+xy[2])/2:.1f}" cy="{(xy[1]+xy[3])/2:.1f}" '
            f'rx="{(xy[2]-xy[0])/2:.1f}" ry="{(xy[3]-xy[1])/2:.1f}" '
            f'fill="{fill_svg}" stroke="{outline or "none"}" stroke-width="{width}"/>'
        )

    def line(
        self,
        points: list[tuple[float, float]],
        fill: str = "#000000",
        width: int = 2,
        dash: str | None = None,
    ) -> None:
        self.draw.line(points, fill=hex_to_rgb(fill), width=width)
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            f'<polyline points="{point_str}" fill="none" stroke="{fill}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        fill: str,
        outline: str | None = None,
        width: int = 1,
    ) -> None:
        self.draw.polygon(points, fill=hex_to_rgb(fill), outline=hex_to_rgb(outline) if outline else None)
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.svg.append(
            f'<polygon points="{point_str}" fill="{fill}" stroke="{outline or "none"}" stroke-width="{width}"/>'
        )

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = "#000000",
        width: int = 4,
        head: int = 16,
        dash: str | None = None,
    ) -> None:
        self.line([start, end], fill=color, width=width, dash=dash)
        sx, sy = start
        ex, ey = end
        angle = math.atan2(ey - sy, ex - sx)
        left = (ex - head * math.cos(angle - math.pi / 7), ey - head * math.sin(angle - math.pi / 7))
        right = (ex - head * math.cos(angle + math.pi / 7), ey - head * math.sin(angle + math.pi / 7))
        self.polygon([end, left, right], fill=color)

    def arc_arrow(
        self,
        points: list[tuple[float, float]],
        color: str = "#000000",
        width: int = 4,
        head: int = 16,
    ) -> None:
        self.line(points, fill=color, width=width)
        if len(points) >= 2:
            self.arrow(points[-2], points[-1], color=color, width=0, head=head)

    def text(
        self,
        xy: tuple[float, float],
        text: str,
        size: int = 28,
        color: str = "#000000",
        bold: bool = False,
        anchor: str = "la",
    ) -> None:
        font = get_font(size, bold=bold)
        self.draw.text(xy, text, fill=hex_to_rgb(color), font=font, anchor=anchor)
        weight = "700" if bold else "400"
        svg_anchor = {"la": "start", "ma": "middle", "ra": "end", "mm": "middle"}.get(anchor, "start")
        dominant = "middle" if anchor in {"lm", "mm", "rm"} else "auto"
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.svg.append(
            f'<text x="{xy[0]:.1f}" y="{xy[1]:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{svg_anchor}" '
            f'dominant-baseline="{dominant}">{safe}</text>'
        )

    def multiline(
        self,
        xy: tuple[float, float],
        text: str,
        size: int = 24,
        color: str = "#000000",
        bold: bool = False,
        width_chars: int = 36,
        line_gap: float = 1.22,
    ) -> None:
        x, y = xy
        lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
            else:
                lines.extend(textwrap.wrap(paragraph, width=width_chars))
        for i, line in enumerate(lines):
            self.text((x, y + i * size * line_gap), line, size=size, color=color, bold=bold)

    def save(self, stem: str) -> None:
        self.svg.append("</svg>")
        (OUT_DIR / f"{stem}.svg").write_text("\n".join(self.svg), encoding="utf-8")
        self.image.save(OUT_DIR / f"{stem}.png", dpi=(300, 300))
        self.image.save(OUT_DIR / f"{stem}.tiff", dpi=(600, 600), compression="tiff_lzw")


def draw_panel_header(c: FigureCanvas, label: str, title: str, x: int, y: int) -> None:
    c.ellipse((x, y - 30, x + 54, y + 24), fill=PALETTE["ink"], outline=PALETTE["ink"])
    c.text((x + 27, y - 3), label, size=25, color=PALETTE["white"], bold=True, anchor="ma")
    c.text((x + 70, y), title, size=31, color=PALETTE["ink"], bold=True)


def draw_small_ellipsoid(c: FigureCanvas, cx: float, cy: float, rx: float, ry: float) -> None:
    c.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=PALETTE["blue_fill"], outline=PALETTE["blue"], width=4)
    # Curved "active" boundary segment.
    pts = []
    for t in np.linspace(-0.4, 1.08, 70):
        x = cx + rx * math.cos(t)
        y = cy - ry * math.sin(t)
        pts.append((x, y))
    c.line(pts, fill=PALETTE["blue"], width=7)


def draw_step_badge(c: FigureCanvas, xy: tuple[float, float], number: str, color: str = PALETTE["orange"]) -> None:
    x, y = xy
    c.ellipse((x - 20, y - 20, x + 20, y + 20), fill=color, outline=PALETTE["white"], width=3)
    c.text((x, y + 1), number, size=23, color=PALETTE["white"], bold=True, anchor="mm")


def draw_matrix(c: FigureCanvas, x: int, y: int, cell: int = 52) -> None:
    values = np.array(
        [
            [1.00, 0.91, 0.25, 0.48],
            [0.91, 1.00, 0.31, 0.52],
            [0.25, 0.31, 1.00, 0.43],
            [0.48, 0.52, 0.43, 1.00],
        ]
    )
    labels = ["f1", "f2", "f3", "f4"]
    for i, lab in enumerate(labels):
        c.text((x + (i + 0.5) * cell, y - 14), lab, size=18, color=PALETTE["muted"], anchor="ma")
        c.text((x - 15, y + (i + 0.62) * cell), lab, size=18, color=PALETTE["muted"], anchor="ra")
    for i in range(4):
        for j in range(4):
            v = values[i, j]
            fill_rgb = alpha_rgb(PALETTE["green"], 0.18 + 0.70 * v)
            c.rect((x + j * cell, y + i * cell, x + (j + 1) * cell, y + (i + 1) * cell), fill="#%02x%02x%02x" % fill_rgb, outline=PALETTE["white"], width=2)
            if i != j:
                c.text((x + (j + 0.5) * cell, y + (i + 0.62) * cell), f"{v:.2f}", size=16, color=PALETTE["ink"], anchor="ma")
    c.rect((x, y, x + 4 * cell, y + 4 * cell), fill="none", outline=PALETTE["panel_edge"], width=3)


def draw_graph(c: FigureCanvas, cx: int, cy: int) -> None:
    nodes = {
        "f1": (cx - 72, cy - 62),
        "f2": (cx + 50, cy - 66),
        "f3": (cx - 35, cy + 68),
        "f4": (cx + 92, cy + 48),
    }
    edges = [
        ("f1", "f2", 8, PALETTE["green"]),
        ("f1", "f4", 3, PALETTE["muted"]),
        ("f2", "f4", 3, PALETTE["muted"]),
        ("f1", "f3", 2, PALETTE["muted"]),
        ("f2", "f3", 2, PALETTE["muted"]),
        ("f3", "f4", 2, PALETTE["muted"]),
    ]
    for a, b, width, col in edges:
        c.line([nodes[a], nodes[b]], fill=col, width=width)
    for lab, (x, y) in nodes.items():
        fill = PALETTE["green"] if lab in {"f1", "f2"} else (PALETTE["purple"] if lab == "f3" else PALETTE["orange"])
        c.ellipse((x - 34, y - 34, x + 34, y + 34), fill=fill, outline=PALETTE["white"], width=4)
        c.text((x, y + 9), lab, size=25, color=PALETTE["white"], bold=True, anchor="ma")


def main() -> None:
    c = FigureCanvas(W, H)

    c.text((75, 72), "Nonlinear Objective Dimensionality Reduction Workflow", size=44, color=PALETTE["ink"], bold=True)
    c.text(
        (75, 120),
        "Sample informative feasible points, linearize locally, aggregate pairwise objective correlations, and detect objective communities.",
        size=25,
        color=PALETTE["muted"],
    )

    panel_y0, panel_y1 = 170, 1220
    panel_w = 520
    gap = 55
    xs = [75 + i * (panel_w + gap) for i in range(4)]
    titles = [
        "Problem + seeds",
        "Selected points",
        "Linearization",
        "Objective groups",
    ]
    labels = list("ABCD")
    for x, label, title in zip(xs, labels, titles):
        c.rect((x, panel_y0, x + panel_w, panel_y1), fill=PALETTE["panel"], outline=PALETTE["panel_edge"], width=3, radius=18)
        draw_panel_header(c, label, title, x + 25, panel_y0 + 58)

    # Flow arrows between panels.
    for i in range(3):
        y = 695
        c.arrow((xs[i] + panel_w + 8, y), (xs[i + 1] - 12, y), color=PALETTE["muted"], width=5, head=22)

    # Panel A.
    ax = xs[0]
    c.rect((ax + 46, 285, ax + 470, 470), fill=PALETTE["white"], outline=PALETTE["panel_edge"], width=3, radius=14)
    c.text((ax + 74, 342), "Nonlinear MaOP", size=28, color=PALETTE["ink"], bold=True)
    c.text((ax + 74, 392), "min {f1(x), ..., fM(x)}", size=25, color=PALETTE["ink"])
    c.text((ax + 74, 432), "s.t. g(x) <= 0,  x in X", size=25, color=PALETTE["ink"])
    draw_small_ellipsoid(c, ax + 260, 745, 190, 140)
    seed_points = [(ax + 105, 760), (ax + 260, 605), (ax + 420, 750)]
    for idx, pt in enumerate(seed_points, 1):
        c.ellipse((pt[0] - 11, pt[1] - 11, pt[0] + 11, pt[1] + 11), fill=PALETTE["ink"], outline=PALETTE["white"], width=3)
        c.text((pt[0], pt[1] - 24), f"f{idx}*", size=19, color=PALETTE["ink"], anchor="ma")
    c.text((ax + 70, 940), "Single-objective optima seed the search.", size=24, color=PALETTE["ink"])
    c.text((ax + 70, 984), "Curved constraints and interior optima are both considered.", size=20, color=PALETTE["muted"])

    # Panel B.
    bx = xs[1]
    draw_small_ellipsoid(c, bx + 260, 625, 200, 148)
    x_n = (bx + 205, 640)
    trial = (bx + 365, 465)
    projected = (bx + 355, 560)
    next_pt = (bx + 330, 592)
    c.ellipse((x_n[0] - 12, x_n[1] - 12, x_n[0] + 12, x_n[1] + 12), fill=PALETTE["ink"], outline=PALETTE["white"], width=3)
    c.text((x_n[0] - 18, x_n[1] + 42), "x_n", size=24, color=PALETTE["ink"], anchor="ra")
    cone = [(x_n[0], x_n[1]), (x_n[0] + 155, x_n[1] - 185), (x_n[0] + 215, x_n[1] - 52)]
    c.polygon(cone, fill=PALETTE["green_fill"], outline=PALETTE["green"], width=3)
    c.arrow(x_n, (x_n[0] + 130, x_n[1] - 140), color=PALETTE["green"], width=5)
    c.arrow(x_n, (x_n[0] + 178, x_n[1] - 45), color=PALETTE["green"], width=5)
    c.text((bx + 66, 352), "Cone of projected objective gradients", size=25, color=PALETTE["green"], bold=True)
    c.arrow(x_n, trial, color=PALETTE["orange"], width=5, dash="9,8")
    c.ellipse((trial[0] - 11, trial[1] - 11, trial[0] + 11, trial[1] + 11), fill=PALETTE["white"], outline=PALETTE["orange"], width=4)
    c.text((trial[0] + 18, trial[1] - 8), "trial point", size=22, color=PALETTE["orange"])
    c.arrow(trial, projected, color=PALETTE["purple"], width=5)
    c.ellipse((next_pt[0] - 13, next_pt[1] - 13, next_pt[0] + 13, next_pt[1] + 13), fill=PALETTE["orange"], outline=PALETTE["white"], width=3)
    c.text((next_pt[0] + 20, next_pt[1] + 10), "x_(n+1)", size=24, color=PALETTE["ink"])
    c.rect((bx + 55, 835, bx + 465, 1065), fill=PALETTE["white"], outline=PALETTE["panel_edge"], width=3, radius=14)
    step_text = [
        ("1", "combine projected gradients"),
        ("2", "step: x_n + alpha d"),
        ("3", "project back to feasible set X"),
        ("4", "repeat from new selected point"),
    ]
    for row, (num, txt) in enumerate(step_text):
        yy = 885 + row * 48
        draw_step_badge(c, (bx + 88, yy - 8), num)
        c.text((bx + 122, yy), txt, size=22, color=PALETTE["ink"])

    # Panel C.
    cx = xs[2]
    draw_small_ellipsoid(c, cx + 260, 485, 180, 125)
    selected = [(cx + 102, 540), (cx + 205, 595), (cx + 290, 500), (cx + 400, 435)]
    for k, pt in enumerate(selected):
        c.ellipse((pt[0] - 10, pt[1] - 10, pt[0] + 10, pt[1] + 10), fill=PALETTE["orange"], outline=PALETTE["white"], width=3)
        slope = [-0.8, -0.35, 0.30, 0.95][k]
        dx = 86
        dy = slope * dx
        c.line([(pt[0] - dx, pt[1] - dy), (pt[0] + dx, pt[1] + dy)], fill=PALETTE["muted"], width=3, dash="8,7")
    c.text((cx + 58, 670), "At each selected point:", size=25, color=PALETTE["ink"], bold=True)
    c.rect((cx + 58, 710, cx + 462, 890), fill=PALETTE["white"], outline=PALETTE["panel_edge"], width=3, radius=14)
    c.text((cx + 86, 762), "g_n(x) = g(x_n) + grad g(x_n)(x - x_n)", size=20, color=PALETTE["ink"])
    c.text((cx + 86, 814), "f_i,n(x) = f_i(x_n) + grad f_i(x_n)(x - x_n)", size=20, color=PALETTE["ink"])
    c.text((cx + 86, 862), "Store Jacobians for constraints and objectives", size=20, color=PALETTE["muted"])
    c.line([(cx + 72, 1015), (cx + 448, 955)], fill=PALETTE["muted"], width=4)
    base = (cx + 225, 990)
    c.arrow(base, (base[0] + 95, base[1] - 85), color=PALETTE["green"], width=5)
    c.arrow(base, (base[0] + 115, base[1] - 18), color=PALETTE["orange"], width=5)
    c.text((cx + 72, 1080), "Score pairs by projected-gradient alignment", size=22, color=PALETTE["ink"])
    c.text((cx + 72, 1120), "S_ij^A = weighted average over k,n", size=21, color=PALETTE["muted"])

    # Panel D.
    dx = xs[3]
    draw_matrix(c, dx + 62, 330, cell=55)
    c.text((dx + 62, 585), "Objective correlation matrix", size=21, color=PALETTE["muted"])
    draw_graph(c, dx + 385, 445)
    c.arrow((dx + 292, 450), (dx + 326, 450), color=PALETTE["muted"], width=4, head=18)
    c.text((dx + 70, 670), "Matrix entries become graph edge weights.", size=23, color=PALETTE["ink"])
    c.text((dx + 70, 712), "Community detection groups correlated objectives.", size=23, color=PALETTE["ink"])
    c.rect((dx + 74, 805, dx + 446, 1065), fill=PALETTE["white"], outline=PALETTE["panel_edge"], width=3, radius=14)
    c.text((dx + 105, 865), "Reduced problem", size=29, color=PALETTE["ink"], bold=True)
    c.text((dx + 105, 925), "G1 = {f1, f2}", size=27, color=PALETTE["green"], bold=True)
    c.text((dx + 105, 978), "G2 = {f3}", size=27, color=PALETTE["purple"], bold=True)
    c.text((dx + 105, 1031), "G3 = {f4}", size=27, color=PALETTE["orange"], bold=True)

    # Bottom note.
    c.rect((75, 1250, 2325, 1308), fill="#f4f6fa", outline=PALETTE["panel_edge"], width=2, radius=16)
    c.text((105, 1288), "Key distinction from the linear algorithm: nonlinear objective interactions are sampled at multiple feasible points before building the correlation graph.", size=24, color=PALETTE["ink"])

    c.save("algorithm_workflow_v2")


if __name__ == "__main__":
    main()
