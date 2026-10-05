from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "examples/ellipse"
OUT_DIR = ROOT / "figures"
OUT_BASE = OUT_DIR / "figure3_dtlz5_known_structure_benchmark_v4"

CURRENT_MATRIX = DATA_DIR / "audit_dtlz5_current_matrix.csv"
STAR_MATRIX = DATA_DIR / "audit_dtlz5_paper_star_matrix.csv"
SUMMARY = DATA_DIR / "audit_dtlz5_current_vs_paper_star_summary.csv"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
    }
)

COLORS = {
    "ink": "#18212b",
    "muted": "#667085",
    "grid": "#d9e0ea",
    "known": "#4f9d69",
    "known_light": "#dfeee5",
    "detected": "#e68619",
    "detected_light": "#f7e2c5",
    "star": "#6f63c6",
    "panel": "#f7f9fb",
    "low": "#f2f5c4",
    "blue": "#385f9d",
}


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    labels = [row["objective"] for row in rows]
    matrix = np.array([[float(row[label]) for label in labels] for row in rows])
    return labels, matrix


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["operator"]: row["group_sets"] for row in rows}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.18,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_known_structure(ax, labels: list[str]) -> None:
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.text(
        0.0,
        0.98,
        "Known DTLZ5 objective structure",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.0,
        0.87,
        "DTLZ5(I = 3, M = 5): one reducible block\nand two singleton objectives",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )

    xs = np.linspace(0.08, 0.92, len(labels))
    y = 0.48
    for idx, (x, lab) in enumerate(zip(xs, labels)):
        if idx < 3:
            face = COLORS["known"]
        elif idx == 3:
            face = "#829ab1"
        else:
            face = "#5c6f82"
        ax.scatter(x, y, s=330, color=face, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(x, y, lab, ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)

    block = patches.FancyBboxPatch(
        (xs[0] - 0.07, y - 0.18),
        xs[2] - xs[0] + 0.14,
        0.36,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=COLORS["known"],
        facecolor=COLORS["known_light"],
        zorder=1,
    )
    ax.add_patch(block)
    ax.text(
        xs[1],
        y - 0.28,
        "known correlated block",
        ha="center",
        va="top",
        fontsize=6.5,
        color=COLORS["known"],
        fontweight="bold",
    )

    ax.text(
        0.0,
        0.12,
        r"Expected groups: $\{f_1,f_2,f_3\}$, $\{f_4\}$, $\{f_5\}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=COLORS["ink"],
    )


def draw_matrix(ax, labels: list[str], matrix: np.ndarray) -> None:
    panel_label(ax, "b")
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(length=0)
    ax.set_title("Projected objective correlation matrix", fontsize=8.2, fontweight="bold", pad=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.74 else COLORS["ink"]
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.6, color=color, fontweight=weight)

    # Known block and detected Leiden block are intentionally both shown.
    ax.add_patch(patches.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=COLORS["known"], linewidth=1.8))
    ax.add_patch(
        patches.Rectangle(
            (-0.5, -0.5),
            4,
            4,
            fill=False,
            edgecolor=COLORS["detected"],
            linewidth=1.3,
            linestyle=(0, (2.4, 1.8)),
        )
    )
    ax.text(
        0.02,
        -0.19,
        "green box: known block; dashed orange: Leiden block",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=5.9,
        ha="left",
        va="top",
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)


def draw_graph(ax, labels: list[str], matrix: np.ndarray) -> None:
    panel_label(ax, "c")
    ax.set_axis_off()
    ax.set_title("Community call from the current implementation", fontsize=8.2, fontweight="bold", pad=8)

    pos = {
        0: (0.18, 0.62),
        1: (0.34, 0.77),
        2: (0.51, 0.61),
        3: (0.66, 0.43),
        4: (0.86, 0.25),
    }

    hull = patches.FancyBboxPatch(
        (0.10, 0.30),
        0.65,
        0.55,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.4,
        linestyle=(0, (3, 2)),
        edgecolor=COLORS["detected"],
        facecolor=COLORS["detected_light"],
        alpha=0.55,
        zorder=0,
    )
    ax.add_patch(hull)
    known = patches.FancyBboxPatch(
        (0.105, 0.52),
        0.48,
        0.32,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=COLORS["known"],
        facecolor=COLORS["known_light"],
        alpha=0.65,
        zorder=1,
    )
    ax.add_patch(known)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            w = matrix[i, j]
            if w < 0.5:
                continue
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            if i < 3 and j < 3:
                edge = COLORS["known"]
            elif j == 3:
                edge = COLORS["detected"]
            else:
                edge = "#9aa7b5"
            ax.plot([x1, x2], [y1, y2], color=edge, lw=0.6 + 3.2 * (w - 0.5), alpha=0.8, zorder=2)

    for i, lab in enumerate(labels):
        x, y = pos[i]
        face = COLORS["known"] if i < 3 else ("#829ab1" if i == 3 else "#5c6f82")
        ax.scatter(x, y, s=520, color=face, edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(x, y, lab, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=6)

    ax.text(
        0.11,
        0.90,
        r"known: $\{f_1,f_2,f_3\}$",
        ha="left",
        va="center",
        color=COLORS["known"],
        fontsize=6.7,
        fontweight="bold",
    )
    ax.text(
        0.11,
        0.25,
        r"Leiden output: $\{f_1,f_2,f_3,f_4\}$, $\{f_5\}$",
        ha="left",
        va="center",
        color=COLORS["detected"],
        fontsize=6.7,
        fontweight="bold",
    )
    ax.text(
        0.11,
        0.08,
        "Edges shown for correlation >= 0.50",
        ha="left",
        va="center",
        color=COLORS["muted"],
        fontsize=6.2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_operator_sensitivity(ax, labels: list[str], current: np.ndarray, star: np.ndarray) -> None:
    panel_label(ax, "d")
    pairs = [
        (0, 1, "within known block"),
        (0, 2, "within known block"),
        (1, 2, "within known block"),
        (0, 3, "intermediate coupling"),
        (1, 3, "intermediate coupling"),
        (2, 3, "intermediate coupling"),
        (0, 4, "near independent"),
        (3, 4, "near independent"),
    ]
    names = [f"{labels[i]}-{labels[j]}" for i, j, _ in pairs]
    y = np.arange(len(pairs))[::-1]
    current_vals = np.array([current[i, j] for i, j, _ in pairs])
    star_vals = np.array([star[i, j] for i, j, _ in pairs])

    for yi, cval, sval, (_, _, cat) in zip(y, current_vals, star_vals, pairs):
        if cat == "within known block":
            color = COLORS["known"]
        elif cat == "intermediate coupling":
            color = COLORS["detected"]
        else:
            color = "#7b8794"
        ax.plot([cval, sval], [yi, yi], color=color, lw=1.0, alpha=0.55)
        ax.scatter(cval, yi - 0.07, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(sval, yi + 0.07, s=34, marker="D", color=COLORS["star"], edgecolor="white", linewidth=0.45, zorder=4)

    ax.axvline(0.5, color="#b4bfca", lw=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y, names, fontsize=6.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("correlation strength", fontsize=7)
    ax.set_title("Current-code vs paper-text operator", fontsize=8.2, fontweight="bold", pad=8)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["known"], markeredgecolor="white", label="current-code", markersize=5),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["star"], markeredgecolor="white", label="paper-text", markersize=5),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=6.4, frameon=False)


def draw_outcome(ax) -> None:
    panel_label(ax, "e")
    ax.set_axis_off()
    ax.text(
        0,
        0.98,
        "Benchmark outcome",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
    )
    rows = [
        ("Known structure", r"$\{f_1,f_2,f_3\}$, $\{f_4\}$, $\{f_5\}$", COLORS["known"]),
        ("Current-code", r"$\{f_1,f_2,f_3,f_4\}$, $\{f_5\}$", COLORS["detected"]),
        ("Paper-text", r"$\{f_1,f_2,f_3,f_4\}$, $\{f_5\}$", COLORS["star"]),
    ]
    y0 = 0.76
    for idx, (left, right, color) in enumerate(rows):
        y = y0 - idx * 0.24
        ax.add_patch(
            patches.Rectangle((0.0, y - 0.085), 0.018, 0.13, transform=ax.transAxes, color=color, clip_on=False)
        )
        ax.text(0.04, y + 0.035, left, transform=ax.transAxes, ha="left", va="center", fontsize=6.7, fontweight="bold", color=COLORS["ink"])
        ax.text(0.04, y - 0.055, right, transform=ax.transAxes, ha="left", va="center", fontsize=6.7, color=COLORS["ink"])

    ax.text(
        0.0,
        0.01,
        "Interpretation: projected correlations recover the intended block,\n"
        "but the present community step also absorbs f4 (~0.55 correlations).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=COLORS["muted"],
        linespacing=1.25,
    )


def write_source_data(labels: list[str], current: np.ndarray, star: np.ndarray) -> None:
    pair_path = OUT_DIR / "figure3_dtlz5_pairwise_source_data.csv"
    with pair_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "i", "j", "current_code", "paper_text", "difference_paper_minus_current"])
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                writer.writerow(
                    [
                        f"{labels[i]}-{labels[j]}",
                        labels[i],
                        labels[j],
                        f"{current[i, j]:.10f}",
                        f"{star[i, j]:.10f}",
                        f"{star[i, j] - current[i, j]:.10f}",
                    ]
                )


def main() -> None:
    labels, current = read_matrix(CURRENT_MATRIX)
    labels_star, star = read_matrix(STAR_MATRIX)
    if labels != labels_star:
        raise ValueError("Matrix label mismatch between current-code and paper-text files.")
    read_summary(SUMMARY)
    write_source_data(labels, current, star)

    fig = plt.figure(figsize=(7.2, 5.75), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.14, 1.16, 1.28],
        height_ratios=[1.0, 1.02],
        left=0.055,
        right=0.985,
        bottom=0.07,
        top=0.81,
        wspace=0.38,
        hspace=0.52,
    )
    fig.text(
        0.055,
        0.965,
        "Figure 3. DTLZ5 known-structure benchmark",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.925,
        "Known structure, audited projected correlations, community output, and current-code versus paper-text operator comparison.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    draw_known_structure(ax_a, labels)
    draw_matrix(ax_b, labels, current)
    draw_graph(ax_c, labels, current)
    draw_operator_sensitivity(ax_d, labels, current, star)
    draw_outcome(ax_e)

    for ext, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600}),
    ]:
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Saved {OUT_BASE.with_suffix('.png')}")
    print(f"Saved {OUT_BASE.with_suffix('.svg')}")
    print(f"Saved {OUT_BASE.with_suffix('.pdf')}")
    print(f"Saved {OUT_BASE.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
