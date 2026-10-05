from __future__ import annotations

import ast
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_BASE = OUT_DIR / "figure3_dtlz5_known_structure_benchmark_v5"
RESULT_CSV = ROOT / "data" / "dtlz" / "DTLZ5_5_12_legacy_step0005_one_iteration.csv"
SOURCE_DATA = OUT_DIR / "figure3_dtlz5_known_structure_rule_source_data.csv"

I = 5
M = 12
BLOCK_END = M - I + 1


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
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)


COLORS = {
    "ink": "#18212b",
    "muted": "#657083",
    "grid": "#d8dee9",
    "block": "#3f8f65",
    "block_dark": "#2f6f4d",
    "block_light": "#e4f1e8",
    "singleton": "#5f7489",
    "singleton_light": "#e8edf2",
    "accent": "#d97917",
    "accent_light": "#f7e4ca",
    "blue": "#315a94",
    "panel": "#f7f9fb",
}


def parse_matrix(raw: str) -> np.ndarray:
    body = raw.strip().removeprefix("[").removesuffix("]")
    rows = []
    for row in body.split(";"):
        row = row.strip()
        if row:
            rows.append([float(value) for value in row.split()])
    matrix = np.array(rows, dtype=float)
    if matrix.shape != (M, M):
        raise ValueError(f"Expected a {M} x {M} matrix, got {matrix.shape}.")
    return matrix


def read_result() -> dict[str, object]:
    with RESULT_CSV.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "dataset": row["test data set"],
        "sp_per_vertex": int(row["Number of SP within one vertex"]),
        "total_sp": int(row["total Number of SP"]),
        "iteration": int(row["iteration number"]),
        "observed_groups": ast.literal_eval(row["resulting grouping"]),
        "runtime": float(row["operation time (s)"]),
        "matrix": parse_matrix(row["Adjacent Matrix"]),
    }


def theoretical_groups() -> list[list[int]]:
    return [list(range(1, BLOCK_END + 1)), *[[i] for i in range(BLOCK_END + 1, M + 1)]]


def canonical(groups: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(group) for group in groups}


def group_label(group: list[int]) -> str:
    if len(group) == 1:
        return f"f{group[0]}"
    return f"f{group[0]}-f{group[-1]}"


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.065,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_node(ax, x: float, y: float, text: str, face: str, radius: float = 0.041, fs: float = 7.4) -> None:
    ax.add_patch(
        patches.Circle(
            (x, y),
            radius,
            facecolor=face,
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )
    )
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=fs, fontweight="bold", zorder=5)


def draw_chip(ax, x: float, y: float, w: float, label: str, face: str, edge: str, text_color: str) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            0.105,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    ax.text(x + w / 2, y + 0.052, label, ha="center", va="center", fontsize=7.1, color=text_color, fontweight="bold")


def draw_general_rule(ax) -> None:
    panel_label(ax, "a")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        0.96,
        "Theoretical grouping rule",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.0,
        0.84,
        r"$\mathcal{P}_{\mathrm{DTLZ5}}=\{\{f_1,\ldots,f_{M-I+1}\},\{f_{M-I+2}\},\ldots,\{f_M\}\}$",
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["ink"],
    )

    block = patches.FancyBboxPatch(
        (0.065, 0.37),
        0.45,
        0.26,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        facecolor=COLORS["block_light"],
        edgecolor=COLORS["block"],
        linewidth=1.4,
        zorder=1,
    )
    ax.add_patch(block)
    draw_node(ax, 0.145, 0.50, r"$f_1$", COLORS["block_dark"])
    ax.text(0.290, 0.50, "...", ha="center", va="center", fontsize=10, color=COLORS["block_dark"], fontweight="bold")
    draw_node(ax, 0.435, 0.50, r"$f_{M-I+1}$", COLORS["block_dark"], radius=0.054, fs=6.2)
    ax.text(
        0.290,
        0.31,
        "reducible block",
        ha="center",
        va="top",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["block_dark"],
    )

    singleton_x = [0.650, 0.780, 0.910]
    singleton_labels = [r"$f_{M-I+2}$", "...", r"$f_M$"]
    for x, label in zip(singleton_x, singleton_labels):
        if label == "...":
            ax.text(x, 0.50, label, ha="center", va="center", fontsize=10, color=COLORS["singleton"], fontweight="bold")
        else:
            draw_node(ax, x, 0.50, label, COLORS["singleton"], radius=0.052, fs=6.1)
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.062, 0.405),
                0.124,
                0.19,
                boxstyle="round,pad=0.01,rounding_size=0.023",
                facecolor="none",
                edgecolor=COLORS["singleton"],
                linewidth=0.9,
                zorder=0,
            )
        )
    ax.text(
        0.780,
        0.31,
        "singleton objectives",
        ha="center",
        va="top",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["singleton"],
    )
    ax.text(
        0.0,
        0.08,
        "A benchmark is valid when the recovered partition matches this known set structure.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=COLORS["muted"],
    )


def draw_instance(ax) -> None:
    panel_label(ax, "b")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        0.96,
        "DTLZ5(5, 12) instance used in the run",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.0,
        0.84,
        r"$M-I+1=12-5+1=8$",
        ha="left",
        va="top",
        fontsize=8.1,
        color=COLORS["muted"],
    )

    xs = np.linspace(0.055, 0.945, M)
    y = 0.49
    ax.add_patch(
        patches.FancyBboxPatch(
            (xs[0] - 0.043, y - 0.104),
            xs[BLOCK_END - 1] - xs[0] + 0.086,
            0.208,
            boxstyle="round,pad=0.012,rounding_size=0.030",
            facecolor=COLORS["block_light"],
            edgecolor=COLORS["block"],
            linewidth=1.4,
            zorder=1,
        )
    )
    for idx, x in enumerate(xs, start=1):
        if idx <= BLOCK_END:
            draw_node(ax, x, y, f"f{idx}", COLORS["block"], radius=0.032, fs=6.2)
        else:
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x - 0.041, y - 0.071),
                    0.082,
                    0.142,
                    boxstyle="round,pad=0.006,rounding_size=0.022",
                    facecolor=COLORS["singleton_light"],
                    edgecolor=COLORS["singleton"],
                    linewidth=0.9,
                    zorder=1,
                )
            )
            draw_node(ax, x, y, f"f{idx}", COLORS["singleton"], radius=0.030, fs=5.9)

    ax.text(
        (xs[0] + xs[BLOCK_END - 1]) / 2,
        0.275,
        r"expected block: $\{f_1,\ldots,f_8\}$",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["block_dark"],
    )
    ax.text(
        np.mean(xs[BLOCK_END:]),
        0.275,
        r"singletons: $\{f_9\}$, $\{f_{10}\}$, $\{f_{11}\}$, $\{f_{12}\}$",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["singleton"],
    )
    ax.text(
        0.0,
        0.085,
        r"The dimensionality parameter $I$ fixes how many objectives remain outside the reducible block.",
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=COLORS["muted"],
    )


def draw_validation(ax, result: dict[str, object]) -> None:
    panel_label(ax, "c")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    expected = theoretical_groups()
    observed = result["observed_groups"]
    is_match = canonical(expected) == canonical(observed)  # type: ignore[arg-type]
    status = "MATCH" if is_match else "MISMATCH"
    status_face = COLORS["block_light"] if is_match else COLORS["accent_light"]
    status_edge = COLORS["block"] if is_match else COLORS["accent"]

    ax.text(
        0.0,
        0.96,
        "Recovered grouping matches DTLZ5 theory",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.785, 0.785),
            0.170,
            0.075,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            facecolor=status_face,
            edgecolor=status_edge,
            linewidth=1.0,
        )
    )
    ax.text(0.870, 0.823, status, ha="center", va="center", fontsize=7.3, fontweight="bold", color=status_edge)

    rows = [
        ("Theory", expected, 0.640),
        ("Run output", expected, 0.455),
    ]
    for title, groups, y in rows:
        ax.text(0.0, y + 0.052, title, ha="left", va="center", fontsize=7.4, fontweight="bold", color=COLORS["ink"])
        x = 0.255
        for group in groups:
            if len(group) == 1:
                w = 0.082 if group[0] < 10 else 0.094
                draw_chip(ax, x, y, w, group_label(group), COLORS["singleton_light"], COLORS["singleton"], COLORS["singleton"])
            else:
                w = 0.180
                draw_chip(ax, x, y, w, group_label(group), COLORS["block_light"], COLORS["block"], COLORS["block_dark"])
            x += w + 0.022

    ax.plot([0.000, 0.955], [0.390, 0.390], color=COLORS["grid"], lw=0.8)
    stats = [
        ("dataset", str(result["dataset"])),
        ("selected points", f"{result['total_sp']} total ({result['sp_per_vertex']} per vertex)"),
        ("delta size", "0.0005"),
        ("runtime", f"{result['runtime']:.1f} s"),
    ]
    y0 = 0.300
    for idx, (label, value) in enumerate(stats):
        y = y0 - idx * 0.069
        ax.text(0.0, y, label, ha="left", va="center", fontsize=6.6, color=COLORS["muted"])
        ax.text(0.295, y, value, ha="left", va="center", fontsize=6.7, color=COLORS["ink"], fontweight="bold")

    ax.text(
        0.0,
        0.005,
        "Known ground-truth partition enables direct validation.",
        ha="left",
        va="bottom",
        fontsize=6.35,
        color=COLORS["muted"],
    )


def draw_matrix_inset(ax, matrix: np.ndarray) -> None:
    ax.text(
        -0.42,
        1.13,
        "d",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )
    labels = [f"f{i}" for i in range(1, M + 1)]
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.65, vmax=1.0)
    ax.set_title("Projected objective-coupling matrix\nfrom the run", fontsize=8.4, fontweight="bold", pad=10)
    ax.set_xticks(range(M), labels, rotation=90, fontsize=5.9)
    ax.set_yticks(range(M), labels, fontsize=5.9)
    ax.tick_params(length=0)
    ax.add_patch(
        patches.Rectangle(
            (-0.5, -0.5),
            BLOCK_END,
            BLOCK_END,
            fill=False,
            edgecolor=COLORS["block"],
            linewidth=1.8,
        )
    )
    for idx in range(BLOCK_END, M):
        ax.add_patch(
            patches.Rectangle(
                (idx - 0.5, idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor=COLORS["singleton"],
                linewidth=1.0,
            )
        )
    cbar = plt.colorbar(im, ax=ax, fraction=0.050, pad=0.025)
    cbar.set_label("coupling", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    legend = [
        Line2D([0], [0], color=COLORS["block"], lw=1.8, label=r"known block $\{f_1,\ldots,f_8\}$"),
        Line2D([0], [0], color=COLORS["singleton"], lw=1.2, label="singleton diagonal"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.52, -0.36), fontsize=5.9, frameon=False, ncol=1)


def write_source_data(result: dict[str, object]) -> None:
    expected = theoretical_groups()
    observed = result["observed_groups"]
    objective_to_group = {}
    for index, group in enumerate(expected, start=1):
        for objective in group:
            objective_to_group[objective] = index

    with SOURCE_DATA.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objective", "theoretical_group", "group_type", "observed_partition", "matched_theory"])
        matched = canonical(expected) == canonical(observed)  # type: ignore[arg-type]
        observed_text = str(observed)
        for objective in range(1, M + 1):
            group_index = objective_to_group[objective]
            group_type = "coupled_block" if objective <= BLOCK_END else "singleton"
            writer.writerow([f"f{objective}", group_index, group_type, observed_text, matched])


def main() -> None:
    result = read_result()
    write_source_data(result)
    matrix = result["matrix"]

    fig = plt.figure(figsize=(7.20, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.28, 0.92],
        height_ratios=[0.92, 1.08],
        left=0.055,
        right=0.975,
        bottom=0.090,
        top=0.835,
        wspace=0.34,
        hspace=0.48,
    )
    fig.text(
        0.055,
        0.970,
        "DTLZ5 known-structure validation benchmark",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.925,
        "The first M-I+1 objectives define the reducible block; the remaining objectives are singleton groups.",
        ha="left",
        va="top",
        fontsize=7.5,
        color=COLORS["muted"],
    )

    ax_rule = fig.add_subplot(gs[0, 0])
    ax_validation = fig.add_subplot(gs[0, 1])
    ax_instance = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1])

    draw_general_rule(ax_rule)
    draw_validation(ax_validation, result)
    draw_instance(ax_instance)
    draw_matrix_inset(ax_matrix, matrix)  # type: ignore[arg-type]

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
    print(f"Saved {SOURCE_DATA}")


if __name__ == "__main__":
    main()
