"""Create publication-ready DTLZ result figures with Matplotlib only."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".matplotlib-cache"),
)

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

COLORS = {
    "ink": "#17212B",
    "muted": "#68758A",
    "grid": "#D9DEE7",
    "full": "#5B6573",
    "mean": "#2F6B9A",
    "max": "#3A9D8F",
    "l2": "#8C6BB1",
    "truth": "#111111",
    "group1": "#477DB3",
    "group2": "#E69F4F",
    "group3": "#56A783",
    "group4": "#9B79B7",
    "group5": "#D66A65",
}
GROUP_COLORS = [COLORS[f"group{i}"] for i in range(1, 6)]


def panel_label(ax, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["ink"],
    )


def export_figure(fig: plt.Figure, stem: str) -> None:
    base = HERE / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def read_affinity() -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(HERE / "dtlz5_m16_affinity_seed0.csv", index_col=0)
    return frame.index.tolist(), frame.to_numpy(dtype=float)


def draw_affinity_network(ax, labels: list[str], matrix: np.ndarray) -> None:
    n_block = 12
    centre = np.array([-0.30, 0.0])
    radius = 0.58
    angles = np.linspace(math.pi * 0.92, math.pi * 2.92, n_block, endpoint=False)
    positions = {
        index: centre + radius * np.array([math.cos(angle), math.sin(angle)])
        for index, angle in enumerate(angles)
    }
    singleton_xy = [(0.62, 0.58), (0.79, 0.20), (0.79, -0.24), (0.60, -0.61)]
    positions.update({n_block + index: np.array(xy) for index, xy in enumerate(singleton_xy)})

    # The dense block is drawn completely; each singleton shows only its strongest
    # off-diagonal link.  The adjacent heatmap contains the unsparsified matrix.
    for i in range(n_block):
        for j in range(i + 1, n_block):
            weight = matrix[i, j]
            ax.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color=COLORS["group1"],
                lw=0.25 + 0.8 * weight,
                alpha=0.10,
                zorder=1,
            )

    labelled_edges: list[tuple[int, int, float]] = []
    for i in range(n_block, len(labels)):
        candidates = matrix[i].copy()
        candidates[i] = -np.inf
        j = int(np.argmax(candidates))
        weight = float(matrix[i, j])
        labelled_edges.append((i, j, weight))
        ax.plot(
            [positions[i][0], positions[j][0]],
            [positions[i][1], positions[j][1]],
            color=COLORS["muted"],
            lw=0.5 + 2.7 * weight,
            alpha=0.62,
            zorder=2,
        )

    for index, label in enumerate(labels):
        group_index = 0 if index < n_block else index - n_block + 1
        face = GROUP_COLORS[group_index]
        circle = Circle(
            positions[index],
            0.073,
            facecolor=face,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            positions[index][0],
            positions[index][1],
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=6.2,
            fontweight="bold",
            zorder=5,
        )

    # Label one representative within-block edge and all displayed singleton links.
    i, j = 0, 1
    midpoint = 0.5 * (positions[i] + positions[j])
    ax.text(midpoint[0] - 0.02, midpoint[1] + 0.03, f"{matrix[i, j]:.3f}", fontsize=5.5, color=COLORS["group1"])
    for edge_index, (i, j, weight) in enumerate(labelled_edges):
        midpoint = 0.5 * (positions[i] + positions[j])
        offsets = [(0.01, 0.045), (0.02, 0.02), (-0.12, -0.03), (0.00, -0.04)]
        dx, dy = offsets[edge_index]
        ax.text(midpoint[0] + dx, midpoint[1] + dy, f"{weight:.3f}", fontsize=5.5, color=COLORS["muted"])

    ax.text(-0.30, 0.72, r"recovered block $\{f_1,\ldots,f_{12}\}$", ha="center", fontsize=6.4, color=COLORS["group1"], fontweight="bold")
    ax.text(0.78, -0.78, "four singleton groups", ha="center", fontsize=6.2, color=COLORS["muted"])
    ax.text(
        -0.98,
        -0.86,
        "Dense block: all links; singletons: strongest link only\n(edge width encodes affinity; full matrix in b)",
        ha="left",
        va="bottom",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.90, 0.90)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Objective-affinity network", loc="left", fontsize=8, fontweight="bold", pad=2)


def draw_affinity_heatmap(ax, labels: list[str], matrix: np.ndarray) -> None:
    cmap = mpl.colormaps["viridis"]
    image = ax.imshow(matrix, vmin=0.15, vmax=1.0, cmap=cmap, interpolation="nearest")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.tick_params(length=0, pad=1.5)
    ax.add_patch(Rectangle((-0.5, -0.5), 12, 12, fill=False, ec=COLORS["group1"], lw=1.5))
    for index, color in zip(range(12, 16), GROUP_COLORS[1:]):
        ax.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, ec=color, lw=1.2))
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("objective affinity", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5, width=0.6)
    ax.set_title("Full affinity matrix", loc="left", fontsize=8, fontweight="bold", pad=2)


def relabel_in_order(values: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    output = np.empty_like(values)
    for index, value in enumerate(values):
        mapping.setdefault(int(value), len(mapping))
        output[index] = mapping[int(value)]
    return output


def draw_partition_comparison(ax) -> None:
    frame = pd.read_csv(HERE / "dtlz5_m16_partitions_seed0.csv")
    y_positions = [2.0, 1.0, 0.0]
    display_names = ["Known", "Average linkage\n(fixed K=5)", "Leiden\n(target K=5)"]
    for row_index, (_, row) in enumerate(frame.iterrows()):
        labels = np.asarray([int(value) for value in str(row["labels"]).split()], dtype=int)
        canonical = relabel_in_order(labels)
        y = y_positions[row_index]
        for objective, group in enumerate(canonical):
            ax.add_patch(
                Rectangle(
                    (objective, y - 0.32),
                    0.94,
                    0.64,
                    facecolor=GROUP_COLORS[int(group) % len(GROUP_COLORS)],
                    edgecolor="white",
                    linewidth=0.45,
                )
            )
        ax.text(-0.45, y, display_names[row_index], ha="right", va="center", fontsize=6.1)
        ax.text(
            16.20,
            y,
            f"K={int(row['num_groups'])}, ARI={float(row['ari_vs_known']):.3f}",
            ha="left",
            va="center",
            fontsize=5.8,
            color=COLORS["ink"] if int(row["exact_match"]) else COLORS["muted"],
        )
    for index in range(16):
        ax.text(index + 0.47, 2.45, f"f{index + 1}", ha="center", va="bottom", rotation=90, fontsize=4.9)
    ax.set_xlim(-3.1, 19.1)
    ax.set_ylim(-0.55, 2.72)
    ax.axis("off")
    ax.set_title("Partition recovery and algorithm dependence", loc="left", fontsize=8, fontweight="bold", pad=2)


def draw_grouping_robustness(ax) -> None:
    raw = pd.read_csv(ROOT / "data" / "dtlz" / "orca_dtlz5_dtlz6_gradient_grouping_raw.csv")
    summary = pd.read_csv(ROOT / "data" / "dtlz" / "orca_dtlz5_dtlz6_gradient_grouping_summary.csv")
    case_order = summary["dataset"].drop_duplicates().tolist()
    x = np.arange(len(case_order), dtype=float)
    methods = [("ORCA", COLORS["mean"], "o"), ("gradient_cosine", COLORS["full"], "s")]
    offsets = {"ORCA": -0.10, "gradient_cosine": 0.10}
    for method, color, marker in methods:
        subset = raw[raw["method"] == method]
        means = []
        for case_index, case in enumerate(case_order):
            values = subset.loc[subset["dataset"] == case, "grouping_seconds"].to_numpy(dtype=float) * 1000.0
            means.append(np.mean(values))
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(
                np.full(len(values), x[case_index] + offsets[method]) + jitter,
                values,
                s=8,
                color=color,
                alpha=0.35,
                linewidth=0,
                zorder=2,
            )
        ax.plot(x + offsets[method], means, color=color, marker=marker, ms=4, lw=1.2, label="ORCA affinity" if method == "ORCA" else "Gradient cosine")

    short_labels = []
    for case in case_order:
        family = "5" if case.startswith("DTLZ5") else "6"
        objectives = "16" if ",16" in case else "20"
        tail = case.split("k_tail=")[1]
        short_labels.append(f"D{family}\nM{objectives}\nk{tail}")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=5.3)
    ax.set_ylabel("Grouping time (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=5.7)
    ax.text(
        0.98,
        0.96,
        "ARI = 1.00 and exact match = 100%\nfor every case, method and seed (60/60)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=COLORS["ink"],
    )
    ax.set_title("Recovery is stable; cost increases for long tails", loc="left", fontsize=8, fontweight="bold", pad=2)


def make_grouping_figure() -> None:
    labels, matrix = read_affinity()
    fig = plt.figure(figsize=(7.2, 5.45))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.22, 0.78], wspace=0.24, hspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    draw_affinity_network(ax_a, labels, matrix)
    draw_affinity_heatmap(ax_b, labels, matrix)
    draw_partition_comparison(ax_c)
    draw_grouping_robustness(ax_d)
    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08)
    export_figure(fig, "figure_dtlz_grouping")


METHOD_STYLE = {
    "full_nsga3": ("Full", COLORS["full"], "s", "filled"),
    "orca_nsga3_mean_natural": ("Mean / natural", COLORS["mean"], "o", "open"),
    "orca_nsga3_mean_equal_eval": ("Mean / equal eval", COLORS["mean"], "o", "filled"),
    "orca_nsga3_max_natural": ("Max / natural", COLORS["max"], "^", "open"),
    "orca_nsga3_max_equal_eval": ("Max / equal eval", COLORS["max"], "^", "filled"),
    "orca_nsga3_l2_natural": ("L2 / natural", COLORS["l2"], "D", "open"),
    "orca_nsga3_l2_equal_eval": ("L2 / equal eval", COLORS["l2"], "D", "filled"),
}


def draw_time_quality(ax, raw: pd.DataFrame, title: str, y_limits: tuple[float, float]) -> None:
    present_methods = [method for method in METHOD_STYLE if method in set(raw["method"])]
    annotation_offsets = {
        "full_nsga3": (4, 5),
        "orca_nsga3_mean_natural": (4, 5),
        "orca_nsga3_mean_equal_eval": (-50, 8),
        "orca_nsga3_max_natural": (4, -9),
        "orca_nsga3_max_equal_eval": (4, -10),
        "orca_nsga3_l2_natural": (4, 5),
        "orca_nsga3_l2_equal_eval": (-45, -10),
    }
    means: dict[str, tuple[float, float]] = {}
    for method in present_methods:
        subset = raw[raw["method"] == method]
        x = subset["total_seconds"].to_numpy(dtype=float)
        y = subset["empirical_igd"].to_numpy(dtype=float)
        label, color, marker, fill = METHOD_STYLE[method]
        face = color if fill == "filled" else "white"
        ax.scatter(x, y, s=12, marker=marker, facecolor=face, edgecolor=color, linewidth=0.65, alpha=0.38, zorder=2)
        x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
        means[method] = (x_mean, y_mean)
        ax.errorbar(
            x_mean,
            y_mean,
            xerr=float(np.std(x, ddof=0)),
            yerr=float(np.std(y, ddof=0)),
            fmt=marker,
            ms=5.0,
            mfc=face,
            mec=color,
            mew=0.9,
            ecolor=color,
            elinewidth=0.75,
            capsize=2,
            zorder=4,
        )
        dx, dy = annotation_offsets[method]
        ax.annotate(label, (x_mean, y_mean), xytext=(dx, dy), textcoords="offset points", fontsize=5.3, color=color)

    for aggregation in ("mean", "max", "l2"):
        natural = f"orca_nsga3_{aggregation}_natural"
        equal = f"orca_nsga3_{aggregation}_equal_eval"
        if natural in means and equal in means:
            ax.plot(
                [means[natural][0], means[equal][0]],
                [means[natural][1], means[equal][1]],
                color=METHOD_STYLE[natural][1],
                lw=0.8,
                alpha=0.45,
                zorder=1,
            )
    ax.set_xlim(0, max(15.8, raw["total_seconds"].max() * 1.06))
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Total time (s)")
    ax.set_ylabel("Empirical IGD in original 16-objective space")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=2)
    ax.text(0.03, 0.04, "lower and left are better", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=COLORS["muted"])


def draw_dtlz6_g(ax, raw: pd.DataFrame) -> None:
    order = [method for method in METHOD_STYLE if method in set(raw["method"])]
    y_positions = np.arange(len(order))[::-1]
    for y, method in zip(y_positions, order):
        subset = raw[raw["method"] == method]
        values = subset["g_median"].to_numpy(dtype=float)
        label, color, marker, fill = METHOD_STYLE[method]
        face = color if fill == "filled" else "white"
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, s=14, marker=marker, facecolor=face, edgecolor=color, linewidth=0.7, alpha=0.7)
        ax.plot([float(np.mean(values))], [y], marker="|", ms=9, mew=1.5, color=color)
    ax.axvline(0.0, color=COLORS["truth"], lw=1.0, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_STYLE[method][0] for method in order], fontsize=5.8)
    ax.set_xlabel(r"Median distance-function value $g$ (target = 0)")
    ax.set_xlim(-0.25, max(9.0, raw["g_median"].max() * 1.05))
    ax.set_title("DTLZ6 convergence, not grouping, is the bottleneck", loc="left", fontsize=8, fontweight="bold", pad=2)
    ax.text(
        0.98,
        0.06,
        "Every method recovered the known partition (ARI = 1),\nbut none reached the g = 0 manifold within this budget.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=COLORS["muted"],
    )


def make_downstream_figure() -> None:
    aggregation = pd.read_csv(ROOT / "data" / "dtlz" / "orca_dtlz5_dtlz6_m16_aggregation_raw.csv")
    dtlz5 = aggregation[aggregation["family"] == "DTLZ5"].copy()
    dtlz6 = pd.read_csv(ROOT / "data" / "dtlz" / "orca_dtlz6_m16_l2_aggregation_raw.csv")
    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.42, wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    draw_time_quality(ax_a, dtlz5, "DTLZ5(5,16): reduction improves quality", (0.08, 0.53))
    draw_time_quality(ax_b, dtlz6, "DTLZ6(5,16): speed–quality trade-off", (0.22, 0.81))
    draw_dtlz6_g(ax_c, dtlz6)
    for label, ax in zip("abc", [ax_a, ax_b, ax_c]):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.10)
    export_figure(fig, "figure_dtlz_downstream")


def true_front_collapsed(num_points: int = 800) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 0.5 * math.pi, num_points)
    return np.cos(theta), np.sin(theta)


def draw_pareto_arc(ax, solutions: pd.DataFrame, representative_seed: int) -> None:
    x_true, y_true = true_front_collapsed()
    ax.plot(x_true, y_true, color=COLORS["truth"], lw=1.5, label="Analytic true front")
    styles = [
        ("Full NSGA-III", COLORS["full"], "s"),
        ("ORCA-reduced NSGA-III (L2)", COLORS["mean"], "o"),
    ]
    for method, color, marker in styles:
        subset = solutions[
            (solutions["seed"] == representative_seed) & (solutions["method"] == method)
        ].copy()
        ax.scatter(
            subset["collapsed_block_l2"],
            subset["f3"],
            s=20 if method.startswith("ORCA") else 18,
            marker=marker,
            facecolor=color if method.startswith("ORCA") else "white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.92,
            label=method,
            zorder=3,
        )
        farthest = subset.nlargest(min(5, len(subset)), "distance_to_true_front")
        nearest_x = np.sqrt(farthest["nearest_true_f1"] ** 2 + farthest["nearest_true_f2"] ** 2)
        for x0, y0, x1, y1 in zip(farthest["collapsed_block_l2"], farthest["f3"], nearest_x, farthest["nearest_true_f3"]):
            ax.plot([x0, x1], [y0, y1], color=color, lw=0.45, alpha=0.45, zorder=1)
    ax.set_xlabel(r"Collapsed correlated block, $(f_1^2+f_2^2)^{1/2}$")
    ax.set_ylabel(r"Singleton objective, $f_3$")
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.02, 1.04)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=5.7)
    ax.set_title("DTLZ5(2,3): solutions versus analytic front", loc="left", fontsize=8, fontweight="bold", pad=2)
    ax.text(
        0.98,
        0.98,
        f"representative paired run: seed {representative_seed}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=COLORS["muted"],
    )


def draw_paired_metric(ax, metrics: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    pivot = metrics.pivot(index="seed", columns="method", values=metric)
    full = pivot["Full NSGA-III"].to_numpy(dtype=float)
    reduced = pivot["ORCA-reduced NSGA-III (L2)"].to_numpy(dtype=float)
    offsets = np.linspace(-0.035, 0.035, len(full))
    for offset, (left, right) in zip(offsets, zip(full, reduced)):
        ax.plot([offset, 1 + offset], [left, right], color=COLORS["grid"], lw=0.9, zorder=1)
        ax.scatter([offset], [left], s=19, marker="s", facecolor="white", edgecolor=COLORS["full"], linewidth=0.8, zorder=2)
        ax.scatter([1 + offset], [right], s=19, marker="o", facecolor=COLORS["mean"], edgecolor=COLORS["mean"], linewidth=0.8, zorder=2)
    ax.scatter([0, 1], [np.mean(full), np.mean(reduced)], s=55, marker="_", color=[COLORS["full"], COLORS["mean"]], linewidth=2.0, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full", "ORCA\nreduced"], fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=2)
    ratio = np.mean(reduced) / np.mean(full)
    ax.text(0.98, 0.96, f"mean ratio = {ratio:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color=COLORS["mean"])


def make_pareto_figure() -> None:
    solutions = pd.read_csv(HERE / "dtlz3_pareto_solutions.csv")
    metrics = pd.read_csv(HERE / "dtlz3_pareto_metrics.csv")
    solutions = solutions[solutions["family"] == "DTLZ5"].copy()
    metrics = metrics[metrics["family"] == "DTLZ5"].copy()
    igd = metrics.pivot(index="seed", columns="method", values="igd_true")
    paired_improvement = igd["Full NSGA-III"] - igd["ORCA-reduced NSGA-III (L2)"]
    representative_seed = int((paired_improvement - paired_improvement.median()).abs().idxmin())
    fig = plt.figure(figsize=(7.2, 2.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.82, 0.82], wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    draw_pareto_arc(ax_a, solutions, representative_seed)
    draw_paired_metric(ax_b, metrics, "gd_true", "Convergence to front", "GD to true front")
    draw_paired_metric(ax_c, metrics, "igd_true", "Front coverage", "IGD from true front")
    for label, ax in zip("abc", [ax_a, ax_b, ax_c]):
        panel_label(ax, label)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.20)
    export_figure(fig, "figure_dtlz_true_front")


def main() -> None:
    make_grouping_figure()
    make_downstream_figure()
    make_pareto_figure()
    print(f"Figures written to {HERE}")


if __name__ == "__main__":
    main()
