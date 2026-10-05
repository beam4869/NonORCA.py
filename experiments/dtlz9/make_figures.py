"""Create publication-ready Phase 2-3 figures from saved CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MM = 1.0 / 25.4
COLORS = {
    "Raw-Cosine": "#7A7A7A",
    "Single-Constraint-Mean": "#D18F00",
    "ORCA-current": "#D55E00",
    "ORCA-joint": "#0072B2",
    "Pearson": "#009E73",
    "Spearman": "#CC79A7",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def full_pf_pearson_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Pearson relation on the complete DTLZ9 front under uniform theta."""
    # On the exact M=5 front, f_1=...=f_4=cos(theta), f_5=sin(theta),
    # theta ~ Uniform[0, pi/2]. These moments give the population correlation
    # between cos(theta) and sin(theta), avoiding finite-sample dependence.
    cross_correlation = (1.0 / np.pi - 4.0 / np.pi**2) / (0.5 - 4.0 / np.pi**2)
    signed = np.ones((5, 5), dtype=float)
    signed[:4, 4] = cross_correlation
    signed[4, :4] = cross_correlation
    adjacency = 0.5 * (1.0 + signed)
    return signed, adjacency


def make_matrix_figure(results_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(results_dir / "phase2_representative_matrices.csv")
    methods = ["Raw-Cosine", "ORCA-current", "PF-Pearson"]
    titles = ["Raw objective gradients", "Current method", "PF Pearson correlation"]
    strength_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "orca_strength",
        ["#F7F4EF", "#DCE8EB", "#9BC4C6", "#4C8C92", "#23545B"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(165 * MM, 58 * MM), constrained_layout=True)
    image = None
    for index, (ax, method, title) in enumerate(zip(axes, methods, titles)):
        if method == "PF-Pearson":
            _, matrix = full_pf_pearson_matrices()
        else:
            part = frame[frame["method"] == method]
            matrix = part.pivot(index="objective_i", columns="objective_j", values="adjacency").to_numpy()
        if np.nanmin(matrix) < 0.0 or np.nanmax(matrix) > 1.0:
            raise ValueError(f"{method} correlation strengths must lie in [0, 1]")
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=strength_cmap, interpolation="nearest")
        ax.set_title(title, pad=5)
        ax.set_xticks(np.arange(5), labels=np.arange(1, 6))
        ax.set_yticks(np.arange(5), labels=np.arange(1, 6))
        ax.set_xlabel("Objective j")
        if index == 0:
            ax.set_ylabel("Objective i")
        else:
            ax.set_yticklabels([])
        for i in range(5):
            for j in range(5):
                value = matrix[i, j]
                color = "white" if value > 0.68 else "#202124"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color=color)
        panel_label(ax, chr(ord("a") + index))
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_label("Correlation strength")
    fig.suptitle("DTLZ9 exact Pareto front, M = 5 and n = 50", fontsize=9)
    save_all(fig, output_dir, "figure1_exact_pf_correlation_matrices")


def make_recovery_figure(results_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "phase3_grouping_recovery_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 128 * MM), constrained_layout=True)

    exact = summary[
        (summary["regime"] == "exact_pf")
        & (summary["block_size"] == 10)
        & (summary["grouping_mode"] == "unknown_K")
    ]
    ax = axes[0, 0]
    for method in ("ORCA-current", "ORCA-joint"):
        part = exact[exact["method"] == method].sort_values("M")
        ax.plot(part["M"], part["within_mean"], "o-", color=COLORS[method], label=f"{method}: within")
        ax.plot(part["M"], part["between_mean"], "s--", color=COLORS[method], label=f"{method}: between")
    ax.axhline(0, color="#BBBBBB", lw=0.7)
    ax.set_xticks([3, 5, 10, 20])
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("Number of objectives (M)")
    ax.set_ylabel("Mean signed edge strength")
    ax.set_title("Exact-front edge strengths")
    ax.legend(frameon=False, ncol=2, handlelength=2.4, columnspacing=0.8)
    panel_label(ax, "A")

    ax = axes[0, 1]
    for method in ("Raw-Cosine", "ORCA-current", "ORCA-joint"):
        part = exact[exact["method"] == method].sort_values("M")
        ax.plot(part["M"], part["estimated_groups_mean"], "o-", color=COLORS[method], label=method)
    ax.axhline(2, color="black", lw=0.8, ls=":", label="Truth")
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xlabel("Number of objectives (M)")
    ax.set_ylabel("Estimated number of groups")
    ax.set_title("Unknown-K recovery on exact front")
    ax.legend(frameon=False)
    panel_label(ax, "B")

    ax = axes[1, 0]
    methods = ("ORCA-current", "ORCA-joint", "Pearson")
    for regime_prefix, linestyle, label_suffix in (
        ("active_fraction_q", "-", "shared offset"),
        ("heterogeneous_contamination_q", "--", "heterogeneous"),
    ):
        for method in methods:
            part = summary[
                summary["regime"].str.startswith(regime_prefix)
                & (summary["M"] == 10)
                & (summary["block_size"] == 10)
                & (summary["grouping_mode"] == "unknown_K")
                & (summary["method"] == method)
            ].copy()
            if part.empty:
                continue
            part["q"] = part["regime"].str.extract(r"q([0-9.]+)").astype(float)
            part = part.sort_values("q")
            ax.plot(
                part["q"],
                part["exact_recovery_rate"],
                marker="o",
                ls=linestyle,
                color=COLORS[method],
                label=f"{method} ({label_suffix})",
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Designed fraction of active constraints (q)")
    ax.set_ylabel("Exact recovery rate")
    ax.set_title("Partial-intersection robustness, M = 10")
    method_handles = [
        mpl.lines.Line2D([], [], color=COLORS[method], marker="o", label=method)
        for method in methods
    ]
    regime_handles = [
        mpl.lines.Line2D([], [], color="black", ls="-", label="Shared offset"),
        mpl.lines.Line2D([], [], color="black", ls="--", label="Heterogeneous"),
    ]
    first_legend = ax.legend(handles=method_handles, frameon=False, loc="lower right", title="Method")
    ax.add_artist(first_legend)
    ax.legend(handles=regime_handles, frameon=False, loc="lower center", title="Contamination")
    panel_label(ax, "C")

    ax = axes[1, 1]
    selected = summary[
        (summary["regime"] == "orca_selected_points")
        & (summary["method"] == "ORCA-joint")
        & (summary["grouping_mode"] == "unknown_K")
    ].sort_values(["block_size", "M"])
    for block_size, marker in ((1, "o"), (10, "s")):
        part = selected[selected["block_size"] == block_size]
        ax.plot(
            part["M"], part["exact_recovery_rate"], marker=marker, color=COLORS["ORCA-joint"],
            label=f"Recovery, n/M={block_size}",
        )
        ax.plot(
            part["M"], part["active_fraction_mean"], marker=marker, ls="--", color="#666666",
            label=f"Full-intersection points, n/M={block_size}",
        )
    ax.set_xticks([3, 5, 10])
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Number of objectives (M)")
    ax.set_ylabel("Fraction")
    ax.set_title("Existing ORCA point selection")
    ax.legend(frameon=False)
    panel_label(ax, "D")

    save_all(fig, output_dir, "figure2_phase3_grouping_recovery")


def export_figure_data(results_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "phase3_grouping_recovery_summary.csv")
    keep = summary[
        summary["regime"].eq("exact_pf")
        | summary["regime"].str.startswith("active_fraction_q")
        | summary["regime"].str.startswith("heterogeneous_contamination_q")
        | summary["regime"].eq("orca_selected_points")
    ]
    keep.to_csv(output_dir / "figure2_source_data.csv", index=False)
    matrices = pd.read_csv(results_dir / "phase2_representative_matrices.csv")
    matrices = matrices[matrices["method"].isin(["Raw-Cosine", "ORCA-current"])].copy()
    signed, adjacency = full_pf_pearson_matrices()
    pearson_rows = []
    for i in range(5):
        for j in range(5):
            pearson_rows.append(
                {
                    "M": 5,
                    "block_size": np.nan,
                    "method": "PF-Pearson",
                    "objective_i": i + 1,
                    "objective_j": j + 1,
                    "signed_strength": signed[i, j],
                    "adjacency": adjacency[i, j],
                }
            )
    matrices = pd.concat([matrices, pd.DataFrame(pearson_rows)], ignore_index=True)
    matrices["normalized_correlation_strength"] = matrices["adjacency"]
    matrices["sampling_measure"] = np.where(
        matrices["method"].eq("PF-Pearson"),
        "complete exact front; theta uniform on [0, pi/2]; analytic population correlation",
        "pointwise derivative geometry",
    )
    matrices.to_csv(output_dir / "figure1_source_data.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/dtlz9_phase3"))
    args = parser.parse_args()
    output_dir = args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_matrix_figure(args.results_dir, output_dir)
    make_recovery_figure(args.results_dir, output_dir)
    export_figure_data(args.results_dir, output_dir)
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
