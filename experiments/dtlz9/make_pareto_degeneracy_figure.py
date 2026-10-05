"""Visualize the one-dimensional Pareto-front degeneracy of five-objective DTLZ9."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MM = 1.0 / 25.4
CURVE_COLOR = "#2F6F7E"
GROUP_SHADE = "#DCE8EB"
GREY = "#5F6368"


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.03, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def build_data(count: int = 301) -> pd.DataFrame:
    theta = np.linspace(0.0, 0.5 * np.pi, count)
    shared = np.cos(theta)
    f5 = np.sin(theta)
    return pd.DataFrame(
        {
            "theta_radians": theta,
            "f1": shared,
            "f2": shared,
            "f3": shared,
            "f4": shared,
            "f5": f5,
        }
    )


def make_figure(data: pd.DataFrame, output_dir: Path) -> None:
    shared = data["f1"].to_numpy()
    f5 = data["f5"].to_numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(165 * MM, 68 * MM),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(f5, shared, color=CURVE_COLOR, lw=3.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$f_5=\sin\theta$")
    ax.set_ylabel(r"$f_1=f_2=f_3=f_4=\cos\theta$")
    ax.text(
        0.05,
        0.08,
        r"$f_j=\sqrt{1-f_5^2},\ j=1,\ldots,4$",
        transform=ax.transAxes,
        color=GREY,
    )
    add_panel_label(ax, "a")

    ax = axes[1]
    objective_positions = np.arange(5)
    theta_parallel = np.linspace(0.08, 0.5 * np.pi - 0.08, 11)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5 * np.pi)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "slate_to_rose",
        ["#2F4858", "#526A7B", "#7A6F9B", "#B56576", "#E09F7D"],
    )
    for value in theta_parallel:
        objectives = [np.cos(value)] * 4 + [np.sin(value)]
        ax.plot(objective_positions, objectives, color=cmap(norm(value)), lw=1.5, alpha=0.95)
        ax.scatter(objective_positions, objectives, color=cmap(norm(value)), s=8, zorder=3)
    ax.set_xlim(-0.12, 4.12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(objective_positions, [r"$f_1$", r"$f_2$", r"$f_3$", r"$f_4$", r"$f_5$"])
    ax.set_ylabel("Objective value")
    ax.axvspan(-0.12, 3.12, color=GROUP_SHADE, alpha=0.55, lw=0)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_ticks([0.0, 0.25 * np.pi, 0.5 * np.pi])
    colorbar.set_ticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$"])
    colorbar.set_label(r"$\theta$")
    add_panel_label(ax, "b")

    stem = output_dir / "figure3_dtlz9_pareto_degeneracy"
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dtlz9_phase3/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = build_data()
    data.to_csv(args.output_dir / "figure3_source_data.csv", index=False)
    make_figure(data, args.output_dir)
    print(f"wrote DTLZ9 Pareto-front figure to {args.output_dir}")


if __name__ == "__main__":
    main()
