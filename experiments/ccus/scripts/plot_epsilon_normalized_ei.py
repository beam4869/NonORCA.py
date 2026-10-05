"""Screen and plot the augmented epsilon-constraint TAC--J frontier."""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_epsilon_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

DOMINANCE_TOLERANCE = 1.0e-7
DUPLICATE_TOLERANCE = 1.0e-6
COLORS = {
    "urea": "#D06C78",
    "saline": "#4C78A8",
    "other": "#9AA1AC",
    "frontier": "#5A6170",
    "weighted": "#242832",
    "grid": "#E5E8ED",
    "text": "#303640",
    "shade": "#F4F6F9",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(
            values <= point + DOMINANCE_TOLERANCE, axis=1
        ) & np.any(values < point - DOMINANCE_TOLERANCE, axis=1)
        keep[index] = not np.any(dominates)
    return keep


def deduplicate(frontier: pd.DataFrame) -> pd.DataFrame:
    ordered = frontier.sort_values(["TAC_normalized", "J_outer_normalized"]).copy()
    values = ordered[["TAC_normalized", "J_outer_normalized"]].to_numpy(float)
    positions: list[int] = []
    for position, point in enumerate(values):
        if not positions or np.min(
            np.max(np.abs(values[positions] - point), axis=1)
        ) > DUPLICATE_TOLERANCE:
            positions.append(position)
    return ordered.iloc[positions].reset_index(drop=True)


def knee_index(frontier: pd.DataFrame) -> int:
    values = frontier[["TAC_normalized", "J_outer_normalized"]].to_numpy(float)
    start, end = values[0], values[-1]
    chord = end - start
    offsets = values - start
    cross_product = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    distances = np.abs(cross_product) / np.linalg.norm(chord)
    return int(np.argmax(distances))


def load_and_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_raw.csv"
    )
    keep = nondominated_mask(
        raw[["TAC_normalized", "J_outer_normalized"]].to_numpy(float)
    )
    frontier = deduplicate(raw.loc[keep])
    frontier.insert(
        0,
        "frontier_id",
        [f"EC{index:03d}" for index in range(1, len(frontier) + 1)],
    )
    frontier["point_role"] = "interior"
    frontier.loc[0, "point_role"] = "minimum_TAC"
    frontier.loc[len(frontier) - 1, "point_role"] = "minimum_J"
    knee = knee_index(frontier)
    frontier.loc[knee, "point_role"] = "knee"
    switches = frontier["main_sink"].ne(frontier["main_sink"].shift())
    switches.iloc[0] = False
    frontier["sink_switch"] = switches

    frontier.to_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_frontier.csv",
        index=False,
    )
    summary_columns = [
        "frontier_id",
        "point_role",
        "sink_switch",
        "epsilon_index",
        "epsilon_TAC",
        "TAC",
        "TotEmiss",
        "ISI",
        "J_normalized",
        "TAC_normalized",
        "J_outer_normalized",
        "main_sink",
        "active_sinks",
        "pretreated_fraction",
        "direct_fraction",
        "sink1_share",
        "sink2_share",
        "sink3_share",
        "sink4_share",
        "sink5_share",
        "sink6_share",
        "solution_source",
        "max_abs_equality",
        "max_abs_source_carbon_diagnostic",
        "max_inequality_violation",
        "epsilon_violation",
    ]
    frontier[summary_columns].to_csv(
        RESULT_DIR / "direct_use_expansion_epsilon_normalized_frontier_summary.csv",
        index=False,
    )

    weighted_path = (
        RESULT_DIR / "direct_use_expansion_weighted_sum_supported_frontier.csv"
    )
    weighted = pd.read_csv(weighted_path) if weighted_path.exists() else pd.DataFrame()
    return raw, frontier, weighted


def annotate_point(ax: plt.Axes, row: pd.Series, label: str, offset: tuple[int, int]) -> None:
    ax.annotate(
        label,
        (row["TAC"] / 1.0e6, row["J_normalized"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=6.0,
        color=COLORS["text"],
        arrowprops={"arrowstyle": "-", "color": COLORS["frontier"], "linewidth": 0.65},
    )


def plot_results(frontier: pd.DataFrame, weighted: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.93, wspace=0.31)

    ax = axes[0]
    ax.plot(
        frontier["TAC"] / 1.0e6,
        frontier["J_normalized"],
        color=COLORS["frontier"],
        linewidth=1.05,
        alpha=0.78,
        zorder=1,
    )
    sink_style = {
        "Urea": (COLORS["urea"], "o"),
        "Saline Storage": (COLORS["saline"], "s"),
    }
    for sink, group in frontier.groupby("main_sink", sort=False):
        color, marker = sink_style.get(sink, (COLORS["other"], "D"))
        ax.scatter(
            group["TAC"] / 1.0e6,
            group["J_normalized"],
            s=18,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.45,
            label=f"ε-constraint: {sink}",
            zorder=3,
        )

    if not weighted.empty:
        ax.scatter(
            weighted["TAC"] / 1.0e6,
            weighted["J_normalized"],
            s=58,
            marker="*",
            color=COLORS["weighted"],
            edgecolor="white",
            linewidth=0.55,
            label="Weighted-sum supported points",
            zorder=5,
        )

    minimum_tac = frontier.iloc[0]
    minimum_j = frontier.iloc[-1]
    knee = frontier.loc[frontier["point_role"].eq("knee")].iloc[0]
    annotate_point(
        ax,
        minimum_tac,
        f"Minimum TAC\n{minimum_tac['TAC'] / 1e6:.2f} M, J={minimum_tac['J_normalized']:.3f}",
        (8, -6),
    )
    annotate_point(
        ax,
        knee,
        f"Knee\n{knee['TAC'] / 1e6:.2f} M, J={knee['J_normalized']:.3f}",
        (-2, 25),
    )
    annotate_point(
        ax,
        minimum_j,
        f"Minimum J\n{minimum_j['TAC'] / 1e6:.2f} M, J={minimum_j['J_normalized']:.3f}",
        (-72, 11),
    )
    ax.set_xlabel(r"TAC ($10^6$ cost units yr$^{-1}$)")
    ax.set_ylabel(r"Normalized grouped objective, $J=0.5\hat{E}+0.5\widehat{ISI}$")
    ax.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=5.7, handletextpad=0.5)
    ax.text(-0.17, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1]
    x = frontier["TAC"].to_numpy(float) / 1.0e6
    saline = frontier["sink3_share"].clip(0.0, 1.0).to_numpy(float)
    urea = frontier["sink5_share"].clip(0.0, 1.0).to_numpy(float)
    other = np.clip(1.0 - saline - urea, 0.0, 1.0)
    ax.fill_between(x, 0.0, saline, color=COLORS["saline"], alpha=0.13)
    ax.fill_between(x, 1.0 - urea, 1.0, color=COLORS["urea"], alpha=0.13)
    ax.plot(x, saline, color=COLORS["saline"], linewidth=1.65, label="Saline Storage")
    ax.plot(x, urea, color=COLORS["urea"], linewidth=1.65, label="Urea")
    if np.max(other) > 1.0e-4:
        ax.plot(x, other, color=COLORS["other"], linewidth=1.2, label="Other sinks")
    switch_rows = frontier.loc[frontier["sink_switch"]]
    if not switch_rows.empty:
        switch = switch_rows.iloc[0]
        switch_x = switch["TAC"] / 1.0e6
        ax.axvline(switch_x, color=COLORS["frontier"], linestyle="--", linewidth=0.8)
        ax.text(
            switch_x - 0.14,
            0.56,
            f"Main-sink switch\nTAC={switch_x:.2f} M",
            fontsize=6.0,
            color=COLORS["text"],
            va="center",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    ax.set_xlabel(r"TAC ($10^6$ cost units yr$^{-1}$)")
    ax.set_ylabel("Fraction of captured CO$_2$ sent to sink")
    ax.set_ylim(-0.035, 1.035)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=5.8)
    ax.text(-0.16, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

    stem = FIGURE_DIR / "direct_use_expansion_epsilon_normalized_frontier"
    for extension in ("png", "svg", "pdf", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_analysis(raw: pd.DataFrame, frontier: pd.DataFrame) -> None:
    knee = frontier.loc[frontier["point_role"].eq("knee")].iloc[0]
    switch_rows = frontier.loc[frontier["sink_switch"]]
    switch_text = "No main-sink switch was found."
    if not switch_rows.empty:
        switch = switch_rows.iloc[0]
        previous = frontier.loc[switch.name - 1]
        switch_text = (
            f"The main sink changes from {previous['main_sink']} to "
            f"{switch['main_sink']} between TAC={previous['TAC'] / 1e6:.3f} and "
            f"{switch['TAC'] / 1e6:.3f} million cost units per year."
        )
    residuals = raw[
        [
            "max_abs_equality",
            "max_abs_source_carbon_diagnostic",
            "max_inequality_violation",
            "epsilon_violation",
        ]
    ].max()
    source_counts = raw["solution_source"].value_counts()
    frontier_source_counts = frontier["solution_source"].value_counts()
    lines = [
        "# Augmented epsilon-constraint analysis",
        "",
        "## Formulation",
        "",
        "For each TAC budget epsilon, the direct CCUS NLP was solved as",
        "`min J + 1e-4 * TAC_hat` subject to `TAC <= epsilon`, where "
        "`J = 0.5 * TotEmiss_hat + 0.5 * ISI_hat`. Hats denote the fixed "
        "three-objective payoff-table normalization. The augmentation removes weakly "
        "efficient ties. Ipopt multi-start solutions are local NLP solutions; global "
        "optimality is not claimed.",
        "",
        "## Numerical result",
        "",
        f"- Epsilon levels solved: {len(raw)}",
        f"- Unique nondominated points after tolerance-based deduplication: {len(frontier)}",
        f"- Direct optimized solutions: {int(source_counts.get('optimized', 0))}",
        f"- Retained feasible multistart seeds: {int(source_counts.get('retained_seed', 0))}",
        f"- Final frontier optimized points: {int(frontier_source_counts.get('optimized', 0))}",
        f"- Final frontier retained feasible seeds: {int(frontier_source_counts.get('retained_seed', 0))}",
        f"- Maximum equality residual: {residuals['max_abs_equality']:.3e}",
        f"- Maximum source-carbon diagnostic residual: {residuals['max_abs_source_carbon_diagnostic']:.3e}",
        f"- Maximum inequality violation: {residuals['max_inequality_violation']:.3e}",
        f"- Maximum normalized epsilon violation: {residuals['epsilon_violation']:.3e}",
        "",
        "## Interpretation",
        "",
        f"The minimum-TAC endpoint is TAC={frontier.iloc[0]['TAC'] / 1e6:.3f} million "
        f"with J={frontier.iloc[0]['J_normalized']:.6f}. The minimum-J endpoint is "
        f"TAC={frontier.iloc[-1]['TAC'] / 1e6:.3f} million with "
        f"J={frontier.iloc[-1]['J_normalized']:.6f}.",
        "",
        f"The maximum-distance knee point is TAC={knee['TAC'] / 1e6:.3f} million, "
        f"J={knee['J_normalized']:.6f}, total emissions={knee['TotEmiss']:.6f}, "
        f"and ISI={knee['ISI']:.6f}.",
        "",
        switch_text,
        "",
        "Unlike the normalized weighted sum, which recovered only the two supported "
        "endpoints, the augmented epsilon-constraint formulation recovers the interior "
        "nonconvex trade-off set directly.",
        "",
        "One final nondominated point is a retained feasible multistart seed. A targeted "
        "Ipopt restart from that point converged to a slightly worse local objective, so "
        "the better feasible point was retained and remains explicitly marked by the "
        "`solution_source` field.",
    ]
    (RESULT_DIR / "direct_use_expansion_epsilon_normalized_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    raw, frontier, weighted = load_and_screen()
    plot_results(frontier, weighted)
    write_analysis(raw, frontier)
    print(f"Accepted epsilon solves: {len(raw)}")
    print(f"Unique nondominated points: {len(frontier)}")
    print(frontier["main_sink"].value_counts().to_string())
    print(
        frontier.loc[
            frontier["point_role"].ne("interior") | frontier["sink_switch"],
            ["frontier_id", "point_role", "TAC", "J_normalized", "main_sink"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
