"""Analyze the TAC versus a raw weighted emissions-safety objective.

The grouped objective is ``0.20 * total emissions + 0.80 * ISI`` using the
original objective values without payoff normalization. The source data are
all validated deterministic candidates retained on the three-objective
frontier.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_grouped_pareto_matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

SOURCE_NAMES = ["Ammonia", "Steel", "Refinery", "Power plant"]
SINK_NAMES = [
    "Algae",
    "Greenhouse",
    "Saline storage",
    "Methanol",
    "Urea",
    "Acetic acid",
]
SOURCE_CAPACITY = np.array([357.0, 1260.0, 399.0, 3426.0])
SINK_CAPACITY = np.array([300.0, 1400.0, 1400.0, 756.0, 900.0, 913.0])
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
Z_COLUMNS = [f"z{index}" for index in range(1, 83)]
DOMINANCE_TOLERANCE = 1.0e-7
NUMERICAL_DUPLICATE_TOLERANCE = 1.0e-5
ACTIVE_FLOW_TOLERANCE = 1.0
EMISSIONS_WEIGHT = 0.20
ISI_WEIGHT = 0.80

COLORS = {
    "frontier": "#6179A8",
    "minimum_tac": "#C97A8B",
    "saline_10": "#8979A8",
    "knee": "#C89B4B",
    "sink_switch": "#5B91A1",
    "minimum_ei": "#4F9688",
    "treated": "#D5A5B5",
    "direct": "#8192BE",
    "source_node": "#E8EAF0",
    "sink_node": "#E1EBE5",
    "outline": "#59647A",
    "grid": "#E6E7EA",
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
    """Return the exact pairwise nondominated mask for minimization."""
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(
            values <= point + DOMINANCE_TOLERANCE, axis=1
        ) & np.any(values < point - DOMINANCE_TOLERANCE, axis=1)
        keep[index] = not np.any(dominates)
    return keep


def load_grouped_frontier() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    raw = pd.read_csv(RESULT_DIR / "direct_use_expansion_pareto_frontier.csv")
    structures = pd.read_csv(
        RESULT_DIR / "direct_use_expansion_pareto_system_structures.csv"
    )
    structure_columns = [
        "frontier_id",
        "main_sink",
        "active_sinks",
        "pretreated_fraction",
        "direct_fraction",
        "sink2_share",
        "sink3_share",
        "sink5_share",
    ]
    raw = raw.merge(structures[structure_columns], on="frontier_id", how="left")
    payoff = pd.read_csv(RESULT_DIR / "direct_use_expansion_pareto_payoff_table.csv")
    lower = payoff[OBJECTIVES].min().to_numpy(dtype=float)
    upper = payoff[OBJECTIVES].max().to_numpy(dtype=float)
    ranges = upper - lower

    normalized = (raw[OBJECTIVES].to_numpy(dtype=float) - lower) / ranges
    raw["TAC_normalized"] = normalized[:, 0]
    raw["TotEmiss_normalized"] = normalized[:, 1]
    raw["ISI_normalized"] = normalized[:, 2]
    raw["emissions_contribution"] = EMISSIONS_WEIGHT * raw["TotEmiss"]
    raw["ISI_contribution"] = ISI_WEIGHT * raw["ISI"]
    raw["EI_grouped_raw"] = (
        raw["emissions_contribution"] + raw["ISI_contribution"]
    )

    keep = nondominated_mask(raw[["TAC", "EI_grouped_raw"]].to_numpy())
    frontier = raw.loc[keep].sort_values(
        ["TAC", "EI_grouped_raw"]
    ).copy()

    tac_span = frontier["TAC"].max() - frontier["TAC"].min()
    grouped_span = (
        frontier["EI_grouped_raw"].max() - frontier["EI_grouped_raw"].min()
    )
    frontier["_TAC_dedup_scaled"] = (
        frontier["TAC"] - frontier["TAC"].min()
    ) / tac_span
    frontier["_EI_dedup_scaled"] = (
        frontier["EI_grouped_raw"] - frontier["EI_grouped_raw"].min()
    ) / grouped_span

    unique_indices: list[int] = []
    for index, row in frontier.iterrows():
        point = row[["_TAC_dedup_scaled", "_EI_dedup_scaled"]].to_numpy(
            dtype=float
        )
        if not unique_indices:
            unique_indices.append(index)
            continue
        prior = frontier.loc[
            unique_indices[-1], ["_TAC_dedup_scaled", "_EI_dedup_scaled"]
        ].to_numpy(dtype=float)
        if np.max(np.abs(point - prior)) > NUMERICAL_DUPLICATE_TOLERANCE:
            unique_indices.append(index)
    frontier = (
        frontier.loc[unique_indices]
        .drop(columns=["_TAC_dedup_scaled", "_EI_dedup_scaled"])
        .sort_values("TAC")
        .reset_index(drop=True)
    )
    return raw, frontier, lower, ranges


def select_key_points(frontier: pd.DataFrame) -> pd.DataFrame:
    selected = frontier.copy()
    indices = [
        selected["TAC"].idxmin(),
        (selected["sink3_share"] - 0.10).abs().idxmin(),
        (selected["sink3_share"] - 0.50).abs().idxmin(),
        (selected["sink3_share"] - 0.90).abs().idxmin(),
        selected["EI_grouped_raw"].idxmin(),
    ]
    keys = selected.loc[indices].copy()
    keys["key_id"] = ["A", "B", "C", "D", "E"]
    keys["key_role"] = [
        "Minimum TAC",
        "10% saline allocation",
        "Sink-switch point",
        "90% saline allocation",
        "Minimum raw emissions + ISI",
    ]
    keys["key_color"] = [
        COLORS["minimum_tac"],
        COLORS["saline_10"],
        COLORS["knee"],
        COLORS["sink_switch"],
        COLORS["minimum_ei"],
    ]
    keys["key_marker"] = ["o", "P", "D", "X", "s"]
    return keys.reset_index(drop=True)


def decode_flows(row: pd.Series) -> dict[str, np.ndarray | float]:
    z = row[Z_COLUMNS].to_numpy(dtype=float)
    source_flow = np.maximum(z[:4] * SOURCE_CAPACITY, 0.0)
    treated = np.maximum(
        z[4:28].reshape(6, 4).T * SOURCE_CAPACITY[:, None], 0.0
    )
    direct = np.maximum(
        z[28:52].reshape(6, 4).T * SOURCE_CAPACITY[:, None], 0.0
    )
    treatment_waste = np.maximum(
        z[52:76].reshape(6, 4).T * SOURCE_CAPACITY[:, None], 0.0
    )
    reported_sink_flow = np.maximum(z[76:82] * SINK_CAPACITY, 0.0)
    sink_flow = (treated + direct).sum(axis=0)
    mass_error = np.max(np.abs(reported_sink_flow - sink_flow))
    if mass_error > 1.0e-4:
        raise ValueError(f"Sankey sink mass-balance error is {mass_error:.3e}")
    total_source = source_flow.sum()
    pretreated_fraction = (
        (treated.sum() + treatment_waste.sum()) / total_source
        if total_source > 0
        else 0.0
    )
    return {
        "source_flow": source_flow,
        "treated": treated,
        "direct": direct,
        "treatment_waste": treatment_waste,
        "sink_flow": sink_flow,
        "pretreated_fraction": pretreated_fraction,
        "direct_fraction": direct.sum() / total_source if total_source > 0 else 0.0,
        "mass_error": mass_error,
    }


def export_source_data(
    raw: pd.DataFrame, frontier: pd.DataFrame, keys: pd.DataFrame
) -> dict[str, dict[str, np.ndarray | float]]:
    key_lookup = keys.set_index("frontier_id")[["key_id", "key_role"]]
    output_frontier = frontier[
        [
            "frontier_id",
            "method",
            "index_a",
            "index_b",
            "epsilon_TAC",
            "TAC",
            "TotEmiss",
            "ISI",
            "TAC_normalized",
            "TotEmiss_normalized",
            "ISI_normalized",
            "emissions_contribution",
            "ISI_contribution",
            "EI_grouped_raw",
            "main_sink",
            "active_sinks",
            "sink2_share",
            "sink3_share",
            "sink5_share",
        ]
    ].copy()
    output_frontier["key_id"] = output_frontier["frontier_id"].map(
        key_lookup["key_id"]
    )
    output_frontier["key_role"] = output_frontier["frontier_id"].map(
        key_lookup["key_role"]
    )
    output_frontier.to_csv(
        RESULT_DIR / "direct_use_expansion_TAC_EI_pareto_frontier.csv", index=False
    )

    flows_by_key: dict[str, dict[str, np.ndarray | float]] = {}
    key_rows = []
    link_rows = []
    node_rows = []
    for _, row in keys.iterrows():
        key_id = str(row["key_id"])
        flows = decode_flows(row)
        flows_by_key[key_id] = flows
        source_flow = np.asarray(flows["source_flow"])
        sink_flow = np.asarray(flows["sink_flow"])
        treated = np.asarray(flows["treated"])
        direct = np.asarray(flows["direct"])
        key_rows.append(
            {
                "key_id": key_id,
                "key_role": row["key_role"],
                "frontier_id": row["frontier_id"],
                "epsilon_index": int(row["index_a"]),
                "TAC": row["TAC"],
                "TotEmiss": row["TotEmiss"],
                "ISI": row["ISI"],
                "emissions_contribution": row["emissions_contribution"],
                "ISI_contribution": row["ISI_contribution"],
                "EI_grouped_raw": row["EI_grouped_raw"],
                "greenhouse_share": row["sink2_share"],
                "saline_share": row["sink3_share"],
                "urea_share": row["sink5_share"],
                "pretreated_fraction": flows["pretreated_fraction"],
                "direct_fraction": flows["direct_fraction"],
                "max_sink_mass_error": flows["mass_error"],
            }
        )
        for source_index, source_name in enumerate(SOURCE_NAMES):
            node_rows.append(
                {
                    "key_id": key_id,
                    "node_type": "source",
                    "node_name": source_name,
                    "flow": source_flow[source_index],
                    "capacity": SOURCE_CAPACITY[source_index],
                    "utilization": source_flow[source_index]
                    / SOURCE_CAPACITY[source_index],
                }
            )
            for sink_index, sink_name in enumerate(SINK_NAMES):
                for flow_type, matrix in (("treated", treated), ("direct", direct)):
                    value = matrix[source_index, sink_index]
                    if value > ACTIVE_FLOW_TOLERANCE:
                        link_rows.append(
                            {
                                "key_id": key_id,
                                "key_role": row["key_role"],
                                "source": source_name,
                                "sink": sink_name,
                                "flow_type": flow_type,
                                "flow_kt_per_year": value,
                            }
                        )
        for sink_index, sink_name in enumerate(SINK_NAMES):
            node_rows.append(
                {
                    "key_id": key_id,
                    "node_type": "sink",
                    "node_name": sink_name,
                    "flow": sink_flow[sink_index],
                    "capacity": SINK_CAPACITY[sink_index],
                    "utilization": sink_flow[sink_index] / SINK_CAPACITY[sink_index],
                }
            )

    pd.DataFrame(key_rows).to_csv(
        RESULT_DIR / "direct_use_expansion_TAC_EI_key_points.csv", index=False
    )
    pd.DataFrame(link_rows).to_csv(
        RESULT_DIR / "direct_use_expansion_TAC_EI_sankey_links.csv", index=False
    )
    pd.DataFrame(node_rows).to_csv(
        RESULT_DIR / "direct_use_expansion_TAC_EI_sankey_nodes.csv", index=False
    )
    return flows_by_key


def node_bands(totals: np.ndarray, active: np.ndarray) -> dict[int, tuple[float, float]]:
    indices = np.flatnonzero(active)
    available = 0.68
    gap = 0.055 if len(indices) > 1 else 0.0
    flow_height = available - gap * max(len(indices) - 1, 0)
    scale = flow_height / totals[indices].sum()
    top = 0.88
    bands: dict[int, tuple[float, float]] = {}
    for index in indices:
        height = totals[index] * scale
        bands[index] = (top - height, top)
        top -= height + gap
    return bands


def ribbon_path(
    x0: float,
    x1: float,
    source_band: tuple[float, float],
    target_band: tuple[float, float],
) -> MplPath:
    sy0, sy1 = source_band
    ty0, ty1 = target_band
    cx0 = x0 + 0.38 * (x1 - x0)
    cx1 = x0 + 0.62 * (x1 - x0)
    vertices = [
        (x0, sy0),
        (cx0, sy0),
        (cx1, ty0),
        (x1, ty0),
        (x1, ty1),
        (cx1, ty1),
        (cx0, sy1),
        (x0, sy1),
        (x0, sy0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def draw_sankey(ax, row: pd.Series, flows: dict[str, np.ndarray | float]) -> None:
    treated = np.asarray(flows["treated"])
    direct = np.asarray(flows["direct"])
    source_totals = (treated + direct).sum(axis=1)
    sink_totals = (treated + direct).sum(axis=0)
    source_active = source_totals > ACTIVE_FLOW_TOLERANCE
    sink_active = sink_totals > ACTIVE_FLOW_TOLERANCE
    source_bands = node_bands(source_totals, source_active)
    sink_bands = node_bands(sink_totals, sink_active)

    records = []
    for sink_index in np.flatnonzero(sink_active):
        for source_index in np.flatnonzero(source_active):
            for flow_type, matrix in (("direct", direct), ("treated", treated)):
                value = matrix[source_index, sink_index]
                if value > ACTIVE_FLOW_TOLERANCE:
                    records.append((source_index, sink_index, flow_type, value))

    total_flow = sum(record[3] for record in records)
    flow_scale = (
        sum(top - bottom for bottom, top in source_bands.values()) / total_flow
    )
    source_offsets = {index: 0.0 for index in source_bands}
    sink_offsets = {index: 0.0 for index in sink_bands}
    x_source, x_sink = 0.12, 0.88
    node_width = 0.035

    for source_index, sink_index, flow_type, value in records:
        height = value * flow_scale
        source_bottom = source_bands[source_index][0] + source_offsets[source_index]
        target_bottom = sink_bands[sink_index][0] + sink_offsets[sink_index]
        source_segment = (source_bottom, source_bottom + height)
        target_segment = (target_bottom, target_bottom + height)
        source_offsets[source_index] += height
        sink_offsets[sink_index] += height
        color = COLORS["treated"] if flow_type == "treated" else COLORS["direct"]
        ax.add_patch(
            PathPatch(
                ribbon_path(
                    x_source + node_width,
                    x_sink,
                    source_segment,
                    target_segment,
                ),
                facecolor=color,
                edgecolor="none",
                alpha=0.76,
                zorder=1,
            )
        )

    source_flow = np.asarray(flows["source_flow"])
    sink_flow = np.asarray(flows["sink_flow"])
    for index, (bottom, top) in source_bands.items():
        ax.add_patch(
            Rectangle(
                (x_source, bottom),
                node_width,
                top - bottom,
                facecolor=COLORS["source_node"],
                edgecolor=COLORS["outline"],
                linewidth=0.65,
                zorder=3,
            )
        )
        ax.text(
            x_source - 0.025,
            0.5 * (bottom + top),
            f"{SOURCE_NAMES[index]}\n{source_totals[index]:.0f} delivered\n"
            f"({source_flow[index]:.0f} feed)",
            ha="right",
            va="center",
            fontsize=5.1,
            linespacing=1.05,
        )
    for index, (bottom, top) in sink_bands.items():
        ax.add_patch(
            Rectangle(
                (x_sink, bottom),
                node_width,
                top - bottom,
                facecolor=COLORS["sink_node"],
                edgecolor=COLORS["outline"],
                linewidth=0.65,
                zorder=3,
            )
        )
        ax.text(
            x_sink + node_width + 0.025,
            0.5 * (bottom + top),
            f"{SINK_NAMES[index]}\n{sink_flow[index]:.0f} kt yr$^{{-1}}$\n"
            f"({100 * sink_flow[index] / SINK_CAPACITY[index]:.0f}% cap.)",
            ha="left",
            va="center",
            fontsize=5.1,
            linespacing=1.05,
        )

    ax.text(
        0.50,
        0.045,
        f"TAC {row['TAC'] / 1e6:.2f} M  |  emissions {row['TotEmiss']:.1f} kt yr$^{{-1}}$  |  "
        f"ISI {row['ISI']:.2f}\nPretreated feed {100 * float(flows['pretreated_fraction']):.1f}%",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#3E4350",
    )
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def plot_analysis(
    frontier: pd.DataFrame,
    keys: pd.DataFrame,
    flows_by_key: dict[str, dict[str, np.ndarray | float]],
) -> None:
    fig = plt.figure(figsize=(7.2, 7.0))
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=(1.18, 1.0, 1.0),
        left=0.09,
        right=0.97,
        bottom=0.065,
        top=0.96,
        hspace=0.42,
        wspace=0.62,
    )
    ax = fig.add_subplot(grid[0, :])
    ordered = frontier.sort_values("EI_grouped_raw")
    ax.plot(
        ordered["EI_grouped_raw"],
        ordered["TAC"] / 1e6,
        color=COLORS["frontier"],
        linewidth=1.25,
        zorder=1,
    )
    ax.scatter(
        ordered["EI_grouped_raw"],
        ordered["TAC"] / 1e6,
        s=15,
        facecolor="#AEBAD3",
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )

    annotation_offsets = {
        "A": (-55, 14),
        "B": (7, 12),
        "C": (7, 13),
        "D": (8, -22),
        "E": (10, -21),
    }
    for _, row in keys.iterrows():
        ax.scatter(
            row["EI_grouped_raw"],
            row["TAC"] / 1e6,
            s=52,
            marker=row["key_marker"],
            facecolor=row["key_color"],
            edgecolor="white",
            linewidth=0.75,
            zorder=4,
        )
        ax.annotate(
            f"{row['key_id']}  {row['key_role']}",
            (row["EI_grouped_raw"], row["TAC"] / 1e6),
            xytext=annotation_offsets[row["key_id"]],
            textcoords="offset points",
            fontsize=6.0,
            color="#2F3440",
            arrowprops={
                "arrowstyle": "-",
                "color": row["key_color"],
                "linewidth": 0.65,
            },
        )

    ax.set_xlabel("Raw weighted emissions-safety objective")
    ax.set_ylabel(r"TAC ($10^6$ cost units yr$^{-1}$)")
    ax.margins(x=0.035, y=0.08)
    ax.grid(True, color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    ax.text(-0.075, 1.03, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.text(
        0.015,
        0.045,
        r"$J_{raw}=0.20\,E+0.80\,ISI$; original values, no payoff normalization",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#545B69",
    )

    sankey_slots = [
        grid[1, 0:2],
        grid[1, 2:4],
        grid[1, 4:6],
        grid[2, 1:3],
        grid[2, 3:5],
    ]
    for panel_index, (_, row) in enumerate(keys.iterrows()):
        sankey_ax = fig.add_subplot(sankey_slots[panel_index])
        draw_sankey(sankey_ax, row, flows_by_key[row["key_id"]])
        sankey_ax.set_title(
            f"{chr(ord('b') + panel_index)}   {row['key_id']}: {row['key_role']}",
            loc="left",
            fontsize=7.0,
            fontweight="bold",
            pad=4,
        )

    fig.legend(
        handles=[
            Patch(facecolor=COLORS["treated"], label="Treated stream"),
            Patch(facecolor=COLORS["direct"], label="Direct/untreated stream"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=6.0,
        columnspacing=1.4,
        handlelength=1.5,
    )

    stem = FIGURE_DIR / "direct_use_expansion_TAC_EI_pareto_sankey"
    for extension in ("png", "svg", "pdf", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    raw, frontier, _, _ = load_grouped_frontier()
    keys = select_key_points(frontier)
    flows_by_key = export_source_data(raw, frontier, keys)
    plot_analysis(frontier, keys, flows_by_key)
    print(f"Validated three-objective candidate solutions: {len(raw)}")
    print(f"Unique nondominated frontier points: {len(frontier)}")
    print(
        keys[
            [
                "key_id",
                "key_role",
                "frontier_id",
                "TAC",
                "TotEmiss",
                "ISI",
                "emissions_contribution",
                "ISI_contribution",
                "EI_grouped_raw",
                "sink3_share",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
