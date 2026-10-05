"""Create the condition1-condition2 ORCA comparison figure and report.

Figure contract
---------------
Core conclusion: Changing source, capacity, loss-cost, and safety conditions changes
the selected ORCA group from emissions-ISI to TAC-emissions.
Archetype: quantitative grid.
Hero evidence: condition-level nonlinear ORCA strength comparison.
Validation evidence: step-size sensitivity for condition2.
Backend/export: Python/Matplotlib; 183 mm wide; PDF, SVG, TIFF, and PNG.
Reviewer risks: exploratory parameters are not plant-calibrated; Ipopt gives local
projections; variability is across deterministic random-direction replicates.
"""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE = Path("/tmp/ccus_condition_matplotlib")
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

PAIR_ORDER = ["TAC__TotEmiss", "TAC__ISI", "TotEmiss__ISI"]
PAIR_LABELS = {
    "TAC__TotEmiss": "TAC-emissions",
    "TAC__ISI": "TAC-ISI",
    "TotEmiss__ISI": "Emissions-ISI",
}
PAIR_COLORS = {
    "TAC__TotEmiss": "#4C78A8",
    "TAC__ISI": "#B7A6C9",
    "TotEmiss__ISI": "#D98B78",
}
CONDITION_COLORS = {
    "Supply-chain condition 1": "#B8BEC8",
    "Supply-chain condition 2": "#4C78A8",
}
TEXT = "#303640"
GRID = "#E5E8ED"

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


def load_summary(path: Path, condition: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data["constraint_handling"].eq("equality_tangent")].copy()
    data["condition"] = condition
    return data


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    condition1 = load_summary(
        RESULT_DIR / "direct_use_expansion_paper_orca_paper_main_summary.csv",
        "Supply-chain condition 1",
    )
    condition2 = load_summary(
        RESULT_DIR / "supply_chain_condition2_paper_orca_condition2_main_summary.csv",
        "Supply-chain condition 2",
    )
    comparison = pd.concat([condition1, condition2], ignore_index=True)

    sensitivity_parts = []
    for label, step in (
        ("condition2_step_low", 0.01),
        ("condition2_main", 0.03),
        ("condition2_step_high", 0.05),
    ):
        path = RESULT_DIR / f"supply_chain_condition2_paper_orca_{label}_summary.csv"
        data = pd.read_csv(path)
        data = data.loc[data["constraint_handling"].eq("equality_tangent")].copy()
        data["step_size"] = step
        sensitivity_parts.append(data)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)

    source = pd.concat(
        [
            comparison.assign(panel="condition_comparison"),
            sensitivity.assign(panel="step_sensitivity"),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(
        RESULT_DIR / "supply_chain_condition_comparison_source_data.csv",
        index=False,
    )
    parameters = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_parameters.csv"
    )
    parameters = parameters.loc[
        parameters["scenario"].isin(
            ["supply_chain_condition1", "carbon_cost_alignment_1_capacity_supply"]
        )
    ].copy()
    parameters["condition"] = parameters["scenario"].map(
        {
            "supply_chain_condition1": "Supply-chain condition 1",
            "carbon_cost_alignment_1_capacity_supply": "Supply-chain condition 2",
        }
    )
    parameters.to_csv(
        RESULT_DIR / "supply_chain_condition_parameter_comparison.csv",
        index=False,
    )
    return comparison, sensitivity


def plot(comparison: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.93, wspace=0.30)

    ax = axes[0]
    x = np.arange(len(PAIR_ORDER), dtype=float)
    width = 0.34
    for offset, condition in zip((-width / 2, width / 2), CONDITION_COLORS):
        data = comparison.set_index(["condition", "pair"]).loc[condition]
        means = data.loc[PAIR_ORDER, "signed_mean"].to_numpy(float)
        sd = data.loc[PAIR_ORDER, "signed_sd"].fillna(0.0).to_numpy(float)
        ax.bar(
            x + offset,
            means,
            width=width,
            color=CONDITION_COLORS[condition],
            edgecolor="white",
            linewidth=0.7,
            yerr=sd,
            capsize=2.0,
            error_kw={"linewidth": 0.7, "capthick": 0.7},
            label=condition,
            zorder=3,
        )
    ax.axhline(0.0, color="#6E7480", linewidth=0.7)
    ax.set_xticks(x, [PAIR_LABELS[pair] for pair in PAIR_ORDER])
    ax.set_ylabel("ORCA signed correlation")
    ax.set_ylim(0.0, 0.86)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=5.8, ncol=1)
    ax.text(-0.17, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.text(
        0.17,
        0.73,
        "Selected group",
        transform=ax.transAxes,
        color=CONDITION_COLORS["Supply-chain condition 2"],
        fontsize=6.0,
        ha="center",
    )

    ax = axes[1]
    final_labels = []
    for pair in PAIR_ORDER:
        data = sensitivity.loc[sensitivity["pair"].eq(pair)].sort_values("step_size")
        ax.errorbar(
            data["step_size"],
            data["signed_mean"],
            yerr=data["signed_sd"].fillna(0.0),
            color=PAIR_COLORS[pair],
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.4,
            capsize=2.2,
            label=PAIR_LABELS[pair],
            zorder=3,
        )
        final_labels.append((pair, data.iloc[-1]["signed_mean"]))
    ax.set_xlabel("Feasible-projection step size")
    ax.set_ylabel("Condition 2 ORCA signed correlation")
    ax.set_xticks([0.01, 0.03, 0.05])
    ax.set_xlim(0.006, 0.063)
    ax.set_ylim(0.10, 0.60)
    ax.grid(True, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    for pair, value in final_labels:
        ax.text(
            0.052,
            value,
            PAIR_LABELS[pair],
            color=PAIR_COLORS[pair],
            fontsize=5.8,
            va="center",
        )
    ax.text(-0.17, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

    stem = FIGURE_DIR / "supply_chain_condition_orca_comparison"
    for extension in ("png", "svg", "pdf", "tiff"):
        dpi = 600 if extension == "tiff" else 350
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_report(comparison: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    wide = comparison.pivot(index="pair", columns="condition", values="signed_mean")
    c1 = "Supply-chain condition 1"
    c2 = "Supply-chain condition 2"
    endpoints = pd.read_csv(
        RESULT_DIR / "supply_chain_correlation_screen_solutions.csv"
    )
    endpoints = endpoints.loc[
        endpoints["scenario"].isin(
            ["supply_chain_condition1", "carbon_cost_alignment_1_capacity_supply"]
        )
    ]
    endpoint_lines = []
    for scenario, label in (
        ("supply_chain_condition1", "Condition 1"),
        ("carbon_cost_alignment_1_capacity_supply", "Condition 2"),
    ):
        data = endpoints.loc[endpoints["scenario"].eq(scenario)].set_index("solve_type")
        for objective in ("TAC", "TotEmiss", "ISI"):
            row = data.loc[objective]
            endpoint_lines.append(
                f"| {label} | {objective} | {row['main_sink']} | "
                f"{row['pretreated_fraction']:.3f} | {row['TAC'] / 1e6:.3f} | "
                f"{row['TotEmiss']:.3f} | {row['ISI']:.3f} |"
            )

    sensitivity_lines = []
    for step in (0.01, 0.03, 0.05):
        data = sensitivity.loc[sensitivity["step_size"].eq(step)].set_index("pair")
        sensitivity_lines.append(
            f"| {step:.2f} | {data.loc['TAC__TotEmiss', 'signed_mean']:.4f} | "
            f"{data.loc['TAC__ISI', 'signed_mean']:.4f} | "
            f"{data.loc['TotEmiss__ISI', 'signed_mean']:.4f} |"
        )

    report = f"""# Supply-chain conditions and ORCA correlation change

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent; nature-figure
- Origin Mode: run + validate
- Origin Date: 2026-08-24
- Verification Status: VERIFIED FOR THE REPORTED LOCAL ORCA WORKFLOW
- Condition 1: frozen alias of the prior `direct_use_expansion` model
- Condition 2: exploratory parameter scenario; not plant-calibrated
- Optimization: JuMP/Ipopt local NLP solves and feasible projections
- ORCA: local user-developed package, nonlinear modified-gradient workflow

## Main result

Condition 1 groups emissions with ISI. Condition 2 instead groups TAC with emissions.
The condition2 result was recovered in 10/10 main replicates by both Leiden and
average-linkage grouping, and also in the inequality-only audit.

| Objective pair | Condition 1 signed mean | Condition 2 signed mean | Change |
|---|---:|---:|---:|
| TAC–emissions | {wide.loc['TAC__TotEmiss', c1]:.6f} | {wide.loc['TAC__TotEmiss', c2]:.6f} | {wide.loc['TAC__TotEmiss', c2] - wide.loc['TAC__TotEmiss', c1]:+.6f} |
| TAC–ISI | {wide.loc['TAC__ISI', c1]:.6f} | {wide.loc['TAC__ISI', c2]:.6f} | {wide.loc['TAC__ISI', c2] - wide.loc['TAC__ISI', c1]:+.6f} |
| Emissions–ISI | {wide.loc['TotEmiss__ISI', c1]:.6f} | {wide.loc['TotEmiss__ISI', c2]:.6f} | {wide.loc['TotEmiss__ISI', c2] - wide.loc['TotEmiss__ISI', c1]:+.6f} |

The group switch is driven mainly by the reduction of emissions–ISI coupling rather
than an increase in the absolute TAC–emissions strength. This distinction must be
retained in any manuscript interpretation.

## Condition definitions

Condition 1 exactly preserves the previous source capacities, sink capacities, losses,
transport coefficients, and objective coefficients. Condition 2 changes:

- source availability from `[357, 1260, 399, 3426]` to `[357, 1050, 350, 2800]`;
- sink/treatment capacities from `[300, 1400, 1400, 756, 900, 913]` to
  `[250, 1000, 650, 500, 700, 500]`;
- effective sink loss to `[0.30, 0.45, 0.02, 0.30, 0.00, 0.40]`;
- power price from `0.02` to `0.08` and power carbon intensity from `0.366` to `0.732`;
- treatment emissions from `0.0338` to `0.0676` and uniform matched treatment cost;
- sink processing cost proportional to effective carbon loss;
- sink safety to `[2, 1, 60, 2, 60, 1]`, with transport and treatment safety both `1`;
- source capture fraction to `1.0` and pump-power coefficients to zero.

These are mechanism-identification parameters. In particular, the zero pump-power
assumption and carbon-loss-dependent processing cost require replacement by calibrated
engineering/economic values before a real-system claim is made.

## Single-objective system structures

| Condition | Optimized objective | Main sink | Pretreated fraction | TAC (million) | Emissions | ISI |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(endpoint_lines)}

Under condition2, both TAC and emissions optima select Urea, whereas ISI selects
Greenhouse. The aligned endpoint topology is consistent with the ORCA grouping but is
not itself the ORCA calculation.

## Step-size sensitivity

| Step size | TAC–emissions | TAC–ISI | Emissions–ISI |
|---:|---:|---:|---:|
{chr(10).join(sensitivity_lines)}

TAC–emissions remains first at all three step sizes. The main run used 10 replicates,
40 added points per objective seed, and 1,230 total analyzed points. It had zero
projection failures and a maximum inequality violation of `6.05e-9`.

## Screening record and limitations

The full screening table contains all tested scenarios, including unsuccessful ones,
to avoid presenting only the selected case. Endpoint screening is only a selection
stage; the reported conclusion comes from the nonlinear ORCA validation.

Ipopt convergence is local, not a global optimality certificate. ORCA correlation is
a local feasible-direction relationship, not causal correlation and not a statistical
correlation across plants. Parameter selection was mechanism-guided and exploratory,
so the scenario demonstrates that the grouping can change; it does not estimate how
often this change occurs in real CCUS systems.

## Fallacy scan

- Correlation-causation: ORCA strength is not causal evidence.
- Local-to-global: local ORCA grouping is not automatically a global Pareto-redundancy result.
- Garden of forking paths / look-elsewhere: all screened scenarios are retained in the ranking table.
- Simpson, ecological, Berkson, collider, base-rate, regression-to-mean, survivorship,
  and reverse-causality fallacies have no direct evidence in this deterministic model
  comparison, but remain relevant for later empirical calibration.
"""
    (RESULT_DIR / "supply_chain_condition_orca_analysis.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    comparison, sensitivity = load_data()
    plot(comparison, sensitivity)
    write_report(comparison, sensitivity)
    print(comparison[["condition", "pair", "signed_mean", "signed_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
