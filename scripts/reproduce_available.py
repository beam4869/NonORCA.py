"""Validate and replot recovered manuscript data; does not rerun ORCA or NLPs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OBJECTIVES = ["TAC", "TotEmiss", "ISI"]
PREFIXES = ["direct_use_expansion", "supply_chain_condition2"]
RUNS = ["paper_main", "condition2_main"]
PAIRS = ["TAC__TotEmiss", "TAC__ISI", "TotEmiss__ISI"]
GROUPS = ["TAC + total emissions", "TAC + ISI", "Total emissions + ISI"]
EXPECTED_TABLE4 = [
    [(0.7781, 0.3453), (0.7557, 0.7429), (0.8831, 0.2900)],
    [(0.7513, 0.2868), (0.6678, 0.3098), (0.5944, 0.8609)],
]


def read(name: str, folder: str = "ccus") -> pd.DataFrame:
    return pd.read_csv(DATA / folder / name)


def full_frontier(condition: int) -> pd.DataFrame:
    """Plot scaling uses min/max of the supplied full three-objective frontier."""
    prefix = PREFIXES[condition - 1]
    frame = read(f"{prefix}_pareto_frontier.csv")
    labels = read(f"{prefix}_pareto_system_structures.csv")[["frontier_id", "main_sink"]]
    frame = frame.merge(labels, on="frontier_id", validate="one_to_one")
    for objective in OBJECTIVES:
        values = frame[objective]
        span = values.max() - values.min()
        if not np.isfinite(span) or span <= 0:
            raise ValueError(f"Invalid objective range: {objective}")
        frame[f"{objective}_plot"] = (values - values.min()) / span
    return frame


def audit(output: Path) -> dict:
    checks, notes = [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for entry in manifest["recovered_files"]:
        path = ROOT / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        check("source integrity: " + entry["path"], actual == entry["sha256"])

    table2 = read("dtlz5_5_16_nsga3_summary.csv", "dtlz")
    expected = {
        "objectives": [16, 5, 5], "evaluations": [13600, 3500, 13615],
        "total_time_mean_s": [14.84, 4.95, 16.18], "total_time_sd_s": [2.00, 0.74, 2.09],
        "approx_hv_mean": [2.640, 3.689, 3.891], "approx_hv_sd": [0.266, 0.296, 0.198],
        "empirical_igd_mean": [0.410, 0.174, 0.166], "empirical_igd_sd": [0.047, 0.012, 0.009],
        "seeds": [5, 5, 5],
    }
    for column, values in expected.items():
        check("Table 2: " + column, len(table2) == 3 and np.allclose(table2[column], values, rtol=0, atol=1e-12))
    check("Table 2: timing scope", table2["orca_time_included"].tolist() == ["no", "yes", "yes"])
    table2.to_csv(output / "table2_dtlz5_5_16.csv", index=False)

    parameters = read("supply_chain_condition_parameter_comparison.csv")
    for condition, loss, safety, cost in [
        (1, [.22, .16, 0, .18, .12, .30], [7, 3, 18, 32, 25, 30], [-7, -5, 9, -20, -17, -26]),
        (2, [.30, .45, .02, .30, 0, .40], [2, 1, 60, 2, 60, 1], [32.79, 49.18, 2.19, 32.79, 0, 43.72]),
    ]:
        rows = parameters[parameters.condition.eq(f"Supply-chain condition {condition}")].sort_values("sink_index")
        for col, values in [("effective_carbon_loss", loss), ("safety_index", safety), ("processing_cost", cost)]:
            check(f"Table 3: condition {condition} {col}", len(rows) == 6 and np.allclose(rows[col].round(2), values, rtol=0, atol=1e-12))
    parameters[["condition", "sink", "effective_carbon_loss", "safety_index", "processing_cost"]].to_csv(output / "table3_ccus_parameters.csv", index=False)

    table4 = []
    normalization = []
    for index, (prefix, run) in enumerate(zip(PREFIXES, RUNS)):
        condition = index + 1
        summary = read(f"{prefix}_paper_orca_{run}_summary.csv")
        summary = summary[summary.constraint_handling.eq("equality_tangent")].set_index("pair")
        comparison = read(f"{prefix}_exact_info_loss_quantile21_comparison.csv").set_index("grouping")
        check(f"condition {condition}: A=(1+signed)/2", np.allclose(summary.adjacency_mean, (1 + summary.signed_mean) / 2, rtol=0, atol=1e-12))
        for pair, group, expected_pair in zip(PAIRS, GROUPS, EXPECTED_TABLE4[index]):
            row = comparison.loc[group]
            actual = [float(row.orca_strength_01), float(row.mean_information_loss)]
            check(f"Table 4: condition {condition}, {group}", np.allclose(np.round(actual, 4), expected_pair, rtol=0, atol=1e-12))
            check(f"condition {condition}: two archived ORCA summaries agree for {pair}", np.isclose(summary.loc[pair, "adjacency_mean"], actual[0], rtol=0, atol=1e-12))
            table4.append({"condition": condition, "grouping": group, "correlation_strength": actual[0], "conditional_loss": actual[1]})
        groupings = read(f"{prefix}_paper_orca_{run}_groupings.csv")
        groupings = groupings[groupings.constraint_handling.eq("equality_tangent")]
        nrep = [20, 10][index]
        check(f"condition {condition}: archived main replicate count", len(groupings) == nrep and groupings.replicate.nunique() == nrep)
        grouped = ["TotEmiss", "ISI"] if condition == 1 else ["TAC", "TotEmiss"]
        singleton = "TAC" if condition == 1 else "ISI"
        for method in ["leiden", "average"]:
            same = groupings[f"{method}_{grouped[0]}"] == groupings[f"{method}_{grouped[1]}"]
            different = groupings[f"{method}_{grouped[0]}"] != groupings[f"{method}_{singleton}"]
            check(f"condition {condition}: {method} stored partitions", (same & different).all())
        gradients = read(f"{prefix}_paper_orca_{run}_objective_gradients.csv")
        grad_columns = [f"dz{i}" for i in range(1, 83)]
        check(f"condition {condition}: gradient row count", len(gradients) == nrep * 123 * 3)
        check(f"condition {condition}: finite 82-dimensional gradients", np.isfinite(gradients[grad_columns].to_numpy(float)).all())
        frontier = full_frontier(condition)
        check(f"condition {condition}: frontier size", len(frontier) == [225, 279][index])
        check(f"condition {condition}: unique frontier IDs", frontier.frontier_id.is_unique)
        check(f"condition {condition}: finite raw objectives and decisions", np.isfinite(frontier[OBJECTIVES + [f"z{i}" for i in range(1, 83)]].to_numpy(float)).all())
        for col in ["max_abs_equality", "max_inequality_violation", "epsilon_violation"]:
            check(f"condition {condition}: archived {col}", (frontier[col].fillna(0) <= 1e-6).all(), float(frontier[col].max()))
        for objective in OBJECTIVES:
            discrepancy = float(np.max(np.abs(frontier[f"{objective}_normalized"] - frontier[f"{objective}_plot"])))
            normalization.append({"condition": condition, "objective": objective, "max_abs_stored_vs_frontier_minmax": discrepancy})
        if condition == 2:
            projections = read(f"{prefix}_pareto_pairwise_projection_source.csv")
            merged = projections.merge(frontier, on="frontier_id", validate="many_to_one")
            for axis in ["x", "y"]:
                expected_axis = np.array([row[f"{row[axis + '_objective']}_plot"] for _, row in merged.iterrows()])
                check(f"condition 2: archived projection {axis} agrees with frontier min/max", np.allclose(merged[f"{axis}_normalized"], expected_axis, rtol=0, atol=2e-7))
    pd.DataFrame(table4).to_csv(output / "table4_ccus_grouping.csv", index=False)
    pd.DataFrame(normalization).to_csv(output / "normalization_audit.csv", index=False)
    notes.extend([
        "Checks validate supplied artifacts and rounded manuscript tables, not fresh optimization runs.",
        "Condition 2 stored *_normalized columns differ from min/max of the supplied full frontier (ISI: about 0.007095; TAC: about 0.0000802). Original files are unchanged. New plots use full-frontier min/max, which agrees with archived pairwise plot coordinates within 2e-7.",
        "Local source recovery supplies the nonlinear Python core, ellipse/DTLZ5/6/9 runners, CCUS drivers and Julia lockfile. Run verify_local_sources.py for numerical source-to-result checks.",
        "Julia is not available in the preparation environment; CCUS NLP solves have not been rerun here.",
    ])
    report = {"all_available_checks_passed": all(item["passed"] for item in checks), "checks": checks, "notes": notes,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "full_manuscript_reproduction": False}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def plots(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    colors = ["#356a96", "#bc653e", "#56916c"]
    labels = ["TAC–emissions", "TAC–ISI", "Emissions–ISI"]

    def save(fig, stem):
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for index, (prefix, run) in enumerate(zip(PREFIXES, RUNS)):
        summary = read(f"{prefix}_paper_orca_{run}_summary.csv")
        summary = summary[summary.constraint_handling.eq("equality_tangent")].set_index("pair").loc[PAIRS]
        axes[0].bar(np.arange(3) + (index - .5) * .34, summary.adjacency_mean, width=.34,
                    yerr=summary.signed_sd / 2, capsize=3, label=f"Condition {index + 1}")
    axes[0].set(xticks=np.arange(3), xticklabels=labels, ylim=(0, 1), ylabel="ORCA correlation strength A", title="Archived main-run means")
    axes[0].legend()
    for pair, label, color in zip(PAIRS, labels, colors):
        means, sds = [], []
        for run in ["condition2_step_low", "condition2_main", "condition2_step_high"]:
            summary = read(f"supply_chain_condition2_paper_orca_{run}_summary.csv")
            row = summary[summary.constraint_handling.eq("equality_tangent")].set_index("pair").loc[pair]
            means.append(row.adjacency_mean); sds.append(row.signed_sd / 2)
        axes[1].errorbar([.01, .03, .05], means, yerr=sds, marker="o", color=color, label=label, capsize=3)
    axes[1].set(xlabel="Step size", ylabel="ORCA correlation strength A", title="Condition 2: archived settings", xticks=[.01, .03, .05], ylim=(.5, .86))
    axes[1].legend(fontsize=8)
    save(fig, "ccus_orca_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for condition, ax in enumerate(axes, 1):
        frontier = full_frontier(condition)
        scatter = ax.scatter(frontier.TotEmiss_plot, frontier.ISI_plot, c=frontier.TAC_plot, cmap="viridis", vmin=0, vmax=1, s=16)
        ax.set(xlabel="Normalized total emissions", ylabel="Normalized ISI", title=f"Condition {condition}: {len(frontier)} archived points")
    fig.colorbar(scatter, ax=axes, label="Normalized TAC", shrink=.8)
    save(fig, "ccus_frontier_projections")

    candidates = read("supply_chain_condition2_grouping_projection_source.csv")
    frontier = full_frontier(2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for group, ax in zip(GROUPS, axes):
        selected = candidates[candidates.grouping.eq(group)]
        ax.scatter(frontier.TotEmiss_plot, frontier.ISI_plot, s=11, color="#cccccc", label="Full frontier")
        ax.scatter(selected.TotEmiss_normalized, selected.ISI_normalized, s=17, color="#b34340", label="Archived grouped candidates")
        loss = float(selected.mean_information_loss.iloc[0])
        ax.set(xlabel="Normalized total emissions", ylabel="Normalized ISI", title=f"{group}\nConditional loss = {loss:.4f}")
    axes[0].legend(fontsize=7)
    save(fig, "ccus_condition2_archived_candidates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "available")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(output == ROOT / name or (ROOT / name) in output.parents for name in ["data", "archive", "src", "paper", "docs"]):
        parser.error("Use a separate output directory; archived inputs must not be overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    report = audit(output)
    failed = [item for item in report["checks"] if not item["passed"]]
    if failed:
        for item in failed:
            print("FAIL:", item["name"])
        return 1
    if not args.skip_plots:
        plots(output)
    print(f"PASS: {len(report['checks'])} artifact checks. Outputs: {output}")
    print("For executed computational checks and remaining limits, see docs/reproduction_status.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
