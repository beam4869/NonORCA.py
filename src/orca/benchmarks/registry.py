"""Recommended benchmark coverage for ORCA validation.

This registry records benchmark families that are useful for objective
reduction experiments. It is intentionally descriptive: only lightweight,
locally implemented problems are exposed as executable classes in this package.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkRecommendation:
    """One benchmark-family recommendation."""

    family: str
    role: str
    suggested_cases: tuple[str, ...]
    implementation_status: str


RECOMMENDED_BENCHMARKS: tuple[BenchmarkRecommendation, ...] = (
    BenchmarkRecommendation(
        family="DTLZ5",
        role="Known objective grouping for positive validation.",
        suggested_cases=("DTLZ5(2,3)", "DTLZ5(3,5)", "DTLZ5(5,12)"),
        implementation_status="implemented",
    ),
    BenchmarkRecommendation(
        family="DTLZ6",
        role="Known DTLZ5-like grouping with a different nonlinear distance function.",
        suggested_cases=("DTLZ6(3,5)", "DTLZ6(5,12)"),
        implementation_status="implemented",
    ),
    BenchmarkRecommendation(
        family="Constrained-DTLZ6",
        role="DTLZ6 objective-reduction positive control with a nonconvex decision-space feasible region.",
        suggested_cases=("Constrained-DTLZ6(5,12) with circular hole",),
        implementation_status="implemented",
    ),
    BenchmarkRecommendation(
        family="C-DTLZ",
        role="Standard constrained DTLZ pressure tests with nonconvex objective-space constraints.",
        suggested_cases=("C2-DTLZ2", "C3-DTLZ4"),
        implementation_status="implemented via pymoo adapter",
    ),
    BenchmarkRecommendation(
        family="Minus-DTLZ / inverted DTLZ",
        role="Negative-control candidates for conflict-heavy objective structures.",
        suggested_cases=("Minus-DTLZ2", "Inverted DTLZ1", "Inverted DTLZ2"),
        implementation_status="planned",
    ),
    BenchmarkRecommendation(
        family="WFG1-WFG9",
        role="Stress tests for deceptive, nonseparable, degenerate, and disconnected Pareto fronts.",
        suggested_cases=("WFG1", "WFG2", "WFG3", "WFG4", "WFG6", "WFG9"),
        implementation_status="implemented via optproblems adapter; WFG1/WFG2/WFG3/WFG9 smoke-tested",
    ),
    BenchmarkRecommendation(
        family="MaF1-MaF15",
        role="Many-objective stress tests with irregular, degenerate, and disconnected fronts.",
        suggested_cases=("MaF1", "MaF4", "MaF7", "MaF10", "MaF14", "MaF15"),
        implementation_status="planned",
    ),
    BenchmarkRecommendation(
        family="UF / CEC2009",
        role="Nonlinear Pareto-set validation for unconstrained and constrained cases.",
        suggested_cases=("UF1", "UF4", "UF7", "CF8", "CF9", "CF10"),
        implementation_status="CF8/CF9/CF10 implemented locally from CEC2009 formulas; UF planned",
    ),
    BenchmarkRecommendation(
        family="LIR-CMOP",
        role="Large-infeasible-region constrained tests.",
        suggested_cases=("LIR-CMOP5", "LIR-CMOP6", "LIR-CMOP13", "LIR-CMOP14"),
        implementation_status="implemented from PlatEMO formulas",
    ),
    BenchmarkRecommendation(
        family="ZDT",
        role="Bi-objective smoke tests only; not a many-objective reduction benchmark.",
        suggested_cases=("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"),
        implementation_status="planned",
    ),
)
