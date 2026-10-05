"""Benchmark problems for ORCA validation experiments."""

from orca.benchmarks.constrained import (
    CEC2009CFProblem,
    ConstrainedDTLZ6Problem,
    LIRCMOPProblem,
    PymooConstrainedProblem,
)
from orca.benchmarks.dtlz import DTLZ5Problem, DTLZ6Problem, expected_dtlz5_groups
from orca.benchmarks.lsmop import LSMOPProblem
from orca.benchmarks.registry import BenchmarkRecommendation, RECOMMENDED_BENCHMARKS
from orca.benchmarks.wfg import WFGProblem

__all__ = [
    "BenchmarkRecommendation",
    "CEC2009CFProblem",
    "ConstrainedDTLZ6Problem",
    "DTLZ5Problem",
    "DTLZ6Problem",
    "LSMOPProblem",
    "LIRCMOPProblem",
    "PymooConstrainedProblem",
    "RECOMMENDED_BENCHMARKS",
    "WFGProblem",
    "expected_dtlz5_groups",
]
