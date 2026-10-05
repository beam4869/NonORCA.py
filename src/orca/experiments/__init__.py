"""Reusable experiment helpers for ORCA validation."""

from orca.experiments.baseline_comparison import (
    BaselineGroupingResult,
    adjusted_rand_index,
    compare_grouping_methods,
)
from orca.experiments.downstream_optimizer import (
    DownstreamComparisonResult,
    compare_full_vs_reduced_nsga3,
)
from orca.experiments.reduction_study import (
    ReductionStudyResult,
    ReductionStudyRow,
    run_reduction_study,
    summarize_reduction_rows,
)
from orca.experiments.runtime_scaling import RuntimeScalingRow, run_runtime_scaling_case

__all__ = [
    "BaselineGroupingResult",
    "DownstreamComparisonResult",
    "ReductionStudyResult",
    "ReductionStudyRow",
    "RuntimeScalingRow",
    "adjusted_rand_index",
    "compare_full_vs_reduced_nsga3",
    "compare_grouping_methods",
    "run_reduction_study",
    "run_runtime_scaling_case",
    "summarize_reduction_rows",
]
