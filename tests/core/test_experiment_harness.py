"""Tests for baseline and downstream optimizer experiment helpers."""

from __future__ import annotations

import numpy as np

from orca.benchmarks import DTLZ5Problem, DTLZ6Problem, LSMOPProblem, expected_dtlz5_groups
from orca.experiments import (
    adjusted_rand_index,
    compare_full_vs_reduced_nsga3,
    compare_grouping_methods,
    run_reduction_study,
    run_runtime_scaling_case,
    summarize_reduction_rows,
)


def test_adjusted_rand_index_matches_basic_cases() -> None:
    """ARI should be label-permutation invariant and penalize unrelated labels."""

    assert adjusted_rand_index([1, 1, 2, 2], [7, 7, 8, 8]) == 1.0
    assert adjusted_rand_index([1, 1, 2, 2], [1, 2, 1, 2]) < 0.0


def test_compare_grouping_methods_on_known_dtlz5() -> None:
    """ORCA and simple baselines should be measurable against known DTLZ5 labels."""

    problem = DTLZ5Problem(
        intrinsic_dimension=3,
        num_objectives=5,
        initial_point_strategy="deterministic",
    )
    expected = expected_dtlz5_groups(num_objectives=5, intrinsic_dimension=3)
    results = compare_grouping_methods(
        problem,
        expected,
        num_groups=3,
        sample_points=64,
        random_seed=2,
    )

    by_method = {result.method: result for result in results}
    assert set(by_method) == {"ORCA", "objective_value_correlation", "gradient_cosine", "random_control"}
    assert by_method["ORCA"].groups.tolist() == expected.tolist()
    assert by_method["ORCA"].ari == 1.0
    assert by_method["random_control"].ari < 1.0
    for result in results:
        assert result.adj_matrix.shape == (5, 5)


def test_downstream_nsga3_harness_returns_full_space_metrics() -> None:
    """Full and reduced NSGA-III runs should be compared in original objective space."""

    problem = DTLZ5Problem(
        intrinsic_dimension=3,
        num_objectives=5,
        initial_point_strategy="deterministic",
    )
    expected = expected_dtlz5_groups(num_objectives=5, intrinsic_dimension=3)
    result = compare_full_vs_reduced_nsga3(
        problem,
        expected,
        min_population_size=12,
        n_gen=3,
        random_seed=1,
    )

    assert result.groups.tolist() == expected.tolist()
    assert result.full_objectives.shape[1] == problem.num_objectives()
    assert result.reduced_objectives.shape[1] == problem.num_objectives()
    assert result.reference_objectives.shape[1] == problem.num_objectives()
    assert np.isfinite(result.full_hv)
    assert np.isfinite(result.reduced_hv)
    assert np.isfinite(result.full_igd)
    assert np.isfinite(result.reduced_igd)
    assert result.full_runtime_seconds >= 0.0
    assert result.reduced_runtime_seconds >= 0.0


def test_reduction_study_runs_all_reduction_methods() -> None:
    """A multi-method reduction study should summarize full and reduced runs."""

    problem = DTLZ5Problem(
        intrinsic_dimension=3,
        num_objectives=5,
        initial_point_strategy="deterministic",
    )
    expected = expected_dtlz5_groups(num_objectives=5, intrinsic_dimension=3)
    result = run_reduction_study(
        "DTLZ5(3,5)",
        problem,
        expected,
        num_groups=3,
        seeds=[1],
        n_gen=2,
        min_population_size=8,
        sample_points=32,
    )
    methods = {row.method for row in result.rows}

    assert methods == {"full", "ORCA", "objective_value_correlation", "gradient_cosine"}
    assert len(result.rows) == 4
    assert len(summarize_reduction_rows(result.rows)) == 4


def test_lsmop_and_runtime_scaling_smoke() -> None:
    """LSMOP adapter and runtime-scaling runner should produce timing rows."""

    lsmop = LSMOPProblem("LSMOP1", num_objectives=3, num_variables=30)
    x = lsmop.initial_points()[0]

    assert lsmop.objective_values(x).shape == (3,)
    assert lsmop.objective_gradients(x).shape == (3, 30)
    assert np.max(lsmop.constraint_values(x)) <= 0.0

    dtlz = DTLZ6Problem(
        intrinsic_dimension=3,
        num_objectives=5,
        k_tail=5,
        initial_point_strategy="deterministic",
    )
    row = run_runtime_scaling_case(
        "DTLZ6-smoke",
        dtlz,
        num_groups=3,
        n_gen=2,
        min_population_size=8,
        orca_point_limit=3,
    )

    assert row.orca_grouping_seconds >= 0.0
    assert row.full_optimizer_seconds >= 0.0
    assert row.reduced_optimizer_seconds >= 0.0
    assert row.orca_plus_reduced_seconds >= row.reduced_optimizer_seconds
