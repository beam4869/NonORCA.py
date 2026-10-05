"""Nonlinear benchmark tests for paper-aligned ORCA behavior."""

from __future__ import annotations

import numpy as np

from orca.benchmarks import DTLZ5Problem, DTLZ6Problem, WFGProblem, expected_dtlz5_groups
from orca.config import NonlinearORCAConfig
from orca.data_schema import SampledGradientBlocks
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.gradient_evaluation import evaluate_sampled_gradient_blocks
from orca.nonlinear.main import nonlinear_orca, nonlinear_orca_from_sampled_gradients


def test_nonlinear_sampled_gradients_group_known_correlated_block() -> None:
    """A minimal paper-operator check: f1/f2 correlate, f3 conflicts."""

    points = np.zeros((4, 2), dtype=float)
    objective_gradients = np.tile(
        np.array(
            [
                [-1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        (points.shape[0], 1, 1),
    )
    constraint_jacobians = np.tile(np.array([[[0.0, 1.0]]]), (points.shape[0], 1, 1))
    constraint_values = np.zeros((points.shape[0], 1), dtype=float)

    blocks = SampledGradientBlocks(
        points=points,
        objective_gradients=objective_gradients,
        constraint_jacobians=constraint_jacobians,
        constraint_values=constraint_values,
    )
    result = nonlinear_orca_from_sampled_gradients(
        blocks,
        config=NonlinearORCAConfig(num_groups=2, grouping_method="average_linkage"),
    )

    assert result.groups.tolist() == [1, 1, 2]
    assert result.adj_matrix[0, 1] > 0.99
    assert result.adj_matrix[0, 2] < 0.01
    assert result.adj_matrix[1, 2] < 0.01


def test_dtlz5_benchmark_shapes_and_expected_group_labels() -> None:
    """DTLZ5(I, M) helper exposes the known benchmark structure."""

    problem = DTLZ5Problem(intrinsic_dimension=3, num_objectives=5)
    x = np.full(problem.num_variables(), 0.5, dtype=float)

    assert problem.objective_values(x).shape == (5,)
    assert problem.objective_gradients(x).shape == (5, problem.num_variables())
    assert problem.constraint_values(x).shape == (2 * problem.num_variables(),)
    assert problem.constraint_jacobian(x).shape == (2 * problem.num_variables(), problem.num_variables())
    assert expected_dtlz5_groups(num_objectives=5, intrinsic_dimension=3).tolist() == [1, 1, 1, 2, 3]


def test_dtlz_analytic_gradients_match_finite_difference() -> None:
    """Analytic DTLZ gradients should agree with finite differences away from boundaries."""

    rng = np.random.default_rng(13)
    for problem_cls in [DTLZ5Problem, DTLZ6Problem]:
        analytic = problem_cls(
            intrinsic_dimension=5,
            num_objectives=12,
            gradient_backend="analytic",
            initial_point_strategy="deterministic",
        )
        finite = problem_cls(
            intrinsic_dimension=5,
            num_objectives=12,
            gradient_backend="finite_difference",
            initial_point_strategy="deterministic",
        )
        x = rng.uniform(0.1, 0.9, analytic.num_variables())

        assert np.allclose(analytic.objective_gradients(x), finite.objective_gradients(x), atol=1.0e-7)


def test_dtlz5_fixed_point_generation_stays_feasible() -> None:
    """The nonlinear selected-point step should respect box feasibility."""

    problem = DTLZ5Problem(intrinsic_dimension=2, num_objectives=3, initial_point_strategy="deterministic")
    config = NonlinearORCAConfig(
        num_groups=2,
        num_points_per_seed=2,
        include_seed_points=True,
        random_seed=7,
        step_size=0.03,
        grouping_method="average_linkage",
    )

    points = generate_fixed_points(problem, config)
    values = np.vstack([problem.constraint_values(point) for point in points])

    assert points.shape[1] == problem.num_variables()
    assert np.all(points >= -config.atol)
    assert np.all(points <= 1.0 + config.atol)
    assert np.all(values <= config.atol)


def test_dtlz5_sampled_gradient_pipeline_runs() -> None:
    """Smoke-test DTLZ5 through gradient sampling and nonlinear aggregation."""

    problem = DTLZ5Problem(intrinsic_dimension=2, num_objectives=3, initial_point_strategy="deterministic")
    points = np.asarray(problem.initial_points(), dtype=float)
    blocks = evaluate_sampled_gradient_blocks(problem, points)
    result = nonlinear_orca_from_sampled_gradients(
        blocks,
        config=NonlinearORCAConfig(num_groups=2, grouping_method="average_linkage"),
    )

    assert result.adj_matrix.shape == (3, 3)
    assert result.groups.shape == (3,)
    assert np.allclose(result.adj_matrix, result.adj_matrix.T)
    assert np.all(np.isfinite(result.adj_matrix))


def test_dtlz5_small_known_groupings_are_recovered() -> None:
    """Run small DTLZ5 instances end-to-end against their known group labels."""

    cases = [
        (2, 3, [1, 1, 2]),
        (3, 5, [1, 1, 1, 2, 3]),
    ]

    for intrinsic_dimension, num_objectives, expected_groups in cases:
        problem = DTLZ5Problem(
            intrinsic_dimension=intrinsic_dimension,
            num_objectives=num_objectives,
            initial_point_strategy="deterministic",
        )
        result = nonlinear_orca(
            problem,
            NonlinearORCAConfig(
                num_groups=intrinsic_dimension,
                num_points_per_seed=3,
                include_seed_points=True,
                random_seed=11,
                step_size=0.03,
                grouping_method="average_linkage",
            ),
        )

        assert expected_dtlz5_groups(num_objectives, intrinsic_dimension).tolist() == expected_groups
        assert result.groups.tolist() == expected_groups


def test_dtlz5_5_12_known_grouping_is_recovered() -> None:
    """Recommended many-objective benchmark from the nonlinear paper context."""

    problem = DTLZ5Problem(
        intrinsic_dimension=5,
        num_objectives=12,
        optimizer_backend="scipy",
        gradient_backend="finite_difference",
    )
    result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(
            num_groups=5,
            num_points_per_seed=3,
            include_seed_points=True,
            random_seed=11,
            step_size=0.03,
            grouping_method="average_linkage",
        ),
    )

    assert result.groups.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]


def test_dtlz6_known_groupings_are_recovered() -> None:
    """DTLZ6 shares DTLZ5's known objective correlation structure."""

    cases = [
        (3, 5, [1, 1, 1, 2, 3]),
        (5, 12, [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]),
    ]

    for intrinsic_dimension, num_objectives, expected_groups in cases:
        problem = DTLZ6Problem(intrinsic_dimension=intrinsic_dimension, num_objectives=num_objectives, optimizer_backend="scipy")
        result = nonlinear_orca(
            problem,
            NonlinearORCAConfig(
                num_groups=intrinsic_dimension,
                num_points_per_seed=3,
                include_seed_points=True,
                random_seed=11,
                step_size=0.03,
                grouping_method="average_linkage",
            ),
        )

        assert result.groups.tolist() == expected_groups


def test_dtlz_initial_points_use_single_objective_optimization() -> None:
    """DTLZ initial points should be optimizer-derived single-objective optima."""

    problem = DTLZ5Problem(intrinsic_dimension=3, num_objectives=5, optimizer_backend="scipy")
    seeds = problem.initial_points()

    assert len(seeds) == problem.num_objectives()
    for objective_index, seed in enumerate(seeds):
        values = problem.objective_values(seed)
        assert values[objective_index] <= 1.0e-6


def test_dtlz5_5_12_adjacency_matches_paper_equation_regression() -> None:
    """Numerically pin the DTLZ5(5,12) adjacency produced by Eq. (5)."""

    problem = DTLZ5Problem(
        intrinsic_dimension=5,
        num_objectives=12,
        optimizer_backend="scipy",
        gradient_backend="finite_difference",
    )
    result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(
            num_groups=5,
            num_points_per_seed=3,
            include_seed_points=True,
            random_seed=11,
            step_size=0.03,
            grouping_method="average_linkage",
        ),
    )
    expected = np.array(
        [
            [1.0, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 1.0, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 1.0, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 0.58587021, 1.0, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 0.58587021, 0.58587021, 1.0, 0.58587021, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 1.0, 0.58587021, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 1.0, 0.58587021, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 0.58587021, 1.0, 0.53769439, 0.52076258, 0.45133625, 0.33470556],
            [0.53769439, 0.53769439, 0.53769439, 0.53769439, 0.53769439, 0.53769439, 0.53769439, 0.53769439, 1.0, 0.51631333, 0.45368402, 0.34977252],
            [0.52076258, 0.52076258, 0.52076258, 0.52076258, 0.52076258, 0.52076258, 0.52076258, 0.52076258, 0.51631333, 1.0, 0.434341, 0.32632567],
            [0.45133625, 0.45133625, 0.45133625, 0.45133625, 0.45133625, 0.45133625, 0.45133625, 0.45133625, 0.45368402, 0.434341, 1.0, 0.08712395],
            [0.33470556, 0.33470556, 0.33470556, 0.33470556, 0.33470556, 0.33470556, 0.33470556, 0.33470556, 0.34977252, 0.32632567, 0.08712395, 1.0],
        ]
    )

    assert np.allclose(result.adj_matrix, expected, atol=1.0e-8)


def test_wfg_adapter_matches_optproblems_reference_values() -> None:
    """Smoke-test WFG1/WFG2/WFG3/WFG9 against optproblems reference outputs."""

    expected_values = {
        "WFG1": [2.886792851925874, 0.9732684630579093, 0.9749048137207078],
        "WFG2": [0.3254190290999637, 0.49699190435377366, 6.153846153846154],
        "WFG3": [0.6538461538461539, 1.1538461538461537, 3.1538461538461537],
        "WFG9": [1.1017221799341073, 2.030509910297282, 4.133564616000229],
    }

    for family, expected in expected_values.items():
        problem = WFGProblem(family, num_objectives=3, num_variables=6, k=4)
        x = problem.initial_points()[0]
        assert np.allclose(problem.objective_values(x), np.asarray(expected), atol=1.0e-12)
        assert problem.objective_gradients(x).shape == (3, 6)
        assert problem.constraint_jacobian(x).shape == (12, 6)
