"""Constrained benchmark coverage for nonlinear ORCA."""

from __future__ import annotations

import numpy as np
import pytest

from orca.benchmarks import (
    CEC2009CFProblem,
    ConstrainedDTLZ6Problem,
    LIRCMOPProblem,
    PymooConstrainedProblem,
)
from orca.config import NonlinearORCAConfig
from orca.nonlinear.main import nonlinear_orca


def test_constrained_dtlz6_has_nonconvex_feasible_region_and_known_grouping() -> None:
    """Circular-hole DTLZ6 keeps the DTLZ6 grouping while making feasibility nonconvex."""

    problem = ConstrainedDTLZ6Problem(intrinsic_dimension=5, num_objectives=12, optimizer_backend="scipy")
    left = np.full(problem.num_variables(), 0.5, dtype=float)
    right = left.copy()
    midpoint = left.copy()
    hole_i, hole_j = problem.hole_variables
    left[[hole_i, hole_j]] = [0.50, 0.70]
    right[[hole_i, hole_j]] = [0.90, 0.70]
    midpoint[[hole_i, hole_j]] = [0.70, 0.70]

    assert np.max(problem.constraint_values(left)) <= 0.0
    assert np.max(problem.constraint_values(right)) <= 0.0
    assert np.max(problem.constraint_values(midpoint)) > 0.0

    result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(
            num_groups=5,
            num_points_per_seed=2,
            include_seed_points=True,
            random_seed=11,
            step_size=0.03,
            grouping_method="average_linkage",
        ),
    )

    assert result.groups.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]
    assert np.all(np.isfinite(result.adj_matrix))


@pytest.mark.parametrize(
    ("name", "expected_objectives", "expected_constraints"),
    [
        ("c2dtlz2", [0.5, 0.5, 0.7071067811865475], [-0.13119711930697767]),
        ("c3dtlz4", [2.25, 0.0, 0.0], [-0.265625, -4.0625, -4.0625]),
    ],
)
def test_pymoo_cdtlz_adapter_reference_values(
    name: str,
    expected_objectives: list[float],
    expected_constraints: list[float],
) -> None:
    """C2-DTLZ2/C3-DTLZ4 adapter should match pymoo's reference values."""

    problem = PymooConstrainedProblem(name)  # type: ignore[arg-type]
    x = np.full(problem.num_variables(), 0.5, dtype=float)
    if name == "c3dtlz4":
        x[:] = 0.0

    assert np.allclose(problem.objective_values(x), np.asarray(expected_objectives), atol=1.0e-12)
    assert np.allclose(problem.constraint_values(x)[: len(expected_constraints)], expected_constraints, atol=1.0e-12)


@pytest.mark.parametrize("family", ["LIRCMOP5", "LIRCMOP6", "LIRCMOP13", "LIRCMOP14"])
def test_lircmop_feasible_seeds_and_shapes(family: str) -> None:
    """LIR-CMOP seeds should be feasible under the PlatEMO sign convention g <= 0."""

    problem = LIRCMOPProblem(family)  # type: ignore[arg-type]
    seeds = problem.initial_points()

    assert len(seeds) >= problem.num_objectives() + 1
    for seed in seeds:
        assert seed.shape == (problem.num_variables(),)
        assert problem.objective_values(seed).shape == (problem.num_objectives(),)
        assert np.max(problem.constraint_values(seed)) <= 1.0e-8
        assert problem.objective_gradients(seed).shape == (problem.num_objectives(), problem.num_variables())
        assert problem.constraint_jacobian(seed).shape[1] == problem.num_variables()


@pytest.mark.parametrize(
    ("family", "expected_objectives", "expected_constraint"),
    [
        ("CF8", [1.3434972591590472, 0.8412461185709182, 0.867530913300768], [-7.578859399800164]),
        ("CF9", [1.3434972591590472, 0.8412461185709182, 0.867530913300768], [-7.973332403514086]),
        ("CF10", [3.2700504822156253, 1.957501861797347, 5.007289958041987], [0.6274949140092855]),
    ],
)
def test_cec2009_cf_reference_values(
    family: str,
    expected_objectives: list[float],
    expected_constraint: list[float],
) -> None:
    """Pin CF8-CF10 values against the CEC2009/MOEA Framework formulas."""

    problem = CEC2009CFProblem(family)  # type: ignore[arg-type]
    x = np.zeros(problem.num_variables(), dtype=float)
    x[0] = 0.2
    x[1] = 0.4

    assert np.allclose(problem.objective_values(x), np.asarray(expected_objectives), atol=1.0e-12)
    assert np.allclose(problem.constraint_values(x)[:1], np.asarray(expected_constraint), atol=1.0e-12)


@pytest.mark.parametrize(
    "problem",
    [
        PymooConstrainedProblem("c2dtlz2"),
        PymooConstrainedProblem("c3dtlz4"),
        LIRCMOPProblem("LIRCMOP5"),
        LIRCMOPProblem("LIRCMOP6"),
        LIRCMOPProblem("LIRCMOP13"),
        LIRCMOPProblem("LIRCMOP14"),
        CEC2009CFProblem("CF8"),
        CEC2009CFProblem("CF9"),
        CEC2009CFProblem("CF10"),
    ],
)
def test_constrained_benchmarks_run_through_orca_from_feasible_points(problem) -> None:
    """Smoke-test each constrained benchmark through nonlinear ORCA aggregation."""

    points = np.asarray(problem.initial_points(), dtype=float)
    result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(
            num_groups=min(problem.num_objectives(), 2),
            grouping_method="average_linkage",
        ),
        points=points,
    )

    assert result.adj_matrix.shape == (problem.num_objectives(), problem.num_objectives())
    assert result.groups.shape == (problem.num_objectives(),)
    assert np.allclose(result.adj_matrix, result.adj_matrix.T)
    assert np.all(np.isfinite(result.adj_matrix))
    for point in result.input_data.points:
        assert np.max(problem.constraint_values(point)) <= 1.0e-6
