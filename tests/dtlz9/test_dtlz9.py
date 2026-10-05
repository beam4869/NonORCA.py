"""Deterministic unit tests for DTLZ9 analytical geometry."""

from __future__ import annotations

import numpy as np
import inspect
import sys

from experiments.dtlz9.geometry import (
    expected_signed_matrix,
    joint_signed_matrix,
    joint_tangent_projector,
    matrices_from_sampled_blocks,
    positive_component_groups,
    sampled_blocks,
)
from experiments.dtlz9.problem import DTLZ9Problem


def test_exact_pf_values_constraints_and_block_metadata() -> None:
    problem = DTLZ9Problem(5, 50)
    theta = 0.7
    x = problem.exact_pf_decision(theta)
    values = problem.objective_values(x)

    assert x.shape == (50,)
    assert np.all(x > 0.0)
    assert np.allclose(values[:-1], np.cos(theta), atol=1.0e-13)
    assert np.isclose(values[-1], np.sin(theta), atol=1.0e-13)
    assert np.max(np.abs(problem.constraint_values(x))) <= 1.0e-13
    assert [item.size for item in problem.block_metadata()] == [10] * 5


def test_analytical_objective_and_constraint_jacobians_match_finite_difference() -> None:
    problem = DTLZ9Problem(3, 3)
    x = problem.exact_pf_decision(np.pi / 4.0)
    h = 1.0e-7

    objective_fd = np.zeros_like(problem.objective_gradients(x))
    constraint_fd = np.zeros_like(problem.constraint_jacobian(x))
    for idx in range(problem.num_variables()):
        step = np.zeros(problem.num_variables())
        step[idx] = h
        objective_fd[:, idx] = (
            problem.objective_values(x + step) - problem.objective_values(x - step)
        ) / (2.0 * h)
        constraint_fd[:, idx] = (
            problem.constraint_values(x + step) - problem.constraint_values(x - step)
        ) / (2.0 * h)

    assert np.allclose(problem.objective_gradients(x), objective_fd, rtol=2.0e-7, atol=1.0e-8)
    assert np.allclose(problem.constraint_jacobian(x), constraint_fd, rtol=2.0e-7, atol=1.0e-8)


def test_joint_tangent_matrix_matches_analytical_ground_truth() -> None:
    for m in (3, 5, 10):
        for block_size in (1, 10):
            problem = DTLZ9Problem(m, m * block_size)
            for theta in np.linspace(0.15, 0.5 * np.pi - 0.15, 7):
                x = problem.exact_pf_decision(float(theta))
                matrix, diagnostics = joint_signed_matrix(
                    problem.objective_gradients(x), problem.constraint_jacobian(x)
                )
                assert np.allclose(matrix, expected_signed_matrix(m), atol=1.0e-10)
                assert diagnostics.rank == m - 1
                assert diagnostics.symmetry_residual <= 1.0e-12
                assert diagnostics.idempotence_residual <= 1.0e-12
                assert diagnostics.tangent_residual <= 1.0e-12


def test_normal_extension_and_row_mixing_invariance() -> None:
    rng = np.random.default_rng(17)
    problem = DTLZ9Problem(5, 50)
    x = problem.exact_pf_decision(0.7)
    gradients = problem.objective_gradients(x)
    jacobian = problem.constraint_jacobian(x)
    reference, _ = joint_signed_matrix(gradients, jacobian)

    alpha = rng.normal(size=(5, 4))
    extended, _ = joint_signed_matrix(gradients + alpha @ jacobian, jacobian)
    assert np.allclose(reference, extended, atol=1.0e-10)

    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    mixed = joint_tangent_projector(q @ jacobian).projector
    base = joint_tangent_projector(jacobian).projector
    assert np.allclose(base, mixed, atol=1.0e-10)


def test_current_orca_has_neutral_within_group_edges_and_decaying_competition() -> None:
    alpha = 0.9
    neutral_weight = 1.0 - alpha / 2.0
    for m in (3, 5, 10):
        problem = DTLZ9Problem(m, m)
        points = problem.exact_pf_decisions(np.linspace(0.2, 0.5 * np.pi - 0.2, 9))
        matrices, _ = matrices_from_sampled_blocks(sampled_blocks(problem, points))
        current = matrices["ORCA-current"]
        expected_cross = -1.0 / (1.0 + neutral_weight * (m - 2))
        within = current[:-1, :-1][np.triu_indices(m - 1, k=1)]

        assert np.allclose(within, 0.0, atol=1.0e-12)
        assert np.allclose(current[:-1, -1], expected_cross, atol=1.0e-12)


def test_unknown_k_recovery_distinguishes_current_and_joint() -> None:
    problem = DTLZ9Problem(5, 5)
    points = problem.exact_pf_decisions(np.linspace(0.2, 0.5 * np.pi - 0.2, 9))
    matrices, _ = matrices_from_sampled_blocks(sampled_blocks(problem, points))

    current_groups = positive_component_groups(matrices["ORCA-current"])
    joint_groups = positive_component_groups(matrices["ORCA-joint"])
    assert np.unique(current_groups).size == 5
    assert joint_groups.tolist() == [1, 1, 1, 1, 2]


def main() -> None:
    tests = [
        function
        for name, function in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(function)
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}", flush=True)
        except Exception as exc:  # pragma: no cover - command-line reporting
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {exc}", flush=True)
    print(f"{len(tests) - len(failures)}/{len(tests)} tests passed", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
