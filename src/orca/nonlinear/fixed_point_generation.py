"""Selected fixed-point generation for nonlinear ORCA."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from orca.config import NonlinearORCAConfig
from orca.nonlinear.feasible_projection import project_with_problem_method
from orca.nonlinear.problem_interface import NonlinearORCAProblem
from orca.utils.constraint_projection import tangent_component
from orca.utils.vector_normalization import safe_normalize


def select_constraint_for_direction(
    constraint_values: Optional[np.ndarray],
    constraint_jacobian: np.ndarray,
    *,
    rule: str = "active_or_nearest",
    atol: float = 1.0e-14,
) -> int:
    """Select a constraint normal for fixed-point direction generation.

    ``nearest`` and ``active_or_nearest`` both choose the largest constraint
    value, which is the closest-to-active inequality when constraints are
    represented as ``g_k(x) <= 0``. If constraint values are unavailable, the
    first nonzero Jacobian row is used.
    """

    jac = np.asarray(constraint_jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("constraint_jacobian must be two-dimensional")
    if constraint_values is not None:
        values = np.asarray(constraint_values, dtype=float)
        if values.ndim != 1 or values.shape[0] != jac.shape[0]:
            raise ValueError("constraint_values must match the number of constraint rows")
        if rule not in {"active_or_nearest", "nearest"}:
            raise ValueError(f"Unknown constraint selection rule: {rule}")
        return int(np.argmax(values))

    for idx, row in enumerate(jac):
        if np.linalg.norm(row) > atol:
            return idx
    raise ValueError("No nonzero constraint Jacobian rows are available")


def generate_conic_descent_direction(
    objective_gradients: np.ndarray,
    constraint_normal: Optional[np.ndarray],
    rng: np.random.Generator,
    *,
    normalize_direction: bool = True,
    atol: float = 1.0e-14,
) -> np.ndarray:
    """Generate a random conic combination of projected objective descent directions."""

    grads = np.asarray(objective_gradients, dtype=float)
    if grads.ndim != 2:
        raise ValueError("objective_gradients must have shape (num_objectives, num_variables)")

    projected_dirs = []
    for grad in grads:
        descent = safe_normalize(-grad, atol=atol)
        if constraint_normal is not None and np.linalg.norm(constraint_normal) > atol:
            candidate = tangent_component(descent, constraint_normal, atol=atol)
            candidate = safe_normalize(candidate, atol=atol)
            if np.linalg.norm(candidate) <= atol:
                candidate = descent
        else:
            candidate = descent
        projected_dirs.append(candidate)

    dirs = np.vstack(projected_dirs)
    weights = rng.random(dirs.shape[0])
    weight_sum = float(np.sum(weights))
    if weight_sum <= atol:
        weights = np.ones(dirs.shape[0], dtype=float)
        weight_sum = float(dirs.shape[0])
    delta = np.sum(weights[:, None] * dirs, axis=0) / weight_sum
    if normalize_direction:
        delta = safe_normalize(delta, atol=atol)
    return delta


def generate_fixed_points(
    problem: NonlinearORCAProblem,
    config: Optional[NonlinearORCAConfig] = None,
    *,
    seeds: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Generate selected feasible fixed points for nonlinear ORCA.

    Starting from each seed point, this repeatedly generates a conic direction
    from projected objective descent directions, takes a step, and projects the
    trial point back to the feasible set.
    """

    cfg = config or NonlinearORCAConfig()
    rng = np.random.default_rng(cfg.random_seed)
    seed_points = list(seeds) if seeds is not None else list(problem.initial_points())
    if not seed_points:
        raise ValueError("At least one seed point is required")

    selected: list[np.ndarray] = []

    def keep_point(candidate: np.ndarray) -> bool:
        if not selected:
            return True
        distances = [float(np.linalg.norm(candidate - existing)) for existing in selected]
        return min(distances) >= cfg.min_point_distance

    for seed in seed_points:
        current = project_with_problem_method(problem, np.asarray(seed, dtype=float))
        if cfg.include_seed_points and keep_point(current):
            selected.append(current.copy())

        failures = 0
        accepted_from_seed = 0
        while accepted_from_seed < cfg.num_points_per_seed:
            obj_grads = np.asarray(problem.objective_gradients(current), dtype=float)
            con_jac = np.asarray(problem.constraint_jacobian(current), dtype=float)
            con_vals = np.asarray(problem.constraint_values(current), dtype=float)

            try:
                constraint_idx = select_constraint_for_direction(
                    con_vals,
                    con_jac,
                    rule=cfg.constraint_selection,
                    atol=cfg.atol,
                )
                normal = con_jac[constraint_idx, :]
                delta = generate_conic_descent_direction(
                    obj_grads,
                    normal,
                    rng,
                    normalize_direction=cfg.normalize_generated_direction,
                    atol=cfg.atol,
                )
                trial = current + cfg.step_size * delta
                projected = project_with_problem_method(problem, trial)
            except Exception:
                failures += 1
                if failures >= cfg.max_projection_failures:
                    break
                continue

            if keep_point(projected):
                selected.append(projected.copy())
                current = projected
                accepted_from_seed += 1
                failures = 0
            else:
                failures += 1
                current = projected
                if failures >= cfg.max_projection_failures:
                    break

    if not selected:
        raise RuntimeError("No selected fixed points were generated")
    return np.vstack(selected)
