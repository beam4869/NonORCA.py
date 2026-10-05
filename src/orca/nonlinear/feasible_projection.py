"""Feasible projection helpers for nonlinear ORCA fixed-point generation."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from orca.nonlinear.problem_interface import NonlinearORCAProblem


def project_box_feasible(y: np.ndarray, lower: np.ndarray | float, upper: np.ndarray | float) -> np.ndarray:
    """Project onto box bounds by clipping."""

    return np.clip(np.asarray(y, dtype=float), lower, upper)


def project_with_problem_method(problem: NonlinearORCAProblem, y: np.ndarray) -> np.ndarray:
    """Project using the problem's own ``project_feasible`` method."""

    return np.asarray(problem.project_feasible(np.asarray(y, dtype=float)), dtype=float)


def project_with_scipy(
    y: np.ndarray,
    *,
    constraint_values: Callable[[np.ndarray], np.ndarray],
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    x0: Optional[np.ndarray] = None,
    tol: float = 1.0e-9,
    maxiter: int = 500,
) -> np.ndarray:
    """Project to ``g(x) <= 0`` constraints using SciPy SLSQP.

    This utility is optional and is not used unless called explicitly. It solves

        min_x ||x - y||_2^2
        s.t. g_k(x) <= 0.
    """

    try:
        from scipy.optimize import minimize  # type: ignore
    except ImportError as exc:
        raise ImportError("project_with_scipy requires scipy") from exc

    target = np.asarray(y, dtype=float)
    start = np.asarray(x0, dtype=float) if x0 is not None else target.copy()

    def objective(x: np.ndarray) -> float:
        diff = x - target
        return float(np.dot(diff, diff))

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x, idx=idx: -float(np.asarray(constraint_values(x), dtype=float)[idx]),
        }
        for idx in range(len(np.asarray(constraint_values(start), dtype=float)))
    ]

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": maxiter},
    )
    if not result.success:
        raise RuntimeError(f"Feasible projection failed: {result.message}")
    return np.asarray(result.x, dtype=float)
