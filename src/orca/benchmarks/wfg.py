"""WFG benchmark adapter backed by the optproblems implementation.

The local implementation intentionally delegates objective evaluation to
``optproblems.wfg``. The upstream module states that it reimplements the
Walking Fish Group problems from the original C++ toolkit published by Huband
et al.; this keeps the formulas traceable instead of retyping them here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from orca.nonlinear.problem_interface import NonlinearORCAProblem


WFGFamily = Literal["WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"]


class WFGProblem(NonlinearORCAProblem):
    """Adapter for WFG1-WFG9 minimization problems.

    Parameters
    ----------
    family:
        One of ``"WFG1"`` through ``"WFG9"``.
    num_objectives:
        Number of objectives.
    num_variables:
        Number of decision variables.
    k:
        Number of position variables. WFG requires ``k % (num_objectives - 1) == 0``.
    finite_diff_step:
        Step size for bound-aware finite-difference gradients.
    """

    def __init__(
        self,
        family: WFGFamily,
        *,
        num_objectives: int = 3,
        num_variables: int = 6,
        k: int | None = None,
        finite_diff_step: float = 1.0e-6,
    ) -> None:
        self.family = family.upper()
        self.num_objectives_value = num_objectives
        self.num_variables_value = num_variables
        self.k = k if k is not None else 2 * (num_objectives - 1)
        self.finite_diff_step = finite_diff_step
        self._problem = self._build_problem()
        self.lower_bounds = np.asarray(self._problem.min_bounds, dtype=float)
        self.upper_bounds = np.asarray(self._problem.max_bounds, dtype=float)

    def _build_problem(self):
        try:
            from optproblems import wfg  # type: ignore
        except ImportError as exc:
            raise ImportError("WFGProblem requires optproblems and diversipy") from exc

        try:
            problem_cls = getattr(wfg, self.family)
        except AttributeError as exc:
            raise ValueError(f"Unknown WFG family: {self.family}") from exc
        return problem_cls(self.num_objectives_value, self.num_variables_value, self.k)

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return self.num_objectives_value

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.asarray(self._problem(self.project_feasible(x_arr)), dtype=float)

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.concatenate([x_arr - self.upper_bounds, self.lower_bounds - x_arr])

    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        x_arr = self.project_feasible(np.asarray(x, dtype=float))
        gradients = np.zeros((self.num_objectives_value, self.num_variables_value), dtype=float)
        for var_idx in range(self.num_variables_value):
            h = min(
                self.finite_diff_step,
                max(self.finite_diff_step * 0.1, self.upper_bounds[var_idx] - self.lower_bounds[var_idx]),
            )
            forward = x_arr.copy()
            backward = x_arr.copy()
            forward[var_idx] = min(self.upper_bounds[var_idx], forward[var_idx] + h)
            backward[var_idx] = max(self.lower_bounds[var_idx], backward[var_idx] - h)
            denom = forward[var_idx] - backward[var_idx]
            if denom <= 0.0:
                continue
            gradients[:, var_idx] = (self.objective_values(forward) - self.objective_values(backward)) / denom
        return gradients

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        _ = np.asarray(x, dtype=float)
        eye = np.eye(self.num_variables_value, dtype=float)
        return np.vstack([eye, -eye])

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(y, dtype=float), self.lower_bounds, self.upper_bounds)

    def initial_points(self) -> list[np.ndarray]:
        """Return deterministic WFG seeds using scaled corner/center points."""

        center = 0.5 * (self.lower_bounds + self.upper_bounds)
        seeds = [center]
        for obj_idx in range(self.num_objectives_value):
            seed = center.copy()
            if obj_idx < self.k:
                seed[obj_idx] = self.lower_bounds[obj_idx]
            else:
                seed[obj_idx % self.num_variables_value] = self.upper_bounds[obj_idx % self.num_variables_value]
            seeds.append(seed)
        return seeds
