"""Constrained nonlinear benchmarks for ORCA validation.

This module focuses on nonconvex feasible regions:

* ``ConstrainedDTLZ6Problem`` keeps the DTLZ6 objective-reduction structure and
  adds a circular hole in decision space.
* ``PymooConstrainedProblem`` delegates C2-DTLZ2/C3-DTLZ4 objective and
  constraint values to pymoo's CDTLZ implementations.
* ``LIRCMOPProblem`` implements the PlatEMO LIR-CMOP5/6/13/14 formulas.
* ``CEC2009CFProblem`` implements CEC2009 CF8/CF9/CF10 from the CEC2009 report
  and the MOEA Framework reference implementation.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence

import numpy as np

from orca.benchmarks.dtlz import DTLZ6Problem
from orca.nonlinear.feasible_projection import project_with_scipy
from orca.nonlinear.problem_interface import NonlinearORCAProblem


CDTLZName = Literal["c2dtlz2", "c3dtlz4"]
LIRCMOPName = Literal["LIRCMOP5", "LIRCMOP6", "LIRCMOP13", "LIRCMOP14"]
CEC2009CFName = Literal["CF8", "CF9", "CF10"]


def _box_constraint_values(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.concatenate([x - upper, lower - x])


def _box_constraint_jacobian(num_variables: int) -> np.ndarray:
    eye = np.eye(num_variables, dtype=float)
    return np.vstack([eye, -eye])


def _bound_aware_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    finite_diff_step: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    base = np.asarray(func(x_arr), dtype=float)
    jac = np.zeros((base.size, x_arr.size), dtype=float)
    for var_idx in range(x_arr.size):
        h = min(
            finite_diff_step,
            max(finite_diff_step * 0.1, float(upper[var_idx] - lower[var_idx])),
        )
        forward = x_arr.copy()
        backward = x_arr.copy()
        forward[var_idx] = min(upper[var_idx], forward[var_idx] + h)
        backward[var_idx] = max(lower[var_idx], backward[var_idx] - h)
        denom = forward[var_idx] - backward[var_idx]
        if denom <= 0.0:
            continue
        jac[:, var_idx] = (np.asarray(func(forward), dtype=float) - np.asarray(func(backward), dtype=float)) / denom
    return jac


def _unique_points(points: Sequence[np.ndarray], *, atol: float = 1.0e-10) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        arr = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(arr - existing) <= atol for existing in unique):
            unique.append(arr)
    return unique


class _BoundedFiniteDifferenceProblem(NonlinearORCAProblem):
    """Shared utilities for bounded finite-difference benchmark adapters."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    finite_diff_step: float
    projection_maxiter: int

    def _problem_constraint_values(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _project_with_slsqp(self, y: np.ndarray, starts: Sequence[np.ndarray] = ()) -> np.ndarray:
        clipped = np.clip(np.asarray(y, dtype=float), self.lower_bounds, self.upper_bounds)
        if np.all(self.constraint_values(clipped) <= 1.0e-8):
            return clipped

        bounds = list(zip(self.lower_bounds.tolist(), self.upper_bounds.tolist()))
        candidates = [clipped, *[np.asarray(start, dtype=float) for start in starts]]
        last_error: Exception | None = None
        for start in candidates:
            try:
                projected = project_with_scipy(
                    clipped,
                    constraint_values=self.constraint_values,
                    bounds=bounds,
                    x0=np.clip(start, self.lower_bounds, self.upper_bounds),
                    maxiter=self.projection_maxiter,
                )
                projected = np.clip(projected, self.lower_bounds, self.upper_bounds)
                if np.all(self.constraint_values(projected) <= 1.0e-6):
                    return projected
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"Feasible projection failed from all starts: {last_error}") from last_error
        raise RuntimeError("Feasible projection failed from all starts")

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.concatenate(
            [
                self._problem_constraint_values(x_arr),
                _box_constraint_values(x_arr, self.lower_bounds, self.upper_bounds),
            ]
        )

    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        return _bound_aware_jacobian(
            self.objective_values,
            self.project_feasible(np.asarray(x, dtype=float)),
            self.lower_bounds,
            self.upper_bounds,
            finite_diff_step=self.finite_diff_step,
        )

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        x_arr = self.project_feasible(np.asarray(x, dtype=float))
        problem_jac = _bound_aware_jacobian(
            self._problem_constraint_values,
            x_arr,
            self.lower_bounds,
            self.upper_bounds,
            finite_diff_step=self.finite_diff_step,
        )
        return np.vstack([problem_jac, _box_constraint_jacobian(self.num_variables())])


class ConstrainedDTLZ6Problem(DTLZ6Problem):
    """DTLZ6(I, M) with a circular hole, giving a nonconvex feasible region.

    The DTLZ6 objectives and known objective grouping remain unchanged. The
    extra inequality is represented as

    ``radius**2 - ||x[pair] - center||_2**2 <= 0``.
    """

    def __init__(
        self,
        intrinsic_dimension: int = 5,
        num_objectives: int = 12,
        *,
        hole_center: tuple[float, float] = (0.7, 0.7),
        hole_radius: float = 0.15,
        hole_variables: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(intrinsic_dimension=intrinsic_dimension, num_objectives=num_objectives, **kwargs)
        if hole_radius <= 0.0:
            raise ValueError("hole_radius must be positive")
        self.hole_center = np.asarray(hole_center, dtype=float)
        self.hole_radius = float(hole_radius)
        self.hole_variables = hole_variables or (num_objectives - 1, num_objectives)
        if len(set(self.hole_variables)) != 2:
            raise ValueError("hole_variables must contain two distinct indices")
        if min(self.hole_variables) < 0 or max(self.hole_variables) >= self.num_variables_value:
            raise ValueError("hole_variables are outside the decision vector")

    def _hole_constraint_value(self, x: np.ndarray) -> float:
        pair = x[list(self.hole_variables)]
        diff = pair - self.hole_center
        return float(self.hole_radius**2 - np.dot(diff, diff))

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        base = super().constraint_values(x)
        return np.concatenate([base, [self._hole_constraint_value(np.asarray(x, dtype=float))]])

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        jac = super().constraint_jacobian(x)
        x_arr = np.asarray(x, dtype=float)
        row = np.zeros(self.num_variables_value, dtype=float)
        for local_idx, var_idx in enumerate(self.hole_variables):
            row[var_idx] = -2.0 * (x_arr[var_idx] - self.hole_center[local_idx])
        return np.vstack([jac, row])

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
        if self._hole_constraint_value(x) <= 0.0:
            return x

        idx = list(self.hole_variables)
        direction = x[idx] - self.hole_center
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-14:
            direction = np.array([1.0, 0.0], dtype=float)
            norm = 1.0
        x[idx] = self.hole_center + (self.hole_radius + 1.0e-10) * direction / norm
        return np.clip(x, 0.0, 1.0)


class PymooConstrainedProblem(_BoundedFiniteDifferenceProblem):
    """Adapter for pymoo's C2-DTLZ2 and C3-DTLZ4 implementations."""

    def __init__(
        self,
        name: CDTLZName,
        *,
        num_variables: int | None = None,
        num_objectives: int | None = None,
        finite_diff_step: float = 1.0e-6,
        projection_maxiter: int = 500,
    ) -> None:
        self.name = name.lower()
        self.finite_diff_step = finite_diff_step
        self.projection_maxiter = projection_maxiter

        try:
            from pymoo.problems import get_problem  # type: ignore
        except ImportError as exc:
            raise ImportError("PymooConstrainedProblem requires pymoo") from exc

        kwargs = {}
        if num_variables is not None:
            kwargs["n_var"] = num_variables
        if num_objectives is not None:
            kwargs["n_obj"] = num_objectives
        self._problem = get_problem(self.name, **kwargs)
        self.lower_bounds = np.asarray(self._problem.xl, dtype=float)
        self.upper_bounds = np.asarray(self._problem.xu, dtype=float)
        self.num_problem_constraints = int(getattr(self._problem, "n_ieq_constr", 0))

    def num_variables(self) -> int:
        return int(self._problem.n_var)

    def num_objectives(self) -> int:
        return int(self._problem.n_obj)

    def _evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(np.asarray(x, dtype=float), self.lower_bounds, self.upper_bounds)
        values = self._problem.evaluate(clipped, return_values_of=["F", "G"])
        if isinstance(values, tuple):
            objectives, constraints = values
        else:
            objectives, constraints = values, np.empty(0, dtype=float)
        return np.atleast_1d(np.asarray(objectives, dtype=float)), np.atleast_1d(np.asarray(constraints, dtype=float))

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        objectives, _ = self._evaluate(x)
        return objectives

    def _problem_constraint_values(self, x: np.ndarray) -> np.ndarray:
        _, constraints = self._evaluate(x)
        return constraints

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return self._project_with_slsqp(y, starts=self.initial_points())

    def initial_points(self) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []
        center = 0.5 * (self.lower_bounds + self.upper_bounds)
        candidates.extend([center, self.lower_bounds.copy(), self.upper_bounds.copy()])
        for value in np.linspace(0.0, 1.0, 5):
            candidates.append(np.full(self.num_variables(), value, dtype=float))

        rng = np.random.default_rng(23)
        for _ in range(512):
            candidates.append(rng.uniform(self.lower_bounds, self.upper_bounds))

        feasible = [
            np.clip(point, self.lower_bounds, self.upper_bounds)
            for point in candidates
            if np.all(self.constraint_values(np.clip(point, self.lower_bounds, self.upper_bounds)) <= 1.0e-8)
        ]
        if feasible:
            return _unique_points(feasible)[: max(self.num_objectives() + 1, 4)]
        raise RuntimeError(f"No feasible seed found for {self.name}")


class LIRCMOPProblem(_BoundedFiniteDifferenceProblem):
    """LIR-CMOP5/6/13/14 formulas from PlatEMO."""

    def __init__(
        self,
        family: LIRCMOPName,
        *,
        num_variables: int = 30,
        finite_diff_step: float = 1.0e-6,
        projection_maxiter: int = 500,
    ) -> None:
        self.family = family.upper()
        if self.family not in {"LIRCMOP5", "LIRCMOP6", "LIRCMOP13", "LIRCMOP14"}:
            raise ValueError(f"Unsupported LIR-CMOP family: {family}")
        if num_variables < 4:
            raise ValueError("LIR-CMOP requires at least 4 variables")
        self.num_variables_value = num_variables
        self.finite_diff_step = finite_diff_step
        self.projection_maxiter = projection_maxiter
        self.lower_bounds = np.zeros(num_variables, dtype=float)
        self.upper_bounds = np.ones(num_variables, dtype=float)

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return 3 if self.family in {"LIRCMOP13", "LIRCMOP14"} else 2

    def _g1_g2(self, x: np.ndarray) -> tuple[float, float]:
        sum1 = 0.0
        sum2 = 0.0
        for zero_idx in range(1, self.num_variables_value):
            matlab_idx = zero_idx + 1
            angle = 0.5 * matlab_idx / self.num_variables_value * math.pi * x[0]
            if matlab_idx % 2 == 1:
                sum1 += (x[zero_idx] - math.sin(angle)) ** 2
            else:
                sum2 += (x[zero_idx] - math.cos(angle)) ** 2
        return sum1, sum2

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.clip(np.asarray(x, dtype=float), self.lower_bounds, self.upper_bounds)
        gx = 0.7057
        if self.family in {"LIRCMOP5", "LIRCMOP6"}:
            g1, g2 = self._g1_g2(x_arr)
            second_shape = 1.0 - math.sqrt(x_arr[0]) if self.family == "LIRCMOP5" else 1.0 - x_arr[0] ** 2
            return np.array([x_arr[0] + 10.0 * g1 + gx, second_shape + 10.0 * g2 + gx], dtype=float)

        tail_sum = float(np.sum(10.0 * (x_arr[2:] - 0.5) ** 2))
        scale = 1.7057 + tail_sum
        return np.array(
            [
                scale * math.cos(0.5 * math.pi * x_arr[0]) * math.cos(0.5 * math.pi * x_arr[1]),
                scale * math.cos(0.5 * math.pi * x_arr[0]) * math.sin(0.5 * math.pi * x_arr[1]),
                scale * math.sin(0.5 * math.pi * x_arr[0]),
            ],
            dtype=float,
        )

    def _problem_constraint_values(self, x: np.ndarray) -> np.ndarray:
        objectives = self.objective_values(x)
        if self.family in {"LIRCMOP5", "LIRCMOP6"}:
            if self.family == "LIRCMOP5":
                p = q = np.array([1.6, 2.5], dtype=float)
                a = np.array([2.0, 2.0], dtype=float)
                b = np.array([4.0, 8.0], dtype=float)
            else:
                p = q = np.array([1.8, 2.8], dtype=float)
                a = np.array([2.0, 2.0], dtype=float)
                b = np.array([8.0, 8.0], dtype=float)
            radius = 0.1
            theta = -0.25 * math.pi
            constraints = []
            for idx in range(2):
                shifted_x = objectives[0] - p[idx]
                shifted_y = objectives[1] - q[idx]
                rotated_x = shifted_x * math.cos(theta) - shifted_y * math.sin(theta)
                rotated_y = shifted_x * math.sin(theta) + shifted_y * math.cos(theta)
                constraints.append(radius - rotated_x**2 / a[idx] ** 2 - rotated_y**2 / b[idx] ** 2)
            return np.asarray(constraints, dtype=float)

        radius_sq = float(np.sum(objectives**2))
        constraints = [
            (radius_sq - 9.0) * (4.0 - radius_sq),
            (radius_sq - 3.61) * (3.24 - radius_sq),
        ]
        if self.family == "LIRCMOP14":
            constraints.append((radius_sq - 3.0625) * (2.56 - radius_sq))
        return np.asarray(constraints, dtype=float)

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return self._project_with_slsqp(y, starts=self.initial_points())

    def initial_points(self) -> list[np.ndarray]:
        if self.family in {"LIRCMOP5", "LIRCMOP6"}:
            return self._initial_points_2d()
        return self._initial_points_3d()

    def _initial_points_2d(self) -> list[np.ndarray]:
        seeds: list[np.ndarray] = []
        for x0 in np.linspace(0.0, 1.0, 401):
            point = np.zeros(self.num_variables_value, dtype=float)
            point[0] = x0
            for zero_idx in range(1, self.num_variables_value):
                matlab_idx = zero_idx + 1
                angle = 0.5 * matlab_idx / self.num_variables_value * math.pi * x0
                point[zero_idx] = math.sin(angle) if matlab_idx % 2 == 1 else math.cos(angle)
            if np.all(self.constraint_values(point) <= 1.0e-8):
                seeds.append(point)
        if not seeds:
            raise RuntimeError(f"No feasible seed found for {self.family}")
        return _unique_points(seeds)[: max(self.num_objectives() + 1, 4)]

    def _initial_points_3d(self) -> list[np.ndarray]:
        seeds: list[np.ndarray] = []
        tail_offset = 0.0
        if self.family == "LIRCMOP14":
            tail_offset = math.sqrt((1.75 - 1.7057) / 10.0)
        for x0 in np.linspace(0.1, 0.9, 5):
            for x1 in np.linspace(0.1, 0.9, 5):
                point = np.full(self.num_variables_value, 0.5, dtype=float)
                point[0] = x0
                point[1] = x1
                point[2] = 0.5 + tail_offset
                if np.all(self.constraint_values(point) <= 1.0e-8):
                    seeds.append(point)
        if not seeds:
            raise RuntimeError(f"No feasible seed found for {self.family}")
        return _unique_points(seeds)[: max(self.num_objectives() + 1, 4)]


class CEC2009CFProblem(_BoundedFiniteDifferenceProblem):
    """CEC2009 constrained CF8/CF9/CF10 benchmark problems."""

    def __init__(
        self,
        family: CEC2009CFName,
        *,
        num_variables: int = 10,
        finite_diff_step: float = 1.0e-6,
        projection_maxiter: int = 500,
    ) -> None:
        self.family = family.upper()
        if self.family not in {"CF8", "CF9", "CF10"}:
            raise ValueError(f"Unsupported CEC2009 CF family: {family}")
        if num_variables < 5:
            raise ValueError("CF8-CF10 require at least 5 variables")
        self.num_variables_value = num_variables
        self.finite_diff_step = finite_diff_step
        self.projection_maxiter = projection_maxiter
        tail_bound = 4.0 if self.family == "CF8" else 2.0
        self.lower_bounds = np.concatenate([np.zeros(2, dtype=float), -tail_bound * np.ones(num_variables - 2)])
        self.upper_bounds = np.concatenate([np.ones(2, dtype=float), tail_bound * np.ones(num_variables - 2)])

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return 3

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.clip(np.asarray(x, dtype=float), self.lower_bounds, self.upper_bounds)
        sums = np.zeros(3, dtype=float)
        counts = np.zeros(3, dtype=float)
        for matlab_idx in range(3, self.num_variables_value + 1):
            zero_idx = matlab_idx - 1
            yj = x_arr[zero_idx] - 2.0 * x_arr[1] * math.sin(2.0 * math.pi * x_arr[0] + matlab_idx * math.pi / self.num_variables_value)
            value = 4.0 * yj**2 - math.cos(8.0 * math.pi * yj) + 1.0 if self.family == "CF10" else yj**2
            group = {1: 0, 2: 1, 0: 2}[matlab_idx % 3]
            sums[group] += value
            counts[group] += 1.0
        if np.any(counts == 0.0):
            raise RuntimeError("Each CF objective group must have at least one tail variable")
        return np.array(
            [
                math.cos(0.5 * math.pi * x_arr[0]) * math.cos(0.5 * math.pi * x_arr[1]) + 2.0 * sums[0] / counts[0],
                math.cos(0.5 * math.pi * x_arr[0]) * math.sin(0.5 * math.pi * x_arr[1]) + 2.0 * sums[1] / counts[1],
                math.sin(0.5 * math.pi * x_arr[0]) + 2.0 * sums[2] / counts[2],
            ],
            dtype=float,
        )

    def _safe_denominator(self, value: float) -> float:
        if abs(value) >= 1.0e-12:
            return value
        return 1.0e-12 if value >= 0.0 else -1.0e-12

    def _problem_constraint_values(self, x: np.ndarray) -> np.ndarray:
        objectives = self.objective_values(x)
        denom = self._safe_denominator(1.0 - objectives[2] ** 2)
        ratio_sum = (objectives[0] ** 2 + objectives[1] ** 2) / denom
        ratio_diff = (objectives[0] ** 2 - objectives[1] ** 2) / denom
        n_param = 2.0
        a_param = {"CF8": 4.0, "CF9": 3.0, "CF10": 1.0}[self.family]
        wave = math.sin(n_param * math.pi * (ratio_diff + 1.0))
        if self.family == "CF8":
            report_constraint = ratio_sum - a_param * abs(wave) - 1.0
        else:
            report_constraint = ratio_sum - a_param * wave - 1.0
        return np.array([-report_constraint], dtype=float)

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return self._project_with_slsqp(y, starts=self.initial_points())

    def initial_points(self) -> list[np.ndarray]:
        seeds: list[np.ndarray] = []
        for x0 in np.linspace(0.02, 0.98, 97):
            for x1 in np.linspace(0.02, 0.98, 97):
                point = np.zeros(self.num_variables_value, dtype=float)
                point[0] = x0
                point[1] = x1
                for matlab_idx in range(3, self.num_variables_value + 1):
                    point[matlab_idx - 1] = 2.0 * x1 * math.sin(
                        2.0 * math.pi * x0 + matlab_idx * math.pi / self.num_variables_value
                    )
                if np.all(self.constraint_values(point) <= 1.0e-8):
                    seeds.append(point)
                    if len(seeds) >= max(self.num_objectives() + 1, 6):
                        return _unique_points(seeds)
        if not seeds:
            raise RuntimeError(f"No feasible seed found for {self.family}")
        return _unique_points(seeds)
