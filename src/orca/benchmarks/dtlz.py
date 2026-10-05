"""DTLZ-family benchmark problems for nonlinear ORCA tests.

The DTLZ5(I, M) problem is especially useful for ORCA because its objective
correlation structure is known: objectives 1 through M-I+1 form one correlated
block, and the remaining I-1 objectives are singleton conflicting objectives.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from orca.nonlinear.problem_interface import NonlinearORCAProblem


def expected_dtlz5_groups(num_objectives: int, intrinsic_dimension: int) -> np.ndarray:
    """Return one valid label vector for the known DTLZ5(I, M) grouping."""

    if intrinsic_dimension < 1 or intrinsic_dimension > num_objectives:
        raise ValueError("intrinsic_dimension must be between 1 and num_objectives")
    correlated_count = num_objectives - intrinsic_dimension + 1
    labels = np.ones(num_objectives, dtype=int)
    for obj_idx in range(correlated_count, num_objectives):
        labels[obj_idx] = obj_idx - correlated_count + 2
    return labels


class _DTLZ5LikeProblem(NonlinearORCAProblem):
    """Shared implementation for DTLZ5-like angular benchmarks."""

    def __init__(
        self,
        intrinsic_dimension: int,
        num_objectives: int,
        *,
        k_tail: int = 10,
        finite_diff_step: float = 1.0e-6,
        gradient_backend: Literal["analytic", "finite_difference"] = "analytic",
        initial_point_strategy: Literal["optimize", "deterministic"] = "optimize",
        optimizer_backend: Literal["auto", "ipopt", "scipy"] = "auto",
        optimizer_maxiter: int = 500,
    ) -> None:
        if intrinsic_dimension < 1 or intrinsic_dimension > num_objectives:
            raise ValueError("intrinsic_dimension must be between 1 and num_objectives")
        if k_tail < 1:
            raise ValueError("k_tail must be positive")
        self.intrinsic_dimension = intrinsic_dimension
        self.num_objectives_value = num_objectives
        self.k_tail = k_tail
        self.num_variables_value = num_objectives + k_tail - 1
        self.finite_diff_step = finite_diff_step
        self.gradient_backend = gradient_backend
        self.initial_point_strategy = initial_point_strategy
        self.optimizer_backend = optimizer_backend
        self.optimizer_maxiter = optimizer_maxiter

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return self.num_objectives_value

    def _g_value(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def _g_gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _theta_values(self, x: np.ndarray, g_value: float) -> np.ndarray:
        theta = np.zeros(self.num_objectives_value - 1, dtype=float)
        for idx in range(self.num_objectives_value - 1):
            if idx < self.intrinsic_dimension - 1:
                theta[idx] = 0.5 * math.pi * x[idx]
            else:
                theta[idx] = math.pi * (1.0 + 2.0 * g_value * x[idx]) / (4.0 * (1.0 + g_value))
        return theta

    def _shape_values_and_theta_jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return angular objective shape values and derivatives by theta."""

        m = self.num_objectives_value
        shape = np.zeros(m, dtype=float)
        dshape_dtheta = np.zeros((m, m - 1), dtype=float)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        shape[0] = float(np.prod(cos_theta))
        for theta_idx in range(m - 1):
            if m - 1 == 1:
                product_without = 1.0
            else:
                product_without = float(np.prod(np.delete(cos_theta, theta_idx)))
            dshape_dtheta[0, theta_idx] = -sin_theta[theta_idx] * product_without

        for obj_idx in range(1, m - 1):
            cos_count = m - obj_idx - 1
            sin_idx = cos_count
            cos_product = float(np.prod(cos_theta[:cos_count]))
            shape[obj_idx] = cos_product * sin_theta[sin_idx]
            for theta_idx in range(cos_count):
                if cos_count == 1:
                    product_without = 1.0
                else:
                    product_without = float(np.prod(np.delete(cos_theta[:cos_count], theta_idx)))
                dshape_dtheta[obj_idx, theta_idx] = -sin_theta[theta_idx] * product_without * sin_theta[sin_idx]
            dshape_dtheta[obj_idx, sin_idx] = cos_product * cos_theta[sin_idx]

        shape[-1] = sin_theta[0]
        dshape_dtheta[-1, 0] = cos_theta[0]
        return shape, dshape_dtheta

    def _theta_jacobian(self, x: np.ndarray, g_value: float, g_gradient: np.ndarray) -> np.ndarray:
        theta_jac = np.zeros((self.num_objectives_value - 1, self.num_variables_value), dtype=float)
        for theta_idx in range(self.num_objectives_value - 1):
            if theta_idx < self.intrinsic_dimension - 1:
                theta_jac[theta_idx, theta_idx] = 0.5 * math.pi
            else:
                denom = (1.0 + g_value) ** 2
                theta_jac[theta_idx, :] = 0.25 * math.pi * (2.0 * x[theta_idx] - 1.0) * g_gradient / denom
                theta_jac[theta_idx, theta_idx] += 0.5 * math.pi * g_value / (1.0 + g_value)
        return theta_jac

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.num_variables_value,):
            raise ValueError("x has the wrong number of variables")

        g_value = self._g_value(x_arr)
        theta = self._theta_values(x_arr, g_value)
        shape, _ = self._shape_values_and_theta_jacobian(theta)
        return (1.0 + g_value) * shape

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        return np.concatenate([x_arr - 1.0, -x_arr])

    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        if self.gradient_backend == "finite_difference":
            return self._finite_difference_objective_gradients(x)
        if self.gradient_backend != "analytic":
            raise ValueError(f"Unknown gradient backend: {self.gradient_backend}")

        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.num_variables_value,):
            raise ValueError("x has the wrong number of variables")
        g_value = self._g_value(x_arr)
        g_gradient = self._g_gradient(x_arr)
        theta = self._theta_values(x_arr, g_value)
        shape, dshape_dtheta = self._shape_values_and_theta_jacobian(theta)
        theta_jac = self._theta_jacobian(x_arr, g_value, g_gradient)
        return shape[:, None] * g_gradient[None, :] + (1.0 + g_value) * dshape_dtheta @ theta_jac

    def _finite_difference_objective_gradients(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        h = self.finite_diff_step
        gradients = np.zeros((self.num_objectives_value, self.num_variables_value), dtype=float)
        for var_idx in range(self.num_variables_value):
            step = np.zeros(self.num_variables_value, dtype=float)
            if x_arr[var_idx] - h < 0.0:
                step[var_idx] = h
                gradients[:, var_idx] = (self.objective_values(x_arr + step) - self.objective_values(x_arr)) / h
            elif x_arr[var_idx] + h > 1.0:
                step[var_idx] = h
                gradients[:, var_idx] = (self.objective_values(x_arr) - self.objective_values(x_arr - step)) / h
            else:
                step[var_idx] = h
                gradients[:, var_idx] = (self.objective_values(x_arr + step) - self.objective_values(x_arr - step)) / (2.0 * h)
        return gradients

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        _ = np.asarray(x, dtype=float)
        eye = np.eye(self.num_variables_value, dtype=float)
        return np.vstack([eye, -eye])

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(y, dtype=float), 0.0, 1.0)

    def initial_points(self) -> list[np.ndarray]:
        """Return single-objective optima when an optimizer is available.

        The nonlinear ORCA paper uses the single-objective optimal points as
        initial selected points for DTLZ5. This method therefore tries to solve
        one bound-constrained scalar problem per objective. If Ipopt/cyipopt is
        unavailable, ``optimizer_backend="auto"`` falls back to SciPy SLSQP.
        If no optimizer can be used, deterministic near-extreme seeds are
        returned so lightweight tests can still exercise the workflow.
        """

        if self.initial_point_strategy == "optimize":
            try:
                return self.single_objective_optima()
            except Exception:
                if self.optimizer_backend == "ipopt":
                    raise
        return self.deterministic_initial_points()

    def deterministic_initial_points(self) -> list[np.ndarray]:
        """Return deterministic seed points near objective extremes."""

        base = np.full(self.num_variables_value, 0.5, dtype=float)
        seeds: list[np.ndarray] = []
        for obj_idx in range(self.num_objectives_value):
            seed = base.copy()
            if obj_idx == 0:
                seed[0] = 1.0
            elif obj_idx == self.num_objectives_value - 1:
                seed[0] = 0.0
            else:
                angle_idx = self.num_objectives_value - obj_idx - 1
                seed[min(angle_idx, self.num_variables_value - 1)] = 0.0
            seeds.append(seed)
        return seeds

    def single_objective_optima(self) -> list[np.ndarray]:
        """Optimize each objective individually over the DTLZ box domain."""

        starts = self.deterministic_initial_points()
        optima = []
        for obj_idx, start in enumerate(starts):
            optima.append(self._minimize_single_objective(obj_idx, start))
        return optima

    def _minimize_single_objective(self, objective_index: int, start: np.ndarray) -> np.ndarray:
        if self.optimizer_backend in {"auto", "ipopt"}:
            try:
                return self._minimize_with_ipopt(objective_index, start)
            except ImportError:
                if self.optimizer_backend == "ipopt":
                    raise
            except Exception:
                if self.optimizer_backend == "ipopt":
                    raise

        if self.optimizer_backend in {"auto", "scipy"}:
            return self._minimize_with_scipy(objective_index, start)
        raise ValueError(f"Unknown optimizer backend: {self.optimizer_backend}")

    def _minimize_with_ipopt(self, objective_index: int, start: np.ndarray) -> np.ndarray:
        try:
            from cyipopt import minimize_ipopt  # type: ignore
        except ImportError as exc:
            raise ImportError("cyipopt is required for optimizer_backend='ipopt'") from exc

        bounds = [(0.0, 1.0)] * self.num_variables_value

        def fun(x: np.ndarray) -> float:
            return float(self.objective_values(np.asarray(x, dtype=float))[objective_index])

        def jac(x: np.ndarray) -> np.ndarray:
            return self.objective_gradients(np.asarray(x, dtype=float))[objective_index, :]

        result = minimize_ipopt(
            fun,
            x0=np.asarray(start, dtype=float),
            jac=jac,
            bounds=bounds,
            options={"max_iter": self.optimizer_maxiter, "print_level": 0},
        )
        if not result.success:
            raise RuntimeError(f"Ipopt failed for objective {objective_index + 1}: {result.message}")
        return np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)

    def _minimize_with_scipy(self, objective_index: int, start: np.ndarray) -> np.ndarray:
        try:
            from scipy.optimize import minimize  # type: ignore
        except ImportError as exc:
            raise ImportError("scipy is required for optimizer_backend='scipy'") from exc

        bounds = [(0.0, 1.0)] * self.num_variables_value

        def fun(x: np.ndarray) -> float:
            return float(self.objective_values(np.asarray(x, dtype=float))[objective_index])

        def jac(x: np.ndarray) -> np.ndarray:
            return self.objective_gradients(np.asarray(x, dtype=float))[objective_index, :]

        result = minimize(
            fun,
            x0=np.asarray(start, dtype=float),
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.optimizer_maxiter, "ftol": 1.0e-12, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"SciPy SLSQP failed for objective {objective_index + 1}: {result.message}")
        return np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)


class DTLZ5Problem(_DTLZ5LikeProblem):
    """Box-constrained DTLZ5(I, M) benchmark.

    Parameters
    ----------
    intrinsic_dimension:
        The DTLZ5 ``I`` parameter. The expected ORCA grouping is
        ``{f1, ..., f_{M-I+1}}, {f_{M-I+2}}, ..., {f_M}``.
    num_objectives:
        The DTLZ5 ``M`` parameter.
    k_tail:
        Number of distance variables. The common DTLZ5 choice is 10.
    finite_diff_step:
        Central-difference step used for objective gradients.
    """

    def _g_value(self, x: np.ndarray) -> float:
        return float(np.sum((x[self.num_objectives_value - 1 :] - 0.5) ** 2))

    def _g_gradient(self, x: np.ndarray) -> np.ndarray:
        gradient = np.zeros(self.num_variables_value, dtype=float)
        tail = x[self.num_objectives_value - 1 :]
        gradient[self.num_objectives_value - 1 :] = 2.0 * (tail - 0.5)
        return gradient


class DTLZ6Problem(_DTLZ5LikeProblem):
    """Box-constrained DTLZ6(I, M) benchmark.

    DTLZ6 uses the same angular objective map as DTLZ5 but replaces the
    distance function with ``g(x) = sum(x_i**0.1)`` over the tail variables.
    The expected objective-group structure is the same as DTLZ5(I, M).
    """

    def _g_value(self, x: np.ndarray) -> float:
        tail = np.clip(x[self.num_objectives_value - 1 :], 0.0, None)
        return float(np.sum(tail**0.1))

    def _g_gradient(self, x: np.ndarray) -> np.ndarray:
        gradient = np.zeros(self.num_variables_value, dtype=float)
        tail = np.clip(x[self.num_objectives_value - 1 :], 0.0, None)
        # DTLZ6 has a singular derivative at x_i = 0. Use a tiny positive
        # floor so analytic gradients remain finite at feasible boundaries.
        safe_tail = np.maximum(tail, 1.0e-12)
        gradient[self.num_objectives_value - 1 :] = 0.1 * safe_tail ** (-0.9)
        return gradient
