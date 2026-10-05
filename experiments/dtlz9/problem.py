"""Analytical DTLZ9 problem adapter for nonlinear ORCA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import orca_bridge as _orca_bridge  # noqa: F401
from orca.nonlinear.problem_interface import NonlinearORCAProblem


@dataclass(frozen=True)
class BlockMetadata:
    objective_index: int
    start: int
    stop: int
    size: int


class DTLZ9Problem(NonlinearORCAProblem):
    """Standard inequality-constrained DTLZ9 with analytical derivatives.

    Only the ``M-1`` structural DTLZ9 inequalities are exposed to ORCA.
    Box bounds are enforced by ``project_feasible`` and deliberately excluded
    from the active-constraint list because regular exact-PF decision vectors
    can be numerically very close to zero when ``p=0.1`` and ``n=10M``.
    """

    def __init__(
        self,
        num_objectives: int,
        num_variables: int | None = None,
        *,
        p: float = 0.1,
        theta_min: float = 0.15,
    ) -> None:
        if num_objectives < 2:
            raise ValueError("num_objectives must be at least 2")
        n = num_variables if num_variables is not None else 10 * num_objectives
        if n < num_objectives or n % num_objectives != 0:
            raise ValueError("num_variables must be a positive multiple of num_objectives")
        if not (0.0 < p <= 1.0):
            raise ValueError("p must lie in (0, 1]")
        if not (0.0 < theta_min < 0.25 * np.pi):
            raise ValueError("theta_min must lie in (0, pi/4)")

        self.num_objectives_value = int(num_objectives)
        self.num_variables_value = int(n)
        self.p = float(p)
        self.theta_min = float(theta_min)
        self.block_size = self.num_variables_value // self.num_objectives_value
        self.block_slices = tuple(
            slice(j * self.block_size, (j + 1) * self.block_size)
            for j in range(self.num_objectives_value)
        )

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return self.num_objectives_value

    def block_metadata(self) -> tuple[BlockMetadata, ...]:
        return tuple(
            BlockMetadata(j, block.start, block.stop, self.block_size)
            for j, block in enumerate(self.block_slices)
        )

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.num_variables_value,):
            raise ValueError("x has the wrong number of variables")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError("DTLZ9 decision variables must lie in [0, 1]")
        return arr

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        arr = self._validate_x(x)
        return np.asarray(
            [np.sum(arr[block] ** self.p) for block in self.block_slices],
            dtype=float,
        )

    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        arr = self._validate_x(x)
        if np.any(arr <= 0.0):
            raise ValueError("DTLZ9 derivatives are singular at zero; use regular interior points")
        jac = np.zeros((self.num_objectives_value, self.num_variables_value), dtype=float)
        for j, block in enumerate(self.block_slices):
            jac[j, block] = self.p * arr[block] ** (self.p - 1.0)
        return jac

    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        values = self.objective_values(x)
        return 1.0 - values[:-1] ** 2 - values[-1] ** 2

    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        values = self.objective_values(x)
        obj_jac = self.objective_gradients(x)
        jac = np.zeros((self.num_objectives_value - 1, self.num_variables_value), dtype=float)
        for j in range(self.num_objectives_value - 1):
            jac[j] = -2.0 * values[j] * obj_jac[j] - 2.0 * values[-1] * obj_jac[-1]
        return jac

    def exact_pf_objectives(self, theta: float | np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        if np.any(theta_arr <= 0.0) or np.any(theta_arr >= 0.5 * np.pi):
            raise ValueError("exact PF derivatives require theta strictly inside (0, pi/2)")
        flat = theta_arr.reshape(-1)
        values = np.empty((flat.size, self.num_objectives_value), dtype=float)
        values[:, :-1] = np.cos(flat)[:, None]
        values[:, -1] = np.sin(flat)
        return values[0] if theta_arr.ndim == 0 else values.reshape(theta_arr.shape + (self.num_objectives_value,))

    def decision_from_objectives(self, objectives: Sequence[float]) -> np.ndarray:
        values = np.asarray(objectives, dtype=float)
        if values.shape != (self.num_objectives_value,):
            raise ValueError("objectives have the wrong shape")
        if np.any(values <= 0.0) or np.any(values > self.block_size):
            raise ValueError("objective targets must lie in (0, block_size]")
        x = np.empty(self.num_variables_value, dtype=float)
        for value, block in zip(values, self.block_slices):
            x[block] = (value / self.block_size) ** (1.0 / self.p)
        return x

    def exact_pf_decision(self, theta: float) -> np.ndarray:
        return self.decision_from_objectives(self.exact_pf_objectives(theta))

    def exact_pf_decisions(self, theta: Iterable[float]) -> np.ndarray:
        return np.vstack([self.exact_pf_decision(float(value)) for value in theta])

    def controlled_off_front_decision(
        self,
        theta: float,
        deltas: Sequence[float] | float,
    ) -> np.ndarray:
        values = np.asarray(self.exact_pf_objectives(theta), dtype=float)
        delta_arr = np.broadcast_to(np.asarray(deltas, dtype=float), (self.num_objectives_value - 1,))
        if np.any(delta_arr < 0.0):
            raise ValueError("off-front deltas must be nonnegative")
        upper = np.nextafter(float(self.block_size), 0.0)
        values[:-1] = np.minimum(values[:-1] + delta_arr, upper)
        return self.decision_from_objectives(values)

    def adjoining_surface_decision(
        self,
        theta: float,
        active_indices: Sequence[int],
        *,
        inactive_delta: float = 0.05,
    ) -> np.ndarray:
        active = set(int(idx) for idx in active_indices)
        if not active.issubset(set(range(self.num_objectives_value - 1))):
            raise ValueError("active_indices must refer to the first M-1 constraints")
        deltas = np.full(self.num_objectives_value - 1, inactive_delta, dtype=float)
        for idx in active:
            deltas[idx] = 0.0
        return self.controlled_off_front_decision(theta, deltas)

    def initial_points(self) -> list[np.ndarray]:
        count = max(5, self.num_objectives_value)
        theta = np.linspace(self.theta_min, 0.5 * np.pi - self.theta_min, count)
        return [self.exact_pf_decision(float(value)) for value in theta]

    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        """Return a deterministic feasible repair for selected-point experiments.

        This is not claimed to be the exact Euclidean projection used in the
        ORCA paper. It preserves the final block and raises deficient first-group
        objective blocks to the nearest structural lower bound using equal block
        shares. Exact-PF and controlled-sample experiments bypass this repair.
        """

        x = np.clip(np.asarray(y, dtype=float), 1.0e-30, 1.0)
        if x.shape != (self.num_variables_value,):
            raise ValueError("y has the wrong number of variables")
        values = self.objective_values(x)
        lower = float(np.sqrt(max(0.0, 1.0 - values[-1] ** 2)))
        for j, block in enumerate(self.block_slices[:-1]):
            if values[j] < lower:
                repaired = (lower / self.block_size) ** (1.0 / self.p)
                x[block] = repaired
        return np.clip(x, 1.0e-30, 1.0)
