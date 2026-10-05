"""LSMOP1/5/9 large-scale multiobjective benchmark adapters.

The formulas follow the PlatEMO implementations of LSMOP1, LSMOP5, and
LSMOP9 from Cheng, Jin, and Olhofer's large-scale multiobjective benchmark
suite. These problems are useful runtime stress tests because the default
decision dimension scales as ``D = 100 * M``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from orca.nonlinear.problem_interface import NonlinearORCAProblem


LSMOPFamily = Literal["LSMOP1", "LSMOP5", "LSMOP9"]


class LSMOPProblem(NonlinearORCAProblem):
    """Adapter for LSMOP1, LSMOP5, and LSMOP9.

    Parameters
    ----------
    family:
        One of ``"LSMOP1"``, ``"LSMOP5"``, or ``"LSMOP9"``.
    num_objectives:
        Number of objectives ``M``.
    num_variables:
        Number of variables ``D``. Defaults to ``100 * M``.
    nk:
        Number of subcomponents in each variable group. PlatEMO defaults to 5.
    finite_diff_step:
        Bound-aware finite-difference step used for objective gradients.
    """

    def __init__(
        self,
        family: LSMOPFamily,
        *,
        num_objectives: int = 10,
        num_variables: int | None = None,
        nk: int = 5,
        finite_diff_step: float = 1.0e-6,
    ) -> None:
        self.family = family.upper()
        if self.family not in {"LSMOP1", "LSMOP5", "LSMOP9"}:
            raise ValueError(f"Unsupported LSMOP family: {family}")
        if num_objectives < 2:
            raise ValueError("num_objectives must be at least 2")
        self.num_objectives_value = num_objectives
        self.num_variables_value = num_variables if num_variables is not None else 100 * num_objectives
        if self.num_variables_value <= num_objectives:
            raise ValueError("num_variables must be larger than num_objectives")
        self.nk = nk
        self.finite_diff_step = finite_diff_step
        self.lower_bounds = np.zeros(self.num_variables_value, dtype=float)
        self.upper_bounds = np.concatenate(
            [
                np.ones(num_objectives - 1, dtype=float),
                10.0 * np.ones(self.num_variables_value - num_objectives + 1, dtype=float),
            ]
        )
        self.sublen, self.group_offsets = self._component_lengths()

    def _component_lengths(self) -> tuple[np.ndarray, np.ndarray]:
        c_values = [3.8 * 0.1 * (1.0 - 0.1)]
        for _ in range(1, self.num_objectives_value):
            c_values.append(3.8 * c_values[-1] * (1.0 - c_values[-1]))
        c_arr = np.asarray(c_values, dtype=float)
        sublen = np.floor(c_arr / np.sum(c_arr) * (self.num_variables_value - self.num_objectives_value + 1) / self.nk)
        sublen = np.maximum(sublen.astype(int), 1)
        return sublen, np.concatenate([[0], np.cumsum(sublen * self.nk)])

    def num_variables(self) -> int:
        return self.num_variables_value

    def num_objectives(self) -> int:
        return self.num_objectives_value

    def _transformed_tail(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).copy()
        m = self.num_objectives_value
        d = self.num_variables_value
        idx = np.arange(m - 1, d, dtype=float)
        if self.family == "LSMOP1":
            scale = 1.0 + (idx + 1.0) / d
        else:
            scale = 1.0 + np.cos((idx + 1.0) / d * np.pi / 2.0)
        x_arr[m - 1 : d] = scale * x_arr[m - 1 : d] - x_arr[0] * 10.0
        return x_arr

    def _component_slice(self, objective_index: int, component_index: int) -> slice:
        m = self.num_objectives_value
        start = int(self.group_offsets[objective_index] + m - 1 + component_index * self.sublen[objective_index])
        stop = int(start + self.sublen[objective_index])
        return slice(start, min(stop, self.num_variables_value))

    @staticmethod
    def _sphere(values: np.ndarray) -> float:
        return float(np.sum(values**2))

    @staticmethod
    def _ackley(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(
            20.0
            - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(values**2) / values.size))
            - np.exp(np.sum(np.cos(2.0 * np.pi * values)) / values.size)
            + np.e
        )

    def _g_values(self, transformed: np.ndarray) -> np.ndarray:
        g = np.zeros(self.num_objectives_value, dtype=float)
        for obj_idx in range(self.num_objectives_value):
            for component_idx in range(self.nk):
                values = transformed[self._component_slice(obj_idx, component_idx)]
                if self.family == "LSMOP9" and (obj_idx + 1) % 2 == 0:
                    g[obj_idx] += self._ackley(values)
                else:
                    g[obj_idx] += self._sphere(values)
        g = g / self.sublen / self.nk
        return g

    def objective_values(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.clip(np.asarray(x, dtype=float), self.lower_bounds, self.upper_bounds)
        transformed = self._transformed_tail(x_arr)
        m = self.num_objectives_value
        g = self._g_values(transformed)

        if self.family == "LSMOP1":
            cumulative = np.cumprod(np.concatenate([[1.0], x_arr[: m - 1]]))
            shape = np.flip(cumulative)
            shape[1:] *= 1.0 - x_arr[m - 2 :: -1]
            return (1.0 + g) * shape

        if self.family == "LSMOP5":
            cos_angles = np.cos(x_arr[: m - 1] * np.pi / 2.0)
            sin_angles = np.sin(x_arr[m - 2 :: -1] * np.pi / 2.0)
            cumulative = np.cumprod(np.concatenate([[1.0], cos_angles]))
            shape = np.flip(cumulative)
            shape[1:] *= sin_angles
            shifted_g = g + np.concatenate([g[1:], [0.0]])
            return (1.0 + g + shifted_g) * shape

        g_total = 1.0 + float(np.sum(g))
        values = np.zeros(m, dtype=float)
        values[: m - 1] = x_arr[: m - 1]
        values[m - 1] = (1.0 + g_total) * (
            m - np.sum(values[: m - 1] / (1.0 + g_total) * (1.0 + np.sin(3.0 * np.pi * values[: m - 1])))
        )
        return values

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
        center = 0.5 * (self.lower_bounds + self.upper_bounds)
        seeds = [center]
        for idx in range(min(self.num_objectives_value, 5)):
            seed = center.copy()
            if idx < self.num_objectives_value - 1:
                seed[idx] = self.lower_bounds[idx]
            seeds.append(seed)
        return seeds
