"""Abstract interface for nonlinear ORCA problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class NonlinearORCAProblem(ABC):
    """Interface required by the nonlinear ORCA workflow.

    Implement this interface for a NumPy, Pyomo, CasADi, JAX, or other model.
    Inequality constraints should be represented as ``g_k(x) <= 0``.
    """

    @abstractmethod
    def num_variables(self) -> int:
        """Return the number of decision variables."""

    @abstractmethod
    def num_objectives(self) -> int:
        """Return the number of objective functions."""

    @abstractmethod
    def objective_values(self, x: np.ndarray) -> np.ndarray:
        """Return objective values ``f_i(x)`` with shape ``(num_objectives,)``."""

    @abstractmethod
    def constraint_values(self, x: np.ndarray) -> np.ndarray:
        """Return inequality values ``g_k(x)`` with shape ``(num_constraints,)``."""

    @abstractmethod
    def objective_gradients(self, x: np.ndarray) -> np.ndarray:
        """Return objective gradients with shape ``(num_objectives, num_variables)``."""

    @abstractmethod
    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return constraint Jacobian with shape ``(num_constraints, num_variables)``."""

    @abstractmethod
    def project_feasible(self, y: np.ndarray) -> np.ndarray:
        """Project a trial point back to the feasible set."""

    @abstractmethod
    def initial_points(self) -> Sequence[np.ndarray]:
        """Return seed points, such as single-objective optima or user-supplied points."""
