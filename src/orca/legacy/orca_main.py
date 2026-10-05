"""Simple public entry points for ORCA-style objective grouping in Python."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from orca.legacy.orca_constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GROUPING_METHOD,
    DEFAULT_NUM_GROUPS,
    SMOKE_TEST_NUM_GROUPS,
)
from orca.legacy.orca_utils import (
    LinearBlocks,
    ORCAResults,
    extract_linear_blocks_pyomo,
    orca_from_matrices,
)

# Smoke-test constants. These are only used when this file is executed directly.
SMOKE_TEST_JINEQ = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ]
)
SMOKE_TEST_JEQ = np.zeros((0, 2))
SMOKE_TEST_JOBJ = np.array(
    [
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
    ]
)


def main_from_matrices(
    Jineq: np.ndarray,
    Jobj: np.ndarray,
    Jeq: Optional[np.ndarray] = None,
    *,
    num_groups: int = DEFAULT_NUM_GROUPS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    grouping_method: str = DEFAULT_GROUPING_METHOD,
) -> ORCAResults:
    """Run ORCA from already extracted matrix blocks."""

    return orca_from_matrices(
        Jineq=Jineq,
        Jobj=Jobj,
        Jeq=Jeq,
        num_groups=num_groups,
        alpha=alpha,
        beta=beta,
        grouping_method=grouping_method,
    )


def main_from_pyomo(
    model,
    objective_exprs: Sequence,
    *,
    num_groups: int = DEFAULT_NUM_GROUPS,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    grouping_method: str = DEFAULT_GROUPING_METHOD,
) -> tuple[ORCAResults, LinearBlocks]:
    """Extract linear blocks from a Pyomo model and run ORCA.

    Parameters
    ----------
    model:
        A Pyomo ConcreteModel.
    objective_exprs:
        A sequence of linear Pyomo expressions, one per objective. For the
        ammonia paper these would be [Z, H, Psi, Xi].
    num_groups:
        Desired number of objective groups.
    alpha, beta:
        Logistic weighting parameters. Defaults match the uploaded Julia code,
        not the exact paper values.
    grouping_method:
        "auto", "leiden", or "average_linkage".

    Returns
    -------
    results, blocks
        results contains the correlation matrix and groups; blocks contains the
        extracted Jeq, Jineq, Jobj and variable names.
    """

    blocks = extract_linear_blocks_pyomo(model, objective_exprs)
    results = orca_from_matrices(
        Jineq=blocks.Jineq,
        Jobj=blocks.Jobj,
        Jeq=blocks.Jeq,
        num_groups=num_groups,
        alpha=alpha,
        beta=beta,
        grouping_method=grouping_method,
    )
    return results, blocks


if __name__ == "__main__":
    # Minimal matrix-based smoke test. Replace these matrices with model-derived
    # Jeq, Jineq, and Jobj for real use.
    results = main_from_matrices(
        SMOKE_TEST_JINEQ,
        SMOKE_TEST_JOBJ,
        SMOKE_TEST_JEQ,
        num_groups=SMOKE_TEST_NUM_GROUPS,
    )
    print("Adjacency matrix:")
    print(results.adj_matrix)
    print("Groups:")
    print(results.groups)
