"""Extraction of linear matrix blocks from supported modeling packages."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from orca.data_schema import LinearMatrixBlocks


def _empty_matrix(num_columns: int) -> np.ndarray:
    return np.zeros((0, num_columns), dtype=float)


def extract_linear_blocks_pyomo(model, objective_exprs: Sequence) -> LinearMatrixBlocks:
    """Extract linear ORCA matrix blocks from a Pyomo model.

    Inequalities are standardized as ``<=`` rows:

    * ``body <= upper`` contributes ``grad(body)``.
    * ``lower <= body`` contributes ``-grad(body)``.
    * ``lower <= body <= upper`` contributes both rows.

    Equalities are stored separately in ``Jeq``. The expressions must be linear.
    """

    try:
        import pyomo.environ as pyo  # type: ignore
        from pyomo.repn.standard_repn import generate_standard_repn  # type: ignore
    except ImportError as exc:
        raise ImportError("extract_linear_blocks_pyomo requires Pyomo") from exc

    var_list = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    var_to_idx = {id(var): idx for idx, var in enumerate(var_list)}
    var_names = [var.name for var in var_list]
    num_variables = len(var_list)

    def row_from_expr(expr) -> np.ndarray:
        repn = generate_standard_repn(expr, compute_values=False)
        if repn.nonlinear_expr is not None or repn.quadratic_vars:
            raise ValueError(f"Expression is not linear: {expr}")
        row = np.zeros(num_variables, dtype=float)
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            idx = var_to_idx.get(id(var))
            if idx is not None:
                row[idx] += float(coef)
        return row

    eq_rows: list[np.ndarray] = []
    ineq_rows: list[np.ndarray] = []

    for constraint in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        if constraint.body is None:
            continue
        body_row = row_from_expr(constraint.body)
        has_lower_bound = constraint.lower is not None
        has_upper_bound = constraint.upper is not None

        if constraint.equality:
            eq_rows.append(body_row)
        else:
            if has_upper_bound:
                ineq_rows.append(body_row)
            if has_lower_bound:
                ineq_rows.append(-body_row)

    obj_rows = [row_from_expr(expr) for expr in objective_exprs]

    Jeq = np.vstack(eq_rows) if eq_rows else _empty_matrix(num_variables)
    Jineq = np.vstack(ineq_rows) if ineq_rows else _empty_matrix(num_variables)
    Jobj = np.vstack(obj_rows) if obj_rows else _empty_matrix(num_variables)
    return LinearMatrixBlocks(Jineq=Jineq, Jobj=Jobj, Jeq=Jeq, var_names=var_names)
