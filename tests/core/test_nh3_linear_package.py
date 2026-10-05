"""NH3 regression test for the package-level linear ORCA workflow."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from orca.config import LinearORCAConfig
from orca.data_schema import LinearMatrixBlocks, ORCAInteractionData, ORCAResult
from orca.linear import linear_orca_from_pyomo
from tests.core.test_constants import (
    DA_END_INITIAL,
    DA_START,
    ELECTROLYTIC_H2_WATER_FACTOR,
    ELECTROLYZER_COST,
    ELECTROLYZER_OPERATION_RISK,
    EMISSION_COLUMN,
    EXPECTED_NUM_OBJECTIVES,
    EXPECTED_NUM_VARIABLES_MINIMUM,
    FIRST_TIME,
    FOSSIL_H2_EMISSIONS_FACTOR,
    H2_IDX,
    HORIZON_HOURS,
    H2_NH3_STOICHIOMETRIC_RATIO,
    INITIAL_DA,
    INITIAL_ELECTROLYZERS_ON,
    INITIAL_TIME,
    MATRIX_ABS_TOL,
    MAX_ELECTROLYZERS,
    MAX_FLOW,
    MAX_PURCHASED_H2,
    MAX_RAMP_EVENTS_IN_WINDOW,
    MAX_STORAGE,
    MIN_FLOW,
    MIN_STORAGE,
    N2_IDX,
    N2_NH3_STOICHIOMETRIC_RATIO,
    NH3_IDX,
    NUM_COMPONENTS,
    POWER_PER_MASS,
    PRICE_COLUMN,
    PSA_COST,
    PURCHASED_H2_COST,
    PURCHASED_H2_WATER_FACTOR,
    RAMP_DOWN_LIMIT,
    RAMP_UP_LIMIT,
    RAMP_WINDOW,
    REQUESTED_GROUPS,
    TARGET_NH3_RATE,
    TEST_DATA_FILE,
    TEST_DATA_SHEET,
)


def _format_vector(values: np.ndarray) -> list[float]:
    """Return a Python-list representation similar to Julia vector output."""

    return values.astype(float).tolist()


def _format_matrix(values: np.ndarray) -> str:
    """Return the full matrix so pytest output mirrors the verbose Julia log."""

    return np.array2string(values, threshold=sys.maxsize, max_line_width=sys.maxsize)


def _partition_vector_to_communities(labels: np.ndarray) -> list[list[int]]:
    """Convert labels like [1, 2, 2, 2] to Julia-like [[1], [2, 3, 4]]."""

    communities: list[list[int]] = []
    for label in sorted(np.unique(labels).tolist()):
        communities.append([idx + FIRST_TIME for idx, value in enumerate(labels.tolist()) if value == label])
    return communities


def _load_price_and_emissions() -> tuple[np.ndarray, np.ndarray]:
    """Load the same Excel inputs used by the Julia NH3 toy test."""

    assert TEST_DATA_FILE.is_file()
    data = pd.read_excel(TEST_DATA_FILE, sheet_name=TEST_DATA_SHEET)
    price = data[PRICE_COLUMN].to_numpy(dtype=float)[:HORIZON_HOURS]
    emissions = data[EMISSION_COLUMN].to_numpy(dtype=float)[:HORIZON_HOURS]
    assert len(price) == HORIZON_HOURS
    assert len(emissions) == HORIZON_HOURS
    return price, emissions


def _build_nh3_model(price: np.ndarray, emissions: np.ndarray) -> tuple[pyo.ConcreteModel, list]:
    """Build the Pyomo analog of the NH3 scheduling test model."""

    model = pyo.ConcreteModel()

    model.T = pyo.RangeSet(FIRST_TIME, HORIZON_HOURS)
    model.T0 = pyo.RangeSet(INITIAL_TIME, HORIZON_HOURS)
    model.COMP = pyo.RangeSet(FIRST_TIME, NUM_COMPONENTS)
    model.DA = pyo.RangeSet(DA_START, HORIZON_HOURS)

    wind_power = {time_idx: 0.0 for time_idx in model.T}
    initial_storage = {
        H2_IDX: MIN_STORAGE[H2_IDX],
        N2_IDX: MIN_STORAGE[N2_IDX],
        NH3_IDX: MIN_STORAGE[NH3_IDX],
    }

    model.da = pyo.Var(model.DA, domain=pyo.Binary)
    model.de = pyo.Var(model.T, domain=pyo.NonNegativeIntegers)
    model.ms = pyo.Var(model.COMP, model.T0, domain=pyo.NonNegativeReals)
    model.mp = pyo.Var(model.COMP, model.T0, domain=pyo.NonNegativeReals)
    model.mnh3dev = pyo.Var(model.T, domain=pyo.Reals)
    model.ne = pyo.Var(model.T0, domain=pyo.NonNegativeIntegers)
    model.peb = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.pei = pyo.Var(model.COMP, model.T, domain=pyo.NonNegativeReals)
    model.pes = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.COMP, domain=pyo.NonNegativeReals)
    model.bh2 = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    model.cons = pyo.ConstraintList()

    for time_idx in model.T:
        model.cons.add(-model.de[time_idx] <= 0.0)
        model.cons.add(-model.peb[time_idx] <= 0.0)
        model.cons.add(-model.pes[time_idx] <= 0.0)
        model.cons.add(-model.bh2[time_idx] <= 0.0)
        for comp_idx in model.COMP:
            model.cons.add(-model.ms[comp_idx, time_idx] <= 0.0)
            model.cons.add(-model.mp[comp_idx, time_idx] <= 0.0)
            model.cons.add(-model.pei[comp_idx, time_idx] <= 0.0)
    for comp_idx in model.COMP:
        model.cons.add(-model.u[comp_idx] <= 0.0)

    for time_idx in model.T:
        model.cons.add(
            wind_power[time_idx]
            + model.peb[time_idx]
            - model.pes[time_idx]
            - model.pei[H2_IDX, time_idx]
            - model.pei[N2_IDX, time_idx]
            - model.pei[NH3_IDX, time_idx]
            == 0.0
        )
        for comp_idx in model.COMP:
            model.cons.add(model.pei[comp_idx, time_idx] == POWER_PER_MASS[comp_idx] * model.mp[comp_idx, time_idx])
        model.cons.add(model.mnh3dev[time_idx] == TARGET_NH3_RATE - model.mp[NH3_IDX, time_idx])

    for time_idx in model.T:
        model.cons.add(
            model.ms[H2_IDX, time_idx]
            == model.ms[H2_IDX, time_idx - 1]
            + model.bh2[time_idx]
            + model.mp[H2_IDX, time_idx]
            - H2_NH3_STOICHIOMETRIC_RATIO * model.mp[NH3_IDX, time_idx]
        )
        model.cons.add(
            model.ms[N2_IDX, time_idx]
            == model.ms[N2_IDX, time_idx - 1]
            + model.mp[N2_IDX, time_idx]
            - N2_NH3_STOICHIOMETRIC_RATIO * model.mp[NH3_IDX, time_idx]
        )
    for comp_idx in model.COMP:
        model.cons.add(model.u[comp_idx] == model.ms[comp_idx, INITIAL_TIME] - model.ms[comp_idx, HORIZON_HOURS])

    for time_idx in model.T:
        for comp_idx in (N2_IDX, NH3_IDX):
            model.cons.add(MIN_FLOW[comp_idx] - model.mp[comp_idx, time_idx] <= 0.0)
            model.cons.add(-MAX_FLOW[comp_idx] + model.mp[comp_idx, time_idx] <= 0.0)
        model.cons.add(-model.ne[time_idx] <= 0.0)
        model.cons.add(model.ne[time_idx] - MAX_ELECTROLYZERS <= 0.0)
        model.cons.add(MIN_FLOW[H2_IDX] * model.ne[time_idx] - model.mp[H2_IDX, time_idx] <= 0.0)
        model.cons.add(model.mp[H2_IDX, time_idx] - MAX_FLOW[H2_IDX] * model.ne[time_idx] <= 0.0)
        for comp_idx in model.COMP:
            model.cons.add(MIN_STORAGE[comp_idx] - model.ms[comp_idx, time_idx] <= 0.0)
            model.cons.add(-MAX_STORAGE[comp_idx] + model.ms[comp_idx, time_idx] <= 0.0)
        model.cons.add(model.bh2[time_idx] - MAX_PURCHASED_H2 <= 0.0)

    for time_idx in model.T:
        model.cons.add(-model.de[time_idx] + model.ne[time_idx] - model.ne[time_idx - 1] <= 0.0)
        model.cons.add(
            -model.da[time_idx] * RAMP_DOWN_LIMIT
            - model.mp[NH3_IDX, time_idx]
            + model.mp[NH3_IDX, time_idx - 1]
            <= 0.0
        )
        model.cons.add(
            model.mp[NH3_IDX, time_idx] - model.mp[NH3_IDX, time_idx - 1] - model.da[time_idx] * RAMP_UP_LIMIT
            <= 0.0
        )
        model.cons.add(
            sum(model.da[idx] for idx in range(time_idx + FIRST_TIME - RAMP_WINDOW, time_idx + FIRST_TIME))
            - MAX_RAMP_EVENTS_IN_WINDOW
            <= 0.0
        )

    for comp_idx in model.COMP:
        model.cons.add(model.ms[comp_idx, INITIAL_TIME] == initial_storage[comp_idx])
    model.cons.add(model.ne[INITIAL_TIME] == INITIAL_ELECTROLYZERS_ON)
    model.cons.add(model.mp[NH3_IDX, INITIAL_TIME] == TARGET_NH3_RATE)
    for da_idx in range(DA_START, DA_END_INITIAL + FIRST_TIME):
        model.cons.add(model.da[da_idx] == INITIAL_DA[da_idx])

    cost_expr = sum(
        sum(price[time_idx - FIRST_TIME] * POWER_PER_MASS[comp_idx] * model.mp[comp_idx, time_idx] for comp_idx in model.COMP)
        + ELECTROLYZER_COST * model.mp[H2_IDX, time_idx]
        + PURCHASED_H2_COST * model.bh2[time_idx]
        + PSA_COST * model.mp[N2_IDX, time_idx]
        for time_idx in model.T
    )
    emissions_expr = sum(
        sum(POWER_PER_MASS[comp_idx] * model.mp[comp_idx, time_idx] * emissions[time_idx - FIRST_TIME] for comp_idx in model.COMP)
        for time_idx in model.T
    ) + sum(FOSSIL_H2_EMISSIONS_FACTOR * model.bh2[time_idx] for time_idx in model.T)
    water_expr = sum(ELECTROLYTIC_H2_WATER_FACTOR * model.mp[H2_IDX, time_idx] for time_idx in model.T) + sum(
        PURCHASED_H2_WATER_FACTOR * model.bh2[time_idx] for time_idx in model.T
    )
    safety_expr = sum(
        ELECTROLYZER_OPERATION_RISK * model.mp[H2_IDX, time_idx] / MAX_FLOW[H2_IDX] for time_idx in model.T
    )

    return model, [cost_expr, emissions_expr, water_expr, safety_expr]


def test_nh3_toy_model_runs_package_linear_orca() -> None:
    """Run the NH3 Pyomo model through the new linear package API."""

    print("Testing Running tests...")
    price, emissions = _load_price_and_emissions()
    print(f"ceb: {_format_vector(price)}")
    print(f"gridemm: {_format_vector(emissions)}")
    model, objective_exprs = _build_nh3_model(price, emissions)
    config = LinearORCAConfig(
        num_groups=REQUESTED_GROUPS,
        grouping_method="average_linkage",
    )

    result, blocks = linear_orca_from_pyomo(model, objective_exprs, config=config)

    communities = _partition_vector_to_communities(result.groups)
    print(f"Jeq size   = {blocks.Jeq.shape}")
    print(f"Jobj = {_format_matrix(blocks.Jobj)}")
    print(f"Jineq size = {blocks.Jineq.shape}")
    print(f"Jobj size  = {blocks.Jobj.shape}")
    print(_format_matrix(result.adj_matrix))
    print(f"NLPCorrStrengGenerating results:{communities}")
    print(f"ORCA.main returned: {communities}")

    assert isinstance(result, ORCAResult)
    assert isinstance(blocks, LinearMatrixBlocks)
    assert isinstance(result.interaction_data, ORCAInteractionData)
    assert result.input_data is blocks
    assert result.metadata["algorithm"] == "linear_orca"
    assert result.metadata["num_groups"] == REQUESTED_GROUPS
    assert result.metadata["grouping_method"] == "average_linkage"

    assert blocks.Jobj.shape[0] == EXPECTED_NUM_OBJECTIVES
    assert blocks.Jobj.shape[1] >= EXPECTED_NUM_VARIABLES_MINIMUM
    assert blocks.Jineq.shape[1] == blocks.Jobj.shape[1]
    assert blocks.Jeq is not None
    assert blocks.Jeq.shape[1] == blocks.Jobj.shape[1]
    assert blocks.var_names is not None
    assert len(blocks.var_names) == blocks.Jobj.shape[1]

    assert result.adj_matrix.shape == (EXPECTED_NUM_OBJECTIVES, EXPECTED_NUM_OBJECTIVES)
    assert result.groups.shape == (EXPECTED_NUM_OBJECTIVES,)
    assert np.allclose(np.diag(result.adj_matrix), np.ones(EXPECTED_NUM_OBJECTIVES), atol=MATRIX_ABS_TOL)
    assert np.allclose(result.adj_matrix, result.adj_matrix.T, atol=MATRIX_ABS_TOL)
    assert np.all(np.isfinite(result.adj_matrix))

    interaction_data = result.interaction_data
    assert interaction_data is not None
    assert interaction_data.strengths.shape[:2] == (EXPECTED_NUM_OBJECTIVES, EXPECTED_NUM_OBJECTIVES)
    assert interaction_data.weights.shape == interaction_data.strengths.shape
    assert interaction_data.total_weights.shape == (EXPECTED_NUM_OBJECTIVES, EXPECTED_NUM_OBJECTIVES)
    assert interaction_data.valid_mask is not None
    assert interaction_data.valid_mask.shape == interaction_data.strengths.shape
    assert np.all(np.isfinite(interaction_data.total_weights))
    assert interaction_data.metadata["num_inequality_constraints"] == blocks.Jineq.shape[0]
    assert interaction_data.metadata["num_equality_constraints"] == blocks.Jeq.shape[0]
