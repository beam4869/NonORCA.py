"""Constants for the NH3 Pyomo regression test.

These constants mirror the values used in the uploaded Julia test file while
keeping the test implementation free of hidden numeric literals.
"""

from __future__ import annotations

from pathlib import Path

# Test data.
TEST_DATA_FILE = Path(__file__).resolve().parent / "Jan_2022.xlsx"
TEST_DATA_SHEET = "test_for_pareto_frontier"
PRICE_COLUMN = "Jan 21st 11pm price"
EMISSION_COLUMN = "Jan 21st 11pm emission"

# Time and component indexing.
HORIZON_HOURS = 48
NUM_COMPONENTS = 3
FIRST_TIME = 1
INITIAL_TIME = 0
H2_IDX = 1
N2_IDX = 2
NH3_IDX = 3
DA_START = -2
DA_END_INITIAL = 0

# Process parameters.
POWER_PER_MASS = {H2_IDX: 60.0, N2_IDX: 0.8, NH3_IDX: 2.12}
MIN_FLOW = {H2_IDX: 9.99, N2_IDX: 0.0, NH3_IDX: 750.0}
MAX_FLOW = {H2_IDX: 33.3, N2_IDX: 2250.0, NH3_IDX: 1100.0}
MAX_ELECTROLYZERS = 15
MIN_STORAGE = {H2_IDX: 4035.0, N2_IDX: 22601.0, NH3_IDX: 0.0}
MAX_STORAGE = {H2_IDX: 807197.0, N2_IDX: 4.52e6, NH3_IDX: 0.0}
TARGET_NH3_RATE = 1000.0
RAMP_DOWN_LIMIT = 100.0
RAMP_UP_LIMIT = 40.0
RAMP_WINDOW = 4
MAX_RAMP_EVENTS_IN_WINDOW = 1.0
MAX_PURCHASED_H2 = 100.0
INITIAL_ELECTROLYZERS_ON = 0
INITIAL_DA = {DA_START: 0, -1: 0, DA_END_INITIAL: 0}

# Objective coefficients.
ELECTROLYZER_COST = 0.42
PSA_COST = 0.10
PURCHASED_H2_COST = 2.30
FOSSIL_H2_EMISSIONS_FACTOR = 9.3
ELECTROLYTIC_H2_WATER_FACTOR = 9.0
PURCHASED_H2_WATER_FACTOR = 3.0
ELECTROLYZER_OPERATION_RISK = 1.0
H2_NH3_STOICHIOMETRIC_RATIO = 3.0 / 17.0
N2_NH3_STOICHIOMETRIC_RATIO = 14.0 / 17.0

# ORCA and test assertions.
REQUESTED_GROUPS = 2
EXPECTED_NUM_OBJECTIVES = 4
EXPECTED_NUM_VARIABLES_MINIMUM = 1
MATRIX_ABS_TOL = 1.0e-12
