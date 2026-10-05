"""Constants used by the ORCA Python translation.

Keeping these values here avoids hidden magic numbers in the implementation
files and makes it easier to switch between the uploaded Julia defaults and the
published paper defaults.
"""

from __future__ import annotations

# Numerical tolerances and clipping limits.
NUMERIC_ATOL: float = 1.0e-14
EXPONENT_CLIP_LIMIT: float = 700.0

# Correlation scaling.
SELF_CORRELATION: float = 1.0
MIN_CORRELATION: float = 0.0
MAX_CORRELATION: float = 1.0
CORRELATION_RESCALING_FACTOR: float = 0.5
DEFAULT_ZERO_WEIGHT_CORRELATION: float = 0.5

# Defaults that match the uploaded Julia implementation.
DEFAULT_ALPHA: float = 0.9
DEFAULT_BETA: float = 100.0
DEFAULT_NUM_GROUPS: int = 2
DEFAULT_NUM_SELECTED_POINTS: int = 1
DEFAULT_FILTER_INACTIVE_INEQUALITIES: bool = True
DEFAULT_GROUPING_METHOD: str = "auto"

# Values from the published paper, included for explicit paper-replication runs.
PAPER_ALPHA: float = 1.0
PAPER_BETA: float = 10.0

# Leiden resolution search defaults.
DEFAULT_RESOLUTION_START: float = 0.01
DEFAULT_RESOLUTION_STOP: float = 5.0
DEFAULT_RESOLUTION_STEPS: int = 200
LEIDEN_WEIGHT_ATTRIBUTE: str = "weight"
LEIDEN_GRAPH_MODE: str = "UNDIRECTED"

# Group labeling and validation.
MIN_GROUP_COUNT: int = 1
GROUP_LABEL_START: int = 1

# Fallback agglomerative grouping.
NEGATIVE_INFINITY: float = float("-inf")

# Smoke-test matrices in orca_main.py.
SMOKE_TEST_NUM_GROUPS: int = 2
