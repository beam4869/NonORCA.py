# ORCA Python translation

This folder contains a modular Python translation of the uploaded Julia ORCA code.

## Setup

### Prerequisites

- Python
- Poetry

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
cd ORCA_python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
cd ORCA_python
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies with Poetry

```bash
poetry install
```

If Poetry is configured to create its own virtualenv, you may either keep that behavior or point it at the `.venv` above.

## Files

- `linear/`: linear ORCA entry points and Pyomo matrix extraction.
- `nonlinear/`: nonlinear selected-point generation, gradient evaluation, feasible projection, and entry points.
- `utils/`: shared projection, local-interaction, weighting, aggregation, and grouping utilities.
- `benchmarks/`: lightweight benchmark problem definitions for validation, including DTLZ5(I, M), DTLZ6(I, M), WFG, constrained DTLZ/LIR-CMOP/CEC2009 CF helpers, and a recommendation registry.
- `src/`: legacy translation entry points kept for backward-compatible tests.
- `pyproject.toml`: Poetry dependency file. Run `poetry install` in this folder.

## Install with Poetry

```bash
cd orca_python
poetry install
```

The Poetry file includes all packages used by the scripts:

- `numpy`
- `scipy` for optimizer-backed DTLZ seed generation and feasible projection helpers
- `pyomo`
- `python-igraph`
- `leidenalg`
- `pandas` and `openpyxl` for the Excel-backed NH3 regression test
- `optproblems` and `diversipy` for the WFG adapter
- `pymoo` for the C2-DTLZ2/C3-DTLZ4 constrained-DTLZ adapter
- `pytest` for development tests

## Run tests

From the repository root that contains the `ORCA_python/` package directory:

```bash
cd /Users/hongxuan/Documents/ORCA_python
ORCA_python/.venv/bin/python -m pytest ORCA_python/test/test_nonlinear_benchmarks.py
ORCA_python/.venv/bin/python -m pytest ORCA_python/test/test_constrained_benchmarks.py
ORCA_python/.venv/bin/python -m pytest ORCA_python/test/test_experiment_harness.py
ORCA_python/.venv/bin/python -m pytest ORCA_python/test/test_nh3_linear_package.py
```

Or, if your active environment already has the dependencies:

```bash
pytest ORCA_python/test/test_nonlinear_benchmarks.py
pytest ORCA_python/test/test_constrained_benchmarks.py
pytest ORCA_python/test/test_experiment_harness.py
pytest ORCA_python/test/test_nh3_linear_package.py
```

The older legacy test can still be run from the package directory:

```bash
cd ORCA_python
pytest test/test_nh3.py
```

The NH3 tests read `test/Jan_2022.xlsx`, matching the Julia test data sheet and columns.

## Basic matrix usage

```python
from ORCA_python.config import LinearORCAConfig
from ORCA_python.linear import linear_orca_from_matrices

results = linear_orca_from_matrices(
    Jineq,
    Jobj,
    Jeq,
    config=LinearORCAConfig(num_groups=2),
)
print(results.adj_matrix)
print(results.groups)
```

## Pyomo usage

```python
from ORCA_python.config import LinearORCAConfig
from ORCA_python.linear import linear_orca_from_pyomo

# objective_exprs should be the linear expressions for your objectives, e.g.
# [Z, H, Psi, Xi]
results, blocks = linear_orca_from_pyomo(
    model,
    objective_exprs,
    config=LinearORCAConfig(num_groups=2),
)
print(results.adj_matrix)
print(results.groups)
print(blocks.var_names)
```

## Nonlinear usage

```python
from ORCA_python.config import NonlinearORCAConfig
from ORCA_python.nonlinear import nonlinear_orca

results = nonlinear_orca(
    problem,
    NonlinearORCAConfig(
        num_groups=2,
        step_size=0.03,
        num_points_per_seed=40,
    ),
)
print(results.adj_matrix)
print(results.groups)
```

`problem` must implement `NonlinearORCAProblem`: objective values, inequality
values `g(x) <= 0`, objective gradients, constraint Jacobians, feasible
projection, and initial seed points.

## Benchmark helpers

The package includes NumPy DTLZ5(I, M) and DTLZ6(I, M) helpers for nonlinear validation:

```python
from ORCA_python.benchmarks import DTLZ5Problem, DTLZ6Problem, expected_dtlz5_groups

problem = DTLZ5Problem(intrinsic_dimension=3, num_objectives=5)
print(expected_dtlz5_groups(5, 3))  # [1, 1, 1, 2, 3]

harder_problem = DTLZ6Problem(intrinsic_dimension=5, num_objectives=12)
```

DTLZ initial points default to optimizer-backed single-objective optima. The
code tries Ipopt through `cyipopt` first when available, then falls back to
SciPy SLSQP in `optimizer_backend="auto"`.

WFG1/WFG2/WFG3/WFG9 can be created through the `optproblems`-backed adapter:

```python
from ORCA_python.benchmarks import WFGProblem

wfg1 = WFGProblem("WFG1", num_objectives=3, num_variables=6, k=4)
print(wfg1.objective_values(wfg1.initial_points()[0]))
```

Constrained stress tests are also available:

```python
from ORCA_python.benchmarks import (
    CEC2009CFProblem,
    ConstrainedDTLZ6Problem,
    LIRCMOPProblem,
    PymooConstrainedProblem,
)

dtlz6_hole = ConstrainedDTLZ6Problem(intrinsic_dimension=5, num_objectives=12)
c2_dtlz2 = PymooConstrainedProblem("c2dtlz2")
lir5 = LIRCMOPProblem("LIRCMOP5")
cf8 = CEC2009CFProblem("CF8")
```

`ConstrainedDTLZ6Problem` keeps the DTLZ6 known objective grouping and adds a
circular decision-space hole. `PymooConstrainedProblem` delegates C2-DTLZ2 and
C3-DTLZ4 values to pymoo. `LIRCMOPProblem` implements the PlatEMO
LIR-CMOP5/6/13/14 formulas, and `CEC2009CFProblem` implements CF8/CF9/CF10
from the CEC2009 definitions.

`benchmarks.RECOMMENDED_BENCHMARKS` records the suggested validation roadmap:
DTLZ5/DTLZ6 as implemented positive controls; constrained DTLZ/LIR-CMOP/CF
stress tests; WFG; and remaining planned Minus-DTLZ, inverted DTLZ, MaF, UF,
and ZDT families.

## Experiment helpers

`experiments/` contains reusable helpers for paper-level validation:

```python
from ORCA_python.benchmarks import DTLZ5Problem, expected_dtlz5_groups
from ORCA_python.experiments import compare_grouping_methods, compare_full_vs_reduced_nsga3

problem = DTLZ5Problem(intrinsic_dimension=3, num_objectives=5)
expected = expected_dtlz5_groups(num_objectives=5, intrinsic_dimension=3)

baseline_results = compare_grouping_methods(
    problem,
    expected,
    num_groups=3,
)
for result in baseline_results:
    print(result.method, result.groups, result.ari)

downstream = compare_full_vs_reduced_nsga3(
    problem,
    expected,
    min_population_size=40,
    n_gen=40,
)
print(downstream.full_hv, downstream.reduced_hv)
print(downstream.full_igd, downstream.reduced_igd)
```

The baseline helper compares ORCA against objective-value correlation,
gradient-cosine grouping, and a random control using adjusted Rand index when
known group labels are available. The downstream helper runs NSGA-III on both
the original objectives and group-reduced objectives, then evaluates both
solution sets back in the original objective space using normalized HV and IGD.

## Constants

Numerical defaults are defined in `config.py`, including tolerance, logistic
weighting parameters, selected-point settings, Leiden search bounds, and
default group count. Test-specific model constants are defined in
`test/test_constants.py`.

To reproduce the paper settings instead of the uploaded Julia defaults, call:

```python
from ORCA_python.config import LinearORCAConfig

config = LinearORCAConfig(num_groups=2, alpha_weight=1.0, beta_weight=10.0)
```
