# ORCA

**Objective Reduction Community Algorithm for nonlinear many-objective optimization.**

ORCA uses sampled objective gradients and constraint information to estimate
relationships between objectives and group them for objective reduction. This
repository contains the reusable Python implementation and the DTLZ, ellipse,
and carbon capture, utilization, and storage (CCUS) experiments accompanying:

> Hongxuan Wang and Andrew Allman. *ORCA: The Objective Reduction Community
> Algorithm for nonlinear many-objective optimization problems.*

[中文说明](README.zh-CN.md) · [API guide](docs/api.md) ·
[Reproduce the paper](docs/reproduction.md) ·
[Figure/table map](docs/reproduction_status.md) · [Known issues](docs/known_issues.md)

## Install

Python **3.12** is the validated environment. Run the following commands from
the repository root after cloning or downloading it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
The base install needs NumPy and SciPy; the quickstart uses deterministic
average-linkage grouping and does not need Julia or an external solver.
Install from this checkout: no PyPI publication is claimed.

For Leiden community detection or the Pyomo adapter:

```bash
python -m pip install -e ".[leiden,pyomo]"
```

For paper experiments and the validation suite, use the recorded dependency
versions instead:

```bash
python -m pip install -r requirements.txt
python scripts/check_release.py
```

Alternatively, `python -m pip install -e ".[experiments,leiden,test]"` installs
the experiment dependencies without the complete validation lock. It is not a
promise of the original numerical environment. CCUS optimization additionally
requires the separate Julia environment described below.

## Basic usage

```python
from orca import NonlinearORCAConfig, nonlinear_orca
from orca.benchmarks import DTLZ5Problem

problem = DTLZ5Problem(
    intrinsic_dimension=2,
    num_objectives=3,
    k_tail=2,
    initial_point_strategy="deterministic",
)
config = NonlinearORCAConfig(
    num_groups=2,
    num_points_per_seed=8,
    step_size=0.03,
    random_seed=0,
    grouping_method="average_linkage",
)
result = nonlinear_orca(problem, config)

print(result.adj_matrix)        # Pairwise ORCA graph scores
print(result.groups)            # One-based group label for each objective
print(result.input_data.points) # Selected feasible points
```

This small example demonstrates the API; it is not a manuscript benchmark run.
For a custom model, implement `NonlinearORCAProblem`, or pass precomputed
gradients to `nonlinear_orca_from_sampled_gradients`. See the [API guide](docs/api.md)
for array shapes and constraint conventions. Grouping alone does not solve the
reduced optimization problem; downstream experiment drivers implement that step.

## Repository layout

| Path | Contents |
|---|---|
| `src/orca/` | Installable library: configurations, sampling, derivatives, projections, interaction aggregation and grouping |
| `src/orca/benchmarks/` | Reusable benchmark problem definitions |
| `src/orca/experiments/` | Reusable comparison and downstream optimization helpers; extra dependencies required |
| `examples/` | Minimal quickstart and the manuscript ellipse example |
| `experiments/dtlz/` | DTLZ5/6 structural recovery, NSGA-III and diagnostic drivers |
| `experiments/dtlz9/` | DTLZ9 geometry, algorithm variants, experiments and plotting |
| `experiments/ccus/` | CCUS Julia project, Python analyses and original local result snapshot |
| `data/` | Compact manuscript input/result tables and archived DTLZ run records |
| `results/` | Archived DTLZ9 results |
| `figures/` | Figure-generation code and numerical inputs; historical layouts are identified in the figure/table map |
| `tests/` | Python tests, with the manuscript subset selected by `pytest.ini` |
| `scripts/` | Result audits and plotting entry points |
| `docs/` | API, reproduction instructions, result mapping and provenance |
| `environments/` | Validation dependency lock and historical environment records |
| `validation/` | Dated validation evidence and numerical comparisons |
| `archive/` | Historical implementations and original metadata; not the current execution entry point |
| `outputs/` | Fresh audit/plot/DTLZ outputs; generated locally and ignored by Git |

CCUS results remain next to their original drivers to preserve Julia/Python
file contracts. Some original ellipse, figure and CCUS commands write beside
their sources; the reproduction guide identifies these commands. Audit and
overview-plot entry points write to `outputs/`.

## Reproduce the manuscript

All commands below start at the repository root, in the activated environment.

| Task | Command |
|---|---|
| Python tests | `python -m pytest -q` |
| Stored-data integrity, numerical audits and tests | `python scripts/check_release.py` |
| Overview plots from archived data | `python scripts/reproduce_available.py` |
| CCUS manuscript plots from archived results | `python scripts/replot_ccus.py` |
| Ellipse numerical example | `python examples/ellipse/oval_objective_reduction_demo.py` |
| DTLZ5/6 grouping and follow-up experiments | `python -m experiments.dtlz.run_dtlz5_dtlz6_followup` |
| DTLZ5(5,16), five-seed NSGA-III experiment | `python -m experiments.dtlz.run_dtlz5_5_16_orca_nsga3` |
| DTLZ9 experiment options | `python -m experiments.dtlz9.run_phase1_3 --help` |

The full optimization commands can be substantially slower than the quickstart.
For CCUS, use Julia **1.10.0** and the included project/manifest:

```bash
cd experiments/ccus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The [reproduction guide](docs/reproduction.md) gives the exact condition-specific
sampling, Pareto and information-loss commands, seeds and output locations.

## Reproduction status

The previous preparation passed 41 targeted Python tests and 30 computational
checks, including the ellipse and CCUS postprocessing. Results of the present
layout migration are recorded in [validation/reorganization.md](validation/reorganization.md).

The following limits remain:

- Table 2 says sample standard deviation, while its generator and archived
  numbers use `ddof=0`. Both conventions are compared in the validation files.
- A previous fresh NSGA-III seed completed but differed from historical HV/IGD.
  Exact historical optimizer reproduction is not established.
- The full Fig. 4 DTLZ5(5,20) sensitivity grid has not been located.
- Julia NLP solves have not been rerun in this preparation. CCUS numerical
  audits recompute postprocessing from saved derivatives and NLP solutions.

See [known issues](docs/known_issues.md) for details. This is not a claim of
complete, bit-for-bit reproduction of every manuscript result.

## Additional DTLZ9 initialization test

A new [single-objective-start experiment](results/dtlz9_single_objective_starts/README.md)
compares analytical optimal seeds with interior-Pareto and partially active
controls. Under the existing n=10M sampling settings, optimal endpoints collapse
to one retained point; an M=5,n=50 step/deduplication-scale diagnostic restores
fixed-K recovery. These are additional diagnostics, not replacements for the
archived manuscript results. The core algorithm is unchanged.

## Numerical conventions

- ORCA graph scores use `A = (1 + signed_interaction) / 2`, with neutral value
  `0.5`; they are not Pearson correlation coefficients.
- The fixed-K experiments explicitly supply the number of objective groups.
- Nonlinear inequalities use `g(x) <= 0`. The core sampled-gradient interface
  has no separate equality-Jacobian field; CCUS and DTLZ9 handle their additional
  geometry in their experiment code.
- DTLZ9 `ORCA-current` and `ORCA-joint` are distinct implementations. The
  `equality_tangent` CCUS source label does not make those variants equivalent.
- Generalized DTLZ6(I,M) uses the implemented generalized angular construction;
  see the benchmark source before equating every instance with standard DTLZ6.

## Citation and provenance

Use [CITATION.cff](CITATION.cff) for the manuscript authors and title. Publication
status and DOI should be updated when confirmed.

This repository was reorganized from the author's local source preparation.
Imports and file paths were updated; scientific formulas, parameter defaults
and archived numerical data were preserved. See [migration notes](docs/migration.md)
and [source_manifest.json](docs/source_manifest.json) for the original paths and hashes.

The existing [MIT license](LICENSE) is retained. Historical Python metadata
contains an inconsistent author/email entry; it is preserved separately in
`archive/package_metadata/`, not copied into the active package metadata.
The outstanding attribution question is described in [known issues](docs/known_issues.md).
