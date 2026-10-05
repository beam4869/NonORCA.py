# ORCA for nonlinear many-objective optimization

Research code and numerical data accompanying **ORCA: The Objective Reduction
Community Algorithm for nonlinear many-objective optimization problems**, by
Hongxuan Wang and Andrew Allman.

This release was organized from the author's uploaded **Nonlinear Paper.zip**
on 2026-10-03. The reference manuscript is **NLMaOP_OptE (5).pdf**. The local
folder contains more recent experiments than its embedded August supplement;
the current manuscript, rather than the older supplement README, determines
the figure and table mapping below.

[中文说明](README.zh-CN.md) · [Reproduction commands](docs/reproduction.md) ·
[Figure/table map](docs/reproduction_status.md) · [Known issues](docs/known_issues.md)

## Install and check

Use Python 3.12, from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_release.py
```

The check command validates stored manuscript tables and file integrity,
recomputes the ellipse example and CCUS postprocessing, and runs 41 existing
targeted Python tests. It writes reports under `outputs/`. It does not run
the Julia optimization experiments. The Python dependencies in `requirements.txt`
are the versions used for the new validation; historical lockfiles are retained
separately and do not guarantee identical optimizer trajectories.

To redraw three overview plots from the archived results:

```bash
python scripts/reproduce_available.py
```

## Repository layout

| Path | Purpose |
|---|---|
| `ORCA_python/` | Recovered Python package: nonlinear sampling, gradients, constraint-conditioned interactions, weights, grouping, benchmarks and tests |
| `run_dtlz5_5_16_orca_nsga3.py` | Current Table 2 experiment; five paired seeds, three optimization budgets/methods |
| `run_dtlz5_dtlz6_followup.py` | Table 1 high-dimensional structural recovery and additional downstream experiments |
| `run_dtlz6_m16_l2_aggregation.py` | Additional DTLZ6 aggregation experiments |
| `run_active_constraint_orca_vs_gradient.py` | Active-constraint diagnostic experiments |
| `dtlz9_experiments/` | DTLZ9 definitions, analytical geometry, comparison methods, tests and plotters |
| `ccus_orca_experiments/` | CCUS model, original Julia environment, sampling/Pareto/conditional-loss drivers, Python analysis and archived results |
| `Nonlinear_demonstration_code/` | Ellipse example with 68 selected points; additional historical demonstration/audit scripts |
| `figures/` | Figure generators and their numerical inputs; multiple historical layouts retained |
| `data/dtlz/` | DTLZ per-seed results and summaries; prior rounded manuscript tables are also retained |
| `data/ccus/` | Compact manuscript data snapshot used by the table checker; matches corresponding local CCUS CSVs |
| `results/dtlz9_phase3/` | Stored DTLZ9 validation and recovery results |
| `scripts/` | New release checks and archived-data plotting entry point |
| `validation/` | Reports from execution during this preparation, including optimizer smoke output |
| `archive/` | Historical Julia scripts, earlier DTLZ5(2,3) experiment and original local READMEs |
| `paper/reference_figures/` | Previously recovered reference PDFs for locating manuscript outputs |
| `docs/` | Reproduction instructions, provenance, figure/table mapping and limitations |

## Validation status

- 41 existing Python tests passed.
- 30 numerical checks passed: Table 1's 30 stored recovery runs, Table 2's
  per-seed-to-summary calculations, regenerated ellipse points/matrix/groups,
  both CCUS conditions' ORCA values and partitions, and conditional information
  loss recomputed from stored NLP points.
- A fresh DTLZ5(5,16) seed-0 run completed all three optimization protocols with
  the expected evaluation counts and feasible solutions. Its HV/IGD values
  differ from the historical seed-0 results. Exact historical optimizer
  reproduction is **not established**.
- Julia is unavailable in the preparation environment. CCUS NLP solves were
  **not rerun**; postprocessing used the original saved derivatives and solutions.

Two manuscript/data issues remain explicit: Table 2 calls its variability
"sample standard deviation", while its original generator and numerical values
use `ddof=0`; and the complete Fig. 4 DTLZ5(5,20) sensitivity-grid dataset/runner
has not been located. See [known issues](docs/known_issues.md), including the
recomputed `ddof=1` values. This release should not be described as a complete,
bit-for-bit reproduction of every manuscript result.

## Numerical conventions

`A=(1+signed_interaction)/2` is the ORCA graph score, with neutral value `0.5`.
It is not a Pearson correlation coefficient. Group count `K` is supplied in the
manuscript's fixed-K experiments. CCUS objective order is `TAC`, `TotEmiss`,
`ISI`; Condition 1 retains the historical filename prefix `direct_use_expansion`.
The CCUS analysis label `equality_tangent` is preserved from the source; it
must not be interpreted as proof that the core uses the separate DTLZ9
joint-active-set implementation. DTLZ9 retains distinct `ORCA-current` and
`ORCA-joint` variants.

The DTLZ6(I,M) implementation uses generalized DTLZ5 angles with the DTLZ6
distance function; the older supplement calls this family "D6". It should not
be confused with a claim that every generalized instance is the standard
DTLZ6 benchmark.

## Provenance and scope

The Python core and CCUS model were preserved byte-for-byte. Edits to experiment
drivers resolve repository and output paths; no numerical-method change was
made to the recovered drivers. Source paths and original/delivered SHA-256
hashes are recorded in `docs/source_manifest.json`. New checks, instructions,
CI configuration and result reports are identified separately.

Virtual environments, caches, third-party NAS benchmark trees, large TIFF
exports, presentations and manuscript drafts are excluded. Original numerical
CSV inputs remain available. No Git history, commit or remote push is included
in this preparation.

The root `LICENSE` is the existing MIT license from the author's `NLORCA`
repository. `CITATION.cff` gives the manuscript title without inventing a DOI or
publication status. The embedded Python snapshot has incomplete upstream
attribution metadata; see `docs/known_issues.md` before treating the root license
as evidence of rights in every inherited file.
