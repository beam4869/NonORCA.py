# Running the recovered workflows

Commands assume macOS/Linux, Python 3.12 and `python -m pip install -r requirements.txt`
from the root README, which also installs the local `orca` package.
Run from the repository root unless a block explicitly changes directory.
The source audit is read-only for archived inputs. Original scientific drivers
can overwrite their own result folders; use a working copy for full reruns if
you want to retain the archived snapshot for comparison.

## Release checks and compact figures

```bash
python scripts/check_release.py
python scripts/reproduce_available.py
python scripts/replot_ccus.py
```

Reports and new plots are written under `outputs/`. `replot_ccus.py` stages its
inputs there and calls the original plot functions for both conditions,
including the condition-2 grouping figure omitted from the original `main()`.
The 41-test list is explicit in `check_release.py`; `python -m pytest -q` selects the same
41-test manuscript subset through `pytest.ini`. Legacy ammonia and CCUS
file-contract tests remain outside this default suite.

## Ellipse, Figs. 1–2

```bash
python examples/ellipse/oval_objective_reduction_demo.py
python examples/ellipse/plot_toy_selected_point_process.py
python figures/illustrative_example/make_illustrative_objective_graph.py
```

The example uses seed 14, step 0.24, four initial optima and 16 extra points per
optimum, yielding 68 points at K=3. The first command reruns the numerical
example and writes to its source folder. The plotting scripts retain historical
layout/version names; outputs need not match the reference PDFs byte-for-byte.

## DTLZ5/6 structural recovery and downstream optimization

```bash
python experiments/dtlz/run_dtlz5_dtlz6_followup.py
python experiments/dtlz/run_dtlz5_5_16_orca_nsga3.py
```

The first command runs Table 1 grouping checks **and** additional downstream
experiments. The second runs all five paired Table 2 seeds. Fresh results go to
`outputs/dtlz/`; `ORCA_OUTPUT_DIR` can override this path. Archived results live
in `data/dtlz/`. The complete optimization protocol has not been rerun in this
preparation. The completed single-seed smoke is recorded under `validation/`.
See the numerical-drift and SD notes before comparing fresh output to Table 2.

Additional experiments:

```bash
python experiments/dtlz/run_dtlz6_m16_l2_aggregation.py
python experiments/dtlz/run_active_constraint_orca_vs_gradient.py
```

Fig. 3 uses six network panels. Render the source panels before assembling them:

```bash
python figures/dtlz_paper/network_grid/make_figures_with_original_plot_code.py
python figures/dtlz_paper/network_grid/make_combined_dtlz5_dtlz6_six_panel.py
```

These consume committed CSVs under `source_data/networkx_all_labels/`. The
`make_legible_manuscript_networks.py` four-panel layout is an older alternative,
not the six-panel Fig. 3 of the reference manuscript. Fig. 4's full sensitivity
grid cannot yet be regenerated from identified inputs.

## DTLZ9

```bash
python -m pytest -q tests/dtlz9/test_dtlz9.py
python -m experiments.dtlz9.run_phase1_3 \
  --output-dir outputs/dtlz9_new --seeds 30 --points-per-case 12 \
  --active-tolerance 1e-10
python -m experiments.dtlz9.make_figures --results-dir outputs/dtlz9_new
python -m experiments.dtlz9.make_pareto_degeneracy_figure \
  --output-dir outputs/dtlz9_new/figures
```

To plot the historical results, use `--results-dir results/dtlz9_phase3` in the
figure command. `make_figures.py` selects `ORCA-current` for its primary matrix
panel and also plots broader comparisons, including `ORCA-joint`. These
methods remain distinct. Only the analytical tests, rather than the complete
30-seed phase-3 sweep, are part of the routine validation suite.

## CCUS: original Julia environment

```bash
cd experiments/ccus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

The project was locked with Julia 1.10.0 and includes JuMP, Ipopt, ForwardDiff,
CSV and DataFrames. These Julia commands were not run in this preparation.
All following CCUS commands are from this same `experiments/ccus` directory.

### Main ORCA sampling and aggregation

Condition 1:

```bash
CCUS_ORCA_RUN_LABEL=paper_main CCUS_ORCA_STEP_SIZE=0.03 \
CCUS_ORCA_POINTS_PER_SEED=40 CCUS_ORCA_REPLICATES=20 \
CCUS_ORCA_FIRST_RANDOM_SEED=20250810 \
julia --project=. scripts/run_paper_orca_fixed_points.jl

CCUS_ORCA_RUN_LABEL=paper_main python scripts/run_paper_aligned_orca.py
```

Condition 2:

```bash
CCUS_ORCA_PROFILE=supply_chain_condition2 \
CCUS_ORCA_RUN_LABEL=condition2_main CCUS_ORCA_STEP_SIZE=0.03 \
CCUS_ORCA_POINTS_PER_SEED=40 CCUS_ORCA_REPLICATES=10 \
CCUS_ORCA_FIRST_RANDOM_SEED=20260824 \
julia --project=. scripts/run_paper_orca_fixed_points.jl

CCUS_ORCA_PROFILE=supply_chain_condition2 CCUS_ORCA_RUN_LABEL=condition2_main \
python scripts/run_paper_aligned_orca.py
```

The Python script automatically finds the bundled ORCA package. `ORCA_REPO`
can still override the repository root containing `src/orca/`.

### Pareto approximation

```bash
CCUS_PARETO_PROFILE=direct_use_expansion \
CCUS_PARETO_GRID_TAC=31 CCUS_PARETO_GRID_AUX=7 CCUS_REDUCED_GRID=61 \
julia --project=. scripts/run_pareto_epsilon.jl

CCUS_PARETO_PROFILE=direct_use_expansion \
julia --project=. scripts/repair_pareto_consistency.jl

CCUS_PARETO_PROFILE=direct_use_expansion python scripts/analyze_pareto_orca.py
CCUS_PARETO_PROFILE=direct_use_expansion \
julia --project=. scripts/summarize_pareto_structures.jl
```

For Condition 2, substitute `supply_chain_condition2` for the profile in each
command. Results are local NLP approximations, not global certificates.

### Primary conditional information loss

```bash
CCUS_PARETO_PROFILE=direct_use_expansion CCUS_EXACT_INFO_LABEL=quantile21 \
CCUS_EXACT_RETAINED_LEVELS=11 CCUS_EXACT_EPSILON_LEVELS=21 \
CCUS_EXACT_SOLVE_TIME_LIMIT=5.0 \
julia --project=. scripts/run_exact_information_loss_frontier.jl

CCUS_PARETO_PROFILE=direct_use_expansion CCUS_EXACT_INFO_LABEL=quantile21 \
CCUS_EXACT_RETAINED_RESOLUTIONS=6,11 CCUS_EXACT_EPSILON_RESOLUTIONS=11,21 \
python scripts/analyze_exact_information_loss.py
```

Use the Condition 2 profile in both commands for its corresponding results.
The `quantile21` results are the current Table 4 inputs. Earlier bin/neighbor
loss reports and pilot grids are retained as history and must not be substituted
for the primary conditional-slice metric.

## Additional DTLZ9 single-objective initialization diagnostics

These are new tests, separate from the manuscript's archived phase-3 results.
From the repository root:

```bash
python -m experiments.dtlz9.run_single_objective_starts --output-dir outputs/dtlz9_optima_new
python -m experiments.dtlz9.diagnose_endpoint_sampling --output-dir outputs/dtlz9_diagnosis_new
```

The first command uses ten random seeds, M=3/5/10 and block sizes 1/10,
three initialization families, step 0.01 and three extra points per seed.
The second diagnoses step size and deduplication distance at M=5,n=50.
Use a fresh output directory. Saved decisions, graph matrices, reports and
source hashes are in `results/dtlz9_single_objective_starts/`.
