# CCUS–ORCA experiments

This directory is an independent, reproducible reconstruction of the legacy
CCUS scripts. The original Google Drive files are read-only inputs and are not
modified.

## Model decisions

- All 82 decision variables are represented on an equivalent `[0, 1]` scale.
- Equality and inequality constraints are dimensionless before optimization and
  before ORCA receives their gradients.
- Waste-stream composition is derived from capture efficiency and inlet/outlet
  composition, removing four non-identifiable variables and four redundant
  source-carbon equalities. Those balances remain explicit diagnostics.
- Transport electricity emissions use the same conversion in `NetCapture` and
  `TotEmiss`: `kW-day/year × 24 h/day × kg/kWh ÷ 1e6 = kt/year`.
- Source emissions use the source CO2 fraction, and sink losses use the CO2 flow
  entering the sink.
- Legacy TAC correlations are retained provisionally. Their empirical
  coefficients and unit basis still require source-data verification.

## Commands

```bash
julia --project=. test/runtests.jl
julia --project=. scripts/run_baseline.jl
julia --project=. scripts/run_scenarios_and_export.jl
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/run_orca.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_orca_outputs.py
julia --project=. scripts/run_pareto_epsilon.jl
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/analyze_pareto_orca.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_pareto_outputs.py
julia --project=. scripts/run_diversity_experiments.jl
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/run_diversity_orca.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_diversity_outputs.py
CCUS_PARETO_PROFILE=direct_use_expansion CCUS_PARETO_GRID_TAC=31 CCUS_PARETO_GRID_AUX=7 CCUS_REDUCED_GRID=61 julia --project=. scripts/run_pareto_epsilon.jl
CCUS_PARETO_PROFILE=direct_use_expansion julia --project=. scripts/repair_pareto_consistency.jl
CCUS_PARETO_PROFILE=direct_use_expansion /Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/analyze_pareto_orca.py
CCUS_PARETO_PROFILE=direct_use_expansion julia --project=. scripts/summarize_pareto_structures.jl
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_direct_use_pareto_outputs.py
CCUS_ORCA_RUN_LABEL=paper_main CCUS_ORCA_STEP_SIZE=0.03 CCUS_ORCA_POINTS_PER_SEED=40 CCUS_ORCA_REPLICATES=20 CCUS_ORCA_FIRST_RANDOM_SEED=20250810 julia --project=. scripts/run_paper_orca_fixed_points.jl
CCUS_ORCA_RUN_LABEL=paper_main /Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/run_paper_aligned_orca.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/analyze_paper_orca_results.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_paper_aligned_orca_outputs.py
CCUS_PARETO_PROFILE=direct_use_expansion CCUS_EXACT_INFO_LABEL=quantile21 CCUS_EXACT_RETAINED_LEVELS=11 CCUS_EXACT_EPSILON_LEVELS=21 CCUS_EXACT_SOLVE_TIME_LIMIT=5.0 julia --project=. scripts/run_exact_information_loss_frontier.jl
CCUS_PARETO_PROFILE=direct_use_expansion CCUS_EXACT_INFO_LABEL=quantile21 CCUS_EXACT_RETAINED_RESOLUTIONS=6,11 CCUS_EXACT_EPSILON_RESOLUTIONS=11,21 /Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/analyze_exact_information_loss.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_exact_information_loss_outputs.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/plot_revised_pareto_demonstrations.py
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python test/test_revised_figure_outputs.py
julia --project=. scripts/run_supply_chain_correlation_screen.jl
/Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/analyze_supply_chain_correlation_screen.py
CCUS_ORCA_PROFILE=supply_chain_condition2 CCUS_ORCA_RUN_LABEL=condition2_main CCUS_ORCA_STEP_SIZE=0.03 CCUS_ORCA_POINTS_PER_SEED=40 CCUS_ORCA_REPLICATES=10 CCUS_ORCA_FIRST_RANDOM_SEED=20260824 julia --project=. scripts/run_paper_orca_fixed_points.jl
CCUS_ORCA_PROFILE=supply_chain_condition2 CCUS_ORCA_RUN_LABEL=condition2_main /Users/hongxuan/Documents/ORCA_python/ORCA_python/.venv/bin/python scripts/run_paper_aligned_orca.py
python3 scripts/plot_supply_chain_condition_comparison.py
```

Generated results are written only under `results/`.
The concise interpretation is in `results/preliminary_findings.md`.
The deterministic Pareto/ORCA comparison is in
`results/pareto_orca_analysis.md`.
The sink/preprocessing parameter experiments are in
`results/diversity_findings.md`.
The detailed pointwise and aggregated ORCA interpretation for the four
recommended scenarios is in `results/recommended_scenarios_orca_analysis.md`.
The direct-use three-objective Pareto and information-loss validation is in
`results/direct_use_expansion_pareto_orca_analysis.md`.
The paper-aligned nonlinear ORCA validation, endpoint comparison, and sampling
sensitivity audit are in
`results/direct_use_expansion_paper_aligned_orca_analysis.md`.
The Russell--Allman exact conditional-slice information-loss recalculation is in
`results/direct_use_expansion_exact_information_loss_analysis.md`.
The original direct-use model is frozen as `supply_chain_condition1_parameters()`.
The exploratory loss-cost/capacity/source scenario is
`supply_chain_condition2_parameters()`, and its screened nonlinear ORCA comparison is
in `results/supply_chain_condition_orca_analysis.md`.
