# CCUS example

From the repository root, run `python scripts/replot_ccus.py` to draw the CCUS figures from stored, model-generated data. The four manuscript figures use the outputs ending in `pareto_orca_network_vertical.pdf` and `grouping_projections_vertical.pdf` for Condition 1, and `pareto_orca_network.pdf` and `grouping_projections_information_loss.pdf` for Condition 2. The model equations and coefficients for both conditions are in `src/CCUSModel.jl` relative to this directory. The `results` directory holds the archived data used for these plots and the reported tables.

For fresh calculations, install the repository's Python dependencies, then work in `experiments/ccus`. The Julia environment uses version 1.10.0 with JuMP and Ipopt. Fresh runs write into `results/`, so use a new checkout to preserve a separate copy of the published data. The following commands use a POSIX shell.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'

for profile in direct_use_expansion supply_chain_condition2; do
    export CCUS_ORCA_PROFILE="$profile" CCUS_PARETO_PROFILE="$profile"
    if [ "$profile" = direct_use_expansion ]; then
        export CCUS_ORCA_RUN_LABEL=paper_main CCUS_ORCA_REPLICATES=20
        export CCUS_ORCA_FIRST_RANDOM_SEED=20250810 CCUS_PARETO_SOLVE_TIME_LIMIT=30
    else
        export CCUS_ORCA_RUN_LABEL=condition2_main CCUS_ORCA_REPLICATES=10
        export CCUS_ORCA_FIRST_RANDOM_SEED=20260824 CCUS_PARETO_SOLVE_TIME_LIMIT=8
    fi
    export CCUS_ORCA_POINTS_PER_SEED=40 CCUS_ORCA_STEP_SIZE=0.03
    julia --project=. scripts/run_paper_orca_fixed_points.jl
    python scripts/run_paper_aligned_orca.py

    export CCUS_PARETO_GRID_TAC=31 CCUS_PARETO_GRID_AUX=7 CCUS_REDUCED_GRID=61
    julia --project=. scripts/run_pareto_epsilon.jl
    julia --project=. scripts/repair_pareto_consistency.jl
    python scripts/analyze_pareto_orca.py
    julia --project=. scripts/summarize_pareto_structures.jl

    export CCUS_EXACT_INFO_LABEL=quantile21 CCUS_EXACT_RETAINED_LEVELS=11
    export CCUS_EXACT_EPSILON_LEVELS=21 CCUS_EXACT_SOLVE_TIME_LIMIT=5
    julia --project=. scripts/run_exact_information_loss_frontier.jl
    export CCUS_EXACT_RETAINED_RESOLUTIONS=6,11 CCUS_EXACT_EPSILON_RESOLUTIONS=11,21
    python scripts/analyze_exact_information_loss.py
done
```

The CCUS trajectory generator proposes each step using the inequality with the largest residual. Equality constraints enter the feasibility correction and the reported interaction score. For the published fine-grid comparison, select the 11-retained-level, 21-epsilon-level rows from each condition's `exact_info_loss_quantile21_summary.csv`. The Julia solves are local and can vary with the solver build or time limit.
