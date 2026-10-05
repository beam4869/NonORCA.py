# DTLZ9: sampling from single-objective optima

Date: 2026-10-05. The saved experiment uses 180 sampling runs (three start
families, six problem sizes and ten random seeds) plus 40 factorial diagnostic
runs. The ORCA core and the original DTLZ9 adapter are unchanged.

Read `DTLZ9_SingleObjective_Start_Report.html` for the complete Chinese report.

The original settings collapse to one selected point for n=10M. Every off-
diagonal ORCA-current graph score is neutral (0.5); apparent fixed-K recovery
in original objective order is a tie-breaking artifact. In the M=5,n=50
diagnostic, reducing both step size to 1e-11 and duplicate distance to 1e-16
restores 15–16 points and correct fixed-K=2 groups in all ten random seeds
and all 300 objective-order permutations. This is a case-specific diagnostic,
not a validated general default.

Analytical seed choices achieve each assigned objective's nonnegative lower
bound of zero. Pareto-efficient tie breaking produces two distinct endpoints
across M entries. Zero-coordinate derivatives are singular; the unchanged
repair clips to 1e-30 before differentiation, so the evaluated starting points
are perturbed optima. For block size ten the zero objective becomes 0.01.

The two controls use M seed entries as well: evenly spaced interior Pareto
points and partially active feasible points. The latter differs from the
older diagnostic's two-start budget. Selected-point counts vary because of
deduplication and stopping. All six configurations use step 0.01, three extra
points per seed, active tolerance 1e-10 and duplicate distance 1e-8.

Files:
- `sampling_runs.csv`, `sampling_summary.csv`: feasibility and coverage.
- `grouping_runs.csv`, `grouping_summary.csv`: Raw-Cosine, ORCA-current and
  ORCA-joint, fixed K=2 and positive-component grouping kept separate.
- `runs/*.npz`: actual selected decisions, objectives, constraints, exact and
  repaired seeds, graph matrices and ORCA-current total weights.
- `diagnosis/`: the 2x2 step-size/deduplication-distance experiment.
- `run_metadata.json`: parameters, analytic endpoint checks and source hashes.
- `current_matrix_comparison.png`: mean current-method graph matrices for the
  original and scaled M=5,n=50 settings.

Reproduce from the ORCA repository root after installing `requirements.txt`:

```bash
python -m experiments.dtlz9.run_single_objective_starts --output-dir outputs/dtlz9_optima_new
python -m experiments.dtlz9.diagnose_endpoint_sampling --output-dir outputs/dtlz9_diagnosis_new
```

The unknown-K diagnostic does not establish automatic group-count recovery
for ORCA-current. Original Julia/DTLZ9 projections, variants and historical
manuscript results were not replaced by these new experiments.
