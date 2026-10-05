# Known issues and numerical limits

## Table 2 variability convention

The current manuscript caption says "sample standard deviation". The recovered
`experiments/dtlz/run_dtlz5_5_16_orca_nsga3.py:summarize` computes `std(ddof=0)` for time, HV
and IGD. The archived 15 rows reproduce every original summary value using
that convention. Across five seeds, sample SD is larger by `sqrt(5/4)`.

| Method | Time SD: reported / sample | HV SD: reported / sample | IGD SD: reported / sample |
|---|---|---|---|
| Full | 2.00 / 2.24 | 0.266 / 0.298 | 0.047 / 0.053 |
| Reduced, natural | 0.74 / 0.83 | 0.296 / 0.331 | 0.012 / 0.013 |
| Reduced, equal evaluations | 2.09 / 2.33 | 0.198 / 0.221 | 0.009 / 0.010 |

Two consistent resolutions are possible: describe the existing values as
population SD across the five runs, or use the provided sample-SD values and
update the generator for future results. This preparation preserves the
original driver and manuscript values; it does not silently make that scientific
reporting choice. Full precision is in
`validation/table2_standard_deviation_audit.csv`.

## Fresh NSGA-III run differs from archived trajectory

With the delivered source and the recorded validation environment, seed 0
completed all three methods with 13,600, 3,500 and 13,615 evaluations and no
constraint violations. HV and IGD differed from the historical seed-0 rows.
The comparison is retained in `validation/dtlz5_seed0_comparison.csv`.

The uploaded local folder contains the experiment script and raw metrics, but
its ORCA package is a snapshot embedded in an earlier supplement. A clean Git
commit for the exact Table 2 run and a complete Table-2-specific dependency
lock were not located. Numerical-library differences and snapshot differences
are possible causes; this preparation has not established which causes the
observed drift. Timings also depend on hardware and load. Do not equate an
executable run with exact reproduction of the archived optimization results.

The supplied driver does not persist its final decision vectors or pooled
HV/IGD reference archives. Its original saved outputs are per-seed metrics,
groups, summaries and plots. The separate older DTLZ5(2,3) historical environment
in `archive/earlier_dtlz5_2_3/` is not evidence of the environment for Table 2's
DTLZ5(5,16) experiment.

## Fig. 4 sensitivity grid

The archive contains older Julia sampling experiments and a DTLZ5(5,12)
one-iteration CSV, but no complete ten-repetition grid for the manuscript's
DTLZ5(5,20) heatmap was identified, including inside the nested ZIP archives.
The older Julia sampling/projection conventions differ from the Python code.
No missing measured values have been inferred from the figure or replaced by
synthetic runs.

## Julia, normalization and local solutions

The original Julia 1.10 project and manifest are included. Julia was not
available for fresh execution here. The CCUS checks use the saved feasible
points, derivatives, frontiers and conditional solutions, then recompute the
Python aggregation. They do not independently verify fresh Julia sampling,
Pareto optimization or conditional NLP solves. Ipopt produces local solutions.

The stored Condition 2 `*_normalized` columns use an older scale than the full
frontier extrema. The compact overview plotter explicitly uses full-frontier
min/max and reports the discrepancy, rather than modifying the archived CSVs.
The original primary plotters retain their own normalization conventions.

## Algorithm variants and historical metadata

DTLZ9's `ORCA-current` and `ORCA-joint` are distinct implemented methods. The
current Fig. 5 plotting source selects `ORCA-current`; the joint method is
retained as a separate diagnostic. A CCUS label such as `equality_tangent` is
a source label, not an assertion that the full joint-active-set projector is
used. No variants were merged or renamed during cleanup.

The Python snapshot's historical `archive/package_metadata/pyproject.toml` lists Hongxuan Wang with a
`yunqing@ualberta.ca` email address and contains no separate LICENSE file.
The root MIT license was carried over from the author's existing `NLORCA`
repository. Upstream authorship/permission and that inconsistent historical
metadata need confirmation before attributing the entire inherited snapshot
solely to the manuscript authors. Original metadata was preserved for review. The active root `pyproject.toml`
omits the inconsistent author/email entry and uses the new installable layout;
this is not a resolution of the historical attribution question.

The older supplement references another title, figure numbering and numerical
protocol. Its source is useful, but its README is not the authoritative map for
the current manuscript. The recovered Julia code under `archive/legacy_julia/`
is historical and is not the recommended current execution path.

## DTLZ9 optimal-endpoint initialization and decision-space scale

The additional 2026-10-05 experiment is under
`results/dtlz9_single_objective_starts/`. Exact analytical single-objective
minimizers contain zero variables, where the DTLZ9 derivative is singular.
The current repair raises zeros to 1e-30 before evaluation. With ten variables
per objective this raises the nominal zero objective to 0.01 and removes
initial structural activity.

Equal-share optimal endpoints have positive decision coordinates of 1e-10.
For M=5,n=50 their repaired Euclidean separation is about 7.071e-10, below
the default duplicate distance 1e-8. Using the current step 0.01 and three
additional points per seed retained only one point for every tested n=10M
optimal-start run. Neutral matrices can give apparently correct fixed-K
partitions solely from input-order tie breaking; permutation controls expose
this failure.

A 2x2 diagnostic at M=5,n=50 recovered reliable fixed-K=2 groups only when
both step size and duplicate distance were reduced (1e-11 and 1e-16).
This does not validate those settings across other dimensions or models.
The experiment retains the original repair, bound handling, core and
algorithm variants. See its report for the optimal-solution tie convention
and the distinction between fixed-K and unknown-K results.
