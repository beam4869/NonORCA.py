# Deterministic three-objective Pareto and ORCA validation

## Material Passport

- Source model: reconstructed nonlinear CCUS model in this experiment package.
- Method reference: Russell and Allman, AIChE Journal, Section 4.1 and the
  information-loss definition in Section 3.3.
- Optimization method: deterministic augmented epsilon-constraint
  scalarization; no evolutionary or population-based heuristic was used.
- Plotting backend: Python with matplotlib only.
- Verification status: reproduced model residual tests and output-integrity
  checks passed; global optimality is not claimed.

## Pareto calculation

Three single-objective anchors were first solved to build a payoff table. The
formal full-frontier calculation used a 21 by 21 grid in total-emissions and ISI
epsilon constraints with normalized TAC as the primary objective. Two 5 by 5
orthogonal grids, with total emissions and ISI as the primary objective, audited
the boundary regions. Every scalarization included a `1e-3` augmentation term
to suppress weakly efficient solutions and used three deterministic starting
points; remaining anchor starts were attempted when those failed.

The calculation produced 457 accepted full-frontier candidates and 123
reduced-frontier candidates. After duplicate and dominance filtering, 136
unique nondominated points remained. Their physical objective ranges are:

| Objective | Minimum | Maximum |
|---|---:|---:|
| TAC | 9.0772 million | 10.8160 million |
| Total emissions | 34.6251 kt/y | 121.2989 kt/y |
| ISI | 9.6845 | 34.7976 |

All accepted points terminated as `LOCALLY_SOLVED` or
`ALMOST_LOCALLY_SOLVED`. Maximum observed residuals were
`6.37e-16` for the equality equations, `9.95e-9` for the original
inequalities, and `1.29e-8` for the epsilon constraints.

The embedded 11 by 11 grid was compared with the 21 by 21 grid. The mean and
95th-percentile normalized nearest-front distances were 0.0340 and 0.0882,
respectively. The maximum distance was 0.3862 at an extreme branch, so further
adaptive refinement would still be useful before treating the point cloud as a
fully converged global surface.

## Section 4.1 validation

Russell and Allman's Section 4.1 checks whether the strongest ORCA pair is also
the pair whose grouping discards the least trade-off information on the full
three-dimensional Pareto frontier. Their discrete information-loss metric was
adapted to this continuous frontier in two ways:

1. equal-width slices along the one retained objective, using 8 to 30 slices;
2. fixed-neighbor local slices, using 8 to 30 nearby frontier points.

Objective values were normalized using the minimum and maximum values on the
computed full frontier, as in Equation 10 of the paper. For each slice, the
ranges of the two grouped objectives were added, matching Equation 11.

| Grouped objectives | ORCA strength | Mean information loss, equal bins | Mean information loss, neighbor slices | Mean distance to reduced frontier |
|---|---:|---:|---:|---:|
| Total emissions + ISI | **0.8296** | **0.2893** | **0.3382** | **0.0443** |
| TAC + ISI | 0.6252 | 0.3776 | 0.4952 | 0.7077 |
| TAC + total emissions | 0.7794 | 0.4723 | 0.5735 | 0.8927 |

Total emissions plus ISI had the lowest information loss under all 14 tested
slice definitions. Its reduced-space frontier retained 39 of 41 points as
nondominated and remained much closer to the full frontier. This independently
supports the strongest ORCA edge and the `K=2` grouping that places total
emissions and ISI together while leaving TAC separate.

The full three-edge ranking is only partially supported. ORCA ranks TAC plus
total emissions above TAC plus ISI, whereas both Pareto information-loss
definitions reverse those two. The descriptive Spearman rank correlation is
0.5 and Kendall tau is 0.333, but only three pairs exist, so inferential
significance is not meaningful. The defensible conclusion is therefore:

> ORCA reasonably identifies the best objective pair to group, but its edge
> strength should not yet be interpreted as a calibrated quantitative proxy for
> Pareto information loss in this nonlinear CCUS model.

## Important limitations

- The original article develops the structural correlation method for linear
  formulations and explicitly lists nonlinear extension as future work. The
  present pointwise-gradient ORCA analysis is therefore an empirical nonlinear
  extension, not a direct theorem from that paper.
- Ipopt certifies local nonlinear solutions. A formal global Pareto frontier
  would require a deterministic global nonlinear solver and tractable global
  bounds for this 82-variable model.
- The TAC empirical cost correlations and their unit basis remain provisional.
- Information-loss values for a continuous frontier require a slicing rule.
  The best grouping was invariant to both slicing families and every tested
  resolution, but the numerical loss magnitudes should still be reported with
  their sensitivity range.
