# Supply-chain condition 2: 3D Pareto frontier, 2D projections, and information loss

## 1. Scope and conclusion

This analysis keeps the original four-source, six-sink continuous CCUS supply-chain
model and uses the exploratory `supply_chain_condition2` parameter set. The central
result is that the fine-grid Russell-Allman conditional information-loss ranking agrees
with the nonlinear ORCA ranking:

1. **TAC + total emissions**: ORCA adjacency `0.7513`, exact loss `0.2868`;
2. **TAC + ISI**: ORCA adjacency `0.6678`, exact loss `0.3098`;
3. **Total emissions + ISI**: ORCA adjacency `0.5944`, exact loss `0.8609`.

The strongest conclusion is that total emissions + ISI is the least appropriate pair to
group: it is last under ORCA and has an exact information loss roughly three times the
other two. The distinction between TAC + total emissions and TAC + ISI is much weaker:
their fine-grid losses differ by only `0.0230`, and the two groups exchange ranks when
the epsilon resolution is reduced from 21 to 11 levels.

## 2. Deterministic Pareto method

The three-objective set was generated with an augmented epsilon-constraint
scalarization, not a population or stochastic Pareto heuristic. Three primary-objective
orientations were solved:

- TAC primary: `31 x 31` total-emissions/ISI epsilon grid;
- total-emissions primary: `7 x 7` TAC/ISI epsilon grid;
- ISI primary: `7 x 7` TAC/total-emissions epsilon grid.

Each target used deterministic warm starts and objective anchor points. A point was
accepted only when the solver status, primal status, original model residuals, source
carbon diagnostic, and normalized epsilon violation passed the `1e-6` threshold. The
accepted candidates were combined, deduplicated in normalized objective space, and
filtered by three-objective dominance under minimization.

Ipopt supplies local NLP solutions, so this is a deterministic local Pareto
approximation rather than a globally certified complete frontier. The method is strict
with respect to epsilon-constraint construction and feasibility auditing, but no global
optimality certificate is claimed.

## 3. Three-dimensional frontier and system structure

The full scan accepted `558` three-objective candidates. After deduplication and joint
dominance filtering, `279` unique 3D Pareto points remained. Their raw objective ranges
were:

| Objective | Minimum | Maximum |
|---|---:|---:|
| TAC | `2.1072e6` | `3.3635e7` |
| Total emissions | `13.0506` | `501.5932` |
| ISI | `2.2519` | `39.3863` |

The frontier contains five dominant-sink structures:

| Main sink | Frontier points | Share |
|---|---:|---:|
| Methanol | 146 | 52.3% |
| Urea | 77 | 27.6% |
| Acetic Acid | 50 | 17.9% |
| Greenhouse | 3 | 1.1% |
| Algae | 3 | 1.1% |

The pretreatment fraction spans `0.1244-0.8527` with median `0.6873`; the direct-use
fraction spans `0.1473-0.8756`. Thus condition 2 creates a structurally diverse Pareto
set rather than forcing all objectives to select one sink or one preprocessing regime.

## 4. Literal 2D projections

All 279 points are Pareto efficient in three objectives. After dropping the third
objective and reapplying dominance in each two-objective plane, only the following
subsets remain 2D nondominated:

| Projection | 2D-nondominated points |
|---|---:|
| TAC-total emissions | 2 |
| TAC-ISI | 40 |
| Total emissions-ISI | 60 |

The TAC-total-emissions projection collapsing to two endpoints is direct geometric
evidence that these two objectives move together under condition 2. The other 277 points
remain valid 3D Pareto designs because they trade against ISI; they are not 2D Pareto
points after ISI is omitted.

## 5. Russell-Allman conditional information loss

For every proposed grouping, the ungrouped objective was fixed at 11 retained levels.
At each retained level, 21 epsilon targets were reoptimized to construct a conditional
Pareto slice. The information loss for slice p was calculated as the sum of the
normalized ranges of the two grouped objectives, and the reported loss is the mean over
retained levels. Only retained-active, globally nondominated points were used.

| Grouped objectives | Exact loss | SD across slices | Minimum | Maximum | Rank |
|---|---:|---:|---:|---:|---:|
| TAC + total emissions | **0.2868** | 0.1483 | 0.000012 | 0.4000 | 1 |
| TAC + ISI | 0.3098 | 0.1856 | 0.0000003 | 0.5384 | 2 |
| Total emissions + ISI | 0.8609 | 0.5621 | 0.0000001 | 1.6319 | 3 |

All three groupings represented `11/11` retained levels and all reported retained
levels contained at least two unique points after combining reoptimized and existing
frontier points. Across the `693` exact-slice targets, `561` were accepted and `392`
were retained-active. After the final global nondominance audit, 305 retained-active
generated points remained. The maximum retained-active normalized error was `8.78e-7`.

### Nested-grid sensitivity

| Retained x epsilon levels | TAC + emissions | TAC + ISI | Emissions + ISI | Ranking |
|---|---:|---:|---:|---|
| 6 x 11 | 0.2314 | **0.1706** | 0.7692 | 2-1-3 |
| 6 x 21 | **0.2374** | 0.2736 | 0.7925 | 1-2-3 |
| 11 x 11 | 0.2836 | **0.2402** | 0.8230 | 2-1-3 |
| 11 x 21 | **0.2868** | 0.3098 | 0.8609 | 1-2-3 |

Increasing retained levels from 6 to 11 preserves the ranking at a fixed epsilon
resolution. Increasing epsilon levels from 11 to 21 reverses the first two groups, while
emissions + ISI remains last in all four grids. Therefore the ORCA agreement at 11 x 21
is real but should be described as fine-grid agreement, not resolution-independent proof
that TAC + emissions is unambiguously the best grouping.

## 6. ORCA comparison

| Grouped objectives | ORCA adjacency | Signed ORCA strength | ORCA rank | Exact-loss rank |
|---|---:|---:|---:|---:|
| TAC + total emissions | 0.7513 | 0.5025 | 1 | 1 |
| TAC + ISI | 0.6678 | 0.3356 | 2 | 2 |
| Total emissions + ISI | 0.5944 | 0.1887 | 3 | 3 |

The fine-grid rank agreement supports ORCA as a screening metric for objective grouping
under condition 2. It does not imply a linear calibration between ORCA strength and
information loss: the first two ORCA adjacencies differ visibly, but their exact losses
are close and their across-slice distributions overlap. ORCA should therefore be used to
rank candidate groupings and then validated with Pareto slices, not interpreted as a
direct numerical estimator of loss.

## 7. Numerical audit and limitations

- Main/reduced targets: `741/1242` accepted (`558` three-objective and `183` grouped).
- Nested consistency audit: `205` three-objective holes were re-solved from stricter
  feasible witnesses; `0` were repaired and all `205` remain unresolved local failures.
  The reduced two-objective scans had `0` nested holes and accepted `61/61` levels per
  grouping.
- Maximum accepted raw equality residual: `4.78e-16`.
- Maximum accepted raw inequality violation: `9.98e-9`.
- Maximum accepted raw epsilon violation: `1.01e-8`.
- Maximum exact-slice inequality violation: `1.00e-8`.
- Maximum exact-slice epsilon violation: `1.03e-8`.
- The embedded `31 x 31` versus `16 x 16` TAC-primary comparison gives mean normalized
  fine-to-coarse distance `0.0283`, 95th percentile `0.0505`, and maximum `0.3479`.

The relatively large maximum grid distance and the 205 unresolved local holes mean that
absolute frontier completeness is not globally certified. The robust conclusions are the
sink diversity, the near-collapse of the TAC-emissions 2D projection, and the consistently
poor emissions-ISI grouping. The ordering of the first two groupings is a finer claim and
should always be accompanied by the epsilon-resolution sensitivity table.

## 8. Reproducible outputs

- `supply_chain_condition2_pareto_frontier.csv`: 279-point 3D frontier.
- `supply_chain_condition2_pareto_pairwise_projection_source.csv`: all pairwise
  projections and 2D-nondominance flags.
- `supply_chain_condition2_pareto_system_structures.csv`: sink/preprocessing structure.
- `supply_chain_condition2_exact_info_loss_quantile21_points_audited.csv`: audited
  conditional-slice points.
- `supply_chain_condition2_exact_info_loss_quantile21_slices.csv`: slice-level loss.
- `supply_chain_condition2_exact_info_loss_quantile21_summary.csv`: nested-grid summary.
- `supply_chain_condition2_exact_info_loss_quantile21_comparison.csv`: ORCA comparison.
- `supply_chain_condition2_pareto_3d_and_pairwise_2d.*`: main 3D/2D figure.
- `supply_chain_condition2_grouping_projections_information_loss.*`: grouped projections.
- `supply_chain_condition2_exact_information_loss_quantile21_validation.*`: loss and
  convergence validation.
