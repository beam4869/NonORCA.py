# Preliminary CCUS–ORCA findings

## Outcome

The reconstructed nonlinear model is numerically feasible for the baseline and
ten one-at-a-time parameter scenarios. All 55 required solves in those
scenarios terminated as `LOCALLY_SOLVED` and passed independent residual checks.
The deliberately stressed 90% capture target terminated as
`LOCALLY_INFEASIBLE`, rather than being accepted merely because Ipopt returned a
numerical point.

The baseline local maximum net-capture fraction is 0.872024. The 85% target is
feasible; the 90% target misses feasibility by 0.027976 in the dimensionless
capture constraint. This is a local nonlinear-program result, not a formal
global infeasibility certificate.

## Baseline objective relationships

ORCA was evaluated at four converged points per scenario: the TAC, total
emissions, ISI, and maximum-capture solutions. Active inequalities used a
`1e-6` tolerance, all equality constraints were included, and the ORCA settings
were `alpha=0.9`, `beta=100`.

| Objective pair | ORCA signed correlation |
|---|---:|
| TAC – total emissions | 0.558898 |
| TAC – ISI | 0.250402 |
| Total emissions – ISI | 0.659215 |

With descriptive average-linkage grouping at fixed `K=2`, total emissions and
ISI form one group, while TAC forms the other. This grouping remained unchanged
in all eleven feasible scenarios.

## Largest parameter-related changes

Values below are changes in signed ORCA correlation relative to the baseline.

| Scenario | Objective pair | Change |
|---|---|---:|
| Capture target 60% | TAC – ISI | -0.048917 |
| Grid carbon intensity 0.732 | Total emissions – ISI | +0.040719 |
| Capture target 60% | TAC – total emissions | -0.038473 |
| Sink capacity scale 0.80 | TAC – ISI | -0.029106 |
| Grid carbon intensity 0.732 | TAC – total emissions | -0.027031 |
| Sink capacity scale 0.80 | TAC – total emissions | -0.026532 |
| Grid carbon intensity 0.183 | Total emissions – ISI | -0.026035 |
| Sink loss scale 1.50 | TAC – ISI | -0.023778 |
| Capture efficiency 0.95 | TAC – ISI | +0.018972 |

Doubling the stated electricity price from 0.02 to 0.04 changed every signed
correlation by less than `1.1e-6` in this model. This indicates low sensitivity
under the retained legacy cost coefficients; it should not be generalized until
those coefficients and units are verified.

## Feasibility sensitivity

| Scenario | Local maximum capture fraction |
|---|---:|
| Baseline | 0.872024 |
| Capture efficiency 0.80 | 0.809882 |
| Capture efficiency 0.95 | 0.902864 |
| Grid carbon intensity 0.183 | 0.891882 |
| Grid carbon intensity 0.732 | 0.832308 |
| Sink loss scale 1.50 | 0.871245 |

The capture-efficiency and grid-carbon assumptions have the strongest effect on
the observed feasibility envelope among the tested parameters.

## Validation performed

- The model contains 82 normalized variables, 34 equality constraints, and 172
  inequalities including explicit bounds.
- The analytical starting point satisfies the model and the independently
  evaluated source-carbon balances.
- Forward-mode automatic derivatives agree with central finite differences for
  objective and constraint Jacobians.
- The 34 equality-gradient rows have full row rank at all 44 ORCA points; the
  previous rank-deficient waste-composition formulation has been removed.
- All accepted solutions satisfy equality, source-carbon diagnostic, and
  inequality tolerances of `1e-6`.
- A complete second run reproduced solution quantities and ORCA outputs exactly
  in this environment.

## Model passport and limitations

The source topology and empirical coefficients come from the legacy CCUS Julia
scripts. The reconstruction derives waste-stream composition analytically,
removes redundant equalities, uses dimensionless constraints, corrects carbon
accounting units, and smooths fractional-power cost terms near zero. The
original files were not modified.

The retained TAC empirical equations and their unit basis have not yet been
verified against primary source data. Ipopt provides local solutions, and the
current ORCA pilot samples four endpoint-style points per scenario rather than a
dense Pareto set. Therefore these findings establish a stable, testable pilot
model and sensitivity pattern, but not a final global or publication-ready
claim.
