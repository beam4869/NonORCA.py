# Normalized weighted-sum Pareto analysis

## Objective definitions

Total emissions and ISI are first payoff-normalized:

\[
\widehat{E}=\frac{E-E_{\min}}{E_{\max}-E_{\min}},\qquad
\widehat{ISI}=\frac{ISI-ISI_{\min}}{ISI_{\max}-ISI_{\min}},
\]

with

\[
E_{\min}=32.5004,\quad E_{\max}=124.4612,
\]

\[
ISI_{\min}=4.1691,\quad ISI_{\max}=20.3567.
\]

The grouped objective is

\[
J=0.5\widehat{E}+0.5\widehat{ISI}.
\]

The two-objective payoff endpoints are then used to normalize TAC and \(J\) in the
outer weighted sum:

\[
\min_x\; w_{TAC}\widehat{TAC}(x)+(1-w_{TAC})\widehat{J}(x).
\]

All tables and the Pareto panel report the original TAC, emissions, and ISI together
with the dimensionless grouped objective \(J\).

## Direct optimization protocol

- Scenario: `direct_use_expansion`.
- Weight grid: 101 uniformly spaced values, \(w_{TAC}=0,0.01,\ldots,1\).
- Solver: Ipopt applied directly to the nonlinear CCUS model.
- Deterministic starts: previous-weight warm start, both payoff endpoints, and the six
  best existing topology-diverse candidate starts for each weight.
- Endpoint protection: a weighted solve is retained only if it is at least as good as
  directly evaluating both known feasible endpoints at the same weight.
- All 101 weights returned acceptable feasible solutions.
- Maximum equality residual: \(4.78\times10^{-16}\).
- Maximum inequality violation: \(9.74\times10^{-9}\).

Global optimality is not claimed because Ipopt is a local NLP solver. The endpoint
comparison prevents acceptance of a local solution that is worse than a known feasible
endpoint, but it is not a global-optimality certificate.

## Weighted-sum result

After exact pairwise nondominance screening and numerical duplicate removal, the 101
weighted solves produce only two distinct supported Pareto points:

| Supported point | Weight regime | Main sink | TAC | Total emissions | ISI | \(J\) |
|---|---|---|---:|---:|---:|---:|
| WS01 | \(0.51\leq w_{TAC}\leq1.00\) | Urea | 2.541×10⁶ | 88.319 | 20.357 | 0.8035 |
| WS02 | \(0\leq w_{TAC}\leq0.50\) | Saline Storage | 1.193×10⁷ | 32.500 | 10.108 | 0.1834 |

At \(w_{TAC}=0.50\), the Saline Storage endpoint is marginally preferred because its
outer normalized TAC is 0.9993 rather than exactly 1. At \(w_{TAC}=0.51\), the optimizer
switches discontinuously to the Urea endpoint.

No independent intermediate topology and no Greenhouse-dominant solution is supported by
this weighted-sum scalarization.

## Interpretation

The gray reference envelope contains 54 candidate-set nondominated points between the two
endpoints. None is selected by any of the 101 weighted sums. These points are unsupported
points on a nonconvex minimization frontier: a straight weighted-sum supporting line touches
only the two endpoints.

Therefore, increasing the weight-grid density cannot recover the interior curve. For this
model, weighted sum returns the supported subset of the Pareto frontier, not the complete
Pareto frontier. Recovering the intermediate Urea–Saline allocation structures requires an
epsilon-constraint or another nonconvex-frontier-capable formulation.
