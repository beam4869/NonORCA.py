# Augmented epsilon-constraint analysis

## Formulation

For each TAC budget epsilon, the direct CCUS NLP was solved as
`min J + 1e-4 * TAC_hat` subject to `TAC <= epsilon`, where `J = 0.5 * TotEmiss_hat + 0.5 * ISI_hat`. Hats denote the fixed three-objective payoff-table normalization. The augmentation removes weakly efficient ties. Ipopt multi-start solutions are local NLP solutions; global optimality is not claimed.

## Numerical result

- Epsilon levels solved: 121
- Unique nondominated points after tolerance-based deduplication: 120
- Direct optimized solutions: 119
- Retained feasible multistart seeds: 2
- Final frontier optimized points: 119
- Final frontier retained feasible seeds: 1
- Maximum equality residual: 4.777e-16
- Maximum source-carbon diagnostic residual: 4.777e-16
- Maximum inequality violation: 9.840e-09
- Maximum normalized epsilon violation: 1.005e-08

## Interpretation

The minimum-TAC endpoint is TAC=2.541 million with J=0.803486. The minimum-J endpoint is TAC=11.935 million with J=0.183443.

The maximum-distance knee point is TAC=7.473 million, J=0.642199, total emissions=73.796823, and ISI=17.691158.

The main sink changes from Urea to Saline Storage between TAC=9.508 and 9.586 million cost units per year.

Unlike the normalized weighted sum, which recovered only the two supported endpoints, the augmented epsilon-constraint formulation recovers the interior nonconvex trade-off set directly.

One final nondominated point is a retained feasible multistart seed. A targeted Ipopt restart from that point converged to a slightly worse local objective, so the better feasible point was retained and remains explicitly marked by the `solution_source` field.
