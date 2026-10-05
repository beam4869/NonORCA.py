# Supply-chain conditions and ORCA correlation change

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent; nature-figure
- Origin Mode: run + validate
- Origin Date: 2026-08-24
- Verification Status: VERIFIED FOR THE REPORTED LOCAL ORCA WORKFLOW
- Condition 1: frozen alias of the prior `direct_use_expansion` model
- Condition 2: exploratory parameter scenario; not plant-calibrated
- Optimization: JuMP/Ipopt local NLP solves and feasible projections
- ORCA: local user-developed package, nonlinear modified-gradient workflow

## Main result

Condition 1 groups emissions with ISI. Condition 2 instead groups TAC with emissions.
The condition2 result was recovered in 10/10 main replicates by both Leiden and
average-linkage grouping, and also in the inequality-only audit.

| Objective pair | Condition 1 signed mean | Condition 2 signed mean | Change |
|---|---:|---:|---:|
| TAC–emissions | 0.556256 | 0.502503 | -0.053753 |
| TAC–ISI | 0.511422 | 0.335586 | -0.175835 |
| Emissions–ISI | 0.766282 | 0.188734 | -0.577548 |

The group switch is driven mainly by the reduction of emissions–ISI coupling rather
than an increase in the absolute TAC–emissions strength. This distinction must be
retained in any manuscript interpretation.

## Condition definitions

Condition 1 exactly preserves the previous source capacities, sink capacities, losses,
transport coefficients, and objective coefficients. Condition 2 changes:

- source availability from `[357, 1260, 399, 3426]` to `[357, 1050, 350, 2800]`;
- sink/treatment capacities from `[300, 1400, 1400, 756, 900, 913]` to
  `[250, 1000, 650, 500, 700, 500]`;
- effective sink loss to `[0.30, 0.45, 0.02, 0.30, 0.00, 0.40]`;
- power price from `0.02` to `0.08` and power carbon intensity from `0.366` to `0.732`;
- treatment emissions from `0.0338` to `0.0676` and uniform matched treatment cost;
- sink processing cost proportional to effective carbon loss;
- sink safety to `[2, 1, 60, 2, 60, 1]`, with transport and treatment safety both `1`;
- source capture fraction to `1.0` and pump-power coefficients to zero.

These are mechanism-identification parameters. In particular, the zero pump-power
assumption and carbon-loss-dependent processing cost require replacement by calibrated
engineering/economic values before a real-system claim is made.

## Single-objective system structures

| Condition | Optimized objective | Main sink | Pretreated fraction | TAC (million) | Emissions | ISI |
|---|---|---|---:|---:|---:|---:|
| Condition 1 | TAC | Urea | 0.618 | 2.541 | 88.319 | 20.357 |
| Condition 1 | TotEmiss | Saline Storage | 0.561 | 11.963 | 32.500 | 10.210 |
| Condition 1 | ISI | Greenhouse | 0.000 | 33.247 | 124.461 | 4.169 |
| Condition 2 | TAC | Urea | 0.414 | 2.107 | 13.066 | 39.358 |
| Condition 2 | TotEmiss | Urea | 0.677 | 2.135 | 13.051 | 39.605 |
| Condition 2 | ISI | Greenhouse | 0.853 | 33.637 | 501.592 | 2.252 |

Under condition2, both TAC and emissions optima select Urea, whereas ISI selects
Greenhouse. The aligned endpoint topology is consistent with the ORCA grouping but is
not itself the ORCA calculation.

## Step-size sensitivity

| Step size | TAC–emissions | TAC–ISI | Emissions–ISI |
|---:|---:|---:|---:|
| 0.01 | 0.5475 | 0.3384 | 0.1986 |
| 0.03 | 0.5025 | 0.3356 | 0.1887 |
| 0.05 | 0.5083 | 0.3215 | 0.1884 |

TAC–emissions remains first at all three step sizes. The main run used 10 replicates,
40 added points per objective seed, and 1,230 total analyzed points. It had zero
projection failures and a maximum inequality violation of `6.05e-9`.

## Screening record and limitations

The full screening table contains all tested scenarios, including unsuccessful ones,
to avoid presenting only the selected case. Endpoint screening is only a selection
stage; the reported conclusion comes from the nonlinear ORCA validation.

Ipopt convergence is local, not a global optimality certificate. ORCA correlation is
a local feasible-direction relationship, not causal correlation and not a statistical
correlation across plants. Parameter selection was mechanism-guided and exploratory,
so the scenario demonstrates that the grouping can change; it does not estimate how
often this change occurs in real CCUS systems.

## Fallacy scan

- Correlation-causation: ORCA strength is not causal evidence.
- Local-to-global: local ORCA grouping is not automatically a global Pareto-redundancy result.
- Garden of forking paths / look-elsewhere: all screened scenarios are retained in the ranking table.
- Simpson, ecological, Berkson, collider, base-rate, regression-to-mean, survivorship,
  and reverse-causality fallacies have no direct evidence in this deterministic model
  comparison, but remain relevant for later empirical calibration.
