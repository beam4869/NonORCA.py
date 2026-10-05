# DTLZ6 objective-network audit

## Primary protocol

- Standard Python `DTLZ6Problem`, analytic gradients, `k_tail=10`.
- Fixed `K=I` average-linkage partitioning.
- Deterministic initial points, one generated point per objective seed, seed
  points included, step size 0.03, active-constraint tolerance `1e-8`.
- Five random seeds (`0`--`4`). Runtime includes point generation, affinity
  construction, and clustering.
- Expected labels are the benchmark labels and were not changed:
  `{f1,...,f_(M-I+1)}` plus `I-1` singleton groups.

## Primary results

| Case | Returned labels (seed 0) | Expected labels | ARI | Exact runs | Off-diagonal affinity range (seed 0) | Median runtime |
|---|---|---|---:|---:|---:|---:|
| DTLZ6(2,3) | `1 2 1` | `1 1 2` | -0.500 | 0/5 | 0.228155--0.263157 | 0.0009 s |
| DTLZ6(3,5) | `1 1 1 2 3` | `1 1 1 2 3` | 1.000 | 5/5 | 0.249659--0.616583 | 0.0021 s |
| DTLZ6(4,7) | `1 1 1 1 2 3 4` | `1 1 1 1 2 3 4` | 1.000 | 5/5 | 0.243787--0.685011 | 0.0046 s |
| DTLZ6(6,10) | `1 1 1 1 1 2 3 4 5 6` | same | 1.000 | 5/5 | 0.206967--0.750602 | 0.0122 s |
| DTLZ6(7,16) | `1` for f1--f10; six singletons | same | 1.000 | 5/5 | 0.180298--0.822033 | 0.0400 s |

For DTLZ6(2,3), the expected within-group edge is `A(f1,f2)=0.228155`,
whereas `A(f1,f3)=0.263157` and `A(f2,f3)=0.262051`. Therefore the wrong
partition is already implied by the computed affinity matrix; this is not a
label-ordering artifact.

## Uniform sensitivity checks

The same five cases and five random seeds were rerun using each of the
following global configurations:

- deterministic initial points with 1, 5, or 10 generated points per seed;
- optimized single-objective initial points with one generated point per seed;
- active constraints only or all box constraints.

Every configuration produced 20/25 exact runs: all five seeds were correct for
the four larger cases, and all five seeds failed for DTLZ6(2,3). Thus no tested
uniform, reasonable point-generation or constraint-selection configuration
made all five requested cases successful.

## Figure implication

Do not present the five-panel DTLZ6 grid as five successful recoveries. Either
replace DTLZ6(2,3) with another verified case, or retain it with a visually
explicit failure/limitation annotation. The other four requested cases are
stable positive examples under the primary protocol.

Raw results are in `dtlz6_network_case_audit_raw.csv`; the compact table is in
`dtlz6_network_case_primary.csv`; seed-0 affinity matrices are stored one file
per case.
