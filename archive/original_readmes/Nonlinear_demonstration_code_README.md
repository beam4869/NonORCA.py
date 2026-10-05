# Simple Demonstration Cases

## Two-dimensional oval nonlinear case

The 2D oval case is the simpler paper-facing demonstration. It calls the local
nonlinear ORCA Python package in `/Users/hongxuan/Documents/ORCA_python/ORCA_python`
and uses four objectives over

```text
(x1 / 1.35)^2 + (x2 / 0.82)^2 <= 1
```

with two nonlinear objectives:

```text
f1(x) = -x1
f2(x) = -0.94*x1 - 0.10*x2 + 0.05*x2^2
f3(x) =  x1
f4(x) = -x2 + 0.18*x1^2
```

Run:

```bash
python3 Nonlinear_demonstration_code/oval_objective_reduction_demo.py
```

Current result:

```text
f1-f2 correlation strength = 0.767
G1 = {f1, f2}
G2 = {f3}
G3 = {f4}
```

See `oval_case_study_notes.md` for the manuscript-facing description and output list.
The process figure `oval_algorithm_steps_2d.png` shows the initial oval plus five
cumulative selected-point panels with local objective gradients and local feasible
constraints tangent to the oval.
The journal-style three-panel schematic `toy_selected_point_process.png` uses
the simplified prescribed-angle toy problem and includes a projection inset.

## Simple Ellipsoid Demonstration

This folder contains a small, solver-free Julia demonstration of the nonlinear objective
dimensionality reduction algorithm on a 3D ellipsoid.

## Problem

All objectives are minimized over the ellipsoid

```text
(x1 / 1.40)^2 + (x2 / 0.90)^2 + (x3 / 0.60)^2 <= 1
```

with four linear objectives:

```text
f1(x) = -x1
f2(x) = -0.99x1 - 0.05x2
f3(x) =  x1
f4(x) = -x3
```

The intended structure is:

```text
{f1, f2} are correlated
{f3} conflicts with f1/f2
{f4} is mostly separate from the x1-x2 tradeoff
```

## Script

Run:

```bash
julia Nonlinear_demonstration_code/ellipsoid_objective_reduction_demo.jl
```

The script uses only Julia standard libraries. It computes:

- closed-form single-objective optima on the ellipsoid,
- projected-gradient selected points using the same conic-combination idea as
  the DTLZ/CCUS code,
- projection of trial points back to the ellipsoid,
- projected-vector objective correlation scores with the same retained-weight
  logic as `NLPCorrStrengGenerating`,
- a simple threshold grouping for this demonstration.

For the full paper algorithm, the threshold grouping can be replaced by the Leiden
community-detection routine used in the DTLZ and CCUS scripts.

## Outputs

- `ellipsoid_selected_points.csv`: selected point trajectory from each objective optimum.
- `ellipsoid_step_details.csv`: current point, weighted projected direction,
  trial point, projected point, and projection distance for every step.
- `ellipsoid_projected_directions.csv`: per-objective descent direction,
  projected direction, projected norm, and random weight for every step.
- `ellipsoid_objective_vertices.csv`: single-objective optima and objective values.
- `ellipsoid_correlation_matrix.csv`: final objective correlation matrix.
- `ellipsoid_raw_alignment_matrix.csv`: unweighted projected-gradient alignments.
- `ellipsoid_total_weight_matrix.csv`: retained weight totals used in the
  weighted objective-correlation calculation.
- `ellipsoid_unprojected_objective_matrix.csv`: raw objective-direction angle
  scores before projecting onto the ellipsoid surface.
- `ellipsoid_groups.txt`: resulting objective groups.

Current result:

```text
G1 = {f1, f2}
G2 = {f3}
G3 = {f4}
```
