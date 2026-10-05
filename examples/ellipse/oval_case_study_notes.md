# Two-dimensional oval nonlinear ORCA demonstration

This case study replaces the 3D ellipsoid demonstration with a planar oval that is easier to visualize in the paper.

## Feasible region

All four objectives are minimized over the filled oval

```text
(x1 / 1.35)^2 + (x2 / 0.82)^2 <= 1.
```

The selected points used by the nonlinear ORCA package calculation lie on the oval boundary and are generated from the four single-objective optima using projected conic combinations of the objective-gradient descent directions.

## Objectives

```text
f1(x) = -x1
f2(x) = -0.94*x1 - 0.10*x2 + 0.05*x2^2
f3(x) =  x1
f4(x) = -x2 + 0.18*x1^2
```

Objectives `f2` and `f4` are nonlinear. The intended structure is:

```text
{f1, f2} are correlated eastward objectives;
{f3} is the opposing westward objective;
{f4} is a mostly separate northward nonlinear objective.
```

## Result

The current run generates 68 selected points using the packaged `orca` implementation. The nonlinear ORCA package projected-gradient correlation matrix is:

```text
        f1      f2      f3      f4
f1   1.000   0.767   0.123   0.030
f2   0.767   1.000   0.115   0.034
f3   0.123   0.115   1.000   0.280
f4   0.030   0.034   0.280   1.000
```

Using the package's deterministic average-linkage grouping with `num_groups = 3`, the recovered groups are:

```text
G1 = {f1, f2}
G2 = {f3}
G3 = {f4}
```

## Generated files

Run:

```bash
python3 examples/ellipse/oval_objective_reduction_demo.py
```

Key outputs:

- `oval_algorithm_demo_2d.png` and `oval_algorithm_demo_2d.tiff`: paper-facing visual.
- `oval_algorithm_steps_2d.png` and `oval_algorithm_steps_2d.tiff`: six-panel process visual showing the four seed-objective selected-point processes, local objective gradients, and local feasible constraints tangent to the oval at selected points.
- `toy_selected_point_process.png/.pdf/.svg/.tiff`: journal-style three-panel schematic for the simplified prescribed-angle toy problem, with Step 0-2 panels and a projection-mechanism inset.
- `toy_selected_point_process_caption.txt`: proposed manuscript caption for the three-panel schematic.
- `oval_correlation_matrix.csv`: final projected-gradient correlation strengths.
- `oval_groups.txt`: recovered objective groups.
- `oval_selected_points.csv`: selected-point trajectory.
- `oval_projected_directions.csv`: objective descent directions and tangent projections at each step.
- `oval_objective_vertices.csv`: single-objective optima and objective values.
