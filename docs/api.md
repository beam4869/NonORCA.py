# Using ORCA with your own model

Install the repository first with `python -m pip install -e .`. Imports now use
`orca`, for example `from orca import NonlinearORCAConfig, nonlinear_orca`.
The package itself does not import the top-level `experiments/` or `examples/`.

## A nonlinear problem

Subclass `orca.nonlinear.NonlinearORCAProblem` and implement:

| Method | Return value |
|---|---|
| `num_variables()` | Number of decision variables, n |
| `num_objectives()` | Number of objective functions, m |
| `objective_values(x)` | Array of shape `(m,)` |
| `constraint_values(x)` | Inequality values of shape `(q,)`, with `g(x) <= 0` feasible |
| `objective_gradients(x)` | Gradient rows of shape `(m, n)` |
| `constraint_jacobian(x)` | Constraint-gradient rows of shape `(q, n)` |
| `project_feasible(y)` | Feasible projected decision vector of shape `(n,)` |
| `initial_points()` | Sequence of feasible seed vectors, such as single-objective optima |

Call `nonlinear_orca(problem, config)`. Returned `ORCAResult` includes
`adj_matrix`, `groups`, `input_data` and `interaction_data`. See the
[ellipse problem](../examples/ellipse/oval_objective_reduction_demo.py) for a
complete custom model and [quickstart](../examples/quickstart.py) for a small call.

## Precomputed derivatives

Create `orca.SampledGradientBlocks` with the following arrays, where N is the
number of sampled feasible points:

| Field | Shape |
|---|---|
| `points` | `(N, n)` |
| `objective_gradients` | `(N, m, n)` |
| `constraint_jacobians` | `(N, q, n)` |
| `constraint_values` | Optional `(N, q)` |

Then call `orca.nonlinear_orca_from_sampled_gradients(blocks, config)`.
This interface has no separate equality-Jacobian input. The CCUS and DTLZ9
experiment implementations handle additional geometry explicitly; they should
not be substituted for the core routine without checking their conventions.

## Configuration and grouping

`NonlinearORCAConfig` controls `num_groups`, `step_size`, `num_points_per_seed`,
`random_seed`, `active_constraint_tolerance` and weighting parameters. Defaults
are documented in [config.py](../src/orca/config.py); paper drivers specify
their own settings. `active_constraint_tolerance=None` retains all sampled
constraint rows in the local interaction calculation.

Use `grouping_method="average_linkage"` for deterministic grouping with the
base install. For Leiden, install `.[leiden]` and use `grouping_method="leiden"`.
The recovered implementation searches a resolution grid and returns the
partition with group count closest to the requested K; check the returned
count when exact K matters. `auto` selects the implementation's fallback logic.

## Linear models

`linear_orca_from_matrices(Jineq, Jobj, Jeq=None, config=...)` accepts objective
and constraint coefficient rows. Inequalities are in `<=` form and equality
rows are supplied separately. `linear_orca_from_pyomo(model, objective_exprs,
config=...)` additionally requires `.[pyomo]` and linear model expressions.

## Outputs and interpretation

`adj_matrix` is an m-by-m ORCA graph score matrix and `groups` contains one-based
labels. Compare partitions rather than assuming label numbers have semantic
meaning. Scores use `A=(1+signed_interaction)/2`, with neutral score 0.5.
The package returns grouping information; selecting representatives or
aggregating objectives and solving a reduced problem are downstream steps.
