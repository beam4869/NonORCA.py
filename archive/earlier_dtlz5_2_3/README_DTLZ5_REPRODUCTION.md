# DTLZ5(2,3) analytic-front reproduction

Run `reproduce_dtlz5_i2_m3_analytic.py` with the project ORCA source tree on
`PYTHONPATH`.  The historical numerical environment recorded in the August 3,
2026 execution log is listed in `requirements_dtlz5_historical.txt`.  The log
reported Python 3.12.11, NumPy 2.4.6, SciPy 1.17.1, pymoo 0.6.1.6, and
Matplotlib 3.10.9.  A Python 3.12.3 reconstruction with those numerical package
versions reproduced every persisted per-seed metric exactly.

Example:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_dtlz5_historical.txt
ORCA_REPO=/path/to/ORCA_python .venv/bin/python reproduce_dtlz5_i2_m3_analytic.py
```

The protocol uses DTLZ5(I=2,M=3) with ten distance variables and paired seeds
0 through 4.  Full NSGA-III uses the 28 Das-Dennis directions first meeting a
minimum population of 24 and runs for 200 generations, giving 5,600 function
evaluations.  Grouped NSGA-III uses the ORCA partition recovered separately for
each seed.  In all five runs it is `{f1,f2}|{f3}`.  The first group is replaced
by its Euclidean norm.  The reduced problem uses 24 directions and 234
generations, giving 5,616 evaluations.

pymoo defaults are retained: floating-point random sampling, tournament
selection by constraint violation and then random choice, SBX crossover with
probability 1 and distribution index 30, polynomial mutation with distribution
index 20, and duplicate elimination.  The final `result.X` nondominated set is
reevaluated in the original three-objective space.

The fixed analytic reference is a 2,001-point uniform angular grid on
`(cos(theta)/sqrt(2), cos(theta)/sqrt(2), sin(theta))`, including both endpoints.
GD is the mean distance from returned points to that grid.  IGD is the mean
distance from grid points to the returned set.  Fixed objective bounds `[0,1]`
make the stated external scaling the identity.  Across-seed uncertainty is the
sample standard deviation with `ddof=1`.  The output also gives a 10,001-point
grid sensitivity calculation.

The manuscript values are reproduced only by the 2,001-point primary grid:

| Method | GD, mean and sample SD | IGD, mean and sample SD |
| --- | ---: | ---: |
| Full NSGA-III | 0.0026212501 and 0.0015642130 | 0.0755112704 and 0.0025871798 |
| Grouped NSGA-III, L2 | 0.0005440092 and 0.0007252059 | 0.0178462799 and 0.0001403602 |

The project Poetry lock lists NumPy 1.26.4 and SciPy 1.14.1, but those were not
the versions used to create the persisted DTLZ5 CSV.  Under that older lock the
grouped trajectories remain identical, while three of five full NSGA-III final
sets change.  The historical DTLZ environment must therefore be disclosed
separately if exact numerical reproduction is required.
