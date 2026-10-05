# DTLZ5 and DTLZ6 experiments

Install the root `requirements.txt`, then run from the repository root:

```bash
python -m experiments.dtlz.run_dtlz5_dtlz6_followup
python -m experiments.dtlz.run_dtlz5_5_16_orca_nsga3
```

The first driver includes structural recovery and additional downstream
experiments. The second runs all five paired Table 2 seeds. New output goes to
`outputs/dtlz/`; `ORCA_OUTPUT_DIR` can override it. Historical output stays in
`data/dtlz/`. Settings and loop sizes are explicit in the driver source.

Additional drivers cover DTLZ6 aggregation and active-constraint diagnostics.
See [reproduction instructions](../../docs/reproduction.md) and
[known issues](../../docs/known_issues.md) before interpreting new optimizer
results as exact reproduction of the manuscript tables.
