# DTLZ9–ORCA Phase 1–3 experiment package

This package evaluates whether ORCA can recover the analytical DTLZ9 objective grouping

\[
\{f_1,\ldots,f_{M-1}\},\quad \{f_M\},
\]

and whether its signed strengths distinguish alignment from competition.

The package imports the existing ORCA implementation through `orca_bridge.py`; it does not modify the ORCA repository.

## Reproduce

From the project root:

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
PYTHONPATH='/Users/hongxuan/Documents/ORCA_python:/Users/hongxuan/Documents/Nonlinear Paper' \
python dtlz9_experiments/test_dtlz9.py
```

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
PYTHONPATH='/Users/hongxuan/Documents/ORCA_python:/Users/hongxuan/Documents/Nonlinear Paper' \
python dtlz9_experiments/run_phase1_3.py \
  --output-dir results/dtlz9_phase3 \
  --seeds 30 \
  --points-per-case 12 \
  --active-tolerance 1e-10
```

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
python dtlz9_experiments/make_figures.py --results-dir results/dtlz9_phase3
```

Set `ORCA_REPO` if the existing ORCA checkout is in a different location.

## Main files

- `problem.py`: analytical DTLZ9 objectives, constraints, Jacobians, exact-front and controlled sample generators.
- `geometry.py`: raw, current single-constraint, joint-tangent, Pearson and Spearman matrices plus grouping metrics.
- `run_phase1_3.py`: deterministic Phase 2 validation and Phase 3 recovery experiments.
- `test_dtlz9.py`: dependency-light executable validation suite.
- `make_figures.py`: SVG/PDF/PNG figure generation from saved CSV files.

Full numerical results and the run manifest are under `results/dtlz9_phase3`.
