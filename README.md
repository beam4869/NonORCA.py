# ORCA for nonlinear many-objective optimization

Code and data for *ORCA: The Objective Reduction Community Algorithm for nonlinear many-objective optimization problems*, by Hongxuan Wang and Andrew Allman. Figure and table numbers refer to the Optimization and Engineering manuscript.

| Manuscript item | Implementation or archived data |
| --- | --- |
| Section 3.3 and Fig. 4 | `experiments/conference_dtlz5/dtlz5_sensitivity.jl` contains the conference implementation of `deltaGeneration`. It averages the projections of unit descent directions over constraint surfaces. The accompanying CSV contains the 81 displayed accuracy percentages for DTLZ5(5,20). |
| Figs. 1–2 | `examples/ellipse/` contains the numerical example, 68 selected points, and correlation matrix. `figures/illustrative_example/` draws the objective graph. |
| Fig. 3 and Table 1 | The DTLZ5/6 panel inputs and rendering scripts are in `figures/dtlz_paper/network_grid/`. The grouping driver is `experiments/dtlz/run_dtlz5_dtlz6_followup.py`. Six cases and five seeds per case are recorded in `data/dtlz/orca_dtlz5_dtlz6_gradient_grouping_raw.csv`. |
| Table 2 and Online Resource 1 | `experiments/dtlz/run_dtlz5_5_16_orca_nsga3.py` runs the five DTLZ5(5,16) seed pairs and baseline. `data/dtlz/` contains per-seed results and summaries for the 16- and 12-objective studies. The 12-objective CSV contains the full and ORCA rows for the 20 published seeds. |
| Figs. 5–6 | `experiments/dtlz9/` contains the DTLZ9 model, local interactions and analytical Pareto geometry. The representative matrices displayed in Fig. 5 use 25 analytical Pareto decision points. `results/dtlz9_phase3/` contains the stored matrices and summary. |
| CCUS, Tables 3–4 and Figs. 7–10 | [Model, trajectories, conditional information loss and plotting commands](experiments/ccus/README.md). The published CCUS trajectory generator selects one inequality per step. |

The original case drivers retain their own sampling choices. In the archived conference script the default step-size list contains only 0.003. The nine-step configuration appears in the same file. The published grid can be plotted directly from its CSV. The archive provides the grid at cell level; individual replicate records for the full grid have not been verified.

## Reproduce figures

From the repository root, install the Python 3.12 dependencies with `python -m pip install -r requirements.txt`. To draw the figures from their stored inputs or analytical models:

```sh
python examples/ellipse/plot_toy_selected_point_process.py
python figures/illustrative_example/make_illustrative_objective_graph.py
python figures/dtlz_paper/network_grid/make_figures_with_original_plot_code.py
python figures/dtlz_paper/network_grid/make_combined_dtlz5_dtlz6_six_panel.py
python experiments/conference_dtlz5/plot_sensitivity.py
python -m experiments.dtlz9.make_figures --results-dir results/dtlz9_phase3
python -m experiments.dtlz9.make_pareto_degeneracy_figure
python scripts/replot_ccus.py
```

To regenerate the ellipse numerical data, run `python examples/ellipse/oval_objective_reduction_demo.py`. For the DTLZ5/6 grouping and downstream studies run the two drivers under `experiments/dtlz/`. For the DTLZ9 simulations run `python -m experiments.dtlz9.run_phase1_3 --output-dir outputs/dtlz9_new`. Use a separate checkout when rerunning a driver that writes to its archived results directory.

The historical conference run requires Julia, a working CPLEX installation and license, and the Julia packages imported at the top of its script. CCUS uses a separate Julia environment in `experiments/ccus/`. Solver builds, random seeds and hardware can affect new optimization runs and timings. The archived Table 2 dispersion uses population standard deviations (`ddof=0`); `validation/table2_standard_deviation_audit.csv` gives both conventions. The 20-seed DTLZ5(5,12) summary uses sample standard deviations.
