# Manuscript-to-code map

Numbering follows the supplied `NLMaOP_OptE (5).pdf`, not the older manuscript
inside the uploaded folder. "Recomputed" below describes the checks actually
executed in this preparation. Historical logs retain their original dates and
are not evidence of a new run.

| Manuscript item | Code and input data | Status |
|---|---|---|
| Section 3, nonlinear ORCA | `src/orca/nonlinear/`, `src/orca/utils/`, `src/orca/config.py` | Scientific implementation retained from embedded supplement; imports migrated to `orca`; targeted tests cover the new layout |
| Section 4.1, Figs. 1–2 | `examples/ellipse/oval_objective_reduction_demo.py`, `oval_selected_points.csv`, `oval_correlation_matrix.csv`; `plot_toy_selected_point_process.py`; `figures/illustrative_example/` | All 68 points, adjacency and groups regenerated and matched; original plotting layouts retained |
| Section 4.2, Fig. 3 | `figures/dtlz_paper/network_grid/source_data/networkx_all_labels/`; `make_figures_with_original_plot_code.py` then `make_combined_dtlz5_dtlz6_six_panel.py` | Six-panel source data and rendering chain recovered; four-panel and five-panel alternatives are historical layouts |
| Table 1 | `experiments/dtlz/run_dtlz5_dtlz6_followup.py`; `data/dtlz/orca_dtlz5_dtlz6_gradient_grouping_raw.csv` | 30 ORCA runs, six cases × five seeds, all ARI=1 and exact recovery; archived rows checked, full high-dimensional rerun not performed |
| Fig. 4 | Related legacy scripts under `archive/legacy_julia/DTLZ_code/` | Complete DTLZ5(5,20) step-size × point-count grid with ten repetitions is still missing |
| Table 2 | `experiments/dtlz/run_dtlz5_5_16_orca_nsga3.py`; `data/dtlz/orca_dtlz5_5_16_nsga3_raw.csv` and matching summary | All 15 method/seed records recovered and summary reproduced; one fresh optimizer seed executed with numerical drift; sample-SD caption mismatch documented |
| Supporting DTLZ5(5,12) | `data/dtlz/dtlz5_5_12_nsga3_summary.csv`; broader reduction-study CSVs and core experiment harness | Rounded SI summary retained; a unique source/environment mapping for that exact summary is not asserted |
| Section 4.3, Figs. 5–6 | `experiments/dtlz9/problem.py`, `geometry.py`, `run_phase1_3.py`, `make_figures.py`, `make_pareto_degeneracy_figure.py`; `results/dtlz9_phase3/` | Analytical tests passed; raw gradients, ORCA-current, ORCA-joint and Pearson kept distinct; full phase-3 stochastic experiment not rerun |
| Section 4.4, Table 3 | `experiments/ccus/src/CCUSModel.jl`; `results/supply_chain_condition_parameter_comparison.csv` | Model exactly matches prior manuscript supplement; both parameter sets checked; Julia NLP not rerun |
| Table 4 | `scripts/run_paper_orca_fixed_points.jl`, `run_paper_aligned_orca.py`, `run_exact_information_loss_frontier.jl`, `analyze_exact_information_loss.py` inside CCUS | ORCA scores/partitions recomputed for 20+10 replicates; both conditional-loss analyses recomputed from saved NLP solutions and matched |
| Figs. 7–8, CCUS condition 1 | `experiments/ccus/scripts/plot_revised_pareto_demonstrations.py`; primary Pareto/grouping inputs | Model, raw optimization records, derivatives and plotting source recovered; no new Pareto NLP solves |
| Figs. 9–10, CCUS condition 2 | `plot_supply_chain_condition2_pareto.py`, `plot_supply_chain_condition_comparison.py`; condition-2 result files | Previously missing reduced-frontier, normalization and screening inputs recovered; no new Pareto NLP solves |

The CCUS test scripts include historical file/layout contracts, some requiring
large TIFFs and earlier reports. Those exports are intentionally excluded from
the release and those tests are not part of `check_release.py`. The release
instead checks the numerical evidence used by the current manuscript.
