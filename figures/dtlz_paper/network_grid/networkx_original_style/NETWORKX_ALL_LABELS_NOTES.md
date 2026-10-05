# DTLZ NetworkX all-edge-label figures

## Delivered figures

- Ten standalone square figures: five DTLZ5 and five DTLZ6 cases.
- Two 183 × 90 mm composite figures: four square panels on the left and one large square panel on the right.
- Every pairwise edge is drawn and every affinity value is printed to three decimal places.

### DTLZ5 panel order

1. DTLZ5(2,3)
2. DTLZ5(3,5)
3. DTLZ5(4,7)
4. DTLZ5(6,10)
5. DTLZ5(7,12)

### DTLZ6 panel order

1. DTLZ6(2,4)
2. DTLZ6(3,5)
3. DTLZ6(4,7)
4. DTLZ6(6,10)
5. DTLZ6(7,12)

Every displayed case recovered the prescribed partition in all five seeds (`ARI=1.00`). DTLZ6(2,3) was not used as a positive panel because the current protocol reproducibly returns `[1,2,1]` instead of `[1,1,2]`.

## Relationship to the earlier plotting code

The new implementation deliberately retains the earlier visual workflow:

- `networkx.Graph`
- `networkx.circular_layout`
- `draw_networkx_nodes`, `draw_networkx_edges`, `draw_networkx_labels`
- `draw_networkx_edge_labels` for every pairwise value
- light-coral for the principal recovered group and distinct colors for singleton groups

The implementation is parameterized rather than fixed to 12 nodes. Affinity matrices and node colors come from the current algorithm output rather than a hard-coded matrix or manually entered expected groups. Numeric strings are rounded to three decimals rather than truncated.

The earlier code placed every number at the exact midpoint of its edge. Because several diametric edges then share the centre of the circle, the revised implementation deterministically staggers labels among five positions along their own edges. No label is removed.

## Methods represented in the figure

- Standard DTLZ5/6, `k_tail=10`.
- Analytic objective gradients.
- Deterministic initial points, one generated point per seed, seed points included.
- Step size `0.03`, active-constraint tolerance `1e-8`.
- Average-linkage grouping with fixed `K=I`.
- Seed 0 affinity is displayed; exact recovery was verified over seeds 0–4.
- Edge values are nonlinear ORCA affinities, not Pearson or Spearman correlations.

## Suggested captions

### DTLZ5

> **Objective-affinity graphs for DTLZ5 instances of increasing dimension.** Nodes represent objectives, and nodes with the same fill color belong to the same group inferred by fixed-`K` average-linkage clustering. Every pairwise edge and its nonlinear ORCA affinity are displayed. The circular layout is fixed by objective index and does not depend on the recovered partition. The displayed affinities are from seed 0; all five cases recovered the prescribed grouping in each of five runs (`ARI=1.00`). The pair `(I,M)` denotes the intrinsic and objective dimensions, respectively.

### DTLZ6

> **Objective-affinity graphs for DTLZ6 instances of increasing dimension.** Visual encoding and the fixed-`K` grouping protocol are identical to those used for DTLZ5. Every edge and affinity value are shown. Each displayed case recovered the prescribed grouping in all five runs (`ARI=1.00`). DTLZ6(2,4) is used as the smallest positive-control case because DTLZ6(2,3) is a reproducible limitation of the current selected-point protocol.

## Reproduction

```bash
python figures/dtlz_paper/network_grid/generate_networkx_figure_data.py
python figures/dtlz_paper/network_grid/make_networkx_original_style_figures.py
```

Each figure is exported as SVG, PDF, 300-dpi PNG and 600-dpi LZW-compressed TIFF.
