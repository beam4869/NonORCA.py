# DTLZ objective-affinity network figures

## Figure contract

- **Core conclusion:** fixed-K nonlinear ORCA recovers the prescribed block-and-singleton objective structure over a sequence of DTLZ5 and DTLZ6 cases with 3–16 objectives.
- **Archetype:** asymmetric network grid; four equal square validation panels on the left and one square hero panel on the right.
- **Backend:** Python/Matplotlib only.
- **Final size:** 183 × 90 mm.
- **Primary evidence:** the large DTLZ5/6(7,16) network.
- **Supporting evidence:** four progressively sized cases and 5-seed exact-recovery annotations.
- **Reviewer risks controlled:** node colors come from recovered labels; objective order is fixed; every edge is retained; the edge scale is shared; fixed `K=I` and the excluded DTLZ6(2,3) failure are disclosed.

## Panel map

### DTLZ5

- a: DTLZ5(2,3)
- b: DTLZ5(3,5)
- c: DTLZ5(4,7)
- d: DTLZ5(6,10)
- e: DTLZ5(7,16)

All five cases recovered the prescribed partition in 5/5 seeds.

### DTLZ6

- a: DTLZ6(2,4)
- b: DTLZ6(3,5)
- c: DTLZ6(4,7)
- d: DTLZ6(6,10)
- e: DTLZ6(7,16)

All five displayed cases recovered the prescribed partition in 5/5 seeds. DTLZ6(2,3) was deliberately not used as a positive panel: it reproducibly returned `[1,2,1]` instead of `[1,1,2]` in 0/5 exact runs (`ARI=-0.50`).

## Data and visual encoding

- Standard DTLZ5/6 definitions, `k_tail=10`.
- Analytic gradients and deterministic initial points.
- One generated point per seed plus the seed points.
- Active-constraint tolerance `1e-8`; step size `0.03`.
- Average-linkage grouping with fixed `K=I`.
- The displayed affinity matrix is seed 0; the partition was checked over seeds 0–4.
- Node order is always objective index, with `f1` at the right and subsequent objectives counter-clockwise. It is not rearranged to make communities appear separated.
- Node fill is assigned from the inferred labels after canonical relabelling by first objective index; expected labels never determine the plotted color.
- The circular ordering and light-coral-plus-singleton color vocabulary follow the earlier plotting convention, while the implementation remains data-driven rather than manually assigning the expected groups.
- Every pairwise edge is drawn on the same raw `[0,1]` scale in every panel; there is no panel-wise min–max normalization or display threshold.
- Edge width and opacity encode ORCA objective affinity. These values are not Pearson/Spearman correlations.
- For `M<=5`, every edge value is labelled. For larger graphs, labels are limited by a deterministic rule: the maximum-spanning-tree edges of the principal recovered block plus the strongest connection from each singleton to that block. All unlabelled edges remain visible.
- Edge labels use three decimal places, matching the precision of the earlier network figure without truncating numeric strings.

## Suggested captions

### DTLZ5

> **Objective-affinity networks and recovered groups across DTLZ5 instances.** Nodes denote objectives, node fill denotes the group inferred by fixed-`K` average linkage, and edge width and opacity denote nonlinear ORCA affinity. All pairwise edges are shown on a common scale; for visual clarity, numerical labels are restricted to all edges when `M<=5` and to a pre-specified set of representative maximum-strength connections at larger `M`. The circular layout is fixed by objective index and is independent of the inferred partition. The displayed affinities are from seed 0, while the prescribed objective partition was recovered in all five runs for every case (`ARI=1.00`). The pair `(I,M)` gives the intrinsic and objective dimensions, respectively.

### DTLZ6

> **Objective-affinity networks and recovered groups across DTLZ6 instances.** Visual encodings and the fixed-`K` grouping protocol are identical to those in the DTLZ5 figure. The displayed five cases recover the prescribed block-and-singleton structure in all five seeds (`ARI=1.00`). DTLZ6(2,4) is used as the smallest positive-control instance because DTLZ6(2,3) is a reproducible limitation of the present selected-point protocol rather than a successful grouping case.

## Reproduction

```bash
python figures/dtlz_paper/network_grid/generate_dtlz5_network_source_data.py
python figures/dtlz_paper/network_grid/generate_dtlz6_network_source_data.py
python figures/dtlz_paper/network_grid/make_dtlz_network_grid.py
```

Each figure is exported as SVG, PDF, 300-dpi PNG, and 600-dpi LZW-compressed TIFF. SVG text remains editable and PDF text uses TrueType embedding.
