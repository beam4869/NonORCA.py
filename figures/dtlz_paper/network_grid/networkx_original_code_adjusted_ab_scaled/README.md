# DTLZ5 and DTLZ6 correlation-network figures

This folder contains the updated paper figures generated from the current,
unchanged correlation matrices and recovered groups. The network panels use the
same NetworkX/Matplotlib drawing sequence as the earlier plotting code.

## Panel cases

- a: DTLZ5/6(2,4)
- b: DTLZ5/6(3,5)
- c: DTLZ5/6(4,7)
- d: DTLZ5/6(6,10)
- e: DTLZ5/6(7,12)

## Requested size changes

- Panel a: node radius 2x and node/edge-label fonts 2x. Because NetworkX
  `node_size` is marker area, the node area is 4x (`13000` to `52000`). Font
  sizes are `34` for node labels and `26` for edge labels.
- Panel b: node radius 1.5x and node/edge-label fonts 1.5x. The node area is
  therefore 2.25x (`13000` to `29250`). Font sizes are `25.5` for node labels
  and `19.5` for edge labels.
- Panels c, d, and e are unchanged from the preceding adjusted version.

The plotting limits for panels a and b include additional white margin so the
larger nodes and labels are not clipped. This does not reduce their physical
marker or font sizes.

## Files

- `individual/`: all ten square panels in SVG, PDF, PNG (300 dpi), and TIFF
  (600 dpi).
- `combined/`: the two five-panel DTLZ5 and DTLZ6 figures in SVG, PDF, PNG, and
  TIFF (600 dpi). Each combined figure places the already-rendered individual
  panels; it does not redraw the networks.

The reproducible plotting script is
`../make_figures_with_original_plot_code.py`.
