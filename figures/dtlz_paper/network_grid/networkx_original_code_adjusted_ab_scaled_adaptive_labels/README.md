# Adaptive correlation-label trial

This folder contains the DTLZ5 and DTLZ6 network figures with deterministic
edge-label placement for the center-crowded chords. The source affinity
matrices, recovered groups, circular node positions, straight edges, node
sizes, and font sizes are unchanged from the preceding adjusted version.

## Label-placement rule

- Every correlation value remains attached to its original straight edge.
- DTLZ5/6(2,4): the two labels whose edges cross the exact center alternate
  between `label_pos=0.35` and `0.65`.
- DTLZ5/6(3,5) and DTLZ5/6(4,7): labels remain at their original edge
  midpoints because these odd-objective circular graphs have no exact
  diameters.
- DTLZ5/6(6,10): the five exact-diameter labels alternate between
  `label_pos=0.25` and `0.75`.
- DTLZ5/6(7,12): the three innermost chord families are distributed into
  three label bands. Their alternating positions are `0.40/0.60`,
  `0.28/0.72`, and `0.35/0.65`, respectively. This separates the labels that
  previously occupied the same central region without random jitter.
- Labels use a small semi-opaque white backing so crossing edges do not pass
  through the numbers.

## Files

- `individual/`: ten standalone panels in SVG, PDF, PNG (300 dpi), and TIFF
  (600 dpi).
- `combined/`: the DTLZ5 and DTLZ6 five-panel figures in SVG, PDF, PNG, and
  TIFF (600 dpi).

The reproducible Python/Matplotlib script is
`../make_figures_with_original_plot_code.py`.
