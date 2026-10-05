# Size-adjusted original-code network figures

This revision keeps the current ORCA affinity matrices and recovered groups,
then draws every standalone network with the earlier NetworkX/Matplotlib
workflow before assembling the five rendered PNG panels.

## Cases

- DTLZ5: (2,4), (3,5), (4,7), (6,10), (7,12)
- DTLZ6: (2,4), (3,5), (4,7), (6,10), (7,12)

DTLZ5(2,4) was checked over seeds 0–4: all five runs recovered
`{f1,f2,f3} | {f4}` exactly, with ARI = 1.

## Requested display changes

- Each left-hand panel is 43.5 mm square rather than 39.5 mm square.
- The four left-hand networks use a larger plotting area.
- Left-hand node labels use 17 pt and edge labels use 13 pt in each 12-inch
  standalone render before the rendered panels are scaled into the composite.
- (2,4) and (3,5) use `node_size=13000`.
- (4,7) and (6,10) retain `node_size=10000`.
- (7,12) uses `node_size=7500` in both DTLZ figures.
- All edge values remain at the default NetworkX midpoint and use the earlier
  `str(value)[0:5]` display rule.

The subsequent `networkx_original_code_adjusted_ab_scaled` revision enlarges
panel a node radii and graph text by 2x, and panel b node radii and graph text
by 1.5x, while preserving this revision.

## Export

Individual SVG/PDF files retain vector networks and editable text.  The
combined SVG/PDF files contain five embedded rendered panels because the
requested combination is a literal image assembly.  Combined TIFF files are
exported at 600 dpi with LZW compression.
