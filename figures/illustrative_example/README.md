# Illustrative objective graph

This graph replaces the displayed adjacency matrix in the illustrative example.
It preserves all six reported off-diagonal values and the fixed-K partition
`{f1, f2} | {f3} | {f4}`. Diagonal values of one remain in the source CSV;
the graph omits self-loops. No new sampling or optimization was performed.

## Files and insertion

- `illustrative_objective_graph_square_v3.pdf`: current vector figure, 89 x 89 mm, embedded fonts.
- `illustrative_objective_graph_square_v3.svg`: current editable vector copy.
- `illustrative_objective_graph_square_v3.png`: current 600 dpi preview.
- `replacement.tex`: figure environment, caption, and replacement lead-in sentence.
- `illustrative_correlation_matrix.csv`: the manuscript's three-decimal matrix.
- `make_illustrative_objective_graph.py`: reproducible Python/Matplotlib drawing.

Copy the PDF beside the current main LaTeX file. Replace the sentence
"The resulting correlation strength matrix is" and its displayed equation with
`replacement.tex`. Keep the existing following paragraph interpreting the groups.
LaTeX will number this figure and the subsequent figures automatically.

The uploaded manuscript has 27 pages and contains changes absent from the local
25-page manuscript. This delivery therefore provides an insertion block for the
latest source without changing the older manuscript.

## Visual mapping and source check

The four nodes form an axis-aligned square: f1/f2 on top and f3/f4 below.
Blue identifies the f1/f2 group, orange identifies f3, and green identifies f4.
Version 3 doubles each node radius relative to version 2, using four times the
marker area (5000 instead of 1250 square points). Axis limits expand slightly
to keep the enlarged node outlines inside the drawing area.
Labels retain three decimals, including trailing zeros. White text on blue and
green and black text on orange maintain contrast. Line width is `0.45 + 2.6 * A_ij` points, so even the weakest
edges remain visible. The two center-crossing edge labels move along their own
edges to avoid overlap. Node positions have no numerical meaning.

The archived full-precision matrix from the 68 selected points rounds to the CSV
exactly. At K=3, average linkage makes one merge: f1/f2 have the unique largest
off-diagonal affinity, 0.767. The other two objectives remain singleton groups.

Reproduce with Python, NumPy, Matplotlib, and NetworkX:

```bash
python make_illustrative_objective_graph.py
```

## Validation

| Check | Result |
|---|---|
| Source matrix | All 16 values equal the archived matrix rounded to three decimals |
| Graph | Four nodes, all six off-diagonal edges, no self-loops |
| Grouping | Unique first average-linkage merge f1/f2, giving three groups |
| PDF | One page, 89 x 89 mm, all fonts embedded |
| Numerical labels | Each of the six exact values occurs once in extracted PDF text |
| SVG | Ten editable text elements |
| PNG | 2102 x 2102 pixels, 600 dpi |
| Visual review | Final PDF rendered and inspected; no overlap or clipping |
| Caption style | All 15 mechanical scan categories return zero hits, including dashes, rhetorical flourishes, and passive voice |
