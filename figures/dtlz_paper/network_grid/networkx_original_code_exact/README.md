# Current ORCA results drawn with the original plotting code

## What is unchanged

- Affinity matrices and recovered groups are read directly from
  `source_data/networkx_all_labels`.
- No algorithm result is recomputed or altered by the plotting script.
- The requested DTLZ5 and DTLZ6 case lists are unchanged.

## What follows the original plot

- `networkx.Graph` and `networkx.circular_layout`
- `node_size=10000`
- black node outlines and default NetworkX edges
- node-label font size 14
- edge-label font size 11
- every pairwise edge label at NetworkX's default midpoint
- the original `str(value)[0:5]` display rule
- the original color sequence, extended by one color for the seventh group

Node colors use the current algorithm's recovered partition rather than a
manually specified expected partition.

## Assembly method

The ten square individual figures are rendered first.  Each DTLZ composite is
then assembled by reading the five finished PNG files and placing them into the
four-small-plus-one-large layout.  The networks are not redrawn in the
composite.

Individual SVG and PDF files retain vector lines and editable text.  Because
the combined figures are literal image assemblies, their SVG and PDF versions
contain embedded raster panels; use the 600-dpi TIFF for manuscript submission
or the individual SVG/PDF files when fully editable network elements are
required.

The later size-adjusted revision is exported separately under
`networkx_original_code_adjusted`; this directory preserves the initial exact
style reconstruction.
