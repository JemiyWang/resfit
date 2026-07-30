# Figure 4 Full-Border Design

## Goal

Add top and right axis borders to both panels of Figure 4.

## Change

Extend `align_libero_figure.py`, the existing vector postprocessor for Figure 4,
to draw the missing top and right borders on both panels of
`figure/fig_libero10_aligned.pdf`.

The added borders will use the existing axis edge color `#52514e`, line width
`0.8`, and the exact endpoints of the current left and bottom borders. This
preserves the current two-panel source figure and avoids relying on the stale
one-panel configuration in `plot_libero10.py`.

The plotted data, curves, uncertainty bands, grid, legend, labels, panel
dimensions, manuscript layout, and caption remain unchanged.

## Generated Artifacts

Regenerate:

- `figure/fig_libero10_aligned.pdf`
- a temporary raster preview for visual inspection

Then rebuild `main.pdf`.

## Verification

Confirm that the postprocessor completes, both Figure 4 panels show all four
axis borders with consistent color and width, no border overlaps or clips plot
content, the legend and layout remain unchanged, and the main manuscript
compiles without fatal or undefined-reference errors.
