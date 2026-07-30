# Figure 6 Full-Border Design

## Goal

Add top and right axis borders to both panels of Figure 6.

## Change

In `plot_staged_vs_pothiql_threading_piece_v2.py`, change the Matplotlib
configuration for `axes.spines.top` and `axes.spines.right` from `False` to
`True`.

The restored borders will inherit the existing `axes.edgecolor` value
`#52514e` and `axes.linewidth` value `0.8`. The data, curves, uncertainty
bands, grid, legend, labels, figure dimensions, and manuscript caption remain
unchanged.

## Generated Artifacts

Regenerate:

- `figure/fig_staged_vs_pothiql_threading_piece_v2.pdf`
- `/tmp/fig_staged_vs_pothiql_threading_piece_v2.png`

Then rebuild `main.pdf`.

## Verification

Confirm that the plotting script completes, both Figure 6 panels show all four
axis borders with consistent color and width, the plot remains legible without
clipping, and the main manuscript compiles without fatal or undefined-reference
errors.
