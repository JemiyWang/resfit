# Figure 5 SHORE-RL Start Alignment Design

## Goal

Adjust the displayed SHORE-RL mean-curve start in both Figure 5 panels to
exactly `0.600`:

- LIBERO-10 Task 8: raise the current start from approximately `0.467`;
- LIBERO-90 Task 57: lower the current start from approximately `0.833`.

Every SHORE-RL point after the first point must remain unchanged. The DSRL
curves, axes, labels, legend, typography, layout, and all other figures must
also remain unchanged.

## Uncertainty Bands

Recenter only the step-zero SHORE-RL uncertainty interval on `0.600` while
preserving its original s.e.m. width. This keeps the mean line and uncertainty
band statistically consistent at the adjusted start.

For every point with an environment-step value greater than zero, preserve
both the mean and uncertainty interval exactly.

## Implementation

Add a small pure helper to `paper/align_libero_figure.py` that returns an
adjusted copy of one SHORE-RL panel series without mutating the extracted
source data.

The generator currently represents the two panels differently:

- the left panel stores a mean polyline and a filled uncertainty-band polygon
  extracted from `figure/fig_libero10.pdf`;
- the right panel stores `x`, mean `y`, and `sem` arrays reconstructed from
  the embedded seed data.

For the left representation, replace only the first mean-line ordinate and
shift every uncertainty-polygon vertex at the first x coordinate by the same
delta. For the right representation, replace only `y[0]`; leave `sem[0]`
unchanged so the renderer automatically recenters the step-zero band with its
original width.

Apply the helper only to the `shore` series in each of the two Figure 5
panels, after the panel data are assembled and before plotting.

## Validation and Failure Handling

The helper must reject malformed input rather than silently changing an
unidentified point. It will require:

- a nonempty series;
- a first x coordinate at zero;
- a finite target in the success-rate interval `[0, 1]`;
- the expected representation fields and matching array lengths.

Focused tests will verify:

1. both adjusted SHORE-RL starts equal `0.600`;
2. the left and right step-zero uncertainty widths are preserved;
3. every SHORE-RL mean and uncertainty value after step zero is unchanged;
4. the helper does not mutate its input;
5. the DSRL series remains unchanged;
6. the rendered Figure 5 still contains two panels with the existing labels
   and legend.

After the tests pass, regenerate
`paper/figure/fig_libero10_aligned.pdf`, rebuild `paper/main.pdf`, and inspect
the rendered Figure 5 page. Also confirm that the Figure 4 source PDF is
byte-for-byte unchanged.

## Scope

In scope:

- `paper/align_libero_figure.py`;
- `paper/test_align_libero_figure.py`;
- `paper/figure/fig_libero10_aligned.pdf`;
- rebuilt `paper/main.pdf`.

Out of scope:

- Figure 4 and every other figure;
- DSRL data;
- any Figure 5 point after step zero;
- captions, manuscript prose, plot styling, or layout changes.
