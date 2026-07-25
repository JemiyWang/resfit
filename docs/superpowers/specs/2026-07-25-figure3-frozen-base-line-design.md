# Figure 3 Frozen-Base Reference Line Design

## Goal

Add a horizontal frozen-base reference to every panel of the main-paper
Figure 3. Each reference height is the corresponding task's green SHORE-RL
curve at zero environment steps, averaged over the SHORE-RL seeds available at
that point.

## Scope

- Update `paper/plot_pouring_lifttray_seeds.py`.
- Regenerate
  `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`,
  which is the asset included by `paper/main.tex`.
- Update the Figure 3 caption in `paper/main.tex`.
- Preserve all curve data, method colors, method line styles, uncertainty
  bands, panel order, axis ranges, titles, and existing legend entries.

## Design

Before plotting, compute one frozen-base value per panel from only the
SHORE-RL (`ours`) runs:

1. Select every SHORE-RL seed with an evaluation at snapped step zero.
2. Average those step-zero success rates.
3. Draw a thin gray dashed horizontal line across that task's full panel at
   the resulting value.

The reference line is rendered below the learning curves so it does not cover
their markers or means. A dedicated `Frozen base` sample is appended to the
shared legend. The existing `Residual RL` curve remains unchanged; its thicker
line and existing dash pattern distinguish it from the thinner reference.

If a panel has no SHORE-RL step-zero evaluation, the plotting script skips the
reference for that panel and reports that the value is unavailable instead of
inventing or borrowing a height.

The caption will state that the gray dashed references mark the per-task
SHORE-RL success at zero environment steps.

## Verification

1. Run the plotting script with its output directed to the exact Figure 3
   asset used by `paper/main.tex`.
2. Confirm that the script reports a finite reference value for all five
   panels.
3. Convert the regenerated PDF to a raster preview and visually inspect that
   every panel has one horizontal reference, the lines start at the green
   curve heights, and the six-entry legend is readable.
4. Compile `paper/main.tex` and inspect the paper page containing Figure 3 for
   clipping, overlap, or caption/layout regressions.
