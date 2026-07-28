# Figure 2 Balanced Layout Design

## Goal

Make Figure 2 more balanced by slightly shrinking both equal-width panels,
increasing the space around the right panel's y-axis title, and returning the
two-entry horizon legend to the curve plot.

## Design

- Increase the main two-panel GridSpec spacing from `wspace=0.18` to
  `wspace=0.20`.
- Keep `width_ratios=(2.66, 2.66)` so the image and curve panels remain equal
  in width and shrink symmetrically.
- Keep `Eval success rate` at 17 pt in DejaVu Serif with its current default
  padding.
- Move the `Long horizon` and `Short horizon` legend into the right side of
  the curve axes using `loc="center right"`.
- Keep the legend at 14 pt, frameless, and arranged vertically in one column.
- Remove the below-axes legend anchor because the legend no longer sits
  outside the plot.
- Preserve all task images, panel tags, curve data, colors, line widths,
  uncertainty bands, tick styling, and axis limits.

At 180 dpi, `wspace=0.20` gives approximately 21 px between the task-image
panel and the vertical y-axis title and approximately 10 px between the title
and the numeric y-axis tick labels.

## Verification

- Update source-level regression assertions for `wspace=0.20` and the internal
  one-column legend.
- Verify the updated assertions fail before changing the plotting script.
- Run all Figure 2 plotting and paper-integration tests after the change.
- Regenerate `paper/figure/fig_vanilla_collapse.pdf`.
- Inspect the standalone figure at 180 dpi for balanced panel sizing, clear
  y-axis-title spacing, and an unobstructive internal legend.
- Compile `paper/main.tex` twice and inspect Figure 2 on page 3.
- Confirm the LaTeX log contains no errors or undefined references.

