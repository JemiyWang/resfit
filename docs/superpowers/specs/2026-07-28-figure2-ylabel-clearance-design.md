# Figure 2 Y-Axis Label Clearance Design

## Goal

Prevent the right-hand curve panel's y-axis title, `Eval success rate`, from
overlapping either the task-image panel on its left or the numeric y-axis tick
labels on its right.

## Current geometry

At 180 dpi with `wspace=0.16`, the space between the image panel and the
leftmost y-axis tick text is about 42 px, which is also the rendered width of
the vertical y-axis title. There is therefore no position that leaves clearance
on both sides without changing the panel spacing.

## Design

Change only the main two-panel grid spacing from `wspace=0.16` to
`wspace=0.18`. Keep the existing y-axis title font, default title-to-tick
padding, equal panel-width ratios, image grid, curve styling, data, colors, and
legend placement unchanged.

The measured result at 180 dpi is approximately 6 px between the image panel
and the y-axis title and 10 px between the title and the numeric tick labels.
The two panels become about 0.8% narrower, which is not visually material.

## Verification

- Add a source regression test requiring `wspace=0.18`.
- Run the Figure 2 plotting tests.
- Regenerate `paper/figure/fig_vanilla_collapse.pdf`.
- Render and visually inspect both the standalone figure and page 3 of
  `paper/main.pdf`.
- Recompile `paper/main.tex` twice and check the log for LaTeX errors or
  undefined references.

