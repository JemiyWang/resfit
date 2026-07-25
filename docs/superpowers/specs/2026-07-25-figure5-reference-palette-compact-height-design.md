# Figure 5 Reference Palette and Compact Height Design

## Goal

Restyle the existing single-column Figure 5 to match the muted palette of
`paper/figure/参考.png` and shorten the bar-chart region without changing data,
statistics, labels, ordering, or paper placement.

## Palette

Use colors sampled from the reference image and map its yellow `Ours` emphasis
to the full method:

- `SHORE-RL`: yellow `#FAD35B`
- `w/o demo-BC + stage`: green `#539955`
- `w/o stage shaping`: gray `#ABACAB`
- `w/o waypoint`: salmon `#CC8675`
- `w/o waypoint + stage`: blue `#80AECA`

The legend patches must use exactly the same mapping and retain the current
row-major visual order. Bar outlines and error bars use dark gray to match the
reference figure.

## Geometry

- Keep the one-column width at 3.35 inches.
- Reduce the figure height from 2.75 inches to 2.30 inches.
- Reallocate the bottom margin so the three-row, two-column 7 pt legend remains
  unclipped while the plotted axes become approximately 20 percent shorter.
- Preserve the 0--1 y-axis, task groups, five-bar ordering, and 7 pt type.

## Verification

- Add tests for the exact five-color semantic mapping, legend/color identity,
  and 3.35 by 2.30 inch figure size.
- Regenerate PDF and PNG outputs from the unchanged 28-seed summaries.
- Visually inspect both the standalone PNG and the Figure 5 page in `main.pdf`.
- Compile `main.tex` and verify that references remain defined.
