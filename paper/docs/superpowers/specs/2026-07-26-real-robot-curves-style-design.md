# Real-Robot Learning-Curve Style Design

## Goal

Redraw `figure/fig_real_robot_curves.pdf` so that its task names match the
Figure 7 caption and its visual language matches
`figure/fig_staged_vs_pothiql_threading_piece_v2.pdf`.

## Scope

- Preserve all six training checkpoints and every plotted success-rate value.
- Change only presentation, ordering, labels, and the corresponding main-paper
  caption.
- Do not alter the real-robot task figure, experiment results, or evaluation
  protocol.

## Task Panels

Use three panels in the same order as Figure 7:

1. Paper-Roll Placement
2. Block Assembly
3. Cup Stacking

## Visual Encoding

- SHORE-RL: green solid line, matching the reference figure.
- ResFit: blue dash-dot line, matching the reference figure's second method.
- Remove point markers.
- Use a two-entry shared legend centered below the panels.
- Do not encode task identity by color; panel titles identify tasks.

## Axes and Layout

- Use the reference figure's DejaVu Serif typography, muted axes, horizontal
  grid lines, hidden top/right spines, and line widths.
- Use `Env steps (k)` on each x-axis, spanning 0 to 500 with 100k intervals.
- Convert stored percentage values to success-rate fractions.
- Use `Eval success rate` on the shared y-axis, spanning 0.0 to 0.5 with 0.1
  intervals.
- Use a three-panel landscape layout with proportions derived from the
  reference figure.

## Paper Caption

Update the main-paper caption to describe success across the six checkpoints
and identify SHORE-RL as green solid and ResFit as blue dash-dot. Remove the
old statement that colors identify tasks and remove the reference to a
six-entry right-side legend.

## Outputs and Verification

- Regenerate both `figure/fig_real_robot_curves.pdf` and
  `figure/fig_real_robot_curves.png`.
- Confirm PDF text contains all three task names, both method labels, and the
  revised axis labels.
- Confirm the plotted numeric data are unchanged apart from percentage-to-
  fraction display conversion.
- Recompile `main.tex` and check for LaTeX errors, undefined citations, and
  undefined references.
