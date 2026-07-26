# Figure 4 and Pages 6–7 Flow Layout Design

## Goal

Compress Figure 4 from a two-column figure to a one-column figure and let
Figures 4–6 and their surrounding text flow naturally in source order. The
layout must remain robust when prose length changes.

## Layout Rules

- Change Figure 4 from `figure*` to the ordinary single-column `figure`
  environment.
- Size Figure 4 to `\columnwidth`.
- Keep Figures 4, 5, and 6 next to their existing logical discussion points
  and preserve their source and numbering order.
- Use normal LaTeX float placement rather than absolute or page-specific
  positioning.
- Do not add `[H]`, forced page breaks, manual vertical offsets, or placeholder
  space.
- Remove the float barrier between the ablation material and the real-world
  extension so that the two-column page builder can fill columns naturally.

## Expected Result

With the current prose, Figure 4 should occupy a single column near the start
of page 6 and the material currently concentrated in the left column should
redistribute across pages 6 and 7. If prose changes later, figures may move to
another column or page as normal floats, without requiring manual layout edits.

## Verification

- Compile the paper from a clean enough LaTeX state to refresh float placement.
- Confirm that Figure 4 is one column wide.
- Confirm that Figures 4, 5, and 6 remain in numeric and reading order.
- Inspect rendered pages 6 and 7 for large avoidable gaps, collisions, clipped
  content, or captions separated from their figures.
- Check the LaTeX log for new float, overfull-box, and reference warnings.
