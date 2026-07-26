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
- Keep Figure 4 adjacent to the base-generality discussion, Figure 5 adjacent
  to the matched component-ablation discussion, and Figure 6 adjacent to the
  stage-bonus-versus-potential discussion.
- Use normal LaTeX float placement rather than absolute or page-specific
  positioning.
- Do not add `[H]`, forced page breaks, manual vertical offsets, or placeholder
  space.
- Allow top, bottom, and float-page placement so the two-column page builder
  has enough freedom to fill columns naturally.
- Keep a semantic float barrier before the ablation subsection so the main
  results figures cannot drift into unrelated ablation prose.
- Keep the semantic float barrier before the real-world extension so Figures
  5 and 6 cannot drift into an unrelated subsection. The barrier is a section
  boundary, not a page-position lock.
- Keep both the real-robot task overview and learning curves full width. Declare
  the two `figure*` floats consecutively at the start of the real-world
  subsection so LaTeX can stack them in source order at the next page top, near
  the task and evaluation text, without forcing a sparse page.

## Expected Result

With the current prose, Figure 4 should occupy a single column immediately after
the main simulation figures, while the ablation and real-world material should
redistribute across pages 6 and 7. If prose changes later, figures may move to
another column or page as normal floats, without requiring manual layout edits.

## Verification

- Compile the paper from a clean enough LaTeX state to refresh float placement.
- Confirm that Figure 4 is one column wide.
- Confirm that Figures 4, 5, and 6 remain in numeric and reading order.
- Confirm that each figure remains next to the paragraph that interprets it
  and does not cross into an unrelated subsection.
- Confirm that pages 6 and 7 use both columns without making any plot or task
  snapshot illegibly small.
- Inspect rendered pages 6 and 7 for large avoidable gaps, collisions, clipped
  content, or captions separated from their figures.
- Check the LaTeX log for new float, overfull-box, and reference warnings.
