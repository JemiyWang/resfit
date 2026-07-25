# Figure 5 Single-Column Layout Design

## Goal

Reformat Figure 5 for one-column placement without changing its data, metric,
method order, colors, or uncertainty definition.

## Layout

- Replace the two side-by-side task panels with one grouped bar chart.
- Use `Pouring` and `ThreePiece` as the two x-axis groups.
- Show the five existing ablation arms as adjacent colored bars in each group.
- Keep the existing method order and color mapping.
- Put a compact two-column legend below the axes.
- Use a physical width near one AAAI column (about 3.35 inches) and a compact
  height near 2.6--2.9 inches.

## Typography and Marks

- Use approximately 8 pt for axis labels and task labels.
- Use approximately 7 pt for tick labels and legend text.
- Retain a 0--1 success-rate axis with sparse ticks.
- Scale bar width, error-bar caps, line widths, and spacing for the smaller
  physical figure rather than shrinking the old full-width rendering.
- Keep error bars at plus/minus one standard error.

## Paper Integration

- Change the LaTeX float from `figure*` to single-column `figure`.
- Include the graphic at `\columnwidth`.
- Preserve the existing Figure 5 label, caption semantics, and body references.
- Keep the caption outside the plotted PDF so it uses the document font.

## Verification

- Extend the plotting test to enforce the single-column grouped-chart geometry.
- Regenerate both PDF and PNG outputs.
- Compile `main.tex` twice and check for undefined references.
- Rasterize the page containing Figure 5 and visually inspect legibility,
  clipping, float placement, and column containment.
