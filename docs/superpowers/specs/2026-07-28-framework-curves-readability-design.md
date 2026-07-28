# Framework Curve Readability Design

## Goal

Make the two evaluation curves in Figure 1 readable after they are scaled into
the small upper-left framework panel, while retaining the current curve data,
colors, confidence bands, canvas size, and editable SVG text.

## Output and preservation

- Generate `paper/figure/framework_curve_failure_readable.svg`.
- Generate `paper/figure/framework_curve_success_readable.svg`.
- Do not overwrite or modify `framework_curve_failure.svg` or
  `framework_curve_success.svg`.
- Keep the existing `render_curve(...)` interface so tests and preview tooling
  can render the same style to temporary paths.

## Visual design

- Keep the 4.5 by 2.0 inch canvas and existing subplot aspect ratio so the new
  files can replace the originals in the framework layout without resizing.
- Increase tick labels from 12 pt to 20 pt.
- Increase axis labels from 14 pt to 24 pt.
- Reduce the y-axis ticks from five to three: `0.0`, `0.5`, and `1.0`.
- Keep the x-axis ticks at 0, 100, 200, 300, 400, and 500.
- Reduce left/bottom spine and tick widths from 1.4 pt to 0.8 pt.
- Reduce tick length from 4 pt to 3 pt and horizontal grid width from 0.8 pt to
  0.6 pt.
- Keep the 2.2 pt mean curves, 20% confidence bands, serif typeface, white
  background, and the existing pink/green colors.
- Adjust subplot margins only as needed to prevent the larger text from being
  clipped.

## Validation

- Add a regression test that calls `main()` with a temporary output directory
  and confirms that only the two `_readable.svg` names are generated.
- Verify the generated SVGs are valid XML, retain editable `<text>` nodes, and
  contain the intended labels, font sizes, colors, opacity, and thin stroke
  widths.
- Render PNG previews and verify that no text is clipped and both plots retain
  identical geometry.
- Compare hashes of the two original SVGs before and after generation to prove
  they were not modified.

## Scope

This trial changes only the shared plot generator, its focused regression
tests, and the two newly named SVG assets. It does not replace the plots inside
`framework.pdf` or change the LaTeX figure reference.
