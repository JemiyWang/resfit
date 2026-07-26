# Framework Evaluation Curves SVG Design

## Goal

Replace the two small raster-like evaluation plots in the upper-left
`Shortening Horizon` panel of `paper/figure/framework.png` with two
publication-ready, independently editable SVG plots.

## Outputs

- `paper/figure/framework_curve_failure.svg`
- `paper/figure/framework_curve_success.svg`

Each SVG will use the same canvas dimensions and plotting geometry so either
plot can be positioned or scaled without changing its internal alignment.

## Visual Design

- Preserve the original semantic colors:
  - failure plot: pale pink line and matching confidence band;
  - success plot: pale green line and matching confidence band.
- Use a white background.
- Keep only the left and bottom axis spines.
- Draw subtle horizontal grid lines behind the data.
- Use a 1.4 px axis stroke and a 2.2 px curve stroke.
- Render the confidence interval at approximately 20% opacity.
- Use a Times New Roman-compatible serif stack.
- Use 12 px tick labels and 14 px axis labels.
- Do not add titles, legends, point markers, or annotations.

## Axes and Labels

- Horizontal axis: `Env steps (k)`, ranging from 0 to 500, with ticks every
  100 steps.
- Vertical axis: `Eval success rate`, ranging from 0.0 to 1.0, with ticks at
  0.00, 0.25, 0.50, 0.75, and 1.00.

## Curve Shape

The source figure does not contain accessible numerical curve data, so the
SVG paths will faithfully approximate the visible trends rather than claim
exact experimental samples.

- Failure curve: begins near 0.25, declines toward roughly 0.10, and retains
  restrained local variation.
- Success curve: begins near 0.70, rises toward roughly 0.95, and retains
  restrained local variation.
- Both main paths will be visually smoother and more legible than the source
  thumbnail while preserving its qualitative behavior.

## Validation

- Confirm both files are well-formed XML and use SVG `viewBox` geometry.
- Rasterize previews and visually inspect axis alignment, clipping, font
  hierarchy, colors, curve trend, and confidence-band opacity.
- Confirm both SVGs remain readable at the approximate size occupied by the
  original plots in `framework.png`.
