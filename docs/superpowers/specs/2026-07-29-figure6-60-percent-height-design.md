# Figure 6 60% Height Design

## Goal

Compress the complete Figure 6 bar-chart canvas to 60% of its current nominal
height while preserving all figure text, data, colors, legend content, and
paper prose.

## Selected Approach

Regenerate the vector chart from its Matplotlib source. Change
`FIGSIZE_INCHES` from `(3.35, 2.30)` to `(3.35, 1.38)`, because
`2.30 * 0.60 = 1.38`. Keep `FONT_SIZE_PT = 7` and retain the existing
`bbox_inches="tight"` export so the source width and the LaTeX scaling remain
consistent with the current figure.

The LaTeX inclusion remains at `0.85\columnwidth`; the caption, label, float
placement, and surrounding manuscript text remain unchanged.

## Files

- Modify `paper/test_plot_ablation_bars_400k.py`.
- Modify `paper/plot_ablation_bars_400k.py`.
- Regenerate `paper/figure/fig_ablation_bars_400k.pdf`.
- Regenerate `paper/figure/fig_ablation_bars_400k.png`.
- Recompile `paper/main.pdf`.

## Test-Driven Implementation

1. Change the figure-contract test to require `(3.35, 1.38)` and run it to
   observe the expected failure against the current `(3.35, 2.30)` source.
2. Change only the source height constant and rerun the targeted tests.
3. Regenerate the PDF and PNG with the existing plotting pipeline.
4. Compile the paper twice to stabilize cross-references.

## Verification

- Confirm the plotting tests pass and the source still requires 7 pt type.
- Compare old and new standalone PDF dimensions and confirm the exported height
  is approximately 60% after tight-bounding-box effects.
- Confirm `main.tex` still contains the same Figure 6 caption and inclusion
  width.
- Render and inspect the containing paper page for clipping, overlap, and text
  legibility.
- Require two successful `pdflatex -halt-on-error` passes and a clean
  `git diff --check` for the scoped files.
