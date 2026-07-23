# Figure 4 Font Alignment Design

## Goal

Make the typography in Figure 4 visually match the current Figure 3 after both
figures are scaled to `\textwidth` in the paper.

## Scale-aware font mapping

The generated Figure 3 PDF is 1170.79 pt wide and Figure 4 is 970.48 pt wide.
Because both are inserted at the same paper width, Figure 4's source font sizes
must be multiplied by `970.48 / 1170.79 ≈ 0.829` relative to Figure 3.

The Figure 4 source settings will therefore be:

- task titles: 15 pt, normal weight
- x-axis label: 15 pt
- tick labels: 13 pt
- y-axis label: 16 pt
- legend: 13 pt

## Scope

Only the explicit typography settings in `paper/plot_ablation_curves.py` will
change. Plot data, curves, colors, line styles, panel dimensions, spacing,
legend contents, captions, and paper layout will remain unchanged.

## Verification

Regenerate `paper/figure/fig_ablation_curves.pdf`, rebuild `paper/main.pdf`,
and visually compare Figure 4 on the paper page with Figure 3. Confirm that the
paper remains portrait and that the build completes successfully.
