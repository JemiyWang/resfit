# Figure 5 400k Final-Window Bar Chart Design

## Goal

Create a first bar-chart version of the component-ablation Figure 5 without
overwriting the current learning-curve figure.

## Metric

- Use a fixed training budget of 400k environment steps.
- For each seed, average `eval/success_rate` over checkpoints satisfying
  `320k < step <= 400k`. With the current 10k evaluation interval, this is the
  eight checkpoints from 330k through 400k.
- Use the mean of those seed-level values as the bar height.
- Show `±1 s.e.m.` across seed-level values when at least two seeds are
  available. The single-seed LiftTray `w/o waypoint` bar has no error bar.
- Do not extrapolate or substitute missing checkpoints. Fail generation if any
  selected seed lacks the complete eight-checkpoint window.

## Layout

- Preserve the current three-panel task order: Pouring, LiftTray, ThreePiece.
- Draw five bars per panel in the existing legend order and colors.
- Share a success-rate y-axis spanning 0 to 1.
- Use one figure-level legend and typography compatible with the current paper.
- Keep the plot readable at the existing full-width Figure 5 size.

## Files

- Add a dedicated plotting script:
  `paper/plot_ablation_bars_400k.py`.
- Generate:
  `paper/figure/fig_ablation_bars_400k.pdf`.
- Generate a review preview:
  `paper/figure/fig_ablation_bars_400k.png`.
- Do not modify `paper/main.tex` or overwrite
  `paper/figure/fig_ablation_curves.pdf` in this first version.

## Verification

- Unit-test the fixed-window selection, per-seed aggregation, cross-seed
  mean/s.e.m., and incomplete-window rejection without contacting W&B.
- Query the existing Figure 5 run inventory and require all 40 selected seeds
  to contain the complete 330k--400k window.
- Inspect the PDF metadata and embedded fonts.
- Render and visually inspect the generated chart.
