# Figure 5 Main-Paper Integration Design

## Goal

Replace the learning-curve version of Figure 5 in `paper/main.tex` with the
fixed-budget 400k bar chart, and align the surrounding main-paper text with the
new statistical view.

## Scope

- Modify only `paper/main.tex` for paper integration.
- Include `figure/fig_ablation_bars_400k.pdf` at `\textwidth`.
- Keep the original curve asset and plotting script available; do not delete them.
- Do not modify the supplementary manuscript in this pass.

## Text Alignment

- Define final-window success relative to the stated comparison budget.
- State that Ablation A uses a common 400k cutoff, eight checkpoints from
  330k through 400k, and 2--3 seeds per arm.
- Rename the internal figure label from `fig:ablcurves` to `fig:ablbars`.
- Rewrite the component-ablation paragraph around the bar values and avoid
  curve-specific language such as endpoints or late-run markers.
- Rewrite the caption to define per-seed averaging, cross-seed means, and
  `\pm 1` s.e.m. error bars.

## Result Description

The main text will report the strongest task-level patterns: SHORE-RL is best
on ThreePieceAssembly, while the no-waypoint arm remains close on Pouring.

## Verification

- Compile `paper/main.tex` twice with `pdflatex -halt-on-error`.
- Require no undefined references to either ablation figure label.
- Inspect the page containing Figure 5 for legibility and float placement.
- Confirm the old curve path and curve-specific caption text are absent.
