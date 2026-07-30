# Table 3 Real-Robot Results Design

## Goal

Replace the unresolved real-robot Table 3 placeholder with values that are
consistent with the learning-curve data and make the frozen-base reference,
ResFit baseline, and SHORE-RL result directly comparable.

## Scope

Only the real-robot evaluation paragraph, Table 3, and its caption in
`main.tex` will change. The plotting data, generated figures, supplementary
tables, and statistical protocol will remain unchanged.

## Result Definition

- The frozen-base column reports the step-0 evaluation, when the deterministic
  residual is zero.
- ResFit and SHORE-RL report final-window success: the mean of the 400k and
  500k evaluations, corresponding to the final 20% of the 500k-step budget.
- Values are reported as success-rate proportions, matching the other compact
  result tables in the main paper.

The resulting cells are:

| Task | Frozen Base | ResFit | SHORE-RL |
|---|---:|---:|---:|
| Paper-Roll Placement | .22 | .28 | **.43** |
| Block Assembly | .10 | .00 | **.29** |
| Cup Stacking | .12 | .02 | **.31** |

## Prose and Caption

The evaluation paragraph will describe all three configurations. The caption
will explicitly distinguish the step-0 frozen-base reference from the
final-window adapted-method results and retain the statement that every
checkpoint uses 50 physical trials from the fixed initial-state set.

The unresolved request for `mean ± s.e.` will be removed because the plotting
source contains only aggregate checkpoint rates, not seed-level or paired
trial-level records from which the paper's requested uncertainty statistics
could be reconstructed.

## Verification

After editing, the manuscript will be checked for:

1. Removal of the Table 3 `P1` placeholders.
2. Agreement between Table 3 and `plot_real_robot_curves.py`.
3. Successful BibTeX and multi-pass pdfLaTeX compilation.
4. Presence of the three methods and all nine values in the compiled PDF.
