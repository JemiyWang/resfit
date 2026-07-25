# Figure 3 ResFit Legend Design

## Goal

Replace the visible `Residual RL` name in main-paper Figure 3 with `ResFit`
and make the Figure 3 caption use the same name.

## Scope

- Update the display label and descriptive text in
  `paper/plot_pouring_lifttray_seeds.py`.
- Update the Figure 3 caption in `paper/main.tex`.
- Regenerate
  `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`.
- Preserve the internal `base` key, W&B run lists, curve data, uncertainty
  bands, colors, line styles, frozen-base references, panel order, axes, and
  all other legend entries.

## Design

The `base` series remains the same ResFit baseline and retains its existing
gray dashed styling. Only its user-facing `label` and `short` strings change
from `Residual RL` to `ResFit`. The script's top-level description will use
`ResFit` as well.

The Figure 3 caption will say that SHORE-RL is compared with ResFit, DSRL,
IBRL, and full-policy IQL, without the obsolete parenthetical explaining that
ResFit is labeled `Residual RL`.

The existing Figure 3 regression test will require the shared legend to contain
`ResFit` and not contain `Residual RL`.

## Verification

1. Observe the updated regression test fail against the current plotting code.
2. Update the plotting strings and observe the test pass.
3. Regenerate the exact PDF included by `paper/main.tex`.
4. Confirm extracted PDF text contains `ResFit` and no `Residual RL`.
5. Visually inspect the standalone figure and the compiled paper page.
6. Compile `paper/main.tex` twice and confirm successful PDF generation.
