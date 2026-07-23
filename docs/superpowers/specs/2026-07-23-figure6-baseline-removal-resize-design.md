# Figure 6 Baseline Removal and Resize Design

## Goal

Remove the frozen-base visual reference from Figure 6 and make the complete
figure slightly smaller in the paper.

## Plot changes

Delete both elements that represent the frozen $\pi_0$ BC reference:

- the horizontal dashed line at success rate 0.76
- the `π₀ BC 0.76 (query-20)` text annotation

All remaining curves, uncertainty bands, axes, labels, fonts, limits, and
legend contents will remain unchanged.

## Paper layout

Change the Figure 6 inclusion width in `paper/main.tex` from `\columnwidth` to
`0.90\columnwidth`. Keep the figure centered and retain the existing
single-column float.

Update the Figure 6 caption to remove the claim that a frozen-base reference is
shown. The surrounding paragraph that discusses the query-20 base as
experimental context will remain unchanged.

## Verification

Regenerate `paper/figure/fig_libero10.pdf`, confirm that the baseline line and
annotation are absent, rebuild `paper/main.pdf`, and inspect the rendered paper
page. Confirm that Figure 6 is visibly but modestly smaller, remains centered,
and that the paper stays portrait.
