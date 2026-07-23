# Figure 5 Typography and Height Design

## Goal

Make Figure 5 typography visually consistent with the current Figure 3 and
slightly increase the plot height so changes in the curves are easier to see.

## Typography

Figure 5 will explicitly use `DejaVu Serif`, matching the font embedded in the
current Figure 3 PDF. Its source font sizes will be adjusted for Figure 5's
single-column inclusion scale:

- task titles: 16 pt, normal weight
- x-axis label: 16 pt
- tick labels: 14 pt
- y-axis label: 17 pt
- legend: 14 pt

## Plot height

Keep the existing two-panel horizontal layout and the current 7.4-inch source
width. Increase the source height from 3.4 inches to 3.65 inches, approximately
7%, to make vertical changes more legible without making Figure 5 visually
dominant.

## Scope

Only the explicit font configuration, font sizes, and source figure height in
`paper/plot_staged_vs_pothiql_threading_piece_v2.py` will change. Plot data,
curves, colors, line styles, axes limits, panel order, legend contents,
caption, inclusion width, and surrounding paper layout will remain unchanged.

## Verification

Regenerate `paper/figure/fig_staged_vs_pothiql_threading_piece_v2.pdf`, confirm
with `pdffonts` that Figures 3 and 5 both embed `DejaVuSerif`, rebuild
`paper/main.pdf`, and inspect the rendered paper page. Confirm that the paper
remains portrait and that the build succeeds.
