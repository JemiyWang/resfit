# Figure 4 Match-Figure-6 Design

## Goal

Make Figure 4 match Figure 6 in physical size and plotting style while
preserving Figure 4's current data, two-panel content, colors, labels, and
400k-step range.

## Approach

Create a dedicated local restyling script that reads the current two-panel
vector source `figure/fig_libero10.pdf`, extracts the green and purple mean
curves and their uncertainty-band polygons, maps their PDF coordinates back to
the existing plot coordinates, and redraws them with Matplotlib.

This avoids stretching text or lines and does not depend on the stale
one-panel configuration in `plot_libero10.py`. It also avoids replacing the
current Task 57 panel with newly fetched or inferred experimental data.

## Figure 6 Style Contract

The redrawn Figure 4 will use the Figure 6 settings from
`plot_staged_vs_pothiql_threading_piece_v2.py`:

- figure size `7.4 x 3.65` inches for two panels;
- DejaVu Serif with base size `17`;
- title size `16`, left aligned, normal weight, padding `6`;
- x-label size `16`;
- y-label size `17`;
- tick-label size `14`, tick width `0.8`, tick length `3.5`;
- axis color `#52514e`, width `0.8`, with all four spines visible;
- horizontal grid color `#dcdcd7`, width `0.7`;
- legend size `14`, two columns, no frame, and Figure 6 spacing;
- uncertainty-band opacity `0.15`;
- tight-layout margins and panel spacing matching Figure 6.

Figure 4 retains its existing visual identities:

- SHORE-RL: green `#008300`, solid line;
- DSRL: purple `#7a3fb5`, solid line;
- panel titles `LIBERO-10 · Task 8` and `LIBERO-90 · Task 57`;
- x range `0–400k` with 100k ticks;
- y range `-0.03–1.03` with 0.25 ticks;
- shared y axis and legend labels `SHORE-RL (Ours)` and `DSRL`.

The mean-line widths will scale to the closest Figure 6 counterparts:
SHORE-RL `2.6` and DSRL `2.3`.

## Components and Data Flow

The restyling script will:

1. open `figure/fig_libero10.pdf` with PyMuPDF;
2. identify the two existing axis rectangles;
3. select curve and band paths by their vector colors;
4. convert PDF coordinates to Figure 4 data coordinates;
5. validate that each panel has one mean path and corresponding band paths for
   both methods;
6. render the reconstructed data with the Figure 6 style contract;
7. save `figure/fig_libero10_aligned.pdf` and a temporary raster preview.

`main.tex` will continue to include the same output path at
`\columnwidth`; no caption or manuscript-layout change is required.

## Failure Handling

The script will fail with a clear assertion if the source PDF dimensions,
panel bounds, curve colors, or expected path counts differ from the known
two-panel source. It will not silently generate a partial or misidentified
figure.

## Verification

Verification will confirm:

- extracted curve endpoints and extrema match the source within plotting
  tolerance;
- both panels contain the two expected mean curves and uncertainty bands;
- the output page aspect ratio matches Figure 6 within the normal
  `bbox_inches="tight"` variation;
- Figure 4 and Figure 6 use the same typography, borders, tick, grid, and legend
  parameters;
- no unexpected diagonal or connecting vector paths are present;
- the paper compiles twice without fatal or undefined-reference errors;
- existing paper plotting tests pass;
- a rendered manuscript page shows Figure 4 visually aligned with Figure 6
  without clipping or overlap.

## Scope

The change will not alter Figure 4 results, captions, task names, curve colors,
axis limits, or other manuscript figures. It will remain local and will not be
pushed to GitHub.
