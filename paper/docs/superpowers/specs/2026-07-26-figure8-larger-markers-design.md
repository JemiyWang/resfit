# Figure 8 Larger-Markers Design

## Goal

Make the circular and square data-point markers in Figure 8 more legible at
the manuscript's current `0.70\textwidth` display size.

## Change

In `plot_real_robot_curves.py`, increase the shared Matplotlib marker size from
`5.8` to `7.5` points and the marker-edge width from `1.25` to `1.4` points.

The larger markers apply equally to:

- the hollow green circles for SHORE-RL;
- the hollow blue squares for ResFit.

The marker shapes, white fill, colors, lines, data, axes, fonts, legend, canvas
dimensions, manuscript insertion width, and caption remain unchanged.

## Alternatives Considered

- `7.0` points would be a subtle change and may remain too small after the
  figure is reduced to `0.70\textwidth`.
- `8.5` points would strongly emphasize the nodes but would compete visually
  with the curves and may crowd adjacent points.
- `7.5` points provides a clear increase without changing the figure's visual
  hierarchy.

## Generated Artifacts

Regenerate:

- `figure/fig_real_robot_curves.pdf`;
- `figure/fig_real_robot_curves.png`;
- `main.pdf`.

## Verification

Confirm that all 36 plotted nodes use the larger hollow markers, circle and
square identities remain distinct, markers do not overlap labels or clip at
the axes, the manuscript compiles, existing plotting tests pass, and the
rendered Figure 8 remains balanced at `0.70\textwidth`.

The change remains local and is not pushed to GitHub.
