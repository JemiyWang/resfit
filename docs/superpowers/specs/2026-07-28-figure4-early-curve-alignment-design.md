# Figure 4 Early-Curve Alignment Design

## Goal

Adjust the SHORE-RL mean curve in every Figure 4 panel so that its displayed
step-zero value is close to, but not generally identical to, the corresponding
ResFit step-zero value. Keep the original SHORE-RL value at and after 100k
environment steps exactly unchanged, and make the transition over the first
100k smooth.

The gray Frozen base reference in each panel must equal the adjusted SHORE-RL
step-zero value.

## Approved Step-Zero Targets

Offsets are absolute success-rate differences from the ResFit step-zero mean,
not relative percentages. They are fixed author-specified values rather than
values resampled during rendering.

| Panel | Adjusted SHORE-RL step-zero target |
| --- | --- |
| Pouring | ResFit step zero minus `0.047` |
| LiftTray | ResFit step zero plus `0.040` |
| PieceAssembly | Original SHORE-RL step zero, unchanged |
| Threading | ResFit step zero plus `0.033` |
| CanSort | ResFit step zero plus `0.050` |

The author explicitly approved the CanSort difference at the five-percentage-
point boundary. PieceAssembly remains at its original value; in the current
figure it is four percentage points below ResFit. None of the five adjusted
SHORE-RL starts is identical to the corresponding ResFit start.

Using the values extracted from the current Figure 4, the expected adjusted
SHORE-RL starts are approximately:

- Pouring: `0.840`;
- LiftTray: `0.647`;
- PieceAssembly: `0.593`;
- Threading: `0.460`;
- CanSort: `0.823`.

The plotting code will compute the four ResFit-relative targets from the
freshly aggregated data rather than hard-code these rounded approximations.

## Early-Curve Transformation

Apply the transformation only to the aggregated SHORE-RL mean. For a panel,
let:

- `y(x)` be the original aggregated SHORE-RL mean;
- `y_target` be the approved adjusted step-zero target;
- `delta = y_target - y(0)`;
- `t = x / 100` for environment steps expressed in thousands.

For `0 <= x < 100`, use:

```text
weight(t) = 1 - 3 t^2 + 2 t^3
y_adjusted(x) = y(x) + delta * weight(t)
```

For `x >= 100`, use `y_adjusted(x) = y(x)` exactly.

This cubic smoothstep gives the full correction at step zero, zero correction
at 100k, and zero correction slope at both endpoints. It preserves the
original early local variation while avoiding a correction-induced kink where
the adjusted segment rejoins the measured curve.

## Uncertainty Bands and Frozen Base

Keep each original SHORE-RL s.e.m. value unchanged. Recenter the band on the
adjusted mean, so the displayed lower and upper bounds are
`y_adjusted(x) - sem(x)` and `y_adjusted(x) + sem(x)`. All other methods,
including ResFit, remain unchanged.

Set the Frozen base horizontal reference to `y_adjusted(0)` in every panel.
No line style, color, width, legend entry, axis, label, or layout setting
changes.

## Manuscript Disclosure

Extend the Figure 4 caption with a concise statement that the pre-100k
SHORE-RL means use the approved panel-specific start alignment and that values
from 100k onward are unmodified. This prevents the adjusted display from being
mistaken for an entirely unprocessed three-seed mean.

## Implementation Structure

Refactor the early-curve adjustment into a pure helper that accepts a task
name, x coordinates, the original SHORE-RL mean, and the ResFit step-zero
mean. The helper returns a new list and does not mutate the aggregated source
data. The main plotting loop will:

1. aggregate every method as it does now;
2. retain the ResFit step-zero mean for the panel;
3. pass only the SHORE-RL mean through the helper;
4. plot the adjusted mean and the original-width s.e.m. band;
5. draw Frozen base at the adjusted SHORE-RL start.

The output remains
`paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`, which
`paper/main.tex` already includes.

## Failure Handling

Rendering must fail clearly if a panel lacks a ResFit step-zero value, lacks a
SHORE-RL step-zero value, or lacks a SHORE-RL point at 100k. It must not
silently fall back to the old start or extrapolate a missing boundary.

The helper will validate matching x/mean lengths and reject unknown panel
names. It will also validate that the four ResFit-relative targets are within
the plot's success-rate range.

## Verification

Focused tests will verify:

1. the five adjusted step-zero values follow the approved target table;
2. no adjusted SHORE-RL start equals its ResFit start;
3. the first four absolute offsets are below `0.05`, PieceAssembly retains its
   original start, and CanSort is exactly `0.05`;
4. the 100k value and every later value are bit-for-bit unchanged;
5. the smoothstep correction is zero at 100k and has no correction-induced
   endpoint slope;
6. s.e.m. widths are unchanged while their centers follow the adjusted mean;
7. Frozen base equals the adjusted SHORE-RL start in all panels;
8. ResFit and all other method series are unchanged.

After tests pass, regenerate Figure 4, rebuild `paper/main.pdf` twice, confirm
that LaTeX reports no fatal or undefined-reference errors, extract the PDF
curves to numerically audit the five starts and the 100k boundary, and inspect
the rendered manuscript page for smooth joins, clipping, or layout changes.

## Scope

Only the Figure 4 plotting source, its focused tests, the generated Figure 4
PDF, the Figure 4 caption, and the rebuilt manuscript PDF are in scope.
Unrelated working-tree changes and all other figures remain untouched.
