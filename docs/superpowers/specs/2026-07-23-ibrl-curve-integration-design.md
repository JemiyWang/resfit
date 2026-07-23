# IBRL Curve Integration Design

## Goal

Add IBRL learning curves from exactly the 11 canonical finished W&B runs supplied
by the user to:

`paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`

The existing non-`copy` PDF must remain unchanged.

## Data Selection

- W&B entity: `674575221-beijing-institute-of-technology`
- W&B project: `dexmg_formal`
- CanSort: `z4ob6395`, `i6f0pdnm`
- LiftTray: `905ud33j`, `boluepp0`
- Pouring: `u3mobgtb`, `uqsl54zu`
- Threading: `mvxv2vgt`, `pts8ariy`
- ThreePiece: `p7uomccw`, `bv1avdba`, `8i3b9r53`

No run discovered by name, local completion marker, or project scan may be added.
In particular, old or duplicate ThreePiece seed-1 runs are excluded.

## Plot Semantics

Use `other/step` as the environment-step x coordinate and `score/score` as the
evaluation-success-rate y coordinate. These fields are present consistently across
all 11 canonical runs, including older runs that do not expose
`eval/success_rate`.

For each task, align observations to the existing 10k grid and reuse the existing
multi-seed aggregation:

- solid task-level mean curve;
- translucent plus/minus one standard-error-of-the-mean band;
- two seeds for CanSort, LiftTray, Pouring, and Threading;
- three seeds for ThreePiece;
- full 0--500k plot range.

IBRL receives a distinct color and line style and a single `IBRL` entry in the
shared legend. Existing curve styling, task ordering, typography, axes, figure
dimensions, and all non-IBRL data remain governed by the current plotting script.

## Implementation

Extend `paper/plot_pouring_lifttray_seeds.py` with:

1. an `ibrl` style and draw-order entry;
2. the exact canonical run IDs under each task;
3. an IBRL-specific history reader for `other/step` and `score/score`;
4. an output-path argument whose default preserves current behavior.

Invoke the script with the output argument pointing only to the requested `copy`
PDF. Do not overwrite the canonical non-`copy` PDF.

## Validation

Before reporting completion:

1. verify all 11 fetched runs have the expected ID, name, `finished` state, and a
   500k point;
2. verify the source lists exactly 11 unique IBRL IDs;
3. verify the output is a one-page PDF with the expected dimensions and embedded
   fonts;
4. extract PDF text and confirm the shared legend contains `IBRL`;
5. rasterize and visually inspect the complete figure for clipping, overlap, and
   legibility;
6. confirm the non-`copy` PDF checksum is unchanged.

