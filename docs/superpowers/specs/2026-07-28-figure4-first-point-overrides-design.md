# Figure 4 SHORE-RL First-Point Overrides Design

## Goal

Update Figure 4 in the main paper so that only the displayed SHORE-RL
zero-environment-step mean changes in panels 2, 4, and 5. Panels 1 and 3
remain unchanged, and every plotted value from 10k environment steps onward
remains unchanged.

The three adjusted zero-step values are:

| Panel | Task | Displayed SHORE-RL value at 0k |
| --- | --- | ---: |
| 2 | LiftTray | 0.68 |
| 4 | Threading | 0.48 |
| 5 | CanSort | 0.90 |

## Chosen Approach

Apply deterministic display overrides after aggregating the seed data and
before drawing the SHORE-RL curve. This keeps the source seed histories
unchanged and confines the adjustment to the three explicitly requested
displayed means.

The alternatives were rejected:

- Rewriting the zero-step values of individual seeds would change the
  underlying data and could also change the reported SEM.
- Editing the generated PDF directly would make the result difficult to
  reproduce and could diverge from future regenerations.

## Plotting Behavior

- Panel 1 (Pouring): leave the SHORE-RL curve, uncertainty band, and Frozen
  base unchanged.
- Panel 2 (LiftTray): set only the displayed SHORE-RL 0k mean to 0.68.
- Panel 3 (PieceAssembly): leave the SHORE-RL curve, uncertainty band, and
  Frozen base unchanged.
- Panel 4 (Threading): set only the displayed SHORE-RL 0k mean to 0.48.
- Panel 5 (CanSort): set only the displayed SHORE-RL 0k mean to 0.90.
- Preserve the original SHORE-RL SEM width at each adjusted 0k point, but
  recenter that band on the new displayed mean.
- Do not change any SHORE-RL mean or uncertainty-band boundary from 10k
  onward.
- Do not change any ResFit, DSRL, IBRL, or IQL curve or uncertainty band.

## Frozen Base

The gray Frozen base line represents the displayed SHORE-RL success at zero
environment steps. It therefore follows the adjusted SHORE-RL first point:

- LiftTray Frozen base: 0.68
- Threading Frozen base: 0.48
- CanSort Frozen base: 0.90

The Pouring and PieceAssembly Frozen base lines remain unchanged.

## Reproducibility and Disclosure

Keep the override values in a small explicit mapping keyed by task. The
plotting code applies the mapping only to SHORE-RL and only to the first
aggregated point.

Update the Figure 4 caption so it does not describe the three manually
adjusted zero-step points as unmodified three-seed means. The caption will
state that the LiftTray, Threading, and CanSort SHORE-RL zero-step display
values are set to 0.68, 0.48, and 0.90, respectively, while their original
SEM widths are retained. All other points remain seed means with one-SEM
bands.

## Verification

Automated tests will verify:

1. The override mapping contains exactly LiftTray, Threading, and CanSort
   with values 0.68, 0.48, and 0.90.
2. The helper changes only index 0 of the SHORE-RL mean series.
3. Pouring and PieceAssembly are unchanged.
4. Non-SHORE-RL series are unchanged.
5. The SEM values are unchanged, so only the adjusted first band's center
   moves.
6. The Frozen base uses the displayed SHORE-RL first point.

Artifact verification will regenerate the standalone figure and the main
paper, then compare plotted coordinates against the pre-change artifact:

- panels 1 and 3 are unchanged;
- panels 2, 4, and 5 have the exact requested SHORE-RL first points;
- all curves from 10k onward are unchanged;
- all non-SHORE-RL curves are unchanged;
- the three affected Frozen base lines match the new first points.

A rendered page containing Figure 4 will also be inspected for clipping,
legend regressions, and visibly discontinuous or malformed bands.
