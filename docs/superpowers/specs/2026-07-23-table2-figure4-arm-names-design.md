# Table 2 and Figure 4 Arm-Name Alignment Design

## Goal

Align the display names in the `Arm` column of Table 2 with the legend names
used in Figure 4.

## Scope

- Modify only the five arm-name cells in Table 2 of `paper/main.tex`.
- Preserve the waypoint, BC, staged, and removed columns.
- Preserve the table caption, surrounding prose, experimental configurations,
  numerical results, figures, and bibliography.
- Do not regenerate Figure 4 because its legend already contains the desired
  names.

## Exact Mapping

| Current Table 2 name | Figure 4 name |
|---|---|
| `Full (SHORE-RL)` | `SHORE-RL` |
| `- staged reward` | `w/o stage shaping` |
| `- waypoint` | `w/o waypoint` |
| `- waypoint - staged` | `w/o waypoint & stage shaping` |
| `waypoint only` | `w/o demo-BC & stage shaping` |

## Verification

- Confirm the five Table 2 labels exactly match the Figure 4 plotting labels.
- Compile `main.tex` through LaTeX/BibTeX.
- Confirm there are no undefined references or citations.
- Visually inspect the page containing Table 2 for clipping or overflow.
