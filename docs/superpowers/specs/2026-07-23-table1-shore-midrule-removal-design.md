# Table 1 SHORE-RL Row Separator Removal

## Goal

Remove the horizontal rule immediately below the `SHORE-RL` row in Table 1 of
the SHORE-RL main paper.

## Scope

- Edit only `paper/main.tex`.
- Remove the `\midrule` immediately following the `SHORE-RL` data row.
- Preserve all table values, row ordering, spacing commands, caption text, and
  every other table rule.
- Do not modify the supplementary material.

## Verification

1. Inspect the source diff and confirm that exactly the intended `\midrule` was
   removed.
2. Rebuild `main.pdf` using the paper's documented LaTeX build sequence.
3. Confirm that the build exits successfully and Table 1 no longer has a
   horizontal rule directly below `SHORE-RL`.
