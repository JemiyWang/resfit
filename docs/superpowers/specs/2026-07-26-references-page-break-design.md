# References Page Break Design

## Goal

Make the References section in `paper/main.pdf` begin on a new physical page
after the seven-page main text.

## Change

Insert `\clearpage` immediately before `\bibliography{references}` in
`paper/main.tex`. `\clearpage` is preferred over `\newpage` because it flushes
pending floats before starting the bibliography.

## Scope

Do not change prose, figures, bibliography styling, citations, or supplementary
material.

## Verification

Rebuild `paper/main.pdf` and confirm:

1. the conclusion remains on physical page 7;
2. the `References` heading first appears on physical page 8;
3. the PDF builds without LaTeX errors.
