# Figure 8 Size Design

## Goal

Reduce the displayed size of the real-robot learning-curve figure in the main
paper.

## Change

In `main.tex`, change Figure 8's `\includegraphics` width from
`0.85\textwidth` to `0.70\textwidth`.

The source PDF, plot data, caption, placement specifier, and surrounding text
remain unchanged.

## Verification

Rebuild `main.pdf` with BibTeX and multiple pdfLaTeX passes, confirm that the
document compiles without fatal or undefined-reference errors, and visually
inspect the page containing Figure 8 to ensure the reduced figure remains
legible and does not introduce a layout regression.
