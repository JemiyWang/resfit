# Pages 4 and 6 Text-Flow Fill Design

## Goal

Fill the avoidable blank regions on pages 4 and 6 with the following prose,
without changing any figure or table source block, size, placement option, or
relative order.

## Design

- Remove the `\FloatBarrier` immediately before `\subsection{Ablations}`.
- Remove the `\FloatBarrier` immediately before
  `\subsection{Real-World Extension}`.
- Do not add or rewrite paper prose.
- Do not edit any `figure`, `figure*`, `table`, caption, label,
  `\includegraphics`, or placement option.
- Let LaTeX continue the next subsection's prose into the available column
  space while retaining the existing float queue and source order.

## Verification

- Compile twice to settle floats and references.
- Confirm Figures 4–8 and Tables 1–3 retain their current page numbers.
- Confirm page 4's lower-right blank region and page 6's lower-left blank
  region are filled by naturally continued prose.
- Confirm the PDF has no new float, reference, citation, or overfull-box
  warnings.
