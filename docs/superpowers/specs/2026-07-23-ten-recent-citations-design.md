# Ten Recent Citations Design

## Goal

Add exactly ten recent, high-quality, highly relevant papers to the main
paper's citation graph without changing any prose, equations, tables, figures,
captions, results, or claims.

## Scope

- Modify only `paper/main.tex` citation commands and
  `paper/references.bib`.
- Prefer work published from 2022 onward.
- Use peer-reviewed papers from ICML, NeurIPS, ICLR, CoRL, RSS, or comparably
  selective robotics/ML venues.
- Verify title, author list, venue, year, and canonical URL using official
  proceedings, OpenReview, PMLR, or the publisher/proceedings website.
- Do not add a paper unless an existing sentence in the main text directly
  supports citing it.

## Distribution

The ten papers will be balanced across the paper's four main literature
interfaces:

1. Three papers on offline-data-assisted online RL or finetuning imitation
   policies.
2. Two papers on frozen-policy residual adaptation or model-agnostic policy
   refinement.
3. Two papers on long-horizon temporal abstraction, hierarchical skills, or
   goal composition.
4. Three papers on world models, imagination learning, or learned simulators
   for robot/RL training.

If a candidate spans two categories, it is counted once and placed at the
single most directly supported sentence.

## Citation Placement

- Add keys only inside existing `\cite{...}` commands where possible.
- Avoid citation dumping: each new paper must support the immediately
  preceding claim.
- Keep the abstract citation-free.
- Do not cite the paper's own experimental results or method-specific design
  choices.
- Do not alter the Figure 5 names `Stage shaping` and
  `Self-derived shaping`.

## Verification

Completion requires all of the following:

- Exactly ten new BibTeX keys relative to the current 27-entry bibliography.
- Every new key is cited in `paper/main.tex`.
- No missing or duplicate BibTeX keys.
- Removing all citation commands from `paper/main.tex` yields the same content
  hash before and after the edit.
- `pdflatex -> bibtex -> pdflatex -> pdflatex` completes successfully.
- The final log contains no undefined citations.

