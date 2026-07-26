# Main-Paper Reference Expansion Design

## Goal

Expand the main-paper bibliography so that the existing nine-page PDF uses
most of page 9, while preserving all body prose exactly and keeping the output
at nine pages.

## Current State

- Technical content occupies pages 1--7.
- References begin on page 8 because of the existing `\clearpage`.
- The bibliography currently contains 34 cited entries.
- Page 9 contains a mostly full left column and an empty right column.
- `references.bib` contains 40 entries, including an uncited Wiewiora reward
  shaping entry that directly supports existing prose.

## Editing Constraints

- Do not alter any body sentence, heading, caption, equation, table, figure, or
  claim.
- Only extend key lists inside existing `\cite{...}` commands in `main.tex`.
- Add only verified BibTeX records to `references.bib`.
- Do not use `\nocite`.
- Do not change the AAAI bibliography style, font size, spacing, margins, or
  column layout.
- Preserve all unrelated worktree changes.
- Keep the compiled paper at exactly nine pages.

## Core References and Citation Placement

Add these nine references first:

| Key | Paper | Existing claim supported |
| --- | --- | --- |
| `rajeswaran2018learning` | DAPG | Online RL can improve an imitation initialization using demonstrations. |
| `nair2020awac` | AWAC | Prior offline data can accelerate subsequent online RL. |
| `silver2018residual` | Residual Policy Learning | Residual learning adds a learned correction to a fixed controller. |
| `harutyunyan2019hindsight` | Hindsight Credit Assignment | Delayed outcomes create a temporal credit-assignment problem. |
| `sutton1999between` | Options framework | Temporal abstraction decomposes behavior into extended subgoals/actions. |
| `wiewiora2003principled` | Principled Methods for Advising RL Agents | Potential differences provide transition-level shaping feedback. |
| `wu2023daydreamer` | DayDreamer | Learned world models provide imagined experience for physical robot learning. |
| `buckman2018steve` | STEVE | Limiting and weighting recursive model rollouts mitigates model error. |
| `black2025pi05` | pi0.5 | The named VLA predicts 50-step continuous action chunks. |

The Wiewiora record already exists in `references.bib`; the other eight records
will be added from their primary publication pages.

## Adaptive Fill

Compile after adding the core set. If page 9 still has substantial whitespace,
add these references in order:

1. `nair2018overcoming`: DDPG from Demonstrations, cited with the existing
   prior-data replay sentence.
2. `dietterich2000maxq`: MAXQ, cited with the existing temporal-abstraction
   sentence.
3. `kalweit2017uncertaintydriven`: uncertainty-driven imagination, cited with
   the existing short imagined-rollout sentence.

Stop as soon as the right column approaches the normal bottom margin. If an
addition creates page 10, remove the last-added reserve reference. If the core
set itself creates page 10, remove core references in this order until the
paper returns to nine pages: `buckman2018steve`,
`harutyunyan2019hindsight`, `rajeswaran2018learning`. The exact pi0.5 citation,
the foundational residual/temporal citations, and the directly applicable
Wiewiora and DayDreamer citations remain protected.

## Explicit Exclusions

Do not cite the existing LTX-Video, T5, or Flow Matching entries in the main
paper. The current prose does not name their architecture, text encoder, or
training objective, so attaching them to a broader RISE or VLA sentence would
overstate what that sentence says.

Do not add broad VLA survey citations such as Octo merely to consume space.
The exact pi0.5 paper is the citation that directly supports the current
50-step action-chunk sentence.

## Files

- Modify: `paper/main.tex`
- Modify: `paper/references.bib`
- Generate: `paper/main.bbl`
- Generate: `paper/main.pdf`

## Acceptance Criteria

- Every newly output bibliography entry is cited through an existing main-text
  citation site.
- A source comparison shows no body-word changes.
- The PDF has exactly nine pages.
- References still begin on page 8.
- Page 9 uses both columns and ends near, but not below, the normal text
  boundary.
- The LaTeX/BibTeX build reports no undefined citations.
