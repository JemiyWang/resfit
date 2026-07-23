# Main-Paper Citation Audit Design

## Scope

Audit only `paper/main.tex` and `paper/references.bib`. The supplementary
material is out of scope.

## Hard Constraints

- Do not rewrite prose in `paper/main.tex`.
- Do not change claims, experimental values, section structure, captions, or tables.
- Changes to `paper/main.tex` are limited to adding or adjusting `\cite{...}` commands.
- Add only genuine, traceable, directly relevant research papers.
- Prefer peer-reviewed papers from ICML, ICLR, CoRL, RSS, AAAI, NeurIPS, and closely
  related top robotics or machine-learning venues.
- Verify title, authors, year, and venue using first-party sources such as official
  proceedings, PMLR, OpenReview, RSS/CoRL pages, or the paper's official arXiv record.
- Do not cite blogs, project announcements, or unverified third-party BibTeX.

## Audit Method

1. Map every external citation claim in the abstract, introduction, related work,
   preliminaries, method, and experiment setup.
2. Classify each claim as already supported, missing support, self-contained
   mathematical definition, implementation detail, or the paper's own empirical claim.
3. Add citations only where an external source materially supports the nearby claim.
4. Correct incomplete or outdated metadata in existing BibTeX entries when a verified
   first-party record is available.
5. Record unsupported or potentially overstated claims separately without changing
   their prose.

## Target Coverage

- behavior-cloning coverage and distribution-shift limitations;
- offline-to-online reinforcement learning from prior demonstrations;
- long-horizon temporal credit assignment and hierarchical/goal-conditioned RL;
- potential-based reward shaping and time-varying potentials;
- world-model policy learning and model-error control in imagined rollouts;
- formal publication metadata for the policies, baselines, and benchmarks already cited.

## Verification

- Every citation key used in `main.tex` must resolve in `references.bib`.
- Every added BibTeX record must match a first-party source.
- `bibtex` and `pdflatex` must complete without undefined citations.
- The compiled bibliography must contain no placeholder author names.
- Any page-count or layout impact must be reported.
