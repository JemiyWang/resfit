# Main-Paper Citation Corrections Design

## Goal

Correct the citation-placement and bibliography issues identified in the
reference audit of `paper/main.pdf` without changing the paper's experimental
claims, method, figures, tables, or evaluation results.

## Considered Approaches

1. **Minimal correction:** fix only the three clear semantic mismatches.
   This has the smallest layout risk but leaves several statements broader than
   their cited sources and leaves recent publication metadata incomplete.
2. **Targeted correction (selected):** fix the three clear mismatches, tighten
   three potentially misleading statements, and complete the audited recent
   bibliography records. This addresses the review risks while keeping the
   change set narrow.
3. **Full bibliography normalization:** replace all 48 records with canonical
   publisher BibTeX. This maximizes metadata uniformity but creates unnecessary
   churn and could disturb the fixed ten-page layout.

The user approved the targeted correction by requesting implementation of the
previously presented audit recommendations.

## Text Changes

- Cite only ResFit for the claim that a frozen base and restricted trainable
  component can improve sample efficiency; retain Ball et al. where the text
  discusses learning from prior data.
- Describe the related prior-data methods as leveraging prior data during
  online learning rather than asserting that every method directly mixes
  demonstrations with newly collected transitions.
- Split the world-model paragraph so that RISE supports action-conditioned
  dynamics and VIPER supports action-free video-prediction rewards.
- Mark critic-error and residual-drift accumulation as the paper's mechanism
  hypothesis rather than a result established by the delayed-credit citations.
- Cite DexMimicGen alone for the provenance of the named DexMimicGen tasks.
- Describe IQL as the authors' full-policy instantiation, not as an online
  finetuning formulation introduced by the original IQL paper.

## Bibliography Changes

- Use the full workshop name for ResFit while retaining its 2025 arXiv record.
- Complete DSRL with PMLR volume 305, pages 258--282, series, publisher, and
  canonical PMLR URL. Preserve the author order printed in the final paper PDF:
  Wagenmaker, Nakamoto, Zhang, et al.
- Complete DexMimicGen with ICRA 2025 pages 16923--16930, DOI
  `10.1109/ICRA55743.2025.11127809`, and the DOI URL.
- Keep RISE as an RSS 2026 proceedings entry because the official project
  reports acceptance; do not invent pages or a DOI before they are available.

## Verification

- Build with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- Require zero undefined citations or references.
- Require exactly 47 cited keys and 47 bibliography entries with identical
  sets after removing the unsupported MimicGen co-citation.
- Preserve the ten-page paper, with References starting on page 9 and ending
  on page 10.
- Inspect the final two reference pages and review the scoped source diff.
