# Supplement Citation Audit Design

## Goal

Complete a focused citation audit of `paper/aaai2027-unified-supp.tex`
without changing any prose, equations, tables, figures, captions, numerical
results, TODOs, or method descriptions.

## Scope

- Modify only citation commands in `paper/aaai2027-unified-supp.tex`.
- Add at most three direct technical-source records to
  `paper/references.bib`.
- Reuse the existing bibliography for HIQL, RLPD, TD3, potential-based
  shaping, and model-error accumulation.
- Add direct sources for LTX-Video, T5, and flow matching because the
  supplement names those external architectures or objectives explicitly.
- Do not add citations to the paper's own implementation choices,
  hyperparameters, datasets collected by the authors, or experimental results.

## Citation Placement

1. Cite HIQL where its mask and high-level navigator are described.
2. Cite potential-based reward shaping where policy-invariance is discussed.
3. Cite RLPD and TD3 at the residual update definition.
4. Cite LTX-Video, T5, and flow matching in the dynamics-model architecture
   paragraph.
5. Cite MBPO where bounded imagination recursion is motivated by compounding
   model error.

## Verification

- The bibliography grows from 37 to 40 unique entries.
- Every new BibTeX key is cited in the supplement.
- No cited key is missing and no BibTeX key is duplicated.
- Removing all citation commands from the supplement preserves the baseline
  SHA-256
  `2ae6bc9b94738c778e31083c8db8a682daafefc4a26e67ef37511437e4b1f39e`.
- `pdflatex -> bibtex -> pdflatex -> pdflatex` succeeds for
  `aaai2027-unified-supp.tex`.

