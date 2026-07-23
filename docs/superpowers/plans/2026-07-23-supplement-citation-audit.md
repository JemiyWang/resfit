# Supplement Citation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add targeted citations to the supplementary paper without changing non-citation content.

**Architecture:** Reuse existing bibliography keys for method foundations and add three direct architecture/objective sources. Validate the citation graph, citation-stripped hash, and standalone supplement build.

**Tech Stack:** LaTeX, BibTeX, Perl validation, official OpenReview/JMLR/arXiv metadata.

## Global Constraints

- Modify only citation commands in `paper/aaai2027-unified-supp.tex`.
- Append only `hacohen2025ltxvideo`, `raffel2020t5`, and `lipman2023flowmatching` to `paper/references.bib`.
- Preserve the citation-stripped supplement SHA-256 `2ae6bc9b94738c778e31083c8db8a682daafefc4a26e67ef37511437e4b1f39e`.
- Preserve all prose, equations, tables, figures, captions, numerical results, TODOs, and method descriptions.

---

### Task 1: Add Three Direct Technical Sources

**Files:**
- Modify: `paper/references.bib`

- [ ] Append the complete official records for LTX-Video (arXiv 2025), T5
  (JMLR 2020), and Flow Matching (ICLR 2023).
- [ ] Verify `bib=40 duplicates=0`.

### Task 2: Add Targeted Supplement Citations

**Files:**
- Modify: `paper/aaai2027-unified-supp.tex`

- [ ] Add `park2023hiql` to the HIQL mask and navigator descriptions.
- [ ] Add `ng1999policy,devlin2012dynamic` to the potential-based shaping and
  policy-invariance statements.
- [ ] Add `ball2023efficient,fujimoto2018addressing` to the RLPD/TD3 update
  description.
- [ ] Add `hacohen2025ltxvideo`, `raffel2020t5`, and
  `lipman2023flowmatching` to the architecture paragraph.
- [ ] Add `janner2019mbpo` to the bounded model-error accumulation statement.
- [ ] Verify the citation-stripped hash remains
  `2ae6bc9b94738c778e31083c8db8a682daafefc4a26e67ef37511437e4b1f39e`.

### Task 3: Validate and Compile

**Files:**
- Verify: `paper/aaai2027-unified-supp.tex`
- Verify: `paper/references.bib`
- Generate: `paper/aaai2027-unified-supp.pdf`

- [ ] Verify `supp_cited=20 bib=40 missing=0 duplicates=0`.
- [ ] Run `pdflatex`, `bibtex aaai2027-unified-supp`, `pdflatex`,
  `pdflatex`.
- [ ] Confirm no undefined citations or overfull boxes in the final log.

