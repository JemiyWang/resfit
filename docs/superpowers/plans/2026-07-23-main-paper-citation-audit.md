# Main-Paper Citation Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add well-supported citations to the SHORE-RL main paper and replace incomplete or outdated BibTeX metadata with verified records.

**Architecture:** Treat `paper/main.tex` as immutable prose: only citation commands may change. Use first-party proceedings and official arXiv records to build verified entries in `paper/references.bib`, then compile the paper and audit the generated bibliography.

**Tech Stack:** LaTeX, BibTeX, AAAI 2027 style, official PMLR/OpenReview/RSS/NeurIPS/arXiv metadata.

## Global Constraints

- Audit only `paper/main.tex` and `paper/references.bib`.
- Do not rewrite prose in `paper/main.tex`.
- Do not change claims, experimental values, section structure, captions, or tables.
- Changes to `paper/main.tex` are limited to adding or adjusting `\cite{...}` commands.
- Do not cite blogs, project announcements, or unverified third-party BibTeX.
- Preserve existing citation keys when correcting metadata so downstream LaTeX remains stable.

---

### Task 1: Build and Verify the Citation Set

**Files:**
- Read: `paper/main.tex`
- Read: `paper/references.bib`

**Interfaces:**
- Consumes: the scope and quality rules in `docs/superpowers/specs/2026-07-23-main-paper-citation-audit-design.md`
- Produces: a verified set of citation keys and first-party source records for Tasks 2 and 3

- [ ] **Step 1: Verify the six new foundational papers against first-party records**

Verify these records:

```text
ross2011dagger       DAgger, AISTATS 2011, PMLR 15:627--635
arjona2019rudder     RUDDER, NeurIPS 2019
nachum2018hiro       HIRO, NeurIPS 2018
fang2022ptp          Planning to Practice, ICLR 2022
hafner2020dreamer    Dreamer, ICLR 2020
janner2019mbpo       MBPO, NeurIPS 2019
```

Expected: title, complete author list, year, venue, and official URL agree with PMLR, OpenReview, or NeurIPS proceedings.

- [ ] **Step 2: Verify formal venues and complete metadata for existing records**

Verify these exact publication states:

```text
hu2023imitation          RSS 2024
wagenmaker2025dsrl       CoRL 2025
black2024pi0             RSS 2025
jia2024dexmimicgen       ICRA 2025
mandlekar2021matters     CoRL 2021 proceedings, published in PMLR 2022
resfit2025residual       arXiv 2509.19301 and ICLR 2026 workshop; replace placeholder authors
yang2026rise             RSS 2026; replace abbreviated authors
sima2026kai0             arXiv 2602.09021; replace abbreviated authors
mandlekar2023mimicgen    CoRL 2023; replace abbreviated authors
liu2023libero            NeurIPS 2023 Datasets and Benchmarks; replace abbreviated authors
```

Expected: no record retains `Authors` or `and others` when the official complete author list is available.

### Task 2: Add Citations Without Rewriting Prose

**Files:**
- Modify: `paper/main.tex`

**Interfaces:**
- Consumes: verified citation keys from Task 1
- Produces: citation-supported prose with byte-for-byte-identical non-citation text

- [ ] **Step 1: Save a prose-only checksum baseline**

Run a script that removes every `\cite{...}` command from `paper/main.tex` and hashes the result.

Expected: one SHA-256 digest saved for comparison after citation insertion.

- [ ] **Step 2: Add citations at the audited claims**

Apply only the following citation-level changes:

```text
Introduction, BC coverage limits:
  add ross2011dagger and mandlekar2021matters

Introduction, sparse delayed feedback diagnosis:
  add arjona2019rudder

Related Work, BC demonstration quality and coverage:
  add ross2011dagger and mandlekar2021matters

Related Work, temporal abstraction into intermediate goals:
  add nachum2018hiro and fang2022ptp

Related Work, learned dynamics for policy-learning experience:
  add hafner2020dreamer and janner2019mbpo

Dense Potential Shaping, potential-difference construction and dynamic-potential caveat:
  add ng1999policy and devlin2012dynamic at the corresponding existing sentences

Learned Dynamics Model, real-interaction cost:
  add resfit2025residual

Residual RL in Imagination, short branched imagined rollouts and bounded recursion:
  add janner2019mbpo
```

Expected: no experimental result, number, caption, table, or author claim changes.

- [ ] **Step 3: Recompute the prose-only checksum**

Expected: the digest exactly matches Step 1, proving that only citation commands changed.

### Task 3: Update and Extend BibTeX

**Files:**
- Modify: `paper/references.bib`

**Interfaces:**
- Consumes: first-party metadata verified in Task 1
- Produces: resolvable BibTeX entries for every key used in `paper/main.tex`

- [ ] **Step 1: Add the six verified foundational records**

Add complete `@inproceedings` records for:

```text
ross2011dagger
arjona2019rudder
nachum2018hiro
fang2022ptp
hafner2020dreamer
janner2019mbpo
```

Each record must include title, complete authors, venue, year, and official URL; include pages, volume, DOI, or PMLR series data when the first-party record provides it.

- [ ] **Step 2: Correct the ten existing records**

Replace incomplete author lists and update formal venues exactly as verified in Task 1. Do not rename citation keys.

- [ ] **Step 3: Audit citation-key resolution**

Extract all keys from `\cite{...}` in `paper/main.tex` and all entry keys from `paper/references.bib`.

Expected: zero missing keys and zero duplicate BibTeX keys.

### Task 4: Compile and Inspect the Submission

**Files:**
- Build: `paper/main.pdf`
- Inspect: `paper/main.log`
- Inspect: `paper/main.bbl`

**Interfaces:**
- Consumes: the updated LaTeX and BibTeX from Tasks 2 and 3
- Produces: a compiled main paper with a resolved, traceable bibliography

- [ ] **Step 1: Run the complete bibliography build**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all four commands exit successfully.

- [ ] **Step 2: Check citation diagnostics**

Search `paper/main.log` for undefined citations/references and `paper/main.bbl` for placeholder authors.

Expected: no undefined citations, no `ResFiT Authors`, and no `and others`.

- [ ] **Step 3: Check bibliography and layout impact**

Record page count before and after, inspect bibliography extraction from `main.pdf`, and report any new overfull boxes.

Expected: all new papers appear in the bibliography; any page-count change or warnings are explicitly reported.

- [ ] **Step 4: Report unsupported claims without editing them**

Report any claim for which no directly supporting high-quality source was found, especially novelty statements. Do not modify those sentences.
