# Pages 4 and 6 Text-Flow Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the blank column regions on pages 4 and 6 with existing following prose while preserving every figure and table definition.

**Architecture:** Remove only the two standalone `\FloatBarrier` commands that stop the two-column text stream before the Ablations and Real-World Extension subsections. Leave every float block and all prose unchanged, then verify page mappings and rendered layout.

**Tech Stack:** AAAI 2027 two-column LaTeX, pdfLaTeX, Poppler.

## Global Constraints

- Do not add or rewrite paper prose.
- Do not edit any figure, table, caption, label, graphic width, placement option, or source order.
- Preserve the current page mapping for Figures 4–8 and Tables 1–3.
- Do not commit `paper/main.tex`, because it contains unrelated pre-existing user changes.

---

### Task 1: Restore uninterrupted text flow

**Files:**
- Modify: `paper/main.tex`
- Test: static source checks and compiled PDF

**Interfaces:**
- Consumes: the two standalone `\FloatBarrier` commands before `Ablations` and `Real-World Extension`
- Produces: uninterrupted normal LaTeX prose flow with the existing float queue unchanged

- [ ] **Step 1: Verify the pre-change condition**

Run:

```bash
rg -n '^\\FloatBarrier$' paper/main.tex
```

Expected: exactly two matches, immediately before the two target subsections.

- [ ] **Step 2: Remove only the two barriers**

Delete only these two standalone lines:

```latex
\FloatBarrier
```

Do not modify either adjacent `\subsection` or any float block.

- [ ] **Step 3: Verify the source constraint**

Run:

```bash
rg -n '^\\FloatBarrier$|begin\{figure|begin\{table|includegraphics' paper/main.tex
```

Expected: no standalone `\FloatBarrier` remains; all figure, table, and graphic
definitions remain present.

- [ ] **Step 4: Compile twice**

Run twice from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both runs exit `0` and produce `main.pdf`.

- [ ] **Step 5: Verify numbering, page mapping, and visual fill**

Run:

```bash
rg -n 'newlabel\{(fig:(libero|ablbars|stagevspot|realtask|realrobot-curves)|tab:(main|ablarms|realrobot))\}' main.aux
pdftocairo -f 4 -l 6 -jpeg -r 160 main.pdf ../temp/text-flow-fill
```

Expected: Figures 4–8 and Tables 1–3 retain their prior pages; existing prose
fills the avoidable page-4 lower-right and page-6 lower-left gaps.

- [ ] **Step 6: Check regressions**

Run:

```bash
rg -n 'Too many unprocessed floats|Float too large|undefined references|undefined citations|Overfull \\hbox|LaTeX Warning:.*(Float|Reference|Citation)' main.log
git diff --check -- paper/main.tex
```

Expected: the warning scan has no matches and `git diff --check` exits `0`.
