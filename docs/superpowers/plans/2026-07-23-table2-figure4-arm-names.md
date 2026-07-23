# Table 2 and Figure 4 Arm-Name Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five arm names in Table 2 exactly match the five legend names in Figure 4.

**Architecture:** Change only the first-column display labels in the existing Table 2 rows. Keep all experimental settings and surrounding paper content unchanged, then validate the exact label set and rebuild the paper.

**Tech Stack:** LaTeX, BibTeX, ripgrep, pdfLaTeX.

## Global Constraints

- Modify only the five arm-name cells in Table 2 of `paper/main.tex`.
- Preserve the waypoint, BC, staged, and removed columns.
- Preserve captions, prose, numerical results, figures, and bibliography.
- Do not regenerate Figure 4.

---

### Task 1: Align the Table 2 Arm Names

**Files:**
- Modify: `paper/main.tex:566-570`

**Interfaces:**
- Consumes: Figure 4 legend labels defined in `paper/plot_ablation_curves.py:46-57`.
- Produces: Five Table 2 labels identical to the Figure 4 legend.

- [ ] **Step 1: Verify the mismatch exists**

Run:

```bash
rg -n 'Full \(SHORE-RL\)|staged reward|waypoint only' paper/main.tex
```

Expected: the current Table 2 names are present at lines 566--570.

- [ ] **Step 2: Replace only the five first-column labels**

Use these exact LaTeX rows:

```latex
\textbf{SHORE-RL}                       & on  & $0.1$ & on  & ---             \\
w/o stage shaping                       & on  & $0.1$ & off & shaping         \\
w/o waypoint                            & off & $0.1$ & on  & waypoint        \\
w/o waypoint \& stage shaping           & off & $0.1$ & off & both structural \\
w/o demo-BC \& stage shaping            & on  & $0$   & off & BC $+$ staged   \\
```

- [ ] **Step 3: Verify exact label parity**

Run:

```bash
rg -n 'SHORE-RL|w/o stage shaping|w/o waypoint|w/o demo-BC' paper/main.tex paper/plot_ablation_curves.py
```

Expected: Table 2 and the Figure 4 plotting script contain the same five display labels.

- [ ] **Step 4: Build the paper**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all four commands exit successfully and produce `paper/main.pdf`.

- [ ] **Step 5: Validate the final output**

Run:

```bash
rg -n -i 'undefined citations|citation.*undefined|undefined references|Warning--' paper/main.log paper/main.blg
pdfinfo paper/main.pdf
```

Expected: no undefined citation/reference warnings; `main.pdf` is readable.

- [ ] **Step 6: Visually inspect Table 2**

Render the page containing Table 2 and confirm that all five arm names are visible without clipping or overlap.
