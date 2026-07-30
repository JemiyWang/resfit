# Figure 8 Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Figure 8 from `0.85\textwidth` to `0.70\textwidth`.

**Architecture:** Make one localized LaTeX width change in `main.tex`, then rebuild and visually inspect the affected page. Preserve all other local edits and generated source assets.

**Tech Stack:** LaTeX, pdfTeX, BibTeX, Poppler

## Global Constraints

- Do not push to GitHub.
- Preserve unrelated local changes.
- Do not alter Figure 8's source PDF, caption, or placement specifier.

---

### Task 1: Resize and Verify Figure 8

**Files:**
- Modify: `main.tex:610`
- Verify: `main.pdf`

**Interfaces:**
- Consumes: `figure/fig_real_robot_curves.pdf`
- Produces: Figure 8 rendered at `0.70\textwidth`

- [ ] **Step 1: Change the width**

Replace:

```latex
\includegraphics[width=0.85\textwidth]{figure/fig_real_robot_curves.pdf}
```

with:

```latex
\includegraphics[width=0.70\textwidth]{figure/fig_real_robot_curves.pdf}
```

- [ ] **Step 2: Verify the source**

Run:

```bash
rg -n 'fig_real_robot_curves.pdf' main.tex
git diff --check -- main.tex
```

Expected: the Figure 8 line contains `width=0.70\textwidth` and the diff check succeeds.

- [ ] **Step 3: Rebuild**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all four commands exit successfully.

- [ ] **Step 4: Inspect the affected page**

Render the page containing Figure 8 to an image and confirm that the plot,
axis labels, legend, and caption remain legible and that no overlap or clipping
was introduced.

- [ ] **Step 5: Keep the change local**

Run:

```bash
git status --short --branch
```

Expected: the modification remains local and no push occurs.
