# Figure 6 Full-Border Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore top and right borders on both Figure 6 panels.

**Architecture:** Change the shared Matplotlib `rcParams` so both axes inherit the same four-sided border style, regenerate the existing PDF, and rebuild the manuscript. Do not alter data acquisition, aggregation, curves, labels, legend, or dimensions.

**Tech Stack:** Python 3, Matplotlib, Weights & Biases API, pdfTeX, BibTeX, Poppler

## Global Constraints

- Preserve unrelated local modifications.
- Keep border color `#52514e` and width `0.8`.
- Do not push to GitHub.

---

### Task 1: Restore and Verify Figure 6 Borders

**Files:**
- Modify: `plot_staged_vs_pothiql_threading_piece_v2.py:116-120`
- Regenerate: `figure/fig_staged_vs_pothiql_threading_piece_v2.pdf`
- Verify: `main.pdf`

**Interfaces:**
- Consumes: existing Figure 6 W&B run IDs and plotting configuration
- Produces: Figure 6 with four visible axis borders on both panels

- [ ] **Step 1: Verify the current border configuration**

Run:

```bash
rg -n 'axes\.spines\.(top|right)' plot_staged_vs_pothiql_threading_piece_v2.py
```

Expected: both settings are `False`.

- [ ] **Step 2: Enable both borders**

Set:

```python
"axes.spines.top": True,
"axes.spines.right": True,
```

- [ ] **Step 3: Verify the source**

Run:

```bash
rg -n 'axes\.spines\.(top|right)' plot_staged_vs_pothiql_threading_piece_v2.py
git diff --check -- plot_staged_vs_pothiql_threading_piece_v2.py
```

Expected: both settings are `True` and the diff check succeeds.

- [ ] **Step 4: Regenerate Figure 6**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python plot_staged_vs_pothiql_threading_piece_v2.py
```

Expected: exit status 0 and a rewritten
`figure/fig_staged_vs_pothiql_threading_piece_v2.pdf`.

- [ ] **Step 5: Rebuild the manuscript**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: every command exits successfully.

- [ ] **Step 6: Inspect Figure 6**

Render the manuscript page containing Figure 6 and confirm that both panels
have visible top and right borders with the same color and width as the
existing left and bottom borders, with no clipping or overlap.

- [ ] **Step 7: Keep the result local**

Run:

```bash
git status --short --branch
```

Expected: the edits remain local and no push occurs.
