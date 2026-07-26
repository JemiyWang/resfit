# Figure 4 and Pages 6–7 Flow Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Figure 4 single-column and let Figures 4–6 flow naturally while keeping each figure next to the prose that interprets it.

**Architecture:** Preserve the existing semantic source order in `paper/main.tex`: the LIBERO figure follows the base-generality paragraph, the matched-ablation figure follows its interpretation, and the reward-form figure follows its controlled-comparison paragraph. Use ordinary single-column floats with flexible top/bottom/float-page placement, while retaining the subsection boundary that prevents ablation figures from drifting into the real-robot discussion.

**Tech Stack:** AAAI 2027 two-column LaTeX, pdfLaTeX, BibTeX, Poppler (`pdfinfo`, `pdftocairo`, `pdftotext`).

## Global Constraints

- Figure 4 uses `figure`, not `figure*`, and its graphic width is exactly `\columnwidth`.
- Figures 4–6 use normal floating placement; do not add `[H]`, page breaks, negative spacing, or page-specific positioning.
- Figure order and semantic anchors remain Figure 4/base generality, Figure 5/component ablation, and Figure 6/reward-form comparison.
- Retain the `\FloatBarrier` before `Real-World Extension` as a semantic subsection boundary.
- Preserve all pre-existing uncommitted edits in `paper/main.tex`; inspect only the narrow task diff and do not commit that already-dirty file.

---

### Task 1: Convert Figures 4–6 to semantic, flexible single-column floats

**Files:**
- Modify: `paper/main.tex:461-564`
- Test: static source assertions against `paper/main.tex`

**Interfaces:**
- Consumes: existing labels `fig:libero`, `fig:ablbars`, and `fig:stagevspot`, plus the existing `\FloatBarrier`
- Produces: Figure 4 as a `[t]` single-column float and Figures 5--6 as `[tbp]` single-column floats, all sized to `\columnwidth`

- [ ] **Step 1: Run the pre-change assertion and verify it fails**

Run:

```bash
rg -U '\\begin\{figure\}\[t\]\n\\centering\n\\includegraphics\[width=\\columnwidth\]\{figure/fig_libero10_aligned.pdf\}' paper/main.tex
```

Expected: exit status `1`, because Figure 4 is currently a `figure*` with width `0.66\textwidth`.

- [ ] **Step 2: Apply the minimal float changes**

In `paper/main.tex`, change only these environment and placement lines:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figure/fig_libero10_aligned.pdf}
...
\end{figure}
```

For `fig:ablbars` and `fig:stagevspot`, keep their graphics, captions, labels, and semantic source positions unchanged, but change each opening line from `\begin{figure}[t]` to:

```latex
\begin{figure}[tbp]
```

Keep the existing `\FloatBarrier` immediately before `\subsection{Real-World Extension}`.

- [ ] **Step 3: Run source assertions**

Run:

```bash
rg -n 'begin\{figure\}\[tbp\]|fig_libero10_aligned|fig_ablation_bars_400k|fig_staged_vs_pothiql_threading_piece_v2|FloatBarrier' paper/main.tex
```

Expected: Figure 4 uses `[t]`, Figures 5 and 6 use `[tbp]`, and all three ordinary figures occur in source order; Figure 4 uses `fig_libero10_aligned`; the barrier follows Figure 6.

- [ ] **Step 4: Inspect the narrow diff**

Run:

```bash
git diff -U3 -- paper/main.tex
```

Expected: within the Figure 4–6 region, the only new task edits are `figure*` to `figure`, `0.66\textwidth` to `\columnwidth`, and `[t]` to `[tbp]`. Pre-existing unrelated diff remains untouched.

### Task 2: Compile and verify semantic layout on pages 6–7

**Files:**
- Regenerate: `paper/main.pdf`, `paper/main.aux`, `paper/main.log`
- Inspect: rendered pages 6 and 7

**Interfaces:**
- Consumes: the float definitions from Task 1 and the existing `paper/main.bbl`
- Produces: a refreshed nine-page paper PDF with ordered, semantically adjacent Figures 4–6

- [ ] **Step 1: Compile twice to settle floats and references**

Run twice from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both commands exit `0` and produce `main.pdf`.

### Task 3: Refine section-level semantic anchors after visual review

**Files:**
- Modify: `paper/main.tex:461-636`
- Inspect: rendered pages 4--7

**Rationale:** The first compile showed that flexible Figure 4--6 floats fixed page 6, but the ablation subsection could begin before the main-result floats and the full-width real-task overview left page 7 sparse. The final layout therefore uses section-semantic barriers and preserves full width for both real-world overview figures.

- [ ] **Step 1: Keep main results with their discussion**

Use `\begin{figure}[t]` and `width=\columnwidth` for Figure 4, then retain a `\FloatBarrier` immediately before `\subsection{Ablations}`.

- [ ] **Step 2: Keep ablations with their discussion**

Use `[tbp]` for Figures 5 and 6 and retain the existing `\FloatBarrier` before `Real-World Extension`.

- [ ] **Step 3: Balance the real-world pages without shrinking the learning curves**

Use full-width `figure*` floats at `\textwidth` for both Figures 7 and 8. Declare them consecutively at the start of `Real-World Extension` so LaTeX can stack them in source order at the next page top while the matching task and evaluation text flows below.

- [ ] **Step 4: Recompile twice and inspect pages 4--7**

Expected: Figure 4 follows the base-generality discussion before the ablation subsection; Figures 5 and 6 remain with their matching ablation paragraphs; Figures 7 and 8 both span page 7 in source order above the task and evaluation text; pages 6 and 7 use both columns without manual page breaks or absolute placement.
