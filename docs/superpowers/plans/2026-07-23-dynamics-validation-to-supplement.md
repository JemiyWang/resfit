# Dynamics Validation to Supplement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the detailed dynamics-model validation figure from the main paper, retain a compact reliability statement there, and place the full three-part validation figure in Supplementary Sec. C.4.4.

**Architecture:** Edit the two independent LaTeX documents together so their prose agrees: `main.tex` points readers to the supplement, while `aaai2027-unified-supp.tex` owns the detailed protocol and figure. Preserve the existing P0 status instead of inventing results, then compile both documents and audit references and logs.

**Tech Stack:** LaTeX, pdfLaTeX, BibTeX, ripgrep, Poppler

## Global Constraints

- The main text retains a compact description of the three held-out validation axes and one P0 TODO for a future headline metric.
- The detailed figure and protocol live in Supplementary Sec. C.4.4.
- Do not modify dynamics-model training, data, metrics, real-robot results, pipeline, or conclusion.
- Do not claim uncompleted experimental results.

---

### Task 1: Move Detailed Dynamics Validation Out of the Main Paper

**Files:**
- Modify: `paper/main.tex:727-745`
- Modify: `paper/aaai2027-unified-supp.tex:979-1010`

**Interfaces:**
- Consumes: Existing main-paper validation paragraph and existing Supplementary Sec. C.4.4 protocol.
- Produces: A compact main-paper pointer and a self-contained supplementary validation subsection with `fig:supp-wmval`.

- [ ] **Step 1: Run pre-change structural assertions**

```bash
rg -n 'fig:wmval|three panels of the validation figure in the main paper' paper/main.tex paper/aaai2027-unified-supp.tex
rg -n 'fig:supp-wmval' paper/aaai2027-unified-supp.tex
```

Expected: the first command finds the old main-paper figure and old supplementary wording; the second returns no match.

- [ ] **Step 2: Replace the main-paper validation block**

Use this compact paragraph and remove the entire `figure` environment:

```latex
\paragraph{Dynamics Model Validation.}
Before any policy training, we evaluate the finetuned dynamics model on held-out
episodes through multi-step imagined-rollout quality, action controllability---including
a negative probe with deliberately perturbed actions---and consistency between imagined
and real potential trajectories. Full protocols, metrics, and visualizations are reported
in Supplementary Sec.~C.4.4. Because the dynamics model serves only as the training
environment rather than an algorithmic contribution, we defer these detailed diagnostics
to the supplementary material.
\todo{P0: after validation is complete, report one headline held-out metric here.}
```

- [ ] **Step 3: Make Supplementary Sec. C.4.4 own the figure**

Change the opening sentence to end with:

```latex
before training any policy against it, along three axes; these are the three
panels of Figure~\ref{fig:supp-wmval}.
```

After the existing P0 TODO, add:

```latex
\begin{figure*}[t]
\centering
\framebox[0.95\textwidth]{\rule{0pt}{4.5cm}
\todo{P0: dynamics-model validation---(i) $H$-step rollout quality,
(ii) action controllability including the negative probe, and
(iii) potential-trajectory consistency}}
\caption{Validation of the finetuned dynamics model on held-out episodes.
The three panels report multi-step rollout quality, action controllability including
deliberately perturbed actions, and imagined/real potential-trajectory consistency.
\todo{P0: replace the placeholder and report the held-out split and metrics.}}
\label{fig:supp-wmval}
\end{figure*}
```

- [ ] **Step 4: Run post-change structural assertions**

```bash
rg -n 'fig:wmval|three panels of the validation figure in the main paper' paper/main.tex paper/aaai2027-unified-supp.tex
rg -n 'Supplementary Sec.~C.4.4|fig:supp-wmval' paper/main.tex paper/aaai2027-unified-supp.tex
```

Expected: the first command returns no match. The second finds the compact main-paper pointer and the supplementary reference/label.

- [ ] **Step 5: Compile the main paper**

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all commands exit `0`; `main.pdf` is produced with no `fig:wmval` reference.

- [ ] **Step 6: Compile the supplementary material**

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error aaai2027-unified-supp.tex
bibtex aaai2027-unified-supp
pdflatex -interaction=nonstopmode -halt-on-error aaai2027-unified-supp.tex
pdflatex -interaction=nonstopmode -halt-on-error aaai2027-unified-supp.tex
```

Expected: all commands exit `0`; `aaai2027-unified-supp.pdf` is produced and `fig:supp-wmval` resolves.

- [ ] **Step 7: Audit logs and rendered text**

```bash
rg -n -i '(^!|undefined references|citation.*undefined|reference.*undefined)' paper/main.log paper/aaai2027-unified-supp.log
rg -n -i 'overfull' paper/main.log paper/aaai2027-unified-supp.log
pdftotext -layout paper/main.pdf /tmp/shore-main.txt
pdftotext -layout paper/aaai2027-unified-supp.pdf /tmp/shore-supp.txt
rg -n 'Dynamics Model Validation|Supplementary Sec. C.4.4' /tmp/shore-main.txt
rg -n 'Validating the Model Before Any Policy Training|Validation of the finetuned dynamics model' /tmp/shore-supp.txt
```

Expected: the first log scan finds no errors or undefined references/citations. The
overfull scan shows no new warning in the modified validation blocks; the supplement's
pre-existing warning around source lines 352--371 may remain. The main PDF contains the
compact pointer, and the supplementary PDF contains the detailed validation subsection
and figure caption.

- [ ] **Step 8: Review the scoped source changes**

```bash
git diff --check -- paper/main.tex paper/aaai2027-unified-supp.tex
```

Expected: no whitespace errors. Only the approved validation blocks change.
