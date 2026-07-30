# Algorithm 1 Critic-Target Notation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Algorithm 1's oversized line-12 target equation with compact pseudocode that emphasizes sampling two of ten target critics and taking their minimum.

**Architecture:** Keep the algorithm and training procedure unchanged. Define a compact minimum target-critic estimate on line 12, then state that the existing n-step TD target uses this estimate when regressing all ten critics.

**Tech Stack:** LaTeX, pdfLaTeX

## Global Constraints

- Modify only the line-12 critic-target presentation and its adjacent regression statement.
- Preserve uniform sampling of two distinct critics from the ten-critic ensemble.
- Preserve the n-step TD target and regression of all ten critics.

---

### Task 1: Compact the line-12 notation

**Files:**
- Modify: `main.tex:397-403`
- Test: `main.pdf`

**Interfaces:**
- Consumes: target critic ensemble `\bar Q_{1:10}` and the existing n-step TD update.
- Produces: compact notation `\bar Q_{\min}` used by the following target-construction statement.

- [ ] **Step 1: Replace the expanded target equation**

Use the following compact pseudocode:

```latex
$12$ & \quad\quad Sample two distinct target critics uniformly:\newline
\makebox[\linewidth][c]{$i,j\sim\operatorname{Unif}(\{1,\ldots,10\}),\ i\neq j;\quad
\bar Q_{\min}\gets\min(\bar Q_i,\bar Q_j)$}\newline
\hspace*{2em}Form the $n$-step TD target with $\bar Q_{\min}$ and regress $Q_{1:10}$ \\
```

- [ ] **Step 2: Compile the main paper**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 and `main.pdf` generated.

- [ ] **Step 3: Inspect Algorithm 1**

Render or extract the algorithm page and confirm that line 12 is visibly shorter, the formula does not overflow, and the statements still express uniform sampling, distinct indices, minimum aggregation, and n-step critic regression.

### Task 2: Simplify the target-network update

**Files:**
- Modify: `main.tex:394-396`
- Test: `main.pdf`

**Interfaces:**
- Consumes: the existing critic-target update and delayed actor-target update.
- Produces: one concise pseudocode statement with the same update schedule.

- [ ] **Step 1: Replace the equations with text**

Use the following line:

```latex
$16$ & \quad\quad Soft-update critic targets and, on delayed steps, actor targets \\
```

- [ ] **Step 2: Compile and inspect the main paper**

Run:

```bash
