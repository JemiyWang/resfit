# Table 3 Real-Robot Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unresolved main-paper Table 3 cells with frozen-base, ResFit, and SHORE-RL real-robot success rates derived from the existing learning-curve data.

**Architecture:** Keep `plot_real_robot_curves.py` as the numerical source of truth. Modify only the real-robot evaluation prose, Table 3, and caption in `main.tex`, preserving unrelated local edits and leaving the supplementary statistical protocol unchanged.

**Tech Stack:** LaTeX, pdfTeX, BibTeX, Poppler `pdftotext`, Git

## Global Constraints

- Do not push any commit to GitHub.
- Preserve all unrelated uncommitted changes in `main.tex`.
- Report the frozen base from step 0.
- Report ResFit and SHORE-RL as the mean of their 400k and 500k checkpoints.
- Do not invent seed-level standard errors or paired-trial statistics.

---

### Task 1: Fill and Validate Main-Paper Table 3

**Files:**
- Modify: `main.tex:644-674`
- Verify against: `plot_real_robot_curves.py:12-26`

**Interfaces:**
- Consumes: aggregate checkpoint success percentages in `DATA`
- Produces: a three-method real-robot comparison in Table 3

- [ ] **Step 1: Verify the source-data arithmetic**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -c "from paper.plot_real_robot_curves import DATA; order=('paper','block','cup'); print([(task, DATA[task]['SHORE-RL'][0]/100, sum(DATA[task]['ResFit'][-2:])/200, sum(DATA[task]['SHORE-RL'][-2:])/200) for task in order])"
```

Expected:

```text
[('paper', 0.22, 0.28, 0.43), ('block', 0.1, 0.0, 0.29), ('cup', 0.12, 0.02, 0.31)]
```

- [ ] **Step 2: Update the evaluation paragraph**

Replace the two-configuration description with:

```latex
\paragraph{Evaluation Configurations.}
The real-robot extension compares each task-specific frozen base, ResFit, and
the same base augmented by a SHORE-RL residual trained in imagination. Within
a task, all configurations use the same frozen base, observation interface,
and evaluation initial-state set. The frozen-base reference is the step-$0$
evaluation; for both adapted methods, final-window success averages the $400$k
and $500$k checkpoints.
```

- [ ] **Step 3: Replace Table 3**

Use:

```latex
\begin{table}[!t]
\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Task & Frozen Base & ResFit & \textbf{SHORE-RL} \\
\midrule
\textsc{Paper-Roll Placement} & .22 & .28 & \textbf{.43} \\
\textsc{Block Assembly}       & .10 & .00 & \textbf{.29} \\
\textsc{Cup Stacking}         & .12 & .02 & \textbf{.31} \\
\bottomrule
\end{tabular}
\caption{Real-robot extension results. The frozen-base column reports the
step-$0$ evaluation; ResFit and SHORE-RL report final-window success averaged
over the $400$k and $500$k checkpoints. Every checkpoint uses $50$ physical
trials from the fixed initial-state set. SHORE-RL is trained entirely inside
the finetuned dynamics model.}
\label{tab:realrobot}
\end{table}
```

- [ ] **Step 4: Check the edited source**

Run:

```bash
rg -n -C 12 'tab:realrobot|Evaluation Configurations|P1' main.tex
git diff --check -- main.tex
```

Expected: Table 3 contains all nine values, its nearby `P1` placeholders are absent, and `git diff --check` exits successfully.

- [ ] **Step 5: Compile the manuscript**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: every command exits with status 0 and `main.pdf` contains 9 pages unless unrelated local pagination changes intentionally alter the count.

- [ ] **Step 6: Verify compiled Table 3**

Run:

```bash
pdftotext -layout main.pdf /tmp/table3-main.txt
rg -n -C 8 'Frozen Base|Paper-Roll Placement|Block Assembly|Cup Stacking' /tmp/table3-main.txt
```

Expected: the extracted table contains `Frozen Base`, `ResFit`, `SHORE-RL`, and the three rows of approved values.

- [ ] **Step 7: Leave the implementation local**

Run:

```bash
git status --short --branch
```

Expected: the Table 3 edit remains local and no `git push` command is executed.
