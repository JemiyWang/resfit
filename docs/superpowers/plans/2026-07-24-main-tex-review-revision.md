# SHORE-RL Main-Text Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise `paper/main.tex` so its claims, contributions, notation, experiment narrative, names, captions, and conclusion satisfy the 24 confirmed review comments.

**Architecture:** Make four reviewable editing passes over the single LaTeX source: paper-level framing, method rigor, experiment/caption alignment, and final verification. Preserve all numerical results and figure assets while replacing inconsistent prose and reducing repetition.

**Tech Stack:** LaTeX, `rg`, `git diff`, `pdflatex`, BibTeX

## Global Constraints

- Modify only `paper/main.tex` during implementation.
- Preserve all reported experimental values, figure assets, bibliography entries, and unrelated working-tree changes.
- Do not claim that every baseline improves before collapsing.
- Use a bounded mechanism analysis, not a formal causal theorem.
- Use late-training collapse as the temporal phenomenon and final-window success rate as the quantitative summary.
- Use `ResFit` as the prose name for the flat demo-anchored residual baseline.
- Do not redraw or rerun experiments in this text-editing pass.
- Do not stage or commit `paper/main.tex`, because it is pre-existing untracked user work.

---

### Task 1: Align the Paper-Level Claim, Contributions, and Background

**Files:**
- Modify: `paper/main.tex:53-239`

**Interfaces:**
- Consumes: The confirmed central claim and metric contract.
- Produces: Consistent terminology and definitions used by Method and Experiments.

- [ ] **Step 1: Record the pre-edit lexical failures**

Run:

```bash
rg -n "improves initially|initial-to-final|peak success|flat demo-anchored residual|Long-Horizon Anti-Collapse|no-anchoer|improving beyond" paper/main.tex
```

Expected: matches in the Abstract, Introduction, contributions, evaluation text, and experiment results.

- [ ] **Step 2: Rewrite the Abstract and Introduction**

Apply these exact semantic changes:

```text
Central result: SHORE-RL achieves high and sustained success rates on
long-horizon manipulation tasks, while the evaluated RL-finetuning methods
collapse below the frozen base policy.

Mechanism boundary: long sparse reward delay gives each residual correction
weak local evidence, while critic error and residual drift can accumulate over
the remainder of the trajectory; this motivates the design but is not claimed
as a uniquely identified cause.

Evaluation contract: learning curves expose late-training collapse and
final-window success rate summarizes sustained performance.
```

Correct the local grammar and spelling errors `way to improving` and
`no-anchoer`. Remove any sentence asserting that residual methods universally
improve before collapsing.

- [ ] **Step 3: Rewrite the three contributions**

Make the three items state:

```text
1. An empirical long-horizon failure pattern plus the bounded sparse-credit,
   critic-error, and residual-drift mechanism analysis.
2. The first waypoint-conditioned frozen-base residual RL method in the
   related-work scope, presented as a base-policy-agnostic plug-in for frozen
   cloned policies.
3. Experimental coverage across long-horizon tasks, two base-policy classes,
   the learned-potential comparison, and the real-robot extension.
```

Use final-window success rate in Item 1 and avoid listing experiment sections
as if they were algorithmic contributions.

- [ ] **Step 4: Rebalance Related Work**

In the temporal-abstraction paragraph, foreground the gap:

```text
Prior hierarchical and waypoint methods learn a high-level target together
with a learned low-level controller. Existing RL-finetuning methods for frozen
imitation policies instead adapt actions, latent noise, or action selection
without using waypoints to shorten the residual objective horizon.
```

Then describe SHORE-RL in one sentence. Compress the world-model paragraph to
its role as a training-environment extension.

- [ ] **Step 5: Tighten Preliminaries**

Remove the duplicated temporal-span sentence. Define `g` as the task-goal
representation. Change the residual definition to include:

```latex
a_t^{\mathrm{res}}\sim
\pi_{\mathrm{res}}(\cdot\mid o_{\leq t},a_t^b)
```

before the executed-action equation, then state that SHORE-RL adds `z_t` to
this conditioning set. Keep only standard goal-conditioned-value background.

- [ ] **Step 6: Check Task 1**

Run:

```bash
rg -n "way to improving|no-anchoer|improves initially|initial-to-final|peak success|flat demo-anchored residual" paper/main.tex
```

Expected: no matches.

---

### Task 2: Make Method Notation Rigorous and Compress the Extension

**Files:**
- Modify: `paper/main.tex:242-387`

**Interfaces:**
- Consumes: `g`, `\pi_b`, and the flat residual definition from Preliminaries.
- Produces: Definition-before-use for `z_t`, `\pi_h`, `\pi_{\mathrm{res}}`,
  `u`, `V_{\mathrm{gc}}`, `V_{\mathrm p}`, and `D`.

- [ ] **Step 1: Add the Figure 1 reference and plug-in framing**

The Method opening must cite `Fig.~\ref{fig:framework}` and describe SHORE-RL
as wrapping, rather than modifying, a frozen base policy.

- [ ] **Step 2: Correct the residual-policy expression**

Replace the bare conditional-density expression with the sampling statement:

```latex
a_t^{\mathrm{res}}\sim
\pi_{\mathrm{res}}(\cdot\mid o_{\leq t},a_t^b,z_t).
```

Keep `z_t=\pi_h(e_t,g)\in\mathbb{R}^{10}` and ensure `g` has already been
defined.

- [ ] **Step 3: Integrate online joint finetuning into Section 4.2**

Keep the material where the goal-conditioned value and navigator are defined,
but remove the separate paragraph heading. State in sequence: offline
initialization, online/offline transition mixture, frozen base/encoder, and the
exclusive role of `V_{\mathrm{gc}}` in waypoint selection.

- [ ] **Step 4: Define update index and reduce value-name repetition**

Introduce `u` exactly as:

```text
Let u denote the residual-training gradient-update index.
```

Use it in `r_{t,u}^{\Phi}`, `\Phi_u`, and `V_{\mathrm p,u}`. Explain the
functional distinction between `V_{\mathrm{gc}}` and `V_{\mathrm p}` once,
then remove repeated comparisons from the latter half of Section 4.3.

- [ ] **Step 5: Compress Section 4.4**

Replace the multi-paragraph learned-dynamics pipeline with one natural
paragraph that defines `D`, pooled mixed-quality finetuning data, frozen
imagined rollouts, re-encoding through `\psi`, bounded rollout recursion,
imagination replay, and the self-derived surrogate reward. Preserve the
extension boundary but avoid retelling Sections 4.1--4.3.

- [ ] **Step 6: Check Task 2**

Run:

```bash
rg -n "\\\\paragraph\\{Online joint finetuning|\\\\paragraph\\{Imagined states|\\\\paragraph\\{Residual RL in imagination|\\\\paragraph\\{The reward is the potential|At residual-training update \\$u\\$" paper/main.tex
```

Expected: no matches.

Run:

```bash
rg -n "Let \\$u\\$ denote|a_t\\^\\{\\\\mathrm\\{res\\}\\}\\\\sim|fig:framework" paper/main.tex
```

Expected: all three definitions/references are present in Method.

---

### Task 3: Align Experiments, Names, Task Order, and Captions

**Files:**
- Modify: `paper/main.tex:392-713`

**Interfaces:**
- Consumes: The final-window metric contract and method names.
- Produces: Figure/Table prose consistent with rendered Figures 3, 5, 6, and 7.

- [ ] **Step 1: Simplify Tasks and Evaluation**

Define only final-window success as the mean over the last 20 percent of
evaluations with seed variation. State that learning curves expose temporal
behavior. Remove initial-to-final and peak-success comparisons.

- [ ] **Step 2: Standardize ResFit**

Rename the flat demo-anchored residual baseline to:

```latex
\emph{(ii)~ResFit}~\cite{resfit2025residual}
```

Use `ResFit` throughout prose and captions. In Figure 3's caption state that
the plot's “Residual RL” legend entry denotes ResFit.

- [ ] **Step 3: Rewrite the main result paragraph**

Rename it:

```latex
\paragraph{Sustained Long-Horizon Performance.}
```

Coordinate the prose with Figure 3 and Table 1. Emphasize SHORE-RL's high,
sustained performance and the baselines' low final-window performance. Use the
short-horizon CanSort curve only as a contrast in temporal composition. Do not
reference any ablation figure or table from this paragraph.

- [ ] **Step 4: Rewrite Figure 3 caption**

Describe success versus environment steps on the four displayed long-horizon
tasks and CanSort; state `2--3` seed means and `\pm1` s.e.m. bands; map
“Residual RL” to ResFit. Remove the absent open-circle description.

- [ ] **Step 5: Restrict the ablation protocol paragraph**

Make the first Section 5.2 paragraph cover only:

```text
five matched arms; common frozen features, data, action scale, seeds, and
budget; the three selected long-horizon tasks; component coverage; and
final-window success aggregation.
```

Do not describe Figure 5 results. Do not call the stage bonus “the potential.”
Explain task coverage positively as tasks spanning the tested
stage-structured long-horizon settings.

- [ ] **Step 6: Align Table 2 and Figure 5**

Replace every caption use of `Full` with `SHORE-RL`. Remove subsection
references from captions. In the Component Ablations paragraph, remove the
`.1--.3` numerical range and use only figure-supported qualitative
comparisons. Discuss final-window success only. Retain Figure 5's open-circle
description because the rendered ThreePiece panel contains one.

- [ ] **Step 7: Align Figure 6**

Change “Per-seed eval success” to success versus environment steps with
seed-mean lines and `\pm1` s.e.m. bands. Keep the two displayed arms and the
stage-shaping versus self-derived-shaping distinction.

- [ ] **Step 8: Align real-robot prose and Figure 7**

Use this order in the task paragraph, table rows, and caption:

```text
pick_paper_roll, build_block, pick_cup
```

Use this matching description order:

```text
precision picking of a rolling object, assembly-style block stacking, and cup
grasping and placement
```

Write Figure 7 panels as `(a)`, `(b)`, and `(c)` in that order. Remove the
caption's subsection reference and draft marker.

- [ ] **Step 9: Compress the real-robot pipeline**

Use approximately `1,000` success-only teleoperated episodes per task. Remove
the randomization draft marker. Compress the preparation and imagined-training
pipeline without losing the separate per-task bases, shared dynamics model,
pooled expert/base-rollout data, self-derived reward, and 50-trial evaluation.
Delete the sentence disclaiming the extension as evidence for the simulation
claim.

- [ ] **Step 10: Check Task 3**

Run:

```bash
rg -n "Full|initial-to-final|peak success|Long-Horizon Anti-Collapse|flat demo-anchored|hand-designed per-stage bonus as the potential|because nothing collapses|\\.1\\$--\\$\\.3|pick\\\\_paper\\\\_roll.*pick\\\\_cup.*build\\\\_block|sim_task|sec:exp-real.*pick" paper/main.tex
```

Expected: no matches.

Run:

```bash
rg -n "Sustained Long-Horizon Performance|Residual RL.*ResFit|1\\{,\\}000|pick\\\\_paper\\\\_roll.*build\\\\_block.*pick\\\\_cup" paper/main.tex
```

Expected: all required revised forms are present.

---

### Task 4: Fix the Conclusion and Verify the Complete Manuscript

**Files:**
- Modify: `paper/main.tex:716-727`
- Verify: `paper/main.tex`

**Interfaces:**
- Consumes: All revised paper-level terminology.
- Produces: A compiling, internally consistent main manuscript.

- [ ] **Step 1: Rewrite the Conclusion**

Attribute collapse to the evaluated existing RL-finetuning baselines, not to a
generic category mislabeled as ResFit. Say that the main simulation study uses
a hand-designed per-stage bonus, without the phrase `as the potential`.
Retain the self-derived-value result within the two tested stage-structured
tasks. Remove the final draft marker.

- [ ] **Step 2: Run lexical and structure checks**

Run:

```bash
rg -n "initial-to-final|peak success|Full|Long-Horizon Anti-Collapse|as the potential|pick\\\\_paper\\\\_roll.*pick\\\\_cup.*build\\\\_block|\\\\todo\\{confirm|pending the remaining" paper/main.tex
```

Expected: no matches.

Run:

```bash
rg -n "^\\\\section|^\\\\subsection|^\\\\paragraph|\\\\caption" paper/main.tex
```

Expected: all intended sections, result paragraphs, and captions appear once.

- [ ] **Step 3: Check the diff**

Run:

```bash
git diff --no-index /dev/null paper/main.tex
git diff --check --no-index /dev/null paper/main.tex
```

Expected: the first command displays the complete untracked manuscript; the
second reports no whitespace errors.

- [ ] **Step 4: Compile**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all commands exit 0 and produce `paper/main.pdf`.

- [ ] **Step 5: Inspect LaTeX diagnostics**

Run:

```bash
rg -n "Undefined|Citation.*undefined|Reference.*undefined|LaTeX Error|Emergency stop|multiply defined" main.log
pdfinfo main.pdf
```

Expected: no undefined citations/references or LaTeX errors; `pdfinfo` reports a
valid PDF.

- [ ] **Step 6: Review all 24 comments against the final source**

Read the final changed passages in Abstract, Introduction, Contributions,
Related Work, Preliminaries, Method 4.1--4.4, Experiments 5.1--5.3, all five
requested captions, Table 2 caption, and Conclusion. Confirm every symbol is
defined before use and no later ablation is cited from the Section 5.1 main
result paragraph.
