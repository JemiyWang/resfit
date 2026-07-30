# SHORE-RL De-similarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebrand `main.tex` from HiRes-RL to SHORE-RL and reframe the prose so a reviewer does not read the paper as a mechanical HIQL + ResFiT mashup — without weakening any citation, changing any result, or denying that the method is (mechanically) two-level.

**Architecture:** A single LaTeX file (`main.tex`) is edited in seven ordered batches. Batches 2–5 are targeted prose edits written against the *current* vocabulary (`subgoal`, `high-level actor`); the global terminology sweep (Batch 6) runs *last* so it normalizes both the original text and the newly written prose in one pass. Every batch ends with a grep check + a clean `pdflatex` compile. No `\label`/`\ref`/`\cite` names change, so cross-references and the bibliography stay intact and a single `pdflatex` pass resolves everything.

**Tech Stack:** LaTeX (AAAI-2027 `aaai2027.sty`, natbib, pre-generated `main.bbl`), `pdflatex`, `grep`/`sed`.

## Global Constraints

- **Working file:** `/mnt/mnt/data/resfit/paper/main.tex` (only this file changes).
- **Final title (verbatim):** `SHORE-RL: Shortening the Horizon for Long-Horizon Residual Reinforcement Learning`
- **Acronym (for reference, do not print a gloss unless a task says so):** **S**hortening the **HO**rizon for … **RE**sidual RL → SHORE.
- **Do NOT change any `\label{}`, `\ref{}`, `\cite{}`, or `\bibliography` names.** Only prose, `\section`/`\subsection`/`\paragraph` *titles*, and the method-name string change. This keeps every cross-reference resolved.
- **Do NOT touch numbers, tables, figures, `\includegraphics`, captions' data, results, or `\todo` markers' technical content.** Force = medium (identity/wording only).
- **Honesty red line:** every `\cite{park2023hiql}` and `\cite{resfit2025residual}` stays; every factual "we build on / we adopt HIQL's objective" sentence stays. The method IS two-level — keep mechanism descriptions (`two-level controller`, `low-level residual`, the 12 `hierarch*` that describe mechanism). We de-*brand* hierarchy as the identity; we do not hide it.
- **No blind sed.** Before any `sed`, `cp main.tex main.tex.bak`, preview with `grep -n`, apply, then re-grep counts and compile. `main.tex.bak` is the per-batch undo.
- **Compile check each batch:** `pdflatex -halt-on-error -interaction=nonstopmode main` in `/mnt/mnt/data/resfit/paper`; expect exit 0, `Output written on main.pdf (10 pages`, and no *new* `LaTeX Warning: ... undefined`, `multiply defined`, or `Undefined control sequence`.
- **Baseline before Batch 1:** `HiRes-RL`=42, `subgoal*`=62, `hierarch*`=12, `high-level`=11.

---

### Task 1: Rename the method string and the title

**Files:**
- Modify: `main.tex` (title lines 45–46; 42 occurrences of `HiRes-RL` throughout)

**Interfaces:**
- Consumes: nothing.
- Produces: the string `SHORE-RL` (42×) and the new `\title{...}`; later tasks reference `SHORE-RL` in prose but their edits target `hierarchical`/`subgoal` substrings that do not include the name, so they are order-independent from this task.

- [ ] **Step 1: Back up and preview**

```bash
cd /mnt/mnt/data/resfit/paper
cp main.tex main.tex.bak
grep -c "HiRes-RL" main.tex          # expect 42
grep -n "\\\\title{" main.tex        # confirm the title is at line 45
```

- [ ] **Step 2: Replace the title (exact Edit, not sed — the title spans a `\\` line break)**

Replace:
```latex
\title{HiRes-RL: Hierarchical Residual Reinforcement Learning for Stable\\
Finetuning of Cloned Manipulation Policies}
```
with:
```latex
\title{SHORE-RL: Shortening the Horizon for Long-Horizon Residual\\
Reinforcement Learning}
```

- [ ] **Step 3: Global-replace the remaining name occurrences**

```bash
sed -i 's/HiRes-RL/SHORE-RL/g' main.tex
```

- [ ] **Step 4: Verify counts**

```bash
grep -c "HiRes-RL" main.tex          # expect 0
grep -c "SHORE-RL" main.tex          # expect 41 (42 minus the title, which no longer contains the bare token)
```
Expected: `HiRes-RL`=0. (Title now reads "SHORE-RL:"; the 41 body occurrences plus the title's `SHORE-RL` = 42 `SHORE-RL` tokens total — `grep -c "SHORE-RL"` counts lines, so accept ≥41; do a visual `grep -n "SHORE-RL" main.tex | head`.)

- [ ] **Step 5: Compile**

```bash
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p1.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p1.log
```
Expected: `exit=0`, `Output written on main.pdf (10 pages`, no undefined/multiply/Undefined-control lines. Then `rm main.tex.bak`.

---

### Task 2: De-brand hierarchy at the four headline positions

**Files:**
- Modify: `main.tex` lines 166, 240, 277, 285 (identity/title positions only)

**Interfaces:**
- Consumes: `SHORE-RL` string from Task 1 (but all four edits target `hierarchical`/`hierarchy` substrings, so they apply regardless of Task 1).
- Produces: subsection title `Horizon-Shortening Residual Policy` (Batch 6 and any later reference must match this spelling).

Keep `two-level controller` and all mechanism-level `hierarch*` untouched — only these four identity/title strings change.

- [ ] **Step 1: Back up**

```bash
cd /mnt/mnt/data/resfit/paper && cp main.tex main.tex.bak
```

- [ ] **Step 2: Contribution #1 — reframe the framework label (line 166–167)**

Replace:
```latex
\item \textbf{SHORE-RL, a hierarchical residual framework for stable long-horizon
finetuning.} We reduce a long-horizon task to a sequence of self-derived latent subgoals:
```
with:
```latex
\item \textbf{SHORE-RL, a horizon-shortening residual framework for stable long-horizon
finetuning.} We \emph{shorten the horizon} the residual must span, reducing a long-horizon
task to a sequence of nearby, self-derived subgoals:
```

- [ ] **Step 3: Preliminaries paragraph header (line 240)**

Replace:
```latex
\paragraph{Goal-conditioned value and hierarchy.}
```
with:
```latex
\paragraph{Goal-conditioned value and subgoal extraction.}
```

- [ ] **Step 4: Method roadmap — de-brand the policy label (line 277)**

Replace (the substring only):
```latex
(\S\ref{sec:hier}) a hierarchical residual policy
```
with:
```latex
(\S\ref{sec:hier}) a horizon-shortening residual policy
```

- [ ] **Step 5: Subsection title (line 285)**

Replace:
```latex
\subsection{Hierarchical Residual Policy}
```
with:
```latex
\subsection{Horizon-Shortening Residual Policy}
```

- [ ] **Step 6: Verify and compile**

```bash
grep -c "hierarchical residual" main.tex        # expect 0
grep -c "and hierarchy" main.tex                # expect 0
grep -n "Horizon-Shortening Residual Policy" main.tex   # expect 1 (line ~285)
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p2.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p2.log
```
Expected: exit=0, 10 pages, clean. Then `rm main.tex.bak`.

---

### Task 3: Abstract + Intro — phenomenon-first hook, name gloss

**Files:**
- Modify: `main.tex` abstract (lines 55–60) and intro (line 122)

**Interfaces:**
- Consumes: `SHORE-RL` (Task 1).
- Produces: no new labels; prose only.

Goal: (a) stop opening the abstract with ResFiT's textbook definition — open with *our* finding; (b) gloss the name where SHORE-RL is introduced so "shorten the horizon" reads as the identity. The four differentiators (frozen base features / no privileged state / residual on a frozen base / cures long-horizon collapse) already appear in the abstract body and intro — do not delete them.

- [ ] **Step 1: Back up**

```bash
cd /mnt/mnt/data/resfit/paper && cp main.tex main.tex.bak
```

- [ ] **Step 2: Abstract opening (lines 55–60) — phenomenon-first**

Replace:
```latex
Residual reinforcement learning (RL) finetunes a behavior-cloning (BC) policy by
freezing the base and learning a small additive residual off-policy. On short-horizon
manipulation this recipe works well; on long-horizon tasks, however, it does not merely
fail to improve---it \emph{degrades} the base, with success rates falling below the frozen
base policy and collapsing as training continues (critic over-estimation and residual drift
over a long, sparse-reward horizon).
```
with:
```latex
Freezing a cloned base policy and learning a small additive off-policy residual is an
effective way to finetune short-horizon manipulation. We find that this recipe breaks
down as the horizon grows: on long-horizon tasks it does not merely fail to improve---it
\emph{degrades} the base, with success rates falling below the frozen base policy and
collapsing as training continues (critic over-estimation and residual drift over a long,
sparse-reward horizon).
```

- [ ] **Step 3: Intro — gloss the name at its introduction (line 122)**

Replace:
```latex
We present \textbf{SHORE-RL}. Our starting point is the observation above: the flat
```
with:
```latex
We present \textbf{SHORE-RL}, which \emph{shortens the horizon} the residual must span---
turning one long-horizon task into a sequence of short ones so the residual always faces a
nearby target. Our starting point is the observation above: the flat
```

- [ ] **Step 4: Verify and compile**

```bash
grep -c "shortens the horizon" main.tex     # expect >=1
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p3.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p3.log
```
Expected: exit=0, 10 pages, clean. Then `rm main.tex.bak`.

---

### Task 4: Preliminaries — explicit "we adopt HIQL's objective" attribution

**Files:**
- Modify: `main.tex` line 241 (attribution sentence before the two HIQL equations)

**Interfaces:**
- Consumes: nothing new.
- Produces: prose only. The two display equations (`eq:hiql-v`, `eq:hiql-hi`) and their labels stay — they are needed for reproducibility and are honestly attributed; we only sharpen the attribution so the equations read as adopted background, not as our derivation.

- [ ] **Step 1: Back up**

```bash
cd /mnt/mnt/data/resfit/paper && cp main.tex main.tex.bak
```

- [ ] **Step 2: Sharpen the attribution (line 241)**

Replace:
```latex
To turn one long horizon into a sequence of short ones we follow HIQL~\cite{park2023hiql}
and learn an offline goal-conditioned value $V(s,g)$, where the goal $g$ is encoded through
a low-dimensional bottleneck $\phi([g,s])$.
```
with:
```latex
As the offline machinery for proposing subgoals we adopt, unchanged, the training
objective of HIQL~\cite{park2023hiql}: an offline goal-conditioned value $V(s,g)$ whose
goal $g$ is encoded through a low-dimensional bottleneck $\phi([g,s])$. We restate it here
only to fix notation; the contribution of \S\ref{sec:hier}--\S\ref{sec:pot} is what we do
with it, not the objective itself.
```

- [ ] **Step 3: Verify and compile**

```bash
grep -c "we adopt, unchanged, the training" main.tex   # expect 1
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p4.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p4.log
```
Expected: exit=0, 10 pages, clean. Then `rm main.tex.bak`.

---

### Task 5: Related Work — reposition HIQL from "we build on" to "tool for a problem it never faced"

**Files:**
- Modify: `main.tex` lines 222–224 (inside the `Hierarchical and goal-conditioned RL` paragraph)

**Interfaces:**
- Consumes: nothing new.
- Produces: prose only; `\cite{park2023hiql}` retained.

- [ ] **Step 1: Back up**

```bash
cd /mnt/mnt/data/resfit/paper && cp main.tex main.tex.bak
```

- [ ] **Step 2: Reposition (lines 222–224)**

Replace:
```latex
goal-conditioned value and a high-level policy over subgoals. SHORE-RL builds on this idea
and puts a \emph{self-derived} hierarchy to work on a problem this line has not addressed:
the long-horizon collapse of residual finetuning. Our subgoal is read from the base policy's
```
with:
```latex
goal-conditioned value and a high-level policy over subgoals. We \emph{use} this value
machinery as a tool rather than claim it as a contribution: SHORE-RL puts it to work on a
setting this line never studied---residual finetuning of a \emph{frozen} base policy, where
a small residual can drift and collapse over a long horizon, a failure mode absent from
offline goal-conditioned RL, which has no base policy and no residual. Our subgoal is read
from the base policy's
```

- [ ] **Step 3: Verify and compile**

```bash
grep -c "builds on this idea" main.tex     # expect 0
grep -c "use} this value" main.tex         # expect 1
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p5.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p5.log
```
Expected: exit=0, 10 pages, clean. Then `rm main.tex.bak`.

---

### Task 6: Global terminology sweep — subgoal→waypoint, high-level actor/policy/module→navigator

**Files:**
- Modify: `main.tex` (all remaining `subgoal*`/`Subgoal*`; the `high-level actor|policy|module` phrases)

**Interfaces:**
- Consumes: all prior edits (this runs last so it normalizes newly written prose too).
- Produces: final vocabulary. `waypoint` replaces `subgoal`; `navigator` replaces `high-level {actor,policy,module}`. The math symbol `z`, `\pi_h`, and the technical term `high-level advantage $A_h$` (line ~256) are preserved.

**Design note (surface to reviewer, not a blocker):** HIQL itself uses "waypoint" once as a synonym for subgoal, so this swap distances us from HIQL's *most repeated* word (`subgoal`) but not from its vocabulary entirely; the gain is breaking the exact `subgoal + high-level policy + goal-conditioned value` triple-phrase match, and `waypoint` fits the "shorten the horizon / short legs" framing. If zero-overlap is wanted later, the same sweep retargets to `leg`/`beacon` trivially.

- [ ] **Step 1: Back up and confirm no label/cite collisions**

```bash
cd /mnt/mnt/data/resfit/paper && cp main.tex main.tex.bak
grep -nE "\\\\(label|ref|cite)\{[^}]*subgoal" main.tex    # expect no output (safe to sweep)
grep -ciE "subgoal" main.tex                              # record N (pre-sweep count)
```

- [ ] **Step 2: Sweep subgoal→waypoint (both cases)**

```bash
sed -i 's/subgoal/waypoint/g; s/Subgoal/Waypoint/g' main.tex
grep -ci "subgoal" main.tex        # expect 0
grep -ci "waypoint" main.tex       # expect == N from Step 1
```

- [ ] **Step 3: Replace the high-level phrases with navigator (targeted, review each)**

```bash
grep -n "high-level" main.tex      # review all; convert actor/policy/module, KEEP "high-level advantage"
```
Apply, via individual Edits, exactly these mappings wherever they occur:
- `high-level actor $\pi_h$` → `navigator $\pi_h$`
- `high-level actor` → `navigator`
- `high-level policy` → `navigator` (in *our*-method sentences; in the HIQL-attribution sentence at line ~250 and the Related-Work line ~222 describing HIQL's own "high-level policy", leave HIQL's term as-is since it quotes HIQL)
- `high-level module` → `navigator`
- **KEEP unchanged:** `high-level advantage $A_h$` (line ~256) and any `$A_h$`/`\pi_h` symbols.

```bash
grep -nE "high-level (actor|module)" main.tex   # expect 0
grep -c "navigator" main.tex                    # expect ~9-11
```

- [ ] **Step 4: Compile and page check**

```bash
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/p6.log 2>&1; echo "exit=$?"
grep -E "Output written|undefined|multiply|Undefined control" /tmp/p6.log
```
Expected: exit=0, `10 pages`, clean. Then `rm main.tex.bak`.

---

### Task 7: Final full-file verification and two-pass compile

**Files:**
- Read-only verification of `main.tex`; produce `main.pdf`.

- [ ] **Step 1: Sweep-consistency greps**

```bash
cd /mnt/mnt/data/resfit/paper
grep -c "HiRes-RL" main.tex                       # 0
grep -ci "subgoal" main.tex                        # 0
grep -nE "high-level (actor|module)" main.tex      # (none)
grep -n "hierarchical\|hierarchy" main.tex         # only mechanism-level mentions remain; no identity/title use
grep -n "SHORE-RL" main.tex | wc -l                # ~41 body + title
```

- [ ] **Step 2: Confirm no `\label`/`\ref`/`\cite` names moved**

```bash
grep -oE "\\\\label\{[^}]*\}" main.tex | sort > /tmp/labels_after.txt
# compare against the pre-change set if captured; otherwise confirm counts:
grep -c "\\\\label{" main.tex      # unchanged vs baseline
grep -c "\\\\ref{" main.tex        # unchanged vs baseline
```
Expected: label/ref/cite name sets identical to baseline (only titles/prose changed).

- [ ] **Step 3: Clean two-pass compile**

```bash
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/f1.log 2>&1; echo "exit1=$?"
pdflatex -halt-on-error -interaction=nonstopmode main >/tmp/f2.log 2>&1; echo "exit2=$?"
grep -E "Output written|Warning.*undefined|multiply defined|Undefined control|Citation.*undefined" /tmp/f2.log
```
Expected: both exit 0, `Output written on main.pdf (10 pages`, no undefined/multiply/citation warnings.

- [ ] **Step 4: Coherence read-through**

Read the abstract, Contributions, Method roadmap (§277), §sec:hier title, Preliminaries §240–259, and the Related-Work HIQL paragraph. Confirm: (a) "shorten the horizon" reads as the identity; (b) every mechanism claim still sits next to the residual/frozen-features/collapse differentiators; (c) no sentence now denies the method is two-level; (d) HIQL/ResFiT citations all present. Note any awkward seams for a follow-up polish pass.

---

## Self-Review

**Spec coverage:**
- Rename 42× + title → Task 1. ✓
- subgoal→waypoint (62) → Task 6. ✓
- high-level actor→navigator (11) → Task 6. ✓
- De-brand hierarchy, keep mechanism → Task 2 (+ red line in Global Constraints). ✓
- 4 differentiators next to mechanism claims → preserved/added in Tasks 3 & 5 (abstract body + Related Work). ✓
- Preliminaries HIQL equations, explicit cite/compress → Task 4. ✓
- Related Work "tool, not contribution" → Task 5. ✓
- "Shorten horizon" anchor → Tasks 2 (labels), 3 (name gloss). ✓
- Medium force / no numbers-figures-results → Global Constraints. ✓
- Honesty (citations + factual sentences kept) → Global Constraints red line + Task 4/5 keep all `\cite`. ✓
- Per-batch grep + compile → every task Step "Verify and compile". ✓

**Placeholder scan:** none — every edit gives exact before/after LaTeX or an exact command with expected output. The one non-enumerated set (Task 6 Step 3 high-level phrases) is bounded by an exact grep and an exact mapping table with explicit keep-list.

**Type/name consistency:** the subsection title `Horizon-Shortening Residual Policy` (Task 2) is referenced by spelling only in Task 7; no `\ref` depends on it (label `sec:hier` unchanged). `navigator`/`waypoint` spellings are consistent across Tasks 5–7.
