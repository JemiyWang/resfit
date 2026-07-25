# Figure 3 ResFit Legend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Figure 3 baseline legend from `Residual RL` to `ResFit` and make the caption use the same name.

**Architecture:** Preserve the existing `base` data key and all numerical and visual settings. Test the shared legend text through the existing fake-W&B plotting regression, then change only user-facing strings, regenerate the exact PDF used by `main.tex`, and recompile the paper.

**Tech Stack:** Python 3, pytest, Matplotlib, W&B API, LaTeX/pdflatex, Poppler PDF tools.

## Global Constraints

- Every visible Figure 3 occurrence of `Residual RL` becomes `ResFit`.
- The Figure 3 caption uses `ResFit` without the obsolete legend-name parenthetical.
- Preserve the internal `base` key, W&B runs, curve data, uncertainty bands, colors, line styles, frozen-base references, panel order, axes, and all other legend entries.
- Regenerate `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`.

---

### Task 1: Test and change the user-facing baseline name

**Files:**
- Modify: `tests/test_figure3_frozen_base_line.py:72-73`
- Modify: `paper/plot_pouring_lifttray_seeds.py:1-2,40`

**Interfaces:**
- Consumes: the shared legend built from `STY["base"]["label"]`.
- Produces: a Figure 3 shared legend containing `ResFit` and no `Residual RL`.

- [ ] **Step 1: Extend the regression test before changing production code**

Replace the final legend assertion with:

```python
    legend_text = [text.get_text() for text in namespace["fig"].legends[0].texts]
    assert "ResFit" in legend_text
    assert "Residual RL" not in legend_text
    assert legend_text[-1] == "Frozen base"
```

- [ ] **Step 2: Run the focused test and observe the expected failure**

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  tests/test_figure3_frozen_base_line.py -q -W ignore::DeprecationWarning
```

Expected: failure because the current legend contains `Residual RL` and not
`ResFit`.

- [ ] **Step 3: Change only the plotting display strings**

Change the script description and the `base` style entry to:

```python
"""Long-horizon anti-collapse (seed-level): SHORE-RL (ours, staged_joint) vs
ResFit, DSRL, IQL, and IBRL.
```

```python
 "base": {"label":"ResFit", "short":"ResFit",
          "color":"#8a8a86", "ls":(0,(4,2)), "lw":2.5, "z":2},
```

- [ ] **Step 4: Run the focused test and syntax check**

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  tests/test_figure3_frozen_base_line.py -q -W ignore::DeprecationWarning
/mnt/mnt/data/envs/resfit-libero/bin/python -m py_compile \
  paper/plot_pouring_lifttray_seeds.py tests/test_figure3_frozen_base_line.py
```

Expected: `1 passed`; syntax check exits zero.

- [ ] **Step 5: Commit the tested source change**

```bash
git add paper/plot_pouring_lifttray_seeds.py \
  tests/test_figure3_frozen_base_line.py
git commit -m "fig: rename Figure 3 baseline to ResFit"
```

### Task 2: Update the caption, regenerate Figure 3, and compile

**Files:**
- Modify: `paper/main.tex:453-456`
- Modify: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`
- Test: `paper/main.pdf`

**Interfaces:**
- Consumes: the tested plotting strings and existing W&B panel definitions.
- Produces: the main-paper Figure 3 and caption consistently using `ResFit`.

- [ ] **Step 1: Remove the obsolete caption parenthetical**

Change the caption passage to:

```tex
with $\pm 1$ s.e.m. bands. SHORE-RL is compared with ResFit, DSRL, IBRL,
and full-policy IQL. The thin gray dashed reference in each panel marks that
task's mean SHORE-RL success at zero environment steps, before residual
learning.}
```

- [ ] **Step 2: Regenerate the exact Figure 3 asset**

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python \
  paper/plot_pouring_lifttray_seeds.py \
  --output "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
```

Expected: exit zero and five numeric frozen-base values.

- [ ] **Step 3: Verify the standalone PDF text and render it**

Run:

```bash
pdftotext \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" -
pdftocairo -png -singlefile -r 130 \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" \
  /tmp/figure3-resfit-legend
```

Expected: extracted text contains `ResFit`, contains no `Residual RL`, and the
rendered legend is unclipped.

- [ ] **Step 4: Compile the paper twice**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both commands exit zero and generate a nine-page `main.pdf`.

- [ ] **Step 5: Verify the compiled page and focused tests**

Run from the repository root:

```bash
pdftotext -f 5 -l 5 paper/main.pdf -
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  tests/test_figure3_frozen_base_line.py -q -W ignore::DeprecationWarning
git diff --check
```

Expected: page 5 contains `ResFit` and no `Residual RL`; test reports
`1 passed`; no whitespace errors.

- [ ] **Step 6: Commit the regenerated paper integration**

```bash
git add paper/main.tex \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
git commit -m "paper: use ResFit in Figure 3 caption"
```
