# Figure 3 Font and Legend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Figure 3 text render near the paper's 10 pt body size and shorten its four legend labels.

**Architecture:** Keep the existing five-panel plot, data retrieval, curve geometry, and
output path unchanged. Change only the centralized `STY` labels and explicit Matplotlib
font-size settings, then regenerate the existing PDF.

**Tech Stack:** Python, Matplotlib, Weights & Biases API, LaTeX/AAAI 2027 style

## Global Constraints

- Modify `paper/plot_pouring_lifttray_seeds.py` and regenerate
  `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf`.
- Do not change data, colors, line styles, panel order, axis ranges, or the paper caption.
- Final legend labels must be exactly `SHORE-RL`, `Residual RL`, `DSRL`, and `IQL`.
- At the PDF's 7-inch rendered width, primary figure text should be approximately 9--10 pt.

---

### Task 1: Update and regenerate Figure 3

**Files:**
- Modify: `paper/plot_pouring_lifttray_seeds.py:27-35,148-197`
- Modify: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf`

**Interfaces:**
- Consumes: Existing `STY` dictionary, plot construction, and W&B run IDs.
- Produces: The same PDF path with updated typography and legend text.

- [ ] **Step 1: Run a failing static regression assertion**

```bash
python - <<'PY'
from pathlib import Path
s = Path("paper/plot_pouring_lifttray_seeds.py").read_text()
for label in ('"label":"SHORE-RL"', '"label":"Residual RL"',
              '"label":"DSRL"', '"label":"IQL"'):
    assert label in s, label
assert '"font.size":22' in s
assert 'fontsize=22' in s
assert 'labelsize=20' in s
PY
```

Expected: `AssertionError` because the old labels still contain parenthetical descriptions
and the source font sizes are 10--13 pt.

- [ ] **Step 2: Apply the minimal source update**

Set the four `STY[*]["label"]` strings to the exact required names. Set the global font to
22 pt, panel titles and axis labels to 22 pt, ticks to 20 pt, frozen-base/pending
annotations to 20 pt, and the legend to 22 pt. These source sizes render at approximately
8.5--9.4 pt after the existing 0.426 paper scaling.

- [ ] **Step 3: Re-run the static assertion**

Run the command from Step 1.

Expected: exit code 0 with no output.

- [ ] **Step 4: Regenerate and inspect the standalone figure**

```bash
cd paper
python plot_pouring_lifttray_seeds.py
pdfinfo figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf
pdftoppm -png -r 120 -singlefile \
  figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf \
  /tmp/figure3-font-preview
```

Expected: script prints `wrote ...fig_long_horizon_anti_collapse_multiseed_curves.pdf`;
`pdfinfo` reports one page; the preview has no clipped or overlapping labels.

- [ ] **Step 5: Compile and inspect the paper**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -f 1 -l 20 -png -r 120 main.pdf /tmp/main-page
```

Expected: `latexmk` exits 0. Locate Figure 3 in the rendered pages and verify that its
labels remain readable without clipping or overlap.

- [ ] **Step 6: Review the diff**

```bash
git diff --check -- paper/plot_pouring_lifttray_seeds.py
git diff --stat -- paper/plot_pouring_lifttray_seeds.py \
  paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf
```

Expected: no whitespace errors; only the plotting source and regenerated Figure 3 PDF are
part of the implementation change.
