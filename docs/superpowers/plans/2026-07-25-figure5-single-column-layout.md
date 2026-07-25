# Figure 5 Single-Column Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reformat Figure 5 as a legible one-column grouped bar chart while preserving all data and statistical definitions.

**Architecture:** Keep the existing W&B inventory and fixed-window aggregation unchanged. Replace only the Matplotlib rendering layer with one grouped axis, then change the LaTeX float from double-column to single-column and validate the rendered paper page.

**Tech Stack:** Python 3, Matplotlib, unittest, LaTeX/pdflatex, pdfinfo, pdftocairo

## Global Constraints

- Preserve the two tasks: `Pouring` and `ThreePiece`.
- Preserve the five existing ablation arms, order, colors, values, and plus/minus one standard-error bars.
- Render at approximately one AAAI column width: 3.35 inches.
- Use approximately 8 pt for axis/task labels and 7 pt for ticks and legend.
- Use a single-column `figure` with `\includegraphics[width=\columnwidth]`.
- Preserve the Figure 5 label, caption semantics, and body references.

---

### Task 1: Single-column grouped chart

**Files:**
- Modify: `paper/test_plot_ablation_bars_400k.py`
- Modify: `paper/plot_ablation_bars_400k.py`
- Regenerate: `paper/figure/fig_ablation_bars_400k.pdf`
- Regenerate: `paper/figure/fig_ablation_bars_400k.png`

**Interfaces:**
- Consumes: `plot_summaries(summaries, out_pdf, out_png)`, the existing summary mapping `{task: {arm: (mean, sem, seed_values)}}`.
- Produces: `FIGSIZE_INCHES: tuple[float, float]` and a one-axis grouped PDF/PNG with two task groups and five bars per group.

- [ ] **Step 1: Add a failing geometry test**

Add imports and assertions that lock the physical layout:

```python
from paper.plot_ablation_bars_400k import FIGSIZE_INCHES

def test_plot_uses_single_column_physical_size(self):
    self.assertEqual(FIGSIZE_INCHES[0], 3.35)
    self.assertLessEqual(FIGSIZE_INCHES[1], 3.0)
```

Extend the artifact test after `plot_summaries(...)`:

```python
image = plt.imread(out_png)
self.assertLess(image.shape[1], 700)
self.assertGreater(image.shape[1], 500)
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper.test_plot_ablation_bars_400k.FigureContractTest -v
```

Expected: FAIL because `FIGSIZE_INCHES` is not yet defined.

- [ ] **Step 3: Implement the grouped chart**

In `paper/plot_ablation_bars_400k.py`, define:

```python
FIGSIZE_INCHES = (3.35, 2.75)
```

Replace the side-by-side `plt.subplots(1, len(PANELS), ...)` rendering with one
axis. Use task centers `[0.0, 1.0]`, a total group width of `0.76`, and offsets
derived from the five arms:

```python
fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
task_names = list(PANELS)
centers = list(range(len(task_names)))
bar_width = 0.76 / len(LEGEND_ORDER)

for arm_index, arm in enumerate(LEGEND_ORDER):
    offset = (arm_index - (len(LEGEND_ORDER) - 1) / 2) * bar_width
    xs = [center + offset for center in centers]
    means = [summaries[task][arm][0] for task in task_names]
    sems = [summaries[task][arm][1] for task in task_names]
    ax.bar(xs, means, width=bar_width * 0.90, ...)
    ax.errorbar(xs, means, yerr=sems, fmt="none", ...)
```

Set `font.size=8`, x/y task labels to 8 pt, ticks and legend to 7 pt, sparse
0--1 y ticks, and a two-column legend below the plot. Keep the caption-only
description of the 330k--400k window and use the concise plotted y label
`Success rate`.

- [ ] **Step 4: Run tests and regenerate the figure**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -v
/mnt/mnt/data/resfit/rise_venv/bin/python paper/plot_ablation_bars_400k.py
pdfinfo paper/figure/fig_ablation_bars_400k.pdf
```

Expected: all tests PASS; the PDF has one page approximately 241 points wide,
and the script reports both PDF and PNG outputs.

- [ ] **Step 5: Commit the chart change**

```bash
git add paper/plot_ablation_bars_400k.py \
  paper/test_plot_ablation_bars_400k.py \
  paper/figure/fig_ablation_bars_400k.pdf \
  paper/figure/fig_ablation_bars_400k.png
git commit -m "fig: reformat Figure 5 for one column"
```

### Task 2: Single-column paper integration

**Files:**
- Modify: `paper/main.tex`
- Regenerate: `paper/main.pdf`

**Interfaces:**
- Consumes: `paper/figure/fig_ablation_bars_400k.pdf`.
- Produces: Figure 5 as a one-column float under label `fig:ablbars`.

- [ ] **Step 1: Add a source-contract failure check**

Run before editing:

```bash
python - <<'PY'
from pathlib import Path
source = Path("paper/main.tex").read_text()
block = source[source.index(r"\includegraphics[width=\textwidth]{figure/fig_ablation_bars_400k.pdf}") - 40:]
assert r"\begin{figure}[t]" in block[:100]
assert r"\includegraphics[width=\columnwidth]{figure/fig_ablation_bars_400k.pdf}" in block[:300]
PY
```

Expected: FAIL because the source still uses `figure*` and `\textwidth`.

- [ ] **Step 2: Change only the Figure 5 float geometry**

Replace:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure/fig_ablation_bars_400k.pdf}
...
\end{figure*}
```

with:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figure/fig_ablation_bars_400k.pdf}
...
\end{figure}
```

Keep the caption text and `\label{fig:ablbars}` unchanged.

- [ ] **Step 3: Run source checks and compile twice**

Run:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
rg -n "undefined references|multiply defined|fig:ablcurves" main.log
```

Expected: both compiles exit 0 and the reference search returns no matches.

- [ ] **Step 4: Render and inspect the Figure 5 page**

Locate Figure 5 with:

```bash
pdftotext -layout main.pdf /tmp/resfit-main-layout.txt
rg -n "Ablation A: matched component ablation" /tmp/resfit-main-layout.txt
```

Rasterize the matching page:

```bash
pdftocairo -f 1 -l 9 -png -r 150 main.pdf /tmp/resfit-main-page
```

Inspect the matching PNG and verify the graphic stays inside one column, labels
are readable, the legend is not clipped, and nearby text/figures do not
overlap.

- [ ] **Step 5: Run final verification**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -v
git diff --check
pdfinfo paper/main.pdf
```

Expected: all tests PASS, no whitespace errors, and `paper/main.pdf` is a valid
compiled PDF.
