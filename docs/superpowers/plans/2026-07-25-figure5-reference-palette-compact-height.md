# Figure 5 Reference Palette and Compact Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the reference image's muted palette to Figure 5, synchronize legend colors, and reduce the plotted height while preserving the figure's data and single-column placement.

**Architecture:** Keep the W&B inventory, fixed-window summaries, bar ordering, labels, and LaTeX integration unchanged. Modify only the Matplotlib style constants and compact geometry, then regenerate the PDF/PNG and compile the paper.

**Tech Stack:** Python 3, Matplotlib, unittest, pdflatex, pdfinfo, pdftocairo

## Global Constraints

- Preserve the two tasks, five arms, 28 seeds, 330k--400k metric, and plus/minus one standard-error bars.
- Preserve one-column width at 3.35 inches and 7 pt type.
- Set figure height to 2.30 inches.
- Use `SHORE-RL #FAD35B`, `w/o demo-BC + stage #539955`, `w/o stage shaping #ABACAB`, `w/o waypoint #CC8675`, and `w/o waypoint + stage #80AECA`.
- Legend patches must use the same `STY` mapping as the bars and retain row-major visual order.
- Use dark-gray bar outlines and error bars.

---

### Task 1: Reference palette and compact geometry

**Files:**
- Modify: `paper/test_plot_ablation_bars_400k.py`
- Modify: `paper/plot_ablation_bars_400k.py`
- Regenerate: `paper/figure/fig_ablation_bars_400k.pdf`
- Regenerate: `paper/figure/fig_ablation_bars_400k.png`

**Interfaces:**
- Consumes: existing `STY`, `LEGEND_ORDER`, `LEGEND_HANDLE_ORDER`, and `plot_summaries(summaries, out_pdf, out_png)`.
- Produces: `FIGSIZE_INCHES == (3.35, 2.30)` and exact semantic reference colors shared by bars and legend patches.

- [ ] **Step 1: Add failing palette and geometry tests**

Update the physical-size assertion:

```python
self.assertEqual(
    ablation_plot.FIGSIZE_INCHES,
    (3.35, 2.30),
)
```

Add an exact semantic palette test:

```python
def test_reference_palette_is_mapped_semantically(self):
    expected = {
        "full": "#FAD35B",
        "subgoal_only": "#539955",
        "no_staged": "#ABACAB",
        "no_subgoal": "#CC8675",
        "no_both": "#80AECA",
    }
    self.assertEqual(
        {arm: STY[arm]["color"] for arm in LEGEND_ORDER},
        expected,
    )
```

Extend the PNG artifact check so the real rendered image contains each exact
fill color:

```python
rgb = (image[:, :, :3] * 255).round().astype(int)
rendered_colors = {tuple(color) for color in rgb.reshape(-1, 3)}
for color in (STY[arm]["color"] for arm in LEGEND_ORDER):
    value = color.lstrip("#")
    expected_rgb = tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    self.assertIn(expected_rgb, rendered_colors)
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper.test_plot_ablation_bars_400k.FigureContractTest -v
```

Expected: FAIL because the current height is 2.75 inches and the current color
mapping differs from the reference palette.

- [ ] **Step 3: Implement the palette and shorter axes**

In `paper/plot_ablation_bars_400k.py`, set:

```python
FIGSIZE_INCHES = (3.35, 2.30)
BAR_EDGE_COLOR = "#4A4742"
```

Replace only the five `STY` colors with the exact mapping in Global
Constraints. Use `BAR_EDGE_COLOR` for both `ax.bar(..., edgecolor=...)` and
`ax.errorbar(..., ecolor=...)`. Preserve the bar and legend iteration orders.

Allocate enough space for the unchanged three-row legend while shortening the
axes:

```python
fig.legend(
    ...,
    bbox_to_anchor=(0.05, 0.015, 0.92, 0.26),
)
fig.subplots_adjust(
    left=0.17,
    right=0.985,
    top=0.985,
    bottom=0.34,
)
```

- [ ] **Step 4: Verify GREEN and regenerate artifacts**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -v
/mnt/mnt/data/resfit/rise_venv/bin/python paper/plot_ablation_bars_400k.py
pdfinfo paper/figure/fig_ablation_bars_400k.pdf
```

Expected: all tests PASS; the figure PDF remains approximately 3.2 inches wide
and becomes approximately 2.3 inches tall.

- [ ] **Step 5: Inspect standalone and paper renderings**

Inspect `paper/figure/fig_ablation_bars_400k.png` and verify that SHORE-RL and
its legend patch are yellow, all other mappings match the specification, no
label is clipped, and the axes are visibly shorter.

Compile and render the Figure 5 page:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
rg -n "undefined references|multiply defined|fig:ablcurves" main.log
pdftocairo -f 6 -l 6 -singlefile -png -r 150 \
  main.pdf /tmp/resfit-main-page6-reference-palette
```

Expected: both compiles exit 0; the reference search has no matches; Figure 5
remains inside the left column with a shorter chart and an unclipped legend.

- [ ] **Step 6: Run final checks and commit**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -v
git diff --check HEAD
sha256sum paper/figure/fig_ablation_bars_400k.pdf \
  paper/figure/fig_ablation_bars_400k.png paper/main.pdf
```

Expected: all tests PASS, no whitespace errors, and all three artifacts have
nonempty hashes.

Commit:

```bash
git add paper/plot_ablation_bars_400k.py \
  paper/test_plot_ablation_bars_400k.py \
  paper/figure/fig_ablation_bars_400k.pdf \
  paper/figure/fig_ablation_bars_400k.png
git commit -m "fig: match Figure 5 reference palette"
```
