# Figure 8 Larger-Markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase Figure 8's hollow circle and square node markers so they remain clearly visible at `0.70\textwidth`.

**Architecture:** Define named marker-size constants in `plot_real_robot_curves.py` and use them for both plotted methods. Add a focused unittest that locks the requested size while leaving all data and other style parameters unchanged.

**Tech Stack:** Python, unittest, Matplotlib, pdfLaTeX, pdftocairo

## Global Constraints

- Set marker size to `7.5` points.
- Set marker-edge width to `1.4` points.
- Apply the change equally to SHORE-RL circles and ResFit squares.
- Preserve hollow white marker fill, shapes, colors, curves, data, axes, fonts, legend, canvas size, caption, and `0.70\textwidth` insertion.
- Do not push changes to GitHub.

---

### Task 1: Enlarge and verify Figure 8 markers

**Files:**
- Modify: `plot_real_robot_curves.py`
- Create: `test_plot_real_robot_curves.py`
- Regenerate: `figure/fig_real_robot_curves.pdf`
- Regenerate: `figure/fig_real_robot_curves.png`
- Regenerate: `main.pdf`

**Interfaces:**
- Consumes: module constants `MARKER_SIZE` and `MARKER_EDGE_WIDTH`.
- Produces: both Figure 8 method traces with `7.5`-point hollow markers and `1.4`-point marker edges.

- [ ] **Step 1: Write the failing marker-style test**

Create `test_plot_real_robot_curves.py`:

```python
import unittest

import plot_real_robot_curves as figure8


class Figure8MarkerStyleTest(unittest.TestCase):
    def test_markers_are_large_enough_for_reduced_manuscript_figure(self):
        self.assertEqual(figure8.MARKER_SIZE, 7.5)
        self.assertEqual(figure8.MARKER_EDGE_WIDTH, 1.4)
        self.assertEqual(figure8.METHOD_STYLES["SHORE-RL"]["marker"], "o")
        self.assertEqual(figure8.METHOD_STYLES["ResFit"]["marker"], "s")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
python -m unittest test_plot_real_robot_curves -v
```

Expected: `ERROR` because `MARKER_SIZE` and `MARKER_EDGE_WIDTH` are not
defined.

- [ ] **Step 3: Add and use the marker constants**

Add:

```python
MARKER_SIZE = 7.5
MARKER_EDGE_WIDTH = 1.4
```

Replace:

```python
markersize=5.8,
markeredgewidth=1.25,
```

with:

```python
markersize=MARKER_SIZE,
markeredgewidth=MARKER_EDGE_WIDTH,
```

- [ ] **Step 4: Run the test to verify GREEN**

Run:

```bash
python -m unittest test_plot_real_robot_curves -v
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Regenerate Figure 8**

Run:

```bash
python plot_real_robot_curves.py
```

Expected: updated PDF and PNG files in `figure/`.

- [ ] **Step 6: Rebuild and test the paper**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -q
```

Expected: both LaTeX passes exit `0`; all 15 existing plotting tests pass.

- [ ] **Step 7: Render and visually inspect Figure 8**

Run:

```bash
pdftocairo -f 7 -l 7 -singlefile -png -r 180 \
  main.pdf /tmp/resfit-main-page7
```

Expected: all 36 nodes are visibly larger at `0.70\textwidth`, remain hollow,
retain circle/square identities, and do not clip or crowd labels.

- [ ] **Step 8: Review and locally commit the focused change**

Run:

```bash
git diff --check
git diff -- plot_real_robot_curves.py test_plot_real_robot_curves.py
git add plot_real_robot_curves.py test_plot_real_robot_curves.py \
  figure/fig_real_robot_curves.pdf figure/fig_real_robot_curves.png
git commit -m "paper: enlarge Figure 8 markers"
```

Expected: only Figure 8 source, test, and generated artifacts are committed;
unrelated user changes remain untouched and nothing is pushed.
