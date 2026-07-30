# Figure 4 Full-Border Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add matching top and right borders to both panels of Figure 4 without changing its data, layout, or legend.

**Architecture:** Extend the existing PyMuPDF vector postprocessor to draw four border segments over the current two-panel source PDF. Regenerate the aligned PDF, rebuild the manuscript, and verify both the vector geometry and rendered page.

**Tech Stack:** Python, PyMuPDF, pdfLaTeX, pdftocairo

## Global Constraints

- Preserve the existing two-panel source figure and all plotted content.
- Use axis edge color `#52514e` and line width `0.8`.
- Match the exact endpoints of the existing left and bottom borders.
- Do not modify the manuscript caption or figure dimensions.
- Do not push changes to GitHub.

---

### Task 1: Add and verify Figure 4 borders

**Files:**
- Modify: `align_libero_figure.py`
- Regenerate: `figure/fig_libero10_aligned.pdf`
- Regenerate: `main.pdf`
- Test: `test_plot_ablation_bars_400k.py`
- Test: `test_plot_pouring_lifttray_seeds.py`

**Interfaces:**
- Consumes: `figure/fig_libero10.pdf` and the existing Figure 4 panel coordinates.
- Produces: `figure/fig_libero10_aligned.pdf` with top and right borders on both panels.

- [ ] **Step 1: Record the pre-change vector geometry**

Run:

```bash
python -c "import fitz; p=fitz.open('figure/fig_libero10_aligned.pdf')[0]; print([(d['width'], d['color'], d['rect']) for d in p.get_drawings() if d['width'] == 0.8])"
```

Expected: the output identifies the existing `0.8`-point `#52514e` left and bottom border segments for both panels, with no matching top and right segments.

- [ ] **Step 2: Add the vector border drawing**

Add this block in `align_libero_figure.py` after applying redactions and before saving:

```python
border_color = (0.3215686275, 0.3176470588, 0.3058823529)
panel_bounds = (
    (54.18437576293945, 258.4043884277344),
    (288.78436279296875, 493.0043640136719),
)
top = 21.32501220703125
bottom = 135.64500427246094

for left, right in panel_bounds:
    page.draw_line(
        fitz.Point(left, top), fitz.Point(right, top),
        color=border_color, width=0.8,
    )
    page.draw_line(
        fitz.Point(right, top), fitz.Point(right, bottom),
        color=border_color, width=0.8,
    )
```

- [ ] **Step 3: Regenerate Figure 4**

Run:

```bash
python align_libero_figure.py
```

Expected: exit code `0` and an updated `figure/fig_libero10_aligned.pdf`.

- [ ] **Step 4: Verify the generated vector geometry**

Run:

```bash
python -c "import fitz; p=fitz.open('figure/fig_libero10_aligned.pdf')[0]; lines=[item for d in p.get_drawings() if d['width'] is not None and abs(d['width']-0.8)<1e-6 and d['color'] and max(abs(a-b) for a,b in zip(d['color'],(0.3215686275,0.3176470588,0.3058823529)))<1e-3 for item in d['items'] if item[0]=='l']; print(len(lines)); assert len(lines) >= 8"
```

Expected: the command prints at least `8` and exits with code `0`, covering four borders per panel.

- [ ] **Step 5: Rebuild and test the paper**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
/mnt/mnt/data/envs/residual/bin/python -m pytest paper/test_plot_ablation_bars_400k.py paper/test_plot_pouring_lifttray_seeds.py -q
```

Expected: both LaTeX passes exit with code `0`, and all paper figure tests pass.

- [ ] **Step 6: Render and visually inspect Figure 4**

Run:

```bash
pdftocairo -f 6 -l 6 -singlefile -png -r 180 main.pdf /tmp/resfit-main-page6
```

Expected: the rendered manuscript page shows complete, consistent borders on both Figure 4 panels, with no clipping or changes to the data, legend, labels, or layout.

- [ ] **Step 7: Review the final diff**

Run:

```bash
git diff --check
git diff -- align_libero_figure.py
git status --short
```

Expected: no whitespace errors; the source diff is limited to the border drawing; generated files are updated; unrelated user changes remain untouched.
