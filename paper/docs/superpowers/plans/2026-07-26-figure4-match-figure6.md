# Figure 4 Match-Figure-6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct Figure 4 from its current vector paths and render it with Figure 6's canvas, typography, axes, grid, line, band, and legend style.

**Architecture:** `align_libero_figure.py` will treat `figure/fig_libero10.pdf` as the immutable result source, extract each panel's colored mean path and uncertainty polygon with PyMuPDF, convert PDF coordinates back to plot coordinates, and redraw them with Matplotlib. A focused unittest module will validate extraction, source preservation, output dimensions, and four-spine styling without requiring network access.

**Tech Stack:** Python 3, PyMuPDF, Matplotlib, unittest, pdfLaTeX, pdftocairo

## Global Constraints

- Preserve Figure 4's current data, two-panel content, colors, labels, and 400k-step range.
- Use a `7.4 x 3.65` inch two-panel canvas.
- Match Figure 6's DejaVu Serif typography and `17/16/17/14/14` base/title/y-label/tick/legend sizes.
- Use `#52514e` axis edges at width `0.8`, with all four spines visible.
- Use `#dcdcd7` horizontal grid lines at width `0.7`.
- Use SHORE-RL line width `2.6`, DSRL line width `2.3`, and uncertainty opacity `0.15`.
- Keep `main.tex`, Figure 4's caption, and all other figures unchanged.
- Do not push changes to GitHub.

---

### Task 1: Extract Figure 4 result paths

**Files:**
- Modify: `align_libero_figure.py`
- Create: `test_align_libero_figure.py`

**Interfaces:**
- Consumes: `extract_panel_series(source: pathlib.Path)`.
- Produces: `list[dict[str, dict[str, list[tuple[float, float]]]]]`, one dictionary per panel, where each method has `line` and `band` point lists in data coordinates.

- [ ] **Step 1: Write the failing extraction test**

Create `test_align_libero_figure.py`:

```python
import unittest

import align_libero_figure as figure4


class ExtractPanelSeriesTest(unittest.TestCase):
    def test_extracts_current_two_panel_results(self):
        panels = figure4.extract_panel_series(figure4.SOURCE)

        self.assertEqual(len(panels), 2)
        self.assertEqual(set(panels[0]), {"shore", "dsrl"})
        self.assertEqual(set(panels[1]), {"shore", "dsrl"})
        self.assertEqual(len(panels[0]["shore"]["line"]), 41)
        self.assertEqual(len(panels[0]["dsrl"]["line"]), 40)
        self.assertEqual(len(panels[1]["shore"]["line"]), 24)
        self.assertEqual(len(panels[1]["dsrl"]["line"]), 40)
        self.assertAlmostEqual(panels[0]["shore"]["line"][-1][0], 400.0, places=3)
        self.assertAlmostEqual(panels[0]["shore"]["line"][-1][1], 1.0, places=3)
        self.assertAlmostEqual(panels[1]["shore"]["line"][-1][0], 230.0, places=3)
        self.assertAlmostEqual(panels[1]["shore"]["line"][-1][1], 1.0, places=3)
        self.assertGreater(len(panels[0]["shore"]["band"]), 80)
        self.assertGreater(len(panels[0]["dsrl"]["band"]), 75)
        self.assertGreater(len(panels[1]["shore"]["band"]), 40)
        self.assertGreater(len(panels[1]["dsrl"]["band"]), 75)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the extraction test to verify RED**

Run:

```bash
python -m unittest test_align_libero_figure.ExtractPanelSeriesTest -v
```

Expected: `ERROR` because `align_libero_figure` has no
`extract_panel_series` function.

- [ ] **Step 3: Implement vector extraction**

Replace the PDF annotation logic in `align_libero_figure.py` with constants and
helpers equivalent to:

```python
PANEL_BOUNDS = (
    (54.18437576293945, 21.32501220703125,
     258.4043884277344, 135.64500427246094),
    (288.78436279296875, 21.32501220703125,
     493.0043640136719, 135.64500427246094),
)
METHODS = {
    "shore": {"color": (0.0, 0.5137255192, 0.0), "line_width": 2.6},
    "dsrl": {"color": (0.4784313738, 0.2470588237, 0.7098039389),
             "line_width": 2.3},
}
XLIM = (0.0, 400.0)
YLIM = (-0.03, 1.03)


def _same_color(actual, expected, tolerance=0.002):
    return actual is not None and max(
        abs(a - b) for a, b in zip(actual, expected)
    ) <= tolerance


def _path_points(drawing):
    points = []
    for item in drawing["items"]:
        if item[0] != "l":
            continue
        start, end = item[1], item[2]
        start_xy = (start.x, start.y)
        end_xy = (end.x, end.y)
        if not points or any(
            abs(a - b) > 1e-3 for a, b in zip(points[-1], start_xy)
        ):
            points.append(start_xy)
        points.append(end_xy)
    return points


def _to_data(point, bounds):
    left, top, right, bottom = bounds
    x = XLIM[0] + (point[0] - left) / (right - left) * (XLIM[1] - XLIM[0])
    y = YLIM[1] - (point[1] - top) / (bottom - top) * (YLIM[1] - YLIM[0])
    return x, y


def _belongs_to_panel(points, bounds):
    return points and abs(points[0][0] - bounds[0]) <= 0.01


def _clip_to_panel(points, bounds):
    left, top, right, bottom = bounds
    return [
        point for point in points
        if left - 1e-3 <= point[0] <= right + 1e-3
        and top - 1e-3 <= point[1] <= bottom + 1e-3
    ]


def extract_panel_series(source=SOURCE):
    document = fitz.open(source)
    if len(document) != 1:
        raise ValueError(f"expected one source page, found {len(document)}")
    page = document[0]
    panels = []
    for bounds in PANEL_BOUNDS:
        panel = {}
        for method, style in METHODS.items():
            line_paths = []
            band_paths = []
            for drawing in page.get_drawings():
                points = _path_points(drawing)
                if not _belongs_to_panel(points, bounds):
                    continue
                if (_same_color(drawing["color"], style["color"])
                        and drawing["width"] is not None
                        and drawing["width"] > 1.5):
                    line_paths.append(points)
                if (_same_color(drawing["fill"], style["color"])
                        and abs((drawing["fill_opacity"] or 0.0) - 0.15) < 0.01):
                    band_paths.append(points)
            if len(line_paths) != 1 or len(band_paths) != 1:
                raise ValueError(
                    f"{method}: expected one line and one band for panel {bounds}, "
                    f"found {len(line_paths)} and {len(band_paths)}"
                )
            line = [_to_data(point, bounds)
                    for point in _clip_to_panel(line_paths[0], bounds)]
            band = [_to_data(point, bounds)
                    for point in _clip_to_panel(band_paths[0], bounds)]
            panel[method] = {"line": line, "band": band}
        panels.append(panel)
    document.close()
    return panels
```

- [ ] **Step 4: Run the extraction test to verify GREEN**

Run:

```bash
python -m unittest test_align_libero_figure.ExtractPanelSeriesTest -v
```

Expected: `Ran 1 test` and `OK`.

---

### Task 2: Render Figure 4 with the Figure 6 style contract

**Files:**
- Modify: `align_libero_figure.py`
- Modify: `test_align_libero_figure.py`
- Regenerate: `figure/fig_libero10_aligned.pdf`
- Regenerate: `main.pdf`

**Interfaces:**
- Consumes: `render_figure(source: pathlib.Path, output: pathlib.Path, preview: pathlib.Path | None)`.
- Produces: a two-panel PDF with Figure 6's style and a temporary PNG preview.

- [ ] **Step 1: Write the failing render-style test**

Add to `test_align_libero_figure.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz


class RenderFigureTest(unittest.TestCase):
    def test_output_matches_figure6_size_and_has_two_panels(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "figure4.pdf"
            figure4.render_figure(figure4.SOURCE, output)
            page = fitz.open(output)[0]
            reference = fitz.open(
                figure4.PAPER
                / "figure"
                / "fig_staged_vs_pothiql_threading_piece_v2.pdf"
            )[0]

            self.assertLess(abs(page.rect.width - reference.rect.width), 3.0)
            self.assertLess(abs(page.rect.height - reference.rect.height), 3.0)
            text = page.get_text()
            self.assertIn("LIBERO-10 · Task 8", text)
            self.assertIn("LIBERO-90 · Task 57", text)
            self.assertIn("SHORE-RL (Ours)", text)
            self.assertIn("DSRL", text)
```

- [ ] **Step 2: Run the render test to verify RED**

Run:

```bash
python -m unittest test_align_libero_figure.RenderFigureTest -v
```

Expected: `ERROR` because `render_figure` does not exist.

- [ ] **Step 3: Implement Figure 6-aligned rendering**

Add Matplotlib rendering with these exact parameters:

```python
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
TITLES = ("LIBERO-10 · Task 8", "LIBERO-90 · Task 57")


def render_figure(source=SOURCE, output=OUTPUT, preview=None):
    panels = extract_panel_series(source)
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 17,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.65), sharey=True, squeeze=False
    )
    axes = axes[0]
    handles = {}
    for ax, title, panel in zip(axes, TITLES, panels):
        for method in ("shore", "dsrl"):
            style = METHODS[method]
            band = panel[method]["band"]
            line = panel[method]["line"]
            ax.fill(
                [point[0] for point in band],
                [point[1] for point in band],
                color=style["hex"],
                alpha=0.15,
                linewidth=0,
                zorder=2,
            )
            plotted, = ax.plot(
                [point[0] for point in line],
                [point[1] for point in line],
                color=style["hex"],
                linewidth=style["line_width"],
                solid_capstyle="round",
                zorder=4,
            )
            handles[method] = plotted
            if line[-1][0] < XLIM[1] - 1:
                ax.plot(
                    line[-1][0],
                    line[-1][1],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor=style["hex"],
                    markeredgewidth=1.4,
                    markersize=6,
                    zorder=5,
                )
        ax.set_title(title, fontsize=16, fontweight="normal", loc="left", pad=6)
        ax.set_xlabel("Env steps (k)", fontsize=16)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.tick_params(axis="both", labelsize=14, width=0.8, length=3.5)
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Eval success rate", fontsize=17)
    fig.legend(
        [handles["shore"], handles["dsrl"]],
        ["SHORE-RL (Ours)", "DSRL"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.4)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    render_figure(SOURCE, OUTPUT, PREVIEW)
    print("wrote", OUTPUT)
```

Add `"hex"` values `#008300` and `#7a3fb5` to the corresponding `METHODS`
entries and import `matplotlib`, `matplotlib.pyplot`, and `MultipleLocator`.

- [ ] **Step 4: Run all Figure 4 tests to verify GREEN**

Run:

```bash
python -m unittest test_align_libero_figure -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Regenerate Figure 4 and verify its PDF dimensions**

Run:

```bash
python align_libero_figure.py
pdfinfo figure/fig_libero10_aligned.pdf
pdfinfo figure/fig_staged_vs_pothiql_threading_piece_v2.pdf
```

Expected: both PDFs have widths and heights within 3 points of each other.

- [ ] **Step 6: Rebuild and test the manuscript**

Run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ..
/mnt/mnt/data/envs/residual/bin/python -m pytest \
  paper/test_plot_ablation_bars_400k.py \
  paper/test_plot_pouring_lifttray_seeds.py -q
```

Expected: both LaTeX passes exit `0`; the 15 existing plotting tests pass.

- [ ] **Step 7: Render and visually inspect the manuscript page**

Run:

```bash
pdftocairo -f 6 -l 6 -singlefile -png -r 180 \
  main.pdf /tmp/resfit-main-page6
```

Expected: Figure 4 and Figure 6 have aligned height, typography, axis styling,
grid, line emphasis, and legend sizing; Figure 4 has no clipping, overlaps,
path leakage, or data changes.

- [ ] **Step 8: Review and commit the focused change**

Run:

```bash
git diff --check
git diff -- align_libero_figure.py test_align_libero_figure.py
git status --short
git add align_libero_figure.py test_align_libero_figure.py \
  figure/fig_libero10_aligned.pdf
git commit -m "paper: align Figure 4 styling with Figure 6"
```

Expected: only the Figure 4 source, test, and generated PDF are included in
this local commit; unrelated user changes remain untouched and nothing is
pushed.
