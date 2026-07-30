# Real-Robot Learning-Curve Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redraw the real-robot learning curves with task names aligned to Figure 7 and method-centric styling aligned to the staged-vs-self-derived reference figure.

**Architecture:** Keep the recorded percentage data as the single source of truth in `plot_real_robot_curves.py`, expose a small plotting contract for deterministic tests, and render success-rate fractions without changing any values. Update only the generated PDF/PNG and the corresponding main-paper caption.

**Tech Stack:** Python 3, Matplotlib, `unittest`, pdfTeX, Poppler `pdftotext`.

## Global Constraints

- Preserve all six checkpoints and every recorded success-rate value.
- Use panel order: Paper-Roll Placement, Block Assembly, Cup Stacking.
- Encode methods, not tasks: SHORE-RL is green solid and ResFit is blue dash-dot.
- Use a two-entry legend centered below the panels and no point markers.
- Use `Env steps (k)` from 0 to 500 and `Eval success rate` from 0.0 to 0.5.
- Keep the main-paper caption consistent with the rendered visual encoding.

---

### Task 1: Lock the Figure Contract with Tests

**Files:**
- Create: `test_plot_real_robot_curves.py`
- Modify: `plot_real_robot_curves.py`

**Interfaces:**
- Consumes: existing `STEPS` and `DATA` percentage-valued series.
- Produces: `TASK_ORDER`, `TASK_LABELS`, `METHOD_STYLES`, `success_fractions(task, method)`, and `plot_figure(out_pdf, out_png)`.

- [ ] **Step 1: Write the failing contract test**

Create `test_plot_real_robot_curves.py`:

```python
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

import plot_real_robot_curves as curves


EXPECTED_DATA = {
    "paper": {
        "SHORE-RL": [22, 18, 30, 38, 44, 42],
        "ResFit": [22, 18, 22, 26, 28, 28],
    },
    "cup": {
        "SHORE-RL": [12, 10, 24, 32, 30, 32],
        "ResFit": [12, 8, 10, 6, 2, 2],
    },
    "block": {
        "SHORE-RL": [10, 6, 16, 24, 30, 28],
        "ResFit": [10, 4, 0, 2, 0, 0],
    },
}


class FigureContractTest(unittest.TestCase):
    def test_task_order_and_names_match_figure_seven(self):
        self.assertEqual(curves.TASK_ORDER, ("paper", "block", "cup"))
        self.assertEqual(
            [curves.TASK_LABELS[key] for key in curves.TASK_ORDER],
            ["Paper-Roll Placement", "Block Assembly", "Cup Stacking"],
        )

    def test_method_styles_match_reference_figure(self):
        self.assertEqual(
            curves.METHOD_STYLES,
            {
                "SHORE-RL": {
                    "color": "#008300",
                    "ls": "-",
                    "lw": 2.6,
                    "z": 6,
                },
                "ResFit": {
                    "color": "#1f5fa9",
                    "ls": (0, (4.5, 1.2, 1, 1.2)),
                    "lw": 2.3,
                    "z": 7,
                },
            },
        )

    def test_percentage_data_are_preserved_and_displayed_as_fractions(self):
        self.assertEqual(curves.DATA, EXPECTED_DATA)
        for task, methods in EXPECTED_DATA.items():
            for method, values in methods.items():
                self.assertEqual(
                    curves.success_fractions(task, method),
                    [value / 100.0 for value in values],
                )

    def test_rendered_figure_has_three_panels_two_lines_and_no_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            out_pdf = Path(directory) / "curves.pdf"
            out_png = Path(directory) / "curves.png"
            fig = curves.plot_figure(out_pdf, out_png)

            self.assertGreater(out_pdf.stat().st_size, 1_000)
            self.assertGreater(out_png.stat().st_size, 1_000)
            self.assertEqual(len(fig.axes), 3)
            for axis, task in zip(fig.axes, curves.TASK_ORDER):
                self.assertEqual(axis.get_title(), curves.TASK_LABELS[task])
                self.assertEqual(len(axis.get_lines()), 2)
                self.assertTrue(
                    all(line.get_marker() in ("None", None, "") for line in axis.get_lines())
                )
                self.assertEqual(axis.get_xlabel(), "Env steps (k)")
            self.assertEqual(fig.axes[0].get_ylabel(), "Eval success rate")
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the contract is initially absent**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m unittest test_plot_real_robot_curves.py -v
```

Expected: FAIL because `TASK_ORDER`, `METHOD_STYLES`, `success_fractions`, and `plot_figure` do not yet all exist.

- [ ] **Step 3: Refactor the plotting script around the tested contract**

In `plot_real_robot_curves.py`, retain `STEPS` and `DATA` unchanged and define:

```python
TASK_ORDER = ("paper", "block", "cup")
TASK_LABELS = {
    "paper": "Paper-Roll Placement",
    "block": "Block Assembly",
    "cup": "Cup Stacking",
}
METHOD_STYLES = {
    "SHORE-RL": {"color": "#008300", "ls": "-", "lw": 2.6, "z": 6},
    "ResFit": {
        "color": "#1f5fa9",
        "ls": (0, (4.5, 1.2, 1, 1.2)),
        "lw": 2.3,
        "z": 7,
    },
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
FIGSIZE_INCHES = (3.7 * len(TASK_ORDER), 3.65)


def success_fractions(task, method):
    return [value / 100.0 for value in DATA[task][method]]
```

Replace the current plotting body with `plot_figure(out_pdf, out_png)`. It must:

```python
def plot_figure(out_pdf, out_png):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    fig, axes = plt.subplots(
        1, len(TASK_ORDER), figsize=FIGSIZE_INCHES, sharey=True, squeeze=False
    )
    axes = axes[0]
    handles = {}
    x_values = [step / 1000.0 for step in STEPS]

    for axis, task in zip(axes, TASK_ORDER):
        for method in ("SHORE-RL", "ResFit"):
            style = METHOD_STYLES[method]
            line, = axis.plot(
                x_values,
                success_fractions(task, method),
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                zorder=style["z"],
                solid_capstyle="round",
            )
            handles[method] = line
        axis.set_title(TASK_LABELS[task], fontsize=16, fontweight="normal", loc="left", pad=6)
        axis.set_xlabel("Env steps (k)", fontsize=16)
        axis.set_xlim(0, 500)
        axis.set_ylim(0.0, 0.5)
        axis.xaxis.set_major_locator(MultipleLocator(100))
        axis.yaxis.set_major_locator(MultipleLocator(0.1))
        axis.tick_params(axis="both", labelsize=14, width=0.8, length=3.5)
        axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Eval success rate", fontsize=17)
    fig.legend(
        [handles["SHORE-RL"], handles["ResFit"]],
        ["SHORE-RL", "ResFit"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.4)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    return fig
```

Make `main()` resolve the existing `figure` directory, call `plot_figure`, and close the returned figure:

```python
def main():
    out_dir = Path(__file__).parent / "figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_figure(
        out_dir / "fig_real_robot_curves.pdf",
        out_dir / "fig_real_robot_curves.png",
    )
    plt.close(fig)
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python -m unittest test_plot_real_robot_curves.py -v
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit the tested plotting contract**

```bash
git add plot_real_robot_curves.py test_plot_real_robot_curves.py
git commit -m "paper: align real-robot curve styling"
```

---

### Task 2: Regenerate the Figure and Integrate It into the Paper

**Files:**
- Modify: `figure/fig_real_robot_curves.pdf`
- Modify: `figure/fig_real_robot_curves.png`
- Modify: `main.tex:600-605`

**Interfaces:**
- Consumes: `plot_figure(out_pdf, out_png)` and the fixed plotting contract from Task 1.
- Produces: publication artifacts and a caption that describes their method-centric encoding.

- [ ] **Step 1: Regenerate the PDF and PNG**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python plot_real_robot_curves.py
```

Expected: exit code 0 and updated `figure/fig_real_robot_curves.pdf` plus `figure/fig_real_robot_curves.png`.

- [ ] **Step 2: Verify the generated PDF labels and axis text**

Run:

```bash
pdftotext -layout figure/fig_real_robot_curves.pdf -
```

Expected text includes, in panel order:

```text
Paper-Roll Placement
Block Assembly
Cup Stacking
Eval success rate
Env steps (k)
SHORE-RL
ResFit
```

The extracted legend must contain only one `SHORE-RL` entry and one `ResFit` entry.

- [ ] **Step 3: Update the main-paper caption**

Replace the current Figure 8 caption with:

```tex
\caption{\textbf{Real-robot learning curves.} Evaluation success rate at six
training checkpoints for \textsc{Paper-Roll Placement}, \textsc{Block Assembly},
and \textsc{Cup Stacking}. Green solid lines denote SHORE-RL, and blue dash-dot
lines denote ResFit.}
```

- [ ] **Step 4: Run source and formatting checks**

Run:

```bash
git diff --check -- plot_real_robot_curves.py test_plot_real_robot_curves.py main.tex
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Compile the paper**

Run:

```bash
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

Expected: exit code 0 and `Output written on main.pdf`.

- [ ] **Step 6: Check for unresolved LaTeX diagnostics**

Run:

```bash
rg -n "undefined|Citation.*undefined|Reference.*undefined|LaTeX Error" main.log
```

Expected: exit code 1 with no matches.

- [ ] **Step 7: Verify the compiled caption**

Run:

```bash
pdftotext -layout main.pdf - | rg -n "Real-robot learning curves|Paper-Roll Placement|Block Assembly|Cup Stacking"
```

Expected: the compiled Figure 8 caption contains the three aligned task names and the new method-centric encoding description.

- [ ] **Step 8: Commit the generated artifacts and paper integration**

```bash
git add main.tex figure/fig_real_robot_curves.pdf figure/fig_real_robot_curves.png
git commit -m "paper: regenerate real-robot learning curves"
```
