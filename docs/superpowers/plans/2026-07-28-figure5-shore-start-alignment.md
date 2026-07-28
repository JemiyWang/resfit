# Figure 5 SHORE-RL Start Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set the SHORE-RL mean start in both Figure 5 panels to exactly `0.600`, recenter only the corresponding step-zero uncertainty intervals, and preserve every other plotted value.

**Architecture:** Add a pure representation-aware alignment helper to the existing Figure 5 generator, then route both assembled SHORE-RL panel series through it before rendering. Keep extraction and plotting behavior unchanged for DSRL and for all points after step zero.

**Tech Stack:** Python 3.10, Matplotlib, PyMuPDF (`fitz`), `unittest`, pdfTeX

## Global Constraints

- Adjust only Figure 5.
- Set both SHORE-RL step-zero mean values to exactly `0.600`.
- Preserve the original step-zero s.e.m. width while recentering it on `0.600`.
- Preserve every SHORE-RL mean and uncertainty value after step zero exactly.
- Preserve DSRL data, axes, labels, legend, typography, layout, captions, and manuscript prose.
- Confirm `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf` retains SHA-256 `5830ff675a5388928a4b763f06be787b04e0bd82bef4354640a2c472ca894478`.
- Do not commit the implementation files: `paper/align_libero_figure.py`, `paper/test_align_libero_figure.py`, and `paper/figure/fig_libero10_aligned.pdf` are pre-existing untracked user files, so committing them would also capture unrelated user work already present in those files.

---

### Task 1: Specify the Alignment Contract with Focused Tests

**Files:**

- Modify: `paper/test_align_libero_figure.py`
- Test: `paper/test_align_libero_figure.py`

**Interfaces:**

- Consumes: `figure4.extract_panel_series(source)` and the two existing panel-series representations.
- Produces: test contract for `figure4.align_shore_start(series, target=0.6) -> dict` and `figure4.prepare_panels(source) -> list[dict]`.

- [ ] **Step 1: Add imports and tests for both data representations**

Add `copy` and the following test class:

```python
import copy


class AlignShoreStartTest(unittest.TestCase):
    def setUp(self):
        self.panels = figure4.extract_panel_series(figure4.SOURCE)

    def test_left_extracted_series_changes_only_step_zero_center(self):
        source = self.panels[0]["shore"]
        before = copy.deepcopy(source)

        adjusted = figure4.align_shore_start(source)

        self.assertEqual(source, before)
        self.assertAlmostEqual(adjusted["line"][0][0], 0.0, places=9)
        self.assertAlmostEqual(adjusted["line"][0][1], 0.6, places=12)
        self.assertEqual(adjusted["line"][1:], before["line"][1:])

        old_zero_band = [y for x, y in before["band"] if abs(x) < 1e-9]
        new_zero_band = [y for x, y in adjusted["band"] if abs(x) < 1e-9]
        self.assertAlmostEqual(
            max(new_zero_band) - min(new_zero_band),
            max(old_zero_band) - min(old_zero_band),
            places=12,
        )
        self.assertEqual(
            [(x, y) for x, y in adjusted["band"] if abs(x) >= 1e-9],
            [(x, y) for x, y in before["band"] if abs(x) >= 1e-9],
        )

    def test_right_array_series_changes_only_step_zero_center(self):
        source = self.panels[1]["shore"]
        before = copy.deepcopy(source)

        adjusted = figure4.align_shore_start(source)

        self.assertEqual(source, before)
        self.assertAlmostEqual(adjusted["x"][0], 0.0, places=9)
        self.assertAlmostEqual(adjusted["y"][0], 0.6, places=12)
        self.assertEqual(adjusted["y"][1:], before["y"][1:])
        self.assertEqual(adjusted["sem"], before["sem"])
        self.assertEqual(adjusted["observed"], before["observed"])
        self.assertAlmostEqual(adjusted["line"][0][1], 0.6, places=12)
        self.assertEqual(adjusted["line"][1:], before["line"][1:])

    def test_prepare_panels_aligns_only_shore(self):
        raw = figure4.extract_panel_series(figure4.SOURCE)
        prepared = figure4.prepare_panels(figure4.SOURCE)

        self.assertEqual(len(prepared), 2)
        for panel in prepared:
            series = panel["shore"]
            start = series["y"][0] if "sem" in series else series["line"][0][1]
            self.assertAlmostEqual(start, 0.6, places=12)
        self.assertEqual(prepared[0]["dsrl"], raw[0]["dsrl"])
        self.assertEqual(prepared[1]["dsrl"], raw[1]["dsrl"])

    def test_alignment_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "step zero"):
            figure4.align_shore_start({"line": [(10.0, 0.2)], "band": []})
        with self.assertRaisesRegex(ValueError, "target"):
            figure4.align_shore_start(
                {"line": [(0.0, 0.2)], "band": [(0.0, 0.1)]},
                target=1.1,
            )
```

- [ ] **Step 2: Run the focused tests and verify the new contract fails**

Run from `paper/`:

```bash
/mnt/mnt/data/envs/residual/bin/python -m unittest \
  test_align_libero_figure.AlignShoreStartTest -v
```

Expected: `ERROR` because `align_shore_start` and `prepare_panels` do not yet
exist.

---

### Task 2: Implement Pure Step-Zero Alignment

**Files:**

- Modify: `paper/align_libero_figure.py:14-17`
- Modify: `paper/align_libero_figure.py:273-326`
- Test: `paper/test_align_libero_figure.py`

**Interfaces:**

- Consumes: extracted series dictionaries in either `line`/`band` or `x`/`y`/`sem` form.
- Produces: `align_shore_start(series, target=0.6) -> dict` and `prepare_panels(source=SOURCE) -> list[dict]`.

- [ ] **Step 1: Add the fixed target and copy support**

Add the import and constant:

```python
import copy

SHORE_START_TARGET = 0.6
```

- [ ] **Step 2: Add the pure alignment helper**

Insert before `_plot_series`:

```python
def align_shore_start(series, target=SHORE_START_TARGET):
    target = float(target)
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be finite and within [0, 1]: {target}")

    adjusted = copy.deepcopy(series)
    if "sem" in adjusted:
        x = adjusted.get("x", [])
        y = adjusted.get("y", [])
        sem = adjusted.get("sem", [])
        if not x or len(x) != len(y) or len(x) != len(sem):
            raise ValueError("x, y, and sem must be nonempty and have matching lengths")
        if not math.isclose(float(x[0]), 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("SHORE-RL series must start at step zero")
        y[0] = target
        if "line" in adjusted:
            if len(adjusted["line"]) != len(x):
                raise ValueError("line and x must have matching lengths")
            adjusted["line"][0] = (adjusted["line"][0][0], target)
        return adjusted

    line = adjusted.get("line", [])
    band = adjusted.get("band", [])
    if not line or not band:
        raise ValueError("line and band must be nonempty")
    x0, original_start = line[0]
    if not math.isclose(float(x0), 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("SHORE-RL series must start at step zero")
    delta = target - float(original_start)
    line[0] = (x0, target)
    adjusted["band"] = [
        (x, y + delta if math.isclose(float(x), float(x0), rel_tol=0.0, abs_tol=1e-9) else y)
        for x, y in band
    ]
    return adjusted
```

- [ ] **Step 3: Add a panel-preparation boundary and use it in rendering**

Add:

```python
def prepare_panels(source=SOURCE):
    panels = extract_panel_series(source)
    task57 = _build_task57_series()
    if len(panels) == 2:
        panels[1]["shore"] = task57["shore"]
        panels[1]["dsrl"] = task57["dsrl"]
    for panel in panels:
        panel["shore"] = align_shore_start(panel["shore"])
    return panels
```

Replace the opening data assembly in `render_figure` with:

```python
def render_figure(source=SOURCE, output=OUTPUT, preview=None):
    panels = prepare_panels(source)
```

- [ ] **Step 4: Run the complete Figure 5 test module**

Run from `paper/`:

```bash
/mnt/mnt/data/envs/residual/bin/python -m unittest \
  test_align_libero_figure.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Check source formatting and unintended changes**

Run:

```bash
git diff --check -- paper/align_libero_figure.py paper/test_align_libero_figure.py
```

Expected: no output.

Because both files were already untracked user work, do not create an
implementation commit.

---

### Task 3: Regenerate Figure 5 and Rebuild the Manuscript

**Files:**

- Modify: `paper/figure/fig_libero10_aligned.pdf`
- Modify: `paper/main.pdf`

**Interfaces:**

- Consumes: `prepare_panels()` and the existing `paper/main.tex`.
- Produces: updated Figure 5 PDF and rebuilt manuscript PDF.

- [ ] **Step 1: Regenerate the standalone Figure 5 asset**

Run from the repository root:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/align_libero_figure.py
```

Expected output:

```text
wrote /mnt/mnt/data/resfit/paper/figure/fig_libero10_aligned.pdf
```

- [ ] **Step 2: Rebuild the manuscript twice**

Run twice from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both runs finish with exit code `0` and report an output PDF with
11 pages.

- [ ] **Step 3: Verify the numeric alignment and preservation contract**

Run from the repository root:

```bash
/mnt/mnt/data/envs/residual/bin/python -c '
import copy
import sys
sys.path.insert(0, "paper")
import align_libero_figure as figure5
raw = figure5.extract_panel_series()
prepared = figure5.prepare_panels()
starts = [
    panel["shore"]["y"][0]
    if "sem" in panel["shore"]
    else panel["shore"]["line"][0][1]
    for panel in prepared
]
assert starts == [0.6, 0.6], starts
assert prepared[0]["shore"]["line"][1:] == raw[0]["shore"]["line"][1:]
assert prepared[1]["shore"]["y"][1:] == raw[1]["shore"]["y"][1:]
assert prepared[0]["dsrl"] == raw[0]["dsrl"]
assert prepared[1]["dsrl"] == raw[1]["dsrl"]
print(starts)
'
```

Expected:

```text
[0.6, 0.6]
```

- [ ] **Step 4: Confirm Figure 4 is byte-for-byte unchanged**

Run:

```bash
sha256sum "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
```

Expected SHA-256:

```text
5830ff675a5388928a4b763f06be787b04e0bd82bef4354640a2c472ca894478
```

- [ ] **Step 5: Check the final PDF and LaTeX diagnostics**

Run:

```bash
pdfinfo paper/main.pdf
rg -n "Fatal error|Undefined control sequence|undefined references|undefined citations" \
  paper/main.log
```

Expected: `paper/main.pdf` has 11 pages; the diagnostic search produces no
matches.

- [ ] **Step 6: Render and inspect the Figure 5 manuscript page**

Run:

```bash
pdftocairo -f 5 -l 5 -png -r 180 \
  paper/main.pdf temp/figure5-start-alignment
```

Inspect `temp/figure5-start-alignment-5.png` and confirm:

- both green curves begin at `0.600`;
- both step-zero green bands remain centered on their curves;
- no curve, band, label, legend, axis, or layout regression is visible.

Do not commit the regenerated PDFs because the standalone Figure 5 asset was
already an untracked user file and `paper/main.pdf` is ignored build output.
