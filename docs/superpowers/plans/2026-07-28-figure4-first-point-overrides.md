# Figure 4 SHORE-RL First-Point Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adjust only the displayed SHORE-RL zero-step means in Figure 4 panels 2, 4, and 5 to 0.68, 0.48, and 0.90, move their Frozen base lines with them, and leave every other plotted value unchanged.

**Architecture:** Put the deterministic display-only transformation in a small pure-Python module that has no W&B or Matplotlib side effects. The existing plotting script will aggregate the original seed histories, pass only the displayed SHORE-RL mean/SEM series through that module, and derive the Frozen base from the same task mapping. The main-paper caption will disclose the three adjusted display values and retained original SEM widths.

**Tech Stack:** Python 3.10, `unittest`, Matplotlib 3.10.9, W&B 0.27.1, pdfLaTeX, Poppler command-line tools.

## Global Constraints

- Panel 1 (Pouring) and panel 3 (PieceAssembly/`ThreePiece`) must remain unchanged.
- The displayed SHORE-RL 0k means must be exactly 0.68 for LiftTray, 0.48 for Threading, and 0.90 for CanSort.
- Every SHORE-RL mean and SEM value from 10k onward must remain unchanged.
- The original SEM width at each adjusted 0k point must remain unchanged and be recentered on the adjusted mean.
- ResFit, DSRL, IBRL, and IQL means and SEM bands must remain unchanged.
- The LiftTray, Threading, and CanSort Frozen base lines must be 0.68, 0.48, and 0.90; the other Frozen base lines must remain unchanged.
- Do not rewrite seed histories and do not edit PDF vector objects directly.
- The Figure 4 caption must disclose the adjusted zero-step display values and retained original SEM widths.
- Preserve all unrelated worktree changes. In particular, do not commit pre-existing modifications in `paper/plot_pouring_lifttray_seeds.py`, `paper/test_plot_pouring_lifttray_seeds.py`, `paper/main.tex`, or the tracked Figure 4 PDF.

---

## File Structure

- Create `paper/figure4_first_point_overrides.py`: pure display transformation and Frozen base lookup.
- Create `paper/test_figure4_first_point_overrides.py`: numerical unit tests and source-integration contract tests.
- Modify `paper/plot_pouring_lifttray_seeds.py`: apply the pure transformation after aggregation and use the displayed first point for Frozen base.
- Modify `paper/main.tex`: disclose the three display overrides in the Figure 4 caption.
- Regenerate `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`: tracked Figure 4 artifact.
- Regenerate `paper/main.pdf`: compiled main-paper deliverable.

### Task 1: Pure display transformation

**Files:**
- Create: `paper/figure4_first_point_overrides.py`
- Create: `paper/test_figure4_first_point_overrides.py`

**Interfaces:**
- Consumes: task key, plot group key, aggregated mean sequence, aggregated SEM sequence, and raw Frozen base.
- Produces: `SHORE_FIRST_POINT_OVERRIDES`, `prepare_display_series(task, group, mean, sem)`, and `displayed_frozen_base(task, raw_base)`.

- [ ] **Step 1: Write the failing numerical tests**

Create `paper/test_figure4_first_point_overrides.py`:

```python
from pathlib import Path
import sys
import unittest


PAPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER_DIR))

from figure4_first_point_overrides import (  # noqa: E402
    SHORE_FIRST_POINT_OVERRIDES,
    displayed_frozen_base,
    prepare_display_series,
)


class Figure4FirstPointOverrideTest(unittest.TestCase):
    def test_exact_override_mapping(self):
        self.assertEqual(
            SHORE_FIRST_POINT_OVERRIDES,
            {"LiftTray": 0.68, "Threading": 0.48, "CanSort": 0.90},
        )

    def test_only_first_shore_point_changes_for_affected_tasks(self):
        original_mean = [0.77, 0.81, 0.86]
        original_sem = [0.03, 0.02, 0.01]
        targets = {"LiftTray": 0.68, "Threading": 0.48, "CanSort": 0.90}

        for task, target in targets.items():
            with self.subTest(task=task):
                mean, sem = prepare_display_series(
                    task, "ours", original_mean, original_sem
                )
                self.assertEqual(mean, [target, 0.81, 0.86])
                self.assertEqual(sem, original_sem)

        self.assertEqual(original_mean, [0.77, 0.81, 0.86])
        self.assertEqual(original_sem, [0.03, 0.02, 0.01])

    def test_panels_one_and_three_are_unchanged(self):
        for task in ("Pouring", "ThreePiece"):
            with self.subTest(task=task):
                mean, sem = prepare_display_series(
                    task, "ours", [0.6, 0.7], [0.04, 0.03]
                )
                self.assertEqual(mean, [0.6, 0.7])
                self.assertEqual(sem, [0.04, 0.03])

    def test_non_shore_groups_are_unchanged(self):
        for group in ("base", "dsrl", "iql", "ibrl"):
            with self.subTest(group=group):
                mean, sem = prepare_display_series(
                    "LiftTray", group, [0.2, 0.3], [0.01, 0.02]
                )
                self.assertEqual(mean, [0.2, 0.3])
                self.assertEqual(sem, [0.01, 0.02])

    def test_empty_series_and_length_mismatch(self):
        self.assertEqual(
            prepare_display_series("LiftTray", "ours", [], []),
            ([], []),
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            prepare_display_series("LiftTray", "ours", [0.5], [])

    def test_frozen_base_follows_displayed_first_point(self):
        self.assertEqual(displayed_frozen_base("LiftTray", 0.77), 0.68)
        self.assertEqual(displayed_frozen_base("Threading", 0.51), 0.48)
        self.assertEqual(displayed_frozen_base("CanSort", 0.99), 0.90)
        self.assertEqual(displayed_frozen_base("Pouring", 0.79), 0.79)
        self.assertEqual(displayed_frozen_base("ThreePiece", 0.59), 0.59)
        self.assertIsNone(displayed_frozen_base("LiftTray", None))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm the missing-module failure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'figure4_first_point_overrides'`.

- [ ] **Step 3: Implement the minimal pure module**

Create `paper/figure4_first_point_overrides.py`:

```python
"""Display-only zero-step adjustments for the main paper's Figure 4."""

from collections.abc import Sequence


SHORE_FIRST_POINT_OVERRIDES = {
    "LiftTray": 0.68,
    "Threading": 0.48,
    "CanSort": 0.90,
}


def prepare_display_series(
    task: str,
    group: str,
    mean: Sequence[float],
    sem: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Return display copies, changing only an eligible SHORE-RL first mean."""
    if len(mean) != len(sem):
        raise ValueError("mean and sem must have the same length")

    display_mean = list(mean)
    display_sem = list(sem)
    if group == "ours" and display_mean and task in SHORE_FIRST_POINT_OVERRIDES:
        display_mean[0] = SHORE_FIRST_POINT_OVERRIDES[task]
    return display_mean, display_sem


def displayed_frozen_base(task: str, raw_base: float | None) -> float | None:
    """Return the Frozen base aligned with the displayed SHORE-RL first point."""
    if raw_base is None:
        return None
    return SHORE_FIRST_POINT_OVERRIDES.get(task, raw_base)
```

- [ ] **Step 4: Run the numerical tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
```

Expected: `Ran 6 tests` and `OK`.

- [ ] **Step 5: Commit only the two new files**

Run:

```bash
git add paper/figure4_first_point_overrides.py paper/test_figure4_first_point_overrides.py
git commit --only paper/figure4_first_point_overrides.py paper/test_figure4_first_point_overrides.py -m "test: define Figure 4 first-point display contract"
```

Expected: the commit contains only the new helper and its tests. Confirm that the pre-existing staged file remains staged:

```bash
git diff --cached --name-only
```

Expected: `docs/superpowers/specs/2026-07-25-method-reproducibility-design.md` remains listed and is not included in the new commit.

### Task 2: Plot integration and caption disclosure

**Files:**
- Modify: `paper/test_figure4_first_point_overrides.py`
- Modify: `paper/plot_pouring_lifttray_seeds.py:20-23,150-160,188-191`
- Modify: `paper/main.tex:459-465`

**Interfaces:**
- Consumes: `prepare_display_series` and `displayed_frozen_base` from Task 1.
- Produces: plotting-source integration that applies overrides only after `agg`, plus a transparent Figure 4 caption.

- [ ] **Step 1: Add failing integration-contract tests**

Add these constants and test class to `paper/test_figure4_first_point_overrides.py`:

```python
PLOT_SOURCE = PAPER_DIR / "plot_pouring_lifttray_seeds.py"
MAIN_TEX = PAPER_DIR / "main.tex"


class Figure4IntegrationContractTest(unittest.TestCase):
    def test_plot_applies_display_transform_after_aggregation(self):
        text = PLOT_SOURCE.read_text()
        aggregation = "xs, mean, sem = agg(seeds)"
        transformation = (
            "mean, sem = prepare_display_series(task, gk, mean, sem)"
        )
        self.assertIn(
            "from figure4_first_point_overrides import",
            text,
        )
        self.assertIn(transformation, text)
        self.assertGreater(text.index(transformation), text.index(aggregation))

    def test_plot_aligns_frozen_base_with_displayed_first_point(self):
        text = PLOT_SOURCE.read_text()
        self.assertIn(
            "BASE[task] = displayed_frozen_base(task, raw_base)",
            text,
        )

    def test_caption_discloses_exact_display_values_and_sem_treatment(self):
        text = MAIN_TEX.read_text()
        for fragment in (
            "zero-step means for",
            "\\textsc{LiftTray}",
            "\\textsc{Threading}",
            "\\textsc{CanSort}",
            "$.68$, $.48$, and $.90$",
            "original zero-step s.e.m. widths",
        ):
            self.assertIn(fragment, text)
```

- [ ] **Step 2: Run the integration tests and confirm they fail**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
```

Expected: the six numerical tests pass and the three integration tests fail because the plotting script and caption do not yet contain the integration.

- [ ] **Step 3: Import and apply the transformation in the plotting script**

Add after the Matplotlib imports in `paper/plot_pouring_lifttray_seeds.py`:

```python
from figure4_first_point_overrides import (
    displayed_frozen_base,
    prepare_display_series,
)
```

Replace the Frozen base calculation with:

```python
BASE = {}
for task, groups in DATA.items():
    firsts = [seed[0] for seed in groups.get("ours", []) if 0 in seed]
    raw_base = sum(firsts) / len(firsts) if firsts else None
    BASE[task] = displayed_frozen_base(task, raw_base)
```

Immediately after each aggregation in the drawing loop, add:

```python
        xs, mean, sem = agg(seeds)
        mean, sem = prepare_display_series(task, gk, mean, sem)
```

Do not apply the helper to the later steady-state reporting aggregation because the report describes the source seed histories, not the display-only first-point treatment.

- [ ] **Step 4: Update only the Figure 4 caption**

Replace the caption body in `paper/main.tex` with:

```tex
\caption{\textbf{Simulation results.} Evaluation success rate
versus environment steps on four long-horizon DexMimicGen tasks and the
shorter-horizon \textsc{CanSort} control. Lines are means across available
seeds with $\pm 1$ s.e.m. bands. The displayed SHORE-RL zero-step means for
\textsc{LiftTray}, \textsc{Threading}, and \textsc{CanSort} are set to
$.68$, $.48$, and $.90$, respectively, while retaining their original
zero-step s.e.m. widths; all other plotted means are seed aggregates.
SHORE-RL is compared with ResFit, DSRL, IBRL, and full-policy IQL. The thin
gray solid reference in each panel marks the displayed SHORE-RL success at
zero environment steps, before residual learning.}
```

- [ ] **Step 5: Run all Figure 4 source tests**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
/mnt/mnt/data/envs/residual/bin/python paper/test_plot_pouring_lifttray_seeds.py
```

Expected: `Ran 9 tests ... OK` and `Ran 3 tests ... OK`.

- [ ] **Step 6: Review the task-specific source diff without committing dirty files**

Run:

```bash
git diff -- paper/plot_pouring_lifttray_seeds.py paper/main.tex
git diff -- paper/test_figure4_first_point_overrides.py
```

Expected: the task-specific additions are present. Do not create a commit for
the plotting script or `main.tex`, because both files contained pre-existing
user modifications before this task and committing them would capture
unrelated work.

### Task 3: Regenerate, audit, and promote Figure 4

**Files:**
- Modify: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`
- Regenerate: `paper/main.pdf`

**Interfaces:**
- Consumes: the integrated plotting source and caption from Task 2.
- Produces: verified standalone Figure 4 and compiled main-paper PDF.

- [ ] **Step 1: Record the pre-regeneration artifact identity**

Run:

```bash
sha256sum "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" paper/main.pdf
pdfinfo "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
pdfinfo paper/main.pdf
```

Expected: both PDFs are readable; Figure 4 has one page and `main.pdf` has the current paper page count. Keep the printed hashes in the execution log for before/after comparison.

- [ ] **Step 2: Generate a candidate standalone figure**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/plot_pouring_lifttray_seeds.py \
  --output /tmp/figure4-first-point-overrides.pdf
```

Expected: exit code 0, `wrote /tmp/figure4-first-point-overrides.pdf`, and the normal per-task statistics report.

- [ ] **Step 3: Validate the candidate PDF structure**

Run:

```bash
pdfinfo /tmp/figure4-first-point-overrides.pdf
pdftotext /tmp/figure4-first-point-overrides.pdf -
```

Expected: one readable PDF page containing the five panel titles, axis labels,
and legend labels.

- [ ] **Step 4: Render and visually inspect the candidate**

Run:

```bash
pdftocairo -singlefile -jpeg -r 160 \
  /tmp/figure4-first-point-overrides.pdf \
  /tmp/figure4-first-point-overrides
```

Inspect `/tmp/figure4-first-point-overrides.jpg`. Expected:

- five unclipped panels and a complete legend;
- panels 1 and 3 retain their original starts;
- panels 2, 4, and 5 start at 0.68, 0.48, and 0.90;
- their gray Frozen base lines pass through the corresponding green first points;
- no malformed or detached uncertainty band appears at 0k.

- [ ] **Step 5: Re-run the exact numerical contract immediately before promotion**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
```

Expected: `Ran 9 tests` and `OK`. These tests exercise the same transformation called by the drawing loop and prove that only the eligible mean index 0 changes, all later means and SEM widths are preserved, and all non-SHORE groups are unchanged.

- [ ] **Step 6: Promote the verified candidate**

Copy `/tmp/figure4-first-point-overrides.pdf` over:

```text
paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf
```

Use a non-destructive file replacement only after Steps 3–5 pass. Do not touch any similarly named untracked Figure 4 PDFs.

- [ ] **Step 7: Compile the main paper twice**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both runs exit 0 and write `main.pdf` with no fatal LaTeX errors.

- [ ] **Step 8: Verify the final PDFs and disclosed caption**

Run:

```bash
pdfinfo "figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
pdfinfo main.pdf
pdftotext main.pdf /tmp/main-figure4-final.txt
rg -n "zero-step means|original zero-step|Simulation results" /tmp/main-figure4-final.txt
sha256sum "figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" main.pdf
```

Expected: both PDFs are readable, their hashes differ from the pre-regeneration
hashes, and extracted text contains the adjusted-point disclosure.

- [ ] **Step 9: Render and inspect the final main-paper page containing Figure 4**

Confirm that Figure 4 remains on the current page 5, then render that page:

```bash
pdftotext -f 5 -l 5 main.pdf - | rg "Simulation results"
pdftocairo -f 5 -l 5 -jpeg -r 150 main.pdf /tmp/main-figure4-final
```

Expected: the text check matches the Figure 4 caption, and
`/tmp/main-figure4-final-5.jpg` is created. Inspect it for figure clipping,
caption overflow, broken cross-references, and unintended layout changes.

- [ ] **Step 10: Run final verification and report scope**

Run:

```bash
/mnt/mnt/data/envs/residual/bin/python paper/test_figure4_first_point_overrides.py
/mnt/mnt/data/envs/residual/bin/python paper/test_plot_pouring_lifttray_seeds.py
git status --short
git diff --check
```

Expected: all 12 Figure 4 tests pass, `git diff --check` reports no whitespace
errors, and the worktree still contains unrelated pre-existing changes. Report
the exact modified files and explicitly state that no implementation commit
was created for already-dirty files.
