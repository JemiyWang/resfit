# Figure 5 400k Final-Window Bar Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a two-panel bar-chart version of Figure 5 using fixed-budget 330k--400k final-window success.

**Architecture:** Add one self-contained plotting module with pure metric and plotting functions plus a W&B-backed `main()`. Unit tests exercise the metric and rendering paths with synthetic data, while the production run uses the exact 28-run inventory selected by the current Figure 5 script.

**Tech Stack:** Python 3.11, unittest, Matplotlib, Weights & Biases API

## Global Constraints

- Fix the budget at 400,000 environment steps.
- Average exactly the eight checkpoints with `320,000 < step <= 400,000`.
- Aggregate seed-level window means with mean and `±1 s.e.m.`; use no error bar for a single seed.
- Reject incomplete windows rather than extrapolating.
- Preserve the two task panels (Pouring and ThreePiece), five arm labels, colors, and shared 0--1 success-rate axis.
- Keep the original curve PDF; Task 3 formally replaces its `main.tex` inclusion with the 400k bar-chart PDF.

---

### Task 1: Fixed-Window Metric and Rendering Contract

**Files:**
- Create: `paper/test_plot_ablation_bars_400k.py`
- Create: `paper/plot_ablation_bars_400k.py`

**Interfaces:**
- Consumes: seed histories as `dict[int, float]`, keyed by snapped environment step.
- Produces: `window_mean(points) -> float`, `summarize_seeds(seeds) -> tuple[float, float | None, list[float]]`, and `plot_summaries(summaries, out_pdf, out_png) -> None`.

- [ ] **Step 1: Write failing unit tests**

Create tests that require:

```python
expected_steps = list(range(330_000, 400_001, 10_000))
points = {step: step / 1_000_000 for step in range(0, 500_001, 10_000)}
self.assertAlmostEqual(window_mean(points), sum(points[s] for s in expected_steps) / 8)
```

and:

```python
incomplete = dict(points)
del incomplete[370_000]
with self.assertRaisesRegex(ValueError, "370000"):
    window_mean(incomplete)
```

Use three constant-window seeds with values `0.2`, `0.4`, and `0.6` to require
`mean == 0.4`, seed values `[0.2, 0.4, 0.6]`, and
`s.e.m. == statistics.stdev([0.2, 0.4, 0.6]) / sqrt(3)`. Require `sem is None`
for a single seed. Pass synthetic summaries to `plot_summaries` and require
both temporary PDF and PNG outputs to be non-empty.

- [ ] **Step 2: Run tests and verify the initial failure**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest paper/test_plot_ablation_bars_400k.py -v
```

Expected: fail because `paper.plot_ablation_bars_400k` does not exist.

- [ ] **Step 3: Implement the minimal pure functions**

Define these constants and functions:

```python
BUDGET = 400_000
WINDOW_STEPS = tuple(range(330_000, BUDGET + 1, 10_000))

def window_mean(points):
    missing = [step for step in WINDOW_STEPS if step not in points]
    if missing:
        raise ValueError(f"incomplete 400k final window; missing steps: {missing}")
    return statistics.fmean(points[step] for step in WINDOW_STEPS)

def summarize_seeds(seeds):
    values = [window_mean(seed) for seed in seeds]
    if not values:
        raise ValueError("at least one seed is required")
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return statistics.fmean(values), sem, values
```

Implement `plot_summaries` as three shared-y panels. In each panel, draw the
five arms in `LEGEND_ORDER`, use the current Figure 5 colors and labels, draw a
black capped error bar only when `sem is not None`, and write both requested
formats.

- [ ] **Step 4: Run the metric and rendering tests**

Run the Step 2 command again.

Expected: all tests pass without any W&B access.

- [ ] **Step 5: Commit the tested metric and rendering implementation**

```bash
git add paper/test_plot_ablation_bars_400k.py paper/plot_ablation_bars_400k.py
git commit -m "feat: add Figure 5 400k bar chart"
```

### Task 2: Production Data Pull and Figure Verification

**Files:**
- Modify: `paper/plot_ablation_bars_400k.py`
- Generate: `paper/figure/fig_ablation_bars_400k.pdf`
- Generate: `paper/figure/fig_ablation_bars_400k.png`

**Interfaces:**
- Consumes: the exact task/arm/run inventory from `paper/plot_ablation_curves.py` and W&B field `eval/success_rate`.
- Produces: a validated 28-seed summary and the two figure files.

- [ ] **Step 1: Add the exact production inventory and pull path**

Copy the selected `PANELS`, `STY`, and arm ordering from the current Figure 5
script. Instantiate `wandb.Api(timeout=120)` only inside `main()`. For every
run, snap `_step` to the nearest 10k, collect `eval/success_rate`, call
`summarize_seeds`, and print each arm's seed values, mean, and s.e.m. before
rendering.

- [ ] **Step 2: Add an inventory regression test**

Require:

```python
self.assertEqual(list(PANELS), ["Pouring", "ThreePiece"])
self.assertTrue(all(set(arms) == set(LEGEND_ORDER) for arms in PANELS.values()))
self.assertEqual(sum(len(runs) for arms in PANELS.values() for runs in arms.values()), 28)
```

- [ ] **Step 3: Run the complete offline test file**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest paper/test_plot_ablation_bars_400k.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Generate the production figure**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python paper/plot_ablation_bars_400k.py
```

Expected: all 28 seeds pass the eight-checkpoint completeness gate and the
script reports both output paths.

- [ ] **Step 5: Verify the artifacts**

Run:

```bash
pdfinfo paper/figure/fig_ablation_bars_400k.pdf
pdffonts paper/figure/fig_ablation_bars_400k.pdf
python -m unittest paper/test_plot_ablation_bars_400k.py -v
git diff --check
```

Expected: one-page PDF, embedded fonts, passing tests, and no whitespace errors.
Render the PDF to PNG and inspect the three panels, legend, clipped text, and
error bars across both retained task panels.

- [ ] **Step 6: Commit the generated first version**

```bash
git add paper/plot_ablation_bars_400k.py paper/test_plot_ablation_bars_400k.py \
  paper/figure/fig_ablation_bars_400k.pdf paper/figure/fig_ablation_bars_400k.png
git commit -m "fig: render Figure 5 at 400k"
```

### Task 3: Integrate the Bar Chart into the Main Paper

**Files:**
- Modify: `paper/main.tex:361-363,470-527`
- Consume: `paper/figure/fig_ablation_bars_400k.pdf`
- Generate: `paper/main.pdf`

**Interfaces:**
- Consumes: the verified 400k per-seed statistics printed by the plotting script.
- Produces: Figure 5 with an aligned label, caption, protocol, and result narrative.

- [ ] **Step 1: Run a failing source contract**

Run:

```bash
python -c 'from pathlib import Path; s=Path("paper/main.tex").read_text(); assert "fig_ablation_bars_400k.pdf" in s; assert "\\label{fig:ablbars}" in s; assert "\\ref{fig:ablbars}" in s; assert "fig_ablation_curves.pdf" not in s; assert "fig:ablcurves" not in s'
```

Expected: fail because the main paper still includes the curve PDF and old label.

- [ ] **Step 2: Align the LaTeX source**

Update `paper/main.tex` in six coordinated places:

- Define final-window success over the last 20% of the stated comparison budget.
- State Ablation A's common 400k cutoff, eight 330k--400k checkpoints, and 2--3 seeds.
- Rewrite the results paragraph with the verified ThreePiece and Pouring patterns.
- Include `figure/fig_ablation_bars_400k.pdf` at `\textwidth`.
- Rename the reference and label to `fig:ablbars`.
- Define bar height as the cross-seed mean of per-seed window averages and error bars as `\pm 1` s.e.m.

- [ ] **Step 3: Re-run the source contract**

Run the Step 1 command again. Expected: exit 0, with no curve path or stale label.

- [ ] **Step 4: Compile the main paper twice**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both commands exit 0 and resolve Figure 5 references.

- [ ] **Step 5: Verify references and artifact structure**

```bash
rg -n "undefined|multiply defined|fig:ablcurves" paper/main.log paper/main.tex
pdfinfo paper/main.pdf
```

Expected: no stale label or reference warnings and a valid eight-page paper PDF.

- [ ] **Step 6: Inspect Figure 5 in context**

Render the page containing Figure 5 and inspect bar and legend legibility, caption wrapping, float placement, and surrounding prose.
