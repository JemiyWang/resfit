# Figure 3 Frozen-Base Reference Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin gray horizontal frozen-base reference to every main-paper Figure 3 panel at that panel's mean SHORE-RL step-zero success.

**Architecture:** Keep the existing W&B data-loading and plotting pipeline intact. Change the per-panel `BASE` statistic to use only `ours` step-zero evaluations, render that value below the learning curves, expose one reference sample in the shared legend, and document the encoding in the caption.

**Tech Stack:** Python 3, Matplotlib, W&B API, pytest, LaTeX/pdflatex, Poppler/PDF inspection tools.

## Global Constraints

- Use only SHORE-RL (`ours`) seeds with an evaluation at snapped step zero when computing each panel's reference.
- Skip the reference and report it as unavailable if a panel has no qualifying SHORE-RL step-zero evaluation.
- Preserve all curve data, method colors, method line styles, uncertainty bands, panel order, axis ranges, titles, and existing legend entries.
- Regenerate the exact asset included by `paper/main.tex`: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`.
- Describe the gray dashed reference in the Figure 3 caption.

---

## File Structure

- `paper/plot_pouring_lifttray_seeds.py`: computes the per-panel reference, draws it, and adds the shared legend entry.
- `tests/test_figure3_frozen_base_line.py`: runs the plotting script against a deterministic fake W&B API and checks the reference source, geometry, and legend.
- `paper/main.tex`: explains the new Figure 3 visual encoding.
- `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`: regenerated main-paper Figure 3.

### Task 1: Add a deterministic Figure 3 reference-line regression test

**Files:**
- Create: `tests/test_figure3_frozen_base_line.py`
- Test: `tests/test_figure3_frozen_base_line.py`

**Interfaces:**
- Consumes: the top-level plotting pipeline in `paper/plot_pouring_lifttray_seeds.py`.
- Produces: a regression test requiring `BASE[task]` to equal the SHORE-RL step-zero mean, one `_frozen_base` line per panel, and a `Frozen base` shared-legend entry.

- [ ] **Step 1: Write the failing regression test**

```python
import runpy
import sys
import types

import pytest


OURS_IDS = {
    "3nsvrbob", "b4yerjy2", "yo3rvi0t",
    "e7sntzx9", "372ah0gx", "qodyz8ea",
    "kmtsayff", "m2s74dqm", "cgtmwrv3",
    "0cojzxec", "cbmgm6c0", "s2rbsvxp",
    "9a8903ei", "j955v4ah",
}


class FakeRun:
    def __init__(self, run_id):
        self.run_id = run_id

    def history(self, **_kwargs):
        start = 0.8 if self.run_id in OURS_IDS else 0.2
        return [
            {"_step": 0, "eval/success_rate": start},
            {"_step": 10_000, "eval/success_rate": start + 0.05},
        ]

    def scan_history(self, **_kwargs):
        return [
            {"other/step": 0, "score/score": 0.3},
            {"other/step": 10_000, "score/score": 0.35},
        ]


class FakeApi:
    def __init__(self, **_kwargs):
        pass

    def run(self, path):
        return FakeRun(path.rsplit("/", 1)[-1])


def test_figure3_frozen_base_uses_shore_step_zero(monkeypatch, tmp_path):
    fake_wandb = types.SimpleNamespace(Api=FakeApi)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    output = tmp_path / "figure3.pdf"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_pouring_lifttray_seeds.py", "--output", str(output)],
    )

    namespace = runpy.run_path("paper/plot_pouring_lifttray_seeds.py")

    assert output.exists()
    assert all(value == pytest.approx(0.8) for value in namespace["BASE"].values())
    for axis in namespace["axes"]:
        references = [
            line for line in axis.lines if line.get_label() == "_frozen_base"
        ]
        assert len(references) == 1
        assert list(references[0].get_ydata()) == pytest.approx([0.8, 0.8])
    legend_text = [text.get_text() for text in namespace["fig"].legends[0].texts]
    assert legend_text[-1] == "Frozen base"
```

- [ ] **Step 2: Run the test and verify the current implementation fails**

Run:

```bash
pytest tests/test_figure3_frozen_base_line.py -v
```

Expected: one failure at the `BASE` assertion because the current script derives `BASE` from the gray `base` group and draws no `_frozen_base` lines.

- [ ] **Step 3: Commit the regression test**

```bash
git add tests/test_figure3_frozen_base_line.py
git commit -m "test: cover Figure 3 frozen-base reference"
```

### Task 2: Compute and draw the SHORE-RL step-zero references

**Files:**
- Modify: `paper/plot_pouring_lifttray_seeds.py:149-215`
- Test: `tests/test_figure3_frozen_base_line.py`

**Interfaces:**
- Consumes: `DATA[task]["ours"]`, whose seed dictionaries map snapped environment steps to success rates.
- Produces: `BASE: dict[str, float | None]`, a `_frozen_base` line in each qualifying panel, and the `Frozen base` legend sample.

- [ ] **Step 1: Change the reference computation to the green SHORE-RL starts**

Replace the current base-recipe comment and `BASE` loop with:

```python
# frozen base per panel = the green SHORE-RL recipe's own step-zero evaluation.
# At step zero the residual is still zero, so averaging the available SHORE-RL
# seeds gives the height from which the green mean curve starts.
BASE = {}
for task, groups in DATA.items():
    firsts = [seed[0] for seed in groups.get("ours", []) if 0 in seed]
    BASE[task] = sum(firsts) / len(firsts) if firsts else None
```

- [ ] **Step 2: Define a visually distinct thin-gray reference style**

Add immediately after `INK, MUTED, GRID`:

```python
FROZEN_STYLE = {
    "label": "Frozen base",
    "color": "#5f5f5b",
    "ls": (0, (1.5, 2.2)),
    "lw": 1.35,
    "z": 1,
}
```

The short, thin dash differs from the existing thicker `Residual RL` dash while remaining a gray dashed reference.

- [ ] **Step 3: Draw one reference below the curves in each panel**

Insert at the start of the panel loop, before `pending = []`:

```python
    if BASE[task] is not None:
        handles["frozen"] = ax.axhline(
            BASE[task],
            color=FROZEN_STYLE["color"],
            ls=FROZEN_STYLE["ls"],
            lw=FROZEN_STYLE["lw"],
            zorder=FROZEN_STYLE["z"],
            label="_frozen_base",
        )
```

- [ ] **Step 4: Append the reference to the shared legend without changing method order**

Replace the legend-order and label expressions with:

```python
order = [k for k in ("ours", "base", "dsrl", "iql", "ibrl", "frozen") if k in handles]
legend_labels = {
    **{key: style["label"] for key, style in STY.items()},
    "frozen": FROZEN_STYLE["label"],
}
leg = fig.legend(
    [handles[k] for k in order],
    [legend_labels[k] for k in order],
    loc="lower center",
    ncol=len(order),
    frameon=False,
    fontsize=16,
    bbox_to_anchor=(0.5, -0.05),
    columnspacing=1.8,
    handlelength=2.4,
)
```

- [ ] **Step 5: Run the focused regression test**

Run:

```bash
pytest tests/test_figure3_frozen_base_line.py -v
```

Expected: `1 passed`; the temporary PDF is created, all five `BASE` values are `0.8`, every panel has one `_frozen_base` line at `0.8`, and the legend ends with `Frozen base`.

- [ ] **Step 6: Check syntax and whitespace**

Run:

```bash
python -m py_compile paper/plot_pouring_lifttray_seeds.py tests/test_figure3_frozen_base_line.py
git diff --check -- paper/plot_pouring_lifttray_seeds.py tests/test_figure3_frozen_base_line.py
```

Expected: both commands exit zero with no output.

- [ ] **Step 7: Commit the plotting behavior**

```bash
git add paper/plot_pouring_lifttray_seeds.py tests/test_figure3_frozen_base_line.py
git commit -m "fig: add Figure 3 frozen-base references"
```

### Task 3: Regenerate Figure 3 and integrate the caption

**Files:**
- Modify: `paper/main.tex:449-455`
- Modify: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`
- Test: compiled `paper/main.pdf`

**Interfaces:**
- Consumes: the updated plotting script and live W&B runs already enumerated in `PANELS`.
- Produces: the main-paper Figure 3 PDF with five references and a caption that defines them.

- [ ] **Step 1: Update the Figure 3 caption**

Replace the final two caption sentences with:

```tex
with $\pm 1$ s.e.m. bands. SHORE-RL is compared with ResFit (labeled
\emph{Residual RL} in the legend), DSRL, IBRL, and full-policy IQL. The thin
gray dashed reference in each panel marks that task's mean SHORE-RL success at
zero environment steps, before residual learning.
```

- [ ] **Step 2: Regenerate the exact PDF used by the paper**

Run:

```bash
python paper/plot_pouring_lifttray_seeds.py \
  --output "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
```

Expected: the command exits zero, prints the output path, and reports a numeric `frozen base` value for Pouring, LiftTray, ThreePiece, Threading, and CanSort.

- [ ] **Step 3: Check PDF structure and legend text**

Run:

```bash
pdfinfo "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
pdftotext "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" -
```

Expected: a valid one-page PDF; extracted text includes `Frozen base` once and all five task titles.

- [ ] **Step 4: Render and inspect the standalone figure**

Run:

```bash
pdftocairo -png -singlefile -r 180 \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" \
  /tmp/figure3-frozen-base
```

Open `/tmp/figure3-frozen-base.png` with the image viewer. Expected: all five panels contain one thin gray horizontal dashed reference at the green curve's starting height; the legend is readable and unclipped.

- [ ] **Step 5: Compile the paper**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: both passes exit zero and produce `paper/main.pdf` without undefined Figure 3 references or overfull content caused by the new caption.

- [ ] **Step 6: Render and inspect the paper page containing Figure 3**

Figure 3 is on page 5 of the current nine-page main paper. Render that page:

```bash
pdftocairo -f 5 -l 5 -png -singlefile -r 180 \
  main.pdf /tmp/main-figure3-frozen-base
```

Open `/tmp/main-figure3-frozen-base.png` with the image viewer. Expected: Figure 3, its six-entry legend, and caption are readable without clipping or overlap at `\textwidth`.

- [ ] **Step 7: Run final focused checks**

Run from the repository root:

```bash
pytest tests/test_figure3_frozen_base_line.py -v
git diff --check -- paper/plot_pouring_lifttray_seeds.py paper/main.tex \
  tests/test_figure3_frozen_base_line.py
git status --short
```

Expected: `1 passed`; no whitespace errors; status lists only the intended Figure 3 source, caption, test, and generated-asset changes in addition to the user's pre-existing unrelated worktree changes.

- [ ] **Step 8: Commit the regenerated paper integration**

```bash
git add paper/main.tex \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
git commit -m "paper: explain Figure 3 frozen-base references"
```
