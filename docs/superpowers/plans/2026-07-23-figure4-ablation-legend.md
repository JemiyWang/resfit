# Figure 4 Ablation Legend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the five Figure 4 legend entries to the approved SHORE-RL and `w/o` terminology, then regenerate and visually verify the PDF.

**Architecture:** Keep the existing plotting and WandB data pipeline unchanged. Edit only the `label` values in the `STY` mapping, rerun the existing plotting script, and inspect the regenerated PDF for exact text and layout.

**Tech Stack:** Python, Matplotlib, WandB API, Poppler (`pdftotext`/`pdftocairo`)

## Global Constraints

- Use exactly these labels: `SHORE-RL`, `w/o stage shaping`, `w/o waypoint`, `w/o waypoint & stage shaping`, and `w/o demo-BC & stage shaping`.
- Do not modify experiment data, run selection, curves, colors, line styles, legend order, panel layout, or manuscript text.

---

### Task 1: Rename and Regenerate the Figure 4 Legend

**Files:**
- Modify: `paper/plot_ablation_curves.py:45-58`
- Regenerate: `paper/figure/fig_ablation_curves.pdf`

**Interfaces:**
- Consumes: Existing `STY` dictionary and existing plotting/data-fetch pipeline.
- Produces: A Figure 4 PDF whose legend contains the five approved labels.

- [ ] **Step 1: Run the pre-change label assertion and verify it fails**

```bash
python -c 'from pathlib import Path; s=Path("paper/plot_ablation_curves.py").read_text(); labels=["\"label\": \"SHORE-RL\"","\"label\": \"w/o stage shaping\"","\"label\": \"w/o waypoint\"","\"label\": \"w/o waypoint & stage shaping\"","\"label\": \"w/o demo-BC & stage shaping\""]; assert all(x in s for x in labels)'
```

Expected: `AssertionError`, because the approved labels are not yet all present.

- [ ] **Step 2: Apply the minimal label-only implementation**

Replace the five `STY` labels with:

```python
STY = {
    "full": {"label": "SHORE-RL",
             "color": "#008300", "ls": "-", "lw": 2.6, "z": 7},
    "no_staged": {"label": "w/o stage shaping",
                  "color": "#2a6fd6", "ls": (0, (5, 1)), "lw": 2.2, "z": 6},
    "no_subgoal": {"label": "w/o waypoint",
                   "color": "#d1622b", "ls": (0, (5, 1)), "lw": 2.2, "z": 5},
    "no_both": {"label": "w/o waypoint & stage shaping",
                "color": "#7b3fa0", "ls": (0, (3, 1, 1, 1)), "lw": 2.0, "z": 4},
    "subgoal_only": {"label": "w/o demo-BC & stage shaping",
                     "color": "#0f9b8e", "ls": (0, (4, 2)), "lw": 2.2, "z": 6},
}
```

- [ ] **Step 3: Run the label assertion and verify it passes**

```bash
python -c 'from pathlib import Path; s=Path("paper/plot_ablation_curves.py").read_text(); labels=["\"label\": \"SHORE-RL\"","\"label\": \"w/o stage shaping\"","\"label\": \"w/o waypoint\"","\"label\": \"w/o waypoint & stage shaping\"","\"label\": \"w/o demo-BC & stage shaping\""]; assert all(x in s for x in labels); print("all Figure 4 labels present")'
```

Expected: `all Figure 4 labels present`.

- [ ] **Step 4: Regenerate the figure**

```bash
python paper/plot_ablation_curves.py
```

Expected: exit code `0` and `wrote /mnt/mnt/data/resfit/paper/figure/fig_ablation_curves.pdf`.

- [ ] **Step 5: Verify the PDF content and layout**

```bash
pdftotext -layout paper/figure/fig_ablation_curves.pdf -
pdftocairo -png -singlefile -r 160 paper/figure/fig_ablation_curves.pdf /tmp/fig_ablation_curves_review
```

Expected: extracted text contains all five approved labels. The rendered image shows no overlapping, clipped, or off-canvas legend text.

- [ ] **Step 6: Review the scoped diff**

```bash
git diff --check -- paper/plot_ablation_curves.py paper/figure/fig_ablation_curves.pdf
git diff -- paper/plot_ablation_curves.py
```

Expected: no whitespace errors; the source diff changes only the five label strings.
