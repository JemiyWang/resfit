# IBRL Curve Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mean plus/minus one SEM IBRL curves from exactly 11 canonical W&B runs to the requested copy of the five-task learning-curve PDF.

**Architecture:** Extend the existing self-contained plotting script with an IBRL style, explicit per-task canonical run lists, and a history reader that uses the common IBRL fields `other/step` and `score/score`. Add a CLI output override so execution can update only the `copy.pdf` target while retaining the script's existing default output behavior.

**Tech Stack:** Python 3.11, W&B API, Matplotlib, Python `unittest`, Poppler command-line PDF inspection tools.

## Global Constraints

- Consume only the 11 canonical run IDs enumerated in the approved design.
- Exclude old or duplicate ThreePiece seed-1 runs.
- Use `other/step` for IBRL environment steps and `score/score` for IBRL evaluation success rate.
- Reuse the existing 10k alignment, mean, and plus/minus one SEM aggregation.
- Keep the existing five-panel ordering, typography, dimensions, axes, and non-IBRL styling.
- Overwrite only `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`.
- Preserve the checksum of `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf`.

---

### Task 1: Add a source-level IBRL plotting contract

**Files:**
- Create: `paper/test_plot_pouring_lifttray_seeds.py`
- Modify: `paper/plot_pouring_lifttray_seeds.py`

**Interfaces:**
- Consumes: the existing `STY`, `DRAW_ORDER`, `PANELS`, `pull`, and PDF save path in `paper/plot_pouring_lifttray_seeds.py`.
- Produces: an `ibrl` group in every panel, IBRL history selection inside `pull(proj: str, rid: str, group: str) -> dict[int, float]`, and a `--output` CLI option.

- [ ] **Step 1: Write the failing source-contract test**

```python
from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).with_name("plot_pouring_lifttray_seeds.py")
EXPECTED_IBRL_IDS = {
    "z4ob6395", "i6f0pdnm",
    "905ud33j", "boluepp0",
    "u3mobgtb", "uqsl54zu",
    "mvxv2vgt", "pts8ariy",
    "p7uomccw", "bv1avdba", "8i3b9r53",
}


class IBRLPlotContractTest(unittest.TestCase):
    def test_exact_canonical_ids_and_common_history_fields(self):
        text = SOURCE.read_text()
        actual = set(re.findall(r'\\("dexmg_formal","([a-z0-9]{8})"\\)', text))
        self.assertEqual(actual, EXPECTED_IBRL_IDS)
        self.assertIn('"score/score"', text)
        self.assertIn('"other/step"', text)

    def test_style_draw_order_and_output_override(self):
        text = SOURCE.read_text()
        self.assertRegex(text, r'"ibrl"\\s*:\\s*\\{"label":"IBRL"')
        self.assertRegex(text, r'DRAW_ORDER\\s*=.*"ibrl"')
        self.assertIn('"--output"', text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest -v paper/test_plot_pouring_lifttray_seeds.py
```

Expected: both tests fail because the source does not yet contain an IBRL group or `--output`.

- [ ] **Step 3: Add IBRL style and the exact run mapping**

Add an IBRL style with a distinct purple color and dash pattern:

```python
"ibrl": {"label":"IBRL", "short":"IBRL",
         "color":"#7b4ab5", "ls":(0,(6,2,1,2)), "lw":2.3, "z":4},
```

Set:

```python
DRAW_ORDER = ["base", "dsrl", "iql", "ibrl", "ours"]
```

Add the following literal entries to the matching task dictionaries:

```python
"CanSort":   [("dexmg_formal","z4ob6395"), ("dexmg_formal","i6f0pdnm")]
"LiftTray":  [("dexmg_formal","905ud33j"), ("dexmg_formal","boluepp0")]
"Pouring":   [("dexmg_formal","u3mobgtb"), ("dexmg_formal","uqsl54zu")]
"Threading": [("dexmg_formal","mvxv2vgt"), ("dexmg_formal","pts8ariy")]
"ThreePiece":[("dexmg_formal","p7uomccw"), ("dexmg_formal","bv1avdba"),
              ("dexmg_formal","8i3b9r53")]
```

- [ ] **Step 4: Add method-aware history loading**

Change the function signature and IBRL branch to:

```python
def pull(proj, rid, group):
    r = api.run(f"{ENT}/{proj}/{rid}")
    if group == "ibrl":
        h = r.scan_history(keys=["other/step", "score/score"], page_size=1000)
        step_key, value_key = "other/step", "score/score"
    else:
        h = r.history(keys=["eval/success_rate"], samples=10000, pandas=False)
        step_key, value_key = "_step", "eval/success_rate"
    d = {}
    for x in h:
        v, s = x.get(value_key), x.get(step_key)
        if v is None or s is None:
            continue
        d[gridkey(int(s))] = float(v)
    return d
```

Update collection to pass the group key:

```python
DATA = {
    task: {gk: [pull(p, r, gk) for p, r in runs]
           for gk, runs in groups.items()}
    for task, groups in PANELS.items()
}
```

Include `ibrl` in the legend order:

```python
order = [k for k in ("ours", "base", "dsrl", "iql", "ibrl") if k in handles]
```

- [ ] **Step 5: Add an output-path override**

Import `argparse`, parse the option before saving, and preserve the existing path as default:

```python
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="/mnt/mnt/data/resfit/paper/figure/"
            "fig_long_horizon_anti_collapse_multiseed_curves.pdf",
)
args = parser.parse_args()
out_pdf = args.output
```

- [ ] **Step 6: Run the focused test and verify it passes**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python -m unittest -v paper/test_plot_pouring_lifttray_seeds.py
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 7: Commit the source contract**

```bash
git add paper/plot_pouring_lifttray_seeds.py paper/test_plot_pouring_lifttray_seeds.py
git commit -m "feat: add canonical IBRL learning curves"
```

### Task 2: Generate and validate the requested PDF

**Files:**
- Modify: `paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`

**Interfaces:**
- Consumes: the `--output PATH` interface created by Task 1 and read-only W&B history for the 11 canonical runs.
- Produces: a one-page five-panel PDF with an IBRL mean curve, SEM band, and shared legend entry.

- [ ] **Step 1: Record the protected canonical PDF checksum**

Run:

```bash
sha256sum paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf
```

Expected: checksum `173be6576ce8e5e7a94aa2d7303dcefbd68db9647b9bbe884dd4dd150e54e31d`.

- [ ] **Step 2: Generate only the requested copy**

Run:

```bash
/mnt/mnt/data/resfit/rise_venv/bin/python paper/plot_pouring_lifttray_seeds.py \
  --output "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
```

Expected: the script prints `wrote paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf`, and each IBRL group reports `union_to=500k` with the expected seed count.

- [ ] **Step 3: Verify all canonical W&B run identities and terminal points**

Run a read-only W&B validation that asserts the exact run name, `finished` state,
and presence of `other/step == 500000` for every ID:

```python
EXPECTED = {
    "z4ob6395": "ibrl_bc_can_sort_random_fix_d05_wu50_mix05_seed1_20260713_235501",
    "i6f0pdnm": "ibrl_bc_can_sort_random_fix_d05_wu50_mix05_seed2_20260723_014529",
    "905ud33j": "ibrl_bc_lift_tray_prop_fix_d05_wu50_mix05_seed1_20260716_104308",
    "boluepp0": "ibrl_bc_lift_tray_prop_fix_d05_wu50_mix05_seed2_20260716_104308",
    "u3mobgtb": "ibrl_bc_pouring_prop_fix_d05_wu50_mix05_seed1_20260716_104308",
    "uqsl54zu": "ibrl_bc_pouring_prop_fix_d05_wu50_mix05_seed2_20260716_104308",
    "mvxv2vgt": "ibrl_bc_threading_fix_d05_wu50_mix05_seed1_20260713_230240",
    "pts8ariy": "ibrl_bc_threading_fix_d05_wu50_mix05_seed2_20260716_111916",
    "p7uomccw": "ibrl_bc_three_piece_fix_d05_wu50_mix05_seed1_20260714_230443",
    "bv1avdba": "ibrl_bc_three_piece_fix_d05_wu50_mix05_seed2_20260714_000730",
    "8i3b9r53": "ibrl_bc_three_piece_fix_d05_wu50_mix05_seed3_20260716_111916",
}
```

Expected: validation prints `validated 11 canonical IBRL runs`.

- [ ] **Step 4: Validate the PDF structure, fonts, legend, and protected checksum**

Run:

```bash
pdfinfo "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
pdffonts "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
pdftotext -layout \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" - | rg IBRL
sha256sum paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf
```

Expected: one page; page size approximately `1171 x 296 pt`; embedded DejaVu Serif
fonts; extracted legend text contains `IBRL`; protected checksum remains
`173be6576ce8e5e7a94aa2d7303dcefbd68db9647b9bbe884dd4dd150e54e31d`.

- [ ] **Step 5: Rasterize and visually inspect**

Run:

```bash
pdftoppm -png -r 180 -singlefile \
  "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf" \
  /tmp/ibrl-curve-integration
```

Inspect `/tmp/ibrl-curve-integration.png`.

Expected: all five panels are legible; each includes a purple IBRL curve and SEM
band where nonzero; the five-entry shared legend is unclipped and non-overlapping;
titles, axis labels, and ticks remain aligned.

- [ ] **Step 6: Commit the generated artifact**

```bash
git add "paper/figure/fig_long_horizon_anti_collapse_multiseed_curves copy.pdf"
git commit -m "fig: add IBRL curves to long-horizon comparison"
```

