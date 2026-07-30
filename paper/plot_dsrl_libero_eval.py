"""DSRL (noise-space RL on pi0) on LIBERO -- eval success-rate curves, 1x2 panels:
LIBERO-10 task8 (moka pots) | LIBERO-90 task63 (bowl stacking).

Style follows plot_pouring_lifttray_seeds.py (shared y-axis, frozen-base reference
line, open marker = group not yet at target, one shared legend); the DSRL colour is
reused verbatim from that script's STY["dsrl"] so both figures read as one system.

Two deliberate departures from the reference script, both forced by the data:
  * SOURCE: reads the LOCAL .wandb transaction logs, not the wandb API. These runs
    log to entity 'robot_vla' (chj's key), not ours, and their eval videos never
    finished uploading -- the local logs are the complete record.
  * X-AXIS: uses 'env_steps', NOT '_step'. In DSRL '_step' counts gradient updates
    (= (samples - start_online_updates) * multi_grad_step); real environment
    interaction is 'env_steps' = samples * query_freq. The two differ by a constant
    +10k offset here. The reference script's x-axis is env steps, so matching it
    means reading env_steps.

Single seed (seed 0) per task, so there is no s.e.m. band -- the reference script's
multi-seed aggregation would be a lie with n=1. Per-panel note states the seed count.

The two frozen-base lines are NOT the same kind of baseline, and the panels say so:
pi0_libero is trained on `physical-intelligence/libero` (40 tasks: spatial/object/
goal/libero_10, 1693 demos). "put both moka pots on the stove" is task_index 6 of
that set -- in-distribution, base 0.76. libero_90 is absent from it entirely, so
task63 is zero-shot -- base 0.00, no positive samples, and SAC never bootstraps.
Reading the two 'frozen base' lines as one type of reference would be misleading.

moka was still training at write time (open marker = tail). task63 was stopped at
~160k env steps on 2026-07-16: its outcome was already determined.
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

# ---- style: lifted from plot_pouring_lifttray_seeds.py so the figures match ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
DSRL = {"label": "DSRL (noise-space RL)", "color": "#2a6fd6", "ls": (0, (5, 1)), "lw": 2.0, "z": 3}

PANELS = {
    "LIBERO-10 task8: moka pots": {
        "path": "/mnt/mnt/data/chj/dsrl_tmp/moka_h520_eval50_tmp/tmp5dxknfd1/dsrl_oneshot_libero_10_t8_2026_07_15_11_58_31_0000--s-0_seed0/wandb/run-20260715_115837-dsrl_oneshot_libero_10_t8_2026_07_15_11_58_31_0000--s-0_seed0/run-dsrl_oneshot_libero_10_t8_2026_07_15_11_58_31_0000--s-0_seed0.wandb",
        "desc": "put both moka pots on the stove",
        "regime": "in-distribution",  # = task_index 6 of pi0's training set
        "status": "running",
    },
    "LIBERO-90 task63: bowl stacking": {
        "path": "/mnt/mnt/data/chj/dsrl_tmp/task63_h520_eval50_tmp/tmpzvvqtjb_/dsrl_oneshot_libero_90_t63_2026_07_15_12_22_20_0000--s-0_seed0/wandb/run-20260715_122228-dsrl_oneshot_libero_90_t63_2026_07_15_12_22_20_0000--s-0_seed0/run-dsrl_oneshot_libero_90_t63_2026_07_15_12_22_20_0000--s-0_seed0.wandb",
        "desc": "stack the left bowl on the right bowl, place in tray",
        "regime": "zero-shot",  # libero_90 absent from pi0's training set
        "status": "stopped 2026-07-16",
    },
}
XMAX = 420  # max_steps=400k gradient updates -> ~410k env steps


def pull(path):
    """(env_steps_k, success_rate) pairs from a local .wandb transaction log."""
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    pts = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break  # tail of a live run: trailing partial record
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in rec.history.item:
            k = it.key if it.key else ".".join(it.nested_key)
            try:
                row[k] = json.loads(it.value_json)
            except Exception:
                pass
        if "evaluation/success_rate" in row and "env_steps" in row:
            pts.append((row["env_steps"] / 1000.0, float(row["evaluation/success_rate"])))
    return sorted(pts)


DATA = {t: pull(c["path"]) for t, c in PANELS.items()}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
handle = None
for ax, (task, cfg) in zip(axes, PANELS.items()):
    pts = DATA[task]
    # frozen base = eval at env_steps 0, i.e. before any gradient update lands
    # (start_online_updates=500). This is pi0 driven by the untrained noise actor.
    bsr = pts[0][1] if pts and pts[0][0] == 0 else None
    if bsr is not None:
        ax.axhline(bsr, color=INK, lw=1.0, ls=(0, (1, 2)), zorder=1)
        # keep this label short: moka's curve runs right up to the base line, and a
        # longer string would be overdrawn by it. The regime goes in the note above.
        va, off = ("bottom", 0.02) if bsr < 0.1 else ("top", -0.02)
        ax.text(XMAX - 8, bsr + off, f"frozen base {bsr:.2f}",
                va=va, ha="right", fontsize=7.5, color=MUTED)
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    ln, = ax.plot(xs, ys, color=DSRL["color"], lw=DSRL["lw"], ls=DSRL["ls"],
                  zorder=DSRL["z"], solid_capstyle="round")
    handle = ln
    if xs and xs[-1] < XMAX - 20:  # open marker: did not reach the 400k-step target
        ax.plot(xs[-1], ys[-1], "o", mfc="white", mec=DSRL["color"], mew=1.4, ms=6,
                zorder=DSRL["z"] + 1)
    # regime lives here (not on the base line) -- it is the fact that distinguishes
    # the two panels' baselines, so it must be legible even where the curve is dense
    ax.text(0.5, 0.955, f"n=1 seed · {cfg['regime']} · {cfg['status']} ({xs[-1]:.0f}k)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
            color=MUTED, style="italic")
    ax.set_title(task, fontsize=9.8, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Env steps (k)")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate")

fig.legend([handle], [DSRL["label"]], loc="lower center", ncol=1, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.05), handlelength=2.4)
fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.4)
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_dsrl_libero_eval_curves.pdf"
out_png = "/tmp/claude-0/-mnt-mnt-data/83f072b1-ad87-4f66-b2fe-a1683b04c240/scratchpad/fig_dsrl_libero_eval_curves.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf)
print("wrote", out_png)

# ---- report stats (ss = mean of last 20% of evals, as in the reference script) ----
for task, pts in DATA.items():
    ys = [y for _, y in pts]
    if not ys:
        print(f"\n[{task}] no eval data")
        continue
    tail = ys[max(1, int(len(ys) * 0.8)):] or ys
    ss = sum(tail) / len(tail)
    print(f"\n[{task}]  frozen base {ys[0]:.2f}  n_eval={len(ys)}  to={pts[-1][0]:.0f}k env steps")
    print(f"  latest={ys[-1]:.2f}  peak={max(ys):.2f}  min={min(ys):.2f}  "
          f"steady-state(last 20%)={ss:.2f}")
    print(f"  vs frozen base: {ss - ys[0]:+.2f}   (50-episode s.e. ~ "
          f"{math.sqrt(max(ss, 1e-9) * (1 - ss) / 50):.3f})")
