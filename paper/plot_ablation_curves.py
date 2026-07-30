"""Component ablation of SHORE-RL, seed-level eval curves. Same visual language as
plot_pouring_lifttray_seeds.py (shared y-axis, mean +/-1 s.e.m. band over seeds,
faint per-seed traces, open marker for runs not yet at 500k, one
shared legend). Self-contained: pulls fresh from wandb.

Each ARM is one configuration of the full method with some component(s) removed; all
arms share the same actfeat base, gc_value ckpt and action_scale, so a curve that sits
lower than 'Full' isolates the contribution of the removed component.

EVAL_NUM_ENVS note: success rate is not invariant to this evaluator knob, so legacy
eval_num_envs=4/1 ablations are deliberately excluded. Every run plotted here uses
eval_num_envs=8. The ThreePiece waypoint-only arm currently has three running seeds and is
shown through their latest available evaluations (open endpoint).

Extend by adding entries to ARMS (new component to drop) and to PANELS[task][arm]
(the seed runs). Empty arm lists render nothing (no pending note -- ablation, not
baseline comparison)."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb

api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"
CR = "dexmg-chunk-residual"

# arm styling shared across panels so one legend covers all.
# NB: removing subgoal necessarily disables joint finetuning too (high_actor, which
# joint updates, is part of the subgoal machinery) -- so the "-subgoal" arms are
# "-subgoal -joint". BC(0.1) is held fixed across every arm.
# CAVEAT -- stage_balanced differs by panel (verified against wandb configs):
#  * Pouring & ThreePiece: the `nostage` modes pass --no_stage_balanced alongside
#    --reward_shaping none, so "- staged reward" and "- subgoal - staged reward" also drop
#    stage-balanced offline sampling (stage_balanced True->False). Those two arms are
#    "- all stage privilege", not a single-variable ablation of just the shaping term.
#  * LiftTray: the same-named arms keep stage_balanced=True (zero unexpected diffs vs full),
#    so there "- staged reward" removes only the shaping term -- a clean single variable.
# The "- subgoal" arm keeps --stage_balanced on every panel and so is always clean.
# So the "- staged reward" curve is not strictly the same ablation across panels.
STY = {
    "full": {"label": "SHORE-RL",
             "color": "#008300", "ls": "-", "lw": 2.6, "z": 7},
    "no_staged": {"label": "w/o stage shaping",
                  "color": "#2a6fd6", "ls": (0, (5, 1)), "lw": 2.2, "z": 6},
    "no_subgoal": {"label": "w/o waypoint",
                   "color": "#d1622b", "ls": (0, (5, 1)), "lw": 2.2, "z": 5},
    "no_both": {"label": "w/o waypoint & stage shaping",
                "color": "#7b3fa0", "ls": (0, (3, 1, 1, 1)), "lw": 2.0, "z": 4},
    # waypoint kept (joint is bundled with it), but BC and staged reward both removed
    # (= "-BC -staged"). The only ablation arm besides Full that has run to 500k (2 seeds).
    "subgoal_only": {"label": "w/o demo-BC & stage shaping",
                     "color": "#0f9b8e", "ls": (0, (4, 2)), "lw": 2.2, "z": 6},
}
# Per-panel draw order (full last = on top). All displayed arms use eval_num_envs=8;
# earlier envs=4/1 runs remain excluded. A task falls back to DEFAULT_ORDER if unlisted.
DEFAULT_ORDER = ["subgoal_only", "full"]
DRAW_ORDER_BY_TASK = {
    "Pouring": ["no_both", "no_subgoal", "no_staged", "subgoal_only", "full"],
    "LiftTray": ["no_both", "no_subgoal", "no_staged", "subgoal_only", "full"],
    "ThreePiece": ["no_both", "no_subgoal", "no_staged", "subgoal_only", "full"],
}
ALL_DRAWN = sorted({a for order in DRAW_ORDER_BY_TASK.values() for a in order}
                   | set(DEFAULT_ORDER))  # union: which arms to fetch at all

PANELS = {
    "Pouring": {
        "full":         [(CR, "3nsvrbob"), (CR, "b4yerjy2"), (CR, "yo3rvi0t")],  # 500k
        "subgoal_only": [(CR, "02oot9gd"), (CR, "hgn6n19p")],                    # 500k, 2 seeds
        # 2026-07-17 rerun batch (*_h20p2e8_*_r1), seeds 1/2/3, all at eval_num_envs=8 so
        # they share full's ruler (replaces the 07-16 envs=4/1 batch). base/as/exec match
        # full; verified via wandb config. no_subgoal is a clean single-variable arm; the two
        # nostage arms additionally carry --no_stage_balanced (stage_balanced True->False),
        # so "- staged reward" is really "- staged reward - stage-balanced sampling" (script
        # design, see run_pouring_ablation_seed.sh).
        # Use only seeds 2/3 for this curve; seed 1 (le5ejf1k) is intentionally excluded.
        "no_staged":    [(CR, "s3padi0r"), (CR, "t788eu2h")],                    # envs=8, 2 seeds
        "no_subgoal":   [(CR, "m0w28c01"), (CR, "7pwb2cf6"), (CR, "7005hpgz")],  # envs=8, clean
        "no_both":      [(CR, "9j5i5nnd"), (CR, "31fzc3uc"), (CR, "dma01plo")],  # envs=8
    },
    "LiftTray": {
        "full":         [(CR, "e7sntzx9"), (CR, "372ah0gx"), (CR, "qodyz8ea")],  # 500k
        "subgoal_only": [(CR, "dxeu4uj4"), (CR, "5q371vwt")],                    # 500k, 2 seeds
        # 2026-07-17 batch, seeds 1/2/3, all eval_num_envs=8 (share full's ruler), base/as/exec
        # match full. Unlike Pouring/ThreePiece, these LiftTray no_staged/no_both arms keep
        # stage_balanced=True (zero unexpected diffs vs full) -> here "- staged reward" is a
        # clean single-variable removal of just the shaping term. Replaces the earlier envs=4
        # seed3 run l40r30o4.
        "no_staged":    [(CR, "hyp39jad"), (CR, "d6qindb1"), (CR, "dz9ubgf9")],
        "no_both":      [(CR, "8ztodpmb"), (CR, "lkfs0nhu"), (CR, "pvaq28fk")],
        # no_subgoal completed seeds 3/2/1; all use envs=8 and have matching configs apart
        # from seed. khbe8135 is the completed seed1 rerun replacing crashed dadxiil2.
        "no_subgoal":   [(CR, "nbgai06b"), (CR, "pm9ctit1"), (CR, "khbe8135")],
    },
    "ThreePiece": {
        # full = Fig.2's ThreePiece 'ours' (staged_joint), 3 seeds @500k, envs=8.
        "full":         [(CR, "kmtsayff"), (CR, "m2s74dqm"), (CR, "cgtmwrv3")],
        # 2026-07-17 rerun batch (*_h20p2e8_*_r1), seeds 1/2/3, all at eval_num_envs=8 so
        # they share full's ruler (replaces the earlier envs=4/1 batch
        # 5ayko5og/n188wvf4/3eazw8gu, jlgjnh1p/vqds8wb7/8x0enx52, lenjuxf3/lag7t9ej/pcxbjkyr).
        # base(piecce/best)/as/exec match full. no_subgoal is a clean single-variable arm; the
        # two nostage arms carry --no_stage_balanced (stage_balanced True->False), as on
        # Pouring.
        "no_staged":    [(CR, "15z3ijc6"), (CR, "uvrakupm"), (CR, "exu0ta17")],
        "no_subgoal":   [(CR, "r50syxbq"), (CR, "3icsrj6r"), (CR, "ya51qauh")],  # clean
        "no_both":      [(CR, "ihfya5hb"), (CR, "xvo0bumf"), (CR, "6t0y1m6d")],
        # 2026-07-22 no-BC/no-staged arm, seeds 1/2/3, eval_num_envs=8. For seed1 use the
        # latest rerun w8y426z6; do not include crashed predecessor 93plpkvq.
        "subgoal_only": [(CR, "w8y426z6"), (CR, "abfqmeox"), (CR, "zcii68rz")],
    },
}


def gridkey(s):                      # snap eval step to nearest 10k; step 0/1 -> 0
    return int(round(s / 10000.0)) * 10000


def pull(proj, rid):
    r = api.run(f"{ENT}/{proj}/{rid}")
    h = r.history(keys=["eval/success_rate"], samples=10000, pandas=False)
    d = {}
    for x in h:
        v = x.get("eval/success_rate")
        if v is None:
            continue
        d[gridkey(int(x["_step"]))] = float(v)
    return d


def agg(seeds):
    """mean +/- s.e.m. at each gridkey over seeds present there; band only where >=2
    seeds overlap, collapses to the bare line for a single seed."""
    allk = sorted(set().union(*[set(s.keys()) for s in seeds])) if seeds else []
    xs, mean, sem = [], [], []
    for k in allk:
        vals = [s[k] for s in seeds if k in s]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        xs.append(k / 1000.0)
        mean.append(m)
        sem.append(math.sqrt(var) / math.sqrt(len(vals)))
    return xs, mean, sem


def order_for(task):
    return DRAW_ORDER_BY_TASK.get(task, DEFAULT_ORDER)

# Fetch only the arms some panel actually draws (ALL_DRAWN). Hidden arms stay in PANELS as a
# record but are not pulled -- saves wandb round-trips and avoids a transient fetch error on
# a run we are not plotting anyway. Re-adding an arm to a panel's order makes it fetch again.
DATA = {task: {arm: [pull(p, r) for p, r in arms[arm]]
               for arm in order_for(task) if arm in arms}
        for task, arms in PANELS.items()}

# frozen base = pooled step-0 across this panel's DRAWN arms (residual==0 -> same base policy).
# Pool ONLY over drawn arms: a hidden arm at a different eval_num_envs would drag the
# reference line onto a mixed ruler.
BASE = {}
for task, arms in DATA.items():
    firsts = [s[0] for arm in order_for(task) for s in arms.get(arm, []) if 0 in s]
    BASE[task] = sum(firsts) / len(firsts) if firsts else None

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.family": "DejaVu Serif",
                     "font.size": 17, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK})

n = len(PANELS)
fig, axes = plt.subplots(1, n, figsize=(7.8 if n == 1 else 4.6 * n, 3.4),
                         sharey=True, squeeze=False)
axes = axes[0]
handles = {}
for ax, task in zip(axes, PANELS):
    for arm in order_for(task):
        seeds = DATA[task].get(arm, [])
        if not seeds:
            continue
        st = STY[arm]
        c = st["color"]
        # per-seed traces intentionally omitted: show only the mean line + s.e.m. band.
        xs, mean, sem = agg(seeds)
        ax.fill_between(xs, [m - e for m, e in zip(mean, sem)], [m + e for m, e in zip(mean, sem)],
                        color=c, alpha=0.15, lw=0, zorder=st["z"] - 1)
        ln, = ax.plot(xs, mean, color=c, lw=st["lw"], ls=st["ls"], zorder=st["z"],
                      solid_capstyle="round")
        if xs[-1] < 495:                 # open marker: arm not yet at 500k
            ax.plot(xs[-1], mean[-1], "o", mfc="white", mec=c, mew=1.4, ms=6, zorder=st["z"] + 1)
        handles[arm] = ln
    ax.set_title(task, fontsize=15, fontweight="normal",
                 loc="left", pad=6)
    ax.set_xlabel("Env steps (k)", fontsize=15)
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=13, width=0.8, length=3.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate", fontsize=16)

# legend covers every arm drawn in any panel; canonical order = the richest panel's order
LEGEND_ORDER = max(DRAW_ORDER_BY_TASK.values(), key=len)[::-1]  # full first
order = [k for k in LEGEND_ORDER if k in handles]
ncol = 3 if len(order) >= 5 else len(order)   # 5 arms -> 3 columns (2 rows)
fig.legend([handles[k] for k in order], [STY[k]["label"] for k in order],
           loc="lower center", ncol=ncol, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handlelength=2.2)
fig.tight_layout(rect=[0, 0.15, 1, 1], w_pad=1.4)
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_ablation_curves.pdf"
out_png = "/tmp/fig_ablation_curves.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf)


def ss(seeds):                            # steady-state = mean of last 20% of evals
    out = []
    for s in seeds:
        pts = [v for _, v in sorted(s.items())]
        if not pts:
            continue
        tail = pts[max(1, int(len(pts) * 0.8)):] or pts
        out.append(sum(tail) / len(tail))
    return out


for task in PANELS:
    print(f"\n[{task}] frozen base ~ {BASE[task]:.3f}" if BASE[task] is not None else f"\n[{task}]")
    for arm in order_for(task)[::-1]:
        seeds = DATA[task].get(arm, [])
        s = ss(seeds)
        if not s:
            print(f"  {arm:10s} (none)")
            continue
        xs, mean, sem = agg(seeds)
        m = sum(s) / len(s)
        var = sum((v - m) ** 2 for v in s) / (len(s) - 1) if len(s) > 1 else 0.0
        print(f"  {arm:10s} to={xs[-1]:.0f}k n_seed={len(s)} ss per-seed={[round(v,3) for v in s]} "
              f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(s)):.3f} s.e.m.")
