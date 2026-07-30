"""Staged-reward (full) vs PotHIQL, seed-level eval curves for four long-horizon dual-arm
tasks. Same visual language as plot_ablation_curves.py: shared y-axis, frozen-base
reference (pooled step-0 across the panel's arms, residual==0 there), mean +/-1 s.e.m.
band over seeds, faint per-seed traces, open marker for runs not yet at 500k, one shared
legend. Self-contained: pulls fresh from wandb.

The two arms per panel differ only in the reward the residual RL optimizes (verified from
wandb configs):
  full     -- reward_shaping=staged,    potential_source=stage  ("_staged_joint")
  pothiql  -- reward_shaping=potential,  potential_source=hiql   ("_pothiql_joint")
Everything else (actfeat base, subgoal_conditioned=True, joint high-actor finetuning,
demo_bc_coef=0.1, sg15) is held fixed, so a gap isolates the reward term.

DATA CAVEAT -- the pothiql arm did not run to 500k on every task (wandb states below):
  Pouring     pothiql: 2 seeds finished @500k.
  LiftTray    pothiql: 1 seed finished @500k + 1 crashed @330k.
  Threading   pothiql: 2 seeds crashed @240-260k (no 500k run with subgoal on; the only
              *finished* threading pothiql, threading_best_pothiql/i4n3b5zz, is a DIFFERENT
              config -- subgoal_conditioned=False -- and is intentionally NOT used here).
  ThreePiece  pothiql: 1 seed crashed @250k.
Every sub-500k arm draws an open marker at its tail; do not read those tails as converged."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb

api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"
CR = "dexmg-chunk-residual"

STY = {
    "full":    {"label": "Full  (staged reward + joint)",
                "color": "#008300", "ls": "-", "lw": 2.6, "z": 6},
    "pothiql": {"label": "PotHIQL  (potential reward + joint)",
                "color": "#d1622b", "ls": (0, (5, 1.4)), "lw": 2.3, "z": 5},
}
DRAW_ORDER = ["pothiql", "full"]        # full on top

PANELS = {
    "Pouring": {
        "full":    [(CR, "3nsvrbob"), (CR, "b4yerjy2"), (CR, "yo3rvi0t")],   # 3 seeds @500k
        "pothiql": [(CR, "yzsw1r4u"), (CR, "toj957n8")],                     # 2 seeds @500k
    },
    "LiftTray": {
        "full":    [(CR, "e7sntzx9"), (CR, "372ah0gx"), (CR, "qodyz8ea")],   # 3 seeds @500k
        "pothiql": [(CR, "j54jpnjn"), (CR, "somalim5")],                     # 500k + crashed@330k
    },
    "Threading": {
        "full":    [(CR, "cbmgm6c0"), (CR, "s2rbsvxp"), (CR, "ulguni4f")],   # 3 runs @500k
        "pothiql": [(CR, "y91hfmqf"), (CR, "fxe2pvvx")],                     # crashed @260k/240k
    },
    "ThreePiece": {
        "full":    [(CR, "kmtsayff"), (CR, "m2s74dqm"), (CR, "cgtmwrv3")],   # 3 seeds @500k
        "pothiql": [(CR, "t5ohavo2")],                                       # crashed @250k
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


DATA = {task: {arm: [pull(p, r) for p, r in runs] for arm, runs in arms.items()}
        for task, arms in PANELS.items()}

# frozen base = pooled step-0 across this panel's arms (residual==0 -> same base policy)
BASE = {}
for task, arms in DATA.items():
    firsts = [s[0] for seeds in arms.values() for s in seeds if 0 in s]
    BASE[task] = sum(firsts) / len(firsts) if firsts else None

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK})

n = len(PANELS)
fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.4), sharey=True, squeeze=False)
axes = axes[0]
handles = {}
for ax, task in zip(axes, PANELS):
    bsr = BASE[task]
    if bsr is not None:
        ax.axhline(bsr, color=INK, lw=1.0, ls=(0, (1, 2)), zorder=1)
        ax.text(492, bsr - 0.02, f"frozen base {bsr:.2f}", va="top", ha="right",
                fontsize=7.5, color=MUTED)
    for arm in DRAW_ORDER:
        seeds = DATA[task].get(arm, [])
        if not seeds:
            continue
        st = STY[arm]
        c = st["color"]
        for s in seeds:                  # faint per-seed traces
            pts = sorted(s.items())
            ax.plot([k / 1000 for k, _ in pts], [v for _, v in pts],
                    color=c, lw=0.8, alpha=0.20, zorder=st["z"] - 1, solid_capstyle="round")
        xs, mean, sem = agg(seeds)
        ax.fill_between(xs, [m - e for m, e in zip(mean, sem)], [m + e for m, e in zip(mean, sem)],
                        color=c, alpha=0.15, lw=0, zorder=st["z"] - 1)
        ln, = ax.plot(xs, mean, color=c, lw=st["lw"], ls=st["ls"], zorder=st["z"],
                      solid_capstyle="round")
        if xs[-1] < 495:                 # open marker: arm not yet at 500k
            ax.plot(xs[-1], mean[-1], "o", mfc="white", mec=c, mew=1.4, ms=6, zorder=st["z"] + 1)
        handles[arm] = ln
    ax.set_title(f"{task} (long-horizon, dual-arm)", fontsize=9.6, fontweight="bold",
                 loc="left", pad=6)
    ax.set_xlabel("Env steps (k)")
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate")

order = [k for k in DRAW_ORDER[::-1] if k in handles]
fig.legend([handles[k] for k in order], [STY[k]["label"] for k in order],
           loc="lower center", ncol=len(order), frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.8, handlelength=2.4)
fig.text(0.5, -0.075, "open marker = run stopped before 500k (crashed / partial)",
         ha="center", va="top", fontsize=7.2, color=MUTED)
fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=1.4)
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_staged_vs_pothiql_curves.pdf"
out_png = "/tmp/fig_staged_vs_pothiql_curves.png"
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
    for arm in DRAW_ORDER[::-1]:
        seeds = DATA[task].get(arm, [])
        s = ss(seeds)
        if not s:
            print(f"  {arm:8s} (none)")
            continue
        xs, mean, sem = agg(seeds)
        m = sum(s) / len(s)
        var = sum((v - m) ** 2 for v in s) / (len(s) - 1) if len(s) > 1 else 0.0
        print(f"  {arm:8s} to={xs[-1]:.0f}k n_seed={len(s)} ss per-seed={[round(v,3) for v in s]} "
              f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(s)):.3f} s.e.m.")
