"""Long-horizon anti-collapse (seed-level): SHORE-RL (ours, staged_joint) vs
Residual RL, DSRL, IQL, and IBRL.
1x5 panels: Pouring | LiftTray | ThreePiece | Threading | CanSort(short-horizon ref).
Shared y-axis and legend; each method uses a mean +/-1 s.e.m. band over seeds.
Self-contained: pulls fresh from wandb.

'ours' = same config across tasks (subgoal+BC0.1+staged reward+online joint, as0.05).
'base' = flat residual (no subgoal/BC/potential). NOT hyperparameter-matched across
  panels: Pouring/LiftTray/ThreePiece/Threading use the as005 recipe (action_scale 0.05,
  gamma 0.99, n_step 3, stddev 0.05, off1000ep; Pouring/LiftTray additionally exec10).
  CanSort uses the newer as005 exec10 run with n_step=5, UTD=4, buffer=300k, off100ep.
  Run names do not encode every setting -- pull the wandb config and diff before adding a
  run here.
'iql' = dexmg-iql-baseline.
Empty group lists (e.g. Threading base, CanSort ours/base) render a 'pending' note."""
import argparse
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="/mnt/mnt/data/resfit/paper/figure/"
            "fig_long_horizon_anti_collapse_multiseed_curves.pdf",
)
args = parser.parse_args()

api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"

# shared styling so a single legend covers all panels; 'short' used in pending notes
STY = {
 "ours": {"label":"SHORE-RL", "short":"SHORE-RL",
          "color":"#008300", "ls":"-", "lw":3.0, "z":5},
 "base": {"label":"Residual RL", "short":"Residual RL",
          "color":"#8a8a86", "ls":(0,(4,2)), "lw":2.5, "z":2},
 "dsrl": {"label":"DSRL", "short":"DSRL",
          "color":"#2a6fd6", "ls":(0,(5,1)), "lw":2.3, "z":3},
 "iql":  {"label":"IQL", "short":"IQL",
          "color":"#d1622b", "ls":(0,(3,1,1,1)), "lw":2.3, "z":4},
 "ibrl": {"label":"IBRL", "short":"IBRL",
          "color":"#7b4ab5", "ls":(0,(6,2,1,2)), "lw":2.3, "z":4},
}
DRAW_ORDER = ["base", "dsrl", "iql", "ibrl", "ours"]   # ours drawn last (on top)
IQ = "dexmg-iql-baseline"
PANELS = {
 "Pouring": {
   "ours": [("dexmg-chunk-residual","3nsvrbob"),("dexmg-chunk-residual","b4yerjy2"),
            ("dexmg-chunk-residual","yo3rvi0t")],
   # exec10: base ACT n_action_steps=10, matching ours' replanning period. Pouring/
   # LiftTray/ThreePiece/Threading/CanSort base runs are now exec10-aligned.
   # CanSort base uses the newer as005 exec10 run, which is still running.
   "base": [("dexmg-pouring-final","u6w2z2or"),      # seed1  500k  ss .229
            ("dexmg-pouring-final","mcfdqa8u"),      # seed2  500k  ss .022
            ("dexmg-pouring-final","riis3xwv")],     # seed3  500k  ss .035
   "iql":  [(IQ,"ml8d8jtb"),(IQ,"bkgpkmkx"),(IQ,"yodzg5yn")],   # seed42/43/44(44 running)
   "dsrl": [("dsrl","xu9hxe55"),("dsrl","7ikv6dxj"),("dsrl","9csa026x")],  # seed1(500k)/2(~180k)/3(early)
   "ibrl": [("dexmg_formal","u3mobgtb"),("dexmg_formal","uqsl54zu")],
 },
 "LiftTray": {
   "ours": [("dexmg-chunk-residual","e7sntzx9"),("dexmg-chunk-residual","372ah0gx"),
            ("dexmg-chunk-residual","qodyz8ea")],
   "base": [("dexmg-lifttray-final","w8ddpjfu"),         # seed1  500k  ss .051
            ("dexmg-lifttray-final","pogkle3m"),         # seed2  500k  ss .027
            ("dexmg-lifttray-final","zudp5btn")],        # seed3  500k  ss .002
   "iql":  [(IQ,"s48pszl0"),(IQ,"s167w2w3"),(IQ,"qd0s9l3p")],
   "dsrl": [("dsrl","77sql6ye"),("dsrl","xvoktkyg"),("dsrl","q1h6ihvm")],  # seed1(500k)/2(~400k)/3(early)
   "ibrl": [("dexmg_formal","905ud33j"),("dexmg_formal","boluepp0")],
 },
 "ThreePiece": {
   "ours": [("dexmg-chunk-residual","kmtsayff"),("dexmg-chunk-residual","m2s74dqm"),
            ("dexmg-chunk-residual","cgtmwrv3")],
   # exec10-aligned base (n_action_steps=10, same piece/best weights as ours);
   # 3 seeds, all 500k, collapsed (3-seed mean ss .021).
   "base": [("dexmg-threepiece-final","iqcf78ts"),       # seed1  500k  ss .020
            ("dexmg-threepiece-final","t98cav6q"),       # seed2  500k  ss .016
            ("dexmg-threepiece-final","8ip6bc72")],      # seed3  500k  ss .027
   "iql":  [(IQ,"jrh0fi4i"),(IQ,"gr9y6vii"),(IQ,"rkic8mfh")],
   "dsrl": [("dsrl","vv9pnppm"),("dsrl","yourc0wx"),("dsrl","d6wevn3l")],  # seed1(500k)/2,3(early)
   "ibrl": [("dexmg_formal","p7uomccw"),("dexmg_formal","bv1avdba"),
            ("dexmg_formal","8i3b9r53")],
 },
 "Threading": {
   "ours": [("dexmg-chunk-residual","0cojzxec"),("dexmg-chunk-residual","cbmgm6c0"),
            ("dexmg-chunk-residual","s2rbsvxp")],
   # exec10-aligned base (n_action_steps=10, same threading/best weights as ours);
   # 3 seeds, all 500k, collapsed (3-seed mean ss .009).
   "base": [("dexmg-twoarmthreading-final","lnea5hv7"),  # seed1  500k  ss .004
            ("dexmg-twoarmthreading-final","16ef0q0k"),  # seed2  500k  ss .005
            ("dexmg-twoarmthreading-final","fkdazfsb")], # seed3  500k  ss .018
   "iql":  [(IQ,"ll3aqv8c"),(IQ,"5g2om1ko"),(IQ,"1rg76aor")],
   "dsrl": [("dsrl","i351q2a1"),("dsrl","h95kruxi"),("dsrl","t9xp6k6p")],  # seed1(500k)/2(~100k)/3(early)
   "ibrl": [("dexmg_formal","mvxv2vgt"),("dexmg_formal","pts8ariy")],
 },
 "CanSort": {                                         # short-horizon reference panel
   "ours": [("dexmg-chunk-residual","9a8903ei"),      # seed1  as005_joint (running ~40k, ss~0.96)
            ("dexmg-chunk-residual","j955v4ah")],     # seed2  as005_joint (running ~40k, ss~1.00)
   "base": [("dexmg-cansorting-final","it6scym6"),    # seed1  500k  ss .991
            ("dexmg-cansorting-final","3fxjk2ro"),    # seed2  500k  ss .993
            ("dexmg-cansorting-final","ajphvpon")],   # seed3289280101  500k  ss .991
   "iql":  [(IQ,"de339r01"),(IQ,"cgvljajm"),(IQ,"49m46qrv")],
   "dsrl": [("dsrl","ikmx14r8"),("dsrl","m9ts6sc1"),("dsrl","zxotez5b")],  # seed1(500k)/2,3(early)
   "ibrl": [("dexmg_formal","z4ob6395"),("dexmg_formal","i6f0pdnm")],
 },
}

def gridkey(s):            # snap eval step to nearest 10k; step 0/1 -> 0
    return int(round(s/10000.0))*10000

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

def agg(seeds):
    """mean +/- s.e.m. at each gridkey over the seeds PRESENT there (union of steps).
    Extends to the longest seed; the s.e.m. band shows only where >=2 seeds overlap and
    collapses to the bare line where a single seed remains -- a visual cue of seed count
    (used e.g. for DSRL, whose seed2/3 are still short so only seed1 reaches the tail)."""
    allk = sorted(set().union(*[set(s.keys()) for s in seeds])) if seeds else []
    xs, mean, sem = [], [], []
    for k in allk:
        vals = [s[k] for s in seeds if k in s]
        if not vals: continue
        m = sum(vals)/len(vals)
        var = sum((v-m)**2 for v in vals)/(len(vals)-1) if len(vals) > 1 else 0.0
        xs.append(k/1000.0); mean.append(m); sem.append(math.sqrt(var)/math.sqrt(len(vals)))
    return xs, mean, sem

# ---- gather ----
DATA = {task: {gk: [pull(p,r,gk) for p,r in runs] for gk,runs in groups.items()}
        for task, groups in PANELS.items()}
# frozen base per panel = the flat base-recipe's own starting eval (residual ~ 0), 'base' group
# ONLY. ours/iql come from different repos/eval-harnesses; averaging their starts would draw a
# misleading mid-line. Aligning the base recipe's replanning period to ours (exec10) closes only
# part of that harness gap -- LiftTray: same frozen base = 0.78 under ours, 0.38 at exec20 and
# 0.56 at exec10 under baseline's -- so the two harnesses still differ by something else.
# The base recipe visibly starts here and collapses -> honest reference. (M.2.)
BASE = {}
for task, groups in DATA.items():
    firsts = [s[0] for s in groups.get("base", []) if 0 in s]
    BASE[task] = sum(firsts)/len(firsts) if firsts else None

# ---- style ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.family":"serif",
 "font.serif":["TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],
 "font.size":18,"pdf.fonttype":42,"ps.fonttype":42,
 "axes.edgecolor":MUTED,"axes.linewidth":0.8,"axes.spines.top":False,
 "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK})

fig, axes = plt.subplots(1, 5, figsize=(16.6, 4.1), sharey=True)
handles = {}
for ax, task in zip(axes, PANELS):
    pending = []
    for gk in DRAW_ORDER:                 # base, iql, ours (ours on top)
        seeds = DATA[task].get(gk, [])
        if not seeds:
            if gk in ("ours", "base"):    # note only the arms we intend to fill later
                pending.append(STY[gk]["short"])
            continue
        st = STY[gk]; c = st["color"]
        # faint per-seed traces removed per request; only mean line + s.e.m. band shown
        xs, mean, sem = agg(seeds)
        ax.fill_between(xs, [m-e for m,e in zip(mean,sem)], [m+e for m,e in zip(mean,sem)],
                        color=c, alpha=0.15, lw=0, zorder=st["z"]-1)
        ln, = ax.plot(xs, mean, color=c, lw=st["lw"], ls=st["ls"], zorder=st["z"],
                      solid_capstyle="round")
        if xs[-1] < 495:                  # open marker: group not yet at 500k
            ax.plot(xs[-1], mean[-1], "o", mfc="white", mec=c, mew=1.4, ms=6, zorder=st["z"]+1)
        handles[gk] = ln
    if pending:
        ax.text(0.5, 0.10, " / ".join(pending) + ": pending", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=17, color=MUTED, style="italic")
    ttl = task   # panel title = bare task name (group labels removed per request)
    ax.set_title(ttl, fontsize=18, fontweight="normal", loc="left", pad=8)
    ax.set_xlabel("Env steps (k)", fontsize=18)
    ax.set_xlim(0, 500); ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.tick_params(axis="both", labelsize=16, width=0.8, length=3.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate", fontsize=19)

order = [k for k in ("ours", "base", "dsrl", "iql", "ibrl") if k in handles]
leg = fig.legend([handles[k] for k in order], [STY[k]["label"] for k in order],
                 loc="lower center", ncol=len(order), frameon=False, fontsize=16,
                 bbox_to_anchor=(0.5, -0.05), columnspacing=1.8, handlelength=2.4)
fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.4)
out_pdf = args.output
fig.savefig(out_pdf, bbox_inches="tight")
print("wrote", out_pdf)

# ---- report stats ----
def ss(seeds):  # steady-state per seed = mean of last 20% of that seed's evals (skip empty)
    out = []
    for s in seeds:
        pts = [v for _,v in sorted(s.items())]
        if not pts: continue
        tail = pts[max(1, int(len(pts)*0.8)):] or pts
        out.append(sum(tail)/len(tail))
    return out
for task in PANELS:
    fb = f"{BASE[task]:.3f}" if BASE[task] is not None else "n/a"
    print(f"\n[{task}] frozen base ~ {fb}")
    for gk in DRAW_ORDER:
        seeds = DATA[task].get(gk, [])
        s = ss(seeds)
        if not seeds or not s:
            print(f"  {gk:5s} (pending)"); continue
        xs, mean, sem = agg(seeds)
        m = sum(s)/len(s); var = sum((v-m)**2 for v in s)/(len(s)-1) if len(s)>1 else 0.0
        print(f"  {gk:5s} union_to={xs[-1]:.0f}k  n_seed={len(s)}  ss per-seed={[round(v,3) for v in s]} "
              f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(s)):.3f} s.e.m.")
