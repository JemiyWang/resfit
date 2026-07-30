"""Pouring: SHORE-RL (ours, 3 seeds) vs Residual RL base recipe (3 seeds).
Same aesthetic as plot_main_1x4.py, single panel, mean +/-1 s.e.m. band over seeds,
faint per-seed traces, and a frozen-base reference line. Self-contained: pulls fresh."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb
api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"

GROUPS = {
 "ours": {"label":"SHORE-RL (ours)", "color":"#008300", "ls":"-",  "lw":2.6, "z":5,
   "runs":[("dexmg-chunk-residual","3nsvrbob"),      # seed1
           ("dexmg-chunk-residual","b4yerjy2"),      # seed2
           ("dexmg-chunk-residual","yo3rvi0t")]},    # seed3
 "base": {"label":"Residual RL (base recipe: no subgoal/BC/potential)",
          "color":"#8a8a86", "ls":(0,(4,2)), "lw":2.2, "z":3,
   "runs":[("dexmg-pouring-final","6e89h7g1"),       # seed2
           ("dexmg-pouring-final","caq9q6l9"),       # seed3
           ("dexmg-pouring-final","fkvjqhd2")]},     # seed2113258942
}

def gridkey(s):            # snap eval step to nearest 10k; step 0/1 -> 0
    return int(round(s/10000.0))*10000

def pull(proj, rid):
    r = api.run(f"{ENT}/{proj}/{rid}")
    h = r.history(keys=["eval/success_rate"], samples=10000, pandas=False)
    d = {}
    for x in h:
        v = x.get("eval/success_rate")
        if v is None: continue
        d[gridkey(int(x["_step"]))] = float(v)
    return d

# ---- gather ----
DATA = {}
base_first = []           # frozen-base estimate = first eval (residual ~ 0) per seed
for gk, g in GROUPS.items():
    seeds = [pull(p, r) for p, r in g["runs"]]
    DATA[gk] = seeds
    for s in seeds:
        if 0 in s: base_first.append(s[0])
BASE_SR = sum(base_first)/len(base_first)

def agg(seeds):
    """mean +/- s.e.m. over the steps where ALL seeds have a value."""
    common = sorted(set.intersection(*[set(s.keys()) for s in seeds]))
    xs, mean, sem = [], [], []
    for k in common:
        vals = [s[k] for s in seeds]
        m = sum(vals)/len(vals)
        var = sum((v-m)**2 for v in vals)/(len(vals)-1) if len(vals) > 1 else 0.0
        xs.append(k/1000.0); mean.append(m); sem.append(math.sqrt(var)/math.sqrt(len(vals)))
    return xs, mean, sem

# ---- style ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.family":"sans-serif","font.size":10,
 "axes.edgecolor":MUTED,"axes.linewidth":0.8,"axes.spines.top":False,
 "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK})

fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.6))

# frozen base reference
ax.axhline(BASE_SR, color=INK, lw=1.0, ls=(0,(1,2)), zorder=1)
ax.text(505, BASE_SR, f" frozen base\n {BASE_SR:.2f}", va="center", ha="left",
        fontsize=7.5, color=MUTED, linespacing=1.2)

handles, labels = [], []
for gk, g in GROUPS.items():
    seeds = DATA[gk]; c = g["color"]
    # faint per-seed traces (full extent each)
    for s in seeds:
        pts = sorted(s.items())
        ax.plot([k/1000 for k,_ in pts], [v for _,v in pts],
                color=c, lw=0.8, alpha=0.22, zorder=g["z"]-1, solid_capstyle="round")
    # mean +/- s.e.m. over common steps
    xs, mean, sem = agg(seeds)
    ax.fill_between(xs, [m-e for m,e in zip(mean,sem)], [m+e for m,e in zip(mean,sem)],
                    color=c, alpha=0.16, lw=0, zorder=g["z"]-1)
    ln, = ax.plot(xs, mean, color=c, lw=g["lw"], ls=g["ls"], zorder=g["z"],
                  solid_capstyle="round")
    # open marker if group's common extent stops short of 500k (some seeds still running)
    if xs[-1] < 495:
        ax.plot(xs[-1], mean[-1], "o", mfc="white", mec=c, mew=1.4, ms=6, zorder=g["z"]+1)
    handles.append(ln); labels.append(g["label"])

ax.set_title("Pouring (long-horizon, dual-arm)", fontsize=10.5, fontweight="bold",
             loc="left", pad=6)
ax.set_xlabel("Env steps (k)"); ax.set_ylabel("Eval success rate")
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, 500)
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)

leg = ax.legend(handles, labels, loc="lower center", frameon=True, fontsize=8.0,
                bbox_to_anchor=(0.5, 0.015), handlelength=2.2, labelspacing=0.5)
leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(0.78)

fig.tight_layout()
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_pouring_seeds.pdf"
out_png = "/tmp/claude-0/-mnt-mnt-data/16fd98c4-a900-4205-92db-f825c6303309/scratchpad/fig_pouring_seeds.png"
fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf)

# ---- report stats to console ----
def ss(seeds):  # steady-state per seed = mean of last 20% of that seed's evals
    out = []
    for s in seeds:
        pts = [v for _,v in sorted(s.items())]
        tail = pts[max(1, int(len(pts)*0.8)):]
        out.append(sum(tail)/len(tail))
    return out
print(f"frozen base ~ {BASE_SR:.3f}")
for gk, g in GROUPS.items():
    xs, mean, sem = agg(DATA[gk])
    s = ss(DATA[gk])
    m = sum(s)/len(s); var = sum((v-m)**2 for v in s)/(len(s)-1)
    print(f"{gk:5s} common_to={xs[-1]:.0f}k  steady-state per-seed={[round(v,3) for v in s]} "
          f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(s)):.3f} (s.e.m.)")
