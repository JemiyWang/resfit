"""Pouring: SHORE-RL (ours) vs Residual RL base recipe, 3 seeds each.
rliable-style (Agarwal 2021): center = IQM (interquartile mean), band = 95%
stratified bootstrap CI computed at the RUN level (resample seeds, recompute the
whole IQM curve per resample -> preserves per-run cross-frame correlation).
For n=3 seeds, IQM's 25% trim removes 0 points, so IQM == mean here."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb
api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"

GROUPS = {
 "ours": {"label":"SHORE-RL (ours)", "color":"#008300", "ls":"-",  "lw":2.6, "z":5,
   "runs":[("dexmg-chunk-residual","3nsvrbob"),
           ("dexmg-chunk-residual","b4yerjy2"),
           ("dexmg-chunk-residual","yo3rvi0t")]},
 "base": {"label":"Residual RL (base recipe: no subgoal/BC/potential)",
          "color":"#8a8a86", "ls":(0,(4,2)), "lw":2.2, "z":3,
   "runs":[("dexmg-pouring-final","6e89h7g1"),
           ("dexmg-pouring-final","caq9q6l9"),
           ("dexmg-pouring-final","fkvjqhd2")]},
}

def gridkey(s): return int(round(s/10000.0))*10000
def pull(proj, rid):
    r = api.run(f"{ENT}/{proj}/{rid}")
    h = r.history(keys=["eval/success_rate"], samples=10000, pandas=False)
    return {gridkey(int(x["_step"])): float(x["eval/success_rate"])
            for x in h if x.get("eval/success_rate") is not None}

def iqm_axis0(a):                      # IQM over axis 0 (seeds); n=3 -> trim 0 -> mean
    n = a.shape[0]; cut = int(0.25*n)
    s = np.sort(a, axis=0)
    return s[cut:n-cut].mean(axis=0)

def aggregate(seeds, B=20000, seed=0):
    common = sorted(set.intersection(*[set(s.keys()) for s in seeds]))
    xs = np.array(common)/1000.0
    M = np.array([[s[k] for k in common] for s in seeds])   # (n_seeds, T)
    point = iqm_axis0(M)                                     # IQM curve (full data)
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    idx = rng.integers(0, n, size=(B, n))                   # stratified: resample seeds
    boot = iqm_axis0(M[idx].transpose(1,0,2))               # -> (B, T)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return xs, point, lo, hi

# frozen base = first eval across all seeds (residual ~ 0)
DATA = {gk: [pull(p,r) for p,r in g["runs"]] for gk,g in GROUPS.items()}
base_first = [s[0] for seeds in DATA.values() for s in seeds if 0 in s]
BASE_SR = sum(base_first)/len(base_first)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.family":"sans-serif","font.size":10,
 "axes.edgecolor":MUTED,"axes.linewidth":0.8,"axes.spines.top":False,
 "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK})
fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.6))

ax.axhline(BASE_SR, color=INK, lw=1.0, ls=(0,(1,2)), zorder=1)
ax.text(505, BASE_SR, f" frozen base\n {BASE_SR:.2f}", va="center", ha="left",
        fontsize=7.5, color=MUTED, linespacing=1.2)

handles, labels = [], []
for gk, g in GROUPS.items():
    seeds = DATA[gk]; c = g["color"]
    for s in seeds:                          # faint per-seed traces
        pts = sorted(s.items())
        ax.plot([k/1000 for k,_ in pts], [v for _,v in pts],
                color=c, lw=0.8, alpha=0.22, zorder=g["z"]-1, solid_capstyle="round")
    xs, point, lo, hi = aggregate(seeds)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0, zorder=g["z"]-1)
    ln, = ax.plot(xs, point, color=c, lw=g["lw"], ls=g["ls"], zorder=g["z"],
                  solid_capstyle="round")
    handles.append(ln); labels.append(g["label"])
    print(f"{gk:5s} IQM_final={point[-1]:.3f} CI_final=[{lo[-1]:.3f},{hi[-1]:.3f}] "
          f"IQM_ss(last20%)={point[int(len(point)*0.8):].mean():.3f}")

ax.set_title("Pouring (long-horizon, dual-arm)", fontsize=10.5, fontweight="bold",
             loc="left", pad=6)
ax.set_xlabel("Env steps (k)"); ax.set_ylabel("Eval success rate")
ax.set_ylim(-0.03, 1.03); ax.set_xlim(0, 500)
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
leg = ax.legend(handles, labels, loc="lower center", frameon=True, fontsize=8.0,
                bbox_to_anchor=(0.5, 0.015), handlelength=2.2, labelspacing=0.5,
                title="IQM $\\pm$ 95% stratified bootstrap CI (3 seeds)",
                title_fontsize=7.5)
leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(0.78)
leg._legend_box.align = "left"

fig.tight_layout()
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_pouring_seeds_ci.pdf"
out_png = "/tmp/claude-0/-mnt-mnt-data/16fd98c4-a900-4205-92db-f825c6303309/scratchpad/fig_pouring_seeds_ci.png"
fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf, "frozen base", round(BASE_SR,3))
