import json, math, statistics as st
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import wandb

api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"
CR = "dexmg-chunk-residual"
DSRL_PROJECT = "DSRL_pi0_Libero_moka_h520_eval50"

# DSRL baseline on LIBERO-10 Task 8 ("put both moka pots on the stove"). The three
# matching 50-episode-evaluation seeds are pulled directly from W&B. All reach
# approximately 405k raw environment steps, which snap to the figure's 400k grid.
DSRL_RUNS = [
    "dsrl_oneshot_libero_10_t8_2026_07_15_11_58_31_0000--s-0_seed0",
    "dsrl_oneshot_libero_10_t8_2026_07_19_19_22_41_0000--s-1_seed1",
    "dsrl_oneshot_libero_10_t8_2026_07_20_12_19_15_0000--s-2_seed2",
]


@lru_cache(maxsize=None)
def pull_remote_dsrl_grid(run_name):
    """Fetch {snapped-env-step: success_rate} from Wandb by run name."""
    for run in api.runs(f"{ENT}/{DSRL_PROJECT}"):
        if run.name == run_name:
            h = run.history(
                keys=["evaluation/success_rate", "env_steps"],
                samples=10000,
                pandas=False,
            )
            out = {}
            for x in h:
                sr = x.get("evaluation/success_rate")
                if sr is None:
                    continue
                step = x.get("env_steps")
                if step is None:
                    continue
                out[_gridkey(int(step))] = float(sr)
            if out:
                return out
            raise RuntimeError(f"{run_name}: WandB history exists but no usable evaluation metric")
    raise RuntimeError(f"{run_name}: WandB run not found in {DSRL_PROJECT}")


# 'ours' = pi0-feat subgoal on LIBERO-10 Task 8, 3 seeds, all eval_num_envs=8. Pulled fresh
# from wandb. As of 2026-07-21 this is a clean 3x500k trio (seeds 1/2/3): die3f92s (seed1,
# finished @500k, ss .936), mbdn13bk (seed2, finished @500k, ss 1.00), fs0ik9u1 (seed3,
# finished @500k, ss .945). The seed0 re-run c1q0cnsa CRASHED at 50k, so it is dropped in
# favour of the finished seed1 die3f92s (same seed slot, same eval_num_envs=8) -- no wait,
# no thinned tail: all three now span the full 500k. (die3f92s was excluded earlier only
# because a seed0 re-run was in flight; that re-run crashed, so die3f92s stands.)
OURS_RUNS = [(CR, "die3f92s"), (CR, "mbdn13bk"), (CR, "fs0ik9u1")]


def _gridkey(s):
    return int(round(s / 10000.0)) * 10000


@lru_cache(maxsize=None)
def pull_curve(proj, rid):
    r = api.run(f"{ENT}/{proj}/{rid}")
    h = r.history(keys=["eval/success_rate"], samples=10000, pandas=False)
    d = {}
    for x in h:
        v = x.get("eval/success_rate")
        if v is not None:
            d[_gridkey(int(x["_step"]))] = float(v)
    return d


def agg_seeds(seeds):
    """mean +/- s.e.m. at each gridkey over the seeds present there (union of steps)."""
    allk = sorted(set().union(*[set(s) for s in seeds])) if seeds else []
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


SP = "/tmp"
PAPER = "/mnt/mnt/data/resfit/paper"
D = json.load(open(f"{PAPER}/libero_curves.json"))

# ---- style: identical to paper Fig.1 (plot_main_1x4.py) ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
BLUE, GREEN, PURPLE = "#2a78d6", "#008300", "#7a3fb5"  # CVD-validated 3-hue set
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10.5,
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK})

DASH = (0, (4, 2))
# curve_key -> (color, lw, linestyle, zorder, legend_label)
CURVES = {
    "flat_h10":  (BLUE,   1.8, "-",  3, "flat residual (h10)"),
    "flat_h30":  (BLUE,   1.5, DASH, 2, "flat residual (h30)"),
    "subgoal":   (GREEN,  2.4, "-",  5, "SHORE-RL"),
    "dsrl":      (PURPLE, 2.0, "-",  4, "DSRL"),
}
# per-panel: task title -> list of (curve_key, run_id) for wandb-run curves drawn from D.
# Task 6 panel removed per request; only Task 8. The flat-residual curves (flat h10/h30) are
# also removed per request, so PANELS carries no D-curves. The two curves that remain -- the
# pi0-feat subgoal (ours, 3-seed, from OURS_RUNS) and the DSRL baseline (3-seed, from
# DSRL_RUNS) are all drawn in the pi==0 branch below.
PANELS = [
    ("LIBERO-10 · Task 8", []),
]


n = len(PANELS)
fig, axes = plt.subplots(1, n, figsize=(3.9 if n == 1 else 3.6 * n, 2.7),
                         sharey=True, squeeze=False)
axes = axes[0]
handle_for = {}
for pi, (ax, (title, curves)) in enumerate(zip(axes, PANELS)):
    for ckey, rid in curves:
        rec = D[rid]
        xs, ys = rec["xs_k"], rec["ys"]
        c, lw, ls, z, lab = CURVES[ckey]
        ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
        handle_for.setdefault(ckey, Line2D([0], [0], color=c, lw=lw, ls=ls))
        # open marker at end of a run that never reached 500k (crashed / stopped early)
        if xs[-1] < 450:
            ax.plot(xs[-1], ys[-1], "o", mfc="white", mec=c, mew=1.3, ms=5, zorder=z + 1)
    # ours (pi0-feat subgoal, 3 seeds) + DSRL baseline + frozen pi0-BC reference: Task 8 panel
    if pi == 0:
        # DSRL baseline: 3 seeds (0/1/2), all 50-episode eval on Task 8, pulled
        # directly from W&B through the full 400k plotting budget.
        dseeds = [pull_remote_dsrl_grid(name) for name in DSRL_RUNS]
        dxs, dmean, dsem = agg_seeds(dseeds)
        c, lw, ls, z, lab = CURVES["dsrl"]
        ax.fill_between(dxs, [m - e for m, e in zip(dmean, dsem)],
                        [m + e for m, e in zip(dmean, dsem)],
                        color=c, alpha=0.15, lw=0, zorder=z - 1)
        ax.plot(dxs, dmean, color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
        # DSRL end-of-curve open marker suppressed per author request (2026-07-21): the
        # panel is capped at 400k, DSRL fills the whole visible range (ends ~405k), so no
        # incompleteness circle is drawn on its tail.
        handle_for.setdefault("dsrl", Line2D([0], [0], color=c, lw=lw, ls=ls))
        # ours: 3-seed mean + s.e.m. band (no per-seed traces)
        oseeds = [pull_curve(p, r) for p, r in OURS_RUNS]
        oxs, omean, osem = agg_seeds(oseeds)
        oc, olw, ols, oz, olab = CURVES["subgoal"]
        ax.fill_between(oxs, [m - e for m, e in zip(omean, osem)],
                        [m + e for m, e in zip(omean, osem)],
                        color=oc, alpha=0.15, lw=0, zorder=oz - 1)
        ax.plot(oxs, omean, color=oc, lw=olw, ls=ols, zorder=oz, solid_capstyle="round")
        if oxs[-1] < 450:                 # aggregate ended before 500k
            ax.plot(oxs[-1], omean[-1], "o", mfc="white", mec=oc, mew=1.3, ms=5, zorder=oz + 1)
        handle_for.setdefault("subgoal", Line2D([0], [0], color=oc, lw=olw, ls=ols))
    ax.set_title(title, fontsize=12.0, fontweight="normal", loc="left", pad=5)
    ax.set_xlabel("Env steps (k)", fontsize=12.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, 400)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate", fontsize=12.5)

# shared legend below (ordered)
ORDER = ["flat_h10", "flat_h30", "subgoal", "dsrl"]
order = [k for k in ORDER if k in handle_for]
fig.legend([handle_for[k] for k in order], [CURVES[k][4] for k in order],
           loc="lower center", ncol=4, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, -0.03), columnspacing=1.4, handlelength=2.0)
# ○-marker meaning + eval-protocol caveat live in the LaTeX \caption

fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=1.6)
out_png = f"{SP}/fig_libero10.png"
out_pdf = f"{PAPER}/figure/fig_libero10.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print("wrote", out_png, "and", out_pdf)
for title, curves in PANELS:
    for ckey, rid in curves:
        r = D[rid]
        print(f"{title:22s} {ckey:9s} {rid} state={r['state']:9s} "
              f"end=({r['xs_k'][-1]:.0f}k,{r['ys'][-1]:.2f}) max={max(r['ys']):.2f}")
_oseeds = [pull_curve(p, r) for p, r in OURS_RUNS]
for (p, rid), s in zip(OURS_RUNS, _oseeds):
    last = max(s) if s else -1
    print(f"{'LIBERO-10 · Task 8':22s} {'ours':9s} {rid} n_eval={len(s):2d} "
          f"last={last//1000 if s else -1}k end_sr={s.get(last, float('nan')):.2f}")
_oxs, _omean, _osem = agg_seeds(_oseeds)
print(f"{'LIBERO-10 · Task 8':22s} {'ours(agg)':9s} union_to={_oxs[-1]:.0f}k "
      f"end_mean={_omean[-1]:.2f} steady(last5 of mean)={st.mean(_omean[-5:]):.3f}")
_dseeds = [pull_remote_dsrl_grid(name) for name in DSRL_RUNS]
for i, s in enumerate(_dseeds):
    last = max(s) if s else -1
    tail = sorted(s.values())  # not used for steady; print reach + end
    pts = [v for _, v in sorted(s.items())]
    dt = pts[max(1, int(len(pts) * 0.8)):] or pts
    print(f"{'LIBERO-10 · Task 8':22s} {'dsrl':9s} seed{i} n_eval={len(s):2d} "
          f"last={last//1000 if s else -1}k end_sr={s.get(last, float('nan')):.2f} "
          f"steady(last20%)={sum(dt)/len(dt):.3f}")
_dxs, _dmean, _dsem = agg_seeds(_dseeds)
print(f"{'LIBERO-10 · Task 8':22s} {'dsrl(agg)':9s} 3seed union_to={_dxs[-1]:.0f}k "
      f"end_mean={_dmean[-1]:.2f} steady(last5 of mean)={st.mean(_dmean[-5:]):.3f}")
