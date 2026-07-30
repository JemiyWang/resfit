"""Fig.4 v2 -- staged-reward (full) vs PotHIQL, now with the 07-18/19 PotHIQL re-run.

v2 of plot_staged_vs_pothiql_threading_piece.py. The v1 arms are kept EXACTLY as they
were (same run ids, same colors/linestyles/z-order, same aggregation); v2 only ADDS a
third arm per panel -- the PotHIQL re-runs launched 2026-07-18/19, still in flight.

WHY THE RE-RUN IS NOT JUST "MORE SEEDS" (wandb config diff, 2026-07-20):
  the v1 pothiql runs (y91hfmqf / fxe2pvvx / t5ohavo2) have
      online_finetune_high_actor = False, online_finetune_value = False
  i.e. joint online finetuning was OFF, while the staged "full" arm has both True.
  So the v1 orange arm differs from green in TWO variables (reward AND joint), which
  the paper caption ("identical ... except the reward") does not reflect.
  The new runs set both to True, leaving reward_shaping (staged->potential) +
  potential_source (stage->hiql) -- and their required hiql_value_ckpt /
  offline_stage_cache attachments -- as the ONLY differences vs the green arm.
  The blue arm is therefore the clean Ablation-C contrast; the orange one is not.

The three arms per panel:
  full          -- staged reward + joint          ("_staged_joint")        3 seeds @500k
  pothiql       -- potential reward, joint OFF    ("_pothiql", v1)         crashed 240-260k
  pothiql_joint -- potential reward + joint ON    ("_pothiql_joint_seedN") IN PROGRESS

DATA STATUS -- all blue-arm seeds reached 500k (snapshot 2026-07-21): both panels are
now a full 3-seed comparison at 500k, so no open markers / thinned tails are drawn. The
fade_partial / open-marker logic below is kept but is inert now that every seed is at
500k."""
import math

from aaai_type1_matplotlib import configure_aaai_type1_matplotlib

configure_aaai_type1_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import wandb

api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"
CR = "dexmg-chunk-residual"

STY = {
    "full":    {"label": "Stage shaping",
                "color": "#008300", "ls": "-", "lw": 2.6, "z": 6},
    # v1 "pothiql" arm (potential reward, joint OFF -- old) removed per request
    # 2026-07-20: it was the two-variable contrast (reward AND joint both differ vs
    # full), superseded by the clean joint-ON re-run below. Runs kept in git history.
    # new arm: blue reads clearly against both v1 hues under protan/deutan/tritan
    # (validated); it also carries its own dash-dot pattern so identity never rests
    # on hue alone -- the v1 green/orange pair is only 1.9 dE apart under protan.
    "pothiql_joint": {"label": "Self-derived shaping",
                      "color": "#1f5fa9", "ls": (0, (4.5, 1.2, 1, 1.2)), "lw": 2.3, "z": 7,
                      "fade_partial": True},
}
DRAW_ORDER = ["pothiql_joint", "full"]        # full on top; v1 "pothiql" arm removed

PANELS = {
    "Threading": {
        "full":    [(CR, "cbmgm6c0"), (CR, "s2rbsvxp"), (CR, "ulguni4f")],   # 3 runs @500k
        # v1 "pothiql" (joint OFF) removed 2026-07-20: y91hfmqf / fxe2pvvx (crashed @260k/240k)
        # 07-19 re-run, joint ON, running: seed1 ~453k, seed2 ~220k, seed3 ~217k
        "pothiql_joint": [(CR, "l3u0nkbe"), (CR, "uqtmeu2c"), (CR, "fjo6nmk4")],
    },
    "ThreePiece": {
        "full":    [(CR, "kmtsayff"), (CR, "m2s74dqm"), (CR, "cgtmwrv3")],   # 3 seeds @500k
        # v1 "pothiql" (joint OFF) removed 2026-07-20: t5ohavo2 (crashed @250k)
        # 07-18/19 re-run, joint ON, running: seed1 ~395k, seed2 ~278k, seed3 ~274k
        "pothiql_joint": [(CR, "2guzcnl6"), (CR, "ba6368e0"), (CR, "gkp4obqi")],
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
    xs, mean, sem, nseed = [], [], [], []
    for k in allk:
        vals = [s[k] for s in seeds if k in s]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        xs.append(k / 1000.0)
        mean.append(m)
        sem.append(math.sqrt(var) / math.sqrt(len(vals)))
        nseed.append(len(vals))
    return xs, mean, sem, nseed


DATA = {task: {arm: [pull(p, r) for p, r in runs] for arm, runs in arms.items()}
        for task, arms in PANELS.items()}

# frozen base = pooled step-0 across this panel's arms (residual==0 -> same base policy)
BASE = {}
for task, arms in DATA.items():
    firsts = [s[0] for seeds in arms.values() for s in seeds if 0 in s]
    BASE[task] = sum(firsts) / len(firsts) if firsts else None

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd7"
plt.rcParams.update({"font.size": 17,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": True,
                     "axes.spines.right": True, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK})

n = len(PANELS)
fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.65), sharey=True, squeeze=False)
axes = axes[0]
handles = {}
for ax, task in zip(axes, PANELS):
    # frozen-base reference line intentionally omitted per request (BASE still computed
    # for the stats printout below).
    for arm in DRAW_ORDER:
        seeds = DATA[task].get(arm, [])
        if not seeds:
            continue
        st = STY[arm]
        c = st["color"]
        xs, mean, sem, nseed = agg(seeds)
        ax.fill_between(xs, [m - e for m, e in zip(mean, sem)], [m + e for m, e in zip(mean, sem)],
                        color=c, alpha=0.15, lw=0, zorder=st["z"] - 1)
        if st.get("fade_partial") and len(seeds) > 1:
            # seeds are of unequal length, so the tail of the mean silently drops to
            # fewer seeds. Draw the full-cohort prefix solid and the thinned tail
            # translucent, so a tail carried by one seed cannot be misread as an
            # n-seed mean. (nseed is non-increasing: every seed starts at step 0.)
            full = max(nseed)
            cut = max(i for i, ns in enumerate(nseed) if ns == full) + 1
            ax.plot(xs[cut - 1:], mean[cut - 1:], color=c, lw=st["lw"] * 0.55, ls=st["ls"],
                    alpha=0.5, zorder=st["z"], solid_capstyle="round")
            ln, = ax.plot(xs[:cut], mean[:cut], color=c, lw=st["lw"], ls=st["ls"],
                          zorder=st["z"], solid_capstyle="round")
            if cut < len(xs):            # tick where the cohort thins out
                ax.plot(xs[cut - 1], mean[cut - 1], "|", color=c, ms=7, mew=1.4,
                        zorder=st["z"] + 1)
        else:
            ln, = ax.plot(xs, mean, color=c, lw=st["lw"], ls=st["ls"], zorder=st["z"],
                          solid_capstyle="round")
        if xs[-1] < 495:                 # open marker: arm not yet at 500k
            ax.plot(xs[-1], mean[-1], "o", mfc="white", mec=c, mew=1.4, ms=6, zorder=st["z"] + 1)
        handles[arm] = ln
    display_name = "PieceAssembly" if task == "ThreePiece" else task
    ax.set_title(display_name, fontsize=16, fontweight="normal",
                 loc="left", pad=6)
    ax.set_xlabel("Env steps (k)", fontsize=16)
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=14, width=0.8, length=3.5)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate", fontsize=17)

order = [k for k in DRAW_ORDER[::-1] if k in handles]
fig.legend([handles[k] for k in order], [STY[k]["label"] for k in order],
           loc="lower center", ncol=len(order), frameon=False, fontsize=14,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.4, handlelength=2.4)
# footnote about open markers / thinned tails removed 2026-07-21: all seeds now at
# 500k, so neither is drawn and the note no longer applies.
fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.4)
out_pdf = "/mnt/mnt/data/resfit/paper/figure/fig_staged_vs_pothiql_threading_piece_v2.pdf"
out_png = "/tmp/fig_staged_vs_pothiql_threading_piece_v2.png"
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
        xs, mean, sem, nseed = agg(seeds)
        m = sum(s) / len(s)
        var = sum((v - m) ** 2 for v in s) / (len(s) - 1) if len(s) > 1 else 0.0
        tails = sorted(max(k for k in sd) / 1000 for sd in seeds)
        print(f"  {arm:13s} to={xs[-1]:.0f}k n_seed={len(s)} seed_tails={tails} "
              f"n_at_tail={nseed[-1]} ss per-seed={[round(v,3) for v in s]} "
              f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(s)):.3f} s.e.m.")


# ---- matched-window comparison -------------------------------------------------
# The per-arm steady state above is each arm's OWN last 20%, so it compares the
# blue arm's ~200k window against the green arm's ~450k window. For a like-for-like
# read, also report every arm inside the window all three arms actually cover.
print("\n=== matched window (all arms measured over the same last-40k-of-common-range) ===")
for task in PANELS:
    arms = {a: s for a, s in DATA[task].items() if s}
    common = min(max(k for k in sd) for seeds in arms.values() for sd in seeds)
    lo = max(0, common - 40000)
    print(f"\n[{task}] common range ends at {common/1000:.0f}k -> window [{lo/1000:.0f}k, "
          f"{common/1000:.0f}k], frozen base ~ {BASE[task]:.3f}")
    for arm in DRAW_ORDER[::-1]:
        seeds = arms.get(arm, [])
        if not seeds:
            continue
        per = []
        for sd in seeds:
            vals = [v for k, v in sd.items() if lo <= k <= common]
            if vals:
                per.append(sum(vals) / len(vals))
        if not per:
            print(f"  {arm:13s} (no evals in window)")
            continue
        m = sum(per) / len(per)
        var = sum((v - m) ** 2 for v in per) / (len(per) - 1) if len(per) > 1 else 0.0
        print(f"  {arm:13s} n_seed={len(per)} per-seed={[round(v,3) for v in per]} "
              f"mean={m:.3f} +/- {math.sqrt(var)/math.sqrt(len(per)):.3f} s.e.m.")
