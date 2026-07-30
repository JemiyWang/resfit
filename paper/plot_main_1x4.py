import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# pull final picks fresh so the script is self-contained
import wandb
api = wandb.Api(timeout=120)
ENT = "674575221-beijing-institute-of-technology"; CR = f"{ENT}/dexmg-chunk-residual"
# NOTE: the "unsafe" (goal-cond. V(s,z) potential) arm is intentionally NOT plotted
# in the main figure -- it lives in Table 2 (goal-cond pot unsafe row) + the Fig.2
# drift plot. Only the 4 arms below are drawn here.
PICKS = {
 "ThreePiece": {"vanilla":(f"{ENT}/dexmg-threepiece-final","hda0bw5s"),
                "subgoal":(CR,"bdou2t10"), "hires":(CR,"t5ohavo2"),
                "staged":(CR,"iqjfnogl")},   # hires=pothiql t5ohavo2 to 250k
 "Threading":  {"vanilla":(f"{ENT}/dexmg-twoarmthreading-final","c1p7xd7y"),
                "subgoal":(CR,"17l8wdlt"), "hires":(CR,"y91hfmqf"),
                "staged":(CR,"556z5940")},
 "LiftTray":   {"vanilla":(f"{ENT}/dexmg-lifttray-final","0gb3ot7o"),
                "subgoal":(CR,"ef4ejor0"), "hires":(CR,"somalim5")},
 "Pouring":    {"vanilla":(f"{ENT}/dexmg-pouring-final","j5xcqzsn"),
                "subgoal":(CR,"5ttsseik"), "hires":(CR,"ufrsoegx")},
 "CanSort":    {"vanilla":(f"{ENT}/dexmg-cansorting-final","swoei1ya"),   # resfit-original vanilla, ss0.96
                "subgoal":(CR,"q464siq2")},                                # user's subgoal-only act_feat, ss1.00
}
D={}
for task,arms in PICKS.items():
    D[task]={}
    for arm,(proj,rid) in arms.items():
        r=api.run(f"{proj}/{rid}")
        h=r.history(keys=["eval/success_rate"],samples=5000,pandas=False)
        D[task][arm]=sorted([(int(x["_step"])/1000,float(x["eval/success_rate"])) for x in h if x.get("eval/success_rate") is not None])

# ---- style ----
INK,MUTED,GRID="#0b0b0b","#52514e","#dcdcd7"
STYLE = {  # arm -> (label, color, lw, linestyle, zorder)
 "vanilla":("Residual RL (published; no BC)",     "#8a8a86", 1.6, (0,(4,2)), 2),
 "subgoal":("+ subgoal + BC",                     "#2a78d6", 1.8, "-",       3),
 "hires":  ("SHORE-RL (+ subgoal + BC + $V(s)$)", "#008300", 2.4, "-",       5),
 "staged": ("+ subgoal + BC + stage",             "#4a3aa7", 1.8, "-",       4),
}
ORDER=["vanilla","subgoal","hires","staged"]
plt.rcParams.update({"font.family":"sans-serif","font.size":9,
 "axes.edgecolor":MUTED,"axes.linewidth":0.8,"axes.spines.top":False,
 "axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED,
 "text.color":INK,"axes.labelcolor":INK})

fig,axes=plt.subplots(1,5,figsize=(13.5,2.6),sharey=True)
tasks=["ThreePiece","Threading","LiftTray","Pouring","CanSort"]
handles={}
for ax,task in zip(axes,tasks):
    if task=="CanSort":                   # short-horizon reference panel: tint + note
        ax.set_facecolor("#f5f5f3")
    for arm in ORDER:
        if arm not in D[task]: continue
        pts=D[task][arm]; xs=[x for x,_ in pts]; ys=[v for _,v in pts]
        lab,c,lw,ls,z=STYLE[arm]
        ln,=ax.plot(xs,ys,color=c,lw=lw,ls=ls,zorder=z,solid_capstyle="round")
        handles[arm]=ln
        # open marker at end of a partial run (<450k) to flag "not finished"
        if xs[-1] < 450:
            ax.plot(xs[-1],ys[-1],"o",mfc="white",mec=c,mew=1.3,ms=5,zorder=z+1)
    ttl = "CanSort (short-horiz. ref.)" if task=="CanSort" else task
    if task=="CanSort":
        ax.text(0.5,0.30,"short-horizon control:\nneither published\nnor ours collapses",
                transform=ax.transAxes,ha="center",va="center",fontsize=7.0,
                color="#52514e",style="italic",linespacing=1.3)
    ax.set_title(ttl,fontsize=9.0,fontweight="bold",loc="left",pad=5)
    ax.set_xlabel("Env steps (k)"); ax.set_ylim(-0.03,1.03); ax.set_xlim(0,510)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.grid(axis="y",color=GRID,lw=0.7,zorder=0); ax.set_axisbelow(True)
axes[0].set_ylabel("Eval success rate")

# shared legend below
order=[a for a in ORDER if a in handles]
fig.legend([handles[a] for a in order],[STYLE[a][0] for a in order],
           loc="lower center",ncol=len(order),frameon=False,fontsize=8.5,
           bbox_to_anchor=(0.5,-0.06),columnspacing=1.6,handlelength=2.0)
fig.tight_layout(rect=[0,0.04,1,1],w_pad=1.3)
out_pdf="/mnt/mnt/data/resfit/paper/figure/fig_main_1x4_DRAFT.pdf"
out_png="/tmp/claude-0/-mnt-mnt-data/958f8315-f374-4e53-960c-37daf2fa26f8/scratchpad/fig_main_1x4.png"
fig.savefig(out_pdf,bbox_inches="tight"); fig.savefig(out_png,dpi=140,bbox_inches="tight")
print("wrote",out_pdf)
# report end-steps
for task in tasks:
    print(task, {a:(f"{D[task][a][-1][0]:.0f}k,ss~{D[task][a][-1][1]:.2f}") for a in ORDER if a in D[task]})
