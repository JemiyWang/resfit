"""解析 chunk_residual 训练 log,2x2 画:target_q / residual_norm /
(整体成功率 + 每阶段未回退率) / (best.pt 每阶段到达率柱状)。
用法: python plot_stage_diag.py <log_path> [out_png]
注: 需从能 import resfit 的环境运行(同 eval_stage_reach.py)。"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resfit.rl_finetuning.chunk_residual.stage_log_parse import (
    parse_stage_purity_line,
    load_reach_sidecar,
)

log_path = sys.argv[1] if len(sys.argv) > 1 else "cl1_queue_potential.log"
out_png = sys.argv[2] if len(sys.argv) > 2 else log_path.rsplit(".", 1)[0] + "_stagediag.png"

steps, succ = [], []
tq = {s: [] for s in range(4)}      # target_q per stage
rn = {s: [] for s in range(4)}      # residual_norm per stage
tq_steps = []                        # stage-diag 行对应的 step
purity = {s: [] for s in range(4)}   # 每阶段未回退率
purity_steps = []                    # stage-purity 行对应的 step

cur_step = None
with open(log_path) as f:
    for line in f:
        m = re.search(r"env_steps (\d+)\] eval success_rate=([-0-9.]+)", line)
        if m:
            cur_step = int(m.group(1))
            steps.append(cur_step); succ.append(float(m.group(2)))
            continue
        if line.startswith("[stage-diag]") and cur_step is not None:
            tq_steps.append(cur_step)
            for s in range(4):
                t = re.search(rf"target_q/stage{s}=([-0-9.]+)", line)
                r = re.search(rf"residual_norm/stage{s}=([-0-9.]+)", line)
                tq[s].append(float(t.group(1)) if t else float("nan"))
                rn[s].append(float(r.group(1)) if r else float("nan"))
        elif line.startswith("[stage-purity]") and cur_step is not None:
            pmap = parse_stage_purity_line(line)
            purity_steps.append(cur_step)
            for s in range(4):
                purity[s].append(pmap.get(s, float("nan")))

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# (0,0) target_q per stage
for s in range(4):
    ax[0, 0].plot(tq_steps, tq[s], marker="o", ms=3, color=colors[s], label=f"stage{s}")
ax[0, 0].set_title("target_q per stage (training batch mean)")
ax[0, 0].set_xlabel("env_steps"); ax[0, 0].set_ylabel("target_q")
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

# (0,1) residual_norm per stage
for s in range(4):
    ax[0, 1].plot(tq_steps, rn[s], marker="o", ms=3, color=colors[s], label=f"stage{s}")
ax[0, 1].set_title("residual_norm per stage"); ax[0, 1].set_xlabel("env_steps")
ax[0, 1].set_ylabel("||residual||"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

# (1,0) 整体成功率 + 每阶段未回退率(训练 rollout)
ax[1, 0].plot(steps, succ, marker="s", ms=4, color="black", label="success (overall, eval)")
for s in range(4):
    ax[1, 0].plot(purity_steps, purity[s], marker="o", ms=3, color=colors[s],
                  alpha=.6, label=f"stage{s} stay-rate")
ax[1, 0].set_title("eval success + per-stage non-regress (training rollout)")
ax[1, 0].set_xlabel("env_steps"); ax[1, 0].set_ylabel("rate")
ax[1, 0].set_ylim(-0.02, 1.05); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

# (1,1) best.pt 每阶段到达率(柱状,来自 sidecar)
reach_path = log_path.rsplit(".", 1)[0] + "_reach.json"
reach = load_reach_sidecar(reach_path)
if reach is None:
    ax[1, 1].text(0.5, 0.5, "run re-eval to populate\n(<log>_reach.json missing)",
                  ha="center", va="center", transform=ax[1, 1].transAxes)
    ax[1, 1].set_title("per-stage reach @ best.pt")
else:
    ks = sorted(reach["reach"])
    ax[1, 1].bar([str(k) for k in ks], [reach["reach"][k] for k in ks],
                 color=[colors[k % len(colors)] for k in ks])
    ax[1, 1].set_ylim(0, 1.05)
    ax[1, 1].set_title(f"per-stage reach @ best.pt (step {reach['step']})")
    ax[1, 1].set_xlabel("stage k  (reach = P[max_stage >= k])")
    ax[1, 1].set_ylabel("reach rate")
ax[1, 1].grid(alpha=.3, axis="y")

fig.suptitle(log_path.split("/")[-1], y=1.01)
fig.tight_layout()
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"saved -> {out_png}")
print(f"diag points: {len(tq_steps)}, eval points: {len(steps)}, "
      f"purity points: {len(purity_steps)}")
