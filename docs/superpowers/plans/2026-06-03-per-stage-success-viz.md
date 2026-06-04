# Per-stage 成功率可视化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `plot_stage_diag.py` 改成 2x2(新增每阶段未回退率曲线 + best.pt 每阶段到达率柱状图),并提供一个 num_envs=1 的 reeval 脚本产出到达率 sidecar。

**Architecture:** 纯解析/聚合逻辑下沉到新 package 模块 `stage_log_parse.py`(可被 pytest 导入并 TDD);plot 脚本与 reeval 脚本只做编排,调用纯函数 + 已有的 `stage_reach_rates`。reeval 用单环境绕开 wrapper 单标量 stage 闩锁的多环境污染。

**Tech Stack:** Python, matplotlib (Agg), pytest, torch(仅 reeval 脚本运行时,需 GPU/conda env residual)。

---

## File Structure

- 新建 `resfit/rl_finetuning/chunk_residual/stage_log_parse.py` — 三个纯函数:
  `parse_stage_purity_line` / `load_reach_sidecar` / `fold_episode_max_stage`。
- 新建 `resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py` — 上述单测。
- 改 `outputs_chunk/plot_stage_diag.py` — 1x3 → 2x2,导入上面纯函数。
- 新建 `resfit/rl_finetuning/chunk_residual/reeval_stage_reach.py` — best.pt → `<log>_reach.json`。

仓库根目录:`/data2/RL/residual-offpolicy-rl`。所有 `pytest` 从根目录跑。

---

## Task 1: 纯函数模块 stage_log_parse.py（解析 + 折叠）

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/stage_log_parse.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py`

- [ ] **Step 1: 写失败测试**

写入 `resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py`:

```python
"""stage_log_parse 纯函数:purity 行解析 / reach sidecar 读取 / 单步 stage 折叠。"""
import json

from resfit.rl_finetuning.chunk_residual.stage_log_parse import (
    parse_stage_purity_line,
    load_reach_sidecar,
    fold_episode_max_stage,
)


# --- parse_stage_purity_line ---------------------------------------------
def test_parse_purity_real_line():
    line = ("[stage-purity] regress 3547/9980=35.5%  stage0:0/2404=0%  "
            "stage1:413/1963=21%  stage2:7/1238=1%  stage3:3127/4375=71%\n")
    out = parse_stage_purity_line(line)
    # 未回退率 = 1 - regress/total
    assert out[0] == 1.0          # 0/2404
    assert abs(out[1] - (1 - 413 / 1963)) < 1e-9
    assert abs(out[3] - (1 - 3127 / 4375)) < 1e-9


def test_parse_purity_skips_zero_total():
    # b==0 的桶不应出现在结果里(避免除零)
    out = parse_stage_purity_line("[stage-purity] regress 0/0=0%  stage2:0/0=0%")
    assert 2 not in out


def test_parse_purity_non_purity_line_returns_empty():
    assert parse_stage_purity_line("[env_steps 10000] eval success_rate=0.200") == {}


# --- load_reach_sidecar ---------------------------------------------------
def test_load_reach_sidecar_roundtrip(tmp_path):
    p = tmp_path / "run_reach.json"
    p.write_text(json.dumps({"step": 42, "n_episodes": 50,
                             "reach": {"1": 1.0, "2": 0.8, "3": 0.2}}))
    got = load_reach_sidecar(str(p))
    assert got["step"] == 42
    assert got["reach"] == {1: 1.0, 2: 0.8, 3: 0.2}   # key 转 int


def test_load_reach_sidecar_missing_returns_none(tmp_path):
    assert load_reach_sidecar(str(tmp_path / "nope.json")) is None


# --- fold_episode_max_stage ----------------------------------------------
def test_fold_takes_max():
    assert fold_episode_max_stage(2, 3, reward=0.0, top_stage=4) == 3
    assert fold_episode_max_stage(3, 1, reward=0.0, top_stage=4) == 3


def test_fold_success_jumps_to_top():
    assert fold_episode_max_stage(1, 1, reward=1.0, top_stage=4) == 4


def test_fold_no_success_no_jump():
    assert fold_episode_max_stage(0, 0, reward=0.5, top_stage=4) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named '...stage_log_parse'`

- [ ] **Step 3: 写最小实现**

写入 `resfit/rl_finetuning/chunk_residual/stage_log_parse.py`:

```python
"""chunk_residual 训练 log 的纯解析/聚合(无副作用,供 plot_stage_diag 与 reeval 复用)。"""
from __future__ import annotations

import json
import os
import re

_PURITY_RE = re.compile(r"stage(\d+):(\d+)/(\d+)=")


def parse_stage_purity_line(line: str) -> dict[int, float]:
    """从一行 [stage-purity] 解析每 stage 的未回退率 = 1 - regress/total。

    `stageK:a/b=P%` 中 a=回退步数, b=该桶总步数。b==0 的桶跳过(不入结果)。
    非 [stage-purity] 行返回 {}。
    """
    if "[stage-purity]" not in line:
        return {}
    out: dict[int, float] = {}
    for m in _PURITY_RE.finditer(line):
        k, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if b > 0:
            out[k] = 1.0 - a / b
    return out


def load_reach_sidecar(path: str) -> dict | None:
    """读 <log>_reach.json;不存在返回 None。reach 的 key 还原成 int。"""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    reach = {int(k): float(v) for k, v in data["reach"].items()}
    return {"step": int(data["step"]), "reach": reach}


def fold_episode_max_stage(prev_max: int, max_stage_in_chunk: int,
                           reward: float, top_stage: int) -> int:
    """单步更新 episode 内最高 stage:取最高;若该步成功(reward>=1)→ top_stage。"""
    new = max(int(prev_max), int(max_stage_in_chunk))
    if reward >= 1.0:
        new = max(new, int(top_stage))
    return new
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py -q`
Expected: PASS（8 passed）

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/stage_log_parse.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_log_parse.py
git commit -m "feat(chunk_residual): stage_log_parse 纯函数(purity/reach/fold)+测试"
```

---

## Task 2: plot_stage_diag.py 改 2x2

**Files:**
- Modify: `outputs_chunk/plot_stage_diag.py`（整文件重写）

依赖 Task 1 的 `stage_log_parse`。此任务无单测（渲染脚本），靠在真实 log 上跑出 png 验证。

- [ ] **Step 1: 整文件重写**

把 `outputs_chunk/plot_stage_diag.py` 整体替换为:

```python
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
```

- [ ] **Step 2: 在真实 log 上跑,验证出图且 purity points > 0**

Run:
```bash
cd /data2/RL/residual-offpolicy-rl/outputs_chunk && python plot_stage_diag.py cl1_queue_staged_as0.05.log
```
Expected: 打印 `saved -> cl1_queue_staged_as0.05_stagediag.png`,且 `purity points:` 大于 0;
sidecar 尚未生成,(1,1) 显示 "run re-eval to populate"。不报错即通过。

- [ ] **Step 3: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add outputs_chunk/plot_stage_diag.py
git commit -m "feat(plot): stage-diag 改 2x2,加每阶段未回退率曲线+到达率柱状"
```

---

## Task 3: reeval_stage_reach.py（best.pt → reach sidecar）

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/reeval_stage_reach.py`

依赖 Task 1 的 `fold_episode_max_stage`。需 GPU 才能实跑,故只做"导入冒烟",实跑由用户执行。

- [ ] **Step 1: 写脚本**

写入 `resfit/rl_finetuning/chunk_residual/reeval_stage_reach.py`:

```python
"""重评某 run 的 best.pt:num_envs=1 逐 episode 记最高 stage,产出 <log>_reach.json
(供 plot_stage_diag.py 的 (1,1) 柱状图)。

为何 num_envs=1:ChunkResidualEnvWrapper 的 stage 闩锁是单标量、只读 env0
(chunk_env_wrapper.py:150/170),多环境会互相污染 → 必须单环境才有正确 per-episode 归属。

用法(仓库根目录, conda env residual):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.reeval_stage_reach \
    --run_dir outputs_chunk/cl1_queue_potential \
    --log outputs_chunk/cl1_queue_potential.log
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy
from resfit.rl_finetuning.chunk_residual.stage_reach import stage_reach_rates
from resfit.rl_finetuning.chunk_residual.stage_log_parse import fold_episode_max_stage
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="含 best.pt 的目录")
    p.add_argument("--log", required=True, help="对应 log;sidecar 写到 <log>_reach.json")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=20000, help="安全上限,防卡死")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.run_dir, "best.pt"),
                      map_location=args.device, weights_only=False)
    cfg_args = ckpt["config"]                       # 训练时存的 argparse Namespace
    step = int(ckpt.get("global_step", -1))
    task = cfg_args.task
    n_stages = NUM_STAGES[task]
    top_stage = n_stages - 1

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    meta = LeRobotDataset(cfg_args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=cfg_args.action_scale,
        min_range_per_dim=cfg_args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    base_policy = build_base_policy(cfg_args.base_wandb_id, args.device)
    vec_env = create_vectorized_env(env_name=task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=cfg_args.chunk_length,
                                  reward_shaping_mode="none",
                                  base_action_mode=cfg_args.base_action_mode)

    image_keys = list(base_policy.config.image_features.keys())
    obs, _ = env.reset()
    img_c, img_h, img_w = obs[image_keys[0]].shape[1:]
    state_dim = obs["observation.state"].shape[1]
    action_dim = env.action_dim * cfg_args.chunk_length

    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor.action_scale = cfg_args.action_scale
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.train(False)

    ep_max_stages: list[int] = []
    ep_max = 0
    for _ in range(args.max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        obs, reward, term, trunc, info = env.step(action)
        ep_max = fold_episode_max_stage(ep_max, info.get("max_stage_in_chunk", 0),
                                        float(reward[0]), top_stage)
        if bool((term | trunc).any()):
            ep_max_stages.append(ep_max)
            ep_max = 0
            obs, _ = env.reset()
            if len(ep_max_stages) >= args.n_episodes:
                break

    rates = stage_reach_rates(ep_max_stages, n_stages)
    out = {"step": step, "n_episodes": len(ep_max_stages),
           "reach": {str(k): rates[k] for k in rates}}
    sidecar = args.log.rsplit(".", 1)[0] + "_reach.json"
    with open(sidecar, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[reeval] task={task} step={step} n={len(ep_max_stages)} reach={rates}")
    print(f"  (自洽: reach[{top_stage}] 应 ≈ 该 run 的 success_rate)")
    print(f"saved -> {sidecar}")
    vec_env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 导入冒烟(无 GPU 也能跑)**

Run:
```bash
cd /data2/RL/residual-offpolicy-rl && python -c "import ast; ast.parse(open('resfit/rl_finetuning/chunk_residual/reeval_stage_reach.py').read()); print('syntax ok')"
```
Expected: `syntax ok`（纯语法校验;完整 import 需 conda env residual,留给用户实跑）

- [ ] **Step 3: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/reeval_stage_reach.py
git commit -m "feat(chunk_residual): reeval_stage_reach 脚本(best.pt -> reach sidecar)"
```

---

## 交付后(用户执行,需 GPU / conda env residual)

对两个目标 run 各跑一次 reeval 生成 sidecar,再重画图:

```bash
cd /data2/RL/residual-offpolicy-rl
for r in cl1_queue_potential cl1_queue_staged_as0.05; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data2/tmp \
  python -m resfit.rl_finetuning.chunk_residual.reeval_stage_reach \
    --run_dir outputs_chunk/$r --log outputs_chunk/$r.log
done
cd outputs_chunk
python plot_stage_diag.py cl1_queue_potential.log
python plot_stage_diag.py cl1_queue_staged_as0.05.log
```
校验:reeval 打印的 `reach[top_stage]` 应 ≈ 该 run 的 success_rate(已塌的 run 接近 0)。

---

## Self-Review

- **Spec 覆盖**:D1 plot 2x2(Task 2)、每阶段未回退率(Task 1 解析 + Task 2 (1,0))、
  reach 柱状(Task 1 sidecar 读 + Task 2 (1,1))、D2 reeval 脚本(Task 3,复用已有
  `stage_reach_rates`)、fold 纯函数(Task 1)——全覆盖。原 D2 的 evaluate_dexmg 插桩 +
  train 打印已在 spec 修订为作废(单标量 stage 闩锁多环境会算错),计划据修订版。
- **占位扫描**:无 TBD;每个改代码步骤含完整代码与确切命令/预期。
- **类型一致**:`fold_episode_max_stage(prev_max, max_stage_in_chunk, reward, top_stage)`、
  `load_reach_sidecar` 返回 `{"step":int,"reach":{int:float}}`、sidecar 写出 key 为 str、
  读入转 int —— 跨 Task 1/2/3 一致;`stage_reach_rates` 返回域 1..n_stages-1 与 sidecar/
  柱状图 key 一致。
