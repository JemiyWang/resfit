# HIQL 分层 Phase 2：AWR 高层策略 π^h 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 造高层策略 `π^h(z | s, g)`——给定当前 state `s` 与任务目标 `g`，输出 k 步后子目标的潜表征 `z=φ(s_t, s_{t+k})` 上的高斯分布；用 AWR（优势加权回归）从 Phase 1 冻结的 goal-conditioned value 抽取，离线训练并冻结存盘，供 Phase 3 online 提子目标。

**Architecture:** 新增**纯逻辑模块** `hiql_high_actor.py`（`HighActor` 网络、`awr_weight` 纯函数、`train_high_actor`、`save/load_high_actor`）+ **离线训练脚本** `train_hiql_high_actor.py`。**复用 Phase 1**：`build_gc_data`/`sample_gc_goals`/`stage_entries_from_instant`/`load_gc_value` 与冻结 `GoalConditionedVF` 的 `phi`。AWR 目标 `J = E[ exp(β·(V(s_{t+k},g)−V(s_t,g))) · log π^h(φ(s_t,s_{t+k}) | s,g) ]`（HIQL 式 6），k 步航点 + goal 混采。

**Tech Stack:** PyTorch（`torch.distributions.Normal`、state-independent log_std）、NumPy、pytest。conda env `residual`，命令从仓库根跑。

**前置：** Phase 1 已完成，产出 `outputs_chunk/three_piece_gc_value.pt`（`load_gc_value` 可载入、`v_stats.max>min`）。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`（新） | `HighActor`（MLP→Normal over rep_dim）、`awr_weight`（纯）、`train_high_actor`（用冻结 vf）、`save/load_high_actor`。 |
| `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`（新） | 离线训练 CLI：复用 `read_per_demo_states`+`load_stage_cache`+`build_gc_data`，载入冻结 `gc_value.pt`，串起 `train_high_actor`。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`（新） | Task 1–4 单测。 |

**不动**：Phase 1 的 `hiql_gc_value.py`（只 import 复用）、`hiql_value.py`、`q_agent.py`、`train_chunk_residual.py`。

---

## Task 1: 高层网络（HighActor）

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hiql_high_actor.py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor


def test_high_actor_forward_dist():
    ha = HighActor(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(6, 30)
    g = torch.randn(6, 30)
    dist = ha(s, g)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (6, 10)
    z = dist.rsample()
    assert z.shape == (6, 10)
    # log_prob 对 rep 维求和 -> (B,)
    lp = dist.log_prob(z).sum(-1)
    assert lp.shape == (6,)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_high_actor_forward_dist -q`
Expected: FAIL（`ModuleNotFoundError` / `ImportError: HighActor`）

- [ ] **Step 3: 写最小实现**

```python
# hiql_high_actor.py
"""HIQL 高层策略 π^h(z|s,g)(分层路 Phase 2)。

AWR 从 Phase 1 冻结的 goal-conditioned value 抽取:输出 k 步后子目标潜表征 z=φ(s_t,s_{t+k})
上的高斯。设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import _mlp, sample_gc_goals


class HighActor(nn.Module):
    """π^h(z | s, g):concat(s,g) -> trunk -> mean(rep_dim);log_std 为 state-independent 参数。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.trunk = _mlp(2 * state_dim, hidden, hidden)
        self.mean = nn.Linear(hidden, rep_dim)
        self.log_std = nn.Parameter(torch.zeros(rep_dim))

    def forward(self, s, g):
        h = torch.relu(self.trunk(torch.cat([s, g], dim=-1)))
        mean = self.mean(h)
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        return torch.distributions.Normal(mean, std.expand_as(mean))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_high_actor_forward_dist -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-high): HighActor π^h(z|s,g) 高斯输出"
```

---

## Task 2: AWR 权重（awr_weight 纯函数）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**

```python
def test_awr_weight():
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import awr_weight
    adv = torch.tensor([-1.0, 0.0, 1.0, 100.0])
    w = awr_weight(adv, beta=1.0, clip=100.0)
    assert torch.allclose(w[:3], torch.exp(torch.tensor([-1.0, 0.0, 1.0])), atol=1e-5)
    assert float(w[3]) == 100.0          # exp(100) 被 clip 到 100
    # beta 缩放
    w2 = awr_weight(adv, beta=0.5, clip=100.0)
    assert torch.allclose(w2[2], torch.exp(torch.tensor(0.5)), atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_awr_weight -q`
Expected: FAIL（`ImportError: awr_weight`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_high_actor.py`，在 `HighActor` 之后）

```python
def awr_weight(adv, beta, clip=100.0):
    """AWR 权重 exp(beta·adv),上界 clip(防爆)。adv 为张量,返回同形状张量。"""
    return torch.exp(beta * adv).clamp(max=clip)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_awr_weight -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-high): awr_weight 纯函数(exp(beta·adv) clip)"
```

---

## Task 3: AWR 训练（train_high_actor）

用冻结 `GoalConditionedVF` 算优势 `Ã^h = V(s_{t+k},g) − V(s_t,g)` 与回归目标 `z=φ(s_t,s_{t+k})`，k 步航点 + goal 混采（复用 Phase 1）。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**

```python
def test_train_high_actor_predicts_forward_waypoint():
    # 一维进度链:用 Phase 1 训个小 gc value,再训高层;高层应预测"前向 k 步航点"而非原地
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import train_high_actor

    seq = np.arange(20).reshape(20, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])
    vf, _ = train_gc_value(data, steps=2000, batch_size=16, rep_dim=8, hidden=64,
                           lr=1e-3, ema=0.01, seed=0)
    ha = train_high_actor(data, vf, way_steps=5, beta=1.0, steps=2000,
                          batch_size=16, hidden=64, lr=1e-3, seed=0)

    # 取靠前的 state s_t=states[2],goal=末态;真前向航点=states[7]
    states = data["states"]
    st = states[2:3]
    g = states[-1:].clone()
    with torch.no_grad():
        z_pred = ha(st, g).mean
        z_fwd = vf.phi(st, states[7:8])     # 前向 5 步航点
        z_stay = vf.phi(st, st)             # 原地
    # 预测更接近"前向航点"而非"原地"
    assert (z_pred - z_fwd).norm() < (z_pred - z_stay).norm()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_train_high_actor_predicts_forward_waypoint -q`
Expected: FAIL（`ImportError: train_high_actor`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_high_actor.py`）

```python
def train_high_actor(data, vf, *, way_steps=25, beta=1.0, lr=3e-4,
                     batch_size=256, steps=50_000, hidden=256, seed=0):
    """AWR 抽高层 π^h。vf:冻结 GoalConditionedVF。复用 Phase 1 的 data(build_gc_data)。

    优势 Ã^h = min V(s_{t+k},g) − min V(s_t,g);回归目标 z=vf.phi(s_t, s_{t+k})。
    k 步航点 way=min(s_idx+k, demo末);goal 混采 sample_gc_goals。返回训练后的 HighActor。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vf = copy.deepcopy(vf).eval()
    for p in vf.parameters():
        p.requires_grad_(False)
    states = data["states"]
    D = states.shape[1]
    rep_dim = vf.rep_dim
    ha = HighActor(D, rep_dim, hidden)
    opt = torch.optim.Adam(ha.parameters(), lr=lr)
    s_idx, traj_id = data["s_idx"], data["traj_id"]
    last_arr = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
    n = len(s_idx)
    bs = min(batch_size, n)
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, tj = s_idx[b], traj_id[b]
        wi = np.minimum(si + way_steps, last_arr[b])
        gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                             rng, n_total=len(states))
        s, sw, g = states[si], states[wi], states[gi]
        with torch.no_grad():
            vs1, vs2 = vf(s, g)
            vw1, vw2 = vf(sw, g)
            adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
            w = awr_weight(adv, beta)
            z_tgt = vf.phi(s, sw)                 # base=s_t, target=s_{t+k}
        dist = ha(s, g)
        logp = dist.log_prob(z_tgt).sum(-1)
        loss = -(w * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ha
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_train_high_actor_predicts_forward_waypoint -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-high): train_high_actor(AWR 抽高层,k步航点+混采)"
```

---

## Task 4: 存取（save_high_actor / load_high_actor）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py`

- [ ] **Step 1: 写失败测试**

```python
def test_high_actor_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
        HighActor, save_high_actor, load_high_actor)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "high_actor.pt")
    save_high_actor(p, ha, gc_value_ckpt="gc_value.pt", way_steps=25, beta=1.0)
    ha2, info = load_high_actor(p)
    s, g = torch.randn(3, 30), torch.randn(3, 30)
    assert torch.allclose(ha(s, g).mean, ha2(s, g).mean, atol=1e-6)
    assert info["way_steps"] == 25 and info["gc_value_ckpt"] == "gc_value.pt"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py::test_high_actor_save_load_roundtrip -q`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_high_actor.py`）

```python
def save_high_actor(path, model, *, gc_value_ckpt, way_steps, beta):
    """存 high_actor.pt:权重 + 维度 + 关联的 gc_value_ckpt / way_steps / beta(供 Phase 3 校验)。"""
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "gc_value_ckpt": gc_value_ckpt,
        "way_steps": way_steps,
        "beta": beta,
    }, path)


def load_high_actor(path, map_location="cpu"):
    """读 high_actor.pt,重建 HighActor(eval),返回 (model, info)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = HighActor(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("gc_value_ckpt", "way_steps", "beta")}
    return model, info
```

- [ ] **Step 4: 跑测试确认通过 + 全模块单测**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -q`
Expected: 4 个测试全 PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py
git commit -m "feat(hiql-high): save/load_high_actor 往返"
```

---

## Task 5: 离线训练 CLI（train_hiql_high_actor.py）

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py`

- [ ] **Step 1: 写脚本**

```python
"""离线训练 HIQL 高层 π^h(分层路 Phase 2)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly \
      --stage_cache outputs_chunk/three_piece_stages.npz \
      --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
      --way_steps 25 --output outputs_chunk/three_piece_high_actor.pt

设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import argparse

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    train_high_actor, save_high_actor,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states
from resfit.rl_finetuning.chunk_residual.train_hiql_gc_value import stage_entries_aligned


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL 高层 π^h(Phase 2)")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage_cache", required=True)
    p.add_argument("--gc_value_ckpt", required=True, help="Phase 1 产的冻结 gc_value.pt")
    p.add_argument("--output", default="high_actor.pt")
    p.add_argument("--num_demos", type=int, default=None)
    p.add_argument("--way_steps", type=int, default=25)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    seqs, _, _ = read_per_demo_states(args.hdf5, args.dataset, "eef_piece",
                                      num_demos=args.num_demos)
    seq_lens = [len(s) for s in seqs]
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, args.num_demos, seq_lens)
    data = build_gc_data(seqs, stage_entries)
    vf, info = load_gc_value(args.gc_value_ckpt)
    assert data["states"].shape[1] == vf.state_dim, "state_dim 与 gc_value 不符(state_mode 须同源 eef_piece)"
    print(f"[hiql_high] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={vf.state_dim} rep_dim={vf.rep_dim} way_steps={args.way_steps}")
    ha = train_high_actor(data, vf, way_steps=args.way_steps, beta=args.beta, lr=args.lr,
                          batch_size=args.batch_size, steps=args.steps, hidden=args.hidden,
                          seed=args.seed)
    save_high_actor(args.output, ha, gc_value_ckpt=args.gc_value_ckpt,
                    way_steps=args.way_steps, beta=args.beta)
    print(f"[hiql_high] saved {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟**

```bash
cd /data2/RL/residual-offpolicy-rl && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
  --num_demos 5 --steps 2000 --output /tmp/high_actor_smoke.pt
```
Expected: 打印 `[hiql_high] demos=5 ... rep_dim=10 way_steps=25` 与 `saved`，无报错。

- [ ] **Step 3: 正式训练**

```bash
cd /data2/RL/residual-offpolicy-rl && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value.pt \
  --way_steps 25 --output outputs_chunk/three_piece_high_actor.pt 2>&1 | tee three_piece_high_actor.log
```

- [ ] **Step 4: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_high_actor.py
git commit -m "feat(hiql-high): train_hiql_high_actor 离线训练 CLI"
```

---

## Phase 2 验证 Gate（进入 Phase 3 前）

- [x] 4 个单测全绿：2026-06-09 复跑 `pytest test_hiql_high_actor.py -q` → **4 passed**。
- [x] 正式训练产出 `three_piece_high_actor.pt`（06-08 19:26），`load_high_actor` 载入正常：info 挂在 `gc_value_ckpt=outputs_chunk/three_piece_gc_value.pt`、`way_steps=25`、`beta=1.0`，forward 输出 (B,10) 高斯（log_std≈0.1 量级合理）。
- [x] 子目标合理性手验（spec §4.7 Phase 2 gate）：2026-06-09 用 `verify_high_actor.py` 在 40 条真 demo（1200 采样点、同源 rel_piece 重标准化，std 漂移 ~12% 已修）上验证，**Gate 第3条 PASS**：
  - ⓪ 子目标 z 范数 mean=3.10 ≈ √rep=3.16 → z 落在 φ 流形上；
  - ① 前向步数 `j*−t`（room 态）median=**25.0** mean=25.2 IQR=[24,26]，几乎精确命中 way_steps=25；
  - ② 前向占比 **100%**、原地 0%、倒退 0%；③ 远离末态却塌末端 **0/948=0%**；④ off/expected median=**1.00**。
  - 复验脚本：`resfit/rl_finetuning/chunk_residual/verify_high_actor.py`（一键复跑）；可视化 `outputs_chunk/high_actor_verify.png`。
  - ⚠️ 局限：AWR 回归目标恒为"固定 k=25 步航点"，故"落在 +25"**主要证明回归收敛 + 权重非退化可用**（Gate 第3条所求）；**未单独证明优势加权是否偏向高价值航点**（回归项盖过该信号，离线难测，留 Phase 3 在线 rollout 成功率验）。
  - ✅ **已全面切 geometric（2026-06-09）**：Phase 1 改用 geometric 版 V（`three_piece_gc_value_geom.pt`，对比证实目标侧早期尖坑 7→0、V(s,g=s) 负尾 −19.2→−0.96、状态侧不退化），high_actor 已连带重训为 **`three_piece_high_actor_geom.pt`**（挂 geom V，info 已确认），并用 `verify_high_actor.py` 复验 **Gate 第3条 PASS**：z 范数 3.11、前向步数 median=25.0 IQR[24,26]、前向 100%、塌末态 0/948、off/expected median=mean=1.00。可视化 `outputs_chunk/high_actor_verify_geom.png`。
  - 旧版（stage_entry）`three_piece_gc_value.pt` / `three_piece_high_actor.pt` / `high_actor_verify.png` 保留作对照,未删。**后续 Phase 3 一律用 `_geom` 两件套。**

**Gate 已于 2026-06-09 带证据通过（已切 geometric）。** 通过后进入 Phase 3（低层子目标条件化 + offline buffer 接线 + 执行接线，另出一份 plan，依赖本阶段 `three_piece_high_actor_geom.pt` 与 Phase 1 `three_piece_gc_value_geom.pt`）。

---

## 自查记录（writing-plans self-review）

- **Spec 覆盖**：覆盖 spec §4.1 高层、§4.3 AWR 训练（式 6、β、clip 100、回归目标 φ(s_t,s_{t+k})）、§4.6 的 `hiql_high_actor.py`/`train_hiql_high_actor.py`、§5 高层单测、§4.7 Phase 2 gate。
- **占位符**：无；每步完整代码/命令。
- **类型一致**：`vf.phi(s, g)`（base 在前）与 Phase 1 一致；`build_gc_data` 返回键、`sample_gc_goals(..., n_total=)` 签名复用 Phase 1 一致；`HighActor(state_dim, rep_dim, hidden)` 构造在 Task 1/3/4 一致；`awr_weight(adv, beta, clip)` 在 Task 2/3 一致；`save_high_actor(..., gc_value_ckpt=, way_steps=, beta=)` 与 `load_high_actor` 字段一致；`stage_entries_aligned` 复用 Phase 1 Task 7 定义。
