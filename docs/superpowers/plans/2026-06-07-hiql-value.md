# 离线 HIQL action-free value(③a)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写一个独立离线脚本,从纯成功 demo 学一个冻结的 action-free IQL expectile value V(s)(lowdim 18 维 state + 小 MLP),产出 value.pt,供后续 ③b 当 PBS 势函数 Φ。

**Architecture:** 纯逻辑(loss / target / 网络 / 选段 / 训练循环 / 存取)集中在 `hiql_value.py`(全可单测);hdf5 读取 + 标准化 + CLI 串接放 `train_hiql_value.py`(薄,靠 manual 实跑验证)。学法:goal-reaching 内部 reward(末步=1 否则 0)+ expectile TD(`y=r+γ(1-done)V_target(s')`,EMA target)。state 标准化复用现有 `StateStandardizer`(从 dataset metadata 拉 mean/std,与 RL 训练同源)。

**Tech Stack:** PyTorch、h5py、numpy、现有 `offline_hdf5_buffer.{assemble_state18,STATE18_KEYS,sorted_demo_keys}`、`utils/normalization.StateStandardizer`、conda env `residual`、pytest。

**Spec:** `docs/superpowers/specs/2026-06-07-hiql-value-design.md`

**通用测试命令:** `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest <路径> -v`(所有命令从仓库根跑)

---

## File Structure

- `resfit/rl_finetuning/chunk_residual/hiql_value.py`(新)—— 纯逻辑:`expectile_loss`、`discounted_target`、`ValueMLP`、`build_transitions`、`save_value`/`load_value`、`train_value`。
- `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(新)—— `read_per_demo_states`(复用 assemble_state18 + StateStandardizer)、`build_parser`、`main`。
- `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`(新)—— 全部单测。

每个 Task 的 import 前缀:`from resfit.rl_finetuning.chunk_residual.hiql_value import ...`。

---

### Task 1: expectile_loss(expectile regression 损失)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_hiql_value.py`:

```python
import torch
from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


def test_expectile_half_equals_half_mse():
    diff = torch.tensor([2.0, -3.0, 1.0])
    loss = expectile_loss(diff, 0.5)
    assert torch.allclose(loss, 0.5 * diff.pow(2).mean())


def test_expectile_asymmetric_weights():
    # diff>0 (低估, V<y) 用权重 tau; diff<0 用权重 1-tau
    pos = expectile_loss(torch.tensor([2.0]), 0.7)
    neg = expectile_loss(torch.tensor([-2.0]), 0.7)
    assert torch.allclose(pos, torch.tensor(0.7 * 4.0))
    assert torch.allclose(neg, torch.tensor(0.3 * 4.0))
    assert pos > neg


def test_expectile_returns_scalar():
    loss = expectile_loss(torch.randn(8), 0.7)
    assert loss.shape == torch.Size([])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'expectile_loss'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py`(文件开头):

```python
"""离线 HIQL action-free value(模块 ③a)的纯逻辑。

设计见 docs/superpowers/specs/2026-06-07-hiql-value-design.md。
学法:goal-reaching 内部 reward(末步=1 否则 0)+ expectile TD,EMA target。
"""
import copy

import numpy as np
import torch
import torch.nn as nn


def expectile_loss(diff, expectile):
    """expectile regression 损失 L_tau(u) = |tau - 1[u<0]| * u^2,u=diff=y-V。

    tau>0.5 时对低估(diff>0,V<y)惩罚更重 -> 学上侧 expectile(乐观 value)。
    tau=0.5 退化为 0.5*MSE。返回标量。
    """
    weight = torch.where(diff < 0, 1.0 - expectile, expectile)
    return (weight * diff.pow(2)).mean()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): expectile_loss 纯函数 + 单测(③a)"
```

---

### Task 2: discounted_target(action-free TD target)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_hiql_value.py` 追加(import 行改为同时导入 `discounted_target`):

```python
from resfit.rl_finetuning.chunk_residual.hiql_value import discounted_target


def test_target_done_no_bootstrap():
    y = discounted_target(reward=torch.tensor([1.0]), next_v=torch.tensor([5.0]),
                          done=torch.tensor([1.0]), gamma=0.99)
    assert torch.allclose(y, torch.tensor([1.0]))


def test_target_not_done_bootstraps():
    y = discounted_target(reward=torch.tensor([0.0]), next_v=torch.tensor([5.0]),
                          done=torch.tensor([0.0]), gamma=0.99)
    assert torch.allclose(y, torch.tensor([0.99 * 5.0]))


def test_target_batch():
    y = discounted_target(torch.tensor([0.0, 1.0]), torch.tensor([2.0, 9.0]),
                          torch.tensor([0.0, 1.0]), 0.9)
    assert torch.allclose(y, torch.tensor([1.8, 1.0]))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'discounted_target'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py`(`expectile_loss` 之后)追加:

```python
def discounted_target(reward, next_v, done, gamma):
    """action-free TD target y = r + gamma*(1-done)*V(s')。

    done=1(终止)时 y=r,不 bootstrap(与 critic 的 Q-target 一致)。
    入参均为 [B] 或 [B,1] 张量;done 为 float(0/1)。
    """
    return reward + gamma * (1.0 - done) * next_v
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): discounted_target(done 不 bootstrap)+ 单测(③a)"
```

---

### Task 3: ValueMLP(value 网络)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

追加(import 加 `ValueMLP`):

```python
from resfit.rl_finetuning.chunk_residual.hiql_value import ValueMLP


def test_valuemlp_output_shape():
    m = ValueMLP(state_dim=18, hidden=32)
    out = m(torch.randn(4, 18))
    assert out.shape == (4, 1)


def test_valuemlp_records_dims():
    m = ValueMLP(state_dim=18, hidden=64)
    assert m.state_dim == 18 and m.hidden == 64


def test_valuemlp_trainable():
    m = ValueMLP(18, 32)
    m(torch.randn(4, 18)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'ValueMLP'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py` 追加:

```python
class ValueMLP(nn.Module):
    """lowdim state -> 标量 V(s) 的小 MLP。state_dim/hidden 存为属性,便于 save/load 重建。"""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s):
        return self.net(s)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(9 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): ValueMLP(state_dim/hidden 记录属性)+ 单测(③a)"
```

---

### Task 4: build_transitions(state 序列 -> (s,s',done))

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

追加(import 加 `build_transitions`;文件顶部已 import numpy as np 于测试中—若无则加 `import numpy as np`):

```python
import numpy as np
from resfit.rl_finetuning.chunk_residual.hiql_value import build_transitions


def test_build_transitions_counts_and_done():
    seqs = [np.zeros((3, 2)), np.zeros((2, 2))]  # 3帧->2 trans, 2帧->1 trans
    s, sn, done = build_transitions(seqs)
    assert s.shape == (3, 2) and sn.shape == (3, 2) and done.shape == (3, 1)
    # demo1 done=[0,1], demo2 done=[1]
    assert torch.allclose(done.squeeze(1), torch.tensor([0.0, 1.0, 1.0]))


def test_build_transitions_alignment():
    seq = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    s, sn, done = build_transitions([seq])
    assert torch.allclose(s, torch.tensor([[0., 0.], [1., 1.]]))
    assert torch.allclose(sn, torch.tensor([[1., 1.], [2., 2.]]))


def test_build_transitions_skips_short():
    s, sn, done = build_transitions([np.zeros((1, 2)), np.zeros((3, 2))])
    assert s.shape == (2, 2)  # 仅 3 帧的 demo 贡献(1 帧的跳过)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_transitions'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py` 追加:

```python
def build_transitions(state_seqs):
    """list of [T_i, D] 数组(每条 demo 的标准化 state 序列)-> (s, s_next, done) float32 张量。

    每条长 T 的 demo 产出 T-1 个 transition;done 在该 demo 末 transition=1(到达 goal=轨迹终点)。
    T<2 的 demo 跳过。done 形状 [N,1]。
    """
    s_list, sn_list, done_list = [], [], []
    for seq in state_seqs:
        seq = np.asarray(seq, dtype=np.float32)
        T = seq.shape[0]
        if T < 2:
            continue
        s_list.append(seq[:-1])
        sn_list.append(seq[1:])
        d = np.zeros(T - 1, dtype=np.float32)
        d[-1] = 1.0
        done_list.append(d)
    s = torch.from_numpy(np.concatenate(s_list, axis=0))
    sn = torch.from_numpy(np.concatenate(sn_list, axis=0))
    done = torch.from_numpy(np.concatenate(done_list, axis=0)).unsqueeze(1)
    return s, sn, done
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(12 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): build_transitions(state 序列->(s,s',done),末步 done=1)+ 单测(③a)"
```

---

### Task 5: save_value / load_value(value.pt 存取)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

追加(import 加 `save_value, load_value`):

```python
from resfit.rl_finetuning.chunk_residual.hiql_value import save_value, load_value


def test_save_load_roundtrip(tmp_path):
    m = ValueMLP(6, 16)
    vstats = {"min": 0.0, "max": 1.0, "mean": 0.5}
    mean, std = torch.zeros(6), torch.ones(6)
    p = str(tmp_path / "value.pt")
    save_value(p, m, v_stats=vstats, mean=mean, std=std, dataset_id="dummy/ds")
    m2, info = load_value(p)
    x = torch.randn(3, 6)
    assert torch.allclose(m(x), m2(x))          # 权重一致
    assert m2.state_dim == 6 and m2.hidden == 16  # 维度重建正确
    assert info["v_stats"] == vstats
    assert info["dataset_id"] == "dummy/ds"
    assert torch.allclose(info["mean"], mean) and torch.allclose(info["std"], std)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_value'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py` 追加:

```python
def save_value(path, model, *, v_stats, mean, std, dataset_id):
    """存 value.pt:权重 + 维度 + 训练集 V 统计 + (记录用)state mean/std + dataset_id。

    标准化口径与 RL 训练同源,③b 推理时喂的 state 已标准化,故 mean/std 仅作记录/自校验,③b 不依赖。
    """
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "hidden": model.hidden,
        "v_stats": v_stats,
        "mean": mean,
        "std": std,
        "dataset_id": dataset_id,
    }, path)


def load_value(path, map_location="cpu"):
    """读 value.pt,重建 ValueMLP(eval 模式),返回 (model, info_dict)。

    info_dict 含 v_stats / mean / std / dataset_id。
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = ValueMLP(ckpt["state_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("v_stats", "mean", "std", "dataset_id")}
    return model, info
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(13 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): save_value/load_value(权重+维度+V统计+norm记录)+ 单测(③a)"
```

---

### Task 6: train_value(训练循环 + EMA target + V 统计)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

- [ ] **Step 1: 写失败测试**

追加(import 加 `train_value`):

```python
from resfit.rl_finetuning.chunk_residual.hiql_value import train_value


def test_train_value_returns_model_and_stats():
    seq = np.linspace(0, 1, 10).reshape(-1, 1).astype(np.float32)
    s, sn, done = build_transitions([seq] * 16)
    model, v_stats = train_value(s, sn, done, steps=50, batch_size=32, hidden=16, seed=0)
    assert isinstance(model, ValueMLP)
    assert set(v_stats) == {"min", "max", "mean"}
    assert v_stats["min"] <= v_stats["mean"] <= v_stats["max"]


def test_train_value_monotone_along_trajectory():
    # 一维直线轨迹:越靠近终点(goal)value 应越大
    seq = np.linspace(0, 1, 20).reshape(-1, 1).astype(np.float32)
    s, sn, done = build_transitions([seq] * 64)
    model, _ = train_value(s, sn, done, steps=2000, batch_size=128, hidden=64, seed=0)
    with torch.no_grad():
        vs = model(torch.from_numpy(seq)).squeeze(1)
    assert vs[-1] > vs[0]                       # 终点 value > 起点
    diffs = vs[1:] - vs[:-1]
    assert (diffs > 0).float().mean() > 0.7     # 大体单调递增
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'train_value'`

- [ ] **Step 3: 写最小实现**

在 `hiql_value.py` 追加:

```python
def train_value(s, s_next, done, *, gamma=0.99, expectile=0.7, ema=0.005,
                lr=3e-4, batch_size=256, steps=50000, hidden=256, seed=0):
    """在 (s, s_next, done) 上训 action-free IQL expectile value。

    goal-reaching 内部 reward = done(末步=1 否则 0)。EMA target net 稳定 bootstrap。
    返回 (model, v_stats),v_stats = 训练后全数据上 V 的 {min,max,mean}。
    """
    torch.manual_seed(seed)
    n, d = s.shape
    model = ValueMLP(d, hidden)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    reward = done  # r_t = 1 if done else 0 == done
    bs = min(batch_size, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,))
        with torch.no_grad():
            y = discounted_target(reward[idx], target(s_next[idx]), done[idx], gamma)
        v = model(s[idx])
        loss = expectile_loss(y - v, expectile)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), model.parameters()):
                tp.mul_(1.0 - ema).add_(ema * mp)
    with torch.no_grad():
        allv = model(s)
        v_stats = {"min": float(allv.min()), "max": float(allv.max()), "mean": float(allv.mean())}
    return model, v_stats
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(15 passed)。`test_train_value_monotone_along_trajectory` 跑 2000 步 1D,数秒内完成。若偶发 flaky(单调比例略低),seed 已固定为 0;确认逻辑无误后可不动。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): train_value(expectile TD + EMA target,V 沿轨迹单调)+ 单测(③a)"
```

---

### Task 7: train_hiql_value.py(hdf5 读取 + CLI + 串接)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py`

**Context:** `read_per_demo_states` 复用已核实的现有口径 —— `offline_hdf5_buffer.{STATE18_KEYS, assemble_state18, sorted_demo_keys}` 逐 demo 读 6 个 lowdim key 拼 (T,18),`StateStandardizer.from_dataset_stats(LeRobotDatasetMetadata(dataset).stats["observation.state"])` 标准化。hdf5 读取依赖真数据 + LeRobot metadata,**不进单测**;单测只覆盖 `build_parser` 默认值。`read_per_demo_states`/`main` 靠 Step 6 的 manual 实跑验证。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_hiql_value.py`:

```python
from resfit.rl_finetuning.chunk_residual.train_hiql_value import build_parser


def test_parser_defaults():
    args = build_parser().parse_args(["--hdf5", "x.hdf5", "--dataset", "some/ds"])
    assert args.gamma == 0.99
    assert args.expectile == 0.7
    assert args.ema == 0.005
    assert args.steps == 50000
    assert args.value_hidden == 256
    assert args.output == "value.pt"


def test_parser_overrides():
    args = build_parser().parse_args(
        ["--hdf5", "x.hdf5", "--dataset", "some/ds",
         "--expectile", "0.9", "--steps", "1000", "--output", "v2.pt"])
    assert args.expectile == 0.9 and args.steps == 1000 and args.output == "v2.pt"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: FAIL — `ModuleNotFoundError: ...train_hiql_value`

- [ ] **Step 3: 写最小实现**

创建 `train_hiql_value.py`:

```python
"""离线训练 HIQL action-free value(模块 ③a)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly --output outputs_chunk/three_piece_value.pt

设计见 docs/superpowers/specs/2026-06-07-hiql-value-design.md。
"""
import argparse

import h5py
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from resfit.rl_finetuning.chunk_residual.hiql_value import (
    build_transitions, train_value, save_value,
)
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    STATE18_KEYS, assemble_state18, sorted_demo_keys,
)
from resfit.rl_finetuning.utils.normalization import StateStandardizer


def read_per_demo_states(hdf5_path, dataset_id, device="cpu"):
    """读每条 demo 的标准化 18 维 state 序列(与 RL 训练同源 mean/std)。

    返回 (list[np.ndarray (T,18)], standardizer)。复用 assemble_state18 + STATE18_KEYS。
    """
    meta = LeRobotDatasetMetadata(dataset_id)
    standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=device)
    seqs = []
    with h5py.File(hdf5_path, "r") as f:
        for ep in sorted_demo_keys(list(f["data"].keys())):
            grp = f[f"data/{ep}"]
            obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
            state_raw = assemble_state18(obs_arrays)  # (T,18) np
            state_n = standardizer.standardize(
                torch.as_tensor(state_raw, dtype=torch.float32)).cpu().numpy()
            seqs.append(state_n)
    return seqs, standardizer


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 HIQL action-free value(③a)")
    p.add_argument("--hdf5", required=True, help="源 hdf5(含 data/demo_i/obs/<key>)")
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--output", default="value.pt")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--ema", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    seqs, standardizer = read_per_demo_states(args.hdf5, args.dataset)
    s, s_next, done = build_transitions(seqs)
    print(f"[hiql_value] demos={len(seqs)} transitions={s.shape[0]} state_dim={s.shape[1]}")
    model, v_stats = train_value(
        s, s_next, done, gamma=args.gamma, expectile=args.expectile, ema=args.ema,
        lr=args.lr, batch_size=args.batch_size, steps=args.steps,
        hidden=args.value_hidden, seed=args.seed)
    save_value(args.output, model, v_stats=v_stats,
               mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
               dataset_id=args.dataset)
    print(f"[hiql_value] saved {args.output}; v_stats={v_stats}")


if __name__ == "__main__":
    main()
```

注:`standardizer._mean`/`_std` 是 `StateStandardizer` 的实际属性(见 normalization.py:213 的 `self._mean`/`self._std`);若实现时属性名不符,以该文件实际为准并相应取值。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v`
Expected: PASS(17 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py
git commit -m "feat(hiql): train_hiql_value.py(per-demo state 读取复用现有口径 + CLI)+ 单测(③a)"
```

- [ ] **Step 6: Manual 实跑验证(产出 value.pt)**

用真数据小步实跑,确认整条链路通 + V 沿成功 demo 单调:

Run:
```bash
cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --steps 2000 --output outputs_chunk/three_piece_value_smoke.pt
```
Expected: 打印 `demos=... transitions=... state_dim=18`,正常结束并存出 `outputs_chunk/three_piece_value_smoke.pt`,`v_stats` 的 max>min(value 有区分度)。若 `read_per_demo_states` 的 hdf5 key 或 standardizer 属性名报错,按 `offline_hdf5_buffer.py` / `normalization.py` 的实际定义修正后重跑。这一步不提交产物(smoke ckpt 不入库)。

---

## 验收(全部 Task 完成后)

- 全套单测:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_value.py -v` -> 17 passed。
- 回归(确认没碰坏别的):`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`(stage_budget / demo_bc / relabel / wandb_logging 仍绿)。
- Manual 实跑产出过一个真 value.pt(Step 6)。
- ③a 对 baseline 零影响(纯新增文件,不改任何现有训练/env 代码)。
- 后续:③b 另开 spec,把 value.pt 当 PBS Φ 接进 reward shaping 两端 + `--potential_source {stage,hiql}` default-off。
