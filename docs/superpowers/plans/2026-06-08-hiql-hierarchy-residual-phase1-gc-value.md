# HIQL 分层 Phase 1：goal-conditioned value 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 造一个 goal-conditioned HIQL 价值 `V(s, φ([g,s]))`(含 10 维归一化瓶颈表征 φ + 双 critic 集成 + EMA target，action-free expectile TD，建在 eef_piece 30 维 state 上），离线训练并冻结存盘，供后续高层 AWR(Phase 2)与低层子目标条件(Phase 3)复用。

**Architecture:** 新增**纯逻辑模块** `hiql_gc_value.py`（与单任务 V-as-Φ 的 `hiql_value.py` 正交、不互相 import 破坏）+ **离线训练脚本** `train_hiql_gc_value.py`（复用现成的 `read_per_demo_states` 取 30 维 state、`load_stage_cache` 取逐帧 stage 算入口锚）。goal 混采用 HIQL 口径（当前 0.2/未来 0.5/随机 0.3），未来目标锚在 stage 入口态。每个纯函数/模块先写失败测试再实现。

**Tech Stack:** PyTorch（nn.Module、Adam、EMA target）、NumPy（goal 采样）、pytest、h5py、LeRobot dataset metadata。conda env `residual`，所有命令从仓库根 `/data2/RL/residual-offpolicy-rl`（本机 `/mnt/mnt/data/resfit`）跑。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`（新） | 纯逻辑：`RelativeGoalEncoder`、`GoalConditionedVF`、`stage_entries_from_instant`、`build_gc_data`、`sample_gc_goals`、`train_gc_value`、`save_gc_value`/`load_gc_value`。**只**从 `hiql_value` 复用 `expectile_loss`。 |
| `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`（新） | 离线训练 CLI：复用 `read_per_demo_states`（eef_piece 30 维）+ `load_stage_cache`，对齐 demo 顺序算 stage 入口，串起 `build_gc_data`→`train_gc_value`→`save_gc_value`。 |
| `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`（新） | Task 1–6 的单测。 |

**不动**：`hiql_value.py`、`hiql_potential.py`、`train_hiql_value.py`、`q_agent.py`、`train_chunk_residual.py`（Phase 1 零侵入现有训练）。

---

## Task 1: goal 表征 φ（RelativeGoalEncoder）

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hiql_gc_value.py
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import RelativeGoalEncoder


def test_relative_goal_encoder_shape_and_norm():
    enc = RelativeGoalEncoder(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(8, 30)
    g = torch.randn(8, 30)
    z = enc(g, s)                      # forward(targets=g, bases=s)
    assert z.shape == (8, 10)
    # 归一化到球面半径 sqrt(rep_dim)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.full((8,), float(np.sqrt(10))), atol=1e-4)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_relative_goal_encoder_shape_and_norm -q`
Expected: FAIL（`ModuleNotFoundError` 或 `ImportError: RelativeGoalEncoder`）

- [ ] **Step 3: 写最小实现**

```python
# hiql_gc_value.py
"""goal-conditioned HIQL value(分层路 Phase 1)的纯逻辑。

与单任务 V-as-Φ 的 hiql_value.py 正交:本模块学 V(s, φ([g,s])),带 10 维归一化瓶颈
表征 φ + 双 critic 集成 + EMA target,供高层 AWR(Phase 2)与低层子目标条件(Phase 3)复用。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


def _mlp(in_dim, hidden, out_dim, n_hidden=2):
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class RelativeGoalEncoder(nn.Module):
    """φ([g,s]):concat(targets=g, bases=s) -> MLP -> rep_dim,再归一化到半径 sqrt(rep_dim)。"""

    def __init__(self, state_dim, rep_dim=10, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.net = _mlp(2 * state_dim, hidden, rep_dim)

    def forward(self, g, s):
        rep = self.net(torch.cat([g, s], dim=-1))
        rep = rep / (rep.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return rep
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_relative_goal_encoder_shape_and_norm -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): RelativeGoalEncoder φ([g,s]) 归一化瓶颈表征"
```

---

## Task 2: goal-conditioned 价值网络（GoalConditionedVF）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
def test_goal_conditioned_vf_forward_and_phi():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
    vf = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    s = torch.randn(5, 30)
    g = torch.randn(5, 30)
    v1, v2 = vf(s, g)
    assert v1.shape == (5,) and v2.shape == (5,)
    # phi(base=s, target=g) 与 goal_encoder(targets=g, bases=s) 一致
    z = vf.phi(s, g)
    assert z.shape == (5, 10)
    assert torch.allclose(z, vf.goal_encoder(g, s), atol=1e-6)
    # 参数可训(梯度非空)
    (v1.sum() + v2.sum()).backward()
    assert any(p.grad is not None for p in vf.parameters())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_goal_conditioned_vf_forward_and_phi -q`
Expected: FAIL（`ImportError: GoalConditionedVF`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_gc_value.py`，在 `RelativeGoalEncoder` 之后）

```python
class GoalConditionedVF(nn.Module):
    """V(s, φ([g,s])),双 critic 集成。

    约定 phi(s, g) = goal_encoder(targets=g, bases=s):第一参恒为"基准状态",第二参为
    "目标/子目标状态"。forward(s, g) -> (v1, v2)。
    """

    def __init__(self, state_dim, rep_dim=10, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.rep_dim = rep_dim
        self.hidden = hidden
        self.goal_encoder = RelativeGoalEncoder(state_dim, rep_dim, hidden)
        self.v1 = _mlp(state_dim + rep_dim, hidden, 1)
        self.v2 = _mlp(state_dim + rep_dim, hidden, 1)

    def phi(self, s, g):
        return self.goal_encoder(g, s)

    def forward(self, s, g):
        x = torch.cat([s, self.phi(s, g)], dim=-1)
        return self.v1(x).squeeze(-1), self.v2(x).squeeze(-1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_goal_conditioned_vf_forward_and_phi -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): GoalConditionedVF V(s,φ([g,s])) 双 critic"
```

---

## Task 3: stage 入口下标（stage_entries_from_instant）

stage 入口 = 逐帧瞬时 stage 的"运行最大值"首次达到每个更高 stage 的下标。这是混合子目标里 future-goal 的语义锚。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
def test_stage_entries_from_instant():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import stage_entries_from_instant
    # 瞬时 stage 可抖动(2->1),用运行最大值 latch;入口=latch 递增处
    instant = np.array([0, 0, 1, 1, 2, 1, 2, 3], dtype=np.int8)
    # latch = [0,0,1,1,2,2,2,3];递增发生在 idx 2(->1)、4(->2)、7(->3)
    entries = stage_entries_from_instant(instant)
    assert entries.tolist() == [2, 4, 7]

    # 全 0(没推进):无入口 -> 空数组
    assert stage_entries_from_instant(np.zeros(5, dtype=np.int8)).tolist() == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_stage_entries_from_instant -q`
Expected: FAIL（`ImportError: stage_entries_from_instant`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_gc_value.py`）

```python
def stage_entries_from_instant(instant_stages):
    """逐帧瞬时 stage(int 数组) -> 各更高 stage 首次到达的下标(升序 int64 数组)。

    用运行最大值 latch 消抖;入口=latch 比前一帧大的位置。全 0 返回空数组。
    """
    instant = np.asarray(instant_stages).astype(np.int64)
    if len(instant) == 0:
        return np.empty(0, dtype=np.int64)
    latch = np.maximum.accumulate(instant)
    inc = np.flatnonzero(np.diff(latch, prepend=latch[0] - (latch[0] > 0)) > 0)
    # prepend 取 latch[0]-1(当 latch[0]>0)以便首帧本身若>0 也算入口;否则首帧=0 不算
    return inc.astype(np.int64)
```

> 注：测试里首帧=0，故首帧不计入口；若某 demo 首帧已 latch>0（极少），prepend 逻辑保证它被算作入口。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_stage_entries_from_instant -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): stage_entries_from_instant(latch 消抖求 stage 入口)"
```

---

## Task 4: 扁平数据 + goal 混采（build_gc_data / sample_gc_goals）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
def test_build_gc_data_and_sample_goals():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals
    # 两条 demo:T=4 和 T=3,D=2
    seqs = [np.arange(8).reshape(4, 2).astype(np.float32),
            (np.arange(6).reshape(3, 2) + 100).astype(np.float32)]
    stage_entries = [np.array([2]), np.array([1])]   # demo0 入口在 within-idx 2;demo1 在 1
    data = build_gc_data(seqs, stage_entries)
    # transitions: demo0 有 3 个(0-1,1-2,2-3), demo1 有 2 个 -> 共 5
    assert len(data["s_idx"]) == 5
    assert data["states"].shape == (7, 2)            # 4+3
    # done 仅各 demo 末 transition=1
    assert data["done"].tolist() == [0, 0, 1, 0, 1]
    # demo0 全局下标 0..3,末态=3;入口全局下标=0+2=2
    assert data["last_idx_of"][0] == 3
    assert data["stage_entries_of"][0].tolist() == [2]
    assert data["stage_entries_of"][1].tolist() == [4 + 1]   # demo1 偏移=4

    # goal 采样:全 current -> goal==当前下标
    rng = np.random.default_rng(0)
    s_i = data["s_idx"]
    g = sample_gc_goals(s_i, data["traj_id"], data["last_idx_of"],
                        data["stage_entries_of"], rng, n_total=7,
                        p_curr=1.0, p_traj=0.0, p_rand=0.0)
    assert g.tolist() == list(s_i)

    # 全 future -> goal 落在"同 demo、>=当前、stage 入口或末态"
    g2 = sample_gc_goals(s_i, data["traj_id"], data["last_idx_of"],
                         data["stage_entries_of"], rng, n_total=7,
                         p_curr=0.0, p_traj=1.0, p_rand=0.0)
    for i, gi in zip(s_i, g2):
        d = data["traj_id"][list(s_i).index(i)]
        allowed = set(data["stage_entries_of"][d][data["stage_entries_of"][d] >= i].tolist())
        allowed.add(data["last_idx_of"][d])
        assert gi in allowed
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_build_gc_data_and_sample_goals -q`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_gc_value.py`）

```python
def build_gc_data(seqs, stage_entries):
    """list[(T,D)] 标准化 state + list[within-demo stage 入口下标] -> 扁平训练数据 dict。

    返回:states(N,D tensor)、s_idx/sn_idx(每 transition 的 s/s' 全局下标)、done(tensor)、
    traj_id、last_idx_of(demo->末态全局下标)、stage_entries_of(demo->入口全局下标 array)。
    T<2 的 demo 跳过。
    """
    states, s_idx, sn_idx, done, traj_id = [], [], [], [], []
    last_idx_of, stage_entries_of = {}, {}
    offset = 0
    for d, seq in enumerate(seqs):
        seq = np.asarray(seq, dtype=np.float32)
        T = len(seq)
        if T < 2:
            continue
        states.append(seq)
        g_idx = np.arange(offset, offset + T)
        last_idx_of[d] = int(g_idx[-1])
        ent = np.asarray(stage_entries[d], dtype=np.int64)
        ent = ent[ent < T]                                   # 防越界
        stage_entries_of[d] = g_idx[ent] if len(ent) else g_idx[[-1]]
        for t in range(T - 1):
            s_idx.append(offset + t)
            sn_idx.append(offset + t + 1)
            done.append(1.0 if t == T - 2 else 0.0)
            traj_id.append(d)
        offset += T
    return {
        "states": torch.tensor(np.concatenate(states, axis=0), dtype=torch.float32),
        "s_idx": np.asarray(s_idx, dtype=np.int64),
        "sn_idx": np.asarray(sn_idx, dtype=np.int64),
        "done": torch.tensor(done, dtype=torch.float32),
        "traj_id": np.asarray(traj_id, dtype=np.int64),
        "last_idx_of": last_idx_of,
        "stage_entries_of": stage_entries_of,
    }


def sample_gc_goals(idx, traj_id, last_idx_of, stage_entries_of, rng,
                    *, n_total, p_curr=0.2, p_traj=0.5, p_rand=0.3):
    """HIQL 混采 + stage 入口锚:current(p_curr)/future(p_traj)/random(p_rand)。

    future = 同 demo 中 >= 当前下标的 stage 入口态里均匀取一个,无则取末态。
    idx/traj_id 为 (B,) np 数组。返回 (B,) goal 全局下标。
    """
    B = len(idx)
    goal = rng.integers(0, n_total, size=B)                   # random
    fut = np.empty(B, dtype=np.int64)
    for j in range(B):
        ent = stage_entries_of[int(traj_id[j])]
        cand = ent[ent >= idx[j]]
        fut[j] = int(rng.choice(cand)) if len(cand) else last_idx_of[int(traj_id[j])]
    denom = max(1.0 - p_curr, 1e-8)
    goal = np.where(rng.random(B) < p_traj / denom, fut, goal)   # traj vs random
    goal = np.where(rng.random(B) < p_curr, idx, goal)            # 覆盖 current
    return goal.astype(np.int64)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_build_gc_data_and_sample_goals -q`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): build_gc_data + sample_gc_goals(混采+stage入口锚)"
```

---

## Task 5: action-free expectile 训练（train_gc_value）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
def test_train_gc_value_learns_progress():
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, train_gc_value
    # 一维进度链:state = 标量位置 0..9(包成 (T,1));goal=末态时 V 应随接近末态递增
    seq = np.arange(10).reshape(10, 1).astype(np.float32)
    data = build_gc_data([seq], [np.array([], dtype=np.int64)])  # 无 stage 入口 -> future=末态
    model, v_stats = train_gc_value(
        data, gamma=0.99, expectile=0.7, ema=0.01, lr=1e-3,
        batch_size=9, steps=2000, rep_dim=8, hidden=64, seed=0)
    assert v_stats["max"] > v_stats["min"]                       # 有区分度
    # 对 goal=末态(全局下标 9),V(s) 应大致随 s 接近末态递增
    states = data["states"]
    g = states[9].repeat(10, 1)
    with torch.no_grad():
        v1, v2 = model(states, g)
        v = torch.minimum(v1, v2)
    # 末端 3 步均值 > 起始 3 步均值
    assert v[-3:].mean() > v[:3].mean()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_train_gc_value_learns_progress -q`
Expected: FAIL（`ImportError: train_gc_value`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_gc_value.py`）

```python
def train_gc_value(data, *, gamma=0.99, expectile=0.7, ema=0.005, lr=3e-4,
                   batch_size=256, steps=50_000, rep_dim=10, hidden=256, seed=0):
    """在扁平 GC 数据上训 action-free expectile goal-conditioned value。

    reward r(s,g)=0 if s==g else -1;到达 goal 或 demo 末步都截断 bootstrap。
    双 critic 集成、target 取 min、EMA target。返回 (model, v_stats)。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = data["states"]
    D = states.shape[1]
    model = GoalConditionedVF(D, rep_dim, hidden)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(data["s_idx"])
    bs = min(batch_size, n)
    s_idx, sn_idx, traj_id = data["s_idx"], data["sn_idx"], data["traj_id"]
    done_all = data["done"]
    for _ in range(steps):
        b = rng.integers(0, n, size=bs)
        si, sni, tj = s_idx[b], sn_idx[b], traj_id[b]
        gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"],
                             rng, n_total=len(states))
        s = states[si]
        s_next = states[sni]
        g = states[gi]
        success = torch.tensor(si == gi, dtype=torch.float32)
        reward = success - 1.0                      # 0 / -1
        mask = (1.0 - success) * (1.0 - done_all[b])  # 到达 goal 或 demo 末步都不 bootstrap
        with torch.no_grad():
            nv1, nv2 = target(s_next, g)
            nv = torch.minimum(nv1, nv2)
            y = reward + gamma * mask * nv
        v1, v2 = model(s, g)
        loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, mp in zip(target.parameters(), model.parameters()):
                tp.mul_(1.0 - ema).add_(ema * mp)
    with torch.no_grad():
        gl = np.array([data["last_idx_of"][int(d)] for d in traj_id], dtype=np.int64)
        vv1, vv2 = model(states[s_idx], states[gl])    # V(s, g=该 demo 末态)
        vv = torch.minimum(vv1, vv2)
        v_stats = {"min": float(vv.min()), "max": float(vv.max()), "mean": float(vv.mean())}
    return model, v_stats
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_train_gc_value_learns_progress -q`
Expected: PASS（若偶发不稳，提高 steps 到 3000 或 seed 固定已设；该测试设计为稳定）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): train_gc_value(action-free expectile TD + 双critic min + EMA)"
```

---

## Task 6: 存取（save_gc_value / load_gc_value）

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py`

- [ ] **Step 1: 写失败测试**

```python
def test_gc_value_save_load_roundtrip(tmp_path):
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
        GoalConditionedVF, save_gc_value, load_gc_value)
    model = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=64)
    p = str(tmp_path / "gc_value.pt")
    save_gc_value(p, model, v_stats={"min": -3.0, "max": 0.0, "mean": -1.0},
                  mean=torch.zeros(30), std=torch.ones(30),
                  dataset_id="ds", state_mode="eef_piece",
                  rel_piece_stats=(np.zeros(12), np.ones(12)))
    m2, info = load_gc_value(p)
    s, g = torch.randn(4, 30), torch.randn(4, 30)
    v1a, _ = model(s, g)
    v1b, _ = m2(s, g)
    assert torch.allclose(v1a, v1b, atol=1e-6)
    assert info["state_mode"] == "eef_piece"
    assert info["v_stats"]["min"] == -3.0
    assert info["rel_piece_mean"] is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_save_load_roundtrip -q`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 写最小实现**（追加到 `hiql_gc_value.py`）

```python
def save_gc_value(path, model, *, v_stats, mean, std, dataset_id,
                  state_mode="eef_piece", rel_piece_stats=None):
    """存 gc_value.pt:权重 + 维度(state_dim/rep_dim/hidden) + v_stats + state mean/std
    + dataset_id + state_mode(+ eef_piece 的 rel_piece mean/std,供 Phase 3 online 同源标准化)。"""
    payload = {
        "state_dict": model.state_dict(),
        "state_dim": model.state_dim,
        "rep_dim": model.rep_dim,
        "hidden": model.hidden,
        "v_stats": v_stats,
        "mean": mean,
        "std": std,
        "dataset_id": dataset_id,
        "state_mode": state_mode,
    }
    if rel_piece_stats is not None:
        payload["rel_piece_mean"], payload["rel_piece_std"] = rel_piece_stats
    torch.save(payload, path)


def load_gc_value(path, map_location="cpu"):
    """读 gc_value.pt,重建 GoalConditionedVF(eval),返回 (model, info)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = GoalConditionedVF(ckpt["state_dim"], ckpt["rep_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    info = {k: ckpt[k] for k in ("v_stats", "mean", "std", "dataset_id")}
    info["state_mode"] = ckpt.get("state_mode", "eef_piece")
    info["rel_piece_mean"] = ckpt.get("rel_piece_mean")
    info["rel_piece_std"] = ckpt.get("rel_piece_std")
    return model, info
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py::test_gc_value_save_load_roundtrip -q`
Expected: PASS

- [ ] **Step 5: 跑全模块单测 + 提交**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q`
Expected: 6 个测试全 PASS

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py
git commit -m "feat(hiql-gc): save_gc_value/load_gc_value 往返"
```

---

## Task 7: 离线训练 CLI（train_hiql_gc_value.py）

把纯逻辑串成可跑脚本：复用 `read_per_demo_states`（eef_piece 30 维，含 rel_piece 标准化）取 state，`load_stage_cache` 取逐帧 stage、对齐 demo 顺序算入口锚。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`

- [ ] **Step 1: 写脚本**

```python
"""离线训练 goal-conditioned HIQL value(分层路 Phase 1)。从仓库根跑:

    conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
      --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
      --dataset ankile/dexmg-two-arm-three-piece-assembly \
      --stage_cache outputs_chunk/three_piece_stages.npz \
      --output outputs_chunk/three_piece_gc_value.pt

设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import argparse

import h5py
import numpy as np

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    build_gc_data, train_gc_value, save_gc_value, stage_entries_from_instant,
)
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
    load_stage_cache, sorted_demo_keys,
)
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states


def stage_entries_aligned(hdf5_path, stage_cache, num_demos, seq_lens):
    """按 read_per_demo_states 的同款 demo 顺序读逐帧 stage,算 within-demo 入口下标,
    并裁剪到对应 seq 长度(eef_piece 路会把 seq 截到 min(len(s18),len(rel)))。"""
    stages_by_demo = load_stage_cache(stage_cache)
    with h5py.File(hdf5_path, "r") as f:
        eps = sorted_demo_keys(list(f["data"].keys()))
    if num_demos is not None:
        eps = eps[:num_demos]
    out = []
    for ep, T in zip(eps, seq_lens):
        inst = np.asarray(stages_by_demo[ep])[:T]
        ent = stage_entries_from_instant(inst)
        out.append(ent[ent < T])
    return out


def build_parser():
    p = argparse.ArgumentParser(description="离线训练 goal-conditioned HIQL value(Phase 1)")
    p.add_argument("--hdf5", required=True)
    p.add_argument("--dataset", required=True, help="LeRobot dataset id(取 state norm stats)")
    p.add_argument("--stage_cache", required=True, help="逐帧 stage 缓存 npz(precompute_stage_cache 产)")
    p.add_argument("--output", default="gc_value.pt")
    p.add_argument("--num_demos", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--ema", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--rep_dim", type=int, default=10)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    # state_mode 固定 eef_piece(分层路必须含物体 pose;③a' 假设就绪)
    seqs, standardizer, rel_stats = read_per_demo_states(
        args.hdf5, args.dataset, "eef_piece", num_demos=args.num_demos)
    seq_lens = [len(s) for s in seqs]
    stage_entries = stage_entries_aligned(args.hdf5, args.stage_cache, args.num_demos, seq_lens)
    data = build_gc_data(seqs, stage_entries)
    print(f"[hiql_gc] demos={len(seqs)} transitions={len(data['s_idx'])} "
          f"state_dim={data['states'].shape[1]} rep_dim={args.rep_dim}")
    model, v_stats = train_gc_value(
        data, gamma=args.gamma, expectile=args.expectile, ema=args.ema, lr=args.lr,
        batch_size=args.batch_size, steps=args.steps, rep_dim=args.rep_dim,
        hidden=args.value_hidden, seed=args.seed)
    save_gc_value(args.output, model, v_stats=v_stats,
                  mean=standardizer._mean.cpu(), std=standardizer._std.cpu(),
                  dataset_id=args.dataset, state_mode="eef_piece", rel_piece_stats=rel_stats)
    print(f"[hiql_gc] saved {args.output}; v_stats={v_stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟（小 demo 数，确认链路通）**

Run（先确保 `outputs_chunk/three_piece_stages.npz` 存在；不存在则先 `precompute_stage_cache`，见 `offline_stage_replay`）：
```bash
cd /data2/RL/residual-offpolicy-rl && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --num_demos 5 --steps 2000 --output /tmp/gc_value_smoke.pt
```
Expected: 打印 `[hiql_gc] demos=5 transitions=... state_dim=30 rep_dim=10` 和末尾 `v_stats={...}`（`max>min`），无报错，产出 `/tmp/gc_value_smoke.pt`。

> 注：`read_per_demo_states` 的 eef_piece 路要 replay 全 demo 算 rel_piece，需 EGL；`--num_demos 5` 控制冒烟时长。

- [ ] **Step 3: 正式训练**

```bash
cd /data2/RL/residual-offpolicy-rl && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run -n residual python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --stage_cache outputs_chunk/three_piece_stages.npz \
  --output outputs_chunk/three_piece_gc_value.pt 2>&1 | tee three_piece_gc_value.log
```
Expected: `v_stats` 的 `max>min`（有区分度）。

- [ ] **Step 4: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py
git commit -m "feat(hiql-gc): train_hiql_gc_value 离线训练 CLI(eef_piece + stage 入口锚)"
```

---

## Phase 1 验证 Gate（进入 Phase 2 前）

- [ ] 6 个单测全绿：`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -q`
- [ ] 正式训练产出 `three_piece_gc_value.pt`，`v_stats.max > v_stats.min`（V 对不同 g 有区分度）。
- [ ] 手验单调性：载入 `gc_value.pt`，沿一条成功 demo 取 state、g=该 demo 末态，`V(s,g)` 大致随接近末态递增（与 Task 5 测试同口径，真数据上肉眼/脚本确认）。

通过后进入 Phase 2（高层 AWR `π^h`，将另出一份 plan，依赖本阶段产出的 `gc_value.pt` 与其 `phi`）。

---

## 自查记录（writing-plans self-review）

- **Spec 覆盖**：本 plan 覆盖 spec §4.1 的 GC value + φ、§4.2 的 goal 混采与 stage 入口锚、§4.6 的 `hiql_gc_value.py`/`train_hiql_gc_value.py` 两文件、§5 的 V 单测、§4.7 Phase 1 与其 gate。Phase 2/3（高层、低层接线）属后续 plan，spec 已标注分阶段。
- **占位符**：无 TBD/TODO；每步含完整代码或精确命令。
- **类型一致**：`phi(s, g)`/`forward(s, g)` 参数序（base 在前、target 在后）在 Task 2/5/Gate 一致；`sample_gc_goals` 的 keyword-only `n_total`/`p_*` 在 Task 4/5 调用一致；`build_gc_data` 返回键（`states/s_idx/sn_idx/done/traj_id/last_idx_of/stage_entries_of`）在 Task 4/5/7 一致；`save_gc_value`/`load_gc_value` 字段（含 `rep_dim`）在 Task 6 自洽。
