# HIQL 在线联合微调 (V + high_actor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 3 主训(`train_chunk_residual.py`)里解冻 `V(s,g)` 和 `high_actor`,用 online+offline 混采数据持续联合微调它们,残差 TD3 低层与 `online_rb` 保持零改动。

**Architecture:** 把离线 `train_gc_value` / `train_high_actor` 的单步更新体抽成 buffer-agnostic 的 `value_update_step` / `high_actor_update_step`,离线与在线复用同一份逻辑(同源)。新增一个随 episode 增长的 `OnlineHiqlStore`(`build_gc_data` 结构,几何 goal 采样,无需 stage 检测),与一份从 demo seqs 现建的 offline store 混采,喂给一个 `OnlineHiqlFinetuner`,它持有 V 头/high_actor 的优化器与 EMA target、冻结 φ、暴露 `on_step / on_episode_end / maybe_update` 三个钩子。所有新行为走 flag,默认 OFF 时与现状逐位等价。

**Tech Stack:** Python, PyTorch, NumPy, pytest;resfit chunk_residual 分层 HIQL 路。

## Global Constraints

- **冻 φ**:在线只更新 `vf.v1` / `vf.v2`(critic head)与 `high_actor`;`vf.goal_encoder`(φ)始终 `requires_grad_(False)`。
- **残差 TD3 零回归**:`online_rb`、`add_chunk_transition`、`QAgent.update`、`q_agent.py` 一律不改。
- **默认 OFF 逐位等价**:新 flag `--online_finetune_value` / `--online_finetune_high_actor` 默认 `False`;关闭时整条主训与现状逐位等价。
- **同源**:在线 value/high_actor 更新复用 `value_update_step` / `high_actor_update_step` / `sample_gc_goals`,与离线完全同一套数学;`value_loss_mode` / `value_mask_mode` 从 gc_value ckpt 的 info 读取(不新引入口径)。
- **reward 语义**:V 仍用 goal-conditioned 稀疏 reward `r=success-1`(success = goal 命中当前态),env 任务奖励不进 V。
- **测试纪律**:每个测试文件顶部加模块级 `_cap_torch_threads` fixture(限 1 线程,跑完恢复),与现有 `tests/test_hiql_gc_value.py` 一致。
- **测试运行环境**:resfit residual conda 环境;命令统一 `cd /mnt/mnt/data/resfit && python -m pytest <file> -v`。

---

### Task 1: 抽出 `value_update_step`(hiql_gc_value.py)

把 `train_gc_value` 内层循环的"算 loss + opt.step + EMA"抽成独立函数,离线训练改为调用它。保证逐位等价。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py:185-277`(`train_gc_value` 内层循环;新增 `value_update_step`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_value_update_step.py`(新建)

**Interfaces:**
- Produces: `value_update_step(model, target, opt, s, s_next, g, success, done, *, gamma, expectile, ema, value_loss_mode="shared_min", value_mask_mode="done_aware") -> float`
  - `s, s_next, g`: `[B, D]` 标准化 state(已在 `model`/`target` 所在 device)。
  - `success`: `[B]` float,1 表示 goal 命中当前态(供算 `reward=success-1` 与 mask)。
  - `done`: `[B]` float,demo 末步终止 mask。
  - 行为:计算 IQL expectile loss(按 `value_loss_mode`/`value_mask_mode`)、`opt.step()`、EMA 更新 `target`,返回标量 loss。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_value_update_step.py`)

```python
import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import (
    GoalConditionedVF, value_update_step)
from resfit.rl_finetuning.chunk_residual.hiql_value import expectile_loss


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _manual_shared_min_step(model, target, opt, s, s_next, g, success, done,
                            *, gamma, expectile, ema):
    """逐字复刻 train_gc_value 旧内层(shared_min + done_aware)的一步,作为对照基线。"""
    reward = success - 1.0
    mask = (1.0 - success) * (1.0 - done)
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
    return float(loss)


def test_value_update_step_matches_manual_inline():
    torch.manual_seed(0)
    s = torch.randn(7, 6)
    s_next = torch.randn(7, 6)
    g = torch.randn(7, 6)
    success = torch.tensor([1., 0., 0., 1., 0., 0., 0.])
    done = torch.tensor([0., 0., 1., 0., 0., 1., 0.])

    # 两套模型从同一初始权重出发
    m_a = GoalConditionedVF(6, rep_dim=4, hidden=16)
    m_b = copy.deepcopy(m_a)
    t_a = copy.deepcopy(m_a)
    t_b = copy.deepcopy(m_a)
    for p in t_a.parameters():
        p.requires_grad_(False)
    for p in t_b.parameters():
        p.requires_grad_(False)
    opt_a = torch.optim.Adam(m_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(m_b.parameters(), lr=1e-3)

    la = _manual_shared_min_step(m_a, t_a, opt_a, s, s_next, g, success, done,
                                 gamma=0.99, expectile=0.7, ema=0.005)
    lb = value_update_step(m_b, t_b, opt_b, s, s_next, g, success, done,
                           gamma=0.99, expectile=0.7, ema=0.005)
    assert abs(la - lb) < 1e-9
    for ka, va in m_a.state_dict().items():
        assert torch.equal(va, m_b.state_dict()[ka]), f"model param 不等: {ka}"
    for ka, va in t_a.state_dict().items():
        assert torch.equal(va, t_b.state_dict()[ka]), f"target param 不等: {ka}"


def test_value_update_step_hiql_mode_runs_and_changes_params():
    torch.manual_seed(1)
    s, s_next, g = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    success = torch.zeros(5)
    done = torch.zeros(5)
    m = GoalConditionedVF(4, rep_dim=4, hidden=16)
    before = copy.deepcopy(m.state_dict())
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss = value_update_step(m, t, opt, s, s_next, g, success, done,
                             gamma=0.99, expectile=0.7, ema=0.005,
                             value_loss_mode="hiql", value_mask_mode="hiql")
    assert np.isfinite(loss)
    changed = any(not torch.equal(before[k], m.state_dict()[k]) for k in before)
    assert changed
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_value_update_step.py -v`
Expected: FAIL —— `ImportError: cannot import name 'value_update_step'`

- [ ] **Step 3: 写最小实现**(在 `hiql_gc_value.py` 的 `train_gc_value` 之前新增函数)

```python
def value_update_step(model, target, opt, s, s_next, g, success, done, *,
                      gamma, expectile, ema, value_loss_mode="shared_min",
                      value_mask_mode="done_aware"):
    """GC value 的单步更新(buffer-agnostic;离线/在线共享)。

    s/s_next/g: [B,D] 标准化 state(在 model/target 的 device 上)。
    success: [B] float,1=goal 命中当前态;done: [B] float demo 末步 mask。
    reward=success-1;mask 由 value_mask_mode 决定;loss 由 value_loss_mode 决定。
    opt.step() 后 EMA 更新 target。返回标量 loss。
    """
    if value_loss_mode not in ("shared_min", "hiql"):
        raise ValueError(f"unknown value_loss_mode: {value_loss_mode!r}")
    if value_mask_mode not in ("done_aware", "hiql"):
        raise ValueError(f"unknown value_mask_mode: {value_mask_mode!r}")
    reward = success - 1.0
    if value_mask_mode == "done_aware":
        mask = (1.0 - success) * (1.0 - done)
    else:  # "hiql"
        mask = 1.0 - success
    if value_loss_mode == "shared_min":
        with torch.no_grad():
            nv1, nv2 = target(s_next, g)
            nv = torch.minimum(nv1, nv2)
            y = reward + gamma * mask * nv
        v1, v2 = model(s, g)
        loss = expectile_loss(y - v1, expectile) + expectile_loss(y - v2, expectile)
    else:  # "hiql"
        with torch.no_grad():
            nv1, nv2 = target(s_next, g)
            nv = torch.minimum(nv1, nv2)
            q = reward + gamma * mask * nv
            v1t, v2t = target(s, g)
            v_t = 0.5 * (v1t + v2t)
            adv = q - v_t
            q1 = reward + gamma * mask * nv1
            q2 = reward + gamma * mask * nv2
        v1, v2 = model(s, g)
        loss = (expectile_loss_weighted(adv, q1 - v1, expectile)
                + expectile_loss_weighted(adv, q2 - v2, expectile))
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        for tp, mp in zip(target.parameters(), model.parameters()):
            tp.mul_(1.0 - ema).add_(ema * mp)
    return float(loss)
```

然后把 `train_gc_value` 内层循环(现 `hiql_gc_value.py:240-271`,从 `success = torch.tensor(...)` 到 EMA 那段)替换为:

```python
        success = torch.tensor(si == gi, dtype=torch.float32, device=device)
        value_update_step(model, target, opt, s, s_next, g, success, done_all[b],
                          gamma=gamma, expectile=expectile, ema=ema,
                          value_loss_mode=value_loss_mode, value_mask_mode=value_mask_mode)
```

(注:`s = states[si]`、`s_next = states[sni]`、`g = states[gi]` 仍在循环内 `value_update_step` 调用之前算好,保持不变。)

- [ ] **Step 4: 运行测试确认通过 + 回归现有套件**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_value_update_step.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_gc_value.py -v`
Expected: PASS(新测试全过 + 现有 `test_hiql_gc_value.py` 全部仍过 → 证明抽取逐位等价)

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_gc_value.py resfit/rl_finetuning/chunk_residual/tests/test_value_update_step.py
git commit -m "refactor: extract value_update_step from train_gc_value (bit-equiv)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 抽出 `high_actor_update_step`(hiql_high_actor.py)

把 `train_high_actor` 内层 AWR 更新体抽成独立函数。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py:97-124`(`train_high_actor` 内层;新增 `high_actor_update_step`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_high_actor_update_step.py`(新建)

**Interfaces:**
- Consumes: 无(独立)。
- Produces: `high_actor_update_step(ha, vf, opt, s, sw, g, *, beta, adv_agg="min") -> float`
  - `s, sw, g`: `[B,D]`(`sw`=k 步航点态)。`vf`: 冻结 `GoalConditionedVF`。
  - 行为:`no_grad` 下算 adv(按 `adv_agg`)、AWR 权重、回归目标 `z_tgt=vf.phi(s,sw)`;`ha` 前向算 logp;`-(w*logp).mean()` 反传 + `opt.step()`。返回标量 loss。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_high_actor_update_step.py`)

```python
import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import (
    HighActor, awr_weight, high_actor_update_step)


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _manual_min_step(ha, vf, opt, s, sw, g, *, beta):
    with torch.no_grad():
        vs1, vs2 = vf(s, g)
        vw1, vw2 = vf(sw, g)
        adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
        w = awr_weight(adv, beta)
        z_tgt = vf.phi(s, sw)
    dist = ha(s, g)
    logp = dist.log_prob(z_tgt).sum(-1)
    loss = -(w * logp).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


def test_high_actor_update_step_matches_manual_inline():
    torch.manual_seed(0)
    vf = GoalConditionedVF(6, rep_dim=4, hidden=16)
    for p in vf.parameters():
        p.requires_grad_(False)
    s, sw, g = torch.randn(8, 6), torch.randn(8, 6), torch.randn(8, 6)

    ha_a = HighActor(6, rep_dim=4, hidden=16)
    ha_b = copy.deepcopy(ha_a)
    opt_a = torch.optim.Adam(ha_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(ha_b.parameters(), lr=1e-3)

    la = _manual_min_step(ha_a, vf, opt_a, s, sw, g, beta=1.0)
    lb = high_actor_update_step(ha_b, vf, opt_b, s, sw, g, beta=1.0, adv_agg="min")
    assert abs(la - lb) < 1e-9
    for k, v in ha_a.state_dict().items():
        assert torch.equal(v, ha_b.state_dict()[k]), f"high_actor param 不等: {k}"


def test_high_actor_update_step_mean_agg_runs():
    torch.manual_seed(2)
    vf = GoalConditionedVF(4, rep_dim=4, hidden=16)
    for p in vf.parameters():
        p.requires_grad_(False)
    ha = HighActor(4, rep_dim=4, hidden=16)
    before = copy.deepcopy(ha.state_dict())
    opt = torch.optim.Adam(ha.parameters(), lr=1e-2)
    s, sw, g = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    loss = high_actor_update_step(ha, vf, opt, s, sw, g, beta=1.0, adv_agg="mean")
    assert np.isfinite(loss)
    assert any(not torch.equal(before[k], ha.state_dict()[k]) for k in before)
    with pytest.raises(ValueError):
        high_actor_update_step(ha, vf, opt, s, sw, g, beta=1.0, adv_agg="bogus")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_high_actor_update_step.py -v`
Expected: FAIL —— `ImportError: cannot import name 'high_actor_update_step'`

- [ ] **Step 3: 写最小实现**(在 `hiql_high_actor.py` 的 `train_high_actor` 之前新增)

```python
def high_actor_update_step(ha, vf, opt, s, sw, g, *, beta, adv_agg="min"):
    """high_actor 的 AWR 单步更新(buffer-agnostic;离线/在线共享)。

    s/sw/g: [B,D],sw=k 步航点态。vf:冻结 GoalConditionedVF。
    adv 由 adv_agg 决定('min'|'mean');回归目标 z=vf.phi(s,sw)。返回标量 loss。
    """
    if adv_agg not in ("min", "mean"):
        raise ValueError(f"unknown adv_agg: {adv_agg!r}")
    with torch.no_grad():
        vs1, vs2 = vf(s, g)
        vw1, vw2 = vf(sw, g)
        if adv_agg == "min":
            adv = torch.minimum(vw1, vw2) - torch.minimum(vs1, vs2)
        else:  # "mean"
            adv = 0.5 * (vw1 + vw2) - 0.5 * (vs1 + vs2)
        w = awr_weight(adv, beta)
        z_tgt = vf.phi(s, sw)
    dist = ha(s, g)
    logp = dist.log_prob(z_tgt).sum(-1)
    loss = -(w * logp).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)
```

然后把 `train_high_actor` 内层(现 `hiql_high_actor.py:109-123`,从 `with torch.no_grad():` 到 `opt.step()`)替换为:

```python
        s, sw, g = states[si], states[wi], states[gi]
        high_actor_update_step(ha, vf, opt, s, sw, g, beta=beta, adv_agg=adv_agg)
```

(注:`s, sw, g = states[si], states[wi], states[gi]` 原在 `hiql_high_actor.py:108`,现并入替换块;`wi`/`gi` 计算保持不变。)

- [ ] **Step 4: 运行测试确认通过 + 回归现有套件**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_high_actor_update_step.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_high_actor.py -v`
Expected: PASS(新测试 + 现有 `test_hiql_high_actor.py` 全过 → 抽取逐位等价)

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_high_actor.py resfit/rl_finetuning/chunk_residual/tests/test_high_actor_update_step.py
git commit -m "refactor: extract high_actor_update_step from train_high_actor (bit-equiv)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 抽出 `HiqlSubgoal.encode_state`(hiql_subgoal.py)

把 `subgoal_online` 里"obs → 标准化 vf-输入 state"那段抽成可复用方法,供在线采集用;`subgoal_online` 改为调用它,保证 z 逐位等价。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py:102-131`(`subgoal_online`;新增 `encode_state`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_subgoal_encode_state.py`(新建)

**Interfaces:**
- Produces: `HiqlSubgoal.encode_state(obs, rel_raw=None, prefix_feat=None) -> Tensor [B, vf.state_dim]`(`@torch.no_grad()`);三种 state_mode 的标准化 state(即原 `subgoal_online` 里喂给 `self.ha(s,g)` 的 `s`)。
- `subgoal_online` 行为不变(z 逐位等价)。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_subgoal_encode_state.py`)

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _make_eef_subgoal():
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=32)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=32)
    goal = np.zeros(30, dtype=np.float32)
    return HiqlSubgoal(gc, ha, goal, device="cpu", renorm_subgoal=True,
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))


def test_encode_state_eef_matches_build_state30():
    sg = _make_eef_subgoal()
    state18 = torch.randn(3, 18)
    rel12 = np.random.RandomState(0).randn(3, 12).astype(np.float32)
    s = sg.encode_state(state18, rel12)
    assert s.shape == (3, 30)
    assert torch.allclose(s, sg.build_state30(state18, rel12), atol=1e-6)


def test_subgoal_online_unchanged_by_refactor():
    """encode_state 抽取后,subgoal_online 的 z 与"自己用 encode_state 重算"逐位一致。"""
    sg = _make_eef_subgoal()
    state18 = torch.randn(4, 18)
    rel12 = np.random.RandomState(1).randn(4, 12).astype(np.float32)
    z = sg.subgoal_online(state18, rel12)
    s = sg.encode_state(state18, rel12)
    g = sg.goal.unsqueeze(0).expand(s.shape[0], -1)
    z_ref = sg.ha(s, g).mean
    z_ref = z_ref / (z_ref.norm(dim=-1, keepdim=True) + 1e-8) * (sg.rep_dim ** 0.5)
    assert torch.allclose(z, z_ref, atol=1e-6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_subgoal_encode_state.py -v`
Expected: FAIL —— `AttributeError: 'HiqlSubgoal' object has no attribute 'encode_state'`

- [ ] **Step 3: 写最小实现**(在 `hiql_subgoal.py` 把 `subgoal_online` 重构)

新增 `encode_state` 方法(把现 `subgoal_online` 的 `if self.state_mode == "pi0_feat": ... else: # act_feat` 那段 `s` 的计算原样搬过来,以 `return s` 结尾):

```python
    @torch.no_grad()
    def encode_state(self, obs, rel_raw=None, prefix_feat=None):
        """obs -> [B, vf.state_dim] 标准化 state(喂给 high_actor/vf 的 s)。
        逻辑与原 subgoal_online 完全一致,抽出供在线采集 store 复用。"""
        if self.state_mode == "pi0_feat":
            assert prefix_feat is not None, "pi0_feat 在线需 prefix_feat(从 base policy last_prefix_feat 取)"
            def _to_dev(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().to(dtype=torch.float32, device=self.device)
                return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
            proprio = _to_dev(obs["observation.state"])
            pf = _to_dev(prefix_feat)
            if pf.ndim == 1:
                pf = pf.unsqueeze(0)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            feat = torch.cat([pf, proprio], dim=-1)
            s = (feat - self.feat_mean) / self.feat_std
        elif self.state_mode == "eef_piece":
            state_std = obs["observation.state"] if isinstance(obs, dict) else obs
            s = self.build_state30(state_std, rel_raw)
        else:  # act_feat
            feat = self.extractor.embed_batch(obs)
            s = (feat.to(self.device) - self.feat_mean) / self.feat_std
        return s
```

然后把 `subgoal_online` 改为:

```python
    @torch.no_grad()
    def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
        """eef_piece: obs 传已 std 的 state([B,18]) 或含 observation.state 的 dict(+rel_raw);
        act_feat: obs 传含 images+observation.state 的 dict;
        pi0_feat: obs 含 observation.state(proprio),prefix_feat 传冻结 pi0 prefix 特征(2048 维)。"""
        s = self.encode_state(obs, rel_raw=rel_raw, prefix_feat=prefix_feat)
        g = self.goal.unsqueeze(0).expand(s.shape[0], -1)
        z = self.ha(s, g).mean
        if self.renorm_subgoal:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return z
```

- [ ] **Step 4: 运行测试确认通过 + 回归 subgoal 套件**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_subgoal_encode_state.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_pi0_feat.py -v`
Expected: PASS(新测试 + 现有 subgoal 套件全过)

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_subgoal_encode_state.py
git commit -m "refactor: extract HiqlSubgoal.encode_state from subgoal_online (z bit-equiv)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 在线轨迹 store + 采样器(新建 online_hiql_store.py)

随 episode 增长的 `build_gc_data` 结构(几何 goal,无需 stage 检测)+ value/high_actor 批采样器。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/online_hiql_store.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_store.py`(新建)

**Interfaces:**
- Produces:
  - `OnlineHiqlStore(max_transitions=50_000)`:`.add_episode(states: np[T,D])`、`.__len__()`、`.ready(min_transitions) -> bool`、`.data() -> dict`(`build_gc_data` 形状;FIFO 丢最旧 episode 保持容量;脏标记按需重建)。
  - `sample_value_batch(data, bs, rng, *, future_mode, gamma) -> (s, s_next, g, success, done)`:全为 `[bs,*]` torch tensor(success/done 为 `[bs]`)。
  - `sample_high_actor_batch(data, bs, rng, *, way_steps, future_mode, gamma) -> (s, sw, g)`:`[bs,D]` tensor。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_online_hiql_store.py`)

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.online_hiql_store import (
    OnlineHiqlStore, sample_value_batch, sample_high_actor_batch)


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def test_store_accumulates_and_builds_data():
    store = OnlineHiqlStore(max_transitions=1000)
    assert len(store) == 0 and not store.ready(1)
    store.add_episode(np.arange(8, dtype=np.float32).reshape(4, 2))   # 3 transitions
    store.add_episode((np.arange(6, dtype=np.float32) + 100).reshape(3, 2))  # 2 transitions
    assert len(store) == 5
    assert store.ready(5) and not store.ready(6)
    data = store.data()
    assert data["states"].shape == (7, 2)
    assert len(data["s_idx"]) == 5
    assert data["done"].tolist() == [0, 0, 1, 0, 1]


def test_store_skips_short_episode_and_caches_until_dirty():
    store = OnlineHiqlStore(max_transitions=1000)
    store.add_episode(np.zeros((1, 3), np.float32))   # T=1 -> 跳过
    assert len(store) == 0
    store.add_episode(np.ones((3, 3), np.float32))    # 2 transitions
    d1 = store.data()
    d2 = store.data()
    assert d1 is d2                                   # 未变 -> 缓存复用(同一对象)
    store.add_episode(np.ones((2, 3), np.float32))
    d3 = store.data()
    assert d3 is not d1                               # 脏后重建


def test_store_fifo_drops_oldest_over_cap():
    store = OnlineHiqlStore(max_transitions=4)
    store.add_episode(np.zeros((3, 1), np.float32))   # 2 trans
    store.add_episode(np.ones((3, 1), np.float32))    # 2 trans -> total 4
    store.add_episode(np.full((3, 1), 2.0, np.float32))  # +2 -> 超 4,丢最旧
    assert len(store) <= 4
    # 最旧(全 0)那条应被丢弃
    assert not np.any(store.data()["states"].numpy() == 0.0)


def test_sample_value_batch_shapes_and_success():
    store = OnlineHiqlStore()
    store.add_episode(np.arange(20, dtype=np.float32).reshape(10, 2))
    data = store.data()
    rng = np.random.default_rng(0)
    s, s_next, g, success, done = sample_value_batch(
        data, 32, rng, future_mode="geometric", gamma=0.99)
    assert s.shape == (32, 2) and s_next.shape == (32, 2) and g.shape == (32, 2)
    assert success.shape == (32,) and done.shape == (32,)
    assert set(np.unique(success.numpy()).tolist()) <= {0.0, 1.0}


def test_sample_high_actor_batch_waypoint_in_range():
    store = OnlineHiqlStore()
    store.add_episode(np.arange(40, dtype=np.float32).reshape(20, 2))
    data = store.data()
    rng = np.random.default_rng(0)
    s, sw, g = sample_high_actor_batch(
        data, 16, rng, way_steps=5, future_mode="geometric", gamma=0.99)
    assert s.shape == (16, 2) and sw.shape == (16, 2) and g.shape == (16, 2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_store.py -v`
Expected: FAIL —— `ModuleNotFoundError: ... online_hiql_store`

- [ ] **Step 3: 写最小实现**(新建 `online_hiql_store.py`)

```python
"""HIQL 在线联合微调(Phase 3)的在线轨迹 store + 批采样器。

随 episode 增长的 build_gc_data 结构;goal 用 geometric 采样(只需 last_idx,
无需 stage 检测)。与离线 store 同形,采样器对两者通用。
设计见 docs/superpowers/specs/2026-06-24-hiql-online-joint-finetune-design.md。
"""
from collections import deque

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, sample_gc_goals


class OnlineHiqlStore:
    """完成的 online episode 累积成 build_gc_data 结构;FIFO 控容量,脏标记按需重建。"""

    def __init__(self, max_transitions=50_000):
        self.seqs = deque()
        self.max_transitions = int(max_transitions)
        self._n = 0
        self._data = None

    def add_episode(self, states):
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or len(states) < 2:
            return                       # T<2 无 transition(与 build_gc_data 一致跳过)
        self.seqs.append(states)
        self._n += len(states) - 1       # transition 数 = T-1
        while self._n > self.max_transitions and len(self.seqs) > 1:
            old = self.seqs.popleft()
            self._n -= len(old) - 1
        self._data = None                # 脏

    def __len__(self):
        return self._n

    def ready(self, min_transitions):
        return self._n >= int(min_transitions) and len(self.seqs) >= 1

    def data(self):
        if self._data is None:
            empty = [np.empty(0, dtype=np.int64) for _ in self.seqs]
            self._data = build_gc_data(list(self.seqs), empty)
        return self._data


def _sample_indices(data, bs, rng):
    n = len(data["s_idx"])
    b = rng.integers(0, n, size=bs)
    return b


def sample_value_batch(data, bs, rng, *, future_mode, gamma):
    """从 build_gc_data 形状的 data 采一个 value 训练 batch。返回 torch tensor。"""
    states = data["states"]
    b = _sample_indices(data, bs, rng)
    si, sni, tj = data["s_idx"][b], data["sn_idx"][b], data["traj_id"][b]
    gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"], rng,
                         n_total=len(states), future_mode=future_mode, discount=gamma)
    s = states[si]
    s_next = states[sni]
    g = states[gi]
    success = torch.tensor(si == gi, dtype=torch.float32)
    done = data["done"][b]
    return s, s_next, g, success, done


def sample_high_actor_batch(data, bs, rng, *, way_steps, future_mode, gamma):
    """从 build_gc_data 形状的 data 采一个 high_actor(fixed_waypoint)训练 batch。"""
    states = data["states"]
    b = _sample_indices(data, bs, rng)
    si, tj = data["s_idx"][b], data["traj_id"][b]
    last_arr = np.array([data["last_idx_of"][int(d)] for d in tj], dtype=np.int64)
    wi = np.minimum(si + way_steps, last_arr)
    gi = sample_gc_goals(si, tj, data["last_idx_of"], data["stage_entries_of"], rng,
                         n_total=len(states), future_mode=future_mode, discount=gamma)
    return states[si], states[wi], states[gi]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/online_hiql_store.py resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_store.py
git commit -m "feat: OnlineHiqlStore + value/high_actor batch samplers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 在线联合微调器(新建 online_hiql_finetune.py)

持有 V 头/high_actor 优化器与 EMA target、冻 φ、online+offline 混采,暴露 `on_step / on_episode_end / maybe_update`。

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/online_hiql_finetune.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_finetune.py`(新建)

**Interfaces:**
- Consumes: `value_update_step`(T1)、`high_actor_update_step`(T2)、`OnlineHiqlStore`/`sample_value_batch`/`sample_high_actor_batch`(T4)、`HiqlSubgoal`(含 `.vf`/`.ha`/`.encode_state`,T3)。
- Produces: `OnlineHiqlFinetuner(subgoal, *, offline_seqs, device="cpu", value_lr=1e-5, high_actor_lr=1e-5, way_steps=25, offline_fraction=0.5, batch_size=256, every=1, utd=1, gamma=0.99, expectile=0.7, ema=0.005, future_mode="geometric", value_loss_mode="shared_min", value_mask_mode="done_aware", adv_agg="min", beta=1.0, finetune_value=True, finetune_high_actor=True, max_online_transitions=50_000, seed=0)`
  - `.on_step(s)`:`s` = `[1,D]`(或 `[D]`)当前 vf-输入 state(`subgoal.encode_state` 产物),累入当前 episode。
  - `.on_episode_end()`:把累积 episode 推进 `OnlineHiqlStore`。
  - `.maybe_update(env_steps) -> dict|None`:满足 `env_steps % every == 0` 且 online store `ready` 时,执行 `utd` 次 value/high_actor 更新,返回 metrics(`{"finetune/value_loss":..., "finetune/high_actor_loss":..., "finetune/online_trans":...}`);否则 `None`。
  - 不变式:`subgoal.vf.goal_encoder`(φ)始终 `requires_grad_(False)`。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_online_hiql_finetune.py`)

```python
import copy
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
from resfit.rl_finetuning.chunk_residual.online_hiql_finetune import OnlineHiqlFinetuner


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _make_subgoal(D=6):
    gc = GoalConditionedVF(state_dim=D, rep_dim=4, hidden=16)
    ha = HighActor(state_dim=D, rep_dim=4, hidden=16)
    goal = np.zeros(D, dtype=np.float32)
    # eef_piece 路要 rel_stats;这里直接用 act_feat 风格 feat_stats 太重,改用最简:伪 eef
    return HiqlSubgoal(gc, ha, goal, device="cpu", renorm_subgoal=True,
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(D - 18, np.float32) if D > 18 else np.zeros(0, np.float32),
                                  np.ones(D - 18, np.float32) if D > 18 else np.ones(0, np.float32)))


def _offline_seqs(D=6, n=3):
    return [np.cumsum(np.ones((10, D), np.float32), axis=0) + i for i in range(n)]


def test_finetuner_keeps_phi_frozen():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1, seed=0)
    # φ(goal_encoder)所有参数 requires_grad=False;v1/v2/ha 为 True
    assert all(not p.requires_grad for p in sg.vf.goal_encoder.parameters())
    assert all(p.requires_grad for p in sg.vf.v1.parameters())
    assert all(p.requires_grad for p in sg.vf.v2.parameters())
    assert all(p.requires_grad for p in sg.ha.parameters())


def test_finetuner_maybe_update_noop_until_ready():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1,
                             min_online_transitions=20, seed=0)
    assert ft.maybe_update(1) is None              # online store 空
    for _ in range(3):
        ft.on_step(torch.randn(1, 6))
    ft.on_episode_end()                            # 才 2 transitions < 20
    assert ft.maybe_update(2) is None


def test_finetuner_updates_value_heads_and_high_actor_but_not_phi():
    sg = _make_subgoal()
    phi_before = copy.deepcopy(sg.vf.goal_encoder.state_dict())
    v1_before = copy.deepcopy(sg.vf.v1.state_dict())
    ha_before = copy.deepcopy(sg.ha.state_dict())
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-2, high_actor_lr=1e-2, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=1,
                             min_online_transitions=5, seed=0)
    # 灌几条 online episode
    for ep in range(3):
        for t in range(8):
            ft.on_step(torch.randn(1, 6))
        ft.on_episode_end()
    m = ft.maybe_update(1)
    assert m is not None
    assert np.isfinite(m["finetune/value_loss"])
    assert np.isfinite(m["finetune/high_actor_loss"])
    # φ 不动
    for k in phi_before:
        assert torch.equal(phi_before[k], sg.vf.goal_encoder.state_dict()[k]), f"φ 被改了: {k}"
    # v1 与 ha 动了
    assert any(not torch.equal(v1_before[k], sg.vf.v1.state_dict()[k]) for k in v1_before)
    assert any(not torch.equal(ha_before[k], sg.ha.state_dict()[k]) for k in ha_before)


def test_finetuner_every_gates_update():
    sg = _make_subgoal()
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-3, high_actor_lr=1e-3, batch_size=16,
                             offline_fraction=0.5, way_steps=3, every=10,
                             min_online_transitions=5, seed=0)
    for ep in range(2):
        for t in range(8):
            ft.on_step(torch.randn(1, 6))
        ft.on_episode_end()
    assert ft.maybe_update(3) is None     # 3 % 10 != 0
    assert ft.maybe_update(10) is not None  # 10 % 10 == 0


def test_finetuner_value_only_skips_high_actor():
    sg = _make_subgoal()
    ha_before = copy.deepcopy(sg.ha.state_dict())
    ft = OnlineHiqlFinetuner(sg, offline_seqs=_offline_seqs(), device="cpu",
                             value_lr=1e-2, high_actor_lr=1e-2, batch_size=16,
                             offline_fraction=0.0, way_steps=3, every=1,
                             min_online_transitions=5, finetune_high_actor=False, seed=0)
    for t in range(8):
        ft.on_step(torch.randn(1, 6))
    ft.on_episode_end()
    m = ft.maybe_update(1)
    assert "finetune/high_actor_loss" not in m
    for k in ha_before:
        assert torch.equal(ha_before[k], sg.ha.state_dict()[k]), "high_actor 不该被更新"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_finetune.py -v`
Expected: FAIL —— `ModuleNotFoundError: ... online_hiql_finetune`

- [ ] **Step 3: 写最小实现**(新建 `online_hiql_finetune.py`)

```python
"""HIQL 在线联合微调器(Phase 3):冻 φ,在线更新 V 头 + high_actor,online+offline 混采。

设计见 docs/superpowers/specs/2026-06-24-hiql-online-joint-finetune-design.md。
"""
import copy

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, value_update_step
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import high_actor_update_step
from resfit.rl_finetuning.chunk_residual.online_hiql_store import (
    OnlineHiqlStore, sample_value_batch, sample_high_actor_batch)


class OnlineHiqlFinetuner:
    def __init__(self, subgoal, *, offline_seqs, device="cpu",
                 value_lr=1e-5, high_actor_lr=1e-5, way_steps=25,
                 offline_fraction=0.5, batch_size=256, every=1, utd=1,
                 gamma=0.99, expectile=0.7, ema=0.005, future_mode="geometric",
                 value_loss_mode="shared_min", value_mask_mode="done_aware",
                 adv_agg="min", beta=1.0, finetune_value=True, finetune_high_actor=True,
                 max_online_transitions=50_000, min_online_transitions=2_000, seed=0):
        self.subgoal = subgoal
        self.vf = subgoal.vf
        self.ha = subgoal.ha
        self.device = device
        self.way_steps = int(way_steps)
        self.batch_size = int(batch_size)
        self.every = max(int(every), 1)
        self.utd = max(int(utd), 1)
        self.gamma = gamma
        self.expectile = expectile
        self.ema = ema
        self.future_mode = future_mode
        self.value_loss_mode = value_loss_mode
        self.value_mask_mode = value_mask_mode
        self.adv_agg = adv_agg
        self.beta = beta
        self.finetune_value = bool(finetune_value)
        self.finetune_high_actor = bool(finetune_high_actor)
        self.min_online = int(min_online_transitions)
        self.rng = np.random.default_rng(seed)

        # 混采配比:offline_fraction 决定 V/ha 更新 batch 里 offline 占比
        self.offline_bs = int(round(self.batch_size * offline_fraction))
        self.online_bs = self.batch_size - self.offline_bs
        if self.online_bs <= 0:
            raise ValueError("offline_fraction 过大,online_bs<=0(在线微调需要 online 样本)")

        # 冻 φ,解冻 V 头 + high_actor
        for p in self.vf.parameters():
            p.requires_grad_(False)
        if self.finetune_value:
            for p in list(self.vf.v1.parameters()) + list(self.vf.v2.parameters()):
                p.requires_grad_(True)
            self.target = copy.deepcopy(self.vf).to(device).eval()
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.value_opt = torch.optim.Adam(
                list(self.vf.v1.parameters()) + list(self.vf.v2.parameters()), lr=value_lr)
        if self.finetune_high_actor:
            for p in self.ha.parameters():
                p.requires_grad_(True)
            self.ha_opt = torch.optim.Adam(self.ha.parameters(), lr=high_actor_lr)

        # online / offline store
        self.online = OnlineHiqlStore(max_online_transitions)
        self._ep = []
        if self.offline_bs > 0:
            empty = [np.empty(0, dtype=np.int64) for _ in offline_seqs]
            self.offline_data = build_gc_data(list(offline_seqs), empty)
        else:
            self.offline_data = None

    # ---- 采集钩子 ----
    def on_step(self, s):
        self._ep.append(np.asarray(s.detach().to("cpu"), dtype=np.float32).reshape(-1))

    def on_episode_end(self):
        if len(self._ep) >= 2:
            self.online.add_episode(np.stack(self._ep, axis=0))
        self._ep = []

    # ---- 混采:从 online(+offline)各取一份,拼成大 batch ----
    def _mixed_value_batch(self):
        on = sample_value_batch(self.online.data(), self.online_bs, self.rng,
                                future_mode=self.future_mode, gamma=self.gamma)
        if self.offline_data is not None and self.offline_bs > 0:
            off = sample_value_batch(self.offline_data, self.offline_bs, self.rng,
                                     future_mode=self.future_mode, gamma=self.gamma)
            parts = [torch.cat([a, b], dim=0) for a, b in zip(on, off)]
        else:
            parts = list(on)
        return [p.to(self.device) for p in parts]

    def _mixed_high_actor_batch(self):
        on = sample_high_actor_batch(self.online.data(), self.online_bs, self.rng,
                                     way_steps=self.way_steps, future_mode=self.future_mode,
                                     gamma=self.gamma)
        if self.offline_data is not None and self.offline_bs > 0:
            off = sample_high_actor_batch(self.offline_data, self.offline_bs, self.rng,
                                          way_steps=self.way_steps, future_mode=self.future_mode,
                                          gamma=self.gamma)
            parts = [torch.cat([a, b], dim=0) for a, b in zip(on, off)]
        else:
            parts = list(on)
        return [p.to(self.device) for p in parts]

    # ---- 主更新 ----
    def maybe_update(self, env_steps):
        if env_steps % self.every != 0:
            return None
        if not self.online.ready(self.min_online):
            return None
        v_loss = h_loss = None
        for _ in range(self.utd):
            if self.finetune_value:
                s, s_next, g, success, done = self._mixed_value_batch()
                v_loss = value_update_step(
                    self.vf, self.target, self.value_opt, s, s_next, g, success, done,
                    gamma=self.gamma, expectile=self.expectile, ema=self.ema,
                    value_loss_mode=self.value_loss_mode, value_mask_mode=self.value_mask_mode)
            if self.finetune_high_actor:
                s, sw, g = self._mixed_high_actor_batch()
                h_loss = high_actor_update_step(
                    self.ha, self.vf, self.ha_opt, s, sw, g, beta=self.beta, adv_agg=self.adv_agg)
        metrics = {"finetune/online_trans": float(len(self.online))}
        if v_loss is not None:
            metrics["finetune/value_loss"] = v_loss
        if h_loss is not None:
            metrics["finetune/high_actor_loss"] = h_loss
        return metrics
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_finetune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/online_hiql_finetune.py resfit/rl_finetuning/chunk_residual/tests/test_online_hiql_finetune.py
git commit -m "feat: OnlineHiqlFinetuner (freeze phi, online+offline mixed V/high_actor update)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 加 flags + parser 默认-off 测试(train_chunk_residual.py)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:398-543`(`build_parser`,在 subgoal flag 区段后追加)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_flags.py`(新建)

**Interfaces:**
- Produces(argparse 新增,默认值见下):`--online_finetune_value`(store_true,默认 False)、`--online_finetune_high_actor`(store_true,默认 False)、`--online_value_lr`(float,1e-5)、`--online_high_actor_lr`(float,1e-5)、`--online_finetune_offline_fraction`(float,0.5)、`--online_finetune_every`(int,1)、`--online_finetune_expectile`(float,0.7)、`--online_finetune_ema`(float,0.005)、`--online_finetune_future_mode`(choices stage_entry/geometric,默认 geometric)、`--online_finetune_max_trans`(int,50000)、`--online_finetune_min_trans`(int,2000)。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_online_finetune_flags.py`)

```python
import pytest

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def _req(extra):
    base = ["--base_wandb_id", "x", "--task", "three_piece", "--dataset", "some/ds"]
    return build_parser().parse_args(base + extra)


def test_online_finetune_flags_default_off():
    a = _req([])
    assert a.online_finetune_value is False
    assert a.online_finetune_high_actor is False
    assert a.online_value_lr == 1e-5
    assert a.online_high_actor_lr == 1e-5
    assert a.online_finetune_offline_fraction == 0.5
    assert a.online_finetune_every == 1
    assert a.online_finetune_future_mode == "geometric"


def test_online_finetune_flags_settable():
    a = _req(["--online_finetune_value", "--online_finetune_high_actor",
              "--online_value_lr", "3e-6", "--online_finetune_every", "4",
              "--online_finetune_future_mode", "stage_entry"])
    assert a.online_finetune_value is True
    assert a.online_finetune_high_actor is True
    assert a.online_value_lr == 3e-6
    assert a.online_finetune_every == 4
    assert a.online_finetune_future_mode == "stage_entry"
```

> 注:`base` 里的必填项(`--base_wandb_id` / `--task` / `--dataset`)需与 `build_parser` 实际 required 项一致;实现 Step 1 前先 `grep -n "required=True\|add_argument(\"--base_wandb_id\|--task\|--dataset" train_chunk_residual.py` 核对,缺哪个补哪个(只补够 parse 通过即可,不影响断言)。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_flags.py -v`
Expected: FAIL —— `AttributeError: 'Namespace' object has no attribute 'online_finetune_value'`

- [ ] **Step 3: 写最小实现**(在 `build_parser` 的 `--pi0_feat_cache`/subgoal 区段之后追加)

```python
    # --- HIQL 在线联合微调(spec 2026-06-24;默认全关 = 逐位等价)---
    p.add_argument("--online_finetune_value", action="store_true",
                   help="Phase3 在线微调 V(s,g)(冻 φ,只更新 critic head + EMA target);默认关")
    p.add_argument("--online_finetune_high_actor", action="store_true",
                   help="Phase3 在线微调 high_actor(AWR,用当前 V);默认关")
    p.add_argument("--online_value_lr", type=float, default=1e-5,
                   help="在线 V 微调 LR(小;默认 1e-5)")
    p.add_argument("--online_high_actor_lr", type=float, default=1e-5,
                   help="在线 high_actor 微调 LR(小;默认 1e-5)")
    p.add_argument("--online_finetune_offline_fraction", type=float, default=0.5,
                   help="V/high_actor 在线更新 batch 里 offline demo 占比(RLPD 锚;默认 0.5)")
    p.add_argument("--online_finetune_every", type=int, default=1,
                   help="每多少个 update tick 跑一次在线微调(默认 1)")
    p.add_argument("--online_finetune_expectile", type=float, default=0.7,
                   help="在线 V 微调 expectile τ(默认 0.7)")
    p.add_argument("--online_finetune_ema", type=float, default=0.005,
                   help="在线 V target 的 EMA 系数(默认 0.005)")
    p.add_argument("--online_finetune_future_mode", choices=["stage_entry", "geometric"],
                   default="geometric",
                   help="在线 goal 采样 future_mode(默认 geometric,只需 last_idx 无需 stage 检测)")
    p.add_argument("--online_finetune_max_trans", type=int, default=50_000,
                   help="在线 store 容量(transition 数,FIFO)")
    p.add_argument("--online_finetune_min_trans", type=int, default=2_000,
                   help="在线 store 达到此 transition 数才开始微调")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_flags.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_flags.py
git commit -m "feat: add online-finetune CLI flags (default off = bit-equiv)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 主循环接线(train_chunk_residual.py)

实例化 `OnlineHiqlFinetuner` 并在主循环挂 `on_step / on_episode_end / maybe_update`;捕获各 state_mode 的 demo seqs;默认关时不构造 finetuner(零回归)。

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(subgoal 构造块 `:724-782`、主循环 `:911-985`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_wiring.py`(新建,纯逻辑,不起 env/base_policy)

**Interfaces:**
- Consumes: `OnlineHiqlFinetuner`(T5)、`HiqlSubgoal.encode_state`(T3)。
- 行为:`args.online_finetune_value or args.online_finetune_high_actor` 为真时构造 finetuner;主循环每步 `finetuner.on_step(subgoal.encode_state(...))`、episode done 时 `on_episode_end()`、update 段 `maybe_update(env_steps)` 并把 metrics 并入 wandb。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_online_finetune_wiring.py`)

```python
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import GoalConditionedVF
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import HighActor
from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import maybe_build_finetuner


@pytest.fixture(autouse=True, scope="module")
def _cap_torch_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


class _Args:
    online_finetune_value = False
    online_finetune_high_actor = False
    online_value_lr = 1e-5
    online_high_actor_lr = 1e-5
    online_finetune_offline_fraction = 0.5
    online_finetune_every = 1
    online_finetune_expectile = 0.7
    online_finetune_ema = 0.005
    online_finetune_future_mode = "geometric"
    online_finetune_max_trans = 50_000
    online_finetune_min_trans = 5
    subgoal_way_steps = 3
    gamma = 0.99
    batch_size = 16
    device = "cpu"
    seed = 0


def _make_subgoal():
    gc = GoalConditionedVF(state_dim=30, rep_dim=10, hidden=16)
    ha = HighActor(state_dim=30, rep_dim=10, hidden=16)
    return HiqlSubgoal(gc, ha, np.zeros(30, np.float32), device="cpu",
                       state_mode="eef_piece",
                       rel_stats=(np.zeros(12, np.float32), np.ones(12, np.float32)))


def test_maybe_build_finetuner_off_returns_none():
    a = _Args()
    ft = maybe_build_finetuner(a, _make_subgoal(),
                               [np.ones((10, 30), np.float32)],
                               gc_info={"value_loss_mode": "shared_min",
                                        "value_mask_mode": "done_aware"})
    assert ft is None


def test_maybe_build_finetuner_on_reads_gc_info_modes():
    a = _Args()
    a.online_finetune_value = True
    a.online_finetune_high_actor = True
    ft = maybe_build_finetuner(a, _make_subgoal(),
                               [np.ones((10, 30), np.float32) + i for i in range(3)],
                               gc_info={"value_loss_mode": "hiql",
                                        "value_mask_mode": "hiql"})
    assert ft is not None
    assert ft.value_loss_mode == "hiql" and ft.value_mask_mode == "hiql"
    # 端到端:灌 episode -> maybe_update 出 metrics
    for ep in range(2):
        for t in range(6):
            ft.on_step(torch.randn(1, 30))
        ft.on_episode_end()
    m = ft.maybe_update(1)
    assert m is not None and "finetune/value_loss" in m and "finetune/high_actor_loss" in m
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_wiring.py -v`
Expected: FAIL —— `ImportError: cannot import name 'maybe_build_finetuner'`

- [ ] **Step 3: 写最小实现**

(a) 在 `train_chunk_residual.py` 顶部 import 区加:

```python
from resfit.rl_finetuning.chunk_residual.online_hiql_finetune import OnlineHiqlFinetuner
```

(b) 在模块级(`main` 之前)新增工厂函数:

```python
def maybe_build_finetuner(args, subgoal, finetune_seqs, *, gc_info):
    """开关任一在线微调 flag 时构造 OnlineHiqlFinetuner;否则 None(零回归)。

    value_loss_mode / value_mask_mode 从 gc_value ckpt 的 info 读(同源,不新引口径)。
    finetune_seqs:与 gc_value 同维的离线 demo 标准化 state 序列(eef=30/act_feat=530/pi0=2056)。
    """
    if not (args.online_finetune_value or args.online_finetune_high_actor):
        return None
    assert subgoal is not None, "在线微调需 --subgoal_conditioned"
    return OnlineHiqlFinetuner(
        subgoal, offline_seqs=finetune_seqs, device=args.device,
        value_lr=args.online_value_lr, high_actor_lr=args.online_high_actor_lr,
        way_steps=args.subgoal_way_steps,
        offline_fraction=args.online_finetune_offline_fraction,
        batch_size=args.batch_size, every=args.online_finetune_every,
        gamma=args.gamma, expectile=args.online_finetune_expectile,
        ema=args.online_finetune_ema, future_mode=args.online_finetune_future_mode,
        value_loss_mode=gc_info["value_loss_mode"],
        value_mask_mode=gc_info["value_mask_mode"],
        adv_agg="min",
        finetune_value=args.online_finetune_value,
        finetune_high_actor=args.online_finetune_high_actor,
        max_online_transitions=args.online_finetune_max_trans,
        min_online_transitions=args.online_finetune_min_trans, seed=args.seed)
```

> `adv_agg` 第一版固定 `"min"`(与 high_actor ckpt 训练口径无关项简化;如需对齐 ckpt 的 `adv_agg`,后续从 `load_high_actor` 的 info 读并透传)。

(c) 在 subgoal 构造块里,为三种 state_mode 各记一个统一的 `_finetune_seqs`:
- eef_piece 分支(`:771-781`):在 `_seqs30 = load_or_build_state30(...)` 后加 `_finetune_seqs = _seqs30`。
- act_feat 分支(`:730-752`):`_finetune_seqs = _seqs`(即 `_offline_act_feat_seqs`)。
- pi0_feat 分支(`:753-770`):`_finetune_seqs = _pi0_seqs`。
- 在 `if args.subgoal_conditioned:`(`:724`)之前先 `_finetune_seqs = None`。

紧接 `print(f"[hiql-subgoal] on; ...")`(`:782`)之后加:

```python
    finetuner = maybe_build_finetuner(args, subgoal, _finetune_seqs, gc_info=_subgoal_gc_info) \
        if args.subgoal_conditioned else None
    if finetuner is not None:
        print(f"[online-finetune] on; value={args.online_finetune_value} "
              f"high_actor={args.online_finetune_high_actor} "
              f"offline_frac={args.online_finetune_offline_fraction} "
              f"every={args.online_finetune_every} future={args.online_finetune_future_mode}")
```

(d) 主循环接线:
- 在 `agent.act` 之前、`compute_online_subgoal` 之后(`:914-915` 块内),采集当前 state。在 `if args.subgoal_conditioned:` 块里追加:

```python
            if finetuner is not None:
                _pf = base_policy.last_prefix_feat() if subgoal.state_mode == "pi0_feat" else None
                finetuner.on_step(subgoal.encode_state(obs, rel_raw=cur_rel, prefix_feat=_pf))
```

- episode done 时(`:936` `obs = next_obs` 之前)追加:

```python
        if finetuner is not None and bool(done.any()):
            finetuner.on_episode_end()
```

- 在 update 段(`:964-966` `agent.update(...)` 之后、诊断之前)追加:

```python
                if finetuner is not None:
                    ft_metrics = finetuner.maybe_update(env_steps)
                    if ft_metrics is not None:
                        last_ft_metrics = ft_metrics
```

并在循环外(`:908` `best_sr = 0.0` 附近)初始化 `last_ft_metrics = None`;在 wandb log 段(`:982-984`)把它并入:

```python
                if last_ft_metrics is not None:
                    log_dict.update(last_ft_metrics)
```

- [ ] **Step 4: 运行测试确认通过 + 全量回归**

Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_wiring.py -v`
Expected: PASS

再跑整个 chunk_residual 测试套件确认零回归:
Run: `cd /mnt/mnt/data/resfit && python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全过(此前基线条数 + 本计划新增,无 fail)

- [ ] **Step 5: Commit**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_online_finetune_wiring.py
git commit -m "feat: wire OnlineHiqlFinetuner into Phase3 main loop (default off = zero regression)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 验证与交付(实现完成后,由用户授权跑 GPU)

1. **全量单测**:`python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q` 全过。
2. **默认-off 等价**:`--online_finetune_*` 不传时,主训路径与现状逐位等价(未构造 finetuner)。
3. **真 smoke**:在已跑通的 `three_piece` / `square` 配置上加 `--online_finetune_value --online_finetune_high_actor`,跑 `--smoke` 确认起得来、`[online-finetune] on` 打印正确、不崩。
4. **A/B**:同配置 finetune on vs off,对比 eval success_rate 曲线 + wandb 里 `finetune/value_loss` / `finetune/high_actor_loss` / `finetune/online_trans`,并监控 V(s,g) 分布是否塌方(若塌:降 `--online_value_lr`、提 `--online_finetune_offline_fraction`)。

## 已知限制(spec §7 之外的实现层注记)

- **offline_rb 的 z 不随 ha 更新**:残差 TD3 的 offline 锚 demo transition 里的 `observation.subgoal` 在 buffer 构建时定死(初始 ha);ha 在线漂移后这些 z 变旧。φ 冻结 + 小 LR 下漂移有界,接受;在线 transition 的 z 始终跟随 live ha。
- **act_feat 每步多一次 extractor 前向**:`on_step` 调 `encode_state` 在 act_feat 模式下会再跑一次 ACT encoder(eef_piece/pi0_feat 无此开销;pi0_feat 复用 `last_prefix_feat`)。如成瓶颈,后续让 `compute_online_subgoal` 回传 `s` 复用。
- **adv_agg 第一版固定 min**:与 high_actor ckpt 训练时的 `adv_agg` 解耦;如需严格对齐,从 `load_high_actor` info 透传。
