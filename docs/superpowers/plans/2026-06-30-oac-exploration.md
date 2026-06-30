# OAC 乐观探索接入 resfit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 resfit 的 chunk 残差 RL 加 OAC(乐观 actor-critic)探索,开关式、默认关、关时与现状逐位等价。

**Architecture:** 只改探索采样路径。新增一个纯函数模块做 OAC 偏移数学(可独立单测);`QAgent.act()` 在 `oac_explore=True 且非 eval` 时走新方法 `_act_oac()`,用 critic ensemble 算 Q 乐观上界、对残差均值求梯度、把均值沿梯度偏移后再采样。critic/actor 的学习更新一行不碰。

**Tech Stack:** Python 3.10, PyTorch, torchrl, pytest;conda env `residual`。

## Global Constraints

- 默认关、零回归:`oac_explore=False` 时 `act()` 必须完全等价于现有 `_act_default` 路径。
- 不动学习侧:`update_critic`/`update_actor`/target 路径不得改。
- 偏移在动作空间做(resfit 的 `TruncatedNormal` 无 pre-tanh 层)。
- 梯度取在 critic 真实看到的动作上:`residual_actor=True` 时为 `clamp(base_action + 残差均值, -1, 1)`,否则为残差均值(镜像 `q_agent.py:359-364`)。
- σ_Q 用 critic ensemble 均值/标准差(`unbiased=False`),对任意头数成立(chunk 路 `num_q=10`)。
- 默认值:`oac_explore=False`、`oac_beta_ub=4.0`、`oac_delta=0.5`。
- 测试全程 CPU,不启真训练;依赖 VitEncoder 的测试标 `@pytest.mark.manual`(与现有约定一致)。
- 仅在 `chunk-residual-validation` 分支提交;每个 Task 末尾只 `git add` 本 Task 触碰的文件。

---

### Task 1: OAC 配置项 + CLI flag(wiring)

**Files:**
- Modify: `resfit/rl_finetuning/config/rlpd.py`(`QAgentConfig`,约 97-129 行区间)
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`build_parser` 约 562-731;cfg 透传约 899-904)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py`(新建)

**Interfaces:**
- Produces: `QAgentConfig.oac_explore: bool`、`oac_beta_ub: float`、`oac_delta: float`(默认 False/4.0/0.5)
- Produces: CLI flag `--oac_explore`(store_true)、`--oac_beta_ub`(float)、`--oac_delta`(float)

- [ ] **Step 1: 写失败测试(parser 默认 + 可设 + config 默认)**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py`:

```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
from resfit.rl_finetuning.config.rlpd import QAgentConfig


def _req(extra):
    base = ["--base_wandb_id", "x", "--task", "three_piece", "--dataset", "some/ds"]
    return build_parser().parse_args(base + extra)


def test_oac_flags_default_off():
    a = _req([])
    assert a.oac_explore is False
    assert a.oac_beta_ub == 4.0
    assert a.oac_delta == 0.5


def test_oac_flags_settable():
    a = _req(["--oac_explore", "--oac_beta_ub", "2.0", "--oac_delta", "0.1"])
    assert a.oac_explore is True
    assert a.oac_beta_ub == 2.0
    assert a.oac_delta == 0.1


def test_qagent_config_has_oac_fields_with_defaults():
    cfg = QAgentConfig()
    assert cfg.oac_explore is False
    assert cfg.oac_beta_ub == 4.0
    assert cfg.oac_delta == 0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q`
Expected: FAIL（`AttributeError: 'Namespace' object has no attribute 'oac_explore'` / `QAgentConfig` 无该字段）

- [ ] **Step 3: 加 config 字段**

在 `resfit/rl_finetuning/config/rlpd.py` 的 `QAgentConfig` 内(在 `target_action_noise` 行附近、类内任意字段处)新增:

```python
    # OAC(乐观 actor-critic)探索:仅改探索采样,默认关、关时零回归。见 docs/superpowers/specs/2026-06-30-oac-exploration-design.md
    oac_explore: bool = False
    oac_beta_ub: float = 4.0
    oac_delta: float = 0.5
```

- [ ] **Step 4: 加 CLI flag**

在 `train_chunk_residual.py` `build_parser()` 内(`return p` 之前)新增:

```python
    p.add_argument("--oac_explore", action="store_true",
                   help="开启 OAC 乐观探索(只改探索采样,默认关、关时零回归)")
    p.add_argument("--oac_beta_ub", type=float, default=4.0,
                   help="OAC 乐观上界系数 β_UB(Q_UB = μ_Q + β_UB·σ_Q)")
    p.add_argument("--oac_delta", type=float, default=0.5,
                   help="OAC 均值偏移的 KL 预算 δ(偏移≈√(2δ)·std)")
```

- [ ] **Step 5: 透传进 agent cfg**

在 `train_chunk_residual.py` 设置 `cfg.agent.*` 的那段(`cfg.agent.device = args.device` 之后,约 904 行):

```python
    cfg.agent.oac_explore = args.oac_explore
    cfg.agent.oac_beta_ub = args.oac_beta_ub
    cfg.agent.oac_delta = args.oac_delta
```

- [ ] **Step 6: 运行确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q`
Expected: PASS（3 passed）

- [ ] **Step 7: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/config/rlpd.py \
        resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py
git commit -m "feat: OAC config fields + CLI flags (default off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: OAC 偏移数学纯函数模块

**Files:**
- Create: `resfit/rl_finetuning/off_policy/rl/oac_explore.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py`(追加)

**Interfaces:**
- Produces: `q_upper_bound(q_per_head: Tensor[K,B,1] | Tensor[K,B], beta_ub: float) -> Tensor[B]`
  返回 `mean_k Q_k + beta_ub * std_k Q_k`(std unbiased=False)。
- Produces: `optimistic_mean_shift(mu_T: Tensor[B,A], std_T: Tensor[B,A], q_ub_fn: Callable[[Tensor[B,A]], Tensor[B]], delta: float) -> Tensor[B,A]`
  返回偏移后的均值 `mu_E`(已 detach,确定性);`delta<=0` 时返回 `mu_T` 原值。

- [ ] **Step 1: 写失败测试**

在 `test_oac_explore.py` 追加:

```python
import math
import torch
from resfit.rl_finetuning.off_policy.rl.oac_explore import q_upper_bound, optimistic_mean_shift


def test_q_upper_bound_two_heads_matches_paper():
    # K=2 时 σ = |Q1-Q2|/2,μ = (Q1+Q2)/2
    q = torch.tensor([[[1.0]], [[3.0]]])              # [K=2, B=1, 1]
    out = q_upper_bound(q, beta_ub=2.0)               # μ=2, σ=1 -> 2 + 2*1 = 4
    assert torch.allclose(out, torch.tensor([4.0]))


def test_shift_kl_budget_invariant():
    # 线性 Q_UB: q_ub(a)=sum(w*a) -> grad=w;偏移在 Σ^{-1} 范数下应恰为 sqrt(2δ)
    torch.manual_seed(0)
    B, A = 4, 6
    w = torch.randn(B, A)
    mu_T = torch.randn(B, A)
    std_T = torch.rand(B, A) * 0.1 + 0.01
    delta = 0.3
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: (w * a).sum(-1), delta)
    mu_C = mu_E - mu_T
    sigma = std_T ** 2
    kl_norm_sq = (mu_C ** 2 / sigma).sum(-1)          # 应 ≈ 2δ(每行)
    assert torch.allclose(kl_norm_sq, torch.full((B,), 2 * delta), atol=1e-4)
    # 方向:mu_C 与 Σ·w 同向(逐维同号)
    assert torch.all(torch.sign(mu_C) == torch.sign(sigma * w))


def test_shift_delta_zero_is_noop():
    mu_T = torch.randn(3, 5)
    std_T = torch.rand(3, 5) + 0.01
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: a.sum(-1), delta=0.0)
    assert torch.allclose(mu_E, mu_T)


def test_shift_rows_independent():
    # 改第 0 行的目标权重不应影响第 1 行的偏移
    mu_T = torch.zeros(2, 4)
    std_T = torch.ones(2, 4) * 0.1
    w_a = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    w_b = torch.tensor([[0, 0, 1.0, 0], [0, 1.0, 0, 0]])    # 第1行与 w_a 相同
    mu_a = optimistic_mean_shift(mu_T, std_T, lambda a: (w_a * a).sum(-1), 0.2)
    mu_b = optimistic_mean_shift(mu_T, std_T, lambda a: (w_b * a).sum(-1), 0.2)
    assert torch.allclose(mu_a[1], mu_b[1])                 # 第1行不受第0行变化影响


def test_shift_zero_grad_no_nan():
    mu_T = torch.randn(2, 3)
    std_T = torch.ones(2, 3) * 0.1
    mu_E = optimistic_mean_shift(mu_T, std_T, lambda a: (a * 0.0).sum(-1), 0.5)
    assert torch.isfinite(mu_E).all()
    assert torch.allclose(mu_E, mu_T, atol=1e-3)           # 梯度为0 -> 几乎不偏移
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q`
Expected: FAIL（`ModuleNotFoundError: ... oac_explore`）

- [ ] **Step 3: 写实现**

新建 `resfit/rl_finetuning/off_policy/rl/oac_explore.py`:

```python
# OAC(Optimistic Actor-Critic, Ciosek et al. NeurIPS 2019)探索偏移的纯函数。
# 只做"把探索分布均值沿 Q 乐观上界梯度偏移"的数学,不依赖具体 agent,便于单测。
# 参考: /mnt/mnt/data/wjm/residual/oac-explore/optimistic_exploration.py
from __future__ import annotations

import math
from typing import Callable

import torch


def q_upper_bound(q_per_head: torch.Tensor, beta_ub: float) -> torch.Tensor:
    """Q 乐观上界 μ_Q + β_UB·σ_Q,用 critic ensemble 的均值/标准差。

    q_per_head: [K, B, 1] 或 [K, B];K=critic 头数。
    返回: [B]。K=2 时 σ 退化为 |Q1-Q2|/2(unbiased=False)。
    """
    q = q_per_head.squeeze(-1) if q_per_head.dim() == 3 else q_per_head   # [K, B]
    mu_q = q.mean(0)                                                      # [B]
    sigma_q = q.std(0, unbiased=False)                                    # [B]
    return mu_q + beta_ub * sigma_q


def optimistic_mean_shift(
    mu_T: torch.Tensor,
    std_T: torch.Tensor,
    q_ub_fn: Callable[[torch.Tensor], torch.Tensor],
    delta: float,
) -> torch.Tensor:
    """把均值 mu_T 沿 q_ub_fn 的梯度方向偏移,KL 预算 δ;协方差不变。

    mu_T, std_T: [B, A]。q_ub_fn(action[B,A]) -> Q_UB[B](已含 β_UB·σ)。
    返回偏移后的均值 mu_E[B,A](已 detach)。delta<=0 时原样返回 mu_T。
    数学(对角 Σ=std²):mu_E = mu_T + √(2δ)·Σ·g / sqrt(gᵀΣg),g=∇_a Q_UB|_{mu_T}。
    """
    if delta <= 0.0:
        return mu_T.detach()

    mu_leaf = mu_T.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        q_ub = q_ub_fn(mu_leaf)                       # [B]
        grad = torch.autograd.grad(q_ub.sum(), mu_leaf)[0]   # [B, A]

    sigma = std_T ** 2                                # [B, A]
    denom = torch.sqrt((grad ** 2 * sigma).sum(-1, keepdim=True)) + 1e-6   # [B, 1]
    mu_c = math.sqrt(2.0 * delta) * (sigma * grad) / denom
    return (mu_leaf + mu_c).detach()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q`
Expected: PASS（含 Task1 共 8 passed）

- [ ] **Step 5: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/off_policy/rl/oac_explore.py \
        resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py
git commit -m "feat: pure OAC mean-shift helper (q_upper_bound + optimistic_mean_shift)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `_act_oac` 方法 + `act()` 分支 + `__init__` 读 cfg

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`(import 区;`__init__` 约 68-72;`act` 约 284-309;新增 `_act_oac`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py`(追加 manual 测试)

**Interfaces:**
- Consumes: `optimistic_mean_shift`、`q_upper_bound`(Task 2);`QAgentConfig.oac_*`(Task 1)
- Produces: `QAgent._act_oac(self, obs, stddev) -> Tensor[B, A]`;`QAgent.act` 在 `oac_explore and not eval_mode` 时走该方法

- [ ] **Step 1: 写失败测试(manual,CPU 真 agent)**

在 `test_oac_explore.py` 追加:

```python
import pytest


def _build_agent(oac_explore, beta_ub=4.0, delta=0.5):
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    torch.manual_seed(0)
    C, H, W = 3, 84, 84
    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    cfg.oac_explore = oac_explore
    cfg.oac_beta_ub = beta_ub
    cfg.oac_delta = delta
    agent = QAgent(obs_shape=(C, H, W), prop_shape=(5,), action_dim=12,
                   rl_cameras=["observation.images.agentview"], cfg=cfg,
                   residual_actor=True)
    agent.train(False)
    return agent, C, H, W


def _obs(C, H, W, B=8):
    return {"observation.images.agentview": torch.rand(B, C, H, W),
            "observation.state": torch.randn(B, 5),
            "observation.base_action": torch.tanh(torch.randn(B, 12))}


@pytest.mark.manual
def test_act_oac_returns_finite_shape():
    agent, C, H, W = _build_agent(oac_explore=True)
    with torch.no_grad():
        a = agent.act(_obs(C, H, W), eval_mode=False, stddev=0.05, cpu=True)
    assert a.shape == (8, 12)
    assert torch.isfinite(a).all()


@pytest.mark.manual
def test_act_dispatch_default_when_off():
    # oac_explore=False:_act_oac 不应被调用(置爆炸桩仍不报)
    agent, C, H, W = _build_agent(oac_explore=False)
    agent._act_oac = lambda *a, **k: (_ for _ in ()).throw(AssertionError("OAC 不该被调用"))
    with torch.no_grad():
        agent.act(_obs(C, H, W), eval_mode=False, stddev=0.05, cpu=True)   # 不报即对


@pytest.mark.manual
def test_act_dispatch_default_when_eval():
    # oac_explore=True 但 eval_mode=True:走均值,不走 OAC
    agent, C, H, W = _build_agent(oac_explore=True)
    agent._act_oac = lambda *a, **k: (_ for _ in ()).throw(AssertionError("eval 不该走 OAC"))
    with torch.no_grad():
        agent.act(_obs(C, H, W), eval_mode=True, stddev=0.0, cpu=True)      # 不报即对
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -m manual -q`
Expected: FAIL（`test_act_oac_returns_finite_shape` 报 `AttributeError: ... _act_oac`)

- [ ] **Step 3: 加 import**

`resfit/rl_finetuning/off_policy/rl/q_agent.py` 顶部 import 区,在 `from ...rl.critic import Critic` 之后新增:

```python
from resfit.rl_finetuning.off_policy.rl.oac_explore import optimistic_mean_shift
```

- [ ] **Step 4: `__init__` 读 cfg(约 72 行 `self.subgoal_conditioned = ...` 之后)**

```python
        # OAC 探索(默认关;getattr 兜底老 cfg 实例)
        self.oac_explore = getattr(cfg, "oac_explore", False)
        self.oac_beta_ub = getattr(cfg, "oac_beta_ub", 0.0)
        self.oac_delta = getattr(cfg, "oac_delta", 0.0)
```

- [ ] **Step 5: `act()` 加分支**

把 `act()`(约 295-301)里这段:

```python
        action = self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=None,
            use_target=False,
        )
```

改为:

```python
        if self.oac_explore and not eval_mode:
            action = self._act_oac(obs, stddev)
        else:
            action = self._act_default(
                obs=obs,
                eval_mode=eval_mode,
                stddev=stddev,
                clip=None,
                use_target=False,
            )
```

- [ ] **Step 6: 新增 `_act_oac`(放在 `_act_default` 之后)**

```python
    def _act_oac(self, obs: dict[str, torch.Tensor], stddev: float) -> torch.Tensor:
        """OAC 乐观探索:把探索分布均值沿 Q 乐观上界梯度偏移后采样。

        只用于训练 rollout 探索(eval / target 路径不经此)。
        梯度取在 critic 真实看到的动作上(residual_actor=True 时为 clamp(base+残差均值))。
        """
        from resfit.rl_finetuning.off_policy.rl.oac_explore import q_upper_bound

        dist = self.actor.forward(obs, stddev)        # TruncatedNormal(scaled_mu, std)
        mu_T = dist.loc                               # [B, A] 残差均值
        std_T = dist.scale                            # [B, A]

        feat = obs["feat"]
        prop = self._critic_prop(obs)
        base_action = obs.get("observation.base_action")

        def q_ub_fn(a: torch.Tensor) -> torch.Tensor:
            if self.residual_actor:
                crit_act = torch.clamp(base_action + a, -1.0, 1.0)
            else:
                crit_act = a
            q = self.critic(feat, prop, crit_act)     # [K, B, 1]
            return q_upper_bound(q, self.oac_beta_ub)  # [B]

        mu_E = optimistic_mean_shift(mu_T, std_T, q_ub_fn, self.oac_delta)
        return utils.TruncatedNormal(mu_E, std_T).sample(clip=None)
```

- [ ] **Step 7: 运行确认通过(manual + 非 manual 全跑)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -m manual -q`
Expected: PASS（非 manual 8 passed;manual 3 passed）

- [ ] **Step 8: 零回归回归测试(整套 chunk_residual 测试)**

Run: `cd /mnt/mnt/data/resfit && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -q -m "not manual"`
Expected: PASS（与改动前数量一致,无新失败)

- [ ] **Step 9: 提交**

```bash
cd /mnt/mnt/data/resfit
git add resfit/rl_finetuning/off_policy/rl/q_agent.py \
        resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py
git commit -m "feat: wire OAC optimistic exploration into QAgent.act (toggle, zero-regression when off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 验收(全部 Task 完成后)

- `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py -q`(含 manual)全绿。
- 整套 `chunk_residual/tests`(`-m "not manual"`)零新增失败。
- `oac_explore=False` 时 `act()` 走原 `_act_default`(由 dispatch 测试保证)。
- 冒烟(可选,需 GPU,本计划不含):任挑一任务加 `--oac_explore --oac_delta 0.5 --oac_beta_ub 4.0` 起短跑,确认能正常 rollout/落 buffer。
