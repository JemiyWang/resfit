# residual demo-BC(模块②a)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给残差 actor 实现 BC loss(解开 `assert not residual_actor`),并在 chunk_residual 训练里接上一个 default-off 的 `--demo_bc_coef` 旋钮,用现成 offline GT demo 当 bc_batch 做 DAPG 式 demo 锚(均匀 BC)。

**Architecture:** 残差 BC target = 采取动作 − 基座动作(纯函数 `bc_target`,非残差时 = 采取动作、不取 base);`_compute_actor_bc_loss` 解 assert 后用它;train 脚本 demo_bc_coef>0 时从 offline_rb 采 bc_batch 传给现成的 `update_actor_rft`(bc_loss_dynamic 保持 0=均匀)。demo_bc_coef=0(默认)时 bc_batch=None,走原 `update_actor`,逐位等价 baseline。

**Tech Stack:** PyTorch；tensordict（集成测试）；pytest（conda env `residual`，命令前缀 `conda run -n residual`，从仓库根目录跑）。

设计依据：`docs/superpowers/specs/2026-06-06-residual-demo-bc-design.md`。前置:模块①stage_budget 已实现。

---

## 文件结构

- `resfit/rl_finetuning/off_policy/rl/q_agent.py` —— 模块顶部加纯函数 `bc_target`;`_compute_actor_bc_loss` 解 assert + 用 `bc_target`(残差时取 base_action,非残差不取)。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— `--demo_bc_coef` + config 设置 + 依赖断言 + update 调用接 bc_batch。
- `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py` —— 新建(bc_target light + manual 集成 + CLI)。

测试运行(全部):`conda run -n residual python -m pytest <path> -v`(仓库根目录,默认排除 `-m manual`)。
**GPU 约束:GPU 全被在跑实验占,本计划不启动任何 CUDA 训练;所有测试 CPU 跑。**

---

## Task 1: 纯函数 bc_target

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`(模块顶部,import 之后、`class QAgent` 之前)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py`(新建)

- [ ] **Step 1: 写失败测试**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py`:

```python
import torch

from resfit.rl_finetuning.off_policy.rl.q_agent import bc_target


def test_bc_target_residual_subtracts_base():
    action = torch.tensor([[1.0, 2.0, 3.0]])
    base = torch.tensor([[0.5, 1.0, 1.0]])
    out = bc_target(action, base, residual_actor=True)
    assert torch.allclose(out, action - base)


def test_bc_target_demo_zero_when_action_equals_base():
    # offline GT demo: base==action -> 残差 target=0
    a = torch.randn(4, 6)
    out = bc_target(a, a.clone(), residual_actor=True)
    assert torch.allclose(out, torch.zeros_like(a))


def test_bc_target_non_residual_returns_action_ignores_base():
    # 非残差 actor: target=action,且不解引用 base(传 None 也不报错)
    action = torch.tensor([[1.0, 2.0]])
    out = bc_target(action, None, residual_actor=False)
    assert torch.allclose(out, action)
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -v`
Expected: FAIL（`ImportError: cannot import name 'bc_target'`）

- [ ] **Step 3: 实现 bc_target**

在 `resfit/rl_finetuning/off_policy/rl/q_agent.py` 的 import 块之后、`class QAgent(nn.Module):` 之前插入:

```python
def bc_target(action, base_action, residual_actor):
    """残差 actor 的 BC 目标 = 采取动作 − 基座动作;非残差 actor = 采取动作(不解引用 base_action)。"""
    return action - base_action if residual_actor else action
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/q_agent.py \
        resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py
git commit -m "feat(demo_bc): bc_target 纯函数(残差=action-base,非残差=action) + 单测"
```

---

## Task 2: _compute_actor_bc_loss 解 assert + 用 bc_target

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`（`_compute_actor_bc_loss`，约第 473-493 行）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py`

- [ ] **Step 1: 追加失败测试(manual 集成,CPU)**

在 `test_demo_bc.py` 末尾追加:

```python
import math
import pytest
from tensordict import TensorDict
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

_C, _H, _W = 3, 84, 84
_CAM = "observation.images.agentview"
_SD, _FLAT, _NS, _BS = 5, 12, 5, 4


def _residual_agent(bc_coef):
    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    cfg.bc_loss_coef = bc_coef
    cfg.bc_loss_dynamic = 0
    agent = QAgent(obs_shape=(_C, _H, _W), prop_shape=(_SD,), action_dim=_FLAT,
                   rl_cameras=[_CAM], cfg=cfg, residual_actor=True, num_stages=_NS)
    agent.train(True)
    agent.actor_target.train(True)
    return agent


def _obs():
    return {
        _CAM: torch.rand(_BS, _C, _H, _W),
        "observation.state": torch.randn(_BS, _SD),
        "observation.base_action": torch.tanh(torch.randn(_BS, _FLAT)),
        "observation.stage_id": torch.randint(0, _NS, (_BS, 1)).float(),
    }


def _rl_batch():
    return TensorDict({
        "obs": TensorDict(_obs(), batch_size=[_BS]),
        "action": torch.tanh(torch.randn(_BS, _FLAT)),
        ("next", "reward"): torch.zeros(_BS),
        "gamma": torch.full((_BS,), 0.99),
        "nonterminal": torch.ones(_BS),
        ("next", "obs"): TensorDict(_obs(), batch_size=[_BS]),
    }, batch_size=[_BS])


def _bc_batch():
    return TensorDict({
        "obs": TensorDict(_obs(), batch_size=[_BS]),
        "action": torch.tanh(torch.randn(_BS, _FLAT)),
    }, batch_size=[_BS])


@pytest.mark.manual
def test_residual_bc_update_runs_and_reports_bc_loss():
    """残差 actor 带 bc_batch 跑 update -> update_actor_rft -> _compute_actor_bc_loss(assert 已解)。
    metrics 含有限的 rft/bc_loss + actor_loss_total。依赖 VitEncoder,故 manual;CPU。"""
    torch.manual_seed(0)
    agent = _residual_agent(bc_coef=0.1)
    metrics = agent.update(_rl_batch(), stddev=0.05, update_actor=True,
                           bc_batch=_bc_batch(), ref_agent=agent)
    assert math.isfinite(metrics["rft/bc_loss"])
    assert math.isfinite(metrics["train/actor_loss_total"])


@pytest.mark.manual
def test_default_off_no_bc_loss_metric():
    """bc_batch=None(默认关)走 update_actor(非 rft),无 rft/bc_loss。"""
    torch.manual_seed(0)
    agent = _residual_agent(bc_coef=0.0)
    metrics = agent.update(_rl_batch(), stddev=0.05, update_actor=True,
                           bc_batch=None, ref_agent=None)
    assert "rft/bc_loss" not in metrics
    assert math.isfinite(metrics["train/actor_loss_total"])
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -k residual_bc_update -m manual -v`
Expected: FAIL（`AssertionError: Not implemented` —— `_compute_actor_bc_loss` 第一行的 `assert not self.residual_actor`)

- [ ] **Step 3: 改 _compute_actor_bc_loss**

`q_agent.py` 的 `_compute_actor_bc_loss`(约第 473-493 行)整体替换为(删 assert、用 bc_target、残差才取 base_action):

```python
    def _compute_actor_bc_loss(self, batch, *, backprop_encoder):
        obs: dict[str, torch.Tensor] = batch["obs"]

        assert "feat" not in obs, "safety check"
        obs["feat"] = self._encode(obs, augment=True)

        if not backprop_encoder:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        base_action = obs["observation.base_action"] if self.residual_actor else None
        target = bc_target(batch["action"], base_action, self.residual_actor)
        loss = nn.functional.mse_loss(pred_action, target, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss  # noqa: RET504
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -v`
Expected: PASS（Task1 的 3 个 light + 本 Task 2 个 manual 集成）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/q_agent.py \
        resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py
git commit -m "feat(demo_bc): 残差 actor 的 _compute_actor_bc_loss(解 assert,target=action-base)"
```

---

## Task 3: train 脚本接线(--demo_bc_coef + bc_batch)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`（`build_parser`；`main` config 约 248-251 / 断言 / update 调用约 341）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py`

- [ ] **Step 1: 追加失败测试(CLI light)**

在 `test_demo_bc.py` 末尾追加:

```python
def test_cli_demo_bc_coef_default_zero():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args([]).demo_bc_coef == 0.0


def test_cli_demo_bc_coef_parses():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args(["--demo_bc_coef", "0.1"]).demo_bc_coef == 0.1
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -k cli -v`
Expected: FAIL（argparse `unrecognized arguments: --demo_bc_coef` / 或 `demo_bc_coef` AttributeError）

- [ ] **Step 3a: build_parser 加 --demo_bc_coef**

`train_chunk_residual.py` 的 `build_parser` 里(放在 `--stage_budget` 附近),加:

```python
    p.add_argument("--demo_bc_coef", type=float, default=0.0,
                   help="残差 actor 的 demo-BC 权重(模块②a);0=关(逐位等价 baseline)。"
                        ">0 需 offline_fraction>0(bc_batch 取自 offline_rb)、--actor raw")
```

- [ ] **Step 3b: main config 设置 + 依赖断言**

`main` 里 `cfg.agent.actor.action_scale = args.action_scale`(约第 251 行)之后,加:

```python
    cfg.agent.bc_loss_coef = args.demo_bc_coef
    cfg.agent.bc_loss_dynamic = 0          # 均匀 BC(②a 不开 DAPG 动态)
    if args.demo_bc_coef > 0:
        assert args.actor == "raw", "demo_bc 第一版只支持 --actor raw"
        assert args.offline_fraction > 0, \
            "demo_bc_coef>0 需 offline_fraction>0(bc_batch 取自 offline_rb)"
```

- [ ] **Step 3c: update 调用接 bc_batch**

`main` 训练循环里的(约第 341 行):

```python
                m_upd = agent.update(batch, args.stddev, update_actor, bc_batch=None, ref_agent=None)
```

替换为:

```python
                bc_batch = None
                if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:
                    bc_batch = offline_rb.sample(args.batch_size).to(args.device, non_blocking=True)
                m_upd = agent.update(batch, args.stddev, update_actor,
                                     bc_batch=bc_batch,
                                     ref_agent=(agent if bc_batch is not None else None))
```

- [ ] **Step 4: 运行测试确认通过(全 CPU)**

CLI light + 全文件:
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -m "not manual" -v`
Expected: PASS（light:bc_target 3 + cli 2 = 5)

manual 集成再跑一遍确认没被接线破坏:
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -m manual -v`
Expected: PASS（2 个 manual）

- [ ] **Step 5: 轻量接线验证(不启动 CUDA 训练)**

```bash
conda run -n residual python -c "from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser; a=build_parser().parse_args(['--demo_bc_coef','0.1']); assert a.demo_bc_coef==0.1; print('OK parser')"
grep -n "demo_bc_coef\|bc_batch=bc_batch\|offline_rb.sample(args.batch_size)" resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
```
Expected: 打印 `OK parser`;grep 看到 config 设置(bc_loss_coef=demo_bc_coef)、断言、update 接 bc_batch、offline_rb.sample。

- [ ] **Step 6: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py
git commit -m "feat(demo_bc): train --demo_bc_coef 接线(bc_batch 取自 offline_rb)+ CLI 测试"
```

---

## 完成后(不属本实现,A/B 在 plan/run 阶段)

见 spec §6:nas10 强 base + best 基座 + `--offline_fraction 0.5`,A=`--demo_bc_coef 0` vs B=扫 0.01/0.1,
看 success/stage-reach 不降、残差更受控,旁证 wandb `rft/bc_loss`。这一发只确认消费端造好且 demo-BC 不砸;
真正治稀疏靠 ②b(在线 relabel,复用本消费端)。
