# cl=1 step 级 stage-conditioned residual 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 obs 里已存在但从未喂网络的 `observation.stage_id`，以 one-hot 形式喂进 raw residual actor 与 critic，全程由开关控制、默认关闭（关闭时行为与现状逐字节相同）。

**Architecture:** 新增 `off_policy/rl/stage_utils.py`（stage one-hot 工具，纯函数易测）。Actor 自己从 obs 取 stage_id 在 forward 内 append（因为 Actor.forward 接 obs dict）。Critic 的 forward 不接 obs，拿不到 stage_id，故由 QAgent 把 stage one-hot 拼进 `prop` 再传给 critic（critic 源码不改，只是 prop_dim 加宽）。两个独立 config 开关 `--stage_conditioned`（生效）与 `--stage_budget_mode`（仅占位）。

**Tech Stack:** PyTorch；pytest（`uv run pytest`，排除 `-m "not manual"`）；torchrl TensorDict（仅 manual 烟雾用）。

**关联：** 设计文档 `docs/superpowers/specs/2026-06-04-stage-conditioned-residual-design.md`。

**与 spec 的一处微调：** spec 写 helper 放 "chunk_residual 包内"，本计划改放 `off_policy/rl/stage_utils.py`，理由是 actor.py/q_agent.py 在 `off_policy/rl` 下，从 `chunk_residual` import 会形成 `off_policy → chunk_residual` 反向依赖。功能不变。

---

## 文件结构

- **新建** `resfit/rl_finetuning/off_policy/rl/stage_utils.py` — stage one-hot 编码与 prop 拼接纯函数。
- **新建** `resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py` — helper 单测。
- **新建** `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py` — Actor / Critic stage-conditioning 单测 + manual 烟雾。
- **改** `resfit/rl_finetuning/off_policy/rl/actor.py` — Actor 加 `stage_conditioned`/`num_stages`，prop_dim 加宽，forward append one-hot。
- **改** `resfit/rl_finetuning/off_policy/rl/q_agent.py` — QAgent 加 `stage_conditioned`/`num_stages`，critic prop_dim 加宽，`_critic_prop` 拼接，替换 6 个 critic 主路径调用点，向 Actor 传参。
- **改** `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` — argparse 两开关，从 `NUM_STAGES` 取段数，构造 QAgent 时传参。

---

## Task 1: stage_utils 纯函数 + 单测

**Files:**
- Create: `resfit/rl_finetuning/off_policy/rl/stage_utils.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py`

- [ ] **Step 1: 写失败测试**

Create `resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py`:

```python
import torch
from resfit.rl_finetuning.off_policy.rl.stage_utils import stage_onehot, append_stage


def test_stage_onehot_basic():
    sid = torch.tensor([[0.0], [3.0], [4.0]])      # [B,1] float
    oh = stage_onehot(sid, num_stages=5)
    assert oh.shape == (3, 5)
    assert oh.dtype == torch.float32
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([0, 3, 4]))
    # 每行恰好一个 1
    assert torch.equal(oh.sum(dim=-1), torch.ones(3))


def test_stage_onehot_accepts_1d():
    sid = torch.tensor([1.0, 2.0])                 # [B] 也接受
    oh = stage_onehot(sid, num_stages=5)
    assert oh.shape == (2, 5)
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([1, 2]))


def test_stage_onehot_clamps_out_of_range():
    sid = torch.tensor([[-1.0], [9.0]])            # 越界被 clamp 到 [0,4]
    oh = stage_onehot(sid, num_stages=5)
    assert torch.equal(oh.argmax(dim=-1), torch.tensor([0, 4]))


def test_append_stage_widens_prop():
    prop = torch.randn(4, 7)
    sid = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    out = append_stage(prop, sid, num_stages=5)
    assert out.shape == (4, 7 + 5)
    # 前 7 维与原 prop 完全一致
    assert torch.equal(out[:, :7], prop)
    # 不同 stage → 后 5 维不同
    assert not torch.equal(out[0, 7:], out[3, 7:])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py -v`
Expected: FAIL，`ModuleNotFoundError: ... stage_utils`。

- [ ] **Step 3: 写实现**

Create `resfit/rl_finetuning/off_policy/rl/stage_utils.py`:

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""stage-conditioned residual 用的轻量工具。

stage_id 在 obs 里是 float([B,1] 或 [B])，取值 0..num_stages-1（瞬时阶段，可回退）。
这里把它转成 one-hot 并可拼到 prop 向量上。纯函数，无副作用，便于单测。
"""
import torch
import torch.nn.functional as F


def stage_onehot(stage_id: torch.Tensor, num_stages: int) -> torch.Tensor:
    """stage_id [B,1] 或 [B] float -> one-hot [B, num_stages] float32。

    越界值 clamp 到 [0, num_stages-1]（瞬时 stage 理论上不越界，clamp 仅作防御）。
    """
    idx = stage_id.reshape(stage_id.shape[0]).long().clamp_(0, num_stages - 1)
    oh = F.one_hot(idx, num_classes=num_stages)
    return oh.to(dtype=torch.float32, device=stage_id.device)


def append_stage(prop: torch.Tensor, stage_id: torch.Tensor, num_stages: int) -> torch.Tensor:
    """把 stage one-hot 拼到 prop 末尾：[B, P] -> [B, P + num_stages]。"""
    return torch.cat([prop, stage_onehot(stage_id, num_stages)], dim=-1)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py -v`
Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/stage_utils.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_utils.py
git commit -m "feat(stage): stage_onehot/append_stage 工具 + 单测"
```

---

## Task 2: Actor stage-conditioning

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/actor.py:68-78`（构造）、`:150-166`（forward）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`

- [ ] **Step 1: 写失败测试**

Create `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`:

```python
import torch
from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor

REPR, PATCH, PROP, FLAT = 32, 16, 3, 20
B, NUM_STAGES = 6, 5


def _actor(stage_conditioned):
    cfg = ActorConfig()  # spatial_emb=0 默认 → 走 residual 兼容的 else 分支
    return Actor(REPR, PATCH, PROP, FLAT, cfg, residual_actor=True,
                 stage_conditioned=stage_conditioned, num_stages=NUM_STAGES)


def _obs(with_stage, stage_val=0):
    o = {
        "feat": torch.randn(B, REPR // PATCH, PATCH),
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
    }
    if with_stage:
        o["observation.stage_id"] = torch.full((B, 1), float(stage_val))
    return o


def test_actor_off_does_not_require_stage_id():
    # 关闭时 forward 不读 stage_id（obs 里没有也不报错），输出 shape 不变
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=False).eval()
    dist = actor.forward(_obs(with_stage=False), std=0.0)
    assert dist.mean.shape == (B, FLAT)


def test_actor_on_shape_ok():
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=True).eval()
    dist = actor.forward(_obs(with_stage=True, stage_val=2), std=0.0)
    assert dist.mean.shape == (B, FLAT)


def test_actor_on_is_sensitive_to_stage():
    # 同一 actor、同一非 stage 输入，仅改 stage_id → 输出应不同（证明 stage 真的喂进去了）
    torch.manual_seed(0)
    actor = _actor(stage_conditioned=True).eval()
    base = _obs(with_stage=True, stage_val=0)
    o0 = dict(base); o0["observation.stage_id"] = torch.zeros(B, 1)
    o3 = dict(base); o3["observation.stage_id"] = torch.full((B, 1), 3.0)
    m0 = actor.forward(o0, std=0.0).mean
    m3 = actor.forward(o3, std=0.0).mean
    assert not torch.allclose(m0, m3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py -v`
Expected: FAIL，`Actor.__init__() got an unexpected keyword argument 'stage_conditioned'`。

- [ ] **Step 3: 改 Actor 构造**

In `resfit/rl_finetuning/off_policy/rl/actor.py`, add the import near the top (after line 8):

```python
from resfit.rl_finetuning.off_policy.rl.stage_utils import stage_onehot
```

Replace the constructor signature + early body (lines 69-78):

```python
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False,
                 stage_conditioned: bool = False, num_stages: int = 0):
        super().__init__()

        self.prop_dim = prop_dim
        self.residual_actor = residual_actor
        self.stage_conditioned = stage_conditioned
        self.num_stages = num_stages
        self.cfg = cfg

        if residual_actor:
            # The residual actor takes the base action as input alongside the state
            self.prop_dim += action_dim

        if stage_conditioned:
            # stage-conditioned: stage one-hot 作为额外 prop 维度喂入
            self.prop_dim += num_stages
```

- [ ] **Step 4: 改 Actor.forward**

In `resfit/rl_finetuning/off_policy/rl/actor.py`, replace the `all_input` block (lines 158-166):

```python
        all_input = [feat]
        if self.prop_dim > 0:
            prop = obs["observation.state"]
            all_input.append(prop)
            if self.residual_actor:
                # The residual actor takes the base action as input alongside the state
                all_input.append(obs["observation.base_action"])
            if self.stage_conditioned:
                all_input.append(stage_onehot(obs["observation.stage_id"], self.num_stages))

        policy_input = torch.cat(all_input, dim=-1)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py -v`
Expected: 3 passed。

- [ ] **Step 6: 跑现有 actor 相关测试确认无回归**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py -v`
Expected: 全 passed（flow actor 不传新参数，默认关闭，行为不变）。

- [ ] **Step 7: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/actor.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py
git commit -m "feat(stage): raw Actor 支持 stage-conditioning（默认关）"
```

---

## Task 3: Critic 端 stage 敏感性（验证"拼进 prop"有效）

Critic 源码不改（它本就接受任意 `prop_dim`）。本任务用单测证明：把 stage one-hot 拼进 prop 喂 critic，Q 会随 stage 变化——这是 QAgent 接线（Task 4）的正确性前提。

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`（追加）

- [ ] **Step 1: 追加失败测试**

Append to `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`:

```python
from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.rl.critic import Critic
from resfit.rl_finetuning.off_policy.rl.stage_utils import append_stage


def _critic(prop_dim):
    cfg = CriticConfig()
    cfg.loss.type = "mse"            # 标量 Q，断言简单
    return Critic(repr_dim=REPR, patch_repr_dim=PATCH, prop_dim=prop_dim,
                  action_dim=FLAT, cfg=cfg).eval()


def test_critic_q_sensitive_to_stage_in_prop():
    torch.manual_seed(0)
    critic = _critic(prop_dim=PROP + NUM_STAGES)   # prop 已加宽
    feat = torch.randn(B, REPR // PATCH, PATCH)
    prop = torch.randn(B, PROP)
    act = torch.tanh(torch.randn(B, FLAT))
    sid0 = torch.zeros(B, 1)
    sid3 = torch.full((B, 1), 3.0)
    # 用 forward：mse 下确定返回 [num_q, B, 1]，shape 可预测
    q0 = critic.forward(feat, append_stage(prop, sid0, NUM_STAGES), act)
    q3 = critic.forward(feat, append_stage(prop, sid3, NUM_STAGES), act)
    assert q0.shape[-2] == B and q0.shape[-1] == 1
    assert not torch.allclose(q0, q3)
```

注：`CriticConfig.loss` 是 `CriticLossCfg`，其 `type` 字段控制 head 类型；非 `hl_gauss`/`c51`（即 `"mse"`）时 `Critic.forward` 直接返回逐头标量 Q `[num_q, B, 1]`（见 critic.py:325-326）。

- [ ] **Step 2: 运行测试确认通过**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py::test_critic_q_sensitive_to_stage_in_prop -v`
Expected: PASS（critic 已能消化加宽 prop 且对 stage 敏感，无需改 critic 源码）。

若该测试因 `CriticConfig.loss.type` 字段名不符而出错，先 Read `resfit/rl_finetuning/config/rlpd.py` 顶部的 `CriticLossCfg` 定义，按真实字段名设置 mse 模式，再跑。

- [ ] **Step 3: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py
git commit -m "test(stage): critic 对拼入 prop 的 stage 敏感"
```

---

## Task 4: QAgent 接线

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`（构造 21-86、`_critic_prop` 新增、6 个 critic 调用点）

- [ ] **Step 1: 加 import 与构造参数**

In `resfit/rl_finetuning/off_policy/rl/q_agent.py`, add import after line 18:

```python
from resfit.rl_finetuning.off_policy.rl.stage_utils import append_stage
```

Replace the `__init__` signature (lines 22-30) to add two params:

```python
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg: QAgentConfig,
        residual_actor: bool = False,
        stage_conditioned: bool = False,
        num_stages: int = 0,
    ):
```

Add after line 58 (`self.residual_actor = residual_actor`):

```python
        self.stage_conditioned = stage_conditioned
        self.num_stages = num_stages
```

- [ ] **Step 2: critic prop_dim 加宽 + 向 Actor 传参**

In `resfit/rl_finetuning/off_policy/rl/q_agent.py`, replace the critic+actor construction (lines 78-86):

```python
        # create critics & actor
        # stage-conditioned 时把 stage one-hot 作为额外 prop 维度喂给 critic
        critic_prop_dim = prop_dim + (self.num_stages if self.stage_conditioned else 0)
        self.critic = Critic(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=critic_prop_dim,
            action_dim=action_dim,
            cfg=self.cfg.critic,
        )
        self.actor = Actor(repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor,
                           residual_actor=residual_actor,
                           stage_conditioned=self.stage_conditioned, num_stages=self.num_stages)
```

- [ ] **Step 3: 新增 `_critic_prop` 辅助方法**

In `resfit/rl_finetuning/off_policy/rl/q_agent.py`, add this method right after `_maybe_unsqueeze_` (after line 249, before `def act`):

```python
    def _critic_prop(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """critic 用的 prop 向量。stage-conditioned 时把 stage one-hot 拼到末尾，
        与构造时加宽的 critic prop_dim 对齐。关闭时原样返回 observation.state。"""
        prop = obs["observation.state"]
        if self.stage_conditioned:
            prop = append_stage(prop, obs["observation.stage_id"], self.num_stages)
        return prop
```

- [ ] **Step 4: 替换 6 个 critic 主路径调用点**

In `resfit/rl_finetuning/off_policy/rl/q_agent.py`, make these exact replacements. Each swaps the 2nd positional arg (the prop) from a raw `*_obs["observation.state"]` to `self._critic_prop(*_obs)`.

Line 334 (target q_value):
```python
            target_all = self.critic_target.q_value(next_obs["feat"], self._critic_prop(next_obs), next_action)
```

Line 345 (hl_gauss current):
```python
            q_per_head, logits_per_head = self.critic(obs["feat"], self._critic_prop(obs), action, return_logits=True)
```

Line 351 (c51 current):
```python
            q_per_head, logits_per_head = self.critic(obs["feat"], self._critic_prop(obs), action, return_logits=True)
```

Lines 355-357 (c51 target):
```python
                _, next_logits = self.critic_target(
                    next_obs["feat"], self._critic_prop(next_obs), next_action, return_logits=True
                )
```

Line 377 (mse current):
```python
            q_all = self.critic(obs["feat"], self._critic_prop(obs), action).squeeze(-1)  # [K,B]
```

Line 447 (actor loss q_value_for_policy):
```python
        q = self.critic.q_value_for_policy(obs["feat"], self._critic_prop(obs), combined_action)
```

**不改** lines 564-565（`update_actor_rft` 的 bc 路径）：该路径仅非 residual 用（`_compute_actor_bc_loss` 对 residual_actor `assert False`），且第一版 stage-conditioning 只配 residual cl=1，不与 rft+bc 组合。保持原状，避免对没有 `stage_id` 的 bc_obs 触发拼接。

- [ ] **Step 5: 跑回归测试（关闭 stage 时行为不变）**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/ -v -m "not manual"`
Expected: 全 passed。关闭 stage 时 `_critic_prop` 原样返回 state、`critic_prop_dim==prop_dim`、Actor 不传新行为，等价改前。

- [ ] **Step 6: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/q_agent.py
git commit -m "feat(stage): QAgent 把 stage 拼进 critic prop + 透传给 actor（默认关）"
```

---

## Task 5: train_chunk_residual.py 接线

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`（import、argparse、num_stages、QAgent 构造）

- [ ] **Step 1: import NUM_STAGES**

In `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`, near the other chunk_residual imports (around line 37), add:

```python
from resfit.rl_finetuning.chunk_residual.stage_detectors import NUM_STAGES
```

- [ ] **Step 2: 加两个 argparse 开关**

In `main()`'s argparser, add after line 145 (`--stage_balanced` line):

```python
    p.add_argument("--stage_conditioned", action="store_true",
                   help="把 stage_id one-hot 喂进 actor/critic（§22 分段修正；默认关=baseline）")
    p.add_argument("--stage_budget_mode", choices=["none"], default="none",
                   help="按阶段缩放残差上限（§18.3）。第一版仅占位，逻辑未实现")
```

- [ ] **Step 3: 取 num_stages 并传给 QAgent**

In `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`, replace the QAgent construction (lines 230-232):

```python
    num_stages = NUM_STAGES.get(args.task, 1)   # 无检测器任务退化为 1 段
    if args.stage_conditioned:
        assert args.actor == "raw", "stage-conditioning 第一版只支持 --actor raw（flow 注入未接 stage）"
        assert args.task in NUM_STAGES, f"--stage_conditioned 需要 {args.task} 有 stage 检测器"
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True,
                   stage_conditioned=args.stage_conditioned, num_stages=num_stages)
```

- [ ] **Step 4: 冒烟 import / argparse（不真训）**

Run: `uv run python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual as m; print('import ok')"`
Expected: 打印 `import ok`，无 import 错误。

Run: `uv run python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual --help 2>&1 | grep -E "stage_conditioned|stage_budget_mode"`
Expected: 两个新开关出现在 help 里。

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(stage): train 脚本加 --stage_conditioned/--stage_budget_mode 开关"
```

---

## Task 6: QAgent.update 端到端烟雾（manual）

完整 QAgent 依赖 VitEncoder（较重，需构造图像），标 `manual` 不进默认 CI；需要时手动跑，端到端覆盖 Task 4 的 critic 调用点替换。

**Files:**
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`（追加）

- [ ] **Step 1: 追加 manual 烟雾测试**

Append to `resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py`:

```python
import math
import pytest
from tensordict import TensorDict
from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


@pytest.mark.manual
def test_qagent_update_smoke_stage_on():
    torch.manual_seed(0)
    C, H, W = 3, 96, 96
    cam = "observation.images.cam"
    state_dim, flat = 5, 12
    bs, ns = 4, 5

    cfg = QAgentConfig()
    cfg.device = "cpu"
    cfg.critic.loss.type = "mse"
    agent = QAgent(obs_shape=(C, H, W), prop_shape=(state_dim,), action_dim=flat,
                   rl_cameras=[cam], cfg=cfg, residual_actor=True,
                   stage_conditioned=True, num_stages=ns)
    agent.train(True)
    agent.actor_target.train(True)

    def _obs():
        return {
            cam: torch.rand(bs, C, H, W),
            "observation.state": torch.randn(bs, state_dim),
            "observation.base_action": torch.tanh(torch.randn(bs, flat)),
            "observation.stage_id": torch.randint(0, ns, (bs, 1)).float(),
        }

    batch = TensorDict({
        "obs": TensorDict(_obs(), batch_size=[bs]),
        "action": torch.tanh(torch.randn(bs, flat)),
        ("next", "reward"): torch.zeros(bs),
        "gamma": torch.full((bs,), 0.99),
        "nonterminal": torch.ones(bs),
        ("next", "obs"): TensorDict(_obs(), batch_size=[bs]),
    }, batch_size=[bs])

    metrics = agent.update(batch, stddev=0.05, update_actor=True)
    assert math.isfinite(metrics["train/critic_loss"])
    assert math.isfinite(metrics["train/actor_loss_total"])
```

注：若 `VitEncoder` 对输入分辨率有约束导致构造/前向报错，先 Read `resfit/rl_finetuning/off_policy/networks/encoder.py` 的 `VitEncoder` 与 `VitEncoderConfig`，把 `(C,H,W)` 调成其支持的尺寸（patch 整除）。batch 的 key 结构对照 `q_agent.py::update`（lines 606-611）：`obs`/`action`/`("next","reward")`/`gamma`/`nonterminal`/`("next","obs")`。

- [ ] **Step 2: 手动跑该烟雾（可选，需要时）**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py::test_qagent_update_smoke_stage_on -v -m manual`
Expected: PASS，critic/actor loss 均 finite。

- [ ] **Step 3: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py
git commit -m "test(stage): QAgent.update 端到端烟雾（manual）"
```

---

## 收尾验证

- [ ] **全量默认测试通过**

Run: `uv run pytest resfit/rl_finetuning/chunk_residual/tests/ -v -m "not manual"`
Expected: 全 passed，含 Task 1-4 新测与所有现有测（关闭 stage 时无回归）。

- [ ] **lint**

Run: `uv run ruff check resfit/rl_finetuning/off_policy/rl/stage_utils.py resfit/rl_finetuning/off_policy/rl/actor.py resfit/rl_finetuning/off_policy/rl/q_agent.py resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
Expected: 无错误（或 `--fix` 后无错误）。

---

## 实现完成后（不在本计划编码范围）

A/B 实验由用户手动 GPU 跑：A=`--stage_conditioned` 关，B=开，同超参，cl=1 queue 模式。
看 `eval_stage_reach` 的 reach3→reach4 与 success。先决条件：确认 residual 真长出来
（`actor_lr 1e-6` 可能两边 residual_norm 都≈0；必要时调 `5e-6` 或给足步数）。
