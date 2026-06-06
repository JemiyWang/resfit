# stage_budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 chunk_residual 残差 actor 加一个 default-off 的"逐阶段残差幅度预算"开关,在精插入阶段(stage3/4)机械地把残差输出包络(均值+探索噪声)缩小。

**Architecture:** 全部改动落在 `Actor.forward` 一处:`scaled_mu = mu * action_scale` 之后按当前 stage_id 乘一个预算因子,均值和 std 一起缩;因为 actor/actor_target/rollout/eval/actor-loss 全走这条 forward,一处改、五处一致。stage_budget=None 时逐字节等价 baseline。只缩输出、不把 stage 喂进网络(区别于已证伪的 stage_conditioning)。

**Tech Stack:** PyTorch；torchrl/tensordict（仅集成测试用）；pytest（conda env `residual`，命令前缀 `conda run -n residual`）。

设计依据：`docs/superpowers/specs/2026-06-06-stage-budget-design.md`。

---

## 文件结构

- `resfit/rl_finetuning/off_policy/rl/stage_utils.py` —— 加两个纯函数 `stage_budget_factor`、`parse_stage_budget`（与现有 `stage_onehot` 同处，可单测）。
- `resfit/rl_finetuning/off_policy/rl/actor.py` —— `Actor.__init__` 加 `stage_budget` 形参 + 注册 buffer；`forward` 内施加预算。
- `resfit/rl_finetuning/off_policy/rl/q_agent.py` —— `QAgent.__init__` 加 `stage_budget` 形参，透传给构造 `Actor` 的一处（actor_target 靠 deepcopy 继承）。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` —— `build_parser` 用 `--stage_budget` 取代占位 `--stage_budget_mode`；`main` 里 `parse_stage_budget` + 断言 `--actor raw` + 传 QAgent。
- `resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py` —— 新建测试文件（纯函数、forward、QAgent 集成 manual）。

测试运行约定（全部）：从仓库根目录、conda env `residual`：
`conda run -n residual python -m pytest <path> -v`（默认排除 `-m manual`）。

---

## Task 1: 纯函数 stage_budget_factor + parse_stage_budget

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/stage_utils.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py`

- [ ] **Step 1: 写失败测试**

新建 `resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py`:

```python
import torch

from resfit.rl_finetuning.off_policy.rl.stage_utils import (
    stage_budget_factor, parse_stage_budget,
)

NUM_STAGES = 5
BUDGET = [1.0, 1.0, 1.0, 0.3, 0.1]


def test_stage_budget_factor_mixed():
    budget = torch.tensor(BUDGET)
    sid = torch.tensor([[0.], [3.], [4.], [1.]])
    f = stage_budget_factor(sid, budget, NUM_STAGES)
    assert f.shape == (4, 1)
    assert torch.allclose(f.reshape(-1), torch.tensor([1.0, 0.3, 0.1, 1.0]))


def test_stage_budget_factor_clamps_out_of_range():
    budget = torch.tensor([1.0, 0.5, 0.2])
    sid = torch.tensor([[5.], [-1.]])          # 越界 -> clamp 到 idx 2 和 0
    f = stage_budget_factor(sid, budget, 3)
    assert torch.allclose(f.reshape(-1), torch.tensor([0.2, 1.0]))


def test_parse_stage_budget_none():
    assert parse_stage_budget(None, NUM_STAGES) is None


def test_parse_stage_budget_ok():
    assert parse_stage_budget("1,1,1,0.3,0.1", NUM_STAGES) == [1.0, 1.0, 1.0, 0.3, 0.1]


def test_parse_stage_budget_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        parse_stage_budget("1,1,0.1", NUM_STAGES)   # 长度 3 != 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -v`
Expected: FAIL（`ImportError: cannot import name 'stage_budget_factor'`）

- [ ] **Step 3: 实现两个纯函数**

在 `resfit/rl_finetuning/off_policy/rl/stage_utils.py` 末尾追加：

```python
def stage_budget_factor(stage_id: torch.Tensor, budget: torch.Tensor, num_stages: int) -> torch.Tensor:
    """stage_id [B,1] 或 [B] float -> 每样本残差幅度乘子 [B,1] float32。

    budget: 1-D 张量，长度 num_stages，budget[k] = stage k 的乘子。
    越界 stage_id clamp 到 [0, num_stages-1]。返回乘子在 stage_id 的 device 上。
    """
    idx = stage_id.reshape(stage_id.shape[0]).long().clamp_(0, num_stages - 1)
    fac = budget.to(idx.device)[idx]
    return fac.reshape(-1, 1).to(torch.float32)


def parse_stage_budget(arg: "str | None", num_stages: int) -> "list[float] | None":
    """解析 --stage_budget 字符串 -> list[float]；None 表关。长度必须 == num_stages。"""
    if arg is None:
        return None
    vals = [float(x) for x in arg.split(",")]
    if len(vals) != num_stages:
        raise ValueError(f"stage_budget 长度 {len(vals)} != num_stages {num_stages}")
    return vals
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -v`
Expected: PASS（5 个纯函数测试全绿）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/stage_utils.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py
git commit -m "feat(stage_budget): stage_budget_factor + parse_stage_budget 纯函数 + 单测"
```

---

## Task 2: Actor 施加预算（mean + std 都缩）

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/actor.py`（import 行 10；`Actor.__init__` 70-78；`forward` 183-186）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py`

- [ ] **Step 1: 追加失败测试**

在 `test_stage_budget.py` 顶部 import 后追加（复用现有 test_stage_conditioned 的构造惯例）：

```python
from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.rl.actor import Actor

REPR, PATCH, PROP, FLAT = 32, 16, 3, 20
B = 6


def _actor(stage_budget):
    cfg = ActorConfig()                       # spatial_emb=0 默认 -> residual 兼容 else 分支
    return Actor(REPR, PATCH, PROP, FLAT, cfg, residual_actor=True,
                 num_stages=NUM_STAGES, stage_budget=stage_budget)


def _obs(stage_vals):                          # stage_vals: 长度 B 的 list
    return {
        "feat": torch.randn(B, REPR // PATCH, PATCH),
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
        "observation.stage_id": torch.tensor(stage_vals, dtype=torch.float32).reshape(B, 1),
    }


def test_actor_budget_off_scale_uniform():
    # 关:scale 是均匀标量 0.05(TruncatedNormal float -> ones_like*scale),不随 stage 变
    torch.manual_seed(0)
    actor = _actor(stage_budget=None).eval()
    dist = actor.forward(_obs([0, 1, 2, 3, 4, 0]), std=0.05)
    assert torch.allclose(dist.scale, torch.full_like(dist.loc, 0.05))


def test_actor_budget_scales_mean_and_std():
    # 开:同一 actor(同权重),mean 与 std 都被逐样本 factor 缩
    torch.manual_seed(0)
    actor = _actor(stage_budget=BUDGET).eval()
    obs = _obs([0, 3, 4, 1, 2, 3])
    factor = torch.tensor([1.0, 0.3, 0.1, 1.0, 1.0, 0.3]).reshape(B, 1)
    dist_on = actor.forward(obs, std=0.05)
    actor.stage_budget = None                  # 关掉同一 actor 取 baseline
    dist_off = actor.forward(obs, std=0.05)
    assert torch.allclose(dist_on.loc, dist_off.loc * factor, atol=1e-6)
    assert torch.allclose(dist_on.scale, torch.full_like(dist_on.loc, 0.05) * factor, atol=1e-6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -k actor -v`
Expected: FAIL（`Actor.__init__() got an unexpected keyword argument 'stage_budget'`）

- [ ] **Step 3: 改 Actor**

3a. `actor.py` 第 10 行 import 加上 `stage_budget_factor`：

```python
from resfit.rl_finetuning.off_policy.rl.stage_utils import stage_onehot, stage_budget_factor
```

3b. `Actor.__init__` 签名（第 70-71 行）末尾加形参 `stage_budget`：

```python
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: ActorConfig, residual_actor: bool = False,
                 stage_conditioned: bool = False, num_stages: int = 0,
                 stage_budget: "list[float] | None" = None):
```

3c. 在 `self.cfg = cfg`（第 78 行）之后追加存储 + 注册 buffer：

```python
        self.stage_budget = stage_budget
        if stage_budget is not None:
            assert len(stage_budget) == num_stages, \
                f"stage_budget 长度 {len(stage_budget)} != num_stages {num_stages}"
            self.register_buffer("_stage_budget", torch.tensor(stage_budget, dtype=torch.float32))
```

3d. `forward` 内,把 `scaled_mu = mu * self.cfg.action_scale`（第 183 行）与
`action_dist = utils.TruncatedNormal(scaled_mu, std)`（第 186 行）之间插入预算块：

```python
        scaled_mu = mu * self.cfg.action_scale

        if self.stage_budget is not None:
            # 只缩输出包络,不改网络输入(区别于 stage_conditioning)
            factor = stage_budget_factor(
                obs["observation.stage_id"], self._stage_budget, self.num_stages).to(scaled_mu.device)
            scaled_mu = scaled_mu * factor
            std = std * factor                  # float * [B,1] -> [B,1]，TruncatedNormal 接受张量 scale 并广播

        action_dist = utils.TruncatedNormal(scaled_mu, std)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -v`
Expected: PASS（含 Task 1 的 5 个 + 本任务 2 个）

并跑现有 actor/stage 回归测试确认没踩坏：
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_conditioned.py -m "not manual" -v`
Expected: PASS（stage_conditioned 行为不变；stage_budget 默认 None 不影响）

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/actor.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py
git commit -m "feat(stage_budget): Actor.forward 按 stage 缩残差均值+std（默认关=baseline）"
```

---

## Task 3: QAgent 透传 + train 脚本接线 + 端到端集成测试

**Files:**
- Modify: `resfit/rl_finetuning/off_policy/rl/q_agent.py`（`__init__` 签名 23-32；建 `Actor` 处 93-95）
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`（`build_parser` 153-154；`main` 252-259）
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py`

- [ ] **Step 1: 追加失败测试（集成 manual + CLI）**

在 `test_stage_budget.py` 末尾追加：

```python
import pytest


def test_cli_stage_budget_default_none():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args([])
    assert args.stage_budget is None


def test_cli_stage_budget_parses_string():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args(["--stage_budget", "1,1,1,0.3,0.1"])
    assert args.stage_budget == "1,1,1,0.3,0.1"


@pytest.mark.manual
def test_qagent_act_budget_only_scales_output():
    """核心性质:同一 obs 只改 stage_id,残差只被乘 factor、网络计算不变。
    (eval_mode + stddev=0 => 取均值;stage4 budget=0.1 => a4 == a0 * 0.1)
    依赖 VitEncoder,故 manual。"""
    import math
    from resfit.rl_finetuning.config.rlpd import QAgentConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    torch.manual_seed(0)
    C, H, W = 3, 84, 84
    cam = "observation.images.agentview"
    state_dim, flat, ns = 5, 12, 5
    budget = [1.0, 1.0, 1.0, 1.0, 0.1]                 # 只在 stage4 卡死
    cfg = QAgentConfig(); cfg.device = "cpu"; cfg.critic.loss.type = "mse"
    agent = QAgent(obs_shape=(C, H, W), prop_shape=(state_dim,), action_dim=flat,
                   rl_cameras=[cam], cfg=cfg, residual_actor=True,
                   num_stages=ns, stage_budget=budget)
    agent.train(False)
    base = {cam: torch.rand(8, C, H, W),
            "observation.state": torch.randn(8, state_dim),
            "observation.base_action": torch.tanh(torch.randn(8, flat))}
    o0 = dict(base); o0["observation.stage_id"] = torch.zeros(8, 1)        # budget 1.0
    o4 = dict(base); o4["observation.stage_id"] = torch.full((8, 1), 4.0)  # budget 0.1
    with torch.no_grad():
        a0 = agent.act(o0, eval_mode=True, stddev=0.0, cpu=True)
        a4 = agent.act(o4, eval_mode=True, stddev=0.0, cpu=True)
    assert torch.allclose(a4, a0 * 0.1, atol=1e-5)     # 只缩输出,不改网络计算
    assert math.isfinite(a0.abs().sum().item())
```

- [ ] **Step 2: 运行 CLI 测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -k cli -v`
Expected: FAIL（`--stage_budget` 还不是合法参数 -> argparse SystemExit；或 `args.stage_budget` AttributeError）

- [ ] **Step 3a: QAgent 透传 stage_budget**

`q_agent.py` `__init__` 签名（第 30-32 行附近）`num_stages: int = 0,` 之后加：

```python
        stage_budget: "list[float] | None" = None,
```

建 `self.actor`（第 93-95 行）加上 `stage_budget=stage_budget`：

```python
        self.actor = Actor(repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor,
                           residual_actor=residual_actor,
                           stage_conditioned=self.stage_conditioned, num_stages=self.num_stages,
                           stage_budget=stage_budget)
```

（第 98 行 `self.actor_target = copy.deepcopy(self.actor)` 自动继承,无需改;critic 不动。）

- [ ] **Step 3b: train 脚本接线**

`train_chunk_residual.py` `build_parser` 内,删掉占位 `--stage_budget_mode`（第 153-154 行）：

```python
    p.add_argument("--stage_budget_mode", choices=["none"], default="none",
                   help="按阶段缩放残差上限（§18.3）。第一版仅占位，逻辑未实现")
```

替换为：

```python
    p.add_argument("--stage_budget", default=None,
                   help="逐阶段残差幅度乘子,逗号分隔,长度=num_stages(如 '1,1,1,0.3,0.1');不传=关(§18.3)")
```

`main` 内,在 `num_stages = NUM_STAGES.get(args.task, 1)`（第 252 行）与建 agent 之间,加解析+断言：

```python
    from resfit.rl_finetuning.off_policy.rl.stage_utils import parse_stage_budget
    stage_budget = parse_stage_budget(args.stage_budget, num_stages)
    if stage_budget is not None:
        assert args.actor == "raw", "stage_budget 第一版只支持 --actor raw"
        assert args.task in NUM_STAGES, f"--stage_budget 需要 {args.task} 有 stage 检测器"
```

建 `agent = QAgent(...)`（第 256-259 行）加上 `stage_budget=stage_budget`：

```python
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True,
                   stage_conditioned=args.stage_conditioned, num_stages=num_stages,
                   stage_budget=stage_budget)
```

- [ ] **Step 4: 运行测试确认通过**

CLI（轻量）：
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -m "not manual" -v`
Expected: PASS（全部轻量测试绿）

集成（manual，CPU 可跑，约几十秒）：
Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py -k budget_only_scales_output -m manual -v`
Expected: PASS（`a4 == a0 * 0.1`，证明只缩输出、不改网络计算）

- [ ] **Step 5: baseline 逐位等价冒烟（关键回归）**

确认不传 `--stage_budget` 时 train 脚本仍能起、行为同改造前(用 --smoke 少量步)：
Run:
```bash
conda run -n residual env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --chunk_length 1 --base_action_mode queue \
  --base_wandb_id /data2/RL/residual-offpolicy-rl/bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/best \
  --dataset ankile/dexmg-two-arm-three-piece-assembly --smoke
```
Expected: 正常跑完 smoke(退出码 0),无 stage_budget 相关报错。
（注:`--smoke` 自动 wandb disabled;此步只验证接线没破 baseline 启动,不评效果。）

- [ ] **Step 6: 提交**

```bash
git add resfit/rl_finetuning/off_policy/rl/q_agent.py \
        resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_stage_budget.py
git commit -m "feat(stage_budget): QAgent 透传 + train --stage_budget 接线 + 集成测试"
```

---

## 完成后(不属本实现,A/B 在 plan/run 阶段)

实现完成后的下一发实验(见 spec §6):用 nas10 强 base(`--base_n_action_steps 10`
`--offline_fraction 0.5` `--reward_shaping staged` `--base_action_mode queue` `--chunk_length 1`),
A = 不传 `--stage_budget`(baseline)、B = `--stage_budget "1,1,1,0.3,0.1"`,
看 `eval_stage_reach` 的 reach3->reach4 与 success,旁证 stage-diag 里 stage3/4 的 residual_norm 下降。

后续模块:② RPL relay relabeling、③ HIQL-Φ 各自另开 spec+plan。
