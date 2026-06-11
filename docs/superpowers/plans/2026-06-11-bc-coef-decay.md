# BC 系数线性衰减 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 demo-BC 系数加一个随训练步数线性衰减的开关 `--bc_coef_final`,默认 `None` 时逐位等价现状。

**Architecture:** 衰减数学抽成无重依赖的纯函数 `linear_bc_coef`(单独可测);训练循环每次 actor update 按 `env_steps` 算当前系数写回 `agent.cfg.bc_loss_coef`,`q_agent.py:605` 自然读到;agent / q_agent 源码不动。默认 `None` 不进衰减分支,正在跑的 run 不受影响。

**Tech Stack:** Python / PyTorch;pytest;wandb。测试用 `conda run -n residual`(base python 无 `lerobot.common.*`)。

参考 spec:`docs/superpowers/specs/2026-06-11-bc-coef-decay-design.md`

**环境前缀(所有 pytest 命令都用):**
```bash
cd /mnt/mnt/data/resfit
ENVRUN="conda run -n residual --no-capture-output env PYTHONPATH=/mnt/mnt/data/resfit OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
```

---

## File Structure

- **新建** `resfit/rl_finetuning/chunk_residual/bc_schedule.py` — 纯函数 `linear_bc_coef`,单一职责(衰减数学),无重依赖。
- **新建** `resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py` — 纯函数单测。
- **改** `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` — import + `--bc_coef_final` flag + 校验 + 循环调用 + wandb 记录。
- **改** `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py` — 追加 `--bc_coef_final` 的 parser 测试(沿用 101-108 现有范式)。

---

### Task 1: 衰减纯函数 `linear_bc_coef`

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/bc_schedule.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py`

- [ ] **Step 1: 写失败测试**

Create `resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py`:

```python
import pytest

from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef


def test_start_returns_c0():
    assert linear_bc_coef(0, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.1)


def test_end_returns_c_final():
    assert linear_bc_coef(500_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.01)


def test_midpoint_is_average():
    assert linear_bc_coef(250_000, c0=0.1, c_final=0.0, total_steps=500_000) == pytest.approx(0.05)


def test_beyond_end_clipped_to_c_final():
    assert linear_bc_coef(900_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.01)


def test_negative_steps_clipped_to_c0():
    assert linear_bc_coef(-100, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.1)


def test_zero_total_steps_returns_c_final():
    # 防除零:total_steps<=0 退化为终值
    assert linear_bc_coef(0, c0=0.1, c_final=0.01, total_steps=0) == pytest.approx(0.01)


def test_flat_when_final_equals_c0():
    for t in (0, 123_456, 500_000):
        assert linear_bc_coef(t, c0=0.1, c_final=0.1, total_steps=500_000) == pytest.approx(0.1)


def test_concrete_quarter_point():
    # c0=0.1, c_final=0.01, t=125000/500000=0.25 -> 0.1 + (0.01-0.1)*0.25 = 0.0775
    assert linear_bc_coef(125_000, c0=0.1, c_final=0.01, total_steps=500_000) == pytest.approx(0.0775)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$ENVRUN pytest resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named '...bc_schedule'`

- [ ] **Step 3: 写最小实现**

Create `resfit/rl_finetuning/chunk_residual/bc_schedule.py`:

```python
"""demo-BC 系数的训练步数调度(模块②a 增强)。

当前 BC 系数在 train_chunk_residual.py 建 agent 时一次性写死(cfg.agent.bc_loss_coef)。
本模块提供线性衰减:训练早期强约束(锚专家防塌方)、后期放手(残差自由探索)。
纯函数、无重依赖,便于单测;由训练循环每步调用后写回 agent.cfg.bc_loss_coef。
"""


def linear_bc_coef(env_steps, *, c0, c_final, total_steps):
    """BC 系数线性衰减。

    env_steps 从 0 到 total_steps 时,返回值从 c0 线性变到 c_final;区间外 clip
    (env_steps<0 -> c0,env_steps>=total_steps -> c_final)。total_steps<=0 时
    退化为 c_final(防除零)。

    参数:
      env_steps:   当前已采样环境步数。
      c0:          起始系数(= --demo_bc_coef)。
      c_final:     终值系数(= --bc_coef_final,floor)。
      total_steps: 衰减区间长度(= --total_env_steps)。
    """
    if total_steps <= 0:
        return c_final
    progress = min(max(env_steps / total_steps, 0.0), 1.0)
    return c0 + (c_final - c0) * progress
```

- [ ] **Step 4: 跑测试确认通过**

Run: `$ENVRUN pytest resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py -q`
Expected: PASS(8 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/bc_schedule.py \
        resfit/rl_finetuning/chunk_residual/tests/test_bc_schedule.py
git commit -m "feat: linear_bc_coef 衰减纯函数 + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: train_chunk_residual 接入 `--bc_coef_final`

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - import(34 行附近)、flag(319 行后)、校验(480 行后)、循环(682 块)、wandb(705 行)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py`(追加 parser 测试)

- [ ] **Step 1: 写失败测试**(追加到 `tests/test_demo_bc.py` 末尾)

```python
def test_cli_bc_coef_final_default_none():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args([]).bc_coef_final is None


def test_cli_bc_coef_final_parses():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    assert build_parser().parse_args(["--bc_coef_final", "0.01"]).bc_coef_final == 0.01
```

- [ ] **Step 2: 跑测试确认失败**

Run: `$ENVRUN pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -q -k bc_coef_final`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'bc_coef_final'`

- [ ] **Step 3a: 加 import**(在 `train_chunk_residual.py:34` 的 `from ...q_agent import QAgent` 之后新增一行)

```python
from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef
```

- [ ] **Step 3b: 加 flag**(在 `--demo_bc_coef` 的 argparse 块之后、`--relabel` 之前,即当前第 319 行 help 文本结束后插入)

```python
    p.add_argument("--bc_coef_final", type=float, default=None,
                   help="demo-BC 系数线性衰减的终值(floor);不传=固定 demo_bc_coef(逐位等价)。"
                        "传值 v 则 bc_loss_coef 从 demo_bc_coef 线性降到 v(区间 0→total_env_steps)。"
                        "需 demo_bc_coef>0 且 0<=v<=demo_bc_coef")
```

- [ ] **Step 3c: 加校验**(在 `demo_bc_coef>0` 的 assert 块之后,即当前第 480 行 `"demo_bc_coef>0 需 offline_fraction>0(...)"` 这条 assert 结束后插入)

```python
    if args.bc_coef_final is not None:
        assert args.demo_bc_coef > 0, \
            "--bc_coef_final 需 --demo_bc_coef>0(BC 未开则衰减无意义)"
        assert 0.0 <= args.bc_coef_final <= args.demo_bc_coef, \
            f"--bc_coef_final 需在 [0, demo_bc_coef={args.demo_bc_coef}] 内,得到 {args.bc_coef_final}"
```

- [ ] **Step 3d: 循环里写回当前系数**(在第 682 行 `if args.demo_bc_coef > 0 and update_actor and offline_rb is not None:` 之后、`if args.relabel:` 之前插入;缩进与块内 `if args.relabel:` 一致)

```python
                    if args.bc_coef_final is not None:
                        agent.cfg.bc_loss_coef = linear_bc_coef(
                            env_steps, c0=args.demo_bc_coef,
                            c_final=args.bc_coef_final, total_steps=args.total_env_steps)
```

- [ ] **Step 3e: wandb 记录当前系数**(把当前第 705 行 `wandb.log(build_train_log_dict(m_upd, lrs, buf_sizes), step=env_steps)` 替换为下面三行,缩进对齐)

```python
                log_dict = build_train_log_dict(m_upd, lrs, buf_sizes)
                log_dict["rft/bc_coef_cur"] = agent.cfg.bc_loss_coef
                wandb.log(log_dict, step=env_steps)
```

- [ ] **Step 4: 跑 parser 测试确认通过**

Run: `$ENVRUN pytest resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py -q`
Expected: PASS(原有 + 2 新 = 全 passed)

- [ ] **Step 5: 提交**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py \
        resfit/rl_finetuning/chunk_residual/tests/test_demo_bc.py
git commit -m "feat: --bc_coef_final 接入 train(衰减写回 agent.cfg.bc_loss_coef + wandb)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 全量回归 + 默认逐位等价核对

**Files:**(不改代码,只验证)

- [ ] **Step 1: 跑 chunk_residual 全量测试**

Run: `$ENVRUN pytest resfit/rl_finetuning/chunk_residual/tests/ -q`
Expected: 全 passed(= 旧基线 302 + 本次新增 10:test_bc_schedule 8 + test_demo_bc 2),0 failed。若有 skip 属既有环境跳过。

- [ ] **Step 2: 默认等价核对**(确认不传 flag 时行为不变)

Run: `$ENVRUN python -c "from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser; a=build_parser().parse_args([]); print('bc_coef_final=', a.bc_coef_final); assert a.bc_coef_final is None"`
Expected: 打印 `bc_coef_final= None`,无 AssertionError。

- [ ] **Step 3: 衰减数学抽样核对**(纯函数端到端值)

Run: `$ENVRUN python -c "from resfit.rl_finetuning.chunk_residual.bc_schedule import linear_bc_coef as f; print(f(0,c0=0.1,c_final=0.01,total_steps=500000), f(250000,c0=0.1,c_final=0.01,total_steps=500000), f(500000,c0=0.1,c_final=0.01,total_steps=500000))"`
Expected: `0.1 0.055 0.01`

- [ ] **Step 4: 无新提交**(回归通过即可;若 Step 1-3 暴露问题,回到对应 Task 修复并重跑)

---

## Self-Review

**1. Spec coverage:**
- spec §2 形状 linear → Task 1 `linear_bc_coef`。
- spec §2 区间 0→total → Task 1 `total_steps=total_env_steps`、Task 2 Step 3d 传 `args.total_env_steps`。
- spec §2 终值 floor + 总开关合一(`--bc_coef_final` 默认 None)→ Task 2 Step 3b + Task 3 Step 2。
- spec §2 校验(demo_bc_coef>0 且 0≤final≤demo_bc_coef)→ Task 2 Step 3c。
- spec §2 wandb rft/bc_coef_cur → Task 2 Step 3e。
- spec §3.3 向后兼容(默认 None 逐位等价)→ Task 3 Step 2。
- spec §5 文件清单 → 全覆盖。

**2. Placeholder scan:** 无 TBD/TODO;每个 code step 都给完整代码。

**3. Type consistency:** `linear_bc_coef(env_steps, *, c0, c_final, total_steps)` 在 Task 1 定义、Task 2 Step 3d 与 Task 3 Step 3 调用签名一致(关键字参数 c0/c_final/total_steps)。`args.bc_coef_final` 名称在 flag/校验/循环/测试中一致。
